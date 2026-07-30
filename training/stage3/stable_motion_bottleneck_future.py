"""Future-visible armor predictor with a supervised physical motion bottleneck.

The causal history encoder is allowed to estimate only four stable quantities:
target-center velocity in the tracker frame and physical yaw rate.  Both the
learned trajectory field and the modulo-four selector consume that predicted
four-vector plus S-owned q0 relative geometry.  Physical truth is never a
forward input; it is used only by :func:`stable_motion_future_loss`.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .anonymous_vehicle_motion_v2 import _masked_window_mean, target_roles
from .continuous_invariant_anonymous_future import V3_FORWARD_FIELDS
from .increment_invariant_anonymous_future import (
    IncrementOnlyMotionContext,
    IncrementInvariantAnonymousFutureModel,
)


MOTION_STATE_FIELDS = ("velocity_x", "velocity_y", "velocity_z", "yaw_rate")


class StableIncrementMotionContext(IncrementOnlyMotionContext):
    """Pure temporal state before every q0 geometry or raw-origin injection."""

    model_family = "stable-increment-motion-context-v5"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # displacement xyz, elapsed, velocity xyz, time-to-q0, primary,
        # |switch|, |cumulative switch|, velocity-valid, first-visible = 13.
        self.history_projection = nn.Sequential(
            nn.Linear(13, self.channels), nn.LayerNorm(self.channels), nn.SiLU(),
        )
        del self.q0_projection

    @property
    def config(self) -> dict[str, Any]:
        parent = dict(super().config)
        parent.update({
            "family": self.model_family,
            "temporal_features": (
                "same-handle displacement/dt/velocity/time/switch/masks only"
            ),
            "first_visible_origin_offset": False,
            "q0_geometry_in_motion_estimator": False,
            "raw_history_origin_feature": False,
        })
        return parent

    def forward(
        self,
        history_obs_rel_m: torch.Tensor,
        history_obs_mask: torch.Tensor,
        history_primary_mask: torch.Tensor,
        history_event_mask: torch.Tensor,
        history_time_s: torch.Tensor,
        history_dt_s: torch.Tensor,
        history_switch_step: torch.Tensor,
        q0_relation_m: torch.Tensor,
        q0_sigma_m: torch.Tensor,
        q0_confidence: torch.Tensor,
        q0_age_s: torch.Tensor,
        q0_support_class: torch.Tensor,
        q0_supported: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del history_dt_s, q0_sigma_m, q0_confidence, q0_age_s, q0_support_class
        if history_obs_rel_m.ndim != 4 or history_obs_rel_m.shape[2:] != (4, 3):
            raise ValueError("history observations must have shape [B,T,4,3]")
        batch, events = history_obs_rel_m.shape[:2]
        if history_obs_mask.shape != (batch, events, 4):
            raise ValueError("history observation mask has the wrong shape")
        if history_primary_mask.shape != history_obs_mask.shape:
            raise ValueError("history primary mask has the wrong shape")
        if any(value.shape != (batch, events) for value in (
            history_event_mask, history_time_s, history_switch_step,
        )):
            raise ValueError("history scalar fields must have shape [B,T]")
        if q0_relation_m.shape != (batch, 4, 3) or q0_supported.shape != (batch, 4):
            raise ValueError("q0 geometry fields have the wrong shape")

        active = history_event_mask.to(torch.bool)
        visible = history_obs_mask.to(torch.bool) & active.unsqueeze(-1)
        primary = history_primary_mask.to(torch.bool) & active.unsqueeze(-1)
        if bool(torch.any(active.sum(dim=1) < 8)):
            raise ValueError("context requires at least eight active events")
        if bool(torch.any(visible.sum(dim=2)[active] < 1)):
            raise ValueError("active events need an observation")
        if bool(torch.any(visible.sum(dim=2) > 2)):
            raise ValueError("at most two anonymous handles may be visible")
        if bool(torch.any(primary.sum(dim=2)[active] != 1)):
            raise ValueError("active events need exactly one primary handle")
        if bool(torch.any(primary & ~visible)):
            raise ValueError("primary handle must be visible")
        if bool(torch.any(active & (history_time_s > 1e-6))):
            raise ValueError("history cannot contain future events")
        if bool(torch.any(active & ~torch.isfinite(history_time_s))):
            raise ValueError("active history time must be finite")
        valid_switch = torch.isin(
            history_switch_step.to(torch.long),
            history_switch_step.new_tensor((-1, 0, 1), dtype=torch.long),
        )
        if bool(torch.any(active & ~valid_switch)):
            raise ValueError("history switch values must be -1, 0 or +1")
        if bool(torch.any(visible & ~torch.isfinite(history_obs_rel_m).all(dim=-1))):
            raise ValueError("visible history coordinates must be finite")

        last_event = self._last_active(active)
        last_time = history_time_s.gather(1, last_event[:, None]).squeeze(1)
        if bool(torch.any(torch.abs(last_time) > 1e-6)):
            raise ValueError("last active event must be q0")
        rows = torch.arange(batch, device=active.device)
        current_primary = primary[rows, last_event]
        clean_obs = torch.where(
            visible.unsqueeze(-1), history_obs_rel_m,
            torch.zeros_like(history_obs_rel_m),
        )
        clean_time = torch.where(active, history_time_s, torch.zeros_like(history_time_s))
        clean_switch = torch.where(
            active, history_switch_step, torch.zeros_like(history_switch_step),
        )
        cumulative = torch.cumsum(clean_switch, dim=1)
        cumulative_q0 = cumulative.gather(1, last_event[:, None])
        cumulative_relative = torch.where(
            active, cumulative - cumulative_q0, torch.zeros_like(cumulative),
        )
        elapsed, local_velocity, velocity_valid = self.visible_differences(
            clean_obs, visible, clean_time,
        )
        displacement = torch.where(
            velocity_valid.unsqueeze(-1), local_velocity * elapsed.unsqueeze(-1),
            torch.zeros_like(local_velocity),
        )
        first_visible = visible & ~velocity_valid
        position_scale = self.position_scale_m
        time_scale = self.history_scale_s
        feature = torch.cat((
            displacement / position_scale,
            (elapsed / time_scale).unsqueeze(-1),
            local_velocity * (time_scale / position_scale),
            (clean_time / time_scale)[:, :, None, None].expand(-1, -1, 4, 1),
            primary.to(clean_obs.dtype).unsqueeze(-1),
            clean_switch.abs().to(clean_obs.dtype)[:, :, None, None].expand(-1, -1, 4, 1),
            (cumulative_relative.abs().to(clean_obs.dtype) / 6.0)[:, :, None, None].expand(-1, -1, 4, 1),
            velocity_valid.to(clean_obs.dtype).unsqueeze(-1),
            first_visible.to(clean_obs.dtype).unsqueeze(-1),
        ), dim=-1)
        compact, compact_mask, visible_count = self._compact_visible(feature, visible)
        sequence = self.history_projection(compact)
        sequence = sequence.permute(0, 1, 3, 2).reshape(
            batch * 4, self.channels, events,
        )
        lane_mask = compact_mask.reshape(batch * 4, events)
        for block in self.temporal:
            sequence = block(sequence, lane_mask)
        gather = (visible_count - 1).clamp_min(0).reshape(-1)
        lane_last = sequence.gather(
            2, gather[:, None, None].expand(-1, self.channels, 1),
        ).squeeze(2).reshape(batch, 4, self.channels)
        lane_last = torch.where(
            (visible_count > 0).unsqueeze(-1), lane_last, torch.zeros_like(lane_last),
        )
        lane_mean = sequence.sum(dim=2).reshape(batch, 4, self.channels)
        lane_mean = lane_mean / visible_count.clamp_min(1).unsqueeze(-1)
        handle_state = torch.cat((lane_last, lane_mean), dim=-1)
        for message in self.messages:
            handle_state = message(handle_state)
        vehicle_state = torch.cat((
            handle_state.mean(dim=1), handle_state.amax(dim=1),
        ), dim=-1)
        return {
            "handle_state": handle_state,
            "vehicle_state": vehicle_state,
            "primary_index": current_primary.to(torch.long).argmax(dim=1),
            "history_active_count": active.sum(dim=1),
            "history_visible_count": visible_count,
            "history_visible_elapsed_s": elapsed,
            "history_local_velocity_mps": local_velocity,
            "history_velocity_valid": velocity_valid,
        }


class StableMotionBottleneckAnonymousFutureModel(
    IncrementInvariantAnonymousFutureModel
):
    """Learned future field whose only temporal interface is a physical 4D code."""

    model_family = "stable-motion-bottleneck-anonymous-future-v5"

    def __init__(
        self,
        *,
        velocity_scale_mps: tuple[float, float, float],
        yaw_rate_scale_rad_s: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if len(velocity_scale_mps) != 3 or any(
            not math.isfinite(float(value)) or float(value) <= 0
            for value in velocity_scale_mps
        ):
            raise ValueError("velocity scales must contain three positive values")
        if not math.isfinite(float(yaw_rate_scale_rad_s)) or yaw_rate_scale_rad_s <= 0:
            raise ValueError("yaw-rate scale must be positive")
        self.register_buffer(
            "motion_state_scale",
            torch.tensor(
                (*velocity_scale_mps, yaw_rate_scale_rad_s), dtype=torch.float32,
            ),
        )
        self.context = StableIncrementMotionContext(
            channels=self.channels, dropout=self.dropout,
            message_layers=self.message_layers,
            position_scale_m=self.position_scale_m,
            history_scale_s=self.history_scale_s,
        )
        # The high-dimensional context terminates here.  It may estimate the
        # interpretable physical state, but no decoder reads it directly.
        self.motion_state_head = nn.Sequential(
            nn.LayerNorm(4 * self.channels),
            nn.Linear(4 * self.channels, 2 * self.channels), nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(2 * self.channels, self.channels), nn.SiLU(),
            nn.Linear(self.channels, 4), nn.Tanh(),
        )
        self.motion_state_encoder = nn.Sequential(
            nn.Linear(4, 2 * self.channels), nn.LayerNorm(2 * self.channels),
            nn.SiLU(), nn.Linear(2 * self.channels, 4 * self.channels), nn.SiLU(),
        )
        # Decoder inputs are motion code (4C) and one geometry code (C).
        self.trajectory_coefficient_head = nn.Sequential(
            nn.Linear(5 * self.channels, 2 * self.channels), nn.SiLU(),
            nn.Dropout(self.dropout), nn.Linear(2 * self.channels, self.channels),
            nn.SiLU(), nn.Linear(self.channels, self.basis_count * 3),
        )
        # Selector inputs are motion code, primary geometry and role geometry.
        self.role_coefficient_head = nn.Sequential(
            nn.Linear(6 * self.channels, 2 * self.channels),
            nn.LayerNorm(2 * self.channels), nn.SiLU(), nn.Dropout(self.dropout),
            nn.Linear(2 * self.channels, self.channels), nn.SiLU(),
            nn.Linear(self.channels, self.basis_count),
        )
        nn.init.zeros_(self.motion_state_head[-2].bias)
        nn.init.zeros_(self.trajectory_coefficient_head[-1].bias)
        nn.init.zeros_(self.role_coefficient_head[-1].bias)

    @property
    def config(self) -> dict[str, Any]:
        parent = dict(super().config)
        parent.update({
            "family": self.model_family,
            "motion_state_fields": list(MOTION_STATE_FIELDS),
            "motion_state_scale": self.motion_state_scale.detach().cpu().tolist(),
            "motion_state_supervision": "physical truth loss only",
            "decoder_temporal_input": "predicted 4D motion state only",
            "decoder_geometry_input": "q0 relation and supported mask only",
            "decoder_reads_high_dimensional_context": False,
            "selector_reads_high_dimensional_context": False,
            "future_gradient_to_motion_encoder": False,
            "teacher_forcing_motion_state": False,
            "physics_decoder": False,
            "truth_state_input": False,
        })
        return parent

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        detach_motion_code: bool = True,
        detach_selector_context: bool = False,
    ) -> dict[str, torch.Tensor]:
        missing = set(V3_FORWARD_FIELDS) - set(batch)
        if missing:
            raise ValueError(f"v5 future forward fields missing: {sorted(missing)}")
        history = self.context(
            batch["history_obs_rel_m"], batch["history_obs_mask"],
            batch["history_primary_mask"], batch["history_event_mask"],
            batch["history_time_s"], batch["history_dt_s"],
            batch["history_switch_step"], batch["q0_relation_m"],
            batch["q0_sigma_m"], batch["q0_confidence"], batch["q0_age_s"],
            batch["q0_support_class"], batch["q0_supported"],
        )
        motion_state_normalized = self.motion_state_head(history["vehicle_state"])
        decoder_motion = (
            motion_state_normalized.detach()
            if detach_motion_code else motion_state_normalized
        )
        current = batch["current_position_m"]
        if current.ndim != 2 or current.shape[1] != 3:
            raise ValueError("current position must have shape [B,3]")
        batch_size = current.shape[0]
        tau = self._tau(batch["tau_s"], batch_size)
        primary = history["primary_index"]
        relative_role = torch.arange(4, device=current.device)[None]
        ordered_handle = torch.remainder(primary[:, None] + relative_role, 4)
        q0_relation = batch["q0_relation_m"].gather(
            1, ordered_handle.unsqueeze(-1).expand(-1, -1, 3),
        ).clone()
        q0_relation[:, 0] = 0.0
        q0_supported = batch["q0_supported"].gather(1, ordered_handle)
        decoded = self.decode_ordered(
            current_position_m=current,
            tau_s=tau,
            ordered_q0_relation_m=q0_relation,
            ordered_q0_supported=q0_supported,
            motion_state_normalized=decoder_motion,
            detach_selector_context=detach_selector_context,
        )
        return {
            **history,
            "motion_state_normalized": motion_state_normalized,
            "motion_state_physical": (
                motion_state_normalized * self.motion_state_scale.to(
                    motion_state_normalized.dtype,
                )
            ),
            **decoded,
        }

    def decode_ordered(
        self,
        *,
        current_position_m: torch.Tensor,
        tau_s: torch.Tensor,
        ordered_q0_relation_m: torch.Tensor,
        ordered_q0_supported: torch.Tensor,
        motion_state_normalized: torch.Tensor,
        detach_selector_context: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Decode only an ordered q0 geometry, a 4D motion state and query time."""
        current = current_position_m
        if current.ndim != 2 or current.shape[1] != 3:
            raise ValueError("current position must have shape [B,3]")
        batch_size = current.shape[0]
        tau = self._tau(tau_s, batch_size)
        if ordered_q0_relation_m.shape != (batch_size, 4, 3):
            raise ValueError("ordered q0 relation must have shape [B,4,3]")
        if ordered_q0_supported.shape != (batch_size, 4):
            raise ValueError("ordered q0 supported must have shape [B,4]")
        if motion_state_normalized.shape != (batch_size, 4):
            raise ValueError("motion state must have shape [B,4]")
        q0_relation = ordered_q0_relation_m.clone()
        q0_relation[:, 0] = 0.0
        # GeometryOnlyHandleEncoder reads relation xyz and supported at index 4.
        # All quality/session fingerprint slots remain exact zero.
        geometry_feature = q0_relation.new_zeros((batch_size, 4, 13))
        geometry_feature[..., :3] = q0_relation / self.position_scale_m
        geometry_feature[..., 4] = ordered_q0_supported.to(q0_relation.dtype)
        geometry = self.handle_encoder(geometry_feature)
        motion_code = self.motion_state_encoder(motion_state_normalized)

        coefficient_feature = torch.cat((
            motion_code[:, None].expand(-1, 4, -1), geometry,
        ), dim=-1)
        coefficient = self.trajectory_coefficient_head(
            coefficient_feature,
        ).reshape(batch_size, 4, self.basis_count, 3)
        query_feature = self._query_feature(tau, self.trained_horizon_s)
        basis = self.time_basis(query_feature)
        dynamic = torch.einsum(
            "bqr,bhrc->bqhc", basis, coefficient,
        ) / math.sqrt(float(self.basis_count))
        # A stationary physical code has exactly zero learned dynamic motion;
        # q0 geometry alone cannot become a session-motion shortcut.
        motion_gate = torch.linalg.vector_norm(
            motion_state_normalized, dim=-1,
        ).clamp(max=1.0)[:, None, None, None]
        residual = torch.tanh(dynamic) * self.residual_scale_m * motion_gate
        tau_scale = (tau / self.trained_horizon_s)[:, :, None, None]
        role_delta = q0_relation[:, None] + tau_scale * residual
        role_position = current[:, None, None] + role_delta

        role_feature = torch.cat((
            motion_code[:, None].expand(-1, 4, -1),
            geometry[:, :1].expand(-1, 4, -1), geometry,
        ), dim=-1)
        selector_basis = basis.detach() if detach_selector_context else basis
        if detach_selector_context:
            role_feature = role_feature.detach()
        role_coefficient = self.role_coefficient_head(role_feature)
        role_logits = torch.einsum("bhr,bqr->bqh", role_coefficient, selector_basis)
        zero_query = tau == 0
        zero_role_logits = torch.full_like(role_logits, -torch.inf)
        zero_role_logits[..., 0] = 0.0
        role_logits = torch.where(zero_query.unsqueeze(-1), zero_role_logits, role_logits)
        role_probability = torch.softmax(role_logits.float(), dim=-1).to(
            motion_code.dtype,
        )
        selected_role = role_logits.argmax(dim=-1)
        selected_delta = role_delta.gather(
            2, selected_role[:, :, None, None].expand(-1, -1, 1, 3),
        ).squeeze(2)
        selected_position = current[:, None] + selected_delta
        return {
            "motion_code": motion_code,
            "time_basis": basis,
            "trajectory_coefficient": coefficient,
            "role_coefficient": role_coefficient,
            "role_logits": role_logits,
            "role_probability": role_probability,
            "selected_role": selected_role,
            "role_delta_m": role_delta,
            "role_position_m": role_position,
            "delta_m": selected_delta,
            "position_m": selected_position,
        }


def stable_motion_future_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    motion_weight: float = 0.0,
    trajectory_weight: float = 0.0,
    trend_weight: float = 0.0,
    role_weight: float = 0.0,
    distance_risk_weight: float = 0.0,
    huber_beta_m: float = 0.02,
    motion_huber_beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Separated physical-state and direct learned-future objectives."""
    target_motion = batch.get("target_motion_state_normalized")
    if target_motion is None or target_motion.shape != prediction["motion_state_normalized"].shape:
        raise ValueError("v5 loss requires a [B,4] normalized motion-state target")
    motion_coordinate = F.smooth_l1_loss(
        prediction["motion_state_normalized"], target_motion,
        reduction="none", beta=motion_huber_beta,
    )
    velocity = motion_coordinate[:, :3].mean(dim=-1).mean()
    yaw_rate = motion_coordinate[:, 3].mean()
    motion = velocity + yaw_rate

    mask = batch["target_query_mask"].to(torch.bool) & (batch["tau_s"] > 0)
    if not bool(mask.any()):
        raise ValueError("v5 future loss requires a positive-time query")
    role = target_roles(batch["target_switch_count"], mask)
    conditional = prediction["role_position_m"].gather(
        2, role[:, :, None, None].expand(-1, -1, 1, 3),
    ).squeeze(2)
    target = batch["truth_current_position_m"][:, None] + batch["target_visible_delta_m"]
    coordinate = F.smooth_l1_loss(
        conditional, target, reduction="none", beta=huber_beta_m,
    ).mean(dim=-1)
    trajectory = _masked_window_mean(coordinate, mask)
    per_window_trend: list[torch.Tensor] = []
    for sample in range(mask.shape[0]):
        indices = torch.nonzero(mask[sample], as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        order = indices[torch.argsort(batch["tau_s"][sample, indices])]
        per_window_trend.append(F.smooth_l1_loss(
            conditional[sample, order[1:]] - conditional[sample, order[:-1]],
            target[sample, order[1:]] - target[sample, order[:-1]],
            reduction="mean", beta=huber_beta_m,
        ))
    trend = torch.stack(per_window_trend).mean() if per_window_trend else trajectory * 0.0
    role_ce = F.cross_entropy(
        prediction["role_logits"].transpose(1, 2).float(), role, reduction="none",
    )
    role_loss = _masked_window_mean(role_ce, mask)
    role_distance = torch.linalg.vector_norm(
        prediction["role_position_m"].detach() - target[:, :, None], dim=-1,
    )
    distance_risk = _masked_window_mean(
        (prediction["role_probability"] * role_distance).sum(dim=-1), mask,
    )
    objective = (
        motion_weight * motion
        + trajectory_weight * trajectory
        + trend_weight * trend
        + role_weight * role_loss
        + distance_risk_weight * distance_risk
    )
    return objective, {
        "objective": objective,
        "motion": motion,
        "velocity": velocity,
        "yaw_rate": yaw_rate,
        "trajectory": trajectory,
        "trend": trend,
        "role": role_loss,
        "distance_risk": distance_risk,
    }


__all__ = [
    "MOTION_STATE_FIELDS",
    "StableMotionBottleneckAnonymousFutureModel",
    "stable_motion_future_loss",
]
