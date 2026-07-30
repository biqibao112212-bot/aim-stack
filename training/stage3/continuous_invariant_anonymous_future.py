"""Continuous, translation-equivariant anonymous future predictor.

This v3 model removes the latent expert/router interface used by v2.  A
history window produces one continuous trajectory field over four anonymous
cyclic roles.  Future truth is used only by the loss; physical IDs, session
identity, motion labels and absolute world position are not trajectory or
selector features.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .anonymous_vehicle_motion_v2 import (
    VisibilityAwareAnonymousVehicleFutureModel,
    VisibilityDrivenMotionContext,
    _masked_window_mean,
    target_roles,
)


V3_FORWARD_FIELDS = (
    "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
    "history_event_mask", "history_time_s", "history_dt_s",
    "history_switch_step", "q0_relation_m", "q0_sigma_m",
    "q0_confidence", "q0_age_s", "q0_support_class", "q0_supported",
    "current_position_m", "tau_s",
)


class ContinuousInvariantAnonymousFutureModel(
    VisibilityAwareAnonymousVehicleFutureModel
):
    """Learn a single continuous future field and a modulo-four role selector.

    ``current_position_m`` is deliberately consumed only at the final
    ``position = current + delta`` operation.  Consequently, translating the
    complete scene translates every predicted position by exactly the same
    amount while leaving trajectory coefficients, deltas and selector scores
    unchanged.
    """

    model_family = "continuous-invariant-anonymous-future-v3"

    def __init__(
        self,
        *,
        channels: int = 96,
        dropout: float = 0.05,
        message_layers: int = 3,
        trained_horizon_s: float = 0.55,
        maximum_absolute_step: int = 6,
        position_scale_m: float = 1.0,
        history_scale_s: float = 0.32,
        residual_scale_m: float = 2.0,
        basis_count: int = 8,
    ) -> None:
        # Initialize the mature causal/C4 helper implementation, then replace
        # every v2 head whose contract contained absolute position or experts.
        super().__init__(
            channels=channels,
            dropout=dropout,
            message_layers=message_layers,
            trained_horizon_s=trained_horizon_s,
            maximum_absolute_step=maximum_absolute_step,
            position_scale_m=position_scale_m,
            history_scale_s=history_scale_s,
            residual_scale_m=residual_scale_m,
            basis_count=basis_count,
            latent_experts=2,
        )
        self.context = VisibilityDrivenMotionContext(
            channels=channels,
            dropout=dropout,
            message_layers=message_layers,
            position_scale_m=position_scale_m,
            history_scale_s=history_scale_s,
        )
        self.trajectory_coefficient_head = nn.Sequential(
            nn.Linear(7 * channels, 2 * channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(2 * channels, channels), nn.SiLU(),
            nn.Linear(channels, basis_count * 3),
        )
        del self.motion_regime_gate
        del self.latent_experts
        self.role_coefficient_head = nn.Sequential(
            nn.Linear(8 * channels, 2 * channels), nn.LayerNorm(2 * channels),
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(2 * channels, channels),
            nn.SiLU(), nn.Linear(channels, basis_count),
        )
        del self.exact_selector_context
        del self.direction_score_head
        del self.crossing_interval_head
        del self.temperature_head
        nn.init.zeros_(self.trajectory_coefficient_head[-1].bias)
        nn.init.zeros_(self.role_coefficient_head[-1].bias)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "family": self.model_family,
            "channels": self.channels,
            "dropout": self.dropout,
            "message_layers": self.message_layers,
            "trained_horizon_s": self.trained_horizon_s,
            "maximum_absolute_step": self.maximum_absolute_step,
            "position_scale_m": self.position_scale_m,
            "history_scale_s": self.history_scale_s,
            "residual_scale_m": self.residual_scale_m,
            "basis_count": self.basis_count,
            "motion_context": self.context.config,
            "trajectory": (
                "one history-conditioned coefficient field plus shared "
                "learned continuous-time basis"
            ),
            "trajectory_experts": False,
            "future_best_expert_target": False,
            "translation_equivariant": True,
            "absolute_position_usage": "final current-plus-delta only",
            "selector_primary_target": "relative physical role modulo four",
            "selector_auxiliary_target": False,
            "physics_decoder": False,
            "physical_id_input": False,
            "motion_class_input": False,
            "session_identity_input": False,
            "truth_state_input": False,
            "future_pnp_input": False,
        }

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        detach_selector_context: bool = False,
    ) -> dict[str, torch.Tensor]:
        missing = set(V3_FORWARD_FIELDS) - set(batch)
        if missing:
            raise ValueError(f"v3 future forward fields missing: {sorted(missing)}")
        history = self.context(
            batch["history_obs_rel_m"], batch["history_obs_mask"],
            batch["history_primary_mask"], batch["history_event_mask"],
            batch["history_time_s"], batch["history_dt_s"],
            batch["history_switch_step"], batch["q0_relation_m"],
            batch["q0_sigma_m"], batch["q0_confidence"], batch["q0_age_s"],
            batch["q0_support_class"], batch["q0_supported"],
        )
        current = batch["current_position_m"]
        if current.ndim != 2 or current.shape[1] != 3:
            raise ValueError("current position must have shape [B,3]")
        batch_size = current.shape[0]
        tau = self._tau(batch["tau_s"], batch_size)

        primary = history["primary_index"]
        relative_role = torch.arange(4, device=current.device)[None]
        ordered_handle = torch.remainder(primary[:, None] + relative_role, 4)
        ordered_state = history["handle_state"].gather(
            1, ordered_handle.unsqueeze(-1).expand(-1, -1, 2 * self.channels),
        )
        q0_relation = batch["q0_relation_m"].gather(
            1, ordered_handle.unsqueeze(-1).expand(-1, -1, 3),
        ).clone()
        q0_relation[:, 0] = 0.0
        q0_sigma = batch["q0_sigma_m"].gather(
            1, ordered_handle.unsqueeze(-1).expand(-1, -1, 3),
        )
        q0_confidence = batch["q0_confidence"].gather(1, ordered_handle)
        q0_age = batch["q0_age_s"].gather(1, ordered_handle)
        q0_support = batch["q0_support_class"].to(torch.long).gather(
            1, ordered_handle,
        )
        q0_supported = batch["q0_supported"].gather(1, ordered_handle)
        handle_feature = torch.cat((
            q0_relation / self.position_scale_m,
            q0_confidence.unsqueeze(-1),
            q0_supported.to(q0_relation.dtype).unsqueeze(-1),
            (q0_age.clamp(0.0, 10.0 * self.history_scale_s)
             / self.history_scale_s).unsqueeze(-1),
            q0_sigma.clamp_min(0.0) / self.position_scale_m,
            F.one_hot(q0_support, num_classes=4).to(q0_relation.dtype),
        ), dim=-1)
        encoded_handle = self.handle_encoder(handle_feature)
        vehicle = history["vehicle_state"]

        coefficient_feature = torch.cat((
            vehicle[:, None].expand(-1, 4, -1), ordered_state, encoded_handle,
        ), dim=-1)
        coefficient = self.trajectory_coefficient_head(
            coefficient_feature,
        ).reshape(batch_size, 4, self.basis_count, 3)
        query_feature = self._query_feature(tau, self.trained_horizon_s)
        basis = self.time_basis(query_feature)
        dynamic = torch.einsum(
            "bqr,bhrc->bqhc", basis, coefficient,
        ) / math.sqrt(float(self.basis_count))
        residual = torch.tanh(dynamic) * self.residual_scale_m
        tau_scale = (tau / self.trained_horizon_s)[:, :, None, None]
        role_delta = q0_relation[:, None] + tau_scale * residual
        role_position = current[:, None, None] + role_delta

        role_feature = torch.cat((
            vehicle[:, None].expand(-1, 4, -1),
            ordered_state[:, :1].expand(-1, 4, -1),
            ordered_state,
        ), dim=-1)
        selector_basis = basis.detach() if detach_selector_context else basis
        if detach_selector_context:
            role_feature = role_feature.detach()
        role_coefficient = self.role_coefficient_head(role_feature)
        role_logits = torch.einsum("bhr,bqr->bqh", role_coefficient, selector_basis)
        zero_query = tau == 0
        zero_role_logits = torch.full_like(role_logits, -torch.inf)
        zero_role_logits[..., 0] = 0.0
        role_logits = torch.where(
            zero_query.unsqueeze(-1), zero_role_logits, role_logits,
        )
        role_probability = torch.softmax(role_logits.float(), dim=-1).to(
            vehicle.dtype,
        )
        selected_role = role_logits.argmax(dim=-1)

        selected_delta = role_delta.gather(
            2, selected_role[:, :, None, None].expand(-1, -1, 1, 3),
        ).squeeze(2)
        selected_position = current[:, None] + selected_delta
        return {
            **history,
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


def continuous_future_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    trajectory_weight: float = 1.0,
    trend_weight: float = 0.25,
    role_weight: float = 1.0,
    distance_risk_weight: float = 0.25,
    huber_beta_m: float = 0.02,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Direct future-position objective with no future-derived expert target."""
    mask = batch["target_query_mask"].to(torch.bool) & (batch["tau_s"] > 0)
    if not bool(mask.any()):
        raise ValueError("v3 future loss requires a positive-time query")
    role = target_roles(batch["target_switch_count"], mask)
    conditional = prediction["role_position_m"].gather(
        2, role[:, :, None, None].expand(-1, -1, 1, 3),
    ).squeeze(2)
    target = (
        batch["truth_current_position_m"][:, None]
        + batch["target_visible_delta_m"]
    )
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
    trend = (
        torch.stack(per_window_trend).mean()
        if per_window_trend else trajectory * 0.0
    )

    role_ce = F.cross_entropy(
        prediction["role_logits"].transpose(1, 2).float(), role,
        reduction="none",
    )
    role_loss = _masked_window_mean(role_ce, mask)
    role_distance = torch.linalg.vector_norm(
        prediction["role_position_m"].detach() - target[:, :, None], dim=-1,
    )
    distance_risk = _masked_window_mean(
        (prediction["role_probability"] * role_distance).sum(dim=-1), mask,
    )
    objective = (
        trajectory_weight * trajectory
        + trend_weight * trend
        + role_weight * role_loss
        + distance_risk_weight * distance_risk
    )
    return objective, {
        "objective": objective,
        "trajectory": trajectory,
        "trend": trend,
        "role": role_loss,
        "distance_risk": distance_risk,
    }
