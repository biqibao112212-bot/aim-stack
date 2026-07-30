"""Visibility-driven anonymous vehicle motion and future-role predictor.

Version two intentionally lives beside, rather than replaces, the immutable
v1 diagnostic model.  Four lanes are window-local anonymous cyclic handles.
Only observations where a handle is visible update that handle's temporal
stream; physical IDs, motion labels, truth state and future PnP are never
forward inputs.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .anonymous_vehicle_motion import FORWARD_FIELDS, SymmetricCyclicMessageBlock
from .observable_future_model import MaskedCausalResidualBlock


def _masked_window_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average queries within each window, then give every window equal mass."""
    if value.shape != mask.shape:
        raise ValueError("masked window mean inputs disagree")
    valid = mask.to(torch.bool)
    count = valid.sum(dim=1)
    keep = count > 0
    if not bool(keep.any()):
        raise ValueError("masked window mean needs at least one valid window")
    total = torch.where(valid, value, torch.zeros_like(value)).sum(dim=1)
    return (total[keep] / count[keep].to(value.dtype)).mean()


def target_roles(
    target_switch_count: torch.Tensor,
    target_query_mask: torch.Tensor,
) -> torch.Tensor:
    if target_switch_count.shape != target_query_mask.shape:
        raise ValueError("target switch and query masks disagree")
    return torch.remainder(target_switch_count.to(torch.long), 4)


class VisibilityDrivenMotionContext(nn.Module):
    """Encode four asynchronous anonymous handle streams.

    Events are compacted independently for every handle before the shared
    causal encoder.  Therefore inserting or modifying an event where a handle
    is invisible cannot advance that handle's temporal sequence.
    """

    model_family = "visibility-driven-anonymous-motion-context-v2"

    def __init__(
        self,
        *,
        channels: int = 96,
        dropout: float = 0.05,
        message_layers: int = 3,
        position_scale_m: float = 1.0,
        history_scale_s: float = 0.32,
    ) -> None:
        super().__init__()
        if channels < 32 or message_layers < 2:
            raise ValueError("invalid visibility-driven context capacity")
        if position_scale_m <= 0 or history_scale_s <= 0:
            raise ValueError("context scales must be positive")
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.message_layers = int(message_layers)
        self.position_scale_m = float(position_scale_m)
        self.history_scale_s = float(history_scale_s)
        # xyz, time-to-q0, elapsed since this handle was last visible,
        # primary, |event switch|, |cumulative switch to q0|, velocity xyz,
        # velocity-valid and first-visible markers = 13.
        self.history_projection = nn.Sequential(
            nn.Linear(13, channels), nn.LayerNorm(channels), nn.SiLU(),
        )
        self.temporal = nn.ModuleList(
            MaskedCausalResidualBlock(channels, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16)
        )
        # q0 relation/sigma, confidence, age, support onehot, supported,
        # current-primary marker = 14.
        self.q0_projection = nn.Sequential(
            nn.Linear(14, 2 * channels), nn.LayerNorm(2 * channels), nn.SiLU(),
        )
        self.messages = nn.ModuleList(
            SymmetricCyclicMessageBlock(2 * channels, dropout)
            for _ in range(message_layers)
        )

    @property
    def config(self) -> dict[str, Any]:
        return {
            "family": self.model_family,
            "channels": self.channels,
            "dropout": self.dropout,
            "message_layers": self.message_layers,
            "position_scale_m": self.position_scale_m,
            "history_scale_s": self.history_scale_s,
            "temporal_update": "per-handle visible events only",
            "elapsed_time": "previous visible event of the same handle",
            "handle_semantics": "window-local anonymous cyclic memory",
            "c4_equivariant": True,
            "direction_reflection_consistent": True,
            "physical_id_input": False,
            "motion_class_input": False,
            "truth_state_input": False,
            "future_pnp_input": False,
        }

    @staticmethod
    def _last_active(mask: torch.Tensor) -> torch.Tensor:
        index = torch.arange(mask.shape[1], device=mask.device)[None]
        last = torch.where(mask, index, torch.full_like(index, -1)).amax(dim=1)
        if bool(torch.any(last < 0)):
            raise ValueError("each context sample needs active history")
        return last

    @staticmethod
    def visible_differences(
        clean_obs: torch.Tensor,
        visible: torch.Tensor,
        history_time_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return elapsed time, local velocity and valid flag per handle."""
        batch, events, handles = visible.shape
        index = torch.arange(events, device=visible.device)[None, :, None]
        visible_index = torch.where(
            visible, index, torch.full_like(index, -1),
        )
        last_seen = visible_index.cummax(dim=1).values
        previous = torch.cat((
            torch.full_like(last_seen[:, :1], -1), last_seen[:, :-1],
        ), dim=1)
        previous_safe = previous.clamp_min(0).permute(0, 2, 1)
        obs_by_handle = clean_obs.permute(0, 2, 1, 3)
        previous_obs = obs_by_handle.gather(
            2, previous_safe.unsqueeze(-1).expand(-1, -1, -1, 3),
        ).permute(0, 2, 1, 3)
        time_by_handle = history_time_s[:, None].expand(-1, handles, -1)
        previous_time = time_by_handle.gather(2, previous_safe).permute(0, 2, 1)
        elapsed = history_time_s[:, :, None] - previous_time
        valid = visible & (previous >= 0) & (elapsed > 1e-7)
        elapsed = torch.where(valid, elapsed, torch.zeros_like(elapsed))
        velocity = torch.where(
            valid.unsqueeze(-1),
            (clean_obs - previous_obs) / elapsed.clamp_min(1e-7).unsqueeze(-1),
            torch.zeros_like(clean_obs),
        )
        return elapsed, velocity, valid

    @staticmethod
    def _compact_visible(
        feature: torch.Tensor,
        visible: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compact each handle's visible events without changing their order."""
        batch, events, handles, feature_count = feature.shape
        count = visible.sum(dim=1)
        rank = (visible.to(torch.long).cumsum(dim=1) - 1).clamp_min(0)
        compact = feature.new_zeros(batch, handles, events, feature_count)
        compact.scatter_add_(
            2,
            rank.permute(0, 2, 1).unsqueeze(-1).expand(
                -1, -1, -1, feature_count,
            ),
            (feature * visible.unsqueeze(-1)).permute(0, 2, 1, 3),
        )
        compact_mask = (
            torch.arange(events, device=feature.device)[None, None]
            < count[:, :, None]
        )
        return compact, compact_mask, count

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
        del history_dt_s  # Per-handle elapsed time is derived from timestamps.
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
        if q0_relation_m.shape != (batch, 4, 3) or q0_sigma_m.shape != (batch, 4, 3):
            raise ValueError("q0 relation/sigma must have shape [B,4,3]")
        if any(value.shape != (batch, 4) for value in (
            q0_confidence, q0_age_s, q0_support_class, q0_supported,
        )):
            raise ValueError("q0 scalar fields must have shape [B,4]")

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
        if bool(torch.any(current_primary.sum(dim=1) != 1)):
            raise ValueError("q0 must have exactly one primary handle")

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
        first_visible = visible & ~velocity_valid
        position_scale = self.position_scale_m
        time_scale = self.history_scale_s
        feature = torch.cat((
            clean_obs / position_scale,
            (clean_time / time_scale)[:, :, None, None].expand(-1, -1, 4, 1),
            (elapsed / time_scale).unsqueeze(-1),
            primary.to(clean_obs.dtype).unsqueeze(-1),
            clean_switch.abs().to(clean_obs.dtype)[:, :, None, None].expand(-1, -1, 4, 1),
            (cumulative_relative.abs().to(clean_obs.dtype) / 6.0)[:, :, None, None].expand(-1, -1, 4, 1),
            local_velocity * (time_scale / position_scale),
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
            (visible_count > 0).unsqueeze(-1), lane_last,
            torch.zeros_like(lane_last),
        )
        lane_mean = sequence.sum(dim=2).reshape(batch, 4, self.channels)
        lane_mean = lane_mean / visible_count.clamp_min(1).unsqueeze(-1)
        temporal_state = torch.cat((lane_last, lane_mean), dim=-1)

        support = q0_support_class.to(torch.long)
        if bool(torch.any((support < 0) | (support > 3))):
            raise ValueError("q0 support class must be within [0,3]")
        support_onehot = F.one_hot(support, num_classes=4).to(q0_relation_m.dtype)
        safe_age = torch.where(
            torch.isfinite(q0_age_s), q0_age_s,
            torch.full_like(q0_age_s, 10.0 * time_scale),
        ).clamp(0.0, 10.0 * time_scale)
        q0_feature = torch.cat((
            q0_relation_m / position_scale,
            q0_sigma_m.clamp_min(0.0) / position_scale,
            q0_confidence.unsqueeze(-1),
            (safe_age / time_scale).unsqueeze(-1),
            support_onehot,
            q0_supported.to(q0_relation_m.dtype).unsqueeze(-1),
            current_primary.to(q0_relation_m.dtype).unsqueeze(-1),
        ), dim=-1)
        handle_state = temporal_state + self.q0_projection(q0_feature)
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


class VisibilityAwareAnonymousVehicleFutureModel(nn.Module):
    """Predict four role trajectories and select a modulo-four physical role."""

    model_family = "visibility-aware-anonymous-vehicle-future-v2"

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
        latent_experts: int = 3,
    ) -> None:
        super().__init__()
        if trained_horizon_s <= 0 or maximum_absolute_step < 4:
            raise ValueError("invalid horizon or candidate range")
        if residual_scale_m <= 0 or basis_count < 4 or latent_experts < 2:
            raise ValueError("invalid continuous trajectory capacity")
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.message_layers = int(message_layers)
        self.trained_horizon_s = float(trained_horizon_s)
        self.maximum_absolute_step = int(maximum_absolute_step)
        self.position_scale_m = float(position_scale_m)
        self.history_scale_s = float(history_scale_s)
        self.residual_scale_m = float(residual_scale_m)
        self.basis_count = int(basis_count)
        self.latent_experts = int(latent_experts)
        self.context = VisibilityDrivenMotionContext(
            channels=channels, dropout=dropout, message_layers=message_layers,
            position_scale_m=position_scale_m, history_scale_s=history_scale_s,
        )
        self.handle_encoder = nn.Sequential(
            nn.Linear(13, channels), nn.LayerNorm(channels), nn.SiLU(),
            nn.Linear(channels, channels), nn.SiLU(),
        )
        self.time_basis = nn.Sequential(
            nn.Linear(11, channels), nn.SiLU(),
            nn.Linear(channels, basis_count), nn.Tanh(),
        )
        trajectory_features = 7 * channels + 3
        self.trajectory_coefficient_head = nn.Sequential(
            nn.Linear(trajectory_features, 2 * channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(2 * channels, channels), nn.SiLU(),
            nn.Linear(channels, latent_experts * basis_count * 3),
        )
        self.motion_regime_gate = nn.Sequential(
            nn.Linear(4 * channels, channels), nn.SiLU(),
            nn.Linear(channels, latent_experts),
        )
        # A shared role scorer is evaluated on primary/following/opposite/
        # previous states. Reflection therefore swaps role 1 and 3 exactly.
        self.role_coefficient_head = nn.Sequential(
            nn.Linear(8 * channels + 3, 2 * channels), nn.LayerNorm(2 * channels),
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(2 * channels, channels),
            nn.SiLU(), nn.Linear(channels, basis_count),
        )
        # Auxiliary ordered signed-crossing model. Its probability is
        # normalized within each modulo-four role and cannot change final XYZ.
        exact_features = 12 * channels + 3
        self.exact_selector_context = nn.Sequential(
            nn.Linear(exact_features, 2 * channels), nn.LayerNorm(2 * channels),
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(2 * channels, channels),
            nn.SiLU(),
        )
        self.direction_score_head = nn.Linear(channels, 1)
        self.crossing_interval_head = nn.Linear(channels, maximum_absolute_step)
        self.temperature_head = nn.Linear(channels, 1)
        nn.init.zeros_(self.trajectory_coefficient_head[-1].bias)
        nn.init.zeros_(self.role_coefficient_head[-1].bias)
        nn.init.zeros_(self.direction_score_head.bias)
        nn.init.zeros_(self.crossing_interval_head.weight)
        nn.init.constant_(self.crossing_interval_head.bias, -1.5)
        nn.init.zeros_(self.temperature_head.weight)
        nn.init.zeros_(self.temperature_head.bias)

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
            "latent_experts": self.latent_experts,
            "motion_context": self.context.config,
            "trajectory": "tau-independent coefficients plus shared learned time basis",
            "motion_regime": "history-inferred latent mixture",
            "selector_primary_target": "relative physical role modulo four",
            "selector_auxiliary_target": "exact signed crossing count",
            "tau_zero_contract": "all handles at q0; selected role zero",
            "physics_decoder": False,
            "physical_id_input": False,
            "motion_class_input": False,
            "future_pnp_input": False,
        }

    def _tau(self, tau_s: torch.Tensor, batch: int) -> torch.Tensor:
        if tau_s.ndim == 1:
            tau_s = tau_s[None].expand(batch, -1)
        if tau_s.ndim != 2 or tau_s.shape[0] != batch:
            raise ValueError("tau must have shape [Q] or [B,Q]")
        if bool(torch.any(~torch.isfinite(tau_s))) or bool(torch.any(tau_s < 0)):
            raise ValueError("tau must be finite and nonnegative")
        if bool(torch.any(tau_s > self.trained_horizon_s + 1e-6)):
            raise ValueError("tau exceeds the trained horizon")
        return tau_s

    @staticmethod
    def _query_feature(tau: torch.Tensor, horizon: float) -> torch.Tensor:
        normalized = tau / horizon
        feature = [
            normalized, normalized.square(), torch.sqrt(normalized.clamp_min(0.0)),
        ]
        for frequency in (1.0, 2.0, 4.0, 8.0):
            phase = 2.0 * torch.pi * frequency * normalized
            feature.extend((torch.sin(phase), torch.cos(phase)))
        return torch.stack(feature, dim=-1)

    def _validate_candidates(
        self, candidate_step: torch.Tensor, candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = candidate_mask.to(torch.bool)
        if candidate_step.shape != mask.shape:
            raise ValueError("candidate step/mask disagree")
        if candidate_step.dtype.is_floating_point and bool(torch.any(
            mask & (candidate_step != candidate_step.round())
        )):
            raise ValueError("candidate steps must be exact integers")
        step = candidate_step.to(torch.long)
        expected = torch.arange(
            -self.maximum_absolute_step, self.maximum_absolute_step + 1,
            device=step.device,
        )
        if bool(torch.any(mask.sum(dim=1) != expected.numel())):
            raise ValueError("selector requires the complete signed candidate range")
        ordered = torch.sort(torch.where(mask, step, step.new_full((), 10_000)), dim=1).values
        if ordered.shape[1] != expected.numel() or bool(torch.any(ordered != expected[None])):
            raise ValueError("candidate steps must be the unique complete signed range")
        return step

    @staticmethod
    def _role_rows(step: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        role = torch.remainder(step, 4)
        match = mask[:, None] & (
            role[:, None] == torch.arange(4, device=step.device)[None, :, None]
        )
        if bool(torch.any(match.sum(dim=-1) < 1)):
            raise ValueError("candidate range must contain every modulo-four role")
        return match.to(torch.long).argmax(dim=-1)

    def _raw_exact_probability(
        self,
        selector_state: torch.Tensor,
        step: torch.Tensor,
        mask: torch.Tensor,
        tau: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        direction_logits = self.direction_score_head(selector_state).squeeze(-1)
        direction_probability = torch.softmax(direction_logits.float(), dim=-1).to(
            selector_state.dtype
        )
        minimum_gap = self.trained_horizon_s * 1e-4
        crossing_interval = minimum_gap + self.trained_horizon_s * F.softplus(
            self.crossing_interval_head(selector_state)
        )
        crossing_time = crossing_interval.cumsum(dim=-1)
        temperature = (
            0.01 * self.trained_horizon_s
            + 0.09 * self.trained_horizon_s
            * torch.sigmoid(self.temperature_head(selector_state).squeeze(-1))
        )
        cumulative = torch.sigmoid((
            tau[:, :, None, None] - crossing_time[:, None]
        ).float() / temperature[:, None, :, None].float())
        magnitude = torch.cat((
            1.0 - cumulative[..., :1],
            cumulative[..., :-1] - cumulative[..., 1:],
            cumulative[..., -1:],
        ), dim=-1).clamp_min(0.0)
        magnitude = magnitude / magnitude.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        magnitude_index = step.abs()[:, None, None].expand(-1, tau.shape[1], 2, -1)
        magnitude_mass = magnitude.gather(3, magnitude_index)
        negative = direction_probability[:, None, 0, None] * magnitude_mass[:, :, 0]
        positive = direction_probability[:, None, 1, None] * magnitude_mass[:, :, 1]
        zero_mass = (direction_probability[:, None, :, None] * magnitude_mass).sum(dim=2)
        mass = torch.where(
            step[:, None] < 0, negative,
            torch.where(step[:, None] > 0, positive, zero_mass),
        )
        mass = torch.where(mask[:, None], mass, torch.zeros_like(mass))
        mass = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        zero_query = tau == 0
        zero_row = mask & (step == 0)
        exact_zero = torch.where(
            zero_row[:, None], torch.ones_like(mass), torch.zeros_like(mass),
        )
        mass = torch.where(zero_query.unsqueeze(-1), exact_zero, mass)
        return {
            "raw_crossing_probability": mass,
            "direction_logits": direction_logits,
            "direction_probability": direction_probability,
            "crossing_interval_s": crossing_interval,
            "crossing_time_s": crossing_time,
            "crossing_temperature_s": temperature,
        }

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        detach_selector_context: bool = False,
    ) -> dict[str, torch.Tensor]:
        missing = set(FORWARD_FIELDS) - set(batch)
        if missing:
            raise ValueError(f"v2 future forward fields missing: {sorted(missing)}")
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
        mask = batch["candidate_mask"].to(torch.bool)
        step = self._validate_candidates(batch["candidate_step"], mask)
        if batch["candidate_relation_m"].shape != (*step.shape, 3):
            raise ValueError("candidate relation has the wrong shape")
        if any(batch[name].shape != step.shape for name in (
            "candidate_confidence", "candidate_supported",
        )):
            raise ValueError("candidate metadata has the wrong shape")
        tau = self._tau(batch["tau_s"], batch_size)
        primary = history["primary_index"]
        relative_role = torch.arange(4, device=current.device)[None]
        ordered_handle = torch.remainder(primary[:, None] + relative_role, 4)
        ordered_state = history["handle_state"].gather(
            1, ordered_handle.unsqueeze(-1).expand(-1, -1, 2 * self.channels),
        )
        q0_relation = batch["q0_relation_m"].gather(
            1, ordered_handle.unsqueeze(-1).expand(-1, -1, 3),
        )
        q0_relation[:, 0] = 0.0
        q0_sigma = batch["q0_sigma_m"].gather(
            1, ordered_handle.unsqueeze(-1).expand(-1, -1, 3),
        )
        q0_confidence = batch["q0_confidence"].gather(1, ordered_handle)
        q0_age = batch["q0_age_s"].gather(1, ordered_handle)
        q0_support = batch["q0_support_class"].to(torch.long).gather(1, ordered_handle)
        q0_supported = batch["q0_supported"].gather(1, ordered_handle)
        handle_feature = torch.cat((
            q0_relation / self.position_scale_m,
            q0_confidence.unsqueeze(-1),
            q0_supported.to(q0_relation.dtype).unsqueeze(-1),
            (q0_age.clamp(0.0, 10.0 * self.history_scale_s) / self.history_scale_s).unsqueeze(-1),
            q0_sigma.clamp_min(0.0) / self.position_scale_m,
            F.one_hot(q0_support, num_classes=4).to(q0_relation.dtype),
        ), dim=-1)
        encoded_handle = self.handle_encoder(handle_feature)
        vehicle = history["vehicle_state"]
        coefficient_feature = torch.cat((
            vehicle[:, None].expand(-1, 4, -1), ordered_state,
            encoded_handle,
            (current / self.position_scale_m)[:, None].expand(-1, 4, -1),
        ), dim=-1)
        coefficient = self.trajectory_coefficient_head(coefficient_feature).reshape(
            batch_size, 4, self.latent_experts, self.basis_count, 3,
        )
        regime_logits = self.motion_regime_gate(vehicle)
        regime_probability = torch.softmax(regime_logits.float(), dim=-1).to(vehicle.dtype)
        query_feature = self._query_feature(tau, self.trained_horizon_s)
        basis = self.time_basis(query_feature)
        expert_dynamic = torch.einsum("bqr,bherc->bqhec", basis, coefficient)
        mixed_dynamic = torch.einsum(
            "be,bqhec->bqhc", regime_probability, expert_dynamic,
        ) / math.sqrt(float(self.basis_count))
        residual = torch.tanh(mixed_dynamic) * self.residual_scale_m
        tau_scale = (tau / self.trained_horizon_s)[:, :, None, None]
        role_delta = q0_relation[:, None] + tau_scale * residual
        role_position = current[:, None, None] + role_delta

        role_feature = torch.cat((
            vehicle[:, None].expand(-1, 4, -1),
            ordered_state[:, :1].expand(-1, 4, -1),
            ordered_state,
            (current / self.position_scale_m)[:, None].expand(-1, 4, -1),
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
        role_probability = torch.softmax(role_logits.float(), dim=-1).to(vehicle.dtype)
        selected_role = role_logits.argmax(dim=-1)

        primary_state = ordered_state[:, 0]
        following_state = ordered_state[:, 1]
        opposite_state = ordered_state[:, 2]
        previous_state = ordered_state[:, 3]
        common = (vehicle, primary_state, opposite_state, current / self.position_scale_m)
        negative_feature = torch.cat((
            common[0], common[1], previous_state, common[2], following_state, common[3],
        ), dim=-1)
        positive_feature = torch.cat((
            common[0], common[1], following_state, common[2], previous_state, common[3],
        ), dim=-1)
        exact_feature = torch.stack((negative_feature, positive_feature), dim=1)
        if detach_selector_context:
            exact_feature = exact_feature.detach()
        exact_state = self.exact_selector_context(exact_feature)
        exact = self._raw_exact_probability(exact_state, step, mask, tau)
        raw_probability = exact["raw_crossing_probability"]
        candidate_role = torch.remainder(step, 4)
        raw_role_mass = raw_probability.new_zeros(batch_size, tau.shape[1], 4)
        raw_role_mass.scatter_add_(
            2, candidate_role[:, None].expand(-1, tau.shape[1], -1), raw_probability,
        )
        within_role = raw_probability / raw_role_mass.gather(
            2, candidate_role[:, None].expand(-1, tau.shape[1], -1),
        ).clamp_min(1e-12)
        switch_probability = within_role * role_probability.gather(
            2, candidate_role[:, None].expand(-1, tau.shape[1], -1),
        )
        switch_probability = torch.where(
            mask[:, None], switch_probability, torch.zeros_like(switch_probability),
        )
        switch_probability = switch_probability / switch_probability.sum(
            dim=-1, keepdim=True,
        ).clamp_min(1e-12)
        switch_logits = torch.log(switch_probability.clamp_min(1e-12)).masked_fill(
            ~mask[:, None], -torch.inf,
        )
        selected_exact_row = switch_logits.argmax(dim=-1)
        selected_switch_step = step.gather(1, selected_exact_row)

        candidate_position = role_position.gather(
            2,
            candidate_role[:, None, :, None].expand(-1, tau.shape[1], -1, 3),
        )
        candidate_delta = candidate_position - current[:, None, None]
        selected_position = role_position.gather(
            2, selected_role[:, :, None, None].expand(-1, -1, 1, 3),
        ).squeeze(2)
        role_rows = self._role_rows(step, mask)
        result = {
            **history,
            **exact,
            "time_basis": basis,
            "trajectory_coefficient": coefficient,
            "motion_regime_logits": regime_logits,
            "motion_regime_probability": regime_probability,
            "role_coefficient": role_coefficient,
            "role_logits": role_logits,
            "role_probability": role_probability,
            "selected_role": selected_role,
            "candidate_role": candidate_role,
            "candidate_handle": torch.remainder(primary[:, None] + candidate_role, 4),
            "role_candidate_row": role_rows,
            "role_delta_m": role_delta,
            "role_position_m": role_position,
            "conditional_delta_m": candidate_delta,
            "conditional_position_m": candidate_position,
            "switch_logits": switch_logits,
            "switch_probability": switch_probability,
            "selected_candidate_row_aux": selected_exact_row,
            "selected_switch_step_aux": selected_switch_step,
            "delta_m": selected_position - current[:, None],
            "position_m": selected_position,
        }
        return result


def visibility_aware_future_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    trajectory_weight: float = 1.0,
    trend_weight: float = 0.25,
    role_weight: float = 1.0,
    exact_crossing_weight: float = 0.15,
    distance_risk_weight: float = 0.25,
    joint_position_weight: float = 0.0,
    regime_balance_weight: float = 0.01,
    regime_entropy_weight: float = 0.002,
    huber_beta_m: float = 0.02,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = batch["target_query_mask"].to(torch.bool) & (batch["tau_s"] > 0)
    if not bool(mask.any()):
        raise ValueError("v2 future loss requires a positive-time query")
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
        predicted_difference = conditional[sample, order[1:]] - conditional[sample, order[:-1]]
        target_difference = target[sample, order[1:]] - target[sample, order[:-1]]
        per_window_trend.append(F.smooth_l1_loss(
            predicted_difference, target_difference,
            reduction="mean", beta=huber_beta_m,
        ))
    trend = torch.stack(per_window_trend).mean() if per_window_trend else trajectory * 0.0

    role_ce = F.cross_entropy(
        prediction["role_logits"].transpose(1, 2).float(), role,
        reduction="none",
    )
    role_loss = _masked_window_mean(role_ce, mask)
    exact_row = (
        batch["candidate_step"][:, None].to(torch.long)
        == batch["target_switch_count"][:, :, None].to(torch.long)
    ) & batch["candidate_mask"][:, None].to(torch.bool)
    if bool(torch.any(mask & (exact_row.sum(dim=-1) != 1))):
        raise ValueError("each valid target needs one exact candidate")
    exact_target = exact_row.to(torch.long).argmax(dim=-1)
    exact_ce = F.cross_entropy(
        prediction["switch_logits"].transpose(1, 2).float(), exact_target,
        reduction="none",
    )
    exact_crossing = _masked_window_mean(exact_ce, mask)

    role_distance = torch.linalg.vector_norm(
        prediction["role_position_m"].detach() - target[:, :, None], dim=-1,
    )
    distance_risk = _masked_window_mean(
        (prediction["role_probability"] * role_distance).sum(dim=-1), mask,
    )
    soft_position = (
        prediction["role_probability"].unsqueeze(-1)
        * prediction["role_position_m"].detach()
    ).sum(dim=2)
    joint_coordinate = F.smooth_l1_loss(
        soft_position, target, reduction="none", beta=huber_beta_m,
    ).mean(dim=-1)
    joint_position = _masked_window_mean(joint_coordinate, mask)

    regime = prediction["motion_regime_probability"].float()
    mean_regime = regime.mean(dim=0)
    uniform = torch.full_like(mean_regime, 1.0 / mean_regime.numel())
    regime_balance = (mean_regime - uniform).square().mean()
    regime_entropy = -(regime.clamp_min(1e-12).log() * regime).sum(dim=-1).mean()
    objective = (
        trajectory_weight * trajectory
        + trend_weight * trend
        + role_weight * role_loss
        + exact_crossing_weight * exact_crossing
        + distance_risk_weight * distance_risk
        + joint_position_weight * joint_position
        + regime_balance_weight * regime_balance
        + regime_entropy_weight * regime_entropy
    )
    return objective, {
        "objective": objective,
        "trajectory": trajectory,
        "trend": trend,
        "role": role_loss,
        "exact_crossing": exact_crossing,
        "distance_risk": distance_risk,
        "joint_position": joint_position,
        "regime_balance": regime_balance,
        "regime_entropy": regime_entropy,
    }
