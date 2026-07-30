"""Anonymous multi-handle motion context and future-visible-target heads.

The forward contract is causal and observation-only.  Four handles are
window-local cyclic memory locations, not persistent physical armor IDs.
Motion class, truth state, session identity and future PnP are deliberately
absent from every model input.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .observable_future_model import MaskedCausalResidualBlock


FORWARD_FIELDS = (
    "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
    "history_event_mask", "history_time_s", "history_dt_s",
    "history_switch_step", "q0_relation_m", "q0_sigma_m",
    "q0_confidence", "q0_age_s", "q0_support_class", "q0_supported",
    "current_position_m", "candidate_relation_m", "candidate_step",
    "candidate_mask", "candidate_confidence", "candidate_supported",
    "tau_s",
)


class SymmetricCyclicMessageBlock(nn.Module):
    """C4-equivariant and direction-reflection-consistent ring message."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(3 * channels, 2 * channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(2 * channels, channels),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or state.shape[1] != 4:
            raise ValueError("cyclic state must have shape [B,4,C]")
        previous = torch.roll(state, shifts=1, dims=1)
        following = torch.roll(state, shifts=-1, dims=1)
        neighbor_mean = 0.5 * (previous + following)
        neighbor_difference = torch.abs(previous - following)
        message = self.update(torch.cat((
            state, neighbor_mean, neighbor_difference,
        ), dim=-1))
        return self.norm(state + message)


class AnonymousVehicleMotionContext(nn.Module):
    """Encode four anonymous causal lanes into handle and vehicle latents."""

    model_family = "anonymous-vehicle-motion-context-v1"

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
            raise ValueError("invalid MotionContext capacity")
        if position_scale_m <= 0 or history_scale_s <= 0:
            raise ValueError("MotionContext scales must be positive")
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.message_layers = int(message_layers)
        self.position_scale_m = float(position_scale_m)
        self.history_scale_s = float(history_scale_s)
        # xyz, time, dt, primary, visible, signed event switch, cumulative
        # switch relative to q0, same-handle local velocity xyz, velocity-valid,
        # event-valid = 14 features.
        self.history_projection = nn.Sequential(
            nn.Linear(14, channels), nn.LayerNorm(channels), nn.SiLU(),
        )
        self.temporal = nn.ModuleList(
            MaskedCausalResidualBlock(channels, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16)
        )
        # q0 relation/sigma, confidence, age, four support classes, supported,
        # current-primary marker = 14 features.
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
            "handle_semantics": "window-local anonymous cyclic memory",
            "c4_equivariant": True,
            "direction_reflection_consistent": True,
            "physical_id_input": False,
            "motion_class_input": False,
            "truth_state_input": False,
            "future_pnp_input": False,
        }

    @staticmethod
    def _last_index(mask: torch.Tensor) -> torch.Tensor:
        indices = torch.arange(mask.shape[1], device=mask.device)[None]
        result = torch.where(
            mask, indices, torch.full_like(indices, -1),
        ).amax(dim=1)
        if bool(torch.any(result < 0)):
            raise ValueError("every MotionContext sample needs an active history")
        return result

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
        if history_obs_rel_m.ndim != 4 or history_obs_rel_m.shape[2:] != (4, 3):
            raise ValueError("history observations must have shape [B,T,4,3]")
        batch, events = history_obs_rel_m.shape[:2]
        if history_obs_mask.shape != (batch, events, 4):
            raise ValueError("history observation mask has the wrong shape")
        if history_primary_mask.shape != history_obs_mask.shape:
            raise ValueError("history primary mask has the wrong shape")
        scalar_shape = (batch, events)
        if any(value.shape != scalar_shape for value in (
            history_event_mask, history_time_s, history_dt_s,
            history_switch_step,
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
            raise ValueError("MotionContext requires at least eight active events")
        if bool(torch.any(visible.sum(dim=2)[active] < 1)):
            raise ValueError("active MotionContext events need an observation")
        if bool(torch.any(visible.sum(dim=2) > 2)):
            raise ValueError("at most two anonymous handles may be observed")
        if bool(torch.any(primary.sum(dim=2)[active] != 1)):
            raise ValueError("active events need exactly one primary handle")
        if bool(torch.any(primary & ~visible)):
            raise ValueError("primary handle must be observed")
        if bool(torch.any(active & (history_time_s > 1e-6))):
            raise ValueError("MotionContext history cannot contain future events")
        if bool(torch.any(active & ~torch.isfinite(history_time_s))):
            raise ValueError("active history time must be finite")
        if bool(torch.any(active & ~torch.isin(
            history_switch_step.to(torch.long),
            history_switch_step.new_tensor((-1, 0, 1), dtype=torch.long),
        ))):
            raise ValueError("history switch values must be -1, 0 or +1")
        if bool(torch.any(visible & ~torch.isfinite(history_obs_rel_m).all(dim=-1))):
            raise ValueError("visible history coordinates must be finite")

        last_event = self._last_index(active)
        last_time = history_time_s.gather(1, last_event[:, None]).squeeze(1)
        if bool(torch.any(torch.abs(last_time) > 1e-6)):
            raise ValueError("last active MotionContext event must be q0")
        current_primary = primary[
            torch.arange(batch, device=primary.device), last_event,
        ]
        if bool(torch.any(current_primary.sum(dim=1) != 1)):
            raise ValueError("q0 must have exactly one primary handle")

        clean_obs = torch.where(
            visible.unsqueeze(-1), history_obs_rel_m,
            torch.zeros_like(history_obs_rel_m),
        )
        clean_time = torch.where(active, history_time_s, torch.zeros_like(history_time_s))
        clean_dt = torch.where(active, history_dt_s, torch.zeros_like(history_dt_s))
        clean_switch = torch.where(
            active, history_switch_step, torch.zeros_like(history_switch_step),
        )
        cumulative = torch.cumsum(clean_switch, dim=1)
        cumulative_q0 = cumulative.gather(1, last_event[:, None])
        cumulative_relative = torch.where(
            active, cumulative - cumulative_q0, torch.zeros_like(cumulative),
        )

        local_velocity = torch.zeros_like(clean_obs)
        velocity_valid = torch.zeros_like(visible)
        pair_valid = (
            visible[:, 1:] & visible[:, :-1]
            & active[:, 1:, None] & active[:, :-1, None]
            & (clean_dt[:, 1:, None] > 0)
        )
        pair_velocity = (
            (clean_obs[:, 1:] - clean_obs[:, :-1])
            / clean_dt[:, 1:, None, None].clamp_min(1e-6)
        )
        local_velocity[:, 1:] = torch.where(
            pair_valid.unsqueeze(-1), pair_velocity,
            torch.zeros_like(pair_velocity),
        )
        velocity_valid[:, 1:] = pair_valid
        position_scale = self.position_scale_m
        time_scale = self.history_scale_s
        features = torch.cat((
            clean_obs / position_scale,
            (clean_time / time_scale)[:, :, None, None].expand(-1, -1, 4, 1),
            (clean_dt / time_scale)[:, :, None, None].expand(-1, -1, 4, 1),
            primary.to(clean_obs.dtype).unsqueeze(-1),
            visible.to(clean_obs.dtype).unsqueeze(-1),
            clean_switch.abs().to(clean_obs.dtype)[:, :, None, None].expand(-1, -1, 4, 1),
            (cumulative_relative.abs().to(clean_obs.dtype) / 6.0)[:, :, None, None].expand(-1, -1, 4, 1),
            local_velocity * (time_scale / position_scale),
            velocity_valid.to(clean_obs.dtype).unsqueeze(-1),
            active.to(clean_obs.dtype)[:, :, None, None].expand(-1, -1, 4, 1),
        ), dim=-1)
        sequence = self.history_projection(features)
        sequence = sequence.permute(0, 2, 3, 1).reshape(
            batch * 4, self.channels, events,
        )
        lane_mask = active[:, None].expand(-1, 4, -1).reshape(batch * 4, events)
        for block in self.temporal:
            sequence = block(sequence, lane_mask)
        gather = last_event[:, None].expand(-1, 4).reshape(-1)
        lane_last = sequence.gather(
            2, gather[:, None, None].expand(-1, self.channels, 1),
        ).squeeze(2).reshape(batch, 4, self.channels)
        lane_mean = sequence.sum(dim=2).reshape(batch, 4, self.channels)
        lane_mean = lane_mean / active.sum(dim=1).clamp_min(1)[:, None, None]
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
            "history_velocity_count": velocity_valid.sum(dim=(1, 2)),
        }


class AnonymousVehicleFutureModel(nn.Module):
    """Joint learned candidate trajectories and ordered visible-target choice."""

    model_family = "anonymous-vehicle-visible-future-v1"

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
    ) -> None:
        super().__init__()
        if trained_horizon_s <= 0 or maximum_absolute_step < 1:
            raise ValueError("invalid future horizon or candidate range")
        if residual_scale_m <= 0:
            raise ValueError("trajectory residual scale must be positive")
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.trained_horizon_s = float(trained_horizon_s)
        self.maximum_absolute_step = int(maximum_absolute_step)
        self.position_scale_m = float(position_scale_m)
        self.history_scale_s = float(history_scale_s)
        self.residual_scale_m = float(residual_scale_m)
        self.context = AnonymousVehicleMotionContext(
            channels=channels, dropout=dropout, message_layers=message_layers,
            position_scale_m=position_scale_m, history_scale_s=history_scale_s,
        )
        # relation, confidence, supported, age, sigma xyz, support class.  The
        # signed/revolution step is deliberately absent: rows k and k+4 refer
        # to the same anonymous q0 role and must share one physical trajectory.
        self.candidate_encoder = nn.Sequential(
            nn.Linear(13, channels), nn.LayerNorm(channels), nn.SiLU(),
            nn.Linear(channels, channels), nn.SiLU(),
        )
        query_features = 11
        trajectory_features = 7 * channels + 3 + query_features
        self.trajectory_head = nn.Sequential(
            nn.Linear(trajectory_features, 2 * channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(2 * channels, channels), nn.SiLU(),
            nn.Linear(channels, 3),
        )
        # One shared direction encoder is evaluated twice: primary->previous
        # and primary->following.  A handle reflection swaps these two rows,
        # giving exact direction-logit/interval equivariance by construction.
        selector_features = 12 * channels + 3
        self.selector_context = nn.Sequential(
            nn.Linear(selector_features, 2 * channels), nn.LayerNorm(2 * channels),
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(2 * channels, channels),
            nn.SiLU(),
        )
        self.direction_score_head = nn.Linear(channels, 1)
        self.crossing_interval_head = nn.Linear(channels, maximum_absolute_step)
        self.temperature_head = nn.Linear(channels, 1)
        nn.init.zeros_(self.trajectory_head[-1].bias)
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
            "trained_horizon_s": self.trained_horizon_s,
            "maximum_absolute_step": self.maximum_absolute_step,
            "position_scale_m": self.position_scale_m,
            "history_scale_s": self.history_scale_s,
            "residual_scale_m": self.residual_scale_m,
            "motion_context": self.context.config,
            "tau_zero_contract": "candidate q0 relation; selected step zero is exact identity",
            "selector": "sample-conditioned direction plus ordered crossing intervals",
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
        features = [
            normalized, normalized.square(), torch.sqrt(normalized.clamp_min(0.0)),
        ]
        for frequency in (1.0, 2.0, 4.0, 8.0):
            phase = 2.0 * torch.pi * frequency * normalized
            features.extend((torch.sin(phase), torch.cos(phase)))
        return torch.stack(features, dim=-1)

    def _candidate_probability(
        self,
        selector_state: torch.Tensor,
        candidate_step: torch.Tensor,
        candidate_mask: torch.Tensor,
        tau: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask = candidate_mask.to(torch.bool)
        rounded = candidate_step.to(torch.long)
        if bool(torch.any(mask & (rounded.abs() > self.maximum_absolute_step))):
            raise ValueError("candidate step exceeds configured range")
        required = 2 * self.maximum_absolute_step + 1
        if bool(torch.any(mask.sum(dim=1) != required)):
            raise ValueError("selector requires the complete signed candidate range")
        if selector_state.ndim != 3 or selector_state.shape[1] != 2:
            raise ValueError("selector state must have shape [B,2,C]")
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
        magnitude_index = rounded.abs()[:, None, None, :].expand(
            -1, tau.shape[1], 2, -1,
        )
        magnitude_mass = magnitude.gather(3, magnitude_index)
        negative_mass = (
            direction_probability[:, None, 0, None] * magnitude_mass[:, :, 0]
        )
        positive_mass = (
            direction_probability[:, None, 1, None] * magnitude_mass[:, :, 1]
        )
        zero_mass = (
            direction_probability[:, None, :, None] * magnitude_mass
        ).sum(dim=2)
        signed_mass = torch.where(
            rounded[:, None] < 0, negative_mass,
            torch.where(rounded[:, None] > 0, positive_mass, zero_mass),
        )
        probability = torch.where(
            mask[:, None], signed_mass, torch.zeros_like(signed_mass),
        )
        probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        logits = torch.log(probability.clamp_min(1e-12)).masked_fill(
            ~mask[:, None], -torch.inf,
        )
        zero = tau == 0
        current = mask & (rounded == 0)
        if bool(torch.any(current.sum(dim=1) != 1)):
            raise ValueError("candidate set needs exactly one step-zero row")
        zero_logits = torch.where(
            current[:, None], torch.zeros_like(logits),
            torch.full_like(logits, -torch.inf),
        )
        logits = torch.where(zero.unsqueeze(-1), zero_logits, logits)
        probability = torch.softmax(logits.float(), dim=-1).to(selector_state.dtype)
        selected_row = logits.argmax(dim=-1)
        return {
            "switch_logits": logits,
            "switch_probability": probability,
            "selected_candidate_row": selected_row,
            "selected_switch_step": rounded.gather(1, selected_row),
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
            raise ValueError(f"vehicle future forward fields missing: {sorted(missing)}")
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
        relation = batch["candidate_relation_m"]
        step = batch["candidate_step"].to(torch.long)
        mask = batch["candidate_mask"].to(torch.bool)
        if relation.ndim != 3 or relation.shape[0] != batch_size or relation.shape[2] != 3:
            raise ValueError("candidate relation must have shape [B,K,3]")
        if step.shape != relation.shape[:2] or mask.shape != step.shape:
            raise ValueError("candidate tensors disagree")
        # Every step congruent to zero addresses the current anonymous handle.
        # Its q0 relation is structurally the origin, including revolution rows
        # such as -4 and +4.  Do not rely on a composer rounding this exactly.
        current_role = torch.remainder(step, 4) == 0
        relation = torch.where(
            current_role.unsqueeze(-1), torch.zeros_like(relation), relation,
        )
        tau = self._tau(batch["tau_s"], batch_size)
        primary = history["primary_index"]
        handle = torch.remainder(primary[:, None] + step, 4)
        handle_state = history["handle_state"].gather(
            1, handle.unsqueeze(-1).expand(-1, -1, 2 * self.channels),
        )
        q0_sigma = batch["q0_sigma_m"].gather(
            1, handle.unsqueeze(-1).expand(-1, -1, 3),
        )
        q0_age = batch["q0_age_s"].gather(1, handle)
        q0_support = batch["q0_support_class"].to(torch.long).gather(1, handle)
        support_onehot = F.one_hot(q0_support, num_classes=4).to(relation.dtype)
        candidate_feature = torch.cat((
            relation / self.position_scale_m,
            batch["candidate_confidence"].unsqueeze(-1),
            batch["candidate_supported"].to(relation.dtype).unsqueeze(-1),
            (q0_age.clamp(0.0, 10.0 * self.history_scale_s) / self.history_scale_s).unsqueeze(-1),
            q0_sigma.clamp_min(0.0) / self.position_scale_m,
            support_onehot,
        ), dim=-1)
        candidate_state = self.candidate_encoder(candidate_feature)
        vehicle = history["vehicle_state"]
        query_feature = self._query_feature(tau, self.trained_horizon_s)
        query_count, candidate_count = tau.shape[1], relation.shape[1]
        trajectory_feature = torch.cat((
            vehicle[:, None, None].expand(-1, query_count, candidate_count, -1),
            handle_state[:, None].expand(-1, query_count, -1, -1),
            candidate_state[:, None].expand(-1, query_count, -1, -1),
            (current / self.position_scale_m)[:, None, None].expand(
                -1, query_count, candidate_count, -1,
            ),
            query_feature[:, :, None].expand(-1, -1, candidate_count, -1),
        ), dim=-1)
        residual = torch.tanh(self.trajectory_head(trajectory_feature)) * self.residual_scale_m
        tau_scale = (tau / self.trained_horizon_s)[:, :, None, None]
        conditional_delta = relation[:, None] + tau_scale * residual
        conditional_delta = torch.where(
            mask[:, None, :, None], conditional_delta,
            torch.zeros_like(conditional_delta),
        )
        conditional_position = current[:, None, None] + conditional_delta

        relative_handle = torch.remainder(
            primary[:, None] + torch.arange(4, device=primary.device)[None], 4,
        )
        ordered_handle = history["handle_state"].gather(
            1, relative_handle.unsqueeze(-1).expand(-1, -1, 2 * self.channels),
        )
        primary_state = ordered_handle[:, 0]
        opposite_state = ordered_handle[:, 2]
        following_state = ordered_handle[:, 1]
        previous_state = ordered_handle[:, 3]
        common_selector = (
            vehicle, primary_state, opposite_state,
            current / self.position_scale_m,
        )
        negative_feature = torch.cat((
            common_selector[0], common_selector[1], previous_state,
            common_selector[2], following_state, common_selector[3],
        ), dim=-1)
        positive_feature = torch.cat((
            common_selector[0], common_selector[1], following_state,
            common_selector[2], previous_state, common_selector[3],
        ), dim=-1)
        selector_feature = torch.stack((
            negative_feature, positive_feature,
        ), dim=1)
        if detach_selector_context:
            selector_feature = selector_feature.detach()
        selector_state = self.selector_context(selector_feature)
        selected = self._candidate_probability(selector_state, step, mask, tau)
        row = selected["selected_candidate_row"]
        selected_delta = conditional_delta.gather(
            2, row[:, :, None, None].expand(-1, -1, 1, 3),
        ).squeeze(2)
        result = {
            **history,
            **selected,
            "candidate_handle": handle,
            "conditional_delta_m": conditional_delta,
            "conditional_position_m": conditional_position,
            "delta_m": selected_delta,
            "position_m": current[:, None] + selected_delta,
        }
        return result


def target_candidate_rows(
    candidate_step: torch.Tensor,
    candidate_mask: torch.Tensor,
    target_switch_count: torch.Tensor,
    target_query_mask: torch.Tensor,
) -> torch.Tensor:
    match = (
        candidate_mask[:, None].to(torch.bool)
        & (candidate_step[:, None].to(torch.long) == target_switch_count[:, :, None].to(torch.long))
    )
    valid = target_query_mask.to(torch.bool)
    if bool(torch.any(valid & (match.sum(dim=-1) != 1))):
        raise ValueError("every valid target query needs one candidate row")
    return match.to(torch.long).argmax(dim=-1)


def anonymous_vehicle_future_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    trajectory_weight: float = 1.0,
    trend_weight: float = 0.25,
    switch_weight: float = 1.0,
    distance_risk_weight: float = 0.25,
    joint_position_weight: float = 0.1,
    huber_beta_m: float = 0.02,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # tau=0 is a structural identity and an upstream q0 diagnostic.  It is not
    # optimizable while Mapper/S/H are frozen, so learned losses use only
    # positive-time queries.
    mask = batch["target_query_mask"].to(torch.bool) & (batch["tau_s"] > 0)
    if not bool(mask.any()):
        raise ValueError("vehicle future loss requires a valid query")
    row = target_candidate_rows(
        batch["candidate_step"], batch["candidate_mask"],
        batch["target_switch_count"], mask,
    )
    conditional = prediction["conditional_position_m"].gather(
        2, row[:, :, None, None].expand(-1, -1, 1, 3),
    ).squeeze(2)
    target = (
        batch["truth_current_position_m"][:, None]
        + batch["target_visible_delta_m"]
    )
    per_coordinate = F.smooth_l1_loss(
        conditional, target, reduction="none", beta=huber_beta_m,
    ).mean(dim=-1)
    trajectory = per_coordinate[mask].mean()

    trend_terms: list[torch.Tensor] = []
    for sample in range(mask.shape[0]):
        indices = torch.nonzero(mask[sample], as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        order = indices[torch.argsort(batch["tau_s"][sample, indices])]
        predicted_difference = conditional[sample, order[1:]] - conditional[sample, order[:-1]]
        target_difference = target[sample, order[1:]] - target[sample, order[:-1]]
        trend_terms.append(F.smooth_l1_loss(
            predicted_difference, target_difference,
            reduction="mean", beta=huber_beta_m,
        ))
    trend = torch.stack(trend_terms).mean() if trend_terms else trajectory * 0.0

    logits = prediction["switch_logits"]
    switch = F.cross_entropy(logits[mask].float(), row[mask])
    target_expanded = target[:, :, None]
    candidate_distance = torch.linalg.vector_norm(
        prediction["conditional_position_m"].detach() - target_expanded,
        dim=-1,
    )
    distance_risk = (
        prediction["switch_probability"] * candidate_distance
    ).sum(dim=-1)[mask].mean()
    soft_position = (
        prediction["switch_probability"].unsqueeze(-1)
        * prediction["conditional_position_m"]
    ).sum(dim=2)
    joint_position = F.smooth_l1_loss(
        soft_position[mask], target[mask], reduction="mean", beta=huber_beta_m,
    )
    objective = (
        trajectory_weight * trajectory
        + trend_weight * trend
        + switch_weight * switch
        + distance_risk_weight * distance_risk
        + joint_position_weight * joint_position
    )
    return objective, {
        "objective": objective,
        "trajectory": trajectory,
        "trend": trend,
        "switch": switch,
        "distance_risk": distance_risk,
        "joint_position": joint_position,
    }
