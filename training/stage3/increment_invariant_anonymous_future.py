"""Increment-only, time-scaled anonymous future predictor.

The temporal branch never receives a raw historical position.  It observes
only same-handle causal increments and their real elapsed times.  Relative q0
geometry is injected once through a separate branch, which prevents historical
range/pose fingerprints from becoming a shortcut for a capture session.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .anonymous_vehicle_motion import SymmetricCyclicMessageBlock
from .anonymous_vehicle_motion_v2 import VisibilityDrivenMotionContext
from .continuous_invariant_anonymous_future import (
    ContinuousInvariantAnonymousFutureModel,
    continuous_future_loss,
)
from .observable_future_model import MaskedCausalResidualBlock


class GeometryOnlyHandleEncoder(nn.Module):
    """Encode q0 relation/support while ignoring all quality fingerprints."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, channels), nn.LayerNorm(channels), nn.SiLU(),
            nn.Linear(channels, channels), nn.SiLU(),
        )

    def forward(self, legacy_feature: torch.Tensor) -> torch.Tensor:
        if legacy_feature.shape[-1] != 13:
            raise ValueError("handle feature must retain the 13-field boundary")
        # V3 boundary layout: relation xyz, confidence, supported, age,
        # sigma xyz, support one-hot.  Only relation and supported are causal
        # motion-law inputs in v4.
        feature = torch.cat((legacy_feature[..., :3], legacy_feature[..., 4:5]), dim=-1)
        return self.encoder(feature)


class IncrementOnlyMotionContext(VisibilityDrivenMotionContext):
    """Encode anonymous handle streams without raw historical coordinates."""

    model_family = "increment-only-anonymous-motion-context-v4"

    def __init__(
        self,
        *,
        channels: int = 96,
        dropout: float = 0.05,
        message_layers: int = 3,
        position_scale_m: float = 1.0,
        history_scale_s: float = 0.32,
    ) -> None:
        super().__init__(
            channels=channels, dropout=dropout, message_layers=message_layers,
            position_scale_m=position_scale_m, history_scale_s=history_scale_s,
        )
        # displacement xyz, elapsed, velocity xyz, time-to-q0, primary,
        # |switch|, |cumulative switch|, velocity-valid, first-visible and
        # first-visible offset to this handle's q0 relation xyz = 16.
        self.history_projection = nn.Sequential(
            nn.Linear(16, channels), nn.LayerNorm(channels), nn.SiLU(),
        )
        # q0 relation xyz, supported, current-primary and normalized visible
        # count.  Confidence/sigma/support class/age are deliberately excluded.
        self.q0_projection = nn.Sequential(
            nn.Linear(6, 2 * channels), nn.LayerNorm(2 * channels), nn.SiLU(),
        )
        self.temporal = nn.ModuleList(
            MaskedCausalResidualBlock(channels, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16)
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
            "temporal_features": (
                "same-handle displacement/dt/velocity plus first-to-q0 offset and masks"
            ),
            "raw_history_position_feature": False,
            "q0_geometry_injection": "relation/support only, separate from temporal",
            "q0_quality_features": False,
            "c4_equivariant": True,
            "direction_reflection_consistent": True,
            "physical_id_input": False,
            "motion_class_input": False,
            "session_identity_input": False,
            "truth_state_input": False,
            "future_pnp_input": False,
        }

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
        del history_dt_s
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
        displacement = torch.where(
            velocity_valid.unsqueeze(-1), local_velocity * elapsed.unsqueeze(-1),
            torch.zeros_like(local_velocity),
        )
        first_visible = visible & ~velocity_valid
        first_offset_to_q0 = torch.where(
            first_visible.unsqueeze(-1),
            clean_obs - q0_relation_m[:, None],
            torch.zeros_like(clean_obs),
        )
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
            first_offset_to_q0 / position_scale,
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

        q0_feature = torch.cat((
            q0_relation_m / position_scale,
            q0_supported.to(q0_relation_m.dtype).unsqueeze(-1),
            current_primary.to(q0_relation_m.dtype).unsqueeze(-1),
            (visible_count.to(q0_relation_m.dtype) / max(float(events), 1.0)).unsqueeze(-1),
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


class IncrementInvariantAnonymousFutureModel(
    ContinuousInvariantAnonymousFutureModel
):
    """V4 continuous F with increment-only temporal and geometry-only q0 paths."""

    model_family = "increment-invariant-anonymous-future-v4"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.context = IncrementOnlyMotionContext(
            channels=self.channels, dropout=self.dropout,
            message_layers=self.message_layers,
            position_scale_m=self.position_scale_m,
            history_scale_s=self.history_scale_s,
        )
        self.handle_encoder = GeometryOnlyHandleEncoder(self.channels)

    @property
    def config(self) -> dict[str, Any]:
        parent = dict(super().config)
        parent.update({
            "family": self.model_family,
            "motion_context": self.context.config,
            "raw_history_position_feature": False,
            "q0_quality_features": False,
            "time_scaling_augmentation": "trainer-owned synchronized causal time",
        })
        return parent


__all__ = [
    "GeometryOnlyHandleEncoder",
    "IncrementOnlyMotionContext",
    "IncrementInvariantAnonymousFutureModel",
    "continuous_future_loss",
]
