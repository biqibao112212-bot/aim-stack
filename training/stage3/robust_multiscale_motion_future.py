"""Robust multi-scale causal state estimator for the learned future field.

The estimator replaces the v5 adjacent-difference ``last/mean/max`` path.  It
constructs causal same-handle displacement edges at several physical time
scales and an unordered two-visible-armor relative-motion stream.  Learned
bounded reweighting aggregates noisy edges before an availability-aware scale
fusion emits the same four-dimensional physical-motion bottleneck used by the
existing learned decoder and selector.

No absolute position, range, session identity, motion class, physical armor ID,
future observation or truth label is accepted by the state estimator.  Truth
is used only by :func:`robust_multiscale_motion_future_loss`.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .anonymous_vehicle_motion import SymmetricCyclicMessageBlock
from .anonymous_vehicle_motion_v2 import VisibilityDrivenMotionContext
from .continuous_invariant_anonymous_future import V3_FORWARD_FIELDS
from .stable_motion_bottleneck_future import (
    StableMotionBottleneckAnonymousFutureModel,
    stable_motion_future_loss,
)


class _RobustEdgeConsensus(nn.Module):
    """Two-pass bounded neural M-estimator over causal edge tokens."""

    def __init__(self, input_features: int, channels: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_features, channels), nn.LayerNorm(channels), nn.SiLU(),
            nn.Linear(channels, channels), nn.SiLU(),
        )
        self.reliability = nn.Sequential(
            nn.Linear(3 * channels + 2, channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(channels, 1),
        )

    def forward(
        self,
        token: torch.Tensor,
        valid: torch.Tensor,
        scale_embedding: torch.Tensor,
        elapsed_s: torch.Tensor,
        scale_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # token [N,T,K,F], valid/dt [N,T,K], embedding [K,C].
        latent = self.projection(token) + scale_embedding[None, None]
        valid_f = valid.to(latent.dtype)
        support = valid_f.sum(dim=1)
        provisional = (latent * valid_f.unsqueeze(-1)).sum(dim=1)
        provisional = provisional / support.clamp_min(1).unsqueeze(-1)
        difference = (latent - provisional[:, None]).abs()
        ratio = elapsed_s / scale_s[None, None].clamp_min(1e-6)
        quality = torch.stack((
            torch.log1p(elapsed_s.clamp_min(0.0) / 0.01),
            torch.log(ratio.clamp_min(1e-6)).abs(),
        ), dim=-1)
        gate_feature = torch.cat((
            latent,
            provisional[:, None].expand_as(latent),
            difference,
            quality,
        ), dim=-1)
        # A positive floor prevents one learned score from erasing all causal
        # evidence at a scale; normalization below still reflects reliability.
        raw_weight = 0.05 + 0.95 * torch.sigmoid(
            self.reliability(gate_feature).squeeze(-1)
        )
        weight = raw_weight * valid_f
        denominator = weight.sum(dim=1).clamp_min(1e-6)
        consensus = (latent * weight.unsqueeze(-1)).sum(dim=1)
        consensus = consensus / denominator.unsqueeze(-1)
        dispersion = (
            ((latent - consensus[:, None]).square() * weight.unsqueeze(-1)).sum(dim=1)
            / denominator.unsqueeze(-1)
        ).clamp_min(1e-8).sqrt()
        available = support > 0
        state = torch.cat((consensus, dispersion), dim=-1)
        state = torch.where(available.unsqueeze(-1), state, torch.zeros_like(state))
        effective = torch.where(available, denominator, torch.zeros_like(denominator))
        squared_mass = (weight.square()).sum(dim=1).clamp_min(1e-8)
        effective_sample_size = torch.where(
            available, denominator.square() / squared_mass,
            torch.zeros_like(denominator),
        )
        return state, available, effective, effective_sample_size


class RobustMultiScaleIncrementMotionContext(nn.Module):
    """Causal anonymous multi-scale increment and relative-pair encoder."""

    model_family = "robust-multiscale-increment-motion-context-v6"

    def __init__(
        self,
        *,
        channels: int = 96,
        dropout: float = 0.05,
        message_layers: int = 3,
        position_scale_m: float = 1.0,
        history_scale_s: float = 0.32,
        lag_scales_s: tuple[float, ...] = (0.01, 0.03, 0.07, 0.15, 0.28),
    ) -> None:
        super().__init__()
        if channels < 32 or message_layers < 2:
            raise ValueError("invalid robust multi-scale context capacity")
        if position_scale_m <= 0 or history_scale_s <= 0:
            raise ValueError("context scales must be positive")
        if len(lag_scales_s) < 3 or any(
            not math.isfinite(float(value)) or float(value) <= 0
            for value in lag_scales_s
        ):
            raise ValueError("lag scales must contain at least three positive values")
        numeric_scales = tuple(float(value) for value in lag_scales_s)
        if tuple(sorted(numeric_scales)) != numeric_scales or len(set(numeric_scales)) != len(
            numeric_scales
        ):
            raise ValueError("lag scales must be strictly ordered")
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.message_layers = int(message_layers)
        self.position_scale_m = float(position_scale_m)
        self.history_scale_s = float(history_scale_s)
        self.register_buffer(
            "lag_scales_s", torch.tensor(lag_scales_s, dtype=torch.float32),
        )
        self.scale_embedding = nn.Parameter(torch.empty(len(lag_scales_s), channels))
        nn.init.normal_(self.scale_embedding, std=0.02)
        # Same-handle token: displacement xyz, velocity xyz, velocity contrast
        # xyz, log elapsed, endpoint-to-q0, primary, interval switch, lag ratio.
        self.handle_consensus = _RobustEdgeConsensus(14, channels, dropout)
        # Unordered pair token: delta of rr^T (6), its rate (6), time and switch.
        self.pair_consensus = _RobustEdgeConsensus(16, channels, dropout)
        self.messages = nn.ModuleList(
            SymmetricCyclicMessageBlock(2 * channels, dropout)
            for _ in range(message_layers)
        )
        self.scale_vehicle_projection = nn.Sequential(
            nn.Linear(6 * channels + 4, 4 * channels),
            nn.LayerNorm(4 * channels), nn.SiLU(),
            nn.Linear(4 * channels, 4 * channels), nn.SiLU(),
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
            "lag_scales_s": self.lag_scales_s.detach().cpu().tolist(),
            "same_handle_edges": "causal nearest prior at fixed physical time scales",
            "time_scale_support": "non-overlapping geometric-mean bands",
            "pair_edges": "unordered simultaneous two-armor rrT temporal increments",
            "aggregation": "two-pass bounded learned robust consensus",
            "absolute_position_or_range_input": False,
            "q0_geometry_or_quality_input": False,
            "physical_id_input": False,
            "session_or_motion_class_input": False,
            "truth_or_future_input": False,
            "c4_invariant_output": True,
            "direction_reflection_invariant_output": True,
        }

    @staticmethod
    def _last_active(mask: torch.Tensor) -> torch.Tensor:
        return VisibilityDrivenMotionContext._last_active(mask)

    @staticmethod
    def _compact(
        value: torch.Tensor, visible: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return VisibilityDrivenMotionContext._compact_visible(value, visible)

    def _lag_bank(
        self,
        value: torch.Tensor,
        time_s: torch.Tensor,
        valid_event: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select one strictly earlier causal predecessor per physical scale."""
        streams, events, dimensions = value.shape
        if time_s.shape != (streams, events) or valid_event.shape != (streams, events):
            raise ValueError("lag-bank stream fields have incompatible shapes")
        current_index = torch.arange(events, device=value.device)[None, :, None]
        prior_index = torch.arange(events, device=value.device)[None, None, :]
        dt_all = time_s[:, :, None] - time_s[:, None, :]
        causal = (
            valid_event[:, :, None] & valid_event[:, None, :]
            & (prior_index < current_index) & (dt_all > 1e-7)
        )
        scales = self.lag_scales_s.to(dtype=value.dtype, device=value.device)
        ratio = dt_all[:, :, :, None] / scales[None, None, None]
        middle = torch.sqrt(scales[:-1] * scales[1:])
        lower = torch.cat((scales[:1] * 0.5, middle))
        upper = torch.cat((middle, scales[-1:] * 1.5))
        candidate = (
            causal.unsqueeze(-1)
            & (dt_all[:, :, :, None] >= lower[None, None, None])
            & (dt_all[:, :, :, None] < upper[None, None, None])
        )
        cost = torch.log(ratio.clamp_min(1e-7)).abs()
        cost = torch.where(candidate, cost, torch.full_like(cost, torch.inf))
        selected = cost.argmin(dim=2)
        edge_valid = candidate.any(dim=2)
        selected_safe = selected.clamp(0, max(events - 1, 0))
        prior_value = value[:, None].expand(-1, events, -1, -1).gather(
            2, selected_safe.unsqueeze(-1).expand(-1, -1, -1, dimensions),
        )
        prior_time = time_s[:, None].expand(-1, events, -1).gather(2, selected_safe)
        elapsed = time_s[:, :, None] - prior_time
        delta = value[:, :, None] - prior_value
        delta = torch.where(edge_valid.unsqueeze(-1), delta, torch.zeros_like(delta))
        elapsed = torch.where(edge_valid, elapsed, torch.zeros_like(elapsed))
        selected = torch.where(edge_valid, selected, torch.full_like(selected, -1))
        return delta, elapsed, edge_valid, selected

    @staticmethod
    def _symmetric_pair_shape(
        observations: torch.Tensor, visible: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the six unique entries of (a-b)(a-b)^T without ordering a,b."""
        pair_valid = visible.sum(dim=2) == 2
        # Subtract first in FP32.  The algebraically equivalent
        # 2*sum(xxT)-sum(x)sum(x)T form catastrophically loses translation
        # invariance under BF16 autocast, exactly where combined motion is hard.
        device_type = observations.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            value = observations.float()
            pair_difference = value[:, :, :, None] - value[:, :, None, :]
            pair_mask = visible[:, :, :, None] & visible[:, :, None, :]
            pair_outer = 0.5 * torch.einsum(
                "bthkc,bthkd,bthk->btcd",
                pair_difference, pair_difference, pair_mask.to(value.dtype),
            )
            shape = torch.stack((
                pair_outer[..., 0, 0], pair_outer[..., 1, 1], pair_outer[..., 2, 2],
                pair_outer[..., 0, 1], pair_outer[..., 0, 2], pair_outer[..., 1, 2],
            ), dim=-1)
            shape = torch.where(
                pair_valid.unsqueeze(-1), shape, torch.zeros_like(shape),
            )
        return shape, pair_valid

    @staticmethod
    def _pool_states(
        state: torch.Tensor, support: torch.Tensor, dimension: int,
    ) -> torch.Tensor:
        weight = support.to(state.dtype).unsqueeze(-1)
        count = weight.sum(dim=dimension).clamp_min(1.0)
        mean = (state * weight).sum(dim=dimension) / count
        centered = state - mean.unsqueeze(dimension)
        dispersion = (
            (centered.square() * weight).sum(dim=dimension) / count
        ).clamp_min(1e-8).sqrt()
        return torch.cat((mean, dispersion), dim=-1)

    def forward(
        self,
        history_obs_rel_m: torch.Tensor,
        history_obs_mask: torch.Tensor,
        history_primary_mask: torch.Tensor,
        history_event_mask: torch.Tensor,
        history_time_s: torch.Tensor,
        history_switch_step: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
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
        if bool(torch.any(last_time.abs() > 1e-6)):
            raise ValueError("last active event must be q0")
        rows = torch.arange(batch, device=active.device)
        current_primary = primary[rows, last_event]

        clean_obs = torch.where(
            visible.unsqueeze(-1), history_obs_rel_m, torch.zeros_like(history_obs_rel_m),
        )
        clean_time = torch.where(active, history_time_s, torch.zeros_like(history_time_s))
        clean_switch = torch.where(
            active, history_switch_step, torch.zeros_like(history_switch_step),
        )
        cumulative = torch.cumsum(clean_switch, dim=1)

        packed_feature, packed_mask, visible_count = self._compact(
            torch.cat((
                clean_obs,
                clean_time[:, :, None, None].expand(-1, -1, 4, 1),
                primary.to(clean_obs.dtype).unsqueeze(-1),
                cumulative.to(clean_obs.dtype)[:, :, None, None].expand(-1, -1, 4, 1),
            ), dim=-1),
            visible,
        )
        packed_obs = packed_feature[..., :3]
        packed_time = packed_feature[..., 3]
        packed_primary = packed_feature[..., 4]
        packed_cumulative = packed_feature[..., 5]
        streams = batch * 4
        flat_obs = packed_obs.reshape(streams, events, 3)
        flat_time = packed_time.reshape(streams, events)
        flat_mask = packed_mask.reshape(streams, events)
        delta, elapsed, edge_valid, prior_index = self._lag_bank(
            flat_obs, flat_time, flat_mask,
        )
        velocity = torch.where(
            edge_valid.unsqueeze(-1),
            delta / elapsed.clamp_min(1e-7).unsqueeze(-1),
            torch.zeros_like(delta),
        )
        velocity_mean = (
            velocity * edge_valid.unsqueeze(-1).to(velocity.dtype)
        ).sum(dim=2) / edge_valid.sum(dim=2).clamp_min(1).unsqueeze(-1)
        velocity_contrast = velocity - velocity_mean[:, :, None]
        prior_safe = prior_index.clamp_min(0)
        prior_cumulative = packed_cumulative.reshape(streams, events)[:, None].expand(
            -1, events, -1
        ).gather(2, prior_safe)
        endpoint_cumulative = packed_cumulative.reshape(streams, events)[:, :, None]
        interval_switch = (endpoint_cumulative - prior_cumulative).abs()
        scale = self.lag_scales_s.to(dtype=clean_obs.dtype, device=clean_obs.device)
        handle_token = torch.cat((
            delta / self.position_scale_m,
            velocity * (self.history_scale_s / self.position_scale_m),
            velocity_contrast * (self.history_scale_s / self.position_scale_m),
            torch.log1p(elapsed / 0.01).unsqueeze(-1),
            (flat_time[:, :, None] / self.history_scale_s).unsqueeze(-1).expand(
                -1, -1, len(scale), -1
            ),
            packed_primary.reshape(streams, events)[:, :, None, None].expand(
                -1, -1, len(scale), -1
            ),
            (interval_switch / 6.0).unsqueeze(-1),
            (elapsed / scale[None, None]).unsqueeze(-1),
        ), dim=-1)
        (
            handle_scale, handle_available, handle_weight_mass, handle_ess,
        ) = self.handle_consensus(
            handle_token, edge_valid, self.scale_embedding, elapsed, scale,
        )
        scale_count = len(scale)
        handle_scale = handle_scale.reshape(batch, 4, scale_count, 2 * self.channels)
        handle_available = handle_available.reshape(batch, 4, scale_count)
        handle_weight_mass = handle_weight_mass.reshape(batch, 4, scale_count)
        handle_ess = handle_ess.reshape(batch, 4, scale_count)
        message_state = handle_scale.permute(0, 2, 1, 3).reshape(
            batch * scale_count, 4, 2 * self.channels,
        )
        for message in self.messages:
            message_state = message(message_state)
        message_state = message_state.reshape(
            batch, scale_count, 4, 2 * self.channels,
        ).permute(0, 2, 1, 3)
        message_state = torch.where(
            handle_available.unsqueeze(-1), message_state, torch.zeros_like(message_state),
        )
        handle_vehicle = self._pool_states(
            message_state.permute(0, 2, 1, 3),
            handle_ess.permute(0, 2, 1),
            dimension=2,
        )

        pair_shape, pair_event = self._symmetric_pair_shape(clean_obs, visible)
        pair_feature, pair_mask, _ = self._compact(
            torch.cat((
                pair_shape,
                clean_time.unsqueeze(-1),
                cumulative.to(clean_obs.dtype).unsqueeze(-1),
            ), dim=-1).unsqueeze(2),
            pair_event.unsqueeze(-1),
        )
        pair_feature = pair_feature[:, 0]
        pair_mask = pair_mask[:, 0]
        pair_value = pair_feature[..., :6]
        pair_time = pair_feature[..., 6]
        pair_cumulative = pair_feature[..., 7]
        pair_delta, pair_elapsed, pair_edge_valid, pair_prior = self._lag_bank(
            pair_value, pair_time, pair_mask,
        )
        pair_rate = torch.where(
            pair_edge_valid.unsqueeze(-1),
            pair_delta / pair_elapsed.clamp_min(1e-7).unsqueeze(-1),
            torch.zeros_like(pair_delta),
        )
        pair_prior_safe = pair_prior.clamp_min(0)
        pair_prior_cumulative = pair_cumulative[:, None].expand(
            -1, events, -1
        ).gather(2, pair_prior_safe)
        pair_interval_switch = (
            pair_cumulative[:, :, None] - pair_prior_cumulative
        ).abs()
        pair_position_scale = self.position_scale_m * self.position_scale_m
        pair_token = torch.cat((
            pair_delta / pair_position_scale,
            pair_rate * (self.history_scale_s / pair_position_scale),
            torch.log1p(pair_elapsed / 0.01).unsqueeze(-1),
            (pair_time[:, :, None] / self.history_scale_s).unsqueeze(-1).expand(
                -1, -1, scale_count, -1
            ),
            (pair_interval_switch / 6.0).unsqueeze(-1),
            (pair_elapsed / scale[None, None]).unsqueeze(-1),
        ), dim=-1)
        (
            pair_scale, pair_available, pair_weight_mass, pair_ess,
        ) = self.pair_consensus(
            pair_token, pair_edge_valid, self.scale_embedding, pair_elapsed, scale,
        )
        handle_scale_available = handle_available.any(dim=1)
        scale_available = handle_scale_available | pair_available
        scale_coordinate_available = torch.cat((
            handle_scale_available.unsqueeze(-1).expand(-1, -1, 3),
            scale_available.unsqueeze(-1),
        ), dim=-1)
        pair_scale = torch.where(
            pair_available.unsqueeze(-1), pair_scale, torch.zeros_like(pair_scale),
        )
        reliability_feature = torch.stack((
            torch.log1p(handle_ess.sum(dim=1)),
            handle_available.sum(dim=1).to(clean_obs.dtype) / 4.0,
            torch.log1p(pair_ess),
            pair_available.to(clean_obs.dtype),
        ), dim=-1)
        vehicle_input = torch.cat((handle_vehicle, pair_scale, reliability_feature), dim=-1)
        scale_vehicle = self.scale_vehicle_projection(vehicle_input)
        scale_vehicle = torch.where(
            scale_available.unsqueeze(-1), scale_vehicle, torch.zeros_like(scale_vehicle),
        )
        if bool(torch.any(~scale_coordinate_available.any(dim=1).all(dim=-1))):
            raise ValueError("no causal multi-scale edge is available")
        return {
            "scale_vehicle_state": scale_vehicle,
            "scale_handle_vehicle_state": handle_vehicle,
            "scale_available": scale_available,
            "scale_coordinate_available": scale_coordinate_available,
            "scale_reliability_feature": reliability_feature,
            "scale_handle_available": handle_available,
            "scale_handle_weight_mass": handle_weight_mass.sum(dim=1),
            "scale_handle_effective_sample_size": handle_ess.sum(dim=1),
            "scale_edge_weight_mass": handle_weight_mass.sum(dim=1) + pair_weight_mass,
            "scale_effective_sample_size": handle_ess.sum(dim=1) + pair_ess,
            "scale_pair_available": pair_available,
            "lag_prior_index": prior_index.reshape(batch, 4, events, scale_count),
            "lag_edge_valid": edge_valid.reshape(batch, 4, events, scale_count),
            "handle_state": message_state.mean(dim=2),
            "vehicle_state": (
                scale_vehicle * scale_available.unsqueeze(-1).to(scale_vehicle.dtype)
            ).sum(dim=1) / scale_available.sum(dim=1).clamp_min(1).unsqueeze(-1),
            "primary_index": current_primary.to(torch.long).argmax(dim=1),
            "history_active_count": active.sum(dim=1),
            "history_visible_count": visible_count,
        }


class AvailabilityAwareMotionStateHead(nn.Module):
    """Deep-supervised per-scale states with coordinate-wise causal fusion."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        width = 4 * channels
        self.scale_state = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 2 * channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(2 * channels, channels), nn.SiLU(),
            nn.Linear(channels, 4), nn.Tanh(),
        )
        self.scale_log_variance = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, channels), nn.SiLU(),
            nn.Linear(channels, 4),
        )
        self.fusion_score = nn.Sequential(
            nn.LayerNorm(width + 12), nn.Linear(width + 12, channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(channels, 4),
        )
        nn.init.zeros_(self.scale_state[-2].bias)
        nn.init.zeros_(self.scale_log_variance[-1].bias)
        nn.init.zeros_(self.fusion_score[-1].bias)

    def forward(
        self,
        scale_vehicle_state: torch.Tensor,
        scale_coordinate_available: torch.Tensor,
        scale_reliability_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if scale_vehicle_state.ndim != 3:
            raise ValueError("scale vehicle state must have shape [B,K,C]")
        if scale_coordinate_available.shape != (*scale_vehicle_state.shape[:2], 4):
            raise ValueError("coordinate scale availability has the wrong shape")
        if scale_reliability_feature.shape != (*scale_vehicle_state.shape[:2], 4):
            raise ValueError("scale reliability features have the wrong shape")
        scale_state = self.scale_state(scale_vehicle_state)
        scale_log_variance = self.scale_log_variance(scale_vehicle_state).clamp(-5.0, 5.0)
        fusion_feature = torch.cat((
            scale_vehicle_state, scale_state, scale_log_variance,
            scale_reliability_feature,
        ), dim=-1)
        logits = self.fusion_score(fusion_feature)
        logits = logits.masked_fill(~scale_coordinate_available, -torch.inf)
        weight = torch.softmax(logits.float(), dim=1).to(scale_state.dtype)
        fused = (weight * scale_state).sum(dim=1)
        fused_log_variance = (
            weight * scale_log_variance
        ).sum(dim=1)
        return {
            "motion_state_normalized": fused,
            "scale_motion_state_normalized": scale_state,
            "scale_motion_log_variance": scale_log_variance,
            "scale_motion_weight": weight,
            "motion_log_variance": fused_log_variance,
        }


class RobustMultiScaleMotionBottleneckFutureModel(
    StableMotionBottleneckAnonymousFutureModel
):
    """V6 state estimator with the unchanged learned trajectory interface."""

    model_family = "robust-multiscale-motion-bottleneck-future-v6"

    def __init__(
        self,
        *,
        lag_scales_s: tuple[float, ...] = (0.01, 0.03, 0.07, 0.15, 0.28),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.context = RobustMultiScaleIncrementMotionContext(
            channels=self.channels, dropout=self.dropout,
            message_layers=self.message_layers,
            position_scale_m=self.position_scale_m,
            history_scale_s=self.history_scale_s,
            lag_scales_s=lag_scales_s,
        )
        self.motion_state_head = AvailabilityAwareMotionStateHead(
            self.channels, self.dropout,
        )

    @property
    def config(self) -> dict[str, Any]:
        parent = dict(super().config)
        parent.update({
            "family": self.model_family,
            "motion_context": self.context.config,
            "motion_state_fusion": "coordinate-wise available-scale softmax",
            "scale_uncertainty_decoder_input": False,
            "decoder_temporal_input": "fused predicted 4D motion state only",
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
            raise ValueError(f"v6 future forward fields missing: {sorted(missing)}")
        state_output = self.estimate_motion_state(
            history_obs_rel_m=batch["history_obs_rel_m"],
            history_obs_mask=batch["history_obs_mask"],
            history_primary_mask=batch["history_primary_mask"],
            history_event_mask=batch["history_event_mask"],
            history_time_s=batch["history_time_s"],
            history_switch_step=batch["history_switch_step"],
        )
        history = state_output["history"]
        state = state_output["state"]
        motion_state_normalized = state["motion_state_normalized"]
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
            **state,
            "motion_state_physical": (
                motion_state_normalized * self.motion_state_scale.to(
                    motion_state_normalized.dtype,
                )
            ),
            **decoded,
        }

    def estimate_motion_state(
        self,
        *,
        history_obs_rel_m: torch.Tensor,
        history_obs_mask: torch.Tensor,
        history_primary_mask: torch.Tensor,
        history_event_mask: torch.Tensor,
        history_time_s: torch.Tensor,
        history_switch_step: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Estimate state through the exact causal, observation-only API."""
        history = self.context(
            history_obs_rel_m, history_obs_mask, history_primary_mask,
            history_event_mask, history_time_s, history_switch_step,
        )
        state = self.motion_state_head(
            history["scale_vehicle_state"],
            history["scale_coordinate_available"],
            history["scale_reliability_feature"],
        )
        state["motion_state_physical"] = (
            state["motion_state_normalized"] * self.motion_state_scale.to(
                state["motion_state_normalized"].dtype,
            )
        )
        return {"history": history, "state": state}


def robust_multiscale_motion_future_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    **weights: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """V5 future losses plus deep supervision of every available state scale."""
    objective, components = stable_motion_future_loss(prediction, batch, **weights)
    motion_objective, motion_components = robust_multiscale_motion_state_loss(
        prediction, batch,
    )
    motion_weight = float(weights.get("motion_weight", 0.0))
    objective = objective + motion_weight * (
        motion_objective - motion_components["motion"]
    )
    components = dict(components)
    components["scale_aux"] = motion_components["scale_aux"]
    components["scale_heteroscedastic"] = motion_components[
        "scale_heteroscedastic"
    ]
    components["objective"] = objective
    return objective, components


def robust_multiscale_motion_state_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """State-only loss that never reads future/q0/decoder fields."""
    target = batch.get("target_motion_state_normalized")
    scale_state = prediction.get("scale_motion_state_normalized")
    available = prediction.get("scale_coordinate_available")
    log_variance = prediction.get("scale_motion_log_variance")
    if target is None or scale_state is None or available is None or log_variance is None:
        raise ValueError("v6 loss requires supervised multi-scale state outputs")
    final_coordinate = F.smooth_l1_loss(
        prediction["motion_state_normalized"], target,
        reduction="none", beta=0.1,
    )
    velocity = final_coordinate[:, :3].mean(dim=-1).mean()
    yaw_rate = final_coordinate[:, 3].mean()
    motion = velocity + yaw_rate
    coordinate = F.smooth_l1_loss(
        scale_state, target[:, None].expand_as(scale_state),
        reduction="none", beta=0.1,
    )
    available_f = available.to(coordinate.dtype)
    denominator = available_f.sum().clamp_min(1.0)
    scale_aux = (coordinate * available_f).sum() / denominator
    heteroscedastic = (
        (
            coordinate * torch.exp(-log_variance)
            + 0.05 * F.softplus(log_variance)
        )
        * available_f
    ).sum() / denominator
    objective = motion + 0.30 * scale_aux + 0.05 * heteroscedastic
    zero = objective * 0.0
    return objective, {
        "objective": objective,
        "motion": motion,
        "velocity": velocity,
        "yaw_rate": yaw_rate,
        "scale_aux": scale_aux,
        "scale_heteroscedastic": heteroscedastic,
        "trajectory": zero,
        "trend": zero,
        "role": zero,
        "distance_risk": zero,
    }


__all__ = [
    "AvailabilityAwareMotionStateHead",
    "RobustMultiScaleIncrementMotionContext",
    "RobustMultiScaleMotionBottleneckFutureModel",
    "robust_multiscale_motion_future_loss",
    "robust_multiscale_motion_state_loss",
]
