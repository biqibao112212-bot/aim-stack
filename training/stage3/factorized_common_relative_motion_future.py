"""Factorized causal motion state for combined translation and rotation.

V7 keeps the deployed six-field observation API and four-dimensional physical
state.  Internally it prevents the failure mode measured in v6 by separating
unordered relative-shape angular evidence, anonymous common-translation
evidence and vertical velocity.  The sole cross-branch path is detached
predicted yaw conditioning a learned planar residual.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .anonymous_vehicle_motion_v2 import VisibilityDrivenMotionContext
from .continuous_invariant_anonymous_future import V3_FORWARD_FIELDS
from .observable_future_pnp_ab import state_dict_sha256
from .robust_multiscale_motion_future import _RobustEdgeConsensus
from .stable_motion_bottleneck_future import (
    StableMotionBottleneckAnonymousFutureModel,
    stable_motion_future_loss,
)


def _weighted_pool(
    value: torch.Tensor, support: torch.Tensor, dimension: int,
) -> torch.Tensor:
    weight = support.to(value.dtype).unsqueeze(-1)
    denominator = weight.sum(dim=dimension).clamp_min(1.0)
    mean = (value * weight).sum(dim=dimension) / denominator
    centered = value - mean.unsqueeze(dimension)
    dispersion = (
        (centered.square() * weight).sum(dim=dimension) / denominator
    ).clamp_min(1e-8).sqrt()
    return torch.cat((mean, dispersion), dim=-1)


class FactorizedCommonRelativeMotionContext(nn.Module):
    """Build independent common-motion and relative-angular scale evidence."""

    model_family = "factorized-common-relative-motion-context-v7"

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
            raise ValueError("invalid factorized context capacity")
        if position_scale_m <= 0 or history_scale_s <= 0:
            raise ValueError("context scales must be positive")
        numeric_scales = tuple(float(value) for value in lag_scales_s)
        if (
            len(numeric_scales) < 3
            or any(not math.isfinite(value) or value <= 0 for value in numeric_scales)
            or tuple(sorted(numeric_scales)) != numeric_scales
            or len(set(numeric_scales)) != len(numeric_scales)
        ):
            raise ValueError("lag scales must be distinct, positive and ordered")
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.message_layers = int(message_layers)
        self.position_scale_m = float(position_scale_m)
        self.history_scale_s = float(history_scale_s)
        self.register_buffer(
            "lag_scales_s", torch.tensor(numeric_scales, dtype=torch.float32),
        )
        scale_count = len(numeric_scales)
        self.translation_scale_embedding = nn.Parameter(
            torch.empty(scale_count, channels)
        )
        self.rotation_scale_embedding = nn.Parameter(torch.empty(scale_count, channels))
        nn.init.normal_(self.translation_scale_embedding, std=0.02)
        nn.init.normal_(self.rotation_scale_embedding, std=0.02)

        # Translation tokens contain only same-handle or same-anonymous-set
        # displacement evidence.  Relative-shape latent never enters here.
        self.handle_translation_consensus = _RobustEdgeConsensus(
            10, channels, dropout,
        )
        self.centroid_translation_consensus = _RobustEdgeConsensus(
            10, channels, dropout,
        )
        self.translation_projection = nn.Sequential(
            nn.Linear(6 * channels + 5, 4 * channels),
            nn.LayerNorm(4 * channels), nn.SiLU(),
            nn.Linear(4 * channels, 2 * channels), nn.SiLU(),
        )
        self.translation_refinement = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(2 * channels), nn.Linear(2 * channels, 2 * channels),
                nn.SiLU(), nn.Dropout(dropout), nn.Linear(2 * channels, 2 * channels),
            )
            for _ in range(message_layers)
        )

        # Angular tokens retain current and prior normalized rrT, their delta
        # and rate, and the signed matrix commutator.  This avoids the v6 loss
        # of orientation context from using delta(rrT) alone.
        self.pair_rotation_consensus = _RobustEdgeConsensus(
            35, channels, dropout,
        )
        self.curvature_rotation_consensus = _RobustEdgeConsensus(
            20, channels, dropout,
        )
        self.curvature_projection = nn.Sequential(
            nn.Linear(4 * channels, 2 * channels),
            nn.LayerNorm(2 * channels), nn.SiLU(),
        )
        self.rotation_projection = nn.Sequential(
            nn.Linear(2 * channels + 3, 2 * channels),
            nn.LayerNorm(2 * channels), nn.SiLU(),
            nn.Linear(2 * channels, 2 * channels), nn.SiLU(),
        )
        self.rotation_refinement = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(2 * channels), nn.Linear(2 * channels, 2 * channels),
                nn.SiLU(), nn.Dropout(dropout), nn.Linear(2 * channels, 2 * channels),
            )
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
            "lag_scales_s": self.lag_scales_s.detach().cpu().tolist(),
            "time_scale_support": "non-overlapping geometric-mean bands",
            "angular_evidence": (
                "normalized unordered rrT with common-ramp-invariant curvature fallback"
            ),
            "translation_evidence": (
                "same-handle increments plus same-unordered-set pair centroids"
            ),
            "vertical_evidence": "translation branch without angular conditioning",
            "physical_id_input": False,
            "absolute_position_or_range_input": False,
            "q0_geometry_or_quality_input": False,
            "session_or_motion_class_input": False,
            "truth_or_future_input": False,
            "pair_latent_to_translation": False,
            "translation_latent_to_angular": False,
            "c4_invariant_output": True,
            "handle_reflection_invariant_output": True,
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
        """Choose one strictly earlier predecessor in each disjoint lag band."""
        streams, events, dimensions = value.shape
        if time_s.shape != (streams, events) or valid_event.shape != (streams, events):
            raise ValueError("lag-bank stream fields have incompatible shapes")
        current_index = torch.arange(events, device=value.device)[None, :, None]
        prior_index = torch.arange(events, device=value.device)[None, None, :]
        elapsed_all = time_s[:, :, None] - time_s[:, None, :]
        causal = (
            valid_event[:, :, None] & valid_event[:, None, :]
            & (prior_index < current_index) & (elapsed_all > 1e-7)
        )
        scales = self.lag_scales_s.to(dtype=value.dtype, device=value.device)
        middle = torch.sqrt(scales[:-1] * scales[1:])
        lower = torch.cat((scales[:1] * 0.5, middle))
        upper = torch.cat((middle, scales[-1:] * 1.5))
        ratio = elapsed_all[:, :, :, None] / scales[None, None, None]
        candidate = (
            causal.unsqueeze(-1)
            & (elapsed_all[:, :, :, None] >= lower[None, None, None])
            & (elapsed_all[:, :, :, None] < upper[None, None, None])
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
    def _normalized_pair_shape(
        observations: torch.Tensor, visible: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return trace-normalized unordered rrT in FP32."""
        pair_valid = visible.sum(dim=2) == 2
        with torch.autocast(device_type=observations.device.type, enabled=False):
            value = observations.float()
            difference = value[:, :, :, None] - value[:, :, None, :]
            pair_mask = visible[:, :, :, None] & visible[:, :, None, :]
            outer = 0.5 * torch.einsum(
                "bthkc,bthkd,bthk->btcd",
                difference, difference, pair_mask.to(value.dtype),
            )
            trace = torch.diagonal(outer, dim1=-2, dim2=-1).sum(dim=-1)
            normalized = outer / trace.clamp_min(1e-8)[..., None, None]
            shape = torch.stack((
                normalized[..., 0, 0], normalized[..., 1, 1],
                normalized[..., 2, 2], normalized[..., 0, 1],
                normalized[..., 0, 2], normalized[..., 1, 2],
            ), dim=-1)
            shape = torch.where(pair_valid.unsqueeze(-1), shape, torch.zeros_like(shape))
        return shape, pair_valid

    @staticmethod
    def _shape_matrix(shape: torch.Tensor) -> torch.Tensor:
        xx, yy, zz, xy, xz, yz = shape.unbind(dim=-1)
        return torch.stack((
            xx, xy, xz, xy, yy, yz, xz, yz, zz,
        ), dim=-1).reshape(*shape.shape[:-1], 3, 3)

    @staticmethod
    def _edge_token(
        delta: torch.Tensor,
        velocity: torch.Tensor,
        elapsed: torch.Tensor,
        endpoint_time: torch.Tensor,
        interval_switch: torch.Tensor,
        scales: torch.Tensor,
        position_scale_m: float,
        history_scale_s: float,
    ) -> torch.Tensor:
        return torch.cat((
            delta / position_scale_m,
            velocity * (history_scale_s / position_scale_m),
            torch.log1p(elapsed / 0.01).unsqueeze(-1),
            (endpoint_time[:, :, None] / history_scale_s).unsqueeze(-1).expand(
                -1, -1, len(scales), -1,
            ),
            (interval_switch / 6.0).unsqueeze(-1),
            (elapsed / scales[None, None]).unsqueeze(-1),
        ), dim=-1)

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
        scales = self.lag_scales_s.to(dtype=clean_obs.dtype, device=clean_obs.device)
        scale_count = len(scales)

        # Anonymous same-handle translation evidence.
        packed, packed_mask, visible_count = self._compact(
            torch.cat((
                clean_obs,
                clean_time[:, :, None, None].expand(-1, -1, 4, 1),
                cumulative.to(clean_obs.dtype)[:, :, None, None].expand(-1, -1, 4, 1),
            ), dim=-1),
            visible,
        )
        packed_obs, packed_time, packed_cumulative = (
            packed[..., :3], packed[..., 3], packed[..., 4],
        )
        streams = batch * 4
        flat_obs = packed_obs.reshape(streams, events, 3)
        flat_time = packed_time.reshape(streams, events)
        flat_mask = packed_mask.reshape(streams, events)
        handle_delta, handle_elapsed, handle_valid, handle_prior = self._lag_bank(
            flat_obs, flat_time, flat_mask,
        )
        handle_velocity = torch.where(
            handle_valid.unsqueeze(-1),
            handle_delta / handle_elapsed.clamp_min(1e-7).unsqueeze(-1),
            torch.zeros_like(handle_delta),
        )
        handle_prior_safe = handle_prior.clamp_min(0)
        flat_cumulative = packed_cumulative.reshape(streams, events)
        handle_prior_cumulative = flat_cumulative[:, None].expand(
            -1, events, -1,
        ).gather(2, handle_prior_safe)
        handle_interval_switch = (
            flat_cumulative[:, :, None] - handle_prior_cumulative
        ).abs()
        handle_token = self._edge_token(
            handle_delta, handle_velocity, handle_elapsed, flat_time,
            handle_interval_switch, scales, self.position_scale_m,
            self.history_scale_s,
        )
        handle_state, handle_available, handle_mass, handle_ess = (
            self.handle_translation_consensus(
                handle_token, handle_valid, self.translation_scale_embedding,
                handle_elapsed, scales,
            )
        )
        handle_state = handle_state.reshape(batch, 4, scale_count, 2 * self.channels)
        handle_available = handle_available.reshape(batch, 4, scale_count)
        handle_mass = handle_mass.reshape(batch, 4, scale_count)
        handle_ess = handle_ess.reshape(batch, 4, scale_count)
        handle_vehicle = _weighted_pool(
            handle_state.permute(0, 2, 1, 3),
            handle_ess.permute(0, 2, 1), dimension=2,
        )

        # Centroid increments are admitted only when both endpoints contain the
        # same unordered window-local handle set.  Slot values are anonymous
        # memory, never physical plate identity.
        pair_event = visible.sum(dim=2) == 2
        centroid = (
            clean_obs * visible.unsqueeze(-1).to(clean_obs.dtype)
        ).sum(dim=2) / visible.sum(dim=2).clamp_min(1).unsqueeze(-1)
        centroid_delta, centroid_elapsed, centroid_valid, centroid_prior = self._lag_bank(
            centroid, clean_time, pair_event,
        )
        centroid_prior_safe = centroid_prior.clamp_min(0)
        prior_set = visible[:, None].expand(-1, events, -1, -1).gather(
            2, centroid_prior_safe.unsqueeze(-1).expand(-1, -1, -1, 4),
        )
        same_set = (prior_set == visible[:, :, None]).all(dim=-1)
        centroid_valid = centroid_valid & same_set
        centroid_delta = torch.where(
            centroid_valid.unsqueeze(-1), centroid_delta, torch.zeros_like(centroid_delta),
        )
        centroid_elapsed = torch.where(
            centroid_valid, centroid_elapsed, torch.zeros_like(centroid_elapsed),
        )
        centroid_velocity = torch.where(
            centroid_valid.unsqueeze(-1),
            centroid_delta / centroid_elapsed.clamp_min(1e-7).unsqueeze(-1),
            torch.zeros_like(centroid_delta),
        )
        centroid_prior_cumulative = cumulative[:, None].expand(
            -1, events, -1,
        ).gather(2, centroid_prior_safe)
        centroid_switch = (cumulative[:, :, None] - centroid_prior_cumulative).abs()
        centroid_token = self._edge_token(
            centroid_delta, centroid_velocity, centroid_elapsed, clean_time,
            centroid_switch, scales, self.position_scale_m, self.history_scale_s,
        )
        centroid_state, centroid_available, centroid_mass, centroid_ess = (
            self.centroid_translation_consensus(
                centroid_token, centroid_valid, self.translation_scale_embedding,
                centroid_elapsed, scales,
            )
        )
        translation_available = handle_available.any(dim=1) | centroid_available
        translation_reliability = torch.stack((
            torch.log1p(handle_ess.sum(dim=1)),
            handle_available.sum(dim=1).to(clean_obs.dtype) / 4.0,
            torch.log1p(centroid_ess),
            centroid_available.to(clean_obs.dtype),
            torch.log1p(handle_mass.sum(dim=1) + centroid_mass),
        ), dim=-1)
        translation_input = torch.cat((
            handle_vehicle,
            torch.where(
                centroid_available.unsqueeze(-1), centroid_state,
                torch.zeros_like(centroid_state),
            ),
            translation_reliability,
        ), dim=-1)
        translation_scale_latent = self.translation_projection(translation_input)
        for refinement in self.translation_refinement:
            translation_scale_latent = translation_scale_latent + refinement(
                translation_scale_latent,
            )
        translation_scale_latent = torch.where(
            translation_available.unsqueeze(-1), translation_scale_latent,
            torch.zeros_like(translation_scale_latent),
        )

        # Unordered relative-shape angular evidence.
        pair_shape, pair_event = self._normalized_pair_shape(clean_obs, visible)
        pair_delta, pair_elapsed, pair_valid, pair_prior = self._lag_bank(
            pair_shape, clean_time, pair_event,
        )
        pair_prior_safe = pair_prior.clamp_min(0)
        prior_shape = pair_shape[:, None].expand(-1, events, -1, -1).gather(
            2, pair_prior_safe.unsqueeze(-1).expand(-1, -1, -1, 6),
        )
        current_shape = pair_shape[:, :, None].expand(-1, -1, scale_count, -1)
        prior_matrix = self._shape_matrix(prior_shape)
        current_matrix = self._shape_matrix(current_shape)
        commutator = current_matrix @ prior_matrix - prior_matrix @ current_matrix
        commutator_unique = torch.stack((
            commutator[..., 0, 1], commutator[..., 0, 2], commutator[..., 1, 2],
        ), dim=-1)
        frobenius_dot = (current_matrix * prior_matrix).sum(dim=(-1, -2))
        current_norm = current_matrix.square().sum(dim=(-1, -2)).clamp_min(1e-8).sqrt()
        prior_norm = prior_matrix.square().sum(dim=(-1, -2)).clamp_min(1e-8).sqrt()
        pair_prior_cumulative = cumulative[:, None].expand(
            -1, events, -1,
        ).gather(2, pair_prior_safe)
        pair_switch = (cumulative[:, :, None] - pair_prior_cumulative).abs()
        pair_support = pair_valid.sum(dim=1).to(clean_obs.dtype)
        pair_rate = torch.where(
            pair_valid.unsqueeze(-1),
            pair_delta / pair_elapsed.clamp_min(1e-7).unsqueeze(-1),
            torch.zeros_like(pair_delta),
        )
        pair_token = torch.cat((
            current_shape, prior_shape, pair_delta, pair_rate,
            commutator_unique,
            frobenius_dot.unsqueeze(-1), current_norm.unsqueeze(-1),
            prior_norm.unsqueeze(-1),
            torch.log1p(pair_elapsed / 0.01).unsqueeze(-1),
            (clean_time[:, :, None] / self.history_scale_s).unsqueeze(-1).expand(
                -1, -1, scale_count, -1,
            ),
            (pair_switch / 6.0).unsqueeze(-1),
            (pair_elapsed / scales[None, None]).unsqueeze(-1),
            torch.log1p(pair_support)[:, None, :, None].expand(-1, events, -1, -1),
        ), dim=-1)
        pair_rotation, pair_available, pair_mass, pair_ess = (
            self.pair_rotation_consensus(
                pair_token, pair_valid, self.rotation_scale_embedding,
                pair_elapsed, scales,
            )
        )

        # Same-handle curvature is a fallback only when no pair edge exists at
        # that scale.  It uses only differences of consecutive velocities, so
        # a common constant-velocity ramp cancels exactly.  Absolute velocity
        # direction and magnitude are deliberately forbidden from yaw.
        prior_velocity = handle_velocity[:, None].expand(
            -1, events, -1, -1, -1,
        ).gather(
            2,
            handle_prior_safe.unsqueeze(2).unsqueeze(-1).expand(
                -1, -1, 1, -1, 3,
            ),
        ).squeeze(2)
        prior_edge_valid = handle_valid[:, None].expand(
            -1, events, -1, -1,
        ).gather(2, handle_prior_safe.unsqueeze(2)).squeeze(2)
        prior_elapsed = handle_elapsed[:, None].expand(
            -1, events, -1, -1,
        ).gather(2, handle_prior_safe.unsqueeze(2)).squeeze(2)
        curvature_valid = handle_valid & prior_edge_valid
        acceleration = torch.where(
            curvature_valid.unsqueeze(-1), handle_velocity - prior_velocity,
            torch.zeros_like(handle_velocity),
        )
        prior_acceleration = acceleration[:, None].expand(
            -1, events, -1, -1, -1,
        ).gather(
            2,
            handle_prior_safe.unsqueeze(2).unsqueeze(-1).expand(
                -1, -1, 1, -1, 3,
            ),
        ).squeeze(2)
        prior_curvature_valid = curvature_valid[:, None].expand(
            -1, events, -1, -1,
        ).gather(2, handle_prior_safe.unsqueeze(2)).squeeze(2)
        fallback_valid = curvature_valid & prior_curvature_valid
        acceleration_norm = acceleration.norm(dim=-1)
        prior_acceleration_norm = prior_acceleration.norm(dim=-1)
        unit_acceleration = acceleration / acceleration_norm.clamp_min(1e-6).unsqueeze(-1)
        unit_prior_acceleration = (
            prior_acceleration
            / prior_acceleration_norm.clamp_min(1e-6).unsqueeze(-1)
        )
        curvature_token = torch.cat((
            acceleration * (self.history_scale_s / self.position_scale_m),
            prior_acceleration * (self.history_scale_s / self.position_scale_m),
            (acceleration - prior_acceleration)
            * (self.history_scale_s / self.position_scale_m),
            torch.cross(unit_prior_acceleration, unit_acceleration, dim=-1),
            (unit_prior_acceleration * unit_acceleration).sum(dim=-1, keepdim=True),
            (acceleration_norm * self.history_scale_s / self.position_scale_m).unsqueeze(-1),
            (
                prior_acceleration_norm * self.history_scale_s
                / self.position_scale_m
            ).unsqueeze(-1),
            torch.log1p(handle_elapsed / 0.01).unsqueeze(-1),
            torch.log1p(prior_elapsed / 0.01).unsqueeze(-1),
            (flat_time[:, :, None] / self.history_scale_s).unsqueeze(-1).expand(
                -1, -1, scale_count, -1,
            ),
            (handle_interval_switch / 6.0).unsqueeze(-1),
            (handle_elapsed / scales[None, None]).unsqueeze(-1),
        ), dim=-1)
        curvature_state, curvature_available, curvature_mass, curvature_ess = (
            self.curvature_rotation_consensus(
                curvature_token, fallback_valid, self.rotation_scale_embedding,
                handle_elapsed, scales,
            )
        )
        curvature_state = curvature_state.reshape(
            batch, 4, scale_count, 2 * self.channels,
        )
        curvature_available = curvature_available.reshape(batch, 4, scale_count)
        curvature_mass = curvature_mass.reshape(batch, 4, scale_count)
        curvature_ess = curvature_ess.reshape(batch, 4, scale_count)
        fallback_available = curvature_available.any(dim=1)
        fallback_state = self.curvature_projection(_weighted_pool(
            curvature_state.permute(0, 2, 1, 3),
            curvature_ess.permute(0, 2, 1), dimension=2,
        ))
        use_pair = pair_available
        rotation_available = use_pair | fallback_available
        selected_rotation = torch.where(
            use_pair.unsqueeze(-1), pair_rotation, fallback_state,
        )
        selected_ess = torch.where(
            use_pair, pair_ess, curvature_ess.sum(dim=1),
        )
        rotation_reliability = torch.stack((
            torch.log1p(selected_ess),
            use_pair.to(clean_obs.dtype),
            (~use_pair & fallback_available).to(clean_obs.dtype),
        ), dim=-1)
        rotation_scale_latent = self.rotation_projection(torch.cat((
            selected_rotation, rotation_reliability,
        ), dim=-1))
        for refinement in self.rotation_refinement:
            rotation_scale_latent = rotation_scale_latent + refinement(
                rotation_scale_latent,
            )
        rotation_scale_latent = torch.where(
            rotation_available.unsqueeze(-1), rotation_scale_latent,
            torch.zeros_like(rotation_scale_latent),
        )

        if bool(torch.any(~translation_available.any(dim=1))):
            raise ValueError("no causal common-motion edge is available")
        return {
            "translation_scale_latent": translation_scale_latent,
            "rotation_scale_latent": rotation_scale_latent,
            "translation_scale_available": translation_available,
            "rotation_scale_available": rotation_available,
            "translation_scale_reliability": translation_reliability,
            "rotation_scale_reliability": rotation_reliability,
            "scale_coordinate_available": torch.cat((
                translation_available.unsqueeze(-1).expand(-1, -1, 3),
                rotation_available.unsqueeze(-1),
            ), dim=-1),
            "scale_available": translation_available | rotation_available,
            "scale_handle_available": handle_available,
            "scale_pair_available": pair_available,
            "scale_centroid_same_set_available": centroid_available,
            "scale_curvature_fallback_available": fallback_available,
            "scale_edge_weight_mass": (
                handle_mass.sum(dim=1) + centroid_mass + pair_mass
                + curvature_mass.sum(dim=1)
            ),
            "scale_effective_sample_size": (
                handle_ess.sum(dim=1) + centroid_ess + pair_ess
                + curvature_ess.sum(dim=1)
            ),
            "lag_prior_index": handle_prior.reshape(
                batch, 4, events, scale_count,
            ),
            "lag_edge_valid": handle_valid.reshape(
                batch, 4, events, scale_count,
            ),
            "primary_index": current_primary.to(torch.long).argmax(dim=1),
            "history_active_count": active.sum(dim=1),
            "history_visible_count": visible_count,
        }


class FactorizedMotionStateHead(nn.Module):
    """Separate planar, vertical and angular scale heads and fusers."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        width = 2 * channels
        self.planar_base = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(channels, 2),
        )
        self.planar_rotation_residual = nn.Sequential(
            nn.LayerNorm(width + 4), nn.Linear(width + 4, channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(channels, 2), nn.Tanh(),
        )
        self.vertical_state = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(channels, 1), nn.Tanh(),
        )
        self.angular_state = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(channels, 1), nn.Tanh(),
        )
        self.translation_log_variance = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, channels), nn.SiLU(),
            nn.Linear(channels, 3),
        )
        self.angular_log_variance = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, channels), nn.SiLU(),
            nn.Linear(channels, 1),
        )
        self.planar_fusion = nn.Sequential(
            nn.LayerNorm(width + 9), nn.Linear(width + 9, channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(channels, 2),
        )
        self.vertical_fusion = nn.Sequential(
            nn.LayerNorm(width + 7), nn.Linear(width + 7, channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(channels, 1),
        )
        self.angular_fusion = nn.Sequential(
            nn.LayerNorm(width + 5), nn.Linear(width + 5, channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(channels, 1),
        )
        self.interaction_enabled = True
        self.training_phase = "joint_calibration"

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        masked = logits.masked_fill(~available, -torch.inf)
        return torch.softmax(masked.float(), dim=1).to(logits.dtype)

    def forward(
        self,
        translation_scale_latent: torch.Tensor,
        rotation_scale_latent: torch.Tensor,
        translation_scale_available: torch.Tensor,
        rotation_scale_available: torch.Tensor,
        translation_scale_reliability: torch.Tensor,
        rotation_scale_reliability: torch.Tensor,
        *,
        rotation_condition_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if translation_scale_latent.ndim != 3 or rotation_scale_latent.ndim != 3:
            raise ValueError("factorized scale latents must have shape [B,K,C]")
        if translation_scale_latent.shape != rotation_scale_latent.shape:
            raise ValueError("translation and rotation scale latent shapes differ")
        batch, scale_count = translation_scale_latent.shape[:2]
        if translation_scale_available.shape != (batch, scale_count):
            raise ValueError("translation availability shape differs")
        if rotation_scale_available.shape != (batch, scale_count):
            raise ValueError("rotation availability shape differs")

        yaw = self.angular_state(rotation_scale_latent)
        yaw_log_variance = self.angular_log_variance(
            rotation_scale_latent,
        ).clamp(-5.0, 5.0)
        planar_pre = self.planar_base(translation_scale_latent)
        if rotation_condition_override is None:
            rotation_condition = yaw.detach()
        else:
            override = rotation_condition_override.to(
                dtype=yaw.dtype, device=yaw.device,
            )
            if override.shape == (batch,):
                override = override[:, None]
            if override.shape == (batch, 1):
                override = override[:, None].expand(-1, scale_count, -1)
            if override.shape != (batch, scale_count, 1):
                raise ValueError("rotation-condition override has the wrong shape")
            rotation_condition = override.detach()
        detached_rotation = torch.cat((
            rotation_condition, rotation_scale_reliability.detach(),
        ), dim=-1)
        planar_residual = 0.5 * self.planar_rotation_residual(torch.cat((
            translation_scale_latent, detached_rotation,
        ), dim=-1))
        if not self.interaction_enabled:
            planar_residual = torch.zeros_like(planar_residual)
        planar = torch.tanh(planar_pre + planar_residual)
        vertical = self.vertical_state(translation_scale_latent)
        translation = torch.cat((planar, vertical), dim=-1)
        translation_log_variance = self.translation_log_variance(
            translation_scale_latent,
        ).clamp(-5.0, 5.0)

        planar_fusion_feature = torch.cat((
            translation_scale_latent, planar,
            translation_log_variance[..., :2], translation_scale_reliability,
        ), dim=-1)
        vertical_fusion_feature = torch.cat((
            translation_scale_latent, vertical,
            translation_log_variance[..., 2:], translation_scale_reliability,
        ), dim=-1)
        planar_logits = self.planar_fusion(planar_fusion_feature)
        vertical_logits = self.vertical_fusion(vertical_fusion_feature)
        planar_available = translation_scale_available.unsqueeze(-1).expand(
            -1, -1, 2,
        )
        vertical_available = translation_scale_available.unsqueeze(-1)
        planar_weight = self._masked_softmax(
            planar_logits, planar_available,
        )
        vertical_weight = self._masked_softmax(
            vertical_logits, vertical_available,
        )
        angular_fusion_feature = torch.cat((
            rotation_scale_latent, yaw, yaw_log_variance,
            rotation_scale_reliability,
        ), dim=-1)
        angular_logits = self.angular_fusion(angular_fusion_feature)
        # A few equal-budget short histories contain neither two pair events nor
        # a three-edge curvature chain.  Preserve those samples: use a learned
        # zero-evidence prior at scale zero for fused yaw, while the public
        # coordinate-availability mask remains false so deep supervision skips
        # nonexistent evidence.
        angular_fusion_available = rotation_scale_available.clone()
        missing_angular = ~angular_fusion_available.any(dim=1)
        angular_fusion_available[missing_angular, 0] = True
        angular_available = angular_fusion_available.unsqueeze(-1)
        angular_weight = self._masked_softmax(angular_logits, angular_available)
        scale_state = torch.cat((translation, yaw), dim=-1)
        scale_log_variance = torch.cat((
            translation_log_variance, yaw_log_variance,
        ), dim=-1)
        scale_weight = torch.cat((planar_weight, vertical_weight, angular_weight), dim=-1)
        fused_planar = (planar_weight * planar).sum(dim=1)
        fused_vertical = (vertical_weight * vertical).sum(dim=1)
        fused_yaw = (angular_weight * yaw).sum(dim=1)
        fused = torch.cat((fused_planar, fused_vertical, fused_yaw), dim=-1)
        fused_log_variance = torch.cat((
            (planar_weight * translation_log_variance[..., :2]).sum(dim=1),
            (vertical_weight * translation_log_variance[..., 2:]).sum(dim=1),
            (angular_weight * yaw_log_variance).sum(dim=1),
        ), dim=-1)
        return {
            "motion_state_normalized": fused,
            "scale_motion_state_normalized": scale_state,
            "scale_motion_log_variance": scale_log_variance,
            "scale_motion_weight": scale_weight,
            "motion_log_variance": fused_log_variance,
            "scale_planar_base_normalized": torch.tanh(planar_pre),
            "scale_planar_rotation_residual_normalized": planar_residual,
            "rotation_condition_detached": detached_rotation,
        }


class FactorizedCommonRelativeMotionStateV7(
    StableMotionBottleneckAnonymousFutureModel
):
    """V7 state estimator with the unchanged learned future interface."""

    model_family = "factorized-common-relative-motion-state-v7"
    state_substage_schedule = {
        "angular_specialization": [1, 250],
        "translation_specialization": [251, 600],
        "joint_calibration": [601, 800],
    }

    def __init__(
        self,
        *,
        lag_scales_s: tuple[float, ...] = (0.01, 0.03, 0.07, 0.15, 0.28),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.context = FactorizedCommonRelativeMotionContext(
            channels=self.channels, dropout=self.dropout,
            message_layers=self.message_layers,
            position_scale_m=self.position_scale_m,
            history_scale_s=self.history_scale_s,
            lag_scales_s=lag_scales_s,
        )
        self.motion_state_head = FactorizedMotionStateHead(
            self.channels, self.dropout,
        )

    @property
    def config(self) -> dict[str, Any]:
        parent = dict(super().config)
        parent.update({
            "family": self.model_family,
            "motion_context": self.context.config,
            "motion_state_factorization": (
                "yaw-only relative branch -> detached conditioning -> planar; z separate"
            ),
            "state_substage_schedule": self.state_substage_schedule,
            "training_only_common_velocity_ramp": True,
            "scale_uncertainty_decoder_input": False,
            "decoder_temporal_input": "fused predicted 4D motion state only",
        })
        return parent

    @staticmethod
    def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
        module.requires_grad_(trainable)
        module.train(trainable)

    def state_branch_hashes(self) -> dict[str, str]:
        """Hash the separated trainable branches at substage boundaries."""
        angular_prefixes = (
            "context.rotation_scale_embedding",
            "context.pair_rotation_consensus.",
            "context.curvature_rotation_consensus.",
            "context.curvature_projection.",
            "context.rotation_projection.",
            "context.rotation_refinement.",
            "motion_state_head.angular_state.",
            "motion_state_head.angular_log_variance.",
            "motion_state_head.angular_fusion.",
        )
        translation_prefixes = (
            "context.translation_scale_embedding",
            "context.handle_translation_consensus.",
            "context.centroid_translation_consensus.",
            "context.translation_projection.",
            "context.translation_refinement.",
            "motion_state_head.planar_base.",
            "motion_state_head.planar_rotation_residual.",
            "motion_state_head.vertical_state.",
            "motion_state_head.translation_log_variance.",
            "motion_state_head.planar_fusion.",
            "motion_state_head.vertical_fusion.",
        )
        state = self.state_dict()
        return {
            "angular": state_dict_sha256({
                name: value for name, value in state.items()
                if any(
                    name == prefix or name.startswith(prefix)
                    for prefix in angular_prefixes
                )
            }),
            "translation_vertical": state_dict_sha256({
                name: value for name, value in state.items()
                if any(
                    name == prefix or name.startswith(prefix)
                    for prefix in translation_prefixes
                )
            }),
        }

    def set_state_training_update(self, stage_update: int) -> str:
        if not 1 <= int(stage_update) <= 800:
            raise ValueError("v7 state update must be inside 1..800")
        angular_context = (
            self.context.pair_rotation_consensus,
            self.context.curvature_rotation_consensus,
            self.context.curvature_projection,
            self.context.rotation_projection,
            *self.context.rotation_refinement,
        )
        translation_context = (
            self.context.handle_translation_consensus,
            self.context.centroid_translation_consensus,
            self.context.translation_projection,
            *self.context.translation_refinement,
        )
        angular_head = (
            self.motion_state_head.angular_state,
            self.motion_state_head.angular_log_variance,
            self.motion_state_head.angular_fusion,
        )
        translation_head = (
            self.motion_state_head.planar_base,
            self.motion_state_head.planar_rotation_residual,
            self.motion_state_head.vertical_state,
            self.motion_state_head.translation_log_variance,
            self.motion_state_head.planar_fusion,
            self.motion_state_head.vertical_fusion,
        )
        if stage_update <= 250:
            phase = "angular_specialization"
            angular_trainable, translation_trainable = True, False
            self.motion_state_head.interaction_enabled = False
        elif stage_update <= 600:
            phase = "translation_specialization"
            angular_trainable, translation_trainable = False, True
            self.motion_state_head.interaction_enabled = True
        else:
            phase = "joint_calibration"
            angular_trainable = translation_trainable = True
            self.motion_state_head.interaction_enabled = True
        for module in angular_context + angular_head:
            self._set_module_trainable(module, angular_trainable)
        for module in translation_context + translation_head:
            self._set_module_trainable(module, translation_trainable)
        self.context.rotation_scale_embedding.requires_grad_(angular_trainable)
        self.context.translation_scale_embedding.requires_grad_(translation_trainable)
        self.motion_state_head.training_phase = phase
        return phase

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
        history = self.context(
            history_obs_rel_m, history_obs_mask, history_primary_mask,
            history_event_mask, history_time_s, history_switch_step,
        )
        state = self.motion_state_head(
            history["translation_scale_latent"],
            history["rotation_scale_latent"],
            history["translation_scale_available"],
            history["rotation_scale_available"],
            history["translation_scale_reliability"],
            history["rotation_scale_reliability"],
        )
        state["motion_state_physical"] = (
            state["motion_state_normalized"] * self.motion_state_scale.to(
                state["motion_state_normalized"].dtype,
            )
        )
        return {"history": history, "state": state}

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        detach_motion_code: bool = True,
        detach_selector_context: bool = False,
    ) -> dict[str, torch.Tensor]:
        missing = set(V3_FORWARD_FIELDS) - set(batch)
        if missing:
            raise ValueError(f"v7 future forward fields missing: {sorted(missing)}")
        state_output = self.estimate_motion_state(
            history_obs_rel_m=batch["history_obs_rel_m"],
            history_obs_mask=batch["history_obs_mask"],
            history_primary_mask=batch["history_primary_mask"],
            history_event_mask=batch["history_event_mask"],
            history_time_s=batch["history_time_s"],
            history_switch_step=batch["history_switch_step"],
        )
        history, state = state_output["history"], state_output["state"]
        normalized = state["motion_state_normalized"]
        decoder_motion = normalized.detach() if detach_motion_code else normalized
        current = batch["current_position_m"]
        if current.ndim != 2 or current.shape[1] != 3:
            raise ValueError("current position must have shape [B,3]")
        tau = self._tau(batch["tau_s"], current.shape[0])
        primary = history["primary_index"]
        relative_role = torch.arange(4, device=current.device)[None]
        ordered_handle = torch.remainder(primary[:, None] + relative_role, 4)
        q0_relation = batch["q0_relation_m"].gather(
            1, ordered_handle.unsqueeze(-1).expand(-1, -1, 3),
        ).clone()
        q0_relation[:, 0] = 0.0
        q0_supported = batch["q0_supported"].gather(1, ordered_handle)
        decoded = self.decode_ordered(
            current_position_m=current, tau_s=tau,
            ordered_q0_relation_m=q0_relation,
            ordered_q0_supported=q0_supported,
            motion_state_normalized=decoder_motion,
            detach_selector_context=detach_selector_context,
        )
        return {**history, **state, **decoded}


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def factorized_motion_state_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    coordinates: str = "all",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Branch-specific state loss; reads no future or q0 decoder field."""
    target = batch.get("target_motion_state_normalized")
    scale_state = prediction.get("scale_motion_state_normalized")
    available = prediction.get("scale_coordinate_available")
    log_variance = prediction.get("scale_motion_log_variance")
    if target is None or scale_state is None or available is None or log_variance is None:
        raise ValueError("v7 loss requires supervised factorized state outputs")
    if coordinates not in {"all", "angular", "translation"}:
        raise ValueError("unknown v7 state-loss coordinate set")
    coordinate_mask = torch.ones(4, dtype=torch.bool, device=target.device)
    if coordinates == "angular":
        coordinate_mask[:3] = False
    elif coordinates == "translation":
        coordinate_mask[3] = False
    final_error = F.smooth_l1_loss(
        prediction["motion_state_normalized"], target,
        reduction="none", beta=0.1,
    )
    velocity = final_error[:, :3].mean()
    yaw_rate = final_error[:, 3].mean()
    if coordinates == "angular":
        motion = yaw_rate
    elif coordinates == "translation":
        motion = velocity
    else:
        motion = velocity + yaw_rate
    scale_error = F.smooth_l1_loss(
        scale_state, target[:, None].expand_as(scale_state),
        reduction="none", beta=0.1,
    )
    selected_available = available & coordinate_mask[None, None]
    scale_aux = _masked_mean(scale_error, selected_available)
    heteroscedastic = _masked_mean(
        scale_error * torch.exp(-log_variance)
        + 0.05 * F.softplus(log_variance),
        selected_available,
    )
    objective = motion + 0.25 * scale_aux + 0.05 * heteroscedastic
    zero = objective * 0.0
    return objective, {
        "objective": objective, "motion": motion,
        "velocity": velocity, "yaw_rate": yaw_rate,
        "scale_aux": scale_aux, "scale_heteroscedastic": heteroscedastic,
        "ramp_yaw_invariance": zero, "ramp_translation_equivariance": zero,
        "trajectory": zero, "trend": zero, "role": zero,
        "distance_risk": zero,
    }


def apply_common_velocity_ramp(
    batch: dict[str, torch.Tensor],
    velocity_ramp_mps: torch.Tensor,
    motion_state_scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply p(t)+u*t and v+u without modifying the source batch."""
    observations = batch["history_obs_rel_m"]
    if velocity_ramp_mps.shape != (observations.shape[0], 3):
        raise ValueError("velocity ramp must have shape [B,3]")
    if bool(torch.any(velocity_ramp_mps[:, 2] != 0)):
        raise ValueError("v7 common ramp is planar only")
    changed = dict(batch)
    offset = (
        batch["history_time_s"][:, :, None, None]
        * velocity_ramp_mps[:, None, None, :]
    )
    valid = (
        batch["history_event_mask"].to(torch.bool)[:, :, None]
        & batch["history_obs_mask"].to(torch.bool)
    )
    changed["history_obs_rel_m"] = torch.where(
        valid.unsqueeze(-1), observations + offset, observations,
    )
    if motion_state_scale.shape != (4,) or bool(torch.any(motion_state_scale <= 0)):
        raise ValueError("motion-state scale must be a positive four-vector")
    target = batch["target_motion_state_normalized"].clone()
    target[:, :3] = target[:, :3] + velocity_ramp_mps / motion_state_scale[:3]
    changed["target_motion_state_normalized"] = target
    if "target_motion_state_physical" in batch:
        physical = batch["target_motion_state_physical"].clone()
        physical[:, :3] = physical[:, :3] + velocity_ramp_mps
        changed["target_motion_state_physical"] = physical
    changed["velocity_ramp_mps"] = velocity_ramp_mps
    return changed


def factorized_state_train_step(
    model: FactorizedCommonRelativeMotionStateV7,
    batch: dict[str, torch.Tensor],
    stage_update: int,
    stage_total: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    """Fixed v7 three-substage state training with paired ramp constraints."""
    if int(stage_total) != 800:
        raise ValueError("v7 state schedule is fixed to 800 updates")
    phase = model.set_state_training_update(stage_update)
    fields = {
        name: batch[name] for name in (
            "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
            "history_event_mask", "history_time_s", "history_switch_step",
        )
    }
    original_output = model.estimate_motion_state(**fields)
    original = {**original_output["history"], **original_output["state"]}
    batch_size = batch["history_obs_rel_m"].shape[0]
    selected = torch.rand(batch_size, device=batch["history_obs_rel_m"].device) < 0.5
    if not bool(selected.any()):
        selected[0] = True
    ramp = torch.zeros(
        batch_size, 3, dtype=batch["history_obs_rel_m"].dtype,
        device=batch["history_obs_rel_m"].device,
    )
    ramp[selected, :2] = (
        torch.rand(int(selected.sum()), 2, device=ramp.device, dtype=ramp.dtype)
        * 1.2 - 0.6
    )
    augmented_batch = apply_common_velocity_ramp(
        batch, ramp, model.motion_state_scale.to(ramp.dtype),
    )
    augmented_output = model.estimate_motion_state(**{
        name: augmented_batch[name] for name in fields
    })
    augmented = {**augmented_output["history"], **augmented_output["state"]}
    coordinate_set = {
        "angular_specialization": "angular",
        "translation_specialization": "translation",
        "joint_calibration": "all",
    }[phase]
    original_loss, original_components = factorized_motion_state_loss(
        original, batch, coordinates=coordinate_set,
    )
    augmented_loss, augmented_components = factorized_motion_state_loss(
        augmented, augmented_batch, coordinates=coordinate_set,
    )
    yaw_invariance = F.smooth_l1_loss(
        augmented["motion_state_normalized"][selected, 3],
        original["motion_state_normalized"][selected, 3],
        beta=0.02,
    )
    normalized_ramp = ramp[:, :2] / model.motion_state_scale[:2].to(ramp.dtype)
    translation_equivariance = F.smooth_l1_loss(
        (
            augmented["motion_state_normalized"][selected, :2]
            - original["motion_state_normalized"][selected, :2]
        ),
        normalized_ramp[selected], beta=0.02,
    )
    consistency = (
        yaw_invariance if phase == "angular_specialization"
        else translation_equivariance if phase == "translation_specialization"
        else yaw_invariance + translation_equivariance
    )
    objective = 0.5 * (original_loss + augmented_loss) + 0.20 * consistency
    components = dict(original_components)
    for key in (
        "motion", "velocity", "yaw_rate", "scale_aux",
        "scale_heteroscedastic",
    ):
        components[key] = 0.5 * (
            original_components[key] + augmented_components[key]
        )
    components.update({
        "objective": objective,
        "ramp_yaw_invariance": yaw_invariance,
        "ramp_translation_equivariance": translation_equivariance,
        "state_substage": phase,
        "state_substage_endpoint": stage_update in {250, 600, 800},
    })
    return original, objective, components


def factorized_motion_future_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    **weights: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    objective, components = stable_motion_future_loss(prediction, batch, **weights)
    state_objective, state_components = factorized_motion_state_loss(
        prediction, batch,
    )
    motion_weight = float(weights.get("motion_weight", 0.0))
    objective = objective + motion_weight * (
        state_objective - state_components["motion"]
    )
    components = dict(components)
    components["scale_aux"] = state_components["scale_aux"]
    components["scale_heteroscedastic"] = state_components[
        "scale_heteroscedastic"
    ]
    components["objective"] = objective
    return objective, components


__all__ = [
    "FactorizedCommonRelativeMotionContext",
    "FactorizedMotionStateHead",
    "FactorizedCommonRelativeMotionStateV7",
    "apply_common_velocity_ramp",
    "factorized_motion_state_loss",
    "factorized_state_train_step",
    "factorized_motion_future_loss",
]
