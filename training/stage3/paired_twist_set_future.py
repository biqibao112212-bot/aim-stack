"""Anonymous geometry--velocity paired-set state estimator (V9 probe).

V8 pooled handle velocity, pair differential and geometry independently before
joining them.  V9 keeps geometry and velocity paired on every causal local
edge, then uses a permutation-invariant latent-query set network to emit one
physically consistent four-dimensional twist.  It retains the exact deployed
six-field causal boundary and never accepts identity, class, truth or future
inputs.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .anonymous_vehicle_motion_v2 import VisibilityDrivenMotionContext
from .continuous_invariant_anonymous_future import V3_FORWARD_FIELDS
from .factorized_common_relative_motion_future import apply_common_velocity_ramp
from .joint_rigid_flow_probe import (
    LOCAL_LAG_SCALES_S,
    _deterministic_probe_ramp,
)
from .stable_motion_bottleneck_future import StableMotionBottleneckAnonymousFutureModel


class _CrossAttentionBlock(nn.Module):
    """Update three expert latents from differently masked token subsets."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        if width % 4:
            raise ValueError("paired-set attention width must be divisible by four")
        self.attention = nn.MultiheadAttention(
            width, 4, dropout=dropout, batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(4 * width, width),
        )

    def forward(
        self,
        latent: torch.Tensor,
        token: torch.Tensor,
        expert_token_valid: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 3 or token.ndim != 3 or expert_token_valid.ndim != 3:
            raise ValueError("paired-set attention ranks differ")
        if latent.shape[:2] != expert_token_valid.shape[:2]:
            raise ValueError("paired-set expert masks differ")
        updated = []
        for expert in range(latent.shape[1]):
            valid = expert_token_valid[:, expert]
            if bool(torch.any(~valid.any(dim=1))):
                raise ValueError("every paired-set expert needs at least one token")
            query = latent[:, expert:expert + 1]
            attended, _ = self.attention(
                query, token, token, key_padding_mask=~valid,
                need_weights=False,
            )
            value = self.attention_norm(query + attended)
            value = value + self.feed_forward(value)
            updated.append(value)
        return torch.cat(updated, dim=1)


class AnonymousPairedTwistTokenContext(nn.Module):
    """Build local handle and same-set pair tokens without early pooling."""

    model_family = "anonymous-paired-twist-token-context-v9-probe"

    def __init__(
        self,
        *,
        channels: int = 96,
        dropout: float = 0.05,
        message_layers: int = 3,
        position_scale_m: float = 1.0,
        history_scale_s: float = 0.32,
        lag_scales_s: tuple[float, ...] = LOCAL_LAG_SCALES_S,
    ) -> None:
        super().__init__()
        numeric_scales = tuple(float(value) for value in lag_scales_s)
        if numeric_scales != LOCAL_LAG_SCALES_S:
            raise ValueError("v9 probe is fixed to local 10/30/70-ms scales")
        if channels < 32 or message_layers < 2:
            raise ValueError("invalid paired-set capacity")
        if position_scale_m <= 0 or history_scale_s <= 0:
            raise ValueError("paired-set scales must be positive")
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.message_layers = int(message_layers)
        self.position_scale_m = float(position_scale_m)
        self.history_scale_s = float(history_scale_s)
        self.width = 2 * self.channels
        self.register_buffer(
            "lag_scales_s", torch.tensor(numeric_scales, dtype=torch.float32),
        )
        self.scale_embedding = nn.Parameter(
            torch.empty(len(numeric_scales), self.width),
        )
        self.type_embedding = nn.Parameter(torch.empty(2, self.width))
        nn.init.normal_(self.scale_embedding, std=0.02)
        nn.init.normal_(self.type_embedding, std=0.02)
        # Handle geometry (12) stays separate from kinematics (14) so the
        # validation-only pairing intervention can permute only geometry.
        self.handle_encoder = nn.Sequential(
            nn.Linear(26, self.width), nn.LayerNorm(self.width), nn.SiLU(),
            nn.Linear(self.width, self.width), nn.SiLU(),
        )
        # Pair geometry (12) plus local differential/time evidence (13).
        self.pair_encoder = nn.Sequential(
            nn.Linear(25, self.width), nn.LayerNorm(self.width), nn.SiLU(),
            nn.Linear(self.width, self.width), nn.SiLU(),
        )

    @property
    def config(self) -> dict[str, Any]:
        return {
            "family": self.model_family,
            "channels": self.channels,
            "token_width": self.width,
            "dropout": self.dropout,
            "message_layers": self.message_layers,
            "position_scale_m": self.position_scale_m,
            "history_scale_s": self.history_scale_s,
            "lag_scales_s": self.lag_scales_s.detach().cpu().tolist(),
            "handle_token": "paired endpoint geometry and same-handle local velocity",
            "pair_token": "current-primary-oriented same-set local pair differential",
            "early_geometry_velocity_pooling": False,
            "long_projective_yaw_lags": False,
            "curvature_fallback": False,
            "handle_embedding": False,
            "physical_id_input": False,
            "session_or_motion_class_input": False,
            "absolute_position_or_range_input": False,
            "q0_geometry_or_quality_input": False,
            "truth_or_future_input": False,
            "token_set_permutation_invariant": True,
        }

    @staticmethod
    def _last_active(mask: torch.Tensor) -> torch.Tensor:
        return VisibilityDrivenMotionContext._last_active(mask)

    def _lag_bank(
        self,
        value: torch.Tensor,
        time_s: torch.Tensor,
        valid_event: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        streams, events, dimensions = value.shape
        if time_s.shape != (streams, events) or valid_event.shape != (streams, events):
            raise ValueError("paired-set lag fields differ")
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
        prior = value[:, None].expand(-1, events, -1, -1).gather(
            2, selected_safe.unsqueeze(-1).expand(-1, -1, -1, dimensions),
        )
        prior_time = time_s[:, None].expand(-1, events, -1).gather(
            2, selected_safe,
        )
        elapsed = torch.where(
            edge_valid, time_s[:, :, None] - prior_time,
            torch.zeros_like(prior_time),
        )
        delta = torch.where(
            edge_valid.unsqueeze(-1), value[:, :, None] - prior,
            torch.zeros_like(prior),
        )
        selected = torch.where(edge_valid, selected, torch.full_like(selected, -1))
        return delta, elapsed, edge_valid, selected

    def _same_set_pair_bank(
        self,
        pair: torch.Tensor,
        time_s: torch.Tensor,
        valid: torch.Tensor,
        visible: torch.Tensor,
        primary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, events, dimensions = pair.shape
        current_index = torch.arange(events, device=pair.device)[None, :, None]
        prior_index = torch.arange(events, device=pair.device)[None, None, :]
        elapsed_all = time_s[:, :, None] - time_s[:, None, :]
        same_set = (visible[:, :, None] == visible[:, None, :]).all(dim=-1)
        causal = (
            valid[:, :, None] & valid[:, None, :] & same_set
            & (prior_index < current_index) & (elapsed_all > 1e-7)
        )
        scales = self.lag_scales_s.to(dtype=pair.dtype, device=pair.device)
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
        prior = pair[:, None].expand(-1, events, -1, -1).gather(
            2, selected_safe.unsqueeze(-1).expand(-1, -1, -1, dimensions),
        )
        prior_primary = primary[:, None].expand(-1, events, -1, -1).gather(
            2, selected_safe.unsqueeze(-1).expand(-1, -1, -1, 4),
        )
        current_primary = primary[:, :, None].expand(-1, -1, len(scales), -1)
        same_primary = (current_primary & prior_primary).any(dim=-1)
        prior = torch.where(same_primary.unsqueeze(-1), prior, -prior)
        prior_time = time_s[:, None].expand(-1, events, -1).gather(
            2, selected_safe,
        )
        elapsed = torch.where(
            edge_valid, time_s[:, :, None] - prior_time,
            torch.zeros_like(prior_time),
        )
        selected = torch.where(edge_valid, selected, torch.full_like(selected, -1))
        return prior, elapsed, edge_valid, selected

    @staticmethod
    def _gather_event(
        value: torch.Tensor, selected: torch.Tensor,
    ) -> torch.Tensor:
        # value [S,T,F], selected [S,T,K].
        dimensions = value.shape[-1]
        return value[:, None].expand(-1, value.shape[1], -1, -1).gather(
            2, selected.clamp_min(0).unsqueeze(-1).expand(-1, -1, -1, dimensions),
        )

    @staticmethod
    def _roll_grouped_geometry(
        geometry: torch.Tensor, valid: torch.Tensor,
    ) -> torch.Tensor:
        """Roll time within each anonymous stream and scale independently.

        Geometry type, scale, support and every kinematic/time field remain in
        their original marginal group.  Applying the same rule independently
        to every temporary handle makes the intervention invariant to cyclic or
        reflected handle relabelling.
        """
        if geometry.ndim != 5 or valid.shape != geometry.shape[:-1]:
            raise ValueError("grouped pairing intervention fields differ")
        result = geometry.clone()
        for row in range(geometry.shape[0]):
            for stream in range(geometry.shape[1]):
                for scale in range(geometry.shape[3]):
                    index = torch.nonzero(
                        valid[row, stream, :, scale], as_tuple=False,
                    ).flatten()
                    if index.numel() > 1:
                        result[row, stream, index, scale] = geometry[
                            row, stream, torch.roll(index, 1), scale,
                        ]
        return result

    def _validate_inputs(
        self,
        history_obs_rel_m: torch.Tensor,
        history_obs_mask: torch.Tensor,
        history_primary_mask: torch.Tensor,
        history_event_mask: torch.Tensor,
        history_time_s: torch.Tensor,
        history_switch_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            raise ValueError("history scalar fields have the wrong shape")
        active = history_event_mask.to(torch.bool)
        visible = history_obs_mask.to(torch.bool) & active.unsqueeze(-1)
        primary = history_primary_mask.to(torch.bool) & active.unsqueeze(-1)
        if bool(torch.any(active.sum(dim=1) < 8)):
            raise ValueError("paired-set context requires eight active events")
        if bool(torch.any((visible.sum(dim=2)[active] < 1) | (visible.sum(dim=2)[active] > 2))):
            raise ValueError("active events require one or two observations")
        if bool(torch.any(primary.sum(dim=2)[active] != 1)):
            raise ValueError("active events require one primary")
        if bool(torch.any(primary & ~visible)):
            raise ValueError("primary must be visible")
        if bool(torch.any(active & (history_time_s > 1e-6))):
            raise ValueError("history cannot contain future events")
        if bool(torch.any(active & ~torch.isfinite(history_time_s))):
            raise ValueError("history time must be finite")
        if bool(torch.any(visible & ~torch.isfinite(history_obs_rel_m).all(dim=-1))):
            raise ValueError("visible coordinates must be finite")
        valid_switch = torch.isin(
            history_switch_step.to(torch.long),
            history_switch_step.new_tensor((-1, 0, 1), dtype=torch.long),
        )
        if bool(torch.any(active & ~valid_switch)):
            raise ValueError("history switch must be -1, 0 or +1")
        last = self._last_active(active)
        if bool(torch.any(history_time_s.gather(1, last[:, None]).abs() > 1e-6)):
            raise ValueError("last active event must be q0")
        return active, visible, primary

    def _tokens(
        self,
        history_obs_rel_m: torch.Tensor,
        history_obs_mask: torch.Tensor,
        history_primary_mask: torch.Tensor,
        history_event_mask: torch.Tensor,
        history_time_s: torch.Tensor,
        history_switch_step: torch.Tensor,
        *,
        break_pairing: bool,
    ) -> dict[str, torch.Tensor]:
        active, visible, primary = self._validate_inputs(
            history_obs_rel_m, history_obs_mask, history_primary_mask,
            history_event_mask, history_time_s, history_switch_step,
        )
        batch, events = active.shape
        scale_count = len(LOCAL_LAG_SCALES_S)
        clean_obs = torch.where(
            visible.unsqueeze(-1), history_obs_rel_m, torch.zeros_like(history_obs_rel_m),
        )
        clean_time = torch.where(active, history_time_s, torch.zeros_like(history_time_s))
        clean_switch = torch.where(
            active, history_switch_step, torch.zeros_like(history_switch_step),
        )
        cumulative = torch.cumsum(clean_switch, dim=1)
        visible_count = visible.sum(dim=2).clamp_min(1)
        centroid = clean_obs.sum(dim=2) / visible_count.unsqueeze(-1)
        centered = torch.where(
            visible.unsqueeze(-1), clean_obs - centroid[:, :, None],
            torch.zeros_like(clean_obs),
        )

        # Keep handle identity only long enough to create a same-stream edge;
        # it is removed before the permutation-invariant token pool.
        streams = batch * 4
        flat_obs = clean_obs.permute(0, 2, 1, 3).reshape(streams, events, 3)
        flat_centered = centered.permute(0, 2, 1, 3).reshape(streams, events, 3)
        flat_valid = visible.permute(0, 2, 1).reshape(streams, events)
        flat_primary = primary.permute(0, 2, 1).reshape(streams, events)
        flat_time = clean_time[:, None].expand(-1, 4, -1).reshape(streams, events)
        flat_cumulative = cumulative[:, None].expand(-1, 4, -1).reshape(
            streams, events,
        )
        flat_count = visible_count[:, None].expand(-1, 4, -1).reshape(streams, events)
        delta, elapsed, handle_valid, handle_prior_index = self._lag_bank(
            flat_obs, flat_time, flat_valid,
        )
        handle_prior_safe = handle_prior_index.clamp_min(0)
        prior_obs = self._gather_event(flat_obs, handle_prior_safe)
        prior_centered = self._gather_event(flat_centered, handle_prior_safe)
        prior_primary = self._gather_event(
            flat_primary.to(flat_obs.dtype).unsqueeze(-1), handle_prior_safe,
        ).squeeze(-1)
        prior_count = self._gather_event(
            flat_count.to(flat_obs.dtype).unsqueeze(-1), handle_prior_safe,
        ).squeeze(-1)
        prior_cumulative = self._gather_event(
            flat_cumulative.to(flat_obs.dtype).unsqueeze(-1), handle_prior_safe,
        ).squeeze(-1)
        current_obs = flat_obs[:, :, None].expand(-1, -1, scale_count, -1)
        current_centered = flat_centered[:, :, None].expand_as(current_obs)
        velocity = torch.where(
            handle_valid.unsqueeze(-1), delta / elapsed.clamp_min(1e-7).unsqueeze(-1),
            torch.zeros_like(delta),
        )
        scales = self.lag_scales_s.to(dtype=flat_obs.dtype, device=flat_obs.device)
        handle_geometry = torch.cat((
            current_centered / self.position_scale_m,
            prior_centered / self.position_scale_m,
            current_obs / self.position_scale_m,
            prior_obs / self.position_scale_m,
        ), dim=-1)
        handle_kinematics = torch.cat((
            delta / self.position_scale_m,
            velocity * (self.history_scale_s / self.position_scale_m),
            torch.log1p(elapsed / 0.01).unsqueeze(-1),
            (flat_time[:, :, None] / self.history_scale_s).unsqueeze(-1).expand(
                -1, -1, scale_count, -1,
            ),
            ((flat_cumulative[:, :, None] - prior_cumulative).abs() / 6.0).unsqueeze(-1),
            flat_primary[:, :, None, None].to(flat_obs.dtype).expand(
                -1, -1, scale_count, -1,
            ),
            prior_primary.unsqueeze(-1),
            (elapsed / scales[None, None]).unsqueeze(-1),
            (flat_count[:, :, None].to(flat_obs.dtype) / 2.0).unsqueeze(-1).expand(
                -1, -1, scale_count, -1,
            ),
            (prior_count / 2.0).unsqueeze(-1),
        ), dim=-1)
        handle_geometry = handle_geometry.reshape(
            batch, 4, events, scale_count, 12,
        )
        handle_kinematics = handle_kinematics.reshape(
            batch, 4, events, scale_count, 14,
        )
        handle_valid_grouped = handle_valid.reshape(batch, 4, events, scale_count)
        if break_pairing:
            handle_geometry = self._roll_grouped_geometry(
                handle_geometry, handle_valid_grouped,
            )
        handle_geometry = handle_geometry.reshape(batch, -1, 12)
        handle_kinematics = handle_kinematics.reshape(batch, -1, 14)
        handle_token_valid = handle_valid_grouped.reshape(batch, -1)
        handle_token = self.handle_encoder(torch.cat((
            handle_geometry, handle_kinematics,
        ), dim=-1))
        handle_scale = self.scale_embedding[None, None].expand(
            streams, events, -1, -1,
        ).reshape(batch, -1, self.width)
        handle_token = handle_token + handle_scale + self.type_embedding[0]
        handle_token = torch.where(
            handle_token_valid.unsqueeze(-1), handle_token,
            torch.zeros_like(handle_token),
        )

        # Primary-oriented pair vectors preserve sign without permanent IDs.
        primary_position = (
            clean_obs * primary.unsqueeze(-1).to(clean_obs.dtype)
        ).sum(dim=2)
        secondary = visible & ~primary
        secondary_position = (
            clean_obs * secondary.unsqueeze(-1).to(clean_obs.dtype)
        ).sum(dim=2)
        pair_event = (visible.sum(dim=2) == 2) & (secondary.sum(dim=2) == 1)
        pair = torch.where(
            pair_event.unsqueeze(-1), secondary_position - primary_position,
            torch.zeros_like(primary_position),
        )
        pair_prior, pair_elapsed, pair_valid, pair_prior_index = (
            self._same_set_pair_bank(
                pair, clean_time, pair_event, visible, primary,
            )
        )
        current_pair = pair[:, :, None].expand_as(pair_prior)
        current_norm = current_pair.norm(dim=-1)
        prior_norm = pair_prior.norm(dim=-1)
        current_unit = current_pair / current_norm.clamp_min(1e-6).unsqueeze(-1)
        prior_unit = pair_prior / prior_norm.clamp_min(1e-6).unsqueeze(-1)
        pair_delta = torch.where(
            pair_valid.unsqueeze(-1), current_pair - pair_prior,
            torch.zeros_like(current_pair),
        )
        pair_rate = torch.where(
            pair_valid.unsqueeze(-1),
            pair_delta / pair_elapsed.clamp_min(1e-7).unsqueeze(-1),
            torch.zeros_like(pair_delta),
        )
        pair_prior_safe = pair_prior_index.clamp_min(0)
        pair_prior_cumulative = cumulative[:, None].expand(-1, events, -1).gather(
            2, pair_prior_safe,
        )
        prior_primary_mask = primary[:, None].expand(-1, events, -1, -1).gather(
            2, pair_prior_safe.unsqueeze(-1).expand(-1, -1, -1, 4),
        )
        current_primary_mask = primary[:, :, None].expand(
            -1, -1, scale_count, -1,
        )
        primary_swap = ~(
            current_primary_mask & prior_primary_mask
        ).any(dim=-1)
        pair_geometry = torch.cat((
            current_pair / self.position_scale_m,
            pair_prior / self.position_scale_m,
            current_unit,
            prior_unit,
        ), dim=-1)
        pair_kinematics = torch.cat((
            pair_delta / self.position_scale_m,
            pair_rate * (self.history_scale_s / self.position_scale_m),
            (current_norm / self.position_scale_m).unsqueeze(-1),
            (prior_norm / self.position_scale_m).unsqueeze(-1),
            torch.log1p(pair_elapsed / 0.01).unsqueeze(-1),
            (clean_time[:, :, None, None] / self.history_scale_s).expand(
                -1, -1, scale_count, -1,
            ),
            ((cumulative[:, :, None] - pair_prior_cumulative).abs() / 6.0).unsqueeze(-1),
            primary_swap.to(clean_obs.dtype).unsqueeze(-1),
            (pair_elapsed / scales[None, None]).unsqueeze(-1),
        ), dim=-1)
        pair_geometry = pair_geometry.reshape(
            batch, 1, events, scale_count, 12,
        )
        pair_kinematics = pair_kinematics.reshape(
            batch, 1, events, scale_count, 13,
        )
        pair_valid_grouped = pair_valid[:, None]
        if break_pairing:
            pair_geometry = self._roll_grouped_geometry(
                pair_geometry, pair_valid_grouped,
            )
        pair_geometry = pair_geometry.reshape(batch, -1, 12)
        pair_kinematics = pair_kinematics.reshape(batch, -1, 13)
        pair_token_valid = pair_valid_grouped.reshape(batch, -1)
        pair_token = self.pair_encoder(torch.cat((
            pair_geometry, pair_kinematics,
        ), dim=-1))
        pair_scale = self.scale_embedding[None, None].expand(
            batch, events, -1, -1,
        ).reshape(batch, -1, self.width)
        pair_token = pair_token + pair_scale + self.type_embedding[1]
        pair_token = torch.where(
            pair_token_valid.unsqueeze(-1), pair_token, torch.zeros_like(pair_token),
        )

        token = torch.cat((handle_token, pair_token), dim=1)
        token_valid = torch.cat((handle_token_valid, pair_token_valid), dim=1)
        pair_scale_available = pair_valid.any(dim=1)
        handle_endpoint_time = flat_time[:, :, None].expand(
            -1, -1, scale_count,
        ).reshape(batch, -1)
        pair_endpoint_time = clean_time[:, :, None].expand(
            -1, -1, scale_count,
        ).reshape(batch, -1)
        endpoint_time = torch.cat((handle_endpoint_time, pair_endpoint_time), dim=1)
        local_token_valid = token_valid & (endpoint_time >= -0.105)
        local_available = local_token_valid.any(dim=1)
        if bool(torch.any(~local_available)):
            raise ValueError("paired-set local expert lacks recent causal support")
        any_pair = pair_scale_available.any(dim=1)
        steady_available = (
            (active.sum(dim=1) == 32) & pair_scale_available.all(dim=1)
        )
        fallback_available = ~any_pair
        last = self._last_active(active)
        rows = torch.arange(batch, device=active.device)
        stats = torch.stack((
            active.sum(dim=1).to(clean_obs.dtype) / 32.0,
            pair_scale_available.sum(dim=1).to(clean_obs.dtype) / scale_count,
            pair_scale_available.any(dim=1).to(clean_obs.dtype),
            handle_token_valid.sum(dim=1).to(clean_obs.dtype) / 128.0,
            pair_token_valid.sum(dim=1).to(clean_obs.dtype) / 32.0,
            pair_event[rows, last].to(clean_obs.dtype),
            visible[rows, last].sum(dim=1).to(clean_obs.dtype) / 2.0,
        ), dim=-1)
        return {
            "_token": token,
            "_token_valid": token_valid,
            "_local_token_valid": local_token_valid,
            "_steady_token_valid": token_valid,
            "_handle_token_valid": torch.cat((
                handle_token_valid, torch.zeros_like(pair_token_valid),
            ), dim=1),
            "_pair_token_valid": torch.cat((
                torch.zeros_like(handle_token_valid), pair_token_valid,
            ), dim=1),
            "_expert_available": torch.stack((
                local_available, steady_available, fallback_available,
            ), dim=1),
            "_router_stats": stats,
            "history_active_count": active.sum(dim=1),
            "primary_index": primary[rows, last].to(torch.long).argmax(dim=1),
            "pair_flow_available": pair_scale_available,
            "pair_flow_edge_valid": pair_valid,
            "current_pair_available": pair_event[rows, last],
            "paired_token_count": token_valid.sum(dim=1),
        }

    def forward(
        self,
        history_obs_rel_m: torch.Tensor,
        history_obs_mask: torch.Tensor,
        history_primary_mask: torch.Tensor,
        history_event_mask: torch.Tensor,
        history_time_s: torch.Tensor,
        history_switch_step: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self._tokens(
            history_obs_rel_m, history_obs_mask, history_primary_mask,
            history_event_mask, history_time_s, history_switch_step,
            break_pairing=False,
        )

    def forward_broken_pairing(
        self,
        history_obs_rel_m: torch.Tensor,
        history_obs_mask: torch.Tensor,
        history_primary_mask: torch.Tensor,
        history_event_mask: torch.Tensor,
        history_time_s: torch.Tensor,
        history_switch_step: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self._tokens(
            history_obs_rel_m, history_obs_mask, history_primary_mask,
            history_event_mask, history_time_s, history_switch_step,
            break_pairing=True,
        )


class PairedTwistSetHead(nn.Module):
    """Fuse local, steady-history and handle-only experts with one twist weight."""

    def __init__(self, channels: int, dropout: float, message_layers: int) -> None:
        super().__init__()
        self.width = 2 * int(channels)
        self.expert_query = nn.Parameter(torch.empty(3, self.width))
        nn.init.normal_(self.expert_query, std=0.02)
        self.blocks = nn.ModuleList(
            _CrossAttentionBlock(self.width, dropout)
            for _ in range(int(message_layers))
        )
        self.proposal = nn.Sequential(
            nn.LayerNorm(self.width), nn.Linear(self.width, self.width), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(self.width, 4), nn.Tanh(),
        )
        self.log_variance = nn.Sequential(
            nn.LayerNorm(self.width), nn.Linear(self.width, channels), nn.SiLU(),
            nn.Linear(channels, 4),
        )
        self.router = nn.Sequential(
            nn.LayerNorm(3 * self.width + 7),
            nn.Linear(3 * self.width + 7, channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(channels, 3),
        )

    def forward(
        self,
        token: torch.Tensor,
        token_valid: torch.Tensor,
        local_valid: torch.Tensor,
        steady_valid: torch.Tensor,
        handle_valid: torch.Tensor,
        expert_available: torch.Tensor,
        router_stats: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = token.shape[0]
        if bool(torch.any(~token_valid.any(dim=1))):
            raise ValueError("paired-set head needs causal tokens")
        # Local reads recent complete evidence; steady reads the complete
        # handle+pair history only under history32/pair3 support; fallback reads
        # handle evidence only and is enabled only when all pair support is absent.
        if expert_available.shape != (batch, 3):
            raise ValueError("paired-set expert availability differs")
        safe_steady = torch.where(
            expert_available[:, 1, None], steady_valid, token_valid,
        )
        safe_fallback = torch.where(
            expert_available[:, 2, None], handle_valid, token_valid,
        )
        expert_valid = torch.stack((local_valid, safe_steady, safe_fallback), dim=1)
        latent = self.expert_query[None].expand(batch, -1, -1)
        for block in self.blocks:
            latent = block(latent, token, expert_valid)
        proposal = self.proposal(latent)
        log_variance = self.log_variance(latent).clamp(-5.0, 5.0)
        logits = self.router(torch.cat((latent.flatten(1), router_stats), dim=-1))
        logits = logits.masked_fill(~expert_available, -torch.inf)
        weight = torch.softmax(logits.float(), dim=1).to(proposal.dtype)
        state = (weight.unsqueeze(-1) * proposal).sum(dim=1)
        state_log_variance = (weight.unsqueeze(-1) * log_variance).sum(dim=1)
        return {
            "motion_state_normalized": state,
            "motion_log_variance": state_log_variance,
            "expert_motion_state_normalized": proposal,
            "expert_motion_log_variance": log_variance,
            "expert_weight": weight,
            "expert_available": expert_available,
        }


class AnonymousPairedTwistSetProbe(StableMotionBottleneckAnonymousFutureModel):
    """V9 probe with the unchanged learned future interface frozen downstream."""

    model_family = "anonymous-paired-twist-set-probe-v9"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.context = AnonymousPairedTwistTokenContext(
            channels=self.channels, dropout=self.dropout,
            message_layers=self.message_layers,
            position_scale_m=self.position_scale_m,
            history_scale_s=self.history_scale_s,
        )
        self.motion_state_head = PairedTwistSetHead(
            self.channels, self.dropout, self.message_layers,
        )

    @property
    def config(self) -> dict[str, Any]:
        parent = dict(super().config)
        parent.update({
            "family": self.model_family,
            "motion_context": self.context.config,
            "motion_state_fusion": (
                "permutation-invariant local/steady/handle latent queries with "
                "one shared expert weight for the complete 4D twist"
            ),
            "per_coordinate_scale_selection": False,
            "decoder_temporal_input": "fused predicted 4D motion state only",
            "physical_id_input": False,
            "motion_class_input": False,
            "session_identity_input": False,
            "truth_state_input": False,
        })
        return parent

    @staticmethod
    def _field_names() -> tuple[str, ...]:
        return (
            "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
            "history_event_mask", "history_time_s", "history_switch_step",
        )

    def _estimate(
        self, fields: dict[str, torch.Tensor], *, break_pairing: bool,
    ) -> dict[str, dict[str, torch.Tensor]]:
        history = (
            self.context.forward_broken_pairing(**fields)
            if break_pairing else self.context(**fields)
        )
        state = self.motion_state_head(
            history["_token"], history["_token_valid"],
            history["_local_token_valid"], history["_steady_token_valid"],
            history["_handle_token_valid"], history["_expert_available"],
            history["_router_stats"],
        )
        state["motion_state_physical"] = (
            state["motion_state_normalized"] * self.motion_state_scale.to(
                state["motion_state_normalized"].dtype,
            )
        )
        public_history = {
            name: value for name, value in history.items() if not name.startswith("_")
        }
        return {"history": public_history, "state": state}

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
        return self._estimate({
            "history_obs_rel_m": history_obs_rel_m,
            "history_obs_mask": history_obs_mask,
            "history_primary_mask": history_primary_mask,
            "history_event_mask": history_event_mask,
            "history_time_s": history_time_s,
            "history_switch_step": history_switch_step,
        }, break_pairing=False)

    def estimate_motion_state_broken_pairing(
        self,
        *,
        history_obs_rel_m: torch.Tensor,
        history_obs_mask: torch.Tensor,
        history_primary_mask: torch.Tensor,
        history_event_mask: torch.Tensor,
        history_time_s: torch.Tensor,
        history_switch_step: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        return self._estimate({
            "history_obs_rel_m": history_obs_rel_m,
            "history_obs_mask": history_obs_mask,
            "history_primary_mask": history_primary_mask,
            "history_event_mask": history_event_mask,
            "history_time_s": history_time_s,
            "history_switch_step": history_switch_step,
        }, break_pairing=True)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        detach_motion_code: bool = True,
        detach_selector_context: bool = False,
    ) -> dict[str, torch.Tensor]:
        missing = set(V3_FORWARD_FIELDS) - set(batch)
        if missing:
            raise ValueError(f"v9 probe future fields missing: {sorted(missing)}")
        output = self.estimate_motion_state(**{
            name: batch[name] for name in self._field_names()
        })
        history, state = output["history"], output["state"]
        motion = state["motion_state_normalized"]
        decoder_motion = motion.detach() if detach_motion_code else motion
        current = batch["current_position_m"]
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
        return {**history, **state, **decoded}


def paired_twist_state_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise only the unified twist; never read future or q0 fields."""
    target = batch.get("target_motion_state_normalized")
    predicted = prediction.get("motion_state_normalized")
    log_variance = prediction.get("motion_log_variance")
    if target is None or predicted is None or log_variance is None:
        raise ValueError("v9 state loss requires unified twist and uncertainty")
    coordinate = F.smooth_l1_loss(
        predicted, target, reduction="none", beta=0.1,
    )
    planar = coordinate[:, :2].mean()
    vertical = coordinate[:, 2].mean()
    yaw = coordinate[:, 3].mean()
    velocity = planar + 0.25 * vertical
    motion = velocity + yaw
    heteroscedastic = (
        coordinate * torch.exp(-log_variance)
        + 0.05 * F.softplus(log_variance)
    ).mean()
    objective = motion + 0.05 * heteroscedastic
    zero = objective.new_zeros(())
    return objective, {
        "objective": objective,
        "motion": motion,
        "velocity": velocity,
        "yaw_rate": yaw,
        "planar_velocity": planar,
        "vertical_velocity": vertical,
        "scale_aux": zero,
        "scale_heteroscedastic": heteroscedastic,
        "trajectory": zero,
        "trend": zero,
        "role": zero,
        "distance_risk": zero,
    }


def paired_twist_probe_train_step(
    model: StableMotionBottleneckAnonymousFutureModel,
    batch: dict[str, torch.Tensor],
    stage_update: int,
    stage_total: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    """Fixed 200-update paired-set loss with variant-independent ramp draws."""
    if int(stage_total) != 200 or not 1 <= int(stage_update) <= 200:
        raise ValueError("v9 structural probe is fixed to 200 updates")
    field_names = AnonymousPairedTwistSetProbe._field_names()
    original_output = model.estimate_motion_state(**{
        name: batch[name] for name in field_names
    })
    original = {**original_output["history"], **original_output["state"]}
    selected, ramp = _deterministic_probe_ramp(
        batch["history_obs_rel_m"].shape[0],
        dtype=batch["history_obs_rel_m"].dtype,
        device=batch["history_obs_rel_m"].device,
        stage_update=stage_update,
    )
    augmented_batch = apply_common_velocity_ramp(
        batch, ramp, model.motion_state_scale.to(ramp.dtype),
    )
    augmented_output = model.estimate_motion_state(**{
        name: augmented_batch[name] for name in field_names
    })
    augmented = {**augmented_output["history"], **augmented_output["state"]}
    original_loss, original_components = paired_twist_state_loss(original, batch)
    augmented_loss, augmented_components = paired_twist_state_loss(
        augmented, augmented_batch,
    )
    yaw_invariance = F.smooth_l1_loss(
        augmented["motion_state_normalized"][selected, 3],
        original["motion_state_normalized"][selected, 3], beta=0.02,
    )
    normalized_ramp = ramp[:, :2] / model.motion_state_scale[:2].to(ramp.dtype)
    translation_equivariance = F.smooth_l1_loss(
        augmented["motion_state_normalized"][selected, :2]
        - original["motion_state_normalized"][selected, :2],
        normalized_ramp[selected], beta=0.02,
    )
    objective = 0.5 * (original_loss + augmented_loss) + 0.20 * (
        yaw_invariance + translation_equivariance
    )
    components = dict(original_components)
    for name in (
        "motion", "velocity", "yaw_rate", "planar_velocity",
        "vertical_velocity", "scale_heteroscedastic",
    ):
        components[name] = 0.5 * (
            original_components[name] + augmented_components[name]
        )
    components.update({
        "objective": objective,
        "ramp_yaw_invariance": yaw_invariance,
        "ramp_translation_equivariance": translation_equivariance,
        "state_substage": "paired_twist_structural_probe",
        "state_substage_endpoint": False,
    })
    return original, objective, components


__all__ = [
    "AnonymousPairedTwistTokenContext",
    "PairedTwistSetHead",
    "AnonymousPairedTwistSetProbe",
    "paired_twist_state_loss",
    "paired_twist_probe_train_step",
]
