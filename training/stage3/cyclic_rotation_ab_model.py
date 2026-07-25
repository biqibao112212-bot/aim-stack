"""Rotation-only V21 F-layer alternatives with deterministic direction.

Both alternatives consume the frozen V19 S state.  Rotation direction is a
causal geometric fact derived from visible history; it is neither a learned
output nor a supervised classification task.
"""

from __future__ import annotations

import torch
from torch import nn

from .cyclic_future_model import MotionEvidenceEncoder, decode_rotation
from .cyclic_track_model import CyclicMessageBlock, _expanded_tau


def _cross_xy(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def project_direct_planar_rigid(
    q0_m: torch.Tensor,
    raw_position_m: torch.Tensor,
    track_mask: torch.Tensor,
) -> torch.Tensor:
    """Project direct query predictions to one center-free planar rigid pose.

    The projection is a weighted 2-D Procrustes fit against each sample's own
    q0 tracks.  It exposes no center, radius, phase, velocity or yaw rate.  A
    one-track state keeps the directly predicted translation and uses identity
    rotation; two or more tracks determine the closest proper rotation.
    """
    if q0_m.ndim != 3 or q0_m.shape[1:] != (4, 3):
        raise ValueError("q0_m must have shape [B,4,3]")
    if raw_position_m.ndim != 4 or raw_position_m.shape[2:] != (4, 3):
        raise ValueError("raw_position_m must have shape [B,Q,4,3]")
    if track_mask.shape != q0_m.shape[:2]:
        raise ValueError("track_mask must have shape [B,4]")
    weight = track_mask.to(q0_m.dtype)[:, None, :, None]
    count = weight.sum(dim=2, keepdim=True).clamp_min(1.0)
    source = q0_m[:, None, :, :2].expand(-1, raw_position_m.shape[1], -1, -1)
    target = raw_position_m[..., :2]
    source_centroid = (weight * source).sum(dim=2, keepdim=True) / count
    target_centroid = (weight * target).sum(dim=2, keepdim=True) / count
    centered_source = source - source_centroid
    centered_target = target - target_centroid
    dot = (weight * centered_source * centered_target).sum(dim=(2, 3))
    cross = (weight.squeeze(-1) * _cross_xy(
        centered_source, centered_target,
    )).sum(dim=2)
    squared_norm = dot.square() + cross.square()
    norm = torch.sqrt(squared_norm.clamp_min(1e-12))
    enough = (track_mask.sum(dim=1) >= 2)[:, None] & (squared_norm > 1e-12)
    cosine = torch.where(enough, dot / norm, torch.ones_like(dot))
    sine = torch.where(enough, cross / norm, torch.zeros_like(cross))
    x = centered_source[..., 0]
    y = centered_source[..., 1]
    rotated = torch.stack((
        cosine[:, :, None] * x - sine[:, :, None] * y,
        sine[:, :, None] * x + cosine[:, :, None] * y,
    ), dim=-1)
    projected_xy = rotated + target_centroid
    projected_z = q0_m[:, None, :, 2:3].expand(
        -1, raw_position_m.shape[1], -1, -1,
    )
    return torch.cat((projected_xy, projected_z), dim=-1)


def deterministic_rotation_direction(
    obs: torch.Tensor,
    obs_mask: torch.Tensor,
    event_mask: torch.Tensor,
    event_time_s: torch.Tensor,
    position_mean: torch.Tensor,
    position_std: torch.Tensor,
    *,
    minimum_score: float = 1e-5,
    maximum_edge_gap_s: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return causal planar rotation sign, validity and evidence source.

    Source 1 is the signed change of a co-visible adjacent directed edge.
    Source 2 is signed curvature of one repeatedly visible temporary track.
    Edge evidence wins because common translation cancels from it.  No future
    query, target, center, radius, phase, motion class or learned parameter is
    used.  Samples without enough history remain invalid instead of guessing.
    """
    if obs.ndim != 4 or obs.shape[2:] != (4, 3):
        raise ValueError("obs must have shape [B,T,4,3]")
    if (
        obs_mask.shape != obs.shape[:3]
        or event_mask.shape != obs.shape[:2]
        or event_time_s.shape != obs.shape[:2]
    ):
        raise ValueError("direction masks do not match observation history")
    if minimum_score <= 0 or maximum_edge_gap_s <= 0:
        raise ValueError("direction thresholds must be positive")
    visible = obs_mask.to(torch.bool) & event_mask.to(torch.bool).unsqueeze(-1)
    physical = obs * position_std.view(1, 1, 1, 3)
    physical = physical + position_mean.view(1, 1, 1, 3)
    physical = torch.where(visible.unsqueeze(-1), physical, torch.zeros_like(physical))
    batch = physical.shape[0]
    xy = physical[..., :2]
    edge = torch.roll(xy, shifts=-1, dims=2) - xy
    pair_visible = visible & torch.roll(visible, shifts=-1, dims=2)
    time_gap = event_time_s[:, 1:] - event_time_s[:, :-1]
    repeated_edge = (
        pair_visible[:, 1:] & pair_visible[:, :-1]
        & (time_gap > 0)[:, :, None]
        & (time_gap <= maximum_edge_gap_s)[:, :, None]
    )
    edge_cross = _cross_xy(edge[:, :-1], edge[:, 1:])
    edge_dot = (edge[:, :-1] * edge[:, 1:]).sum(dim=-1)
    edge_angle = torch.atan2(edge_cross, edge_dot)
    edge_score = torch.where(
        repeated_edge, edge_angle, torch.zeros_like(edge_angle),
    ).sum(dim=(1, 2))

    first_delta = xy[:, 1:-1] - xy[:, :-2]
    second_delta = xy[:, 2:] - xy[:, 1:-1]
    curve_gap_left = event_time_s[:, 1:-1] - event_time_s[:, :-2]
    curve_gap_right = event_time_s[:, 2:] - event_time_s[:, 1:-1]
    curve_support = (
        visible[:, :-2] & visible[:, 1:-1] & visible[:, 2:]
        & (curve_gap_left > 0)[:, :, None]
        & (curve_gap_right > 0)[:, :, None]
        & (curve_gap_left <= maximum_edge_gap_s)[:, :, None]
        & (curve_gap_right <= maximum_edge_gap_s)[:, :, None]
    )
    denominator = (
        torch.linalg.vector_norm(first_delta, dim=-1)
        * torch.linalg.vector_norm(second_delta, dim=-1)
    ).clamp_min(1e-8)
    normalized_cross = _cross_xy(first_delta, second_delta) / denominator
    curve_score = torch.where(
        curve_support, normalized_cross, torch.zeros_like(normalized_cross),
    ).sum(dim=(1, 2))

    edge_valid = edge_score.abs() > minimum_score
    curve_valid = curve_score.abs() > minimum_score
    score = torch.where(edge_valid, edge_score, curve_score)
    valid = edge_valid | curve_valid
    sign = torch.where(score >= 0, torch.ones_like(score), -torch.ones_like(score))
    sign = torch.where(valid, sign, torch.zeros_like(sign))
    source = torch.where(
        edge_valid, torch.ones(batch, device=obs.device, dtype=torch.long),
        torch.where(
            curve_valid,
            torch.full((batch,), 2, device=obs.device, dtype=torch.long),
            torch.zeros(batch, device=obs.device, dtype=torch.long),
        ),
    )
    return sign.detach(), valid.detach(), source.detach()


class _RotationABBase(nn.Module):
    def __init__(
        self,
        position_mean: torch.Tensor,
        position_std: torch.Tensor,
        *,
        channels: int,
        dropout: float,
        history_events: int,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.history_events = int(history_events)
        self.evidence = MotionEvidenceEncoder(channels, dropout, history_events)
        self.register_buffer("position_mean", position_mean.float().clone())
        self.register_buffer("position_std", position_std.float().clone())

    def _common(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        primary_mask: torch.Tensor,
        event_mask: torch.Tensor,
        event_time_s: torch.Tensor,
        switch_step: torch.Tensor,
        s_state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state, pooled = self.evidence(
            obs, obs_mask, primary_mask, event_mask, event_time_s,
            switch_step, s_state, self.position_mean, self.position_std,
        )
        direction, direction_valid, direction_source = (
            deterministic_rotation_direction(
                obs, obs_mask, event_mask, event_time_s,
                self.position_mean, self.position_std,
            )
        )
        task_valid = s_state["q0_valid"] & (
            s_state["current_visible"] | s_state["anchor_composed"]
        )
        future_valid = task_valid & direction_valid[:, None]
        return state, pooled, direction, direction_source, future_valid


class ParametricRotationFutureExpertV2(_RotationABBase):
    """Center-free rigid decoder with externally determined direction."""

    model_family = "cyclic-center-free-rotation-magnitude-expert-v2"

    def __init__(
        self,
        position_mean: torch.Tensor,
        position_std: torch.Tensor,
        *,
        channels: int = 48,
        dropout: float = 0.05,
        history_events: int = 32,
        max_speed_mps: float = 7.0,
        max_omega_rad_s: float = 20.0,
    ) -> None:
        super().__init__(
            position_mean, position_std, channels=channels, dropout=dropout,
            history_events=history_events,
        )
        self.max_speed_mps = float(max_speed_mps)
        self.max_omega_rad_s = float(max_omega_rad_s)
        self.head = nn.Sequential(
            nn.Linear(3 * channels, 2 * channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(2 * channels, 3),
        )
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():
            self.head[-1].bias[2] = -2.0

    def forward(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        primary_mask: torch.Tensor,
        event_mask: torch.Tensor,
        event_time_s: torch.Tensor,
        switch_step: torch.Tensor,
        tau: torch.Tensor,
        s_state: dict[str, torch.Tensor],
        *,
        q0_override_m: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        state, pooled, direction, direction_source, future_valid = self._common(
            obs, obs_mask, primary_mask, event_mask, event_time_s,
            switch_step, s_state,
        )
        batch = obs.shape[0]
        row = torch.arange(batch, device=obs.device)
        primary_index = s_state["primary_index"].to(torch.long)
        raw = self.head(torch.cat((state[row, primary_index], pooled), dim=-1))
        primary_velocity = self.max_speed_mps * torch.tanh(raw[:, :2])
        omega_magnitude = self.max_omega_rad_s * torch.sigmoid(raw[:, 2])
        omega = direction * omega_magnitude
        q0_m = s_state["q0_m"] if q0_override_m is None else q0_override_m
        if self.training and q0_override_m is not None:
            raise ValueError("q0 override is evaluation-only")
        position = decode_rotation(
            q0_m, primary_index, primary_velocity, omega, tau,
        )
        return {
            "position_m": position,
            "delta_m": position - q0_m[:, None],
            "future_valid": future_valid,
            "primary_index": primary_index,
            "primary_velocity_xy_mps": primary_velocity,
            "omega_magnitude_rad_s": omega_magnitude,
            "omega_rad_s": omega,
            "direction_sign": direction,
            "direction_valid": direction != 0,
            "direction_source": direction_source,
        }

    def config(self) -> dict[str, object]:
        return {
            "family": self.model_family,
            "channels": self.channels,
            "dropout": self.dropout,
            "history_events": self.history_events,
            "max_speed_mps": self.max_speed_mps,
            "max_omega_rad_s": self.max_omega_rad_s,
            "direction": "deterministic causal history geometry; never learned",
            "direction_history": "all available causal observation events",
            "future_output": "center-free rigid decode from primary velocity and signed magnitude",
            "q0_head": False,
            "fixed_geometry": False,
            "cyclic_equivariance": "C4 exact in eval mode",
        }


class DirectRotationTrajectoryExpert(_RotationABBase):
    """Continuous query-conditioned direct future-delta expert."""

    model_family = "cyclic-direct-rotation-future-delta-expert-v1"

    def __init__(
        self,
        position_mean: torch.Tensor,
        position_std: torch.Tensor,
        *,
        channels: int = 48,
        dropout: float = 0.05,
        history_events: int = 32,
        max_tau_s: float = 0.5,
    ) -> None:
        super().__init__(
            position_mean, position_std, channels=channels, dropout=dropout,
            history_events=history_events,
        )
        self.max_tau_s = float(max_tau_s)
        if self.max_tau_s <= 0:
            raise ValueError("max_tau_s must be positive")
        self.edge = nn.Sequential(
            nn.Linear(8, channels), nn.LayerNorm(channels), nn.SiLU(),
        )
        self.query_message = nn.ModuleList(
            CyclicMessageBlock(channels, dropout) for _ in range(2)
        )
        self.head = nn.Sequential(
            nn.Linear(3 * channels + 4, 2 * channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(2 * channels, 2),
        )
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        primary_mask: torch.Tensor,
        event_mask: torch.Tensor,
        event_time_s: torch.Tensor,
        switch_step: torch.Tensor,
        tau: torch.Tensor,
        s_state: dict[str, torch.Tensor],
        *,
        q0_override_m: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        state, pooled, direction, direction_source, future_valid = self._common(
            obs, obs_mask, primary_mask, event_mask, event_time_s,
            switch_step, s_state,
        )
        q0_m = s_state["q0_m"] if q0_override_m is None else q0_override_m
        if self.training and q0_override_m is not None:
            raise ValueError("q0 override is evaluation-only")
        scale = self.position_std.mean().clamp_min(1e-6)
        following = torch.roll(q0_m, shifts=-1, dims=1) - q0_m
        previous = torch.roll(q0_m, shifts=1, dims=1) - q0_m
        following_valid = s_state["q0_valid"] & torch.roll(
            s_state["q0_valid"], shifts=-1, dims=1,
        )
        previous_valid = s_state["q0_valid"] & torch.roll(
            s_state["q0_valid"], shifts=1, dims=1,
        )
        following = torch.where(
            following_valid.unsqueeze(-1), following, torch.zeros_like(following),
        )
        previous = torch.where(
            previous_valid.unsqueeze(-1), previous, torch.zeros_like(previous),
        )
        edge_feature = torch.cat((
            following / scale,
            previous / scale,
            following_valid.to(q0_m.dtype).unsqueeze(-1),
            previous_valid.to(q0_m.dtype).unsqueeze(-1),
        ), dim=-1)
        state = state + self.edge(edge_feature)
        for message in self.query_message:
            state = message(state)
        batch, tracks, _ = state.shape
        tau = _expanded_tau(tau, batch)
        u = tau / self.max_tau_s
        query = torch.stack((u, u.square(), u.pow(3), direction[:, None].expand_as(u)), dim=-1)
        query_count = tau.shape[1]
        track_context = torch.cat((
            state, pooled[:, None].expand(-1, tracks, -1),
        ), dim=-1)
        expanded_track = track_context[:, None].expand(-1, query_count, -1, -1)
        expanded_query = query[:, :, None].expand(-1, -1, tracks, -1)
        raw_delta_xy = self.head(torch.cat((expanded_track, expanded_query), dim=-1))
        delta_xy = u[:, :, None, None] * raw_delta_xy
        raw_delta = torch.cat((
            delta_xy, torch.zeros_like(delta_xy[..., :1]),
        ), dim=-1)
        raw_position = q0_m[:, None] + raw_delta
        position = project_direct_planar_rigid(q0_m, raw_position, future_valid)
        position = torch.where(
            (tau == 0)[:, :, None, None], q0_m[:, None], position,
        )
        delta = position - q0_m[:, None]
        return {
            "position_m": position,
            "delta_m": delta,
            "future_valid": future_valid,
            "primary_index": s_state["primary_index"].to(torch.long),
            "direction_sign": direction,
            "direction_valid": direction != 0,
            "direction_source": direction_source,
        }

    def config(self) -> dict[str, object]:
        return {
            "family": self.model_family,
            "channels": self.channels,
            "dropout": self.dropout,
            "history_events": self.history_events,
            "max_tau_s": self.max_tau_s,
            "direction": "deterministic causal history geometry; never learned",
            "direction_history": "all available causal observation events",
            "future_output": "direct continuous q0-relative trajectory with per-query rigid projection",
            "forbidden_outputs": ["center", "radius", "phase", "velocity", "omega", "acceleration"],
            "q0_head": False,
            "fixed_geometry": False,
            "cyclic_equivariance": "C4 exact in eval mode",
        }
