"""Center-free continuous future propagation for temporary cyclic tracks.

The frozen S layer owns q0 reconstruction.  These experts consume the same
causal history plus the frozen q0/edge state and predict only motion relative
to q0.  No center, phase, radius, height, or physical slot identity is used.
"""

from __future__ import annotations

import torch
from torch import nn

from .cyclic_track_model import CyclicContextEncoder, CyclicMessageBlock, _expanded_tau


DYNAMIC_EXPERTS = ("translation", "rotation", "combined")


def planar_j(value: torch.Tensor) -> torch.Tensor:
    """Apply the +90 degree planar rotation J while preserving shape."""
    if value.shape[-1] != 2:
        raise ValueError("planar_j requires a final dimension of two")
    return torch.stack((-value[..., 1], value[..., 0]), dim=-1)


def _stable_c2(theta: torch.Tensor) -> torch.Tensor:
    """Return (1-cos(theta))/theta^2 with a stable small-angle branch."""
    square = theta.square()
    series = 0.5 - square / 24.0 + square.square() / 720.0
    regular = (1.0 - torch.cos(theta)) / square.clamp_min(1e-12)
    return torch.where(theta.abs() < 0.1, series, regular)


def _stable_c3(theta: torch.Tensor) -> torch.Tensor:
    """Return (theta-sin(theta))/theta^2 with a stable small-angle branch."""
    square = theta.square()
    series = theta / 6.0 - theta * square / 120.0 + theta * square.square() / 5040.0
    regular = (theta - torch.sin(theta)) / square.clamp_min(1e-12)
    return torch.where(theta.abs() < 0.1, series, regular)


def _stable_sinc(theta: torch.Tensor) -> torch.Tensor:
    """Return sin(theta)/theta with a stable small-angle branch."""
    square = theta.square()
    series = 1.0 - square / 6.0 + square.square() / 120.0
    regular = torch.sin(theta) / theta.abs().clamp_min(1e-12) * theta.sign()
    return torch.where(theta.abs() < 0.1, series, regular)


def _stable_cosc(theta: torch.Tensor) -> torch.Tensor:
    """Return (1-cos(theta))/theta with a stable small-angle branch."""
    square = theta.square()
    series = theta / 2.0 - theta * square / 24.0 + theta * square.square() / 720.0
    regular = (1.0 - torch.cos(theta)) / theta.abs().clamp_min(1e-12) * theta.sign()
    return torch.where(theta.abs() < 0.1, series, regular)


def decode_translation(
    q0_m: torch.Tensor, velocity_mps: torch.Tensor, tau_s: torch.Tensor,
) -> torch.Tensor:
    """Propagate a common 3-D constant translation velocity."""
    if q0_m.ndim != 3 or q0_m.shape[1:] != (4, 3):
        raise ValueError("q0_m must have shape [B,4,3]")
    if velocity_mps.shape != (q0_m.shape[0], 3):
        raise ValueError("velocity_mps must have shape [B,3]")
    tau_s = _expanded_tau(tau_s, q0_m.shape[0])
    return q0_m[:, None] + tau_s[:, :, None, None] * velocity_mps[:, None, None]


def decode_rotation(
    q0_m: torch.Tensor,
    primary_index: torch.Tensor,
    primary_velocity_xy_mps: torch.Tensor,
    omega_rad_s: torch.Tensor,
    tau_s: torch.Tensor,
) -> torch.Tensor:
    """Propagate pure yaw rotation without representing a center or radius."""
    batch = q0_m.shape[0]
    if q0_m.ndim != 3 or q0_m.shape[1:] != (4, 3):
        raise ValueError("q0_m must have shape [B,4,3]")
    if primary_index.shape != (batch,):
        raise ValueError("primary_index must have shape [B]")
    if primary_velocity_xy_mps.shape != (batch, 2):
        raise ValueError("primary velocity must have shape [B,2]")
    if omega_rad_s.shape != (batch,):
        raise ValueError("omega must have shape [B]")
    tau_s = _expanded_tau(tau_s, batch)
    row = torch.arange(batch, device=q0_m.device)
    primary_xy = q0_m[row, primary_index, :2]
    relative_xy = q0_m[..., :2] - primary_xy[:, None]
    track_velocity = (
        primary_velocity_xy_mps[:, None]
        + omega_rad_s[:, None, None] * planar_j(relative_xy)
    )
    theta = omega_rad_s[:, None] * tau_s
    a = tau_s * _stable_sinc(theta)
    b = tau_s * _stable_cosc(theta)
    delta_xy = (
        a[:, :, None, None] * track_velocity[:, None]
        + b[:, :, None, None] * planar_j(track_velocity)[:, None]
    )
    delta = torch.cat((delta_xy, torch.zeros_like(delta_xy[..., :1])), dim=-1)
    return q0_m[:, None] + delta


def decode_combined(
    q0_m: torch.Tensor,
    primary_index: torch.Tensor,
    primary_velocity_mps: torch.Tensor,
    primary_acceleration_xy_mps2: torch.Tensor,
    omega_rad_s: torch.Tensor,
    tau_s: torch.Tensor,
) -> torch.Tensor:
    """Propagate constant center translation plus constant yaw rate.

    The identifiable state is the primary track's total instantaneous velocity
    and planar acceleration together with omega.  This avoids the gauge between
    an unobserved center velocity and an anchor's rotational tangent velocity.
    """
    batch = q0_m.shape[0]
    if q0_m.ndim != 3 or q0_m.shape[1:] != (4, 3):
        raise ValueError("q0_m must have shape [B,4,3]")
    if primary_index.shape != (batch,):
        raise ValueError("primary_index must have shape [B]")
    if primary_velocity_mps.shape != (batch, 3):
        raise ValueError("primary velocity must have shape [B,3]")
    if primary_acceleration_xy_mps2.shape != (batch, 2):
        raise ValueError("primary acceleration must have shape [B,2]")
    if omega_rad_s.shape != (batch,):
        raise ValueError("omega must have shape [B]")
    tau_s = _expanded_tau(tau_s, batch)
    row = torch.arange(batch, device=q0_m.device)
    primary_xy = q0_m[row, primary_index, :2]
    relative_xy = q0_m[..., :2] - primary_xy[:, None]
    track_velocity_xy = (
        primary_velocity_mps[:, None, :2]
        + omega_rad_s[:, None, None] * planar_j(relative_xy)
    )
    track_acceleration_xy = (
        primary_acceleration_xy_mps2[:, None]
        - omega_rad_s[:, None, None].square() * relative_xy
    )
    theta = omega_rad_s[:, None] * tau_s
    c2 = tau_s.square() * _stable_c2(theta)
    c3 = tau_s.square() * _stable_c3(theta)
    delta_xy = (
        tau_s[:, :, None, None] * track_velocity_xy[:, None]
        + c2[:, :, None, None] * track_acceleration_xy[:, None]
        + c3[:, :, None, None] * planar_j(track_acceleration_xy)[:, None]
    )
    delta_z = (
        tau_s[:, :, None, None]
        * primary_velocity_mps[:, None, None, 2:3]
    ).expand(-1, -1, 4, -1)
    return q0_m[:, None] + torch.cat((delta_xy, delta_z), dim=-1)


class MotionEvidenceEncoder(nn.Module):
    """Encode q0-relative causal observations and frozen S quality state."""

    def __init__(self, channels: int, dropout: float, history_events: int) -> None:
        super().__init__()
        self.context = CyclicContextEncoder(channels, dropout, history_events)
        # valid, sigma, visible, q0-observed, anchor-composed, cold,
        # track age, edge valid, edge sigma, edge age, clockwise, counterclockwise
        self.quality = nn.Sequential(
            nn.Linear(12, channels), nn.LayerNorm(channels), nn.SiLU(),
        )
        self.quality_message = CyclicMessageBlock(channels, dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        primary_mask: torch.Tensor,
        event_mask: torch.Tensor,
        event_time_s: torch.Tensor,
        switch_step: torch.Tensor,
        s_state: dict[str, torch.Tensor],
        position_mean: torch.Tensor,
        position_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q0_normalized = (s_state["q0_m"] - position_mean) / position_std
        relative_obs = torch.where(
            obs_mask.unsqueeze(-1), obs - q0_normalized[:, None],
            torch.zeros_like(obs),
        )
        track_state, _, _, _ = self.context(
            relative_obs, obs_mask, primary_mask, event_mask,
            event_time_s, switch_step,
        )
        finite_track_age = torch.where(
            torch.isfinite(s_state["age_s"]), s_state["age_s"],
            torch.full_like(s_state["age_s"], 10.0),
        ).clamp(0.0, 10.0)
        finite_edge_age = torch.where(
            torch.isfinite(s_state["edge_age_s"]), s_state["edge_age_s"],
            torch.full_like(s_state["edge_age_s"], 10.0),
        ).clamp(0.0, 10.0)
        scale = position_std.mean().clamp_min(1e-6)
        quality = torch.stack((
            s_state["q0_valid"].to(obs.dtype),
            s_state["q0_sigma_m"].squeeze(-1) / scale,
            s_state["current_visible"].to(obs.dtype),
            s_state["q0_observed"].to(obs.dtype),
            s_state["anchor_composed"].to(obs.dtype),
            s_state["cold"].to(obs.dtype),
            finite_track_age,
            s_state["edge0_valid"].to(obs.dtype),
            s_state["edge0_sigma_m"].squeeze(-1) / scale,
            finite_edge_age,
            s_state["clockwise"].to(obs.dtype),
            s_state["counterclockwise"].to(obs.dtype),
        ), dim=-1)
        state = self.norm(track_state + self.quality(quality))
        state = self.quality_message(state)
        pooled = torch.cat((state.mean(dim=1), state.amax(dim=1)), dim=-1)
        return state, pooled


class CyclicFutureMotionExpert(nn.Module):
    """One parameter-independent dynamic motion expert after frozen S."""

    model_family = "cyclic-center-free-future-motion-expert-v1"

    def __init__(
        self,
        expert: str,
        position_mean: torch.Tensor,
        position_std: torch.Tensor,
        *,
        channels: int = 48,
        dropout: float = 0.05,
        history_events: int = 32,
        max_speed_mps: float = 7.0,
        max_acceleration_mps2: float = 100.0,
        max_omega_rad_s: float = 20.0,
    ) -> None:
        super().__init__()
        if expert not in DYNAMIC_EXPERTS:
            raise ValueError(f"unsupported dynamic expert: {expert}")
        if tuple(position_mean.shape) != (3,) or tuple(position_std.shape) != (3,):
            raise ValueError("position normalization must have shape [3]")
        if bool(torch.any(position_std <= 0)):
            raise ValueError("position_std must be positive")
        if min(max_speed_mps, max_acceleration_mps2, max_omega_rad_s) <= 0:
            raise ValueError("motion bounds must be positive")
        self.expert = expert
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.history_events = int(history_events)
        self.max_speed_mps = float(max_speed_mps)
        self.max_acceleration_mps2 = float(max_acceleration_mps2)
        self.max_omega_rad_s = float(max_omega_rad_s)
        self.evidence = MotionEvidenceEncoder(channels, dropout, history_events)
        output_count = {"translation": 3, "rotation": 3, "combined": 6}[expert]
        self.head = nn.Sequential(
            nn.Linear(3 * channels, 2 * channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(2 * channels, output_count),
        )
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("position_mean", position_mean.float().clone())
        self.register_buffer("position_std", position_std.float().clone())

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
        state, pooled = self.evidence(
            obs, obs_mask, primary_mask, event_mask, event_time_s,
            switch_step, s_state, self.position_mean, self.position_std,
        )
        batch = obs.shape[0]
        row = torch.arange(batch, device=obs.device)
        primary_index = s_state["primary_index"].to(torch.long)
        primary_state = state[row, primary_index]
        raw = self.head(torch.cat((primary_state, pooled), dim=-1))
        q0_m = s_state["q0_m"] if q0_override_m is None else q0_override_m
        if self.training and q0_override_m is not None:
            raise ValueError("q0 override is evaluation-only")
        if q0_m.shape != s_state["q0_m"].shape:
            raise ValueError("q0_override_m must match the frozen S q0 shape")

        if self.expert == "translation":
            velocity = self.max_speed_mps * torch.tanh(raw)
            position = decode_translation(q0_m, velocity, tau)
            motion = {
                "velocity_mps": velocity,
                "omega_rad_s": torch.zeros(batch, device=obs.device, dtype=obs.dtype),
            }
        elif self.expert == "rotation":
            primary_velocity = self.max_speed_mps * torch.tanh(raw[:, :2])
            omega = self.max_omega_rad_s * torch.tanh(raw[:, 2])
            position = decode_rotation(
                q0_m, primary_index, primary_velocity, omega, tau,
            )
            motion = {
                "primary_velocity_xy_mps": primary_velocity,
                "omega_rad_s": omega,
            }
        else:
            primary_velocity = self.max_speed_mps * torch.tanh(raw[:, :3])
            primary_acceleration = (
                self.max_acceleration_mps2 * torch.tanh(raw[:, 3:5])
            )
            omega = self.max_omega_rad_s * torch.tanh(raw[:, 5])
            position = decode_combined(
                q0_m, primary_index, primary_velocity,
                primary_acceleration, omega, tau,
            )
            motion = {
                "primary_velocity_mps": primary_velocity,
                "primary_acceleration_xy_mps2": primary_acceleration,
                "omega_rad_s": omega,
            }
        if self.expert == "translation":
            valid = s_state["q0_valid"] & s_state["current_visible"]
        else:
            valid = s_state["q0_valid"] & (
                s_state["current_visible"] | s_state["anchor_composed"]
            )
        return {
            "position_m": position,
            "delta_m": position - q0_m[:, None],
            "future_valid": valid,
            "primary_index": primary_index,
            **motion,
        }

    def config(self) -> dict[str, object]:
        return {
            "family": self.model_family,
            "expert": self.expert,
            "channels": self.channels,
            "dropout": self.dropout,
            "history_events": self.history_events,
            "max_speed_mps": self.max_speed_mps,
            "max_acceleration_mps2": self.max_acceleration_mps2,
            "max_omega_rad_s": self.max_omega_rad_s,
            "future_output": "frozen q0 plus continuous center-free rigid delta",
            "q0_head": False,
            "fixed_geometry": False,
            "combined_calls_other_experts": False,
            "cyclic_equivariance": "C4 exact in eval mode",
        }
