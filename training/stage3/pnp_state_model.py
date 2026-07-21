"""Paired PnP-history models for explicit-state versus implicit-pose A/B."""

from __future__ import annotations

import torch
from torch import nn

from .physical_model import TemporalArmorSetEncoder


def _expanded_tau(tau: torch.Tensor, batch: int) -> torch.Tensor:
    if tau.ndim == 1:
        tau = tau.unsqueeze(0).expand(batch, -1)
    if tau.ndim != 2 or tau.shape[0] != batch:
        raise ValueError("tau must have shape [Q] or [B,Q]")
    return tau


class _RigidGeometryDecoder(nn.Module):
    """Decode query-specific centers and unit-complex phases into four slots."""

    def __init__(self, geometry: torch.Tensor) -> None:
        super().__init__()
        if tuple(geometry.shape) != (4, 3):
            raise ValueError("geometry must have shape [4,3]")
        # The template is expressed from the real target rotation center.
        # Re-centering it around the armor centroid creates fictitious motion.
        self.register_buffer("geometry", geometry.to(dtype=torch.float32).clone())

    def forward(self, center: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        if center.ndim != 3 or center.shape[-1] != 3:
            raise ValueError("center must have shape [B,Q,3]")
        if phase.shape != (*center.shape[:2], 2):
            raise ValueError("phase must have shape [B,Q,2]")
        norm = torch.linalg.vector_norm(phase.float(), dim=-1, keepdim=True)
        identity = torch.zeros_like(phase)
        identity[..., 0] = 1.0
        phase = torch.where(
            norm > 1e-6, phase / norm.to(phase.dtype).clamp_min(1e-6), identity
        )
        cosine, sine = phase[..., 0], phase[..., 1]
        gx = self.geometry[:, 0].view(1, 1, 4)
        gy = self.geometry[:, 1].view(1, 1, 4)
        gz = self.geometry[:, 2].view(1, 1, 4).expand(center.shape[0], center.shape[1], -1)
        x = cosine[:, :, None] * gx - sine[:, :, None] * gy
        y = sine[:, :, None] * gx + cosine[:, :, None] * gy
        relative = torch.stack((x, y, gz), dim=-1)
        return center[:, :, None, :] + relative


class ExplicitStatePnPAdapter(nn.Module):
    """Infer one constant-twist state and propagate it with frozen mechanics.

    The state is re-estimated from the complete causal input window on every
    forward call. It is not a recurrent EKF state and is never supplied as an
    input label. Position losses train it through the frozen rigid decoder.
    """

    model_family = "pnp-explicit-constant-twist-state-v1"

    def __init__(
        self,
        geometry: torch.Tensor,
        *,
        input_features: int = 7,
        channels: int = 96,
        dropout: float = 0.05,
        center_reference: torch.Tensor | None = None,
        center_scale: torch.Tensor | None = None,
        maximum_speed_mps: float = 3.5,
        maximum_yaw_rate_rad_s: float = 15.0,
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.maximum_speed_mps = float(maximum_speed_mps)
        self.maximum_yaw_rate_rad_s = float(maximum_yaw_rate_rad_s)
        self.encoder = TemporalArmorSetEncoder(input_features, channels, dropout)
        self.state_head = nn.Sequential(
            nn.Linear(channels, 192), nn.SiLU(),
            nn.Linear(192, 192), nn.SiLU(), nn.Linear(192, 8),
        )
        # A random 15-rad/s-scale initial motion is a poor neutral point for an
        # unordered four-fold periodic set loss. Start translation/yaw rates at
        # exactly zero; their rows still receive gradients and learn motion.
        with torch.no_grad():
            self.state_head[-1].weight[3:6].zero_()
            self.state_head[-1].bias[3:6].zero_()
            self.state_head[-1].weight[7].zero_()
            self.state_head[-1].bias[7].zero_()
        reference = torch.zeros(3) if center_reference is None else center_reference
        scale = torch.ones(3) if center_scale is None else center_scale
        if tuple(reference.shape) != (3,) or tuple(scale.shape) != (3,):
            raise ValueError("center reference and scale must have shape [3]")
        if torch.any(scale <= 0):
            raise ValueError("center scale must be positive")
        self.register_buffer("center_reference", reference.to(torch.float32).clone())
        self.register_buffer("center_scale", scale.to(torch.float32).clone())
        self.decoder = _RigidGeometryDecoder(geometry)

    def forward(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        event_mask: torch.Tensor,
        event_time_s: torch.Tensor,
        tau: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(obs, obs_mask, event_mask, event_time_s)
        latent = self.state_head(encoded)
        center0 = self.center_reference + self.center_scale * latent[:, 0:3]
        velocity = self.maximum_speed_mps * torch.tanh(latent[:, 3:6])
        yaw0 = torch.pi * torch.tanh(latent[:, 6])
        phase0 = torch.stack((torch.cos(yaw0), torch.sin(yaw0)), dim=-1)
        omega = self.maximum_yaw_rate_rad_s * torch.tanh(latent[:, 7])
        tau = _expanded_tau(tau, obs.shape[0])
        center = center0[:, None, :] + velocity[:, None, :] * tau[:, :, None]
        angle = omega[:, None] * tau
        cosine, sine = torch.cos(angle), torch.sin(angle)
        phase = torch.stack(
            (
                phase0[:, None, 0] * cosine - phase0[:, None, 1] * sine,
                phase0[:, None, 1] * cosine + phase0[:, None, 0] * sine,
            ),
            dim=-1,
        )
        position = self.decoder(center, phase)
        return {
            "position_mean": position,
            "center0": center0,
            "velocity": velocity,
            "phase0": phase0,
            "omega": omega,
            "query_center": center,
            "query_phase": phase,
        }

    def config(self) -> dict[str, object]:
        return {
            "family": self.model_family,
            "input_features": self.input_features,
            "channels": self.channels,
            "dropout": self.dropout,
            "maximum_speed_mps": self.maximum_speed_mps,
            "maximum_yaw_rate_rad_s": self.maximum_yaw_rate_rad_s,
            "geometry": self.decoder.geometry.detach().cpu().tolist(),
            "center_reference": self.center_reference.detach().cpu().tolist(),
            "center_scale": self.center_scale.detach().cpu().tolist(),
        }


class ImplicitQueryPosePredictor(nn.Module):
    """Predict each query pose independently without a shared motion state."""

    model_family = "pnp-implicit-query-pose-v1"

    def __init__(
        self,
        geometry: torch.Tensor,
        *,
        input_features: int = 7,
        channels: int = 96,
        dropout: float = 0.05,
        center_reference: torch.Tensor | None = None,
        center_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.encoder = TemporalArmorSetEncoder(input_features, channels, dropout)
        self.query_head = nn.Sequential(
            nn.Linear(channels + 3, 192), nn.SiLU(),
            nn.Linear(192, 192), nn.SiLU(), nn.Linear(192, 4),
        )
        reference = torch.zeros(3) if center_reference is None else center_reference
        scale = torch.ones(3) if center_scale is None else center_scale
        if tuple(reference.shape) != (3,) or tuple(scale.shape) != (3,):
            raise ValueError("center reference and scale must have shape [3]")
        if torch.any(scale <= 0):
            raise ValueError("center scale must be positive")
        self.register_buffer("center_reference", reference.to(torch.float32).clone())
        self.register_buffer("center_scale", scale.to(torch.float32).clone())
        self.decoder = _RigidGeometryDecoder(geometry)

    def forward(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        event_mask: torch.Tensor,
        event_time_s: torch.Tensor,
        tau: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(obs, obs_mask, event_mask, event_time_s)
        tau = _expanded_tau(tau, obs.shape[0])
        expanded = encoded[:, None, :].expand(-1, tau.shape[1], -1)
        query = torch.cat(
            (expanded, tau[:, :, None], tau.square()[:, :, None], tau.pow(3)[:, :, None]),
            dim=-1,
        )
        latent = self.query_head(query)
        center = self.center_reference + self.center_scale * latent[..., 0:3]
        yaw = torch.pi * torch.tanh(latent[..., 3])
        phase = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
        position = self.decoder(center, phase)
        return {
            "position_mean": position,
            "query_center": center,
            "query_phase": phase,
        }

    def config(self) -> dict[str, object]:
        return {
            "family": self.model_family,
            "input_features": self.input_features,
            "channels": self.channels,
            "dropout": self.dropout,
            "geometry": self.decoder.geometry.detach().cpu().tolist(),
            "center_reference": self.center_reference.detach().cpu().tolist(),
            "center_scale": self.center_scale.detach().cpu().tolist(),
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
