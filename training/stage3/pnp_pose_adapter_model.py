"""Causal PnP-history adapter for the current canonical rigid pose.

This module deliberately does not predict velocity, yaw rate, or future pose.
It restores the current physical center and canonical phase from observations
whose timestamps are no later than the current anchor.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .causal_physical_state_model import FixedGeometryDecoder
from .physical_model import TemporalArmorSetEncoder


class CurrentPnPPoseAdapter(nn.Module):
    """Restore one current rigid pose from unordered causal PnP histories."""

    model_family = "causal-current-pnp-pose-adapter-v1"

    def __init__(
        self,
        geometry: torch.Tensor,
        observation_mean: torch.Tensor,
        observation_std: torch.Tensor,
        *,
        channels: int = 96,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if tuple(geometry.shape) != (4, 3):
            raise ValueError("geometry must have shape [4,3]")
        if tuple(observation_mean.shape) != (3,) or tuple(observation_std.shape) != (3,):
            raise ValueError("observation normalization must have shape [3]")
        if bool(torch.any(observation_std <= 0)):
            raise ValueError("observation_std must be positive")
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.encoder = TemporalArmorSetEncoder(8, channels, dropout)
        self.pose_head = nn.Sequential(
            nn.Linear(channels + 6, 192), nn.SiLU(),
            nn.Linear(192, 192), nn.SiLU(), nn.Linear(192, 5),
        )
        self.decoder = FixedGeometryDecoder(geometry)
        self.register_buffer("observation_mean", observation_mean.float().clone())
        self.register_buffer("observation_std", observation_std.float().clone())
        radius = torch.sqrt(geometry[:, :2].float().square().sum(dim=-1).mean())
        self.register_buffer("geometry_radius", radius)
        self.register_buffer(
            "center_residual_scale",
            observation_std.float().clamp(min=0.15, max=0.75),
        )
        # Start from the latest PnP centroid and the permutation-invariant
        # modulo-quarter-turn yaw anchor. Small weights keep gradients flowing
        # into the encoder from the first update.
        with torch.no_grad():
            nn.init.normal_(self.pose_head[-1].weight, mean=0.0, std=1e-3)
            self.pose_head[-1].bias.zero_()
            self.pose_head[-1].bias[3] = 1.0

    @staticmethod
    def _last_indices(effective_event: torch.Tensor) -> torch.Tensor:
        time = effective_event.shape[1]
        index = torch.arange(time, device=effective_event.device).view(1, time)
        return torch.where(
            effective_event, index, torch.full_like(index, -1)
        ).amax(dim=1)

    def _relative_observations(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        event_mask: torch.Tensor,
        event_time_s: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        if obs.ndim != 4 or obs.shape[-1] < 5:
            raise ValueError("obs must have shape [B,T,4,F>=5]")
        effective = (
            obs_mask.to(torch.bool)
            & event_mask.to(torch.bool).unsqueeze(-1)
            & torch.isfinite(obs[..., :5]).all(dim=-1)
        )
        effective_event = effective.any(dim=2) & torch.isfinite(event_time_s)
        last = self._last_indices(effective_event)
        if bool((last < 0).any()):
            raise ValueError("every PnP history must contain a valid event")

        xyz = obs[..., :3].float() * self.observation_std + self.observation_mean
        mask = effective.unsqueeze(-1)
        count = effective.sum(dim=2, keepdim=True).clamp_min(1).to(xyz.dtype)
        frame_centroid = torch.where(mask, xyz, torch.zeros_like(xyz)).sum(dim=2) / count
        gather3 = last.view(-1, 1, 1).expand(-1, 1, 3)
        latest_centroid = frame_centroid.gather(1, gather3).squeeze(1)

        shape = (xyz - frame_centroid.unsqueeze(2)) / self.geometry_radius.clamp_min(1e-6)
        frame_delta = (
            (frame_centroid - latest_centroid.unsqueeze(1))
            / self.observation_std.clamp_min(0.15)
        ).unsqueeze(2).expand(-1, -1, obs.shape[2], -1)
        relative = torch.cat((shape, frame_delta, obs[..., 3:5].float()), dim=-1)
        relative = torch.where(mask, relative, torch.zeros_like(relative))

        gather4 = last.view(-1, 1, 1).expand(-1, 1, obs.shape[2])
        latest_mask = effective.gather(1, gather4).squeeze(1)
        latest_yaw = obs[..., 3:5].float().gather(
            1, last.view(-1, 1, 1, 1).expand(-1, 1, obs.shape[2], 2)
        ).squeeze(1)
        yaw_identity = torch.zeros_like(latest_yaw)
        yaw_identity[..., 1] = 1.0
        latest_yaw = torch.where(
            latest_mask.unsqueeze(-1), latest_yaw, yaw_identity
        )
        yaw = torch.atan2(latest_yaw[..., 0], latest_yaw[..., 1])
        weight = latest_mask.to(yaw.dtype)
        four_sine = (torch.sin(4.0 * yaw) * weight).sum(dim=1)
        four_cosine = (torch.cos(4.0 * yaw) * weight).sum(dim=1)
        principal = 0.25 * torch.atan2(four_sine, four_cosine)
        phase_anchor = torch.stack((torch.cos(principal), torch.sin(principal)), dim=-1)
        latest_count = latest_mask.sum(dim=1, keepdim=True).to(xyz.dtype) / 4.0
        return (
            relative, latest_centroid, phase_anchor, latest_count,
            effective, effective_event,
        )

    def forward(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        event_mask: torch.Tensor,
        event_time_s: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        (
            relative, latest_centroid, phase_anchor, latest_count,
            effective_mask, effective_event,
        ) = (
            self._relative_observations(obs, obs_mask, event_mask, event_time_s)
        )
        encoded = self.encoder(relative, effective_mask, effective_event, event_time_s)
        absolute_anchor = (latest_centroid - self.observation_mean) / self.observation_std
        latent = self.pose_head(torch.cat(
            (encoded, absolute_anchor.to(encoded.dtype), phase_anchor.to(encoded.dtype),
             latest_count.to(encoded.dtype)), dim=-1,
        )).float()
        center = latest_centroid + self.center_residual_scale * latent[:, :3]
        delta = F.normalize(latent[:, 3:5], dim=-1, eps=1e-8)
        invalid_delta = torch.linalg.vector_norm(latent[:, 3:5], dim=-1, keepdim=True) < 1e-6
        identity = torch.zeros_like(delta)
        identity[:, 0] = 1.0
        delta = torch.where(invalid_delta, identity, delta)
        phase = torch.stack((
            phase_anchor[:, 0] * delta[:, 0] - phase_anchor[:, 1] * delta[:, 1],
            phase_anchor[:, 1] * delta[:, 0] + phase_anchor[:, 0] * delta[:, 1],
        ), dim=-1)
        position = self.decoder(center[:, None], phase[:, None])
        return {
            "position_mean": position,
            "center": center,
            "phase": phase,
            "latest_pnp_centroid": latest_centroid,
            "phase_anchor": phase_anchor,
        }

    def config(self) -> dict[str, object]:
        return {
            "family": self.model_family,
            "channels": self.channels,
            "dropout": self.dropout,
            "relative_input_features": [
                "candidate_minus_frame_centroid_xyz_over_geometry_radius",
                "frame_centroid_minus_latest_centroid_xyz_over_train_scale",
                "pnp_sin_yaw",
                "pnp_cos_yaw",
            ],
            "geometry": self.decoder.geometry.detach().cpu().tolist(),
            "observation_mean": self.observation_mean.detach().cpu().tolist(),
            "observation_std": self.observation_std.detach().cpu().tolist(),
            "center_residual_scale": self.center_residual_scale.detach().cpu().tolist(),
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
