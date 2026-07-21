"""Physical-only Stage-3 predictors with an unchanged position output API."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .model import CausalResidualBlock


class TemporalArmorSetEncoder(nn.Module):
    """Encode irregular armor sets without collapsing them to sum/max alone."""

    def __init__(self, input_features: int = 5, channels: int = 96, dropout: float = 0.05) -> None:
        super().__init__()
        if input_features < 3:
            raise ValueError("physical encoder needs at least xyz")
        if channels % 4:
            raise ValueError("channels must be divisible by four")
        self.input_features = int(input_features)
        self.channels = int(channels)
        token_channels = channels // 2
        self.armor_mlp = nn.Sequential(
            nn.Linear(input_features, token_channels), nn.SiLU(),
            nn.Linear(token_channels, token_channels), nn.SiLU(),
        )
        self.slot_queries = nn.Parameter(torch.randn(4, token_channels) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            token_channels, num_heads=4, dropout=dropout, batch_first=True
        )
        frame_input = 4 * token_channels + token_channels * 2 + 1 + 2
        self.frame_projection = nn.Sequential(
            nn.Linear(frame_input, channels), nn.LayerNorm(channels), nn.SiLU()
        )
        self.tcn = nn.ModuleList(
            CausalResidualBlock(channels, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16, 32, 64, 128)
        )

    def forward(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        event_mask: torch.Tensor,
        event_time_s: torch.Tensor,
    ) -> torch.Tensor:
        if obs.shape[-1] != self.input_features:
            raise ValueError(
                f"expected {self.input_features} input features, got {obs.shape[-1]}"
            )
        batch, time, armor, _ = obs.shape
        tokens = self.armor_mlp(obs).reshape(batch * time, armor, -1)
        flat_mask = obs_mask.reshape(batch * time, armor).to(torch.bool)
        # MultiheadAttention cannot consume a row whose every key is masked.
        # Padded frames get one zero dummy key and are zeroed again afterwards.
        all_missing = ~flat_mask.any(dim=1)
        safe_mask = flat_mask.clone()
        safe_mask[all_missing, 0] = True
        tokens = tokens.clone()
        tokens[all_missing, 0] = 0.0
        queries = self.slot_queries.unsqueeze(0).expand(batch * time, -1, -1)
        slots, _ = self.cross_attention(
            queries, tokens, tokens, key_padding_mask=~safe_mask, need_weights=False
        )
        mask_float = safe_mask.unsqueeze(-1).to(tokens.dtype)
        count = mask_float.sum(dim=1).clamp_min(1.0)
        token_mean = (tokens * mask_float).sum(dim=1) / count
        minimum = torch.finfo(tokens.dtype).min
        token_max = torch.where(safe_mask.unsqueeze(-1), tokens, minimum).amax(dim=1)
        token_max = torch.where(all_missing.unsqueeze(-1), torch.zeros_like(token_max), token_max)
        slots = slots.reshape(batch, time, -1)
        token_mean = token_mean.reshape(batch, time, -1)
        token_max = token_max.reshape(batch, time, -1)
        count_feature = obs_mask.to(tokens.dtype).sum(dim=2, keepdim=True) / 4.0
        previous_time = torch.cat((event_time_s[:, :1], event_time_s[:, :-1]), dim=1)
        previous_valid = torch.cat(
            (torch.zeros_like(event_mask[:, :1]), event_mask[:, :-1]), dim=1
        )
        delta_time = torch.where(
            event_mask & previous_valid,
            event_time_s - previous_time,
            torch.zeros_like(event_time_s),
        )
        time_features = torch.stack(
            (event_time_s.clamp(-15.0, 0.0), delta_time.clamp(0.0, 1.0)), dim=-1
        ).to(tokens.dtype)
        frame = self.frame_projection(torch.cat(
            (slots, token_mean, token_max, count_feature, time_features), dim=-1
        ))
        frame = frame * event_mask.to(frame.dtype).unsqueeze(-1)
        sequence = frame.transpose(1, 2)
        for block in self.tcn:
            sequence = block(sequence)
            sequence = sequence * event_mask.to(sequence.dtype).unsqueeze(1)
        return sequence[:, :, -1]


class AnchoredDeltaPredictor(nn.Module):
    """Direct position model split into q0 state and a zero-at-q0 displacement."""

    model_family = "anchored-direct-delta-v1"

    def __init__(self, input_features: int = 5, channels: int = 96, dropout: float = 0.05) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.encoder = TemporalArmorSetEncoder(input_features, channels, dropout)
        self.anchor_head = nn.Sequential(
            nn.Linear(channels, 192), nn.SiLU(), nn.Linear(192, 12)
        )
        self.delta_head = nn.Sequential(
            nn.Linear(channels + 3, 192), nn.SiLU(),
            nn.Linear(192, 192), nn.SiLU(), nn.Linear(192, 12),
        )

    def forward(self, obs: torch.Tensor, obs_mask: torch.Tensor,
                event_mask: torch.Tensor, event_time_s: torch.Tensor,
                tau: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self.encoder(obs, obs_mask, event_mask, event_time_s)
        if tau.ndim == 1:
            tau = tau.unsqueeze(0).expand(obs.shape[0], -1)
        anchor = self.anchor_head(state).reshape(-1, 4, 3)
        expanded = state.unsqueeze(1).expand(-1, tau.shape[1], -1)
        query = torch.cat(
            (expanded, tau.unsqueeze(-1), tau.square().unsqueeze(-1), tau.pow(3).unsqueeze(-1)),
            dim=-1,
        )
        rate = self.delta_head(query).reshape(-1, tau.shape[1], 4, 3)
        delta = tau[:, :, None, None] * rate
        position = anchor[:, None] + delta
        return {"position_mean": position, "anchor_position": anchor, "delta": delta}

    def config(self) -> dict[str, int | float | str]:
        return {
            "family": self.model_family, "input_features": self.input_features,
            "channels": self.channels, "dropout": self.dropout,
        }


class RigidMotionPredictor(nn.Module):
    """Learn a rigid constant-twist latent and expose only future positions."""

    model_family = "rigid-latent-decoder-v1"

    def __init__(self, geometry: torch.Tensor, input_features: int = 5,
                 channels: int = 96, dropout: float = 0.05) -> None:
        super().__init__()
        if tuple(geometry.shape) != (4, 3):
            raise ValueError("geometry must have shape [4,3]")
        self.input_features = int(input_features)
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.encoder = TemporalArmorSetEncoder(input_features, channels, dropout)
        self.state_head = nn.Sequential(
            nn.Linear(channels, 192), nn.SiLU(), nn.Linear(192, 9)
        )
        # A zero two-vector has no well-defined yaw.  Bias the initial phase
        # towards the identity rotation so the fixed geometry never collapses
        # during the first optimization steps.
        with torch.no_grad():
            self.state_head[-1].bias[6] = 1.0
        # geometry_template.relative_position_m is already expressed from the
        # real target rotation center.  Re-centering around the arithmetic
        # armor centroid would introduce a yaw-dependent fictitious translation.
        self.register_buffer("geometry", geometry.to(dtype=torch.float32).clone())

    def forward(self, obs: torch.Tensor, obs_mask: torch.Tensor,
                event_mask: torch.Tensor, event_time_s: torch.Tensor,
                tau: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(obs, obs_mask, event_mask, event_time_s)
        latent = self.state_head(encoded)
        center0 = latent[:, 0:3]
        velocity = 3.5 * torch.tanh(latent[:, 3:6])
        phase = F.normalize(latent[:, 6:8], dim=-1, eps=1e-6)
        omega = 15.0 * torch.tanh(latent[:, 8])
        if tau.ndim == 1:
            tau = tau.unsqueeze(0).expand(obs.shape[0], -1)
        center = center0[:, None, :] + velocity[:, None, :] * tau[:, :, None]
        angle = omega[:, None] * tau
        cos_delta, sin_delta = torch.cos(angle), torch.sin(angle)
        cos_yaw = phase[:, 0:1] * cos_delta - phase[:, 1:2] * sin_delta
        sin_yaw = phase[:, 1:2] * cos_delta + phase[:, 0:1] * sin_delta
        gx = self.geometry[:, 0].view(1, 1, 4)
        gy = self.geometry[:, 1].view(1, 1, 4)
        gz = self.geometry[:, 2].view(1, 1, 4).expand(obs.shape[0], tau.shape[1], -1)
        x = cos_yaw[:, :, None] * gx - sin_yaw[:, :, None] * gy
        y = sin_yaw[:, :, None] * gx + cos_yaw[:, :, None] * gy
        relative = torch.stack((x, y, gz), dim=-1)
        position = center[:, :, None, :] + relative
        return {
            "position_mean": position,
            "center0": center0,
            "velocity": velocity,
            "phase": phase,
            "omega": omega,
        }

    def config(self) -> dict[str, int | float | str | list[list[float]]]:
        return {
            "family": self.model_family, "input_features": self.input_features,
            "channels": self.channels, "dropout": self.dropout,
            "geometry": self.geometry.detach().cpu().tolist(),
        }
