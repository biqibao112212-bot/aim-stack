"""Causal set encoder + TCN with arbitrary-time four-armor decoder."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dilation = dilation
        self.conv1 = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def _conv(self, layer: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        left = 2 * self.dilation
        return layer(F.pad(x, (left, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T]. LayerNorm is applied in the channel-last view to avoid
        # batch statistics leaking across padded/missing time slots.
        residual = x
        y = self._conv(self.conv1, x).transpose(1, 2)
        y = F.silu(self.norm1(y)).transpose(1, 2)
        y = self.dropout(y)
        y = self._conv(self.conv2, y).transpose(1, 2)
        y = F.silu(self.norm2(y)).transpose(1, 2)
        y = self.dropout(y)
        return residual + y


class Stage3TCN(nn.Module):
    """Input and output shapes are fixed in the armor dimension only.

    obs: [B,T,4,5] = xyz, sin(yaw), cos(yaw)
    obs_mask: [B,T,4], event_mask: [B,T], event_time_s: [B,T], tau: [B,Q]
    """

    def __init__(self, channels: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.armor_mlp = nn.Sequential(
            nn.Linear(5, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU()
        )
        self.frame_projection = nn.Sequential(
            nn.Linear(32 + 32 + 1 + 2, channels), nn.SiLU()
        )
        self.tcn = nn.ModuleList(
            [CausalResidualBlock(channels, dilation, dropout) for dilation in
             (1, 2, 4, 8, 16, 32, 64, 128)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(channels + 2, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU()
        )
        self.position_mean = nn.Linear(128, 12)
        self.position_logvar = nn.Linear(128, 12)
        self.normal = nn.Linear(128, 12)
        self.motion_logits = nn.Linear(channels, 4)

    def encode(self, obs: torch.Tensor, obs_mask: torch.Tensor,
               event_mask: torch.Tensor, event_time_s: torch.Tensor) -> torch.Tensor:
        tokens = self.armor_mlp(obs)
        mask = obs_mask.to(dtype=torch.bool).unsqueeze(-1)
        count = mask.to(tokens.dtype).sum(dim=2)
        token_sum = (tokens * mask.to(tokens.dtype)).sum(dim=2)
        minimum = torch.finfo(tokens.dtype).min
        token_max = torch.where(mask, tokens, minimum).amax(dim=2)
        token_max = torch.where(count > 0, token_max, torch.zeros_like(token_max))
        count_feature = (count / 4.0).clamp(0.0, 1.0)
        previous_time = torch.cat((event_time_s[:, :1], event_time_s[:, :-1]), dim=1)
        delta_time = event_time_s - previous_time
        previous_valid = torch.cat(
            (torch.zeros_like(event_mask[:, :1]), event_mask[:, :-1]), dim=1
        )
        delta_time = torch.where(event_mask & previous_valid, delta_time, torch.zeros_like(delta_time))
        time_features = torch.stack(
            (event_time_s.clamp(-10.0, 0.0), delta_time.clamp(0.0, 1.0)), dim=-1
        ).to(tokens.dtype)
        pooled = torch.cat((token_sum, token_max, count_feature, time_features), dim=-1)
        frame = self.frame_projection(pooled)
        frame = frame * event_mask.to(frame.dtype).unsqueeze(-1)
        x = frame.transpose(1, 2)
        for block in self.tcn:
            x = block(x)
            x = x * event_mask.to(x.dtype).unsqueeze(1)
        return x[:, :, -1]

    def forward(self, obs: torch.Tensor, obs_mask: torch.Tensor,
                event_mask: torch.Tensor, event_time_s: torch.Tensor,
                tau: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self.encode(obs, obs_mask, event_mask, event_time_s)
        if tau.ndim == 1:
            tau = tau.unsqueeze(0).expand(obs.shape[0], -1)
        query_count = tau.shape[1]
        expanded = state.unsqueeze(1).expand(-1, query_count, -1)
        query = torch.cat((expanded, tau.unsqueeze(-1), tau.square().unsqueeze(-1)), dim=-1)
        decoded = self.decoder(query.reshape(-1, query.shape[-1]))
        position_mean = self.position_mean(decoded).reshape(-1, query_count, 4, 3)
        position_logvar = self.position_logvar(decoded).reshape(-1, query_count, 4, 3)
        position_logvar = position_logvar.clamp(-6.907755, 0.0)
        normal = self.normal(decoded).reshape(-1, query_count, 4, 3)
        normal = F.normalize(normal, dim=-1, eps=1e-6)
        return {
            "position_mean": position_mean,
            "position_logvar": position_logvar,
            "normal": normal,
            "motion_logits": self.motion_logits(state),
        }
