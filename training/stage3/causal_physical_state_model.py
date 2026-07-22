"""Neural fixed-slot physical A/B models with no analytic state recovery."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .physical_model import CausalResidualBlock


def _expanded_tau(tau: torch.Tensor, batch: int) -> torch.Tensor:
    if tau.ndim == 1:
        return tau.unsqueeze(0).expand(batch, -1)
    if tau.ndim != 2 or tau.shape[0] != batch:
        raise ValueError("tau must have shape [Q] or [B,Q]")
    return tau


class FixedSlotHistoryEncoder(nn.Module):
    """Encode four persistent armor histories independently, then fuse them.

    No displacement division, least-squares fit, velocity, yaw, or yaw-rate
    reconstruction is performed here.  The only temporal coordinate is each
    observation's real anchor-relative timestamp.
    """

    def __init__(
        self, input_features: int = 5, channels: int = 64,
        dropout: float = 0.05, history_events: int = 32,
    ) -> None:
        super().__init__()
        if input_features < 3:
            raise ValueError("fixed-slot encoder requires xyz features")
        if channels % 4:
            raise ValueError("channels must be divisible by four")
        if history_events < 8 or history_events > 200:
            raise ValueError("history_events must be within [8,200]")
        self.input_features = int(input_features)
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.history_events = int(history_events)
        self.projection = nn.Sequential(
            nn.Linear(input_features + 1, channels),
            nn.LayerNorm(channels), nn.SiLU(),
        )
        self.temporal = nn.ModuleList(
            CausalResidualBlock(channels, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16)
        )
        self.fusion = nn.Sequential(
            nn.Linear(4 * channels + 4, 2 * channels),
            nn.LayerNorm(2 * channels), nn.SiLU(),
            nn.Linear(2 * channels, channels),
            nn.LayerNorm(channels), nn.SiLU(),
        )

    def forward(
        self, obs: torch.Tensor, obs_mask: torch.Tensor,
        event_mask: torch.Tensor, event_time_s: torch.Tensor,
    ) -> torch.Tensor:
        if obs.ndim != 4 or obs.shape[2] != 4 or obs.shape[3] != self.input_features:
            raise ValueError("obs must have shape [B,T,4,F]")
        if obs_mask.shape != obs.shape[:3] or event_mask.shape != obs.shape[:2]:
            raise ValueError("history masks do not match obs")
        if event_time_s.shape != obs.shape[:2]:
            raise ValueError("event_time_s must have shape [B,T]")
        obs = obs[:, -self.history_events:]
        obs_mask = obs_mask[:, -self.history_events:]
        event_mask = event_mask[:, -self.history_events:]
        event_time_s = event_time_s[:, -self.history_events:]
        batch, time, slots, _ = obs.shape
        valid = (
            obs_mask.to(torch.bool)
            & event_mask.to(torch.bool).unsqueeze(-1)
            & torch.isfinite(obs).all(dim=-1)
            & torch.isfinite(event_time_s).unsqueeze(-1)
            & (event_time_s <= 1e-6).unsqueeze(-1)
        )
        clean = torch.where(valid.unsqueeze(-1), obs, torch.zeros_like(obs))
        timestamp = torch.where(
            valid, event_time_s.unsqueeze(-1).expand(-1, -1, slots),
            torch.zeros_like(valid, dtype=event_time_s.dtype),
        )
        feature = torch.cat((clean, timestamp.unsqueeze(-1)), dim=-1)
        # [B,T,4,F] -> four independent histories sharing one encoder.
        feature = feature.permute(0, 2, 1, 3).reshape(
            batch * slots, time, self.input_features + 1
        )
        slot_mask = valid.permute(0, 2, 1).reshape(batch * slots, time)
        sequence = self.projection(feature).transpose(1, 2)
        for block in self.temporal:
            sequence = block(sequence)
            sequence = sequence * slot_mask.to(sequence.dtype).unsqueeze(1)
        indices = torch.arange(time, device=obs.device).view(1, time)
        last = torch.where(
            slot_mask, indices, torch.full_like(indices, -1)
        ).amax(dim=1).clamp_min(0)
        gather = last.view(-1, 1, 1).expand(-1, self.channels, 1)
        encoded = sequence.gather(2, gather).squeeze(2)
        present = slot_mask.any(dim=1)
        encoded = encoded * present.to(encoded.dtype).unsqueeze(-1)
        encoded = encoded.reshape(batch, slots, self.channels)
        present = present.reshape(batch, slots)
        fused = torch.cat((encoded.flatten(1), present.to(encoded.dtype)), dim=-1)
        return self.fusion(fused)


class FixedGeometryDecoder(nn.Module):
    """Frozen FP32 rigid decoder shared by both experimental arms."""

    def __init__(self, geometry: torch.Tensor) -> None:
        super().__init__()
        if tuple(geometry.shape) != (4, 3):
            raise ValueError("geometry must have shape [4,3]")
        self.register_buffer("geometry", geometry.float().clone())

    def forward(self, center: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        if center.ndim != 3 or center.shape[-1] != 3:
            raise ValueError("center must have shape [B,Q,3]")
        if phase.shape != center.shape[:2] + (2,):
            raise ValueError("phase must have shape [B,Q,2]")
        with torch.autocast(device_type=center.device.type, enabled=False):
            center32 = center.float()
            phase32 = F.normalize(phase.float(), dim=-1, eps=1e-8)
            cosine, sine = phase32[..., 0], phase32[..., 1]
            gx = self.geometry[:, 0].view(1, 1, 4)
            gy = self.geometry[:, 1].view(1, 1, 4)
            gz = self.geometry[:, 2].view(1, 1, 4).expand(
                center.shape[0], center.shape[1], -1
            )
            x = cosine.unsqueeze(-1) * gx - sine.unsqueeze(-1) * gy
            y = sine.unsqueeze(-1) * gx + cosine.unsqueeze(-1) * gy
            relative = torch.stack((x, y, gz), dim=-1)
            return center32.unsqueeze(2) + relative


class _StateBase(nn.Module):
    def _init_common(
        self, geometry: torch.Tensor, position_mean: torch.Tensor,
        position_std: torch.Tensor, input_features: int, channels: int,
        dropout: float, history_events: int,
    ) -> None:
        if tuple(position_mean.shape) != (3,) or tuple(position_std.shape) != (3,):
            raise ValueError("position normalization must have shape [3]")
        if bool(torch.any(position_std <= 0)):
            raise ValueError("position_std must be positive")
        self.input_features = int(input_features)
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.history_events = int(history_events)
        self.encoder = FixedSlotHistoryEncoder(
            input_features, channels, dropout, history_events
        )
        self.decoder = FixedGeometryDecoder(geometry)
        center_reference = position_mean.float() - geometry.float().mean(dim=0)
        self.register_buffer("center_reference", center_reference.clone())
        self.register_buffer("center_scale", position_std.float().clamp_min(0.25))

    def _center(self, raw: torch.Tensor) -> torch.Tensor:
        # The scale conditions optimization but must not limit the reachable
        # workspace to the train mean plus/minus one standard deviation.
        return self.center_reference + self.center_scale * raw

    @staticmethod
    def _phase(raw_yaw: torch.Tensor) -> torch.Tensor:
        yaw = torch.pi * torch.tanh(raw_yaw)
        return torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)


class ExplicitStatePhysicalPredictor(_StateBase):
    """A: infer one shared state, then apply frozen constant-twist physics."""

    model_family = "fixed-slot-neural-explicit-state-v1"

    def __init__(
        self, geometry: torch.Tensor, position_mean: torch.Tensor,
        position_std: torch.Tensor, input_features: int = 5,
        channels: int = 64, dropout: float = 0.05,
        history_events: int = 32, maximum_speed_mps: float = 3.5,
        maximum_yaw_rate_rad_s: float = 15.0,
    ) -> None:
        super().__init__()
        self._init_common(
            geometry, position_mean, position_std, input_features,
            channels, dropout, history_events,
        )
        self.maximum_speed_mps = float(maximum_speed_mps)
        self.maximum_yaw_rate_rad_s = float(maximum_yaw_rate_rad_s)
        self.state_head = nn.Sequential(
            nn.Linear(channels, 192), nn.SiLU(),
            nn.Linear(192, 192), nn.SiLU(), nn.Linear(192, 8),
        )
        nn.init.normal_(self.state_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.state_head[-1].bias)

    def forward(
        self, obs: torch.Tensor, obs_mask: torch.Tensor,
        event_mask: torch.Tensor, event_time_s: torch.Tensor,
        tau: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(obs, obs_mask, event_mask, event_time_s)
        raw = self.state_head(encoded)
        center0 = self._center(raw[:, :3])
        velocity = self.maximum_speed_mps * torch.tanh(raw[:, 3:6])
        phase0 = self._phase(raw[:, 6])
        omega = self.maximum_yaw_rate_rad_s * torch.tanh(raw[:, 7])
        tau = _expanded_tau(tau, obs.shape[0])
        center = center0[:, None] + tau.unsqueeze(-1) * velocity[:, None]
        angle = tau * omega[:, None]
        cosine, sine = torch.cos(angle), torch.sin(angle)
        phase = torch.stack((
            phase0[:, None, 0] * cosine - phase0[:, None, 1] * sine,
            phase0[:, None, 1] * cosine + phase0[:, None, 0] * sine,
        ), dim=-1)
        return {
            "position_mean": self.decoder(center, phase),
            "query_center": center, "query_phase": phase,
            "center0": center0, "velocity": velocity,
            "phase0": phase0, "omega": omega,
        }

    def config(self) -> dict[str, object]:
        return {
            "family": self.model_family, "input_features": self.input_features,
            "channels": self.channels, "dropout": self.dropout,
            "history_events": self.history_events,
            "maximum_speed_mps": self.maximum_speed_mps,
            "maximum_yaw_rate_rad_s": self.maximum_yaw_rate_rad_s,
            "geometry": self.decoder.geometry.detach().cpu().tolist(),
            "position_mean": (
                self.center_reference + self.decoder.geometry.mean(dim=0)
            ).detach().cpu().tolist(),
            "position_std": self.center_scale.detach().cpu().tolist(),
        }


class ImplicitQueryPhysicalPredictor(_StateBase):
    """B: infer each query pose directly, with no shared velocity or omega."""

    model_family = "fixed-slot-neural-query-pose-v1"

    def __init__(
        self, geometry: torch.Tensor, position_mean: torch.Tensor,
        position_std: torch.Tensor, input_features: int = 5,
        channels: int = 64, dropout: float = 0.05,
        history_events: int = 32,
    ) -> None:
        super().__init__()
        self._init_common(
            geometry, position_mean, position_std, input_features,
            channels, dropout, history_events,
        )
        self.query_head = nn.Sequential(
            nn.Linear(channels + 3, 192), nn.SiLU(),
            nn.Linear(192, 192), nn.SiLU(), nn.Linear(192, 4),
        )
        nn.init.normal_(self.query_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.query_head[-1].bias)

    def forward(
        self, obs: torch.Tensor, obs_mask: torch.Tensor,
        event_mask: torch.Tensor, event_time_s: torch.Tensor,
        tau: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(obs, obs_mask, event_mask, event_time_s)
        tau = _expanded_tau(tau, obs.shape[0])
        expanded = encoded.unsqueeze(1).expand(-1, tau.shape[1], -1)
        query = torch.cat((
            expanded, tau.unsqueeze(-1), tau.square().unsqueeze(-1),
            tau.pow(3).unsqueeze(-1),
        ), dim=-1)
        raw = self.query_head(query)
        center = self._center(raw[..., :3])
        phase = self._phase(raw[..., 3])
        return {
            "position_mean": self.decoder(center, phase),
            "query_center": center, "query_phase": phase,
        }

    def config(self) -> dict[str, object]:
        return {
            "family": self.model_family, "input_features": self.input_features,
            "channels": self.channels, "dropout": self.dropout,
            "history_events": self.history_events,
            "geometry": self.decoder.geometry.detach().cpu().tolist(),
            "position_mean": (
                self.center_reference + self.decoder.geometry.mean(dim=0)
            ).detach().cpu().tolist(),
            "position_std": self.center_scale.detach().cpu().tolist(),
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
