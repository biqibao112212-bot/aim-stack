"""Hierarchical moving-only refinement for translation versus combined motion."""

from __future__ import annotations

import torch
from torch import nn

from .causal_physical_state_model import (
    FixedSlotHistoryEncoder,
    IndependentMotionExpertSystem,
)


class MovingRefinementSystem(nn.Module):
    """Keep a frozen v14 system and refine only its moving route."""

    model_family = "fixed-slot-moving-refinement-v1"
    route_names = ("stationary", "translation", "rotation", "combined")

    def __init__(
        self, base: IndependentMotionExpertSystem, *,
        position_mean: torch.Tensor, position_std: torch.Tensor,
        refinement_channels: int = 64, refinement_dropout: float = 0.05,
        history_events: int = 32,
    ) -> None:
        super().__init__()
        if refinement_channels % 4:
            raise ValueError("refinement channels must be divisible by four")
        if history_events != base.history_events:
            raise ValueError("refinement history must match the frozen base")
        self.base = base
        self.history_events = int(history_events)
        self.refinement_channels = int(refinement_channels)
        self.refinement_dropout = float(refinement_dropout)
        geometry = base.decoder.geometry.detach().float()
        radius = torch.sqrt(geometry[:, :2].square().sum(dim=-1).mean())
        self.register_buffer("relative_scale_m", radius.clamp_min(1e-3))
        self.register_buffer(
            "position_mean", position_mean.detach().float().clone(),
        )
        self.register_buffer(
            "position_std", position_std.detach().float().clone(),
        )
        self.refinement_encoder = FixedSlotHistoryEncoder(
            input_features=5, channels=refinement_channels,
            dropout=refinement_dropout, history_events=history_events,
        )
        self.refinement_head = nn.Sequential(
            nn.Linear(refinement_channels, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1),
        )
        nn.init.normal_(self.refinement_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.refinement_head[-1].bias)
        self.freeze_base()

    def freeze_base(self) -> None:
        self.base.requires_grad_(False)
        self.base.eval()

    @property
    def decoder(self) -> nn.Module:
        """Expose the frozen decoder through the common validation interface."""
        return self.base.decoder

    def train(self, mode: bool = True) -> "MovingRefinementSystem":
        super().train(mode)
        self.base.eval()
        return self

    def relative_observation(
        self, obs: torch.Tensor, obs_mask: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Remove per-event common translation without temporal differencing."""
        valid = obs_mask.to(torch.bool) & event_mask.to(torch.bool).unsqueeze(-1)
        physical = obs[..., :3] * self.position_std + self.position_mean
        physical = torch.where(
            valid.unsqueeze(-1), physical, torch.zeros_like(physical),
        )
        count = valid.sum(dim=2, keepdim=True).clamp_min(1).to(physical.dtype)
        center = physical.sum(dim=2, keepdim=True) / count.unsqueeze(-1)
        relative = (physical - center) / self.relative_scale_m
        relative = torch.where(
            valid.unsqueeze(-1), relative, torch.zeros_like(relative),
        )
        slot = torch.where(
            valid.unsqueeze(-1), obs[..., 3:5], torch.zeros_like(obs[..., 3:5]),
        )
        return torch.cat((relative, slot), dim=-1)

    def refinement_logit(
        self, obs: torch.Tensor, obs_mask: torch.Tensor,
        event_mask: torch.Tensor, event_time_s: torch.Tensor,
    ) -> torch.Tensor:
        relative = self.relative_observation(obs, obs_mask, event_mask)
        encoded = self.refinement_encoder(
            relative, obs_mask, event_mask, event_time_s,
        )
        return self.refinement_head(encoded).squeeze(-1)

    @staticmethod
    def _expanded_tau(tau: torch.Tensor, batch: int) -> torch.Tensor:
        if tau.ndim == 1:
            tau = tau.unsqueeze(0).expand(batch, -1)
        if tau.ndim != 2 or tau.shape[0] != batch:
            raise ValueError("tau must have shape [Q] or [B,Q]")
        return tau

    def forward(
        self, obs: torch.Tensor, obs_mask: torch.Tensor,
        event_mask: torch.Tensor, event_time_s: torch.Tensor,
        tau: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            base = self.base(
                obs, obs_mask, event_mask, event_time_s, tau,
            )
        refinement_logit = self.refinement_logit(
            obs, obs_mask, event_mask, event_time_s,
        )
        refinement_probability = torch.sigmoid(refinement_logit).clamp(
            1e-6, 1.0 - 1e-6,
        )
        move_probability = base["move_probability"]
        coarse_move = move_probability >= 0.5
        nonmove_probability = base["router_probability"][:, (0, 2)]
        nonmove_rotate_probability = (
            nonmove_probability[:, 1] / nonmove_probability.sum(dim=-1).clamp_min(1e-6)
        ).clamp(1e-6, 1.0 - 1e-6)
        coarse_nonmove_rotate = nonmove_rotate_probability >= 0.5
        refined_combined = refinement_probability >= 0.5
        route_index = torch.where(
            coarse_move,
            torch.where(
                refined_combined,
                torch.full_like(base["route_index"], 3),
                torch.full_like(base["route_index"], 1),
            ),
            torch.where(
                coarse_nonmove_rotate,
                torch.full_like(base["route_index"], 2),
                torch.zeros_like(base["route_index"]),
            ),
        )
        route_probability = torch.stack((
            (1.0 - move_probability) * (1.0 - nonmove_rotate_probability),
            move_probability * (1.0 - refinement_probability),
            (1.0 - move_probability) * nonmove_rotate_probability,
            move_probability * refinement_probability,
        ), dim=-1).clamp_min(1e-8)
        route_probability = route_probability / route_probability.sum(
            dim=-1, keepdim=True,
        )
        router_logit = torch.log(route_probability)
        translation_route = route_index == 1
        rotation_route = route_index == 2
        combined_route = route_index == 3
        velocity = torch.where(
            translation_route[:, None], base["translation_velocity"],
            torch.where(
                combined_route[:, None], base["combined_velocity"],
                torch.zeros_like(base["translation_velocity"]),
            ),
        )
        omega = torch.where(
            rotation_route, base["rotation_omega"],
            torch.where(
                combined_route, base["combined_omega"],
                torch.zeros_like(base["rotation_omega"]),
            ),
        )
        tau = self._expanded_tau(tau, obs.shape[0])
        center = base["center0"][:, None] + tau.unsqueeze(-1) * velocity[:, None]
        angle = tau * omega[:, None]
        cosine, sine = torch.cos(angle), torch.sin(angle)
        phase0 = base["phase0"]
        phase = torch.stack((
            phase0[:, None, 0] * cosine - phase0[:, None, 1] * sine,
            phase0[:, None, 1] * cosine + phase0[:, None, 0] * sine,
        ), dim=-1)
        rotate_probability = (
            route_probability[:, 2] + route_probability[:, 3]
        ).clamp(1e-6, 1.0 - 1e-6)
        result = dict(base)
        result.update({
            "position_mean": self.base.decoder(center, phase),
            "query_center": center,
            "query_phase": phase,
            "router_logit": router_logit,
            "router_probability": route_probability,
            "route_index": route_index,
            "move_logit": torch.logit(move_probability),
            "move_probability": move_probability,
            "move_active": coarse_move,
            "rotate_logit": torch.logit(rotate_probability),
            "rotate_probability": rotate_probability,
            "rotate_active": rotation_route | combined_route,
            "velocity": velocity,
            "omega": omega,
            "refinement_logit": refinement_logit,
            "refinement_probability": refinement_probability,
        })
        return result

    def config(self) -> dict[str, object]:
        return {
            "family": self.model_family,
            "history_events": self.history_events,
            "refinement_channels": self.refinement_channels,
            "refinement_dropout": self.refinement_dropout,
            "relative_scale_m": float(self.relative_scale_m.detach().cpu()),
            "position_mean": self.position_mean.detach().cpu().tolist(),
            "position_std": self.position_std.detach().cpu().tolist(),
            "base_config": self.base.config(),
        }
