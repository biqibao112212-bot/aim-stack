"""C4-equivariant clean-physics predictor for four temporary armor tracks."""

from __future__ import annotations

import torch
from torch import nn

from .model import CausalResidualBlock


ROUTE_NAMES = ("stationary", "translation", "rotation", "combined")


def _expanded_tau(tau: torch.Tensor, batch: int) -> torch.Tensor:
    if tau.ndim == 1:
        return tau.unsqueeze(0).expand(batch, -1)
    if tau.ndim != 2 or tau.shape[0] != batch:
        raise ValueError("tau must have shape [Q] or [B,Q]")
    return tau


class SharedTrackTemporalEncoder(nn.Module):
    """Apply one causal temporal network identically to all four tracks."""

    def __init__(
        self, channels: int = 48, dropout: float = 0.05,
        history_events: int = 32,
    ) -> None:
        super().__init__()
        if channels < 8 or channels % 4:
            raise ValueError("channels must be at least 8 and divisible by four")
        if not 8 <= history_events <= 200:
            raise ValueError("history_events must be within [8,200]")
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.history_events = int(history_events)
        # xyz, anchor-relative time, visible flag, primary flag, switch step.
        self.projection = nn.Sequential(
            nn.Linear(7, channels), nn.LayerNorm(channels), nn.SiLU(),
        )
        self.temporal = nn.ModuleList(
            CausalResidualBlock(channels, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16)
        )

    def forward(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        primary_mask: torch.Tensor,
        event_mask: torch.Tensor,
        event_time_s: torch.Tensor,
        switch_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.ndim != 4 or obs.shape[2:] != (4, 3):
            raise ValueError("obs must have shape [B,T,4,3]")
        if obs_mask.shape != obs.shape[:3] or primary_mask.shape != obs.shape[:3]:
            raise ValueError("track masks must match obs [B,T,4]")
        if event_mask.shape != obs.shape[:2]:
            raise ValueError("event_mask must have shape [B,T]")
        if event_time_s.shape != obs.shape[:2] or switch_step.shape != obs.shape[:2]:
            raise ValueError("time and switch_step must have shape [B,T]")
        if bool(torch.any(obs_mask.sum(dim=-1) > 2)):
            raise ValueError("at most two tracks may be visible per event")
        active = event_mask.to(torch.bool)
        visible_count = obs_mask.sum(dim=-1)
        if bool(torch.any(active & ((visible_count < 1) | (visible_count > 2)))):
            raise ValueError("active events require one or two visible tracks")
        if bool(torch.any(~active & (visible_count != 0))):
            raise ValueError("padded events cannot contain visible tracks")
        primary_count = primary_mask.sum(dim=-1)
        if bool(torch.any(active & (primary_count != 1))):
            raise ValueError("active events require exactly one primary track")
        if bool(torch.any(~active & (primary_count != 0))):
            raise ValueError("padded events cannot contain a primary track")
        if bool(torch.any(primary_mask & ~obs_mask)):
            raise ValueError("the primary track must be visible")
        if bool(torch.any(~torch.isin(switch_step, switch_step.new_tensor((-1, 0, 1))))):
            raise ValueError("switch_step values must be -1, 0, or +1")
        active_finite = torch.isfinite(obs).all(dim=-1)
        if bool(torch.any(obs_mask & ~active_finite)):
            raise ValueError("visible track coordinates must be finite")
        primary_index = primary_mask.to(torch.long).argmax(dim=-1)
        previous_active = torch.cat(
            (torch.zeros_like(active[:, :1]), active[:, :-1]), dim=1
        )
        consecutive = active & previous_active
        previous_index = torch.cat(
            (primary_index[:, :1], primary_index[:, :-1]), dim=1
        )
        delta = (primary_index - previous_index) % 4
        if bool(torch.any(consecutive & (delta == 2))):
            raise ValueError("primary track cannot jump to the opposite track")
        expected_switch = torch.where(
            delta == 1, torch.ones_like(switch_step),
            torch.where(delta == 3, -torch.ones_like(switch_step),
                        torch.zeros_like(switch_step)),
        )
        if bool(torch.any(consecutive & (switch_step != expected_switch))):
            raise ValueError("switch_step is inconsistent with primary history")
        first_active = active & ~previous_active
        if bool(torch.any(first_active & (switch_step != 0))):
            raise ValueError("the first active event must have switch_step zero")

        obs = obs[:, -self.history_events:]
        obs_mask = obs_mask[:, -self.history_events:].to(torch.bool)
        primary_mask = primary_mask[:, -self.history_events:].to(torch.bool)
        event_mask = event_mask[:, -self.history_events:].to(torch.bool)
        event_time_s = event_time_s[:, -self.history_events:]
        switch_step = switch_step[:, -self.history_events:]
        batch, time, tracks, _ = obs.shape
        finite_event = (
            event_mask & torch.isfinite(event_time_s)
            & torch.isfinite(switch_step) & (event_time_s <= 1e-6)
        )
        if bool(torch.any(event_mask & ~finite_event)):
            raise ValueError("active events require finite non-future time and switch")
        visible = obs_mask & finite_event.unsqueeze(-1) & torch.isfinite(obs).all(dim=-1)
        clean_xyz = torch.where(visible.unsqueeze(-1), obs, torch.zeros_like(obs))
        safe_time = torch.where(finite_event, event_time_s, torch.zeros_like(event_time_s))
        safe_switch = torch.where(
            finite_event, switch_step, torch.zeros_like(switch_step)
        )
        feature = torch.cat((
            clean_xyz,
            safe_time[:, :, None, None].expand(-1, -1, tracks, 1),
            visible.to(obs.dtype).unsqueeze(-1),
            (primary_mask & finite_event.unsqueeze(-1)).to(obs.dtype).unsqueeze(-1),
            safe_switch[:, :, None, None].expand(-1, -1, tracks, 1),
        ), dim=-1)
        feature = feature.permute(0, 2, 1, 3).reshape(batch * tracks, time, 7)
        sequence = self.projection(feature).transpose(1, 2)
        temporal_mask = finite_event[:, None].expand(-1, tracks, -1).reshape(
            batch * tracks, time
        )
        for block in self.temporal:
            sequence = block(sequence)
            sequence = sequence * temporal_mask.to(sequence.dtype).unsqueeze(1)
        indices = torch.arange(time, device=obs.device).view(1, time)
        last_event = torch.where(
            temporal_mask, indices, torch.full_like(indices, -1)
        ).amax(dim=1)
        if bool(torch.any(last_event < 0)):
            raise ValueError("each sample must contain at least one valid event")
        gather = last_event.view(-1, 1, 1).expand(-1, self.channels, 1)
        encoded = sequence.gather(2, gather).squeeze(2).reshape(
            batch, tracks, self.channels
        )

        per_track_indices = indices.view(1, 1, time).expand(batch, tracks, -1)
        track_visible = visible.permute(0, 2, 1)
        last_seen = torch.where(
            track_visible, per_track_indices, torch.full_like(per_track_indices, -1)
        ).amax(dim=2)
        seen = last_seen >= 0
        safe_last_seen = last_seen.clamp_min(0)
        normalized_by_track = obs.permute(0, 2, 1, 3)
        last_xyz = normalized_by_track.gather(
            2, safe_last_seen[..., None, None].expand(-1, -1, 1, 3)
        ).squeeze(2)
        last_xyz = torch.where(seen.unsqueeze(-1), last_xyz, torch.zeros_like(last_xyz))
        count = seen.sum(dim=1, keepdim=True).clamp_min(1).to(obs.dtype)
        visible_mean = last_xyz.sum(dim=1, keepdim=True) / count.unsqueeze(-1)
        base = torch.where(seen.unsqueeze(-1), last_xyz, visible_mean.expand_as(last_xyz))
        return encoded, base, seen


class CyclicMessageBlock(nn.Module):
    """Directed circular neighbor message passing with slot-shared weights."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(3 * channels, 2 * channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(2 * channels, channels),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or state.shape[1] != 4:
            raise ValueError("cyclic state must have shape [B,4,C]")
        previous = torch.roll(state, shifts=1, dims=1)
        following = torch.roll(state, shifts=-1, dims=1)
        return self.norm(state + self.update(torch.cat((state, previous, following), dim=-1)))


class CyclicContextEncoder(nn.Module):
    """C4-equivariant per-track context and invariant pooled context."""

    def __init__(
        self, channels: int, dropout: float, history_events: int,
        message_layers: int = 3,
    ) -> None:
        super().__init__()
        if message_layers < 2:
            raise ValueError("at least two cyclic message layers are required")
        self.temporal = SharedTrackTemporalEncoder(
            channels, dropout, history_events
        )
        self.messages = nn.ModuleList(
            CyclicMessageBlock(channels, dropout) for _ in range(message_layers)
        )

    def forward(
        self, obs: torch.Tensor, obs_mask: torch.Tensor,
        primary_mask: torch.Tensor, event_mask: torch.Tensor,
        event_time_s: torch.Tensor, switch_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state, base, seen = self.temporal(
            obs, obs_mask, primary_mask, event_mask, event_time_s, switch_step
        )
        for message in self.messages:
            state = message(state)
        pooled = torch.cat((state.mean(dim=1), state.amax(dim=1)), dim=-1)
        return state, pooled, base, seen


class CyclicTrajectoryExpert(nn.Module):
    """One independent direct trajectory specialist with shared track decoder."""

    def __init__(
        self, channels: int, dropout: float, history_events: int,
        position_mean: torch.Tensor, position_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.context = CyclicContextEncoder(channels, dropout, history_events)
        context_features = 3 * channels
        self.anchor_head = nn.Sequential(
            nn.Linear(context_features, 2 * channels), nn.SiLU(),
            nn.Linear(2 * channels, 3),
        )
        self.rate_head = nn.Sequential(
            nn.Linear(context_features + 3, 2 * channels), nn.SiLU(),
            nn.Linear(2 * channels, 2 * channels), nn.SiLU(),
            nn.Linear(2 * channels, 3),
        )
        if tuple(position_mean.shape) != (3,) or tuple(position_std.shape) != (3,):
            raise ValueError("position normalization must have shape [3]")
        if bool(torch.any(position_std <= 0)):
            raise ValueError("position_std must be positive")
        self.register_buffer("position_mean", position_mean.float().clone())
        self.register_buffer("position_std", position_std.float().clone())
        nn.init.normal_(self.anchor_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.anchor_head[-1].bias)
        nn.init.normal_(self.rate_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.rate_head[-1].bias)

    def forward(
        self, obs: torch.Tensor, obs_mask: torch.Tensor,
        primary_mask: torch.Tensor, event_mask: torch.Tensor,
        event_time_s: torch.Tensor, switch_step: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        state, pooled, base, _ = self.context(
            obs, obs_mask, primary_mask, event_mask, event_time_s, switch_step
        )
        batch, tracks, _ = state.shape
        tau = _expanded_tau(tau, batch)
        global_state = pooled[:, None].expand(-1, tracks, -1)
        track_context = torch.cat((state, global_state), dim=-1)
        anchor = base + self.anchor_head(track_context)
        query_count = tau.shape[1]
        expanded = track_context[:, None].expand(-1, query_count, -1, -1)
        time = tau[:, :, None, None].expand(-1, -1, tracks, 1)
        time_feature = torch.cat((time, time.square(), time.pow(3)), dim=-1)
        normalized = anchor[:, None] + time * self.rate_head(
            torch.cat((expanded, time_feature), dim=-1)
        )
        position = self.position_mean + self.position_std * normalized
        return position


class CyclicTrackExpertSystem(nn.Module):
    """Four independent trajectory experts plus a C4-invariant router."""

    model_family = "cyclic-equivariant-independent-track-experts-v1"
    route_names = ROUTE_NAMES

    def __init__(
        self, position_mean: torch.Tensor, position_std: torch.Tensor,
        *, channels: int = 48, dropout: float = 0.05,
        history_events: int = 32,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.history_events = int(history_events)
        self.experts = nn.ModuleList(
            CyclicTrajectoryExpert(
                channels, dropout, history_events, position_mean, position_std
            ) for _ in ROUTE_NAMES
        )
        self.router_context = CyclicContextEncoder(
            channels, dropout, history_events
        )
        self.router_head = nn.Sequential(
            nn.Linear(2 * channels, 2 * channels), nn.SiLU(),
            nn.Linear(2 * channels, len(ROUTE_NAMES)),
        )

    def forward(
        self, obs: torch.Tensor, obs_mask: torch.Tensor,
        primary_mask: torch.Tensor, event_mask: torch.Tensor,
        event_time_s: torch.Tensor, switch_step: torch.Tensor,
        tau: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        expert_position = torch.stack([
            expert(
                obs, obs_mask, primary_mask, event_mask,
                event_time_s, switch_step, tau,
            ) for expert in self.experts
        ], dim=1)
        _, router_state, _, _ = self.router_context(
            obs, obs_mask, primary_mask, event_mask, event_time_s, switch_step
        )
        router_logit = self.router_head(router_state)
        router_probability = torch.softmax(router_logit, dim=-1)
        route_index = router_logit.argmax(dim=-1)
        selected = expert_position[
            torch.arange(obs.shape[0], device=obs.device), route_index
        ]
        return {
            "expert_position": expert_position,
            "router_logit": router_logit,
            "router_probability": router_probability,
            "route_index": route_index,
            "position_mean": selected,
        }

    def config(self) -> dict[str, object]:
        return {
            "family": self.model_family,
            "channels": self.channels,
            "dropout": self.dropout,
            "history_events": self.history_events,
            "route_names": list(self.route_names),
            "position_mean": self.experts[0].position_mean.detach().cpu().tolist(),
            "position_std": self.experts[0].position_std.detach().cpu().tolist(),
            "cyclic_equivariance": "C4 exact in eval mode",
            "slot_features": False,
            "fixed_geometry": False,
            "combined_is_independent": True,
        }
