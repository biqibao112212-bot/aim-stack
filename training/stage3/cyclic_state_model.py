"""C4-equivariant q0 state recovery for temporary cyclic armor tracks.

The model restores only the current state.  Track indices are temporary cyclic
memory handles, never physical slot identities.  Current clean observations are
passed through exactly; only previously observed hidden tracks are learned.
"""

from __future__ import annotations

import torch
from torch import nn

from .cyclic_track_model import CyclicContextEncoder
from .model import CausalResidualBlock


def current_track_support(
    obs_mask: torch.Tensor,
    primary_mask: torch.Tensor,
    event_mask: torch.Tensor,
    event_time_s: torch.Tensor,
    *,
    history_events: int,
) -> dict[str, torch.Tensor]:
    """Return causal current/warm/cold masks for the consumed history only."""
    if obs_mask.ndim != 3 or obs_mask.shape[-1] != 4:
        raise ValueError("obs_mask must have shape [B,T,4]")
    if primary_mask.shape != obs_mask.shape:
        raise ValueError("primary_mask must match obs_mask")
    if event_mask.shape != obs_mask.shape[:2]:
        raise ValueError("event_mask must have shape [B,T]")
    if event_time_s.shape != obs_mask.shape[:2]:
        raise ValueError("event_time_s must have shape [B,T]")
    if not 1 <= history_events <= obs_mask.shape[1]:
        raise ValueError("history_events is outside the supplied history")

    visible = obs_mask[:, -history_events:].to(torch.bool)
    primary = primary_mask[:, -history_events:].to(torch.bool)
    active = event_mask[:, -history_events:].to(torch.bool)
    time = event_time_s[:, -history_events:]
    if bool(torch.any(visible & ~active.unsqueeze(-1))):
        raise ValueError("inactive events cannot contain visible tracks")
    if bool(torch.any(primary & ~visible)):
        raise ValueError("primary tracks must be visible")
    if bool(torch.any(active & (visible.sum(dim=-1) < 1))):
        raise ValueError("active events require at least one visible track")
    if bool(torch.any(active & (visible.sum(dim=-1) > 2))):
        raise ValueError("active events permit at most two visible tracks")

    batch, events, _ = visible.shape
    indices = torch.arange(events, device=visible.device).view(1, events)
    last_event = torch.where(
        active, indices, torch.full_like(indices, -1)
    ).amax(dim=1)
    if bool(torch.any(last_event < 0)):
        raise ValueError("every sample requires a causal event")
    gather_track = last_event[:, None, None].expand(-1, 1, 4)
    current_visible = visible.gather(1, gather_track).squeeze(1)
    current_primary = primary.gather(1, gather_track).squeeze(1)
    if bool(torch.any(current_primary.sum(dim=-1) != 1)):
        raise ValueError("the current event requires exactly one primary")

    seen = visible.any(dim=1)
    warm_hidden = seen & ~current_visible
    cold = ~seen
    pair_visible = visible & torch.roll(visible, shifts=-1, dims=2)
    pair_seen = pair_visible.any(dim=1)
    per_track_index = indices[:, :, None].expand(batch, -1, 4)
    last_seen_index = torch.where(
        visible, per_track_index, torch.full_like(per_track_index, -1)
    ).amax(dim=1)
    safe_last_seen = last_seen_index.clamp_min(0)
    last_seen_time = time[:, :, None].expand(-1, -1, 4).gather(
        1, safe_last_seen[:, None]
    ).squeeze(1)
    current_time = time.gather(1, last_event[:, None]).squeeze(1)
    current_event_is_q0 = current_time.abs() <= 1e-6
    q0_observed = current_visible & current_event_is_q0[:, None]
    age_s = (current_time[:, None] - last_seen_time).clamp_min(0.0)
    age_s = torch.where(seen, age_s, torch.full_like(age_s, float("inf")))

    primary_index = current_primary.to(torch.long).argmax(dim=-1)
    row = torch.arange(batch, device=visible.device)
    clockwise = torch.zeros_like(current_visible)
    counterclockwise = torch.zeros_like(current_visible)
    clockwise[row, (primary_index + 1) % 4] = True
    counterclockwise[row, (primary_index - 1) % 4] = True
    adjacent = clockwise | counterclockwise
    left_edge_support = (
        torch.roll(current_visible, shifts=1, dims=1)
        & torch.roll(pair_seen, shifts=1, dims=1)
    )
    right_edge_support = (
        torch.roll(current_visible, shifts=-1, dims=1) & pair_seen
    )
    edge_warm = warm_hidden & adjacent & (left_edge_support | right_edge_support)
    self_warm = warm_hidden & adjacent & ~edge_warm
    relevant_edge = torch.zeros_like(pair_seen)
    relevant_edge[row, primary_index] = True
    relevant_edge[row, (primary_index - 1) % 4] = True
    return {
        "current_event_index": last_event,
        "current_visible": current_visible,
        "q0_observed": q0_observed,
        "current_event_time_s": current_time,
        "current_primary": current_primary,
        "primary_index": primary_index,
        "seen": seen,
        "warm_hidden": warm_hidden,
        "cold": cold,
        "self_warm": self_warm,
        "edge_warm": edge_warm,
        "age_s": age_s,
        "adjacent": adjacent,
        "clockwise": clockwise,
        "counterclockwise": counterclockwise,
        "pair_seen": pair_seen,
        "relevant_edge": relevant_edge,
    }


class SharedEdgeTemporalEncoder(nn.Module):
    """Encode observed directed adjacent edges with weights shared over C4."""

    def __init__(self, channels: int, dropout: float, history_events: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.history_events = int(history_events)
        # normalized edge xyz, relative time, pair-visible flag, switch step.
        self.projection = nn.Sequential(
            nn.Linear(6, channels), nn.LayerNorm(channels), nn.SiLU(),
        )
        self.temporal = nn.ModuleList(
            CausalResidualBlock(channels, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16)
        )

    def forward(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        event_mask: torch.Tensor,
        event_time_s: torch.Tensor,
        switch_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = obs[:, -self.history_events:]
        visible = obs_mask[:, -self.history_events:].to(torch.bool)
        active = event_mask[:, -self.history_events:].to(torch.bool)
        time = event_time_s[:, -self.history_events:]
        switch = switch_step[:, -self.history_events:]
        pair_visible = visible & torch.roll(visible, shifts=-1, dims=2)
        edge = torch.roll(obs, shifts=-1, dims=2) - obs
        clean_edge = torch.where(
            pair_visible.unsqueeze(-1), edge, torch.zeros_like(edge)
        )
        batch, events, edges, _ = clean_edge.shape
        feature = torch.cat((
            clean_edge,
            time[:, :, None, None].expand(-1, -1, edges, 1),
            pair_visible.to(obs.dtype).unsqueeze(-1),
            switch[:, :, None, None].expand(-1, -1, edges, 1),
        ), dim=-1)
        feature = feature.permute(0, 2, 1, 3).reshape(batch * edges, events, 6)
        sequence = self.projection(feature).transpose(1, 2)
        temporal_mask = active[:, None].expand(-1, edges, -1).reshape(
            batch * edges, events
        )
        sequence = sequence * temporal_mask.to(sequence.dtype).unsqueeze(1)
        for block in self.temporal:
            sequence = block(sequence)
            sequence = sequence * temporal_mask.to(sequence.dtype).unsqueeze(1)
        indices = torch.arange(events, device=obs.device).view(1, events)
        last_event = torch.where(
            temporal_mask, indices, torch.full_like(indices, -1)
        ).amax(dim=1)
        gather = last_event.view(-1, 1, 1).expand(-1, self.channels, 1)
        encoded = sequence.gather(2, gather).squeeze(2).reshape(
            batch, edges, self.channels
        )

        per_edge_indices = indices.view(1, 1, events).expand(batch, edges, -1)
        pair_by_edge = pair_visible.permute(0, 2, 1)
        last_pair = torch.where(
            pair_by_edge, per_edge_indices, torch.full_like(per_edge_indices, -1)
        ).amax(dim=2)
        pair_seen = last_pair >= 0
        safe_pair = last_pair.clamp_min(0)
        edge_by_index = clean_edge.permute(0, 2, 1, 3)
        base = edge_by_index.gather(
            2, safe_pair[..., None, None].expand(-1, -1, 1, 3)
        ).squeeze(2)
        base = torch.where(pair_seen.unsqueeze(-1), base, torch.zeros_like(base))
        current_pair = pair_visible.gather(
            1, last_event.reshape(batch, edges)[:, :1, None].expand(-1, 1, 4)
        ).squeeze(1)
        return encoded, base, pair_seen, current_pair


class CyclicStateRestorer(nn.Module):
    """Restore q0 positions for visible and causally warm cyclic tracks."""

    model_family = "cyclic-equivariant-q0-state-restorer-v1"

    def __init__(
        self,
        position_mean: torch.Tensor,
        position_std: torch.Tensor,
        *,
        channels: int = 48,
        dropout: float = 0.05,
        history_events: int = 32,
    ) -> None:
        super().__init__()
        if tuple(position_mean.shape) != (3,) or tuple(position_std.shape) != (3,):
            raise ValueError("position normalization must have shape [3]")
        if bool(torch.any(position_std <= 0)):
            raise ValueError("position_std must be positive")
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.history_events = int(history_events)
        self.context = CyclicContextEncoder(
            channels, dropout, history_events, message_layers=3
        )
        self.edge_temporal = SharedEdgeTemporalEncoder(
            channels, dropout, history_events
        )
        self.track_edge_update = nn.Sequential(
            nn.Linear(3 * channels, 2 * channels), nn.SiLU(),
            nn.Linear(2 * channels, channels),
        )
        self.track_edge_norm = nn.LayerNorm(channels)
        features = 5 * channels
        self.position_head = nn.Sequential(
            nn.Linear(features, 2 * channels), nn.SiLU(),
            nn.Linear(2 * channels, 3),
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(features, channels), nn.SiLU(), nn.Linear(channels, 1),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(3 * channels, 2 * channels), nn.SiLU(),
            nn.Linear(2 * channels, 3),
        )
        self.register_buffer("position_mean", position_mean.float().clone())
        self.register_buffer("position_std", position_std.float().clone())
        nn.init.normal_(self.position_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.position_head[-1].bias)
        nn.init.zeros_(self.sigma_head[-1].weight)
        nn.init.constant_(self.sigma_head[-1].bias, -2.0)
        nn.init.normal_(self.edge_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.edge_head[-1].bias)

    def forward(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        primary_mask: torch.Tensor,
        event_mask: torch.Tensor,
        event_time_s: torch.Tensor,
        switch_step: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        state, pooled, base, _ = self.context(
            obs, obs_mask, primary_mask, event_mask, event_time_s, switch_step
        )
        edge_state, edge_base, pair_seen, current_pair = self.edge_temporal(
            obs, obs_mask, event_mask, event_time_s, switch_step
        )
        support = current_track_support(
            obs_mask, primary_mask, event_mask, event_time_s,
            history_events=self.history_events,
        )
        batch, tracks, _ = state.shape
        if not torch.equal(pair_seen, support["pair_seen"]):
            raise RuntimeError("edge and support histories disagree")
        left_edge_state = torch.roll(edge_state, shifts=1, dims=1)
        state = self.track_edge_norm(
            state + self.track_edge_update(torch.cat((
                state, left_edge_state, edge_state,
            ), dim=-1))
        )
        pooled = torch.cat((state.mean(dim=1), state.amax(dim=1)), dim=-1)
        global_state = pooled[:, None].expand(-1, tracks, -1)
        context = torch.cat((
            state, global_state, left_edge_state, edge_state,
        ), dim=-1)
        restored_normalized = base + self.position_head(context)

        event_index = support["current_event_index"]
        absolute_event_index = event_index + (obs.shape[1] - self.history_events)
        current_obs = obs.gather(
            1, absolute_event_index[:, None, None, None].expand(-1, 1, 4, 3)
        ).squeeze(1)
        q0_normalized = torch.where(
            support["q0_observed"].unsqueeze(-1),
            current_obs,
            restored_normalized,
        )
        q0_m = self.position_mean + self.position_std * q0_normalized

        edge_context = torch.cat((
            edge_state, pooled[:, None].expand(-1, tracks, -1),
        ), dim=-1)
        restored_edge_normalized = edge_base + self.edge_head(edge_context)
        current_edge_normalized = (
            torch.roll(current_obs, shifts=-1, dims=1) - current_obs
        )
        current_pair_q0 = (
            current_pair
            & support["current_event_time_s"].abs().le(1e-6)[:, None]
        )
        edge0_normalized = torch.where(
            current_pair_q0.unsqueeze(-1), current_edge_normalized,
            restored_edge_normalized,
        )
        edge0_m = self.position_std * edge0_normalized

        learned_sigma = torch.nn.functional.softplus(self.sigma_head(context)) + 1e-4
        learned_sigma_m = learned_sigma * self.position_std.mean()
        sigma_m = torch.where(
            support["q0_observed"].unsqueeze(-1),
            torch.full_like(learned_sigma_m, 1e-6),
            learned_sigma_m,
        )
        sigma_m = torch.where(
            support["cold"].unsqueeze(-1),
            torch.full_like(sigma_m, 1.0),
            sigma_m,
        )
        confidence = torch.where(
            support["seen"], torch.exp(-sigma_m.squeeze(-1) / 0.1),
            torch.zeros_like(sigma_m.squeeze(-1)),
        )
        result = {
            "q0_m": q0_m,
            "q0_valid": support["seen"],
            "q0_sigma_m": sigma_m,
            "confidence": confidence,
            "track_context": state,
            "edge_context": edge_state,
            "edge0_m": edge0_m,
            "edge0_valid": support["pair_seen"],
            "current_pair_visible": current_pair,
            "current_pair_q0_observed": current_pair_q0,
            **support,
        }
        return result

    def config(self) -> dict[str, object]:
        return {
            "family": self.model_family,
            "channels": self.channels,
            "dropout": self.dropout,
            "history_events": self.history_events,
            "position_mean": self.position_mean.detach().cpu().tolist(),
            "position_std": self.position_std.detach().cpu().tolist(),
            "track_identity": "temporary cyclic state handles only",
            "q0_observed_identity_bypass": True,
            "stale_visible_propagated_to_q0": True,
            "cold_tracks_are_invalid": True,
            "sample_specific_adjacent_edge_memory": True,
            "fixed_geometry": False,
            "slot_features": False,
        }
