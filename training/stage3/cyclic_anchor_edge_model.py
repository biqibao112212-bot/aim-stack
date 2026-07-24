"""Anchor-relative q0 restoration for temporary cyclic armor tracks.

Visible tracks remain the causal q0 anchors.  A hidden adjacent track is never
regressed as an unrelated absolute point: its q0 position is composed from the
current primary anchor and a directed, sample-specific edge.  The edge may be
supported by simultaneous pair observations or by asynchronous observations of
both endpoint handles.  No center, phase, fixed radius/height, or slot identity
is used.
"""

from __future__ import annotations

import torch
from torch import nn

from .cyclic_state_model import CyclicStateRestorer


class CyclicAnchorEdgeRestorer(CyclicStateRestorer):
    """Restore q0 with a visible anchor plus learned directed cyclic edges."""

    model_family = "cyclic-equivariant-anchor-edge-q0-restorer-v2"

    def __init__(
        self,
        position_mean: torch.Tensor,
        position_std: torch.Tensor,
        *,
        channels: int = 48,
        dropout: float = 0.05,
        history_events: int = 32,
    ) -> None:
        super().__init__(
            position_mean,
            position_std,
            channels=channels,
            dropout=dropout,
            history_events=history_events,
        )
        # For edge i -> i+1 use both asynchronous endpoint states, the
        # simultaneous-pair state when available, and a C4-invariant summary.
        edge_features = 5 * channels
        self.async_edge_head = nn.Sequential(
            nn.Linear(edge_features, 2 * channels),
            nn.SiLU(),
            nn.Linear(2 * channels, 3),
        )
        self.edge_sigma_head = nn.Sequential(
            nn.Linear(edge_features, channels),
            nn.SiLU(),
            nn.Linear(channels, 1),
        )
        # A loaded v1 foundation therefore starts as an exact anchor + existing
        # edge audit, before the asynchronous residual learns anything.
        nn.init.zeros_(self.async_edge_head[-1].weight)
        nn.init.zeros_(self.async_edge_head[-1].bias)
        nn.init.zeros_(self.edge_sigma_head[-1].weight)
        nn.init.constant_(self.edge_sigma_head[-1].bias, -2.0)

    @staticmethod
    def new_parameter_prefixes() -> tuple[str, ...]:
        return ("async_edge_head.", "edge_sigma_head.")

    def forward(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        primary_mask: torch.Tensor,
        event_mask: torch.Tensor,
        event_time_s: torch.Tensor,
        switch_step: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        result = super().forward(
            obs, obs_mask, primary_mask, event_mask, event_time_s, switch_step
        )
        state = result["track_context"]
        next_state = torch.roll(state, shifts=-1, dims=1)
        edge_state = result["edge_context"]
        pooled = torch.cat((state.mean(dim=1), state.amax(dim=1)), dim=-1)
        edge_context = torch.cat((
            state,
            next_state,
            edge_state,
            pooled[:, None].expand(-1, state.shape[1], -1),
        ), dim=-1)

        foundation_edge_normalized = result["edge0_m"] / self.position_std
        restored_edge_normalized = (
            foundation_edge_normalized + self.async_edge_head(edge_context)
        )
        # If both endpoints are observed at q0, preserve their exact measured
        # difference even while the new residual head is training.
        edge0_normalized = torch.where(
            result["current_pair_q0_observed"].unsqueeze(-1),
            foundation_edge_normalized,
            restored_edge_normalized,
        )
        edge0_m = self.position_std * edge0_normalized

        seen = result["seen"].to(torch.bool)
        edge0_supported = seen & torch.roll(seen, shifts=-1, dims=1)
        edge_age_s = torch.maximum(
            result["age_s"], torch.roll(result["age_s"], shifts=-1, dims=1)
        )
        edge_age_s = torch.where(
            edge0_supported,
            edge_age_s,
            torch.full_like(edge_age_s, float("inf")),
        )

        direct_q0_m = result["q0_m"]
        direct_q0_normalized = (
            direct_q0_m - self.position_mean
        ) / self.position_std
        batch = direct_q0_m.shape[0]
        row = torch.arange(batch, device=direct_q0_m.device)
        primary_index = result["primary_index"]
        anchor_normalized = direct_q0_normalized[row, primary_index]
        forward_edge = edge0_normalized[row, primary_index]
        forward_supported = edge0_supported[row, primary_index]

        warm_hidden = result["warm_hidden"].to(torch.bool)
        compose_clockwise = (
            warm_hidden
            & result["clockwise"].to(torch.bool)
            & forward_supported[:, None]
        )
        compose_counterclockwise = (
            warm_hidden
            & result["counterclockwise"].to(torch.bool)
            & edge0_supported
        )
        anchor_composed = compose_clockwise | compose_counterclockwise

        clockwise_candidate = anchor_normalized + forward_edge
        counterclockwise_candidate = (
            anchor_normalized[:, None] - edge0_normalized
        )
        q0_normalized = torch.where(
            compose_clockwise.unsqueeze(-1),
            clockwise_candidate[:, None].expand(-1, 4, -1),
            direct_q0_normalized,
        )
        q0_normalized = torch.where(
            compose_counterclockwise.unsqueeze(-1),
            counterclockwise_candidate,
            q0_normalized,
        )
        q0_m = self.position_mean + self.position_std * q0_normalized

        learned_edge_sigma_m = (
            torch.nn.functional.softplus(self.edge_sigma_head(edge_context))
            + 1e-4
        ) * self.position_std.mean()
        edge_sigma_m = torch.where(
            result["current_pair_q0_observed"].unsqueeze(-1),
            torch.full_like(learned_edge_sigma_m, 1e-6),
            learned_edge_sigma_m,
        )
        edge_sigma_m = torch.where(
            edge0_supported.unsqueeze(-1),
            edge_sigma_m,
            torch.ones_like(edge_sigma_m),
        )
        direct_sigma_m = result["q0_sigma_m"]
        anchor_sigma_m = direct_sigma_m[row, primary_index]
        forward_sigma_m = edge_sigma_m[row, primary_index]
        clockwise_sigma_m = torch.sqrt(
            anchor_sigma_m.square() + forward_sigma_m.square()
        )
        counterclockwise_sigma_m = torch.sqrt(
            anchor_sigma_m[:, None].square() + edge_sigma_m.square()
        )
        q0_sigma_m = torch.where(
            compose_clockwise.unsqueeze(-1),
            clockwise_sigma_m[:, None].expand(-1, 4, -1),
            direct_sigma_m,
        )
        q0_sigma_m = torch.where(
            compose_counterclockwise.unsqueeze(-1),
            counterclockwise_sigma_m,
            q0_sigma_m,
        )

        q0_valid = result["current_visible"].to(torch.bool) | anchor_composed
        q0_sigma_m = torch.where(
            q0_valid.unsqueeze(-1), q0_sigma_m, torch.ones_like(q0_sigma_m)
        )
        confidence = torch.where(
            q0_valid,
            torch.exp(-q0_sigma_m.squeeze(-1) / 0.1),
            torch.zeros_like(q0_sigma_m.squeeze(-1)),
        )
        result.update({
            "q0_m": q0_m,
            "q0_valid": q0_valid,
            "q0_sigma_m": q0_sigma_m,
            "confidence": confidence,
            "direct_q0_m": direct_q0_m,
            "anchor_composed": anchor_composed,
            "edge0_foundation_m": result["edge0_m"],
            "edge0_m": edge0_m,
            "edge0_valid": edge0_supported,
            "edge0_supported": edge0_supported,
            "edge0_async_supported": edge0_supported & ~result["pair_seen"],
            "edge0_sigma_m": edge_sigma_m,
            "edge_age_s": edge_age_s,
        })
        return result

    def config(self) -> dict[str, object]:
        config = super().config()
        config.update({
            "family": self.model_family,
            "hidden_position_parameterization": "current anchor plus directed edge",
            "asynchronous_endpoint_edge_support": True,
            "direct_hidden_absolute_head_is_deployment_output": False,
            "current_visible_anchor": "current primary q0 state",
        })
        return config
