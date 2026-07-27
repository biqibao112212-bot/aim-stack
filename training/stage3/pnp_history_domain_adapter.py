"""Anonymous closed-window PnP history adapter at the frozen-F boundary."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class PnPHistoryDomainAdapter(nn.Module):
    """Map a noisy anonymous selected history into the clean-F history domain."""

    model_family = "anonymous-pnp-history-domain-adapter-v1"

    def __init__(
        self,
        *,
        channels: int = 64,
        dropout: float = 0.05,
        history_events: int = 32,
        position_scale_m: float = 1.0,
        history_scale_s: float = 0.32,
        switch_scale: float = 6.0,
    ) -> None:
        super().__init__()
        if channels < 16 or channels % 2:
            raise ValueError("history adapter channels must be even and >=16")
        if history_events < 2 or min(
            position_scale_m, history_scale_s, switch_scale
        ) <= 0:
            raise ValueError("invalid history adapter scales")
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.history_events = int(history_events)
        self.position_scale_m = float(position_scale_m)
        self.history_scale_s = float(history_scale_s)
        self.switch_scale = float(switch_scale)
        self.input_projection = nn.Sequential(
            nn.Linear(10, channels), nn.LayerNorm(channels), nn.SiLU()
        )
        self.window_smoother = nn.GRU(
            channels, channels // 2, num_layers=2, batch_first=True,
            dropout=dropout, bidirectional=True,
        )
        self.residual_head = nn.Sequential(
            nn.Linear(channels + 3, channels), nn.SiLU(),
            nn.Linear(channels, 3),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "family": self.model_family,
            "channels": self.channels,
            "dropout": self.dropout,
            "history_events": self.history_events,
            "position_scale_m": self.position_scale_m,
            "history_scale_s": self.history_scale_s,
            "switch_scale": self.switch_scale,
            "physical_id_input": False,
            "motion_class_input": False,
            "primary_mask_input": False,
            "candidate_input": False,
            "future_or_target_input": False,
            "history_semantics": "anonymous visibility-selected relative trajectory",
            "window_causality": "all valid timestamps are <= q0; no future field",
            "per_event_online_causality": False,
            "q0_contract": "bit-exact zero relative position",
        }

    def forward(
        self,
        history_position_rel_m: torch.Tensor,
        history_time_s: torch.Tensor,
        history_dt_s: torch.Tensor,
        history_switch_step: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if history_position_rel_m.ndim != 3 or history_position_rel_m.shape[-1] != 3:
            raise ValueError("history positions must have shape [B,T,3]")
        batch, events = history_position_rel_m.shape[:2]
        if events != self.history_events:
            raise ValueError("history adapter length differs from configuration")
        shape = (batch, events)
        if any(value.shape != shape for value in (
            history_time_s, history_dt_s, history_switch_step, history_mask,
        )):
            raise ValueError("history adapter scalar inputs must have shape [B,T]")
        mask = history_mask.to(torch.bool)
        finite_position = torch.isfinite(history_position_rel_m).all(dim=-1)
        finite_scalar = (
            torch.isfinite(history_time_s)
            & torch.isfinite(history_dt_s)
            & torch.isfinite(history_switch_step)
        )
        if bool(torch.any(mask & ~(finite_position & finite_scalar))):
            raise ValueError("valid history adapter inputs must be finite")
        if bool(torch.any(mask.sum(dim=1) < 2)):
            raise ValueError("history adapter requires two valid events")
        if bool(torch.any(mask & (history_time_s > 1e-6))):
            raise ValueError("history adapter forbids observations after q0")
        pair_valid = mask[:, :, None] & mask[:, None, :]
        ordered_pair = torch.triu(
            torch.ones(events, events, dtype=torch.bool, device=mask.device),
            diagonal=1,
        )
        nonincreasing = history_time_s[:, :, None] >= history_time_s[:, None, :]
        if bool(torch.any(pair_valid & ordered_pair & nonincreasing)):
            raise ValueError("history adapter times must be strictly increasing")
        q0 = mask & history_time_s.abs().le(1e-6)
        if bool(torch.any(q0.sum(dim=1) != 1)):
            raise ValueError("history adapter requires exactly one q0 event")

        position = torch.where(
            mask.unsqueeze(-1), history_position_rel_m,
            torch.zeros_like(history_position_rel_m),
        )
        time = torch.where(mask, history_time_s, torch.zeros_like(history_time_s))
        dt = torch.where(mask, history_dt_s, torch.zeros_like(history_dt_s))
        switch = torch.where(
            mask, history_switch_step, torch.zeros_like(history_switch_step)
        ).to(position.dtype)
        normalized = position / self.position_scale_m
        features = torch.cat((
            normalized,
            normalized.square(),
            (time / self.history_scale_s).unsqueeze(-1),
            (dt / self.history_scale_s).unsqueeze(-1),
            (switch / self.switch_scale).unsqueeze(-1),
            mask.to(position.dtype).unsqueeze(-1),
        ), dim=-1)
        projected = self.input_projection(features)
        projected = torch.where(
            mask.unsqueeze(-1), projected, torch.zeros_like(projected)
        )
        smoothed, _ = self.window_smoother(projected)
        residual = self.residual_head(torch.cat((smoothed, normalized), dim=-1))
        residual_m = residual * self.position_scale_m
        residual_m = torch.where(
            mask.unsqueeze(-1) & ~q0.unsqueeze(-1), residual_m,
            torch.zeros_like(residual_m),
        )
        corrected = torch.where(
            mask.unsqueeze(-1), position + residual_m,
            torch.zeros_like(position),
        )
        corrected = torch.where(
            q0.unsqueeze(-1), torch.zeros_like(corrected), corrected
        )
        return {
            "corrected_history_position_rel_m": corrected,
            "residual_m": residual_m,
            "history_mask": mask,
            "event_context": smoothed,
        }
