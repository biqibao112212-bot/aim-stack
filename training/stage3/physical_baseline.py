"""Deterministic rigid motion operator used as the physical-core oracle."""

from __future__ import annotations

import torch
from torch import nn


class RigidTwoFrameRollout(nn.Module):
    """Infer the latest constant twist from two exact position sets.

    The input positions may be normalized; mean/std buffers restore metres.
    Output remains the common ``[B,Q,4,3]`` position API.  This module has no
    trainable parameters and therefore cannot hide physical error in fitting.
    """

    model_family = "rigid-two-frame-analytic-v1"

    def __init__(self, position_mean: torch.Tensor, position_std: torch.Tensor) -> None:
        super().__init__()
        if tuple(position_mean.shape) != (3,) or tuple(position_std.shape) != (3,):
            raise ValueError("position mean/std must have shape [3]")
        if torch.any(position_std <= 0):
            raise ValueError("position std must be positive")
        self.register_buffer("position_mean", position_mean.to(torch.float32))
        self.register_buffer("position_std", position_std.to(torch.float32))

    def forward(
        self, obs: torch.Tensor, obs_mask: torch.Tensor,
        event_mask: torch.Tensor, event_time_s: torch.Tensor,
        tau: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if obs.shape[-1] != 3:
            raise ValueError("analytic physical core requires xyz truth history")
        full_event = event_mask & obs_mask.all(dim=2)
        valid_count = full_event.sum(dim=1)
        if torch.any(valid_count < 2):
            raise ValueError("analytic physical core requires two complete truth events")
        batch, time = event_mask.shape
        index = torch.arange(time, device=obs.device).view(1, -1).expand(batch, -1)
        invalid = torch.full_like(index, -1)
        last_index = torch.where(full_event, index, invalid).amax(dim=1)
        before_last = full_event & (index < last_index[:, None])
        previous_index = torch.where(before_last, index, invalid).amax(dim=1)

        def gather_time(value: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
            shape = [batch, 1] + list(value.shape[2:])
            gather_index = selected.view(batch, 1, *([1] * (value.ndim - 2))).expand(*shape)
            return value.gather(1, gather_index).squeeze(1)

        position_m = obs * self.position_std + self.position_mean
        previous = gather_time(position_m, previous_index)
        latest = gather_time(position_m, last_index)
        previous_time = event_time_s.gather(1, previous_index[:, None]).squeeze(1)
        latest_time = event_time_s.gather(1, last_index[:, None]).squeeze(1)
        dt = latest_time - previous_time
        if torch.any(dt <= 0):
            raise ValueError("truth event times must be strictly increasing")

        previous_center = previous.mean(dim=1)
        latest_center = latest.mean(dim=1)
        velocity = (latest_center - previous_center) / dt[:, None]
        previous_relative = previous - previous_center[:, None]
        latest_relative = latest - latest_center[:, None]
        dot = (
            previous_relative[..., 0] * latest_relative[..., 0]
            + previous_relative[..., 1] * latest_relative[..., 1]
        ).sum(dim=1)
        cross = (
            previous_relative[..., 0] * latest_relative[..., 1]
            - previous_relative[..., 1] * latest_relative[..., 0]
        ).sum(dim=1)
        omega = torch.atan2(cross, dot) / dt
        if tau.ndim == 1:
            tau = tau.unsqueeze(0).expand(batch, -1)
        horizon = tau - latest_time[:, None]
        center = latest_center[:, None] + velocity[:, None] * horizon[:, :, None]
        angle = omega[:, None] * horizon
        cosine, sine = torch.cos(angle), torch.sin(angle)
        x = (
            cosine[:, :, None] * latest_relative[:, None, :, 0]
            - sine[:, :, None] * latest_relative[:, None, :, 1]
        )
        y = (
            sine[:, :, None] * latest_relative[:, None, :, 0]
            + cosine[:, :, None] * latest_relative[:, None, :, 1]
        )
        z = latest_relative[:, None, :, 2].expand(-1, tau.shape[1], -1)
        position = center[:, :, None] + torch.stack((x, y, z), dim=-1)
        return {
            "position_mean": position,
            "velocity": velocity,
            "omega": omega,
            "latest_time_s": latest_time,
        }

    def config(self) -> dict[str, object]:
        return {
            "family": self.model_family,
            "position_mean": self.position_mean.detach().cpu().tolist(),
            "position_std": self.position_std.detach().cpu().tolist(),
        }


class ExactStateRigidRollout(nn.Module):
    """Apply the exact constant-twist equation to the q0 four-plate truth."""

    model_family = "rigid-exact-state-analytic-v1"

    def forward(
        self, anchor_position: torch.Tensor, center0: torch.Tensor,
        velocity: torch.Tensor, yaw_rate: torch.Tensor, tau: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if anchor_position.ndim != 3 or anchor_position.shape[1:] != (4, 3):
            raise ValueError("anchor_position must have shape [B,4,3]")
        if velocity.shape != (anchor_position.shape[0], 3):
            raise ValueError("velocity must have shape [B,3]")
        if center0.shape != (anchor_position.shape[0], 3):
            raise ValueError("center0 must have shape [B,3]")
        if yaw_rate.shape != (anchor_position.shape[0],):
            raise ValueError("yaw_rate must have shape [B]")
        if tau.ndim == 1:
            tau = tau.unsqueeze(0).expand(anchor_position.shape[0], -1)
        relative0 = anchor_position - center0[:, None]
        center = center0[:, None] + velocity[:, None] * tau[:, :, None]
        angle = yaw_rate[:, None] * tau
        cosine, sine = torch.cos(angle), torch.sin(angle)
        x = (
            cosine[:, :, None] * relative0[:, None, :, 0]
            - sine[:, :, None] * relative0[:, None, :, 1]
        )
        y = (
            sine[:, :, None] * relative0[:, None, :, 0]
            + cosine[:, :, None] * relative0[:, None, :, 1]
        )
        z = relative0[:, None, :, 2].expand(-1, tau.shape[1], -1)
        return {
            "position_mean": center[:, :, None] + torch.stack((x, y, z), dim=-1)
        }

    def config(self) -> dict[str, str]:
        return {"family": self.model_family}
