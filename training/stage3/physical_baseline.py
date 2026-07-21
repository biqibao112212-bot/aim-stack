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


class RigidPoseLeastSquaresRollout(nn.Module):
    """Causal input-sufficiency oracle fitted from fixed-slot pose history.

    This is deliberately parameter-free and is not the learned A/B candidate.
    It tests whether the admitted history itself contains enough information to
    recover constant translation and yaw motion without future labels.
    """

    model_family = "fixed-slot-rigid-pose-least-squares-v1"

    def __init__(
        self, geometry: torch.Tensor, position_mean: torch.Tensor,
        position_std: torch.Tensor, fit_history_s: float = 1.0,
        fit_events: int = 4,
    ) -> None:
        super().__init__()
        if tuple(geometry.shape) != (4, 3):
            raise ValueError("geometry must have shape [4,3]")
        if tuple(position_mean.shape) != (3,) or tuple(position_std.shape) != (3,):
            raise ValueError("position mean/std must have shape [3]")
        if torch.any(position_std <= 0) or fit_history_s <= 0 or fit_events < 2:
            raise ValueError("normalization and fit history must be positive")
        centered_xy = geometry[:, :2] - geometry[:, :2].mean(dim=0)
        if torch.sum(centered_xy.square()) <= 1e-8:
            raise ValueError("geometry is degenerate for planar rigid-pose fitting")
        self.fit_history_s = float(fit_history_s)
        self.fit_events = int(fit_events)
        self.register_buffer("geometry", geometry.to(torch.float32).clone())
        self.register_buffer("position_mean", position_mean.to(torch.float32).clone())
        self.register_buffer("position_std", position_std.to(torch.float32).clone())

    @staticmethod
    def _line_fit(time_s: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_time = time_s.mean()
        mean_value = value.mean(dim=0)
        centered_time = time_s - mean_time
        denominator = centered_time.square().sum()
        if denominator <= 1e-12:
            raise ValueError("least-squares pose history has no time baseline")
        slope = (centered_time.unsqueeze(-1) * (value - mean_value)).sum(dim=0) / denominator
        at_zero = mean_value - slope * mean_time
        return at_zero, slope

    def forward(
        self, obs: torch.Tensor, obs_mask: torch.Tensor,
        event_mask: torch.Tensor, event_time_s: torch.Tensor,
        tau: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        full_event = event_mask.to(torch.bool) & obs_mask.to(torch.bool).all(dim=2)
        if not bool((full_event.sum(dim=1) >= 2).all()):
            raise ValueError("least-squares oracle needs two complete fixed-slot events")
        if tau.ndim == 1:
            tau = tau.unsqueeze(0).expand(obs.shape[0], -1)
        with torch.autocast(device_type=obs.device.type, enabled=False):
            geometry = self.geometry.float()
            clean = torch.where(
                full_event[:, :, None, None], obs[..., :3].float(),
                torch.zeros_like(obs[..., :3].float()),
            )
            position = clean * self.position_std.float() + self.position_mean.float()
            observed_mean = position.mean(dim=2)
            observed_xy_centered = position[..., :2] - observed_mean[:, :, None, :2]
            geometry_xy_centered = geometry[:, :2] - geometry[:, :2].mean(dim=0)
            dot = (
                observed_xy_centered[..., 0] * geometry_xy_centered[None, None, :, 0]
                + observed_xy_centered[..., 1] * geometry_xy_centered[None, None, :, 1]
            ).sum(dim=2)
            cross = (
                geometry_xy_centered[None, None, :, 0] * observed_xy_centered[..., 1]
                - geometry_xy_centered[None, None, :, 1] * observed_xy_centered[..., 0]
            ).sum(dim=2)
            raw_yaw = torch.atan2(cross, dot)
            raw_yaw = torch.where(full_event, raw_yaw, torch.zeros_like(raw_yaw))
            increment = torch.zeros_like(raw_yaw)
            increment[:, 0] = torch.where(full_event[:, 0], raw_yaw[:, 0], 0.0)
            raw_difference = raw_yaw[:, 1:] - raw_yaw[:, :-1]
            wrapped_difference = torch.atan2(torch.sin(raw_difference), torch.cos(raw_difference))
            increment[:, 1:] = torch.where(
                full_event[:, 1:], wrapped_difference, torch.zeros_like(wrapped_difference)
            )
            yaw = torch.cumsum(increment, dim=1)
            phase = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
            gx, gy = geometry[:, 0], geometry[:, 1]
            rotated = torch.stack((
                phase[..., 0, None] * gx - phase[..., 1, None] * gy,
                phase[..., 1, None] * gx + phase[..., 0, None] * gy,
                geometry[:, 2].view(1, 1, 4).expand(obs.shape[0], obs.shape[1], -1),
            ), dim=-1)
            center = (position - rotated).mean(dim=2)

            reverse_rank = torch.flip(
                torch.cumsum(torch.flip(full_event.to(torch.int64), dims=(1,)), dim=1),
                dims=(1,),
            )
            fit_mask = (
                full_event & (reverse_rank <= self.fit_events)
                & (event_time_s >= -self.fit_history_s)
            )
            fallback = full_event & (reverse_rank <= 2)
            fit_mask = torch.where(
                (fit_mask.sum(dim=1) >= 2).unsqueeze(1), fit_mask, fallback
            )
            weight = fit_mask.float()
            count = weight.sum(dim=1)
            mean_time = (weight * event_time_s).sum(dim=1) / count
            mean_center = (weight.unsqueeze(-1) * center).sum(dim=1) / count.unsqueeze(-1)
            mean_yaw = (weight * yaw).sum(dim=1) / count
            centered_time = event_time_s - mean_time.unsqueeze(1)
            denominator = (weight * centered_time.square()).sum(dim=1)
            if not bool((denominator > 1e-12).all()):
                raise ValueError("least-squares pose history has no time baseline")
            velocity = (
                weight.unsqueeze(-1) * centered_time.unsqueeze(-1)
                * (center - mean_center.unsqueeze(1))
            ).sum(dim=1) / denominator.unsqueeze(-1)
            omega = (
                weight * centered_time * (yaw - mean_yaw.unsqueeze(1))
            ).sum(dim=1) / denominator
            center0 = mean_center - velocity * mean_time.unsqueeze(-1)
            yaw0 = mean_yaw - omega * mean_time
            query_center = center0[:, None, :] + tau.float().unsqueeze(-1) * velocity[:, None, :]
            query_yaw = yaw0[:, None] + tau.float() * omega[:, None]
            cosine, sine = torch.cos(query_yaw), torch.sin(query_yaw)
            relative = torch.stack((
                cosine[:, :, None] * gx - sine[:, :, None] * gy,
                sine[:, :, None] * gx + cosine[:, :, None] * gy,
                geometry[:, 2].view(1, 1, 4).expand(obs.shape[0], tau.shape[1], -1),
            ), dim=-1)
        return {
            "position_mean": query_center[:, :, None, :] + relative,
            "center": query_center,
            "angle": query_yaw,
            "center0": center0,
            "velocity": velocity,
            "yaw0": yaw0,
            "omega": omega,
        }

    def config(self) -> dict[str, object]:
        return {
            "family": self.model_family,
            "fit_history_s": self.fit_history_s,
            "fit_events": self.fit_events,
            "geometry": self.geometry.detach().cpu().tolist(),
            "position_mean": self.position_mean.detach().cpu().tolist(),
            "position_std": self.position_std.detach().cpu().tolist(),
        }
