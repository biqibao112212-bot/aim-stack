"""Small, interpretable q0-only objective for the current-pose adapter."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .pnp_state_targets import _query_pose_from_fixed_truth


def current_pose_loss(
    output: dict[str, torch.Tensor],
    target_q0: torch.Tensor,
    geometry: torch.Tensor,
    *,
    huber_beta_m: float = 0.02,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if target_q0.ndim != 3 or target_q0.shape[1:] != (4, 3):
        raise ValueError("target_q0 must have shape [B,4,3]")
    if huber_beta_m <= 0:
        raise ValueError("huber_beta_m must be positive")
    if not torch.isfinite(target_q0).all():
        raise ValueError("current pose loss refuses non-finite truth")
    target_center, target_phase = _query_pose_from_fixed_truth(
        target_q0[:, None], geometry
    )
    target_center = target_center[:, 0]
    target_phase = target_phase[:, 0]
    center = F.smooth_l1_loss(
        output["center"].float(), target_center,
        beta=huber_beta_m, reduction="mean",
    )
    radius = torch.sqrt(geometry[:, :2].float().square().sum(dim=-1).mean())
    phase = F.smooth_l1_loss(
        radius * output["phase"].float(), radius * target_phase,
        beta=huber_beta_m, reduction="mean",
    )
    total = center + phase
    return total, {"center": center, "phase": phase}
