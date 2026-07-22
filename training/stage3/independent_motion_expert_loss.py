"""Training-only objectives for independent rigid-motion specialists."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .pnp_state_targets import truth_trajectory_targets


def _masked_zero_huber(
    value: torch.Tensor, mask: torch.Tensor, beta: float,
) -> torch.Tensor:
    if bool(mask.any()):
        return F.smooth_l1_loss(
            value[mask], torch.zeros_like(value[mask]),
            beta=beta, reduction="mean",
        )
    return value.sum() * 0.0


def _balanced_router_loss(
    logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor,
) -> torch.Tensor:
    group_losses = []
    for route in range(4):
        group = valid & (labels == route)
        if bool(group.any()):
            group_losses.append(F.cross_entropy(logits[group], labels[group]))
    if not group_losses:
        return logits.sum() * 0.0
    return torch.stack(group_losses).mean()


def independent_motion_expert_loss(
    prediction: dict[str, torch.Tensor], target: torch.Tensor,
    tau: torch.Tensor, rule_query: torch.Tensor, geometry: torch.Tensor, *,
    huber_beta_m: float = 0.005, reference_horizon_s: float = 0.5,
    rotation_weight: float = 1.0, combined_velocity_weight: float = 1.0,
    combined_rotation_weight: float = 1.0, router_weight: float = 0.1,
    move_negative_mps: float = 0.01, move_positive_mps: float = 0.10,
    rotate_negative_rad_s: float = 0.05,
    rotate_positive_rad_s: float = 0.20,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train pure-rotation, joint-combined, and four-class routing branches.

    Future truth is used only to create detached supervision labels.  The pose
    and translation foundations have no objective here and remain frozen.
    """
    required = {
        "rotation_omega", "combined_velocity", "combined_omega",
        "router_logit",
    }
    missing = required - prediction.keys()
    if missing:
        raise ValueError(
            f"independent expert output is missing: {sorted(missing)}"
        )
    if target.ndim != 4 or target.shape[2:] != (4, 3):
        raise ValueError("target must have shape [B,Q,4,3]")
    if tau.shape != rule_query.shape or tau.shape[:2] != target.shape[:2]:
        raise ValueError("tau/rule_query must match target [B,Q]")
    if target.shape[1] < 4:
        raise ValueError("independent expert supervision requires q0 through q3")
    positive_scales = (
        huber_beta_m, reference_horizon_s, rotation_weight,
        combined_velocity_weight, combined_rotation_weight, router_weight,
    )
    if min(positive_scales) <= 0:
        raise ValueError("independent expert loss scales must be positive")
    if not 0 <= move_negative_mps < move_positive_mps:
        raise ValueError("move router thresholds are invalid")
    if not 0 <= rotate_negative_rad_s < rotate_positive_rad_s:
        raise ValueError("rotation router thresholds are invalid")

    truth = truth_trajectory_targets(
        target[:, :4], tau[:, :4], geometry, rule_queries=4,
    )
    eligible = (
        rule_query[:, :4].to(torch.bool).all(dim=1)
        & truth["constant_motion"]
    )
    truth_velocity = truth["velocity"].detach()
    truth_omega = truth["omega"].detach()
    speed = torch.linalg.vector_norm(truth_velocity, dim=-1)
    abs_omega = truth_omega.abs()
    move_positive = speed >= move_positive_mps
    move_negative = speed <= move_negative_mps
    rotate_positive = abs_omega >= rotate_positive_rad_s
    rotate_negative = abs_omega <= rotate_negative_rad_s
    valid = eligible & (move_positive | move_negative) & (
        rotate_positive | rotate_negative
    )
    labels = (
        move_positive.to(torch.long)
        + 2 * rotate_positive.to(torch.long)
    ).detach()
    pure_rotation = valid & (labels == 2)
    combined = valid & (labels == 3)
    radius_m = torch.sqrt(
        geometry[:, :2].float().square().sum(dim=-1).mean()
    )

    rotation = _masked_zero_huber(
        radius_m * reference_horizon_s
        * (prediction["rotation_omega"] - truth_omega),
        pure_rotation, huber_beta_m,
    )
    combined_velocity = _masked_zero_huber(
        reference_horizon_s
        * (prediction["combined_velocity"] - truth_velocity),
        combined, huber_beta_m,
    )
    combined_rotation = _masked_zero_huber(
        radius_m * reference_horizon_s
        * (prediction["combined_omega"] - truth_omega),
        combined, huber_beta_m,
    )
    router = _balanced_router_loss(
        prediction["router_logit"], labels, valid,
    )
    total = (
        rotation_weight * rotation
        + combined_velocity_weight * combined_velocity
        + combined_rotation_weight * combined_rotation
        + router_weight * router
    )
    return total, {
        "rotation_expert": rotation,
        "combined_velocity": combined_velocity,
        "combined_rotation": combined_rotation,
        "router": router,
        "eligible_fraction": eligible.float().mean(),
        "router_valid_fraction": valid.float().mean(),
        "pure_rotation_fraction": pure_rotation.float().mean(),
        "combined_fraction": combined.float().mean(),
    }
