"""Interpretable factor-aware objective for router-only fine-tuning."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .pnp_state_targets import truth_trajectory_targets


def _balanced_four_class_ce(
    logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor, *,
    label_smoothing: float,
) -> torch.Tensor:
    losses = []
    for route in range(4):
        group = valid & (labels == route)
        if bool(group.any()):
            losses.append(F.cross_entropy(
                logits[group], labels[group], label_smoothing=label_smoothing,
            ))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _balanced_binary_bce(
    logits: torch.Tensor, positive: torch.Tensor, valid: torch.Tensor, *,
    label_smoothing: float,
) -> torch.Tensor:
    losses = []
    for truth in (False, True):
        group = valid & (positive == truth)
        if bool(group.any()):
            target = positive[group].to(logits.dtype)
            if label_smoothing:
                target = target * (1.0 - label_smoothing) + 0.5 * label_smoothing
            losses.append(F.binary_cross_entropy_with_logits(
                logits[group], target,
            ))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def router_only_loss(
    router_logits: torch.Tensor, target: torch.Tensor,
    tau: torch.Tensor, rule_query: torch.Tensor, geometry: torch.Tensor, *,
    four_class_weight: float = 1.0, move_factor_weight: float = 1.0,
    rotate_factor_weight: float = 1.0, label_smoothing: float = 0.02,
    move_negative_mps: float = 0.01, move_positive_mps: float = 0.10,
    rotate_negative_rad_s: float = 0.05,
    rotate_positive_rad_s: float = 0.20,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise the four routes and their two physical binary factors.

    Future truth is detached and used only to construct the same accepted
    constant-motion labels as v13.  The predictor still receives observations,
    masks, real timestamps and query time only.
    """
    if router_logits.ndim != 2 or router_logits.shape[1] != 4:
        raise ValueError("router logits must have shape [B,4]")
    if target.ndim != 4 or target.shape[2:] != (4, 3):
        raise ValueError("target must have shape [B,Q,4,3]")
    if tau.shape != rule_query.shape or tau.shape[:2] != target.shape[:2]:
        raise ValueError("tau/rule_query must match target [B,Q]")
    if target.shape[1] < 4:
        raise ValueError("router supervision requires q0 through q3")
    weights = (four_class_weight, move_factor_weight, rotate_factor_weight)
    if min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("router loss weights must be nonnegative and nonzero")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label smoothing must be within [0,1)")
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
    speed = torch.linalg.vector_norm(truth["velocity"].detach(), dim=-1)
    abs_omega = truth["omega"].detach().abs()
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

    four_class = _balanced_four_class_ce(
        router_logits, labels, valid, label_smoothing=label_smoothing,
    )
    move_logit = torch.logsumexp(router_logits[:, (1, 3)], dim=-1) - (
        torch.logsumexp(router_logits[:, (0, 2)], dim=-1)
    )
    rotate_logit = torch.logsumexp(router_logits[:, (2, 3)], dim=-1) - (
        torch.logsumexp(router_logits[:, (0, 1)], dim=-1)
    )
    move_factor = _balanced_binary_bce(
        move_logit, move_positive, valid, label_smoothing=label_smoothing,
    )
    rotate_factor = _balanced_binary_bce(
        rotate_logit, rotate_positive, valid, label_smoothing=label_smoothing,
    )
    total = (
        four_class_weight * four_class
        + move_factor_weight * move_factor
        + rotate_factor_weight * rotate_factor
    )
    return total, {
        "four_class": four_class,
        "move_factor": move_factor,
        "rotate_factor": rotate_factor,
        "eligible_fraction": eligible.float().mean(),
        "valid_fraction": valid.float().mean(),
        "stationary_fraction": (valid & (labels == 0)).float().mean(),
        "translation_fraction": (valid & (labels == 1)).float().mean(),
        "rotation_fraction": (valid & (labels == 2)).float().mean(),
        "combined_fraction": (valid & (labels == 3)).float().mean(),
    }
