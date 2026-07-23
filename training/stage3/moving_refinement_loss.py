"""Moving-only binary objective for translation versus combined motion."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .pnp_state_targets import truth_trajectory_targets


def moving_refinement_loss(
    logit: torch.Tensor, target: torch.Tensor, tau: torch.Tensor,
    rule_query: torch.Tensor, geometry: torch.Tensor, *,
    label_smoothing: float = 0.01,
    move_positive_mps: float = 0.10,
    rotate_negative_rad_s: float = 0.05,
    rotate_positive_rad_s: float = 0.20,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Classify moving samples as translation (0) or combined (1)."""
    if logit.ndim != 1 or logit.shape[0] != target.shape[0]:
        raise ValueError("refinement logit must have shape [B]")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label smoothing must be within [0,1)")
    truth = truth_trajectory_targets(
        target[:, :4], tau[:, :4], geometry, rule_queries=4,
    )
    eligible = (
        rule_query[:, :4].to(torch.bool).all(dim=1)
        & truth["constant_motion"]
    )
    speed = torch.linalg.vector_norm(truth["velocity"].detach(), dim=-1)
    abs_omega = truth["omega"].detach().abs()
    moving = speed >= move_positive_mps
    translation = eligible & moving & (abs_omega <= rotate_negative_rad_s)
    combined = eligible & moving & (abs_omega >= rotate_positive_rad_s)
    valid = translation | combined
    losses = []
    for group, value in ((translation, 0.0), (combined, 1.0)):
        if bool(group.any()):
            target_value = torch.full_like(logit[group], value)
            if label_smoothing:
                target_value = (
                    target_value * (1.0 - label_smoothing)
                    + 0.5 * label_smoothing
                )
            losses.append(F.binary_cross_entropy_with_logits(
                logit[group], target_value,
            ))
    total = torch.stack(losses).mean() if losses else logit.sum() * 0.0
    probability = torch.sigmoid(logit)
    return total, {
        "binary_ce": total,
        "eligible_fraction": eligible.float().mean(),
        "valid_fraction": valid.float().mean(),
        "translation_fraction": translation.float().mean(),
        "combined_fraction": combined.float().mean(),
        "translation_probability_mean": (
            probability[translation].mean() if bool(translation.any())
            else probability.sum() * 0.0
        ),
        "combined_probability_mean": (
            probability[combined].mean() if bool(combined.any())
            else probability.sum() * 0.0
        ),
    }
