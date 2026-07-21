"""Deterministic physical-only objective for Stage-3 predictors."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .physical_metrics import q0_permutation


def physical_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    tau: torch.Tensor,
    *,
    query_mask: torch.Tensor | None = None,
    huber_beta_m: float = 0.05,
    state_weight: float = 2.0,
    motion_weight: float = 1.0,
    absolute_weight: float = 1.0,
    rigidity_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    if huber_beta_m <= 0:
        raise ValueError("huber_beta_m must be positive")
    if not torch.allclose(tau[:, 0], torch.zeros_like(tau[:, 0]), atol=0.0, rtol=0.0):
        raise ValueError("physical training requires exact tau=0 in query zero")
    aligned, best_index, _ = q0_permutation(prediction, target)
    if query_mask is None:
        query_mask = torch.ones_like(tau, dtype=torch.bool)
    if query_mask.shape != tau.shape:
        raise ValueError("query_mask must have the same [B,Q] shape as tau")
    query_mask = query_mask.to(dtype=torch.bool)
    if not torch.all(query_mask[:, 0]):
        raise ValueError("query zero must always be active")

    def masked_huber(
        value: torch.Tensor, reference: torch.Tensor, active: torch.Tensor
    ) -> torch.Tensor:
        element = F.smooth_l1_loss(
            value, reference, beta=huber_beta_m, reduction="none"
        )
        mask = active
        while mask.ndim < element.ndim:
            mask = mask.unsqueeze(-1)
        expanded = mask.expand_as(element).to(element.dtype)
        return (element * expanded).sum() / expanded.sum().clamp_min(1.0)

    state = F.smooth_l1_loss(
        aligned[:, 0], target[:, 0], beta=huber_beta_m, reduction="mean"
    )
    predicted_delta = aligned[:, 1:] - aligned[:, :1]
    target_delta = target[:, 1:] - target[:, :1]
    motion = masked_huber(predicted_delta, target_delta, query_mask[:, 1:])
    absolute = masked_huber(aligned, target, query_mask)
    pair_i, pair_j = torch.triu_indices(4, 4, offset=1, device=prediction.device)
    predicted_pair = torch.linalg.vector_norm(
        aligned[:, :, pair_i] - aligned[:, :, pair_j], dim=-1
    )
    target_pair = torch.linalg.vector_norm(
        target[:, :, pair_i] - target[:, :, pair_j], dim=-1
    )
    rigidity = masked_huber(predicted_pair, target_pair, query_mask)
    total = (
        state_weight * state + motion_weight * motion
        + absolute_weight * absolute + rigidity_weight * rigidity
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "state_huber": float(state.detach().cpu()),
        "motion_huber": float(motion.detach().cpu()),
        "absolute_huber": float(absolute.detach().cpu()),
        "rigidity_huber": float(rigidity.detach().cpu()),
        "mean_permutation": float(best_index.float().mean().detach().cpu()),
        "active_query_fraction": float(query_mask.float().mean().detach().cpu()),
    }
