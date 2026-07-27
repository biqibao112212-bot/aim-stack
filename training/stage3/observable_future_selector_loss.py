"""Selector-only objectives for a frozen observable-target F trajectory."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .observable_future_loss import _balanced_group_mean, _target_candidate_row


def observable_future_selector_loss(
    prediction: dict[str, torch.Tensor],
    candidate_step: torch.Tensor,
    candidate_mask: torch.Tensor,
    tau_s: torch.Tensor,
    target_switch_count: torch.Tensor,
    target_visible_delta_m: torch.Tensor,
    target_query_mask: torch.Tensor,
    *,
    switch_weight: float = 1.0,
    macro_balance_weight: float = 0.5,
    switch_focal_gamma: float = 2.0,
    distance_cost_weight: float = 1.0,
    distance_cost_scale_m: float = 0.3,
    distance_cost_cap: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train only anonymous branch selection.

    The distance term uses detached conditional trajectories.  It can shape
    the switch logits according to the physical consequence of a wrong branch,
    but it can never update the already accepted trajectory predictor.
    """
    if (
        min(switch_weight, distance_cost_weight, switch_focal_gamma) < 0
        or switch_weight + distance_cost_weight <= 0
        or not 0.0 <= macro_balance_weight <= 1.0
        or min(distance_cost_scale_m, distance_cost_cap) <= 0
    ):
        raise ValueError("observable F selector loss scales are invalid")
    logits = prediction.get("switch_logits")
    conditional_delta = prediction.get("conditional_delta_m")
    if logits is None or logits.ndim != 3:
        raise ValueError("prediction switch_logits must have shape [B,Q,K]")
    if conditional_delta is None or conditional_delta.shape != logits.shape + (3,):
        raise ValueError("conditional_delta_m must have shape [B,Q,K,3]")
    if target_visible_delta_m.shape != logits.shape[:2] + (3,):
        raise ValueError("target_visible_delta_m must have shape [B,Q,3]")
    if candidate_step.shape != (logits.shape[0], logits.shape[2]):
        raise ValueError("candidate_step does not match prediction candidates")
    if candidate_mask.shape != candidate_step.shape:
        raise ValueError("candidate_mask does not match candidate_step")
    if tau_s.ndim == 1:
        tau = tau_s.unsqueeze(0).expand(logits.shape[0], -1)
    else:
        tau = tau_s
    if tau.shape != logits.shape[:2] or not bool(torch.isfinite(tau).all()):
        raise ValueError("tau_s must match prediction queries and be finite")

    true_row, query_mask = _target_candidate_row(
        candidate_step, candidate_mask, target_switch_count, target_query_mask
    )
    optimization_mask = query_mask & (tau > 0)
    if not bool(optimization_mask.any()):
        raise ValueError("observable F selector batch has no learned query")
    valid_candidate = candidate_mask.to(torch.bool)
    masked_logits = logits.float().masked_fill(
        ~valid_candidate[:, None, :], -torch.inf
    )
    if not bool(torch.isfinite(masked_logits[optimization_mask]).any(dim=-1).all()):
        raise ValueError("eligible selector query has no finite candidate logit")

    switch_element = F.cross_entropy(
        masked_logits.reshape(-1, masked_logits.shape[-1]),
        true_row.reshape(-1), reduction="none",
    ).reshape_as(true_row)
    if switch_focal_gamma > 0:
        correct_probability = torch.exp(-switch_element)
        switch_element = (
            (1.0 - correct_probability).pow(switch_focal_gamma)
            * switch_element
        )
    switch, switch_no, switch_yes = _balanced_group_mean(
        switch_element, optimization_mask, target_switch_count,
        macro_balance_weight,
    )

    # Only logits receive gradient.  Relative-to-true excess cost keeps the
    # semantic target authoritative even if a wrong frozen branch happens to
    # land closer to the target because of trajectory regression error.
    detached_delta = conditional_delta.detach()
    candidate_error_m = torch.linalg.vector_norm(
        detached_delta - target_visible_delta_m[:, :, None, :], dim=-1
    )
    true_error_m = candidate_error_m.gather(2, true_row.unsqueeze(-1)).squeeze(-1)
    excess_cost = (candidate_error_m - true_error_m.unsqueeze(-1)).clamp_min(0.0)
    normalized_cost = (excess_cost / distance_cost_scale_m).clamp_max(
        distance_cost_cap
    )
    normalized_cost = torch.where(
        valid_candidate[:, None, :], normalized_cost,
        torch.zeros_like(normalized_cost),
    ).detach()
    probability = torch.softmax(masked_logits, dim=-1)
    distance_element = (probability * normalized_cost).sum(dim=-1)
    distance_cost, distance_no, distance_yes = _balanced_group_mean(
        distance_element, optimization_mask, target_switch_count,
        macro_balance_weight,
    )
    objective = switch_weight * switch + distance_cost_weight * distance_cost
    return objective, {
        "objective": objective,
        "switch": switch,
        "switch_no_change": switch_no,
        "switch_changed": switch_yes,
        "distance_cost": distance_cost,
        "distance_cost_no_change": distance_no,
        "distance_cost_changed": distance_yes,
        "mean_true_branch_error_m": true_error_m[optimization_mask].mean().detach(),
    }
