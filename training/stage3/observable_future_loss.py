"""Branch-conditional objectives and metrics for anonymous observable F."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _target_candidate_row(
    candidate_step: torch.Tensor,
    candidate_mask: torch.Tensor,
    target_switch_count: torch.Tensor,
    target_query_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidate_step.ndim != 2 or candidate_mask.shape != candidate_step.shape:
        raise ValueError("candidate_step/mask must have shape [B,K]")
    if target_switch_count.ndim != 2 or target_query_mask.shape != target_switch_count.shape:
        raise ValueError("target switch/mask must have shape [B,Q]")
    if target_switch_count.shape[0] != candidate_step.shape[0]:
        raise ValueError("candidate and query batch sizes disagree")
    query_mask = target_query_mask.to(torch.bool)
    matches = (
        candidate_mask.to(torch.bool)[:, None, :]
        & (candidate_step[:, None, :] == target_switch_count[:, :, None])
    )
    coverage = matches.sum(dim=-1)
    if bool(torch.any(query_mask & (coverage != 1))):
        raise ValueError("eligible target branch is missing or duplicated")
    return matches.to(torch.long).argmax(dim=-1), query_mask


def _balanced_group_mean(
    value: torch.Tensor,
    query_mask: torch.Tensor,
    target_switch_count: torch.Tensor,
    macro_balance_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups: list[torch.Tensor] = []
    zero = value.sum() * 0.0
    no_switch_mask = query_mask & (target_switch_count == 0)
    switched_mask = query_mask & (target_switch_count != 0)
    no_switch = value[no_switch_mask].mean() if bool(no_switch_mask.any()) else zero
    switched = value[switched_mask].mean() if bool(switched_mask.any()) else zero
    if bool(no_switch_mask.any()):
        groups.append(no_switch)
    if bool(switched_mask.any()):
        groups.append(switched)
    if not groups:
        raise ValueError("observable F batch has no eligible query")
    # Blend the ordinary query mean with a signed-step macro mean.  A pure
    # macro loss can over-amplify a handful of switched queries and sacrifice
    # the common continuous branch; a pure micro loss erases rare multi-turn
    # branches.  The blend keeps both objectives explicit.
    step_groups: list[torch.Tensor] = []
    for step in torch.unique(target_switch_count[query_mask]):
        role = query_mask & (target_switch_count == step)
        step_groups.append(value[role].mean())
    macro = torch.stack(step_groups).mean()
    micro = value[query_mask].mean()
    balanced = macro_balance_weight * macro + (1.0 - macro_balance_weight) * micro
    return balanced, no_switch, switched


def observable_future_loss(
    prediction: dict[str, torch.Tensor],
    candidate_step: torch.Tensor,
    candidate_mask: torch.Tensor,
    tau_s: torch.Tensor,
    target_switch_count: torch.Tensor,
    target_visible_delta_m: torch.Tensor,
    target_query_mask: torch.Tensor,
    *,
    huber_beta_m: float = 0.01,
    switch_weight: float = 1.0,
    position_weight: float = 10.0,
    position_mse_weight: float = 0.0,
    rate_weight: float = 0.0,
    rate_huber_beta_mps: float = 0.02,
    rate_tau_floor_s: float = 0.05,
    position_tail_weight: float = 0.0,
    position_tail_fraction: float = 0.2,
    trend_weight: float = 0.0,
    macro_balance_weight: float = 0.5,
    position_macro_balance_weight: float | None = None,
    switch_focal_gamma: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train classification plus the position of the true branch only."""
    position_balance = (
        macro_balance_weight
        if position_macro_balance_weight is None
        else position_macro_balance_weight
    )
    if (
        min(huber_beta_m, rate_huber_beta_mps, rate_tau_floor_s) <= 0
        or min(switch_weight, position_weight, position_mse_weight,
               rate_weight, position_tail_weight, trend_weight) < 0
        or not 0.0 <= macro_balance_weight <= 1.0
        or not 0.0 <= position_balance <= 1.0
        or switch_focal_gamma < 0
        or not 0.0 < position_tail_fraction <= 1.0
    ):
        raise ValueError("observable F loss scales are invalid")
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
        raise ValueError("observable F batch has no learned nonzero-time query")
    flat_logits = logits.float().reshape(-1, logits.shape[-1])
    flat_row = true_row.reshape(-1)
    switch_element = F.cross_entropy(
        flat_logits, flat_row, reduction="none"
    ).reshape_as(true_row)
    if switch_focal_gamma > 0:
        correct_probability = torch.exp(-switch_element)
        switch_element = (
            (1.0 - correct_probability).pow(switch_focal_gamma)
            * switch_element
        )
    switch, switch_no, switch_yes = _balanced_group_mean(
        switch_element, optimization_mask, target_switch_count, macro_balance_weight
    )

    gather = true_row[:, :, None, None].expand(-1, -1, 1, 3)
    true_delta = conditional_delta.gather(2, gather).squeeze(2)
    position_element = F.smooth_l1_loss(
        true_delta, target_visible_delta_m, beta=huber_beta_m, reduction="none"
    ).mean(dim=-1)
    position, position_no, position_yes = _balanced_group_mean(
        position_element, optimization_mask, target_switch_count, position_balance
    )
    position_mse_element = (
        (true_delta - target_visible_delta_m).square().mean(dim=-1)
    )
    position_mse, _, _ = _balanced_group_mean(
        position_mse_element, optimization_mask, target_switch_count,
        position_balance,
    )
    position_error = torch.linalg.vector_norm(
        true_delta - target_visible_delta_m, dim=-1
    )
    eligible_position_error = position_error[optimization_mask]
    tail_count = max(
        1, int(math.ceil(position_tail_fraction * eligible_position_error.numel()))
    )
    position_tail = eligible_position_error.topk(tail_count).values.mean()
    # A positive floor keeps near-boundary visibility switches from producing
    # unbounded 1/tau gradients while still conditioning the short-time trend.
    safe_tau = torch.where(
        optimization_mask,
        tau.clamp_min(rate_tau_floor_s),
        torch.ones_like(tau),
    )
    predicted_average_rate = true_delta / safe_tau.unsqueeze(-1)
    target_average_rate = target_visible_delta_m / safe_tau.unsqueeze(-1)
    rate_element = F.smooth_l1_loss(
        predicted_average_rate, target_average_rate,
        beta=rate_huber_beta_mps, reduction="none",
    ).mean(dim=-1)
    rate, _, _ = _balanced_group_mean(
        rate_element, optimization_mask, target_switch_count, position_balance
    )

    trend = conditional_delta.sum() * 0.0
    pair_count = query_mask.new_zeros((), dtype=torch.long)
    if trend_weight > 0:
        predicted_pair = true_delta[:, :, None, :] - true_delta[:, None, :, :]
        target_pair = (
            target_visible_delta_m[:, :, None, :]
            - target_visible_delta_m[:, None, :, :]
        )
        pair_mask = (
            optimization_mask[:, :, None] & optimization_mask[:, None, :]
            & (tau[:, :, None] > tau[:, None, :])
        )
        pair_count = pair_mask.sum()
        if bool(pair_mask.any()):
            trend_element = F.smooth_l1_loss(
                predicted_pair, target_pair, beta=huber_beta_m, reduction="none"
            ).mean(dim=-1)
            trend = trend_element[pair_mask].mean()
    objective = (
        switch_weight * switch
        + position_weight * position
        + position_mse_weight * position_mse
        + rate_weight * rate
        + position_tail_weight * position_tail
        + trend_weight * trend
    )
    return objective, {
        "objective": objective,
        "switch": switch,
        "switch_no_change": switch_no,
        "switch_changed": switch_yes,
        "position": position,
        "position_no_change": position_no,
        "position_changed": position_yes,
        "position_mse": position_mse,
        "position_tail": position_tail,
        "rate": rate,
        "trend": trend,
        "trend_pair_count": pair_count,
    }


@torch.no_grad()
def observable_future_batch_metrics(
    prediction: dict[str, torch.Tensor],
    candidate_step: torch.Tensor,
    candidate_mask: torch.Tensor,
    target_switch_count: torch.Tensor,
    target_visible_delta_m: torch.Tensor,
    target_query_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return mergeable counts and error vectors; callers compute percentiles."""
    true_row, query_mask = _target_candidate_row(
        candidate_step, candidate_mask, target_switch_count, target_query_mask
    )
    logits = prediction["switch_logits"]
    conditional = prediction["conditional_delta_m"]
    selected_row = prediction.get("selected_candidate_row")
    if selected_row is None:
        selected_row = logits.argmax(dim=-1)
    if selected_row.shape != logits.shape[:2]:
        raise ValueError("selected_candidate_row must have shape [B,Q]")
    selected_step = candidate_step.gather(1, selected_row)
    correct = query_mask & (selected_step == target_switch_count)
    gather_true = true_row[:, :, None, None].expand(-1, -1, 1, 3)
    conditional_true = conditional.gather(2, gather_true).squeeze(2)
    gather_hard = selected_row[:, :, None, None].expand(-1, -1, 1, 3)
    hard = conditional.gather(2, gather_hard).squeeze(2)
    conditional_error = torch.linalg.vector_norm(
        conditional_true - target_visible_delta_m, dim=-1
    )
    hard_error = torch.linalg.vector_norm(hard - target_visible_delta_m, dim=-1)
    switched = query_mask & (target_switch_count != 0)
    return {
        "eligible_count": query_mask.sum(),
        "correct_count": correct.sum(),
        "switched_count": switched.sum(),
        "switched_correct_count": (switched & correct).sum(),
        "conditional_error_m": conditional_error[query_mask],
        "hard_error_m": hard_error[query_mask],
    }
