"""Interpretable objectives for independent cyclic-track experts."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _balanced_router_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses = []
    for route in range(4):
        group = labels == route
        if bool(group.any()):
            losses.append(F.cross_entropy(logits[group], labels[group]))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _track_role_weights(
    primary_index: torch.Tensor,
    current_visible_mask: torch.Tensor,
    *, opposite_weight: float,
) -> torch.Tensor:
    if primary_index.ndim != 1 or current_visible_mask.shape != (primary_index.shape[0], 4):
        raise ValueError("current track roles must have shape [B] and [B,4]")
    batch = primary_index.shape[0]
    weights = torch.full(
        (batch, 4), float(opposite_weight),
        device=primary_index.device, dtype=torch.float32,
    )
    row = torch.arange(batch, device=primary_index.device)
    weights[row, primary_index] = 1.0
    weights[row, (primary_index - 1) % 4] = 1.0
    weights[row, (primary_index + 1) % 4] = 1.0
    return torch.maximum(weights, current_visible_mask.to(weights.dtype))


def _weighted_huber(
    prediction: torch.Tensor, target: torch.Tensor,
    role_weight: torch.Tensor, query_mask: torch.Tensor, beta_m: float,
) -> torch.Tensor:
    error = F.smooth_l1_loss(
        prediction, target, beta=beta_m, reduction="none"
    ).mean(dim=-1)
    weight = role_weight[:, None].expand_as(error) * query_mask[:, :, None].to(
        error.dtype
    )
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


def _motion_delta_huber(
    prediction: torch.Tensor, target: torch.Tensor,
    role_weight: torch.Tensor, query_mask: torch.Tensor, beta_m: float,
) -> torch.Tensor:
    predicted_delta = prediction - prediction[:, :1]
    target_delta = target - target[:, :1]
    # q0 is identically zero in both deltas and would only dilute the motion
    # term.  It remains available as a validation identity, not supervision.
    return _weighted_huber(
        predicted_delta[:, 1:], target_delta[:, 1:], role_weight,
        query_mask[:, 1:], beta_m,
    )


def _self_rigid_huber(
    prediction: torch.Tensor, query_mask: torch.Tensor, beta_m: float,
) -> torch.Tensor:
    pair = prediction[:, :, :, None] - prediction[:, :, None, :]
    distance = torch.linalg.vector_norm(pair, dim=-1)
    drift = distance - distance[:, :1]
    upper = torch.triu(
        torch.ones((4, 4), dtype=torch.bool, device=prediction.device), diagonal=1
    )
    values = drift[..., upper]
    if bool(query_mask.any()):
        selected = values[query_mask]
        return F.smooth_l1_loss(
            selected, torch.zeros_like(selected), beta=beta_m, reduction="mean"
        )
    return values.sum() * 0.0


def cyclic_track_expert_loss(
    prediction: dict[str, torch.Tensor],
    target: torch.Tensor,
    motion_class: torch.Tensor,
    rule_query: torch.Tensor,
    current_primary_index: torch.Tensor,
    current_visible_mask: torch.Tensor,
    *,
    huber_beta_m: float = 0.01,
    motion_delta_weight: float = 0.5,
    rigid_weight: float = 0.05,
    router_weight: float = 0.1,
    opposite_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train only the matching raw expert plus a balanced four-class router.

    There is no cyclic-shift search: target and prediction use the same local
    tracker labels.  The rigid term preserves each prediction's own q0 pair
    distances and never references a radius, height, center, or template.
    """
    expert = prediction.get("expert_position")
    router_logit = prediction.get("router_logit")
    if expert is None or router_logit is None:
        raise ValueError("cyclic-track output requires expert_position and router_logit")
    if expert.ndim != 5 or expert.shape[1] != 4 or expert.shape[3:] != (4, 3):
        raise ValueError("expert_position must have shape [B,4,Q,4,3]")
    if target.shape != expert.shape[:1] + expert.shape[2:]:
        raise ValueError("target must have shape [B,Q,4,3]")
    if motion_class.shape != target.shape[:1] or router_logit.shape != (target.shape[0], 4):
        raise ValueError("motion_class/router shapes do not match the batch")
    if rule_query.shape != target.shape[:2]:
        raise ValueError("rule_query must have shape [B,Q]")
    if not bool(rule_query[:, 0].to(torch.bool).all()):
        raise ValueError("q0 must be an eligible rule query for every sample")
    if bool(torch.any((motion_class < 0) | (motion_class > 3))):
        raise ValueError("motion_class must be within [0,3]")
    positive = (
        huber_beta_m, motion_delta_weight, rigid_weight,
        router_weight, opposite_weight,
    )
    if min(positive) <= 0 or opposite_weight > 1:
        raise ValueError("cyclic-track loss scales are invalid")

    role_weight = _track_role_weights(
        current_primary_index, current_visible_mask,
        opposite_weight=opposite_weight,
    )
    position_losses = []
    delta_losses = []
    rigid_losses = []
    parts: dict[str, torch.Tensor] = {}
    names = ("stationary", "translation", "rotation", "combined")
    for route, name in enumerate(names):
        group = motion_class == route
        if bool(group.any()):
            raw = expert[group, route]
            truth = target[group]
            weights = role_weight[group]
            query = rule_query[group].to(torch.bool)
            position = _weighted_huber(
                raw, truth, weights, query, huber_beta_m
            )
            delta = _motion_delta_huber(
                raw, truth, weights, query, huber_beta_m
            )
            rigid = _self_rigid_huber(raw, query, huber_beta_m)
            position_losses.append(position)
            delta_losses.append(delta)
            rigid_losses.append(rigid)
        else:
            zero = expert[:, route].sum() * 0.0
            position = delta = rigid = zero
        parts[f"{name}_position"] = position
        parts[f"{name}_motion_delta"] = delta
        parts[f"{name}_self_rigid"] = rigid
    position_loss = torch.stack(position_losses).mean()
    delta_loss = torch.stack(delta_losses).mean()
    rigid_loss = torch.stack(rigid_losses).mean()
    router_loss = _balanced_router_ce(router_logit, motion_class)
    total = (
        position_loss + motion_delta_weight * delta_loss
        + rigid_weight * rigid_loss + router_weight * router_loss
    )
    parts.update({
        "position": position_loss,
        "motion_delta": delta_loss,
        "self_rigid": rigid_loss,
        "router": router_loss,
    })
    return total, parts
