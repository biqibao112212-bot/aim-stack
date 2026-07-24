"""Compact objectives for center-free future motion experts."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def truth_omega_from_future(
    future_position: torch.Tensor,
    tau: torch.Tensor,
    rule_query: torch.Tensor,
    track_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive a loss-only yaw-rate label from eligible adjacent truth edges.

    The shortest eligible positive query is used per sample to reduce angle
    wrapping.  Cold/opposite tracks cannot influence this auxiliary label.
    """
    if future_position.ndim != 4 or future_position.shape[2:] != (4, 3):
        raise ValueError("future_position must have shape [B,Q,4,3]")
    if tau.shape != future_position.shape[:2] or rule_query.shape != tau.shape:
        raise ValueError("tau/rule_query must match future positions")
    if track_mask.shape != (future_position.shape[0], 4):
        raise ValueError("track_mask must have shape [B,4]")
    q0 = future_position[:, 0, :, :2]
    edge0 = torch.roll(q0, shifts=-1, dims=1) - q0
    edge_support = track_mask.to(torch.bool) & torch.roll(
        track_mask.to(torch.bool), shifts=-1, dims=1,
    )
    usable_edge = edge_support & (edge0.square().sum(dim=-1) > 1e-8)
    best_dt = torch.full_like(tau[:, 0], float("inf"))
    omega = torch.zeros_like(tau[:, 0])
    support = torch.zeros_like(rule_query[:, 0], dtype=torch.bool)
    weight = usable_edge.to(edge0.dtype)
    for query in range(1, future_position.shape[1]):
        qt = future_position[:, query, :, :2]
        edget = torch.roll(qt, shifts=-1, dims=1) - qt
        cross = edge0[..., 0] * edget[..., 1] - edge0[..., 1] * edget[..., 0]
        dot = (edge0 * edget).sum(dim=-1)
        angle = torch.atan2((cross * weight).sum(dim=-1), (dot * weight).sum(dim=-1))
        dt = tau[:, query]
        candidate = (
            rule_query[:, 0].to(torch.bool)
            & rule_query[:, query].to(torch.bool)
            & usable_edge.any(dim=-1)
            & torch.isfinite(angle)
            & torch.isfinite(dt)
            & (dt > 1e-4)
        )
        take = candidate & (dt < best_dt)
        omega = torch.where(take, angle / dt.clamp_min(1e-4), omega)
        best_dt = torch.where(take, dt, best_dt)
        support |= take
    return omega.detach(), support.detach()


def task_track_mask(
    expert: str,
    prediction: dict[str, torch.Tensor],
    s_state: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return the deterministic role mask used by loss and validation."""
    if expert == "translation":
        expected = s_state["q0_valid"] & s_state["current_visible"]
    elif expert in {"rotation", "combined"}:
        expected = s_state["q0_valid"] & (
            s_state["current_visible"] | s_state["anchor_composed"]
        )
    else:
        raise ValueError(f"unsupported dynamic expert: {expert}")
    actual = prediction.get("future_valid")
    if actual is None or actual.shape != expected.shape:
        raise ValueError("prediction future_valid is missing or malformed")
    if not torch.equal(actual.to(torch.bool), expected.to(torch.bool)):
        raise ValueError("model and objective task masks disagree")
    return expected.to(torch.bool)


def _balanced_delta_loss(
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
    rule_query: torch.Tensor,
    visible: torch.Tensor,
    warm: torch.Tensor,
    beta_m: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    element = F.smooth_l1_loss(
        predicted_delta, target_delta, beta=beta_m, reduction="none"
    ).mean(dim=-1)
    groups: list[torch.Tensor] = []
    visible_losses: list[torch.Tensor] = []
    warm_losses: list[torch.Tensor] = []
    for query in range(1, predicted_delta.shape[1]):
        eligible_query = rule_query[:, query].to(torch.bool)
        visible_mask = eligible_query[:, None] & visible
        warm_mask = eligible_query[:, None] & warm
        if bool(visible_mask.any()):
            value = element[:, query][visible_mask].mean()
            visible_losses.append(value)
            groups.append(value)
        if bool(warm_mask.any()):
            value = element[:, query][warm_mask].mean()
            warm_losses.append(value)
            groups.append(value)
    if not groups:
        raise ValueError("future expert batch has no eligible non-q0 supervision")
    zero = predicted_delta.sum() * 0.0
    return torch.stack(groups).mean(), {
        "visible_delta": torch.stack(visible_losses).mean() if visible_losses else zero,
        "warm_delta": torch.stack(warm_losses).mean() if warm_losses else zero,
    }


def cyclic_future_expert_loss(
    expert: str,
    prediction: dict[str, torch.Tensor],
    s_state: dict[str, torch.Tensor],
    future_position: torch.Tensor,
    tau: torch.Tensor,
    rule_query: torch.Tensor,
    *,
    huber_beta_m: float = 0.01,
    omega_weight: float = 0.10,
    omega_sign_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train motion delta only; q0 reconstruction remains frozen in S."""
    if huber_beta_m <= 0 or min(omega_weight, omega_sign_weight) < 0:
        raise ValueError("future loss scales are invalid")
    predicted_delta = prediction.get("delta_m")
    if predicted_delta is None or predicted_delta.shape != future_position.shape:
        raise ValueError("prediction delta must match future_position [B,Q,4,3]")
    if tau.shape != future_position.shape[:2] or rule_query.shape != tau.shape:
        raise ValueError("tau/rule_query must match future_position [B,Q]")
    if not bool(rule_query[:, 0].to(torch.bool).all()):
        raise ValueError("q0 must be eligible for every sample")
    if not bool(torch.isfinite(tau).all()) or bool((tau < 0).any()):
        raise ValueError("future query tau must be finite and non-negative")
    if not torch.equal(tau[:, 0], torch.zeros_like(tau[:, 0])):
        raise ValueError("future query zero must be exact q0")
    target_delta = future_position - future_position[:, :1]
    track_mask = task_track_mask(expert, prediction, s_state)
    visible = track_mask & s_state["current_visible"].to(torch.bool)
    warm = track_mask & s_state["anchor_composed"].to(torch.bool)
    delta, role_parts = _balanced_delta_loss(
        predicted_delta, target_delta, rule_query, visible, warm, huber_beta_m,
    )
    zero = predicted_delta.sum() * 0.0
    omega_loss = zero
    sign_loss = zero
    omega_support_fraction = zero.detach()
    if expert in {"rotation", "combined"}:
        predicted_omega = prediction.get("omega_rad_s")
        if predicted_omega is None or predicted_omega.shape != (future_position.shape[0],):
            raise ValueError("rotating experts require omega_rad_s [B]")
        truth_omega, support = truth_omega_from_future(
            future_position, tau, rule_query, track_mask,
        )
        if bool(support.any()):
            omega_loss = F.smooth_l1_loss(
                predicted_omega[support], truth_omega[support],
                beta=0.1, reduction="mean",
            )
            sign_target = (truth_omega[support] > 0).to(predicted_omega.dtype)
            sign_loss = F.binary_cross_entropy_with_logits(
                predicted_omega[support], sign_target,
            )
        omega_support_fraction = support.float().mean()
    total = delta + omega_weight * omega_loss + omega_sign_weight * sign_loss
    return total, {
        "objective": total,
        "delta": delta,
        "omega": omega_loss,
        "omega_sign": sign_loss,
        "omega_support_fraction": omega_support_fraction,
        "visible_delta": role_parts["visible_delta"],
        "warm_delta": role_parts["warm_delta"],
    }
