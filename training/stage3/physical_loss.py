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


def _masked_huber(
    value: torch.Tensor, reference: torch.Tensor, active: torch.Tensor,
    beta_m: float,
) -> torch.Tensor:
    element = F.smooth_l1_loss(value, reference, beta=beta_m, reduction="none")
    mask = active.to(torch.bool)
    while mask.ndim < element.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(element).to(element.dtype)
    return (element * expanded).sum() / expanded.sum().clamp_min(1.0)


def causal_physical_base_loss(
    prediction: dict[str, torch.Tensor], target: torch.Tensor, tau: torch.Tensor,
    rule_query: torch.Tensor, *, huber_beta_m: float = 0.005,
    q0_weight: float = 2.0, absolute_weight: float = 1.0,
    motion_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """A objective: fixed-slot q0, future position, and motion delta."""
    position = prediction["position_mean"]
    if position.shape != target.shape or position.shape[2:] != (4, 3):
        raise ValueError("causal physical position/target must be [B,Q,4,3]")
    if tau.shape != rule_query.shape or tau.shape[:2] != position.shape[:2]:
        raise ValueError("tau and rule_query must match [B,Q]")
    if not torch.allclose(tau[:, 0], torch.zeros_like(tau[:, 0]), atol=0.0, rtol=0.0):
        raise ValueError("causal physical query zero must be exact tau=0")
    active = rule_query.to(torch.bool)
    if not torch.all(active[:, 0]):
        raise ValueError("causal physical query zero must always be active")
    q0 = F.smooth_l1_loss(
        position[:, 0], target[:, 0], beta=huber_beta_m, reduction="mean"
    )
    absolute = _masked_huber(position, target, active, huber_beta_m)
    predicted_delta = position[:, 1:] - position[:, :1]
    target_delta = target[:, 1:] - target[:, :1]
    motion = _masked_huber(
        predicted_delta, target_delta, active[:, 1:], huber_beta_m
    )
    if min(q0_weight, absolute_weight, motion_weight) <= 0:
        raise ValueError("causal physical loss weights must be positive")
    total = q0_weight * q0 + absolute_weight * absolute + motion_weight * motion
    return total, {"q0": q0, "absolute": absolute, "motion": motion}


def causal_physical_history_regularizers(
    future_prediction: dict[str, torch.Tensor],
    history_prediction: dict[str, torch.Tensor],
    history_position_m: torch.Tensor,
    obs_mask: torch.Tensor,
    event_mask: torch.Tensor,
    event_time_s: torch.Tensor,
    tau: torch.Tensor,
    rule_query: torch.Tensor,
    *,
    geometry_rms_radius_m: float,
    huber_beta_m: float = 0.005,
    constant_history_s: float = 0.2,
    constant_history_events: int = 4,
    minimum_abs_tau_s: float = 0.005,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """B-only history reconstruction and state-label-free shared motion."""
    if geometry_rms_radius_m <= 0:
        raise ValueError("geometry_rms_radius_m must be positive")
    if constant_history_events < 2:
        raise ValueError("constant_history_events must be at least two")
    event = event_mask.to(torch.bool)
    reverse_rank = torch.flip(
        torch.cumsum(torch.flip(event.to(torch.int64), dims=(1,)), dim=1),
        dims=(1,),
    )
    qualified_history = event & (reverse_rank <= constant_history_events)
    history_active = obs_mask.to(torch.bool) & qualified_history.unsqueeze(-1)
    history = _masked_huber(
        history_prediction["position_mean"], history_position_m,
        history_active, huber_beta_m,
    )

    recent_history = (
        qualified_history
        & (event_time_s >= -constant_history_s)
        & (event_time_s.abs() >= minimum_abs_tau_s)
    )
    future_active = (
        rule_query.to(torch.bool)
        & (tau.abs() >= minimum_abs_tau_s)
    )
    if "query_horizon" not in future_prediction or "query_horizon" not in history_prediction:
        raise ValueError("shared motion requires anchor-relative query horizons")
    all_tau = torch.cat((
        history_prediction["query_horizon"], future_prediction["query_horizon"]
    ), dim=1)
    all_active = torch.cat((recent_history, future_active), dim=1)
    all_center = torch.cat((
        history_prediction["delta_center"], future_prediction["delta_center"]
    ), dim=1)
    all_angle = torch.cat((
        history_prediction["delta_angle"], future_prediction["delta_angle"]
    ), dim=1)
    weight = all_active.to(all_tau.dtype)
    denominator = (weight * all_tau.square()).sum(dim=1).clamp_min(1e-8)
    velocity = (
        weight.unsqueeze(-1) * all_tau.unsqueeze(-1) * all_center
    ).sum(dim=1) / denominator.unsqueeze(-1)
    omega = (weight * all_tau * all_angle).sum(dim=1) / denominator
    center_residual = all_center - all_tau.unsqueeze(-1) * velocity.unsqueeze(1)
    angle_residual_m = geometry_rms_radius_m * (
        all_angle - all_tau * omega.unsqueeze(1)
    )
    shared_value = torch.cat((center_residual, angle_residual_m.unsqueeze(-1)), dim=-1)
    shared = _masked_huber(
        shared_value, torch.zeros_like(shared_value), all_active, huber_beta_m
    )
    return history + shared, {
        "history": history,
        "shared": shared,
        "shared_active_fraction": all_active.float().mean(),
    }
