"""Interpretable training-only losses for factorized motion experts."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .pnp_state_targets import truth_trajectory_targets


def _zero_huber(value: torch.Tensor, beta: float) -> torch.Tensor:
    return F.smooth_l1_loss(
        value, torch.zeros_like(value), beta=beta, reduction="mean",
    )


def _masked_zero_huber(
    value: torch.Tensor, mask: torch.Tensor, beta: float,
) -> torch.Tensor:
    if bool(mask.any()):
        return _zero_huber(value[mask], beta)
    return value.sum() * 0.0


def _balanced_gate_loss(
    logit: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor,
) -> torch.Tensor:
    positive_loss = F.binary_cross_entropy_with_logits(
        logit[positive], torch.ones_like(logit[positive]), reduction="mean",
    ) if bool(positive.any()) else logit.sum() * 0.0
    negative_loss = F.binary_cross_entropy_with_logits(
        logit[negative], torch.zeros_like(logit[negative]), reduction="mean",
    ) if bool(negative.any()) else logit.sum() * 0.0
    if bool(positive.any()) and bool(negative.any()):
        return 0.5 * (positive_loss + negative_loss)
    return positive_loss + negative_loss


def factorized_expert_loss(
    prediction: dict[str, torch.Tensor], target: torch.Tensor,
    tau: torch.Tensor, rule_query: torch.Tensor, geometry: torch.Tensor, *,
    huber_beta_m: float = 0.005, reference_horizon_s: float = 0.5,
    expert_weight: float = 1.0, gate_weight: float = 0.1,
    move_negative_mps: float = 0.01, move_positive_mps: float = 0.10,
    rotate_negative_rad_s: float = 0.05,
    rotate_positive_rad_s: float = 0.20,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise q0 pose, positive experts and causal gates without leakage.

    Future truth constructs detached loss labels only. Predictor forward inputs
    remain observation positions, masks, real timestamps and query tau.
    """
    required = {
        "center0", "phase0", "velocity_expert", "move_logit",
        "omega_expert", "rotate_logit", "position_mean",
    }
    missing = required - prediction.keys()
    if missing:
        raise ValueError(f"factorized expert output is missing: {sorted(missing)}")
    if target.ndim != 4 or target.shape[2:] != (4, 3):
        raise ValueError("target must have shape [B,Q,4,3]")
    if tau.shape != rule_query.shape or tau.shape[:2] != target.shape[:2]:
        raise ValueError("tau/rule_query must match target [B,Q]")
    if target.shape[1] < 4:
        raise ValueError("factorized expert supervision requires q0 through q3")
    if min(huber_beta_m, reference_horizon_s, expert_weight, gate_weight) <= 0:
        raise ValueError("factorized expert loss scales must be positive")
    if not 0 <= move_negative_mps < move_positive_mps:
        raise ValueError("move gate thresholds are invalid")
    if not 0 <= rotate_negative_rad_s < rotate_positive_rad_s:
        raise ValueError("rotate gate thresholds are invalid")

    truth = truth_trajectory_targets(
        target[:, :4], tau[:, :4], geometry, rule_queries=4,
    )
    active = rule_query[:, :4].to(torch.bool).all(dim=1) & truth["constant_motion"]
    if not bool(active.any()):
        zero = prediction["position_mean"].sum() * 0.0
        return zero, {
            "center0": zero.detach(), "phase0": zero.detach(),
            "velocity_expert": zero.detach(), "omega_expert": zero.detach(),
            "move_gate": zero.detach(), "rotate_gate": zero.detach(),
            "active_fraction": active.float().mean().detach(),
        }

    geometry_radius_m = torch.sqrt(
        geometry[:, :2].float().square().sum(dim=-1).mean()
    )
    truth_velocity = truth["velocity"]
    truth_omega = truth["omega"]
    speed = torch.linalg.vector_norm(truth_velocity, dim=-1)
    abs_omega = truth_omega.abs()
    move_positive = active & (speed >= move_positive_mps)
    move_negative = active & (speed <= move_negative_mps)
    rotate_positive = active & (abs_omega >= rotate_positive_rad_s)
    rotate_negative = active & (abs_omega <= rotate_negative_rad_s)

    center0 = _zero_huber(
        prediction["center0"][active] - truth["center0"][active],
        huber_beta_m,
    )
    phase0 = _zero_huber(
        geometry_radius_m
        * (prediction["phase0"][active] - truth["phase0"][active]),
        huber_beta_m,
    )
    velocity_expert = _masked_zero_huber(
        reference_horizon_s
        * (prediction["velocity_expert"] - truth_velocity),
        move_positive, huber_beta_m,
    )
    omega_expert = _masked_zero_huber(
        geometry_radius_m * reference_horizon_s
        * (prediction["omega_expert"] - truth_omega),
        rotate_positive, huber_beta_m,
    )
    move_gate = _balanced_gate_loss(
        prediction["move_logit"], move_positive, move_negative,
    )
    rotate_gate = _balanced_gate_loss(
        prediction["rotate_logit"], rotate_positive, rotate_negative,
    )
    total = (
        center0 + phase0 + expert_weight * (velocity_expert + omega_expert)
        + gate_weight * (move_gate + rotate_gate)
    )
    return total, {
        "center0": center0, "phase0": phase0,
        "velocity_expert": velocity_expert, "omega_expert": omega_expert,
        "move_gate": move_gate, "rotate_gate": rotate_gate,
        "active_fraction": active.float().mean(),
        "move_positive_fraction": move_positive.float().mean(),
        "rotate_positive_fraction": rotate_positive.float().mean(),
    }
