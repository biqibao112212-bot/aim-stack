"""Trajectory-space objectives for the V21 rotation A/B comparison."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .cyclic_future_loss import truth_omega_from_future


def _mean_or_zero(values: list[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    return torch.stack(values).mean() if values else reference.sum() * 0.0


def _role_query_groups(
    element: torch.Tensor,
    rule_query: torch.Tensor,
    visible: torch.Tensor,
    warm: torch.Tensor,
) -> list[torch.Tensor]:
    groups: list[torch.Tensor] = []
    for query in range(1, element.shape[1]):
        query_valid = rule_query[:, query].to(torch.bool)[:, None]
        for role in (visible, warm):
            mask = query_valid & role
            if bool(mask.any()):
                groups.append(element[:, query][mask])
    return groups


def cyclic_rotation_ab_loss(
    prediction: dict[str, torch.Tensor],
    s_state: dict[str, torch.Tensor],
    future_position: torch.Tensor,
    tau: torch.Tensor,
    rule_query: torch.Tensor,
    *,
    architecture: str,
    huber_beta_m: float = 0.01,
    tail_weight: float = 0.20,
    edge_weight: float = 0.15,
    rigid_weight: float = 0.02,
    omega_magnitude_weight: float = 0.05,
    max_omega_rad_s: float = 20.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Optimize future trajectory; rotation direction is never a loss target."""
    if architecture not in {"parametric_v2", "direct_trajectory"}:
        raise ValueError(f"unsupported rotation A/B architecture: {architecture}")
    if min(
        huber_beta_m, max_omega_rad_s,
    ) <= 0 or min(
        tail_weight, edge_weight, rigid_weight,
        omega_magnitude_weight,
    ) < 0:
        raise ValueError("rotation A/B loss scales are invalid")
    predicted_delta = prediction.get("delta_m")
    if predicted_delta is None or predicted_delta.shape != future_position.shape:
        raise ValueError("prediction delta must match future_position [B,Q,4,3]")
    valid = prediction.get("future_valid")
    direction_valid = prediction.get("direction_valid")
    if valid is None or valid.shape != future_position.shape[:1] + (4,):
        raise ValueError("future_valid must have shape [B,4]")
    if direction_valid is None or direction_valid.shape != future_position.shape[:1]:
        raise ValueError("direction_valid must have shape [B]")
    expected = s_state["q0_valid"] & (
        s_state["current_visible"] | s_state["anchor_composed"]
    ) & direction_valid[:, None]
    if not torch.equal(valid.to(torch.bool), expected.to(torch.bool)):
        raise ValueError("rotation A/B task mask disagrees with direction validity")
    if not bool(valid.any()):
        raise ValueError("rotation A/B batch has no direction-qualified tracks")

    target_delta = future_position - future_position[:, :1]
    coordinate_error = F.smooth_l1_loss(
        predicted_delta, target_delta, beta=huber_beta_m, reduction="none",
    ).mean(dim=-1)
    visible = valid & s_state["current_visible"].to(torch.bool)
    warm = valid & s_state["anchor_composed"].to(torch.bool)
    coordinate_groups = _role_query_groups(
        coordinate_error, rule_query, visible, warm,
    )
    if not coordinate_groups:
        raise ValueError("rotation A/B batch has no eligible future supervision")
    trajectory = torch.stack([group.mean() for group in coordinate_groups]).mean()

    tail_groups: list[torch.Tensor] = []
    vector_error = torch.linalg.vector_norm(predicted_delta - target_delta, dim=-1)
    for group in _role_query_groups(vector_error, rule_query, visible, warm):
        count = max(1, int((group.numel() + 9) // 10))
        tail_groups.append(torch.topk(group, count, largest=True).values.mean())
    tail = _mean_or_zero(tail_groups, predicted_delta)

    predicted_position = prediction["position_m"]
    truth_edge = torch.roll(future_position, shifts=-1, dims=2) - future_position
    predicted_edge = torch.roll(predicted_position, shifts=-1, dims=2) - predicted_position
    pair_valid = valid & torch.roll(valid, shifts=-1, dims=1)
    edge_groups: list[torch.Tensor] = []
    for query in range(1, predicted_delta.shape[1]):
        mask = rule_query[:, query].to(torch.bool)[:, None] & pair_valid
        if bool(mask.any()):
            value = F.smooth_l1_loss(
                predicted_edge[:, query], truth_edge[:, query],
                beta=huber_beta_m, reduction="none",
            ).mean(dim=-1)
            edge_groups.append(value[mask].mean())
    edge = _mean_or_zero(edge_groups, predicted_delta)

    q0_distance = torch.cdist(predicted_position[:, 0], predicted_position[:, 0])
    predicted_distance = torch.cdist(predicted_position, predicted_position)
    pair_matrix = valid[:, :, None] & valid[:, None, :]
    rigid_groups: list[torch.Tensor] = []
    for query in range(1, predicted_delta.shape[1]):
        mask = rule_query[:, query].to(torch.bool)[:, None, None] & pair_matrix
        if bool(mask.any()):
            drift = (predicted_distance[:, query] - q0_distance).abs()
            rigid_groups.append(drift[mask].mean())
    rigid = _mean_or_zero(rigid_groups, predicted_delta)

    omega_magnitude = predicted_delta.sum() * 0.0
    omega_support_fraction = direction_valid.float().mean().detach()
    if architecture == "parametric_v2":
        predicted_magnitude = prediction.get("omega_magnitude_rad_s")
        if predicted_magnitude is None:
            raise ValueError("parametric_v2 requires omega magnitude output")
        truth_omega, support = truth_omega_from_future(
            future_position, tau, rule_query, valid,
        )
        support = support & direction_valid
        if bool(support.any()):
            omega_magnitude = F.smooth_l1_loss(
                predicted_magnitude[support] / max_omega_rad_s,
                truth_omega[support].abs() / max_omega_rad_s,
                beta=0.01, reduction="mean",
            )
        omega_support_fraction = support.float().mean().detach()

    total = (
        trajectory
        + tail_weight * tail
        + edge_weight * edge
        + rigid_weight * rigid
        + omega_magnitude_weight * omega_magnitude
    )
    return total, {
        "objective": total,
        "trajectory": trajectory,
        "tail": tail,
        "edge": edge,
        "rigid": rigid,
        "omega_magnitude": omega_magnitude,
        "omega_support_fraction": omega_support_fraction,
        "direction_coverage": direction_valid.float().mean().detach(),
    }
