"""Interpretable objective for anchor-relative cyclic q0 restoration."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cyclic_anchor_edge_loss(
    prediction: dict[str, torch.Tensor],
    target_q0_m: torch.Tensor,
    motion_class: torch.Tensor,
    *,
    huber_beta_m: float = 0.01,
    sigma_weight: float = 0.05,
    edge_weight: float = 1.0,
    recent_age_s: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train visible q0 propagation and dynamically supported directed edges.

    Motion class is used only to balance supervised groups; it is never a model
    input.  Cold targets and hidden stationary/translation targets remain absent.
    Rotation/combined hidden targets are balanced by support kind and age so the
    stale asynchronous tail cannot be hidden by abundant recent observations.
    """
    q0 = prediction["q0_m"]
    sigma = prediction["q0_sigma_m"].squeeze(-1)
    if q0.shape != target_q0_m.shape or q0.ndim != 3 or q0.shape[1:] != (4, 3):
        raise ValueError("q0 prediction and target must have shape [B,4,3]")
    if motion_class.shape != q0.shape[:1]:
        raise ValueError("motion_class must have shape [B]")
    if (
        huber_beta_m <= 0 or sigma_weight < 0 or edge_weight < 0
        or recent_age_s <= 0
    ):
        raise ValueError("anchor-edge loss scales are invalid")

    dynamic = (motion_class == 2) | (motion_class == 3)
    q0_observed = prediction["q0_observed"].to(torch.bool)
    visible_to_propagate = (
        prediction["current_visible"].to(torch.bool) & ~q0_observed
    )
    self_warm = prediction["self_warm"].to(torch.bool)
    pair_warm = prediction["edge_warm"].to(torch.bool)
    age = prediction["age_s"]
    recent = age <= recent_age_s
    selected = prediction["anchor_composed"].to(torch.bool) & dynamic[:, None]

    position_component = F.smooth_l1_loss(
        q0, target_q0_m, beta=huber_beta_m, reduction="none"
    ).mean(dim=-1)
    position_groups: list[torch.Tensor] = []
    for route in range(4):
        group = (motion_class == route)[:, None] & visible_to_propagate
        if bool(group.any()):
            position_groups.append(position_component[group].mean())
    for route in (2, 3):
        route_mask = (motion_class == route)[:, None]
        for support_mask in (self_warm, pair_warm):
            for age_mask in (recent, ~recent):
                group = route_mask & support_mask & age_mask
                if bool(group.any()):
                    position_groups.append(position_component[group].mean())
    position = (
        torch.stack(position_groups).mean()
        if position_groups else q0.sum() * 0.0
    )

    predicted_edge = prediction["edge0_m"]
    target_edge = torch.roll(target_q0_m, shifts=-1, dims=1) - target_q0_m
    edge_component = F.smooth_l1_loss(
        predicted_edge, target_edge, beta=huber_beta_m, reduction="none"
    ).mean(dim=-1)
    edge_supported = prediction["edge0_supported"].to(torch.bool)
    pair_seen = prediction["pair_seen"].to(torch.bool)
    relevant = prediction["relevant_edge"].to(torch.bool)
    edge_age = prediction["edge_age_s"]
    edge_recent = edge_age <= recent_age_s
    edge_groups: list[torch.Tensor] = []
    for route in (2, 3):
        route_mask = (motion_class == route)[:, None]
        for support_mask in (pair_seen, edge_supported & ~pair_seen):
            for age_mask in (edge_recent, ~edge_recent):
                group = (
                    route_mask & relevant & edge_supported
                    & support_mask & age_mask
                )
                if bool(group.any()):
                    edge_groups.append(edge_component[group].mean())
    edge = (
        torch.stack(edge_groups).mean()
        if edge_groups else predicted_edge.sum() * 0.0
    )

    error_m = torch.linalg.vector_norm(q0.detach() - target_q0_m, dim=-1)
    calibrated = selected | visible_to_propagate
    calibration = (
        F.smooth_l1_loss(
            sigma[calibrated], error_m[calibrated],
            beta=huber_beta_m, reduction="mean",
        )
        if bool(calibrated.any()) else sigma.sum() * 0.0
    )
    total = position + edge_weight * edge + sigma_weight * calibration
    return total, {
        "warm_adjacent_position": position,
        "supported_edge": edge,
        "sigma_calibration": calibration,
        "selected_fraction": selected.to(q0.dtype).mean(),
        "visible_propagated_fraction": visible_to_propagate.to(q0.dtype).mean(),
        "async_edge_fraction": (
            edge_supported & ~pair_seen & relevant & dynamic[:, None]
        ).to(q0.dtype).mean(),
    }
