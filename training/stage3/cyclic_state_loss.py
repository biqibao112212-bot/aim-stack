"""Loss for the independently trained cyclic q0 state restorer."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cyclic_state_loss(
    prediction: dict[str, torch.Tensor],
    target_q0_m: torch.Tensor,
    motion_class: torch.Tensor,
    *,
    huber_beta_m: float = 0.01,
    sigma_weight: float = 0.05,
    edge_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise only warm adjacent tracks for rotation and combined motion.

    Current visible tracks are an exact identity bypass.  Cold tracks, and
    hidden tracks in stationary/translation samples, are intentionally absent
    from the deterministic position objective.
    """
    q0 = prediction["q0_m"]
    sigma = prediction["q0_sigma_m"].squeeze(-1)
    warm = prediction["warm_hidden"].to(torch.bool)
    adjacent = prediction["adjacent"].to(torch.bool)
    if q0.shape != target_q0_m.shape or q0.ndim != 3 or q0.shape[1:] != (4, 3):
        raise ValueError("q0 prediction and target must have shape [B,4,3]")
    if motion_class.shape != q0.shape[:1]:
        raise ValueError("motion_class must have shape [B]")
    if huber_beta_m <= 0 or sigma_weight < 0 or edge_weight < 0:
        raise ValueError("loss scales are invalid")
    dynamic = (motion_class == 2) | (motion_class == 3)
    q0_observed = prediction["q0_observed"].to(torch.bool)
    visible_to_propagate = (
        prediction["current_visible"].to(torch.bool) & ~q0_observed
    )
    self_warm = prediction["self_warm"].to(torch.bool)
    edge_warm = prediction["edge_warm"].to(torch.bool)
    selected = warm & adjacent & dynamic[:, None]
    component = F.smooth_l1_loss(
        q0, target_q0_m, beta=huber_beta_m, reduction="none"
    ).mean(dim=-1)
    position_groups = []
    for route in range(4):
        group = (motion_class == route)[:, None] & visible_to_propagate
        if bool(group.any()):
            position_groups.append(component[group].mean())
    for route in (2, 3):
        route_mask = motion_class == route
        for support_mask in (self_warm, edge_warm):
            group = route_mask[:, None] & support_mask
            if bool(group.any()):
                position_groups.append(component[group].mean())
    if position_groups:
        position = torch.stack(position_groups).mean()
        error_m = torch.linalg.vector_norm(
            q0.detach() - target_q0_m, dim=-1
        )
        calibrated = selected | visible_to_propagate
        calibration = F.smooth_l1_loss(
            sigma[calibrated], error_m[calibrated], beta=huber_beta_m,
            reduction="mean",
        )
    else:
        position = q0.sum() * 0.0
        calibration = sigma.sum() * 0.0
    predicted_edge = prediction["edge0_m"]
    target_edge = torch.roll(target_q0_m, shifts=-1, dims=1) - target_q0_m
    edge_component = F.smooth_l1_loss(
        predicted_edge, target_edge, beta=huber_beta_m, reduction="none"
    ).mean(dim=-1)
    edge_selected = (
        prediction["pair_seen"].to(torch.bool)
        & prediction["relevant_edge"].to(torch.bool)
        & dynamic[:, None]
    )
    edge_groups = []
    for route in (2, 3):
        group = (motion_class == route)[:, None] & edge_selected
        if bool(group.any()):
            edge_groups.append(edge_component[group].mean())
    edge = (
        torch.stack(edge_groups).mean()
        if edge_groups else predicted_edge.sum() * 0.0
    )
    total = position + edge_weight * edge + sigma_weight * calibration
    return total, {
        "warm_adjacent_position": position,
        "observed_edge": edge,
        "sigma_calibration": calibration,
        "selected_fraction": selected.to(q0.dtype).mean(),
        "visible_propagated_fraction": visible_to_propagate.to(q0.dtype).mean(),
    }
