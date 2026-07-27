"""Trajectory-space objectives for the V21 rotation A/B comparison."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .cyclic_future_loss import truth_omega_from_future


def _mean_or_zero(values: list[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    return torch.stack(values).mean() if values else reference.sum() * 0.0


def _top_fraction_mean(value: torch.Tensor, fraction: float) -> torch.Tensor:
    if value.numel() == 0:
        raise ValueError("tail group must not be empty")
    count = max(1, int((value.numel() * fraction) + 0.999999))
    return torch.topk(value, count, largest=True).values.mean()


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
    edge_chord_weight: float = 0.0,
    relation_probe_weight: float = 0.0,
    q3_tail_weight: float = 0.0,
    max_omega_rad_s: float = 20.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Optimize future trajectory; rotation direction is never a loss target."""
    if architecture not in {
        "parametric_v2", "direct_trajectory",
        "parametric_relational_v3", "direct_relational_trajectory",
        "direct_ordered_relational_trajectory",
    }:
        raise ValueError(f"unsupported rotation A/B architecture: {architecture}")
    if (
        future_position.ndim != 4
        or future_position.shape[2:] != (4, 3)
        or tau.shape != future_position.shape[:2]
        or rule_query.shape != future_position.shape[:2]
    ):
        raise ValueError("future position, tau and rule-query shapes do not match")
    if not bool(torch.isfinite(tau).all()) or bool(torch.any(tau < 0)):
        raise ValueError("future query tau must be finite and nonnegative")
    if not bool(torch.all(tau[:, 0].abs() <= 1e-7)):
        raise ValueError("query zero must be the q0 timestamp")
    if not bool(rule_query[:, 0].to(torch.bool).all()):
        raise ValueError("query zero must be eligible for every sample")
    if min(
        huber_beta_m, max_omega_rad_s,
    ) <= 0 or min(
        tail_weight, edge_weight, rigid_weight,
        omega_magnitude_weight, edge_chord_weight,
        relation_probe_weight, q3_tail_weight,
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
    warm = (
        valid & s_state["anchor_composed"].to(torch.bool)
        & ~s_state["current_visible"].to(torch.bool)
    )
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

    q3_tail_groups: list[torch.Tensor] = []
    if predicted_delta.shape[1] > 3:
        q3_eligible = (
            rule_query[:, 0].to(torch.bool)
            & rule_query[:, 3].to(torch.bool)
            & (tau[:, 3] > 0)
        )[:, None]
        for role in (visible, warm):
            mask = q3_eligible & role
            if bool(mask.any()):
                q3_tail_groups.append(_top_fraction_mean(
                    vector_error[:, 3][mask], 0.10,
                ))
    q3_tail = (
        torch.stack(q3_tail_groups).amax()
        if q3_tail_groups else predicted_delta.sum() * 0.0
    )

    predicted_position = prediction["position_m"]
    if architecture == "direct_ordered_relational_trajectory":
        truth_edge = torch.roll(target_delta, shifts=-1, dims=2) - target_delta
        predicted_edge = (
            torch.roll(predicted_delta, shifts=-1, dims=2) - predicted_delta
        )
    else:
        truth_edge = torch.roll(future_position, shifts=-1, dims=2) - future_position
        predicted_edge = (
            torch.roll(predicted_position, shifts=-1, dims=2) - predicted_position
        )
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

    predicted_edge_delta = (
        torch.roll(predicted_delta, shifts=-1, dims=2) - predicted_delta
    )
    target_edge_delta = (
        torch.roll(target_delta, shifts=-1, dims=2) - target_delta
    )
    s_q0_edge = torch.roll(s_state["q0_m"], shifts=-1, dims=1) - s_state["q0_m"]
    raw_s_edge_length = torch.linalg.vector_norm(s_q0_edge, dim=-1)
    s_edge_length = raw_s_edge_length.clamp_min(1e-4)
    truth_q0_edge = (
        torch.roll(future_position[:, 0], shifts=-1, dims=1)
        - future_position[:, 0]
    )
    raw_truth_edge_length = torch.linalg.vector_norm(truth_q0_edge, dim=-1)
    truth_edge_length = raw_truth_edge_length.clamp_min(1e-4)
    predicted_chord_ratio = torch.linalg.vector_norm(
        predicted_edge_delta, dim=-1,
    ) / s_edge_length[:, None]
    target_chord_ratio = torch.linalg.vector_norm(
        target_edge_delta, dim=-1,
    ) / truth_edge_length[:, None]
    chord_element = F.smooth_l1_loss(
        predicted_chord_ratio, target_chord_ratio,
        beta=0.02, reduction="none",
    )
    visible_pair = visible & torch.roll(visible, shifts=-1, dims=1)
    warm_pair = warm & torch.roll(warm, shifts=-1, dims=1)
    mixed_pair = (
        (visible & torch.roll(warm, shifts=-1, dims=1))
        | (warm & torch.roll(visible, shifts=-1, dims=1))
    )
    pair_roles = (visible_pair, mixed_pair, warm_pair)
    chord_groups: list[torch.Tensor] = []
    q0_query_valid = rule_query[:, 0].to(torch.bool)[:, None]
    for query in range(1, predicted_delta.shape[1]):
        eligible = (
            q0_query_valid
            & rule_query[:, query].to(torch.bool)[:, None]
            & (tau[:, query] > 0)[:, None]
            & pair_valid
            & (raw_s_edge_length > 1e-4)
            & (raw_truth_edge_length > 1e-4)
        )
        for pair_role in pair_roles:
            mask = eligible & pair_role
            if bool(mask.any()):
                chord_groups.append(chord_element[:, query][mask].mean())
    edge_chord = _mean_or_zero(chord_groups, predicted_delta)

    relation_probe = predicted_delta.sum() * 0.0
    relation_probe_ratio = prediction.get("relation_edge_chord_ratio")
    if relation_probe_ratio is not None:
        if relation_probe_ratio.shape != predicted_chord_ratio.shape:
            raise ValueError("relation edge-chord probe shape does not match queries")
        relation_probe_support = prediction.get("relational_local_track_support")
        if (
            relation_probe_support is None
            or relation_probe_support.shape != pair_valid.shape
        ):
            raise ValueError("relation chord probe requires local track support")
        probe_element = F.smooth_l1_loss(
            relation_probe_ratio, target_chord_ratio,
            beta=0.02, reduction="none",
        )
        probe_groups: list[torch.Tensor] = []
        for query in range(1, predicted_delta.shape[1]):
            eligible = (
                q0_query_valid
                & rule_query[:, query].to(torch.bool)[:, None]
                & (tau[:, query] > 0)[:, None]
                & pair_valid
                & (raw_s_edge_length > 1e-4)
                & (raw_truth_edge_length > 1e-4)
                & relation_probe_support.to(torch.bool)
            )
            for pair_role in pair_roles:
                mask = eligible & pair_role
                if bool(mask.any()):
                    probe_groups.append(probe_element[:, query][mask].mean())
        relation_probe = _mean_or_zero(probe_groups, predicted_delta)
    elif relation_probe_weight > 0:
        raise ValueError("relation probe weight requires a relation-only chord head")

    omega_magnitude = predicted_delta.sum() * 0.0
    omega_support_fraction = direction_valid.float().mean().detach()
    if architecture in {"parametric_v2", "parametric_relational_v3"}:
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
        + edge_chord_weight * edge_chord
        + relation_probe_weight * relation_probe
        + q3_tail_weight * q3_tail
    )
    parts = {
        "objective": total,
        "trajectory": trajectory,
        "tail": tail,
        "edge": edge,
        "rigid": rigid,
        "edge_chord": edge_chord,
        "relation_probe": relation_probe,
        "q3_tail": q3_tail,
        "omega_magnitude": omega_magnitude,
        "omega_support_fraction": omega_support_fraction,
        "direction_coverage": direction_valid.float().mean().detach(),
    }
    if "relational_edge_support" in prediction:
        parts["relational_edge_coverage"] = prediction[
            "relational_edge_support"
        ].float().mean().detach()
        parts["relational_curve_coverage"] = prediction[
            "relational_curve_support"
        ].float().mean().detach()
    return total, parts
