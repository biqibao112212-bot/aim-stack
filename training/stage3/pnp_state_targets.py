"""Training-only physical state targets and trajectory state extraction.

These functions consume future physical truth only in the loss/evaluator. They
are never part of a predictor forward graph or deployment input contract.
"""

from __future__ import annotations

import math

import torch


def _query_pose_from_fixed_truth(
    position: torch.Tensor, geometry: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover the exact simulator center and phase from fixed truth slots."""
    if position.ndim != 4 or position.shape[2:] != (4, 3):
        raise ValueError("position must have shape [B,Q,4,3]")
    if tuple(geometry.shape) != (4, 3):
        raise ValueError("geometry must have shape [4,3]")
    with torch.autocast(device_type=position.device.type, enabled=False):
        position = position.float()
        geometry = geometry.float()
        geometry_xy = geometry[:, :2]
        geometry_centered = geometry_xy - geometry_xy.mean(dim=0)
        observed_xy = position[..., :2]
        observed_centered = observed_xy - observed_xy.mean(dim=2, keepdim=True)
        dot = (
            observed_centered[..., 0] * geometry_centered[None, None, :, 0]
            + observed_centered[..., 1] * geometry_centered[None, None, :, 1]
        ).sum(dim=2)
        cross = (
            geometry_centered[None, None, :, 0] * observed_centered[..., 1]
            - geometry_centered[None, None, :, 1] * observed_centered[..., 0]
        ).sum(dim=2)
        yaw = torch.atan2(cross, dot)
        phase = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
        gx, gy = geometry[:, 0], geometry[:, 1]
        relative = torch.stack((
            phase[..., 0, None] * gx - phase[..., 1, None] * gy,
            phase[..., 1, None] * gx + phase[..., 0, None] * gy,
            geometry[:, 2].view(1, 1, 4).expand(position.shape[0], position.shape[1], -1),
        ), dim=-1)
        center = (position - relative).mean(dim=2)
    return center, phase


def _trajectory_state(
    center: torch.Tensor, phase: torch.Tensor, tau: torch.Tensor,
    *, rule_queries: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit center0/v and obtain alias-safe q0-to-q1 yaw0/omega."""
    if center.ndim != 3 or center.shape[-1] != 3:
        raise ValueError("center must have shape [B,Q,3]")
    if phase.shape != (*center.shape[:2], 2) or tau.shape != center.shape[:2]:
        raise ValueError("phase/tau must match center [B,Q]")
    rule_queries = center.shape[1] if rule_queries is None else rule_queries
    if rule_queries < 2 or center.shape[1] < rule_queries:
        raise ValueError("trajectory state requires at least two rule queries")
    with torch.autocast(device_type=center.device.type, enabled=False):
        center = center.float()
        phase = phase.float()
        tau = tau.float()
        rule_tau = tau[:, :rule_queries]
        rule_center = center[:, :rule_queries]
        mean_tau = rule_tau.mean(dim=1, keepdim=True)
        mean_center = rule_center.mean(dim=1, keepdim=True)
        centered_tau = rule_tau - mean_tau
        denominator = centered_tau.square().sum(dim=1).clamp_min(1e-8)
        velocity = (
            centered_tau[:, :, None] * (rule_center - mean_center)
        ).sum(dim=1) / denominator[:, None]
        center0 = mean_center[:, 0] - velocity * mean_tau

        q1_dt = tau[:, 1] - tau[:, 0]
        if bool((q1_dt <= 0).any()):
            raise ValueError("q1 must have a positive horizon")
        # The dataset contract guarantees |omega| <= 15 rad/s and q1 < 0.125 s,
        # hence |omega*q1| < pi and this signed difference is alias-free.
        q0, q1 = phase[:, 0], phase[:, 1]
        cross = q0[:, 0] * q1[:, 1] - q0[:, 1] * q1[:, 0]
        dot = q0[:, 0] * q1[:, 0] + q0[:, 1] * q1[:, 1]
        delta = torch.atan2(cross, dot)
        omega = delta / q1_dt
        phase0 = phase[:, 0]
    return center0, velocity, phase0, omega


def truth_trajectory_targets(
    position: torch.Tensor,
    tau: torch.Tensor,
    geometry: torch.Tensor,
    *,
    rule_queries: int | None = None,
    center_consistency_tolerance_m: float = 0.001,
    yaw_consistency_tolerance_rad: float = 0.001,
    maximum_abs_omega_rad_s: float = 15.0,
) -> dict[str, torch.Tensor]:
    """Build detached physical labels and a constant-motion eligibility mask."""
    minimum_c4_asymmetry = geometry_c4_asymmetry_m(geometry)
    if minimum_c4_asymmetry <= 0.005:
        raise ValueError(
            "full relative yaw requires geometry C4 asymmetry above 5 mm"
        )
    center, phase = _query_pose_from_fixed_truth(position, geometry)
    rule_queries = position.shape[1] if rule_queries is None else rule_queries
    center0, velocity, phase0, omega = _trajectory_state(
        center, phase, tau, rule_queries=rule_queries
    )
    if not torch.allclose(
        tau[:, 0], torch.zeros_like(tau[:, 0]), atol=0.0, rtol=0.0
    ):
        raise ValueError("query zero must have exact tau=0")
    q1_dt = tau[:, 1] - tau[:, 0]
    if float(q1_dt.abs().max().detach().cpu()) * maximum_abs_omega_rad_s >= math.pi:
        raise ValueError("q1 layout can alias the allowed yaw-rate range")
    rule_tau = tau[:, :rule_queries].float()
    fitted_center = center0[:, None] + rule_tau[:, :, None] * velocity[:, None]
    center_residual = torch.linalg.vector_norm(
        fitted_center - center[:, :rule_queries], dim=-1
    ).amax(dim=1)
    angle = omega[:, None] * rule_tau
    cosine, sine = torch.cos(angle), torch.sin(angle)
    fitted_phase = torch.stack((
        phase0[:, None, 0] * cosine - phase0[:, None, 1] * sine,
        phase0[:, None, 1] * cosine + phase0[:, None, 0] * sine,
    ), dim=-1)
    actual = phase[:, :rule_queries]
    yaw_residual = torch.atan2(
        fitted_phase[..., 0] * actual[..., 1] - fitted_phase[..., 1] * actual[..., 0],
        fitted_phase[..., 0] * actual[..., 0] + fitted_phase[..., 1] * actual[..., 1],
    ).abs().amax(dim=1)
    eligible = (
        (center_residual <= center_consistency_tolerance_m)
        & (yaw_residual <= yaw_consistency_tolerance_rad)
        & (omega.abs() <= maximum_abs_omega_rad_s + 1e-3)
    )
    return {
        "query_center": center.detach(),
        "query_phase": phase.detach(),
        "center0": center0.detach(),
        "velocity": velocity.detach(),
        "phase0": phase0.detach(),
        "omega": omega.detach(),
        "constant_motion": eligible.detach(),
        "center_consistency_residual_m": center_residual.detach(),
        "yaw_consistency_residual_rad": yaw_residual.detach(),
        "geometry_c4_asymmetry_m": torch.as_tensor(
            minimum_c4_asymmetry, dtype=torch.float32, device=position.device
        ),
    }


def decoded_trajectory_state(
    output: dict[str, torch.Tensor], tau: torch.Tensor, geometry: torch.Tensor,
    *, rule_queries: int | None = None,
) -> dict[str, torch.Tensor]:
    """Re-parse decoded positions, then apply the same extractor to A or B."""
    center, phase = _query_pose_from_fixed_truth(output["position_mean"], geometry)
    center0, velocity, phase0, omega = _trajectory_state(
        center, phase, tau,
        rule_queries=rule_queries,
    )
    return {
        "query_center": center,
        "query_phase": phase,
        "center0": center0,
        "velocity": velocity,
        "phase0": phase0,
        "omega": omega,
    }


def geometry_c4_asymmetry_m(geometry: torch.Tensor) -> float:
    """Minimum unordered-set distance to a non-trivial 90-degree rotation."""
    if tuple(geometry.shape) != (4, 3):
        raise ValueError("geometry must have shape [4,3]")
    with torch.autocast(device_type=geometry.device.type, enabled=False):
        geometry = geometry.float()
        values = []
        for quarter_turns in (1, 2, 3):
            angle = quarter_turns * math.pi / 2.0
            cosine, sine = math.cos(angle), math.sin(angle)
            rotated = geometry.clone()
            rotated[:, 0] = cosine * geometry[:, 0] - sine * geometry[:, 1]
            rotated[:, 1] = sine * geometry[:, 0] + cosine * geometry[:, 1]
            pair = torch.linalg.vector_norm(
                rotated[:, None, :] - geometry[None, :, :], dim=-1
            )
            symmetric = 0.5 * (
                pair.amin(dim=1).mean() + pair.amin(dim=0).mean()
            )
            values.append(float(symmetric.detach().cpu()))
    return min(values)
