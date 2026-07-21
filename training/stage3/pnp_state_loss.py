"""Common identity-free set objective for the PnP explicit/implicit A/B."""

from __future__ import annotations

import torch
import torch.nn.functional as F


from .pnp_state_metrics import SET_POLICY
from .pnp_state_targets import decoded_trajectory_state, truth_trajectory_targets


def pnp_state_position_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    tau: torch.Tensor,
    *,
    huber_beta_m: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise both arms through the same unordered future physical sets."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must both have shape [B,Q,4,3]")
    if prediction.shape[2:] != (4, 3) or tau.shape != prediction.shape[:2]:
        raise ValueError("physical PnP tensors require [B,Q,4,3] and tau [B,Q]")
    if huber_beta_m <= 0:
        raise ValueError("huber_beta_m must be positive")
    if not torch.allclose(tau[:, 0], torch.zeros_like(tau[:, 0]), atol=0.0, rtol=0.0):
        raise ValueError("query zero must be exact tau=0")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("position loss refuses non-finite active values")

    prediction = prediction.float()
    target = target.float()
    pair = torch.linalg.vector_norm(
        prediction[:, :, :, None, :] - target[:, :, None, :, :], dim=-1
    )
    predicted_to_target = pair.amin(dim=-1)
    target_to_prediction = pair.amin(dim=-2)

    def robust_set(value: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(
            value, torch.zeros_like(value), beta=huber_beta_m, reduction="mean"
        )

    q0 = 0.5 * (
        robust_set(predicted_to_target[:, 0])
        + robust_set(target_to_prediction[:, 0])
    )
    absolute = 0.5 * (
        robust_set(predicted_to_target) + robust_set(target_to_prediction)
    )
    predicted_centroid = prediction.mean(dim=2)
    target_centroid = target.mean(dim=2)
    motion = F.smooth_l1_loss(
        predicted_centroid[:, 1:] - predicted_centroid[:, :1],
        target_centroid[:, 1:] - target_centroid[:, :1],
        beta=huber_beta_m,
        reduction="mean",
    )
    total = 2.0 * q0 + absolute + motion
    return total, {"q0": q0, "absolute": absolute, "motion": motion}


def pnp_state_constrained_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    tau: torch.Tensor,
    geometry: torch.Tensor,
    *,
    huber_beta_m: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Common trajectory-derived state supervision for both A and B.

    A receives no latent-only shortcut: velocity and omega are re-extracted
    from its decoded query trajectory using exactly the same operator as B.
    """
    truth = truth_trajectory_targets(target, tau, geometry)
    active = truth["constant_motion"]
    if not bool(active.any()):
        zero = output["position_mean"].float().sum() * 0.0
        return zero, {
            name: zero.detach() for name in (
                "q0", "absolute", "motion", "center", "center_delta",
                "velocity", "yaw_delta", "omega",
            )
        } | {
            "constant_motion_fraction": torch.zeros_like(zero.detach()),
        }
    query_count = output["position_mean"].shape[1]
    prediction = output["position_mean"][active, :query_count]
    active_target = target[active, :query_count]
    active_tau = tau[active, :query_count]
    base, base_parts = pnp_state_position_loss(
        prediction, active_target, active_tau, huber_beta_m=huber_beta_m
    )
    predicted_state = decoded_trajectory_state(output, tau, geometry)
    query_center = predicted_state["query_center"][active]
    target_center = truth["query_center"][active]

    def zero_huber(value_m: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(
            value_m, torch.zeros_like(value_m), beta=huber_beta_m, reduction="mean"
        )

    center = zero_huber(query_center - target_center)
    center_delta = zero_huber(
        (query_center[:, 1:] - query_center[:, :1])
        - (target_center[:, 1:] - target_center[:, :1])
    )
    reference_horizon_s = 0.5
    velocity = zero_huber(
        reference_horizon_s * (
            predicted_state["velocity"][active] - truth["velocity"][active]
        )
    )
    predicted_phase = predicted_state["query_phase"][active]
    target_phase = truth["query_phase"][active]
    predicted_relative = torch.stack((
        predicted_phase[:, :1, 0] * predicted_phase[..., 0]
        + predicted_phase[:, :1, 1] * predicted_phase[..., 1],
        predicted_phase[:, :1, 0] * predicted_phase[..., 1]
        - predicted_phase[:, :1, 1] * predicted_phase[..., 0],
    ), dim=-1)
    target_relative = torch.stack((
        target_phase[:, :1, 0] * target_phase[..., 0]
        + target_phase[:, :1, 1] * target_phase[..., 1],
        target_phase[:, :1, 0] * target_phase[..., 1]
        - target_phase[:, :1, 1] * target_phase[..., 0],
    ), dim=-1)
    geometry_radius_m = torch.sqrt(
        geometry[:, :2].float().square().sum(dim=-1).mean()
    )
    yaw_delta = zero_huber(
        geometry_radius_m * (predicted_relative[:, 1:] - target_relative[:, 1:])
    )
    omega = zero_huber(
        geometry_radius_m * reference_horizon_s * (
            predicted_state["omega"][active] - truth["omega"][active]
        )
    )
    active_fraction = active.float().mean()
    total = active_fraction * (
        base + center + 2.0 * center_delta + 4.0 * velocity
        + yaw_delta + 4.0 * omega
    )
    return total, {
        "q0": base_parts["q0"],
        "absolute": base_parts["absolute"],
        "motion": base_parts["motion"],
        "center": center,
        "center_delta": center_delta,
        "velocity": velocity,
        "yaw_delta": yaw_delta,
        "omega": omega,
        "constant_motion_fraction": active_fraction,
    }
