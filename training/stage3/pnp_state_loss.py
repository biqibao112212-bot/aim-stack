"""Common identity-free set objective for the PnP explicit/implicit A/B."""

from __future__ import annotations

import torch
import torch.nn.functional as F


from .pnp_state_metrics import SET_POLICY


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
