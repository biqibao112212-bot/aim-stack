"""Identity-free set metrics for PnP-history physical prediction."""

from __future__ import annotations

import torch


SET_POLICY = (
    "unordered four-armor sets; symmetric nearest-set distance; "
    "no truth slot, no permutation enumeration, no per-query identity"
)


def pnp_state_set_batch_errors(
    prediction: torch.Tensor, target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must both have shape [B,Q,4,3]")
    if prediction.shape[2:] != (4, 3):
        raise ValueError("PnP state set metrics require four xyz points")
    prediction = prediction.float()
    target = target.float()
    pair = torch.linalg.vector_norm(
        prediction[:, :, :, None, :] - target[:, :, None, :, :], dim=-1
    )
    predicted_to_target = pair.amin(dim=-1)
    target_to_predicted = pair.amin(dim=-2)
    symmetric = 0.5 * (
        predicted_to_target.mean(dim=-1) + target_to_predicted.mean(dim=-1)
    )
    hausdorff = torch.maximum(
        predicted_to_target.amax(dim=-1), target_to_predicted.amax(dim=-1)
    )
    predicted_centroid = prediction.mean(dim=2)
    target_centroid = target.mean(dim=2)
    centroid = torch.linalg.vector_norm(predicted_centroid - target_centroid, dim=-1)
    centroid_motion = torch.linalg.vector_norm(
        (predicted_centroid[:, 1:] - predicted_centroid[:, :1])
        - (target_centroid[:, 1:] - target_centroid[:, :1]),
        dim=-1,
    )
    pair_i, pair_j = torch.triu_indices(4, 4, offset=1, device=prediction.device)
    predicted_lengths = torch.linalg.vector_norm(
        prediction[:, :, pair_i] - prediction[:, :, pair_j], dim=-1
    ).sort(dim=-1).values
    target_lengths = torch.linalg.vector_norm(
        target[:, :, pair_i] - target[:, :, pair_j], dim=-1
    ).sort(dim=-1).values
    shape = torch.abs(predicted_lengths - target_lengths).mean(dim=-1)
    return {
        "state_q0_m": symmetric[:, 0],
        "absolute_set_m": symmetric,
        "hausdorff_set_m": hausdorff,
        "centroid_m": centroid,
        "centroid_motion_m": centroid_motion,
        "rigid_shape_m": shape,
    }
