"""Pure metrics for Stage-3 physical-position predictors.

The only slot assignment contract in this module is selected from the four
physical truth plates at query zero.  That assignment is then held fixed for
every later query.  This prevents future truth from improving the apparent
current-state estimate and makes state and motion errors directly comparable.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch


PERMUTATIONS = tuple(itertools.permutations(range(4)))
PERMUTATION_POLICY = "q0-all4-truth-l2-lex-v1"


def q0_permutation(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align ``prediction`` once from q0 and return aligned values and gaps.

    Args:
        prediction: ``[B,Q,4,3]`` predicted positions.
        target: ``[B,Q,4,3]`` physical truth positions.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must both have shape [B,Q,4,3]")
    if prediction.shape[1] < 1 or prediction.shape[2:] != (4, 3):
        raise ValueError("physical tensors require at least q0 and four xyz slots")
    permutation = torch.tensor(PERMUTATIONS, dtype=torch.long, device=prediction.device)
    candidates = prediction[:, :, permutation, :].permute(0, 2, 1, 3, 4)
    q0_score = torch.linalg.vector_norm(
        candidates[:, :, 0] - target[:, None, 0], dim=-1
    ).mean(dim=-1)
    ordered, _ = q0_score.sort(dim=1)
    best_index = q0_score.argmin(dim=1)
    gather = best_index.view(-1, 1, 1, 1, 1).expand(
        -1, 1, prediction.shape[1], 4, 3
    )
    aligned = candidates.gather(1, gather).squeeze(1)
    gap = ordered[:, 1] - ordered[:, 0]
    return aligned, best_index, gap


def physical_batch_errors(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, torch.Tensor]:
    aligned, permutation_index, permutation_gap = q0_permutation(prediction, target)
    point = torch.linalg.vector_norm(aligned - target, dim=-1)
    absolute = point.mean(dim=-1)
    predicted_delta = aligned - aligned[:, :1]
    target_delta = target - target[:, :1]
    motion = torch.linalg.vector_norm(predicted_delta - target_delta, dim=-1).mean(dim=-1)
    centroid = torch.linalg.vector_norm(
        aligned[:, 0].mean(dim=1) - target[:, 0].mean(dim=1), dim=-1
    )
    predicted_centered = aligned[:, 0] - aligned[:, 0].mean(dim=1, keepdim=True)
    target_centered = target[:, 0] - target[:, 0].mean(dim=1, keepdim=True)
    centered_shape = torch.linalg.vector_norm(
        predicted_centered - target_centered, dim=-1
    ).mean(dim=-1)

    # Best planar rotation after removing translation.  The remaining error is
    # a true non-rigid/template residual and is zero for a translated/rotated
    # copy of the target four-plate geometry.
    dot = (
        predicted_centered[..., 0] * target_centered[..., 0]
        + predicted_centered[..., 1] * target_centered[..., 1]
    ).sum(dim=1)
    cross = (
        predicted_centered[..., 0] * target_centered[..., 1]
        - predicted_centered[..., 1] * target_centered[..., 0]
    ).sum(dim=1)
    yaw = torch.atan2(cross, dot)
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    x = cos_yaw[:, None] * predicted_centered[..., 0] - sin_yaw[:, None] * predicted_centered[..., 1]
    y = sin_yaw[:, None] * predicted_centered[..., 0] + cos_yaw[:, None] * predicted_centered[..., 1]
    rotated = torch.stack((x, y, predicted_centered[..., 2]), dim=-1)
    rigid_residual = torch.linalg.vector_norm(rotated - target_centered, dim=-1).mean(dim=-1)
    return {
        "aligned": aligned,
        "permutation_index": permutation_index,
        "permutation_gap_m": permutation_gap,
        "state_q0_m": absolute[:, 0],
        "absolute_pg_m": absolute,
        "motion_delta_m": motion,
        "centroid_q0_m": centroid,
        "centered_shape_q0_m": centered_shape,
        "rigid_residual_q0_m": rigid_residual,
        "yaw_alignment_rad": yaw,
    }


def summary(values: np.ndarray | list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return {"count": 0}
    if not np.all(np.isfinite(array)):
        raise ValueError("physical metric contains non-finite values")
    return {
        "count": int(len(array)),
        "mean_m": float(array.mean()),
        "median_m": float(np.quantile(array, 0.50)),
        "p90_m": float(np.quantile(array, 0.90)),
        "p95_m": float(np.quantile(array, 0.95)),
        "p99_m": float(np.quantile(array, 0.99)),
    }

