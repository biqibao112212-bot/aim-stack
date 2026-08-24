"""Laplace EPro-PnP MVP loss; target pose is consumed only on the loss side."""

from __future__ import annotations

import torch

from .gpu_pnp import GpuPnPResult, project_points


def _cost(rotation: torch.Tensor, translation: torch.Tensor, image_points: torch.Tensor,
          object_points: torch.Tensor, intrinsics: torch.Tensor, weights: torch.Tensor,
          covariance: torch.Tensor) -> torch.Tensor:
    projected, camera = project_points(rotation, translation, object_points, intrinsics)
    residual = image_points - projected
    identity = torch.eye(2, dtype=covariance.dtype, device=covariance.device)
    scaled = covariance / weights.clamp_min(1.0e-6)[..., None, None]
    cholesky, info = torch.linalg.cholesky_ex(scaled + 1.0e-5 * identity)
    safe = torch.where((info == 0)[..., None, None], cholesky, identity)
    white = torch.linalg.solve_triangular(safe, residual[..., None], upper=False).squeeze(-1)
    result = white.square().sum(dim=(1, 2))
    valid = (info == 0).all(dim=1) & (camera[..., 2] > 1.0e-5).all(dim=1) & torch.isfinite(result)
    return torch.where(valid, result, torch.full_like(result, torch.inf))


def laplace_epro_nll(pnp: GpuPnPResult, image_points: torch.Tensor, object_points: torch.Tensor,
                     intrinsics: torch.Tensor, target_rotation: torch.Tensor,
                     target_translation: torch.Tensor, *, weights: torch.Tensor | None = None,
                     covariance: torch.Tensor | None = None) -> torch.Tensor:
    """Approximate EPro partition with online MAP modes and Hessian covariance.

    The online solver output is already fixed before target pose enters this
    function.  A future V20 step may replace this Laplace normalizer with AMIS;
    this MVP already trains a normalized pose distribution rather than a raw
    coordinate-only regressor.
    """
    batch, count = image_points.shape[:2]
    if weights is None:
        weights = torch.ones((batch, count), dtype=image_points.dtype, device=image_points.device)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6) * count
    if covariance is None:
        identity2 = torch.eye(2, dtype=image_points.dtype, device=image_points.device)
        covariance = identity2.expand(batch, count, 2, 2)
    target_cost = _cost(target_rotation, target_translation, image_points, object_points,
                        intrinsics, weights, covariance)
    covariance_local = 0.5 * (pnp.covariance_local + pnp.covariance_local.transpose(-1, -2))
    identity6 = torch.eye(6, dtype=image_points.dtype, device=image_points.device)
    cholesky, info = torch.linalg.cholesky_ex(covariance_local + 1.0e-6 * identity6)
    log_sqrt_det = torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1).clamp_min(1.0e-12)).sum(dim=-1).clamp(-20.0, 20.0)
    mode_valid = pnp.valid & (info == 0) & torch.isfinite(pnp.objective)
    minimum_objective = torch.where(mode_valid, pnp.objective, torch.full_like(pnp.objective, torch.inf)).amin(dim=1)
    relative_objective = pnp.objective - minimum_objective[:, None]
    log_partition_modes = torch.where(
        mode_valid, -0.5 * relative_objective + log_sqrt_det,
        torch.full_like(pnp.objective, -torch.inf),
    )
    log_partition = torch.logsumexp(log_partition_modes, dim=1)
    # If the labelled pose has lower energy than every returned online mode,
    # MAP search has not found a trustworthy local optimum.  Such early dense
    # samples train support/UV only; they must not create a fake EPro gradient.
    solver_consistent = target_cost + 1.0e-3 >= minimum_objective
    valid = (torch.isfinite(target_cost) & torch.isfinite(log_partition)
             & mode_valid.any(dim=1) & solver_consistent)
    # Correspondence-count normalization keeps the same probabilistic energy
    # scale for four sparse anchors and 64 dense points.
    per_sample = (0.5 * (target_cost - minimum_objective) + log_partition) / float(count)
    if not valid.any():
        return image_points.sum() * 0.0
    return per_sample[valid].mean()
