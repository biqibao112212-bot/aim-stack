"""CUDA-friendly supervision and coordinate transforms for armor-pose research."""

from __future__ import annotations

import torch


PATCH_HEIGHT = 64
PATCH_WIDTH = 128

# Canonical image-facing convention.  The production object axes are reversed
# relative to this UV convention; `canonical_uv_to_object_points` owns that
# conversion explicitly so nobody silently changes corner order.
CANONICAL_CORNER_UV = torch.tensor(
    [[-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0]],
    dtype=torch.float32,
)


def patch_grid(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return `[H,W,2]` pixel centers in patch x/y coordinates."""
    y, x = torch.meshgrid(
        torch.arange(PATCH_HEIGHT, device=device, dtype=dtype),
        torch.arange(PATCH_WIDTH, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x, y), dim=-1)


def map_points_homography(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply batched homographies to `[B,N,2]` points."""
    if points.ndim != 3 or points.shape[-1] != 2 or transform.shape != (points.shape[0], 3, 3):
        raise ValueError("homography input contract changed")
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    mapped = homogeneous @ transform.transpose(1, 2)
    denominator = mapped[..., 2:3]
    safe = torch.where(denominator.abs() >= 1.0e-8, denominator, denominator.sign() * 1.0e-8)
    safe = torch.where(safe == 0, torch.full_like(safe, 1.0e-8), safe)
    return mapped[..., :2] / safe


def gaussian_corner_heatmaps(target_patch: torch.Tensor, *, sigma_px: float = 1.5) -> torch.Tensor:
    """Generate normalized `[B,4,H,W]` heatmaps entirely on the input device."""
    if target_patch.ndim != 3 or target_patch.shape[1:] != (4, 2) or sigma_px <= 0:
        raise ValueError("target_patch must be [B,4,2] and sigma_px positive")
    grid = patch_grid(device=target_patch.device, dtype=target_patch.dtype)
    residual = grid[None, None] - target_patch[:, :, None, None]
    logits = -0.5 * residual.square().sum(dim=-1) / (sigma_px * sigma_px)
    probability = torch.softmax(logits.flatten(2), dim=-1).reshape_as(logits)
    return probability


def heatmap_moments(logits: torch.Tensor, *, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean `[B,K,2]`, covariance `[B,K,2,2]`, and entropy `[B,K]`."""
    if logits.ndim != 4 or logits.shape[-2:] != (PATCH_HEIGHT, PATCH_WIDTH) or temperature <= 0:
        raise ValueError("heatmap logits must be [B,K,64,128]")
    probability = torch.softmax((logits / temperature).flatten(2), dim=-1)
    grid = patch_grid(device=logits.device, dtype=logits.dtype).reshape(-1, 2)
    mean = probability @ grid
    centered = grid[None, None] - mean[:, :, None]
    covariance = torch.einsum("bkn,bkni,bknj->bkij", probability, centered, centered)
    identity = torch.eye(2, dtype=logits.dtype, device=logits.device)
    covariance = covariance + 1.0e-4 * identity
    entropy = -(probability * probability.clamp_min(1.0e-12).log()).sum(dim=-1)
    return mean, covariance, entropy


def calibrated_patch_grid_moments(
    residual_logits: torch.Tensor, raw_patch: torch.Tensor, *, prior_sigma_px: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return calibrated patch-space moments for a raw-corner prior.

    The returned logits include the raw Gaussian prior.  The coordinate grid
    used for the moments is translated by the (possibly boundary-truncated)
    prior's bias, so identically zero residual logits preserve ``raw_patch``
    exactly, including corners close to or outside the 128x64 context.
    """
    if residual_logits.ndim != 4 or residual_logits.shape[1:] != (4, PATCH_HEIGHT, PATCH_WIDTH):
        raise ValueError("residual_logits must be [B,4,64,128]")
    if raw_patch.shape != (residual_logits.shape[0], 4, 2) or prior_sigma_px <= 0.0:
        raise ValueError("raw_patch must be [B,4,2] and prior_sigma_px positive")
    grid = patch_grid(device=residual_logits.device, dtype=residual_logits.dtype).reshape(-1, 2)
    prior_residual = grid[None, None] - raw_patch[:, :, None]
    prior_logits = -0.5 * prior_residual.square().sum(dim=-1) / (prior_sigma_px * prior_sigma_px)
    combined_logits = prior_logits + residual_logits.flatten(2)
    prior_probability = torch.softmax(prior_logits, dim=-1)
    probability = torch.softmax(combined_logits, dim=-1)
    prior_mean = prior_probability @ grid
    calibrated_grid = grid[None, None] + (raw_patch - prior_mean)[:, :, None]
    mean = torch.einsum("bkn,bkni->bki", probability, calibrated_grid)
    centered = calibrated_grid - mean[:, :, None]
    covariance = torch.einsum("bkn,bkni,bknj->bkij", probability, centered, centered)
    identity = torch.eye(2, dtype=residual_logits.dtype, device=residual_logits.device)
    covariance = covariance + 1.0e-4 * identity
    entropy = -(probability * probability.clamp_min(1.0e-12).log()).sum(dim=-1)
    return mean, covariance, entropy, combined_logits.reshape_as(residual_logits)


def calibrated_grid_moments(residual_logits: torch.Tensor, raw_patch: torch.Tensor,
                            raw_full: torch.Tensor, inverse_transform: torch.Tensor,
                            *, prior_sigma_px: float = 2.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full-image moments whose zero residual preserves every raw corner exactly."""
    if residual_logits.ndim != 4 or residual_logits.shape[1:] != (4, PATCH_HEIGHT, PATCH_WIDTH):
        raise ValueError("residual_logits must be [B,4,64,128]")
    if raw_patch.shape != (residual_logits.shape[0], 4, 2) or raw_full.shape != raw_patch.shape:
        raise ValueError("raw corner contract changed")
    grid_patch = patch_grid(device=residual_logits.device, dtype=residual_logits.dtype).reshape(-1, 2)
    mapped_grid = map_points_homography(
        grid_patch[None].expand(residual_logits.shape[0], -1, -1), inverse_transform
    )
    prior_residual = grid_patch[None, None] - raw_patch[:, :, None]
    prior_logits = -0.5 * prior_residual.square().sum(dim=-1) / (prior_sigma_px * prior_sigma_px)
    prior_probability = torch.softmax(prior_logits, dim=-1)
    probability = torch.softmax(prior_logits + residual_logits.flatten(2), dim=-1)
    prior_mean = prior_probability @ mapped_grid
    calibrated_grid = mapped_grid[:, None] + (raw_full - prior_mean)[:, :, None]
    mean = torch.einsum("bkn,bkni->bki", probability, calibrated_grid)
    centered = calibrated_grid - mean[:, :, None]
    covariance = torch.einsum("bkn,bkni,bknj->bkij", probability, centered, centered)
    identity = torch.eye(2, dtype=residual_logits.dtype, device=residual_logits.device)
    covariance = covariance + 1.0e-4 * identity
    entropy = -(probability * probability.clamp_min(1.0e-12).log()).sum(dim=-1)
    return mean, covariance, entropy


def pixel_to_uv_homography(target_patch: torch.Tensor) -> torch.Tensor:
    """Solve target-patch pixel -> canonical UV homographies by batched DLT."""
    if target_patch.ndim != 3 or target_patch.shape[1:] != (4, 2):
        raise ValueError("target_patch must be [B,4,2]")
    batch = target_patch.shape[0]
    xy = target_patch
    uv = CANONICAL_CORNER_UV.to(device=xy.device, dtype=xy.dtype).expand(batch, -1, -1)
    x, y = xy.unbind(dim=-1)
    u, v = uv.unbind(dim=-1)
    ones, zeros = torch.ones_like(x), torch.zeros_like(x)
    row_u = torch.stack((x, y, ones, zeros, zeros, zeros, -u * x, -u * y), dim=-1)
    row_v = torch.stack((zeros, zeros, zeros, x, y, ones, -v * x, -v * y), dim=-1)
    matrix = torch.stack((row_u, row_v), dim=2).reshape(batch, 8, 8)
    right = torch.stack((u, v), dim=2).reshape(batch, 8, 1)
    identity = torch.eye(8, dtype=xy.dtype, device=xy.device).expand(batch, -1, -1)
    solution = torch.linalg.solve(matrix.transpose(1, 2) @ matrix + 1.0e-6 * identity,
                                  matrix.transpose(1, 2) @ right).squeeze(-1)
    last = torch.ones((batch, 1), dtype=xy.dtype, device=xy.device)
    return torch.cat((solution, last), dim=1).reshape(batch, 3, 3)


def dense_surface_labels(target_patch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return nominal projected-support mask, UV, and interior edge distance.

    This is an amodal planar support derived from the nominal target quad.  It
    is deliberately not called an occlusion-tested visible foreground mask.
    """
    transform = pixel_to_uv_homography(target_patch)
    grid = patch_grid(device=target_patch.device, dtype=target_patch.dtype)
    points = grid.reshape(1, -1, 2).expand(target_patch.shape[0], -1, -1)
    uv = map_points_homography(points, transform).reshape(target_patch.shape[0], PATCH_HEIGHT, PATCH_WIDTH, 2)
    margin = 1.0 - uv.abs()
    mask = (margin.amin(dim=-1) >= 0.0).to(dtype=target_patch.dtype)
    edge_distance = margin.amin(dim=-1).clamp(0.0, 1.0) * mask
    return mask[:, None], uv.permute(0, 3, 1, 2), edge_distance[:, None]


def canonical_uv_to_object_points(uv: torch.Tensor) -> torch.Tensor:
    """Map arbitrary `[...,2]` canonical UV to the fixed nominal armor plane."""
    if uv.shape[-1] != 2:
        raise ValueError("canonical UV must end in two coordinates")
    scale = torch.tensor([0.0675, 0.0275], dtype=uv.dtype, device=uv.device)
    xy = -uv * scale
    return torch.cat((xy, torch.zeros_like(xy[..., :1])), dim=-1)
