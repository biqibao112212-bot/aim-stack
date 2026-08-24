"""Offline-only supervised losses for sparse and dense truth-free inference heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from training.corner_pnp.pnp import NOMINAL_OBJECT_POINTS_M

from .dense_correspondence_head import stratified_correspondences
from .gpu_pnp import (
    GpuPnPResult,
    observable_initialization_from_result,
    solve_weighted_planar_pnp,
)
from .epro_loss import laplace_epro_nll
from .labels import dense_surface_labels, gaussian_corner_heatmaps
from .sparse_prob_head import ProbabilisticCornerNet, SparsePrediction


@dataclass
class LossOutput:
    total: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
    pnp: GpuPnPResult


def _gaussian_nll(target: torch.Tensor, mean: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(2, dtype=target.dtype, device=target.device)
    covariance = covariance + 1.0e-5 * identity
    cholesky, info = torch.linalg.cholesky_ex(covariance)
    if torch.any(info != 0):
        raise FloatingPointError("predicted covariance is not SPD")
    residual = target - mean
    white = torch.linalg.solve_triangular(cholesky, residual[..., None], upper=False).squeeze(-1)
    return 0.5 * white.square().sum(dim=-1) + torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)).sum(dim=-1)


def _physical_errors(pnp: GpuPnPResult, reference_translation: torch.Tensor) -> tuple[torch.Tensor, ...]:
    estimate = pnp.translation_m[:, 0]
    delta = estimate - reference_translation
    z = reference_translation[:, 2].clamp_min(1.0e-4)
    position_mm = 1.0e3 * torch.linalg.vector_norm(delta, dim=1)
    depth_mm = 1.0e3 * delta[:, 2].abs()
    ray_mrad = 1.0e3 * torch.linalg.vector_norm(delta[:, :2] / z[:, None], dim=1)
    return position_mm, depth_mm, ray_mrad


def _cvar95(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    selected = value[valid]
    if selected.numel() == 0:
        return value.sum() * 0.0
    count = max(1, int((selected.numel() + 9) // 10))
    return selected.topk(count).values.mean()


def sparse_loss(model: ProbabilisticCornerNet, batch: dict[str, object]) -> LossOutput:
    online = batch["online"]
    target = batch["supervision"]
    assert isinstance(online, dict) and isinstance(target, dict)
    prediction: SparsePrediction = model(
        online["patch_rgb"], online["detector_geometry"], online["raw_corners_px"],
        online["raw_corners_patch_px"], online["patch_to_image_h"], online["raw_scale_px"],
    )
    target_patch = target["target_corners_patch_px"]
    in_context = (
        (target_patch[..., 0] >= -0.5) & (target_patch[..., 0] <= 127.5)
        & (target_patch[..., 1] >= -0.5) & (target_patch[..., 1] <= 63.5)
    )
    target_heatmap = gaussian_corner_heatmaps(target_patch.clamp(target_patch.new_tensor([0.0, 0.0]),
                                                                 target_patch.new_tensor([127.0, 63.0])))
    grid_kl_per_corner = F.kl_div(
        F.log_softmax(prediction.heatmap_logits.flatten(2), dim=-1), target_heatmap.flatten(2),
        reduction="none",
    ).sum(dim=-1)
    grid_kl = (grid_kl_per_corner * in_context).sum() / in_context.sum().clamp_min(1)
    context_bce = F.binary_cross_entropy_with_logits(prediction.visibility_logits, in_context.to(prediction.visibility_logits.dtype))
    tail_nll = _gaussian_nll(target["target_corners_px"], prediction.tail_image_mean,
                             prediction.tail_covariance).mean()
    corner_nll = _gaussian_nll(target["target_corners_px"], prediction.image_mean,
                               prediction.image_covariance).mean()
    object_points = torch.as_tensor(NOMINAL_OBJECT_POINTS_M, dtype=prediction.image_mean.dtype,
                                    device=prediction.image_mean.device)[None].expand(prediction.image_mean.shape[0], -1, -1)
    pnp = solve_weighted_planar_pnp(
        prediction.image_mean, object_points, online["intrinsics"], covariance=prediction.image_covariance,
    )
    valid = pnp.valid[:, 0]
    position, depth, ray = _physical_errors(pnp, target["reference_translation_m"])
    physical = (_cvar95(position / 10.0, valid) + _cvar95(depth / 10.0, valid)
                + _cvar95(ray, valid))
    epro_nll = laplace_epro_nll(
        pnp, prediction.image_mean, object_points, online["intrinsics"],
        target["reference_rotation"], target["reference_translation_m"],
        covariance=prediction.image_covariance,
    )
    invalid_penalty = (~valid).to(prediction.image_mean.dtype).mean()
    corner_rms = torch.sqrt((prediction.image_mean - target["target_corners_px"]).square().mean())
    total = (grid_kl + 0.25 * context_bce + 0.5 * tail_nll + corner_nll
             + 0.01 * epro_nll + 0.001 * physical + 0.1 * invalid_penalty)
    return LossOutput(total, {
        "grid_kl": grid_kl, "context_bce": context_bce, "tail_nll": tail_nll,
        "corner_nll": corner_nll, "corner_rms_px": corner_rms, "epro_nll": epro_nll,
        "physical_cvar": physical, "invalid_fraction": invalid_penalty,
        "position_mm_mean": position[valid].mean() if valid.any() else position.sum() * 0.0,
        "depth_mm_mean": depth[valid].mean() if valid.any() else depth.sum() * 0.0,
        "ray_mrad_mean": ray[valid].mean() if valid.any() else ray.sum() * 0.0,
    }, pnp)


def dense_loss(model: torch.nn.Module, batch: dict[str, object], *, correspondence_count: int = 64,
               pose_weight: float = 0.0) -> LossOutput:
    online = batch["online"]
    target = batch["supervision"]
    assert isinstance(online, dict) and isinstance(target, dict)
    prediction = model(online["patch_rgb"], online["detector_geometry"], online["raw_corners_patch_px"])
    support, target_uv, target_edge = dense_surface_labels(target["target_corners_patch_px"])
    support_bce = F.binary_cross_entropy_with_logits(prediction.support_logits, support)
    probability = torch.sigmoid(prediction.support_logits)
    dice = 1.0 - (2.0 * (probability * support).sum() + 1.0) / (probability.sum() + support.sum() + 1.0)
    uv_residual = prediction.canonical_uv - target_uv
    uv_nll_map = 0.5 * torch.exp(-prediction.log_variance) * uv_residual.square().sum(dim=1, keepdim=True) + prediction.log_variance
    uv_nll = (uv_nll_map * support).sum() / support.sum().clamp_min(1.0)
    uv_l1 = (F.smooth_l1_loss(prediction.canonical_uv, target_uv, reduction="none", beta=0.05).sum(dim=1, keepdim=True)
             * support).sum() / support.sum().clamp_min(1.0)
    uv_rmse = torch.sqrt((uv_residual.square().sum(dim=1, keepdim=True) * support).sum()
                         / support.sum().clamp_min(1.0))
    predicted_support = probability >= 0.5
    target_support = support > 0.5
    support_iou = ((predicted_support & target_support).sum().to(support.dtype)
                   / (predicted_support | target_support).sum().clamp_min(1).to(support.dtype))
    edge = (F.smooth_l1_loss(prediction.edge_distance, target_edge, reduction="none") * support).sum() / support.sum().clamp_min(1.0)
    corner_scale = target["target_corners_patch_px"].new_tensor([128.0, 64.0])
    final_corner = F.smooth_l1_loss(
        prediction.predicted_corners_patch / corner_scale,
        target["target_corners_patch_px"] / corner_scale,
        beta=0.02,
    )
    zero = final_corner * 0.0
    corner_heatmap_kl = zero
    corner_context_bce = zero
    corner_local = zero
    corner_tail = zero
    if prediction.corner_heatmap_logits is not None:
        if (prediction.local_corners_patch is None or prediction.tail_corners_patch is None
                or prediction.in_context_logits is None):
            raise RuntimeError("projective dense head omitted a corner-mixture output")
        target_patch = target["target_corners_patch_px"]
        in_context = (
            (target_patch[..., 0] >= -0.5) & (target_patch[..., 0] <= 127.5)
            & (target_patch[..., 1] >= -0.5) & (target_patch[..., 1] <= 63.5)
        )
        clamped_target = target_patch.clamp(
            target_patch.new_tensor([0.0, 0.0]), target_patch.new_tensor([127.0, 63.0]),
        )
        target_heatmap = gaussian_corner_heatmaps(clamped_target)
        heatmap_kl_per_corner = F.kl_div(
            F.log_softmax(prediction.corner_heatmap_logits.flatten(2), dim=-1),
            target_heatmap.flatten(2), reduction="none",
        ).sum(dim=-1)
        corner_heatmap_kl = (
            (heatmap_kl_per_corner * in_context).sum() / in_context.sum().clamp_min(1)
        )
        corner_context_bce = F.binary_cross_entropy_with_logits(
            prediction.in_context_logits, in_context.to(prediction.in_context_logits.dtype),
        )
        local_per_corner = F.smooth_l1_loss(
            prediction.local_corners_patch / corner_scale,
            target_patch / corner_scale, reduction="none", beta=0.02,
        ).sum(dim=-1)
        corner_local = (local_per_corner * in_context).sum() / in_context.sum().clamp_min(1)
        corner_tail = F.smooth_l1_loss(
            prediction.tail_corners_patch / corner_scale,
            target_patch / corner_scale, beta=0.02,
        )
    correspondences = stratified_correspondences(prediction, online["patch_to_image_h"], count=correspondence_count)
    initialization = None
    if model.config.get("architecture") == "spatial_projective_v2":
        batch = correspondences.image_points.shape[0]
        raw_object_points = torch.as_tensor(
            NOMINAL_OBJECT_POINTS_M,
            dtype=correspondences.image_points.dtype,
            device=correspondences.image_points.device,
        )[None].expand(batch, -1, -1)
        raw_pnp = solve_weighted_planar_pnp(
            online["raw_corners_px"], raw_object_points, online["intrinsics"],
        )
        initialization = observable_initialization_from_result(raw_pnp)
    pnp = solve_weighted_planar_pnp(
        correspondences.image_points, correspondences.object_points, online["intrinsics"],
        weights=correspondences.weights, initialization=initialization,
    )
    valid = pnp.valid[:, 0]
    position, depth, ray = _physical_errors(pnp, target["reference_translation_m"])
    physical = (_cvar95(position / 10.0, valid) + _cvar95(depth / 10.0, valid)
                + _cvar95(ray, valid))
    identity2 = torch.eye(2, dtype=prediction.support_logits.dtype, device=prediction.support_logits.device)
    dense_covariance = identity2.expand(*correspondences.image_points.shape[:2], 2, 2)
    epro_nll = laplace_epro_nll(
        pnp, correspondences.image_points, correspondences.object_points, online["intrinsics"],
        target["reference_rotation"], target["reference_translation_m"],
        weights=correspondences.weights, covariance=dense_covariance,
    )
    invalid_penalty = (~valid).to(prediction.support_logits.dtype).mean()
    total = (support_bce + dice + 0.5 * uv_nll + 5.0 * uv_l1 + 5.0 * final_corner + 0.25 * edge
             + corner_heatmap_kl + 0.25 * corner_context_bce + 0.5 * corner_local + 0.5 * corner_tail
             + pose_weight * 0.01 * epro_nll + pose_weight * 0.001 * physical
             + pose_weight * 0.1 * invalid_penalty)
    return LossOutput(total, {
        "support_bce": support_bce, "support_dice_loss": dice, "support_iou": support_iou,
        "uv_nll": uv_nll, "uv_l1": uv_l1, "uv_rmse": uv_rmse,
        "homography_corner_loss": final_corner, "corner_final_loss": final_corner,
        "corner_heatmap_kl": corner_heatmap_kl, "corner_context_bce": corner_context_bce,
        "corner_local_loss": corner_local, "corner_tail_loss": corner_tail,
        "edge_loss": edge, "epro_nll": epro_nll, "physical_cvar": physical, "invalid_fraction": invalid_penalty,
        "position_mm_mean": position[valid].mean() if valid.any() else position.sum() * 0.0,
        "depth_mm_mean": depth[valid].mean() if valid.any() else depth.sum() * 0.0,
        "ray_mrad_mean": ray[valid].mean() if valid.any() else ray.sum() * 0.0,
    }, pnp)
