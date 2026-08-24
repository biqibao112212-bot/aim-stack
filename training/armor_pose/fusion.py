"""Truth-free correspondence-level fusion of sparse anchors and dense surface points."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from training.corner_pnp.pnp import NOMINAL_OBJECT_POINTS_M

from .dense_correspondence_head import DenseCorrespondenceNet, DensePrediction, stratified_correspondences
from .gpu_pnp import GpuPnPResult, solve_weighted_planar_pnp
from .sparse_prob_head import ProbabilisticCornerNet, SparsePrediction


@dataclass
class FusionPrediction:
    sparse: SparsePrediction
    dense: DensePrediction
    pnp: GpuPnPResult
    image_points: torch.Tensor
    object_points: torch.Tensor
    weights: torch.Tensor
    covariance: torch.Tensor


class SparseDensePoseNet(nn.Module):
    family = "same-frame-sparse-dense-correspondence-fusion-v1"

    def __init__(self, *, dense_count: int = 64) -> None:
        super().__init__()
        self.sparse = ProbabilisticCornerNet()
        self.dense = DenseCorrespondenceNet()
        self.dense_count = int(dense_count)

    def forward(self, patch: torch.Tensor, geometry: torch.Tensor, raw_full: torch.Tensor,
                raw_patch: torch.Tensor, inverse_transform: torch.Tensor, scale: torch.Tensor,
                intrinsics: torch.Tensor) -> FusionPrediction:
        sparse = self.sparse(patch, geometry, raw_full, raw_patch, inverse_transform, scale)
        dense = self.dense(patch, geometry)
        dense_set = stratified_correspondences(dense, inverse_transform, count=self.dense_count)
        batch = patch.shape[0]
        sparse_object = torch.as_tensor(NOMINAL_OBJECT_POINTS_M, dtype=patch.dtype, device=patch.device)[None].expand(batch, -1, -1)
        sparse_quality = torch.reciprocal(
            torch.diagonal(sparse.image_covariance, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(1.0e-4)
        )
        sparse_weight = sparse_quality / sparse_quality.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        dense_weight = dense_set.weights / dense_set.weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        weights = torch.cat((sparse_weight, dense_weight), dim=1)
        image_points = torch.cat((sparse.image_mean, dense_set.image_points), dim=1)
        object_points = torch.cat((sparse_object, dense_set.object_points), dim=1)
        identity2 = torch.eye(2, dtype=patch.dtype, device=patch.device)
        dense_covariance = identity2.expand(batch, self.dense_count, 2, 2)
        covariance = torch.cat((sparse.image_covariance, dense_covariance), dim=1)
        pnp = solve_weighted_planar_pnp(
            image_points, object_points, intrinsics, weights=weights, covariance=covariance,
        )
        return FusionPrediction(sparse, dense, pnp, image_points, object_points, weights, covariance)

    @property
    def config(self) -> dict[str, object]:
        return {
            "family": self.family, "dense_correspondence_count": self.dense_count,
            "fusion": "separately mass-normalized sparse and dense correspondence groups",
            "online_truth_input": False, "temporal_input": False,
        }
