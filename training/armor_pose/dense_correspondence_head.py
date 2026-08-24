"""Dense canonical-surface prediction and spatially covered correspondence sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import ArmorPoseBackbone
from .labels import canonical_uv_to_object_points, map_points_homography, patch_grid


@dataclass
class DensePrediction:
    support_logits: torch.Tensor
    canonical_uv: torch.Tensor
    log_variance: torch.Tensor
    edge_distance: torch.Tensor


@dataclass
class DenseCorrespondences:
    image_points: torch.Tensor
    object_points: torch.Tensor
    weights: torch.Tensor
    patch_points: torch.Tensor


class DenseCorrespondenceNet(nn.Module):
    family = "same-frame-dense-canonical-armor-surface-v1"

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ArmorPoseBackbone()
        channels = self.backbone.output_channels
        self.support = nn.Conv2d(channels, 1, 1)
        self.uv = nn.Conv2d(channels, 2, 1)
        self.log_variance = nn.Conv2d(channels, 1, 1)
        self.edge = nn.Conv2d(channels, 1, 1)

    def forward(self, patch: torch.Tensor, geometry: torch.Tensor) -> DensePrediction:
        feature = self.backbone(patch, geometry)
        return DensePrediction(
            support_logits=self.support(feature),
            canonical_uv=torch.tanh(self.uv(feature)),
            log_variance=self.log_variance(feature).clamp(-6.0, 6.0),
            edge_distance=torch.sigmoid(self.edge(feature)),
        )

    @property
    def config(self) -> dict[str, object]:
        return {
            "family": self.family,
            "online_inputs": ["patch", "detector_geometry"],
            "online_truth_input": False,
            "temporal_input": False,
        }


def stratified_correspondences(prediction: DensePrediction, inverse_transform: torch.Tensor,
                               *, count: int = 64) -> DenseCorrespondences:
    """Pick one maximum-confidence point per fixed cell; no truth is consulted."""
    if count not in {32, 64, 128}:
        raise ValueError("count must be one of the predeclared 32/64/128 values")
    batch, _, height, width = prediction.support_logits.shape
    rows = 4 if count <= 64 else 8
    columns = count // rows
    if height % rows or width % columns:
        raise ValueError("predeclared grid does not tile the ROI")
    cell_h, cell_w = height // rows, width // columns
    score = F.logsigmoid(prediction.support_logits) - 0.5 * prediction.log_variance
    score_cells = score.reshape(batch, 1, rows, cell_h, columns, cell_w).permute(0, 2, 4, 1, 3, 5)
    flat_index = score_cells.reshape(batch, count, -1).argmax(dim=-1)
    local_y = torch.div(flat_index, cell_w, rounding_mode="floor")
    local_x = flat_index.remainder(cell_w)
    cell_y = torch.arange(rows, device=score.device).repeat_interleave(columns)[None]
    cell_x = torch.arange(columns, device=score.device).repeat(rows)[None]
    y = cell_y * cell_h + local_y
    x = cell_x * cell_w + local_x
    linear = y * width + x
    uv = prediction.canonical_uv.flatten(2).transpose(1, 2).gather(
        1, linear[..., None].expand(-1, -1, 2)
    )
    raw_weight = (torch.sigmoid(prediction.support_logits) * torch.exp(-prediction.log_variance)).flatten(2).squeeze(1)
    weights = raw_weight.gather(1, linear).clamp(1.0e-4, 1.0e4)
    grid = patch_grid(device=score.device, dtype=score.dtype).reshape(-1, 2)
    patch_points = grid[linear]
    image_points = map_points_homography(patch_points, inverse_transform)
    return DenseCorrespondences(
        image_points=image_points,
        object_points=canonical_uv_to_object_points(uv),
        weights=weights,
        patch_points=patch_points,
    )
