"""Dense canonical-surface prediction and spatially covered correspondence sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import ArmorPoseBackbone
from .labels import (
    calibrated_patch_grid_moments,
    canonical_uv_to_object_points,
    dense_surface_labels,
    map_points_homography,
    patch_grid,
)


@dataclass
class DensePrediction:
    support_logits: torch.Tensor
    canonical_uv: torch.Tensor
    log_variance: torch.Tensor
    edge_distance: torch.Tensor
    predicted_corners_patch: torch.Tensor
    corner_heatmap_logits: torch.Tensor | None = None
    local_corners_patch: torch.Tensor | None = None
    local_corner_covariance: torch.Tensor | None = None
    local_corner_entropy: torch.Tensor | None = None
    tail_corners_patch: torch.Tensor | None = None
    in_context_logits: torch.Tensor | None = None
    in_context_probability: torch.Tensor | None = None


@dataclass
class DenseCorrespondences:
    image_points: torch.Tensor
    object_points: torch.Tensor
    weights: torch.Tensor
    patch_points: torch.Tensor


class SpatialBinCornerTail(nn.Module):
    """Continuous corner tail that preserves where evidence occurs in the ROI."""

    bins = (4, 8)

    def __init__(self, channels: int, *, hidden: int = 128) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(self.bins)
        self.mlp = nn.Sequential(
            nn.Flatten(), nn.Linear(channels * self.bins[0] * self.bins[1], hidden), nn.SiLU(),
            nn.Linear(hidden, 4 * 3),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pool(feature)).reshape(-1, 4, 3)


class LegacyDenseCorrespondenceNet(nn.Module):
    """Byte-compatible V19 global-average corner/UV head for old checkpoints."""

    family = "same-frame-dense-canonical-armor-surface-v1"

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ArmorPoseBackbone()
        channels = self.backbone.output_channels
        self.support = nn.Conv2d(channels, 1, 1)
        self.uv = nn.Conv2d(channels, 2, 1)
        self.log_variance = nn.Conv2d(channels, 1, 1)
        self.edge = nn.Conv2d(channels, 1, 1)
        self.corner_delta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 64), nn.SiLU(), nn.Linear(64, 8),
        )
        for layer in (self.support, self.uv, self.log_variance, self.edge):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.corner_delta[-1].weight)
        nn.init.zeros_(self.corner_delta[-1].bias)

    def forward(self, patch: torch.Tensor, geometry: torch.Tensor,
                raw_patch: torch.Tensor) -> DensePrediction:
        feature = self.backbone(patch, geometry)
        normalized_delta = 2.0 * torch.tanh(self.corner_delta(feature).reshape(-1, 4, 2))
        patch_scale = patch.new_tensor([128.0, 64.0])
        predicted_corners = raw_patch + normalized_delta * patch_scale
        raw_support, raw_uv, raw_edge = dense_surface_labels(predicted_corners)
        return DensePrediction(
            support_logits=torch.where(
                raw_support > 0.5, raw_support.new_tensor(4.0), raw_support.new_tensor(-4.0),
            ) + self.support(feature),
            canonical_uv=raw_uv + 0.10 * torch.tanh(self.uv(feature)),
            log_variance=self.log_variance(feature).clamp(-6.0, 6.0),
            edge_distance=(raw_edge + 0.25 * torch.tanh(self.edge(feature))).clamp(0.0, 1.0),
            predicted_corners_patch=predicted_corners,
        )

    @property
    def config(self) -> dict[str, object]:
        return {
            "family": self.family,
            "architecture": "legacy_global_average_v1",
            "online_inputs": ["patch", "detector_geometry", "raw_corners_patch"],
            "online_truth_input": False,
            "temporal_input": False,
        }


class DenseCorrespondenceNet(nn.Module):
    family = "same-frame-projective-dense-canonical-armor-surface-v2"
    tail_displacement_patch_multiples = 2.0
    nonprojective_uv_residual_enabled = False
    nonprojective_uv_residual_scale = 0.0

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ArmorPoseBackbone()
        channels = self.backbone.output_channels
        self.corner_heatmaps = nn.Conv2d(channels, 4, 1)
        self.support = nn.Conv2d(channels, 1, 1)
        self.log_variance = nn.Conv2d(channels, 1, 1)
        self.edge = nn.Conv2d(channels, 1, 1)
        self.corner_tail = SpatialBinCornerTail(channels)
        for layer in (self.corner_heatmaps, self.support, self.log_variance, self.edge):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, patch: torch.Tensor, geometry: torch.Tensor,
                raw_patch: torch.Tensor) -> DensePrediction:
        if raw_patch.shape != (patch.shape[0], 4, 2):
            raise ValueError("raw_patch must be [B,4,2]")
        feature = self.backbone(patch, geometry)
        residual_logits = self.corner_heatmaps(feature)
        local_corners, local_covariance, local_entropy, combined_logits = calibrated_patch_grid_moments(
            residual_logits, raw_patch,
        )
        tail = self.corner_tail(feature)
        patch_scale = patch.new_tensor([128.0, 64.0])
        tail_delta = self.tail_displacement_patch_multiples * torch.tanh(tail[..., :2]) * patch_scale
        tail_corners = raw_patch + tail_delta
        in_context_logits = tail[..., 2]
        in_context_probability = torch.sigmoid(in_context_logits)
        predicted_corners = (
            in_context_probability[..., None] * local_corners
            + (1.0 - in_context_probability)[..., None] * tail_corners
        )
        raw_support, projective_uv, raw_edge = dense_surface_labels(predicted_corners)
        return DensePrediction(
            support_logits=torch.where(raw_support > 0.5, raw_support.new_tensor(4.0), raw_support.new_tensor(-4.0))
            + self.support(feature),
            # V20's first projective trial deliberately has no free per-pixel
            # UV residual: every correspondence belongs to the single plane
            # induced by the four predicted corners.
            canonical_uv=projective_uv,
            log_variance=self.log_variance(feature).clamp(-6.0, 6.0),
            edge_distance=(raw_edge + 0.25 * torch.tanh(self.edge(feature))).clamp(0.0, 1.0),
            corner_heatmap_logits=combined_logits,
            local_corners_patch=local_corners,
            local_corner_covariance=local_covariance,
            local_corner_entropy=local_entropy,
            tail_corners_patch=tail_corners,
            in_context_logits=in_context_logits,
            in_context_probability=in_context_probability,
            predicted_corners_patch=predicted_corners,
        )

    @property
    def config(self) -> dict[str, object]:
        return {
            "family": self.family,
            "architecture": "spatial_projective_v2",
            "online_inputs": ["patch", "detector_geometry", "raw_corners_patch"],
            "online_truth_input": False,
            "temporal_input": False,
            "corner_parameterization": "raw-prior calibrated heatmap plus spatial-bin continuous tail mixture",
            "spatial_tail_bins": list(SpatialBinCornerTail.bins),
            "tail_displacement_patch_multiples": self.tail_displacement_patch_multiples,
            "canonical_uv_parameterization": "single homography induced by predicted corners",
            "nonprojective_uv_residual_enabled": self.nonprojective_uv_residual_enabled,
            "nonprojective_uv_residual_scale": self.nonprojective_uv_residual_scale,
            "zero_initialization_preserves_raw_patch": True,
        }


def build_dense_correspondence_net(
    *, architecture: str | None = None, model_config: dict[str, object] | None = None,
) -> nn.Module:
    """Reconstruct either a preserved V19 head or the projective V20 head."""
    configured_family = str(model_config.get("family")) if model_config is not None else None
    configured_architecture = (
        str(model_config.get("architecture"))
        if model_config is not None and model_config.get("architecture") else None
    )
    selected = architecture or configured_architecture
    if selected is None:
        if configured_family == LegacyDenseCorrespondenceNet.family:
            selected = "legacy_global_average_v1"
        elif configured_family == DenseCorrespondenceNet.family:
            selected = "spatial_projective_v2"
    if selected == "legacy_global_average_v1":
        return LegacyDenseCorrespondenceNet()
    if selected == "spatial_projective_v2":
        return DenseCorrespondenceNet()
    raise ValueError(f"unsupported dense architecture: {selected!r}")


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
