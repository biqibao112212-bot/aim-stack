"""Probabilistic four-corner prediction with a truth-free forward signature."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .backbone import ArmorPoseBackbone
from .labels import calibrated_grid_moments


@dataclass
class SparsePrediction:
    heatmap_logits: torch.Tensor
    visibility_logits: torch.Tensor
    entropy: torch.Tensor
    image_mean: torch.Tensor
    image_covariance: torch.Tensor
    local_image_mean: torch.Tensor
    tail_image_mean: torch.Tensor
    in_context_probability: torch.Tensor
    tail_covariance: torch.Tensor


class ProbabilisticCornerNet(nn.Module):
    family = "same-frame-four-heatmap-probabilistic-corners-v1"

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ArmorPoseBackbone()
        self.heatmaps = nn.Conv2d(self.backbone.output_channels, 4, 1)
        self.tail = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(self.backbone.output_channels, 64), nn.SiLU(),
            nn.Linear(64, 4 * 6),
        )
        nn.init.zeros_(self.heatmaps.weight)
        nn.init.zeros_(self.heatmaps.bias)
        nn.init.zeros_(self.tail[-1].weight)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(self, patch: torch.Tensor, geometry: torch.Tensor, raw_full: torch.Tensor,
                raw_patch: torch.Tensor, inverse_transform: torch.Tensor,
                scale: torch.Tensor) -> SparsePrediction:
        if raw_full.shape != (patch.shape[0], 4, 2) or raw_patch.shape != raw_full.shape or scale.shape != (patch.shape[0],):
            raise ValueError("raw corner/scale online contract changed")
        feature = self.backbone(patch, geometry)
        logits = self.heatmaps(feature)
        local_image_mean, local_covariance, entropy = calibrated_grid_moments(
            logits, raw_patch, raw_full, inverse_transform
        )
        tail = self.tail(feature).reshape(-1, 4, 6)
        tail_residual = 4.0 * torch.tanh(tail[..., :2])
        tail_image_mean = raw_full + tail_residual * scale[:, None, None]
        tail_std_norm = torch.exp(tail[..., 2:4].clamp(-5.0, 0.6931471805599453))
        rho = 0.95 * torch.tanh(tail[..., 4])
        sigma_x, sigma_y = tail_std_norm.unbind(dim=-1)
        tail_covariance_norm = torch.stack(
            (sigma_x.square(), rho * sigma_x * sigma_y,
             rho * sigma_x * sigma_y, sigma_y.square()), dim=-1
        ).reshape(-1, 4, 2, 2)
        identity = torch.eye(2, dtype=patch.dtype, device=patch.device)
        tail_covariance = tail_covariance_norm * scale[:, None, None, None].square() + 1.0e-4 * identity
        in_context_probability = torch.sigmoid(tail[..., 5])
        mixture = in_context_probability[..., None]
        image_mean = mixture * local_image_mean + (1.0 - mixture) * tail_image_mean
        local_delta = local_image_mean - image_mean
        tail_delta = tail_image_mean - image_mean
        image_covariance = (
            in_context_probability[..., None, None]
            * (local_covariance + local_delta[..., :, None] * local_delta[..., None, :])
            + (1.0 - in_context_probability)[..., None, None]
            * (tail_covariance + tail_delta[..., :, None] * tail_delta[..., None, :])
        )
        return SparsePrediction(
            heatmap_logits=logits,
            visibility_logits=tail[..., 5],
            entropy=entropy,
            image_mean=image_mean,
            image_covariance=image_covariance,
            local_image_mean=local_image_mean,
            tail_image_mean=tail_image_mean,
            in_context_probability=in_context_probability,
            tail_covariance=tail_covariance,
        )

    @property
    def config(self) -> dict[str, object]:
        return {
            "family": self.family,
            "online_inputs": ["patch", "detector_geometry", "raw_corners", "raw_corners_patch", "inverse_transform", "raw_scale"],
            "online_truth_input": False,
            "temporal_input": False,
            "corner_order": ["bl", "tl", "tr", "br"],
        }
