"""Small full-resolution encoder shared by sparse and dense offline branches."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _block(input_channels: int, output_channels: int, *, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(min(8, output_channels), output_channels),
        nn.SiLU(),
        nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, output_channels), output_channels),
        nn.SiLU(),
    )


class ArmorPoseBackbone(nn.Module):
    """Encode one detector-conditioned ROI without truth or temporal inputs."""

    output_channels = 32

    def __init__(self, geometry_dimension: int = 15) -> None:
        super().__init__()
        self.geometry_dimension = int(geometry_dimension)
        self.stem = _block(3, 24)
        self.down1 = _block(24, 48, stride=2)
        self.down2 = _block(48, 80, stride=2)
        self.down3 = _block(80, 128, stride=2)
        self.geometry = nn.Sequential(nn.Linear(self.geometry_dimension, 64), nn.SiLU(), nn.Linear(64, 128))
        self.up2 = _block(128 + 80, 80)
        self.up1 = _block(80 + 48, 48)
        self.up0 = _block(48 + 24, self.output_channels)

    def forward(self, patch: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        if patch.ndim != 4 or patch.shape[1:] != (3, 64, 128):
            raise ValueError("patch must be [B,3,64,128]")
        if geometry.shape != (patch.shape[0], self.geometry_dimension):
            raise ValueError("detector geometry contract changed")
        level0 = self.stem(patch)
        level1 = self.down1(level0)
        level2 = self.down2(level1)
        level3 = self.down3(level2)
        level3 = level3 + self.geometry(geometry)[:, :, None, None]
        decoded2 = self.up2(torch.cat((F.interpolate(level3, size=level2.shape[-2:], mode="bilinear", align_corners=False), level2), dim=1))
        decoded1 = self.up1(torch.cat((F.interpolate(decoded2, size=level1.shape[-2:], mode="bilinear", align_corners=False), level1), dim=1))
        return self.up0(torch.cat((F.interpolate(decoded1, size=level0.shape[-2:], mode="bilinear", align_corners=False), level0), dim=1))
