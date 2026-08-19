from __future__ import annotations

import torch

from training.stage3.train_image_corner_repair_formal import (
    CornerHeatmapPriorReliabilityNet,
)


def test_prior_heatmap_starts_at_exact_zero_correction() -> None:
    model = CornerHeatmapPriorReliabilityNet().eval()
    with torch.no_grad():
        output = model(
            torch.zeros((2, 3, 64, 128), dtype=torch.float32),
            torch.zeros((2, 15), dtype=torch.float32),
            torch.ones(2, dtype=torch.float32),
        )
    torch.testing.assert_close(output[:, :8], torch.zeros_like(output[:, :8]), atol=1.0e-7, rtol=0.0)


def test_gaussian_target_distribution_is_normalized() -> None:
    model = CornerHeatmapPriorReliabilityNet().eval()
    target = model.target_distribution(torch.zeros((3, 8), dtype=torch.float32))
    assert target.shape == (3, 4, 64 * 128)
    torch.testing.assert_close(
        target.sum(dim=2), torch.ones((3, 4)), atol=1.0e-6, rtol=0.0
    )
