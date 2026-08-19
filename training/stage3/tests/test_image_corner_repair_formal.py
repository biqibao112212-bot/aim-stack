from __future__ import annotations

import numpy as np

from training.stage3.train_image_corner_repair_formal import (
    context_transform,
    normalized_context_predictions_to_full,
    normalized_context_targets,
)


def sample_raw() -> np.ndarray:
    return np.asarray(
        [[610.0, 570.0], [615.0, 525.0], [705.0, 530.0], [700.0, 575.0]],
        dtype=np.float32,
    )


def test_context_transform_round_trips_points() -> None:
    raw = sample_raw()
    transform, inverse = context_transform(raw)
    homogeneous = np.column_stack((raw, np.ones(4, dtype=np.float32)))
    projected = homogeneous @ transform.T
    projected = projected[:, :2] / projected[:, 2:]
    restored = np.column_stack((projected, np.ones(4, dtype=np.float32))) @ inverse.T
    restored = restored[:, :2] / restored[:, 2:]
    np.testing.assert_allclose(restored, raw, atol=1.0e-4)


def test_normalized_target_inverse_recovers_full_pixel_residual() -> None:
    raw = sample_raw()[None]
    exact = raw + np.asarray(
        [[[-2.0, 3.0], [1.5, -1.0], [2.5, 0.5], [-1.0, 2.0]]],
        dtype=np.float32,
    )
    normalized = normalized_context_targets(raw, exact)
    restored = normalized_context_predictions_to_full(raw, normalized)
    np.testing.assert_allclose(restored, (exact - raw).reshape(1, 8), atol=2.0e-4)


def test_zero_normalized_residual_preserves_raw_corners() -> None:
    raw = np.stack((sample_raw(), sample_raw() + np.asarray([35.0, -20.0], dtype=np.float32)))
    restored = normalized_context_predictions_to_full(raw, np.zeros((2, 8), dtype=np.float32))
    np.testing.assert_allclose(restored, 0.0, atol=2.0e-4)
