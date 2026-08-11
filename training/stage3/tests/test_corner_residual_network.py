from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from training.stage3.corner_residual_network import (
    FEATURE_DIM,
    JointCornerResidualMLP,
    Standardization,
    observable_features,
    polygon_signed_area,
)
from training.stage3.train_corner_residual_network import (
    inner_group,
    split_groups,
    validation_groups,
)


def sample_corners() -> np.ndarray:
    return np.asarray(
        [[600.0, 560.0], [602.0, 520.0], [700.0, 522.0], [698.0, 562.0]],
        dtype=np.float32,
    )


def test_observable_feature_contract_is_finite_and_fixed_width() -> None:
    features = observable_features(sample_corners())
    assert features.shape == (FEATURE_DIM,)
    assert np.isfinite(features).all()


def test_features_are_invariant_to_uniform_pixel_scale_about_image_center() -> None:
    corners = sample_corners()
    center = np.asarray([720.0, 540.0], dtype=np.float32)
    smaller = center + 0.5 * (corners - center)
    first = observable_features(corners)
    second = observable_features(smaller)
    np.testing.assert_allclose(first[:8], second[:8], atol=1.0e-6)
    assert not np.isclose(first[10], second[10])


def test_polygon_area_requires_four_ordered_points() -> None:
    assert abs(float(polygon_signed_area(sample_corners()))) > 1.0
    try:
        polygon_signed_area(np.zeros((3, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid corner shape was accepted")


def test_model_starts_as_identity_correction() -> None:
    model = JointCornerResidualMLP()
    prediction = model(torch.randn(5, FEATURE_DIM))
    torch.testing.assert_close(prediction, torch.zeros_like(prediction))


def test_standardization_uses_train_rows_and_round_trips_targets() -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(20, FEATURE_DIM)).astype(np.float32)
    y = rng.normal(size=(20, 8)).astype(np.float32)
    standardization = Standardization.fit(x, y)
    restored = standardization.denormalize_targets(
        standardization.normalize_targets(y)
    )
    np.testing.assert_allclose(restored, y, atol=1.0e-6)


def test_leave_session_out_keeps_a_complete_segment_for_inner_validation() -> None:
    frame = pd.DataFrame(
        {
            "session_id": ["session-a"] * 4 + ["session-b"] * 4,
            "segment_index": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    outer_groups = split_groups(frame, "leave_session_out")
    inner_groups = validation_groups(frame, "leave_session_out")
    test_mask = outer_groups == "session-a"
    available = sorted(set(inner_groups[~test_mask]))
    selected = inner_group("leave_session_out", "session-a", available)
    validation_mask = (inner_groups == selected) & ~test_mask
    train_mask = ~(test_mask | validation_mask)
    assert test_mask.sum() == 4
    assert validation_mask.sum() == 2
    assert train_mask.sum() == 2
    assert not np.any(train_mask & validation_mask)
