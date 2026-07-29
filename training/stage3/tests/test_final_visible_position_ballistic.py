from __future__ import annotations

import numpy as np

from training.stage3.evaluate_final_visible_position_ballistic import (
    DISPLAY_BODY_PERCENTILE,
    TruthState,
    _ballistic_label,
    _plot,
    _table_rows,
)


def test_stationary_ballistic_label_uses_causal_model_range_only_for_tau() -> None:
    q0 = np.asarray(
        [[3.8, 0.0, 0.0], [4.0, 0.2, 0.0], [4.2, 0.0, 0.0], [4.0, -0.2, 0.0]],
        dtype=np.float32,
    )
    history = np.repeat(q0[None, :, :], 32, axis=0)
    state = TruthState(
        history_position_m=history,
        event_mask=np.ones(32, dtype=np.bool_),
        event_time_s=np.linspace(-0.31, 0.0, 32, dtype=np.float32),
        q0_position_m=q0,
        center_m=np.asarray([4.0, 0.0, 0.0], dtype=np.float32),
        velocity_mps=np.zeros(3, dtype=np.float32),
        yaw_rate_rad_s=0.0,
    )
    label = _ballistic_label(
        state,
        np.asarray([4.4, 0.0, 0.0], dtype=np.float32),
        reverse_direction=False,
        bullet_speed_mps=22.0,
        dense_step_s=0.001,
    )
    assert np.isclose(label["flight_time_s"], 0.2)
    assert np.isclose(label["estimated_distance_m"], 4.4)
    assert np.isclose(label["truth_distance_m"], 3.8)
    assert label["target_switch_count"] == 0
    assert np.array_equal(label["target_visible_delta_m"], np.zeros(3, dtype=np.float32))

    observed_stream = _ballistic_label(
        state,
        np.asarray([4.4, 0.0, 0.0], dtype=np.float32),
        truth_current_position_m=q0[1],
        reverse_direction=False,
        bullet_speed_mps=22.0,
        dense_step_s=0.001,
    )
    assert np.isclose(observed_stream["truth_distance_m"], np.linalg.norm(q0[1]))
    assert observed_stream["target_switch_count"] == -1
    assert np.allclose(
        q0[1] + observed_stream["target_visible_delta_m"], q0[0]
    )


def test_distance_table_uses_fixed_one_metre_bins() -> None:
    queries = {
        "truth_distance_m": np.asarray([2.2, 2.8, 6.2], dtype=np.float32),
        "estimated_distance_m": np.asarray([2.3, 2.7, 6.3], dtype=np.float32),
        "final_error_m": np.asarray([0.05, 0.15, 0.30], dtype=np.float32),
        "flight_time_s": np.asarray([0.1, 0.12, 0.28], dtype=np.float32),
    }
    rows = _table_rows(queries)
    two_to_three = next(row for row in rows if row["distance_bin_m"] == "[2,3)")
    overall = next(row for row in rows if row["distance_bin_m"] == "[1,7) overall")
    assert two_to_three["count"] == 2
    assert np.isclose(two_to_three["p50_error_mm"], 100.0)
    assert np.isclose(two_to_three["coverage_le_100mm"], 0.5)
    assert overall["count"] == 3


def test_plot_caps_the_tail_at_the_declared_body_percentile(tmp_path) -> None:
    queries = {
        "truth_distance_m": np.linspace(2.0, 6.5, 100, dtype=np.float32),
        "final_error_m": np.linspace(0.0, 1.0, 100, dtype=np.float32),
    }
    output = tmp_path / "scatter.png"
    audit = _plot(output, "rotation", queries)
    assert output.is_file()
    assert audit["display_body_percentile"] == DISPLAY_BODY_PERCENTILE
    assert audit["overflow_count"] == 5
