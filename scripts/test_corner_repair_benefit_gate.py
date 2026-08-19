from __future__ import annotations

import copy

import numpy as np

from train_corner_repair_benefit_gate import (
    FEATURE_NAMES,
    Sample,
    gate_metrics,
    runtime_feature_rows,
)


def candidate(observation_id: int, offset_x: float, proposal_shift: float = 0.5) -> dict:
    raw = np.asarray(
        [
            [600.0 + offset_x, 500.0],
            [600.0 + offset_x, 460.0],
            [700.0 + offset_x, 460.0],
            [700.0 + offset_x, 500.0],
        ]
    )
    proposed = raw.copy()
    proposed[:, 0] += proposal_shift
    corners = lambda values: {
        name: point.tolist()
        for name, point in zip(("bl", "tl", "tr", "br"), values)
    }
    solution = {
        "solver_solution_index": 0,
        "tvec_m": [offset_x / 1000.0, 0.0, 4.0],
        "reprojection_rms_px": 0.1,
        "reprojection_max_px": 0.15,
        "observed_yaw_rad": 0.1,
        "selected": True,
    }
    return {
        "candidate_rank": observation_id,
        "observation_id": observation_id,
        "detector_confidence": 0.8,
        "detector_type": "small",
        "repair": {
            "raw_corners_px": corners(raw),
            "model_proposed_corners_px": corners(proposed),
            "applied": True,
            "reliability_probability": 0.9,
            "predicted_correction_rms_px": proposal_shift,
            "predicted_correction_max_px": proposal_shift,
        },
        "raw_pnp": {"candidates": [solution]},
    }


def frame(seq: int, candidates: list[dict]) -> dict:
    return {
        "producer_epoch": 7,
        "frame_seq": seq,
        "timestamp_ns": 1_000_000_000 + seq * 20_000_000,
        "candidates": candidates,
    }


def camera() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([[800.0, 0.0, 640.0], [0.0, 800.0, 480.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
    )


def test_runtime_features_are_future_causal() -> None:
    matrix, distortion = camera()
    first = frame(0, [candidate(0, -100.0)])
    future_a = frame(1, [candidate(0, -95.0)])
    future_b = frame(1, [candidate(0, 500.0, proposal_shift=10.0)])
    only_first = runtime_feature_rows([copy.deepcopy(first)], matrix, distortion)
    with_future_a = runtime_feature_rows([copy.deepcopy(first), future_a], matrix, distortion)
    with_future_b = runtime_feature_rows([copy.deepcopy(first), future_b], matrix, distortion)
    identity = (7, 0, 1_000_000_000, 0)
    np.testing.assert_array_equal(only_first[identity], with_future_a[identity])
    np.testing.assert_array_equal(only_first[identity], with_future_b[identity])


def test_past_candidate_permutation_does_not_change_current_features() -> None:
    matrix, distortion = camera()
    past_a = candidate(0, -100.0)
    past_b = candidate(1, 100.0)
    current = candidate(0, -95.0)
    ordered = runtime_feature_rows(
        [frame(0, [copy.deepcopy(past_a), copy.deepcopy(past_b)]), frame(1, [copy.deepcopy(current)])],
        matrix,
        distortion,
    )
    permuted = runtime_feature_rows(
        [frame(0, [copy.deepcopy(past_b), copy.deepcopy(past_a)]), frame(1, [copy.deepcopy(current)])],
        matrix,
        distortion,
    )
    identity = (7, 1, 1_020_000_000, 0)
    np.testing.assert_allclose(ordered[identity], permuted[identity], rtol=0.0, atol=0.0)


def test_reject_all_is_exact_raw_noninferiority() -> None:
    values = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    sample = Sample(
        session="s",
        split="train",
        mode="spin",
        identity=(1, 2, 3, 0),
        features=values,
        old_applied=True,
        outcome="HARM",
        raw={
            "angular_error_deg": 0.1,
            "radial_error_abs_mm": 10.0,
            "transverse_error_mm": 5.0,
        },
        proposed={
            "angular_error_deg": 1.0,
            "radial_error_abs_mm": 100.0,
            "transverse_error_mm": 50.0,
        },
    )
    result = gate_metrics([sample], np.asarray([0.0]), 1.01)
    row = result["sessions"][0]
    assert result["feasible"]
    assert row["gate_applied"] == 0
    assert row["gated_angular_p95_deg"] == row["raw_angular_p95_deg"]
    assert row["gated_radial_p95_mm"] == row["raw_radial_p95_mm"]
    assert row["gated_transverse_p95_mm"] == row["raw_transverse_p95_mm"]


def test_feature_contract_has_no_truth_or_planned_mode() -> None:
    joined = " ".join(FEATURE_NAMES).lower()
    for forbidden in ("truth", "relative_slot", "session", "mode", "frame_seq", "timestamp"):
        assert forbidden not in joined
