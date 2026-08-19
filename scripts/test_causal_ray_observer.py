#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from causal_ray_observer import CausalRayObserver, ObserverConfig


def candidate(u_offset: float, v_offset: float = 0.0) -> dict:
    return {
        "valid": True,
        "camera_tvec_m": [u_offset, v_offset, 2.0],
        "yaw_absolute_rad": 0.1,
        "detector_number": 3,
    }


def frame(index: int, candidates: list[dict], **extra) -> dict:
    result = {
        "schema_version": "autoaim-observer-frame-v1",
        "session_id": "test-session",
        "producer_epoch": 7,
        "frame_seq": 100 + index,
        "capture_timestamp_ns": 1_000_000_000 + index * 10_000_000,
        "observation_sink_status": "ok",
        "candidates": candidates,
    }
    result.update(extra)
    return result


def observation_projection(output: dict) -> str:
    selected = copy.deepcopy(output)
    for handle in selected["anonymous_handles"]:
        handle.pop("measurement_quality_features", None)
    return json.dumps(selected, sort_keys=True)


def test_permutation_and_frame_local_metadata_do_not_change_output() -> None:
    forward = CausalRayObserver()
    reverse = CausalRayObserver()
    for index in range(10):
        left = candidate(-0.2 + 0.002 * index)
        right = candidate(0.3 + 0.002 * index)
        left["observation_index"] = 0
        right["observation_index"] = 1
        output_forward = forward.update(frame(index, [left, right]))
        left["observation_index"] = 19
        right["observation_index"] = 3
        output_reverse = reverse.update(frame(index, [right, left]))
        assert observation_projection(output_forward) == observation_projection(output_reverse)
    assert output_forward["observer_status"] == "OBSERVED_ANONYMOUS"
    assert output_forward["physical_identity_resolved"] is False


def test_empty_frames_become_stale_then_reacquire_from_new_handles() -> None:
    observer = CausalRayObserver()
    for index in range(8):
        output = observer.update(frame(index, [candidate(0.01 * index)]))
    assert output["observer_status"] == "OBSERVED_ANONYMOUS"
    old_handle = output["anonymous_handles"][0]["ephemeral_handle_id"]
    output = observer.update(frame(8, []))
    assert output["observer_status"] == "ACQUIRING"
    assert output["anonymous_handles"] == []
    output = observer.update(frame(14, []))
    assert output["observer_status"] == "STALE"
    output = observer.update(frame(15, []))
    assert output["observer_status"] == "STALE"
    output = observer.update(frame(16, [candidate(0.15)]))
    assert output["observer_status"] == "REACQUIRING"
    assert output["anonymous_handles"][0]["ephemeral_handle_id"] != old_handle
    assert output["prediction_valid"] is False


def test_too_many_candidates_and_close_assignment_fail_closed() -> None:
    observer = CausalRayObserver()
    output = observer.update(frame(0, [candidate(value) for value in (-0.4, -0.2, 0.0, 0.2, 0.4)]))
    assert output["observer_status"] == "AMBIGUOUS_SET"
    assert output["set_ambiguity_status"] == "too_many_candidates"

    observer = CausalRayObserver(ObserverConfig(ambiguity_margin_deg=1.0))
    observer.update(frame(0, [candidate(-0.01), candidate(0.01)]))
    output = observer.update(frame(1, [candidate(0.0)]))
    assert output["observer_status"] == "AMBIGUOUS_SET"
    assert output["status_reason"] == ["close_assignment_cost"]


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ({"observation_sink_status": "overflow"}, "observation_sink_failure"),
        ({"frame_seq": 100}, "frame_sequence_duplicate_or_regression"),
        ({"capture_timestamp_ns": 1_000_000_000}, "timestamp_duplicate_or_regression"),
        ({"producer_epoch": 8}, "session_or_epoch_change"),
    ],
)
def test_invalid_stream_conditions_fail_closed(mutation: dict, reason: str) -> None:
    observer = CausalRayObserver()
    observer.update(frame(0, [candidate(0.0)]))
    output = observer.update(frame(1, [candidate(0.01)], **mutation))
    assert output["observer_status"] == "INVALID_STREAM"
    assert output["status_reason"] == [reason]
    assert output["anonymous_handles"] == []


def test_irregular_real_time_changes_rate_and_future_does_not_change_past() -> None:
    regular = CausalRayObserver(ObserverConfig(min_history_events=2))
    irregular = CausalRayObserver(ObserverConfig(min_history_events=2))
    regular.update(frame(0, [candidate(0.0)]))
    first_output = regular.update(frame(1, [candidate(0.02)]))
    frozen = copy.deepcopy(first_output)
    regular.update(frame(2, [candidate(0.04)]))
    assert first_output == frozen

    irregular.update(frame(0, [candidate(0.0)]))
    delayed = frame(1, [candidate(0.02)], capture_timestamp_ns=1_020_000_000)
    irregular_output = irregular.update(delayed)
    regular_rate = first_output["anonymous_handles"][0]["du_dt_deg_s"]
    irregular_rate = irregular_output["anonymous_handles"][0]["du_dt_deg_s"]
    assert regular_rate == pytest.approx(2.0 * irregular_rate, rel=1e-3)


def test_truth_like_fields_are_not_runtime_inputs() -> None:
    clean = CausalRayObserver()
    contaminated = CausalRayObserver()
    for index in range(4):
        base = frame(index, [candidate(0.01 * index)])
        with_truth = copy.deepcopy(base)
        with_truth.update({"truth_slot": 3, "future_truth": [999, 999, 999]})
        with_truth["candidates"][0]["truth_slot"] = index % 4
        assert clean.update(base) == contaminated.update(with_truth)
