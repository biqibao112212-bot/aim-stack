from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]


def load_module(name: str, relative: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_opencv_camera_point_uses_exposure_world_pose() -> None:
    prediction = load_module(
        "test_linux_corner_prediction",
        "scripts/evaluate-linux-corner-repair-local-prediction.py",
    )
    exposure = {
        "camera_position_world_m": [10.0, 20.0, 30.0],
        "camera_quaternion_world_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    actual = prediction.camera_to_world(np.asarray([1.0, 2.0, 3.0]), exposure)
    np.testing.assert_allclose(actual, [13.0, 19.0, 28.0], atol=1e-12)


def test_pose_capture_qualification_is_create_once_and_hash_bound(tmp_path: Path) -> None:
    qualifier = load_module(
        "test_linux_pose_qualifier",
        "scripts/qualify-linux-pose-frame-capture.py",
    )
    frames = tmp_path / "frames"
    frames.mkdir()
    raw = frames / "7_11_13.rgba"
    raw.write_bytes(b"\x5a" * qualifier.NATIVE_RGBA_BYTES)
    event = {
        "producer_epoch": 7,
        "frame_seq": 11,
        "timestamp_ns": 13,
        "width": 1440,
        "height": 1080,
        "payload_bytes": qualifier.NATIVE_RGBA_BYTES,
        "pixel_format": "rgba32",
        "raw_rgba_file": "frames/7_11_13.rgba",
    }
    exposure = {
        "schema_version": "aim-stack.exposure-frame/1",
        "producer_epoch": 7,
        "frame_seq": 11,
        "timestamp_ns": 13,
        "state_flags": 7,
        "online_target_truth_read": False,
        "future_truth_included": False,
    }
    (tmp_path / "capture-events.jsonl").write_text(
        json.dumps(event) + "\n", encoding="utf-8"
    )
    (tmp_path / "exposure-states.jsonl").write_text(
        json.dumps(exposure) + "\n", encoding="utf-8"
    )

    result = qualifier.qualify(tmp_path)
    assert result["capture"]["frame_count"] == 1
    assert result["exposure"]["coverage_fraction"] == 1.0
    ledger = json.loads((tmp_path / "tcp-identities.jsonl").read_text())
    assert ledger["payload_sha256"] == ledger["raw_rgba_sha256"]

    try:
        qualifier.qualify(tmp_path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("qualification must refuse to overwrite evidence")
