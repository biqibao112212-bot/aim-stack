#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def load_module():
    path = Path(__file__).with_name("detect-linux-corner-repair-rows.py")
    spec = importlib.util.spec_from_file_location("corner_detector_rows_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detection_row(score_logit: float = 5.0) -> np.ndarray:
    row = np.zeros(22, dtype=np.float32)
    row[:8] = [100, 200, 180, 120, 180, 200, 100, 120]
    row[8] = score_logit
    row[9:13] = [0.1, 2.0, 0.2, 0.0]
    row[13:22] = [0.0, 0.1, 0.2, 4.0, 0.3, 0.4, 0.5, 0.6, 0.7]
    return row


def test_decode_diagnostics_and_class_metadata() -> None:
    module = load_module()
    output = np.stack([detection_row(), detection_row(-20.0)])
    detections, diagnostics = module.decode_with_diagnostics(output, 0.25, 0.45)
    assert len(detections) == 1
    assert diagnostics["raw_output_rows"] == 2
    assert diagnostics["below_score_rows"] == 1
    assert diagnostics["post_nms_candidates"] == 1
    detection = detections[0]
    assert detection["color_index"] == 1
    assert detection["number_index"] == 3
    assert detection["number"] == 3
    assert detection["armor_type"] == "small"
    assert np.asarray(detection["corners"]).shape == (4, 2)


def test_nms_rejection_is_counted() -> None:
    module = load_module()
    output = np.stack([detection_row(5.0), detection_row(4.0)])
    detections, diagnostics = module.decode_with_diagnostics(output, 0.25, 0.45)
    assert len(detections) == 1
    assert diagnostics["pre_nms_candidates"] == 2
    assert diagnostics["nms_rejected_candidates"] == 1
