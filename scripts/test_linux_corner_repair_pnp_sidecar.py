#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from replay_linux_corner_repair_pnp_candidates import (
    CORNER_ORDER,
    SMALL_POINTS_MM,
    ordered_corner_array,
    solve_ippe_candidates,
    valid_proposal,
)


def test_ippe_candidate_enumeration_closes_nominal_projection() -> None:
    matrix = np.asarray([[1303.6753, 0, 720], [0, 1303.6753, 540], [0, 0, 1]], dtype=float)
    rvec = np.asarray([0.05, -0.2, 0.01], dtype=float)
    tvec = np.asarray([80.0, -40.0, 3200.0], dtype=float)
    corners, _ = cv2.projectPoints(SMALL_POINTS_MM, rvec, tvec, matrix, np.zeros(5))
    candidates, failure = solve_ippe_candidates(
        corners.reshape(4, 2), matrix, np.zeros(5), SMALL_POINTS_MM
    )
    assert failure is None
    assert candidates
    assert candidates[0]["selected"] is True
    assert candidates[0]["reprojection_rms_px"] < 1e-3
    assert candidates[0]["tvec_m"] == pytest.approx((tvec / 1000.0).tolist(), abs=1e-5)


def test_corner_contract_and_proposal_guards() -> None:
    candidate = {
        "raw_corners_order": list(CORNER_ORDER),
        "raw_corners_px": {
            "bl": [10, 20], "tl": [10, 10], "tr": [30, 10], "br": [30, 20]
        },
    }
    corners = ordered_corner_array(candidate)
    assert corners.tolist() == [[10, 20], [10, 10], [30, 10], [30, 20]]
    assert valid_proposal(corners, 1440, 1080) is None
    outside = corners.copy()
    outside[0, 0] = -1
    assert valid_proposal(outside, 1440, 1080) == "model_proposal_outside_image_x"
