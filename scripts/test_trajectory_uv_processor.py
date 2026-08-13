#!/usr/bin/env python3
"""Focused regression tests for the offline u/v processor."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from trajectory_uv_processor import (
    CausalUvRidgeProcessor,
    ProcessorConfig,
    UvRidgeModel,
    cv_forecast,
    make_common_feature,
)


def synthetic_examples() -> list[dict]:
    result = []
    for run in range(8):
        times = np.arange(16, dtype=float) * 0.01
        for horizon in (0.05, 0.10, 0.20):
            u = 1.0 + run * 0.1 + 3.0 * times + 0.4 * times**2
            v = -0.5 + 0.2 * run - 1.5 * times + 0.2 * times**2
            cv = cv_forecast(times, u, v, horizon)
            actual = cv + np.asarray([0.02 * horizon, -0.01 * horizon])
            result.append(
                {
                    "example_id": f"run={run}|h={horizon}",
                    "horizon_s": horizon,
                    "times": times,
                    "history_u": u,
                    "history_v": v,
                    "feature_uv": make_common_feature(times, u, v, horizon),
                    "cv": cv,
                    "hold": np.asarray([u[-1], v[-1]]),
                    "actual": actual,
                    "residual_to_cv": actual - cv,
                }
            )
    return result


class ProcessorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = UvRidgeModel.fit(
            synthetic_examples(),
            ProcessorConfig(max_train_examples_per_horizon=100),
            {"test": True},
        )

    def test_model_round_trip_and_runtime_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            self.model.save(path)
            loaded = UvRidgeModel.load(path)
        self.assertFalse(loaded.to_dict()["truth_is_runtime_input"])
        self.assertEqual(
            loaded.to_dict()["runtime_input_contract"],
            ["timestamp_s", "u_deg", "v_deg"],
        )
        self.assertEqual(set(loaded.horizons), set(self.model.horizons))

    def test_ridge_and_gap_fallback_modes(self) -> None:
        processor = CausalUvRidgeProcessor(self.model)
        for index in range(16):
            processor.update(index * 0.01, 1.0 + index * 0.03, -0.5 - index * 0.015)
        prediction = processor.predict(0.10)
        self.assertEqual(prediction.method, "ridge_uv_residual")
        self.assertGreater(prediction.confidence, 0.0)
        processor.update(1.0, 2.0, -1.0)
        self.assertEqual(processor.predict(0.10).method, "hold_fallback")
        processor.update(1.01, 2.03, -1.01)
        self.assertEqual(processor.predict(0.10).method, "hold_fallback")
        processor.update(1.02, 2.06, -1.02)
        self.assertEqual(processor.predict(0.10).method, "hold_fallback")

    def test_kalman_is_explicit_diagnostic_mode(self) -> None:
        model = UvRidgeModel.fit(
            synthetic_examples(),
            ProcessorConfig(
                max_train_examples_per_horizon=100,
                enable_kalman_fallback=True,
            ),
        )
        processor = CausalUvRidgeProcessor(model)
        processor.update(0.0, 0.0, 0.0)
        processor.update(0.01, 0.1, 0.0)
        processor.update(0.02, 0.2, 0.0)
        self.assertEqual(processor.predict(0.10).method, "kalman_fallback")

    def test_timestamp_regression_is_rejected(self) -> None:
        processor = CausalUvRidgeProcessor(self.model)
        processor.update(1.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            processor.update(1.0, 0.1, 0.1)


if __name__ == "__main__":
    unittest.main()
