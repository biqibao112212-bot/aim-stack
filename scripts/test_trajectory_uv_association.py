#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from trajectory_uv_association import (
    CyclicSegmentAssociator,
    LearnedCausalAssociator,
    PairLogisticModel,
    pair_feature,
)


class AssociationTest(unittest.TestCase):
    def test_model_round_trip_and_causal_assignment(self) -> None:
        features = []
        labels = []
        history = [(0.0, 0.0, 0.0), (0.1, 0.1, 0.0)]
        for index in range(40):
            features.append(pair_feature(history, 0.2, 0.2 + index * 0.001, 0.0))
            labels.append(1)
            features.append(pair_feature(history, 0.2, 4.0 + index * 0.01, 2.0))
            labels.append(0)
        model = PairLogisticModel.fit(np.vstack(features), np.asarray(labels))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.json"
            model.save(path)
            loaded = PairLogisticModel.load(path)
        self.assertFalse(loaded.payload["truth_is_runtime_input"])
        associator = LearnedCausalAssociator(loaded, birth_probability=0.5)
        first = associator.update([{"t_s": 0.0, "u": 0.0, "v": 0.0}])
        self.assertEqual(first[0][1], 0)
        second = associator.update([{"t_s": 0.1, "u": np.radians(0.1), "v": 0.0}])
        self.assertEqual(second[0][1], 0)

    def test_cyclic_births_and_short_continuity(self) -> None:
        associator = CyclicSegmentAssociator(continuity_gate_deg=1.0, reacquire_timeout_s=0.5)
        first = associator.update([{"t_s": 0.0, "u": 0.0, "v": 0.0}])
        self.assertEqual(first[0][1], 0)
        continued = associator.update([{"t_s": 0.1, "u": np.radians(0.1), "v": 0.0}])
        self.assertEqual(continued[0][1], 0)
        self.assertEqual(continued[0][0]["association_mode"], "continuity")
        self.assertGreater(continued[0][0]["association_confidence"], 0.0)
        born = associator.update([{"t_s": 1.0, "u": np.radians(4.0), "v": 0.0}])
        self.assertEqual(born[0][1], 1)
        self.assertEqual(born[0][0]["association_mode"], "cyclic_birth")


if __name__ == "__main__":
    unittest.main()
