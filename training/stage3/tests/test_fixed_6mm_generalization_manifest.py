from __future__ import annotations

import unittest

from training.stage3.generate_fixed_6mm_generalization_manifest import (
    capture_envelope,
    generate_records,
)


class Fixed6mmGeneralizationManifestTest(unittest.TestCase):
    def test_generator_is_deterministic_and_every_combined_path_is_safe(self) -> None:
        first, rejected_first = generate_records(2026072911, "fixed-6mm-test")
        second, rejected_second = generate_records(2026072911, "fixed-6mm-test")
        self.assertEqual(first, second)
        self.assertEqual(rejected_first, rejected_second)
        self.assertEqual(12, len(first))
        self.assertEqual(6, sum(record["mode"] == "spin" for record in first))
        self.assertEqual(6, sum(record["mode"] == "linear_and_spin" for record in first))
        self.assertEqual(12, len({record["session_id"] for record in first}))
        for record in first:
            self.assertEqual("wide_6mm", record["camera_profile"])
            self.assertFalse(record["dual_focal"])
            self.assertLessEqual(record["distance_m"], 6.5)
            if record["mode"] == "linear_and_spin":
                self.assertTrue(record["capture_envelope"]["accepted"])

    def test_known_behind_camera_path_is_rejected(self) -> None:
        envelope = capture_envelope(
            distance_m=2.5257400170310795,
            direction_deg=177.99829160697192,
            linear_speed_mps=2.3154964440247463,
            linear_span_m=8.0,
        )
        self.assertFalse(envelope["accepted"])
        self.assertLess(envelope["min_forward_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
