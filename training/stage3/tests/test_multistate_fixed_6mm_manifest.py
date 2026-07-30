from __future__ import annotations

import unittest

from training.stage3.generate_multistate_fixed_6mm_manifest import (
    SCHEMA_VERSION,
    generate_records,
)


class MultistateFixed6mmManifestTest(unittest.TestCase):
    def test_generation_is_deterministic_safe_and_varies_within_session(self) -> None:
        first, first_rejected = generate_records(
            2026073001, "multistate-test", spin_count=4, combined_count=4
        )
        second, second_rejected = generate_records(
            2026073001, "multistate-test", spin_count=4, combined_count=4
        )
        self.assertEqual(first, second)
        self.assertEqual(first_rejected, second_rejected)
        self.assertEqual(8, len(first))
        self.assertEqual(8, len({record["session_id"] for record in first}))

        for record in first:
            self.assertEqual(SCHEMA_VERSION, record["schema_version"])
            self.assertEqual("wide_6mm", record["camera_profile"])
            self.assertFalse(record["dual_focal"])
            self.assertEqual(12, len(record["segments"]))
            self.assertAlmostEqual(36.0, record["duration_s"])
            self.assertEqual(
                list(range(12)),
                [segment["segment_index"] for segment in record["segments"]],
            )
            stationary = [
                segment for segment in record["segments"]
                if segment["mode"] == "stationary"
            ]
            self.assertEqual(1, len(stationary))
            self.assertNotIn(stationary[0]["segment_index"], (0, 11))
            active = [
                segment for segment in record["segments"]
                if segment["mode"] != "stationary"
            ]
            self.assertEqual(11, len(active))
            self.assertEqual(11, len({segment["spin_rad_s"] for segment in active}))
            self.assertTrue(any(segment["spin_rad_s"] > 0 for segment in active))
            self.assertTrue(any(segment["spin_rad_s"] < 0 for segment in active))
            if record["mode"] == "spin":
                self.assertTrue(all(segment["mode"] == "spin" for segment in active))
                self.assertTrue(all(segment["linear_speed_mps"] == 0 for segment in active))
            else:
                self.assertTrue(
                    all(segment["mode"] == "linear_and_spin" for segment in active)
                )
                self.assertEqual(11, len({segment["linear_speed_mps"] for segment in active}))
                self.assertTrue(
                    all(segment["capture_envelope"]["accepted"] for segment in active)
                )
                for segment in active:
                    self.assertGreaterEqual(segment["linear_span_m"], 0.5)

    def test_invalid_schedule_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_records(1, "bad", segments_per_session=2)
        with self.assertRaises(ValueError):
            generate_records(1, "bad", segment_duration_s=float("nan"))


if __name__ == "__main__":
    unittest.main()
