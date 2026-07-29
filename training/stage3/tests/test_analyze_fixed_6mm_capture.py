from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.stage3.analyze_fixed_6mm_capture import analyze


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


class Fixed6mmCaptureAuditTest(unittest.TestCase):
    def test_audit_joins_exact_truth_and_counts_observation_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session_id = "capture-spin-00"
            manifest = root / "manifest.jsonl"
            _write_jsonl(manifest, [{"session_id": session_id, "mode": "spin"}])

            raw = root / "raw"
            raw.mkdir()
            observations = raw / "observations.jsonl"
            truth = raw / "truth.jsonl"
            base = {"producer_epoch": 7, "timestamp_ns": 100}
            _write_jsonl(
                observations,
                [
                    {
                        **base,
                        "frame_seq": 1,
                        "camera_profile_id": "wide_6mm",
                        "armors": [{"valid": True, "detector_number": 3}],
                    },
                    {
                        **base,
                        "frame_seq": 2,
                        "camera_profile_id": "wide_6mm",
                        "armors": [],
                    },
                ],
            )
            truth_rows = []
            for frame_seq, x in ((1, 3.0), (2, 4.0)):
                truth_rows.append(
                    {
                        **base,
                        "frame_seq": frame_seq,
                        "has_exact_exposure_truth": True,
                        "selected_target_id": 9,
                        "exposure_state": {"camera_position_world_m": [0.0, 0.0, 0.0]},
                        "ground_truth": {
                            "targets": [{"target_id": 9, "world_position_m": [x, 0.0, 0.0]}]
                        },
                    }
                )
            _write_jsonl(truth, truth_rows)

            evidence = root / "evidence" / session_id
            evidence.mkdir(parents=True)
            (evidence / "session_result.json").write_text(
                json.dumps(
                    {
                        "session_id": session_id,
                        "observations": str(observations),
                        "truth": str(truth),
                        "camera_profile": "wide_6mm",
                        "dual_focal": False,
                    }
                ),
                encoding="utf-8",
            )

            report = analyze(manifest, evidence.parent)
            self.assertTrue(report["all_fixed_6mm"])
            self.assertEqual(2, report["aggregates"]["overall"]["frames"])
            self.assertEqual(50.0, report["aggregates"]["overall"]["target3_pct"])
            self.assertEqual(50.0, report["aggregates"]["overall"]["empty_pct"])
            self.assertEqual(3.0, report["aggregates"]["overall"]["distance_min_m"])
            self.assertEqual(4.0, report["aggregates"]["overall"]["distance_max_m"])


if __name__ == "__main__":
    unittest.main()
