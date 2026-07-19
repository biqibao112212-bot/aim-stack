#!/usr/bin/env python3
"""Extract all per-exposure solved armors without tracker selection.

The resulting JSONL is the raw observation-set artifact for the target-3
slow-spin experiment.  It deliberately omits tracked_id, tracked_armor and
jump_flag.  observation_id is frame-local only and is never treated as a
cross-frame identity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def target_candidate(armor: dict[str, object]) -> bool:
    distance_mm = armor.get("distance_mm")
    return (
        armor.get("color") == 1
        and armor.get("type") == "small"
        and finite(distance_mm)
        and 500.0 <= float(distance_mm) <= 12000.0
    )


def selected_reprojection(armor: dict[str, object]) -> Optional[float]:
    for candidate in armor.get("pnp_candidates") or []:
        if isinstance(candidate, dict) and candidate.get("selected") and finite(candidate.get("reprojection_error_px")):
            return float(candidate["reprojection_error_px"])
    return None


def compact_observation(
    armor: dict[str, object], detector_by_id: dict[int, dict[str, object]]
) -> dict[str, object]:
    observation_id = int(armor.get("observation_id", -1))
    detector = detector_by_id.get(observation_id, {})
    return {
        "observation_id": observation_id,
        "number": int(armor.get("number", -1)),
        "color": int(armor.get("color", -1)),
        "type": str(armor.get("type", "")),
        "target_candidate": target_candidate(armor),
        "detector_confidence": detector.get("confidence"),
        "center_px": armor.get("center_px"),
        "pnp_vertices_px": armor.get("pnp_vertices_px"),
        "pnp_candidates": armor.get("pnp_candidates", []),
        "pnp_ab": armor.get("pnp_ab"),
        "position_m": armor.get("position_m"),
        "armor_yaw_deg": armor.get("armor_yaw_deg"),
        "armor_yaw_absolute_deg": armor.get("armor_yaw_absolute_deg"),
        "image_yaw_deg": armor.get("image_yaw_deg"),
        "image_pitch_down_positive_deg": armor.get("image_pitch_down_positive_deg"),
        "distance_mm": armor.get("distance_mm"),
        "distance_to_image_center_px": armor.get("distance_to_image_center_px"),
        "selected_reprojection_error_px": selected_reprojection(armor),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    frames_with_observations = 0
    frames_with_target_candidates = 0
    observation_count = 0
    target_candidate_count = 0
    target_counts: dict[str, int] = {}
    all_number_counts: dict[str, int] = {}
    per_frame_counts: list[int] = []

    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            solved = record.get("solved_armors") or []
            detector = record.get("detector") or {}
            detector_by_id = {
                int(item["observation_id"]): item
                for item in detector.get("candidates") or []
                if isinstance(item, dict) and finite(item.get("observation_id"))
            }
            observations = [
                compact_observation(armor, detector_by_id)
                for armor in solved
                if isinstance(armor, dict)
            ]
            target_observations = [
                observation for observation in observations if observation["target_candidate"]
            ]
            frame_count += 1
            frames_with_observations += bool(observations)
            frames_with_target_candidates += bool(target_observations)
            observation_count += len(observations)
            target_candidate_count += len(target_observations)
            per_frame_counts.append(len(target_observations))
            for observation in observations:
                key = str(observation["number"])
                all_number_counts[key] = all_number_counts.get(key, 0) + 1
                if observation["target_candidate"]:
                    target_counts[key] = target_counts.get(key, 0) + 1

            output_record = {
                "source_image_seq": record.get("source_image_seq"),
                "source_capture_timestamp_ns": record.get("source_capture_timestamp_ns"),
                "vision_completion_timestamp_ns": record.get("vision_completion_timestamp_ns"),
                "input_gimbal_yaw_deg": record.get("input_gimbal_yaw_deg"),
                "input_gimbal_pitch_deg": record.get("input_gimbal_pitch_deg"),
                "input_gimbal_roll_deg": record.get("input_gimbal_roll_deg"),
                "camera": record.get("camera"),
                "calibration": record.get("calibration"),
                "observations": observations,
                "target_candidate_count": len(target_observations),
            }
            destination.write(json.dumps(output_record, ensure_ascii=False, separators=(",", ":")))
            destination.write("\n")

    summary = {
        "source": str(input_path),
        "frames": frame_count,
        "frames_with_observations": frames_with_observations,
        "frames_with_target_candidates": frames_with_target_candidates,
        "all_solved_observations": observation_count,
        "target_candidate_observations": target_candidate_count,
        "target_candidate_number_counts": target_counts,
        "all_solved_number_counts": all_number_counts,
        "target_observations_per_frame": {
            "p50": sorted(per_frame_counts)[len(per_frame_counts) // 2] if per_frame_counts else None,
            "max": max(per_frame_counts) if per_frame_counts else None,
            "mean": sum(per_frame_counts) / len(per_frame_counts) if per_frame_counts else None,
            "frames_with_2_or_more": sum(count >= 2 for count in per_frame_counts),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
