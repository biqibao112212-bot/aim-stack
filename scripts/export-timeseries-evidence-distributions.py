#!/usr/bin/env python3
"""Losslessly export Stage3 time-key, availability, u/v and set-transition evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fields: list[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def frame_key(record: dict) -> tuple[str, int, int, int]:
    return (
        str(record.get("session_id", "")),
        int(record["producer_epoch"]),
        int(record["frame_seq"]),
        int(record["timestamp_ns"]),
    )


def finite_tvec(armor: dict) -> bool:
    values = armor.get("camera_tvec_m")
    return (
        isinstance(values, list)
        and len(values) == 3
        and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values)
    )


def valid_armor(armor: dict) -> bool:
    return armor.get("valid") is not False and finite_tvec(armor)


def camera_angles_deg(armor: dict) -> tuple[float | None, float | None]:
    if not finite_tvec(armor):
        return None, None
    x, y, z = (float(value) for value in armor["camera_tvec_m"])
    z = max(z, 1e-6)
    return math.degrees(math.atan2(x, z)), math.degrees(math.atan2(y, z))


def detector_signature(observation: dict) -> str:
    values = []
    for armor in observation.get("armors", []):
        if not valid_armor(armor):
            continue
        values.append(
            (
                int(armor.get("detector_number", -1)),
                int(armor.get("detector_color", -1)),
                int(armor.get("detector_type", -1)),
            )
        )
    return json.dumps(sorted(values), separators=(",", ":"))


def quantiles(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return {"n": 0, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "n": len(ordered),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def missing_streaks(
    truth_rows: list[dict], present_keys: set[tuple], layer: str, common: dict
) -> list[dict]:
    ordered = sorted(truth_rows, key=lambda row: (int(row["timestamp_ns"]), int(row["frame_seq"])))
    if not ordered:
        return []
    timestamps = [int(row["timestamp_ns"]) for row in ordered]
    typical_delta_ns = (
        int(sorted(b - a for a, b in zip(timestamps, timestamps[1:]) if b > a)[
            max(0, (len(timestamps) - 2) // 2)
        ])
        if len(timestamps) > 1
        else 0
    )
    result: list[dict] = []
    start: int | None = None
    for index, record in enumerate(ordered):
        present = frame_key(record) in present_keys
        if not present and start is None:
            start = index
        if present and start is not None:
            first = ordered[start]
            last = ordered[index - 1]
            result.append(
                {
                    **common,
                    "layer": layer,
                    "start_frame_seq": int(first["frame_seq"]),
                    "start_timestamp_ns": int(first["timestamp_ns"]),
                    "end_frame_seq": int(last["frame_seq"]),
                    "end_timestamp_ns": int(last["timestamp_ns"]),
                    "missing_truth_frames": index - start,
                    "duration_ms": (int(record["timestamp_ns"]) - int(first["timestamp_ns"])) * 1e-6,
                    "right_censored": False,
                }
            )
            start = None
    if start is not None:
        first = ordered[start]
        last = ordered[-1]
        result.append(
            {
                **common,
                "layer": layer,
                "start_frame_seq": int(first["frame_seq"]),
                "start_timestamp_ns": int(first["timestamp_ns"]),
                "end_frame_seq": int(last["frame_seq"]),
                "end_timestamp_ns": int(last["timestamp_ns"]),
                "missing_truth_frames": len(ordered) - start,
                "duration_ms": (int(last["timestamp_ns"]) - int(first["timestamp_ns"]) + typical_delta_ns) * 1e-6,
                "right_censored": True,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    accepted_root = args.accepted_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    analysis_paths = {
        "spin": accepted_root / "spin" / "angular-facing-v1" / "analysis_summary.json",
        "combined": accepted_root / "combined" / "angular-facing-v1" / "analysis_summary.json",
    }
    source_roots: dict[Path, tuple[str, set[float]]] = {}
    for label, path in analysis_paths.items():
        analysis = read_json(path)
        accepted_scales = {float(value) for value in analysis["included_radial_scales"]}
        for root in analysis["roots"]:
            source_roots[Path(root).resolve()] = (label, accepted_scales)

    frame_rows: list[dict] = []
    detection_rows: list[dict] = []
    event_rows: list[dict] = []
    transition_rows: list[dict] = []
    streak_rows: list[dict] = []
    run_rows: list[dict] = []
    source_hashes: dict[str, dict[str, str]] = {}
    duplicate_truth_keys = 0
    duplicate_observation_keys = 0

    for source_root, (analysis_motion, accepted_scales) in sorted(
        source_roots.items(), key=lambda item: str(item[0])
    ):
        for run_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
            manifest_path = run_dir / "collection_run_manifest.json"
            truth_path = run_dir / "truth.jsonl"
            observation_path = run_dir / "stage3_observations.jsonl"
            summary_path = run_dir / "summary.json"
            if not all(path.exists() for path in (manifest_path, truth_path, observation_path, summary_path)):
                continue
            manifest = read_json(manifest_path)
            radial_scale = float(manifest["radial_scale"])
            if not any(abs(radial_scale - accepted) <= 1e-9 for accepted in accepted_scales):
                continue
            truths = read_jsonl(truth_path)
            observations = read_jsonl(observation_path)
            motion = "spin" if manifest.get("motion_mode") == "spin" else "combined"
            if motion != analysis_motion:
                raise ValueError(f"motion label mismatch: {run_dir}")
            common = {
                "motion": motion,
                "source_root": source_root.name,
                "run": run_dir.name,
                "radial_scale": radial_scale,
                "distance_m": float(manifest["requested_distance_m"]),
                "repeat": int(manifest["repeat"]),
            }
            truth_map = {frame_key(row): row for row in truths}
            observation_map = {frame_key(row): row for row in observations}
            duplicate_truth_keys += len(truths) - len(truth_map)
            duplicate_observation_keys += len(observations) - len(observation_map)
            observation_keys = set(observation_map)
            valid_event_keys = {
                key
                for key, row in observation_map.items()
                if any(valid_armor(armor) for armor in row.get("armors", []))
            }

            sorted_truths = sorted(truths, key=lambda row: (int(row["timestamp_ns"]), int(row["frame_seq"])))
            previous_truth_timestamp: int | None = None
            previous_observation_timestamp: int | None = None
            previous_observation_seq: int | None = None
            for truth in sorted_truths:
                key = frame_key(truth)
                observation = observation_map.get(key)
                timestamp_ns = int(truth["timestamp_ns"])
                truth_interval_ms = (
                    None if previous_truth_timestamp is None else (timestamp_ns - previous_truth_timestamp) * 1e-6
                )
                previous_truth_timestamp = timestamp_ns
                armor_count = len(observation.get("armors", [])) if observation else 0
                valid_count = (
                    sum(valid_armor(armor) for armor in observation.get("armors", [])) if observation else 0
                )
                observation_interval_ms = None
                observation_seq_delta = None
                if observation is not None:
                    observation_interval_ms = (
                        None
                        if previous_observation_timestamp is None
                        else (timestamp_ns - previous_observation_timestamp) * 1e-6
                    )
                    observation_seq_delta = (
                        None
                        if previous_observation_seq is None
                        else int(truth["frame_seq"]) - previous_observation_seq
                    )
                    previous_observation_timestamp = timestamp_ns
                    previous_observation_seq = int(truth["frame_seq"])
                frame_rows.append(
                    {
                        **common,
                        "session_id": str(truth.get("session_id", "")),
                        "producer_epoch": int(truth["producer_epoch"]),
                        "frame_seq": int(truth["frame_seq"]),
                        "timestamp_ns": timestamp_ns,
                        "truth_exact": bool(truth.get("has_exact_exposure_truth")),
                        "observation_frame_available": observation is not None,
                        "valid_event": valid_count > 0,
                        "armor_count": armor_count,
                        "valid_armor_count": valid_count,
                        "invalid_armor_count": armor_count - valid_count,
                        "truth_interval_ms": truth_interval_ms,
                        "observation_interval_ms": observation_interval_ms,
                        "observation_frame_seq_delta": observation_seq_delta,
                        "gimbal_pose_timestamp_ns": (
                            observation.get("gimbal_pose_timestamp_ns") if observation else None
                        ),
                        "gimbal_pose_timestamp_delta_ns": (
                            int(observation["gimbal_pose_timestamp_ns"]) - timestamp_ns
                            if observation and observation.get("gimbal_pose_timestamp_ns") is not None
                            else None
                        ),
                        "gimbal_pose_exposure_matched": (
                            observation.get("gimbal_pose_exposure_matched") if observation else None
                        ),
                        "tracker_world_transform_exposure_matched": (
                            observation.get("tracker_world_transform_exposure_matched")
                            if observation
                            else None
                        ),
                        "position_contract": observation.get("position_contract") if observation else None,
                        "camera_gimbal_extrinsic_from_config": (
                            observation.get("camera_gimbal_extrinsic_from_config")
                            if observation
                            else None
                        ),
                    }
                )

            for key, observation in observation_map.items():
                if key in truth_map:
                    continue
                armors = observation.get("armors", [])
                frame_rows.append(
                    {
                        **common,
                        "session_id": key[0],
                        "producer_epoch": key[1],
                        "frame_seq": key[2],
                        "timestamp_ns": key[3],
                        "truth_exact": False,
                        "observation_frame_available": True,
                        "valid_event": any(valid_armor(armor) for armor in armors),
                        "armor_count": len(armors),
                        "valid_armor_count": sum(valid_armor(armor) for armor in armors),
                        "invalid_armor_count": sum(not valid_armor(armor) for armor in armors),
                        "truth_interval_ms": None,
                        "observation_interval_ms": None,
                        "observation_frame_seq_delta": None,
                        "gimbal_pose_timestamp_ns": observation.get("gimbal_pose_timestamp_ns"),
                        "gimbal_pose_timestamp_delta_ns": (
                            int(observation["gimbal_pose_timestamp_ns"]) - key[3]
                            if observation.get("gimbal_pose_timestamp_ns") is not None
                            else None
                        ),
                        "gimbal_pose_exposure_matched": observation.get("gimbal_pose_exposure_matched"),
                        "tracker_world_transform_exposure_matched": observation.get(
                            "tracker_world_transform_exposure_matched"
                        ),
                        "position_contract": observation.get("position_contract"),
                        "camera_gimbal_extrinsic_from_config": observation.get(
                            "camera_gimbal_extrinsic_from_config"
                        ),
                    }
                )

            sorted_observations = sorted(
                observations, key=lambda row: (int(row["timestamp_ns"]), int(row["frame_seq"]))
            )
            previous_event: dict | None = None
            previous_observation: dict | None = None
            for observation in sorted_observations:
                timestamp_ns = int(observation["timestamp_ns"])
                valid = [armor for armor in observation.get("armors", []) if valid_armor(armor)]
                for fallback_index, armor in enumerate(observation.get("armors", [])):
                    u_deg, v_deg = camera_angles_deg(armor)
                    tvec = armor.get("camera_tvec_m") if finite_tvec(armor) else [None, None, None]
                    detection_rows.append(
                        {
                            **common,
                            "session_id": str(observation.get("session_id", "")),
                            "producer_epoch": int(observation["producer_epoch"]),
                            "frame_seq": int(observation["frame_seq"]),
                            "timestamp_ns": timestamp_ns,
                            "observation_index": int(armor.get("observation_index", fallback_index)),
                            "valid": valid_armor(armor),
                            "detector_number": armor.get("detector_number"),
                            "detector_color": armor.get("detector_color"),
                            "detector_type": armor.get("detector_type"),
                            "camera_x_m": tvec[0],
                            "camera_y_m": tvec[1],
                            "camera_z_m": tvec[2],
                            "u_deg": u_deg,
                            "v_deg": v_deg,
                            "yaw_absolute_rad": armor.get("yaw_absolute_rad"),
                            "reprojection_rms_px": armor.get("reprojection_rms_px"),
                            "reprojection_max_px": armor.get("reprojection_max_px"),
                        }
                    )
                if valid:
                    event_rows.append(
                        {
                            **common,
                            "session_id": str(observation.get("session_id", "")),
                            "producer_epoch": int(observation["producer_epoch"]),
                            "frame_seq": int(observation["frame_seq"]),
                            "timestamp_ns": timestamp_ns,
                            "valid_armor_count": len(valid),
                            "preceding_valid_event_interval_ms": (
                                None
                                if previous_event is None
                                else (timestamp_ns - int(previous_event["timestamp_ns"])) * 1e-6
                            ),
                            "preceding_valid_event_frame_seq_delta": (
                                None
                                if previous_event is None
                                else int(observation["frame_seq"]) - int(previous_event["frame_seq"])
                            ),
                        }
                    )
                    previous_event = observation
                if previous_observation is not None:
                    previous_count = len(previous_observation.get("armors", []))
                    previous_valid = sum(
                        valid_armor(armor) for armor in previous_observation.get("armors", [])
                    )
                    current_count = len(observation.get("armors", []))
                    current_valid = len(valid)
                    previous_signature = detector_signature(previous_observation)
                    current_signature = detector_signature(observation)
                    transition_rows.append(
                        {
                            **common,
                            "previous_frame_seq": int(previous_observation["frame_seq"]),
                            "previous_timestamp_ns": int(previous_observation["timestamp_ns"]),
                            "current_frame_seq": int(observation["frame_seq"]),
                            "current_timestamp_ns": timestamp_ns,
                            "delta_ms": (timestamp_ns - int(previous_observation["timestamp_ns"])) * 1e-6,
                            "frame_seq_delta": int(observation["frame_seq"])
                            - int(previous_observation["frame_seq"]),
                            "previous_armor_count": previous_count,
                            "current_armor_count": current_count,
                            "armor_count_delta": current_count - previous_count,
                            "previous_valid_count": previous_valid,
                            "current_valid_count": current_valid,
                            "valid_count_delta": current_valid - previous_valid,
                            "previous_detector_signature": previous_signature,
                            "current_detector_signature": current_signature,
                            "same_detector_signature": previous_signature == current_signature,
                        }
                    )
                previous_observation = observation

            streak_rows.extend(missing_streaks(truths, observation_keys, "observation_frame", common))
            streak_rows.extend(missing_streaks(truths, valid_event_keys, "valid_event", common))
            source_hashes[f"{motion}:{source_root.name}:{run_dir.name}"] = {
                "collection_run_manifest.json": sha256(manifest_path),
                "truth.jsonl": sha256(truth_path),
                "stage3_observations.jsonl": sha256(observation_path),
                "summary.json": sha256(summary_path),
            }
            run_rows.append(
                {
                    **common,
                    "truth_frames": len(truths),
                    "exact_truth_frames": sum(bool(row.get("has_exact_exposure_truth")) for row in truths),
                    "observation_frames": len(observations),
                    "full_key_join_frames": len(set(truth_map).intersection(observation_map)),
                    "usable_exact_truth_join_frames": sum(
                        bool(truth_map[key].get("has_exact_exposure_truth"))
                        for key in set(truth_map).intersection(observation_map)
                    ),
                    "valid_events": len(valid_event_keys),
                    "detections": sum(len(row.get("armors", [])) for row in observations),
                    "valid_detections": sum(
                        valid_armor(armor)
                        for row in observations
                        for armor in row.get("armors", [])
                    ),
                }
            )

    empirical_rows: list[dict] = []
    empirical_sources: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for row in frame_rows:
        sample = f"{row['motion']}:{row['run']}:{row['frame_seq']}:{row['timestamp_ns']}"
        for metric in ("truth_interval_ms", "observation_interval_ms"):
            value = row.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                empirical_sources[(str(row["motion"]), metric)].append((sample, float(value)))
    for row in event_rows:
        value = row.get("preceding_valid_event_interval_ms")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            sample = f"{row['motion']}:{row['run']}:{row['frame_seq']}:{row['timestamp_ns']}"
            empirical_sources[(str(row["motion"]), "valid_event_interval_ms")].append(
                (sample, float(value))
            )
    for row in streak_rows:
        sample = (
            f"{row['motion']}:{row['run']}:{row['layer']}:"
            f"{row['start_frame_seq']}:{row['start_timestamp_ns']}"
        )
        empirical_sources[(str(row["motion"]), f"{row['layer']}_missing_streak_ms")].append(
            (sample, float(row["duration_ms"]))
        )
    for (motion, metric), values in sorted(empirical_sources.items()):
        ordered = sorted(values, key=lambda item: (item[1], item[0]))
        count = len(ordered)
        for rank, (sample, value) in enumerate(ordered, 1):
            empirical_rows.append(
                {
                    "motion": motion,
                    "metric": metric,
                    "sample_key": sample,
                    "rank": rank,
                    "sample_count": count,
                    "value": value,
                    "empirical_cdf": rank / count,
                    "empirical_survival": (count - rank) / count,
                }
            )

    count_frequency = Counter(
        (str(row["motion"]), int(row["armor_count"]), int(row["valid_armor_count"]))
        for row in frame_rows
        if row["observation_frame_available"]
    )
    frequency_rows = [
        {
            "motion": motion,
            "armor_count": armor_count,
            "valid_armor_count": valid_count,
            "frames": count,
        }
        for (motion, armor_count, valid_count), count in sorted(count_frequency.items())
    ]

    paths = {
        "frame_availability_samples.csv": (frame_rows, list(frame_rows[0])),
        "detection_uv_samples.csv": (detection_rows, list(detection_rows[0])),
        "valid_event_interval_samples.csv": (event_rows, list(event_rows[0])),
        "candidate_set_transition_samples.csv": (transition_rows, list(transition_rows[0])),
        "missing_streak_samples.csv": (streak_rows, list(streak_rows[0])),
        "run_summary_samples.csv": (run_rows, list(run_rows[0])),
        "empirical_distributions.csv": (empirical_rows, list(empirical_rows[0])),
        "candidate_count_frequency.csv": (frequency_rows, list(frequency_rows[0])),
    }
    for name, (rows, fields) in paths.items():
        write_csv(output / name, fields, rows)

    interval_summary = {
        f"{motion}:{metric}": quantiles([value for _, value in values])
        for (motion, metric), values in sorted(empirical_sources.items())
    }
    observation_frame_rows = [row for row in frame_rows if row["observation_frame_available"]]
    position_contract_counts = Counter(str(row.get("position_contract")) for row in observation_frame_rows)
    summary = {
        "schema_version": "stage3-timeseries-evidence-distribution-v1",
        "authority": "accepted 120-run truth-gated spin/combined observation matrix",
        "join_key": ["session_id", "producer_epoch", "frame_seq", "timestamp_ns"],
        "uv_definition": {
            "source": "camera_tvec_m in OpenCV [right, down, forward] metres",
            "u": "degrees(atan2(camera_x, camera_z))",
            "v": "degrees(atan2(camera_y, camera_z))",
        },
        "runs": len(run_rows),
        "frame_rows": len(frame_rows),
        "detection_rows": len(detection_rows),
        "valid_event_rows": len(event_rows),
        "candidate_transition_rows": len(transition_rows),
        "missing_streak_rows": len(streak_rows),
        "empirical_distribution_rows": len(empirical_rows),
        "truth_frames": sum(int(row["truth_frames"]) for row in run_rows),
        "exact_truth_frames": sum(int(row["exact_truth_frames"]) for row in run_rows),
        "observation_frames": sum(int(row["observation_frames"]) for row in run_rows),
        "full_key_join_frames": sum(int(row["full_key_join_frames"]) for row in run_rows),
        "usable_exact_truth_join_frames": sum(
            int(row["usable_exact_truth_join_frames"]) for row in run_rows
        ),
        "valid_events": sum(int(row["valid_events"]) for row in run_rows),
        "detections": sum(int(row["detections"]) for row in run_rows),
        "valid_detections": sum(int(row["valid_detections"]) for row in run_rows),
        "observation_frame_contract_audit": {
            "gimbal_pose_exposure_matched": sum(
                row.get("gimbal_pose_exposure_matched") is True for row in observation_frame_rows
            ),
            "gimbal_pose_exposure_mismatch": sum(
                row.get("gimbal_pose_exposure_matched") is not True for row in observation_frame_rows
            ),
            "tracker_world_transform_exposure_matched": sum(
                row.get("tracker_world_transform_exposure_matched") is True
                for row in observation_frame_rows
            ),
            "tracker_world_transform_exposure_mismatch": sum(
                row.get("tracker_world_transform_exposure_matched") is not True
                for row in observation_frame_rows
            ),
            "configured_camera_gimbal_extrinsic": sum(
                row.get("camera_gimbal_extrinsic_from_config") is True
                for row in observation_frame_rows
            ),
            "position_contract_counts": dict(sorted(position_contract_counts.items())),
        },
        "zero_observation_runs": [
            f"{row['motion']}:{row['run']}" for row in run_rows if int(row["observation_frames"]) == 0
        ],
        "duplicate_truth_keys": duplicate_truth_keys,
        "duplicate_observation_keys": duplicate_observation_keys,
        "interval_and_streak_navigation": interval_summary,
        "identity_boundary": (
            "observation_index and detector signature are frame-local diagnostics only; "
            "candidate-set transitions do not assign a physical armor identity"
        ),
        "distribution_policy": (
            "sample CSVs and exact ranked empirical distributions are authoritative; "
            "this summary is navigation only"
        ),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = output / "TIMESERIES_EVIDENCE_REPORT.md"
    report_path.write_text(
        "\n".join(
            [
                "# PnP 后因果时间序列完整证据导出",
                "",
                "本目录逐帧保留 exact-exposure join、空帧、有效事件、候选集合变化和全部 camera-tvec u/v。",
                "汇总统计仅用于导航，不能替代逐样本 CSV 或精确经验分布。",
                "",
                f"- runs: {summary['runs']}",
                f"- truth frames: {summary['truth_frames']}",
                f"- observation frames: {summary['observation_frames']}",
                f"- valid events: {summary['valid_events']}",
                f"- detections: {summary['detections']}",
                f"- missing streak samples: {summary['missing_streak_rows']}",
                "",
                "物理槽位和 truth 不是这些原始时间序列文件的在线输入。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    script_path = Path(__file__).resolve()
    output_files = [output / name for name in paths] + [summary_path, report_path]
    manifest = {
        "schema_version": "stage3-timeseries-evidence-retention-v1",
        "protected": True,
        "analysis_source": {str(script_path): sha256(script_path)},
        "source_files": source_hashes,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in output_files
        },
        "retention_class": "protected point-level PnP time-series evidence",
    }
    (output / "retention_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
