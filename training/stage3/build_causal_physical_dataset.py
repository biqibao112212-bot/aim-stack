"""Build clean, causally segmented physical histories without PnP coordinates.

The source truth-history artifact contains all four exact armor positions at
every observation event.  This derivative preserves the real event timestamps
and cuts histories at causal gap discontinuities.  The qualification policy
keeps all four exact slots; an optional facing-count policy is a diagnostic
visibility proxy that still excludes PnP coordinates. Slot indices stay in the
simulator geometry contract and no permutation search is used.

Only train and validation shards are admitted.  Test shards are never opened.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .build_truth_history_dataset import (
    _load_jsonl,
    _nearest_indices,
    _parse_truth,
    _resolve_truth,
    _sha256,
    _write_json,
)


SCHEMA_VERSION = "stage3-causal-physical-v1"
MAX_PHYSICALLY_VISIBLE = 2
GAP_MULTIPLIER = 2.5
DEFAULT_CONSTANT_MOTION_HISTORY_EVENTS = 4


def _last_causal_segment(mask: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Keep the suffix after the last abnormal gap using prefix-only times."""
    active = np.flatnonzero(mask)
    result = np.zeros_like(mask, dtype=np.bool_)
    if active.size == 0:
        return result
    if active.size < 3:
        result[active] = True
        return result
    times = time_s[active].astype(np.float64)
    delta = np.diff(times)
    positive = delta[delta > 0]
    if positive.size == 0:
        result[active[-1:]] = True
        return result
    typical = float(np.median(positive))
    discontinuities = np.flatnonzero(
        (~np.isfinite(delta)) | (delta <= 0) | (delta > GAP_MULTIPLIER * typical)
    )
    start = int(discontinuities[-1] + 1) if discontinuities.size else 0
    result[active[start:]] = True
    return result


def _facing_slots(frame: Any, requested_count: int) -> tuple[list[int], bool]:
    """Select exact physical slots from camera-facing geometry, never PnP xyz."""
    camera = np.asarray(frame.camera_origin_world_m, dtype=np.float64)
    scores: list[tuple[float, int]] = []
    for armor in frame.armors:
        position = np.asarray(armor.position_world_m, dtype=np.float64)
        direction = camera - position
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            raise ValueError("camera and armor positions coincide")
        score = float(
            np.dot(np.asarray(armor.outward_normal_world, dtype=np.float64), direction / norm)
        )
        scores.append((score, int(armor.relative_slot)))
    scores.sort(key=lambda item: (-item[0], item[1]))
    overflow = requested_count > MAX_PHYSICALLY_VISIBLE
    count = min(max(int(requested_count), 1), MAX_PHYSICALLY_VISIBLE)
    return sorted(slot for _, slot in scores[:count]), overflow


def _build_shard(task: dict[str, Any]) -> dict[str, Any]:
    truth_history_path = Path(task["truth_history_shard"])
    observation_path = Path(task["observation_shard"])
    output_path = Path(task["output_shard"])
    frames = _parse_truth(Path(task["truth_path"]), str(task["truth_sha256"]))
    timestamps = np.asarray([frame.timestamp_ns for frame in frames], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("truth timestamps are not strictly increasing")
    velocities = np.asarray([frame.velocity_world_mps for frame in frames], dtype=np.float64)
    yaw_rates = np.asarray([frame.yaw_rate_rad_s for frame in frames], dtype=np.float64)
    origins = np.asarray([frame.target_origin_world_m for frame in frames], dtype=np.float64)
    yaws = np.asarray([frame.yaw_rad for frame in frames], dtype=np.float64)
    target_ids = np.asarray([frame.target_id for frame in frames], dtype=np.int64)
    geometry_hashes = np.asarray([str(frame.geometry_hash) for frame in frames], dtype=np.str_)
    epochs = np.asarray([frame.producer_epoch for frame in frames], dtype=np.int64)

    with np.load(truth_history_path, allow_pickle=False) as loaded:
        physical = {key: loaded[key] for key in loaded.files}
    with np.load(observation_path, allow_pickle=False) as loaded:
        observation = {key: loaded[key] for key in loaded.files}
    if not np.array_equal(physical["t0_ns"], observation["t0_ns"]):
        raise ValueError("truth-history and observation samples are not aligned")
    if physical["event_time_s"].shape != observation["event_time_s"].shape:
        raise ValueError("truth-history and observation event shapes differ")

    raw_position = physical["truth_obs"].astype(np.float32, copy=True)
    visible_position = np.zeros_like(raw_position)
    visible_mask = np.zeros_like(physical["truth_obs_mask"], dtype=np.bool_)
    event_mask = np.zeros_like(physical["event_mask"], dtype=np.bool_)
    segment_start_time_s = np.zeros((raw_position.shape[0],), dtype=np.float32)
    visible_counts = np.zeros((5,), dtype=np.int64)
    overflow_events = 0
    reset_windows = 0
    dropped_short_windows = 0
    dropped_nonconstant_history = 0
    maximum_timestamp_error_ns = 0
    continuity_checked_events = 0
    keep_sample = np.ones((raw_position.shape[0],), dtype=np.bool_)

    for sample in range(raw_position.shape[0]):
        source_event = observation["event_mask"][sample].astype(np.bool_, copy=False)
        suffix = _last_causal_segment(source_event, physical["event_time_s"][sample])
        if not np.array_equal(suffix, source_event):
            reset_windows += 1
        active = np.flatnonzero(suffix)
        if active.size == 0:
            raise ValueError("causal segmentation removed every history event")
        if active.size < int(task["minimum_events"]):
            keep_sample[sample] = False
            dropped_short_windows += 1
            continue
        segment_start_time_s[sample] = physical["event_time_s"][sample, active[0]]
        anchor_index, anchor_error = _nearest_indices(
            timestamps, np.asarray([int(physical["t0_ns"][sample])], dtype=np.int64)
        )
        if int(anchor_error[0]) != 0:
            raise ValueError("causal physical anchor truth is not exact at t0")
        anchor = frames[int(anchor_index[0])]
        event_timestamps = int(physical["t0_ns"][sample]) + np.rint(
            physical["event_time_s"][sample, active].astype(np.float64) * 1e9
        ).astype(np.int64)
        truth_index, timestamp_error = _nearest_indices(timestamps, event_timestamps)
        maximum_timestamp_error_ns = max(
            maximum_timestamp_error_ns, int(timestamp_error.max(initial=0))
        )
        if np.any(timestamp_error > 2_000):
            raise ValueError("physical visibility truth lookup exceeded 2 us")
        fit_events = int(task["constant_motion_history_events"])
        fit_start = int(truth_index[max(0, len(truth_index) - fit_events)])
        fit_end = int(anchor_index[0])
        left, right = sorted((fit_start, fit_end))
        interval = slice(left, right + 1)
        interval_time_s = (
            timestamps[interval] - timestamps[fit_start]
        ).astype(np.float64) / 1e9
        velocity_change = np.linalg.norm(
            velocities[interval] - velocities[fit_start], axis=1
        ).max(initial=0.0)
        yaw_rate_change = np.abs(
            yaw_rates[interval] - yaw_rates[fit_start]
        ).max(initial=0.0)
        expected_origin = (
            origins[fit_start][None, :]
            + velocities[fit_start][None, :] * interval_time_s[:, None]
        )
        position_residual = np.linalg.norm(
            origins[interval] - expected_origin, axis=1
        ).max(initial=0.0)
        expected_yaw = yaws[fit_start] + yaw_rates[fit_start] * interval_time_s
        yaw_delta = yaws[interval] - expected_yaw
        yaw_residual = np.abs(
            np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))
        ).max(initial=0.0)
        same_identity = (
            np.all(epochs[interval] == anchor.producer_epoch)
            and np.all(target_ids[interval] == anchor.target_id)
            and np.all(geometry_hashes[interval] == str(anchor.geometry_hash))
        )
        if not (
            same_identity
            and velocity_change <= 1e-6
            and yaw_rate_change <= 1e-6
            and position_residual <= 1e-4
            and yaw_residual <= 1e-4
        ):
            keep_sample[sample] = False
            dropped_nonconstant_history += 1
            continue
        for tensor_event, frame_index in zip(active, truth_index):
            frame = frames[int(frame_index)]
            if (
                frame.producer_epoch != anchor.producer_epoch
                or frame.target_id != anchor.target_id
                or str(frame.geometry_hash) != str(anchor.geometry_hash)
            ):
                raise ValueError(
                    "causal physical history crossed producer/target/geometry identity"
                )
            continuity_checked_events += 1
            requested = int(observation["obs_mask"][sample, tensor_event].sum())
            if requested <= 0:
                raise ValueError("active observation event has no visible candidates")
            if task["visibility_policy"] == "complete":
                slots, overflow = [0, 1, 2, 3], False
            elif task["visibility_policy"] == "facing-count":
                slots, overflow = _facing_slots(frames[int(frame_index)], requested)
            else:
                raise ValueError(f"unsupported visibility policy: {task['visibility_policy']}")
            overflow_events += int(overflow)
            visible_counts[len(slots)] += 1
            visible_position[sample, tensor_event, slots] = raw_position[
                sample, tensor_event, slots
            ]
            visible_mask[sample, tensor_event, slots] = True
            event_mask[sample, tensor_event] = True

    if not np.any(keep_sample):
        return {
            "path": None,
            "split": task["split"],
            "session_id": task["session_id"],
            "sample_count": 0,
            "source_sample_count": int(raw_position.shape[0]),
            "dropped_short_windows": dropped_short_windows,
            "dropped_nonconstant_history": dropped_nonconstant_history,
            "sha256": None,
            "bytes": 0,
            "normalization_count": 0,
            "normalization_sum": [0.0, 0.0, 0.0],
            "normalization_sum_square": [0.0, 0.0, 0.0],
            "visible_counts": visible_counts.tolist(),
            "overflow_events": overflow_events,
            "reset_windows": reset_windows,
            "maximum_timestamp_error_ns": maximum_timestamp_error_ns,
            "continuity_checked_events": continuity_checked_events,
        }
    output = {
        "history_position_m": visible_position[keep_sample],
        "history_obs_mask": visible_mask[keep_sample],
        "event_mask": event_mask[keep_sample],
        "event_time_s": physical["event_time_s"][keep_sample].astype(np.float32, copy=False),
        "segment_start_time_s": segment_start_time_s[keep_sample],
        "tau": physical["tau"][keep_sample].astype(np.float32, copy=False),
        "future_position": physical["future_position"][keep_sample].astype(np.float32, copy=False),
        "motion_class": physical["motion_class"][keep_sample].astype(np.int64, copy=False),
        "rule_query": physical["rule_query"][keep_sample].astype(np.bool_, copy=False),
        "distance_m": physical["distance_m"][keep_sample].astype(np.float32, copy=False),
        "session_id": physical["session_id"][keep_sample],
        "t0_ns": physical["t0_ns"][keep_sample].astype(np.int64, copy=False),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)
    valid = output["history_position_m"][output["history_obs_mask"]]
    return {
        "path": str(output_path),
        "split": task["split"],
        "session_id": task["session_id"],
        "sample_count": int(keep_sample.sum()),
        "source_sample_count": int(raw_position.shape[0]),
        "dropped_short_windows": dropped_short_windows,
        "dropped_nonconstant_history": dropped_nonconstant_history,
        "sha256": _sha256(output_path),
        "bytes": output_path.stat().st_size,
        "normalization_count": int(valid.shape[0]),
        "normalization_sum": valid.astype(np.float64).sum(axis=0).tolist(),
        "normalization_sum_square": np.square(valid.astype(np.float64)).sum(axis=0).tolist(),
        "visible_counts": visible_counts.tolist(),
        "overflow_events": overflow_events,
        "reset_windows": reset_windows,
        "maximum_timestamp_error_ns": maximum_timestamp_error_ns,
        "continuity_checked_events": continuity_checked_events,
    }


def build(args: argparse.Namespace) -> Path:
    truth_history = Path(args.truth_history).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite causal physical dataset: {output}")
    output.mkdir(parents=True)
    truth_manifest_path = truth_history / "dataset_manifest.json"
    truth_manifest = json.loads(truth_manifest_path.read_text(encoding="utf-8"))
    if truth_manifest.get("schema_version") != "stage3-truth-history-v1":
        raise ValueError("source must be the qualified truth-history v1 dataset")
    if not bool(truth_manifest.get("qualification_passed", False)):
        raise ValueError("truth-history source is not qualified")
    if bool(truth_manifest.get("test_accessed", True)):
        raise ValueError("truth-history source must record test_accessed=false")
    observation_root = Path(str(truth_manifest["source_dataset"])).resolve()
    observation_manifest_path = observation_root / "dataset_manifest.json"
    observation_manifest = json.loads(
        observation_manifest_path.read_text(encoding="utf-8")
    )
    if observation_manifest.get("schema_version") != "stage3-dataset-v3":
        raise ValueError("truth-history source must bind a v3 observation dataset")
    raw_root = observation_root.parent.parent
    sources = {
        str(item["session_id"]): item
        for item in _load_jsonl(
            observation_root / str(observation_manifest["canonical_sources"])
        )
    }
    observation_shards = {
        (str(item["split"]), str(item["session_ids"][0])): item
        for item in observation_manifest["shards"]
        if str(item["split"]) in {"train", "validation"}
    }
    tasks: list[dict[str, Any]] = []
    split_sessions = {"train": 0, "validation": 0}
    requested_sessions = set(args.session_id or [])
    found_sessions: set[str] = set()
    for shard in truth_manifest["shards"]:
        split = str(shard["split"])
        if split not in split_sessions:
            continue
        session_id = str(shard["session_ids"][0])
        if requested_sessions and session_id not in requested_sessions:
            continue
        if args.session_limit > 0 and split_sessions[split] >= args.session_limit:
            continue
        observation_shard = observation_shards[(split, session_id)]
        source = sources[session_id]
        truth_path = _resolve_truth(raw_root, source)
        tasks.append({
            "truth_history_shard": str(truth_history / str(shard["path"])),
            "observation_shard": str(observation_root / str(observation_shard["path"])),
            "output_shard": str(output / "shards" / Path(str(shard["path"])).name),
            "split": split,
            "session_id": session_id,
            "truth_path": str(truth_path),
            "truth_sha256": str(source["truth_sha256"]),
            "visibility_policy": args.visibility_policy,
            "minimum_events": args.minimum_events,
            "constant_motion_history_events": args.constant_motion_history_events,
        })
        split_sessions[split] += 1
        found_sessions.add(session_id)
    missing_sessions = requested_sessions - found_sessions
    if missing_sessions:
        raise ValueError(
            f"requested causal physical sessions are absent from train/validation: "
            f"{sorted(missing_sessions)}"
        )
    if not tasks or not all(split_sessions.values()):
        raise ValueError(f"incomplete causal physical selection: {split_sessions}")

    results: list[dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_build_shard, task): task for task in tasks}
            for completed, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                print(json.dumps({
                    "completed": completed,
                    "total": len(tasks),
                    "session_id": result["session_id"],
                    "split": result["split"],
                    "samples": result["sample_count"],
                }), flush=True)
    except Exception:
        _write_json(output / "build_failed.json", {
            "status": "failed", "completed_shards": len(results),
            "truth_history": str(truth_history),
        })
        raise
    results.sort(key=lambda item: (item["split"], item["session_id"]))
    admitted = [item for item in results if int(item["sample_count"]) > 0]
    train = [item for item in admitted if item["split"] == "train"]
    if not train or not any(item["split"] == "validation" for item in admitted):
        raise ValueError("causal history policy removed an entire split")
    count = sum(int(item["normalization_count"]) for item in train)
    total = np.sum([item["normalization_sum"] for item in train], axis=0, dtype=np.float64)
    total_square = np.sum(
        [item["normalization_sum_square"] for item in train], axis=0, dtype=np.float64
    )
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 1e-12)
    normalization = {
        "schema_version": "stage3-causal-physical-normalization-v1",
        "source_split": "train",
        "position_m": {"count": count, "mean": mean.tolist(), "std": np.sqrt(variance).tolist()},
    }
    normalization_path = output / "normalization.json"
    _write_json(normalization_path, normalization)
    shutil.copy2(truth_history / "geometry_template.json", output / "geometry_template.json")
    source_file = Path(__file__).resolve()
    visible_totals = np.sum([item["visible_counts"] for item in results], axis=0)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "qualification_passed": True,
        "source_truth_history": str(truth_history),
        "source_truth_history_manifest_sha256": _sha256(truth_manifest_path),
        "source_observation_dataset": str(observation_root),
        "source_observation_manifest_sha256": _sha256(observation_manifest_path),
        "test_accessed": False,
        "splits": ["train", "validation"],
        "session_count": len(admitted),
        "requested_session_count": len(results),
        "zero_sample_sessions": [
            item["session_id"] for item in results if int(item["sample_count"]) == 0
        ],
        "requested_sessions": sorted(requested_sessions),
        "sample_count": sum(int(item["sample_count"]) for item in results),
        "source_sample_count": sum(int(item["source_sample_count"]) for item in results),
        "dropped_short_windows": sum(
            int(item["dropped_short_windows"]) for item in results
        ),
        "dropped_nonconstant_history": sum(
            int(item["dropped_nonconstant_history"]) for item in results
        ),
        "normalization": normalization_path.name,
        "normalization_sha256": _sha256(normalization_path),
        "geometry_template": "geometry_template.json",
        "geometry_template_sha256": _sha256(output / "geometry_template.json"),
        "identity_contract": {
            "policy": "causal-cyclic-fixed-slots-v1",
            "permutation_search": False,
            "discontinuity": f"gap > {GAP_MULTIPLIER} * window median positive dt",
            "reacquisition": "new segment",
            "minimum_events_before_prediction": args.minimum_events,
            "constant_motion_fit_events": args.constant_motion_history_events,
            "constant_motion_history_to_t0": {
                "maximum_velocity_change_mps": 1e-6,
                "maximum_yaw_rate_change_rad_s": 1e-6,
                "maximum_position_residual_m": 1e-4,
                "maximum_yaw_residual_rad": 1e-4,
            },
            "slot_source": "exact relative_slot label for the assumed-correct external causal tracker",
            "history_continuity": "same producer_epoch, target_id, and geometry_hash as t0",
        },
        "visibility_contract": {
            "position_source": "exact exposure physical truth only",
            "policy": args.visibility_policy,
            "candidate_count_source": (
                "not used" if args.visibility_policy == "complete"
                else "observation mask count only; PnP coordinates excluded"
            ),
            "selection": (
                "all four exact physical slots" if args.visibility_policy == "complete"
                else "top camera-facing physical slots; diagnostic visibility proxy"
            ),
            "maximum_slots": 4 if args.visibility_policy == "complete" else MAX_PHYSICALLY_VISIBLE,
            "visible_event_counts": {
                "one": int(visible_totals[1]), "two": int(visible_totals[2]),
                "three": int(visible_totals[3]), "four": int(visible_totals[4]),
            },
            "overflow_candidate_events": sum(int(item["overflow_events"]) for item in results),
            "reset_windows": sum(int(item["reset_windows"]) for item in results),
        },
        "maximum_timestamp_error_ns": max(int(item["maximum_timestamp_error_ns"]) for item in results),
        "continuity_checked_events": sum(
            int(item["continuity_checked_events"]) for item in results
        ),
        "builder_source_sha256": {
            source_file.name: _sha256(source_file),
            "build_truth_history_dataset.py": _sha256(source_file.parent / "build_truth_history_dataset.py"),
            "truth_history_dataset.py": _sha256(source_file.parent / "truth_history_dataset.py"),
            "schema.py": _sha256(source_file.parent / "schema.py"),
        },
        "shards": [{
            "path": str(Path(item["path"]).relative_to(output)),
            "split": item["split"],
            "session_ids": [item["session_id"]],
            "sample_count": item["sample_count"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        } for item in admitted],
    }
    _write_json(output / "dataset_manifest.json", manifest)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-history", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--session-limit", type=int, default=0)
    parser.add_argument("--minimum-events", type=int, default=8)
    parser.add_argument(
        "--constant-motion-history-events", type=int,
        default=DEFAULT_CONSTANT_MOTION_HISTORY_EVENTS,
        help=(
            "number of most recent observation events that must share one "
            "constant velocity/yaw-rate interval through t0"
        ),
    )
    parser.add_argument(
        "--session-id", action="append", default=[],
        help="explicit train/validation session to include; may be repeated",
    )
    parser.add_argument(
        "--visibility-policy", choices=("complete", "facing-count"), default="complete"
    )
    args = parser.parse_args()
    if (
        args.workers < 1 or args.session_limit < 0 or args.minimum_events < 2
        or not 2 <= args.constant_motion_history_events <= 200
        or args.minimum_events > 200
        or args.minimum_events < args.constant_motion_history_events
    ):
        parser.error(
            "workers must be positive, session-limit non-negative, and "
            "2 <= constant-motion-history-events <= minimum-events <= 200"
        )
    print(build(args))


if __name__ == "__main__":
    main()
