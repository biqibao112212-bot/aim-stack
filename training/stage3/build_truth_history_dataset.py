"""Derive causal truth-history inputs from the already captured Stage-3 data.

Only train and validation shards are built.  Each input event uses the exact
exposure truth at an observation timestamp, expressed in the same anchor
tracker frame as the existing future-position label.  No simulator run or new
capture is involved and the held-out test shards are never opened.
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

from .schema import TruthFrame


SCHEMA_VERSION = "stage3-truth-history-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.append(json.loads(line))
    return result


def _resolve_truth(raw_root: Path, source: dict[str, Any]) -> Path:
    declared = Path(str(source["truth"]))
    if declared.is_file():
        return declared
    session_root = raw_root / str(source["session_id"])
    candidates = sorted(session_root.glob("run-*/truth.jsonl"))
    expected = str(source["truth_sha256"])
    matching = [path for path in candidates if _sha256(path) == expected]
    if len(matching) != 1:
        raise FileNotFoundError(
            f"cannot uniquely repair truth path for {source['session_id']}: {matching}"
        )
    return matching[0]


def _rotation_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quaternion
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _nearest_indices(timestamps: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(timestamps, queries, side="left")
    right = np.clip(right, 0, len(timestamps) - 1)
    left = np.clip(right - 1, 0, len(timestamps) - 1)
    choose_left = np.abs(timestamps[left] - queries) <= np.abs(timestamps[right] - queries)
    index = np.where(choose_left, left, right)
    return index, np.abs(timestamps[index] - queries)


def _parse_truth(path: Path, expected_sha256: str) -> list[TruthFrame]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"truth source hash mismatch: {path}")
    frames: list[TruthFrame] = []
    for line in payload.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if bool(record.get("has_exact_exposure_truth", False)):
            frames.append(TruthFrame.from_mapping(record))
    frames.sort(key=lambda frame: (frame.producer_epoch, frame.timestamp_ns, frame.frame_seq))
    if not frames:
        raise ValueError(f"truth source has no exact frames: {path}")
    return frames


def _build_shard(task: dict[str, Any]) -> dict[str, Any]:
    source_shard = Path(task["source_shard"])
    output_shard = Path(task["output_shard"])
    truth_path = Path(task["truth_path"])
    frames = _parse_truth(truth_path, str(task["truth_sha256"]))
    with np.load(source_shard, allow_pickle=False) as loaded:
        source = {key: loaded[key] for key in loaded.files}
    session_ids = {str(value) for value in source["session_id"]}
    if session_ids != {str(task["session_id"])}:
        raise ValueError(f"source shard is not a single expected session: {session_ids}")

    timestamps = np.asarray([frame.timestamp_ns for frame in frames], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"truth timestamps are not strictly increasing: {truth_path}")
    positions_world = np.asarray([
        [armor.position_world_m for armor in frame.armors] for frame in frames
    ], dtype=np.float64)
    velocities = np.asarray([frame.velocity_world_mps for frame in frames], dtype=np.float64)
    yaw_rates = np.asarray([frame.yaw_rate_rad_s for frame in frames], dtype=np.float64)
    target_origins = np.asarray([
        frame.target_origin_world_m for frame in frames
    ], dtype=np.float64)
    target_yaws = np.asarray([frame.yaw_rad for frame in frames], dtype=np.float64)
    target_ids = np.asarray([frame.target_id for frame in frames], dtype=np.int64)
    geometry_hashes = np.asarray([frame.geometry_hash for frame in frames], dtype=np.str_)
    epochs = np.asarray([frame.producer_epoch for frame in frames], dtype=np.int64)

    sample_count, event_count = source["event_mask"].shape
    truth_obs = np.zeros((sample_count, event_count, 4, 3), dtype=np.float32)
    truth_mask = np.repeat(source["event_mask"][:, :, None], 4, axis=2).astype(np.bool_)
    anchor_velocity = np.zeros((sample_count, 3), dtype=np.float32)
    anchor_yaw_rate = np.zeros((sample_count,), dtype=np.float32)
    anchor_center_position = np.zeros((sample_count, 3), dtype=np.float32)
    rule_query = np.ones_like(source["tau"], dtype=np.bool_)
    maximum_history_timestamp_error_ns = 0
    maximum_q0_position_error_m = 0.0
    for sample_index in range(sample_count):
        t0 = int(source["t0_ns"][sample_index])
        anchor_index, anchor_error = _nearest_indices(timestamps, np.asarray([t0], dtype=np.int64))
        if int(anchor_error[0]) != 0:
            raise ValueError(f"anchor truth is not exact at {task['session_id']}/{t0}")
        anchor = frames[int(anchor_index[0])]
        rotation = _rotation_matrix(anchor.chassis_quaternion_world_wxyz)
        world_up_in_anchor = np.asarray([0.0, 0.0, 1.0]) @ rotation
        if (
            np.linalg.norm(world_up_in_anchor[:2]) > 1e-5
            or world_up_in_anchor[2] < 1.0 - 1e-5
        ):
            raise ValueError(
                "planar yaw-rate rollout requires world +Z aligned with anchor +Z"
            )
        anchor_velocity[sample_index] = (
            np.asarray(anchor.velocity_world_mps, dtype=np.float64) @ rotation
        ).astype(np.float32)
        anchor_yaw_rate[sample_index] = np.float32(anchor.yaw_rate_rad_s)
        if anchor.target_origin_world_m is None:
            raise ValueError("exact truth is missing the target rotation center")
        anchor_center_position[sample_index] = (
            (
                np.asarray(anchor.target_origin_world_m, dtype=np.float64)
                - np.asarray(anchor.gimbal_origin_world_m, dtype=np.float64)
            ) @ rotation
        ).astype(np.float32)
        valid_slots = np.flatnonzero(source["event_mask"][sample_index])
        event_queries = t0 + np.rint(
            source["event_time_s"][sample_index, valid_slots].astype(np.float64) * 1e9
        ).astype(np.int64)
        history_indices, timestamp_error = _nearest_indices(timestamps, event_queries)
        maximum_history_timestamp_error_ns = max(
            maximum_history_timestamp_error_ns,
            int(timestamp_error.max(initial=0)),
        )
        if np.any(epochs[history_indices] != anchor.producer_epoch):
            raise ValueError("truth history crossed producer epochs")
        delta_world = positions_world[history_indices] - np.asarray(
            anchor.gimbal_origin_world_m, dtype=np.float64
        )[None, None, :]
        truth_obs[sample_index, valid_slots] = (delta_world @ rotation).astype(np.float32)

        raw_q0 = (
            positions_world[int(anchor_index[0])]
            - np.asarray(anchor.gimbal_origin_world_m, dtype=np.float64)[None, :]
        ) @ rotation
        q0_error = np.linalg.norm(
            raw_q0.astype(np.float32) - source["future_position"][sample_index, 0], axis=-1
        ).max()
        maximum_q0_position_error_m = max(maximum_q0_position_error_m, float(q0_error))

        query_indices, query_error = _nearest_indices(
            timestamps, source["future_timestamp_ns"][sample_index].astype(np.int64)
        )
        if np.any(query_error != 0):
            raise ValueError("future truth timestamp is not exact")
        for query_index, future_index in enumerate(query_indices):
            if query_index == 0:
                continue
            left, right = sorted((int(anchor_index[0]), int(future_index)))
            segment_velocity = velocities[left:right + 1]
            segment_yaw_rate = yaw_rates[left:right + 1]
            segment_time_s = (
                timestamps[left:right + 1] - timestamps[int(anchor_index[0])]
            ).astype(np.float64) / 1e9
            velocity_change = np.linalg.norm(
                segment_velocity - segment_velocity[:1], axis=1
            ).max(initial=0.0)
            yaw_rate_change = np.max(
                np.abs(segment_yaw_rate - segment_yaw_rate[:1]), initial=0.0
            )
            expected_origin = (
                target_origins[int(anchor_index[0])][None, :]
                + velocities[int(anchor_index[0])][None, :] * segment_time_s[:, None]
            )
            position_residual = np.linalg.norm(
                target_origins[left:right + 1] - expected_origin, axis=1
            ).max(initial=0.0)
            expected_yaw = (
                target_yaws[int(anchor_index[0])]
                + yaw_rates[int(anchor_index[0])] * segment_time_s
            )
            yaw_delta = target_yaws[left:right + 1] - expected_yaw
            yaw_residual = np.max(
                np.abs(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))), initial=0.0
            )
            same_identity = (
                np.all(epochs[left:right + 1] == anchor.producer_epoch)
                and np.all(target_ids[left:right + 1] == anchor.target_id)
                and np.all(geometry_hashes[left:right + 1] == str(anchor.geometry_hash))
            )
            rule_query[sample_index, query_index] = (
                same_identity
                and velocity_change <= 1e-6 and yaw_rate_change <= 1e-6
                and position_residual <= 1e-4 and yaw_residual <= 1e-4
            )

    if maximum_history_timestamp_error_ns > 2_000:
        raise ValueError(
            f"float event timestamp reconstruction exceeded 2 us: "
            f"{maximum_history_timestamp_error_ns} ns"
        )
    if maximum_q0_position_error_m > 5e-5:
        raise ValueError(f"truth history frame mismatch: {maximum_q0_position_error_m} m")
    distance_m = np.linalg.norm(source["future_position"][:, 0].mean(axis=1), axis=1).astype(np.float32)
    output = {
        "truth_obs": truth_obs,
        "truth_obs_mask": truth_mask,
        "event_mask": source["event_mask"].astype(np.bool_, copy=False),
        "event_time_s": source["event_time_s"].astype(np.float32, copy=False),
        "tau": source["tau"].astype(np.float32, copy=False),
        "future_timestamp_ns": source["future_timestamp_ns"].astype(np.int64, copy=False),
        "future_position": source["future_position"].astype(np.float32, copy=False),
        "motion_class": source["motion_class"].astype(np.int64, copy=False),
        "anchor_velocity_mps": anchor_velocity,
        "anchor_yaw_rate_rad_s": anchor_yaw_rate,
        "anchor_center_position_m": anchor_center_position,
        "rule_query": rule_query,
        "distance_m": distance_m,
        "session_id": source["session_id"],
        "t0_ns": source["t0_ns"].astype(np.int64, copy=False),
    }
    segment_fields = (
        "motion_command_epoch", "motion_segment_start_ns",
        "motion_segment_end_ns", "history_start_ns", "future_end_ns",
        "window_constant_motion",
    )
    if "motion_command_epoch" in source:
        if not (
            np.all(source["window_constant_motion"])
            and np.all(source["history_start_ns"] >= source["motion_segment_start_ns"])
            and np.all(source["future_end_ns"] < source["motion_segment_end_ns"])
            and np.all(source["future_end_ns"] == source["future_timestamp_ns"].max(axis=1))
        ):
            raise ValueError("source contains a cross-segment or nonconstant model window")
        for name in segment_fields:
            output[name] = source[name].copy()
    output_shard.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_shard, **output)
    valid_values = truth_obs[truth_mask]
    return {
        "path": str(output_shard),
        "split": task["split"],
        "session_id": task["session_id"],
        "sample_count": sample_count,
        "sha256": _sha256(output_shard),
        "bytes": output_shard.stat().st_size,
        "normalization_count": int(valid_values.shape[0]),
        "normalization_sum": valid_values.astype(np.float64).sum(axis=0).tolist(),
        "normalization_sum_square": np.square(valid_values.astype(np.float64)).sum(axis=0).tolist(),
        "maximum_history_timestamp_error_ns": maximum_history_timestamp_error_ns,
        "maximum_q0_position_error_m": maximum_q0_position_error_m,
        "rule_query_count": int(rule_query.sum()),
        "query_count": int(rule_query.size),
    }


def build(args: argparse.Namespace) -> Path:
    source_dataset = Path(args.source_dataset).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite truth-history dataset: {output}")
    output.mkdir(parents=True)
    manifest_path = source_dataset / "dataset_manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema_version") != "stage3-dataset-v3" or not bool(
        source_manifest.get("qualification_passed", False)
    ):
        raise ValueError("source must be the qualified Stage-3 v3 dataset")
    raw_root = source_dataset.parent.parent
    sources = {
        str(item["session_id"]): item
        for item in _load_jsonl(source_dataset / str(source_manifest["canonical_sources"]))
    }
    selected: list[dict[str, Any]] = []
    split_counts = {"train": 0, "validation": 0}
    for shard in source_manifest["shards"]:
        split = str(shard["split"])
        if split not in split_counts:
            continue
        if args.session_limit > 0 and split_counts[split] >= args.session_limit:
            continue
        session_ids = [str(value) for value in shard["session_ids"]]
        if len(session_ids) != 1:
            raise ValueError("truth-history builder requires one-session source shards")
        session_id = session_ids[0]
        source = sources[session_id]
        truth_path = _resolve_truth(raw_root, source)
        selected.append({
            "source_shard": str(source_dataset / str(shard["path"])),
            "output_shard": str(output / "shards" / Path(str(shard["path"])).name),
            "split": split,
            "session_id": session_id,
            "truth_path": str(truth_path),
            "truth_sha256": str(source["truth_sha256"]),
        })
        split_counts[split] += 1
    if not selected or not all(split_counts.values()):
        raise ValueError(f"no complete train/validation selection: {split_counts}")

    results: list[dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_build_shard, task): task for task in selected}
            for completed, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                print(json.dumps({
                    "completed": completed,
                    "total": len(selected),
                    "session_id": result["session_id"],
                    "split": result["split"],
                    "samples": result["sample_count"],
                }), flush=True)
    except Exception:
        _write_json(output / "build_failed.json", {
            "status": "failed", "completed_shards": len(results),
            "source_dataset": str(source_dataset),
        })
        raise
    results.sort(key=lambda item: (item["split"], item["session_id"]))
    train = [item for item in results if item["split"] == "train"]
    count = sum(int(item["normalization_count"]) for item in train)
    total = np.sum([item["normalization_sum"] for item in train], axis=0, dtype=np.float64)
    total_square = np.sum(
        [item["normalization_sum_square"] for item in train], axis=0, dtype=np.float64
    )
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 1e-12)
    normalization = {
        "schema_version": "stage3-truth-history-normalization-v1",
        "source_split": "train",
        "position_m": {
            "count": count,
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
        },
    }
    normalization_path = output / "normalization.json"
    _write_json(normalization_path, normalization)
    shutil.copy2(source_dataset / "geometry_template.json", output / "geometry_template.json")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "qualification_passed": True,
        "source_dataset": str(source_dataset),
        "source_dataset_manifest_sha256": _sha256(manifest_path),
        **({
            "capture_contract": str(source_manifest["capture_contract"]),
            "capture_contract_sha256": str(source_manifest["capture_contract_sha256"]),
        } if "capture_contract_sha256" in source_manifest else {}),
        "test_accessed": False,
        "splits": ["train", "validation"],
        "session_count": len(results),
        "sample_count": sum(int(item["sample_count"]) for item in results),
        "normalization": normalization_path.name,
        "normalization_sha256": _sha256(normalization_path),
        "geometry_template": "geometry_template.json",
        "geometry_template_sha256": _sha256(output / "geometry_template.json"),
        "maximum_history_timestamp_error_ns": max(
            int(item["maximum_history_timestamp_error_ns"]) for item in results
        ),
        "maximum_q0_position_error_m": max(
            float(item["maximum_q0_position_error_m"]) for item in results
        ),
        "rule_query_fraction": (
            sum(int(item["rule_query_count"]) for item in results)
            / sum(int(item["query_count"]) for item in results)
        ),
        "rule_query_policy": {
            "same_producer_epoch_target_and_geometry": True,
            "maximum_velocity_change_mps": 1e-6,
            "maximum_yaw_rate_change_rad_s": 1e-6,
            "maximum_constant_velocity_position_residual_m": 1e-4,
            "maximum_constant_yaw_residual_rad": 1e-4,
        },
        "builder_source_sha256": {
            name: _sha256(Path(__file__).resolve().parent / name)
            for name in (
                "build_truth_history_dataset.py", "schema.py",
                "truth_history_dataset.py",
            )
        },
        "shards": [{
            "path": str(Path(item["path"]).relative_to(output)),
            "split": item["split"],
            "session_ids": [item["session_id"]],
            "sample_count": item["sample_count"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        } for item in results],
    }
    _write_json(output / "dataset_manifest.json", manifest)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--session-limit", type=int, default=0,
        help="diagnostic limit applied independently to train and validation",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.session_limit < 0:
        parser.error("workers must be positive and session-limit non-negative")
    print(build(args))


if __name__ == "__main__":
    main()
