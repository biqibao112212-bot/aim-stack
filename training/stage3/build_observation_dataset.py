"""Derive a masked future-observation training dataset from qualified v3 shards.

The raw captures and the qualified v3 dataset are immutable.  This builder
adds exact-exposure observation labels and two history quality channels to a
new v4 directory without recollecting data or opening test data during
training.  A missing exact frame is unknown; an exact frame with zero valid
candidates is an explicit all-invisible label.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .analyze_triangle_errors import _match_observation_to_truth
from .dataset import _observation_position_v3, load_camera_gimbal_extrinsic
from .schema import ObservationFrame, iter_json_records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_raw(path: Path, extrinsic: Any) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in iter_json_records(path):
        frame = ObservationFrame.from_mapping(record)
        if frame.timestamp_ns in result:
            raise ValueError(f"duplicate observation timestamp: {path}:{frame.timestamp_ns}")
        slots: list[tuple[np.ndarray | None, float]] = []
        valid_positions: list[np.ndarray] = []
        valid_quality: list[float] = []
        for armor in frame.armors:
            finite = armor.valid and all(math.isfinite(v) for v in (*armor.position_m, armor.yaw_rad))
            if not finite:
                slots.append((None, 0.0))
                continue
            position = _observation_position_v3(frame, armor.position_m, extrinsic)
            quality = armor.reprojection_rms_px
            quality_value = float(quality) if quality is not None and math.isfinite(float(quality)) else 0.0
            slots.append((position, quality_value))
            valid_positions.append(position)
            valid_quality.append(quality_value)
        result[frame.timestamp_ns] = {
            "slots": slots,
            "valid_positions": valid_positions,
            "valid_quality": valid_quality,
            "valid_count": len(valid_positions),
        }
    return result


def _augment_history(
    arrays: dict[str, np.ndarray], raw: dict[int, dict[str, Any]], index: int
) -> np.ndarray:
    original = arrays["obs"][index].astype(np.float32, copy=False)
    augmented = np.zeros((*original.shape[:2], 7), dtype=np.float32)
    augmented[..., :5] = original
    t0_ns = int(arrays["t0_ns"][index])
    for event in range(original.shape[0]):
        if not bool(arrays["event_mask"][index, event]):
            continue
        timestamp = t0_ns + int(round(float(arrays["event_time_s"][index, event]) * 1e9))
        frame = raw.get(timestamp)
        if frame is None:
            continue
        augmented[event, :, 5] = 0.0
        augmented[event, :, 6] = min(float(frame["valid_count"]), 4.0) / 4.0
        for slot, item in enumerate(frame["slots"][:4]):
            if item[0] is not None and bool(arrays["obs_mask"][index, event, slot]):
                augmented[event, slot, 5] = float(item[1])
    return augmented


def _future_labels(
    arrays: dict[str, np.ndarray], raw: dict[int, dict[str, Any]], index: int
) -> dict[str, np.ndarray]:
    query_count = int(arrays["future_position"].shape[1])
    position = np.zeros((query_count, 4, 3), dtype=np.float32)
    mask = np.zeros((query_count, 4), dtype=np.bool_)
    frame_available = np.zeros((query_count,), dtype=np.bool_)
    frame_usable = np.zeros((query_count,), dtype=np.bool_)
    ambiguous = np.zeros((query_count,), dtype=np.bool_)
    quality = np.zeros((query_count, 4), dtype=np.float32)
    for query, timestamp_raw in enumerate(arrays["future_timestamp_ns"][index]):
        frame = raw.get(int(timestamp_raw))
        if frame is None:
            continue
        frame_available[query] = True
        if int(frame["valid_count"]) > 4:
            ambiguous[query] = True
            continue
        if not frame["valid_positions"]:
            frame_usable[query] = True
            continue
        matched = _match_observation_to_truth(
            np.asarray(frame["valid_positions"], dtype=np.float64),
            arrays["future_position"][index, query].astype(np.float64),
        )
        if matched is None:
            continue
        _, assignment = matched
        frame_usable[query] = True
        for truth_slot, observation_row in assignment.items():
            position[query, truth_slot] = frame["valid_positions"][observation_row]
            quality[query, truth_slot] = frame["valid_quality"][observation_row]
            mask[query, truth_slot] = True
    return {
        "future_observation_position": position,
        "future_observation_mask": mask,
        "future_observation_frame_available": frame_available,
        "future_observation_frame_usable": frame_usable,
        "future_observation_ambiguous": ambiguous,
        "future_observation_reprojection_rms": quality,
    }


def build(args: argparse.Namespace) -> Path:
    source = Path(args.source_dataset).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v4 dataset: {output}")
    source_manifest_path = source / "dataset_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema_version") != "stage3-dataset-v3" or not source_manifest.get("qualification_passed"):
        raise ValueError("source must be a qualified stage3-dataset-v3")
    extrinsic_path = Path(str(source_manifest["camera_gimbal_extrinsic_yaml"])).resolve()
    if _sha256(extrinsic_path) != str(source_manifest["camera_gimbal_extrinsic_sha256"]):
        raise ValueError("source calibration hash mismatch")
    extrinsic = load_camera_gimbal_extrinsic(extrinsic_path)
    canonical = {
        str(item["session_id"]): item
        for item in iter_json_records(source / str(source_manifest["canonical_sources"]))
    }
    raw_root = source.parent.parent.parent / "autoaim-stage3-v1"
    output.mkdir(parents=True)
    (output / "shards").mkdir()
    (output / "build_state.json").write_text(
        json.dumps({"schema_version": "stage3-dataset-v4-observation", "status": "in_progress"}, indent=2),
        encoding="utf-8",
    )

    quality_values: list[np.ndarray] = []
    shard_manifest: list[dict[str, Any]] = []
    source_shards = [item for item in source_manifest["shards"] if item["split"] in {"train", "validation", "test"}]
    for ordinal, shard in enumerate(source_shards, 1):
        shard_path = source / str(shard["path"])
        with np.load(shard_path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        session_ids = {str(value) for value in arrays["session_id"]}
        if len(session_ids) != 1:
            raise ValueError(f"v3 shard is not one-session-per-shard: {shard_path}")
        session_id = next(iter(session_ids))
        source_record = canonical.get(session_id)
        if source_record is None:
            raise ValueError(f"canonical source missing for {session_id}")
        observation_path = Path(str(source_record["observations"])).resolve()
        if not observation_path.is_file():
            matches = list(raw_root.glob(f"*/run-*/observations.jsonl"))
            matches = [path for path in matches if path.parent.parent.name == session_id]
            if len(matches) != 1:
                raise FileNotFoundError(f"canonical observations unavailable for {session_id}")
            observation_path = matches[0].resolve()
        if _sha256(observation_path) != str(source_record["observation_sha256"]):
            raise ValueError(f"raw observation hash mismatch for {session_id}")
        raw = _load_raw(observation_path, extrinsic)
        count = len(arrays["session_id"])
        new_arrays = {key: arrays[key] for key in arrays}
        new_arrays["obs"] = np.stack([_augment_history(arrays, raw, i) for i in range(count)])
        future = {key: [] for key in (
            "future_observation_position", "future_observation_mask",
            "future_observation_frame_available", "future_observation_frame_usable",
            "future_observation_ambiguous", "future_observation_reprojection_rms",
        )}
        for i in range(count):
            labels = _future_labels(arrays, raw, i)
            for key, value in labels.items():
                future[key].append(value)
        new_arrays.update({key: np.stack(value) for key, value in future.items()})
        if shard["split"] == "train":
            valid_quality = new_arrays["obs"][new_arrays["obs_mask"]][:, 5]
            if len(valid_quality):
                quality_values.append(valid_quality.astype(np.float64))
        relative = Path(str(shard["path"]))
        relative = relative.with_name(
            relative.name.replace("train-", "obs-train-")
            .replace("validation-", "obs-validation-")
            .replace("test-", "obs-test-")
        )
        destination = output / relative
        np.savez_compressed(destination, **new_arrays)
        shard_manifest.append({
            "path": relative.as_posix(),
            "split": str(shard["split"]),
            "sample_count": count,
            "session_ids": [session_id],
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
        })
        print(json.dumps({"progress": f"{ordinal}/{len(source_shards)}", "session_id": session_id, "samples": count}, sort_keys=True), flush=True)

    quality = np.concatenate(quality_values) if quality_values else np.zeros((0,), dtype=np.float64)
    quality_mean = float(quality.mean()) if len(quality) else 0.0
    quality_std = float(quality.std()) if len(quality) else 1.0
    quality_std = max(quality_std, 1e-6)
    normalization = json.loads((source / str(source_manifest["normalization"])).read_text(encoding="utf-8"))
    normalization["schema_version"] = "stage3-normalization-v2-observation"
    normalization["obs_quality"] = {"mean": [quality_mean, 0.0], "std": [quality_std, 1.0], "features": ["reprojection_rms_px", "valid_candidate_fraction"]}
    (output / "normalization.json").write_text(json.dumps(normalization, indent=2, sort_keys=True), encoding="utf-8")
    for name in ("geometry_template.json", "qualification_report.json", "splits.json", "canonical_sources.jsonl"):
        shutil.copy2(source / name, output / name)
    artifact_paths = {
        "normalization": output / "normalization.json",
        "geometry_template": output / "geometry_template.json",
        "qualification_report": output / "qualification_report.json",
        "session_splits": output / "splits.json",
        "canonical_sources": output / "canonical_sources.jsonl",
    }
    dataset_manifest = {
        "schema_version": "stage3-dataset-v4-observation",
        "source_v3_dataset": str(source),
        "source_v3_manifest_sha256": _sha256(source_manifest_path),
        "qualification_passed": True,
        "session_count": int(source_manifest["session_count"]),
        "sample_count": int(source_manifest["sample_count"]),
        "observation_schema": source_manifest["observation_schema"],
        "truth_schema": source_manifest.get("truth_schema", "stage3-truth-v1"),
        "camera_gimbal_extrinsic_yaml": str(extrinsic_path),
        "camera_gimbal_extrinsic_sha256": _sha256(extrinsic_path),
        "tensor_contract": {
            "history_shape": [200, 4, 7],
            "history_features": ["x", "y", "z", "sin_yaw", "cos_yaw", "reprojection_rms_px", "valid_candidate_fraction"],
            "future_physical_shape": [8, 4, 3],
            "future_observation_shape": [8, 4, 3],
            "future_observation_mask_shape": [8, 4],
            "future_observation_frame_available_shape": [8],
            "missing_exact_frame_policy": "mask_all_observation_losses",
            "zero_candidate_policy": "visible_negative_label",
            "ambiguous_more_than_four_policy": "mask_all_observation_losses",
        },
        "normalization": "normalization.json",
        "geometry_template": "geometry_template.json",
        "qualification_report": "qualification_report.json",
        "splits": "splits.json",
        "canonical_sources": "canonical_sources.jsonl",
        "artifact_sha256": {name: _sha256(path) for name, path in artifact_paths.items()},
        "shards": shard_manifest,
    }
    (output / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output / "build_state.json").write_text(json.dumps({"schema_version": "stage3-dataset-v4-observation", "status": "complete", "dataset_manifest": "dataset_manifest.json"}, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--output", required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
