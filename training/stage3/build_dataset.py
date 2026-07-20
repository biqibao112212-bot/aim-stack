"""Build the canonical, qualified Stage-3 dataset from accepted sessions.

This command discovers only ``<evidence-root>/<session-id>/session_result.json``
files named by the formal manifest.  Retries, first-article captures, and other
diagnostic directories are therefore never selected by a recursive glob.
Raw captures are immutable inputs; the builder writes a new, non-overwriting
directory containing hashes, qualification evidence, split-specific tensor
shards, and train-only normalization statistics.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .dataset import (
    CameraGimbalExtrinsic,
    build_samples,
    load_camera_gimbal_extrinsic,
    samples_to_arrays,
)
from .schema import ObservationFrame, TruthFrame, iter_json_records, schema_fingerprint


DATASET_SCHEMA_VERSION = "stage3-dataset-v3"
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.60, "validation": 0.20, "test": 0.20}


def _json_dump(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _write_json(path: Path, value: object) -> None:
    path.write_text(_json_dump(value) + "\n", encoding="utf-8")


def _normalise_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _load_formal_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in iter_json_records(path):
        record = dict(raw)
        session_id = str(record.get("session_id", ""))
        if not session_id:
            raise ValueError(f"manifest record has no session_id: {path}")
        if session_id in seen:
            raise ValueError(f"duplicate manifest session_id: {session_id}")
        if str(record.get("schema_version", "")) != "stage3-manifest-v1":
            raise ValueError(f"unsupported manifest schema for {session_id}")
        seen.add(session_id)
        records.append(record)
    if not records:
        raise ValueError("formal manifest is empty")
    return records


def discover_canonical_sources(
    manifest_records: Iterable[Mapping[str, Any]], evidence_root: Path, raw_root: Path
) -> list[dict[str, Any]]:
    """Resolve one accepted raw pair per manifest session without recursion."""

    sources: list[dict[str, Any]] = []
    observation_paths: set[Path] = set()
    truth_paths: set[Path] = set()
    for manifest in manifest_records:
        session_id = str(manifest["session_id"])
        captured_manifest_path = evidence_root / f".manifest-{session_id}.json"
        if not captured_manifest_path.is_file():
            raise ValueError(f"missing captured session manifest: {captured_manifest_path}")
        captured_manifest = json.loads(captured_manifest_path.read_text(encoding="utf-8-sig"))
        if json.dumps(captured_manifest, sort_keys=True, separators=(",", ":")) != json.dumps(
            dict(manifest), sort_keys=True, separators=(",", ":")
        ):
            raise ValueError(f"captured manifest differs from master record: {session_id}")
        result_path = evidence_root / session_id / "session_result.json"
        if not result_path.is_file():
            raise ValueError(f"missing canonical session_result.json: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8-sig"))
        if str(result.get("session_id", "")) != session_id:
            raise ValueError(f"session_result id mismatch: {result_path}")
        observations = _normalise_path(str(result.get("observations", "")))
        truth = _normalise_path(str(result.get("truth", "")))
        for kind, path in (("observations", observations), ("truth", truth)):
            if not path.is_file() or path.stat().st_size <= 0:
                raise ValueError(f"canonical {kind} is missing or empty: {path}")
            if not path.is_relative_to(raw_root):
                raise ValueError(f"canonical {kind} escapes protected raw root: {path}")
            if path.parent.parent.name != session_id or not path.parent.name.startswith("run-"):
                raise ValueError(f"canonical {kind} is not inside the accepted run directory: {path}")
        if observations.name != "observations.jsonl" or truth.name != "truth.jsonl":
            raise ValueError(f"canonical raw filenames are invalid for {session_id}")
        if observations in observation_paths:
            raise ValueError(f"observation file reused by multiple sessions: {observations}")
        if truth in truth_paths:
            raise ValueError(f"truth file reused by multiple sessions: {truth}")
        observation_paths.add(observations)
        truth_paths.add(truth)
        sources.append({
            "session_id": session_id,
            "session_result": str(result_path.resolve()),
            "session_result_sha256": _sha256_file(result_path),
            "captured_manifest": str(captured_manifest_path.resolve()),
            "captured_manifest_sha256": _sha256_file(captured_manifest_path),
            "observations": str(observations),
            "truth": str(truth),
        })
    return sources


def _speed_band(value: float) -> str:
    if value < 0.02:
        return "zero"
    if value < 1.0:
        return "low"
    if value < 2.0:
        return "medium"
    return "high"


def _omega_band(value: float) -> str:
    value = abs(value)
    if value < 0.02:
        return "zero"
    if value < 5.0:
        return "low"
    if value < 10.0:
        return "medium"
    return "high"


def _tags(record: Mapping[str, Any]) -> tuple[str, ...]:
    omega = float(record.get("spin_rad_s", 0.0))
    sign = "zero" if abs(omega) < 0.02 else ("positive" if omega > 0 else "negative")
    distance_bin = int(record.get("distance_bin", max(0, min(2, int((float(record["distance_m"]) - 1.0) // 2.5)))))
    direction_sector = int(record.get("direction_sector", int(float(record.get("direction_deg", 0.0)) // 45.0) % 8))
    mode = str(record.get("mode", "unknown"))
    return (
        f"mode={mode}",
        f"distance_bin={distance_bin}",
        f"direction_sector={direction_sector}",
        f"speed_band={_speed_band(float(record.get('linear_speed_mps', 0.0)))}",
        f"omega_band={_omega_band(omega)}",
        f"spin_sign={sign}",
        f"mode_distance={mode}:{distance_bin}",
    )


def stratified_session_split(records: list[Mapping[str, Any]], seed: int) -> dict[str, list[str]]:
    """Greedily balance several discrete conditions while keeping exact sizes."""

    count = len(records)
    target_sizes = {
        "train": int(round(count * SPLIT_RATIOS["train"])),
        "validation": int(round(count * SPLIT_RATIOS["validation"])),
    }
    target_sizes["test"] = count - target_sizes["train"] - target_sizes["validation"]
    tag_totals: Counter[str] = Counter(tag for record in records for tag in _tags(record))
    desired = {
        split: {tag: total * SPLIT_RATIOS[split] for tag, total in tag_totals.items()}
        for split in SPLIT_NAMES
    }
    assigned_tags = {split: Counter() for split in SPLIT_NAMES}
    splits = {split: [] for split in SPLIT_NAMES}

    def stable_key(record: Mapping[str, Any]) -> tuple[float, str]:
        session_id = str(record["session_id"])
        rarity = -sum(1.0 / tag_totals[tag] for tag in _tags(record))
        digest = hashlib.sha256(f"{seed}:{session_id}".encode("utf-8")).hexdigest()
        return rarity, digest

    for record in sorted(records, key=stable_key):
        record_tags = _tags(record)
        candidates = [split for split in SPLIT_NAMES if len(splits[split]) < target_sizes[split]]
        if not candidates:
            raise AssertionError("split capacity exhausted before all sessions were assigned")

        def score(split: str) -> tuple[float, float, str]:
            delta = 0.0
            for tag in record_tags:
                before = assigned_tags[split][tag] - desired[split][tag]
                after = assigned_tags[split][tag] + 1 - desired[split][tag]
                delta += (after * after - before * before) / max(desired[split][tag], 1.0)
            fill = (len(splits[split]) + 1) / max(target_sizes[split], 1)
            return delta, fill, split

        selected = min(candidates, key=score)
        session_id = str(record["session_id"])
        splits[selected].append(session_id)
        assigned_tags[selected].update(record_tags)

    for values in splits.values():
        values.sort()
    all_ids = [session_id for values in splits.values() for session_id in values]
    if len(all_ids) != len(set(all_ids)) or len(all_ids) != count:
        raise AssertionError("session split is not a disjoint cover")
    return splits


def _stream_jsonl(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"record at {path}:{line_number} is not an object")
            yield value, digest


def _audit_observations(
    path: Path, session_id: str
) -> tuple[dict[str, Any], set[tuple[str, int, int, int]], list[ObservationFrame]]:
    count = zero = multiple = more_than_four = invalid_armor = 0
    keys: set[tuple[str, int, int, int]] = set()
    last_by_epoch: dict[int, tuple[int, int]] = {}
    frames: list[ObservationFrame] = []
    digest: hashlib._Hash | None = None  # type: ignore[attr-defined]
    for record, digest in _stream_jsonl(path):
        for required in (
            "schema_version", "session_id", "producer_epoch", "frame_seq", "timestamp_ns",
            "armors", "tracker_origin_world_ros_m", "tracker_gimbal_quaternion_world_wxyz",
            "camera_origin_world_ros_m", "camera_quaternion_world_wxyz",
            "gimbal_yaw_deg", "gimbal_pitch_deg",
        ):
            if required not in record:
                raise ValueError(f"observation field {required} is missing in {path}")
        if not bool(record.get("gimbal_pose_exposure_matched", False)):
            raise ValueError(f"observation has no exposure-matched gimbal pose in {path}")
        if not bool(record.get("tracker_world_transform_exposure_matched", False)):
            raise ValueError(f"observation has no exposure-matched tracker transform in {path}")
        frame = ObservationFrame.from_mapping(record)
        if frame.tracker_origin_world_m is None or frame.tracker_quaternion_world_wxyz is None:
            raise ValueError(f"observation tracker transform is missing in {path}")
        if frame.camera_origin_world_m is None or frame.camera_quaternion_world_wxyz is None:
            raise ValueError(f"observation camera transform is missing in {path}")
        if frame.session_id != session_id:
            raise ValueError(f"observation session mismatch in {path}: {frame.session_id}")
        if frame.key in keys:
            raise ValueError(f"duplicate observation key in {path}: {frame.key}")
        keys.add(frame.key)
        frames.append(frame)
        order = (frame.timestamp_ns, frame.frame_seq)
        previous = last_by_epoch.get(frame.producer_epoch)
        if previous is not None and order <= previous:
            raise ValueError(f"non-monotonic observation stream in {path}")
        last_by_epoch[frame.producer_epoch] = order
        count += 1
        zero += int(len(frame.armors) == 0)
        multiple += int(len(frame.armors) > 1)
        more_than_four += int(len(frame.armors) > 4)
        invalid_armor += sum(
            int(not armor.valid or not all(math.isfinite(value) for value in (*armor.position_m, armor.yaw_rad)))
            for armor in frame.armors
        )
    if digest is None or count == 0:
        raise ValueError(f"observation stream is empty: {path}")
    return ({
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "records": count,
        "zero_candidate_records": zero,
        "multiple_candidate_records": multiple,
        "more_than_four_candidate_records": more_than_four,
        "invalid_armor_records": invalid_armor,
    }, keys, frames)


def _quaternion_angle_deg(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


def _audit_truth(
    path: Path, session_id: str, max_ego_translation_drift_m: float,
    max_ego_angle_drift_deg: float, min_stable_duration_s: float,
) -> tuple[
    dict[str, Any], set[tuple[str, int, int, int]], list[TruthFrame]
]:
    exact = unavailable = 0
    keys: set[tuple[str, int, int, int]] = set()
    last_by_epoch: dict[int, tuple[int, int]] = {}
    geometry_hashes: set[str] = set()
    first_origin: tuple[float, float, float] | None = None
    first_quaternion: tuple[float, float, float, float] | None = None
    max_translation = 0.0
    max_angle = 0.0
    poses: list[tuple[int, tuple[float, float, float], tuple[float, float, float, float]]] = []
    geometry_template: dict[str, Any] | None = None
    frames: list[TruthFrame] = []
    digest: hashlib._Hash | None = None  # type: ignore[attr-defined]
    for record, digest in _stream_jsonl(path):
        if not bool(record.get("has_exact_exposure_truth", False)):
            unavailable += 1
            continue
        frame = TruthFrame.from_mapping(record)
        if frame.session_id != session_id:
            raise ValueError(f"truth session mismatch in {path}: {frame.session_id}")
        key = (frame.session_id, frame.producer_epoch, frame.frame_seq, frame.timestamp_ns)
        if key in keys:
            raise ValueError(f"duplicate truth key in {path}: {key}")
        keys.add(key)
        frames.append(frame)
        order = (frame.timestamp_ns, frame.frame_seq)
        previous = last_by_epoch.get(frame.producer_epoch)
        if previous is not None and order <= previous:
            raise ValueError(f"non-monotonic truth stream in {path}")
        last_by_epoch[frame.producer_epoch] = order
        exact += 1
        if frame.geometry_hash is None:
            raise ValueError(f"truth geometry hash is missing in {path}")
        geometry_hashes.add(frame.geometry_hash)
        if geometry_template is None:
            ground_truth = record["ground_truth"]
            targets = ground_truth["targets"]
            target = targets[int(record["selected_target_index"])]
            armors = sorted(target["armors"], key=lambda item: int(item["relative_slot"]))
            geometry_template = {
                "geometry_hash": frame.geometry_hash,
                "radius_even_m": float(target["radius_even_m"]),
                "radius_odd_m": float(target["radius_odd_m"]),
                "armor_height_m": float(target["armor_height_m"]),
                "armors": [{
                    "relative_slot": int(item["relative_slot"]),
                    "relative_position_m": [float(value) for value in item["relative_position_m"]],
                    "outward_normal": [float(value) for value in item["outward_normal"]],
                    "relative_yaw_rad": float(item.get("relative_yaw_rad", 0.0)),
                } for item in armors],
            }
        if first_origin is None:
            first_origin = frame.chassis_origin_world_m
            first_quaternion = frame.chassis_quaternion_world_wxyz
        assert first_quaternion is not None
        translation = math.sqrt(sum(
            (current - initial) ** 2
            for current, initial in zip(frame.chassis_origin_world_m, first_origin)
        ))
        max_translation = max(max_translation, translation)
        max_angle = max(max_angle, _quaternion_angle_deg(
            frame.chassis_quaternion_world_wxyz, first_quaternion
        ))
        poses.append((
            frame.timestamp_ns, frame.chassis_origin_world_m,
            frame.chassis_quaternion_world_wxyz,
        ))
    if digest is None or exact == 0:
        raise ValueError(f"truth stream contains no exact exposure truth: {path}")
    if len(geometry_hashes) != 1:
        raise ValueError(f"truth geometry drift in {path}: {sorted(geometry_hashes)}")
    reference_origin = poses[-1][1]
    reference_quaternion = poses[-1][2]
    last_unstable = -1
    stable_translation = 0.0
    stable_angle = 0.0
    for index, (_, origin, quaternion) in enumerate(poses):
        translation = math.sqrt(sum((a - b) ** 2 for a, b in zip(origin, reference_origin)))
        angle = _quaternion_angle_deg(quaternion, reference_quaternion)
        if translation > max_ego_translation_drift_m or angle > max_ego_angle_drift_deg:
            last_unstable = index
    stable_start_index = last_unstable + 1
    stable_start_ns = poses[stable_start_index][0]
    for _, origin, quaternion in poses[stable_start_index:]:
        stable_translation = max(stable_translation, math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(origin, reference_origin)
        )))
        stable_angle = max(stable_angle, _quaternion_angle_deg(quaternion, reference_quaternion))
    stable_duration_s = (poses[-1][0] - stable_start_ns) / 1e9
    if stable_duration_s < min_stable_duration_s:
        raise ValueError(
            f"fixed-ego stable suffix is too short in {session_id}: {stable_duration_s:.3f} s"
        )
    return ({
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "exact_records": exact,
        "unavailable_records": unavailable,
        "geometry_hash": next(iter(geometry_hashes)),
        "total_pose_change_from_first_m": max_translation,
        "total_pose_change_from_first_deg": max_angle,
        "stable_start_timestamp_ns": stable_start_ns,
        "unstable_prefix_exact_records": stable_start_index,
        "stable_duration_s": stable_duration_s,
        "max_stable_ego_translation_drift_m": stable_translation,
        "max_stable_ego_angle_drift_deg": stable_angle,
        "geometry_template": geometry_template,
    }, keys, frames)


@dataclass
class RunningVectorStats:
    width: int
    count: int = 0

    def __post_init__(self) -> None:
        self.total = np.zeros((self.width,), dtype=np.float64)
        self.total_square = np.zeros((self.width,), dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape(-1, self.width)
        if flat.size == 0:
            return
        self.count += flat.shape[0]
        self.total += flat.sum(axis=0)
        self.total_square += np.square(flat).sum(axis=0)

    def finish(self) -> dict[str, Any]:
        if self.count == 0:
            raise ValueError("cannot finalize empty normalization statistics")
        mean = self.total / self.count
        variance = np.maximum(self.total_square / self.count - np.square(mean), 1e-12)
        return {"count": self.count, "mean": mean.tolist(), "std": np.sqrt(variance).tolist()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _geometry_numeric_vector(template: Mapping[str, Any]) -> np.ndarray:
    values = [
        float(template["radius_even_m"]),
        float(template["radius_odd_m"]),
        float(template["armor_height_m"]),
    ]
    for armor in sorted(template["armors"], key=lambda item: int(item["relative_slot"])):
        values.extend(float(value) for value in armor["relative_position_m"])
        values.extend(float(value) for value in armor["outward_normal"])
        values.append(float(armor["relative_yaw_rad"]))
    return np.asarray(values, dtype=np.float64)


def _flush_shard(
    output: Path, split: str, shard_index: int, samples: list[Any], session_ids: list[str]
) -> dict[str, Any]:
    arrays = samples_to_arrays(samples)
    relative = Path("shards") / f"{split}-{shard_index:04d}.npz"
    path = output / relative
    np.savez_compressed(path, **arrays)
    return {
        "path": relative.as_posix(),
        "split": split,
        "sample_count": len(samples),
        "session_ids": sorted(session_ids),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _process_session_to_shard(task: Mapping[str, Any]) -> dict[str, Any]:
    source = task["source"]
    manifest = task["manifest"]
    session_id = str(source["session_id"])
    split = str(task["split"])
    observations_path = Path(source["observations"])
    truth_path = Path(source["truth"])
    observation_audit, observation_keys, observation_frames = _audit_observations(
        observations_path, session_id
    )
    truth_audit, truth_keys, truth_frames = _audit_truth(
        truth_path, session_id, float(task["max_ego_translation_drift_m"]),
        float(task["max_ego_angle_drift_deg"]), float(task["min_stable_duration_s"]),
    )
    exact_coverage = len(observation_keys & truth_keys) / max(len(observation_keys), 1)
    if exact_coverage != 1.0:
        raise ValueError(
            f"observation/truth exact-key coverage is not 100% for {session_id}: "
            f"{exact_coverage:.9f}"
        )
    seed = int(task["seed"])
    session_seed = seed ^ int.from_bytes(
        hashlib.sha256(session_id.encode("utf-8")).digest()[:4], "little"
    )
    diagnostics: dict[str, object] = {}
    extrinsic = CameraGimbalExtrinsic(
        np.asarray(task["R_camera2gimbal"], dtype=np.float64),
        np.asarray(task["t_camera2gimbal_m"], dtype=np.float64),
    )
    samples = build_samples(
        observations_path, truth_path, {session_id: manifest}, extrinsic=extrinsic,
        augment=False,
        seed=session_seed, diagnostics=diagnostics,
        min_history_timestamp_ns=int(truth_audit["stable_start_timestamp_ns"]),
        preloaded_observations=observation_frames, preloaded_truth=truth_frames,
    )
    rejection_counts = diagnostics.get("rejection_counts", Counter())
    if isinstance(rejection_counts, Counter):
        diagnostics["rejection_counts"] = dict(sorted(rejection_counts.items()))
    session_report = {
        "session_id": session_id,
        "split": split,
        "manifest": manifest,
        "observations": observation_audit,
        "truth": truth_audit,
        "exact_anchor_key_coverage": exact_coverage,
        "tensorization": diagnostics,
    }
    canonical_index = {
        **source,
        "observation_bytes": observation_audit["bytes"],
        "observation_sha256": observation_audit["sha256"],
        "truth_bytes": truth_audit["bytes"],
        "truth_sha256": truth_audit["sha256"],
    }
    shard = None
    if samples:
        arrays = samples_to_arrays(samples)
        relative = Path("shards") / f"{split}-{session_id}.npz"
        path = Path(task["output"]) / relative
        np.savez_compressed(path, **arrays)
        shard = {
            "path": relative.as_posix(),
            "split": split,
            "sample_count": len(samples),
            "session_ids": [session_id],
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return {
        "session_report": session_report,
        "canonical_index": canonical_index,
        "shard": shard,
    }


def build_dataset(args: argparse.Namespace) -> Path:
    manifest_path = _normalise_path(args.manifest)
    evidence_root = _normalise_path(args.evidence_root)
    raw_root = _normalise_path(args.raw_root)
    output = _normalise_path(args.output)
    extrinsic_yaml = _normalise_path(args.extrinsic_yaml)
    extrinsic = load_camera_gimbal_extrinsic(extrinsic_yaml)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dataset output: {output}")
    output.mkdir(parents=True)
    (output / "shards").mkdir()
    state_path = output / "build_state.json"
    _write_json(state_path, {"schema_version": DATASET_SCHEMA_VERSION, "status": "in_progress"})

    manifest_records = _load_formal_manifest(manifest_path)
    if args.max_sessions > 0:
        manifest_records = manifest_records[: args.max_sessions]
    sources = discover_canonical_sources(manifest_records, evidence_root, raw_root)
    records_by_id = {str(record["session_id"]): record for record in manifest_records}
    splits = stratified_session_split(manifest_records, args.seed)
    split_by_session = {
        session_id: split for split, session_ids in splits.items() for session_id in session_ids
    }
    _write_json(output / "splits.json", {
        "schema_version": "stage3-session-splits-v2",
        "seed": args.seed,
        "strategy": "greedy-multitag-session-disjoint-60-20-20",
        **splits,
    })

    shard_manifest: list[dict[str, Any]] = []
    session_reports: list[dict[str, Any]] = []
    canonical_index: list[dict[str, Any]] = []
    obs_stats = RunningVectorStats(3)
    target_stats = RunningVectorStats(3)
    tau_stats = RunningVectorStats(1)

    tasks = [{
        "source": source,
        "manifest": records_by_id[str(source["session_id"])],
        "split": split_by_session[str(source["session_id"])],
        "seed": args.seed,
        "max_ego_translation_drift_m": args.max_ego_translation_drift_m,
        "max_ego_angle_drift_deg": args.max_ego_angle_drift_deg,
        "min_stable_duration_s": args.min_stable_duration_s,
        "output": str(output),
        "R_camera2gimbal": extrinsic.rotation.tolist(),
        "t_camera2gimbal_m": extrinsic.translation_m.tolist(),
    } for source in sources]

    def consume(results: Iterable[dict[str, Any]]) -> None:
        for ordinal, result in enumerate(results, 1):
            report = result["session_report"]
            shard = result["shard"]
            session_reports.append(report)
            canonical_index.append(result["canonical_index"])
            if shard is not None:
                shard_manifest.append(shard)
                if shard["split"] == "train":
                    with np.load(output / shard["path"], allow_pickle=False) as arrays:
                        valid_obs = arrays["obs"][arrays["obs_mask"]]
                        obs_stats.update(valid_obs[:, :3])
                        target_stats.update(arrays["future_position"])
                        tau_stats.update(arrays["tau"][..., None])
            print(json.dumps({
                "progress": f"{ordinal}/{len(sources)}",
                "session_id": report["session_id"],
                "split": report["split"],
                "samples": int(report["tensorization"].get("sample_count", 0)),
            }, sort_keys=True), flush=True)

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            consume(executor.map(_process_session_to_shard, tasks, chunksize=1))
    else:
        consume(map(_process_session_to_shard, tasks))

    with (output / "canonical_sources.jsonl").open("w", encoding="utf-8") as handle:
        for record in canonical_index:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    templates = [report["truth"]["geometry_template"] for report in session_reports]
    geometry_hashes = {str(value["geometry_hash"]) for value in templates}
    if len(geometry_hashes) != 1:
        raise ValueError(f"fixed geometry hash differs across canonical sessions: {sorted(geometry_hashes)}")
    canonical_geometry = _geometry_numeric_vector(templates[0])
    maximum_geometry_delta = max(
        float(np.max(np.abs(_geometry_numeric_vector(value) - canonical_geometry)))
        for value in templates
    )
    if maximum_geometry_delta > args.max_geometry_numeric_drift:
        raise ValueError(
            "fixed geometry numeric drift exceeds tolerance: "
            f"{maximum_geometry_delta:.9g} > {args.max_geometry_numeric_drift:.9g}"
        )
    _write_json(output / "geometry_template.json", {
        "schema_version": "stage3-geometry-template-v1",
        "maximum_cross_session_numeric_delta": maximum_geometry_delta,
        "numeric_drift_tolerance": args.max_geometry_numeric_drift,
        **templates[0],
    })

    normalization = {
        "schema_version": "stage3-normalization-v1",
        "source_split": "train",
        "obs_xyz": {**obs_stats.finish(), "applied": True},
        "future_position": {**target_stats.finish(), "applied": False, "unit": "metre"},
        "tau_effective_s": {**tau_stats.finish(), "applied": False, "unit": "second"},
    }
    _write_json(output / "normalization.json", normalization)

    split_sample_counts = Counter()
    for shard in shard_manifest:
        split_sample_counts[str(shard["split"])] += int(shard["sample_count"])
    zero_sample_sessions = [
        report["session_id"] for report in session_reports
        if int(report["tensorization"].get("sample_count", 0)) == 0
    ]
    zero_sample_fraction = len(zero_sample_sessions) / max(len(session_reports), 1)
    qualification_passed = (
        zero_sample_fraction <= args.max_zero_sample_fraction and
        all(split_sample_counts[name] > 0 for name in SPLIT_NAMES)
    )
    qualification = {
        "schema_version": "stage3-qualification-v2",
        "qualification_passed": qualification_passed,
        "session_count": len(session_reports),
        "zero_sample_sessions": zero_sample_sessions,
        "zero_sample_fraction": zero_sample_fraction,
        "max_zero_sample_fraction": args.max_zero_sample_fraction,
        "split_session_counts": {name: len(splits[name]) for name in SPLIT_NAMES},
        "split_sample_counts": dict(split_sample_counts),
        "sessions": session_reports,
    }
    _write_json(output / "qualification_report.json", qualification)
    artifact_paths = {
        "formal_manifest": manifest_path,
        "canonical_sources": output / "canonical_sources.jsonl",
        "session_splits": output / "splits.json",
        "qualification_report": output / "qualification_report.json",
        "normalization": output / "normalization.json",
        "geometry_template": output / "geometry_template.json",
    }
    dataset_manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "observation_schema": ["stage3-observation-v1", "stage3-observation-v2"],
        "truth_schema": "stage3-truth-v1",
        "schema_fingerprint": schema_fingerprint(),
        "seed": args.seed,
        "manifest": str(manifest_path),
        "evidence_root": str(evidence_root),
        "raw_root": str(raw_root),
        "camera_gimbal_extrinsic_yaml": str(extrinsic_yaml),
        "camera_gimbal_extrinsic_sha256": _sha256_file(extrinsic_yaml),
        "session_count": len(session_reports),
        "sample_count": int(sum(split_sample_counts.values())),
        "shard_strategy": "one-session-per-shard",
        "build_workers": args.workers,
        "tensor_contract": {
            "history_selection": "latest_200_valid_observation_events",
            "minimum_recent_history_seconds": 0.2,
            "maximum_latest_valid_age_seconds": 0.05,
            "maximum_armors_per_frame": 4,
            "observation_shape": [200, 4, 5],
            "event_mask_shape": [200],
            "event_time_shape": [200],
            "event_time_semantics": "real observation timestamp minus anchor timestamp in seconds",
            "padding": "left_only; event_mask=false",
            "target_shape": [8, 4, 3],
            "query_tau_requested_seconds": [0.0, 0.1, 0.2, 0.5],
            "additional_random_queries_per_anchor": 4,
            "random_query_range_seconds": [0.0, 0.5],
            "random_query_seed_key": "seed/session/epoch/frame/timestamp",
            "query_tau_semantics": "effective matched future timestamp minus anchor timestamp",
            "history_position_semantics": "calibrated camera-gimbal extrinsic v1; legacy observation v1 is reversibly migrated",
            "target_origin": "anchor exposure gimbal pivot from tracker_origin_world_ros_m",
            "target_axes": "anchor exposure chassis forward-left-up",
            "unit_position": "metre",
            "unit_time": "second",
        },
        "split_session_counts": {name: len(splits[name]) for name in SPLIT_NAMES},
        "split_sample_counts": dict(split_sample_counts),
        "shards": shard_manifest,
        "qualification_passed": qualification_passed,
        "normalization": "normalization.json",
        "canonical_sources": "canonical_sources.jsonl",
        "qualification_report": "qualification_report.json",
        "geometry_template": "geometry_template.json",
        "artifact_sha256": {
            name: _sha256_file(path) for name, path in artifact_paths.items()
        },
        "builder_source_sha256": {
            name: _sha256_file(Path(__file__).with_name(name))
            for name in ("build_dataset.py", "dataset.py", "schema.py")
        },
    }
    _write_json(output / "dataset_manifest.json", dataset_manifest)
    _write_json(state_path, {
        "schema_version": DATASET_SCHEMA_VERSION,
        "status": "complete" if qualification_passed else "qualification_failed",
        "dataset_manifest": "dataset_manifest.json",
    })
    print(_json_dump(dataset_manifest), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extrinsic-yaml", required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--sessions-per-shard", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--max-ego-translation-drift-m", type=float, default=1e-4)
    parser.add_argument("--max-ego-angle-drift-deg", type=float, default=0.01)
    parser.add_argument("--min-stable-duration-s", type=float, default=2.0)
    parser.add_argument("--max-zero-sample-fraction", type=float, default=0.10)
    parser.add_argument("--max-geometry-numeric-drift", type=float, default=1e-4)
    args = parser.parse_args()
    if args.sessions_per_shard <= 0:
        parser.error("--sessions-per-shard must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_sessions and args.max_sessions < 5:
        parser.error("--max-sessions must be zero or at least five")
    build_dataset(args)


if __name__ == "__main__":
    main()
