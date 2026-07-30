"""Build anonymous observable-target F labels from qualified clean truth.

The source r4 endpoints are too sparse to unwrap visibility switches.  This
builder joins each r4 row to its qualified truth-history anchor state and uses
that label-only physical truth to sample a 1 ms future stream.  The analytic
rollout is checked against every eligible exact endpoint.  No rollout equation
or physical identity is exported to the learned F model.
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

from .observable_future_dataset import (
    DEFAULT_CANDIDATE_STEPS,
    SCHEMA_VERSION,
    VISIBILITY_POLICY,
    construct_observable_future_sample,
)


SEGMENT_AUDIT_FIELDS = (
    "motion_command_epoch",
    "motion_segment_start_ns",
    "motion_segment_end_ns",
    "history_start_ns",
    "future_end_ns",
    "window_constant_motion",
)


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


def _rollout_dense_truth(
    q0_position_m: np.ndarray,
    center_m: np.ndarray,
    velocity_mps: np.ndarray,
    yaw_rate_rad_s: float,
    dense_time_s: np.ndarray,
) -> np.ndarray:
    offset = q0_position_m.astype(np.float64) - center_m.astype(np.float64)[None, :]
    angle = float(yaw_rate_rad_s) * dense_time_s
    cosine = np.cos(angle)
    sine = np.sin(angle)
    result = np.empty((dense_time_s.size, 4, 3), dtype=np.float64)
    result[..., 0] = (
        center_m[0] + velocity_mps[0] * dense_time_s[:, None]
        + offset[None, :, 0] * cosine[:, None]
        - offset[None, :, 1] * sine[:, None]
    )
    result[..., 1] = (
        center_m[1] + velocity_mps[1] * dense_time_s[:, None]
        + offset[None, :, 0] * sine[:, None]
        + offset[None, :, 1] * cosine[:, None]
    )
    result[..., 2] = (
        center_m[2] + velocity_mps[2] * dense_time_s[:, None]
        + offset[None, :, 2]
    )
    return result.astype(np.float32)


def _dense_times(tau_s: np.ndarray, step_s: float) -> np.ndarray:
    maximum = float(np.max(tau_s))
    regular = np.arange(0.0, maximum + step_s, step_s, dtype=np.float64)
    regular = regular[regular <= maximum + 1e-12]
    merged = np.unique(np.concatenate((regular, tau_s.astype(np.float64))))
    if merged[0] != 0.0:
        merged = np.concatenate((np.zeros(1, dtype=np.float64), merged))
    return merged


def _join_truth_rows(
    source: dict[str, np.ndarray], truth: dict[str, np.ndarray]
) -> np.ndarray:
    truth_index: dict[tuple[str, int], int] = {}
    for index, (session, t0) in enumerate(zip(truth["session_id"], truth["t0_ns"])):
        key = (str(session), int(t0))
        if key in truth_index:
            raise ValueError(f"duplicate truth-history anchor: {key}")
        truth_index[key] = index
    result = np.empty(source["t0_ns"].shape[0], dtype=np.int64)
    for index, (session, t0) in enumerate(zip(source["session_id"], source["t0_ns"])):
        key = (str(session), int(t0))
        if key not in truth_index:
            raise ValueError(f"r4 row has no truth-history anchor: {key}")
        result[index] = truth_index[key]
    return result


def _build_shard(task: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(task["source_shard"])
    truth_path = Path(task["truth_shard"])
    output_path = Path(task["output_shard"])
    with np.load(source_path, allow_pickle=False) as loaded:
        source = {key: loaded[key] for key in loaded.files}
    with np.load(truth_path, allow_pickle=False) as loaded:
        truth = {key: loaded[key] for key in loaded.files}
    truth_rows = _join_truth_rows(source, truth)
    samples: list[dict[str, np.ndarray]] = []
    motion_class: list[int] = []
    drop_counts: dict[str, int] = {}
    maximum_rollout_error_m = 0.0
    minimum_switch = 0
    maximum_switch = 0
    uncovered_queries = 0
    eligible_queries = 0
    for source_index, truth_index in enumerate(truth_rows):
        tau = source["tau"][source_index].astype(np.float32, copy=False)
        zero = np.flatnonzero(tau == 0.0)
        if zero.size < 1:
            raise ValueError("source row does not contain an exact q0")
        q0_position = source["future_position"][source_index, int(zero[0])]
        dense_time = _dense_times(tau, float(task["dense_step_s"]))
        dense_position = _rollout_dense_truth(
            q0_position,
            truth["anchor_center_position_m"][truth_index],
            truth["anchor_velocity_mps"][truth_index],
            float(truth["anchor_yaw_rate_rad_s"][truth_index]),
            dense_time,
        )
        # Preserve the exact captured truth at every sparse query while using
        # the qualified analytic state only to fill the gaps between queries.
        endpoint_error: list[float] = []
        for query, query_time in enumerate(tau):
            distance = np.abs(dense_time - float(query_time))
            minimum = float(distance.min())
            matches = np.flatnonzero(distance == minimum)
            if matches.size != 1 or minimum > float(task["query_tolerance_s"]):
                raise ValueError("dense time construction lost a source query")
            dense_index = int(matches[0])
            error = np.linalg.norm(
                dense_position[dense_index] - source["future_position"][source_index, query],
                axis=-1,
            ).max()
            if bool(source["rule_query"][source_index, query]):
                endpoint_error.append(float(error))
            dense_position[dense_index] = source["future_position"][source_index, query]
        row_error = max(endpoint_error, default=0.0)
        maximum_rollout_error_m = max(maximum_rollout_error_m, row_error)
        if row_error > float(task["maximum_rollout_error_m"]):
            reason = "rollout_endpoint_mismatch"
            drop_counts[reason] = drop_counts.get(reason, 0) + 1
            continue
        if int(source["motion_class"][source_index]) != int(
            truth["motion_class"][truth_index]
        ):
            raise ValueError("r4 and truth-history motion classes disagree")
        has_segment_audit = "motion_command_epoch" in source
        if has_segment_audit:
            if any(name not in source or name not in truth for name in SEGMENT_AUDIT_FIELDS):
                raise ValueError("multistate source is missing a segment audit field")
            for name in SEGMENT_AUDIT_FIELDS:
                if not np.array_equal(source[name][source_index], truth[name][truth_index]):
                    raise ValueError(f"r4 and truth-history segment audit differs: {name}")
            if (
                not bool(source["window_constant_motion"][source_index])
                or not bool(np.all(source["rule_query"][source_index]))
                or not bool(np.all(truth["rule_query"][truth_index]))
            ):
                raise ValueError("multistate observable row is not one fully valid motion epoch")
        try:
            sample = construct_observable_future_sample(
                source["history_position_m"][source_index],
                source["event_mask"][source_index],
                source["event_time_s"][source_index],
                dense_position,
                dense_time,
                tau,
                source["rule_query"][source_index],
                history_events=32,
                candidate_steps=DEFAULT_CANDIDATE_STEPS,
                tie_epsilon_m=float(task["tie_epsilon_m"]),
                query_match_tolerance_s=float(task["query_tolerance_s"]),
            )
        except ValueError as error:
            reason = str(error).replace(" ", "_")[:96]
            drop_counts[reason] = drop_counts.get(reason, 0) + 1
            continue
        sample["session_id"] = np.asarray(str(source["session_id"][source_index]))
        sample["t0_ns"] = np.asarray(source["t0_ns"][source_index], dtype=np.int64)
        sample["rule_query"] = source["rule_query"][source_index].astype(
            np.bool_, copy=True
        )
        if has_segment_audit:
            for name in SEGMENT_AUDIT_FIELDS:
                sample[name] = np.asarray(source[name][source_index]).copy()
        eligible = source["rule_query"][source_index].astype(np.bool_, copy=False)
        covered = sample["target_query_mask"]
        eligible_queries += int(eligible.sum())
        uncovered_queries += int((eligible & ~covered).sum())
        supervised_switch = sample["target_switch_count"][sample["target_query_mask"]]
        minimum_switch = min(minimum_switch, int(supervised_switch.min(initial=0)))
        maximum_switch = max(maximum_switch, int(supervised_switch.max(initial=0)))
        samples.append(sample)
        motion_class.append(int(source["motion_class"][source_index]))
    if not samples:
        raise ValueError(f"observable F shard produced no samples: {source_path}")
    arrays = {
        key: np.stack([sample[key] for sample in samples], axis=0)
        for key in samples[0]
    }
    arrays["motion_class"] = np.asarray(motion_class, dtype=np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return {
        "path": str(output_path),
        "split": task["split"],
        "session_id": task["session_id"],
        "source_sample_count": int(source["t0_ns"].shape[0]),
        "sample_count": len(samples),
        "sha256": _sha256(output_path),
        "bytes": output_path.stat().st_size,
        "drop_counts": drop_counts,
        "maximum_rollout_error_m": maximum_rollout_error_m,
        "minimum_switch_count": minimum_switch,
        "maximum_switch_count": maximum_switch,
        "eligible_query_count": eligible_queries,
        "uncovered_query_count": uncovered_queries,
        "segment_audited_sample_count": (
            len(samples) if "motion_command_epoch" in arrays else 0
        ),
    }


def build(args: argparse.Namespace) -> Path:
    source_dir = Path(args.source_dataset).resolve()
    output_dir = Path(args.output).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite observable F dataset: {output_dir}")
    source_manifest_path = source_dir / "dataset_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema_version") != "stage3-causal-physical-v1":
        raise ValueError("source must be the causal physical r4 schema")
    if not bool(source_manifest.get("qualification_passed", False)):
        raise ValueError("source causal physical dataset is not qualified")
    if bool(source_manifest.get("test_accessed", True)):
        raise ValueError("source causal physical dataset accessed test")
    truth_dir = Path(str(source_manifest["source_truth_history"])).resolve()
    truth_manifest_path = truth_dir / "dataset_manifest.json"
    truth_manifest = json.loads(truth_manifest_path.read_text(encoding="utf-8"))
    if truth_manifest.get("schema_version") != "stage3-truth-history-v1":
        raise ValueError("bound truth-history schema mismatch")
    if bool(truth_manifest.get("test_accessed", True)):
        raise ValueError("bound truth-history dataset accessed test")
    truth_by_session: dict[str, dict[str, Any]] = {}
    for item in truth_manifest["shards"]:
        if item["split"] not in {"train", "validation"}:
            continue
        for session in item["session_ids"]:
            if str(session) in truth_by_session:
                raise ValueError(f"duplicate truth-history session shard: {session}")
            truth_by_session[str(session)] = item

    selected: list[dict[str, Any]] = []
    split_session_count = {"train": 0, "validation": 0}
    for item in source_manifest["shards"]:
        split = str(item["split"])
        if split not in split_session_count:
            continue
        if args.session_limit > 0 and split_session_count[split] >= args.session_limit:
            continue
        sessions = [str(value) for value in item["session_ids"]]
        if len(sessions) != 1:
            raise ValueError("observable F builder requires one-session shards")
        session = sessions[0]
        if session not in truth_by_session:
            raise ValueError(f"missing truth-history session: {session}")
        selected.append({
            "source_shard": str(source_dir / str(item["path"])),
            "truth_shard": str(truth_dir / str(truth_by_session[session]["path"])),
            "output_shard": str(output_dir / "shards" / Path(str(item["path"])).name),
            "split": split,
            "session_id": session,
            "dense_step_s": float(args.dense_step_ms) / 1000.0,
            "tie_epsilon_m": float(args.tie_epsilon_m),
            "query_tolerance_s": float(args.query_tolerance_us) / 1e6,
            "maximum_rollout_error_m": float(args.maximum_rollout_error_m),
        })
        split_session_count[split] += 1
    if not selected or not all(split_session_count.values()):
        raise ValueError(f"no complete train/validation selection: {split_session_count}")

    output_dir.mkdir(parents=True)
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
                    "samples": result["sample_count"],
                }), flush=True)
    except Exception:
        _write_json(output_dir / "build_failed.json", {
            "status": "failed",
            "completed_shards": len(results),
            "source_dataset": str(source_dir),
        })
        raise
    results.sort(key=lambda item: (item["split"], item["session_id"]))
    drop_counts: dict[str, int] = {}
    for item in results:
        for reason, count in item["drop_counts"].items():
            drop_counts[reason] = drop_counts.get(reason, 0) + int(count)
    minimum_switch = min(int(item["minimum_switch_count"]) for item in results)
    maximum_switch = max(int(item["maximum_switch_count"]) for item in results)
    if minimum_switch < min(DEFAULT_CANDIDATE_STEPS) or maximum_switch > max(
        DEFAULT_CANDIDATE_STEPS
    ):
        raise ValueError("observed switch count exceeds the declared candidate head")
    uncovered = sum(int(item["uncovered_query_count"]) for item in results)
    eligible = sum(int(item["eligible_query_count"]) for item in results)
    segment_audited_count = sum(
        int(item["segment_audited_sample_count"]) for item in results
    )
    sample_count = sum(int(item["sample_count"]) for item in results)
    if segment_audited_count not in {0, sample_count}:
        raise ValueError("observable dataset mixes ACK-audited and legacy rows")
    segment_audit_enabled = segment_audited_count == sample_count
    formal_segment_audit_required = "capture_contract_sha256" in source_manifest
    if formal_segment_audit_required and not segment_audit_enabled:
        raise ValueError("frozen multistate capture lost segment audit provenance")
    if uncovered:
        raise ValueError(f"candidate head leaves {uncovered}/{eligible} eligible queries uncovered")
    shutil.copy2(source_dir / "geometry_template.json", output_dir / "geometry_template.json")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "qualification_passed": True,
        "test_accessed": False,
        "splits": ["train", "validation"],
        "source_dataset": str(source_dir),
        "source_dataset_manifest_sha256": _sha256(source_manifest_path),
        "truth_history_dataset": str(truth_dir),
        "truth_history_manifest_sha256": _sha256(truth_manifest_path),
        **({
            "capture_contract": str(source_manifest["capture_contract"]),
            "capture_contract_sha256": str(source_manifest["capture_contract_sha256"]),
        } if formal_segment_audit_required else {}),
        "sample_count": sample_count,
        "source_sample_count": sum(int(item["source_sample_count"]) for item in results),
        "session_count": len(results),
        "history_events": 32,
        "candidate_steps": list(DEFAULT_CANDIDATE_STEPS),
        "minimum_switch_count": minimum_switch,
        "maximum_switch_count": maximum_switch,
        "visibility_policy": VISIBILITY_POLICY,
        "dense_label_policy": {
            "source": "qualified q0 physical truth plus truth-only constant-motion anchor state",
            "step_ms": float(args.dense_step_ms),
            "inference_physics_decoder": False,
            "maximum_endpoint_rollout_error_m": max(
                float(item["maximum_rollout_error_m"]) for item in results
            ),
            "maximum_allowed_endpoint_rollout_error_m": float(args.maximum_rollout_error_m),
        },
        "drop_counts": drop_counts,
        "eligible_query_count": eligible,
        "uncovered_query_count": uncovered,
        "physical_identity_exported": False,
        "segment_audit": {
            "enabled": segment_audit_enabled,
            "required_by_capture_contract": formal_segment_audit_required,
            "fields": list(SEGMENT_AUDIT_FIELDS),
            "source": str(source_dir),
            "source_manifest_sha256": _sha256(source_manifest_path),
            "audited_sample_count": segment_audited_count,
            "join_mismatch_count": 0,
            "window_constant_motion_required": segment_audit_enabled,
            "all_rule_query_required": segment_audit_enabled,
        },
        "geometry_template": "geometry_template.json",
        "geometry_template_sha256": _sha256(output_dir / "geometry_template.json"),
        "builder_source_sha256": {
            "build_observable_future_dataset.py": _sha256(Path(__file__)),
            "observable_future_dataset.py": _sha256(
                Path(__file__).with_name("observable_future_dataset.py")
            ),
        },
        "shards": [{
            "path": str(Path(item["path"]).relative_to(output_dir)),
            "split": item["split"],
            "session_ids": [item["session_id"]],
            "sample_count": item["sample_count"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        } for item in results],
    }
    _write_json(output_dir / "dataset_manifest.json", manifest)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--session-limit", type=int, default=0)
    parser.add_argument("--dense-step-ms", type=float, default=1.0)
    parser.add_argument("--tie-epsilon-m", type=float, default=1e-6)
    parser.add_argument("--query-tolerance-us", type=float, default=2.0)
    parser.add_argument("--maximum-rollout-error-m", type=float, default=2e-4)
    args = parser.parse_args()
    if (
        args.workers < 1 or args.session_limit < 0 or args.dense_step_ms <= 0
        or args.tie_epsilon_m < 0 or args.query_tolerance_us < 0
        or args.maximum_rollout_error_m <= 0
    ):
        parser.error("observable F builder arguments are invalid")
    print(build(args))


if __name__ == "__main__":
    main()
