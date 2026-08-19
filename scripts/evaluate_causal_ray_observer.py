#!/usr/bin/env python3
"""Replay retained complete frame ledgers through the causal ray observer.

This is a structural development replay.  Truth is never supplied to the
observer.  Older captures use the observation/truth recorder only to recover
the run-level session/epoch identity missing from frame_events.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from causal_ray_observer import CausalRayObserver, ObserverConfig


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["frame_seq"]), int(row["timestamp_ns"])


def ledger_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["source_sequence"]), int(row["capture_timestamp_ns"])


def run_identity(run_dir: Path, observations: list[dict[str, Any]]) -> tuple[str, int]:
    if observations:
        first = observations[0]
        return str(first["session_id"]), int(first["producer_epoch"])
    truths = read_jsonl(run_dir / "truth.jsonl")
    if not truths:
        raise ValueError(f"no source identity evidence: {run_dir}")
    first = truths[0]
    return str(first["session_id"]), int(first["producer_epoch"])


def build_runtime_frames(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ledger = read_jsonl(run_dir / "frame_events.jsonl")
    observations = read_jsonl(run_dir / "stage3_observations.jsonl")
    observation_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for observation in observations:
        key = source_key(observation)
        if key in observation_by_key:
            raise ValueError(f"duplicate observation key {key}: {run_dir}")
        observation_by_key[key] = observation
    ledger_keys = [ledger_key(row) for row in ledger]
    if len(ledger_keys) != len(set(ledger_keys)):
        raise ValueError(f"duplicate ledger key: {run_dir}")
    unknown = set(observation_by_key).difference(ledger_keys)
    if unknown:
        raise ValueError(f"observation keys absent from complete ledger: {run_dir}: {len(unknown)}")
    session_id, producer_epoch = run_identity(run_dir, observations)
    frames = []
    for event in ledger:
        frame_seq, timestamp_ns = ledger_key(event)
        observation = observation_by_key.get((frame_seq, timestamp_ns))
        frames.append(
            {
                "schema_version": "autoaim-observer-frame-v1",
                "session_id": session_id,
                "producer_epoch": producer_epoch,
                "frame_seq": frame_seq,
                "capture_timestamp_ns": timestamp_ns,
                "observation_sink_status": "ok",
                "gimbal_pose_exposure_matched": (
                    None if observation is None else observation.get("gimbal_pose_exposure_matched")
                ),
                "offline_observation_frame_present": observation is not None,
                "candidates": [] if observation is None else observation.get("armors", []),
            }
        )
    return frames, {
        "ledger_frames": len(ledger),
        "observation_frames": len(observations),
        "exact_joined_observation_frames": len(observation_by_key),
        "nearest_or_interpolated_joins": 0,
    }


def comparable_output(output: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(output))
    return value


def missing_streaks(
    frames: list[dict[str, Any]], present: Callable[[dict[str, Any]], bool]
) -> list[dict[str, Any]]:
    result = []
    start = None
    previous = None
    count = 0
    for frame in frames + [None]:
        missing = frame is not None and not present(frame)
        if missing:
            if start is None:
                start = int(frame["capture_timestamp_ns"])
                count = 0
            previous = int(frame["capture_timestamp_ns"])
            count += 1
        elif start is not None:
            result.append(
                {
                    "frames": count,
                    "span_s": 0.0 if previous is None else (previous - start) * 1e-9,
                }
            )
            start = previous = None
            count = 0
    return result


def replay_run(run_dir: Path, config: ObserverConfig) -> dict[str, Any]:
    manifest = json.loads((run_dir / "collection_run_manifest.json").read_text(encoding="utf-8-sig"))
    frames, join = build_runtime_frames(run_dir)
    observer = CausalRayObserver(config)
    permutation_observer = CausalRayObserver(config)
    statuses: Counter[str] = Counter()
    ambiguity_reasons: Counter[str] = Counter()
    latencies_us = []
    valid_candidates = 0
    raw_candidates = 0
    physical_identity_violations = 0
    prediction_valid_violations = 0
    permutation_failures = 0
    for frame in frames:
        start = time.perf_counter_ns()
        output = observer.update(frame)
        latencies_us.append((time.perf_counter_ns() - start) / 1000.0)
        permuted = dict(frame)
        permuted["candidates"] = list(reversed(frame["candidates"]))
        permutation_output = permutation_observer.update(permuted)
        if comparable_output(output) != comparable_output(permutation_output):
            permutation_failures += 1
        statuses[str(output["observer_status"])] += 1
        for reason in output["status_reason"]:
            if output["observer_status"] == "AMBIGUOUS_SET":
                ambiguity_reasons[str(reason)] += 1
        raw_candidates += int(output["candidate_count"])
        valid_candidates += int(output["valid_candidate_count"])
        physical_identity_violations += int(bool(output["physical_identity_resolved"]))
        prediction_valid_violations += int(bool(output["prediction_valid"]))
    observation_streaks = missing_streaks(
        frames, lambda frame: bool(frame["offline_observation_frame_present"])
    )
    valid_event_streaks = missing_streaks(frames, lambda frame: bool(frame["candidates"]))
    return {
        "run": run_dir.name,
        "source_root": str(run_dir.parent),
        "motion": manifest.get("motion_mode"),
        "distance_m": float(manifest.get("requested_distance_m", float("nan"))),
        "radial_scale": float(manifest.get("radial_scale", float("nan"))),
        "repeat": int(manifest.get("repeat", -1)),
        **join,
        "raw_candidates": raw_candidates,
        "valid_candidates": valid_candidates,
        "zero_candidate_frames": sum(not frame["candidates"] for frame in frames),
        "more_than_four_candidate_frames": sum(len(frame["candidates"]) > 4 for frame in frames),
        "observation_frame_missing_streaks": len(observation_streaks),
        "valid_event_missing_streaks": len(valid_event_streaks),
        "missing_streaks": len(observation_streaks) + len(valid_event_streaks),
        "missing_streak_max_frames": max(
            (row["frames"] for row in observation_streaks + valid_event_streaks), default=0
        ),
        "missing_streak_max_span_s": max(
            (row["span_s"] for row in observation_streaks + valid_event_streaks), default=0.0
        ),
        "observed_anonymous_frames": statuses["OBSERVED_ANONYMOUS"],
        "ambiguous_frames": statuses["AMBIGUOUS_SET"],
        "stale_frames": statuses["STALE"],
        "invalid_stream_frames": statuses["INVALID_STREAM"],
        "status_counts": dict(statuses),
        "ambiguity_reasons": dict(ambiguity_reasons),
        "permutation_failures": permutation_failures,
        "physical_identity_violations": physical_identity_violations,
        "prediction_valid_violations": prediction_valid_violations,
        "latency_p50_us": float(np.percentile(latencies_us, 50)) if latencies_us else float("nan"),
        "latency_p95_us": float(np.percentile(latencies_us, 95)) if latencies_us else float("nan"),
        "latency_p99_us": float(np.percentile(latencies_us, 99)) if latencies_us else float("nan"),
        "latency_max_us": max(latencies_us, default=float("nan")),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key, value in row.items():
            if isinstance(value, dict):
                continue
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-radial-scale", nargs="+", type=float)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = ObserverConfig()
    run_dirs = []
    for root in args.root:
        for run_dir in sorted(path for path in root.resolve().iterdir() if path.is_dir()):
            manifest_path = run_dir / "collection_run_manifest.json"
            if not manifest_path.exists() or not (run_dir / "frame_events.jsonl").exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            scale = float(manifest.get("radial_scale", float("nan")))
            if args.include_radial_scale and not any(
                abs(scale - allowed) <= 1e-9 for allowed in args.include_radial_scale
            ):
                continue
            run_dirs.append(run_dir)
    rows = [replay_run(run_dir, config) for run_dir in run_dirs]
    if not rows:
        raise ValueError("no qualifying runs")
    write_csv(output / "observer_contract_runs.csv", rows)
    total_frames = sum(int(row["ledger_frames"]) for row in rows)
    summary = {
        "schema_version": "autoaim-causal-ray-observer-replay-v1",
        "claim": "development structural replay only; not Linux-1.3.1 deployment acceptance",
        "runs": len(rows),
        "frames": total_frames,
        "observation_frames": sum(int(row["observation_frames"]) for row in rows),
        "zero_candidate_frames": sum(int(row["zero_candidate_frames"]) for row in rows),
        "more_than_four_candidate_frames": sum(
            int(row["more_than_four_candidate_frames"]) for row in rows
        ),
        "missing_streaks": sum(int(row["missing_streaks"]) for row in rows),
        "observation_frame_missing_streaks": sum(
            int(row["observation_frame_missing_streaks"]) for row in rows
        ),
        "valid_event_missing_streaks": sum(
            int(row["valid_event_missing_streaks"]) for row in rows
        ),
        "longest_missing_streak_s": max(float(row["missing_streak_max_span_s"]) for row in rows),
        "observed_anonymous_fraction": sum(
            int(row["observed_anonymous_frames"]) for row in rows
        )
        / total_frames,
        "ambiguous_fraction": sum(int(row["ambiguous_frames"]) for row in rows) / total_frames,
        "stale_fraction": sum(int(row["stale_frames"]) for row in rows) / total_frames,
        "ambiguity_reasons": dict(
            Counter(
                reason
                for row in rows
                for reason, count in row["ambiguity_reasons"].items()
                for _ in range(int(count))
            )
        ),
        "structural_gates": {
            "exact_key_only": all(int(row["nearest_or_interpolated_joins"]) == 0 for row in rows),
            "candidate_permutation_invariance": all(
                int(row["permutation_failures"]) == 0 for row in rows
            ),
            "physical_identity_guard": all(
                int(row["physical_identity_violations"]) == 0 for row in rows
            ),
            "prediction_guard": all(int(row["prediction_valid_violations"]) == 0 for row in rows),
            "invalid_stream_free_on_retained_inputs": all(
                int(row["invalid_stream_frames"]) == 0 for row in rows
            ),
        },
        "offline_python_latency_us": {
            "run_macro_p50_of_p50": float(np.percentile([row["latency_p50_us"] for row in rows], 50)),
            "run_macro_p95_of_p95": float(np.percentile([row["latency_p95_us"] for row in rows], 95)),
            "run_macro_p99_of_p99": float(np.percentile([row["latency_p99_us"] for row in rows], 99)),
            "max": max(float(row["latency_max_us"]) for row in rows),
            "production_claim": False,
        },
        "source_boundary": (
            "frame_events is the complete consumed-frame ledger; stage3 observations supply the complete "
            "post-PnP solved-armors set for exact matching keys. Run-level session/epoch is reconstructed "
            "from recorder provenance. Truth target/slot/motion fields are not observer inputs."
        ),
        "config": config.__dict__,
    }
    (output / "observer_contract_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in output.iterdir()
        if path.is_file() and path.name != "retention_manifest.json"
    }
    (output / "retention_manifest.json").write_text(
        json.dumps(
            {
                "classification": "long_term_private_development_evidence",
                "deletion_allowed": False,
                "source_roots": [str(path.resolve()) for path in args.root],
                "artifacts": artifacts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
