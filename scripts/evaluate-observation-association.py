#!/usr/bin/env python3
"""Evaluate a causal, detector-only armor association baseline.

The association path never reads truth. Truth is joined after association only
to score track consistency. Track IDs are run-local and are compared to
physical slots through the best global permutation, so the score is invariant
to arbitrary initialization labels.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_grid_analysis(repo_root: Path):
    path = repo_root / "scripts" / "analyze-stage3-truth-grid.py"
    spec = importlib.util.spec_from_file_location("stage3_grid_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wrap_rad(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def camera_angles(tvec: list[float]) -> tuple[float, float]:
    x, y, z = (float(value) for value in tvec)
    z = max(z, 1e-6)
    return math.atan2(x, z), math.atan2(y, z)


def valid_detections(observation: dict, t_s: float, truth_slots: dict[int, int]) -> list[dict]:
    detections: list[dict] = []
    for detection_index, armor in enumerate(observation.get("armors", [])):
        tvec = armor.get("camera_tvec_m")
        if not tvec or len(tvec) != 3 or not all(math.isfinite(float(value)) for value in tvec):
            continue
        if armor.get("valid") is False:
            continue
        u, v = camera_angles(tvec)
        yaw = armor.get("yaw_absolute_rad")
        detections.append(
            {
                "detection_index": detection_index,
                "t_s": t_s,
                "u": u,
                "v": v,
                "yaw": float(yaw) if yaw is not None and math.isfinite(float(yaw)) else None,
                "truth_slot": truth_slots.get(detection_index),
            }
        )
    return detections


class CausalAssociator:
    def __init__(self, gate_deg: float, use_yaw: bool, max_tracks: int = 4) -> None:
        self.gate_rad = math.radians(gate_deg)
        self.use_yaw = use_yaw
        self.max_tracks = max_tracks
        self.tracks: list[dict | None] = [None] * max_tracks

    def _predict(self, track: dict, t_s: float) -> tuple[float, float, float | None]:
        dt = max(0.0, min(t_s - track["t_s"], 0.35))
        return (
            track["u"] + track["du_dt"] * dt,
            track["v"] + track["dv_dt"] * dt,
            None if track["yaw"] is None else track["yaw"] + track["dyaw_dt"] * dt,
        )

    def _cost(self, track: dict, detection: dict) -> float:
        predicted_u, predicted_v, predicted_yaw = self._predict(track, detection["t_s"])
        position = math.hypot(wrap_rad(detection["u"] - predicted_u), detection["v"] - predicted_v)
        if not self.use_yaw or predicted_yaw is None or detection["yaw"] is None:
            return position
        yaw_error = abs(wrap_rad(detection["yaw"] - predicted_yaw))
        # Yaw is an auxiliary cue. Position remains the dominant cue because
        # the detector's orientation estimate is noisier near edge-on views.
        return math.hypot(position, 0.35 * yaw_error)

    def _update(self, track_id: int, detection: dict) -> None:
        previous = self.tracks[track_id]
        if previous is None:
            self.tracks[track_id] = {
                "t_s": detection["t_s"],
                "u": detection["u"],
                "v": detection["v"],
                "yaw": detection["yaw"],
                "du_dt": 0.0,
                "dv_dt": 0.0,
                "dyaw_dt": 0.0,
            }
            return
        dt = detection["t_s"] - previous["t_s"]
        if dt > 1e-6:
            self.tracks[track_id] = {
                "t_s": detection["t_s"],
                "u": detection["u"],
                "v": detection["v"],
                "yaw": detection["yaw"],
                "du_dt": (detection["u"] - previous["u"]) / dt,
                "dv_dt": (detection["v"] - previous["v"]) / dt,
                "dyaw_dt": (
                    0.0
                    if detection["yaw"] is None or previous["yaw"] is None
                    else wrap_rad(detection["yaw"] - previous["yaw"]) / dt
                ),
            }

    def _choose_assignment(self, detections: list[dict]) -> list[int | None]:
        if not detections:
            return []
        active = [index for index, track in enumerate(self.tracks) if track is not None]
        free = [index for index, track in enumerate(self.tracks) if track is None]
        new_cost = self.gate_rad * 0.95
        drop_cost = self.gate_rad
        track_columns = active + free
        drop_offset = len(track_columns)
        large = max(1000.0, 1000.0 * self.gate_rad)
        costs = np.full((len(detections), len(track_columns) + len(detections)), large)
        for row, detection in enumerate(detections):
            for column, track_id in enumerate(track_columns):
                if track_id in free:
                    costs[row, column] = new_cost
                else:
                    value = self._cost(self.tracks[track_id], detection)
                    if value <= self.gate_rad:
                        costs[row, column] = value
            costs[row, drop_offset + row] = drop_cost
        row_indices, column_indices = linear_sum_assignment(costs)
        result: list[int | None] = [None] * len(detections)
        for row, column in zip(row_indices, column_indices):
            if column < len(track_columns) and costs[row, column] < large:
                result[row] = track_columns[column]
        return result

    def update(self, detections: list[dict]) -> list[tuple[dict, int | None]]:
        choices = self._choose_assignment(detections)
        result: list[tuple[dict, int | None]] = []
        for detection, choice in zip(detections, choices):
            track_id: int | None = choice
            if track_id is not None:
                self._update(track_id, detection)
            result.append((detection, track_id))
        return result


def best_slot_mapping(rows: list[dict]) -> tuple[dict[int, int], int]:
    track_ids = sorted({int(row["track_id"]) for row in rows if row["track_id"] is not None})
    matrix = {(track_id, slot): 0 for track_id in track_ids for slot in range(4)}
    for row in rows:
        if row["track_id"] is not None and row["truth_slot"] is not None:
            matrix[(int(row["track_id"]), int(row["truth_slot"]))] += 1
    best_score = -1
    best: dict[int, int] = {}
    for slots in itertools.permutations(range(4), len(track_ids)):
        score = sum(matrix[(track_id, slot)] for track_id, slot in zip(track_ids, slots))
        if score > best_score:
            best_score = score
            best = dict(zip(track_ids, slots))
    return best, max(best_score, 0)


def prepare_run(run_dir: Path, source_root: Path, grid_analysis) -> dict:
    truths = read_jsonl(run_dir / "truth.jsonl")
    observations = read_jsonl(run_dir / "stage3_observations.jsonl")
    truth_map = {grid_analysis.make_key(record): record for record in truths}
    manifest = json.loads((run_dir / "collection_run_manifest.json").read_text(encoding="utf-8-sig"))
    requested_distance = float(manifest["requested_distance_m"])
    first_timestamp = min((int(record["timestamp_ns"]) for record in truths), default=0)
    events: list[list[dict]] = []
    valid_total = 0
    for observation in observations:
        truth = truth_map.get(grid_analysis.make_key(observation))
        if truth is None:
            continue
        target = grid_analysis.select_active_target(truth, requested_distance)
        if target is None:
            continue
        projected = grid_analysis.target_slot_points(truth, target)
        truth_assignments = {
            row["detection_index"]: row["slot"]
            for row in grid_analysis.assign_observations(observation, projected)
        }
        t_s = (int(observation["timestamp_ns"]) - first_timestamp) * 1e-9
        detections = valid_detections(observation, t_s, truth_assignments)
        valid_total += len(detections)
        events.append(detections)
    return {
        "run": f"{source_root.name}/{run_dir.name}",
        "source_root": str(source_root),
        "scale": float(manifest["radial_scale"]),
        "distance_m": requested_distance,
        "repeat": int(manifest["repeat"]),
        "events": events,
        "valid_detections": valid_total,
    }


def evaluate_prepared(prepared: dict, gate_deg: float, use_yaw: bool) -> dict:
    associator = CausalAssociator(gate_deg, use_yaw)
    rows: list[dict] = []
    for detections in prepared["events"]:
        for detection, track_id in associator.update(detections):
            rows.append(
                {
                    "t_s": detection["t_s"],
                    "detection_index": detection["detection_index"],
                    "track_id": track_id,
                    "truth_slot": detection["truth_slot"],
                }
            )
    scored = [row for row in rows if row["truth_slot"] is not None]
    mapping, correct = best_slot_mapping(scored)
    for row in scored:
        row["mapped_slot"] = mapping.get(row["track_id"])
    errors = [row for row in scored if row["mapped_slot"] != row["truth_slot"]]
    track_counts = defaultdict(int)
    track_slot_counts: dict[int, defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))
    for row in scored:
        if row["track_id"] is None:
            continue
        track_counts[int(row["track_id"])] += 1
        track_slot_counts[int(row["track_id"])][int(row["truth_slot"])] += 1
    purities = [max(counts.values()) / sum(counts.values()) for counts in track_slot_counts.values() if counts]
    return {
        "run": prepared["run"],
        "source_root": prepared["source_root"],
        "scale": prepared["scale"],
        "distance_m": prepared["distance_m"],
        "repeat": prepared["repeat"],
        "gate_deg": gate_deg,
        "use_yaw": use_yaw,
        "valid_detections": prepared["valid_detections"],
        "scored_detections": len(scored),
        "associated_detections": sum(row["track_id"] is not None for row in scored),
        "track_count": len(track_counts),
        "global_mapping_accuracy": correct / len(scored) if scored else float("nan"),
        "identity_error_rate": len(errors) / len(scored) if scored else float("nan"),
        "mean_track_purity": float(np.mean(purities)) if purities else float("nan"),
        "max_track_purity": float(np.max(purities)) if purities else float("nan"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path, nargs="+")
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-radial-scale",
        type=float,
        nargs="+",
        help="evaluate only runs whose manifest radial_scale matches one of these values",
    )
    args = parser.parse_args()
    roots = [path.resolve() for path in args.root]
    output = (args.output or roots[0] / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    grid_analysis = load_grid_analysis(args.repo.resolve())
    prepared_runs: list[dict] = []
    for root in roots:
        run_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        for run_dir in run_dirs:
            if not (run_dir / "stage3_observations.jsonl").exists():
                continue
            if args.include_radial_scale:
                manifest = json.loads(
                    (run_dir / "collection_run_manifest.json").read_text(encoding="utf-8-sig")
                )
                scale = float(manifest["radial_scale"])
                if not any(abs(scale - selected) <= 1e-9 for selected in args.include_radial_scale):
                    continue
            prepared_runs.append(prepare_run(run_dir, root, grid_analysis))
    rows: list[dict] = []
    for use_yaw in (False, True):
        for gate_deg in (8.0, 15.0, 25.0):
            for prepared in prepared_runs:
                rows.append(evaluate_prepared(prepared, gate_deg, use_yaw))
    write_csv(output / "observation_only_association_runs.csv", rows)
    by_variant: dict[tuple[bool, float], list[dict]] = defaultdict(list)
    for row in rows:
        by_variant[(bool(row["use_yaw"]), float(row["gate_deg"]))].append(row)
    summary_rows: list[dict] = []
    for (use_yaw, gate_deg), variant_rows in sorted(by_variant.items()):
        scored_rows = [row for row in variant_rows if int(row["scored_detections"]) > 0]
        for field in ("global_mapping_accuracy", "identity_error_rate", "mean_track_purity"):
            values = np.asarray([float(row[field]) for row in scored_rows], dtype=float)
            summary_rows.append(
                {
                    "use_yaw": use_yaw,
                    "gate_deg": gate_deg,
                    "metric": field,
                    "run_count": len(scored_rows),
                    "all_run_count": len(variant_rows),
                    "mean": float(np.nanmean(values)),
                    "p05": float(np.nanpercentile(values, 5)),
                    "p50": float(np.nanpercentile(values, 50)),
                    "p95": float(np.nanpercentile(values, 95)),
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                }
            )
    write_csv(output / "observation_only_association_summary.csv", summary_rows)
    summary = {
        "schema_version": 1,
        "kind": "causal_observation_only_association",
        "runtime_input_contract": "timestamp, camera_tvec_m, yaw_absolute_rad and valid detection fields only; detector_number is ignored",
        "evaluation_contract": "truth is joined after association only to score global track-to-slot permutation and track purity",
        "variants": len(by_variant),
        "included_radial_scales": sorted(set(float(value) for value in args.include_radial_scale)) if args.include_radial_scale else "all",
        "runs": len({row["run"] for row in rows}),
        "runs_with_scored_detections": len({row["run"] for row in rows if int(row["scored_detections"]) > 0}),
        "artifacts": ["observation_only_association_runs.csv", "observation_only_association_summary.csv"],
    }
    (output / "observation_only_association_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
