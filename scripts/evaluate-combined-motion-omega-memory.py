#!/usr/bin/env python3
"""Causal cross-window omega-memory sweep for combined rigid motion.

The LOS metric is frozen from the development sweep at depth squared-weight
0.1, 20 mm Huber delta, and 400 ms local history.  This experiment changes
only how the current omega estimate is regularized by earlier causal windows.
The sealed test session is not loaded.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SESSION_PREFIX = "stage3-generalization-fixed6mm-20260729-v1-combined-"
DEV_SESSION_SUFFIXES = ("00", "01", "02", "03", "05")
HISTORY_WINDOW_S = 0.40
HORIZONS_S = (0.05, 0.10, 0.20)
DEPTH_WEIGHT = 0.1
HUBER_DELTA_M = 0.02
MEMORY_METHODS = (
    "raw",
    "median_3",
    "median_7",
    "median_15",
    "median_31",
    "ema_0.2",
    "ema_0.5",
    "cumulative_median_diagnostic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-stride", type=int, default=12)
    parser.add_argument("--omega-grid-step", type=float, default=1.0)
    parser.add_argument("--max-omega", type=float, default=16.0)
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty evidence table: {path}")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty evidence table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def causal_omega(method: str, raw: list[float]) -> float:
    if method == "raw":
        return float(raw[-1])
    if method.startswith("median_"):
        count = int(method.split("_")[1])
        return float(np.median(raw[-count:]))
    if method == "cumulative_median_diagnostic":
        return float(np.median(raw))
    if method.startswith("ema_"):
        alpha = float(method.split("_")[1])
        estimate = float(raw[0])
        for value in raw[1:]:
            estimate = alpha * float(value) + (1.0 - alpha) * estimate
        return estimate
    raise ValueError(method)


def quantiles(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "n": int(array.size),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def summarize(rows: list[dict[str, Any]], include_session: bool) -> list[dict[str, Any]]:
    keys = ["split_role"]
    if include_session:
        keys.append("session_id")
    keys.extend(["future_regime", "memory_method", "horizon_s"])
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        output = dict(zip(keys, key))
        for metric in ("error_cross_depth_m", "error_depth_m", "error_3d_m"):
            for name, value in quantiles(row[metric] for row in group).items():
                output[f"{metric}_{name}"] = value
        result.append(output)
    return result


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    workspace = args.workspace.resolve() if args.workspace else repo.parents[1]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    base_path = repo / "scripts" / "evaluate-combined-motion-factorization.py"
    los_path = repo / "scripts" / "evaluate-combined-motion-los-fit.py"
    base = load_module("combined_factorization_v1_memory", base_path)
    los = load_module("combined_los_fit_v1_memory", los_path)
    dataset_root = workspace / "dataset" / "autoaim-stage3-v1"
    runtime_root = workspace / "runtime" / "stage3-generalization-fixed6mm-20260729-v1"

    prediction_rows: list[dict[str, Any]] = []
    omega_rows: list[dict[str, Any]] = []
    source_paths: set[Path] = {Path(__file__).resolve(), base_path, los_path}
    for suffix in DEV_SESSION_SUFFIXES:
        session_id = SESSION_PREFIX + suffix
        frames, _manifest, sources = base.load_session(dataset_root, runtime_root, session_id)
        source_paths.update(sources)
        split_role = "validation" if suffix == "00" else "train"
        geometry = frames[0].armor_local_m
        raw_history: list[float] = []
        raw_timestamps: list[int] = []
        for anchor_index in range(0, len(frames), args.anchor_stride):
            anchor = frames[anchor_index]
            current_slots = np.flatnonzero(anchor.observed_mask)
            if current_slots.size == 0:
                continue
            history = base.history_observations(
                frames, anchor_index, HISTORY_WINDOW_S, "pnp"
            )
            if history is None:
                continue
            times, slots, positions = history
            primary = int(current_slots[np.argmin(anchor.observed_range_m[current_slots])])
            raw_omega, _, raw_loss = los.estimate_weighted_omega(
                base,
                times,
                slots,
                positions,
                geometry,
                anchor.tracker_to_world,
                depth_weight=DEPTH_WEIGHT,
                huber_delta_m=HUBER_DELTA_M,
                max_omega=args.max_omega,
                grid_step=args.omega_grid_step,
            )
            raw_history.append(raw_omega)
            raw_timestamps.append(anchor.timestamp_ns)
            for method in MEMORY_METHODS:
                omega = causal_omega(method, raw_history)
                coefficient, fit_loss = los.fit_weighted_rigid(
                    base,
                    times,
                    slots,
                    positions,
                    geometry,
                    omega,
                    anchor.tracker_to_world,
                    depth_weight=DEPTH_WEIGHT,
                    huber_delta_m=HUBER_DELTA_M,
                    robust=True,
                )
                omega_rows.append(
                    {
                        "split_role": split_role,
                        "session_id": session_id,
                        "timestamp_ns": anchor.timestamp_ns,
                        "segment_id": anchor.segment_id,
                        "memory_method": method,
                        "raw_omega_rad_s": raw_omega,
                        "model_omega_rad_s": omega,
                        "truth_omega_rad_s": anchor.yaw_rate_rad_s,
                        "absolute_error_rad_s": abs(omega - anchor.yaw_rate_rad_s),
                        "sign_correct": int(
                            math.copysign(1.0, omega or 1.0)
                            == math.copysign(1.0, anchor.yaw_rate_rad_s)
                        ),
                        "fit_loss": fit_loss,
                        "raw_fit_loss": raw_loss,
                        "causal_estimate_count": len(raw_history),
                        "causal_memory_span_s": (
                            anchor.timestamp_ns - raw_timestamps[max(0, len(raw_timestamps) - 31)]
                        )
                        / 1e9,
                    }
                )
                for horizon_s in HORIZONS_S:
                    future = base.nearest_future(frames, anchor_index, horizon_s)
                    if future is None:
                        continue
                    effective_horizon_s = (
                        future.timestamp_ns - anchor.timestamp_ns
                    ) / 1e9
                    prediction = base.predict_rigid(
                        coefficient, geometry, primary, omega, effective_horizon_s
                    )
                    truth = future.armor_world_m[primary]
                    row = {
                        "split_role": split_role,
                        "session_id": session_id,
                        "timestamp_ns": anchor.timestamp_ns,
                        "segment_id": anchor.segment_id,
                        "future_segment_id": future.segment_id,
                        "future_regime": (
                            "constant_twist"
                            if future.segment_id == anchor.segment_id
                            else "cross_reversal"
                        ),
                        "history_window_s": HISTORY_WINDOW_S,
                        "horizon_s": horizon_s,
                        "effective_horizon_s": effective_horizon_s,
                        "memory_method": method,
                        "primary_slot": primary,
                        "visible_count_at_anchor": int(current_slots.size),
                        "raw_omega_rad_s": raw_omega,
                        "model_omega_rad_s": omega,
                        "truth_omega_rad_s": anchor.yaw_rate_rad_s,
                        "fit_loss": fit_loss,
                        "prediction_x_m": float(prediction[0]),
                        "prediction_y_m": float(prediction[1]),
                        "prediction_z_m": float(prediction[2]),
                        "truth_x_m": float(truth[0]),
                        "truth_y_m": float(truth[1]),
                        "truth_z_m": float(truth[2]),
                    }
                    row.update(
                        base.error_components(prediction, truth, anchor.tracker_to_world)
                    )
                    prediction_rows.append(row)

    condition_summary = summarize(prediction_rows, include_session=True)
    pooled_summary = summarize(prediction_rows, include_session=False)
    write_csv_gz(output / "prediction_distribution.csv.gz", prediction_rows)
    write_csv_gz(output / "omega_distribution.csv.gz", omega_rows)
    write_csv(output / "prediction_condition_summary.csv", condition_summary)
    write_csv(output / "prediction_pooled_summary.csv", pooled_summary)
    artifacts = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append(
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    write_json(
        output / "manifest.json",
        {
            "schema_version": "combined-motion-omega-memory-development-v1",
            "test_split_accessed": False,
            "sealed_test_session": SESSION_PREFIX + "04",
            "frozen_los_fit": {
                "history_window_s": HISTORY_WINDOW_S,
                "depth_squared_weight": DEPTH_WEIGHT,
                "huber_delta_m": HUBER_DELTA_M,
            },
            "candidate_memory_methods": list(MEMORY_METHODS),
            "causality": {
                "current_and_past_window_estimates_only": True,
                "future_truth_estimator_input": False,
                "cumulative_median_is_diagnostic_for_constant_session_omega": True,
                "truth_segment_boundary_used_by_local_v1_history": True,
            },
            "selection_contract": {
                "train_sessions": [SESSION_PREFIX + suffix for suffix in ("01", "02", "03", "05")],
                "validation_session": SESSION_PREFIX + "00",
                "no_test_tuning": True,
            },
            "sources": [
                {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in sorted(source_paths, key=str)
            ],
            "artifacts": artifacts,
            "row_counts": {
                "predictions": len(prediction_rows),
                "omega_estimates": len(omega_rows),
                "condition_summaries": len(condition_summary),
                "pooled_summaries": len(pooled_summary),
            },
        },
    )


if __name__ == "__main__":
    main()
