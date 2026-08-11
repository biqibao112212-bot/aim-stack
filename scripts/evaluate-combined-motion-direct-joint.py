#!/usr/bin/env python3
"""Joint long-window omega and reversal-phase fit on combined motion.

The configured bounded-linear axis/speed/span are held fixed.  Omega and the
triangle-wave phase are jointly selected from causal PnP armor histories,
without reconstructing a center per frame.  Development sessions only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize_scalar


SESSION_PREFIX = "stage3-generalization-fixed6mm-20260729-v1-combined-"
DEV_SESSION_SUFFIXES = ("00", "01", "02", "03", "05")
HISTORY_WINDOW_S = 4.0
HORIZONS_S = (0.05, 0.10, 0.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-stride", type=int, default=24)
    parser.add_argument("--omega-grid-step", type=float, default=1.0)
    parser.add_argument("--max-omega", type=float, default=16.0)
    parser.add_argument("--phase-grid-count", type=int, default=25)
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


def joint_fit(
    direct,
    base,
    times_s: np.ndarray,
    slots: np.ndarray,
    positions_m: np.ndarray,
    geometry_m: np.ndarray,
    tracker_to_world: np.ndarray,
    *,
    speed_mps: float,
    span_m: float,
    axis: np.ndarray,
    max_omega: float,
    omega_grid_step: float,
    phase_grid_count: int,
    robust: bool,
) -> tuple[float, float, np.ndarray, float]:
    grid = np.arange(-max_omega, max_omega + 0.5 * omega_grid_step, omega_grid_step)

    def fit_at(omega: float, final_robust: bool):
        return direct.estimate_phase(
            base,
            times_s,
            slots,
            positions_m,
            geometry_m,
            tracker_to_world,
            omega=float(omega),
            speed_mps=speed_mps,
            span_m=span_m,
            axis=axis,
            phase_grid_count=phase_grid_count,
            robust=final_robust,
        )

    grid_results = [fit_at(float(omega), False) for omega in grid]
    grid_losses = np.asarray([result[2] for result in grid_results])
    best = int(np.argmin(grid_losses))
    lower = max(-max_omega, float(grid[best] - omega_grid_step))
    upper = min(max_omega, float(grid[best] + omega_grid_step))

    def objective(omega: float) -> float:
        return fit_at(float(omega), False)[2]

    if upper - lower > 1e-9:
        result = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
        omega = float(result.x)
    else:
        omega = float(grid[best])
    phase, coefficient, loss = fit_at(omega, robust)
    return omega, phase, coefficient, loss


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
    keys.extend(["input_domain", "future_regime", "method", "horizon_s"])
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        item = dict(zip(keys, key))
        for metric in ("error_cross_depth_m", "error_depth_m", "error_3d_m"):
            for name, value in quantiles(row[metric] for row in group).items():
                item[f"{metric}_{name}"] = value
        result.append(item)
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
    direct_path = repo / "scripts" / "evaluate-combined-motion-direct-reflection.py"
    base = load_module("combined_factorization_v1_joint", base_path)
    direct = load_module("combined_direct_reflection_v1_joint", direct_path)
    dataset_root = workspace / "dataset" / "autoaim-stage3-v1"
    runtime_root = workspace / "runtime" / "stage3-generalization-fixed6mm-20260729-v1"

    prediction_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    source_paths: set[Path] = {Path(__file__).resolve(), base_path, direct_path}
    for suffix in DEV_SESSION_SUFFIXES:
        session_id = SESSION_PREFIX + suffix
        frames, manifest, sources = base.load_session(dataset_root, runtime_root, session_id)
        source_paths.update(sources)
        split_role = "validation" if suffix == "00" else "train"
        geometry = frames[0].armor_local_m
        direction = np.deg2rad(float(manifest["direction_deg"]))
        axis = np.asarray([np.cos(direction), np.sin(direction), 0.0])
        speed_mps = float(manifest["linear_speed_mps"])
        span_m = float(manifest["linear_span_m"])
        for anchor_index in range(0, len(frames), args.anchor_stride):
            anchor = frames[anchor_index]
            current_slots = np.flatnonzero(anchor.observed_mask)
            if current_slots.size == 0:
                continue
            primary = int(current_slots[np.argmin(anchor.observed_range_m[current_slots])])
            histories = {
                domain: direct.direct_history(
                    base, frames, anchor_index, HISTORY_WINDOW_S, domain
                )
                for domain in ("clean", "pnp")
            }
            if any(history is None for history in histories.values()):
                continue
            fits = {}
            for domain in ("clean", "pnp"):
                times, slots, positions = histories[domain]
                omega, phase, coefficient, fit_loss = joint_fit(
                    direct,
                    base,
                    times,
                    slots,
                    positions,
                    geometry,
                    anchor.tracker_to_world,
                    speed_mps=speed_mps,
                    span_m=span_m,
                    axis=axis,
                    max_omega=args.max_omega,
                    omega_grid_step=args.omega_grid_step,
                    phase_grid_count=args.phase_grid_count,
                    robust=domain == "pnp",
                )
                method = "direct_joint_omega_phase"
                fits[domain] = (method, omega, phase, coefficient, fit_loss)
                fit_rows.append(
                    {
                        "split_role": split_role,
                        "session_id": session_id,
                        "timestamp_ns": anchor.timestamp_ns,
                        "input_domain": domain,
                        "method": method,
                        "history_window_s": HISTORY_WINDOW_S,
                        "model_omega_rad_s": omega,
                        "truth_omega_rad_s": anchor.yaw_rate_rad_s,
                        "absolute_omega_error_rad_s": abs(omega - anchor.yaw_rate_rad_s),
                        "omega_sign_correct": int(np.sign(omega) == np.sign(anchor.yaw_rate_rad_s)),
                        "phase_m": phase,
                        "span_m": span_m,
                        "speed_mps": speed_mps,
                        "fit_loss": fit_loss,
                        "history_event_count": int(len(times)),
                        "history_time_span_s": float(np.ptp(times)),
                    }
                )
            for horizon_s in HORIZONS_S:
                future = base.nearest_future(frames, anchor_index, horizon_s)
                if future is None:
                    continue
                effective_horizon_s = (
                    future.timestamp_ns - anchor.timestamp_ns
                ) / 1e9
                truth = future.armor_world_m[primary]
                future_regime = (
                    "constant_twist"
                    if future.segment_id == anchor.segment_id
                    else "cross_reversal"
                )
                for domain, (method, omega, phase, coefficient, fit_loss) in fits.items():
                    prediction = direct.predict_direct(
                        base,
                        coefficient,
                        geometry,
                        primary,
                        omega=omega,
                        phase_m=phase,
                        speed_mps=speed_mps,
                        span_m=span_m,
                        axis=axis,
                        horizon_s=effective_horizon_s,
                    )
                    row = {
                        "split_role": split_role,
                        "session_id": session_id,
                        "timestamp_ns": anchor.timestamp_ns,
                        "segment_id": anchor.segment_id,
                        "future_segment_id": future.segment_id,
                        "input_domain": domain,
                        "history_window_s": HISTORY_WINDOW_S,
                        "future_regime": future_regime,
                        "method": method,
                        "horizon_s": horizon_s,
                        "effective_horizon_s": effective_horizon_s,
                        "primary_slot": primary,
                        "visible_count_at_anchor": int(current_slots.size),
                        "model_omega_rad_s": omega,
                        "truth_omega_rad_s": anchor.yaw_rate_rad_s,
                        "phase_m": phase,
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
    write_csv_gz(output / "joint_fit_distribution.csv.gz", fit_rows)
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
            "schema_version": "combined-motion-direct-joint-development-v1",
            "test_split_accessed": False,
            "sealed_test_session": SESSION_PREFIX + "04",
            "model": "direct configured bounded translation with jointly searched omega and triangle phase",
            "history_window_s": HISTORY_WINDOW_S,
            "configured_kinematics": ["path axis", "path span", "linear speed"],
            "candidate_grid": {
                "max_abs_omega_rad_s": args.max_omega,
                "omega_grid_step_rad_s": args.omega_grid_step,
                "phase_grid_count": args.phase_grid_count,
                "anchor_stride_frames": args.anchor_stride,
            },
            "causality": {
                "history_up_to_anchor_only": True,
                "future_truth_estimator_input": False,
                "truth_slot_identity": "offline non-deployable analysis handle",
            },
            "sources": [
                {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in sorted(source_paths, key=str)
            ],
            "artifacts": artifacts,
            "row_counts": {
                "predictions": len(prediction_rows),
                "joint_fits": len(fit_rows),
                "condition_summaries": len(condition_summary),
                "pooled_summaries": len(pooled_summary),
            },
        },
    )


if __name__ == "__main__":
    main()
