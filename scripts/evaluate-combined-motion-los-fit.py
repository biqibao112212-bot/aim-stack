#!/usr/bin/env python3
"""Development-only LOS-aware fit sweep for combined rigid motion.

This script reuses the immutable Stage3 combined-motion sessions and the
factorized rigid model, but changes only the residual metric used during
fitting.  It is intentionally development-only: session 04 remains sealed.

The fit is solved in the anchor tracker frame.  Cross-depth coordinates keep
unit weight while the depth coordinate receives ``depth_weight``.  Complete
per-prediction and per-omega distributions are retained for every candidate.
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
from scipy.optimize import minimize_scalar


DEV_SESSION_SUFFIXES = ("00", "01", "02", "03", "05")
SESSION_PREFIX = "stage3-generalization-fixed6mm-20260729-v1-combined-"
DEPTH_WEIGHTS = (1.0, 0.1, 0.01, 0.001)
HUBER_DELTAS_M = (0.02, 0.05, 0.10)
HISTORY_WINDOWS_S = (0.25, 0.40)
HORIZONS_S = (0.05, 0.10, 0.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-stride", type=int, default=12)
    parser.add_argument("--omega-grid-step", type=float, default=1.0)
    parser.add_argument("--max-omega", type=float, default=16.0)
    return parser.parse_args()


def load_base(repo: Path):
    path = repo / "scripts" / "evaluate-combined-motion-factorization.py"
    spec = importlib.util.spec_from_file_location("combined_factorization_v1", path)
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


def fit_weighted_rigid(
    base,
    times_s: np.ndarray,
    slots: np.ndarray,
    positions_m: np.ndarray,
    geometry_m: np.ndarray,
    omega: float,
    tracker_to_world: np.ndarray,
    *,
    depth_weight: float,
    huber_delta_m: float,
    robust: bool,
) -> tuple[np.ndarray, float]:
    """Fit the v1 model using an anisotropic tracker-frame Huber metric."""
    design, z_offset = base.rigid_design(times_s, slots, geometry_m, omega)
    target = positions_m.reshape(-1).copy()
    target[2::3] -= z_offset[2::3]
    blocks = design.reshape(-1, 3, design.shape[1])
    targets = target.reshape(-1, 3)

    metric = np.diag([math.sqrt(depth_weight), 1.0, 1.0]) @ tracker_to_world.T
    weighted_blocks = np.einsum("ab,nbc->nac", metric, blocks)
    weighted_targets = np.einsum("ab,nb->na", metric, targets)
    flat_design = weighted_blocks.reshape(-1, design.shape[1])
    flat_target = weighted_targets.reshape(-1)

    event_weight = np.ones(len(times_s), dtype=np.float64)
    coefficient = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(8 if robust else 1):
        root = np.sqrt(np.repeat(event_weight, 3))
        coefficient = np.linalg.lstsq(
            flat_design * root[:, None], flat_target * root, rcond=None
        )[0]
        residual = (flat_design @ coefficient - flat_target).reshape(-1, 3)
        norm = np.linalg.norm(residual, axis=1)
        event_weight = np.minimum(
            1.0, huber_delta_m / np.maximum(norm, np.finfo(float).eps)
        )

    residual = (flat_design @ coefficient - flat_target).reshape(-1, 3)
    norm = np.linalg.norm(residual, axis=1)
    if robust:
        delta = huber_delta_m
        loss = float(
            np.mean(
                np.where(
                    norm <= delta,
                    0.5 * norm**2,
                    delta * (norm - 0.5 * delta),
                )
            )
        )
    else:
        loss = float(np.mean(norm**2))
    return coefficient, loss


def estimate_weighted_omega(
    base,
    times_s: np.ndarray,
    slots: np.ndarray,
    positions_m: np.ndarray,
    geometry_m: np.ndarray,
    tracker_to_world: np.ndarray,
    *,
    depth_weight: float,
    huber_delta_m: float,
    max_omega: float,
    grid_step: float,
) -> tuple[float, np.ndarray, float]:
    def fit_at(omega: float, robust: bool) -> tuple[np.ndarray, float]:
        return fit_weighted_rigid(
            base,
            times_s,
            slots,
            positions_m,
            geometry_m,
            omega,
            tracker_to_world,
            depth_weight=depth_weight,
            huber_delta_m=huber_delta_m,
            robust=robust,
        )

    grid = np.arange(-max_omega, max_omega + 0.5 * grid_step, grid_step)
    losses = np.asarray([fit_at(float(omega), False)[1] for omega in grid])
    best = int(np.argmin(losses))
    lower = max(-max_omega, float(grid[best] - grid_step))
    upper = min(max_omega, float(grid[best] + grid_step))
    if upper - lower > 1e-9:
        result = minimize_scalar(
            lambda value: fit_at(float(value), False)[1],
            bounds=(lower, upper),
            method="bounded",
        )
        omega = float(result.x)
    else:
        omega = float(grid[best])
    coefficient, loss = fit_at(omega, True)
    return omega, coefficient, loss


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


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "split_role",
        "session_id",
        "history_window_s",
        "horizon_s",
        "future_regime",
        "omega_source",
        "depth_weight",
        "huber_delta_m",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    result: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        item = dict(zip(keys, group_key))
        for metric in ("error_cross_depth_m", "error_depth_m", "error_3d_m"):
            for name, value in quantiles(row[metric] for row in group_rows).items():
                item[f"{metric}_{name}"] = value
        result.append(item)
    return result


def pooled_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "split_role",
        "history_window_s",
        "horizon_s",
        "future_regime",
        "omega_source",
        "depth_weight",
        "huber_delta_m",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        item = dict(zip(keys, group_key))
        for metric in ("error_cross_depth_m", "error_depth_m", "error_3d_m"):
            for name, value in quantiles(row[metric] for row in group_rows).items():
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
    base = load_base(repo)
    dataset_root = workspace / "dataset" / "autoaim-stage3-v1"
    runtime_root = workspace / "runtime" / "stage3-generalization-fixed6mm-20260729-v1"

    prediction_rows: list[dict[str, Any]] = []
    omega_rows: list[dict[str, Any]] = []
    source_paths: set[Path] = {Path(__file__).resolve(), Path(base.__file__).resolve()}
    for suffix in DEV_SESSION_SUFFIXES:
        session_id = SESSION_PREFIX + suffix
        frames, _manifest, sources = base.load_session(dataset_root, runtime_root, session_id)
        source_paths.update(sources)
        split_role = "validation" if suffix == "00" else "train"
        geometry = frames[0].armor_local_m
        for anchor_index in range(0, len(frames), args.anchor_stride):
            anchor = frames[anchor_index]
            current_slots = np.flatnonzero(anchor.observed_mask)
            if current_slots.size == 0:
                continue
            primary = int(current_slots[np.argmin(anchor.observed_range_m[current_slots])])
            for window_s in HISTORY_WINDOWS_S:
                history = base.history_observations(
                    frames, anchor_index, window_s, "pnp"
                )
                if history is None:
                    continue
                times, slots, positions = history
                for depth_weight in DEPTH_WEIGHTS:
                    # The coarse/refined omega search uses ordinary weighted
                    # residuals and is independent of the later Huber delta.
                    # Search once, then robustly refit coefficients per delta.
                    estimated_omega, _, _ = estimate_weighted_omega(
                        base,
                        times,
                        slots,
                        positions,
                        geometry,
                        anchor.tracker_to_world,
                        depth_weight=depth_weight,
                        huber_delta_m=HUBER_DELTAS_M[0],
                        max_omega=args.max_omega,
                        grid_step=args.omega_grid_step,
                    )
                    for huber_delta_m in HUBER_DELTAS_M:
                        oracle_coefficient, oracle_loss = fit_weighted_rigid(
                            base,
                            times,
                            slots,
                            positions,
                            geometry,
                            anchor.yaw_rate_rad_s,
                            anchor.tracker_to_world,
                            depth_weight=depth_weight,
                            huber_delta_m=huber_delta_m,
                            robust=True,
                        )
                        estimated_coefficient, estimated_loss = fit_weighted_rigid(
                            base,
                            times,
                            slots,
                            positions,
                            geometry,
                            estimated_omega,
                            anchor.tracker_to_world,
                            depth_weight=depth_weight,
                            huber_delta_m=huber_delta_m,
                            robust=True,
                        )
                        omega_rows.append(
                            {
                                "split_role": split_role,
                                "session_id": session_id,
                                "timestamp_ns": anchor.timestamp_ns,
                                "segment_id": anchor.segment_id,
                                "history_window_s": window_s,
                                "depth_weight": depth_weight,
                                "huber_delta_m": huber_delta_m,
                                "truth_omega_rad_s": anchor.yaw_rate_rad_s,
                                "estimated_omega_rad_s": estimated_omega,
                                "absolute_error_rad_s": abs(
                                    estimated_omega - anchor.yaw_rate_rad_s
                                ),
                                "sign_correct": int(
                                    math.copysign(1.0, estimated_omega or 1.0)
                                    == math.copysign(1.0, anchor.yaw_rate_rad_s)
                                ),
                                "fit_loss": estimated_loss,
                                "history_event_count": int(len(times)),
                                "history_time_span_s": float(np.ptp(times)),
                            }
                        )
                        fits = (
                            ("oracle", anchor.yaw_rate_rad_s, oracle_coefficient, oracle_loss),
                            ("estimated", estimated_omega, estimated_coefficient, estimated_loss),
                        )
                        for horizon_s in HORIZONS_S:
                            future = base.nearest_future(frames, anchor_index, horizon_s)
                            if future is None:
                                continue
                            effective_horizon_s = (
                                future.timestamp_ns - anchor.timestamp_ns
                            ) / 1e9
                            future_regime = (
                                "constant_twist"
                                if future.segment_id == anchor.segment_id
                                else "cross_reversal"
                            )
                            truth = future.armor_world_m[primary]
                            for omega_source, omega, coefficient, loss in fits:
                                prediction = base.predict_rigid(
                                    coefficient,
                                    geometry,
                                    primary,
                                    omega,
                                    effective_horizon_s,
                                )
                                row = {
                                    "split_role": split_role,
                                    "session_id": session_id,
                                    "timestamp_ns": anchor.timestamp_ns,
                                    "segment_id": anchor.segment_id,
                                    "history_window_s": window_s,
                                    "horizon_s": horizon_s,
                                    "effective_horizon_s": effective_horizon_s,
                                    "future_regime": future_regime,
                                    "primary_slot": primary,
                                    "visible_count_at_anchor": int(current_slots.size),
                                    "omega_source": omega_source,
                                    "depth_weight": depth_weight,
                                    "huber_delta_m": huber_delta_m,
                                    "truth_omega_rad_s": anchor.yaw_rate_rad_s,
                                    "model_omega_rad_s": omega,
                                    "fit_loss": loss,
                                    "prediction_x_m": float(prediction[0]),
                                    "prediction_y_m": float(prediction[1]),
                                    "prediction_z_m": float(prediction[2]),
                                    "truth_x_m": float(truth[0]),
                                    "truth_y_m": float(truth[1]),
                                    "truth_z_m": float(truth[2]),
                                }
                                row.update(
                                    base.error_components(
                                        prediction, truth, anchor.tracker_to_world
                                    )
                                )
                                prediction_rows.append(row)

    summary_rows = summarize(prediction_rows)
    pooled_rows = pooled_summary(prediction_rows)
    write_csv_gz(output / "prediction_distribution.csv.gz", prediction_rows)
    write_csv_gz(output / "omega_distribution.csv.gz", omega_rows)
    write_csv(output / "prediction_condition_summary.csv", summary_rows)
    write_csv(output / "prediction_pooled_summary.csv", pooled_rows)
    artifacts = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append(
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    write_json(
        output / "manifest.json",
        {
            "schema_version": "combined-motion-los-fit-development-v1",
            "test_split_accessed": False,
            "sealed_test_session": SESSION_PREFIX + "04",
            "development_sessions": [SESSION_PREFIX + suffix for suffix in DEV_SESSION_SUFFIXES],
            "selection_contract": {
                "objective": "minimize PnP cross-depth error; retain depth and 3d as constraints",
                "train_sessions": [SESSION_PREFIX + suffix for suffix in ("01", "02", "03", "05")],
                "validation_session": SESSION_PREFIX + "00",
                "no_test_tuning": True,
            },
            "candidate_grid": {
                "depth_squared_weights": list(DEPTH_WEIGHTS),
                "huber_deltas_m": list(HUBER_DELTAS_M),
                "history_windows_s": list(HISTORY_WINDOWS_S),
                "horizons_s": list(HORIZONS_S),
                "omega_sources": ["oracle", "estimated"],
                "omega_grid_step_rad_s": args.omega_grid_step,
                "max_abs_omega_rad_s": args.max_omega,
                "anchor_stride_frames": args.anchor_stride,
            },
            "metric_contract": {
                "fit_frame": "anchor tracker frame",
                "depth_axis": "tracker x",
                "cross_depth": "sqrt(tracker_y_error^2 + tracker_z_error^2)",
                "complete_signed_tracker_components_retained": True,
            },
            "sources": [
                {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in sorted(source_paths, key=str)
            ],
            "artifacts": artifacts,
            "row_counts": {
                "predictions": len(prediction_rows),
                "omega_estimates": len(omega_rows),
                "condition_summaries": len(summary_rows),
                "pooled_summaries": len(pooled_rows),
            },
        },
    )


if __name__ == "__main__":
    main()
