#!/usr/bin/env python3
"""Directly fit bounded translation plus rigid rotation to armor tracks.

Unlike center-first reversal experiments, this model never reconstructs a
center per frame.  For configured path axis, speed and span it searches the
causal triangle-wave phase, while fitting the shared center/rotation phase
directly to all visible armor observations.  The development split only is
loaded and all candidate-window distributions are retained.
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


SESSION_PREFIX = "stage3-generalization-fixed6mm-20260729-v1-combined-"
DEV_SESSION_SUFFIXES = ("00", "01", "02", "03", "05")
DIRECT_HISTORY_WINDOWS_S = (1.0, 2.0, 4.0)
LOCAL_HISTORY_WINDOW_S = 0.25
HORIZONS_S = (0.05, 0.10, 0.20)
DEPTH_WEIGHT = 0.1
HUBER_DELTA_M = 0.02
OMEGA_MEMORY_COUNT = 31


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-stride", type=int, default=12)
    parser.add_argument("--omega-grid-step", type=float, default=1.0)
    parser.add_argument("--max-omega", type=float, default=16.0)
    parser.add_argument("--phase-grid-count", type=int, default=49)
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


def triangle_value(phase_m: np.ndarray | float, span_m: float) -> np.ndarray:
    phase = np.mod(np.asarray(phase_m, dtype=np.float64), 2.0 * span_m)
    return np.where(phase <= span_m, phase, 2.0 * span_m - phase)


def translation_delta(
    times_s: np.ndarray,
    phase_m: float,
    speed_mps: float,
    span_m: float,
    axis: np.ndarray,
) -> np.ndarray:
    scalar = triangle_value(phase_m + speed_mps * times_s, span_m)
    anchor_scalar = float(triangle_value(phase_m, span_m))
    return (scalar - anchor_scalar)[:, None] * axis[None, :]


def direct_history(base, frames, anchor_index: int, window_s: float, domain: str):
    anchor = frames[anchor_index]
    start_ns = anchor.timestamp_ns - int(round(window_s * 1e9))
    rows = []
    first_timestamp = anchor.timestamp_ns
    for index in range(anchor_index, -1, -1):
        frame = frames[index]
        if frame.timestamp_ns < start_ns:
            break
        if not frame.observed_mask.any():
            continue
        first_timestamp = min(first_timestamp, frame.timestamp_ns)
        for slot_raw in np.flatnonzero(frame.observed_mask):
            slot = int(slot_raw)
            position = (
                frame.armor_world_m[slot]
                if domain == "clean"
                else frame.observed_world_m[slot]
            )
            rows.append(
                ((frame.timestamp_ns - anchor.timestamp_ns) / 1e9, slot, position)
            )
    if not rows or (anchor.timestamp_ns - first_timestamp) / 1e9 < 0.90 * window_s:
        return None
    rows.reverse()
    return (
        np.asarray([row[0] for row in rows], dtype=np.float64),
        np.asarray([row[1] for row in rows], dtype=np.int64),
        np.stack([row[2] for row in rows]),
    )


def fit_direct(
    base,
    times_s: np.ndarray,
    slots: np.ndarray,
    positions_m: np.ndarray,
    geometry_m: np.ndarray,
    tracker_to_world: np.ndarray,
    *,
    omega: float,
    phase_m: float,
    speed_mps: float,
    span_m: float,
    axis: np.ndarray,
    robust: bool,
) -> tuple[np.ndarray, float]:
    full_design, z_offset = base.rigid_design(times_s, slots, geometry_m, omega)
    design = full_design[:, [0, 1, 2, 6, 7]]
    translated = positions_m - translation_delta(
        times_s, phase_m, speed_mps, span_m, axis
    )
    target = translated.reshape(-1).copy()
    target[2::3] -= z_offset[2::3]
    blocks = design.reshape(-1, 3, design.shape[1])
    targets = target.reshape(-1, 3)
    metric = np.diag([math.sqrt(DEPTH_WEIGHT), 1.0, 1.0]) @ tracker_to_world.T
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
            1.0, HUBER_DELTA_M / np.maximum(norm, np.finfo(float).eps)
        )
    residual = (flat_design @ coefficient - flat_target).reshape(-1, 3)
    norm = np.linalg.norm(residual, axis=1)
    if robust:
        loss = float(
            np.mean(
                np.where(
                    norm <= HUBER_DELTA_M,
                    0.5 * norm**2,
                    HUBER_DELTA_M * (norm - 0.5 * HUBER_DELTA_M),
                )
            )
        )
    else:
        loss = float(np.mean(norm**2))
    return coefficient, loss


def estimate_phase(
    base,
    times_s: np.ndarray,
    slots: np.ndarray,
    positions_m: np.ndarray,
    geometry_m: np.ndarray,
    tracker_to_world: np.ndarray,
    *,
    omega: float,
    speed_mps: float,
    span_m: float,
    axis: np.ndarray,
    phase_grid_count: int,
    robust: bool,
) -> tuple[float, np.ndarray, float]:
    period_m = 2.0 * span_m
    grid = np.linspace(0.0, period_m, phase_grid_count, endpoint=False)

    # For a fixed omega the design matrix is independent of translation
    # phase.  Reuse one pseudoinverse throughout the phase search; only the
    # translated target vector changes.  The selected phase is still robustly
    # refit below, so this is an exact acceleration of the original ordinary
    # least-squares search rather than a change of experiment.
    full_design, z_offset = base.rigid_design(times_s, slots, geometry_m, omega)
    design = full_design[:, [0, 1, 2, 6, 7]]
    blocks = design.reshape(-1, 3, design.shape[1])
    metric = np.diag([math.sqrt(DEPTH_WEIGHT), 1.0, 1.0]) @ tracker_to_world.T
    weighted_blocks = np.einsum("ab,nbc->nac", metric, blocks)
    flat_design = weighted_blocks.reshape(-1, design.shape[1])
    pseudoinverse = np.linalg.pinv(flat_design)

    def objective(phase: float) -> float:
        translated = positions_m - translation_delta(
            times_s,
            float(phase % period_m),
            speed_mps,
            span_m,
            axis,
        )
        target = translated.reshape(-1).copy()
        target[2::3] -= z_offset[2::3]
        weighted_target = np.einsum(
            "ab,nb->na", metric, target.reshape(-1, 3)
        ).reshape(-1)
        coefficient = pseudoinverse @ weighted_target
        residual = (flat_design @ coefficient - weighted_target).reshape(-1, 3)
        return float(np.mean(np.linalg.norm(residual, axis=1) ** 2))

    losses = np.asarray([objective(float(phase)) for phase in grid])
    best = int(np.argmin(losses))
    step = period_m / phase_grid_count
    result = minimize_scalar(
        objective,
        bounds=(float(grid[best] - step), float(grid[best] + step)),
        method="bounded",
    )
    phase = float(result.x % period_m)
    coefficient, loss = fit_direct(
        base,
        times_s,
        slots,
        positions_m,
        geometry_m,
        tracker_to_world,
        omega=omega,
        phase_m=phase,
        speed_mps=speed_mps,
        span_m=span_m,
        axis=axis,
        robust=robust,
    )
    return phase, coefficient, loss


def predict_direct(
    base,
    coefficient: np.ndarray,
    geometry_m: np.ndarray,
    slot: int,
    *,
    omega: float,
    phase_m: float,
    speed_mps: float,
    span_m: float,
    axis: np.ndarray,
    horizon_s: float,
) -> np.ndarray:
    expanded = np.zeros(8, dtype=np.float64)
    expanded[:3] = coefficient[:3]
    expanded[6:8] = coefficient[3:5]
    rigid = base.predict_rigid(expanded, geometry_m, slot, omega, horizon_s)
    delta = translation_delta(
        np.asarray([horizon_s]), phase_m, speed_mps, span_m, axis
    )[0]
    return rigid + delta


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
    keys.extend(
        ["input_domain", "history_window_s", "future_regime", "method", "horizon_s"]
    )
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
    los_path = repo / "scripts" / "evaluate-combined-motion-los-fit.py"
    base = load_module("combined_factorization_v1_direct", base_path)
    los = load_module("combined_los_fit_v1_direct", los_path)
    dataset_root = workspace / "dataset" / "autoaim-stage3-v1"
    runtime_root = workspace / "runtime" / "stage3-generalization-fixed6mm-20260729-v1"

    prediction_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    source_paths: set[Path] = {Path(__file__).resolve(), base_path, los_path}
    for suffix in DEV_SESSION_SUFFIXES:
        session_id = SESSION_PREFIX + suffix
        frames, manifest, sources = base.load_session(dataset_root, runtime_root, session_id)
        source_paths.update(sources)
        split_role = "validation" if suffix == "00" else "train"
        geometry = frames[0].armor_local_m
        direction = math.radians(float(manifest["direction_deg"]))
        axis = np.asarray([math.cos(direction), math.sin(direction), 0.0])
        speed_mps = float(manifest["linear_speed_mps"])
        span_m = float(manifest["linear_span_m"])
        raw_omega_history: list[float] = []
        for anchor_index in range(0, len(frames), args.anchor_stride):
            anchor = frames[anchor_index]
            current_slots = np.flatnonzero(anchor.observed_mask)
            if current_slots.size == 0:
                continue
            local = base.history_observations(
                frames, anchor_index, LOCAL_HISTORY_WINDOW_S, "pnp"
            )
            if local is None:
                continue
            local_times, local_slots, local_positions = local
            raw_omega, _, _ = los.estimate_weighted_omega(
                base,
                local_times,
                local_slots,
                local_positions,
                geometry,
                anchor.tracker_to_world,
                depth_weight=DEPTH_WEIGHT,
                huber_delta_m=HUBER_DELTA_M,
                max_omega=args.max_omega,
                grid_step=args.omega_grid_step,
            )
            raw_omega_history.append(raw_omega)
            memory_omega = float(np.median(raw_omega_history[-OMEGA_MEMORY_COUNT:]))
            primary = int(current_slots[np.argmin(anchor.observed_range_m[current_slots])])
            for history_window_s in DIRECT_HISTORY_WINDOWS_S:
                histories = {
                    domain: direct_history(
                        base, frames, anchor_index, history_window_s, domain
                    )
                    for domain in ("clean", "pnp")
                }
                if any(history is None for history in histories.values()):
                    continue
                fits = {}
                for domain, omega, method in (
                    ("clean", anchor.yaw_rate_rad_s, "direct_oracle_omega"),
                    ("pnp", anchor.yaw_rate_rad_s, "direct_oracle_omega"),
                    ("pnp", memory_omega, "direct_memory_omega"),
                ):
                    times, slots, positions = histories[domain]
                    phase, coefficient, fit_loss = estimate_phase(
                        base,
                        times,
                        slots,
                        positions,
                        geometry,
                        anchor.tracker_to_world,
                        omega=omega,
                        speed_mps=speed_mps,
                        span_m=span_m,
                        axis=axis,
                        phase_grid_count=args.phase_grid_count,
                        robust=domain == "pnp",
                    )
                    fits[(domain, method)] = (omega, phase, coefficient, fit_loss)
                    phase_rows.append(
                        {
                            "split_role": split_role,
                            "session_id": session_id,
                            "timestamp_ns": anchor.timestamp_ns,
                            "history_window_s": history_window_s,
                            "input_domain": domain,
                            "method": method,
                            "model_omega_rad_s": omega,
                            "truth_omega_rad_s": anchor.yaw_rate_rad_s,
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
                    for (domain, method), (omega, phase, coefficient, fit_loss) in fits.items():
                        prediction = predict_direct(
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
                            "history_window_s": history_window_s,
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
                            base.error_components(
                                prediction, truth, anchor.tracker_to_world
                            )
                        )
                        prediction_rows.append(row)

    condition_summary = summarize(prediction_rows, include_session=True)
    pooled_summary = summarize(prediction_rows, include_session=False)
    write_csv_gz(output / "prediction_distribution.csv.gz", prediction_rows)
    write_csv_gz(output / "phase_fit_distribution.csv.gz", phase_rows)
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
            "schema_version": "combined-motion-direct-reflection-development-v1",
            "test_split_accessed": False,
            "sealed_test_session": SESSION_PREFIX + "04",
            "model": "direct armor fit: configured bounded-linear triangle wave plus rigid yaw rotation",
            "no_per_frame_center_reconstruction": True,
            "configured_kinematics": ["path axis", "path span", "linear speed"],
            "candidate_history_windows_s": list(DIRECT_HISTORY_WINDOWS_S),
            "frozen_observation_fit": {
                "depth_squared_weight": DEPTH_WEIGHT,
                "huber_delta_m": HUBER_DELTA_M,
                "omega_memory_count": OMEGA_MEMORY_COUNT,
                "phase_grid_count": args.phase_grid_count,
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
                "phase_fits": len(phase_rows),
                "condition_summaries": len(condition_summary),
                "pooled_summaries": len(pooled_summary),
            },
        },
    )


if __name__ == "__main__":
    main()
