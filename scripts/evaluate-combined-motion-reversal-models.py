#!/usr/bin/env python3
"""Evaluate bounded-translation reversal models for combined rigid motion.

This development-only experiment keeps the factorized rotation model and
compares future center-motion hypotheses:

* continue / hold / immediate reverse;
* an oracle best-of-three set-coverage diagnostic;
* bounded reflection with oracle motion parameters;
* bounded reflection learned causally from past truth center state;
* bounded reflection learned causally from past PnP latent-center fits.

The oracle rows are interventions, not deployable predictors.  Complete
per-prediction distributions are retained and the sealed test is not loaded.
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
HISTORY_WINDOW_S = 0.25
HORIZONS_S = (0.05, 0.10, 0.20)
DEPTH_WEIGHT = 0.1
HUBER_DELTA_M = 0.02
OMEGA_MEMORY_COUNT = 31
METHODS = (
    "continue",
    "hold_center",
    "immediate_reverse",
    "best_of_three_oracle_set",
    "reflection_oracle_kinematics",
    "reflection_causal_truth_state",
    "reflection_causal_latent_minmax",
    "reflection_causal_latent_known_span",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-stride", type=int, default=6)
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


def normalized(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return None
    return np.asarray(vector, dtype=np.float64) / norm


def principal_axis(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 3:
        return None
    planar = np.asarray(points, dtype=np.float64)[:, :2]
    centered = planar - np.mean(planar, axis=0)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    if singular[0] < 1e-4:
        return None
    axis = np.asarray([vh[0, 0], vh[0, 1], 0.0], dtype=np.float64)
    return normalized(axis)


def reflected_scalar(position: float, velocity: float, lower: float, upper: float, dt: float) -> float:
    span = upper - lower
    if span <= 1e-6:
        return position + velocity * dt
    clipped = float(np.clip(position, lower, upper))
    offset = clipped - lower
    phase = offset if velocity >= 0.0 else 2.0 * span - offset
    wrapped = (phase + abs(velocity) * dt) % (2.0 * span)
    return lower + (wrapped if wrapped <= span else 2.0 * span - wrapped)


def reflection_delta(
    current_center: np.ndarray,
    current_velocity: np.ndarray,
    axis: np.ndarray,
    lower: float,
    upper: float,
    horizon_s: float,
    *,
    speed_override: float | None = None,
) -> np.ndarray:
    scalar = float(current_center @ axis)
    velocity = float(current_velocity @ axis)
    if speed_override is not None:
        velocity = math.copysign(float(speed_override), velocity or 1.0)
    future = reflected_scalar(scalar, velocity, lower, upper, horizon_s)
    # The bounded-translation hypothesis is one-dimensional.  Any fitted
    # velocity perpendicular to the learned path is observation noise, not a
    # physical center-motion degree of freedom.
    return axis * (future - scalar)


def learned_bounds(
    centers: list[np.ndarray],
    velocities: list[np.ndarray],
    *,
    known_span: float | None,
) -> tuple[np.ndarray, float, float, float] | None:
    if len(centers) < 8:
        return None
    points = np.stack(centers)
    axis = principal_axis(points)
    if axis is None:
        return None
    projected_velocity = np.asarray([velocity @ axis for velocity in velocities])
    # A bounded model is activated only after both travel directions have
    # actually been observed.  This uses no future state.
    tolerance = max(0.05, 0.10 * float(np.median(np.abs(projected_velocity))))
    if not (
        np.any(projected_velocity > tolerance)
        and np.any(projected_velocity < -tolerance)
    ):
        return None
    projected = points @ axis
    observed_lower = float(np.quantile(projected, 0.01))
    observed_upper = float(np.quantile(projected, 0.99))
    if known_span is None:
        lower, upper = observed_lower, observed_upper
    else:
        midpoint = 0.5 * (observed_lower + observed_upper)
        lower = midpoint - 0.5 * known_span
        upper = midpoint + 0.5 * known_span
    speed = float(np.median(np.abs(projected_velocity[np.abs(projected_velocity) > tolerance])))
    if upper - lower < 0.05 or speed < 0.05:
        return None
    return axis, lower, upper, speed


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
        item["model_active_fraction"] = float(
            np.mean([row["model_active"] for row in group])
        )
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
    base = load_module("combined_factorization_v1_reversal", base_path)
    los = load_module("combined_los_fit_v1_reversal", los_path)
    dataset_root = workspace / "dataset" / "autoaim-stage3-v1"
    runtime_root = workspace / "runtime" / "stage3-generalization-fixed6mm-20260729-v1"

    prediction_rows: list[dict[str, Any]] = []
    source_paths: set[Path] = {Path(__file__).resolve(), base_path, los_path}
    for suffix in DEV_SESSION_SUFFIXES:
        session_id = SESSION_PREFIX + suffix
        frames, manifest, sources = base.load_session(dataset_root, runtime_root, session_id)
        source_paths.update(sources)
        split_role = "validation" if suffix == "00" else "train"
        geometry = frames[0].armor_local_m
        global_centers = np.stack([frame.center_world_m for frame in frames])
        global_axis = principal_axis(global_centers)
        if global_axis is None:
            raise RuntimeError(f"no translation axis for {session_id}")
        global_projection = global_centers @ global_axis
        global_lower = float(np.min(global_projection))
        global_upper = float(np.max(global_projection))
        global_speed = float(manifest["linear_speed_mps"])

        raw_omega_history: list[float] = []
        latent_centers: dict[str, list[np.ndarray]] = {"clean": [], "pnp": []}
        latent_velocities: dict[str, list[np.ndarray]] = {"clean": [], "pnp": []}
        for anchor_index in range(0, len(frames), args.anchor_stride):
            anchor = frames[anchor_index]
            current_slots = np.flatnonzero(anchor.observed_mask)
            if current_slots.size == 0:
                continue
            clean_history = base.history_observations(
                frames, anchor_index, HISTORY_WINDOW_S, "clean"
            )
            pnp_history = base.history_observations(
                frames, anchor_index, HISTORY_WINDOW_S, "pnp"
            )
            if clean_history is None or pnp_history is None:
                continue
            clean_times, clean_slots, clean_positions = clean_history
            pnp_times, pnp_slots, pnp_positions = pnp_history
            raw_omega, _, _ = los.estimate_weighted_omega(
                base,
                pnp_times,
                pnp_slots,
                pnp_positions,
                geometry,
                anchor.tracker_to_world,
                depth_weight=DEPTH_WEIGHT,
                huber_delta_m=HUBER_DELTA_M,
                max_omega=args.max_omega,
                grid_step=args.omega_grid_step,
            )
            raw_omega_history.append(raw_omega)
            pnp_omega = float(np.median(raw_omega_history[-OMEGA_MEMORY_COUNT:]))
            clean_coefficient, _ = base.fit_rigid(
                clean_times,
                clean_slots,
                clean_positions,
                geometry,
                anchor.yaw_rate_rad_s,
                robust=False,
                huber_delta_m=HUBER_DELTA_M,
            )
            pnp_coefficient, _ = los.fit_weighted_rigid(
                base,
                pnp_times,
                pnp_slots,
                pnp_positions,
                geometry,
                pnp_omega,
                anchor.tracker_to_world,
                depth_weight=DEPTH_WEIGHT,
                huber_delta_m=HUBER_DELTA_M,
                robust=True,
            )
            fits = {
                "clean": (clean_coefficient, anchor.yaw_rate_rad_s),
                "pnp": (pnp_coefficient, pnp_omega),
            }
            for domain, (coefficient, _omega) in fits.items():
                latent_centers[domain].append(coefficient[:3].copy())
                latent_velocities[domain].append(coefficient[3:6].copy())

            truth_past_centers = [frame.center_world_m for frame in frames[: anchor_index + 1]]
            truth_past_velocities = [frame.velocity_world_mps for frame in frames[: anchor_index + 1]]
            truth_learned = learned_bounds(
                truth_past_centers, truth_past_velocities, known_span=None
            )
            if truth_learned is not None:
                truth_axis, _, _, truth_speed = truth_learned
                truth_projection = np.stack(truth_past_centers) @ truth_axis
                truth_learned = (
                    truth_axis,
                    float(np.min(truth_projection)),
                    float(np.max(truth_projection)),
                    truth_speed,
                )
            primary = int(current_slots[np.argmin(anchor.observed_range_m[current_slots])])
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
                for domain, (coefficient, omega) in fits.items():
                    current_center = coefficient[:3]
                    current_velocity = coefficient[3:6]
                    ordinary = base.predict_rigid(
                        coefficient, geometry, primary, omega, effective_horizon_s
                    )
                    ordinary_center = current_center + current_velocity * effective_horizon_s
                    rotation_part = ordinary - ordinary_center
                    center_predictions: dict[str, tuple[np.ndarray, bool]] = {
                        "continue": (ordinary_center, True),
                        "hold_center": (current_center, True),
                        "immediate_reverse": (
                            current_center - current_velocity * effective_horizon_s,
                            True,
                        ),
                    }
                    oracle_delta = reflection_delta(
                        anchor.center_world_m,
                        anchor.velocity_world_mps,
                        global_axis,
                        global_lower,
                        global_upper,
                        effective_horizon_s,
                        speed_override=global_speed,
                    )
                    center_predictions["reflection_oracle_kinematics"] = (
                        current_center + oracle_delta,
                        True,
                    )
                    if truth_learned is None:
                        center_predictions["reflection_causal_truth_state"] = (
                            ordinary_center,
                            False,
                        )
                    else:
                        axis, lower, upper, speed = truth_learned
                        delta = reflection_delta(
                            anchor.center_world_m,
                            anchor.velocity_world_mps,
                            axis,
                            lower,
                            upper,
                            effective_horizon_s,
                            speed_override=speed,
                        )
                        center_predictions["reflection_causal_truth_state"] = (
                            current_center + delta,
                            True,
                        )
                    for method, known_span in (
                        ("reflection_causal_latent_minmax", None),
                        (
                            "reflection_causal_latent_known_span",
                            float(manifest["linear_span_m"]),
                        ),
                    ):
                        learned = learned_bounds(
                            latent_centers[domain],
                            latent_velocities[domain],
                            known_span=known_span,
                        )
                        if learned is None:
                            center_predictions[method] = (ordinary_center, False)
                        else:
                            axis, lower, upper, speed = learned
                            delta = reflection_delta(
                                current_center,
                                current_velocity,
                                axis,
                                lower,
                                upper,
                                effective_horizon_s,
                                speed_override=speed,
                            )
                            center_predictions[method] = (
                                current_center + delta,
                                True,
                            )

                    simple_predictions = [
                        center_predictions[name][0] + rotation_part
                        for name in ("continue", "hold_center", "immediate_reverse")
                    ]
                    best_prediction = min(
                        simple_predictions, key=lambda value: np.linalg.norm(value - truth)
                    )
                    center_predictions["best_of_three_oracle_set"] = (
                        best_prediction - rotation_part,
                        True,
                    )
                    for method in METHODS:
                        predicted_center, active = center_predictions[method]
                        prediction = predicted_center + rotation_part
                        row = {
                            "split_role": split_role,
                            "session_id": session_id,
                            "timestamp_ns": anchor.timestamp_ns,
                            "segment_id": anchor.segment_id,
                            "future_segment_id": future.segment_id,
                            "input_domain": domain,
                            "future_regime": future_regime,
                            "history_window_s": HISTORY_WINDOW_S,
                            "horizon_s": horizon_s,
                            "effective_horizon_s": effective_horizon_s,
                            "method": method,
                            "model_active": int(active),
                            "primary_slot": primary,
                            "visible_count_at_anchor": int(current_slots.size),
                            "truth_omega_rad_s": anchor.yaw_rate_rad_s,
                            "model_omega_rad_s": omega,
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
            "schema_version": "combined-motion-reversal-development-v1",
            "test_split_accessed": False,
            "sealed_test_session": SESSION_PREFIX + "04",
            "frozen_local_fit": {
                "history_window_s": HISTORY_WINDOW_S,
                "depth_squared_weight": DEPTH_WEIGHT,
                "huber_delta_m": HUBER_DELTA_M,
                "omega_memory_count": OMEGA_MEMORY_COUNT,
            },
            "candidate_methods": list(METHODS),
            "method_scope": {
                "best_of_three_oracle_set": "non-deployable set coverage diagnostic",
                "reflection_oracle_kinematics": "oracle axis, endpoints, speed and direction state",
                "reflection_causal_truth_state": "past/current truth center and velocity only",
                "reflection_causal_latent_minmax": "past/current fitted latent centers and velocities only",
                "reflection_causal_latent_known_span": "past/current latent state plus configured path span",
            },
            "causality": {
                "future_truth_used_only_for_scoring_except_explicit_oracle_set_selection": True,
                "truth_segment_boundary_used_by_local_v1_history": True,
                "causal_reflection_activates_after_both_directions_observed": True,
            },
            "sources": [
                {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in sorted(source_paths, key=str)
            ],
            "artifacts": artifacts,
            "row_counts": {
                "predictions": len(prediction_rows),
                "condition_summaries": len(condition_summary),
                "pooled_summaries": len(pooled_summary),
            },
        },
    )


if __name__ == "__main__":
    main()
