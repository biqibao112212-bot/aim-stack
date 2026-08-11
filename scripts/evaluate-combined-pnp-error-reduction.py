#!/usr/bin/env python3
"""Replay downstream PnP correction and causal coordinate interventions.

The anchor, future, identity and physical-model contracts match the retained
144-session combined-motion validation.  Cross-fitted corrections use the
model trained without the scored session fold.  Every truth-restoration arm is
an offline causal ablation with identical timestamps, gaps and visible arcs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HISTORY_S = 4.0
HORIZONS_S = (0.05, 0.10, 0.20)
MAX_OMEGA = 16.0
OMEGA_GRID_STEP = 1.0
PHASE_GRID_COUNT = 25
GATE_M = 0.055
GATES_M = (0.010, 0.025, 0.055, 0.100, 0.200)
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
ALPHAS = (0.75, 0.50, 0.25)
_MODULE_CACHE: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )
    parser.add_argument("--session-limit", type=int, default=None)
    parser.add_argument(
        "--arm-filter",
        default=None,
        help="Optional comma-separated subset for a targeted follow-up replay.",
    )
    return parser.parse_args()


def load_module(name: str, path: Path):
    key = str(path.resolve())
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    spec = importlib.util.spec_from_file_location(f"{name}_{os.getpid()}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[key] = module
    return module


def modules(repo: Path) -> tuple[Any, Any, Any, Any, Any]:
    large = load_module(
        "combined_reduction_large",
        repo / "scripts" / "evaluate-combined-motion-large-scale.py",
    )
    screen = load_module(
        "combined_reduction_screen",
        repo / "scripts" / "analyze-combined-pnp-downstream-correction.py",
    )
    base, direct, joint = large.modules(repo)
    return large, screen, base, direct, joint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_conditions(manifest_path: Path) -> list[dict[str, Any]]:
    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("mode") == "linear_and_spin":
                    rows.append(row)
    return rows


def arm_names(
    promoted: str | None, oracle: str, include_cross_yz: bool = False
) -> list[str]:
    names = ["raw_pnp"]
    if promoted is not None:
        names.append(f"crossfit_{promoted}")
        if include_cross_yz:
            names.append(f"crossfit_{promoted}_cross_yz")
    names.append(f"oracle_{oracle}")
    names.extend(
        (
            "truth_depth_only",
            "truth_tracker_y_only",
            "truth_tracker_z_only",
            "truth_cross_depth_yz",
        )
    )
    names.extend(f"pnp_residual_alpha_{alpha:.2f}" for alpha in ALPHAS)
    names.append("clean_exact")
    return names


def build_arm_histories(
    repo: Path,
    workspace: Path,
    condition: dict[str, Any],
    models_path: Path,
    arm_filter: str | None = None,
) -> tuple[list[Any], dict[str, Any], list[str]]:
    large, screen, base, _, _ = modules(repo)
    frames, audit, _ = large.load_formal_session(repo, workspace, condition)
    payload = json.loads(models_path.read_text(encoding="utf-8-sig"))
    promoted = payload.get("promoted_deployable_arm")
    oracle = str(payload["best_oracle_arm"])
    requested = {
        name.strip() for name in (arm_filter or "").split(",") if name.strip()
    }
    cross_yz_name = (
        f"crossfit_{promoted}_cross_yz" if promoted is not None else None
    )
    names = arm_names(
        promoted,
        oracle,
        include_cross_yz=cross_yz_name is not None and cross_yz_name in requested,
    )
    for frame in frames:
        frame.intervention_world_m = {
            name: np.full((4, 3), np.nan, dtype=np.float64) for name in names
        }
        for slot_raw in np.flatnonzero(frame.observed_mask):
            slot = int(slot_raw)
            observed_world = frame.observed_world_m[slot]
            truth_world = frame.armor_world_m[slot]
            observed_tracker = (
                observed_world - frame.tracker_origin_world_m
            ) @ frame.tracker_to_world
            truth_tracker = (
                truth_world - frame.tracker_origin_world_m
            ) @ frame.tracker_to_world
            tracker_values = {
                "truth_depth_only": np.asarray(
                    [truth_tracker[0], observed_tracker[1], observed_tracker[2]]
                ),
                "truth_tracker_y_only": np.asarray(
                    [observed_tracker[0], truth_tracker[1], observed_tracker[2]]
                ),
                "truth_tracker_z_only": np.asarray(
                    [observed_tracker[0], observed_tracker[1], truth_tracker[2]]
                ),
                "truth_cross_depth_yz": np.asarray(
                    [observed_tracker[0], truth_tracker[1], truth_tracker[2]]
                ),
            }
            frame.intervention_world_m["raw_pnp"][slot] = observed_world
            if promoted is not None:
                frame.intervention_world_m[f"crossfit_{promoted}"][slot] = observed_world
                if cross_yz_name in names:
                    frame.intervention_world_m[cross_yz_name][slot] = observed_world
            frame.intervention_world_m[f"oracle_{oracle}"][slot] = observed_world
            for name, tracker in tracker_values.items():
                frame.intervention_world_m[name][slot] = (
                    frame.tracker_origin_world_m + tracker @ frame.tracker_to_world.T
                )
            for alpha in ALPHAS:
                frame.intervention_world_m[f"pnp_residual_alpha_{alpha:.2f}"][slot] = (
                    truth_world + alpha * (observed_world - truth_world)
                )
            frame.intervention_world_m["clean_exact"][slot] = truth_world

    runtime_root = workspace / "runtime" / "stage3-formal-20260720-v2"
    _, observation_path, _, result_path = large.select_run_paths(runtime_root, condition)
    start_ns = base.scene_motion_start_ns(result_path)
    frame_by_timestamp = {frame.timestamp_ns: frame for frame in frames}
    feature_rows: list[dict[str, Any]] = []
    references: list[tuple[Any, int]] = []
    with observation_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            timestamp_ns = int(record["timestamp_ns"])
            if timestamp_ns < start_ns:
                continue
            frame = frame_by_timestamp.get(timestamp_ns)
            if frame is None:
                continue
            valid = []
            for armor in record.get("armors", []):
                if not bool(armor.get("valid", False)):
                    continue
                position = np.asarray(armor.get("position_m", []), dtype=np.float64)
                yaw = armor.get("yaw_absolute_rad", armor.get("yaw_rad"))
                if position.shape != (3,) or not np.isfinite(position).all():
                    continue
                yaw_value = (
                    float(yaw)
                    if yaw is not None and math.isfinite(float(yaw))
                    else None
                )
                valid.append((position, yaw_value))
            if not valid or len(valid) > 4:
                continue
            local = np.stack([item[0] for item in valid])
            world = frame.tracker_origin_world_m[None, :] + local @ frame.tracker_to_world.T
            slots, ambiguous = base.best_assignment(world, frame.armor_world_m)
            if ambiguous:
                continue
            for observation_index, slot_raw in enumerate(slots):
                slot = int(slot_raw)
                if valid[observation_index][1] is None:
                    continue
                truth_tracker = (
                    frame.armor_world_m[slot] - frame.tracker_origin_world_m
                ) @ frame.tracker_to_world
                radial_tracker = (
                    frame.armor_world_m[slot] - frame.center_world_m
                ) @ frame.tracker_to_world
                observed = local[observation_index]
                feature_rows.append(
                    {
                        "session_id": condition["session_id"],
                        "fold": screen.session_fold(str(condition["session_id"])),
                        "timestamp_ns": timestamp_ns,
                        "slot": slot,
                        "camera_profile_id": str(record.get("camera_profile_id", "missing")),
                        "candidate_count": len(valid),
                        "distance_m": float(condition["distance_m"]),
                        "linear_speed_mps": float(condition["linear_speed_mps"]),
                        "spin_rad_s": float(condition["spin_rad_s"]),
                        "direction_sector": int(condition["direction_sector"]),
                        "obs_x_m": float(observed[0]),
                        "obs_y_m": float(observed[1]),
                        "obs_z_m": float(observed[2]),
                        "pnp_yaw_rad": float(valid[observation_index][1]),
                        "truth_phase_rad": math.atan2(
                            float(radial_tracker[1]), float(radial_tracker[0])
                        ),
                        "truth_x_m": float(truth_tracker[0]),
                        "truth_y_m": float(truth_tracker[1]),
                        "truth_z_m": float(truth_tracker[2]),
                    }
                )
                references.append((frame, slot))

    if not feature_rows:
        return frames, audit, names
    fold = screen.session_fold(str(condition["session_id"]))
    fold_models = payload["models"][str(fold)]
    values: dict[str, np.ndarray] = {}
    if promoted is not None:
        corrected = screen.predict_model(
            feature_rows, fold_models[promoted]
        )
        values[f"crossfit_{promoted}"] = corrected
        if cross_yz_name in names:
            observed, _ = screen.targets(feature_rows)
            values[cross_yz_name] = np.column_stack(
                (observed[:, 0], corrected[:, 1], corrected[:, 2])
            )
    values[f"oracle_{oracle}"] = screen.predict_model(feature_rows, fold_models[oracle])

    for index, (frame, slot) in enumerate(references):
        for name, positions in values.items():
            tracker = positions[index]
            frame.intervention_world_m[name][slot] = (
                frame.tracker_origin_world_m + tracker @ frame.tracker_to_world.T
            )
    audit["correction_fold"] = fold
    audit["promoted_deployable_arm"] = promoted
    audit["best_oracle_arm"] = oracle
    audit["intervention_event_count"] = len(feature_rows)
    return frames, audit, names


def intervention_history(
    frames: list[Any], anchor_index: int, arm: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    anchor = frames[anchor_index]
    start_ns = anchor.timestamp_ns - int(round(HISTORY_S * 1e9))
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
            position = frame.intervention_world_m[arm][slot]
            if np.isfinite(position).all():
                rows.append(
                    ((frame.timestamp_ns - anchor.timestamp_ns) / 1e9, slot, position)
                )
    if not rows or (anchor.timestamp_ns - first_timestamp) / 1e9 < 0.90 * HISTORY_S:
        return None
    rows.reverse()
    return (
        np.asarray([row[0] for row in rows], dtype=np.float64),
        np.asarray([row[1] for row in rows], dtype=np.int64),
        np.stack([row[2] for row in rows]),
    )


def evaluate_session(task: tuple[str, str, str, dict[str, Any], str | None]) -> dict[str, Any]:
    repo_raw, workspace_raw, models_raw, condition, arm_filter = task
    repo = Path(repo_raw)
    workspace = Path(workspace_raw)
    models_path = Path(models_raw)
    session_id = str(condition["session_id"])
    try:
        large, _, base, direct, joint = modules(repo)
        frames, audit, names = build_arm_histories(
            repo, workspace, condition, models_path, arm_filter
        )
        if arm_filter:
            requested = [name.strip() for name in arm_filter.split(",") if name.strip()]
            missing = sorted(set(requested) - set(names))
            if missing:
                raise ValueError(f"unknown arm filter values: {missing}")
            names = requested
        geometry = frames[0].armor_local_m
        direction = math.radians(float(condition["direction_deg"]))
        axis = np.asarray([math.cos(direction), math.sin(direction), 0.0])
        speed_mps = float(condition["linear_speed_mps"])
        span_m = float(condition["linear_span_m"])
        anchors = large.plan_anchors(frames)
        predictions: list[dict[str, Any]] = []
        fits: list[dict[str, Any]] = []
        coverage: list[dict[str, Any]] = []
        for anchor_index, role_labels in sorted(anchors.items()):
            anchor = frames[anchor_index]
            strata = ["regular_grid"] if "regular_grid" in role_labels else []
            if any(label.startswith("reversal_stress_") for label in role_labels):
                strata.append("reversal_stress")
            current_slots = np.flatnonzero(anchor.observed_mask)
            common = {
                "session_id": session_id,
                "timestamp_ns": anchor.timestamp_ns,
                "anchor_index": anchor_index,
                "role_labels": ";".join(sorted(role_labels)),
                "distance_bin": int(condition["distance_bin"]),
                "distance_m": float(condition["distance_m"]),
                "linear_speed_mps": speed_mps,
                "linear_speed_bin": large.speed_bin(speed_mps),
                "truth_omega_rad_s": float(anchor.yaw_rate_rad_s),
                "absolute_yaw_rate_bin": large.omega_bin(anchor.yaw_rate_rad_s),
                "direction_sector": int(condition["direction_sector"]),
                "camera_profile_id": str(anchor.camera_profile_id),
                "truth_segment_id": int(anchor.segment_id),
            }
            if current_slots.size == 0:
                for stratum in strata:
                    coverage.append(
                        {
                            **common,
                            "sample_stratum": stratum,
                            "status": "no_current_observation",
                            "visible_count_at_anchor": 0,
                            "primary_slot": -1,
                            "available_arm_count": 0,
                        }
                    )
                continue
            primary = int(current_slots[np.argmin(anchor.observed_range_m[current_slots])])
            histories = {name: intervention_history(frames, anchor_index, name) for name in names}
            available = [name for name, history in histories.items() if history is not None]
            if len(available) != len(names):
                for stratum in strata:
                    coverage.append(
                        {
                            **common,
                            "sample_stratum": stratum,
                            "status": "incomplete_common_arm_history",
                            "visible_count_at_anchor": int(current_slots.size),
                            "primary_slot": primary,
                            "available_arm_count": len(available),
                        }
                    )
                continue
            arm_fits = {}
            failed = None
            for name in names:
                times, slots, positions = histories[name]
                try:
                    arm_fits[name] = joint.joint_fit(
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
                        max_omega=MAX_OMEGA,
                        omega_grid_step=OMEGA_GRID_STEP,
                        phase_grid_count=PHASE_GRID_COUNT,
                        robust=name != "clean_exact",
                    )
                except Exception as error:
                    failed = f"{name}:{type(error).__name__}"
                    break
            if failed is not None:
                for stratum in strata:
                    coverage.append(
                        {
                            **common,
                            "sample_stratum": stratum,
                            "status": f"fit_failure:{failed}",
                            "visible_count_at_anchor": int(current_slots.size),
                            "primary_slot": primary,
                            "available_arm_count": len(available),
                        }
                    )
                continue
            for stratum in strata:
                coverage.append(
                    {
                        **common,
                        "sample_stratum": stratum,
                        "status": "evaluated_common_anchor",
                        "visible_count_at_anchor": int(current_slots.size),
                        "primary_slot": primary,
                        "available_arm_count": len(available),
                    }
                )
            for name, (omega, phase, coefficient, loss) in arm_fits.items():
                history = histories[name]
                fits.append(
                    {
                        **common,
                        "input_arm": name,
                        "history_event_count": int(len(history[0])),
                        "history_time_span_s": float(np.ptp(history[0])),
                        "model_omega_rad_s": float(omega),
                        "absolute_omega_error_rad_s": abs(
                            float(omega) - float(anchor.yaw_rate_rad_s)
                        ),
                        "omega_sign_correct": int(
                            np.sign(float(omega)) == np.sign(float(anchor.yaw_rate_rad_s))
                        ),
                        "translation_phase_m": float(phase),
                        "fit_loss": float(loss),
                    }
                )
            for horizon_s in HORIZONS_S:
                future = base.nearest_future(frames, anchor_index, horizon_s)
                if future is None:
                    continue
                effective_horizon_s = (future.timestamp_ns - anchor.timestamp_ns) / 1e9
                truth = future.armor_world_m[primary]
                future_regime = (
                    "constant_twist"
                    if future.segment_id == anchor.segment_id
                    else "cross_reversal"
                )
                for name, (omega, phase, coefficient, loss) in arm_fits.items():
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
                    for stratum in strata:
                        row = {
                            **common,
                            "sample_stratum": stratum,
                            "future_regime": future_regime,
                            "future_segment_id": int(future.segment_id),
                            "input_arm": name,
                            "horizon_s": horizon_s,
                            "effective_horizon_s": effective_horizon_s,
                            "primary_slot": primary,
                            "visible_count_at_anchor": int(current_slots.size),
                            "model_omega_rad_s": float(omega),
                            "fit_loss": float(loss),
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
                        row["cross_depth_within_55mm"] = int(
                            row["error_cross_depth_m"] <= GATE_M
                        )
                        predictions.append(row)
        audit["planned_unique_anchor_count"] = len(anchors)
        audit["evaluated_common_anchor_rows"] = sum(
            row["status"] == "evaluated_common_anchor" for row in coverage
        )
        return {
            "session_id": session_id,
            "status": "ok",
            "predictions": predictions,
            "fits": fits,
            "coverage": coverage,
            "audit": audit,
            "arms": names,
        }
    except Exception as error:
        return {
            "session_id": session_id,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "predictions": [],
            "fits": [],
            "coverage": [],
            "audit": {**condition, "status": "error"},
            "arms": [],
        }


def summarize_values(values: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
    }
    for quantile in QUANTILES:
        result[f"p{int(round(100 * quantile)):02d}"] = float(np.quantile(values, quantile))
    for gate in GATES_M:
        result[f"cdf_le_{int(round(gate * 1000)):03d}mm"] = float(np.mean(values <= gate))
    return result


def grouped_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ("sample_stratum", "future_regime", "input_arm", "horizon_s")
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        common = dict(zip(keys, key))
        for metric in (
            "error_cross_depth_m",
            "error_depth_m",
            "error_3d_m",
            "error_tracker_y_m",
            "error_tracker_z_m",
        ):
            values = np.asarray(
                [abs(float(row[metric])) for row in group], dtype=np.float64
            )
            result.append({**common, "metric": metric, **summarize_values(values)})
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_distributions(output: Path, predictions: list[dict[str, Any]], names: list[str]) -> None:
    subset = [
        row
        for row in predictions
        if row["sample_stratum"] == "regular_grid" and row["future_regime"] == "constant_twist"
    ]
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(names)))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3), sharey=True)
    for axis, horizon in zip(axes, HORIZONS_S):
        for color, name in zip(colors, names):
            values = np.sort(
                np.asarray(
                    [
                        row["error_cross_depth_m"]
                        for row in subset
                        if row["input_arm"] == name and float(row["horizon_s"]) == horizon
                    ],
                    dtype=np.float64,
                )
            )
            if values.size == 0:
                continue
            y = np.arange(1, values.size + 1) / values.size
            axis.plot(values * 1000.0, y, color=color, label=name, linewidth=1.5)
        axis.axvline(55.0, color="#222222", linestyle="--", linewidth=1.0)
        axis.set_xlim(0.0, 500.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_title(f"{int(round(horizon * 1000))} ms")
        axis.set_xlabel("Cross-depth prediction error (mm)")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Empirical CDF")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Combined motion: PnP correction and coordinate interventions")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / "prediction_intervention_ecdf.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "prediction_intervention_ecdf.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    workspace = args.workspace.resolve() if args.workspace else repo.parents[1]
    models_path = args.models.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    manifest_path = (
        workspace
        / "dataset"
        / "autoaim-stage3-v1"
        / "stage3-20260719-v1"
        / "session_manifest.jsonl"
    )
    conditions = read_conditions(manifest_path)
    if args.session_limit is not None:
        conditions = conditions[: args.session_limit]
    tasks = [
        (str(repo), str(workspace), str(models_path), condition, args.arm_filter)
        for condition in conditions
    ]
    predictions: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    failures = []
    names: list[str] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(evaluate_session, tasks):
            if result["status"] != "ok":
                failures.append(result)
                continue
            predictions.extend(result["predictions"])
            fits.extend(result["fits"])
            coverage.extend(result["coverage"])
            audits.append(result["audit"])
            names = result["arms"]
    (output / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"{len(failures)} session failures")
    predictions.sort(
        key=lambda row: (
            row["session_id"],
            row["timestamp_ns"],
            row["sample_stratum"],
            row["horizon_s"],
            row["input_arm"],
        )
    )
    fits.sort(key=lambda row: (row["session_id"], row["timestamp_ns"], row["input_arm"]))
    coverage.sort(
        key=lambda row: (row["session_id"], row["timestamp_ns"], row["sample_stratum"])
    )
    summary = grouped_summary(predictions)
    write_csv_gz(output / "prediction_distribution.csv.gz", predictions)
    write_csv_gz(output / "fit_distribution.csv.gz", fits)
    write_csv(output / "anchor_coverage.csv", coverage)
    write_csv(output / "session_audit.csv", audits)
    write_csv(output / "prediction_summary.csv", summary)
    plot_distributions(output, predictions, names)
    manifest = {
        "schema_version": "combined-pnp-error-reduction-replay-v1",
        "formal_complete": args.session_limit is None and len(conditions) == 144,
        "sessions": len(conditions),
        "arms": names,
        "prediction_rows": len(predictions),
        "fit_rows": len(fits),
        "coverage_rows": len(coverage),
        "models": {"path": str(models_path), "sha256": sha256(models_path)},
        "contract": str(
            repo / "modules" / "autoaim" / "docs" / "combined_motion_pnp_error_reduction_contract.json"
        ),
        "sources": {
            "dataset_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        },
        "artifacts": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
