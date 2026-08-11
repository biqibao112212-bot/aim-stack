#!/usr/bin/env python3
"""Frozen final acceptance for the combined-motion research stage.

The method and all numerical parameters are locked by
``combined_motion_final_selection_contract.json``.  ``--session-suffix 00``
is available only to smoke-test the evaluation implementation.  The formal
one-time acceptance uses suffix 04 and does not tune from its results.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SESSION_PREFIX = "stage3-generalization-fixed6mm-20260729-v1-combined-"
TEST_SUFFIX = "04"
LOCAL_HISTORY_S = 0.40
DIRECT_HISTORY_S = 4.0
HORIZONS_S = (0.05, 0.10, 0.20)
ANCHOR_STRIDE = 24
DEPTH_WEIGHT = 0.1
HUBER_DELTA_M = 0.02
OMEGA_MEMORY_COUNT = 31
OMEGA_GRID_STEP = 1.0
MAX_OMEGA = 16.0
PHASE_GRID_COUNT = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-suffix", choices=("00", "04"), default="04")
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
    keys = ("input_domain", "future_regime", "method", "horizon_s")
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
    direct_path = repo / "scripts" / "evaluate-combined-motion-direct-reflection.py"
    joint_path = repo / "scripts" / "evaluate-combined-motion-direct-joint.py"
    contract_path = (
        repo
        / "modules"
        / "autoaim"
        / "docs"
        / "combined_motion_final_selection_contract.json"
    )
    base = load_module("combined_factorization_final", base_path)
    los = load_module("combined_los_final", los_path)
    direct = load_module("combined_direct_final", direct_path)
    joint = load_module("combined_joint_final", joint_path)

    dataset_root = workspace / "dataset" / "autoaim-stage3-v1"
    runtime_root = workspace / "runtime" / "stage3-generalization-fixed6mm-20260729-v1"
    session_id = SESSION_PREFIX + args.session_suffix
    frames, condition, sources = base.load_session(dataset_root, runtime_root, session_id)
    geometry = frames[0].armor_local_m
    direction = math.radians(float(condition["direction_deg"]))
    axis = np.asarray([math.cos(direction), math.sin(direction), 0.0])
    speed_mps = float(condition["linear_speed_mps"])
    span_m = float(condition["linear_span_m"])

    prediction_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    raw_omega_history: list[float] = []
    common_anchor_count = 0
    for anchor_index in range(0, len(frames), ANCHOR_STRIDE):
        anchor = frames[anchor_index]
        current_slots = np.flatnonzero(anchor.observed_mask)
        if current_slots.size == 0:
            continue
        local_pnp = base.history_observations(
            frames, anchor_index, LOCAL_HISTORY_S, "pnp"
        )
        local_clean = base.history_observations(
            frames, anchor_index, LOCAL_HISTORY_S, "clean"
        )
        if local_pnp is None or local_clean is None:
            continue
        pnp_times, pnp_slots, pnp_positions = local_pnp
        clean_times, clean_slots, clean_positions = local_clean
        raw_omega, raw_coefficient, raw_loss = los.estimate_weighted_omega(
            base,
            pnp_times,
            pnp_slots,
            pnp_positions,
            geometry,
            anchor.tracker_to_world,
            depth_weight=DEPTH_WEIGHT,
            huber_delta_m=HUBER_DELTA_M,
            max_omega=MAX_OMEGA,
            grid_step=OMEGA_GRID_STEP,
        )
        raw_omega_history.append(raw_omega)
        direct_pnp = direct.direct_history(
            base, frames, anchor_index, DIRECT_HISTORY_S, "pnp"
        )
        direct_clean = direct.direct_history(
            base, frames, anchor_index, DIRECT_HISTORY_S, "clean"
        )
        if direct_pnp is None or direct_clean is None:
            continue
        common_anchor_count += 1
        primary = int(current_slots[np.argmin(anchor.observed_range_m[current_slots])])
        memory_omega = float(np.median(raw_omega_history[-OMEGA_MEMORY_COUNT:]))
        memory_coefficient, memory_loss = los.fit_weighted_rigid(
            base,
            pnp_times,
            pnp_slots,
            pnp_positions,
            geometry,
            memory_omega,
            anchor.tracker_to_world,
            depth_weight=DEPTH_WEIGHT,
            huber_delta_m=HUBER_DELTA_M,
            robust=True,
        )
        oracle_local_coefficient, oracle_local_loss = los.fit_weighted_rigid(
            base,
            pnp_times,
            pnp_slots,
            pnp_positions,
            geometry,
            anchor.yaw_rate_rad_s,
            anchor.tracker_to_world,
            depth_weight=DEPTH_WEIGHT,
            huber_delta_m=HUBER_DELTA_M,
            robust=True,
        )
        iso_omega, iso_coefficient, iso_loss = base.estimate_omega(
            pnp_times,
            pnp_slots,
            pnp_positions,
            geometry,
            max_omega=MAX_OMEGA,
            grid_step=OMEGA_GRID_STEP,
            robust=True,
            huber_delta_m=0.05,
        )
        clean_local_coefficient, clean_local_loss = base.fit_rigid(
            clean_times,
            clean_slots,
            clean_positions,
            geometry,
            anchor.yaw_rate_rad_s,
            robust=False,
            huber_delta_m=HUBER_DELTA_M,
        )
        slot_active = pnp_slots == primary
        cv_coefficient = base.fit_cv(
            pnp_times[slot_active], pnp_positions[slot_active]
        )

        pnp_direct_times, pnp_direct_slots, pnp_direct_positions = direct_pnp
        clean_direct_times, clean_direct_slots, clean_direct_positions = direct_clean
        direct_omega, direct_phase, direct_coefficient, direct_loss = joint.joint_fit(
            direct,
            base,
            pnp_direct_times,
            pnp_direct_slots,
            pnp_direct_positions,
            geometry,
            anchor.tracker_to_world,
            speed_mps=speed_mps,
            span_m=span_m,
            axis=axis,
            max_omega=MAX_OMEGA,
            omega_grid_step=OMEGA_GRID_STEP,
            phase_grid_count=PHASE_GRID_COUNT,
            robust=True,
        )
        direct_oracle_phase, direct_oracle_coefficient, direct_oracle_loss = (
            direct.estimate_phase(
                base,
                pnp_direct_times,
                pnp_direct_slots,
                pnp_direct_positions,
                geometry,
                anchor.tracker_to_world,
                omega=anchor.yaw_rate_rad_s,
                speed_mps=speed_mps,
                span_m=span_m,
                axis=axis,
                phase_grid_count=PHASE_GRID_COUNT,
                robust=True,
            )
        )
        clean_direct_omega, clean_direct_phase, clean_direct_coefficient, clean_direct_loss = (
            joint.joint_fit(
                direct,
                base,
                clean_direct_times,
                clean_direct_slots,
                clean_direct_positions,
                geometry,
                anchor.tracker_to_world,
                speed_mps=speed_mps,
                span_m=span_m,
                axis=axis,
                max_omega=MAX_OMEGA,
                omega_grid_step=OMEGA_GRID_STEP,
                phase_grid_count=PHASE_GRID_COUNT,
                robust=False,
            )
        )
        fit_rows.append(
            {
                "session_id": session_id,
                "timestamp_ns": anchor.timestamp_ns,
                "segment_id": anchor.segment_id,
                "truth_omega_rad_s": anchor.yaw_rate_rad_s,
                "los_raw_omega_rad_s": raw_omega,
                "los_memory31_omega_rad_s": memory_omega,
                "v1_isotropic_omega_rad_s": iso_omega,
                "direct_joint_omega_rad_s": direct_omega,
                "clean_direct_joint_omega_rad_s": clean_direct_omega,
                "los_raw_absolute_error_rad_s": abs(raw_omega - anchor.yaw_rate_rad_s),
                "los_memory31_absolute_error_rad_s": abs(memory_omega - anchor.yaw_rate_rad_s),
                "v1_isotropic_absolute_error_rad_s": abs(iso_omega - anchor.yaw_rate_rad_s),
                "direct_joint_absolute_error_rad_s": abs(direct_omega - anchor.yaw_rate_rad_s),
                "clean_direct_joint_absolute_error_rad_s": abs(clean_direct_omega - anchor.yaw_rate_rad_s),
                "direct_phase_m": direct_phase,
                "direct_oracle_phase_m": direct_oracle_phase,
                "clean_direct_phase_m": clean_direct_phase,
                "los_raw_fit_loss": raw_loss,
                "los_memory31_fit_loss": memory_loss,
                "los_oracle_fit_loss": oracle_local_loss,
                "v1_isotropic_fit_loss": iso_loss,
                "direct_joint_fit_loss": direct_loss,
                "direct_oracle_fit_loss": direct_oracle_loss,
                "clean_local_fit_loss": clean_local_loss,
                "clean_direct_fit_loss": clean_direct_loss,
                "omega_memory_count": min(len(raw_omega_history), OMEGA_MEMORY_COUNT),
                "local_history_event_count": int(len(pnp_times)),
                "direct_history_event_count": int(len(pnp_direct_times)),
            }
        )

        local_models = {
            "hold": ("hold", anchor.observed_world_m[primary], math.nan, 0.0),
            "v1_isotropic_single_window": (
                "rigid",
                iso_coefficient,
                iso_omega,
                iso_loss,
            ),
            "los_raw_omega": ("rigid", raw_coefficient, raw_omega, raw_loss),
            "los_memory31": (
                "rigid",
                memory_coefficient,
                memory_omega,
                memory_loss,
            ),
            "los_oracle_omega": (
                "rigid",
                oracle_local_coefficient,
                anchor.yaw_rate_rad_s,
                oracle_local_loss,
            ),
        }
        if cv_coefficient is not None:
            local_models["same_slot_world_cv"] = (
                "cv",
                cv_coefficient,
                math.nan,
                math.nan,
            )
        clean_models = {
            "clean_local_oracle_omega": (
                "rigid",
                clean_local_coefficient,
                anchor.yaw_rate_rad_s,
                clean_local_loss,
            )
        }
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
            predictions: list[tuple[str, str, np.ndarray, float, float, float]] = []
            for method, (kind, model, omega, loss) in local_models.items():
                if kind == "hold":
                    prediction = np.asarray(model)
                elif kind == "cv":
                    prediction = np.asarray([1.0, effective_horizon_s]) @ np.asarray(model)
                else:
                    prediction = base.predict_rigid(
                        np.asarray(model), geometry, primary, float(omega), effective_horizon_s
                    )
                predictions.append(("pnp", method, prediction, float(omega), float(loss), LOCAL_HISTORY_S))
            for method, (kind, model, omega, loss) in clean_models.items():
                prediction = base.predict_rigid(
                    np.asarray(model), geometry, primary, float(omega), effective_horizon_s
                )
                predictions.append(("clean", method, prediction, float(omega), float(loss), LOCAL_HISTORY_S))
            for domain, method, omega, phase, coefficient, loss in (
                (
                    "pnp",
                    "direct_joint_omega_phase",
                    direct_omega,
                    direct_phase,
                    direct_coefficient,
                    direct_loss,
                ),
                (
                    "pnp",
                    "direct_oracle_omega",
                    anchor.yaw_rate_rad_s,
                    direct_oracle_phase,
                    direct_oracle_coefficient,
                    direct_oracle_loss,
                ),
                (
                    "clean",
                    "clean_direct_joint_omega_phase",
                    clean_direct_omega,
                    clean_direct_phase,
                    clean_direct_coefficient,
                    clean_direct_loss,
                ),
            ):
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
                predictions.append(
                    (domain, method, prediction, float(omega), float(loss), DIRECT_HISTORY_S)
                )
            for domain, method, prediction, omega, loss, history_s in predictions:
                row = {
                    "session_id": session_id,
                    "timestamp_ns": anchor.timestamp_ns,
                    "segment_id": anchor.segment_id,
                    "future_segment_id": future.segment_id,
                    "input_domain": domain,
                    "future_regime": future_regime,
                    "method": method,
                    "history_window_s": history_s,
                    "horizon_s": horizon_s,
                    "effective_horizon_s": effective_horizon_s,
                    "primary_slot": primary,
                    "visible_count_at_anchor": int(current_slots.size),
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
                    base.error_components(prediction, truth, anchor.tracker_to_world)
                )
                prediction_rows.append(row)

    summary_rows = summarize(prediction_rows)
    write_csv_gz(output / "prediction_distribution.csv.gz", prediction_rows)
    write_csv_gz(output / "fit_distribution.csv.gz", fit_rows)
    write_csv(output / "prediction_summary.csv", summary_rows)
    write_json(output / "condition.json", condition)
    source_paths = set(sources) | {
        Path(__file__).resolve(),
        base_path,
        los_path,
        direct_path,
        joint_path,
        contract_path,
    }
    artifacts = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append(
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    git_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    write_json(
        output / "manifest.json",
        {
            "schema_version": "combined-motion-final-acceptance-v1",
            "session_id": session_id,
            "formal_sealed_test": args.session_suffix == TEST_SUFFIX,
            "test_split_accessed": args.session_suffix == TEST_SUFFIX,
            "post_test_tuning_allowed": False,
            "selection_contract": str(contract_path),
            "selection_contract_sha256": sha256(contract_path),
            "frozen_candidate_revision": git_revision,
            "common_anchor_count": common_anchor_count,
            "prediction_row_count": len(prediction_rows),
            "fit_row_count": len(fit_rows),
            "evaluation_contract": {
                "anchor_stride_frames": ANCHOR_STRIDE,
                "local_history_s": LOCAL_HISTORY_S,
                "direct_history_s": DIRECT_HISTORY_S,
                "horizons_s": list(HORIZONS_S),
                "same_physical_plate_scoring": True,
                "future_truth_estimator_input": False,
                "truth_slot_identity": "offline non-deployable analysis handle",
                "constant_twist_and_cross_reversal_separated": True,
            },
            "sources": [
                {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in sorted(source_paths, key=str)
            ],
            "artifacts": artifacts,
        },
    )


if __name__ == "__main__":
    main()
