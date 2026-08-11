#!/usr/bin/env python3
"""Large-scale frozen validation of the combined-motion predictor.

This evaluator is intentionally not a tuner.  It consumes every formal
``linear_and_spin`` session from the frozen Stage3 manifest, resolves the
authoritative successful run through ``session_result.json``, and preserves
row-level predictions, fit results, data-quality audits, and coverage losses.
The exact protocol is frozen in
``combined_motion_large_scale_validation_contract.json``.
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
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np


MODE = "linear_and_spin"
HISTORY_S = 4.0
CV_HISTORY_S = 0.4
REGULAR_INTERVAL_S = 2.0
HORIZONS_S = (0.05, 0.10, 0.20)
REVERSAL_LEADS_S = (0.05, 0.10, 0.20)
MAX_OMEGA = 16.0
OMEGA_GRID_STEP = 1.0
PHASE_GRID_COUNT = 25
GATE_M = 0.055
_MODULES: tuple[Any, Any, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )
    parser.add_argument(
        "--session-limit",
        type=int,
        default=None,
        help="Smoke-test only. A limited run is never marked formal/complete.",
    )
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def modules(repo: Path) -> tuple[Any, Any, Any]:
    global _MODULES
    if _MODULES is None:
        suffix = str(os.getpid())
        base = load_module(
            f"combined_large_base_{suffix}",
            repo / "scripts" / "evaluate-combined-motion-factorization.py",
        )
        direct = load_module(
            f"combined_large_direct_{suffix}",
            repo / "scripts" / "evaluate-combined-motion-direct-reflection.py",
        )
        joint = load_module(
            f"combined_large_joint_{suffix}",
            repo / "scripts" / "evaluate-combined-motion-direct-joint.py",
        )
        _MODULES = base, direct, joint
    return _MODULES


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty evidence table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty evidence table: {path}")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "p50": math.nan, "p90": math.nan, "p95": math.nan,
                "p99": math.nan, "max": math.nan}
    return {
        "n": int(array.size),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def select_run_paths(runtime_root: Path, condition: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    session_id = condition["session_id"]
    manifest_path = runtime_root / f".manifest-{session_id}.json"
    result_path = runtime_root / session_id / "session_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    observation_path = Path(result["observations"])
    truth_path = Path(result["truth"])
    if not observation_path.is_file() or not truth_path.is_file():
        raise FileNotFoundError(
            f"authoritative run is missing for {session_id}: "
            f"{observation_path}, {truth_path}"
        )
    return truth_path, observation_path, manifest_path, result_path


def load_formal_session(
    repo: Path,
    workspace: Path,
    condition: dict[str, Any],
) -> tuple[list[Any], dict[str, Any], list[dict[str, Any]]]:
    base, _, _ = modules(repo)
    runtime_root = workspace / "runtime" / "stage3-formal-20260720-v2"
    truth_path, observation_path, manifest_path, result_path = select_run_paths(
        runtime_root, condition
    )
    runtime_condition = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if runtime_condition != condition:
        raise ValueError(f"dataset/runtime manifest mismatch: {condition['session_id']}")
    start_ns = base.scene_motion_start_ns(result_path)

    truth_total = 0
    truth_pre_start = 0
    truth_missing_selected = 0
    truth_duplicate_timestamp = 0
    truth_seen: set[int] = set()
    frames: list[Any] = []
    with truth_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            truth_total += 1
            record = json.loads(line)
            timestamp_ns = int(record["timestamp_ns"])
            if timestamp_ns < start_ns:
                truth_pre_start += 1
                continue
            if timestamp_ns in truth_seen:
                truth_duplicate_timestamp += 1
                continue
            truth_seen.add(timestamp_ns)
            target = base.selected_target(record)
            if target is None:
                truth_missing_selected += 1
                continue
            local = np.asarray(
                [armor["chassis_local_position_m"] for armor in target["armors"]],
                dtype=np.float64,
            )
            if local.shape != (4, 3):
                raise ValueError(
                    f"expected four armor slots in {condition['session_id']}, got {local.shape}"
                )
            center = np.asarray(target["world_position_m"], dtype=np.float64)
            target_rotation = base.quaternion_matrix(target["world_quaternion_wxyz"])
            armor_world = center[None, :] + local @ target_rotation.T
            exposure = record["exposure_state"]
            tracker_origin = np.asarray(
                exposure["gimbal_position_world_m"], dtype=np.float64
            )
            tracker_rotation = base.quaternion_matrix(
                exposure["chassis_quaternion_world_wxyz"]
            )
            frame = base.Frame(
                    timestamp_ns=timestamp_ns,
                    center_world_m=center,
                    velocity_world_mps=np.asarray(
                        target["world_velocity_mps"], dtype=np.float64
                    ),
                    yaw_rate_rad_s=float(target["world_vyaw_rad_s"]),
                    armor_world_m=armor_world,
                    armor_local_m=local,
                    tracker_origin_world_m=tracker_origin,
                    tracker_to_world=tracker_rotation,
                    observed_world_m=np.full((4, 3), np.nan, dtype=np.float64),
                    observed_mask=np.zeros(4, dtype=np.bool_),
                    observed_range_m=np.full(4, np.inf, dtype=np.float64),
                    association_ambiguous=False,
                )
            frame.camera_profile_id = "missing"
            frames.append(frame)
    frames.sort(key=lambda frame: frame.timestamp_ns)
    if not frames:
        raise ValueError(f"no post-start truth frames: {condition['session_id']}")
    frame_by_timestamp = {frame.timestamp_ns: frame for frame in frames}

    observation_total = 0
    observation_pre_start = 0
    observation_unmatched_timestamp = 0
    observation_duplicate_timestamp = 0
    observation_transform_unmatched = 0
    observation_ambiguous = 0
    observation_over_four_valid = 0
    observation_seen: set[int] = set()
    candidate_histogram: Counter[int] = Counter()
    camera_profiles: Counter[str] = Counter()
    with observation_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            observation_total += 1
            record = json.loads(line)
            timestamp_ns = int(record["timestamp_ns"])
            if timestamp_ns < start_ns:
                observation_pre_start += 1
                continue
            if timestamp_ns in observation_seen:
                observation_duplicate_timestamp += 1
                continue
            observation_seen.add(timestamp_ns)
            frame = frame_by_timestamp.get(timestamp_ns)
            if frame is None:
                observation_unmatched_timestamp += 1
                continue
            if not bool(record.get("tracker_world_transform_exposure_matched", False)):
                observation_transform_unmatched += 1
            camera_profile = str(record.get("camera_profile_id", "missing"))
            camera_profiles[camera_profile] += 1
            frame.camera_profile_id = camera_profile
            local_rows = []
            for armor in record.get("armors", []):
                if not bool(armor.get("valid", False)):
                    continue
                value = np.asarray(armor.get("position_m", []), dtype=np.float64)
                if value.shape == (3,) and np.isfinite(value).all():
                    local_rows.append(value)
            candidate_histogram[len(local_rows)] += 1
            if not local_rows:
                continue
            if len(local_rows) > 4:
                observation_over_four_valid += 1
                continue
            local_array = np.stack(local_rows)
            world = (
                frame.tracker_origin_world_m[None, :]
                + local_array @ frame.tracker_to_world.T
            )
            slots, ambiguous = base.best_assignment(world, frame.armor_world_m)
            frame.association_ambiguous = bool(ambiguous)
            if ambiguous:
                observation_ambiguous += 1
                continue
            for row, slot_raw in enumerate(slots):
                slot = int(slot_raw)
                frame.observed_world_m[slot] = world[row]
                frame.observed_mask[slot] = True
                frame.observed_range_m[slot] = float(
                    np.linalg.norm(local_array[row, :2])
                )

    segment = 0
    frames[0].segment_id = segment
    reversal_count = 0
    for previous, current in zip(frames, frames[1:]):
        if (
            np.linalg.norm(current.velocity_world_mps - previous.velocity_world_mps)
            > 0.05
            or abs(current.yaw_rate_rad_s - previous.yaw_rate_rad_s) > 0.05
        ):
            segment += 1
            reversal_count += 1
        current.segment_id = segment

    geometry = frames[0].armor_local_m
    geometry_drift_m = max(
        float(np.max(np.linalg.norm(frame.armor_local_m - geometry, axis=1)))
        for frame in frames
    )
    observed_frames = sum(bool(frame.observed_mask.any()) for frame in frames)
    observed_event_count = sum(int(frame.observed_mask.sum()) for frame in frames)
    raw_sources = []
    for kind, path in (
        ("truth", truth_path),
        ("observations", observation_path),
        ("runtime_manifest", manifest_path),
        ("session_result", result_path),
    ):
        raw_sources.append(
            {
                "session_id": condition["session_id"],
                "kind": kind,
                "path": str(path),
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
        )
    audit = {
        **condition,
        "status": "loaded",
        "selected_run": truth_path.parent.name,
        "motion_start_ns": start_ns,
        "truth_records_total": truth_total,
        "truth_records_pre_start": truth_pre_start,
        "truth_frames_post_start": len(frames),
        "truth_missing_selected": truth_missing_selected,
        "truth_duplicate_timestamp": truth_duplicate_timestamp,
        "observation_records_total": observation_total,
        "observation_records_pre_start": observation_pre_start,
        "observation_records_post_start": observation_total - observation_pre_start,
        "observation_unmatched_timestamp": observation_unmatched_timestamp,
        "observation_duplicate_timestamp": observation_duplicate_timestamp,
        "observation_transform_unmatched": observation_transform_unmatched,
        "observation_ambiguous_assignment": observation_ambiguous,
        "observation_over_four_valid": observation_over_four_valid,
        "candidate_count_histogram_json": json.dumps(
            dict(sorted(candidate_histogram.items())), separators=(",", ":")
        ),
        "camera_profile_histogram_json": json.dumps(
            dict(sorted(camera_profiles.items())), separators=(",", ":")
        ),
        "truth_frames_with_observation": observed_frames,
        "truth_frames_without_observation": len(frames) - observed_frames,
        "associated_observation_event_count": observed_event_count,
        "observation_frame_coverage": observed_frames / len(frames),
        "truth_duration_s": (frames[-1].timestamp_ns - frames[0].timestamp_ns) / 1e9,
        "truth_reversal_count": reversal_count,
        "armor_geometry_max_drift_m": geometry_drift_m,
    }
    return frames, audit, raw_sources


def nearest_index(timestamps: np.ndarray, target_ns: int) -> int:
    index = int(np.searchsorted(timestamps, target_ns))
    candidates = [min(index, len(timestamps) - 1)]
    if index > 0:
        candidates.append(index - 1)
    return min(candidates, key=lambda item: abs(int(timestamps[item]) - target_ns))


def plan_anchors(frames: list[Any]) -> dict[int, set[str]]:
    timestamps = np.asarray([frame.timestamp_ns for frame in frames], dtype=np.int64)
    roles: dict[int, set[str]] = defaultdict(set)
    first_target = frames[0].timestamp_ns + int(round(HISTORY_S * 1e9))
    last_target = frames[-1].timestamp_ns - int(round(max(HORIZONS_S) * 1e9))
    target = first_target
    while target <= last_target:
        index = nearest_index(timestamps, target)
        if abs(int(timestamps[index]) - target) <= 50_000_000:
            roles[index].add("regular_grid")
        target += int(round(REGULAR_INTERVAL_S * 1e9))

    reversal_indices = [
        index
        for index in range(1, len(frames))
        if frames[index].segment_id != frames[index - 1].segment_id
    ]
    for reversal_index in reversal_indices:
        reversal_ns = frames[reversal_index].timestamp_ns
        for lead_s in REVERSAL_LEADS_S:
            target_ns = reversal_ns - int(round(lead_s * 1e9))
            before = int(np.searchsorted(timestamps, target_ns, side="right")) - 1
            if before < 0:
                continue
            if abs(int(timestamps[before]) - target_ns) > 50_000_000:
                continue
            if timestamps[before] - timestamps[0] < int(round(HISTORY_S * 1e9)):
                continue
            roles[before].add(f"reversal_stress_{int(round(lead_s * 1000)):03d}ms")
    return roles


def cv_history(
    frames: list[Any], anchor_index: int, slot: int, domain: str
) -> tuple[np.ndarray, np.ndarray] | None:
    anchor = frames[anchor_index]
    start_ns = anchor.timestamp_ns - int(round(CV_HISTORY_S * 1e9))
    rows = []
    for index in range(anchor_index, -1, -1):
        frame = frames[index]
        if frame.timestamp_ns < start_ns:
            break
        if not frame.observed_mask[slot]:
            continue
        position = (
            frame.observed_world_m[slot]
            if domain == "pnp"
            else frame.armor_world_m[slot]
        )
        rows.append(((frame.timestamp_ns - anchor.timestamp_ns) / 1e9, position))
    if len(rows) < 4:
        return None
    rows.reverse()
    times = np.asarray([row[0] for row in rows], dtype=np.float64)
    if float(np.ptp(times)) < 0.05:
        return None
    return times, np.stack([row[1] for row in rows])


def speed_bin(value: float) -> str:
    if value < 1.0:
        return "[0,1)"
    if value < 2.0:
        return "[1,2)"
    return "[2,3.1]"


def omega_bin(value: float) -> str:
    value = abs(value)
    if value < 5.0:
        return "[0,5)"
    if value < 10.0:
        return "[5,10)"
    return "[10,15.1]"


def evaluate_session(task: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    repo_raw, workspace_raw, condition = task
    repo = Path(repo_raw)
    workspace = Path(workspace_raw)
    session_id = condition["session_id"]
    try:
        base, direct, joint = modules(repo)
        frames, session_audit, raw_sources = load_formal_session(
            repo, workspace, condition
        )
        geometry = frames[0].armor_local_m
        direction_rad = math.radians(float(condition["direction_deg"]))
        axis = np.asarray(
            [math.cos(direction_rad), math.sin(direction_rad), 0.0],
            dtype=np.float64,
        )
        speed_mps = float(condition["linear_speed_mps"])
        span_m = float(condition["linear_span_m"])
        anchors = plan_anchors(frames)
        prediction_rows: list[dict[str, Any]] = []
        fit_rows: list[dict[str, Any]] = []
        anchor_rows: list[dict[str, Any]] = []

        for anchor_index, role_labels in sorted(anchors.items()):
            anchor = frames[anchor_index]
            strata = ["regular_grid"] if "regular_grid" in role_labels else []
            stress_labels = sorted(
                label for label in role_labels if label.startswith("reversal_stress_")
            )
            if stress_labels:
                strata.append("reversal_stress")
            base_anchor = {
                "session_id": session_id,
                "timestamp_ns": anchor.timestamp_ns,
                "anchor_index": anchor_index,
                "role_labels": ";".join(sorted(role_labels)),
                "distance_bin": int(condition["distance_bin"]),
                "distance_m": float(condition["distance_m"]),
                "linear_speed_mps": speed_mps,
                "linear_speed_bin": speed_bin(speed_mps),
                "truth_omega_rad_s": float(anchor.yaw_rate_rad_s),
                "absolute_yaw_rate_bin": omega_bin(anchor.yaw_rate_rad_s),
                "direction_sector": int(condition["direction_sector"]),
                "camera_profile_id": str(anchor.camera_profile_id),
                "truth_segment_id": int(anchor.segment_id),
            }
            current_slots = np.flatnonzero(anchor.observed_mask)
            if current_slots.size == 0:
                for stratum in strata:
                    anchor_rows.append(
                        {
                            **base_anchor,
                            "sample_stratum": stratum,
                            "status": "no_current_observation",
                            "visible_count_at_anchor": 0,
                            "primary_slot": -1,
                            "direct_history_event_count": 0,
                            "direct_history_span_s": 0.0,
                            "cv_pnp_available": 0,
                            "cv_clean_available": 0,
                        }
                    )
                continue
            primary = int(
                current_slots[np.argmin(anchor.observed_range_m[current_slots])]
            )
            pnp_history = direct.direct_history(
                base, frames, anchor_index, HISTORY_S, "pnp"
            )
            clean_history = direct.direct_history(
                base, frames, anchor_index, HISTORY_S, "clean"
            )
            if pnp_history is None or clean_history is None:
                for stratum in strata:
                    anchor_rows.append(
                        {
                            **base_anchor,
                            "sample_stratum": stratum,
                            "status": "insufficient_direct_history",
                            "visible_count_at_anchor": int(current_slots.size),
                            "primary_slot": primary,
                            "direct_history_event_count": 0,
                            "direct_history_span_s": 0.0,
                            "cv_pnp_available": 0,
                            "cv_clean_available": 0,
                        }
                    )
                continue
            pnp_times, pnp_slots, pnp_positions = pnp_history
            clean_times, clean_slots, clean_positions = clean_history
            try:
                pnp_fit = joint.joint_fit(
                    direct,
                    base,
                    pnp_times,
                    pnp_slots,
                    pnp_positions,
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
                clean_fit = joint.joint_fit(
                    direct,
                    base,
                    clean_times,
                    clean_slots,
                    clean_positions,
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
                pnp_oracle = direct.estimate_phase(
                    base,
                    pnp_times,
                    pnp_slots,
                    pnp_positions,
                    geometry,
                    anchor.tracker_to_world,
                    omega=anchor.yaw_rate_rad_s,
                    speed_mps=speed_mps,
                    span_m=span_m,
                    axis=axis,
                    phase_grid_count=PHASE_GRID_COUNT,
                    robust=True,
                )
                clean_oracle = direct.estimate_phase(
                    base,
                    clean_times,
                    clean_slots,
                    clean_positions,
                    geometry,
                    anchor.tracker_to_world,
                    omega=anchor.yaw_rate_rad_s,
                    speed_mps=speed_mps,
                    span_m=span_m,
                    axis=axis,
                    phase_grid_count=PHASE_GRID_COUNT,
                    robust=False,
                )
            except Exception as error:
                for stratum in strata:
                    anchor_rows.append(
                        {
                            **base_anchor,
                            "sample_stratum": stratum,
                            "status": f"fit_failure:{type(error).__name__}",
                            "visible_count_at_anchor": int(current_slots.size),
                            "primary_slot": primary,
                            "direct_history_event_count": int(len(pnp_times)),
                            "direct_history_span_s": float(np.ptp(pnp_times)),
                            "cv_pnp_available": 0,
                            "cv_clean_available": 0,
                        }
                    )
                continue

            cv_models: dict[str, np.ndarray | None] = {}
            for domain in ("pnp", "clean"):
                history = cv_history(frames, anchor_index, primary, domain)
                cv_models[domain] = (
                    None
                    if history is None
                    else base.fit_cv(history[0], history[1])
                )
            for stratum in strata:
                anchor_rows.append(
                    {
                        **base_anchor,
                        "sample_stratum": stratum,
                        "status": "evaluated_common_anchor",
                        "visible_count_at_anchor": int(current_slots.size),
                        "primary_slot": primary,
                        "direct_history_event_count": int(len(pnp_times)),
                        "direct_history_span_s": float(np.ptp(pnp_times)),
                        "cv_pnp_available": int(cv_models["pnp"] is not None),
                        "cv_clean_available": int(cv_models["clean"] is not None),
                    }
                )

            fits = {
                ("pnp", "direct_joint"): pnp_fit,
                ("clean", "direct_joint"): clean_fit,
                (
                    "pnp",
                    "direct_oracle_omega",
                ): (
                    float(anchor.yaw_rate_rad_s),
                    float(pnp_oracle[0]),
                    pnp_oracle[1],
                    float(pnp_oracle[2]),
                ),
                (
                    "clean",
                    "direct_oracle_omega",
                ): (
                    float(anchor.yaw_rate_rad_s),
                    float(clean_oracle[0]),
                    clean_oracle[1],
                    float(clean_oracle[2]),
                ),
            }
            for (domain, method), (omega, phase, coefficient, loss) in fits.items():
                fit_rows.append(
                    {
                        **base_anchor,
                        "input_domain": domain,
                        "method": method,
                        "history_window_s": HISTORY_S,
                        "history_event_count": int(len(pnp_times)),
                        "history_time_span_s": float(np.ptp(pnp_times)),
                        "model_omega_rad_s": float(omega),
                        "absolute_omega_error_rad_s": abs(
                            float(omega) - float(anchor.yaw_rate_rad_s)
                        ),
                        "omega_sign_correct": int(
                            np.sign(float(omega))
                            == np.sign(float(anchor.yaw_rate_rad_s))
                        ),
                        "phase_m": float(phase),
                        "fit_loss": float(loss),
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
                predictions: list[tuple[str, str, np.ndarray, float, float]] = [
                    (
                        "pnp",
                        "hold",
                        anchor.observed_world_m[primary],
                        math.nan,
                        math.nan,
                    ),
                    (
                        "clean",
                        "hold",
                        anchor.armor_world_m[primary],
                        math.nan,
                        math.nan,
                    ),
                ]
                for domain in ("pnp", "clean"):
                    cv_model = cv_models[domain]
                    if cv_model is not None:
                        predictions.append(
                            (
                                domain,
                                "same_slot_cv_400ms",
                                np.asarray([1.0, effective_horizon_s]) @ cv_model,
                                math.nan,
                                math.nan,
                            )
                        )
                    for method in ("direct_joint", "direct_oracle_omega"):
                        omega, phase, coefficient, loss = fits[(domain, method)]
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
                            (
                                domain,
                                method,
                                prediction,
                                float(omega),
                                float(loss),
                            )
                        )
                for stratum in strata:
                    for domain, method, prediction, model_omega, loss in predictions:
                        row = {
                            **base_anchor,
                            "sample_stratum": stratum,
                            "future_regime": future_regime,
                            "future_segment_id": int(future.segment_id),
                            "input_domain": domain,
                            "method": method,
                            "history_window_s": (
                                0.0
                                if method == "hold"
                                else CV_HISTORY_S
                                if method == "same_slot_cv_400ms"
                                else HISTORY_S
                            ),
                            "horizon_s": horizon_s,
                            "effective_horizon_s": effective_horizon_s,
                            "primary_slot": primary,
                            "visible_count_at_anchor": int(current_slots.size),
                            "model_omega_rad_s": model_omega,
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
                        row["cross_depth_within_55mm"] = int(
                            row["error_cross_depth_m"] <= GATE_M
                        )
                        prediction_rows.append(row)

        session_audit["planned_unique_anchor_count"] = len(anchors)
        session_audit["planned_regular_anchor_count"] = sum(
            "regular_grid" in labels for labels in anchors.values()
        )
        session_audit["planned_reversal_stress_anchor_count"] = sum(
            any(label.startswith("reversal_stress_") for label in labels)
            for labels in anchors.values()
        )
        session_audit["evaluated_regular_anchor_count"] = sum(
            row["sample_stratum"] == "regular_grid"
            and row["status"] == "evaluated_common_anchor"
            for row in anchor_rows
        )
        session_audit["evaluated_reversal_stress_anchor_count"] = sum(
            row["sample_stratum"] == "reversal_stress"
            and row["status"] == "evaluated_common_anchor"
            for row in anchor_rows
        )
        return {
            "session_id": session_id,
            "status": "ok",
            "predictions": prediction_rows,
            "fits": fit_rows,
            "anchors": anchor_rows,
            "session_audit": session_audit,
            "raw_sources": raw_sources,
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
            "anchors": [],
            "session_audit": {
                **condition,
                "status": "load_or_evaluation_error",
                "error_type": type(error).__name__,
                "error": str(error),
            },
            "raw_sources": [],
        }


def evaluate_and_store_session(
    task: tuple[str, str, dict[str, Any], str]
) -> dict[str, Any]:
    """Persist one completed session before returning to the parent process.

    Formal evaluation is intentionally expensive.  Per-session atomic shards
    make the execution resumable and prevent a late summary error from losing
    already completed model fits.
    """
    repo_raw, workspace_raw, condition, part_dir_raw = task
    part_dir = Path(part_dir_raw)
    part_path = part_dir / f"{condition['session_id']}.json.gz"
    if part_path.is_file():
        with gzip.open(part_path, "rt", encoding="utf-8") as handle:
            cached = json.load(handle)
        return {
            "session_id": cached["session_id"],
            "status": cached["status"],
            "prediction_count": len(cached["predictions"]),
            "part_path": str(part_path),
            "cached": True,
        }
    result = evaluate_session((repo_raw, workspace_raw, condition))
    temporary = part_path.with_suffix(part_path.suffix + f".{os.getpid()}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, separators=(",", ":"))
    os.replace(temporary, part_path)
    return {
        "session_id": result["session_id"],
        "status": result["status"],
        "prediction_count": len(result["predictions"]),
        "part_path": str(part_path),
        "cached": False,
    }


def grouped_prediction_summary(
    rows: list[dict[str, Any]], keys: list[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        item = dict(zip(keys, key))
        for metric in ("error_cross_depth_m", "error_depth_m", "error_3d_m"):
            for name, value in quantiles(row[metric] for row in group).items():
                item[f"{metric}_{name}"] = value
        item["cross_depth_within_55mm_rate"] = float(
            np.mean([row["cross_depth_within_55mm"] for row in group])
        )
        item["session_count"] = len({row["session_id"] for row in group})
        output.append(item)
    return output


def make_condition_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    common = [
        "sample_stratum",
        "future_regime",
        "input_domain",
        "method",
        "horizon_s",
    ]
    for dimension in (
        "distance_bin",
        "linear_speed_bin",
        "absolute_yaw_rate_bin",
        "direction_sector",
        "camera_profile_id",
    ):
        grouped = grouped_prediction_summary(rows, [dimension, *common])
        for row in grouped:
            row["condition_dimension"] = dimension
            row["condition_value"] = row.pop(dimension)
            output.append(
                {
                    "condition_dimension": row.pop("condition_dimension"),
                    "condition_value": row.pop("condition_value"),
                    **row,
                }
            )
    return output


def make_session_summaries(
    rows: list[dict[str, Any]], planned_sessions: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keys = [
        "session_id",
        "sample_stratum",
        "future_regime",
        "input_domain",
        "method",
        "horizon_s",
    ]
    session_rows = grouped_prediction_summary(rows, keys)
    macro_keys = keys[1:]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in session_rows:
        groups[tuple(row[key] for key in macro_keys)].append(row)
    macro_rows = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        item = dict(zip(macro_keys, key))
        p95s = [row["error_cross_depth_m_p95"] for row in group]
        for name, value in quantiles(p95s).items():
            item[f"session_cross_depth_p95_{name}"] = value
        item["sessions_with_rows"] = len(group)
        item["planned_session_count"] = len(planned_sessions)
        item["session_coverage_rate"] = len(group) / len(planned_sessions)
        item["sessions_p95_within_55mm"] = sum(value <= GATE_M for value in p95s)
        item["sessions_p95_within_55mm_rate_among_covered"] = float(
            np.mean(np.asarray(p95s) <= GATE_M)
        )
        item["sessions_p95_within_55mm_rate_all_planned"] = sum(
            value <= GATE_M for value in p95s
        ) / len(planned_sessions)
        macro_rows.append(item)
    return session_rows, macro_rows


def make_coverage_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["sample_stratum", "status"]
    counts: Counter[tuple[str, str]] = Counter(
        (row["sample_stratum"], row["status"]) for row in rows
    )
    totals = Counter(row["sample_stratum"] for row in rows)
    return [
        {
            "sample_stratum": stratum,
            "status": status,
            "anchor_count": count,
            "stratum_planned_anchor_count": totals[stratum],
            "rate": count / totals[stratum],
        }
        for (stratum, status), count in sorted(counts.items())
    ]


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    workspace = args.workspace.resolve() if args.workspace else repo.parents[1]
    output = args.output.resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite completed evidence {output}")
    output.mkdir(parents=True, exist_ok=True)
    part_dir = output / "session_parts"
    part_dir.mkdir(exist_ok=True)

    dataset_manifest = (
        workspace
        / "dataset"
        / "autoaim-stage3-v1"
        / "stage3-20260719-v1"
        / "session_manifest.jsonl"
    )
    conditions = [
        json.loads(line)
        for line in dataset_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    conditions = [condition for condition in conditions if condition["mode"] == MODE]
    conditions.sort(key=lambda condition: condition["session_id"])
    full_session_count = len(conditions)
    if args.session_limit is not None:
        if args.session_limit <= 0:
            raise ValueError("--session-limit must be positive")
        conditions = conditions[: args.session_limit]

    prediction_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    raw_sources: list[dict[str, Any]] = []
    failures = []
    tasks = [
        (str(repo), str(workspace), condition, str(part_dir))
        for condition in conditions
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(evaluate_and_store_session, task): task[2]["session_id"]
            for task in tasks
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed += 1
            print(
                f"[{completed:03d}/{len(tasks):03d}] {result['session_id']} "
                f"{result['status']} predictions={result['prediction_count']} "
                f"cached={result['cached']}",
                flush=True,
            )

    for condition in conditions:
        part_path = part_dir / f"{condition['session_id']}.json.gz"
        if not part_path.is_file():
            raise RuntimeError(f"missing completed session shard: {part_path}")
        with gzip.open(part_path, "rt", encoding="utf-8") as handle:
            result = json.load(handle)
        prediction_rows.extend(result["predictions"])
        fit_rows.extend(result["fits"])
        anchor_rows.extend(result["anchors"])
        session_rows.append(result["session_audit"])
        raw_sources.extend(result["raw_sources"])
        if result["status"] != "ok":
            failures.append(
                {
                    "session_id": result["session_id"],
                    "error_type": result["error_type"],
                    "error": result["error"],
                    "traceback": result["traceback"],
                }
            )

    prediction_rows.sort(
        key=lambda row: (
            row["session_id"],
            row["timestamp_ns"],
            row["sample_stratum"],
            row["horizon_s"],
            row["input_domain"],
            row["method"],
        )
    )
    fit_rows.sort(
        key=lambda row: (
            row["session_id"], row["timestamp_ns"], row["input_domain"], row["method"]
        )
    )
    anchor_rows.sort(
        key=lambda row: (row["session_id"], row["timestamp_ns"], row["sample_stratum"])
    )
    session_rows.sort(key=lambda row: row["session_id"])
    raw_sources.sort(key=lambda row: (row["session_id"], row["kind"]))
    if not prediction_rows:
        raise RuntimeError("no predictions were produced")

    micro_summary = grouped_prediction_summary(
        prediction_rows,
        ["sample_stratum", "future_regime", "input_domain", "method", "horizon_s"],
    )
    condition_summary = make_condition_summary(prediction_rows)
    session_method_summary, macro_summary = make_session_summaries(
        prediction_rows, [condition["session_id"] for condition in conditions]
    )
    coverage_summary = make_coverage_summary(anchor_rows)

    write_csv_gz(output / "prediction_distribution.csv.gz", prediction_rows)
    write_csv_gz(output / "fit_distribution.csv.gz", fit_rows)
    write_csv(output / "anchor_coverage.csv", anchor_rows)
    write_csv(output / "session_data_audit.csv", session_rows)
    write_csv(output / "raw_source_inventory.csv", raw_sources)
    write_csv(output / "prediction_micro_summary.csv", micro_summary)
    write_csv(output / "prediction_condition_summary.csv", condition_summary)
    write_csv(output / "prediction_session_summary.csv", session_method_summary)
    write_csv(output / "prediction_session_macro_summary.csv", macro_summary)
    write_csv(output / "anchor_coverage_summary.csv", coverage_summary)
    write_json(output / "failures.json", failures)

    contract_path = (
        repo
        / "modules"
        / "autoaim"
        / "docs"
        / "combined_motion_large_scale_validation_contract.json"
    )
    script_paths = [
        Path(__file__).resolve(),
        repo / "scripts" / "evaluate-combined-motion-factorization.py",
        repo / "scripts" / "evaluate-combined-motion-direct-reflection.py",
        repo / "scripts" / "evaluate-combined-motion-direct-joint.py",
    ]
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
            "schema_version": "combined-motion-large-scale-validation-v1",
            "formal_complete_run": (
                args.session_limit is None
                and full_session_count == 144
                and len(conditions) == 144
                and not failures
            ),
            "model_or_threshold_tuned_from_this_run": False,
            "post_result_tuning_allowed": False,
            "dataset_manifest": str(dataset_manifest),
            "dataset_manifest_sha256": sha256(dataset_manifest),
            "validation_contract": str(contract_path),
            "validation_contract_sha256": sha256(contract_path),
            "frozen_candidate_revision": git_revision,
            "workers": args.workers,
            "session_limit": args.session_limit,
            "planned_session_count": len(conditions),
            "full_manifest_combined_session_count": full_session_count,
            "successful_session_count": sum(row["status"] == "loaded" for row in session_rows),
            "failed_session_count": len(failures),
            "prediction_row_count": len(prediction_rows),
            "fit_row_count": len(fit_rows),
            "anchor_coverage_row_count": len(anchor_rows),
            "complete_prediction_distribution_retained": True,
            "sampling_contract": {
                "regular_interval_s": REGULAR_INTERVAL_S,
                "reversal_leads_s": list(REVERSAL_LEADS_S),
                "horizons_s": list(HORIZONS_S),
                "regular_and_reversal_stress_reported_separately": True,
            },
            "causality": {
                "history_up_to_anchor_only": True,
                "future_truth_estimator_input": False,
                "future_truth_sampling_and_scoring_only": True,
                "truth_slot_identity": "offline non-deployable analysis handle",
            },
            "code_sources": [
                {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in script_paths
            ],
            "artifacts": artifacts,
        },
    )


if __name__ == "__main__":
    main()
