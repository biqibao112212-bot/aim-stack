#!/usr/bin/env python3
"""Evaluate a piecewise-rigid combined-motion model on sealed Stage3 evidence.

The experiment deliberately separates four questions:

1. Is a constant-translation plus constant-yaw-rate rigid model an exact
   explanation inside one motion-command segment?
2. Does fitting that model directly to armor trajectories outperform a
   same-physical-slot CV baseline without first reconstructing a center per
   frame?
3. What changes when exact, availability-matched armor positions are replaced
   by current PnP observations?
4. How inaccurate is instantaneous one/two-plate center recovery even when
   the true armor offset (radius and orientation) is supplied as an oracle?

The test split is intentionally never opened.  All PnP-to-slot assignments are
truth-only analysis handles; none are deployable inputs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


SESSION_PREFIX = "stage3-generalization-fixed6mm-20260729-v1-combined-"
DEV_SESSION_SUFFIXES = ("00", "01", "02", "03", "05")
SEALED_TEST_SUFFIX = "04"
HORIZONS_S = (0.05, 0.10, 0.20)
HISTORY_WINDOWS_S = (0.15, 0.25, 0.40)
METHODS = ("hold", "slot_cv", "rigid_oracle_omega", "rigid_estimated_omega")
DOMAINS = ("clean", "pnp")
COLORS = {
    "hold": "#7f7f7f",
    "slot_cv": "#0072B2",
    "rigid_oracle_omega": "#009E73",
    "rigid_estimated_omega": "#D55E00",
}


@dataclass
class Frame:
    timestamp_ns: int
    center_world_m: np.ndarray
    velocity_world_mps: np.ndarray
    yaw_rate_rad_s: float
    armor_world_m: np.ndarray
    armor_local_m: np.ndarray
    tracker_origin_world_m: np.ndarray
    tracker_to_world: np.ndarray
    observed_world_m: np.ndarray
    observed_mask: np.ndarray
    observed_range_m: np.ndarray
    association_ambiguous: bool
    segment_id: int = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-stride", type=int, default=6)
    parser.add_argument("--omega-grid-step", type=float, default=0.5)
    parser.add_argument("--max-omega", type=float, default=16.0)
    parser.add_argument("--huber-delta-m", type=float, default=0.05)
    return parser.parse_args()


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


def quaternion_matrix(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in q)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def selected_target(record: dict[str, Any]) -> dict[str, Any] | None:
    truth = record.get("ground_truth")
    if not truth:
        return None
    selected = record.get("selected_target_id")
    return next(
        (target for target in truth.get("targets", []) if target.get("target_id") == selected),
        None,
    )


def scene_motion_start_ns(session_result: Path) -> int:
    result = json.loads(session_result.read_text(encoding="utf-8-sig"))
    timestamps = [int(value) for value in re.findall(r'"timestamp_ns":(\d+)', result["scene_ack"])]
    if not timestamps:
        raise ValueError(f"scene acknowledgement has no timestamp: {session_result}")
    return max(timestamps)


def best_assignment(observed: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return observed-row -> truth-slot assignment and a strict ambiguity flag."""
    count = int(observed.shape[0])
    distances = np.linalg.norm(observed[:, None, :] - truth[None, :, :], axis=-1)
    scored: list[tuple[float, tuple[int, ...]]] = []
    for slots in itertools.permutations(range(4), count):
        cost = float(sum(distances[row, slot] for row, slot in enumerate(slots))) / count
        scored.append((cost, slots))
    scored.sort(key=lambda item: (item[0], item[1]))
    ambiguous = len(scored) > 1 and scored[1][0] - scored[0][0] <= 1e-6
    return np.asarray(scored[0][1], dtype=np.int64), bool(ambiguous)


def load_session(
    dataset_root: Path,
    runtime_root: Path,
    session_id: str,
) -> tuple[list[Frame], dict[str, Any], list[Path]]:
    session_dir = dataset_root / session_id
    runs = sorted(session_dir.glob("run-*"))
    if len(runs) != 1:
        raise ValueError(f"expected one immutable run for {session_id}, got {len(runs)}")
    run = runs[0]
    truth_path = run / "truth.jsonl"
    observation_path = run / "observations.jsonl"
    manifest_path = runtime_root / f".manifest-{session_id}.json"
    result_path = runtime_root / session_id / "session_result.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    start_ns = scene_motion_start_ns(result_path)

    frames: list[Frame] = []
    frame_by_timestamp: dict[int, Frame] = {}
    with truth_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            timestamp_ns = int(record["timestamp_ns"])
            if timestamp_ns < start_ns:
                continue
            target = selected_target(record)
            if target is None:
                continue
            local = np.asarray(
                [armor["chassis_local_position_m"] for armor in target["armors"]],
                dtype=np.float64,
            )
            center = np.asarray(target["world_position_m"], dtype=np.float64)
            target_rotation = quaternion_matrix(target["world_quaternion_wxyz"])
            armor_world = center[None, :] + local @ target_rotation.T
            exposure = record["exposure_state"]
            tracker_origin = np.asarray(exposure["gimbal_position_world_m"], dtype=np.float64)
            # Stage3 position_m is the calibrated tracker/chassis frame.  The
            # gimbal quaternion would rotate it twice; the dataset builder uses
            # this same chassis quaternion for exact labels.
            tracker_rotation = quaternion_matrix(exposure["chassis_quaternion_world_wxyz"])
            frame = Frame(
                timestamp_ns=timestamp_ns,
                center_world_m=center,
                velocity_world_mps=np.asarray(target["world_velocity_mps"], dtype=np.float64),
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
            frames.append(frame)
            frame_by_timestamp[timestamp_ns] = frame

    with observation_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            frame = frame_by_timestamp.get(int(record["timestamp_ns"]))
            if frame is None:
                continue
            local_rows = [
                np.asarray(armor["position_m"], dtype=np.float64)
                for armor in record.get("armors", [])
                if bool(armor.get("valid", False))
                and np.isfinite(np.asarray(armor["position_m"], dtype=np.float64)).all()
            ]
            if not local_rows or len(local_rows) > 4:
                continue
            local_array = np.stack(local_rows)
            world = frame.tracker_origin_world_m[None, :] + local_array @ frame.tracker_to_world.T
            slots, ambiguous = best_assignment(world, frame.armor_world_m)
            frame.association_ambiguous = ambiguous
            if ambiguous:
                continue
            for row, slot in enumerate(slots):
                frame.observed_world_m[int(slot)] = world[row]
                frame.observed_mask[int(slot)] = True
                frame.observed_range_m[int(slot)] = float(np.linalg.norm(local_array[row, :2]))

    # The commanded translation is bounded and reverses at the path ends.
    # Split exactly where truth velocity or yaw rate changes; a single model is
    # only claimed inside one such constant-twist interval.
    segment = 0
    frames[0].segment_id = segment
    for previous, current in zip(frames, frames[1:]):
        if (
            np.linalg.norm(current.velocity_world_mps - previous.velocity_world_mps) > 0.05
            or abs(current.yaw_rate_rad_s - previous.yaw_rate_rad_s) > 0.05
        ):
            segment += 1
        current.segment_id = segment
    return frames, manifest, [truth_path, observation_path, manifest_path, result_path]


def rigid_design(
    times_s: np.ndarray,
    slots: np.ndarray,
    geometry_m: np.ndarray,
    omega: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearized rigid model with shared phase-scale coefficients C and S.

    The 8-vector is [cxyz, vxyz, C, S].  C/S equal cos(theta)/sin(theta)
    for exact geometry, but leaving their norm free makes the fit tolerant to
    one shared radius-scale mismatch without introducing per-frame centers.
    """
    rows: list[np.ndarray] = []
    z_offset: list[float] = []
    for time_s, slot_raw in zip(times_s, slots):
        slot = int(slot_raw)
        qx, qy, qz = geometry_m[slot]
        cosine = math.cos(omega * float(time_s))
        sine = math.sin(omega * float(time_s))
        hx = (cosine * qx - sine * qy, -cosine * qy - sine * qx)
        hy = (sine * qx + cosine * qy, -sine * qy + cosine * qx)
        row = np.zeros(8, dtype=np.float64)
        row[0], row[3], row[6], row[7] = 1.0, time_s, hx[0], hx[1]
        rows.append(row)
        z_offset.append(0.0)
        row = np.zeros(8, dtype=np.float64)
        row[1], row[4], row[6], row[7] = 1.0, time_s, hy[0], hy[1]
        rows.append(row)
        z_offset.append(0.0)
        row = np.zeros(8, dtype=np.float64)
        row[2], row[5] = 1.0, time_s
        rows.append(row)
        z_offset.append(float(qz))
    return np.stack(rows), np.asarray(z_offset, dtype=np.float64)


def fit_rigid(
    times_s: np.ndarray,
    slots: np.ndarray,
    positions_m: np.ndarray,
    geometry_m: np.ndarray,
    omega: float,
    *,
    robust: bool,
    huber_delta_m: float,
) -> tuple[np.ndarray, float]:
    design, z_offset = rigid_design(times_s, slots, geometry_m, omega)
    target = positions_m.reshape(-1).copy()
    target[2::3] -= z_offset[2::3]
    event_weight = np.ones(len(times_s), dtype=np.float64)
    coefficient = np.zeros(8, dtype=np.float64)
    for _ in range(6 if robust else 1):
        coordinate_weight = np.repeat(event_weight, 3)
        root = np.sqrt(coordinate_weight)
        coefficient = np.linalg.lstsq(
            design * root[:, None], target * root, rcond=None
        )[0]
        residual = (design @ coefficient - target).reshape(-1, 3)
        norm = np.linalg.norm(residual, axis=1)
        event_weight = np.minimum(1.0, huber_delta_m / np.maximum(norm, 1e-12))
    residual = (design @ coefficient - target).reshape(-1, 3)
    norm = np.linalg.norm(residual, axis=1)
    if robust:
        delta = huber_delta_m
        loss = float(np.mean(np.where(norm <= delta, 0.5 * norm**2, delta * (norm - 0.5 * delta))))
    else:
        loss = float(np.mean(norm**2))
    return coefficient, loss


def predict_rigid(
    coefficient: np.ndarray,
    geometry_m: np.ndarray,
    slot: int,
    omega: float,
    horizon_s: float,
) -> np.ndarray:
    qx, qy, qz = geometry_m[int(slot)]
    cosine = math.cos(omega * horizon_s)
    sine = math.sin(omega * horizon_s)
    c_phase, s_phase = coefficient[6:8]
    return np.asarray(
        [
            coefficient[0]
            + coefficient[3] * horizon_s
            + (cosine * qx - sine * qy) * c_phase
            + (-cosine * qy - sine * qx) * s_phase,
            coefficient[1]
            + coefficient[4] * horizon_s
            + (sine * qx + cosine * qy) * c_phase
            + (-sine * qy + cosine * qx) * s_phase,
            coefficient[2] + coefficient[5] * horizon_s + qz,
        ],
        dtype=np.float64,
    )


def estimate_omega(
    times_s: np.ndarray,
    slots: np.ndarray,
    positions_m: np.ndarray,
    geometry_m: np.ndarray,
    *,
    max_omega: float,
    grid_step: float,
    robust: bool,
    huber_delta_m: float,
) -> tuple[float, np.ndarray, float]:
    grid = np.arange(-max_omega, max_omega + 0.5 * grid_step, grid_step)
    grid_loss = []
    for omega in grid:
        _, loss = fit_rigid(
            times_s,
            slots,
            positions_m,
            geometry_m,
            float(omega),
            robust=False,
            huber_delta_m=huber_delta_m,
        )
        grid_loss.append(loss)
    best = int(np.argmin(grid_loss))
    lower = max(-max_omega, float(grid[best] - grid_step))
    upper = min(max_omega, float(grid[best] + grid_step))

    def objective(omega: float) -> float:
        return fit_rigid(
            times_s,
            slots,
            positions_m,
            geometry_m,
            omega,
            robust=False,
            huber_delta_m=huber_delta_m,
        )[1]

    if upper - lower > 1e-9:
        refined = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
        omega = float(refined.x)
    else:
        omega = float(grid[best])
    coefficient, loss = fit_rigid(
        times_s,
        slots,
        positions_m,
        geometry_m,
        omega,
        robust=robust,
        huber_delta_m=huber_delta_m,
    )
    return omega, coefficient, loss


def fit_cv(times_s: np.ndarray, positions_m: np.ndarray) -> np.ndarray | None:
    if len(times_s) < 4 or float(np.ptp(times_s)) < 0.05:
        return None
    design = np.column_stack((np.ones(len(times_s)), times_s))
    return np.linalg.lstsq(design, positions_m, rcond=None)[0]


def nearest_future(frames: list[Frame], anchor_index: int, horizon_s: float) -> Frame | None:
    target = frames[anchor_index].timestamp_ns + int(round(horizon_s * 1e9))
    timestamps = np.fromiter((frame.timestamp_ns for frame in frames), dtype=np.int64)
    index = int(np.searchsorted(timestamps, target))
    candidates = [min(index, len(frames) - 1)]
    if index > 0:
        candidates.append(index - 1)
    best = min(candidates, key=lambda item: abs(int(timestamps[item]) - target))
    if abs(int(timestamps[best]) - target) > 6_000_000:
        return None
    return frames[best]


def error_components(prediction: np.ndarray, truth: np.ndarray, tracker_to_world: np.ndarray) -> dict[str, float]:
    world_error = prediction - truth
    tracker_error = tracker_to_world.T @ world_error
    return {
        "error_3d_m": float(np.linalg.norm(world_error)),
        "error_world_planar_m": float(np.linalg.norm(world_error[:2])),
        "error_depth_m": float(abs(tracker_error[0])),
        "error_cross_depth_m": float(np.linalg.norm(tracker_error[1:])),
        "error_tracker_x_m": float(tracker_error[0]),
        "error_tracker_y_m": float(tracker_error[1]),
        "error_tracker_z_m": float(tracker_error[2]),
    }


def collect_center_rows(session_id: str, frames: list[Frame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in frames:
        slots = np.flatnonzero(frame.observed_mask)
        if slots.size == 0:
            continue
        primary = int(slots[np.argmin(frame.observed_range_m[slots])])
        center_estimates = []
        for slot_raw in slots:
            slot = int(slot_raw)
            # Optimistic intervention: exact radius and exact instantaneous
            # orientation are supplied.  Remaining error is therefore a lower
            # bound on center-first recovery from current PnP position.
            offset = frame.armor_world_m[slot] - frame.center_world_m
            estimate = frame.observed_world_m[slot] - offset
            center_estimates.append(estimate)
            item = {
                "session_id": session_id,
                "timestamp_ns": frame.timestamp_ns,
                "estimator": "single_primary" if slot == primary else "single_secondary",
                "visible_count": int(slots.size),
                "slot": slot,
                "adjacent_pair": False,
            }
            item.update(error_components(estimate, frame.center_world_m, frame.tracker_to_world))
            rows.append(item)
        if slots.size >= 2:
            ordered = slots[np.argsort(frame.observed_range_m[slots])][:2]
            adjacent = int((int(ordered[1]) - int(ordered[0])) % 4) in (1, 3)
            estimate = np.mean(
                [
                    frame.observed_world_m[int(slot)]
                    - (frame.armor_world_m[int(slot)] - frame.center_world_m)
                    for slot in ordered
                ],
                axis=0,
            )
            item = {
                "session_id": session_id,
                "timestamp_ns": frame.timestamp_ns,
                "estimator": (
                    "two_plate_adjacent_oracle_offset_mean"
                    if adjacent
                    else "two_plate_nonadjacent_oracle_offset_mean"
                ),
                "visible_count": int(slots.size),
                "slot": -1,
                "adjacent_pair": bool(adjacent),
            }
            item.update(error_components(estimate, frame.center_world_m, frame.tracker_to_world))
            rows.append(item)
    return rows


def history_observations(
    frames: list[Frame],
    anchor_index: int,
    window_s: float,
    domain: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    anchor = frames[anchor_index]
    start_ns = anchor.timestamp_ns - int(round(window_s * 1e9))
    rows: list[tuple[float, int, np.ndarray]] = []
    first_timestamp = anchor.timestamp_ns
    for index in range(anchor_index, -1, -1):
        frame = frames[index]
        if frame.timestamp_ns < start_ns or frame.segment_id != anchor.segment_id:
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
            rows.append(((frame.timestamp_ns - anchor.timestamp_ns) / 1e9, slot, position))
    if not rows or (anchor.timestamp_ns - first_timestamp) / 1e9 < 0.90 * window_s:
        return None
    rows.reverse()
    return (
        np.asarray([row[0] for row in rows], dtype=np.float64),
        np.asarray([row[1] for row in rows], dtype=np.int64),
        np.stack([row[2] for row in rows]),
    )


def evaluate_predictions(
    session_id: str,
    frames: list[Frame],
    *,
    history_windows_s: Iterable[float],
    horizons_s: Iterable[float],
    anchor_stride: int,
    omega_grid_step: float,
    max_omega: float,
    huber_delta_m: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    prediction_rows: list[dict[str, Any]] = []
    omega_rows: list[dict[str, Any]] = []
    representative: dict[str, Any] | None = None
    geometry = frames[0].armor_local_m
    for anchor_index in range(0, len(frames), anchor_stride):
        anchor = frames[anchor_index]
        current_slots = np.flatnonzero(anchor.observed_mask)
        if current_slots.size == 0:
            continue
        primary = int(current_slots[np.argmin(anchor.observed_range_m[current_slots])])
        for window_s in history_windows_s:
            histories = {
                domain: history_observations(frames, anchor_index, window_s, domain)
                for domain in DOMAINS
            }
            if any(value is None for value in histories.values()):
                continue
            fitted: dict[tuple[str, str], tuple[float | None, np.ndarray | None]] = {}
            for domain in DOMAINS:
                times, slots, positions = histories[domain]  # type: ignore[misc]
                robust = domain == "pnp"
                coefficient, _ = fit_rigid(
                    times,
                    slots,
                    positions,
                    geometry,
                    anchor.yaw_rate_rad_s,
                    robust=robust,
                    huber_delta_m=huber_delta_m,
                )
                fitted[(domain, "rigid_oracle_omega")] = (
                    anchor.yaw_rate_rad_s,
                    coefficient,
                )
                estimated, coefficient, loss = estimate_omega(
                    times,
                    slots,
                    positions,
                    geometry,
                    max_omega=max_omega,
                    grid_step=omega_grid_step,
                    robust=robust,
                    huber_delta_m=huber_delta_m,
                )
                fitted[(domain, "rigid_estimated_omega")] = (estimated, coefficient)
                omega_rows.append(
                    {
                        "session_id": session_id,
                        "timestamp_ns": anchor.timestamp_ns,
                        "history_window_s": window_s,
                        "input_domain": domain,
                        "truth_omega_rad_s": anchor.yaw_rate_rad_s,
                        "estimated_omega_rad_s": estimated,
                        "absolute_error_rad_s": abs(estimated - anchor.yaw_rate_rad_s),
                        "sign_correct": int(math.copysign(1.0, estimated or 1.0) == math.copysign(1.0, anchor.yaw_rate_rad_s)),
                        "fit_loss": loss,
                        "history_event_count": int(len(times)),
                        "history_time_span_s": float(np.ptp(times)),
                    }
                )
                target = slots == primary
                fitted[(domain, "slot_cv")] = (
                    None,
                    fit_cv(times[target], positions[target]),
                )
                fitted[(domain, "hold")] = (
                    None,
                    anchor.armor_world_m[primary]
                    if domain == "clean"
                    else anchor.observed_world_m[primary],
                )

            for horizon_s in horizons_s:
                future = nearest_future(frames, anchor_index, horizon_s)
                if future is None:
                    continue
                effective_horizon_s = (
                    future.timestamp_ns - anchor.timestamp_ns
                ) / 1e9
                regime = "constant_twist" if future.segment_id == anchor.segment_id else "cross_reversal"
                truth = future.armor_world_m[primary]
                for domain in DOMAINS:
                    for method in METHODS:
                        omega, model = fitted[(domain, method)]
                        if model is None:
                            continue
                        if method == "hold":
                            prediction = np.asarray(model, dtype=np.float64)
                        elif method == "slot_cv":
                            prediction = np.asarray([1.0, effective_horizon_s]) @ np.asarray(model)
                        else:
                            prediction = predict_rigid(
                                np.asarray(model),
                                geometry,
                                primary,
                                float(omega),
                                effective_horizon_s,
                            )
                        item = {
                            "session_id": session_id,
                            "timestamp_ns": anchor.timestamp_ns,
                            "segment_id": anchor.segment_id,
                            "history_window_s": window_s,
                            "horizon_s": horizon_s,
                            "effective_horizon_s": effective_horizon_s,
                            "future_regime": regime,
                            "input_domain": domain,
                            "method": method,
                            "primary_slot": primary,
                            "visible_count_at_anchor": int(current_slots.size),
                            "truth_omega_rad_s": anchor.yaw_rate_rad_s,
                            "model_omega_rad_s": omega if omega is not None else math.nan,
                            "prediction_x_m": float(prediction[0]),
                            "prediction_y_m": float(prediction[1]),
                            "prediction_z_m": float(prediction[2]),
                            "truth_x_m": float(truth[0]),
                            "truth_y_m": float(truth[1]),
                            "truth_z_m": float(truth[2]),
                        }
                        item.update(error_components(prediction, truth, anchor.tracker_to_world))
                        prediction_rows.append(item)

                if (
                    representative is None
                    and session_id.endswith("00")
                    and abs(window_s - 0.25) < 1e-9
                    and abs(horizon_s - 0.20) < 1e-9
                    and regime == "constant_twist"
                ):
                    times, slots, clean_positions = histories["clean"]  # type: ignore[misc]
                    _, _, pnp_positions = histories["pnp"]  # type: ignore[misc]
                    representative = {
                        "anchor": anchor,
                        "future": future,
                        "primary": primary,
                        "times": times,
                        "slots": slots,
                        "clean_positions": clean_positions,
                        "pnp_positions": pnp_positions,
                        "models": fitted.copy(),
                        "effective_horizon_s": effective_horizon_s,
                    }
    return prediction_rows, omega_rows, representative


def quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "n": int(array.size),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def write_csv_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty evidence table: {path}")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plain_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty summary: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["input_domain"],
            row["history_window_s"],
            row["future_regime"],
            row["method"],
            row["horizon_s"],
        )
        grouped.setdefault(key, []).append(row)
    result = []
    metrics = (
        "error_3d_m",
        "error_world_planar_m",
        "error_depth_m",
        "error_cross_depth_m",
    )
    for key, group in sorted(grouped.items()):
        item = {
            "input_domain": key[0],
            "history_window_s": key[1],
            "future_regime": key[2],
            "method": key[3],
            "horizon_s": key[4],
            "n": len(group),
            "session_count": len({row["session_id"] for row in group}),
        }
        for metric in metrics:
            stats = quantiles(float(row[metric]) for row in group)
            for name in ("p50", "p95", "p99", "max"):
                item[f"{metric}_{name}"] = stats[name]
        result.append(item)
    return result


def summarize_prediction_conditions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["session_id"],
            row["input_domain"],
            row["history_window_s"],
            row["future_regime"],
            row["method"],
            row["horizon_s"],
        )
        grouped.setdefault(key, []).append(row)
    result = []
    for key, group in sorted(grouped.items()):
        item = {
            "session_id": key[0],
            "input_domain": key[1],
            "history_window_s": key[2],
            "future_regime": key[3],
            "method": key[4],
            "horizon_s": key[5],
            "n": len(group),
        }
        for metric in (
            "error_3d_m",
            "error_world_planar_m",
            "error_depth_m",
            "error_cross_depth_m",
        ):
            stats = quantiles(float(row[metric]) for row in group)
            for name in ("p50", "p95", "p99", "max"):
                item[f"{metric}_{name}"] = stats[name]
        result.append(item)
    return result


def summarize_centers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["estimator"] == "single_secondary":
            continue
        grouped.setdefault(str(row["estimator"]), []).append(row)
    result = []
    for estimator, group in sorted(grouped.items()):
        item = {
            "estimator": estimator,
            "n": len(group),
            "session_count": len({row["session_id"] for row in group}),
            "adjacent_pair_fraction": float(np.mean([row["adjacent_pair"] for row in group])),
        }
        for metric in ("error_3d_m", "error_world_planar_m", "error_depth_m", "error_cross_depth_m"):
            stats = quantiles(float(row[metric]) for row in group)
            for name in ("p50", "p95", "p99", "max"):
                item[f"{metric}_{name}"] = stats[name]
        result.append(item)
    return result


def summarize_omega(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["input_domain"], row["history_window_s"], row["session_id"])
        grouped.setdefault(key, []).append(row)
    result = []
    for key, group in sorted(grouped.items()):
        stats = quantiles(float(row["absolute_error_rad_s"]) for row in group)
        result.append(
            {
                "input_domain": key[0],
                "history_window_s": key[1],
                "session_id": key[2],
                "n": len(group),
                "truth_omega_rad_s": float(np.median([row["truth_omega_rad_s"] for row in group])),
                "sign_accuracy": float(np.mean([row["sign_correct"] for row in group])),
                "absolute_error_rad_s_p50": stats["p50"],
                "absolute_error_rad_s_p95": stats["p95"],
                "absolute_error_rad_s_p99": stats["p99"],
            }
        )
    return result


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_gallery(
    sessions: dict[str, list[Frame]], manifests: dict[str, dict[str, Any]], output: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes_flat = axes.ravel()
    slot_colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    for axis, (session_id, frames) in zip(axes_flat, sorted(sessions.items())):
        center = np.stack([frame.center_world_m for frame in frames])
        axis.plot(center[:, 0], center[:, 1], color="black", lw=1.8, label="truth center")
        for slot in range(4):
            truth = np.stack([frame.armor_world_m[slot] for frame in frames])
            axis.plot(truth[:, 0], truth[:, 1], color=slot_colors[slot], lw=0.8, alpha=0.8, label=f"truth slot {slot}")
            observed = np.stack(
                [frame.observed_world_m[slot] for frame in frames if frame.observed_mask[slot]],
                axis=0,
            )
            axis.scatter(observed[:, 0], observed[:, 1], s=2, color=slot_colors[slot], alpha=0.18)
        manifest = manifests[session_id]
        axis.set_title(
            f"{session_id[-2:]}  v={manifest['linear_speed_mps']:.2f} m/s, "
            f"ω={manifest['spin_rad_s']:+.2f} rad/s"
        )
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)
        axis.set_xlabel("world x (m)")
        axis.set_ylabel("world y (m)")
    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=2)
    fig.suptitle("Combined motion is a family of piecewise trochoids, not one global curve", fontsize=15)
    save_figure(fig, output, "trajectory_truth_pnp_gallery")


def plot_center_cdf(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    labels = {
        "single_primary": "one plate + oracle offset",
        "two_plate_adjacent_oracle_offset_mean": "two adjacent plates + oracle offsets",
    }
    for estimator, label in labels.items():
        selected = [row for row in rows if row["estimator"] == estimator]
        for axis, metric, title in (
            (axes[0], "error_cross_depth_m", "cross-depth center error"),
            (axes[1], "error_depth_m", "depth center error"),
        ):
            values = np.sort([float(row[metric]) * 1000 for row in selected])
            axis.plot(values, np.arange(1, len(values) + 1) / len(values), label=label)
            axis.set_xlabel(f"{title} (mm)")
            axis.set_ylabel("empirical CDF")
            axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Instantaneous center recovery remains noisy even with oracle radius/orientation")
    save_figure(fig, output, "center_recovery_complete_cdf")


def plot_prediction_summary(summary: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, domain in zip(axes, DOMAINS):
        for method in METHODS:
            selected = [
                row
                for row in summary
                if row["input_domain"] == domain
                and abs(float(row["history_window_s"]) - 0.25) < 1e-9
                and row["future_regime"] == "constant_twist"
                and row["method"] == method
            ]
            selected.sort(key=lambda row: float(row["horizon_s"]))
            if not selected:
                continue
            axis.plot(
                [float(row["horizon_s"]) * 1000 for row in selected],
                [float(row["error_cross_depth_m_p95"]) * 1000 for row in selected],
                marker="o",
                label=method,
                color=COLORS[method],
            )
        axis.axhline(55, color="black", ls="--", lw=1, label="55 mm diagnostic gate")
        axis.set_title(f"{domain} history")
        axis.set_xlabel("future horizon (ms)")
        axis.set_ylabel("P95 cross-depth error (mm)")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Causal 250 ms history, same physical plate, constant-twist intervals")
    save_figure(fig, output, "prediction_cross_depth_p95")


def plot_omega(omega_rows: list[dict[str, Any]], output: Path) -> None:
    selected = [
        row for row in omega_rows
        if abs(float(row["history_window_s"]) - 0.25) < 1e-9
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, domain in zip(axes, DOMAINS):
        values = [row for row in selected if row["input_domain"] == domain]
        axis.scatter(
            [row["truth_omega_rad_s"] for row in values],
            [row["estimated_omega_rad_s"] for row in values],
            s=7,
            alpha=0.3,
        )
        axis.plot([-16, 16], [-16, 16], color="black", ls="--", lw=1)
        axis.set_xlim(-16.5, 16.5)
        axis.set_ylim(-16.5, 16.5)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(domain)
        axis.set_xlabel("truth ω (rad/s)")
        axis.set_ylabel("causally estimated ω (rad/s)")
        axis.grid(alpha=0.25)
    fig.suptitle("Frequency is identifiable on clean arcs but PnP and short low-rate arcs remain difficult")
    save_figure(fig, output, "omega_estimation_scatter")


def plot_representative(representative: dict[str, Any], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    slots = representative["slots"]
    clean = representative["clean_positions"]
    pnp = representative["pnp_positions"]
    slot_colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    for slot in range(4):
        active = slots == slot
        axis.plot(clean[active, 0], clean[active, 1], color=slot_colors[slot], lw=1.4)
        axis.scatter(pnp[active, 0], pnp[active, 1], color=slot_colors[slot], s=13, alpha=0.35)
    anchor: Frame = representative["anchor"]
    future: Frame = representative["future"]
    primary = int(representative["primary"])
    effective_horizon_s = float(representative["effective_horizon_s"])
    axis.scatter(
        [future.armor_world_m[primary, 0]],
        [future.armor_world_m[primary, 1]],
        marker="*",
        s=180,
        color="black",
        label="future truth (200 ms)",
    )
    geometry = anchor.armor_local_m
    for method in ("slot_cv", "rigid_oracle_omega", "rigid_estimated_omega"):
        omega, model = representative["models"][("pnp", method)]
        if model is None:
            continue
        if method == "slot_cv":
            prediction = np.asarray([1.0, effective_horizon_s]) @ model
        else:
            prediction = predict_rigid(
                model, geometry, primary, float(omega), effective_horizon_s
            )
        axis.scatter(
            [prediction[0]],
            [prediction[1]],
            marker="X",
            s=100,
            color=COLORS[method],
            label=f"{method} prediction",
        )
    axis.plot(
        [frame.center_world_m[0] for frame in (anchor, future)],
        [frame.center_world_m[1] for frame in (anchor, future)],
        color="black",
        lw=2,
        label="truth center motion",
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    axis.set_title("Representative causal history: PnP points, truth arcs and 200 ms predictions")
    save_figure(fig, output, "representative_prediction_overlay")


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    workspace = args.workspace.resolve() if args.workspace else repo.parents[1]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evidence directory: {output}")
    output.mkdir(parents=True)
    dataset_root = workspace / "dataset" / "autoaim-stage3-v1"
    runtime_root = workspace / "runtime" / "stage3-generalization-fixed6mm-20260729-v1"

    sessions: dict[str, list[Frame]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    sources: list[Path] = []
    conditions = []
    segment_rows = []
    for suffix in DEV_SESSION_SUFFIXES:
        session_id = SESSION_PREFIX + suffix
        frames, manifest, session_sources = load_session(dataset_root, runtime_root, session_id)
        sessions[session_id] = frames
        manifests[session_id] = manifest
        sources.extend(session_sources)
        conditions.append(
            {
                "session_id": session_id,
                "split_role": "validation" if suffix == "00" else "train",
                "distance_m": manifest["distance_m"],
                "linear_speed_mps": manifest["linear_speed_mps"],
                "spin_rad_s": manifest["spin_rad_s"],
                "direction_deg": manifest["direction_deg"],
                "frame_count": len(frames),
                "observed_frame_count": sum(frame.observed_mask.any() for frame in frames),
                "two_or_more_plate_frame_count": sum(frame.observed_mask.sum() >= 2 for frame in frames),
            }
        )
        for segment_id in sorted({frame.segment_id for frame in frames}):
            segment_frames = [frame for frame in frames if frame.segment_id == segment_id]
            segment_rows.append(
                {
                    "session_id": session_id,
                    "segment_id": segment_id,
                    "duration_s": (segment_frames[-1].timestamp_ns - segment_frames[0].timestamp_ns) / 1e9,
                    "speed_mps": float(np.linalg.norm(segment_frames[len(segment_frames) // 2].velocity_world_mps)),
                    "yaw_rate_rad_s": segment_frames[len(segment_frames) // 2].yaw_rate_rad_s,
                    "frame_count": len(segment_frames),
                }
            )

    center_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    omega_rows: list[dict[str, Any]] = []
    representative = None
    for session_id, frames in sessions.items():
        center_rows.extend(collect_center_rows(session_id, frames))
        predictions, omegas, example = evaluate_predictions(
            session_id,
            frames,
            history_windows_s=HISTORY_WINDOWS_S,
            horizons_s=HORIZONS_S,
            anchor_stride=args.anchor_stride,
            omega_grid_step=args.omega_grid_step,
            max_omega=args.max_omega,
            huber_delta_m=args.huber_delta_m,
        )
        prediction_rows.extend(predictions)
        omega_rows.extend(omegas)
        if representative is None and example is not None:
            representative = example

    prediction_summary = summarize_predictions(prediction_rows)
    prediction_condition_summary = summarize_prediction_conditions(prediction_rows)
    center_summary = summarize_centers(center_rows)
    omega_summary = summarize_omega(omega_rows)
    write_csv_gz(output / "prediction_distribution.csv.gz", prediction_rows)
    write_csv_gz(output / "center_recovery_distribution.csv.gz", center_rows)
    write_csv_gz(output / "omega_estimation_distribution.csv.gz", omega_rows)
    write_plain_csv(output / "prediction_summary.csv", prediction_summary)
    write_plain_csv(
        output / "prediction_condition_summary.csv", prediction_condition_summary
    )
    write_plain_csv(output / "center_recovery_summary.csv", center_summary)
    write_plain_csv(output / "omega_estimation_summary.csv", omega_summary)
    write_plain_csv(output / "condition_coverage.csv", conditions)
    write_plain_csv(output / "constant_twist_segments.csv", segment_rows)

    plot_trajectory_gallery(sessions, manifests, output)
    plot_center_cdf(center_rows, output)
    plot_prediction_summary(prediction_summary, output)
    plot_omega(omega_rows, output)
    if representative is not None:
        plot_representative(representative, output)

    source_records = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(set(sources), key=str)
    ]
    artifact_records = []
    for path in sorted(output.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        artifact_records.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    write_json(
        output / "manifest.json",
        {
            "schema_version": "combined-motion-factorization-evidence-v1",
            "created_by": str(Path(__file__).resolve()),
            "created_by_sha256": sha256(Path(__file__).resolve()),
            "test_split_accessed": False,
            "sealed_test_session": SESSION_PREFIX + SEALED_TEST_SUFFIX,
            "development_sessions": list(sessions),
            "association_contract": "truth-only injective nearest-3d handles; non-deployable",
            "prediction_contract": {
                "same_physical_plate": True,
                "causal_history_only": True,
                "history_windows_s": list(HISTORY_WINDOWS_S),
                "horizons_s": list(HORIZONS_S),
                "anchor_stride_frames": args.anchor_stride,
                "future_truth_used_only_after_prediction": True,
                "constant_twist_and_cross_reversal_reported_separately": True,
            },
            "fit_configuration": {
                "omega_grid_step_rad_s": args.omega_grid_step,
                "max_abs_omega_rad_s": args.max_omega,
                "pnp_huber_delta_m": args.huber_delta_m,
            },
            "model_contract": (
                "p_j(t)=c0+v*t+R(omega*t)R(theta0)q_j; fixed geometry, "
                "shared phase-scale, no per-frame center reconstruction"
            ),
            "row_counts": {
                "prediction_distribution": len(prediction_rows),
                "center_recovery_distribution": len(center_rows),
                "omega_estimation_distribution": len(omega_rows),
            },
            "sources": source_records,
            "artifacts": artifact_records,
        },
    )
    print(json.dumps({
        "output": str(output),
        "prediction_rows": len(prediction_rows),
        "center_rows": len(center_rows),
        "omega_rows": len(omega_rows),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
