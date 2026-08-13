#!/usr/bin/env python3
"""Compare causal trajectory-processing families with grouped holdouts.

This is an oracle-identity upper-bound evaluation: physical slot labels group
history and truth supplies future labels, but neither slot nor any truth field
is part of the model feature vector. The deployable association problem is
reported separately and is not solved here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HORIZONS = (0.05, 0.10, 0.20)
HISTORY_SIZE = 16
MAX_HISTORY_SPAN_S = 0.75
MAX_CONSECUTIVE_GAP_S = 0.12
EVALUATION_RATE_HZ = 10.0
MLP_SEEDS = (7, 17, 29, 43, 61)
MAX_TRAIN_EXAMPLES = 20000
MAX_EVAL_PER_RUN_HORIZON = 40
FEATURE_ALLOWLIST = (
    "timestamp_ns",
    "u_deg",
    "v_deg",
    "pnp_camera_z_m",
    "pnp_yaw_absolute_rad",
    "observation_armor_count",
    "gimbal_yaw_deg",
    "gimbal_pitch_deg",
    "pnp_camera_z_valid",
    "pnp_yaw_valid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quick", action="store_true", help="run one MLP seed and repeat holdout only")
    parser.add_argument(
        "--split",
        action="append",
        choices=("repeat_holdout", "leave_distance_out", "leave_radius_out", "motion_transfer", "leave_cell_out"),
        help="limit evaluation to selected split families; repeat for multiple families",
    )
    parser.add_argument("--core-methods", action="store_true", help="evaluate screened core methods only")
    parser.add_argument("--mlp-seed-count", type=int, default=len(MLP_SEEDS))
    return parser.parse_args()


def parse_inputs(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    seen_paths = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"analysis input must be LABEL=DIR: {value}")
        label, raw = value.split("=", 1)
        path = Path(raw).resolve()
        if not label or not path.is_dir():
            raise ValueError(value)
        if path in seen_paths:
            raise ValueError(f"duplicate analysis path: {path}")
        seen_paths.add(path)
        result.append((label, path))
    return result


def validate_analysis_source(label: str, analysis: Path) -> str:
    summary_path = analysis / "analysis_summary.json"
    if not summary_path.exists():
        raise ValueError(f"analysis summary is absent: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    roots = [Path(raw).resolve() for raw in summary.get("roots", [])]
    if not roots:
        raise ValueError(f"analysis has no source roots: {analysis}")
    motion_modes = set()
    audited_runs = 0
    for root in roots:
        collection = root / "collection_manifest.json"
        if not collection.exists():
            raise ValueError(f"collection manifest is absent: {collection}")
        collection_manifest = json.loads(collection.read_text(encoding="utf-8-sig"))
        motion_modes.add(str(collection_manifest.get("motion_mode", "")))
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            run_manifest = run_dir / "collection_run_manifest.json"
            if not run_manifest.exists():
                continue
            audit_path = run_dir / "truth_motion_audit.json"
            if not audit_path.exists():
                raise ValueError(f"truth motion audit is absent: {audit_path}")
            audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
            if not audit.get("passed"):
                raise ValueError(f"truth motion audit failed: {audit_path}")
            audited_runs += 1
    if audited_runs == 0 or len(motion_modes) != 1:
        raise ValueError(f"analysis provenance is not one audited motion mode: {analysis}, {motion_modes}")
    motion_mode = next(iter(motion_modes))
    aliases = {"spin": {"spin"}, "linear_and_spin": {"combined", "linear_and_spin"}}
    if label not in aliases.get(motion_mode, {motion_mode}):
        raise ValueError(f"label {label!r} conflicts with audited motion mode {motion_mode!r}")
    return motion_mode


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any, default: float = 0.0) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) else default


def optional_finite(value: Any) -> float:
    return finite(value, float("nan"))


def fill_missing(values: np.ndarray) -> np.ndarray:
    """Causal forward fill; a value is never sourced from a later frame."""
    result = np.zeros_like(values)
    last = 0.0
    for index, value in enumerate(values):
        if math.isfinite(float(value)):
            last = float(value)
        result[index] = last
    return result


def unwrap_degrees(values: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(values)))


def truth_series(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = sorted(rows, key=lambda row: int(row["timestamp_ns"]))
    times = np.asarray([int(row["timestamp_ns"]) * 1e-9 for row in ordered])
    u = unwrap_degrees(np.asarray([float(row["u_deg"]) for row in ordered]))
    v = np.asarray([float(row["v_deg"]) for row in ordered])
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("truth timestamps must be strictly increasing within a run/slot")
    return times, u, v


def polynomial_prediction(times: np.ndarray, values: np.ndarray, horizon_s: float, degree: int) -> float:
    local = times - times[-1]
    design = np.column_stack([local ** power for power in range(degree + 1)])
    weights = np.linalg.lstsq(design, values, rcond=None)[0]
    return float(sum(weights[power] * horizon_s**power for power in range(degree + 1)))


def kf_forecast(
    times: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    horizon_s: float,
    q: float,
    r: float,
) -> np.ndarray:
    def one(values: np.ndarray) -> float:
        state = np.asarray([values[0], 0.0], dtype=float)
        covariance = np.diag([max(r, 1e-9), 100.0])
        observation_matrix = np.asarray([[1.0, 0.0]])
        for index in range(1, len(values)):
            dt = max(float(times[index] - times[index - 1]), 1e-5)
            transition = np.asarray([[1.0, dt], [0.0, 1.0]])
            process = q * np.asarray([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
            state = transition.dot(state)
            covariance = transition.dot(covariance).dot(transition.T) + process
            innovation = values[index] - float(observation_matrix.dot(state))
            innovation_variance = float(observation_matrix.dot(covariance).dot(observation_matrix.T)) + r
            gain = covariance.dot(observation_matrix.T)[:, 0] / max(innovation_variance, 1e-12)
            state = state + gain * innovation
            covariance = (np.eye(2) - np.outer(gain, observation_matrix[0])).dot(covariance)
        return float(state[0] + horizon_s * state[1])

    return np.asarray([one(u), one(v)])


def coordinated_turn_transition(state: np.ndarray, dt: float) -> np.ndarray:
    """Exact constant-turn-rate transition in the observed angular plane."""
    u, v, du, dv, omega = state
    angle = omega * dt
    if abs(omega) < 1e-6:
        return np.asarray([u + du * dt, v + dv * dt, du, dv, omega])
    sine = math.sin(angle)
    cosine = math.cos(angle)
    return np.asarray(
        [
            u + sine / omega * du - (1.0 - cosine) / omega * dv,
            v + (1.0 - cosine) / omega * du + sine / omega * dv,
            cosine * du - sine * dv,
            sine * du + cosine * dv,
            omega,
        ]
    )


def numerical_jacobian(state: np.ndarray, dt: float) -> np.ndarray:
    base = coordinated_turn_transition(state, dt)
    result = np.empty((len(state), len(state)))
    for index in range(len(state)):
        step = 1e-5 * max(1.0, abs(float(state[index])))
        shifted = state.copy()
        shifted[index] += step
        result[:, index] = (coordinated_turn_transition(shifted, dt) - base) / step
    return result


def coordinated_turn_initial_state(times: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    local = times - times[0]
    design = np.column_stack([np.ones_like(local), local])
    u_fit = np.linalg.lstsq(design, u, rcond=None)[0]
    v_fit = np.linalg.lstsq(design, v, rcond=None)[0]
    delta_t = np.diff(times)
    valid = delta_t > 1e-5
    omega = 0.0
    if np.count_nonzero(valid) >= 4:
        du = np.diff(u)[valid] / delta_t[valid]
        dv = np.diff(v)[valid] / delta_t[valid]
        speed = np.hypot(du, dv)
        useful = speed > max(float(np.percentile(speed, 25)), 1e-3)
        if np.count_nonzero(useful) >= 4:
            headings = np.unwrap(np.arctan2(dv[useful], du[useful]))
            mid = ((times[:-1] + times[1:]) * 0.5)[valid][useful]
            omega = float(np.linalg.lstsq(
                np.column_stack([np.ones_like(mid), mid - mid[-1]]), headings, rcond=None
            )[0][1])
    # The fitted observations initialize the state at the last initialization
    # timestamp.  They must not subsequently be replayed as fresh updates.
    return np.asarray(
        [
            u_fit[0] + u_fit[1] * local[-1],
            v_fit[0] + v_fit[1] * local[-1],
            u_fit[1],
            v_fit[1],
            np.clip(omega, -4.0, 4.0),
        ]
    )


def coordinated_turn_process_noise(dt: float, process_scale: float) -> np.ndarray:
    scale = max(dt, 1e-4)
    return np.diag(
        [
            process_scale * scale**3,
            process_scale * scale**3,
            process_scale * scale,
            process_scale * scale,
            0.05 * process_scale * scale,
        ]
    )


def ekf_coordinated_turn_forecast(
    times: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    horizon_s: float,
    process_scale: float,
    measurement_variance: float,
) -> np.ndarray:
    initialization = min(4, len(times))
    state = coordinated_turn_initial_state(
        times[:initialization], u[:initialization], v[:initialization]
    )
    covariance = np.diag([measurement_variance, measurement_variance, 25.0, 25.0, 1.0])
    observation = np.zeros((2, 5))
    observation[0, 0] = 1.0
    observation[1, 1] = 1.0
    measurement = np.eye(2) * measurement_variance
    identity = np.eye(5)
    for index in range(initialization, len(times)):
        dt = max(float(times[index] - times[index - 1]), 1e-5)
        transition_jacobian = numerical_jacobian(state, dt)
        state = coordinated_turn_transition(state, dt)
        covariance = (
            transition_jacobian.dot(covariance).dot(transition_jacobian.T)
            + coordinated_turn_process_noise(dt, process_scale)
        )
        innovation = np.asarray([u[index], v[index]]) - observation.dot(state)
        innovation_covariance = observation.dot(covariance).dot(observation.T) + measurement
        gain = np.linalg.solve(innovation_covariance, observation.dot(covariance)).T
        state = state + gain.dot(innovation)
        update = identity - gain.dot(observation)
        covariance = update.dot(covariance).dot(update.T) + gain.dot(measurement).dot(gain.T)
    return coordinated_turn_transition(state, horizon_s)[:2]


def sigma_points(state: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dimension = len(state)
    alpha, beta, kappa = 0.25, 2.0, 0.0
    lam = alpha**2 * (dimension + kappa) - dimension
    scaled = (dimension + lam) * covariance
    for jitter in (1e-10, 1e-8, 1e-6, 1e-4):
        try:
            root = np.linalg.cholesky(scaled + np.eye(dimension) * jitter)
            break
        except np.linalg.LinAlgError:
            continue
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(scaled)
        root = eigenvectors.dot(np.diag(np.sqrt(np.maximum(eigenvalues, 1e-8))))
    points = [state]
    for index in range(dimension):
        points.extend((state + root[:, index], state - root[:, index]))
    mean_weights = np.full(2 * dimension + 1, 1.0 / (2.0 * (dimension + lam)))
    covariance_weights = mean_weights.copy()
    mean_weights[0] = lam / (dimension + lam)
    covariance_weights[0] = mean_weights[0] + (1.0 - alpha**2 + beta)
    return np.asarray(points), mean_weights, covariance_weights


def ukf_coordinated_turn_forecast(
    times: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    horizon_s: float,
    process_scale: float,
    measurement_variance: float,
) -> np.ndarray:
    initialization = min(4, len(times))
    state = coordinated_turn_initial_state(
        times[:initialization], u[:initialization], v[:initialization]
    )
    covariance = np.diag([measurement_variance, measurement_variance, 25.0, 25.0, 1.0])
    measurement_noise = np.eye(2) * measurement_variance
    for index in range(initialization, len(times)):
        dt = max(float(times[index] - times[index - 1]), 1e-5)
        points, mean_weights, covariance_weights = sigma_points(state, covariance)
        propagated = np.asarray([coordinated_turn_transition(point, dt) for point in points])
        state = np.sum(propagated * mean_weights[:, None], axis=0)
        state_delta = propagated - state
        covariance = (
            np.einsum("i,ij,ik->jk", covariance_weights, state_delta, state_delta)
            + coordinated_turn_process_noise(dt, process_scale)
        )
        propagated, mean_weights, covariance_weights = sigma_points(state, covariance)
        state_delta = propagated - state
        measured_points = propagated[:, :2]
        measured_mean = np.sum(measured_points * mean_weights[:, None], axis=0)
        measured_delta = measured_points - measured_mean
        innovation_covariance = (
            np.einsum("i,ij,ik->jk", covariance_weights, measured_delta, measured_delta)
            + measurement_noise
        )
        cross_covariance = np.einsum(
            "i,ij,ik->jk", covariance_weights, state_delta, measured_delta
        )
        gain = np.linalg.solve(innovation_covariance, cross_covariance.T).T
        state = state + gain.dot(np.asarray([u[index], v[index]]) - measured_mean)
        covariance = covariance - gain.dot(innovation_covariance).dot(gain.T)
        covariance = (covariance + covariance.T) * 0.5
    points, mean_weights, _ = sigma_points(state, covariance)
    forecast = np.asarray([coordinated_turn_transition(point, horizon_s) for point in points])
    return np.sum(forecast * mean_weights[:, None], axis=0)[:2]


def make_feature(
    rows: list[dict],
    times: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    yaw: np.ndarray,
    z_valid: np.ndarray,
    yaw_valid: np.ndarray,
    horizon_s: float,
) -> np.ndarray:
    relative_time = times - times[-1]
    yaw_unwrapped = np.unwrap(yaw)
    u_velocity = polynomial_prediction(times, u, 0.0, 1) - polynomial_prediction(times, u, -0.01, 1)
    v_velocity = polynomial_prediction(times, v, 0.0, 1) - polynomial_prediction(times, v, -0.01, 1)
    u_acceleration = (
        polynomial_prediction(times, u, 0.01, 2)
        - 2.0 * polynomial_prediction(times, u, 0.0, 2)
        + polynomial_prediction(times, u, -0.01, 2)
    )
    v_acceleration = (
        polynomial_prediction(times, v, 0.01, 2)
        - 2.0 * polynomial_prediction(times, v, 0.0, 2)
        + polynomial_prediction(times, v, -0.01, 2)
    )
    current = rows[-1]
    return np.concatenate(
        [
            relative_time,
            u - u[-1],
            v - v[-1],
            z - z[-1],
            np.sin(yaw_unwrapped),
            np.cos(yaw_unwrapped),
            z_valid.astype(float),
            yaw_valid.astype(float),
            np.asarray(
                [
                    u[-1],
                    v[-1],
                    z[-1],
                    u_velocity / 0.01,
                    v_velocity / 0.01,
                    u_acceleration / 0.0001,
                    v_acceleration / 0.0001,
                    finite(current.get("observation_armor_count")),
                    finite(current.get("gimbal_yaw_deg")),
                    finite(current.get("gimbal_pitch_deg")),
                    horizon_s,
                ]
            ),
        ]
    )


def make_common_feature(
    times: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    horizon_s: float,
) -> np.ndarray:
    """Feature vector available to every u/v-only causal method."""
    relative_time = times - times[-1]
    return np.concatenate(
        [
            relative_time,
            u - u[-1],
            v - v[-1],
            np.asarray([u[-1], v[-1], horizon_s]),
        ]
    )


def make_uv_yaw_feature(
    times: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    yaw: np.ndarray,
    yaw_valid: np.ndarray,
    horizon_s: float,
) -> np.ndarray:
    """Feature vector matched to the periodic filter's u/v + PnP-yaw inputs."""
    return np.concatenate(
        [
            make_common_feature(times, u, v, horizon_s),
            np.sin(np.unwrap(yaw)),
            np.cos(np.unwrap(yaw)),
            yaw_valid.astype(float),
        ]
    )


def build_examples(label: str, analysis: Path) -> list[dict]:
    truth = read_jsonl(analysis / "truth_points.jsonl")
    observed = read_jsonl(analysis / "observed_points.jsonl")
    truth_grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    observed_grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in truth:
        truth_grouped[(row["run"], int(row["slot"]))].append(row)
    for row in observed:
        observed_grouped[(row["run"], int(row["slot"]))].append(row)
    examples = []
    for (run, slot), rows in sorted(observed_grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["timestamp_ns"]))
        if len(ordered) < HISTORY_SIZE:
            continue
        truth_times, truth_u, truth_v = truth_series(truth_grouped[(run, slot)])
        all_times = np.asarray([int(row["timestamp_ns"]) * 1e-9 for row in ordered])
        if len(all_times) > 1 and np.any(np.diff(all_times) <= 0.0):
            raise ValueError(f"observation timestamps must be strictly increasing: {run}, slot={slot}")
        all_u = unwrap_degrees(np.asarray([float(row["u_deg"]) for row in ordered]))
        all_v = np.asarray([float(row["v_deg"]) for row in ordered])
        raw_z = np.asarray([optional_finite(row.get("pnp_camera_z_m")) for row in ordered])
        raw_yaw = np.asarray([optional_finite(row.get("pnp_yaw_absolute_rad")) for row in ordered])
        all_z_valid = np.isfinite(raw_z)
        all_yaw_valid = np.isfinite(raw_yaw)
        all_z = fill_missing(raw_z)
        all_yaw = fill_missing(raw_yaw)
        last_example_time = -float("inf")
        for index in range(HISTORY_SIZE - 1, len(ordered)):
            start = index - HISTORY_SIZE + 1
            times = all_times[start : index + 1]
            if times[-1] - times[0] > MAX_HISTORY_SPAN_S or np.max(np.diff(times)) > MAX_CONSECUTIVE_GAP_S:
                continue
            if times[-1] - last_example_time < 1.0 / EVALUATION_RATE_HZ:
                continue
            last_example_time = times[-1]
            u = all_u[start : index + 1]
            v = all_v[start : index + 1]
            z = all_z[start : index + 1]
            yaw = all_yaw[start : index + 1]
            z_valid = all_z_valid[start : index + 1]
            yaw_valid = all_yaw_valid[start : index + 1]
            for horizon_s in HORIZONS:
                future = times[-1] + horizon_s
                if future > truth_times[-1] or future < truth_times[0]:
                    continue
                actual = np.asarray(
                    [np.interp(future, truth_times, truth_u), np.interp(future, truth_times, truth_v)]
                )
                hold = np.asarray([u[-1], v[-1]])
                cv = np.asarray(
                    [
                        polynomial_prediction(times, u, horizon_s, 1),
                        polynomial_prediction(times, v, horizon_s, 1),
                    ]
                )
                ca = np.asarray(
                    [
                        polynomial_prediction(times, u, horizon_s, 2),
                        polynomial_prediction(times, v, horizon_s, 2),
                    ]
                )
                row = ordered[index]
                actual_scale = optional_finite(row.get("actual_radial_scale"))
                if not math.isfinite(actual_scale):
                    raise ValueError(f"actual_radial_scale is required: {analysis}, {run}")
                examples.append(
                    {
                        "motion": label,
                        "run": f"{label}:{run}",
                        "run_name": run,
                        "repeat": int(row["repeat"]),
                        "distance_m": float(row["distance_m"]),
                        "scale": round(actual_scale, 3),
                        "slot": slot,
                        "timestamp_ns": int(row["timestamp_ns"]),
                        "horizon_s": horizon_s,
                        "example_id": (
                            f"{label}:{run}|slot={slot}|t={int(row['timestamp_ns'])}|"
                            f"h={int(round(horizon_s * 1000.0))}ms"
                        ),
                        "times": times,
                        "history_u": u,
                        "history_v": v,
                        "history_z": z,
                        "history_yaw": yaw,
                        "history_yaw_valid": yaw_valid,
                        "feature": make_feature(
                            ordered[start : index + 1], times, u, v, z, yaw, z_valid, yaw_valid, horizon_s
                        ),
                        "feature_uv": make_common_feature(times, u, v, horizon_s),
                        "feature_uv_yaw": make_uv_yaw_feature(
                            times, u, v, yaw, yaw_valid, horizon_s
                        ),
                        "actual": actual,
                        "hold": hold,
                        "cv": cv,
                        "ca": ca,
                        "residual_to_cv": actual - cv,
                    }
                )
    return examples


def error(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    def rays(values: np.ndarray) -> np.ndarray:
        tangent_u = np.tan(np.radians(values[:, 0]))
        tangent_v = np.tan(np.radians(values[:, 1]))
        result = np.column_stack([tangent_u, tangent_v, np.ones(len(values))])
        return result / np.linalg.norm(result, axis=1, keepdims=True)

    dots = np.sum(rays(predicted) * rays(actual), axis=1)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def select_evaluation_cohort(
    examples: list[dict], limit: int = MAX_EVAL_PER_RUN_HORIZON
) -> list[dict]:
    """Select one deterministic cohort shared by every candidate method."""
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for item in examples:
        grouped[(item["run"], item["horizon_s"])].append(item)
    selected = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: (item["timestamp_ns"], item["slot"], item["example_id"]))
        if len(ordered) > limit:
            indices = np.linspace(0, len(ordered) - 1, limit).astype(int)
            ordered = [ordered[index] for index in indices]
        selected.extend(ordered)
    return selected


def stratified_inner_validation(train: list[dict], limit: int) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for item in train:
        grouped[(item["motion"], item["distance_m"], item["scale"])].append(item)
    quota = max(1, limit // max(len(grouped), 1))
    selected = []
    for group in grouped.values():
        repeat = max(item["repeat"] for item in group)
        candidates = [item for item in group if item["repeat"] == repeat]
        if len(candidates) > quota:
            indices = np.linspace(0, len(candidates) - 1, quota).astype(int)
            candidates = [candidates[index] for index in indices]
        selected.extend(candidates)
    return selected[:limit]


def select_kf(train: list[dict]) -> tuple[float, float]:
    train = stratified_inner_validation(train, 240)
    best = (float("inf"), 0.01, 0.01)
    for q in (1e-3, 1e-2, 1e-1, 1.0):
        for r in (1e-2, 1e-1, 1.0):
            predictions = np.vstack(
                [
                    kf_forecast(
                        item["times"], item["history_u"], item["history_v"], item["horizon_s"], q, r
                    )
                    for item in train
                ]
            )
            score = float(np.percentile(error(predictions, np.vstack([item["actual"] for item in train])), 95))
            if score < best[0]:
                best = (score, q, r)
    return best[1], best[2]


def select_coordinated_turn_filter(train: list[dict], kind: str) -> tuple[float, float]:
    train = stratified_inner_validation(train, 160)
    function = ekf_coordinated_turn_forecast if kind == "ekf" else ukf_coordinated_turn_forecast
    best = (float("inf"), 0.01, 0.1)
    for process_scale in (1e-3, 1e-2, 1e-1):
        for measurement_variance in (1e-2, 1e-1, 1.0):
            predictions = np.vstack(
                [
                    function(
                        item["times"],
                        item["history_u"],
                        item["history_v"],
                        item["horizon_s"],
                        process_scale,
                        measurement_variance,
                    )
                    for item in train
                ]
            )
            score = float(
                np.percentile(error(predictions, np.vstack([item["actual"] for item in train])), 95)
            )
            if score < best[0]:
                best = (score, process_scale, measurement_variance)
    return best[1], best[2]


def select_shared_coordinated_turn_filter(train: list[dict]) -> tuple[float, float]:
    train = stratified_inner_validation(train, 120)
    best = (float("inf"), 0.01, 0.1)
    actual = np.vstack([item["actual"] for item in train])
    for process_scale in (1e-3, 1e-2, 1e-1):
        for measurement_variance in (1e-2, 1e-1, 1.0):
            scores = []
            for function in (ekf_coordinated_turn_forecast, ukf_coordinated_turn_forecast):
                predictions = np.vstack(
                    [
                        function(item["times"], item["history_u"], item["history_v"], item["horizon_s"], process_scale, measurement_variance)
                        for item in train
                    ]
                )
                scores.append(float(np.percentile(error(predictions, actual), 95)))
            score = float(np.mean(scores))
            if score < best[0]:
                best = (score, process_scale, measurement_variance)
    return best[1], best[2]


def grouped_metrics(items: list[dict], predictions: np.ndarray, method: str) -> dict:
    actual = np.vstack([item["actual"] for item in items])
    values = error(predictions, actual)
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for item, value in zip(items, values):
        grouped[(item["motion"], item["distance_m"], item["scale"], item["run"])].append(float(value))
    run_p95 = np.asarray([np.percentile(group, 95) for group in grouped.values()])
    condition_grouped: dict[tuple, list[float]] = defaultdict(list)
    for key, group in grouped.items():
        condition_grouped[key[:3]].extend(group)
    condition_p95 = np.asarray([np.percentile(group, 95) for group in condition_grouped.values()])
    return {
        "method": method,
        "samples": len(items),
        "error_p50_deg": float(np.percentile(values, 50)),
        "error_p90_deg": float(np.percentile(values, 90)),
        "error_p95_deg": float(np.percentile(values, 95)),
        "error_p99_deg": float(np.percentile(values, 99)),
        "run_equal_p95_deg": float(np.mean(run_p95)),
        "condition_equal_p95_deg": float(np.mean(condition_p95)),
        "worst_condition_p95_deg": float(np.max(condition_p95)),
    }


def per_run_metrics(items: list[dict], predictions: np.ndarray, method: str, base: dict) -> list[dict]:
    actual = np.vstack([item["actual"] for item in items])
    values = error(predictions, actual)
    grouped: dict[tuple, list[float]] = defaultdict(list)
    metadata = {}
    for item, value in zip(items, values):
        key = (item["motion"], item["distance_m"], item["scale"], item["run"])
        grouped[key].append(float(value))
        metadata[key] = item
    rows = []
    for key, group in grouped.items():
        item = metadata[key]
        rows.append(
            {
                "split": base["split"],
                "fold": base["fold"],
                "horizon_s": base["horizon_s"],
                "test_example_hash": base["test_example_hash"],
                "method": method,
                "motion": key[0],
                "distance_m": key[1],
                "scale": key[2],
                "run": key[3],
                "repeat": item["repeat"],
                "samples": len(group),
                "error_p50_deg": float(np.percentile(group, 50)),
                "error_p90_deg": float(np.percentile(group, 90)),
                "error_p95_deg": float(np.percentile(group, 95)),
                "error_p99_deg": float(np.percentile(group, 99)),
            }
        )
    return rows


def fit_learned(
    train: list[dict],
    test: list[dict],
    kind: str,
    seed: int = 0,
    feature_key: str = "feature",
) -> np.ndarray:
    x_train = np.vstack([item[feature_key] for item in train])
    y_train = np.vstack([item["residual_to_cv"] for item in train])
    if len(train) > MAX_TRAIN_EXAMPLES:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(train), MAX_TRAIN_EXAMPLES, replace=False)
        x_train = x_train[indices]
        y_train = y_train[indices]
    x_test = np.vstack([item[feature_key] for item in test])
    median = np.median(x_train, axis=0)
    x_train = np.where(np.isfinite(x_train), x_train, median)
    x_test = np.where(np.isfinite(x_test), x_test, median)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    if kind == "ridge":
        model = Ridge(alpha=10.0)
    elif kind == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=200,
            random_state=seed,
        )
    else:
        raise ValueError(kind)
    model.fit(x_train, y_train)
    residual = model.predict(x_test)
    return np.vstack([item["cv"] for item in test]) + residual


def split_definitions(
    examples: list[dict], quick: bool, allowed: set[str] | None = None
) -> list[tuple[str, str, list[dict], list[dict]]]:
    result = []
    repeats = sorted({item["repeat"] for item in examples})
    for repeat in (repeats[-1:] if quick else repeats):
        result.append(
            (
                "repeat_holdout",
                f"repeat={repeat}",
                [item for item in examples if item["repeat"] != repeat],
                [item for item in examples if item["repeat"] == repeat],
            )
        )
    if quick:
        return result
    for distance in sorted({item["distance_m"] for item in examples}):
        result.append(
            (
                "leave_distance_out",
                f"distance={distance:g}",
                [item for item in examples if item["distance_m"] != distance],
                [item for item in examples if item["distance_m"] == distance],
            )
        )
    for scale in sorted({item["scale"] for item in examples}):
        result.append(
            (
                "leave_radius_out",
                f"scale={scale:g}",
                [item for item in examples if item["scale"] != scale],
                [item for item in examples if item["scale"] == scale],
            )
        )
    motions = sorted({item["motion"] for item in examples})
    if len(motions) > 1:
        for motion in motions:
            result.append(
                (
                    "motion_transfer",
                    f"test={motion}",
                    [item for item in examples if item["motion"] != motion],
                    [item for item in examples if item["motion"] == motion],
                )
            )
    for distance, scale in sorted({(item["distance_m"], item["scale"]) for item in examples}):
        result.append(
            (
                "leave_cell_out",
                f"distance={distance:g},scale={scale:g}",
                [item for item in examples if (item["distance_m"], item["scale"]) != (distance, scale)],
                [item for item in examples if (item["distance_m"], item["scale"]) == (distance, scale)],
            )
        )
    return [item for item in result if allowed is None or item[0] in allowed]


def evaluate(
    examples: list[dict],
    quick: bool,
    allowed_splits: set[str] | None = None,
    core_methods: bool = False,
    mlp_seed_count: int = len(MLP_SEEDS),
) -> tuple[list[dict], list[dict]]:
    rows = []
    run_rows = []
    if mlp_seed_count < 1 or mlp_seed_count > len(MLP_SEEDS):
        raise ValueError(f"mlp_seed_count must be in [1, {len(MLP_SEEDS)}]")
    seeds = MLP_SEEDS[:1] if quick else MLP_SEEDS[:mlp_seed_count]
    for split_kind, fold, train_all, test_all in split_definitions(examples, quick, allowed_splits):
        overlap = {item["run"] for item in train_all}.intersection(item["run"] for item in test_all)
        if overlap:
            raise ValueError(f"train/test run leakage in {split_kind}/{fold}: {sorted(overlap)[:3]}")
        for horizon in HORIZONS:
            train = [item for item in train_all if item["horizon_s"] == horizon]
            test = [item for item in test_all if item["horizon_s"] == horizon]
            if len(train) < 100 or len(test) < 40:
                continue
            test_example_hash = hashlib.sha256(
                "\n".join(sorted(item["example_id"] for item in test)).encode("utf-8")
            ).hexdigest()
            base = {
                "split": split_kind,
                "fold": fold,
                "horizon_s": horizon,
                "train_samples": len(train),
                "test_example_hash": test_example_hash,
            }

            def record(method: str, predictions: np.ndarray, extra: dict | None = None) -> None:
                row = dict(base)
                row.update(grouped_metrics(test, predictions, method))
                if extra:
                    row.update(extra)
                rows.append(row)
                method_run_rows = per_run_metrics(test, predictions, method, base)
                if extra:
                    for method_run_row in method_run_rows:
                        method_run_row.update(extra)
                run_rows.extend(method_run_rows)

            for method in (("hold", "cv") if core_methods else ("hold", "cv", "ca")):
                record(method, np.vstack([item[method] for item in test]), {"input_tier": "common_uv"})
            q, r = select_kf(train)
            kf_predictions = np.vstack(
                [
                    kf_forecast(item["times"], item["history_u"], item["history_v"], horizon, q, r)
                    for item in test
                ]
            )
            record("kalman_cv", kf_predictions, {"kf_q": q, "kf_r": r, "input_tier": "common_uv"})
            if not core_methods:
              for filter_kind, function, method_name in (
                ("ekf", ekf_coordinated_turn_forecast, "ekf_coordinated_turn"),
                ("ukf", ukf_coordinated_turn_forecast, "ukf_coordinated_turn"),
              ):
                    process_scale, measurement_variance = select_coordinated_turn_filter(train, filter_kind)
                    filter_predictions = np.vstack(
                        [function(item["times"], item["history_u"], item["history_v"], horizon, process_scale, measurement_variance) for item in test]
                    )
                    record(method_name, filter_predictions, {"filter_process_scale": process_scale, "filter_measurement_variance": measurement_variance, "input_tier": "common_uv"})
              shared_process, shared_measurement = select_shared_coordinated_turn_filter(train)
              for function, method_name in (
                    (ekf_coordinated_turn_forecast, "ekf_coordinated_turn_shared"),
                    (ukf_coordinated_turn_forecast, "ukf_coordinated_turn_shared"),
              ):
                    predictions = np.vstack([function(item["times"], item["history_u"], item["history_v"], horizon, shared_process, shared_measurement) for item in test])
                    record(method_name, predictions, {"filter_process_scale": shared_process, "filter_measurement_variance": shared_measurement, "comparison_mode": "shared_parameters", "input_tier": "common_uv"})
            ridge_uv = fit_learned(train, test, "ridge", feature_key="feature_uv")
            record("ridge_uv_residual", ridge_uv, {"input_tier": "common_uv"})
            ridge_uv_yaw = fit_learned(train, test, "ridge", feature_key="feature_uv_yaw")
            record("ridge_uv_yaw_residual", ridge_uv_yaw, {"input_tier": "uv_yaw"})
            if not core_methods:
                ridge_predictions = fit_learned(train, test, "ridge", feature_key="feature")
                record("ridge_residual", ridge_predictions, {"input_tier": "extended_observation"})
            for seed in seeds:
                predictions_uv = fit_learned(train, test, "mlp", seed, feature_key="feature_uv")
                record("mlp_uv_residual", predictions_uv, {"seed": seed, "input_tier": "common_uv"})
                if not core_methods or seed == seeds[0]:
                    predictions_uv_yaw = fit_learned(
                        train, test, "mlp", seed, feature_key="feature_uv_yaw"
                    )
                    record(
                        "mlp_uv_yaw_residual",
                        predictions_uv_yaw,
                        {"seed": seed, "input_tier": "uv_yaw"},
                    )
                if not core_methods:
                    predictions = fit_learned(train, test, "mlp", seed, feature_key="feature")
                    record(
                        "mlp_residual",
                        predictions,
                        {"seed": seed, "input_tier": "extended_observation"},
                    )
    return rows, run_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    aggregate = []
    keys = sorted({(row["split"], row["horizon_s"], row["input_tier"], row["method"]) for row in rows})
    for split, horizon, input_tier, method in keys:
        selected = [
            row
            for row in rows
            if row["split"] == split
            and row["horizon_s"] == horizon
            and row["input_tier"] == input_tier
            and row["method"] == method
        ]
        aggregate.append(
            {
                "split": split,
                "horizon_s": horizon,
                "input_tier": input_tier,
                "method": method,
                "fold_seed_rows": len(selected),
                "condition_equal_p95_deg_mean": float(np.mean([row["condition_equal_p95_deg"] for row in selected])),
                "condition_equal_p95_deg_std": float(np.std([row["condition_equal_p95_deg"] for row in selected])),
                "worst_condition_p95_deg_mean": float(np.mean([row["worst_condition_p95_deg"] for row in selected])),
            }
        )
    return {"aggregate": aggregate}


def plot_methods(output: Path, aggregate: list[dict]) -> None:
    repeat = [row for row in aggregate if row["split"] == "repeat_holdout"]
    method_order = (
        "hold",
        "cv",
        "ca",
        "kalman_cv",
        "ekf_coordinated_turn",
        "ukf_coordinated_turn",
        "ridge_uv_residual",
        "mlp_uv_residual",
    )
    available = {row["method"] for row in repeat if row.get("input_tier") == "common_uv"}
    methods = tuple(method for method in method_order if method in available)
    fig, ax = plt.subplots(figsize=(11.0, 5.4), constrained_layout=True)
    x = np.arange(len(HORIZONS), dtype=float)
    width = 0.8 / max(len(methods), 1)
    colors = (
        "#999999",
        "#0072B2",
        "#56B4E9",
        "#009E73",
        "#CC79A7",
        "#F0E442",
        "#E69F00",
        "#D55E00",
    )
    for index, (method, color) in enumerate(zip(methods, colors)):
        values = []
        errors = []
        for horizon in HORIZONS:
            row = next(
                item
                for item in repeat
                if item["method"] == method
                and item["horizon_s"] == horizon
                and item["input_tier"] == "common_uv"
            )
            values.append(row["condition_equal_p95_deg_mean"])
            errors.append(row["condition_equal_p95_deg_std"])
        ax.bar(
            x + (index - (len(methods) - 1) / 2.0) * width,
            values,
            width=width,
            yerr=errors,
            capsize=2,
            color=color,
            label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(h * 1000)} ms" for h in HORIZONS])
    ax.set_ylabel("Condition-equal angular P95 (deg)")
    ax.set_title("Causal trajectory processing: held-repeat comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(output / "method_comparison_repeat_holdout.png", dpi=260, bbox_inches="tight")
    fig.savefig(output / "method_comparison_repeat_holdout.svg", bbox_inches="tight")
    plt.close(fig)


def availability_rows(motion: str, analysis: Path, examples: list[dict]) -> list[dict]:
    counts: dict[str, int] = defaultdict(int)
    for item in examples:
        counts[item["run"]] += 1
    rows = []
    with (analysis / "run_metrics.csv").open("r", newline="", encoding="utf-8-sig") as handle:
        for source in csv.DictReader(handle):
            run_id = f"{motion}:{source['run']}"
            rows.append(
                {
                    "motion": motion,
                    "run": run_id,
                    "distance_m": float(source["distance_m"]),
                    "scale": float(source["scale"]),
                    "repeat": int(source["repeat"]),
                    "truth_rows": int(source["truth_rows"]),
                    "observation_rows": int(source["observation_rows"]),
                    "observed_assignments": int(source["observed_assignments"]),
                    "eligible_example_rows": counts.get(run_id, 0),
                    "has_any_observation": int(source["observed_assignments"]) > 0,
                    "has_eligible_history": counts.get(run_id, 0) > 0,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    inputs = parse_inputs(args.analysis)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    examples = []
    availability = []
    sources = {}
    source_hash_pairs = set()
    for label, path in inputs:
        motion_mode = validate_analysis_source(label, path)
        truth_hash = sha256_file(path / "truth_points.jsonl")
        observed_hash = sha256_file(path / "observed_points.jsonl")
        pair = (truth_hash, observed_hash)
        if pair in source_hash_pairs:
            raise ValueError(f"duplicate analysis content: {path}")
        source_hash_pairs.add(pair)
        built = build_examples(motion_mode, path)
        examples.extend(built)
        availability.extend(availability_rows(motion_mode, path, built))
        sources[label] = {
            "analysis_dir": str(path),
            "audited_motion_mode": motion_mode,
            "truth_sha256": truth_hash,
            "observed_sha256": observed_hash,
        }
    example_ids = [item["example_id"] for item in examples]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("duplicate stable example_id across analysis inputs")
    examples = select_evaluation_cohort(examples)
    dataset_fingerprint = hashlib.sha256(
        json.dumps(
            {
                label: {
                    "truth_sha256": source["truth_sha256"],
                    "observed_sha256": source["observed_sha256"],
                }
                for label, source in sorted(sources.items())
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    rows, run_rows = evaluate(
        examples,
        args.quick,
        set(args.split) if args.split else None,
        args.core_methods,
        args.mlp_seed_count,
    )
    for row in rows + run_rows:
        row["dataset_fingerprint"] = dataset_fingerprint
    write_csv(output / "method_evaluation_rows.csv", rows)
    write_csv(output / "method_evaluation_run_rows.csv", run_rows)
    write_csv(output / "data_availability_runs.csv", availability)
    result = summarize(rows)
    plot_methods(output, result["aggregate"])
    script = Path(__file__).resolve()
    summary = {
        "kind": "trajectory_processing_method_selection",
        "oracle_identity_upper_bound": True,
        "truth_policy": "truth supplies future labels and oracle physical-slot/sample selection; no truth field, slot, distance, radius, or motion label is a model feature",
        "feature_allowlist": FEATURE_ALLOWLIST,
        "history_size": HISTORY_SIZE,
        "evaluation_rate_hz": EVALUATION_RATE_HZ,
        "evaluation_cohort_per_run_horizon": MAX_EVAL_PER_RUN_HORIZON,
        "horizons_s": HORIZONS,
        "mlp_seeds": MLP_SEEDS[:1] if args.quick else MLP_SEEDS[: args.mlp_seed_count],
        "core_methods": args.core_methods,
        "selected_split_families": args.split or "all",
        "examples": len(examples),
        "dataset_fingerprint": dataset_fingerprint,
        "expected_collection_runs": len(availability),
        "runs_with_any_observation": sum(bool(row["has_any_observation"]) for row in availability),
        "runs_with_eligible_history": sum(bool(row["has_eligible_history"]) for row in availability),
        "run_metric_rows": len(run_rows),
        "sources": sources,
        "script": {"path": str(script), "sha256": sha256_file(script)},
        **result,
    }
    with (output / "method_selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    retention = {
        "classification": "long_term_private_evidence",
        "deletion_allowed": False,
        "script": summary["script"],
        "artifacts": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    with (output / "retention_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(retention, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
