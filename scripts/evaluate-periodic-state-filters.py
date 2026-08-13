#!/usr/bin/env python3
"""Fair oracle-identity EKF/UKF comparison for an observation-phase Fourier state.

Truth is used only for future labels, grouped splits, and post-hoc scoring.  The
filter sees timestamped detector angles and PnP yaw.  Physical slots organize
histories, so this remains an explicitly labelled oracle-identity upper bound.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


# Every method-selection candidate is scored on the exact same stable example
# IDs.  Runtime can be reduced with --quick (fewer folds), never by silently
# subsampling only this method family.
MAX_EVAL_PER_RUN_HORIZON = 40
MAX_INNER_VALIDATION = 48
INITIALIZATION_OBSERVATIONS = 4
CHI2_90_DF2 = 4.605170186
CONFIGS = (
    {"name": "k1_smooth", "harmonics": 1, "q_center": 0.01, "q_phase": 0.01, "q_coeff": 0.001, "r_uv": 0.01, "r_yaw": 0.04},
    {"name": "k1_adaptive", "harmonics": 1, "q_center": 0.10, "q_phase": 0.05, "q_coeff": 0.010, "r_uv": 0.03, "r_yaw": 0.10},
    {"name": "k2_smooth", "harmonics": 2, "q_center": 0.01, "q_phase": 0.01, "q_coeff": 0.001, "r_uv": 0.01, "r_yaw": 0.04},
    {"name": "k2_adaptive", "harmonics": 2, "q_center": 0.10, "q_phase": 0.05, "q_coeff": 0.010, "r_uv": 0.03, "r_yaw": 0.10},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--split",
        action="append",
        choices=("repeat_holdout", "leave_distance_out", "leave_radius_out", "motion_transfer", "leave_cell_out"),
    )
    parser.add_argument("--core-methods", action="store_true", help="run only the shared-parameter EKF/UKF pair")
    return parser.parse_args()


def load_base(repo: Path):
    path = repo / "scripts" / "evaluate-trajectory-processing-methods.py"
    spec = importlib.util.spec_from_file_location("trajectory_method_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wrap_rad(value: float | np.ndarray) -> float | np.ndarray:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def spherical_error_deg(prediction: np.ndarray, actual: np.ndarray) -> float:
    values = np.vstack([prediction, actual])
    rays = np.column_stack(
        [np.tan(np.radians(values[:, 0])), np.tan(np.radians(values[:, 1])), np.ones(2)]
    )
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return float(np.degrees(np.arccos(np.clip(np.dot(rays[0], rays[1]), -1.0, 1.0))))


def state_dimension(harmonics: int) -> int:
    return 6 + 4 * harmonics


def transition(state: np.ndarray, dt: float, harmonics: int) -> np.ndarray:
    result = state.copy()
    result[0] += state[2] * dt
    result[1] += state[3] * dt
    result[4] = float(wrap_rad(state[4] + state[5] * dt))
    return result


def transition_jacobian(dt: float, harmonics: int) -> np.ndarray:
    result = np.eye(state_dimension(harmonics))
    result[0, 2] = dt
    result[1, 3] = dt
    result[4, 5] = dt
    return result


def process_noise(dt: float, config: dict) -> np.ndarray:
    dimension = state_dimension(config["harmonics"])
    result = np.eye(dimension) * 1e-12
    block = np.asarray([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    for position, velocity in ((0, 2), (1, 3)):
        indices = np.ix_([position, velocity], [position, velocity])
        result[indices] += config["q_center"] * block
    result[np.ix_([4, 5], [4, 5])] += config["q_phase"] * block
    result[6:, 6:] += np.eye(dimension - 6) * config["q_coeff"] * dt
    return result


def observation(state: np.ndarray, harmonics: int, include_yaw: bool) -> np.ndarray:
    phi = state[4]
    u, v = state[0], state[1]
    for harmonic in range(1, harmonics + 1):
        offset = 6 + 4 * (harmonic - 1)
        cosine, sine = math.cos(harmonic * phi), math.sin(harmonic * phi)
        u += state[offset] * cosine + state[offset + 1] * sine
        v += state[offset + 2] * cosine + state[offset + 3] * sine
    values = [u, v]
    if include_yaw:
        values.extend((math.cos(2.0 * phi), math.sin(2.0 * phi)))
    return np.asarray(values)


def observation_jacobian(state: np.ndarray, harmonics: int, include_yaw: bool) -> np.ndarray:
    rows = 4 if include_yaw else 2
    result = np.zeros((rows, len(state)))
    result[0, 0] = 1.0
    result[1, 1] = 1.0
    phi = state[4]
    for harmonic in range(1, harmonics + 1):
        offset = 6 + 4 * (harmonic - 1)
        cosine, sine = math.cos(harmonic * phi), math.sin(harmonic * phi)
        result[0, 4] += harmonic * (-state[offset] * sine + state[offset + 1] * cosine)
        result[1, 4] += harmonic * (-state[offset + 2] * sine + state[offset + 3] * cosine)
        result[0, offset : offset + 2] = (cosine, sine)
        result[1, offset + 2 : offset + 4] = (cosine, sine)
    if include_yaw:
        result[2, 4] = -2.0 * math.sin(2.0 * phi)
        result[3, 4] = 2.0 * math.cos(2.0 * phi)
    return result


def initial_state(item: dict, harmonics: int) -> tuple[np.ndarray, np.ndarray, int]:
    count = min(INITIALIZATION_OBSERVATIONS, len(item["times"]))
    times = np.asarray(item["times"][:count], dtype=float)
    local = times - times[-1]
    u = np.asarray(item["history_u"][:count], dtype=float)
    v = np.asarray(item["history_v"][:count], dtype=float)
    yaw = np.asarray(item["history_yaw"][:count], dtype=float)
    yaw_valid = np.asarray(item["history_yaw_valid"][:count], dtype=bool)
    if np.count_nonzero(yaw_valid) >= 4:
        phi = np.full(len(yaw), np.nan)
        phi[yaw_valid] = 0.5 * np.unwrap(2.0 * yaw[yaw_valid])
        phi = np.interp(times, times[yaw_valid], phi[yaw_valid])
        phase_fit = np.linalg.lstsq(
            np.column_stack([np.ones_like(local), local]), phi, rcond=None
        )[0]
        phase_now, omega = float(phase_fit[0]), float(np.clip(phase_fit[1], -4.0, 4.0))
    else:
        phase_now, omega = 0.0, 0.0
        phi = np.full(len(local), phase_now)
    columns = [np.ones_like(local), local]
    for harmonic in range(1, harmonics + 1):
        columns.extend((np.cos(harmonic * phi), np.sin(harmonic * phi)))
    design = np.column_stack(columns)
    regularizer = np.eye(design.shape[1]) * 1e-3
    regularizer[0:2, 0:2] = 0.0
    lhs = design.T.dot(design) + regularizer
    coefficients_u = np.linalg.solve(lhs, design.T.dot(u))
    coefficients_v = np.linalg.solve(lhs, design.T.dot(v))
    state = np.zeros(state_dimension(harmonics))
    state[0] = coefficients_u[0]
    state[1] = coefficients_v[0]
    state[2], state[3] = coefficients_u[1], coefficients_v[1]
    state[4] = float(wrap_rad(phase_now))
    state[5] = omega
    for harmonic in range(harmonics):
        source = 2 + 2 * harmonic
        target = 6 + 4 * harmonic
        state[target : target + 2] = coefficients_u[source : source + 2]
        state[target + 2 : target + 4] = coefficients_v[source : source + 2]
    covariance = np.diag(
        [0.10, 0.10, 4.0, 4.0, 0.25, 0.50] + [1.0] * (4 * harmonics)
    )
    return state, covariance, count


def measurement_for(item: dict, index: int, config: dict) -> tuple[np.ndarray, np.ndarray, bool]:
    include_yaw = bool(item["history_yaw_valid"][index])
    values = [float(item["history_u"][index]), float(item["history_v"][index])]
    diagonal = [config["r_uv"], config["r_uv"]]
    if include_yaw:
        yaw = float(item["history_yaw"][index])
        values.extend((math.cos(2.0 * yaw), math.sin(2.0 * yaw)))
        diagonal.extend((config["r_yaw"], config["r_yaw"]))
    return np.asarray(values), np.diag(diagonal), include_yaw


def ekf(item: dict, config: dict) -> tuple[np.ndarray, np.ndarray, int]:
    harmonics = config["harmonics"]
    state, covariance, initialized = initial_state(item, harmonics)
    times = item["times"]
    gated = 0
    for index in range(initialized, len(times)):
        dt = max(float(times[index] - times[index - 1]), 1e-5)
        f = transition_jacobian(dt, harmonics)
        state = transition(state, dt, harmonics)
        covariance = f.dot(covariance).dot(f.T) + process_noise(dt, config)
        measured, noise, include_yaw = measurement_for(item, index, config)
        h = observation_jacobian(state, harmonics, include_yaw)
        predicted = observation(state, harmonics, include_yaw)
        innovation = measured - predicted
        innovation_covariance = h.dot(covariance).dot(h.T) + noise
        gain = np.linalg.solve(innovation_covariance, h.dot(covariance)).T
        state = state + gain.dot(innovation)
        state[4] = float(wrap_rad(state[4]))
        update = np.eye(len(state)) - gain.dot(h)
        covariance = update.dot(covariance).dot(update.T) + gain.dot(noise).dot(gain.T)
    state = transition(state, float(item["horizon_s"]), harmonics)
    f = transition_jacobian(float(item["horizon_s"]), harmonics)
    covariance = f.dot(covariance).dot(f.T) + process_noise(float(item["horizon_s"]), config)
    h = observation_jacobian(state, harmonics, False)
    return observation(state, harmonics, False), h.dot(covariance).dot(h.T), gated


def sigma_points(state: np.ndarray, covariance: np.ndarray, alpha: float = 0.25):
    dimension = len(state)
    lam = alpha**2 * dimension - dimension
    scaled = (dimension + lam) * covariance
    for jitter in (1e-10, 1e-8, 1e-6, 1e-4):
        try:
            root = np.linalg.cholesky(scaled + np.eye(dimension) * jitter)
            break
        except np.linalg.LinAlgError:
            continue
    else:
        values, vectors = np.linalg.eigh(scaled)
        root = vectors.dot(np.diag(np.sqrt(np.maximum(values, 1e-8))))
    points = [state]
    for index in range(dimension):
        points.extend((state + root[:, index], state - root[:, index]))
    wm = np.full(2 * dimension + 1, 1.0 / (2.0 * (dimension + lam)))
    wc = wm.copy()
    wm[0] = lam / (dimension + lam)
    wc[0] = wm[0] + 1.0 - alpha**2 + 2.0
    return np.asarray(points), wm, wc


def state_mean(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.sum(points * weights[:, None], axis=0)
    result[4] = math.atan2(
        float(np.sum(weights * np.sin(points[:, 4]))),
        float(np.sum(weights * np.cos(points[:, 4]))),
    )
    return result


def state_delta(points: np.ndarray, mean: np.ndarray) -> np.ndarray:
    result = points - mean
    result[:, 4] = wrap_rad(result[:, 4])
    return result


def ukf(item: dict, config: dict) -> tuple[np.ndarray, np.ndarray, int]:
    harmonics = config["harmonics"]
    state, covariance, initialized = initial_state(item, harmonics)
    times = item["times"]
    gated = 0
    for index in range(initialized, len(times)):
        dt = max(float(times[index] - times[index - 1]), 1e-5)
        points, wm, wc = sigma_points(state, covariance)
        propagated = np.asarray([transition(point, dt, harmonics) for point in points])
        state = state_mean(propagated, wm)
        delta = state_delta(propagated, state)
        covariance = np.einsum("i,ij,ik->jk", wc, delta, delta) + process_noise(dt, config)
        propagated, wm, wc = sigma_points(state, covariance)
        delta = state_delta(propagated, state)
        measured, noise, include_yaw = measurement_for(item, index, config)
        projected = np.asarray([observation(point, harmonics, include_yaw) for point in propagated])
        predicted = np.sum(projected * wm[:, None], axis=0)
        projected_delta = projected - predicted
        innovation_covariance = np.einsum("i,ij,ik->jk", wc, projected_delta, projected_delta) + noise
        innovation = measured - predicted
        cross = np.einsum("i,ij,ik->jk", wc, delta, projected_delta)
        gain = np.linalg.solve(innovation_covariance, cross.T).T
        state = state + gain.dot(innovation)
        state[4] = float(wrap_rad(state[4]))
        covariance = covariance - gain.dot(innovation_covariance).dot(gain.T)
        covariance = (covariance + covariance.T) * 0.5
    points, wm, wc = sigma_points(state, covariance)
    propagated = np.asarray([transition(point, float(item["horizon_s"]), harmonics) for point in points])
    forecast_state = state_mean(propagated, wm)
    forecast_delta = state_delta(propagated, forecast_state)
    forecast_covariance = (
        np.einsum("i,ij,ik->jk", wc, forecast_delta, forecast_delta)
        + process_noise(float(item["horizon_s"]), config)
    )
    propagated, wm, wc = sigma_points(forecast_state, forecast_covariance)
    projected = np.asarray([observation(point, harmonics, False) for point in propagated])
    mean = np.sum(projected * wm[:, None], axis=0)
    delta = projected - mean
    return mean, np.einsum("i,ij,ik->jk", wc, delta, delta), gated


def downsample(items: list[dict], limit: int | None = MAX_EVAL_PER_RUN_HORIZON) -> list[dict]:
    if limit is None:
        return list(items)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for item in items:
        grouped[(item["run"], item["horizon_s"])].append(item)
    result = []
    for group in grouped.values():
        if len(group) <= limit:
            result.extend(group)
        else:
            indices = np.linspace(0, len(group) - 1, limit).astype(int)
            result.extend(group[index] for index in indices)
    return result


def split_definitions(examples: list[dict], quick: bool, allowed: set[str] | None = None):
    repeats = sorted({item["repeat"] for item in examples})
    selected_repeats = repeats[-1:] if quick else repeats
    result = [
        ("repeat_holdout", f"repeat={repeat}", [x for x in examples if x["repeat"] != repeat], [x for x in examples if x["repeat"] == repeat])
        for repeat in selected_repeats
    ]
    if quick:
        return result
    for field, name in (("distance_m", "leave_distance_out"), ("scale", "leave_radius_out"), ("motion", "motion_transfer")):
        for value in sorted({item[field] for item in examples}, key=str):
            result.append((name, f"{field}={value}", [x for x in examples if x[field] != value], [x for x in examples if x[field] == value]))
    for distance, scale in sorted({(item["distance_m"], item["scale"]) for item in examples}):
        result.append(("leave_cell_out", f"distance={distance},scale={scale}", [x for x in examples if (x["distance_m"], x["scale"]) != (distance, scale)], [x for x in examples if (x["distance_m"], x["scale"]) == (distance, scale)]))
    return [item for item in result if allowed is None or item[0] in allowed]


def inner_validation(train: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for item in train:
        grouped[(item["motion"], item["distance_m"], item["scale"])].append(item)
    quota = max(1, MAX_INNER_VALIDATION // max(len(grouped), 1))
    selected = []
    for group in grouped.values():
        repeat = max(item["repeat"] for item in group)
        candidates = [item for item in group if item["repeat"] == repeat]
        if len(candidates) > quota:
            indices = np.linspace(0, len(candidates) - 1, quota).astype(int)
            candidates = [candidates[index] for index in indices]
        selected.extend(candidates)
    return selected[:MAX_INNER_VALIDATION]


def score_predictions(
    items: list[dict], outputs: list[tuple[np.ndarray, np.ndarray, int]], method: str, config: dict
) -> tuple[dict, list[dict]]:
    errors, nll, covered = [], [], []
    gated = 0
    by_run: dict[str, list[float]] = defaultdict(list)
    by_condition: dict[tuple, list[float]] = defaultdict(list)
    run_metadata = {}
    for item, (prediction, covariance, gate_count) in zip(items, outputs):
        residual = np.asarray(item["actual"]) - prediction
        value = spherical_error_deg(prediction, np.asarray(item["actual"]))
        covariance = covariance + np.eye(2) * 1e-8
        mahal = float(residual.T.dot(np.linalg.solve(covariance, residual)))
        sign, logdet = np.linalg.slogdet(covariance)
        errors.append(value)
        nll.append(0.5 * (mahal + logdet + 2.0 * math.log(2.0 * math.pi)) if sign > 0 else float("nan"))
        covered.append(mahal <= CHI2_90_DF2)
        gated += gate_count
        by_run[item["run"]].append(value)
        run_metadata[item["run"]] = item
        by_condition[(item["motion"], item["distance_m"], item["scale"])].append(value)
    values = np.asarray(errors)
    run_p95 = [np.percentile(group, 95) for group in by_run.values()]
    condition_p95 = [np.percentile(group, 95) for group in by_condition.values()]
    summary = {
        "method": method,
        "config": config["name"],
        "samples": len(items),
        "error_p50_deg": float(np.percentile(values, 50)),
        "error_p90_deg": float(np.percentile(values, 90)),
        "error_p95_deg": float(np.percentile(values, 95)),
        "error_p99_deg": float(np.percentile(values, 99)),
        "run_equal_p95_deg": float(np.mean(run_p95)),
        "condition_equal_p95_deg": float(np.mean(condition_p95)),
        "worst_condition_p95_deg": float(np.max(condition_p95)),
        "coverage_90": float(np.mean(covered)),
        "nll_mean": float(np.nanmean(nll)),
        "yaw_gated_updates": gated,
    }
    run_rows = []
    for run, group in by_run.items():
        item = run_metadata[run]
        run_rows.append(
            {
                "method": method,
                "config": config["name"],
                "motion": item["motion"],
                "distance_m": item["distance_m"],
                "scale": item["scale"],
                "run": run,
                "repeat": item["repeat"],
                "samples": len(group),
                "error_p50_deg": float(np.percentile(group, 50)),
                "error_p90_deg": float(np.percentile(group, 90)),
                "error_p95_deg": float(np.percentile(group, 95)),
                "error_p99_deg": float(np.percentile(group, 99)),
            }
        )
    return summary, run_rows


def choose_config(validation: list[dict], function) -> dict:
    best = None
    for config in CONFIGS:
        outputs = [function(item, config) for item in validation]
        errors = [spherical_error_deg(output[0], np.asarray(item["actual"])) for item, output in zip(validation, outputs)]
        candidate = (float(np.percentile(errors, 95)), config)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best[1]


def choose_shared_config(validation: list[dict]) -> dict:
    best = None
    for config in CONFIGS:
        method_scores = []
        for function in (ekf, ukf):
            outputs = [function(item, config) for item in validation]
            errors = [
                spherical_error_deg(output[0], np.asarray(item["actual"]))
                for item, output in zip(validation, outputs)
            ]
            method_scores.append(float(np.percentile(errors, 95)))
        candidate = (float(np.mean(method_scores)), config)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best[1]


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    base = load_base(repo)
    examples = []
    sources = {}
    seen_paths = set()
    seen_hashes = set()
    for value in args.analysis:
        label, raw = value.split("=", 1)
        path = Path(raw).resolve()
        if path in seen_paths:
            raise ValueError(f"duplicate analysis path: {path}")
        seen_paths.add(path)
        motion_mode = base.validate_analysis_source(label, path)
        truth_hash = sha256(path / "truth_points.jsonl")
        observed_hash = sha256(path / "observed_points.jsonl")
        if (truth_hash, observed_hash) in seen_hashes:
            raise ValueError(f"duplicate analysis content: {path}")
        seen_hashes.add((truth_hash, observed_hash))
        examples.extend(base.build_examples(motion_mode, path))
        sources[label] = {"path": str(path), "audited_motion_mode": motion_mode, "truth_sha256": truth_hash, "observed_sha256": observed_hash}
    example_ids = [item["example_id"] for item in examples]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("duplicate stable example_id across analysis inputs")
    examples = base.select_evaluation_cohort(examples, MAX_EVAL_PER_RUN_HORIZON)
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
    rows = []
    run_rows = []
    for split, fold, train_all, test_all in split_definitions(
        examples, args.quick, set(args.split) if args.split else None
    ):
        overlap = {item["run"] for item in train_all}.intersection(item["run"] for item in test_all)
        if overlap:
            raise ValueError(f"train/test run leakage in {split}/{fold}: {sorted(overlap)[:3]}")
        for horizon in base.HORIZONS:
            train = [item for item in train_all if item["horizon_s"] == horizon]
            test = [item for item in test_all if item["horizon_s"] == horizon]
            if len(train) < 100 or len(test) < 40:
                continue
            test_example_hash = hashlib.sha256(
                "\n".join(sorted(item["example_id"] for item in test)).encode("utf-8")
            ).hexdigest()
            validation = inner_validation(train)
            shared_config = choose_shared_config(validation)
            for method, function in (("periodic_ekf_shared", ekf), ("periodic_ukf_shared", ukf)):
                outputs = [function(item, shared_config) for item in test]
                row = {"split": split, "fold": fold, "horizon_s": horizon, "train_samples": len(train), "inner_validation_samples": len(validation), "comparison_mode": "shared_parameters", "input_tier": "uv_yaw", "test_example_hash": test_example_hash, "dataset_fingerprint": dataset_fingerprint}
                score, method_run_rows = score_predictions(test, outputs, method, shared_config)
                row.update(score)
                rows.append(row)
                for method_run_row in method_run_rows:
                    method_run_row.update({"split": split, "fold": fold, "horizon_s": horizon, "comparison_mode": "shared_parameters", "input_tier": "uv_yaw", "test_example_hash": test_example_hash, "dataset_fingerprint": dataset_fingerprint})
                run_rows.extend(method_run_rows)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            for method, function in (() if args.core_methods else (("periodic_ekf", ekf), ("periodic_ukf", ukf))):
                config = choose_config(validation, function)
                outputs = [function(item, config) for item in test]
                row = {"split": split, "fold": fold, "horizon_s": horizon, "train_samples": len(train), "inner_validation_samples": len(validation), "comparison_mode": "independently_tuned", "input_tier": "uv_yaw", "test_example_hash": test_example_hash, "dataset_fingerprint": dataset_fingerprint}
                score, method_run_rows = score_predictions(test, outputs, method, config)
                row.update(score)
                rows.append(row)
                for method_run_row in method_run_rows:
                    method_run_row.update({"split": split, "fold": fold, "horizon_s": horizon, "comparison_mode": "independently_tuned", "input_tier": "uv_yaw", "test_example_hash": test_example_hash, "dataset_fingerprint": dataset_fingerprint})
                run_rows.extend(method_run_rows)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "periodic_filter_evaluation_rows.csv", rows)
    write_csv(output / "periodic_filter_run_rows.csv", run_rows)
    aggregate = []
    for key in sorted({(row["split"], row["horizon_s"], row["method"]) for row in rows}):
        selected = [row for row in rows if (row["split"], row["horizon_s"], row["method"]) == key]
        aggregate.append({
            "split": key[0], "horizon_s": key[1], "method": key[2], "folds": len(selected),
            "condition_equal_p95_deg_mean": float(np.mean([row["condition_equal_p95_deg"] for row in selected])),
            "worst_condition_p95_deg_mean": float(np.mean([row["worst_condition_p95_deg"] for row in selected])),
            "coverage_90_mean": float(np.mean([row["coverage_90"] for row in selected])),
        })
    summary = {
        "schema_version": 1,
        "kind": "periodic_state_filter_method_comparison",
        "oracle_identity_upper_bound": True,
        "inference_allowlist": ["timestamp_ns", "u_deg", "v_deg", "pnp_yaw_absolute_rad", "pnp_yaw_valid"],
        "truth_policy": "truth supplies future labels and oracle physical-slot/sample selection; no truth field is a filter input",
        "state": "[center_u, center_v, center_velocity_u, center_velocity_v, observation_phase, phase_rate, Fourier coefficients]",
        "fairness": "shared-parameter rows use identical state/Q/R/initialization; independently-tuned rows use identical inner-fold search budgets",
        "covariance_boundary": "coverage and NLL are diagnostic until run-grouped covariance calibration passes; point-error ranking is the primary evidence",
        "evaluation_points_per_run_horizon": MAX_EVAL_PER_RUN_HORIZON,
        "dataset_fingerprint": dataset_fingerprint,
        "core_methods": args.core_methods,
        "selected_split_families": args.split or "all",
        "run_metric_rows": len(run_rows),
        "sources": sources,
        "aggregate": aggregate,
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    }
    (output / "periodic_filter_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "retention_manifest.json").write_text(json.dumps({"classification": "long_term_private_evidence", "deletion_allowed": False, "sources": sources, "artifacts": ["periodic_filter_evaluation_rows.csv", "periodic_filter_run_rows.csv", "periodic_filter_summary.json"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
