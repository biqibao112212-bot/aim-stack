#!/usr/bin/env python3
"""Analyze observation assumptions and replay EKF/UKF/PF on one locked input.

The filter-family replay is an oracle-association upper bound: the saved
physical armor slot selects the measurement branch, but no numeric truth field
enters a filter. Future truth is read only after each posterior has been
recorded. All three filters share the Tongji 11D state, transition, Q/R,
initialization, observations, timestamps, and scoring anchors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCENARIOS = {
    "spin_8": "原地旋转 8 rad/s",
    "translate_1p5": "平移 1.5 m/s",
    "translate_1_spin_6": "平移 1 m/s＋旋转 6 rad/s",
}
METHODS = {
    "ekf": "EKF",
    "ukf": "UKF",
    "pf": "PF",
}
COLORS = {
    "ekf": "#0072B2",
    "ukf": "#D55E00",
    "pf": "#009E73",
    "truth": "#111111",
    "radial": "#D55E00",
    "tangent": "#0072B2",
    "vertical": "#009E73",
}
HORIZONS_S = (0.1, 0.2, 0.3, 0.5)
MAX_TRUTH_INTERPOLATION_GAP_S = 0.025
WARMUP_S = 2.0
PF_WARM_START_UPDATES = 20
SMALL_ARMOR_WIDTH_M = 0.135
ARMOR_HEIGHT_M = 0.055


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
        "font.size": 9.2,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.3,
        "legend.fontsize": 8.0,
        "legend.frameon": False,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "lines.linewidth": 1.8,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--combined-registry", required=True, type=Path)
    parser.add_argument("--pnp-evidence-registry", required=True, type=Path)
    parser.add_argument("--method-ranking", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--particles", type=int, default=2048)
    return parser.parse_args()


def wrap_rad(value: float | np.ndarray) -> float | np.ndarray:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"{path}:{line_number}: {error}") from error
    if len(rows) < 2:
        raise RuntimeError(f"not enough rows: {path}")
    timestamps = np.asarray([int(row["timestamp_ns"]) for row in rows])
    if np.any(np.diff(timestamps) <= 0):
        raise RuntimeError(f"timestamps are not strictly increasing: {path}")
    return rows


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Creator": "aim-stack"})
    svg = stem.with_suffix(".svg")
    fig.savefig(svg, metadata={"Creator": "aim-stack", "Date": None})
    lines = svg.read_text(encoding="utf-8").splitlines()
    svg.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(fig)


def observation(row: dict) -> tuple[np.ndarray, float, int] | None:
    pose = row.get("primary_pnp")
    truth = row.get("truth")
    if not isinstance(pose, dict) or not isinstance(truth, dict):
        return None
    xyz = pose.get("xyz_m")
    yaw = pose.get("yaw_rad")
    slot = truth.get("matched_armor_slot")
    if not isinstance(xyz, list) or len(xyz) != 3 or yaw is None:
        return None
    if not isinstance(slot, int) or not 0 <= slot < 4:
        return None
    values = np.asarray(xyz, dtype=float)
    if not np.all(np.isfinite(values)) or not math.isfinite(float(yaw)):
        return None
    return values, float(yaw), slot


def truth_armor(row: dict) -> np.ndarray | None:
    truth = row.get("truth")
    if not isinstance(truth, dict):
        return None
    value = truth.get("armor_m")
    if not isinstance(value, list) or len(value) != 3:
        return None
    result = np.asarray(value, dtype=float)
    return result if np.all(np.isfinite(result)) else None


def xyz_to_ypd(xyz: np.ndarray) -> np.ndarray:
    x, y, z = xyz
    horizontal = math.hypot(float(x), float(y))
    return np.asarray(
        [math.atan2(float(y), float(x)), math.atan2(float(z), horizontal), float(np.linalg.norm(xyz))]
    )


def xyz_to_ypd_jacobian(xyz: np.ndarray) -> np.ndarray:
    x, y, z = (float(value) for value in xyz)
    xy2 = max(x * x + y * y, 1e-12)
    horizontal = math.sqrt(xy2)
    range2 = max(xy2 + z * z, 1e-12)
    distance = math.sqrt(range2)
    return np.asarray(
        [
            [-y / xy2, x / xy2, 0.0],
            [-x * z / (horizontal * range2), -y * z / (horizontal * range2), horizontal / range2],
            [x / distance, y / distance, z / distance],
        ]
    )


def transition_matrix(dt: float) -> np.ndarray:
    result = np.eye(11)
    result[0, 1] = dt
    result[2, 3] = dt
    result[4, 5] = dt
    result[6, 7] = dt
    return result


def transition_state(state: np.ndarray, dt: float) -> np.ndarray:
    result = transition_matrix(dt) @ state
    result[6] = float(wrap_rad(result[6]))
    return result


def process_noise(dt: float) -> np.ndarray:
    # Exact non-outpost PWN coefficients from Tongji sp_vision_25.
    a = dt**4 / 4.0
    b = dt**3 / 2.0
    c = dt**2
    result = np.zeros((11, 11))
    for position, velocity in ((0, 1), (2, 3), (4, 5)):
        result[np.ix_([position, velocity], [position, velocity])] = 100.0 * np.asarray([[a, b], [b, c]])
    result[np.ix_([6, 7], [6, 7])] = 400.0 * np.asarray([[a, b], [b, c]])
    return result


def armor_xyz(state: np.ndarray, slot: int) -> np.ndarray:
    odd = slot % 2 == 1
    angle = float(wrap_rad(state[6] + slot * math.pi / 2.0))
    radius = float(state[8] + state[9]) if odd else float(state[8])
    return np.asarray(
        [
            state[0] - radius * math.cos(angle),
            state[2] - radius * math.sin(angle),
            state[4] + state[10] if odd else state[4],
        ]
    )


def measurement_model(state: np.ndarray, slot: int) -> np.ndarray:
    xyz = armor_xyz(state, slot)
    ypd = xyz_to_ypd(xyz)
    return np.asarray([ypd[0], ypd[1], ypd[2], float(wrap_rad(state[6] + slot * math.pi / 2.0))])


def measurement_jacobian(state: np.ndarray, slot: int) -> np.ndarray:
    odd = slot % 2 == 1
    angle = float(wrap_rad(state[6] + slot * math.pi / 2.0))
    radius = float(state[8] + state[9]) if odd else float(state[8])
    xyz_jacobian = np.zeros((4, 11))
    xyz_jacobian[0, 0] = 1.0
    xyz_jacobian[0, 6] = radius * math.sin(angle)
    xyz_jacobian[0, 8] = -math.cos(angle)
    xyz_jacobian[1, 2] = 1.0
    xyz_jacobian[1, 6] = -radius * math.cos(angle)
    xyz_jacobian[1, 8] = -math.sin(angle)
    xyz_jacobian[2, 4] = 1.0
    xyz_jacobian[3, 6] = 1.0
    if odd:
        xyz_jacobian[0, 9] = -math.cos(angle)
        xyz_jacobian[1, 9] = -math.sin(angle)
        xyz_jacobian[2, 10] = 1.0
    transform = np.zeros((4, 4))
    transform[:3, :3] = xyz_to_ypd_jacobian(armor_xyz(state, slot))
    transform[3, 3] = 1.0
    return transform @ xyz_jacobian


def measurement_noise(xyz: np.ndarray, yaw: float) -> np.ndarray:
    center_yaw = math.atan2(float(xyz[1]), float(xyz[0]))
    delta_angle = float(wrap_rad(yaw - center_yaw))
    distance = float(np.linalg.norm(xyz))
    diagonal = [0.004, 0.004, math.log(abs(delta_angle) + 1.0) + 1.0, math.log(abs(distance) + 1.0) / 200.0 + 0.09]
    return np.diag(diagonal)


def measurement_vector(xyz: np.ndarray, yaw: float) -> np.ndarray:
    ypd = xyz_to_ypd(xyz)
    return np.asarray([ypd[0], ypd[1], ypd[2], yaw])


def measurement_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = left - right
    result[0] = float(wrap_rad(result[0]))
    result[1] = float(wrap_rad(result[1]))
    result[3] = float(wrap_rad(result[3]))
    return result


def initial_state(xyz: np.ndarray, yaw: float, slot: int) -> tuple[np.ndarray, np.ndarray]:
    target_yaw = float(wrap_rad(yaw - slot * math.pi / 2.0))
    result = np.asarray(
        [
            xyz[0] + 0.2 * math.cos(yaw), 0.0,
            xyz[1] + 0.2 * math.sin(yaw), 0.0,
            xyz[2], 0.0,
            target_yaw, 0.0,
            0.2, 0.0, 0.0,
        ],
        dtype=float,
    )
    covariance = np.diag([1.0, 64.0, 1.0, 64.0, 1.0, 64.0, 0.4, 100.0, 1.0, 1.0, 1.0])
    return result, covariance


def stabilize(covariance: np.ndarray) -> np.ndarray:
    covariance = (covariance + covariance.T) * 0.5
    values, vectors = np.linalg.eigh(covariance)
    return vectors @ np.diag(np.maximum(values, 1e-10)) @ vectors.T


def sigma_points(state: np.ndarray, covariance: np.ndarray, alpha: float = 0.35) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dimension = len(state)
    beta = 2.0
    lam = alpha**2 * dimension - dimension
    scaled = stabilize((dimension + lam) * covariance)
    root = np.linalg.cholesky(scaled + np.eye(dimension) * 1e-10)
    points = [state]
    for index in range(dimension):
        points.extend((state + root[:, index], state - root[:, index]))
    mean_weights = np.full(2 * dimension + 1, 1.0 / (2.0 * (dimension + lam)))
    covariance_weights = mean_weights.copy()
    mean_weights[0] = lam / (dimension + lam)
    covariance_weights[0] = mean_weights[0] + 1.0 - alpha**2 + beta
    return np.asarray(points), mean_weights, covariance_weights


def circular_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return math.atan2(float(np.sum(weights * np.sin(values))), float(np.sum(weights * np.cos(values))))


def state_mean(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.sum(points * weights[:, None], axis=0)
    result[6] = circular_mean(points[:, 6], weights)
    return result


def state_deltas(points: np.ndarray, mean: np.ndarray) -> np.ndarray:
    result = points - mean
    result[:, 6] = wrap_rad(result[:, 6])
    return result


def measurement_mean(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.sum(points * weights[:, None], axis=0)
    for index in (0, 1, 3):
        result[index] = circular_mean(points[:, index], weights)
    return result


def measurement_deltas(points: np.ndarray, mean: np.ndarray) -> np.ndarray:
    result = points - mean
    for index in (0, 1, 3):
        result[:, index] = wrap_rad(result[:, index])
    return result


@dataclass
class GaussianPosterior:
    state: np.ndarray
    covariance: np.ndarray


def ekf_predict(posterior: GaussianPosterior, dt: float) -> GaussianPosterior:
    matrix = transition_matrix(dt)
    return GaussianPosterior(
        transition_state(posterior.state, dt),
        stabilize(matrix @ posterior.covariance @ matrix.T + process_noise(dt)),
    )


def ekf_update(posterior: GaussianPosterior, xyz: np.ndarray, yaw: float, slot: int) -> GaussianPosterior:
    measured = measurement_vector(xyz, yaw)
    predicted = measurement_model(posterior.state, slot)
    jacobian = measurement_jacobian(posterior.state, slot)
    noise = measurement_noise(xyz, yaw)
    innovation_covariance = jacobian @ posterior.covariance @ jacobian.T + noise
    gain = np.linalg.solve(innovation_covariance, jacobian @ posterior.covariance).T
    state = posterior.state + gain @ measurement_delta(measured, predicted)
    state[6] = float(wrap_rad(state[6]))
    identity = np.eye(len(state))
    update = identity - gain @ jacobian
    covariance = update @ posterior.covariance @ update.T + gain @ noise @ gain.T
    return GaussianPosterior(state, stabilize(covariance))


def ukf_predict(posterior: GaussianPosterior, dt: float) -> GaussianPosterior:
    points, mean_weights, covariance_weights = sigma_points(posterior.state, posterior.covariance)
    propagated = np.asarray([transition_state(point, dt) for point in points])
    state = state_mean(propagated, mean_weights)
    deltas = state_deltas(propagated, state)
    covariance = np.einsum("i,ij,ik->jk", covariance_weights, deltas, deltas) + process_noise(dt)
    return GaussianPosterior(state, stabilize(covariance))


def ukf_update(posterior: GaussianPosterior, xyz: np.ndarray, yaw: float, slot: int) -> GaussianPosterior:
    points, mean_weights, covariance_weights = sigma_points(posterior.state, posterior.covariance)
    state_delta = state_deltas(points, posterior.state)
    projected = np.asarray([measurement_model(point, slot) for point in points])
    predicted = measurement_mean(projected, mean_weights)
    projected_delta = measurement_deltas(projected, predicted)
    noise = measurement_noise(xyz, yaw)
    innovation_covariance = np.einsum("i,ij,ik->jk", covariance_weights, projected_delta, projected_delta) + noise
    cross = np.einsum("i,ij,ik->jk", covariance_weights, state_delta, projected_delta)
    gain = np.linalg.solve(innovation_covariance, cross.T).T
    state = posterior.state + gain @ measurement_delta(measurement_vector(xyz, yaw), predicted)
    state[6] = float(wrap_rad(state[6]))
    covariance = posterior.covariance - gain @ innovation_covariance @ gain.T
    return GaussianPosterior(state, stabilize(covariance))


def batch_transition(particles: np.ndarray, dt: float) -> None:
    particles[:, 0] += particles[:, 1] * dt
    particles[:, 2] += particles[:, 3] * dt
    particles[:, 4] += particles[:, 5] * dt
    particles[:, 6] = wrap_rad(particles[:, 6] + particles[:, 7] * dt)


def batch_measurement(particles: np.ndarray, slot: int) -> np.ndarray:
    odd = slot % 2 == 1
    angle = wrap_rad(particles[:, 6] + slot * math.pi / 2.0)
    radius = particles[:, 8] + particles[:, 9] if odd else particles[:, 8]
    x = particles[:, 0] - radius * np.cos(angle)
    y = particles[:, 2] - radius * np.sin(angle)
    z = particles[:, 4] + particles[:, 10] if odd else particles[:, 4]
    horizontal = np.hypot(x, y)
    return np.column_stack((np.arctan2(y, x), np.arctan2(z, horizontal), np.sqrt(x * x + y * y + z * z), angle))


def constrain_particles(particles: np.ndarray) -> None:
    particles[:, 1] = np.clip(particles[:, 1], -10.0, 10.0)
    particles[:, 3] = np.clip(particles[:, 3], -10.0, 10.0)
    particles[:, 5] = np.clip(particles[:, 5], -5.0, 5.0)
    particles[:, 7] = np.clip(particles[:, 7], -20.0, 20.0)
    particles[:, 8] = np.clip(particles[:, 8], 0.05, 0.5)
    odd_radius = np.clip(particles[:, 8] + particles[:, 9], 0.05, 0.5)
    particles[:, 9] = odd_radius - particles[:, 8]
    particles[:, 10] = np.clip(particles[:, 10], -0.5, 0.5)
    particles[:, 6] = wrap_rad(particles[:, 6])


@dataclass
class ParticlePosterior:
    particles: np.ndarray
    weights: np.ndarray
    rng: np.random.Generator


def initialize_particles(state: np.ndarray, covariance: np.ndarray, count: int, seed: int) -> ParticlePosterior:
    rng = np.random.default_rng(seed)
    root = np.linalg.cholesky(stabilize(covariance) + np.eye(len(state)) * 1e-10)
    particles = state + rng.standard_normal((count, len(state))) @ root.T
    constrain_particles(particles)
    return ParticlePosterior(particles, np.full(count, 1.0 / count), rng)


def pf_predict(posterior: ParticlePosterior, dt: float) -> ParticlePosterior:
    batch_transition(posterior.particles, dt)
    noise = process_noise(dt)
    active = np.arange(8)
    root = np.linalg.cholesky(stabilize(noise[np.ix_(active, active)]) + np.eye(len(active)) * 1e-12)
    posterior.particles[:, active] += posterior.rng.standard_normal((len(posterior.particles), len(active))) @ root.T
    constrain_particles(posterior.particles)
    return posterior


def systematic_resample(posterior: ParticlePosterior) -> None:
    count = len(posterior.weights)
    positions = (posterior.rng.random() + np.arange(count)) / count
    indices = np.searchsorted(np.cumsum(posterior.weights), positions)
    posterior.particles = posterior.particles[indices]
    posterior.weights.fill(1.0 / count)


def pf_update(posterior: ParticlePosterior, xyz: np.ndarray, yaw: float, slot: int) -> ParticlePosterior:
    measured = measurement_vector(xyz, yaw)
    projected = batch_measurement(posterior.particles, slot)
    residual = measured - projected
    for index in (0, 1, 3):
        residual[:, index] = wrap_rad(residual[:, index])
    variance = np.diag(measurement_noise(xyz, yaw))
    log_weights = np.log(posterior.weights + 1e-300) - 0.5 * np.sum(residual * residual / variance, axis=1)
    log_weights -= float(np.max(log_weights))
    posterior.weights = np.exp(log_weights)
    posterior.weights /= float(np.sum(posterior.weights))
    effective = 1.0 / float(np.sum(posterior.weights**2))
    if effective < len(posterior.weights) / 2.0:
        systematic_resample(posterior)
    return posterior


def particle_state(posterior: ParticlePosterior) -> np.ndarray:
    return state_mean(posterior.particles, posterior.weights)


def run_particle_filter(
    rows: list[dict], particle_count: int, scenario: str
) -> list[np.ndarray | None]:
    """Use a short observation-only EKF warm start before particle sampling.

    Drawing the original 11D P0 directly is a poor finite-particle proposal:
    most samples spend their budget on physically impossible radius/velocity
    combinations. Twenty ordinary measurement updates concentrate that same
    prior before the bootstrap PF begins. The scored window starts at 2 s, well
    after this causal warm start.
    """
    snapshots: list[np.ndarray | None] = []
    warm: GaussianPosterior | None = None
    posterior: ParticlePosterior | None = None
    previous_ns: int | None = None
    update_count = 0
    seed = int.from_bytes(
        hashlib.sha256(f"{scenario}:{particle_count}".encode()).digest()[:8],
        "little",
    )

    for row in rows:
        timestamp_ns = int(row["timestamp_ns"])
        current = observation(row)
        if warm is None and posterior is None:
            if current is None:
                snapshots.append(None)
                continue
            xyz, yaw, slot = current
            state, covariance = initial_state(xyz, yaw, slot)
            warm = ekf_update(GaussianPosterior(state, covariance), xyz, yaw, slot)
            update_count = 1
            snapshots.append(warm.state.copy())
            previous_ns = timestamp_ns
            continue

        if previous_ns is None:
            raise RuntimeError("particle-filter timestamp state is inconsistent")
        dt = max((timestamp_ns - previous_ns) * 1e-9, 1e-6)
        previous_ns = timestamp_ns

        if posterior is None:
            if warm is None:
                raise RuntimeError("particle-filter warm start is absent")
            warm = ekf_predict(warm, dt)
            if current is not None:
                warm = ekf_update(warm, *current)
                update_count += 1
            if update_count >= PF_WARM_START_UPDATES:
                posterior = initialize_particles(
                    warm.state, warm.covariance, particle_count, seed
                )
                warm = None
                snapshots.append(particle_state(posterior))
            else:
                snapshots.append(warm.state.copy())
            continue

        posterior = pf_predict(posterior, dt)
        if current is not None:
            posterior = pf_update(posterior, *current)
        snapshots.append(particle_state(posterior))
    return snapshots


def run_filter(rows: list[dict], method: str, particle_count: int, scenario: str) -> list[np.ndarray | None]:
    if method == "pf":
        return run_particle_filter(rows, particle_count, scenario)

    snapshots: list[np.ndarray | None] = []
    gaussian: GaussianPosterior | None = None
    previous_ns: int | None = None

    for row in rows:
        timestamp_ns = int(row["timestamp_ns"])
        current = observation(row)
        if gaussian is None:
            if current is None:
                snapshots.append(None)
                continue
            xyz, yaw, slot = current
            state, covariance = initial_state(xyz, yaw, slot)
            gaussian = GaussianPosterior(state, covariance)
            update = ekf_update if method == "ekf" else ukf_update
            gaussian = update(gaussian, xyz, yaw, slot)
            snapshots.append(gaussian.state.copy())
            previous_ns = timestamp_ns
            continue

        if previous_ns is None:
            raise RuntimeError("filter timestamp state is inconsistent")
        dt = max((timestamp_ns - previous_ns) * 1e-9, 1e-6)
        previous_ns = timestamp_ns
        predict = ekf_predict if method == "ekf" else ukf_predict
        update = ekf_update if method == "ekf" else ukf_update
        gaussian = predict(gaussian, dt)
        if current is not None:
            gaussian = update(gaussian, *current)
        snapshots.append(gaussian.state.copy())
    return snapshots


def rotate_z(points: np.ndarray, yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    result = np.array(points, dtype=float, copy=True)
    result[..., 0] = cosine * points[..., 0] - sine * points[..., 1]
    result[..., 1] = sine * points[..., 0] + cosine * points[..., 1]
    return result


def truth_local_offsets(rows: list[dict]) -> np.ndarray:
    samples: dict[int, list[np.ndarray]] = {slot: [] for slot in range(4)}
    for row in rows:
        truth = row.get("truth")
        if not isinstance(truth, dict):
            continue
        slot = truth.get("matched_armor_slot")
        armor = truth.get("armor_m")
        center = truth.get("center_m")
        yaw = truth.get("yaw_rad")
        if not isinstance(slot, int) or slot not in samples or not isinstance(armor, list) or not isinstance(center, list) or yaw is None:
            continue
        samples[slot].append(rotate_z(np.asarray(armor) - np.asarray(center), -float(yaw)))
    offsets = []
    for slot in range(4):
        if not samples[slot]:
            raise RuntimeError(f"truth slot {slot} has no samples")
        offsets.append(np.median(np.asarray(samples[slot]), axis=0))
    return np.asarray(offsets)


def interpolate_truth(
    times_s: np.ndarray,
    centers: np.ndarray,
    yaw_unwrapped: np.ndarray,
    query_s: float,
) -> tuple[np.ndarray, float] | None:
    right = int(np.searchsorted(times_s, query_s, side="left"))
    if right < len(times_s) and abs(float(times_s[right] - query_s)) < 1e-9:
        return centers[right].copy(), float(yaw_unwrapped[right])
    if right == 0 or right >= len(times_s):
        return None
    left = right - 1
    gap = float(times_s[right] - times_s[left])
    if gap <= 0.0 or gap > MAX_TRUTH_INTERPOLATION_GAP_S:
        return None
    alpha = float((query_s - times_s[left]) / gap)
    return centers[left] + alpha * (centers[right] - centers[left]), float(
        yaw_unwrapped[left] + alpha * (yaw_unwrapped[right] - yaw_unwrapped[left])
    )


def percentile_summary(values: list[float]) -> dict:
    data = np.asarray(values, dtype=float)
    return {
        "n": int(len(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
    }


def score_filter(rows: list[dict], snapshots: list[np.ndarray | None]) -> dict:
    timestamp_ns = np.asarray([int(row["timestamp_ns"]) for row in rows], dtype=np.int64)
    times_s = (timestamp_ns - timestamp_ns[0]).astype(float) * 1e-9
    centers = np.asarray([row["truth"]["center_m"] for row in rows], dtype=float)
    yaw_unwrapped = np.unwrap(np.asarray([row["truth"]["yaw_rad"] for row in rows], dtype=float))
    offsets = truth_local_offsets(rows)
    buckets = {
        horizon: {"normal_radial_abs_m": [], "tangential_abs_m": [], "vertical_abs_m": [], "error_3d_m": [], "small_armor_window": []}
        for horizon in HORIZONS_S
    }
    for index, (row, state) in enumerate(zip(rows, snapshots)):
        current = observation(row)
        if state is None or current is None or times_s[index] < WARMUP_S:
            continue
        slot = current[2]
        for horizon in HORIZONS_S:
            future = interpolate_truth(times_s, centers, yaw_unwrapped, float(times_s[index] + horizon))
            if future is None:
                continue
            future_center, future_yaw = future
            actual = future_center + rotate_z(offsets[slot], future_yaw)
            predicted = armor_xyz(transition_state(state, horizon), slot)
            error = predicted - actual
            normal = actual - future_center
            normal[2] = 0.0
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-9:
                continue
            normal /= norm
            tangent = np.asarray([-normal[1], normal[0], 0.0])
            normal_abs = abs(float(error @ normal))
            tangent_abs = abs(float(error @ tangent))
            vertical_abs = abs(float(error[2]))
            bucket = buckets[horizon]
            bucket["normal_radial_abs_m"].append(normal_abs)
            bucket["tangential_abs_m"].append(tangent_abs)
            bucket["vertical_abs_m"].append(vertical_abs)
            bucket["error_3d_m"].append(float(np.linalg.norm(error)))
            bucket["small_armor_window"].append(
                tangent_abs <= SMALL_ARMOR_WIDTH_M / 2.0 and vertical_abs <= ARMOR_HEIGHT_M / 2.0
            )
    result = {}
    for horizon, bucket in buckets.items():
        item = {
            key: percentile_summary(values)
            for key, values in bucket.items()
            if key != "small_armor_window"
        }
        item["small_armor_window"] = {
            "n": len(bucket["small_armor_window"]),
            "coverage": float(np.mean(bucket["small_armor_window"])) if bucket["small_armor_window"] else None,
        }
        result[str(int(round(horizon * 1000)))] = item
    return result


def contiguous_acf(values: np.ndarray, times_s: np.ndarray, max_lag: int = 20) -> list[float | None]:
    result: list[float | None] = []
    for lag in range(1, max_lag + 1):
        valid = (times_s[lag:] - times_s[:-lag]) <= MAX_TRUTH_INTERPOLATION_GAP_S * lag
        left, right = values[:-lag][valid], values[lag:][valid]
        if len(left) < 3 or float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
            result.append(None)
        else:
            result.append(float(np.corrcoef(left, right)[0, 1]))
    return result


def observation_distribution(rows: list[dict]) -> tuple[dict, dict[str, np.ndarray]]:
    good = [(row, observation(row), truth_armor(row)) for row in rows]
    good = [(row, obs, truth) for row, obs, truth in good if obs is not None and truth is not None]
    times_s = np.asarray([int(row["timestamp_ns"]) for row, _, _ in good], dtype=np.int64) * 1e-9
    predicted = np.asarray([obs[0] for _, obs, _ in good])
    actual = np.asarray([truth for _, _, truth in good])
    error = predicted - actual
    line_of_sight = actual / np.linalg.norm(actual, axis=1, keepdims=True)
    radial = np.sum(error * line_of_sight, axis=1)
    tangent_axes = np.column_stack((-line_of_sight[:, 1], line_of_sight[:, 0], np.zeros(len(line_of_sight))))
    tangent_axes /= np.linalg.norm(tangent_axes, axis=1, keepdims=True)
    tangent = np.sum(error * tangent_axes, axis=1)
    vertical = error[:, 2]
    intervals_ms = np.diff(times_s) * 1000.0
    components = {"normal_radial": radial, "tangential": tangent, "vertical": vertical}
    summary = {
        "exposures": len(rows),
        "matched_pnp": len(good),
        "matched_fraction": len(good) / len(rows),
        "matched_interval_ms": {
            "p50": float(np.percentile(intervals_ms, 50)),
            "p95": float(np.percentile(intervals_ms, 95)),
            "p99": float(np.percentile(intervals_ms, 99)),
            "max": float(np.max(intervals_ms)),
        },
        "components": {},
    }
    for name, values in components.items():
        absolute = np.abs(values)
        p50, p95 = np.percentile(absolute, [50, 95])
        summary["components"][name] = {
            "signed_mean_m": float(np.mean(values)),
            "signed_median_m": float(np.median(values)),
            "abs_p50_m": float(p50),
            "abs_p95_m": float(p95),
            "p95_p50_ratio": float(p95 / max(p50, 1e-12)),
            "lag1_autocorrelation": contiguous_acf(values, times_s, 1)[0],
        }
    arrays = {**components, "times_s": times_s, "intervals_ms": intervals_ms}
    return summary, arrays


def robust_standardized_absolute(values: np.ndarray) -> np.ndarray:
    center = float(np.median(values))
    scale = float(np.median(np.abs(values - center)) / 0.6744897501960817)
    return np.abs(values - center) / max(scale, 1e-12)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def load_historical_method_screening(method_ranking: Path, evidence_registry: Path) -> dict:
    with method_ranking.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    registry = json.loads(evidence_registry.read_text(encoding="utf-8"))
    authority = registry["authorities"]["accepted_120_run_observation_matrix"]
    selected = {
        "history_linear_correction": ("ridge_uv_residual", "common_uv"),
        "periodic_ekf": ("periodic_ekf_shared", "uv_yaw"),
        "periodic_ukf": ("periodic_ukf_shared", "uv_yaw"),
    }
    methods: dict[str, dict] = {}
    for public_name, (method, input_tier) in selected.items():
        values = {}
        for horizon_ms in (50, 100, 200):
            matches = [
                row
                for row in rows
                if row.get("split") == "leave_distance_out"
                and row.get("method") == method
                and row.get("input_tier") == input_tier
                and abs(float(row["horizon_s"]) * 1000.0 - horizon_ms) < 1e-9
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"historical method row mismatch: {method}/{input_tier}/{horizon_ms}: {len(matches)}"
                )
            values[str(horizon_ms)] = float(matches[0]["condition_equal_p95_deg"])
        methods[public_name] = {
            "source_method": method,
            "input_tier": input_tier,
            "condition_equal_angular_error_p95_deg": values,
        }
    return {
        "purpose": "historical candidate screening; separate from the current 1.4.0 3D replay",
        "collection_runs": int(authority["runs"]),
        "usable_observation_histories": int(authority["usable_observation_histories"]),
        "split": "leave_distance_out",
        "horizons_ms": [50, 100, 200],
        "metric": "condition_equal_angular_error_p95_deg",
        "methods": methods,
    }


def plot_observation_assumptions(distributions: dict[str, dict], arrays: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    ax = axes[0, 0]
    positions = np.arange(len(SCENARIOS))
    offsets = {"normal_radial": -0.18, "tangential": 0.18}
    for component, label, marker in (
        ("normal_radial", "沿视线（深度）", "o"),
        ("tangential", "水平切向", "s"),
    ):
        p50 = np.asarray([distributions[name]["components"][component]["abs_p50_m"] for name in SCENARIOS]) * 1000.0
        p95 = np.asarray([distributions[name]["components"][component]["abs_p95_m"] for name in SCENARIOS]) * 1000.0
        color = COLORS["radial" if component == "normal_radial" else "tangent"]
        ax.vlines(positions + offsets[component], p50, p95, color=color, linewidth=2.2)
        ax.scatter(positions + offsets[component], p50, color=color, marker=marker, s=34, label=f"{label} P50")
        ax.scatter(positions + offsets[component], p95, facecolors="none", edgecolors=color, marker=marker, s=42, label=f"{label} P95")
    ax.set_yscale("log")
    ax.set_xticks(positions, ["原地旋转", "平移", "平移＋旋转"])
    ax.set_ylabel("绝对误差（毫米，对数刻度）")
    ax.set_title("A  误差大小取决于方向")
    ax.legend(ncol=2, fontsize=7.2)

    ax = axes[0, 1]
    width = 0.34
    radial_ratios = np.asarray(
        [distributions[name]["components"]["normal_radial"]["p95_p50_ratio"] for name in SCENARIOS]
    )
    tangent_ratios = np.asarray(
        [distributions[name]["components"]["tangential"]["p95_p50_ratio"] for name in SCENARIOS]
    )
    radial_bars = ax.bar(
        positions - width / 2,
        radial_ratios,
        width,
        color=COLORS["radial"],
        label="沿视线（深度）",
    )
    tangent_bars = ax.bar(
        positions + width / 2,
        tangent_ratios,
        width,
        color=COLORS["tangent"],
        label="水平切向",
    )
    ax.axhline(2.91, color="#555555", linestyle="--", linewidth=1.1, label="零均值高斯参考≈2.9")
    ax.bar_label(radial_bars, fmt="%.1f×", fontsize=7.2, padding=2)
    ax.bar_label(tangent_bars, fmt="%.1f×", fontsize=7.2, padding=2)
    ax.set_xticks(positions, ["原地旋转", "平移", "平移＋旋转"])
    ax.set_ylabel("P95 ÷ P50（倍）")
    ax.set_ylim(0, 29)
    ax.set_title("B  少数帧会比典型帧差很多")
    ax.legend(fontsize=7.2)

    ax = axes[1, 0]
    radial_lag1 = np.asarray(
        [distributions[name]["components"]["normal_radial"]["lag1_autocorrelation"] for name in SCENARIOS]
    )
    tangent_lag1 = np.asarray(
        [distributions[name]["components"]["tangential"]["lag1_autocorrelation"] for name in SCENARIOS]
    )
    radial_bars = ax.bar(
        positions - width / 2,
        radial_lag1,
        width,
        color=COLORS["radial"],
        label="沿视线（深度）",
    )
    tangent_bars = ax.bar(
        positions + width / 2,
        tangent_lag1,
        width,
        color=COLORS["tangent"],
        label="水平切向",
    )
    ax.axhline(0.0, color="#555555", linewidth=0.9)
    ax.bar_label(radial_bars, fmt="%.2f", fontsize=7.2, padding=2)
    ax.bar_label(tangent_bars, fmt="%.2f", fontsize=7.2, padding=2)
    ax.set_xticks(positions, ["原地旋转", "平移", "平移＋旋转"])
    ax.set_ylabel("相邻两次误差的相关系数")
    ax.set_ylim(0, 0.7)
    ax.set_title("C  上一帧的误差会延续到下一帧")
    ax.legend(fontsize=7.2)

    ax = axes[1, 1]
    p50 = np.asarray([distributions[name]["matched_interval_ms"]["p50"] for name in SCENARIOS])
    p95 = np.asarray([distributions[name]["matched_interval_ms"]["p95"] for name in SCENARIOS])
    maximum = np.asarray([distributions[name]["matched_interval_ms"]["max"] for name in SCENARIOS])
    ax.vlines(positions, p50, p95, color=COLORS["ekf"], linewidth=2.4)
    ax.scatter(positions, p50, color=COLORS["ekf"], marker="o", s=35, label="P50：典型间隔")
    ax.scatter(positions, p95, facecolors="none", edgecolors=COLORS["ekf"], marker="o", s=44, label="P95：较长间隔")
    ax.scatter(positions, maximum, color=COLORS["pf"], marker="x", s=44, label="最长间隔")
    for index, value in enumerate(maximum):
        ax.annotate(f"{value:.1f}", (positions[index], value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7.2)
    ax.set_xticks(positions, ["原地旋转", "平移", "平移＋旋转"])
    ax.set_ylabel("相邻成功 PnP 的间隔（毫秒）")
    ax.set_ylim(0, 60)
    ax.set_title("D  成功观测不是等间隔到达")
    ax.legend(fontsize=7.2)

    fig.suptitle("三组 20 秒记录：PnP 观测误差有哪些特征", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output / "observation_assumption_check")


def plot_filter_family(metrics: dict, output: Path, particle_count: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), sharey=True)
    horizon_ms = np.asarray([100, 200, 300, 500])
    x_offsets = {"ekf": -4, "ukf": 4, "pf": 0}
    markers = {"ekf": "o", "ukf": "s", "pf": "^"}
    labels = {"ekf": "EKF", "ukf": "UKF", "pf": f"粒子滤波（{particle_count} 个粒子）"}
    for ax, (scenario, title) in zip(axes, SCENARIOS.items()):
        for method in METHODS:
            values = [metrics[scenario][method][str(value)]["error_3d_m"]["p95"] * 100.0 for value in horizon_ms]
            ax.plot(
                horizon_ms + x_offsets[method],
                values,
                marker=markers[method],
                markersize=5.5,
                color=COLORS[method],
                linestyle="--" if method == "ukf" else "-",
                label=labels[method],
                zorder=3 if method == "ukf" else 2,
            )
        ax.set_title(title)
        ax.set_xlabel("预测时长（毫秒）")
        ax.set_xticks(horizon_ms)
    maximum_difference_mm = max(
        abs(
            metrics[scenario]["ekf"][str(horizon)]["error_3d_m"]["p95"]
            - metrics[scenario]["ukf"][str(horizon)]["error_3d_m"]["p95"]
        )
        * 1000.0
        for scenario in SCENARIOS
        for horizon in horizon_ms
    )
    axes[0].text(
        0.03,
        0.94,
        f"EKF 与 UKF 最大差值：{maximum_difference_mm:.1f} 毫米",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8.0,
        color="#444444",
    )
    axes[0].set_ylabel("未来装甲板三维位置误差 P95（厘米）")
    axes[-1].legend(loc="upper left")
    fig.suptitle("同一套 11 维模型和观测：EKF、UKF 与粒子滤波离线回放", fontsize=12.2, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output / "filter_family_replay")


def plot_structure_aware(registry: dict, output: Path) -> None:
    values = registry["sealed_test_cross_depth_p95_mm"]["constant_twist"]
    horizons = np.asarray([50, 100, 200])
    selected = (
        ("same_slot_world_cv", "同一装甲板恒速外推", "#7A7A7A"),
        ("v1_isotropic_single_window", "短窗口、各方向同权", "#CC79A7"),
        ("los_memory31", "按视线方向分权＋跨窗口角速度", "#0072B2"),
        ("direct_joint_omega_phase", "四块装甲板联合轨迹", "#009E73"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
    for key, label, color in selected:
        axes[0].plot(horizons, values[key], marker="o", color=color, label=label)
    axes[0].set_xlabel("预测时长（毫秒）")
    axes[0].set_ylabel("横向位置误差 P95（毫米）")
    axes[0].set_title("A  全部候选方法")
    axes[0].set_xticks(horizons)
    axes[0].legend(fontsize=7.5)
    for key, label, color in selected[-2:]:
        axes[1].plot(horizons, values[key], marker="o", color=color, label=label)
    axes[1].axhline(55.0, color="#D55E00", linestyle="--", linewidth=1.2, label="55 毫米历史诊断线")
    axes[1].set_ylim(0, 70)
    axes[1].set_xlabel("预测时长（毫秒）")
    axes[1].set_ylabel("横向位置误差 P95（毫米）")
    axes[1].set_title("B  两种结构化方法的细节")
    axes[1].set_xticks(horizons)
    axes[1].legend(fontsize=7.5)
    fig.suptitle("历史组合运动实验：逐步加入误差方向和刚体结构", fontsize=12.2, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output / "structure_aware_methods")


def validate_results(distributions: dict, filter_metrics: dict) -> None:
    for scenario in SCENARIOS:
        distribution = distributions[scenario]
        if not 0 < distribution["matched_pnp"] <= distribution["exposures"]:
            raise RuntimeError(f"invalid observation accounting: {scenario}")
        for horizon in (100, 200, 300, 500):
            sample_counts = set()
            for method in METHODS:
                item = filter_metrics[scenario][method][str(horizon)]
                for metric in ("normal_radial_abs_m", "tangential_abs_m", "vertical_abs_m", "error_3d_m"):
                    values = item[metric]
                    if values["n"] < 1000 or not 0.0 <= values["p50"] <= values["p95"]:
                        raise RuntimeError(
                            f"invalid metric: {scenario}/{method}/{horizon}/{metric}"
                        )
                sample_counts.add(item["error_3d_m"]["n"])
            if len(sample_counts) != 1:
                raise RuntimeError(
                    f"methods do not share scoring anchors: {scenario}/{horizon} {sample_counts}"
                )


def write_report(
    output: Path,
    distributions: dict,
    filter_metrics: dict,
    historical_screening: dict,
    particles: int,
) -> None:
    lines = [
        "# 固定输入滤波器研究报告",
        "",
        "## 当前 1.4.0 回放数据",
        "",
        f"- 三种工况，每种约 20 秒；粒子滤波使用 {particles} 个粒子。",
        "- 数值真值不进入滤波器；保存的物理装甲板槽位只选择观测分支。",
        "- 未来真值只在 100/200/300/500 ms 预测完成后用于评分。",
        "",
        "| 工况 | 曝光数 | 匹配 PnP | 匹配率 | 沿视线绝对误差 P50/P95 | 水平切向绝对误差 P50/P95 | 沿视线相邻相关系数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, label in SCENARIOS.items():
        item = distributions[name]
        radial = item["components"]["normal_radial"]
        tangent = item["components"]["tangential"]
        lines.append(
            f"| {label} | {item['exposures']} | {item['matched_pnp']} | {item['matched_fraction']:.3f} | "
            f"{radial['abs_p50_m'] * 1000:.1f}/{radial['abs_p95_m'] * 1000:.1f} mm | "
            f"{tangent['abs_p50_m'] * 1000:.1f}/{tangent['abs_p95_m'] * 1000:.1f} mm | "
            f"{radial['lag1_autocorrelation']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## EKF、UKF 与粒子滤波回放",
            "",
            "下表均为未来装甲板三维位置误差 P95，单位厘米。",
            "",
            "| 工况 | 预测时长 | EKF | UKF | 粒子滤波 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, label in SCENARIOS.items():
        for horizon in (100, 200, 300, 500):
            values = [
                filter_metrics[name][method][str(horizon)]["error_3d_m"]["p95"] * 100
                for method in METHODS
            ]
            lines.append(
                f"| {label} | {horizon} ms | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} |"
            )
    lines.extend(
        [
            "",
            "## 历史方法筛选数据",
            "",
            f"- 原始采集共有 {historical_screening['collection_runs']} 轮，其中 {historical_screening['usable_observation_histories']} 轮形成可评分历史。",
            "- 该实验预测相机射线 u/v，所以指标是条件等权角误差 P95（度），预测时长为 50/100/200 ms。",
            "- 它用于筛选后续候选方法，不与上面的 1.4.0 三维位置误差作数值比较。",
            "",
            "| 历史候选 | 输入 | 50 ms | 100 ms | 200 ms |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    history_labels = {
        "history_linear_correction": "恒速外推＋历史窗口线性校正",
        "periodic_ekf": "周期状态 EKF",
        "periodic_ukf": "周期状态 UKF",
    }
    input_labels = {"common_uv": "u/v", "uv_yaw": "u/v＋yaw"}
    for name in ("history_linear_correction", "periodic_ekf", "periodic_ukf"):
        item = historical_screening["methods"][name]
        values = item["condition_equal_angular_error_p95_deg"]
        lines.append(
            f"| {history_labels[name]} | {input_labels[item['input_tier']]} | "
            f"{values['50']:.3f}° | {values['100']:.3f}° | {values['200']:.3f}° |"
        )
    lines.extend(
        [
            "",
            "## 结果使用范围",
            "",
            "- 当前回放保留每个曝光时间戳和每次 PnP 缺失。",
            "- 当前回放的物理槽位关联隔离了连续估计问题；在线关联仍需单独评价。",
            f"- 粒子滤波是 {particles} 粒子的 bootstrap 实现，先用 20 次 EKF 更新收窄 11 维先验。",
            "- 第三张图来自单独封存的 combined-04 历史实验，指标是 tracker 横向位置误差 P95。",
            "",
        ]
    )
    (output / "RESEARCH_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.particles < 256:
        raise ValueError("particle count must be at least 256")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_root = args.raw_root.resolve()
    method_ranking = args.method_ranking.resolve()
    pnp_evidence_registry = args.pnp_evidence_registry.resolve()
    rows_by_scenario = {
        name: load_jsonl(raw_root / "raw" / f"{name}.jsonl") for name in SCENARIOS
    }

    distributions = {}
    distribution_arrays = {}
    for name, rows in rows_by_scenario.items():
        distributions[name], distribution_arrays[name] = observation_distribution(rows)
    plot_observation_assumptions(distributions, distribution_arrays, output)

    filter_metrics: dict[str, dict] = {}
    for scenario, rows in rows_by_scenario.items():
        filter_metrics[scenario] = {}
        for method in METHODS:
            print(f"replay {scenario} {method}", flush=True)
            snapshots = run_filter(rows, method, args.particles, scenario)
            filter_metrics[scenario][method] = score_filter(rows, snapshots)
    validate_results(distributions, filter_metrics)
    plot_filter_family(filter_metrics, output, args.particles)

    registry = json.loads(args.combined_registry.read_text(encoding="utf-8"))
    plot_structure_aware(registry, output)
    historical_screening = load_historical_method_screening(
        method_ranking,
        pnp_evidence_registry,
    )

    source_files = {name: raw_root / "raw" / f"{name}.jsonl" for name in SCENARIOS}
    provenance = {
        "schema": "aim-stack.filter-direction-analysis/v2",
        "raw_root": str(raw_root),
        "oracle_association_upper_bound": True,
        "truth_filter_input_policy": "numeric truth excluded; physical slot selects measurement branch only",
        "future_truth_policy": "post-hoc scoring only",
        "filters": "same Tongji 11D state, transition, Q/R, initialization, observations and timestamps",
        "particles": args.particles,
        "pf_warm_start_updates": PF_WARM_START_UPDATES,
        "horizons_ms": [int(round(value * 1000)) for value in HORIZONS_S],
        "warmup_s": WARMUP_S,
        "sources": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in source_files.items()
        },
        "combined_registry": {
            "path": str(args.combined_registry.resolve()),
            "sha256": sha256(args.combined_registry.resolve()),
            "note": "separate sealed experiment contract; used only for the structure-aware figure",
        },
        "historical_method_screening": {
            "method_ranking_path": str(method_ranking),
            "method_ranking_sha256": sha256(method_ranking),
            "pnp_evidence_registry_path": str(pnp_evidence_registry),
            "pnp_evidence_registry_sha256": sha256(pnp_evidence_registry),
            "note": "separate 2026-08-09 angular-domain candidate screening",
        },
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    }
    (output / "observation_distribution_summary.json").write_text(
        json.dumps(distributions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "filter_family_summary.json").write_text(
        json.dumps(filter_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "historical_method_screening_summary.json").write_text(
        json.dumps(historical_screening, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(output, distributions, filter_metrics, historical_screening, args.particles)
    artifact_names = [
        "RESEARCH_REPORT.md",
        "observation_distribution_summary.json",
        "filter_family_summary.json",
        "historical_method_screening_summary.json",
        "provenance.json",
        *[
            f"{stem}.{suffix}"
            for stem in (
                "observation_assumption_check",
                "filter_family_replay",
                "structure_aware_methods",
            )
            for suffix in ("png", "svg", "pdf")
        ],
    ]
    retention = {
        "classification": "public_reproducible_research_artifact",
        "deletion_allowed": False,
        "protected_sources_copied_to_git": False,
        "protected_source_policy": "raw JSONL remains under runtime; only hashes and aggregate results are retained",
        "artifacts": {
            name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
            for name in artifact_names
        },
    }
    (output / "retention_manifest.json").write_text(
        json.dumps(retention, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
