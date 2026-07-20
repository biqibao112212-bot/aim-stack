"""Permutation-invariant offline baselines for the Stage-3 gate."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

PERMUTATIONS = tuple(itertools.permutations(range(4)))


def load_geometry_template(path: str | Path) -> dict[str, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    armors = sorted(payload["armors"], key=lambda item: int(item["relative_slot"]))
    if [int(item["relative_slot"]) for item in armors] != [0, 1, 2, 3]:
        raise ValueError("geometry template slots must be 0..3")
    return {
        "position": np.asarray([item["relative_position_m"] for item in armors], dtype=np.float64),
        "normal": np.asarray([item["outward_normal"] for item in armors], dtype=np.float64),
        "yaw": np.asarray([item["relative_yaw_rad"] for item in armors], dtype=np.float64),
    }


def _wrap(value: float | np.ndarray) -> float | np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def _rotate_z(values: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return values @ rotation.T


def _frame_state_candidates(
    frame: np.ndarray, mask: np.ndarray, geometry: dict[str, np.ndarray]
) -> list[tuple[float, np.ndarray, float]]:
    visible = np.flatnonzero(mask)
    if len(visible) == 0:
        return []
    positions = frame[visible, :3].astype(np.float64)
    yaws = np.arctan2(frame[visible, 3], frame[visible, 4]).astype(np.float64)
    candidates: list[tuple[float, np.ndarray, float]] = []
    for slots in itertools.permutations(range(4), len(visible)):
        slot_index = np.asarray(slots, dtype=np.int64)
        theta_values = _wrap(yaws - geometry["yaw"][slot_index])
        theta = math.atan2(float(np.sin(theta_values).mean()), float(np.cos(theta_values).mean()))
        offsets = _rotate_z(geometry["position"][slot_index], theta)
        centers = positions - offsets
        center = centers.mean(axis=0)
        position_cost = float(np.square(centers - center).sum(axis=1).mean())
        angle_cost = float(np.square(_wrap(theta_values - theta)).mean())
        candidates.append((position_cost + 0.05 * angle_cost, center, theta))
    candidates.sort(key=lambda item: item[0])
    return candidates


def _fit_rigid_history(
    obs: np.ndarray, obs_mask: np.ndarray, event_mask: np.ndarray,
    event_time_s: np.ndarray,
    geometry: dict[str, np.ndarray], max_frames: int = 40,
) -> tuple[np.ndarray, float, np.ndarray, float, int] | None:
    frame_indices = np.flatnonzero(event_mask & obs_mask.any(axis=1))[-max_frames:]
    if len(frame_indices) == 0:
        return None
    candidate_sets = [
        _frame_state_candidates(obs[index], obs_mask[index], geometry)
        for index in frame_indices
    ]
    if any(not values for values in candidate_sets):
        return None
    costs = np.asarray([item[0] for item in candidate_sets[0]], dtype=np.float64)
    paths = [[index] for index in range(len(candidate_sets[0]))]
    for frame_number in range(1, len(candidate_sets)):
        previous = candidate_sets[frame_number - 1]
        current = candidate_sets[frame_number]
        next_costs = np.full((len(current),), np.inf)
        next_paths: list[list[int]] = [[] for _ in current]
        for current_index, (emission, center, theta) in enumerate(current):
            for previous_index, (_, previous_center, previous_theta) in enumerate(previous):
                transition = float(np.square(center - previous_center).sum())
                transition += 0.05 * float(_wrap(theta - previous_theta)) ** 2
                value = costs[previous_index] + emission + transition
                if value < next_costs[current_index]:
                    next_costs[current_index] = value
                    next_paths[current_index] = paths[previous_index] + [current_index]
        costs, paths = next_costs, next_paths
    best_path = paths[int(np.argmin(costs))]
    centers = np.stack([
        candidate_sets[index][choice][1] for index, choice in enumerate(best_path)
    ])
    wrapped_theta = np.asarray([
        candidate_sets[index][choice][2] for index, choice in enumerate(best_path)
    ])
    theta = np.unwrap(wrapped_theta)
    times = event_time_s[frame_indices].astype(np.float64)
    if len(times) == 1:
        return centers[-1], float(theta[-1]), np.zeros(3), 0.0, 1
    pair_dt = times[:, None] - times[None, :]
    upper = np.triu(np.ones_like(pair_dt, dtype=bool), 1)
    valid = upper & (np.abs(pair_dt) > 1e-9)
    velocity_slopes = (centers[:, None, :] - centers[None, :, :])[valid] / pair_dt[valid, None]
    angular_slopes = (theta[:, None] - theta[None, :])[valid] / pair_dt[valid]
    velocity = np.median(velocity_slopes, axis=0)
    omega = float(np.median(angular_slopes))
    center_zero = np.median(centers - times[:, None] * velocity, axis=0)
    theta_zero = float(np.median(theta - times * omega))
    return center_zero, theta_zero, velocity, omega, len(times)


def rigid_constant_velocity_yaw_rate(
    obs: np.ndarray, obs_mask: np.ndarray, event_mask: np.ndarray,
    event_time_s: np.ndarray, tau: np.ndarray, geometry: dict[str, np.ndarray],
    *, static: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool]]:
    state = _fit_rigid_history(obs, obs_mask, event_mask, event_time_s, geometry)
    if state is None:
        return (
            np.zeros((len(tau), 4, 3), dtype=np.float32),
            np.zeros((len(tau), 4, 3), dtype=np.float32),
            {"valid": False, "history_states": 0, "fallback_static": True},
        )
    center, theta, velocity, omega, history_states = state
    fallback_static = static or history_states < 2
    if fallback_static:
        velocity = np.zeros(3)
        omega = 0.0
    positions = np.zeros((len(tau), 4, 3), dtype=np.float32)
    normals = np.zeros_like(positions)
    for query_index, seconds in enumerate(tau):
        angle = theta + omega * float(seconds)
        positions[query_index] = center + velocity * float(seconds) + _rotate_z(
            geometry["position"], angle
        )
        normals[query_index] = _rotate_z(geometry["normal"], angle)
    return positions, normals, {
        "valid": True,
        "history_states": history_states,
        "fallback_static": fallback_static,
    }


def _last_frame(obs: np.ndarray, mask: np.ndarray, event_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_frames = np.flatnonzero(event_mask & (mask.any(axis=1)))
    if len(valid_frames) == 0:
        return np.zeros((4, 3), dtype=np.float32), np.zeros((4,), dtype=bool)
    frame = int(valid_frames[-1])
    return obs[frame, :, :3].copy(), mask[frame].copy()


def static_hold(obs: np.ndarray, obs_mask: np.ndarray, event_mask: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position, mask = _last_frame(obs, obs_mask, event_mask)
    positions = np.repeat(position[None], len(tau), axis=0)
    normals = np.zeros_like(positions)
    return positions, normals


def _best_permutation(reference: np.ndarray, reference_mask: np.ndarray, current: np.ndarray, current_mask: np.ndarray) -> tuple[int, ...]:
    best = PERMUTATIONS[0]
    best_cost = float("inf")
    for permutation in PERMUTATIONS:
        cost = 0.0
        count = 0
        for index, source in enumerate(permutation):
            if reference_mask[index] and current_mask[source]:
                cost += float(np.sum((reference[index] - current[source]) ** 2))
                count += 1
        if count and cost / count < best_cost:
            best_cost = cost / count
            best = permutation
    return best


def constant_twist(
    obs: np.ndarray, obs_mask: np.ndarray, event_mask: np.ndarray,
    event_time_s: np.ndarray, tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    latest_pos, latest_mask = _last_frame(obs, obs_mask, event_mask)
    valid_frames = np.flatnonzero(event_mask & (obs_mask.any(axis=1)))
    if len(valid_frames) < 2:
        return np.repeat(latest_pos[None], len(tau), axis=0), np.zeros((len(tau), 4, 3), dtype=np.float32)
    i0, i1 = int(valid_frames[max(0, len(valid_frames) - 20)]), int(valid_frames[-1])
    dt = max(float(event_time_s[i1] - event_time_s[i0]), 1e-6)
    earlier = obs[i0, :, :3]
    earlier_mask = obs_mask[i0]
    permutation = _best_permutation(latest_pos, latest_mask, earlier, earlier_mask)
    aligned = earlier[list(permutation)]
    common = latest_mask & obs_mask[i0, list(permutation)]
    velocity = np.zeros((4, 3), dtype=np.float32)
    velocity[common] = (latest_pos[common] - aligned[common]) / dt
    center = latest_pos[latest_mask].mean(axis=0) if latest_mask.any() else np.zeros(3)
    centered = latest_pos - center
    angles = []
    for index in range(4):
        source = permutation[index]
        if latest_mask[index] and earlier_mask[source]:
            a0 = math.atan2(float(aligned[index, 1] - center[1]), float(aligned[index, 0] - center[0]))
            a1 = math.atan2(float(latest_pos[index, 1] - center[1]), float(latest_pos[index, 0] - center[0]))
            angles.append(math.atan2(math.sin(a1 - a0), math.cos(a1 - a0)) / dt)
    omega = float(np.median(angles)) if angles else 0.0
    result = np.zeros((len(tau), 4, 3), dtype=np.float32)
    for query_index, seconds in enumerate(tau):
        angle = omega * float(seconds)
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        result[query_index] = center + (centered @ rotation.T) + velocity * float(seconds)
    return result, np.zeros_like(result)
