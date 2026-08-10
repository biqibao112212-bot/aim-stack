#!/usr/bin/env python3
"""Evaluate a transparent rm_vision-style 11-state EKF against simple CV.

This is an offline, oracle-identity ablation.  Historical truth is used only in
explicit ``exact_truth`` intervention arms.  Future truth is never supplied to
an estimator; it is used only after prediction for interpolation and scoring.

State (world frame):
    [xc, vxc, yc, vyc, zc, vzc, theta, omega, r_even, r_odd, dz_odd]

The state is the common rm_vision 9-state vehicle-center model with the
upstream external ``another_r`` and ``dz`` geometry folded into the filter so
that the complete target representation is testable as one 11-state EKF.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HORIZONS_S = (0.0, 0.05, 0.10, 0.20)
HISTORY_SIZE = 16
MAX_HISTORY_SPAN_S = 0.75
MAX_HISTORY_GAP_S = 0.12
MAX_FUTURE_BRACKET_S = 0.04
EVALUATION_INTERVAL_S = 0.10
LATERAL_DIAGNOSTIC_GATE_M = 0.055

# Values copied from rm_vision/rm_auto_aim tracker_node.cpp.  The upstream
# target uses a 9-state EKF and keeps another_r/dz beside it.  In this explicit
# 11-state expansion the same radius random-walk scale is applied to r_even,
# r_odd and dz_odd.
SIGMA2_Q_XYZ = 20.0
SIGMA2_Q_YAW = 100.0
SIGMA2_Q_GEOMETRY = 800.0
R_POSITION_SCALE = 0.05
R_YAW = 0.02
INITIAL_RADIUS_M = 0.26
MIN_RADIUS_M = 0.12
MAX_RADIUS_M = 0.40
MAX_DZ_ABS_M = 0.30

MOTION_ORDER = ("stationary", "linear", "spin", "linear_and_spin")
MOTION_LABEL = {
    "stationary": "Stationary",
    "linear": "Translation",
    "spin": "Rotation",
    "linear_and_spin": "Combined",
}
METHOD_LABEL = {
    "hold_camera": "Hold (camera frame; prior baseline)",
    "cv_ols_camera_16": "OLS-CV (camera frame; prior baseline)",
    "cv_ols_world_16": "OLS-CV (world frame)",
    "ekf11_window16_single_slot": "11D EKF (same 16 samples)",
    "ekf11_persistent_single_slot": "11D EKF (causal segment, same slot)",
    "ekf11_persistent_oracle_multislot": "11D EKF (causal segment, oracle multi-slot)",
}
COLORS = {
    "hold_camera": "#777777",
    "cv_ols_camera_16": "#0072B2",
    "cv_ols_world_16": "#56B4E9",
    "ekf11_window16_single_slot": "#E69F00",
    "ekf11_persistent_single_slot": "#CC79A7",
    "ekf11_persistent_oracle_multislot": "#009E73",
}
PLOT_METHODS = (
    "cv_ols_camera_16",
    "cv_ols_world_16",
    "ekf11_window16_single_slot",
    "ekf11_persistent_oracle_multislot",
)

REFERENCE_URLS = {
    "tracker_node": "https://gitlab.com/rm_vision/rm_auto_aim/-/raw/main/armor_tracker/src/tracker_node.cpp",
    "tracker_header": "https://gitlab.com/rm_vision/rm_auto_aim/-/raw/main/armor_tracker/include/armor_tracker/tracker.hpp",
    "tracker_update": "https://gitlab.com/rm_vision/rm_auto_aim/-/raw/main/armor_tracker/src/tracker.cpp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-trajectory-rows", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=0,
        help="Optional deterministic smoke-test limit; 0 means all sessions.",
    )
    return parser.parse_args()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return pd.read_csv(handle, **kwargs)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(str(path), "wt", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
    else:
        with path.open("w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wrap_angle(value: float | np.ndarray) -> float | np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def quaternion_yaw_wxyz(values: Sequence[float]) -> float:
    w, x, y, z = (float(v) for v in values)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_rotation_wxyz(values: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in values)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def percentile(values: Iterable[float], q: float) -> float:
    array = finite(values)
    return float(np.percentile(array, q)) if len(array) else float("nan")


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.8,
        }
    )


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


TRAJECTORY_COLUMNS = [
    "dataset",
    "session_id",
    "motion_mode",
    "configured_distance_m",
    "configured_yaw_deg",
    "configured_linear_speed_mps",
    "configured_spin_rad_s",
    "producer_epoch",
    "frame_seq",
    "timestamp_ns",
    "time_s",
    "truth_slot",
    "is_visible_slot",
    "truth_world_x_m",
    "truth_world_y_m",
    "truth_world_z_m",
    "pnp_world_x_m",
    "pnp_world_y_m",
    "pnp_world_z_m",
    "truth_camera_x_m",
    "truth_camera_y_m",
    "truth_camera_z_m",
    "pnp_camera_x_m",
    "pnp_camera_y_m",
    "pnp_camera_z_m",
    "center_world_x_m",
    "center_world_y_m",
    "center_world_z_m",
]


def load_visible_rows(path: Path, max_sessions: int) -> pd.DataFrame:
    frame = read_csv(path, usecols=TRAJECTORY_COLUMNS)
    frame = frame[frame["is_visible_slot"].astype(bool)].copy()
    finite_columns = [
        "time_s",
        "truth_world_x_m",
        "truth_world_y_m",
        "truth_world_z_m",
        "pnp_world_x_m",
        "pnp_world_y_m",
        "pnp_world_z_m",
        "truth_camera_x_m",
        "truth_camera_y_m",
        "truth_camera_z_m",
        "pnp_camera_x_m",
        "pnp_camera_y_m",
        "pnp_camera_z_m",
        "center_world_x_m",
        "center_world_y_m",
        "center_world_z_m",
    ]
    valid = np.isfinite(np.asarray(frame[finite_columns].values, dtype=float)).all(axis=1)
    frame = frame.loc[valid].copy()
    if max_sessions > 0:
        sessions = sorted(frame["session_id"].unique())[:max_sessions]
        frame = frame[frame["session_id"].isin(sessions)].copy()
    frame["truth_slot"] = frame["truth_slot"].astype(int)
    frame["truth_yaw_world_rad"] = np.arctan2(
        frame["center_world_y_m"] - frame["truth_world_y_m"],
        frame["center_world_x_m"] - frame["truth_world_x_m"],
    )
    return frame.sort_values(["session_id", "time_s", "truth_slot"]).reset_index(drop=True)


def observation_path(dataset_root: Path, session_id: str) -> Path:
    candidates = sorted((dataset_root / session_id).glob("run-*/observations.jsonl"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one observations.jsonl for {session_id}, found {len(candidates)}"
        )
    return candidates[0]


def best_assignment(cost: np.ndarray) -> tuple[int, ...] | None:
    rows, columns = cost.shape
    if rows == 0 or columns < rows:
        return None
    best: tuple[int, ...] | None = None
    best_cost = float("inf")
    for assignment in itertools.permutations(range(columns), rows):
        value = float(sum(cost[row, column] for row, column in enumerate(assignment)))
        if value < best_cost:
            best_cost = value
            best = assignment
    return best


def join_observed_yaw(
    frame: pd.DataFrame, dataset_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    observed_yaw = np.full(len(result), np.nan, dtype=float)
    match_delta = np.full(len(result), np.nan, dtype=float)
    source_index = np.full(len(result), -1, dtype=int)
    camera_origins = np.full((len(result), 3), np.nan, dtype=float)
    rotations_camera_to_world = np.full((len(result), 3, 3), np.nan, dtype=float)
    diagnostics: list[dict] = []

    for session_id, session in result.groupby("session_id", sort=True):
        keyed: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for index, row in session.iterrows():
            keyed[(int(row.producer_epoch), int(row.frame_seq), int(row.timestamp_ns))].append(
                int(index)
            )
        path = observation_path(dataset_root, str(session_id))
        json_frames = 0
        relevant_frames = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                json_frames += 1
                observation = json.loads(line)
                key = (
                    int(observation["producer_epoch"]),
                    int(observation["frame_seq"]),
                    int(observation["timestamp_ns"]),
                )
                indexes = keyed.get(key)
                if not indexes:
                    continue
                relevant_frames += 1
                armors = [armor for armor in observation.get("armors", []) if armor.get("valid", True)]
                if not armors:
                    continue
                expected = np.asarray(
                    [
                        result.loc[index, ["pnp_camera_x_m", "pnp_camera_y_m", "pnp_camera_z_m"]].values
                        for index in indexes
                    ],
                    dtype=float,
                )
                candidates = np.asarray([armor["camera_tvec_m"] for armor in armors], dtype=float)
                cost = np.linalg.norm(expected[:, None, :] - candidates[None, :, :], axis=2)
                assignment = best_assignment(cost)
                if assignment is None:
                    continue
                tracker_yaw = quaternion_yaw_wxyz(
                    observation["tracker_gimbal_quaternion_world_wxyz"]
                )
                rotation_gimbal_to_world = quaternion_rotation_wxyz(
                    observation["camera_quaternion_world_wxyz"]
                )
                rotation_camera_to_gimbal = np.asarray(
                    observation["R_camera2gimbal"], dtype=float
                ).reshape(3, 3)
                rotation_camera_to_world = (
                    rotation_gimbal_to_world @ rotation_camera_to_gimbal
                )
                camera_origin = np.asarray(
                    observation["camera_origin_world_ros_m"], dtype=float
                )
                for local_row, armor_index in enumerate(assignment):
                    index = indexes[local_row]
                    armor = armors[armor_index]
                    observed_yaw[index] = float(
                        wrap_angle(float(armor["yaw_absolute_rad"]) + tracker_yaw)
                    )
                    match_delta[index] = float(cost[local_row, armor_index])
                    source_index[index] = int(armor.get("observation_index", armor_index))
                    camera_origins[index] = camera_origin
                    rotations_camera_to_world[index] = rotation_camera_to_world
        local_indexes = session.index.to_numpy(dtype=int)
        joined = np.isfinite(observed_yaw[local_indexes])
        yaw_error = np.asarray(
            wrap_angle(
                observed_yaw[local_indexes][joined]
                - result.loc[local_indexes[joined], "truth_yaw_world_rad"].to_numpy(dtype=float)
            ),
            dtype=float,
        )
        diagnostics.append(
            {
                "session_id": session_id,
                "motion_mode": str(session["motion_mode"].iloc[0]),
                "rows": int(len(session)),
                "joined_rows": int(np.count_nonzero(joined)),
                "join_rate": float(np.mean(joined)),
                "json_frames": int(json_frames),
                "relevant_json_frames": int(relevant_frames),
                "match_position_delta_p95_m": percentile(match_delta[local_indexes], 95),
                "observed_vs_truth_yaw_abs_p50_rad": percentile(np.abs(yaw_error), 50),
                "observed_vs_truth_yaw_abs_p95_rad": percentile(np.abs(yaw_error), 95),
            }
        )
    result["observed_yaw_world_rad"] = observed_yaw
    result["observation_camera_match_delta_m"] = match_delta
    result["observation_index"] = source_index
    for axis, name in enumerate("xyz"):
        result[f"camera_origin_world_{name}_m"] = camera_origins[:, axis]
    for row in range(3):
        for column in range(3):
            result[f"rotation_camera_to_world_{row}{column}"] = rotations_camera_to_world[
                :, row, column
            ]
    return result, pd.DataFrame(diagnostics)


def frame_transform_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    origins = frame[
        ["camera_origin_world_x_m", "camera_origin_world_y_m", "camera_origin_world_z_m"]
    ].to_numpy(dtype=float)
    rotations = np.empty((len(frame), 3, 3), dtype=float)
    for row in range(3):
        for column in range(3):
            rotations[:, row, column] = frame[
                f"rotation_camera_to_world_{row}{column}"
            ].to_numpy(dtype=float)
    return origins, rotations


def map_world_to_camera(
    world_points: np.ndarray, origins: np.ndarray, rotations_camera_to_world: np.ndarray
) -> np.ndarray:
    points = np.asarray(world_points, dtype=float)
    relative = points - origins
    return np.einsum("nij,nj->ni", rotations_camera_to_world.transpose(0, 2, 1), relative)


def validate_frame_transforms(frame: pd.DataFrame) -> pd.DataFrame:
    diagnostics: list[dict] = []
    for session_id, group in frame.groupby("session_id", sort=True):
        origins, rotations = frame_transform_arrays(group)
        world_pnp = np.asarray(
            group[["pnp_world_x_m", "pnp_world_y_m", "pnp_world_z_m"]].values,
            dtype=float,
        )
        camera_pnp = np.asarray(
            group[["pnp_camera_x_m", "pnp_camera_y_m", "pnp_camera_z_m"]].values,
            dtype=float,
        )
        world_truth = np.asarray(
            group[["truth_world_x_m", "truth_world_y_m", "truth_world_z_m"]].values,
            dtype=float,
        )
        camera_truth = np.asarray(
            group[["truth_camera_x_m", "truth_camera_y_m", "truth_camera_z_m"]].values,
            dtype=float,
        )
        reconstructed_pnp = map_world_to_camera(world_pnp, origins, rotations)
        reconstructed_truth = map_world_to_camera(world_truth, origins, rotations)
        residual = np.vstack(
            [reconstructed_pnp - camera_pnp, reconstructed_truth - camera_truth]
        )
        residual_norm = np.linalg.norm(residual, axis=1)
        determinants = np.linalg.det(rotations)
        diagnostics.append(
            {
                "session_id": session_id,
                "motion_mode": str(group["motion_mode"].iloc[0]),
                "samples": int(2 * len(group)),
                "rotation_determinant_min": float(np.min(determinants)),
                "rotation_determinant_max": float(np.max(determinants)),
                "fit_residual_mean_m": float(np.mean(residual_norm)),
                "fit_residual_p95_m": percentile(residual_norm, 95),
                "fit_residual_max_m": float(np.max(residual_norm)),
            }
        )
    return pd.DataFrame(diagnostics)


class RmVisionEkf11:
    """Minimal EKF with no association/gating/engineering recovery heuristics."""

    def __init__(self, first_position: np.ndarray, first_yaw: float, first_slot: int):
        self.x = np.zeros(11, dtype=float)
        base_yaw = float(wrap_angle(first_yaw - first_slot * math.pi / 2.0))
        radius = INITIAL_RADIUS_M
        self.x[0] = float(first_position[0] + radius * math.cos(first_yaw))
        self.x[2] = float(first_position[1] + radius * math.sin(first_yaw))
        self.x[4] = float(first_position[2])
        self.x[6] = base_yaw
        self.x[8] = radius
        self.x[9] = radius
        self.x[10] = 0.0
        self.p = np.eye(11, dtype=float)

    @staticmethod
    def transition(dt: float) -> tuple[np.ndarray, np.ndarray]:
        f = np.eye(11, dtype=float)
        q = np.zeros((11, 11), dtype=float)
        for position, velocity in ((0, 1), (2, 3), (4, 5)):
            f[position, velocity] = dt
            block = SIGMA2_Q_XYZ * np.asarray(
                [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]],
                dtype=float,
            )
            q[np.ix_([position, velocity], [position, velocity])] = block
        f[6, 7] = dt
        q[np.ix_([6, 7], [6, 7])] = SIGMA2_Q_YAW * np.asarray(
            [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]],
            dtype=float,
        )
        geometry_variance = SIGMA2_Q_GEOMETRY * dt**4 / 4.0
        for index in (8, 9, 10):
            q[index, index] = geometry_variance
        return f, q

    def predict(self, dt: float) -> None:
        if dt < -1e-9:
            raise ValueError(f"negative EKF dt: {dt}")
        if dt <= 0.0:
            return
        f, q = self.transition(dt)
        self.x = f @ self.x
        self.x[6] = float(wrap_angle(self.x[6]))
        self.p = f @ self.p @ f.T + q

    @staticmethod
    def observation(x: np.ndarray, slot: int) -> tuple[np.ndarray, np.ndarray]:
        angle = float(x[6] + slot * math.pi / 2.0)
        radius_index = 8 if slot % 2 == 0 else 9
        radius = float(x[radius_index])
        odd = 1.0 if slot % 2 else 0.0
        prediction = np.asarray(
            [
                x[0] - radius * math.cos(angle),
                x[2] - radius * math.sin(angle),
                x[4] + odd * x[10],
                wrap_angle(angle),
            ],
            dtype=float,
        )
        h = np.zeros((4, 11), dtype=float)
        h[0, 0] = 1.0
        h[0, 6] = radius * math.sin(angle)
        h[0, radius_index] = -math.cos(angle)
        h[1, 2] = 1.0
        h[1, 6] = -radius * math.cos(angle)
        h[1, radius_index] = -math.sin(angle)
        h[2, 4] = 1.0
        h[2, 10] = odd
        h[3, 6] = 1.0
        return prediction, h

    def update(self, position: np.ndarray, yaw: float, slot: int) -> None:
        measurement = np.asarray([position[0], position[1], position[2], yaw], dtype=float)
        prediction, h = self.observation(self.x, slot)
        innovation = measurement - prediction
        innovation[3] = float(wrap_angle(innovation[3]))
        r = np.diag(
            [
                max(abs(R_POSITION_SCALE * measurement[0]), 1e-9),
                max(abs(R_POSITION_SCALE * measurement[1]), 1e-9),
                max(abs(R_POSITION_SCALE * measurement[2]), 1e-9),
                R_YAW,
            ]
        )
        s = h @ self.p @ h.T + r
        gain = np.linalg.solve(s, h @ self.p).T
        self.x = self.x + gain @ innovation
        self.x[6] = float(wrap_angle(self.x[6]))
        identity = np.eye(11)
        ikh = identity - gain @ h
        self.p = ikh @ self.p @ ikh.T + gain @ r @ gain.T
        self.x[8] = float(np.clip(self.x[8], MIN_RADIUS_M, MAX_RADIUS_M))
        self.x[9] = float(np.clip(self.x[9], MIN_RADIUS_M, MAX_RADIUS_M))
        self.x[10] = float(np.clip(self.x[10], -MAX_DZ_ABS_M, MAX_DZ_ABS_M))

    def target_position(self, slot: int) -> np.ndarray:
        return self.observation(self.x, slot)[0][:3]

    def clone(self) -> "RmVisionEkf11":
        clone = object.__new__(RmVisionEkf11)
        clone.x = self.x.copy()
        clone.p = self.p.copy()
        return clone


def select_position(row: pd.Series, source: str) -> np.ndarray:
    prefix = "pnp_world" if source == "current_pnp" else "truth_world"
    return np.asarray(
        [row[f"{prefix}_x_m"], row[f"{prefix}_y_m"], row[f"{prefix}_z_m"]],
        dtype=float,
    )


def select_yaw(row: pd.Series, source: str) -> float:
    return float(
        row["observed_yaw_world_rad"]
        if source == "current_pnp_yaw"
        else row["truth_yaw_world_rad"]
    )


def run_ekf_history(
    history: pd.DataFrame, position_source: str, yaw_source: str
) -> RmVisionEkf11 | None:
    ordered = history.sort_values(["time_s", "truth_slot"])
    if not len(ordered):
        return None
    if yaw_source == "current_pnp_yaw" and not np.isfinite(
        ordered["observed_yaw_world_rad"].to_numpy(dtype=float)
    ).all():
        return None
    first = ordered.iloc[0]
    ekf = RmVisionEkf11(
        select_position(first, position_source),
        select_yaw(first, yaw_source),
        int(first["truth_slot"]),
    )
    previous_time = float(first["time_s"])
    ekf.update(
        select_position(first, position_source),
        select_yaw(first, yaw_source),
        int(first["truth_slot"]),
    )
    for _, row in ordered.iloc[1:].iterrows():
        time_s = float(row["time_s"])
        ekf.predict(max(0.0, time_s - previous_time))
        ekf.update(
            select_position(row, position_source),
            select_yaw(row, yaw_source),
            int(row["truth_slot"]),
        )
        previous_time = time_s
    return ekf


def _cache_one_stream(
    frame: pd.DataFrame,
    indexes: np.ndarray,
    position_source: str,
    yaw_source: str,
    state_output: np.ndarray,
    count_output: np.ndarray,
) -> None:
    ordered = frame.loc[indexes].sort_values(["time_s", "truth_slot"])
    if not len(ordered):
        return
    if yaw_source == "current_pnp_yaw" and not np.isfinite(
        ordered["observed_yaw_world_rad"].to_numpy(dtype=float)
    ).all():
        return
    ekf: RmVisionEkf11 | None = None
    previous_time = 0.0
    count = 0
    for time_s, simultaneous in ordered.groupby("time_s", sort=True):
        if ekf is not None and float(time_s) - previous_time > MAX_HISTORY_GAP_S:
            # A normal online tracker becomes LOST and reinitializes rather than
            # propagating a stale single-armor state across a long invisible arc.
            ekf = None
            count = 0
        if ekf is None:
            first = simultaneous.iloc[0]
            ekf = RmVisionEkf11(
                select_position(first, position_source),
                select_yaw(first, yaw_source),
                int(first["truth_slot"]),
            )
        else:
            ekf.predict(max(0.0, float(time_s) - previous_time))
        for _, row in simultaneous.sort_values("truth_slot").iterrows():
            ekf.update(
                select_position(row, position_source),
                select_yaw(row, yaw_source),
                int(row["truth_slot"]),
            )
            count += 1
        simultaneous_indexes = simultaneous.index.to_numpy(dtype=int)
        state_output[simultaneous_indexes] = ekf.x
        count_output[simultaneous_indexes] = count
        previous_time = float(time_s)


def build_persistent_state_cache(
    frame: pd.DataFrame,
) -> tuple[dict[tuple[str, str, str], np.ndarray], dict[tuple[str, str, str], np.ndarray]]:
    state_cache: dict[tuple[str, str, str], np.ndarray] = {}
    count_cache: dict[tuple[str, str, str], np.ndarray] = {}
    variants = [
        (scope, position_source, yaw_source)
        for scope in ("single", "multi")
        for position_source in ("current_pnp", "exact_truth")
        for yaw_source in ("current_pnp_yaw", "exact_truth_yaw")
    ]
    for key in variants:
        state_cache[key] = np.full((len(frame), 11), np.nan, dtype=float)
        count_cache[key] = np.zeros(len(frame), dtype=int)

    single_groups = list(frame.groupby(["session_id", "truth_slot"], sort=True))
    multi_groups = list(frame.groupby("session_id", sort=True))
    for position_source in ("current_pnp", "exact_truth"):
        for yaw_source in ("current_pnp_yaw", "exact_truth_yaw"):
            key = ("single", position_source, yaw_source)
            for _, group in single_groups:
                _cache_one_stream(
                    frame,
                    group.index.to_numpy(dtype=int),
                    position_source,
                    yaw_source,
                    state_cache[key],
                    count_cache[key],
                )
            key = ("multi", position_source, yaw_source)
            for _, group in multi_groups:
                _cache_one_stream(
                    frame,
                    group.index.to_numpy(dtype=int),
                    position_source,
                    yaw_source,
                    state_cache[key],
                    count_cache[key],
                )
    return state_cache, count_cache


def predict_state(state: np.ndarray, horizon_s: float) -> np.ndarray:
    result = np.asarray(state, dtype=float).copy()
    result[0] += result[1] * horizon_s
    result[2] += result[3] * horizon_s
    result[4] += result[5] * horizon_s
    result[6] = float(wrap_angle(result[6] + result[7] * horizon_s))
    return result


def target_position_from_state(state: np.ndarray, slot: int) -> np.ndarray:
    return RmVisionEkf11.observation(np.asarray(state, dtype=float), slot)[0][:3]


def interpolate_bracket(
    times: np.ndarray, values: np.ndarray, target_s: float
) -> np.ndarray | None:
    index = int(np.searchsorted(times, target_s))
    if index == 0 or index >= len(times):
        return None
    before, after = float(times[index - 1]), float(times[index])
    if after - before > MAX_FUTURE_BRACKET_S:
        return None
    weight = (target_s - before) / max(after - before, 1e-12)
    return values[index - 1] * (1.0 - weight) + values[index] * weight


def map_prediction_at_future_time(
    prediction_world: np.ndarray,
    times: np.ndarray,
    origins: np.ndarray,
    rotations: np.ndarray,
    anchor_index: int,
    future_time_s: float,
    horizon_s: float,
) -> np.ndarray | None:
    if horizon_s == 0.0:
        return map_world_to_camera(
            np.asarray([prediction_world]),
            origins[anchor_index : anchor_index + 1],
            rotations[anchor_index : anchor_index + 1],
        )[0]
    index = int(np.searchsorted(times, future_time_s))
    if index == 0 or index >= len(times):
        return None
    before, after = float(times[index - 1]), float(times[index])
    if after - before > MAX_FUTURE_BRACKET_S:
        return None
    mapped = map_world_to_camera(
        np.repeat(np.asarray(prediction_world, dtype=float)[None, :], 2, axis=0),
        origins[index - 1 : index + 1],
        rotations[index - 1 : index + 1],
    )
    weight = (future_time_s - before) / max(after - before, 1e-12)
    return mapped[0] * (1.0 - weight) + mapped[1] * weight


def score_prediction(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | bool]:
    error = np.asarray(prediction - truth, dtype=float)
    lateral = float(np.hypot(error[0], error[1]))
    depth = float(error[2])
    if prediction[2] <= 1e-9 or truth[2] <= 1e-9:
        yaw_plane = pitch_plane = angular = float("nan")
    else:
        yaw_plane = float(truth[2] * prediction[0] / prediction[2] - truth[0])
        pitch_plane = float(truth[2] * prediction[1] / prediction[2] - truth[1])
        pred_ray = prediction / max(float(np.linalg.norm(prediction)), 1e-12)
        truth_ray = truth / max(float(np.linalg.norm(truth)), 1e-12)
        angular = math.degrees(
            math.acos(float(np.clip(np.dot(pred_ray, truth_ray), -1.0, 1.0)))
        )
    return {
        "camera_error_x_signed_m": float(error[0]),
        "camera_error_y_signed_m": float(error[1]),
        "camera_error_z_signed_m": depth,
        "camera_error_x_abs_m": float(abs(error[0])),
        "camera_error_y_abs_m": float(abs(error[1])),
        "camera_depth_error_abs_m": float(abs(depth)),
        "camera_lateral_xy_error_m": lateral,
        "position_error_3d_m": float(np.linalg.norm(error)),
        "yaw_plane_miss_signed_m": yaw_plane,
        "yaw_plane_miss_abs_m": abs(yaw_plane),
        "pitch_plane_miss_signed_m": pitch_plane,
        "pitch_plane_miss_abs_m": abs(pitch_plane),
        "ray_angular_error_deg": angular,
        "lateral_55mm_diagnostic_pass": lateral <= LATERAL_DIAGNOSTIC_GATE_M,
        "yaw_plane_55mm_pass": abs(yaw_plane) <= LATERAL_DIAGNOSTIC_GATE_M,
    }


def add_prediction_row(
    rows: list[dict],
    common: dict,
    method: str,
    position_source: str,
    yaw_source: str,
    prediction_camera: np.ndarray,
    prediction_world: np.ndarray,
    future_truth_camera: np.ndarray,
    future_truth_world: np.ndarray,
    state: np.ndarray | None,
    history_observations: int,
) -> None:
    state_values = np.asarray(state, dtype=float) if state is not None else np.full(11, np.nan)
    rows.append(
        {
            **common,
            "method": method,
            "position_source": position_source,
            "yaw_source": yaw_source,
            "history_observations": int(history_observations),
            "prediction_camera_x_m": float(prediction_camera[0]),
            "prediction_camera_y_m": float(prediction_camera[1]),
            "prediction_camera_z_m": float(prediction_camera[2]),
            "prediction_world_x_m": float(prediction_world[0]),
            "prediction_world_y_m": float(prediction_world[1]),
            "prediction_world_z_m": float(prediction_world[2]),
            "future_truth_camera_x_m": float(future_truth_camera[0]),
            "future_truth_camera_y_m": float(future_truth_camera[1]),
            "future_truth_camera_z_m": float(future_truth_camera[2]),
            "future_truth_world_x_m": float(future_truth_world[0]),
            "future_truth_world_y_m": float(future_truth_world[1]),
            "future_truth_world_z_m": float(future_truth_world[2]),
            "state_xc_m": float(state_values[0]),
            "state_vxc_mps": float(state_values[1]),
            "state_yc_m": float(state_values[2]),
            "state_vyc_mps": float(state_values[3]),
            "state_zc_m": float(state_values[4]),
            "state_vzc_mps": float(state_values[5]),
            "state_theta_rad": float(state_values[6]),
            "state_omega_rad_s": float(state_values[7]),
            "state_r_even_m": float(state_values[8]),
            "state_r_odd_m": float(state_values[9]),
            "state_dz_odd_m": float(state_values[10]),
            **score_prediction(prediction_camera, future_truth_camera),
        }
    )


def build_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict] = []
    coverage_rows: list[dict] = []
    persistent_states, persistent_counts = build_persistent_state_cache(frame)
    for (session_id, slot), group in frame.groupby(["session_id", "truth_slot"], sort=True):
        ordered = group.sort_values("time_s")
        times = ordered["time_s"].to_numpy(dtype=float)
        truth_camera = ordered[
            ["truth_camera_x_m", "truth_camera_y_m", "truth_camera_z_m"]
        ].to_numpy(dtype=float)
        truth_world = ordered[
            ["truth_world_x_m", "truth_world_y_m", "truth_world_z_m"]
        ].to_numpy(dtype=float)
        pnp_camera = ordered[
            ["pnp_camera_x_m", "pnp_camera_y_m", "pnp_camera_z_m"]
        ].to_numpy(dtype=float)
        pnp_world = ordered[
            ["pnp_world_x_m", "pnp_world_y_m", "pnp_world_z_m"]
        ].to_numpy(dtype=float)
        origins, rotations = frame_transform_arrays(ordered)
        meta = ordered.iloc[0]
        last_evaluation_s = -float("inf")
        candidate_count = accepted_count = 0
        for index in range(HISTORY_SIZE - 1, len(ordered)):
            candidate_count += 1
            history = ordered.iloc[index - HISTORY_SIZE + 1 : index + 1]
            history_times = history["time_s"].to_numpy(dtype=float)
            gaps = np.diff(history_times)
            if (
                history_times[-1] - history_times[0] > MAX_HISTORY_SPAN_S
                or float(np.max(gaps)) > MAX_HISTORY_GAP_S
                or history_times[-1] - last_evaluation_s < EVALUATION_INTERVAL_S
            ):
                continue
            anchor_time = float(history_times[-1])
            anchor_global_index = int(history.index[-1])
            last_evaluation_s = anchor_time
            accepted_count += 1

            local_times = history_times - anchor_time
            design = np.column_stack([np.ones(HISTORY_SIZE), local_times])
            cv_predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
            for position_source, world_values, camera_values in (
                (
                    "current_pnp",
                    pnp_world[index - HISTORY_SIZE + 1 : index + 1],
                    pnp_camera[index - HISTORY_SIZE + 1 : index + 1],
                ),
                (
                    "exact_truth",
                    truth_world[index - HISTORY_SIZE + 1 : index + 1],
                    truth_camera[index - HISTORY_SIZE + 1 : index + 1],
                ),
            ):
                world_coefficients = np.linalg.lstsq(design, world_values, rcond=None)[0]
                camera_coefficients = np.linalg.lstsq(design, camera_values, rcond=None)[0]
                cv_predictions[position_source] = (
                    world_values[-1],
                    camera_values[-1],
                    world_coefficients,
                    camera_coefficients,
                )

            window_filters: dict[tuple[str, str], RmVisionEkf11 | None] = {}
            for position_source in ("current_pnp", "exact_truth"):
                for yaw_source in ("current_pnp_yaw", "exact_truth_yaw"):
                    window_filters[(position_source, yaw_source)] = run_ekf_history(
                        history, position_source, yaw_source
                    )

            for horizon_s in HORIZONS_S:
                future_time = anchor_time + horizon_s
                if horizon_s == 0.0:
                    future_camera = truth_camera[index]
                    future_world = truth_world[index]
                else:
                    future_camera = interpolate_bracket(times, truth_camera, future_time)
                    future_world = interpolate_bracket(times, truth_world, future_time)
                    if future_camera is None or future_world is None:
                        continue
                common = {
                    "session_id": session_id,
                    "truth_slot": int(slot),
                    "motion_mode": str(meta["motion_mode"]),
                    "configured_distance_m": float(meta["configured_distance_m"]),
                    "configured_yaw_deg": float(meta["configured_yaw_deg"]),
                    "configured_linear_speed_mps": float(meta["configured_linear_speed_mps"]),
                    "configured_spin_rad_s": float(meta["configured_spin_rad_s"]),
                    "anchor_time_s": anchor_time,
                    "future_time_s": future_time,
                    "horizon_s": horizon_s,
                    "history_target_slot_samples": HISTORY_SIZE,
                    "history_span_s": float(history_times[-1] - history_times[0]),
                    "history_max_gap_s": float(np.max(gaps)),
                }
                for position_source, (
                    anchor_world,
                    anchor_camera,
                    world_coefficients,
                    camera_coefficients,
                ) in cv_predictions.items():
                    world_prediction = (
                        world_coefficients[0] + world_coefficients[1] * horizon_s
                    )
                    world_prediction_camera = map_prediction_at_future_time(
                        world_prediction,
                        times,
                        origins,
                        rotations,
                        index,
                        future_time,
                        horizon_s,
                    )
                    camera_cv_prediction = (
                        camera_coefficients[0] + camera_coefficients[1] * horizon_s
                    )
                    for method, camera_prediction, stored_world_prediction in (
                        ("hold_camera", anchor_camera, anchor_world),
                        (
                            "cv_ols_camera_16",
                            camera_cv_prediction,
                            np.full(3, np.nan),
                        ),
                        (
                            "cv_ols_world_16",
                            world_prediction_camera,
                            world_prediction,
                        ),
                    ):
                        if camera_prediction is None:
                            continue
                        add_prediction_row(
                            prediction_rows,
                            common,
                            method,
                            position_source,
                            "not_used",
                            camera_prediction,
                            stored_world_prediction,
                            future_camera,
                            future_world,
                            None,
                            HISTORY_SIZE,
                        )
                for position_source in ("current_pnp", "exact_truth"):
                    for yaw_source in ("current_pnp_yaw", "exact_truth_yaw"):
                        window_ekf = window_filters[(position_source, yaw_source)]
                        if window_ekf is not None:
                            projected = window_ekf.clone()
                            projected.predict(horizon_s)
                            world_prediction = projected.target_position(int(slot))
                            camera_prediction = map_prediction_at_future_time(
                                world_prediction,
                                times,
                                origins,
                                rotations,
                                index,
                                future_time,
                                horizon_s,
                            )
                            if camera_prediction is not None:
                                add_prediction_row(
                                    prediction_rows,
                                    common,
                                    "ekf11_window16_single_slot",
                                    position_source,
                                    yaw_source,
                                    camera_prediction,
                                    world_prediction,
                                    future_camera,
                                    future_world,
                                    projected.x,
                                    HISTORY_SIZE,
                                )

                        for scope, method in (
                            ("single", "ekf11_persistent_single_slot"),
                            ("multi", "ekf11_persistent_oracle_multislot"),
                        ):
                            key = (scope, position_source, yaw_source)
                            state = persistent_states[key][anchor_global_index]
                            if not np.isfinite(state).all():
                                continue
                            projected_state = predict_state(state, horizon_s)
                            world_prediction = target_position_from_state(
                                projected_state, int(slot)
                            )
                            camera_prediction = map_prediction_at_future_time(
                                world_prediction,
                                times,
                                origins,
                                rotations,
                                index,
                                future_time,
                                horizon_s,
                            )
                            if camera_prediction is None:
                                continue
                            add_prediction_row(
                                prediction_rows,
                                common,
                                method,
                                position_source,
                                yaw_source,
                                camera_prediction,
                                world_prediction,
                                future_camera,
                                future_world,
                                projected_state,
                                int(persistent_counts[key][anchor_global_index]),
                            )
        coverage_rows.append(
            {
                "session_id": session_id,
                "truth_slot": int(slot),
                "motion_mode": str(meta["motion_mode"]),
                "candidate_anchors": int(candidate_count),
                "accepted_anchors": int(accepted_count),
            }
        )
    return pd.DataFrame(prediction_rows), pd.DataFrame(coverage_rows)


METRICS = (
    "camera_error_x_abs_m",
    "camera_error_y_abs_m",
    "camera_depth_error_abs_m",
    "camera_lateral_xy_error_m",
    "position_error_3d_m",
    "yaw_plane_miss_abs_m",
    "pitch_plane_miss_abs_m",
    "ray_angular_error_deg",
)


def metric_summary(rows: pd.DataFrame, group_cols: list[str], scope: str) -> pd.DataFrame:
    output: list[dict] = []
    for keys, group in rows.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        common = {name: value for name, value in zip(group_cols, keys)}
        for metric in METRICS:
            values = finite(group[metric].values)
            if not len(values):
                continue
            output.append(
                {
                    "scope": scope,
                    **common,
                    "metric": metric,
                    "samples": int(len(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "p50": float(np.percentile(values, 50)),
                    "p75": float(np.percentile(values, 75)),
                    "p90": float(np.percentile(values, 90)),
                    "p95": float(np.percentile(values, 95)),
                    "p99": float(np.percentile(values, 99)),
                    "maximum": float(np.max(values)),
                }
            )
        for metric in ("lateral_55mm_diagnostic_pass", "yaw_plane_55mm_pass"):
            output.append(
                {
                    "scope": scope,
                    **common,
                    "metric": f"{metric}_rate",
                    "samples": int(len(group)),
                    "mean": float(np.mean(group[metric].astype(float))),
                }
            )
    return pd.DataFrame(output)


def build_summaries(predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            metric_summary(
                predictions,
                ["motion_mode", "method", "position_source", "yaw_source", "horizon_s"],
                "motion",
            ),
            metric_summary(
                predictions,
                [
                    "motion_mode",
                    "configured_linear_speed_mps",
                    "configured_spin_rad_s",
                    "method",
                    "position_source",
                    "yaw_source",
                    "horizon_s",
                ],
                "motion_rate",
            ),
            metric_summary(
                predictions,
                [
                    "motion_mode",
                    "configured_distance_m",
                    "method",
                    "position_source",
                    "yaw_source",
                    "horizon_s",
                ],
                "motion_distance",
            ),
        ],
        ignore_index=True,
        sort=False,
    )


def primary_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = summary[
        (summary["scope"] == "motion")
        & (summary["metric"] == "camera_lateral_xy_error_m")
    ].copy()
    rows["p50_mm"] = rows["p50"] * 1000.0
    rows["p90_mm"] = rows["p90"] * 1000.0
    rows["p95_mm"] = rows["p95"] * 1000.0
    rows["p99_mm"] = rows["p99"] * 1000.0
    return rows[
        [
            "motion_mode",
            "method",
            "position_source",
            "yaw_source",
            "horizon_s",
            "samples",
            "p50_mm",
            "p90_mm",
            "p95_mm",
            "p99_mm",
        ]
    ].sort_values(
        ["motion_mode", "position_source", "yaw_source", "method", "horizon_s"]
    )


def histogram_distribution(predictions: pd.DataFrame) -> pd.DataFrame:
    edges = np.asarray(
        [0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.055, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, np.inf],
        dtype=float,
    )
    output: list[dict] = []
    keys = ["motion_mode", "method", "position_source", "yaw_source", "horizon_s"]
    for values, group in predictions.groupby(keys, dropna=False):
        metric = group["camera_lateral_xy_error_m"].to_numpy(dtype=float)
        counts, _ = np.histogram(metric, bins=edges)
        total = max(int(np.sum(counts)), 1)
        cumulative = 0
        common = dict(zip(keys, values))
        for index, count in enumerate(counts):
            cumulative += int(count)
            output.append(
                {
                    **common,
                    "metric": "camera_lateral_xy_error_m",
                    "bin_left_m": float(edges[index]),
                    "bin_right_m": float(edges[index + 1]),
                    "count": int(count),
                    "fraction": float(count / total),
                    "cumulative_fraction": float(cumulative / total),
                }
            )
    return pd.DataFrame(output)


def current_input_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    baseline = predictions[
        (predictions["position_source"] == "current_pnp")
        & (predictions["yaw_source"] == "not_used")
    ]
    ekf = predictions[
        (predictions["position_source"] == "current_pnp")
        & (predictions["yaw_source"] == "current_pnp_yaw")
    ]
    return pd.concat([baseline, ekf], ignore_index=True)


def exact_input_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    baseline = predictions[
        (predictions["position_source"] == "exact_truth")
        & (predictions["yaw_source"] == "not_used")
    ]
    ekf = predictions[
        (predictions["position_source"] == "exact_truth")
        & (predictions["yaw_source"] == "exact_truth_yaw")
    ]
    return pd.concat([baseline, ekf], ignore_index=True)


def plot_p95_by_horizon(predictions: pd.DataFrame, output: Path, exact: bool) -> None:
    selected = exact_input_rows(predictions) if exact else current_input_rows(predictions)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), sharex=True, sharey=False)
    for axis, motion in zip(axes.flat, MOTION_ORDER):
        local = selected[selected["motion_mode"] == motion]
        for method in PLOT_METHODS:
            values = []
            for horizon in HORIZONS_S:
                arm = local[(local["method"] == method) & np.isclose(local["horizon_s"], horizon)]
                values.append(percentile(arm["camera_lateral_xy_error_m"].values, 95) * 1000.0)
            axis.plot(
                np.asarray(HORIZONS_S) * 1000.0,
                values,
                marker="o",
                color=COLORS[method],
                label=METHOD_LABEL[method],
            )
        axis.axhline(55.0, color="#222222", linestyle="--", linewidth=1.1, label="55 mm 2D diagnostic")
        axis.set_title(MOTION_LABEL[motion])
        axis.set_xlabel("Prediction horizon (ms)")
        axis.set_ylabel("P95 camera lateral XY error (mm)")
        axis.legend(loc="best")
    title = "Exact historical pose: motion-model ceiling" if exact else "Current PnP pose: deployable-input comparison"
    fig.suptitle(title)
    save_figure(fig, output, "p95_lateral_xy_exact_pose" if exact else "p95_lateral_xy_current_pose")


def plot_current_ecdf(predictions: pd.DataFrame, output: Path) -> None:
    selected = current_input_rows(predictions)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    for axis, motion in zip(axes.flat, MOTION_ORDER):
        local = selected[
            (selected["motion_mode"] == motion) & np.isclose(selected["horizon_s"], 0.10)
        ]
        for method in PLOT_METHODS:
            values = np.sort(finite(local.loc[local["method"] == method, "camera_lateral_xy_error_m"].values)) * 1000.0
            if not len(values):
                continue
            axis.step(
                values,
                np.arange(1, len(values) + 1) / len(values),
                where="post",
                color=COLORS[method],
                label=f"{METHOD_LABEL[method]} (n={len(values)})",
            )
        axis.axvline(55.0, color="#222222", linestyle="--", linewidth=1.1)
        axis.set_xscale("symlog", linthresh=5.0)
        axis.set_xlim(left=0.0)
        axis.set_ylim(0.0, 1.005)
        axis.set_xlabel("Camera lateral XY error (mm)")
        axis.set_ylabel("Empirical CDF")
        axis.set_title(MOTION_LABEL[motion])
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, loc="lower right")
    fig.suptitle("Complete empirical distributions at 100 ms (current PnP inputs)")
    save_figure(fig, output, "ecdf_lateral_xy_current_pose_100ms")


def plot_input_ablation(predictions: pd.DataFrame, output: Path) -> None:
    selected = predictions[
        (predictions["method"] == "ekf11_persistent_oracle_multislot")
        & np.isclose(predictions["horizon_s"], 0.10)
    ]
    arms = (
        ("current_pnp", "current_pnp_yaw", "Current pos\ncurrent yaw"),
        ("exact_truth", "current_pnp_yaw", "Exact pos\ncurrent yaw"),
        ("current_pnp", "exact_truth_yaw", "Current pos\nexact yaw"),
        ("exact_truth", "exact_truth_yaw", "Exact pos\nexact yaw"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8), sharey=False)
    for axis, motion in zip(axes.flat, MOTION_ORDER):
        local = selected[selected["motion_mode"] == motion]
        values = []
        samples = []
        for position_source, yaw_source, _ in arms:
            arm = local[
                (local["position_source"] == position_source)
                & (local["yaw_source"] == yaw_source)
            ]
            values.append(percentile(arm["camera_lateral_xy_error_m"].values, 95) * 1000.0)
            samples.append(len(arm))
        x = np.arange(len(arms))
        bars = axis.bar(
            x,
            values,
            color=("#D55E00", "#E69F00", "#56B4E9", "#009E73"),
        )
        axis.axhline(55.0, color="#222222", linestyle="--", linewidth=1.1)
        axis.set_xticks(x, [label for _, _, label in arms])
        axis.bar_label(
            bars,
            labels=[f"{value:.1f}\n(n={sample})" for value, sample in zip(values, samples)],
            padding=3,
            fontsize=7,
        )
        axis.margins(y=0.16)
        axis.set_ylabel("P95 camera lateral XY error (mm)")
        axis.set_title(MOTION_LABEL[motion])
    fig.suptitle("11D EKF input ablation at 100 ms: position versus yaw")
    save_figure(fig, output, "ekf11_input_ablation_100ms")


def representative_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    current = current_input_rows(predictions)
    selected: list[pd.DataFrame] = []
    for motion in MOTION_ORDER:
        local = current[(current["motion_mode"] == motion) & np.isclose(current["horizon_s"], 0.10)]
        if not len(local):
            continue
        counts = local.groupby(["session_id", "truth_slot"]).size().sort_values(ascending=False)
        session_id, slot = counts.index[0]
        selected.append(local[(local["session_id"] == session_id) & (local["truth_slot"] == slot)])
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def plot_trajectory_comparison(representatives: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8))
    for axis, motion in zip(axes.flat, MOTION_ORDER):
        local = representatives[representatives["motion_mode"] == motion]
        if not len(local):
            axis.set_visible(False)
            continue
        truth = local.drop_duplicates(["anchor_time_s", "future_time_s"])[
            ["future_time_s", "future_truth_camera_x_m", "future_truth_camera_y_m"]
        ].sort_values("future_time_s")
        axis.plot(
            truth["future_truth_camera_x_m"] * 1000.0,
            truth["future_truth_camera_y_m"] * 1000.0,
            color="#222222",
            label="Future truth",
            zorder=4,
        )
        for method in (
            "cv_ols_camera_16",
            "cv_ols_world_16",
            "ekf11_window16_single_slot",
            "ekf11_persistent_oracle_multislot",
        ):
            arm = local[local["method"] == method].sort_values("future_time_s")
            axis.plot(
                arm["prediction_camera_x_m"] * 1000.0,
                arm["prediction_camera_y_m"] * 1000.0,
                color=COLORS[method],
                alpha=0.85,
                label=METHOD_LABEL[method],
            )
        axis.set_xlabel("Camera x / right (mm)")
        axis.set_ylabel("Camera y / down (mm)")
        axis.set_title(MOTION_LABEL[motion])
        axis.legend(loc="best")
    fig.suptitle("100 ms predicted trajectory versus future truth (representative current-PnP runs)")
    save_figure(fig, output, "trajectory_prediction_truth_current_pose_100ms")


def p95_lookup(
    primary: pd.DataFrame,
    motion: str,
    method: str,
    position_source: str,
    yaw_source: str,
    horizon: float,
) -> tuple[float, int]:
    row = primary[
        (primary["motion_mode"] == motion)
        & (primary["method"] == method)
        & (primary["position_source"] == position_source)
        & (primary["yaw_source"] == yaw_source)
        & np.isclose(primary["horizon_s"], horizon)
    ]
    if not len(row):
        return float("nan"), 0
    return float(row["p95_mm"].iloc[0]), int(row["samples"].iloc[0])


def write_report(
    output: Path,
    primary: pd.DataFrame,
    yaw_diagnostics: pd.DataFrame,
    coordinate_diagnostics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    lines = [
        "# rm_vision-style 11D EKF versus simple CV",
        "",
        "## Experiment contract",
        "",
        "- State: `[xc,vxc,yc,vyc,zc,vzc,theta,omega,r_even,r_odd,dz_odd]`.",
        "- `cv_ols_camera_16` exactly reproduces the prior simple-CV convention in the moving camera frame. `cv_ols_world_16` uses the same 16 target-slot samples in the EKF world frame. They are reported separately and never share an ambiguous `CV` label.",
        "- `cv_ols_world_16` and `ekf11_window16_single_slot` receive the same 16 world-frame samples from the target slot; this isolates estimator form at equal input support.",
        "- Causal-segment EKF arms retain state while observations remain supported; a gap above 120 ms triggers LOST-style reinitialization instead of unsupported blind propagation. The single-slot arm sees only that target slot; the oracle multi-slot arm sees every visible slot and uses truth slot identity.",
        "- `current_pnp` uses measured historical PnP positions; `exact_truth` is an explicit historical-position intervention.",
        "- `current_pnp_yaw` uses the measured armor yaw transformed into world coordinates; `exact_truth_yaw` is a historical-yaw intervention.",
        "- Future truth is used only after prediction for interpolation and scoring. It is never an estimator input.",
        "- Main metric is the user's requested camera lateral vector norm `sqrt(ex^2 + ey^2)`, excluding depth `ez`. Signed x/y and depth errors remain in the sample table.",
        "- The 55 mm line on this 2D norm is a diagnostic stricter than a one-axis yaw-plane gate; it is not a full hit-rate claim.",
        "",
        "## Primary P95 lateral XY results (mm)",
        "",
        "The table below reports current measured inputs. `n` is the number of valid future comparisons, not the number of raw frames.",
        "",
        "| Motion | Horizon | Camera hold | Camera-CV (prior) | World-CV | EKF11 same 16 | EKF11 causal same slot | EKF11 causal oracle multi-slot |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for motion in MOTION_ORDER:
        for horizon in (0.05, 0.10, 0.20):
            values = []
            for method, yaw_source in (
                ("hold_camera", "not_used"),
                ("cv_ols_camera_16", "not_used"),
                ("cv_ols_world_16", "not_used"),
                ("ekf11_window16_single_slot", "current_pnp_yaw"),
                ("ekf11_persistent_single_slot", "current_pnp_yaw"),
                ("ekf11_persistent_oracle_multislot", "current_pnp_yaw"),
            ):
                value, samples = p95_lookup(
                    primary, motion, method, "current_pnp", yaw_source, horizon
                )
                values.append(f"{value:.1f} (n={samples})" if math.isfinite(value) else "NA")
            lines.append(
                f"| {MOTION_LABEL[motion]} | {horizon*1000:.0f} ms | "
                + " | ".join(values)
                + " |"
            )

    lines.extend(
        [
            "",
            "## Exact-input motion-model ceiling",
            "",
            "This arm supplies exact historical position and yaw but still predicts unseen future samples. A large error here is pure model/history/visibility error, not PnP error.",
            "",
        "| Motion | Horizon | Camera-CV exact position | World-CV exact position | EKF11 oracle multi-slot exact pose |",
        "|---|---:|---:|---:|---:|",
        ]
    )
    for motion in MOTION_ORDER:
        for horizon in (0.05, 0.10, 0.20):
            camera_cv, camera_cv_n = p95_lookup(
                primary, motion, "cv_ols_camera_16", "exact_truth", "not_used", horizon
            )
            world_cv, world_cv_n = p95_lookup(
                primary, motion, "cv_ols_world_16", "exact_truth", "not_used", horizon
            )
            ekf, ekf_n = p95_lookup(
                primary,
                motion,
                "ekf11_persistent_oracle_multislot",
                "exact_truth",
                "exact_truth_yaw",
                horizon,
            )
            lines.append(
                f"| {MOTION_LABEL[motion]} | {horizon*1000:.0f} ms | {camera_cv:.1f} (n={camera_cv_n}) | {world_cv:.1f} (n={world_cv_n}) | {ekf:.1f} (n={ekf_n}) |"
            )

    current = current_input_rows(predictions)
    exact = exact_input_rows(predictions)
    lines.extend(
        [
            "",
            "## Validation and limits",
            "",
            f"- Joined measured armor yaw for {int(yaw_diagnostics.joined_rows.sum()):,}/{int(yaw_diagnostics.rows.sum()):,} visible rows ({yaw_diagnostics.joined_rows.sum()/max(yaw_diagnostics.rows.sum(),1):.3%}).",
            f"- World-to-camera rigid-coordinate reconstruction P95 residual across sessions: {percentile(coordinate_diagnostics.fit_residual_p95_m, 95)*1e6:.3f} micrometres (session-P95-of-P95).",
            f"- Current-input prediction rows retained: {len(current):,}; exact-input ceiling rows retained: {len(exact):,}; all intervention rows: {len(predictions):,}.",
            "- Oracle slot identity removes association failure from this comparison. The multi-slot result is therefore an upper-bound diagnostic, not a deployable tracker result.",
            "- This deliberately minimal EKF has no NIS gate, armor-jump recovery, visibility-aware identity logic, acceleration state, or combined-motion-specific predictor. Without the stated 120 ms LOST-style reset, stale single-slot propagation was observed to diverge and is intentionally excluded from the normal comparison.",
            "- Small `n`, especially at long horizons on fast rotation/combined motion, must not be generalized beyond the supported visible arcs.",
            "",
            "## Retained evidence",
            "",
            "- `prediction_samples.csv.gz`: every prediction, signed camera-axis error, 2D lateral norm, depth error, complete state and intervention labels.",
            "- `prediction_metrics.csv`: quantiles through P99 by motion/rate/distance.",
            "- `lateral_xy_histogram_distribution.csv`: fixed-bin full-distribution export; ECDF figures are also retained.",
            "- `joined_visible_observations.csv.gz`: PnP/truth rows plus matched measured and exact yaw.",
            "- `yaw_join_diagnostics.csv`, `coordinate_transform_diagnostics.csv`, and `anchor_coverage.csv`: audit trails for matching, coordinates and sample support.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    print("[1/8] loading visible paired trajectory rows", flush=True)
    frame = load_visible_rows(args.paired_trajectory_rows, args.max_sessions)
    print(f"      {len(frame):,} rows across {frame.session_id.nunique()} sessions", flush=True)

    print("[2/8] joining measured armor yaw from observations.jsonl", flush=True)
    frame, yaw_diagnostics = join_observed_yaw(frame, args.dataset_root)
    write_csv(frame, args.output / "joined_visible_observations.csv.gz")
    write_csv(yaw_diagnostics, args.output / "yaw_join_diagnostics.csv")
    join_rate = float(np.isfinite(frame["observed_yaw_world_rad"]).mean())
    print(f"      yaw join rate: {join_rate:.3%}", flush=True)
    if join_rate < 0.99:
        raise RuntimeError(f"yaw join rate {join_rate:.3%} is below the 99% audit floor")

    print("[3/8] validating per-frame world-to-camera transforms", flush=True)
    coordinate_diagnostics = validate_frame_transforms(frame)
    write_csv(coordinate_diagnostics, args.output / "coordinate_transform_diagnostics.csv")
    coordinate_p95 = float(coordinate_diagnostics["fit_residual_p95_m"].max())
    print(f"      worst session P95 fit residual: {coordinate_p95:.3e} m", flush=True)
    if coordinate_p95 > 1e-5:
        raise RuntimeError(f"coordinate reconstruction residual {coordinate_p95:.3e} m is too large")

    print("[4/8] running hold, OLS-CV and 11D EKF intervention arms", flush=True)
    predictions, coverage = build_predictions(frame)
    write_csv(predictions, args.output / "prediction_samples.csv.gz")
    write_csv(coverage, args.output / "anchor_coverage.csv")
    print(f"      retained {len(predictions):,} prediction rows", flush=True)
    if not len(predictions):
        raise RuntimeError("no prediction rows produced")

    print("[5/8] exporting quantiles and complete fixed-bin distributions", flush=True)
    summary = build_summaries(predictions)
    primary = primary_table(summary)
    histogram = histogram_distribution(predictions)
    write_csv(summary, args.output / "prediction_metrics.csv")
    write_csv(primary, args.output / "primary_p95_lateral_xy_mm.csv")
    write_csv(histogram, args.output / "lateral_xy_histogram_distribution.csv")

    print("[6/8] rendering comparison, ECDF, ablation and trajectory figures", flush=True)
    plot_p95_by_horizon(predictions, args.output, exact=False)
    plot_p95_by_horizon(predictions, args.output, exact=True)
    plot_current_ecdf(predictions, args.output)
    plot_input_ablation(predictions, args.output)
    representatives = representative_rows(predictions)
    write_csv(representatives, args.output / "representative_prediction_trajectories.csv.gz")
    plot_trajectory_comparison(representatives, args.output)

    print("[7/8] writing report", flush=True)
    write_report(args.output, primary, yaw_diagnostics, coordinate_diagnostics, predictions)

    print("[8/8] hashing retained evidence", flush=True)
    manifest = {
        "schema_version": "rmvision-ekf11-ablation-v1",
        "source": {
            "paired_trajectory_rows": str(args.paired_trajectory_rows.resolve()),
            "paired_trajectory_rows_sha256": sha256_file(args.paired_trajectory_rows),
            "dataset_root": str(args.dataset_root.resolve()),
            "sessions": int(frame["session_id"].nunique()),
            "visible_rows": int(len(frame)),
        },
        "reference_urls": REFERENCE_URLS,
        "state": [
            "xc",
            "vxc",
            "yc",
            "vyc",
            "zc",
            "vzc",
            "theta",
            "omega",
            "r_even",
            "r_odd",
            "dz_odd",
        ],
        "parameters": {
            "horizons_s": HORIZONS_S,
            "history_size": HISTORY_SIZE,
            "max_history_span_s": MAX_HISTORY_SPAN_S,
            "max_history_gap_s": MAX_HISTORY_GAP_S,
            "max_future_bracket_s": MAX_FUTURE_BRACKET_S,
            "evaluation_interval_s": EVALUATION_INTERVAL_S,
            "sigma2_q_xyz": SIGMA2_Q_XYZ,
            "sigma2_q_yaw": SIGMA2_Q_YAW,
            "sigma2_q_geometry": SIGMA2_Q_GEOMETRY,
            "r_position_scale": R_POSITION_SCALE,
            "r_yaw": R_YAW,
            "initial_radius_m": INITIAL_RADIUS_M,
            "lateral_metric": "sqrt(camera_error_x^2 + camera_error_y^2), excluding depth z",
        },
        "causality": {
            "future_truth_estimator_input": False,
            "future_truth_use": "interpolation_and_post_prediction_scoring_only",
            "truth_slot_identity": "oracle",
            "exact_truth_arms": "explicit historical-input interventions",
        },
        "outputs": {},
    }
    for path in sorted(args.output.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        manifest["outputs"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    with (args.output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"complete: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
