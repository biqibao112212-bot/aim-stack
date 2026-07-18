#!/usr/bin/env python3
"""Quantify armor PnP sensitivity to detector corner jitter.

This mirrors the simulator AngleSolver path:
  detector/PnP vertices -> SOLVEPNP_IPPE -> R_CAMERA2GIMBAL/T_CAMERA2GIMBAL
  -> current gimbal-pose rotation -> solver position.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np


def _matrix_from_opencv_yaml(text: str, key: str, rows: int, cols: int) -> np.ndarray:
    pattern = rf"{re.escape(key)}:\s*!!opencv-matrix\s*.*?data:\s*\[([^\]]+)\]"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"missing {key} in OpenCV YAML")
    values = [float(item.strip()) for item in match.group(1).replace("\n", " ").split(",")]
    if len(values) != rows * cols:
        raise ValueError(f"{key} has {len(values)} values, expected {rows * cols}")
    return np.array(values, dtype=np.float64).reshape(rows, cols)


def load_param(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    text = path.read_text(encoding="utf-8")
    camera_matrix = _matrix_from_opencv_yaml(text, "CAMERA_MATRIX", 3, 3)
    distortion = _matrix_from_opencv_yaml(text, "RADIAL_DISTORTION", 5, 1)
    r_camera2gimbal = _matrix_from_opencv_yaml(text, "R_CAMERA2GIMBAL", 3, 3)
    t_camera2gimbal_m = _matrix_from_opencv_yaml(text, "T_CAMERA2GIMBAL", 3, 1)
    return camera_matrix, distortion, r_camera2gimbal, t_camera2gimbal_m.reshape(3)


def load_pipeline(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def armor_object_points_mm(armor_type: str) -> np.ndarray:
    if armor_type.lower() == "large":
        width = 112.5
    else:
        width = 67.5
    points = np.array(
        [
            [0.0, width, -27.5],
            [0.0, width, 27.5],
            [0.0, -width, 27.5],
            [0.0, -width, -27.5],
        ],
        dtype=np.float64,
    )
    return np.stack([points[:, 1], points[:, 2], points[:, 0]], axis=1)


def gimbal_pose_rotation(r_camera2gimbal: np.ndarray, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    r_yaw = np.array(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ],
        dtype=np.float64,
    )
    r_pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float64,
    )
    r_camera_pose = r_yaw @ r_pitch
    return r_camera2gimbal @ r_camera_pose @ r_camera2gimbal.T


def solve_position_m(
    corners: np.ndarray,
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    r_camera2gimbal: np.ndarray,
    t_camera2gimbal_m: np.ndarray,
    pose_rotation: np.ndarray,
) -> np.ndarray:
    ok, _rvec, tvec = cv2.solvePnP(
        object_points.astype(np.float64),
        corners.astype(np.float64),
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed")
    cam_point_mm = np.array(tvec, dtype=np.float64).reshape(3)
    gimbal_point_mm = r_camera2gimbal @ cam_point_mm + t_camera2gimbal_m * 1000.0
    return (pose_rotation @ gimbal_point_mm) / 1000.0


def summarize(values: list[float]) -> dict:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", default="aim_sim_bridge/config/param.sim.yaml")
    parser.add_argument("--corners-from", default="aim_sim_bridge/build/debug/aim_pipeline.json")
    parser.add_argument("--sigmas", default="0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--out", default="agent-team/reports/pnp_corner_sensitivity.json")
    args = parser.parse_args()

    camera_matrix, distortion, r_camera2gimbal, t_camera2gimbal_m = load_param(Path(args.param))
    pipeline = load_pipeline(Path(args.corners_from))
    armor = pipeline.get("first_solved") or {}
    vertices = armor.get("pnp_vertices_px") or armor.get("detector_vertices_px")
    if not vertices:
        raise RuntimeError("no PnP vertices found in pipeline JSON")

    corners = np.array([[float(p["x"]), float(p["y"])] for p in vertices], dtype=np.float64)
    object_points = armor_object_points_mm(str(armor.get("type", "small")))
    yaw_deg = float(pipeline.get("input_gimbal_yaw_deg", 0.0))
    pitch_deg = float(pipeline.get("input_gimbal_pitch_deg", 0.0))
    pose_rotation = gimbal_pose_rotation(r_camera2gimbal, pitch_deg, yaw_deg)
    base_position = solve_position_m(
        corners,
        object_points,
        camera_matrix,
        distortion,
        r_camera2gimbal,
        t_camera2gimbal_m,
        pose_rotation,
    )

    rng = np.random.default_rng(0)
    results = {
        "source": str(Path(args.corners_from)),
        "armor_type": armor.get("type", "small"),
        "base_position_m": {
            "x": float(base_position[0]),
            "y": float(base_position[1]),
            "z": float(base_position[2]),
        },
        "input_gimbal_yaw_deg": yaw_deg,
        "input_gimbal_pitch_deg": pitch_deg,
        "sigmas_px": {},
    }
    for sigma in [float(item) for item in args.sigmas.split(",") if item.strip()]:
        errors = []
        axis_errors = {"x": [], "y": [], "z": []}
        for _ in range(args.samples):
            noisy = corners + rng.normal(0.0, sigma, size=corners.shape)
            pos = solve_position_m(
                noisy,
                object_points,
                camera_matrix,
                distortion,
                r_camera2gimbal,
                t_camera2gimbal_m,
                pose_rotation,
            )
            delta = pos - base_position
            errors.append(float(np.linalg.norm(delta)))
            axis_errors["x"].append(abs(float(delta[0])))
            axis_errors["y"].append(abs(float(delta[1])))
            axis_errors["z"].append(abs(float(delta[2])))
        results["sigmas_px"][str(sigma)] = {
            "position_error_m": summarize(errors),
            "axis_abs_error_m": {key: summarize(vals) for key, vals in axis_errors.items()},
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(
        "pnp_corner_sensitivity "
        f"base=({base_position[0]:.6f},{base_position[1]:.6f},{base_position[2]:.6f}) "
        f"out={out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
