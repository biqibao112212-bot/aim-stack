#!/usr/bin/env python3
"""Validate static armor world-position invariance under gimbal rotation.

This deterministic check bypasses detector noise. It projects one fixed armor
plate through the configured camera/gimbal extrinsic for multiple gimbal
yaw/pitch states, solves the plate pose with the same planar PnP convention used
by the copied vivsionn AngleSolver, and transforms the solved center back to the
solver world frame. A calibrated transform should recover the same static armor
center for every visible gimbal pose.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_opencv_matrix(text: str, name: str) -> np.ndarray:
    pattern = re.compile(
        rf"{re.escape(name)}:\s*!!opencv-matrix\s*"
        r"rows:\s*(\d+)\s*"
        r"cols:\s*(\d+)\s*"
        r"dt:\s*\w+\s*"
        r"data:\s*\[([^\]]+)\]",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"{name} not found in OpenCV YAML")
    rows = int(match.group(1))
    cols = int(match.group(2))
    values = [float(part.strip()) for part in match.group(3).split(",")]
    return np.array(values, dtype=np.float64).reshape(rows, cols)


def parse_scalar_int(text: str, name: str, default: int = 0) -> int:
    match = re.search(rf"^{re.escape(name)}:\s*(-?\d+)\s*$", text, re.MULTILINE)
    if not match:
        return default
    return int(match.group(1))


def load_params(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return {
        "extrinsic_enabled": bool(parse_scalar_int(text, "CAMERA_GIMBAL_EXTRINSIC_ENABLED")),
        "R_camera2gimbal": parse_opencv_matrix(text, "R_CAMERA2GIMBAL"),
        "t_camera2gimbal_m": parse_opencv_matrix(text, "T_CAMERA2GIMBAL").reshape(3),
        "camera_matrix": parse_opencv_matrix(text, "CAMERA_MATRIX"),
        "distortion": parse_opencv_matrix(text, "RADIAL_DISTORTION").reshape(-1, 1),
    }


def armor_object_points_m(size: str) -> np.ndarray:
    half_width_m = 0.1125 if size == "large" else 0.0675
    half_height_m = 0.0275
    return np.array(
        [
            [half_width_m, -half_height_m, 0.0],
            [half_width_m, half_height_m, 0.0],
            [-half_width_m, half_height_m, 0.0],
            [-half_width_m, -half_height_m, 0.0],
        ],
        dtype=np.float64,
    )


def current_gimbal_pose_rotation(
    r_camera2gimbal: np.ndarray, yaw_rad: float, pitch_rad: float
) -> np.ndarray:
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    r_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    r_pitch = np.array(
        [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64
    )
    r_camera_pose = r_yaw @ r_pitch
    return r_camera2gimbal @ r_camera_pose @ r_camera2gimbal.T


def project_camera_points(
    camera_points: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray
) -> np.ndarray:
    image_points, _ = cv2.projectPoints(
        camera_points.astype(np.float64),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        camera_matrix,
        distortion,
    )
    return image_points.reshape(-1, 2)


def solve_pnp_ippe_finite(
    object_points_m: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    try:
        ok, rvecs, tvecs, _errors = cv2.solvePnPGeneric(
            object_points_m.astype(np.float64),
            image_points.astype(np.float64),
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        ok, rvecs, tvecs = False, [], []

    if ok:
        for rvec, tvec in zip(rvecs, tvecs):
            if not (np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))):
                continue
            reprojected, _ = cv2.projectPoints(
                object_points_m.astype(np.float64), rvec, tvec, camera_matrix, distortion
            )
            reproj_errors = np.linalg.norm(reprojected.reshape(-1, 2) - image_points, axis=1)
            if np.all(np.isfinite(reproj_errors)):
                candidates.append((float(reproj_errors.mean()), rvec, tvec, reproj_errors))

    if not candidates:
        ok, rvec, tvec = cv2.solvePnP(
            object_points_m.astype(np.float64),
            image_points.astype(np.float64),
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok and np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec)):
            reprojected, _ = cv2.projectPoints(
                object_points_m.astype(np.float64), rvec, tvec, camera_matrix, distortion
            )
            reproj_errors = np.linalg.norm(reprojected.reshape(-1, 2) - image_points, axis=1)
            if np.all(np.isfinite(reproj_errors)):
                candidates.append((float(reproj_errors.mean()), rvec, tvec, reproj_errors))

    if not candidates:
        raise RuntimeError("PnP produced no finite solution")

    _mean_error, rvec, tvec, reproj_errors = min(candidates, key=lambda item: item[0])
    return rvec, tvec, reproj_errors


def solve_sample(
    object_points_m: np.ndarray,
    world_center_m: np.ndarray,
    r_object_to_world: np.ndarray,
    r_pose: np.ndarray,
    params: dict,
) -> dict | None:
    r_camera2gimbal = params["R_camera2gimbal"]
    t_camera2gimbal_m = params["t_camera2gimbal_m"]
    camera_matrix = params["camera_matrix"]
    distortion = params["distortion"]

    world_corners = world_center_m + (r_object_to_world @ object_points_m.T).T
    neutral_gimbal_corners = (r_pose.T @ world_corners.T).T
    camera_corners = (r_camera2gimbal.T @ (neutral_gimbal_corners - t_camera2gimbal_m).T).T
    if np.any(camera_corners[:, 2] <= 0.0):
        return None

    image_points = project_camera_points(camera_corners, camera_matrix, distortion)
    width = camera_matrix[0, 2] * 2.0 + 1.0
    height = camera_matrix[1, 2] * 2.0 + 1.0
    if not (
        np.all(image_points[:, 0] >= 4.0)
        and np.all(image_points[:, 0] <= width - 4.0)
        and np.all(image_points[:, 1] >= 4.0)
        and np.all(image_points[:, 1] <= height - 4.0)
    ):
        return None

    try:
        _rvec, tvec, reproj_errors = solve_pnp_ippe_finite(
            object_points_m, image_points, camera_matrix, distortion
        )
    except RuntimeError:
        return None

    camera_center_est = tvec.reshape(3)
    neutral_gimbal_est = r_camera2gimbal @ camera_center_est + t_camera2gimbal_m
    world_est = r_pose @ neutral_gimbal_est
    error_vec = world_est - world_center_m

    return {
        "image_points": image_points.tolist(),
        "camera_center_est_m": camera_center_est.tolist(),
        "world_center_est_m": world_est.tolist(),
        "world_error_m": error_vec.tolist(),
        "world_error_norm_m": float(np.linalg.norm(error_vec)),
        "reprojection_mean_px": float(reproj_errors.mean()),
        "reprojection_max_px": float(reproj_errors.max()),
    }


def run_validation(args: argparse.Namespace) -> dict:
    params = load_params(args.param)
    object_points_m = armor_object_points_m(args.armor_size)
    r_object_to_world = params["R_camera2gimbal"]

    yaw_values = parse_csv_floats(args.yaw_deg)
    pitch_values = parse_csv_floats(args.pitch_deg)
    ranges = parse_csv_floats(args.target_ranges_m)
    lateral_offsets = parse_csv_floats(args.target_lateral_offsets_m)
    height_offsets = parse_csv_floats(args.target_height_offsets_m)

    samples = []
    skipped = 0
    for distance_m in ranges:
        for lateral_m in lateral_offsets:
            for height_m in height_offsets:
                target_center = (
                    params["R_camera2gimbal"]
                    @ np.array([lateral_m, height_m, distance_m], dtype=np.float64)
                    + params["t_camera2gimbal_m"]
                )
                target_samples = []
                for yaw_deg in yaw_values:
                    for pitch_deg in pitch_values:
                        r_pose = current_gimbal_pose_rotation(
                            params["R_camera2gimbal"],
                            math.radians(yaw_deg),
                            math.radians(pitch_deg),
                        )
                        result = solve_sample(
                            object_points_m,
                            target_center,
                            r_object_to_world,
                            r_pose,
                            params,
                        )
                        if result is None:
                            skipped += 1
                            continue
                        result.update(
                            {
                                "yaw_deg": yaw_deg,
                                "pitch_deg": pitch_deg,
                                "target_center_world_m": target_center.tolist(),
                            }
                        )
                        samples.append(result)
                        target_samples.append(result)

    if not samples:
        raise RuntimeError("no visible validation samples were accepted")

    absolute_errors = np.array([s["world_error_norm_m"] for s in samples], dtype=np.float64)
    reproj_max = np.array([s["reprojection_max_px"] for s in samples], dtype=np.float64)

    stability_errors = []
    for target in {
        tuple(np.round(s["target_center_world_m"], 9)) for s in samples
    }:
        estimates = np.array(
            [
                s["world_center_est_m"]
                for s in samples
                if tuple(np.round(s["target_center_world_m"], 9)) == target
            ],
            dtype=np.float64,
        )
        mean_estimate = estimates.mean(axis=0)
        stability_errors.extend(np.linalg.norm(estimates - mean_estimate, axis=1).tolist())
    stability_errors = np.array(stability_errors, dtype=np.float64)

    thresholds = {
        "max_absolute_error_m": args.max_abs_error_m,
        "mean_absolute_error_m": args.mean_abs_error_m,
        "max_stability_error_m": args.max_stability_error_m,
        "max_reprojection_error_px": args.max_reprojection_px,
    }
    metrics = {
        "sample_count": int(len(samples)),
        "skipped_sample_count": int(skipped),
        "absolute_error_mean_m": float(absolute_errors.mean()),
        "absolute_error_p95_m": float(np.percentile(absolute_errors, 95.0)),
        "absolute_error_max_m": float(absolute_errors.max()),
        "stability_error_mean_m": float(stability_errors.mean()),
        "stability_error_p95_m": float(np.percentile(stability_errors, 95.0)),
        "stability_error_max_m": float(stability_errors.max()),
        "reprojection_error_max_px": float(reproj_max.max()),
    }
    status = (
        "pass"
        if params["extrinsic_enabled"]
        and metrics["absolute_error_max_m"] <= thresholds["max_absolute_error_m"]
        and metrics["absolute_error_mean_m"] <= thresholds["mean_absolute_error_m"]
        and metrics["stability_error_max_m"] <= thresholds["max_stability_error_m"]
        and metrics["reprojection_error_max_px"] <= thresholds["max_reprojection_error_px"]
        else "fail"
    )

    return {
        "status": status,
        "method": "detector_bypassed_static_armor_pnp_yaw_pitch_sweep",
        "assumptions": {
            "solver_world_frame": "x forward, y left, z up; gimbal origin fixed at world origin",
            "camera_frame": "OpenCV +x right, +y down, +z forward",
            "armor_pose": (
                "front-facing at neutral gimbal pose; local armor axes are initialized from "
                "the configured camera axes, then kept fixed while the gimbal rotates"
            ),
            "world_coordinate_note": (
                "This validates the relative gimbal/world solver chain. A full Daedalus "
                "absolute world coordinate adds the robot gimbal origin/chassis pose."
            ),
        },
        "configuration": {
            "param": str(args.param),
            "extrinsic_enabled": params["extrinsic_enabled"],
            "R_camera2gimbal": params["R_camera2gimbal"].tolist(),
            "t_camera2gimbal_m": params["t_camera2gimbal_m"].tolist(),
            "camera_matrix": params["camera_matrix"].tolist(),
            "distortion": params["distortion"].reshape(-1).tolist(),
            "armor_size": args.armor_size,
            "yaw_deg": yaw_values,
            "pitch_deg": pitch_values,
            "target_ranges_m": ranges,
            "target_lateral_offsets_m": lateral_offsets,
            "target_height_offsets_m": height_offsets,
            "target_offset_note": (
                "target ranges/lateral/height offsets are specified in the neutral OpenCV "
                "camera frame before conversion to the solver world frame"
            ),
        },
        "thresholds": thresholds,
        "metrics": metrics,
        "samples": samples,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=repo_root)
    parser.add_argument(
        "--param",
        type=Path,
        default=repo_root / "aim_sim_bridge" / "config" / "param.sim.yaml",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root
        / "agent-team"
        / "reports"
        / "static_armor_pose_invariance.json",
    )
    parser.add_argument("--armor-size", choices=["small", "large"], default="small")
    parser.add_argument("--yaw-deg", default="-20,-15,-10,-5,0,5,10,15,20")
    parser.add_argument("--pitch-deg", default="-12,-8,-4,0,4,8,12")
    parser.add_argument("--target-ranges-m", default="2.0,3.0,4.5")
    parser.add_argument("--target-lateral-offsets-m", default="0.0")
    parser.add_argument("--target-height-offsets-m", default="0.0")
    parser.add_argument("--max-abs-error-m", type=float, default=0.01)
    parser.add_argument("--mean-abs-error-m", type=float, default=0.003)
    parser.add_argument("--max-stability-error-m", type=float, default=0.003)
    parser.add_argument("--max-reprojection-px", type=float, default=1.0)
    args = parser.parse_args()

    result = run_validation(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")

    metrics = result["metrics"]
    print(f"status={result['status']}")
    print(f"samples={metrics['sample_count']} skipped={metrics['skipped_sample_count']}")
    print(f"absolute_error_mean_m={metrics['absolute_error_mean_m']:.9g}")
    print(f"absolute_error_p95_m={metrics['absolute_error_p95_m']:.9g}")
    print(f"absolute_error_max_m={metrics['absolute_error_max_m']:.9g}")
    print(f"stability_error_max_m={metrics['stability_error_max_m']:.9g}")
    print(f"reprojection_error_max_px={metrics['reprojection_error_max_px']:.9g}")
    print(f"report={args.report}")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
