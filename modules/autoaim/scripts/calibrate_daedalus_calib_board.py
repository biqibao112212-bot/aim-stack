#!/usr/bin/env python3
"""Calibrate the simulator camera-to-gimbal extrinsic from Daedalus CALIB.glb.

The script uses the calibration-board asset that the normal Daedalus scene
spawns, detects its checkerboard inner corners from the embedded texture, and
runs a deterministic OpenCV PnP/reprojection validation against simulated board
observations. It is meant to produce a simulator-ground-truth extrinsic for the
copied vivsionn armor pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import cv2
import numpy as np


def quat_to_r(q: list[float]) -> np.ndarray:
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=np.float64,
    )


def rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    magic, version, _length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError(f"{path} is not a GLB v2 file")

    json_chunk = None
    bin_chunk = None
    offset = 12
    while offset < len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == 0x4E4F534A:
            json_chunk = json.loads(chunk.decode("utf-8"))
        elif chunk_type == 0x004E4942:
            bin_chunk = chunk

    if json_chunk is None or bin_chunk is None:
        raise ValueError(f"{path} does not contain JSON and BIN chunks")
    return json_chunk, bin_chunk


def accessor_array(gltf: dict, bin_chunk: bytes, index: int) -> np.ndarray:
    component_types = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }
    type_counts = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
    accessor = gltf["accessors"][index]
    buffer_view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = component_types[accessor["componentType"]]
    count = type_counts[accessor["type"]]
    stride = buffer_view.get("byteStride", np.dtype(dtype).itemsize * count)
    offset = buffer_view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    values = [
        np.frombuffer(bin_chunk, dtype=dtype, count=count, offset=offset + i * stride)
        for i in range(accessor["count"])
    ]
    return np.vstack(values).astype(np.float64)


def embedded_image(gltf: dict, bin_chunk: bytes, image_index: int = 0) -> np.ndarray:
    image = gltf["images"][image_index]
    buffer_view = gltf["bufferViews"][image["bufferView"]]
    offset = buffer_view.get("byteOffset", 0)
    length = buffer_view["byteLength"]
    encoded = np.frombuffer(bin_chunk[offset : offset + length], dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise ValueError("failed to decode embedded calibration-board texture")
    return decoded


def calibration_board_model_points(calib_glb: Path) -> tuple[np.ndarray, dict]:
    gltf, bin_chunk = load_glb(calib_glb)
    node = gltf["nodes"][0]
    primitive = gltf["meshes"][node["mesh"]]["primitives"][0]
    positions = accessor_array(gltf, bin_chunk, primitive["attributes"]["POSITION"])
    texture = embedded_image(gltf, bin_chunk)

    pattern = (10, 7)
    ok, corners = cv2.findChessboardCorners(texture, pattern)
    if not ok:
        raise RuntimeError("OpenCV could not detect the CALIB.glb 10x7 inner corners")
    cv2.cornerSubPix(
        texture,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4),
    )

    height, width = texture.shape
    image_xy = corners.reshape(-1, 2)
    u = image_xy[:, 0] / float(width - 1)
    # The texture's top row maps to glTF v=1 in this asset.
    v = 1.0 - image_xy[:, 1] / float(height - 1)
    min_xyz = positions.min(axis=0)
    max_xyz = positions.max(axis=0)
    mesh_points = np.column_stack(
        [
            min_xyz[0] + u * (max_xyz[0] - min_xyz[0]),
            np.zeros_like(u),
            min_xyz[2] + v * (max_xyz[2] - min_xyz[2]),
        ]
    )

    scale = np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    rotation = quat_to_r(node.get("rotation", [0.0, 0.0, 0.0, 1.0]))
    model_points = (rotation @ (scale[:, None] * mesh_points.T)).T
    grid = model_points.reshape(pattern[1], pattern[0], 3)
    inner_width_m = float(np.linalg.norm(grid[0, -1] - grid[0, 0]))
    inner_height_m = float(np.linalg.norm(grid[-1, 0] - grid[0, 0]))
    return model_points, {
        "pattern_cols": pattern[0],
        "pattern_rows": pattern[1],
        "corner_count": int(model_points.shape[0]),
        "texture_width": int(width),
        "texture_height": int(height),
        "inner_width_m": inner_width_m,
        "inner_height_m": inner_height_m,
        "estimated_square_size_m": float(inner_width_m / (pattern[0] - 1)),
    }


def vehicle_camera_model(vehicle_glb: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gltf, _bin_chunk = load_glb(vehicle_glb)
    nodes_by_name = {node.get("name"): node for node in gltf["nodes"]}
    shot = nodes_by_name["SHOT_DIRECTION"]
    shot_t = np.array(shot["translation"], dtype=np.float64)
    shot_r = quat_to_r(shot["rotation"])
    camera_clearance_m = 0.2
    camera_origin_bevy = shot_t + shot_r @ np.array([0.0, camera_clearance_m, 0.0])
    bevy_camera_r = shot_r @ rz(math.pi / 2.0)

    # Columns are OpenCV camera axes expressed in the raw Bevy gimbal frame.
    r_camera_to_bevy_gimbal = np.column_stack(
        [
            bevy_camera_r @ np.array([1.0, 0.0, 0.0]),
            bevy_camera_r @ np.array([0.0, -1.0, 0.0]),
            bevy_camera_r @ np.array([0.0, 0.0, -1.0]),
        ]
    )
    # Copied vivsionn expects gimbal points as x=forward, y=left, z=up.
    r_bevy_gimbal_to_solver = np.array(
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    r_camera_to_solver = r_bevy_gimbal_to_solver @ r_camera_to_bevy_gimbal
    t_camera_to_solver = r_bevy_gimbal_to_solver @ camera_origin_bevy
    return camera_origin_bevy, r_camera_to_bevy_gimbal, (r_camera_to_solver, t_camera_to_solver)


def look_at_camera_pose(camera_origin: np.ndarray, target: np.ndarray) -> np.ndarray:
    z_axis = target - camera_origin
    z_axis /= np.linalg.norm(z_axis)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def project_points(
    points_world: np.ndarray,
    camera_origin: np.ndarray,
    r_camera_to_world: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    points_camera = (r_camera_to_world.T @ (points_world - camera_origin).T).T
    projected = np.empty((points_camera.shape[0], 2), dtype=np.float64)
    projected[:, 0] = camera_matrix[0, 0] * points_camera[:, 0] / points_camera[:, 2] + camera_matrix[0, 2]
    projected[:, 1] = camera_matrix[1, 1] * points_camera[:, 1] / points_camera[:, 2] + camera_matrix[1, 2]
    return projected


def rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    u, _s, vt = np.linalg.svd(source_centered.T @ target_centered)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = target_mean - r @ source_mean
    return r, t


def run_calibration(repo_root: Path) -> dict:
    daedalus_assets = repo_root / "upstream" / "daedalus" / "assets"
    model_points, board_meta = calibration_board_model_points(daedalus_assets / "CALIB.glb")
    camera_origin_bevy, r_camera_to_bevy_gimbal, true_extrinsic = vehicle_camera_model(
        daedalus_assets / "vehicle.glb"
    )
    r_true, t_true = true_extrinsic

    width, height = 1440, 1080
    fov_y = math.radians(45.0)
    fy = height / (2.0 * math.tan(fov_y / 2.0))
    fx = fy
    camera_matrix = np.array(
        [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros((5, 1), dtype=np.float64)

    board_translations = [
        np.array([1.0, 2.5, 1.0], dtype=np.float64),
        np.array([2.0, 0.5, 2.0], dtype=np.float64),
    ]
    # Camera origins around the lower and upper calibration boards.
    sample_offsets = [
        np.array([-0.45, 0.05, -0.70]),
        np.array([0.35, 0.03, -0.75]),
        np.array([-0.35, 0.30, -0.62]),
        np.array([0.32, -0.18, -0.68]),
        np.array([-0.55, 0.00, -0.90]),
        np.array([0.50, 0.25, -0.85]),
    ]

    all_camera_points = []
    all_solver_points = []
    reprojection_errors = []
    accepted_samples = []

    for board_index, board_t in enumerate(board_translations):
        board_world = board_t + model_points
        board_center = board_world.mean(axis=0)
        for offset in sample_offsets:
            camera_origin = board_center + offset
            r_camera_to_world = look_at_camera_pose(camera_origin, board_center)
            r_gimbal_to_world = r_camera_to_world @ r_camera_to_bevy_gimbal.T
            gimbal_origin = camera_origin - r_gimbal_to_world @ camera_origin_bevy

            image_points = project_points(board_world, camera_origin, r_camera_to_world, camera_matrix)
            if not (
                np.all(image_points[:, 0] > 8.0)
                and np.all(image_points[:, 0] < width - 8.0)
                and np.all(image_points[:, 1] > 8.0)
                and np.all(image_points[:, 1] < height - 8.0)
            ):
                continue

            ok, rvec, tvec = cv2.solvePnP(
                model_points.astype(np.float64),
                image_points.astype(np.float64),
                camera_matrix,
                dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                continue

            r_board_to_camera, _ = cv2.Rodrigues(rvec)
            camera_points = (r_board_to_camera @ model_points.T).T + tvec.reshape(1, 3)
            solver_points = (
                np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
                @ (r_gimbal_to_world.T @ (board_world - gimbal_origin).T)
            ).T
            all_camera_points.append(camera_points)
            all_solver_points.append(solver_points)

            reprojected, _ = cv2.projectPoints(
                model_points.astype(np.float64), rvec, tvec, camera_matrix, dist
            )
            error = np.linalg.norm(reprojected.reshape(-1, 2) - image_points, axis=1)
            reprojection_errors.extend(error.tolist())
            accepted_samples.append(
                {
                    "board_index": board_index,
                    "mean_px": float(error.mean()),
                    "max_px": float(error.max()),
                }
            )

    if not all_camera_points:
        raise RuntimeError("no calibration samples were accepted")

    camera_points = np.vstack(all_camera_points)
    solver_points = np.vstack(all_solver_points)
    r_est, t_est = rigid_transform(camera_points, solver_points)
    fit_error_m = np.linalg.norm((r_est @ camera_points.T).T + t_est - solver_points, axis=1)

    rot_error_deg = math.degrees(
        math.acos(np.clip((np.trace(r_est @ r_true.T) - 1.0) * 0.5, -1.0, 1.0))
    )
    trans_error_m = np.linalg.norm(t_est - t_true)
    reprojection_errors = np.array(reprojection_errors, dtype=np.float64)

    return {
        "status": "pass"
        if reprojection_errors.mean() < 0.25
        and reprojection_errors.max() < 1.0
        and fit_error_m.mean() < 1e-6
        and rot_error_deg < 1e-5
        and trans_error_m < 1e-6
        else "fail",
        "board": board_meta,
        "samples": accepted_samples,
        "sample_count": len(accepted_samples),
        "camera_matrix": camera_matrix.tolist(),
        "R_camera2gimbal": r_est.tolist(),
        "t_camera2gimbal": t_est.tolist(),
        "true_R_camera2gimbal": r_true.tolist(),
        "true_t_camera2gimbal": t_true.tolist(),
        "reprojection_mean_px": float(reprojection_errors.mean()),
        "reprojection_max_px": float(reprojection_errors.max()),
        "rigid_fit_mean_m": float(fit_error_m.mean()),
        "rigid_fit_max_m": float(fit_error_m.max()),
        "rotation_error_deg": float(rot_error_deg),
        "translation_error_m": float(trans_error_m),
        "thresholds": {
            "reprojection_mean_px": 0.25,
            "reprojection_max_px": 1.0,
            "rigid_fit_mean_m": 1e-6,
            "rotation_error_deg": 1e-5,
            "translation_error_m": 1e-6,
        },
    }


def update_param_yaml(param_yaml: Path, result: dict) -> None:
    lines = param_yaml.read_text(encoding="utf-8").splitlines()
    r_data = [value for row in result["R_camera2gimbal"] for value in row]
    t_data = result["t_camera2gimbal"]
    replacement = [
        "CAMERA_GIMBAL_EXTRINSIC_ENABLED: 1",
        "APPLY_AIMING_OFFSET_TO_INTRINSICS: 0",
        "R_CAMERA2GIMBAL: !!opencv-matrix",
        "   rows: 3",
        "   cols: 3",
        "   dt: d",
        "   data: [" + ", ".join(f"{value:.12g}" for value in r_data) + "]",
        "T_CAMERA2GIMBAL: !!opencv-matrix",
        "   rows: 3",
        "   cols: 1",
        "   dt: d",
        "   data: [" + ", ".join(f"{value:.12g}" for value in t_data) + "]",
    ]

    start = next(i for i, line in enumerate(lines) if line.startswith("CAMERA_GIMBAL_EXTRINSIC_ENABLED:"))
    end = start
    while end < len(lines) and not lines[end].startswith("SPIN_DELAY_TIME:"):
        end += 1
    if end == len(lines):
        raise RuntimeError("could not find SPIN_DELAY_TIME after extrinsic block")
    param_yaml.write_text(
        "\n".join(lines[:start] + replacement + lines[end:]) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="workspace root containing upstream/daedalus and aim_sim_bridge",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "agent-team"
        / "reports"
        / "daedalus_calib_board_extrinsic.json",
    )
    parser.add_argument(
        "--update-param",
        action="store_true",
        help="write calibrated extrinsics into aim_sim_bridge/config/param.sim.yaml",
    )
    args = parser.parse_args()

    result = run_calibration(args.workspace)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.update_param:
        update_param_yaml(args.workspace / "aim_sim_bridge" / "config" / "param.sim.yaml", result)

    print(f"status={result['status']}")
    print(f"samples={result['sample_count']}")
    print(f"reprojection_mean_px={result['reprojection_mean_px']:.6g}")
    print(f"reprojection_max_px={result['reprojection_max_px']:.6g}")
    print(f"rigid_fit_mean_m={result['rigid_fit_mean_m']:.6g}")
    print(f"rotation_error_deg={result['rotation_error_deg']:.6g}")
    print(f"translation_error_m={result['translation_error_m']:.6g}")
    print(f"report={args.report}")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
