#!/usr/bin/env python3
"""Audit, align, plot, and score the truth-capable Stage3 trajectory grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STOCK_RADIUS_EVEN_M = 0.21131764
STOCK_RADIUS_ODD_M = 0.21176702


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wrap_pi(value: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(value) + math.pi) % (2.0 * math.pi) - math.pi


def q_rotate(q: list[float], value: list[float]) -> np.ndarray:
    quaternion = np.asarray(q, dtype=float)
    vector = np.asarray(value, dtype=float)
    xyz = quaternion[1:]
    twice_cross = 2.0 * np.cross(xyz, vector)
    return vector + quaternion[0] * twice_cross + np.cross(xyz, twice_cross)


def q_inverse_rotate(q: list[float], value: list[float]) -> np.ndarray:
    quaternion = np.asarray(q, dtype=float)
    return q_rotate([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]], value)


def camera_angles(point_camera: np.ndarray) -> tuple[float, float]:
    z = max(float(point_camera[2]), 1e-6)
    return math.atan2(float(point_camera[0]), z), math.atan2(float(point_camera[1]), z)


def distance(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def count_regressions(records: list[dict], field: str) -> int:
    values = [int(record[field]) for record in records if field in record]
    return sum(current <= previous for previous, current in zip(values, values[1:]))


def select_active_target(truth: dict, requested_distance: float) -> dict | None:
    exposure = truth.get("exposure_state") or {}
    camera = exposure.get("camera_position_world_m")
    if not camera:
        return None
    candidates = [
        target
        for target in (truth.get("ground_truth") or {}).get("targets", [])
        if target.get("armor_label") == 3 and target.get("armor_count") == 4
    ]
    if not candidates:
        return None
    # The native range publishes the controlled robot as another label-3
    # target. The requested rotating target is the farthest complete label-3
    # target from the camera; target_id is intentionally not used across runs.
    return max(candidates, key=lambda target: distance(target["world_position_m"], camera))


def target_slot_points(truth: dict, target: dict) -> dict[int, dict]:
    exposure = truth["exposure_state"]
    camera_position = exposure["camera_position_world_m"]
    camera_quaternion = exposure["camera_quaternion_world_wxyz"]
    target_position = np.asarray(target["world_position_m"], dtype=float)
    target_quaternion = target["world_quaternion_wxyz"]
    result: dict[int, dict] = {}
    for armor in target.get("armors", []):
        world = target_position + q_rotate(target_quaternion, armor["relative_position_m"])
        outward_normal_world = q_rotate(target_quaternion, armor["outward_normal"])
        to_camera = np.asarray(camera_position, dtype=float) - world
        to_camera_norm = float(np.linalg.norm(to_camera))
        facing_score = (
            float(np.dot(outward_normal_world, to_camera / to_camera_norm))
            if to_camera_norm > 1e-9
            else float("nan")
        )
        camera_bevy = q_inverse_rotate(camera_quaternion, world - np.asarray(camera_position, dtype=float))
        # Ground-truth exposure poses use the simulator camera local basis:
        # +X forward, +Y left, +Z up.  Stage3 camera_tvec uses OpenCV
        # [right, down, forward], so both lateral and vertical axes change
        # sign.  The lateral sign is also checked empirically by projecting
        # exposure-matched truth into detector pixels; using +camera_bevy[1]
        # mirrors the target horizontally by tens of pixels.
        camera = np.asarray([-camera_bevy[1], -camera_bevy[2], camera_bevy[0]], dtype=float)
        u, v = camera_angles(camera)
        result[int(armor["relative_slot"])] = {
            "camera_xyz": camera.tolist(),
            "u_rad": u,
            "v_rad": v,
            "u_deg": math.degrees(u),
            "v_deg": math.degrees(v),
            "facing_score": facing_score,
        }
    return result


def target_frame_metadata(truth: dict, target: dict) -> dict:
    """Return truth-only motion covariates for descriptive error analysis.

    These fields are labels/audit metadata. They must never be used as online
    observation-model inputs unless a separate deployable estimator supplies
    an equivalent causal quantity.
    """
    exposure = truth["exposure_state"]
    camera_position = np.asarray(exposure["camera_position_world_m"], dtype=float)
    gimbal_position = np.asarray(exposure["gimbal_position_world_m"], dtype=float)
    camera_quaternion = exposure["camera_quaternion_world_wxyz"]
    target_position = np.asarray(target["world_position_m"], dtype=float)
    target_velocity = np.asarray(target.get("world_velocity_mps", [0.0, 0.0, 0.0]), dtype=float)
    radius_even = float(target.get("radius_even_m", float("nan")))
    radius_odd = float(target.get("radius_odd_m", float("nan")))
    target_camera_bevy = q_inverse_rotate(camera_quaternion, target_position - camera_position)
    target_camera = np.asarray(
        [-target_camera_bevy[1], -target_camera_bevy[2], target_camera_bevy[0]], dtype=float
    )
    target_u, target_v = camera_angles(target_camera)
    return {
        "target_world_x_m": float(target_position[0]),
        "target_world_y_m": float(target_position[1]),
        "target_world_z_m": float(target_position[2]),
        "target_velocity_world_x_mps": float(target_velocity[0]),
        "target_velocity_world_y_mps": float(target_velocity[1]),
        "target_velocity_world_z_mps": float(target_velocity[2]),
        "target_speed_mps": float(np.linalg.norm(target_velocity)),
        "target_distance_camera_m": float(np.linalg.norm(target_position - camera_position)),
        "target_distance_gimbal_m": float(np.linalg.norm(target_position - gimbal_position)),
        "target_radius_even_m": radius_even,
        "target_radius_odd_m": radius_odd,
        "actual_radial_scale": float(
            np.mean([radius_even / STOCK_RADIUS_EVEN_M, radius_odd / STOCK_RADIUS_ODD_M])
        ),
        "target_yaw_rad": float(target["yaw_rad"]),
        "target_vyaw_rad_s": float(target.get("vyaw_rad_s", 0.0)),
        "target_camera_x_m": float(target_camera[0]),
        "target_camera_y_m": float(target_camera[1]),
        "target_camera_z_m": float(target_camera[2]),
        "target_center_u_deg": math.degrees(target_u),
        "target_center_v_deg": math.degrees(target_v),
        "camera_world_x_m": float(camera_position[0]),
        "camera_world_y_m": float(camera_position[1]),
        "camera_world_z_m": float(camera_position[2]),
    }


def make_key(record: dict) -> tuple[str, int, int, int]:
    return (
        str(record.get("session_id", "")),
        int(record["producer_epoch"]),
        int(record["frame_seq"]),
        int(record["timestamp_ns"]),
    )


def assign_observations_angular(observation: dict, projected: dict[int, dict]) -> list[dict]:
    """Legacy audit assignment using only camera azimuth/elevation.

    This is retained for exact reproduction of the first grid analysis.  It is
    not a safe physical-slot assignment because opposite plates can project to
    nearly the same camera ray while having different PnP depths.
    """
    candidates = []
    for detection_index, armor in enumerate(observation.get("armors", [])):
        tvec = armor.get("camera_tvec_m")
        if not tvec or len(tvec) != 3 or not all(math.isfinite(float(v)) for v in tvec):
            continue
        u, v = camera_angles(np.asarray(tvec, dtype=float))
        for slot, point in projected.items():
            du = float(wrap_pi(u - point["u_rad"]))
            dv = v - point["v_rad"]
            candidates.append((du * du + dv * dv, slot, detection_index, u, v, point["u_deg"], point["v_deg"], armor))
    assigned: list[dict] = []
    used_slots: set[int] = set()
    used_detections: set[int] = set()
    for cost, slot, detection_index, u, v, truth_u_deg, truth_v_deg, armor in sorted(candidates, key=lambda item: item[0]):
        if slot in used_slots or detection_index in used_detections:
            continue
        if math.sqrt(cost) > math.radians(8.0):
            continue
        used_slots.add(slot)
        used_detections.add(detection_index)
        assigned.append(
            {
                "slot": slot,
                "detection_index": detection_index,
                "detector_number": armor.get("detector_number"),
                "u_deg": math.degrees(u),
                "v_deg": math.degrees(v),
                "truth_u_deg": truth_u_deg,
                "truth_v_deg": truth_v_deg,
                "match_error_deg": math.degrees(math.sqrt(cost)),
            }
        )
    return assigned


def assign_observations_pnp3d(
    observation: dict,
    projected: dict[int, dict],
    max_position_error_m: float,
    allowed_slots: set[int] | None = None,
) -> list[dict]:
    """Assign detections to physical slots using the complete PnP translation.

    The cost is Euclidean distance in the shared OpenCV camera frame.  A
    one-to-one greedy assignment is sufficient here (normally one active-target
    detection per frame), while the explicit distance gate rejects detections
    from the other native-range targets.  Legacy angular assignments are kept
    on every row so slot aliases remain auditable rather than disappearing.
    """
    angular_by_detection = {
        int(row["detection_index"]): int(row["slot"])
        for row in assign_observations_angular(observation, projected)
    }
    candidates: list[tuple] = []
    per_detection_costs: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for detection_index, armor in enumerate(observation.get("armors", [])):
        tvec = armor.get("camera_tvec_m")
        if not tvec or len(tvec) != 3 or not all(math.isfinite(float(value)) for value in tvec):
            continue
        camera_tvec = np.asarray(tvec, dtype=float)
        u, v = camera_angles(camera_tvec)
        for slot, point in projected.items():
            if allowed_slots is not None and slot not in allowed_slots:
                continue
            truth_camera = np.asarray(point["camera_xyz"], dtype=float)
            position_error = float(np.linalg.norm(camera_tvec - truth_camera))
            du = float(wrap_pi(u - point["u_rad"]))
            dv = float(v - point["v_rad"])
            angular_error = math.degrees(math.hypot(du, dv))
            candidates.append(
                (
                    position_error,
                    slot,
                    detection_index,
                    u,
                    v,
                    angular_error,
                    point,
                    camera_tvec,
                    armor,
                )
            )
            per_detection_costs[detection_index].append((position_error, slot))

    assigned: list[dict] = []
    used_slots: set[int] = set()
    used_detections: set[int] = set()
    for position_error, slot, detection_index, u, v, angular_error, point, camera_tvec, armor in sorted(
        candidates, key=lambda item: item[0]
    ):
        if slot in used_slots or detection_index in used_detections:
            continue
        if position_error > max_position_error_m:
            continue
        alternatives = sorted(per_detection_costs[detection_index])
        second_error = alternatives[1][0] if len(alternatives) > 1 else None
        angular_slot = angular_by_detection.get(detection_index)
        used_slots.add(slot)
        used_detections.add(detection_index)
        assigned.append(
            {
                "slot": slot,
                "detection_index": detection_index,
                "detector_number": armor.get("detector_number"),
                "detector_type": armor.get("detector_type"),
                "u_deg": math.degrees(u),
                "v_deg": math.degrees(v),
                "truth_u_deg": point["u_deg"],
                "truth_v_deg": point["v_deg"],
                "match_error_deg": angular_error,
                "pnp_position_error_m": position_error,
                "pnp_depth_error_m": abs(float(camera_tvec[2]) - float(point["camera_xyz"][2])),
                "pnp_second_position_error_m": second_error,
                "pnp_assignment_margin_m": second_error - position_error if second_error is not None else None,
                "pnp_assignment_margin_ratio": second_error / max(position_error, 1e-9) if second_error is not None else None,
                "pnp_max_position_error_m": max_position_error_m,
                "pnp_camera_x_m": float(camera_tvec[0]),
                "pnp_camera_y_m": float(camera_tvec[1]),
                "pnp_camera_z_m": float(camera_tvec[2]),
                "truth_camera_x_m": float(point["camera_xyz"][0]),
                "truth_camera_y_m": float(point["camera_xyz"][1]),
                "truth_camera_z_m": float(point["camera_xyz"][2]),
                "truth_facing_score": float(point.get("facing_score", float("nan"))),
                "pnp_yaw_absolute_rad": armor.get("yaw_absolute_rad"),
                "pnp_reprojection_rms_px": armor.get("reprojection_rms_px"),
                "pnp_reprojection_max_px": armor.get("reprojection_max_px"),
                "angular_assignment_slot": angular_slot,
                "slot_changed_from_angular": angular_slot is not None and angular_slot != slot,
            }
        )
    return assigned


def assign_observations_angular_facing(
    observation: dict,
    projected: dict[int, dict],
    max_angular_error_deg: float,
) -> list[dict]:
    """Assign by image ray among every physically front-facing truth plate.

    Long-range planar-PnP depth error can exceed the spacing between armor
    slots. The image ray remains tied to the detected plate center, so it is
    used for identity while PnP translation remains an audited measurement.
    No top-N facing rank is imposed and second-best angular margins are kept.
    """
    candidates: list[tuple] = []
    per_detection_costs: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for detection_index, armor in enumerate(observation.get("armors", [])):
        tvec = armor.get("camera_tvec_m")
        if not tvec or len(tvec) != 3 or not all(math.isfinite(float(value)) for value in tvec):
            continue
        camera_tvec = np.asarray(tvec, dtype=float)
        u, v = camera_angles(camera_tvec)
        for slot, point in projected.items():
            facing_score = float(point.get("facing_score", float("nan")))
            if not math.isfinite(facing_score) or facing_score <= 0.0:
                continue
            du = float(wrap_pi(u - point["u_rad"]))
            dv = float(v - point["v_rad"])
            angular_error_deg = math.degrees(math.hypot(du, dv))
            truth_camera = np.asarray(point["camera_xyz"], dtype=float)
            position_error = float(np.linalg.norm(camera_tvec - truth_camera))
            candidates.append(
                (
                    angular_error_deg,
                    slot,
                    detection_index,
                    u,
                    v,
                    point,
                    camera_tvec,
                    position_error,
                    armor,
                )
            )
            per_detection_costs[detection_index].append((angular_error_deg, slot))

    assigned: list[dict] = []
    used_slots: set[int] = set()
    used_detections: set[int] = set()
    for angular_error_deg, slot, detection_index, u, v, point, camera_tvec, position_error, armor in sorted(
        candidates, key=lambda item: item[0]
    ):
        if slot in used_slots or detection_index in used_detections:
            continue
        if angular_error_deg > max_angular_error_deg:
            continue
        alternatives = sorted(per_detection_costs[detection_index])
        second_error = alternatives[1][0] if len(alternatives) > 1 else None
        used_slots.add(slot)
        used_detections.add(detection_index)
        assigned.append(
            {
                "slot": slot,
                "detection_index": detection_index,
                "detector_number": armor.get("detector_number"),
                "detector_type": armor.get("detector_type"),
                "u_deg": math.degrees(u),
                "v_deg": math.degrees(v),
                "truth_u_deg": point["u_deg"],
                "truth_v_deg": point["v_deg"],
                "match_error_deg": angular_error_deg,
                "assignment_cost_deg": angular_error_deg,
                "assignment_second_cost_deg": second_error,
                "assignment_margin_deg": second_error - angular_error_deg if second_error is not None else None,
                "assignment_max_error_deg": max_angular_error_deg,
                "pnp_position_error_m": position_error,
                "pnp_depth_error_m": abs(float(camera_tvec[2]) - float(point["camera_xyz"][2])),
                "pnp_second_position_error_m": None,
                "pnp_assignment_margin_m": None,
                "pnp_assignment_margin_ratio": None,
                "pnp_max_position_error_m": None,
                "pnp_camera_x_m": float(camera_tvec[0]),
                "pnp_camera_y_m": float(camera_tvec[1]),
                "pnp_camera_z_m": float(camera_tvec[2]),
                "truth_camera_x_m": float(point["camera_xyz"][0]),
                "truth_camera_y_m": float(point["camera_xyz"][1]),
                "truth_camera_z_m": float(point["camera_xyz"][2]),
                "truth_facing_score": float(point["facing_score"]),
                "pnp_yaw_absolute_rad": armor.get("yaw_absolute_rad"),
                "pnp_reprojection_rms_px": armor.get("reprojection_rms_px"),
                "pnp_reprojection_max_px": armor.get("reprojection_max_px"),
                "angular_assignment_slot": slot,
                "slot_changed_from_angular": False,
            }
        )
    return assigned


def assign_observations(observation: dict, projected: dict[int, dict]) -> list[dict]:
    """Compatibility alias for the legacy published analysis."""
    return assign_observations_angular(observation, projected)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def periodic_mean_predict(train_phase: np.ndarray, train_value: np.ndarray, query_phase: np.ndarray) -> np.ndarray:
    bins = 128
    normalized = ((train_phase + math.pi) % (2.0 * math.pi)) / (2.0 * math.pi)
    indices = np.minimum((normalized * bins).astype(int), bins - 1)
    means = []
    centers = []
    for index in range(bins):
        values = train_value[indices == index]
        if values.size:
            means.append(float(np.mean(values)))
            centers.append(-math.pi + (index + 0.5) * 2.0 * math.pi / bins)
    if not means:
        return np.full_like(query_phase, float(np.mean(train_value)))
    centers_array = np.asarray(centers)
    means_array = np.asarray(means)
    order = np.argsort(centers_array)
    centers_array = centers_array[order]
    means_array = means_array[order]
    xp = np.concatenate([centers_array[-1:] - 2.0 * math.pi, centers_array, centers_array[:1] + 2.0 * math.pi])
    fp = np.concatenate([means_array[-1:], means_array, means_array[:1]])
    query = ((query_phase + math.pi) % (2.0 * math.pi)) - math.pi
    return np.interp(query, xp, fp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--assignment-mode",
        choices=("angular", "angular_facing", "pnp_3d", "pnp_3d_facing"),
        default="angular",
        help="physical-slot assignment; angular_facing uses the image ray among every front-facing plate",
    )
    parser.add_argument("--angular-assignment-gate-deg", type=float, default=0.75)
    parser.add_argument("--pnp-position-error-floor-m", type=float, default=0.5)
    parser.add_argument("--pnp-position-error-depth-ratio", type=float, default=0.30)
    parser.add_argument(
        "--include-radial-scale",
        action="append",
        type=float,
        help="accept only this requested/verified radial scale; repeat for multiple scales",
    )
    args = parser.parse_args()
    if not math.isfinite(args.angular_assignment_gate_deg) or args.angular_assignment_gate_deg <= 0.0:
        parser.error("--angular-assignment-gate-deg must be finite and positive")
    roots = [path.resolve() for path in args.root]
    output = (args.output or roots[0] / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)

    truth_points: list[dict] = []
    observed_points: list[dict] = []
    run_rows: list[dict] = []
    exact_join_total = 0
    truth_target_failures = 0
    observation_sequence_regressions = 0
    observation_timestamp_regressions = 0
    truth_sequence_regressions = 0
    truth_timestamp_regressions = 0
    observation_duplicate_keys = 0
    truth_duplicate_keys = 0

    run_dirs = sorted(
        path
        for root in roots
        for path in root.iterdir()
        if path.is_dir()
    )
    for run_dir in run_dirs:
        manifest_path = run_dir / "collection_run_manifest.json"
        truth_path = run_dir / "truth.jsonl"
        observation_path = run_dir / "stage3_observations.jsonl"
        summary_path = run_dir / "summary.json"
        if not all(path.exists() for path in (manifest_path, truth_path, observation_path, summary_path)):
            continue
        manifest = read_json(manifest_path)
        summary = read_json(summary_path)
        scale = float(manifest["radial_scale"])
        if args.include_radial_scale and not any(abs(scale - requested) < 1e-6 for requested in args.include_radial_scale):
            continue
        requested_distance = float(manifest["requested_distance_m"])
        repeat = int(manifest["repeat"])
        truths = read_jsonl(truth_path)
        observations = read_jsonl(observation_path)
        truth_sequence_regressions += count_regressions(truths, "frame_seq")
        truth_timestamp_regressions += count_regressions(truths, "timestamp_ns")
        observation_sequence_regressions += count_regressions(observations, "frame_seq")
        observation_timestamp_regressions += count_regressions(observations, "timestamp_ns")
        truth_map = {make_key(record): record for record in truths}
        truth_duplicate_keys += len(truths) - len({make_key(record) for record in truths})
        observation_duplicate_keys += len(observations) - len({make_key(record) for record in observations})
        first_timestamp = min((int(record["timestamp_ns"]) for record in truths), default=0)
        first_yaw_by_target: dict[int, float] = {}
        run_truth_count = 0
        run_observed_count = 0
        for truth in truths:
            target = select_active_target(truth, requested_distance)
            if target is None:
                truth_target_failures += 1
                continue
            target_id = int(target["target_id"])
            if target_id not in first_yaw_by_target:
                first_yaw_by_target[target_id] = float(target["yaw_rad"])
            phase = float(wrap_pi(float(target["yaw_rad"]) - first_yaw_by_target[target_id]))
            slots = target_slot_points(truth, target)
            target_metadata = target_frame_metadata(truth, target)
            elapsed = (int(truth["timestamp_ns"]) - first_timestamp) * 1e-9
            for slot, point in slots.items():
                truth_points.append(
                    {
                        "run": run_dir.name,
                        "scale": scale,
                        "distance_m": requested_distance,
                        "repeat": repeat,
                        "t_s": elapsed,
                        "phase_rad": phase,
                        "slot": slot,
                        "session_id": str(truth.get("session_id", "")),
                        "producer_epoch": int(truth["producer_epoch"]),
                        "frame_seq": int(truth["frame_seq"]),
                        "timestamp_ns": int(truth["timestamp_ns"]),
                        **target_metadata,
                        "truth_facing_score": point["facing_score"],
                        "u_deg": point["u_deg"],
                        "v_deg": point["v_deg"],
                        "camera_x_m": point["camera_xyz"][0],
                        "camera_y_m": point["camera_xyz"][1],
                        "camera_z_m": point["camera_xyz"][2],
                    }
                )
                run_truth_count += 1
        truth_keys = set(truth_map)
        for observation in observations:
            truth = truth_map.get(make_key(observation))
            if truth is None:
                continue
            exact_join_total += 1
            target = select_active_target(truth, requested_distance)
            if target is None:
                continue
            target_id = int(target["target_id"])
            first_yaw = first_yaw_by_target.get(target_id, float(target["yaw_rad"]))
            phase = float(wrap_pi(float(target["yaw_rad"]) - first_yaw))
            elapsed = (int(truth["timestamp_ns"]) - first_timestamp) * 1e-9
            projected = target_slot_points(truth, target)
            target_metadata = target_frame_metadata(truth, target)
            target_depth_m = float(np.median([point["camera_xyz"][2] for point in projected.values()]))
            max_position_error_m = max(
                float(args.pnp_position_error_floor_m),
                float(args.pnp_position_error_depth_ratio) * max(target_depth_m, 0.0),
            )
            if args.assignment_mode == "pnp_3d_facing":
                active_detection_count = 0
                for armor in observation.get("armors", []):
                    tvec = armor.get("camera_tvec_m")
                    if not tvec or len(tvec) != 3 or not all(math.isfinite(float(value)) for value in tvec):
                        continue
                    camera_tvec = np.asarray(tvec, dtype=float)
                    nearest = min(
                        float(np.linalg.norm(camera_tvec - np.asarray(point["camera_xyz"], dtype=float)))
                        for point in projected.values()
                    )
                    active_detection_count += int(nearest <= max_position_error_m)
                visible_count = min(max(active_detection_count, 1), 2)
                allowed_slots = {
                    slot
                    for slot, _ in sorted(
                        ((slot, float(point["facing_score"])) for slot, point in projected.items()),
                        key=lambda item: (-item[1], item[0]),
                    )[:visible_count]
                }
                assignments = assign_observations_pnp3d(
                    observation, projected, max_position_error_m, allowed_slots=allowed_slots
                )
            elif args.assignment_mode == "pnp_3d":
                assignments = assign_observations_pnp3d(observation, projected, max_position_error_m)
            elif args.assignment_mode == "angular_facing":
                assignments = assign_observations_angular_facing(
                    observation, projected, float(args.angular_assignment_gate_deg)
                )
            else:
                assignments = assign_observations_angular(observation, projected)
            for assigned in assignments:
                observed_points.append(
                    {
                        "run": run_dir.name,
                        "scale": scale,
                        "distance_m": requested_distance,
                        "repeat": repeat,
                        "t_s": elapsed,
                        "phase_rad": phase,
                        **target_metadata,
                        "observation_armor_count": len(observation.get("armors", [])),
                        "gimbal_yaw_deg": observation.get("gimbal_yaw_deg"),
                        "gimbal_pitch_deg": observation.get("gimbal_pitch_deg"),
                        "session_id": str(observation.get("session_id", "")),
                        "producer_epoch": int(observation["producer_epoch"]),
                        "frame_seq": int(observation["frame_seq"]),
                        "timestamp_ns": int(observation["timestamp_ns"]),
                        **assigned,
                    }
                )
                run_observed_count += 1
        run_rows.append(
            {
                "run": run_dir.name,
                "scale": scale,
                "distance_m": requested_distance,
                "repeat": repeat,
                "truth_rows": len(truths),
                "observation_rows": len(observations),
                "truth_slot_points": run_truth_count,
                "observed_assignments": run_observed_count,
                "exact_join_rows": len(truth_keys.intersection({make_key(record) for record in observations})),
                "processed_fps": summary.get("processed_fps"),
                "source_sequence_gaps": summary.get("source_sequence_gaps"),
            }
        )

    with (output / "truth_points.jsonl").open("w", encoding="utf-8") as handle:
        for row in truth_points:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "observed_points.jsonl").open("w", encoding="utf-8") as handle:
        for row in observed_points:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metric_rows: list[dict] = []
    grouped_truth: dict[tuple[float, float, int], list[dict]] = defaultdict(list)
    grouped_observed: dict[tuple[float, float, int], list[dict]] = defaultdict(list)
    for row in truth_points:
        grouped_truth[(row["scale"], row["distance_m"], row["slot"])].append(row)
    for row in observed_points:
        grouped_observed[(row["scale"], row["distance_m"], row["slot"])].append(row)
    for key, rows in sorted(grouped_truth.items()):
        scale, distance_m, slot = key
        phases = np.asarray([row["phase_rad"] for row in rows])
        u = np.asarray([row["u_deg"] for row in rows])
        v = np.asarray([row["v_deg"] for row in rows])
        bins = 64
        phase_bin = np.minimum((((phases + math.pi) % (2.0 * math.pi)) / (2.0 * math.pi) * bins).astype(int), bins - 1)
        means_u = np.asarray([np.mean(u[phase_bin == index]) if np.any(phase_bin == index) else np.nan for index in range(bins)])
        means_v = np.asarray([np.mean(v[phase_bin == index]) if np.any(phase_bin == index) else np.nan for index in range(bins)])
        residuals = np.asarray(
            [math.hypot(u_value - means_u[index], v_value - means_v[index]) for u_value, v_value, index in zip(u, v, phase_bin) if math.isfinite(means_u[index])]
        )
        observed = grouped_observed.get(key, [])
        metric_rows.append(
            {
                "scale": scale,
                "distance_m": distance_m,
                "slot": slot,
                "truth_points": len(rows),
                "observed_points": len(observed),
                "observed_rate": len(observed) / len(rows) if rows else 0.0,
                "phase_coverage": float(np.count_nonzero(np.isfinite(means_u)) / bins),
                "median_curve_residual_deg": float(np.median(residuals)) if residuals.size else None,
                "p95_curve_residual_deg": float(np.percentile(residuals, 95)) if residuals.size else None,
                "observed_match_p95_deg": float(np.percentile([row["match_error_deg"] for row in observed], 95)) if observed else None,
                "u_span_deg": float(np.max(u) - np.min(u)),
                "v_span_deg": float(np.max(v) - np.min(v)),
            }
        )

    prediction_rows: list[dict] = []
    condition_runs: dict[tuple[float, float], list[str]] = defaultdict(list)
    for row in run_rows:
        condition_runs[(row["scale"], row["distance_m"])].append(row["run"])
    by_run_slot: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in truth_points:
        by_run_slot[(row["run"], row["slot"])].append(row)
    for condition, runs in sorted(condition_runs.items()):
        scale, distance_m = condition
        test_run = sorted(runs)[-1]
        train_runs = [run for run in sorted(runs) if run != test_run]
        for slot in range(4):
            train = [row for run in train_runs for row in by_run_slot[(run, slot)]]
            test = by_run_slot[(test_run, slot)]
            if not train or not test:
                continue
            train_phase = np.asarray([row["phase_rad"] for row in train])
            train_u = np.asarray([row["u_deg"] for row in train])
            train_v = np.asarray([row["v_deg"] for row in train])
            test_phase = np.asarray([row["phase_rad"] for row in test])
            pred_u = periodic_mean_predict(train_phase, train_u, test_phase)
            pred_v = periodic_mean_predict(train_phase, train_v, test_phase)
            actual_u = np.asarray([row["u_deg"] for row in test])
            actual_v = np.asarray([row["v_deg"] for row in test])
            errors = np.hypot(np.asarray(wrap_pi(np.radians(actual_u - pred_u))) * 180.0 / math.pi, actual_v - pred_v)
            prediction_rows.append(
                {
                    "scale": scale,
                    "distance_m": distance_m,
                    "slot": slot,
                    "train_runs": ",".join(train_runs),
                    "test_run": test_run,
                    "test_points": len(test),
                    "mean_error_deg": float(np.mean(errors)),
                    "median_error_deg": float(np.median(errors)),
                    "p95_error_deg": float(np.percentile(errors, 95)),
                }
            )

    causal_prediction_rows: list[dict] = []
    for horizon_s in (0.05, 0.10, 0.20):
        by_condition_slot: dict[tuple[float, float, int], list[float]] = defaultdict(list)
        for (run, slot), rows in by_run_slot.items():
            ordered = sorted(rows, key=lambda row: row["t_s"])
            if len(ordered) < 8:
                continue
            times = np.asarray([row["t_s"] for row in ordered], dtype=float)
            u_rad = np.unwrap(np.radians(np.asarray([row["u_deg"] for row in ordered], dtype=float)))
            v_rad = np.radians(np.asarray([row["v_deg"] for row in ordered], dtype=float))
            scale = float(ordered[0]["scale"])
            distance_m = float(ordered[0]["distance_m"])
            for index in range(5, len(ordered)):
                target_time = times[index] + horizon_s
                if target_time > times[-1]:
                    break
                history_start = max(0, index - 5)
                history_t = times[history_start : index + 1] - times[index]
                design = np.column_stack([history_t, np.ones_like(history_t)])
                u_slope, u_intercept = np.linalg.lstsq(design, u_rad[history_start : index + 1], rcond=None)[0]
                v_slope, v_intercept = np.linalg.lstsq(design, v_rad[history_start : index + 1], rcond=None)[0]
                predicted_u = u_slope * horizon_s + u_intercept
                predicted_v = v_slope * horizon_s + v_intercept
                actual_u = float(np.interp(target_time, times, u_rad))
                actual_v = float(np.interp(target_time, times, v_rad))
                error_deg = math.degrees(math.hypot(float(wrap_pi(actual_u - predicted_u)), actual_v - predicted_v))
                by_condition_slot[(scale, distance_m, slot)].append(error_deg)
        for (scale, distance_m, slot), errors in sorted(by_condition_slot.items()):
            values = np.asarray(errors, dtype=float)
            causal_prediction_rows.append(
                {
                    "scale": scale,
                    "distance_m": distance_m,
                    "slot": slot,
                    "horizon_s": horizon_s,
                    "samples": int(values.size),
                    "mean_error_deg": float(np.mean(values)),
                    "median_error_deg": float(np.median(values)),
                    "p95_error_deg": float(np.percentile(values, 95)),
                }
            )

    write_csv(output / "run_metrics.csv", run_rows, list(run_rows[0].keys()))
    write_csv(output / "condition_slot_metrics.csv", metric_rows, list(metric_rows[0].keys()))
    if prediction_rows:
        write_csv(output / "historical_repeat_prediction.csv", prediction_rows, list(prediction_rows[0].keys()))
    if causal_prediction_rows:
        write_csv(output / "causal_constant_velocity_prediction.csv", causal_prediction_rows, list(causal_prediction_rows[0].keys()))

    scales = sorted({row["scale"] for row in metric_rows})
    distances = sorted({row["distance_m"] for row in metric_rows})
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    fig, axes = plt.subplots(len(scales), len(distances), figsize=(4.0 * len(distances), 3.2 * len(scales)), squeeze=False, sharex=False, sharey=False)
    for row_index, scale in enumerate(scales):
        for col_index, distance_m in enumerate(distances):
            axis = axes[row_index][col_index]
            for slot, color in enumerate(colors):
                points = [point for point in truth_points if point["scale"] == scale and point["distance_m"] == distance_m and point["slot"] == slot]
                for run in sorted({point["run"] for point in points}):
                    run_points = sorted((point for point in points if point["run"] == run), key=lambda point: point["phase_rad"])
                    axis.plot([p["u_deg"] for p in run_points], [p["v_deg"] for p in run_points], color=color, alpha=0.18, linewidth=0.7)
                if points:
                    phase = np.asarray([point["phase_rad"] for point in points])
                    u = np.asarray([point["u_deg"] for point in points])
                    v = np.asarray([point["v_deg"] for point in points])
                    order = np.argsort(phase)
                    axis.plot(u[order], v[order], color=color, linewidth=1.6, label=f"slot {slot}")
            axis.set_title(f"scale={scale:g}, d={distance_m:g} m")
            axis.set_xlabel("camera azimuth (deg)")
            axis.set_ylabel("camera elevation (deg)")
            axis.grid(alpha=0.2)
            if row_index == 0 and col_index == 0:
                axis.legend(fontsize=8, loc="best")
    fig.suptitle("Truth trajectories of four physical armor slots", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output / "trajectory_grid_truth.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(scales), len(distances), figsize=(4.0 * len(distances), 3.2 * len(scales)), squeeze=False)
    for row_index, scale in enumerate(scales):
        for col_index, distance_m in enumerate(distances):
            axis = axes[row_index][col_index]
            for slot, color in enumerate(colors):
                points = [point for point in observed_points if point["scale"] == scale and point["distance_m"] == distance_m and point["slot"] == slot]
                if points:
                    axis.scatter([p["u_deg"] for p in points], [p["v_deg"] for p in points], s=2, alpha=0.25, color=color, label=f"slot {slot}")
            axis.set_title(f"scale={scale:g}, d={distance_m:g} m")
            axis.set_xlabel("camera azimuth (deg)")
            axis.set_ylabel("camera elevation (deg)")
            axis.grid(alpha=0.2)
            if row_index == 0 and col_index == 0:
                axis.legend(fontsize=8, loc="best")
    fig.suptitle("Observed armor detections assigned to physical slots by truth geometry", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output / "trajectory_grid_observed.png", dpi=180)
    plt.close(fig)

    condition_p95: dict[tuple[float, float], float] = {}
    for row in metric_rows:
        value = row["p95_curve_residual_deg"]
        if value is not None:
            condition_p95.setdefault((row["scale"], row["distance_m"]), []).append(value)
    heatmap = np.full((len(scales), len(distances)), np.nan)
    for i, scale in enumerate(scales):
        for j, distance_m in enumerate(distances):
            values = condition_p95.get((scale, distance_m), [])
            if values:
                heatmap[i, j] = float(np.mean(values))
    fig, axis = plt.subplots(figsize=(8, 4))
    image = axis.imshow(heatmap, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(distances)), [str(value) for value in distances])
    axis.set_yticks(range(len(scales)), [str(value) for value in scales])
    axis.set_xlabel("requested distance (m)")
    axis.set_ylabel("radial scale")
    axis.set_title("P95 within-condition trajectory residual (deg)")
    for i in range(len(scales)):
        for j in range(len(distances)):
            if math.isfinite(heatmap[i, j]):
                axis.text(j, i, f"{heatmap[i, j]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=axis, label="P95 residual (deg)")
    fig.tight_layout()
    fig.savefig(output / "trajectory_concentration_heatmap.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    for horizon_s in (0.05, 0.10, 0.20):
        values = [row["p95_error_deg"] for row in causal_prediction_rows if row["horizon_s"] == horizon_s]
        if values:
            axis.scatter([horizon_s] * len(values), values, alpha=0.35, s=16)
            axis.plot(horizon_s, float(np.median(values)), "o", markersize=7, label=f"{int(horizon_s * 1000)} ms median")
    axis.set_xlabel("prediction horizon")
    axis.set_ylabel("P95 angular error (deg)")
    axis.set_title("Causal constant-velocity baseline on truth trajectories")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "causal_prediction_error.png", dpi=180)
    plt.close(fig)

    summary = {
        "schema_version": 1,
        "kind": "stage3_truth_grid_analysis",
        "roots": [str(root) for root in roots],
        "runs": len(run_rows),
        "truth_slot_points": len(truth_points),
        "observed_assignments": len(observed_points),
        "exact_join_rows": exact_join_total,
        "truth_target_selection_failures": truth_target_failures,
        "truth_sequence_regressions": truth_sequence_regressions,
        "truth_timestamp_regressions": truth_timestamp_regressions,
        "observation_sequence_regressions": observation_sequence_regressions,
        "observation_timestamp_regressions": observation_timestamp_regressions,
        "truth_duplicate_keys": truth_duplicate_keys,
        "observation_duplicate_keys": observation_duplicate_keys,
        "identity_rule": "label-3 complete target farthest from camera per run/frame; physical armor identity is relative_slot 0..3; target_id is not cross-run identity",
        "assignment_mode": args.assignment_mode,
        "assignment_rule": (
            "truth-facing candidate slots plus one-to-one Euclidean nearest neighbor in OpenCV camera xyz from PnP tvec, with depth-scaled rejection gate"
            if args.assignment_mode == "pnp_3d_facing"
            else "one-to-one Euclidean nearest neighbor in OpenCV camera xyz from PnP tvec, with depth-scaled rejection gate"
            if args.assignment_mode == "pnp_3d"
            else "one-to-one nearest image ray among every positive-facing truth plate, with explicit angular gate and second-best margin"
            if args.assignment_mode == "angular_facing"
            else "legacy one-to-one nearest neighbor in camera azimuth/elevation with 8 degree rejection gate"
        ),
        "angular_assignment_gate_deg": args.angular_assignment_gate_deg,
        "pnp_position_error_floor_m": args.pnp_position_error_floor_m,
        "pnp_position_error_depth_ratio": args.pnp_position_error_depth_ratio,
        "included_radial_scales": args.include_radial_scale,
        "plots": [
            "trajectory_grid_truth.png",
            "trajectory_grid_observed.png",
            "trajectory_concentration_heatmap.png",
            "causal_prediction_error.png",
        ],
        "metrics": [
            "condition_slot_metrics.csv",
            "historical_repeat_prediction.csv",
            "causal_constant_velocity_prediction.csv",
            "run_metrics.csv",
        ],
    }
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    source_runs = {}
    for root in roots:
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            files = [
                run_dir / "collection_run_manifest.json",
                run_dir / "truth_motion_audit.json",
                run_dir / "stage3_observations.jsonl",
            ]
            if not all(path.exists() for path in files):
                continue
            manifest = read_json(files[0])
            if args.include_radial_scale and not any(
                abs(float(manifest["radial_scale"]) - requested) < 1e-6
                for requested in args.include_radial_scale
            ):
                continue
            audit = read_json(files[1])
            source_runs[str(run_dir)] = {
                "truth_sha256": audit.get("truth_sha256"),
                "collection_manifest_sha256": sha256(files[0]),
                "truth_audit_sha256": sha256(files[1]),
                "observations_sha256": sha256(files[2]),
            }
    artifacts = {}
    for path in sorted(candidate for candidate in output.iterdir() if candidate.is_file()):
        if path.name == "retention_manifest.json":
            continue
        artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    retention = {
        "schema_version": 1,
        "kind": "stage3_truth_grid_analysis_retention_manifest",
        "classification": "long_term_private_evidence",
        "deletion_allowed": False,
        "source_runs": source_runs,
        "artifacts": artifacts,
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    }
    (output / "retention_manifest.json").write_text(
        json.dumps(retention, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
