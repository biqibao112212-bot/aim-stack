#!/usr/bin/env python3
"""Propagate a frozen Linux image corner repairer through IPPE and the 400 ms LOS expert."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.stage3.train_image_corner_repair_formal import (
    ContextSpatialResidualNet,
    ContextSpatialReliabilityNet,
    CornerHeatmapReliabilityNet,
    context_patch,
    normalized_context_predictions_to_full,
)
from training.stage3.train_image_corner_repair_pilot import digest, load_session_rows


HISTORY_S = 0.40
HORIZONS_S = (0.05, 0.10, 0.20)
DEPTH_WEIGHT = 0.1
HUBER_DELTA_M = 0.02
OMEGA_MEMORY_COUNT = 31
OMEGA_GRID_STEP = 1.0
MAX_OMEGA = 16.0
CORNER_ORDER = ("bl", "tl", "tr", "br")
OBJECT_POINTS_MM = np.asarray(
    [[-67.5, 27.5, 0.0], [-67.5, -27.5, 0.0], [67.5, -27.5, 0.0], [67.5, 27.5, 0.0]],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-manifest", "--test-dataset-manifest", dest="dataset_manifest",
                        required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", default="linear_and_spin")
    parser.add_argument("--anchor-stride", type=int, default=4)
    parser.add_argument(
        "--development-input", action="store_true",
        help="allow a validation-only post-test development manifest that does not authorize model selection",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_module(name: str, path: Path):
    # The retained predictor modules import matplotlib only for their standalone
    # report figures.  This adapter reuses their numerical functions and emits
    # tables, so keep the simulation-ml runtime free of an unused plotting dep.
    if "matplotlib" not in sys.modules:
        matplotlib_stub = types.ModuleType("matplotlib")
        pyplot_stub = types.ModuleType("matplotlib.pyplot")
        matplotlib_stub.pyplot = pyplot_stub
        sys.modules["matplotlib"] = matplotlib_stub
        sys.modules["matplotlib.pyplot"] = pyplot_stub
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_new_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv_gz(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty evidence table: {path}")
    with gzip.open(path, "xt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matrix_from_label(label: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = label["camera"]["intrinsics"]
    matrix = np.asarray(
        [[intrinsics["fx"], 0.0, intrinsics["cx"]], [0.0, intrinsics["fy"], intrinsics["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return matrix, np.asarray(intrinsics["distortion"], dtype=np.float64)


def solve_ippe(corners: np.ndarray, matrix: np.ndarray, distortion: np.ndarray,
               object_points_mm: np.ndarray = OBJECT_POINTS_MM) -> np.ndarray | None:
    result = cv2.solvePnPGeneric(
        object_points_mm.astype(np.float32), corners.astype(np.float32), matrix, distortion,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not bool(result[0]) or not result[1]:
        return None
    candidates: list[tuple[float, int, np.ndarray]] = []
    for index, (rvec, tvec) in enumerate(zip(result[1], result[2])):
        point = np.asarray(tvec, dtype=np.float64).reshape(3)
        if not np.isfinite(point).all() or point[2] <= 0.0:
            continue
        projected, _ = cv2.projectPoints(object_points_mm, rvec, tvec, matrix, distortion)
        rms = float(np.sqrt(np.mean(np.sum(np.square(projected.reshape(4, 2) - corners), axis=1))))
        candidates.append((rms, index, point / 1000.0))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def camera_to_bevy(point: np.ndarray) -> np.ndarray:
    # OpenCV C: +x right,+y down,+z forward -> simulator camera local:
    # +x forward,+y left,+z up.
    return np.asarray([point[2], -point[0], -point[1]], dtype=np.float64)


def quaternion_matrix(values: list[float]) -> np.ndarray:
    w, x, y, z = np.asarray(values, dtype=np.float64)
    norm = math.sqrt(float(w * w + x * x + y * y + z * z))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("invalid exposure quaternion")
    w, x, y, z = (w / norm, x / norm, y / norm, z / norm)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def camera_to_world(point: np.ndarray, exposure: dict[str, object]) -> np.ndarray:
    origin = np.asarray(exposure["camera_position_world_m"], dtype=np.float64)
    rotation = quaternion_matrix(exposure["camera_quaternion_world_wxyz"])
    return origin + rotation @ camera_to_bevy(point)


def measured_ippe_points(label: dict[str, object]) -> np.ndarray:
    corners = np.asarray(label["plate_geometry"]["object_corners_armor_m"], dtype=np.float32)
    # The marker asset uses armor x=right and z=up.  Production IPPE's planar
    # template uses x=right and y=down, so map (x,-z) and retain the label's
    # measured dimensions while dropping only the constant plane-depth offset.
    return np.column_stack((corners[:, 0] * 1000.0, -corners[:, 2] * 1000.0, np.zeros(4, dtype=np.float32)))


def row_corners(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray(
        [[float(row[f"{prefix}_{corner}_{axis}_px"]) for axis in ("x", "y")] for corner in CORNER_ORDER],
        dtype=np.float32,
    )


def frozen_repairs(checkpoint_path: Path, rows_path: Path, session: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = checkpoint.get("architecture")
    if architecture not in {"v2-context-spatial", "v3-context-spatial-reliability", "v4-corner-heatmap-reliability"}:
        raise ValueError("local prediction adapter supports only frozen context-spatial repairers")
    records = [row for row in csv.DictReader(rows_path.open(encoding="utf-8", newline="")) if row["motion_uniform"] == "True"]
    values, _ = load_session_rows(rows_path, session, patch_fn=context_patch)
    if len(records) != len(values["targets"]):
        raise ValueError("repair rows lost alignment")
    if architecture == "v4-corner-heatmap-reliability":
        model = CornerHeatmapReliabilityNet()
    elif architecture == "v3-context-spatial-reliability":
        model = ContextSpatialReliabilityNet()
    else:
        model = ContextSpatialResidualNet()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    geometry = (values["geometry"] - checkpoint["geometry_mean"]) / checkpoint["geometry_std"]
    with torch.no_grad():
        if architecture in {"v3-context-spatial-reliability", "v4-corner-heatmap-reliability"}:
            network_output = model(
                torch.from_numpy(values["images"]), torch.from_numpy(geometry),
                torch.from_numpy(np.asarray([float(row["detector_score"]) for row in records], dtype=np.float32)),
            ).numpy()
            predicted = network_output[:, :8]
            reliability_probability = 1.0 / (1.0 + np.exp(-network_output[:, 8]))
        else:
            predicted = model(torch.from_numpy(values["images"]), torch.from_numpy(geometry)).numpy()
            reliability_probability = np.ones(len(predicted), dtype=np.float32)
    target_mean = checkpoint.get("target_mean", checkpoint.get("target_mean_px"))
    target_std = checkpoint.get("target_std", checkpoint.get("target_std_px"))
    if checkpoint.get("output_standardized", True):
        predicted = predicted * target_std + target_mean
    if checkpoint.get("target_space", "full-pixel-residual") == "context-normalized-residual":
        predicted = normalized_context_predictions_to_full(values["raw_corners"], predicted)
    apply = np.ones(len(predicted), dtype=bool)
    if architecture in {"v3-context-spatial-reliability", "v4-corner-heatmap-reliability"}:
        apply &= reliability_probability >= float(checkpoint["reliability"]["application_probability_threshold"])
    minimum_score = checkpoint.get("minimum_detector_score")
    if minimum_score is not None:
        apply &= np.asarray([float(row["detector_score"]) for row in records]) >= float(minimum_score)
    minimum_correction = checkpoint.get("minimum_predicted_correction_rms_px")
    if minimum_correction is not None:
        apply &= np.sqrt(np.mean(np.square(predicted), axis=1)) >= float(minimum_correction)
    predicted[~apply] = 0.0
    return records, values["raw_corners"] + predicted.reshape(-1, 4, 2), apply


def exact_frames(session: Path, minimum_frame_seq: int, base,
                 exposures: dict[tuple[int, int, int], dict[str, object]]) -> tuple[list[Any], dict[tuple[int, int, int], int]]:
    grouped: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    with (session / "exact-corners.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            label = json.loads(line)
            key = tuple(int(label[name]) for name in ("producer_epoch", "frame_seq", "timestamp_ns"))
            if key[1] >= minimum_frame_seq and label["motion_uniform"]:
                grouped[key].append(label)
    frames: list[Any] = []
    key_to_index: dict[tuple[int, int, int], int] = {}
    previous_velocity: np.ndarray | None = None
    segment_id = 0
    for key in sorted(grouped):
        labels = grouped[key]
        exposure = exposures.get(key)
        if exposure is None:
            continue
        if {int(label["relative_slot"]) for label in labels} != {0, 1, 2, 3}:
            continue
        exact = np.full((4, 3), np.nan, dtype=np.float64)
        omega = 0.0
        for label in labels:
            matrix, distortion = matrix_from_label(label)
            point = solve_ippe(
                np.asarray(label["exact_corners_px"], dtype=np.float32), matrix, distortion,
                measured_ippe_points(label),
            )
            if point is None:
                break
            exact[int(label["relative_slot"])] = camera_to_world(point, exposure)
            omega = float(label["angular_velocity_world_rad_s"][2])
        if not np.isfinite(exact).all():
            continue
        center = exact.mean(axis=0)
        velocity = np.asarray(labels[0]["linear_velocity_world_mps"], dtype=np.float64)
        if (
            previous_velocity is not None
            and np.linalg.norm(previous_velocity) > 1e-6
            and np.linalg.norm(velocity) > 1e-6
            and float(previous_velocity @ velocity) < 0.0
        ):
            segment_id += 1
        previous_velocity = velocity
        tracker_origin = np.asarray(exposure["gimbal_position_world_m"], dtype=np.float64)
        tracker_to_world = quaternion_matrix(exposure["chassis_quaternion_world_wxyz"])
        frame = base.Frame(
            timestamp_ns=key[2], center_world_m=center,
            velocity_world_mps=velocity, yaw_rate_rad_s=omega,
            armor_world_m=exact, armor_local_m=np.zeros((4, 3), dtype=np.float64),
            tracker_origin_world_m=tracker_origin, tracker_to_world=tracker_to_world,
            observed_world_m=np.full((4, 3), np.nan, dtype=np.float64),
            observed_mask=np.zeros(4, dtype=np.bool_), observed_range_m=np.full(4, np.inf),
            association_ambiguous=False, segment_id=segment_id,
        )
        key_to_index[key] = len(frames)
        frames.append(frame)
    if len(frames) < 10:
        raise ValueError("insufficient complete exact frames for local prediction")
    geometry = frames[0].armor_world_m - frames[0].center_world_m
    for frame in frames:
        frame.armor_local_m = geometry.copy()
    return frames, key_to_index


def populate_observations(frames: list[Any], lookup: dict[tuple[int, int, int], int], records: list[dict[str, str]],
                          corners: np.ndarray, labels_by_key: dict[tuple[int, int, int], dict[str, object]],
                          exposures: dict[tuple[int, int, int], dict[str, object]]) -> dict[str, int]:
    failed, duplicates, accepted = 0, 0, 0
    for record, points in zip(records, corners):
        key = tuple(int(record[name]) for name in ("producer_epoch", "frame_seq", "timestamp_ns"))
        index = lookup.get(key)
        label = labels_by_key.get(key)
        exposure = exposures.get(key)
        if index is None or label is None or exposure is None:
            continue
        slot = int(record["relative_slot"])
        frame = frames[index]
        if frame.observed_mask[slot]:
            duplicates += 1
            continue
        matrix, distortion = matrix_from_label(label)
        point = solve_ippe(points, matrix, distortion)
        if point is None:
            failed += 1
            continue
        world = camera_to_world(point, exposure)
        tracker = frame.tracker_to_world.T @ (world - frame.tracker_origin_world_m)
        frame.observed_world_m[slot] = world
        frame.observed_mask[slot] = True
        frame.observed_range_m[slot] = float(np.linalg.norm(tracker[:2]))
        accepted += 1
    return {"accepted": accepted, "pnp_failed": failed, "duplicates": duplicates}


def populate_oracle_best(frames: list[Any], raw_frames: list[Any], repaired_frames: list[Any],
                         truth: list[Any]) -> dict[str, int]:
    raw_selected = repaired_selected = 0
    for index, frame in enumerate(frames):
        for slot_raw in np.flatnonzero(raw_frames[index].observed_mask):
            slot = int(slot_raw)
            candidates = (
                ("raw", raw_frames[index].observed_world_m[slot]),
                ("repaired", repaired_frames[index].observed_world_m[slot]),
            )
            scored = []
            for source, point in candidates:
                error = frame.tracker_to_world.T @ (point - truth[index].armor_world_m[slot])
                scored.append((float(np.linalg.norm(error[1:])), source, point))
            _loss, source, selected = min(scored, key=lambda item: (item[0], item[1]))
            frame.observed_world_m[slot] = selected
            frame.observed_mask[slot] = True
            tracker = frame.tracker_to_world.T @ (selected - frame.tracker_origin_world_m)
            frame.observed_range_m[slot] = float(np.linalg.norm(tracker[:2]))
            if source == "raw":
                raw_selected += 1
            else:
                repaired_selected += 1
    return {"raw_selected": raw_selected, "repaired_selected": repaired_selected}


def populate_causal_guarded(frames: list[Any], raw_frames: list[Any], repaired_frames: list[Any],
                            innovation_margin_m: float = HUBER_DELTA_M) -> dict[str, int]:
    histories: dict[int, list[tuple[int, int, np.ndarray]]] = defaultdict(list)
    raw_selected = repaired_selected = insufficient_history = stale_history = 0
    for index, frame in enumerate(frames):
        for slot_raw in np.flatnonzero(raw_frames[index].observed_mask):
            slot = int(slot_raw)
            raw = raw_frames[index].observed_world_m[slot]
            repaired = repaired_frames[index].observed_world_m[slot]
            history = histories[slot]
            same_segment = [item for item in history if item[1] == frame.segment_id]
            source = "repaired"
            selected = repaired
            if len(same_segment) < 2:
                insufficient_history += 1
            else:
                t0, _segment0, p0 = same_segment[-2]
                t1, _segment1, p1 = same_segment[-1]
                dt = (t1 - t0) / 1e9
                ahead = (frame.timestamp_ns - t1) / 1e9
                if dt <= 1e-6 or dt > 0.15 or ahead < 0.0 or ahead > 0.15:
                    stale_history += 1
                else:
                    reference = p1 + (p1 - p0) * (ahead / dt)
                    metric = np.diag([math.sqrt(DEPTH_WEIGHT), 1.0, 1.0]) @ frame.tracker_to_world.T
                    raw_innovation = float(np.linalg.norm(metric @ (raw - reference)))
                    repaired_innovation = float(np.linalg.norm(metric @ (repaired - reference)))
                    if repaired_innovation > raw_innovation + innovation_margin_m:
                        source, selected = "raw", raw
            frame.observed_world_m[slot] = selected
            frame.observed_mask[slot] = True
            tracker = frame.tracker_to_world.T @ (selected - frame.tracker_origin_world_m)
            frame.observed_range_m[slot] = float(np.linalg.norm(tracker[:2]))
            history.append((frame.timestamp_ns, frame.segment_id, selected.copy()))
            if source == "raw":
                raw_selected += 1
            else:
                repaired_selected += 1
    return {
        "raw_selected": raw_selected, "repaired_selected": repaired_selected,
        "insufficient_history": insufficient_history, "stale_history": stale_history,
        "innovation_margin_m": innovation_margin_m,
    }


def describe(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)), "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)), "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)), "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)), "within_55mm_fraction": float(np.mean(values <= 55.0)),
    }


def evaluate(base, los, truth: list[Any], domains: dict[str, list[Any]], anchor_stride: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prediction_rows: list[dict[str, object]] = []
    omega_rows: list[dict[str, object]] = []
    memory: dict[str, list[float]] = {domain: [] for domain in domains}
    geometry = truth[0].armor_local_m
    for anchor_index in range(0, len(truth), anchor_stride):
        tracker_to_world = truth[anchor_index].tracker_to_world
        common = np.flatnonzero(np.logical_and.reduce([frames[anchor_index].observed_mask for frames in domains.values()]))
        if not len(common):
            continue
        primary = int(common[np.argmin(domains["raw"][anchor_index].observed_range_m[common])])
        fitted: dict[str, tuple[float, np.ndarray]] = {}
        histories: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for domain, frames in domains.items():
            history = base.history_observations(frames, anchor_index, HISTORY_S, "pnp")
            if history is None:
                break
            histories[domain] = history
        if len(histories) != len(domains):
            continue
        for domain, history in histories.items():
            times, slots, positions = history
            omega, _coefficient, raw_loss = los.estimate_weighted_omega(
                base, times, slots, positions, geometry, tracker_to_world,
                depth_weight=DEPTH_WEIGHT, huber_delta_m=HUBER_DELTA_M,
                max_omega=MAX_OMEGA, grid_step=OMEGA_GRID_STEP,
            )
            memory[domain].append(float(omega))
            selected_omega = float(np.median(memory[domain][-OMEGA_MEMORY_COUNT:]))
            coefficient, fit_loss = los.fit_weighted_rigid(
                base, times, slots, positions, geometry, selected_omega, tracker_to_world,
                depth_weight=DEPTH_WEIGHT, huber_delta_m=HUBER_DELTA_M, robust=True,
            )
            fitted[domain] = (selected_omega, coefficient)
            omega_rows.append({
                "timestamp_ns": truth[anchor_index].timestamp_ns, "input_domain": domain,
                "raw_omega_rad_s": omega, "memory_omega_rad_s": selected_omega,
                "truth_omega_rad_s": truth[anchor_index].yaw_rate_rad_s,
                "raw_fit_loss": raw_loss, "memory_fit_loss": fit_loss,
                "memory_count": min(len(memory[domain]), OMEGA_MEMORY_COUNT),
                "history_event_count": len(times),
            })
        for horizon in HORIZONS_S:
            future = base.nearest_future(truth, anchor_index, horizon)
            if future is None:
                continue
            effective = (future.timestamp_ns - truth[anchor_index].timestamp_ns) / 1e9
            expected = future.armor_world_m[primary]
            future_regime = (
                "constant_direction"
                if future.segment_id == truth[anchor_index].segment_id
                else "cross_reversal"
            )
            for domain, (omega, coefficient) in fitted.items():
                predicted = base.predict_rigid(coefficient, geometry, primary, omega, effective)
                errors = base.error_components(predicted, expected, tracker_to_world)
                prediction_rows.append({
                    "timestamp_ns": truth[anchor_index].timestamp_ns, "input_domain": domain,
                    "horizon_s": horizon, "effective_horizon_s": effective, "primary_slot": primary,
                    "anchor_segment_id": truth[anchor_index].segment_id,
                    "future_segment_id": future.segment_id,
                    "future_regime": future_regime,
                    "prediction_x_m": float(predicted[0]), "prediction_y_m": float(predicted[1]),
                    "prediction_z_m": float(predicted[2]), "truth_x_m": float(expected[0]),
                    "truth_y_m": float(expected[1]), "truth_z_m": float(expected[2]), **errors,
                })
    return prediction_rows, omega_rows


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve(strict=True)
    manifest_path = args.dataset_manifest.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prediction evidence: {output}")
    if args.anchor_stride <= 0:
        raise ValueError("--anchor-stride must be positive")
    output.mkdir(parents=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = [entry for entry in manifest["sessions"] if entry["mode"] == args.mode]
    if len(selected) != 1:
        raise ValueError(f"expected one {args.mode} session, found {len(selected)}")
    entry = selected[0]
    session_result = Path(str(entry["session_result"]["path"])).resolve(strict=True)
    rows_path = Path(str(entry["rows"]["path"])).resolve(strict=True)
    if digest(rows_path) != entry["rows"]["sha256"] or digest(session_result) != entry["session_result"]["sha256"]:
        raise ValueError("qualified combined session evidence changed")
    authorization = manifest.get("test_authorization") or {}
    if authorization:
        if authorization.get("repair_checkpoint_sha256") != sha256(checkpoint_path):
            raise PermissionError("test manifest is not authorized for this repair checkpoint")
        input_role = "sealed_test"
    else:
        if not args.development_input or manifest.get("splits") != ["validation"]:
            raise PermissionError(
                "unsealed input requires --development-input and a validation-only manifest"
            )
        input_role = "post_test_development"

    base = load_module("linux_corner_local_base", REPO_ROOT / "scripts/evaluate-combined-motion-factorization.py")
    los = load_module("linux_corner_local_los", REPO_ROOT / "scripts/evaluate-combined-motion-los-fit.py")
    session = session_result.parent
    result = json.loads(session_result.read_text(encoding="utf-8"))
    exposures: dict[tuple[int, int, int], dict[str, object]] = {}
    with (session / "exposure-states.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            exposure = json.loads(line)
            key = tuple(int(exposure[name]) for name in ("producer_epoch", "frame_seq", "timestamp_ns"))
            if key in exposures:
                raise ValueError(f"duplicate exposure identity: {key}")
            exposures[key] = exposure
    truth, lookup = exact_frames(session, int(result["first_eligible_frame_seq"]), base, exposures)
    records, repaired_corners, applied = frozen_repairs(checkpoint_path, rows_path, session)
    raw_corners = np.stack([row_corners(row, "raw") for row in records])
    labels_by_key: dict[tuple[int, int, int], dict[str, object]] = {}
    with (session / "exact-corners.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            label = json.loads(line)
            key = tuple(int(label[name]) for name in ("producer_epoch", "frame_seq", "timestamp_ns"))
            labels_by_key.setdefault(key, label)
    domains = {
        "raw": copy.deepcopy(truth), "repaired": copy.deepcopy(truth),
        "blend_025": copy.deepcopy(truth), "blend_050": copy.deepcopy(truth),
        "blend_075": copy.deepcopy(truth),
        "causal_guarded": copy.deepcopy(truth),
        "oracle_best": copy.deepcopy(truth), "exact_matched": copy.deepcopy(truth),
    }
    domain_audit = {
        "raw": populate_observations(domains["raw"], lookup, records, raw_corners, labels_by_key, exposures),
        "repaired": populate_observations(domains["repaired"], lookup, records, repaired_corners, labels_by_key, exposures),
    }
    for alpha, name in ((0.25, "blend_025"), (0.50, "blend_050"), (0.75, "blend_075")):
        blended = raw_corners + alpha * (repaired_corners - raw_corners)
        domain_audit[name] = populate_observations(
            domains[name], lookup, records, blended, labels_by_key, exposures
        )
    domain_audit["causal_guarded"] = populate_causal_guarded(
        domains["causal_guarded"], domains["raw"], domains["repaired"]
    )
    domain_audit["oracle_best"] = populate_oracle_best(
        domains["oracle_best"], domains["raw"], domains["repaired"], truth
    )
    for index, frame in enumerate(domains["exact_matched"]):
        mask = domains["raw"][index].observed_mask.copy()
        frame.observed_mask = mask
        frame.observed_world_m[mask] = truth[index].armor_world_m[mask]
        frame.observed_range_m[mask] = np.linalg.norm(frame.observed_world_m[mask, :2], axis=1)
    domain_audit["exact_matched"] = {
        "accepted": int(sum(frame.observed_mask.sum() for frame in domains["exact_matched"])),
        "pnp_failed": 0, "duplicates": 0,
    }
    raw_masks = np.stack([frame.observed_mask for frame in domains["raw"]])
    masks = [np.stack([frame.observed_mask for frame in frames]) for frames in domains.values()]
    if any(not np.array_equal(raw_masks, mask) for mask in masks[1:]):
        raise AssertionError("raw, repaired and exact-matched domains do not share an availability mask")
    prediction_rows, omega_rows = evaluate(base, los, truth, domains, args.anchor_stride)
    if not prediction_rows:
        raise ValueError("no eligible 400 ms local prediction anchors")
    prediction = pd.DataFrame(prediction_rows)
    summary_rows: list[dict[str, object]] = []
    for (domain, horizon), group in prediction.groupby(["input_domain", "horizon_s"], sort=True):
        errors = group["error_cross_depth_m"].to_numpy() * 1000.0
        summary_rows.append({"input_domain": domain, "horizon_s": horizon, **describe(errors)})
    summary = pd.DataFrame(summary_rows)
    condition_summary_rows: list[dict[str, object]] = []
    for (domain, horizon, regime), group in prediction.groupby(
        ["input_domain", "horizon_s", "future_regime"], sort=True
    ):
        errors = group["error_cross_depth_m"].to_numpy() * 1000.0
        condition_summary_rows.append({
            "input_domain": domain, "horizon_s": horizon,
            "future_regime": regime, **describe(errors),
        })
    keys = ["timestamp_ns", "horizon_s", "primary_slot"]
    raw = prediction[prediction.input_domain == "raw"][keys + ["error_cross_depth_m"]].rename(columns={"error_cross_depth_m": "raw_error_m"})
    repaired = prediction[prediction.input_domain == "repaired"][keys + ["error_cross_depth_m"]].rename(columns={"error_cross_depth_m": "repaired_error_m"})
    paired = raw.merge(repaired, on=keys, validate="one_to_one")
    paired["delta_mm"] = (paired.repaired_error_m - paired.raw_error_m) * 1000.0
    paired_summary = []
    for horizon, group in paired.groupby("horizon_s", sort=True):
        values = group.delta_mm.to_numpy()
        paired_summary.append({
            "horizon_s": horizon, "count": len(values), "improved_fraction": float(np.mean(values < 0.0)),
            "worsened_fraction": float(np.mean(values > 0.0)), "mean_delta_mm": float(np.mean(values)),
            "p50_delta_mm": float(np.quantile(values, 0.50)), "p90_delta_mm": float(np.quantile(values, 0.90)),
            "p95_delta_mm": float(np.quantile(values, 0.95)),
        })
    regime_keys = ["timestamp_ns", "horizon_s", "primary_slot", "future_regime"]
    raw_regime = prediction[prediction.input_domain == "raw"][
        regime_keys + ["error_cross_depth_m"]
    ].rename(columns={"error_cross_depth_m": "raw_error_m"})
    repaired_regime = prediction[prediction.input_domain == "repaired"][
        regime_keys + ["error_cross_depth_m"]
    ].rename(columns={"error_cross_depth_m": "repaired_error_m"})
    paired_regime = raw_regime.merge(repaired_regime, on=regime_keys, validate="one_to_one")
    paired_regime["delta_mm"] = (
        paired_regime.repaired_error_m - paired_regime.raw_error_m
    ) * 1000.0
    paired_condition_summary = []
    for (horizon, regime), group in paired_regime.groupby(
        ["horizon_s", "future_regime"], sort=True
    ):
        values = group.delta_mm.to_numpy()
        paired_condition_summary.append({
            "horizon_s": horizon, "future_regime": regime, "count": len(values),
            "improved_fraction": float(np.mean(values < 0.0)),
            "worsened_fraction": float(np.mean(values > 0.0)),
            "mean_delta_mm": float(np.mean(values)),
            "p50_delta_mm": float(np.quantile(values, 0.50)),
            "p90_delta_mm": float(np.quantile(values, 0.90)),
            "p95_delta_mm": float(np.quantile(values, 0.95)),
        })
    write_csv_gz(output / "prediction-rows.csv.gz", prediction_rows)
    write_csv_gz(output / "omega-rows.csv.gz", omega_rows)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(condition_summary_rows).to_csv(output / "condition-summary.csv", index=False)
    paired.to_csv(output / "paired-rows.csv", index=False)
    pd.DataFrame(paired_summary).to_csv(output / "paired-summary.csv", index=False)
    pd.DataFrame(paired_condition_summary).to_csv(
        output / "paired-condition-summary.csv", index=False
    )
    audit = {
        "session_id": entry["session_id"], "truth_frames": len(truth),
        "matched_uniform_rows": len(records), "repair_applied_rows": int(applied.sum()),
        "shared_observation_events": int(raw_masks.sum()), "domain": domain_audit,
    }
    manifest_out = {
        "schema_version": "aim-stack.linux-corner-local-prediction/1", "status": "complete",
        "deployable": False,
        "input_role": input_role,
        "scope": "offline truth-slot association; exact corners score future only; exposure-matched camera-world SE(3); frozen 400 ms LOS expert",
        "inputs": {
            "checkpoint": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path)},
            "dataset_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "rows": {"path": str(rows_path), "sha256": sha256(rows_path)},
            "exposure_states": {"path": str(session / "exposure-states.jsonl"),
                                "sha256": sha256(session / "exposure-states.jsonl")},
        },
        "contract": {
            "history_s": HISTORY_S, "horizons_s": HORIZONS_S, "depth_weight": DEPTH_WEIGHT,
            "huber_delta_m": HUBER_DELTA_M, "omega_memory_count": OMEGA_MEMORY_COUNT,
            "omega_grid_step": OMEGA_GRID_STEP, "max_omega_rad_s": MAX_OMEGA,
            "pnp": "unchanged nominal-small-armor SOLVEPNP_IPPE; minimum reprojection candidate",
            "future_truth_used_as_input": False, "truth_slot_identity": "offline analysis only",
            "pose_input": "same-exposure simulator camera/chassis/gimbal pose; production-equivalent coordinate transform; offline capture only",
            "diagnostic_truth_domains": {
                "exact_matched": "same-frame exact observations as an upper-bound history",
                "oracle_best": "same-frame truth selects raw or repaired; non-causal diagnostic only",
            },
        },
        "audit": audit, "summary": summary_rows,
        "condition_summary": condition_summary_rows,
        "paired_summary": paired_summary,
        "paired_condition_summary": paired_condition_summary,
        "artifacts": [
            "prediction-rows.csv.gz", "omega-rows.csv.gz", "summary.csv",
            "condition-summary.csv", "paired-rows.csv", "paired-summary.csv",
            "paired-condition-summary.csv",
        ],
    }
    write_new_json(output / "manifest.json", manifest_out)
    print(json.dumps({
        "output": str(output), "audit": audit, "summary": summary_rows,
        "condition_summary": condition_summary_rows,
        "paired_summary": paired_summary,
        "paired_condition_summary": paired_condition_summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
