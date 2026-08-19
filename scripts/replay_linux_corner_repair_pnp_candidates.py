#!/usr/bin/env python3
"""Apply the frozen repairer and unchanged IPPE to a complete detector sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.stage3.corner_residual_network import observable_features
from training.stage3.train_image_corner_repair_formal import (
    CONTEXT_SCALE,
    ContextSpatialReliabilityNet,
    context_patch,
    normalized_context_predictions_to_full,
)
from training.stage3.train_image_corner_repair_pilot import load_release_ledger, load_release_rgba


CORNER_ORDER = ("bl", "tl", "tr", "br")
SMALL_POINTS_MM = np.asarray(
    [[-67.5, 27.5, 0.0], [-67.5, -27.5, 0.0], [67.5, -27.5, 0.0], [67.5, 27.5, 0.0]],
    dtype=np.float32,
)
LARGE_POINTS_MM = np.asarray(
    [[-112.5, 27.5, 0.0], [-112.5, -27.5, 0.0], [112.5, -27.5, 0.0], [112.5, 27.5, 0.0]],
    dtype=np.float32,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_new_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def calibration(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    intrinsics = payload["intrinsics"]
    matrix = np.asarray(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.asarray(intrinsics["distortion"], dtype=np.float64)
    return matrix, distortion, payload


def ordered_corner_array(candidate: dict[str, Any]) -> np.ndarray:
    if tuple(candidate.get("raw_corners_order", ())) != CORNER_ORDER:
        raise ValueError("detector candidate corner order is not bl,tl,tr,br")
    values = candidate["raw_corners_px"]
    corners = np.asarray([values[name] for name in CORNER_ORDER], dtype=np.float32)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("detector candidate corners are invalid")
    return corners


def corners_json(corners: np.ndarray) -> dict[str, list[float]]:
    return {
        name: [float(point[0]), float(point[1])]
        for name, point in zip(CORNER_ORDER, np.asarray(corners, dtype=float))
    }


def solve_ippe_candidates(
    corners: np.ndarray,
    matrix: np.ndarray,
    distortion: np.ndarray,
    object_points_mm: np.ndarray,
) -> tuple[list[dict[str, Any]], str | None]:
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return [], "invalid_input_corners"
    result = cv2.solvePnPGeneric(
        object_points_mm.astype(np.float32),
        corners.astype(np.float32),
        matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not bool(result[0]) or not result[1]:
        return [], "solvepnp_failed"
    candidates = []
    for solver_index, (rvec, tvec) in enumerate(zip(result[1], result[2])):
        rvec_array = np.asarray(rvec, dtype=np.float64).reshape(3)
        tvec_array = np.asarray(tvec, dtype=np.float64).reshape(3)
        if not np.isfinite(rvec_array).all() or not np.isfinite(tvec_array).all():
            continue
        if tvec_array[2] <= 0.0:
            continue
        projected, _ = cv2.projectPoints(
            object_points_mm, rvec_array, tvec_array, matrix, distortion
        )
        residuals = np.linalg.norm(projected.reshape(4, 2) - corners, axis=1)
        rotation, _ = cv2.Rodrigues(rvec_array)
        normal = rotation[:, 2]
        observed_yaw_rad = math.atan2(float(normal[0]), float(normal[2]))
        candidates.append(
            {
                "solver_solution_index": solver_index,
                "rvec": rvec_array.tolist(),
                "tvec_m": (tvec_array / 1000.0).tolist(),
                "reprojection_rms_px": float(np.sqrt(np.mean(np.square(residuals)))),
                "reprojection_max_px": float(np.max(residuals)),
                "observed_yaw_rad": observed_yaw_rad,
                "positive_depth": True,
            }
        )
    candidates.sort(key=lambda item: (item["reprojection_rms_px"], item["solver_solution_index"]))
    for index, candidate in enumerate(candidates):
        candidate["id"] = index
        candidate["selected"] = index == 0
    return candidates, None if candidates else "no_finite_positive_depth_candidate"


def valid_proposal(corners: np.ndarray, width: int, height: int) -> str | None:
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return "nonfinite_model_proposal"
    if np.any(corners[:, 0] < 0.0) or np.any(corners[:, 0] >= width):
        return "model_proposal_outside_image_x"
    if np.any(corners[:, 1] < 0.0) or np.any(corners[:, 1] >= height):
        return "model_proposal_outside_image_y"
    polygon = np.asarray([corners[0], corners[1], corners[2], corners[3]], dtype=np.float32)
    area = float(cv2.contourArea(polygon))
    if area <= 1.0:
        return "model_proposal_degenerate"
    return None


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--session", required=True, type=Path)
    result.add_argument("--detector-sidecar", required=True, type=Path)
    result.add_argument("--detector-manifest", required=True, type=Path)
    result.add_argument("--checkpoint", required=True, type=Path)
    result.add_argument("--camera-calibration", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--manifest", required=True, type=Path)
    result.add_argument("--batch-size", type=int, default=128)
    return result


def main() -> None:
    args = parser().parse_args()
    session = args.session.resolve(strict=True)
    detector_sidecar = args.detector_sidecar.resolve(strict=True)
    detector_manifest_path = args.detector_manifest.resolve(strict=True)
    checkpoint_path = args.checkpoint.resolve(strict=True)
    calibration_path = args.camera_calibration.resolve(strict=True)
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    if output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite repair/PnP sidecar evidence")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    detector_manifest = json.loads(detector_manifest_path.read_text(encoding="utf-8"))
    if detector_manifest["output_sha256"] != sha256(detector_sidecar):
        raise ValueError("detector sidecar hash does not match its manifest")
    frames = read_jsonl(detector_sidecar)
    if len(frames) != int(detector_manifest["frames"]):
        raise ValueError("detector sidecar frame count mismatch")
    ledger = load_release_ledger(session)
    matrix, distortion, calibration_payload = calibration(calibration_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "v3-context-spatial-reliability":
        raise ValueError("only the formally selected v3 reliability repairer is accepted")
    model = ContextSpatialReliabilityNet()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    samples = []
    images = []
    geometry = []
    scores = []
    for frame_index, frame in enumerate(frames):
        key = tuple(int(frame[name]) for name in ("producer_epoch", "frame_seq", "timestamp_ns"))
        identity = ledger.get(key)
        if identity is None:
            raise ValueError(f"detector frame lacks Release image identity: {key}")
        rgba = None
        for candidate_index, candidate in enumerate(frame["candidates"]):
            raw = ordered_corner_array(candidate)
            if rgba is None:
                rgba = load_release_rgba(session, identity)
            images.append(context_patch(rgba, raw))
            geometry.append(observable_features(raw))
            scores.append(float(candidate["objectness"]))
            samples.append((frame_index, candidate_index, raw))
    image_values = np.asarray(images, dtype=np.float32)
    geometry_values = np.asarray(geometry, dtype=np.float32)
    score_values = np.asarray(scores, dtype=np.float32)
    predicted_parts = []
    reliability_parts = []
    for start in range(0, len(samples), args.batch_size):
        stop = min(start + args.batch_size, len(samples))
        batch_geometry = (
            geometry_values[start:stop] - np.asarray(checkpoint["geometry_mean"])
        ) / np.asarray(checkpoint["geometry_std"])
        with torch.no_grad():
            network_output = model(
                torch.from_numpy(image_values[start:stop]),
                torch.from_numpy(batch_geometry.astype(np.float32)),
                torch.from_numpy(score_values[start:stop]),
            ).numpy()
        predicted_parts.append(network_output[:, :8])
        reliability_parts.append(1.0 / (1.0 + np.exp(-network_output[:, 8])))
    predicted = np.concatenate(predicted_parts) if predicted_parts else np.empty((0, 8), dtype=np.float32)
    reliability = (
        np.concatenate(reliability_parts) if reliability_parts else np.empty(0, dtype=np.float32)
    )
    if checkpoint.get("output_standardized", True) and len(predicted):
        predicted = predicted * np.asarray(
            checkpoint.get("target_std", checkpoint.get("target_std_px"))
        ) + np.asarray(checkpoint.get("target_mean", checkpoint.get("target_mean_px")))
    raw_batch = np.asarray([sample[2] for sample in samples], dtype=np.float32)
    if checkpoint.get("target_space") == "context-normalized-residual" and len(predicted):
        predicted = normalized_context_predictions_to_full(raw_batch, predicted)
    correction = predicted.reshape(-1, 4, 2)
    probability_threshold = float(
        checkpoint["reliability"]["application_probability_threshold"]
    )
    minimum_score = checkpoint.get("minimum_detector_score")
    minimum_correction = checkpoint.get("minimum_predicted_correction_rms_px")
    model_results: dict[tuple[int, int], dict[str, Any]] = {}
    for sample_index, (frame_index, candidate_index, raw) in enumerate(samples):
        proposed = raw + correction[sample_index]
        correction_rms = float(np.sqrt(np.mean(np.square(correction[sample_index]))))
        applied = bool(reliability[sample_index] >= probability_threshold)
        reason = "accepted"
        if not applied:
            reason = "reliability_below_threshold"
        if applied and minimum_score is not None and score_values[sample_index] < float(minimum_score):
            applied, reason = False, "detector_score_below_threshold"
        if applied and minimum_correction is not None and correction_rms < float(minimum_correction):
            applied, reason = False, "predicted_correction_below_threshold"
        proposal_error = valid_proposal(proposed, 1440, 1080)
        if applied and proposal_error is not None:
            applied, reason = False, proposal_error
        selected = proposed if applied else raw
        model_results[(frame_index, candidate_index)] = {
            "raw": raw,
            "proposed": proposed,
            "selected": selected,
            "applied": applied,
            "reason": reason,
            "reliability_probability": float(reliability[sample_index]),
            "correction_rms_px": correction_rms,
            "correction_max_px": float(np.max(np.linalg.norm(correction[sample_index], axis=1))),
        }

    started_ns = time.perf_counter_ns()
    output_frames = []
    pnp_success = pnp_failure = repair_applied = 0
    for frame_index, frame in enumerate(frames):
        output_candidates = []
        for candidate_index, candidate in enumerate(frame["candidates"]):
            repair = model_results[(frame_index, candidate_index)]
            object_points = (
                LARGE_POINTS_MM
                if candidate["decoded_armor_type"] == "large"
                else SMALL_POINTS_MM
            )
            raw_pnp, raw_failure = solve_ippe_candidates(
                repair["raw"], matrix, distortion, object_points
            )
            selected_pnp, selected_failure = solve_ippe_candidates(
                repair["selected"], matrix, distortion, object_points
            )
            selected = selected_pnp[0] if selected_pnp else None
            pnp_success += int(selected is not None)
            pnp_failure += int(selected is None)
            repair_applied += int(repair["applied"])
            output_candidates.append(
                {
                    "candidate_rank": candidate["candidate_rank"],
                    "observation_id": candidate["observation_id"],
                    "detector_confidence": candidate["objectness"],
                    "detector_number": candidate["decoded_number"],
                    "detector_color": candidate["decoded_color"],
                    "detector_type": candidate["decoded_armor_type"],
                    "repair": {
                        "raw_corners_order": list(CORNER_ORDER),
                        "raw_corners_px": corners_json(repair["raw"]),
                        "model_proposed_corners_px": corners_json(repair["proposed"]),
                        "selected_corners_px": corners_json(repair["selected"]),
                        "selected_source": "model" if repair["applied"] else "raw",
                        "applied": repair["applied"],
                        "status_reason": repair["reason"],
                        "reliability_probability": repair["reliability_probability"],
                        "application_probability_threshold": probability_threshold,
                        "predicted_correction_rms_px": repair["correction_rms_px"],
                        "predicted_correction_max_px": repair["correction_max_px"],
                    },
                    "raw_pnp": {
                        "object_model": candidate["decoded_armor_type"],
                        "candidates": raw_pnp,
                        "failure_reason": raw_failure,
                    },
                    "selected_pnp": {
                        "object_model": candidate["decoded_armor_type"],
                        "candidates": selected_pnp,
                        "failure_reason": selected_failure,
                    },
                    "valid": selected is not None,
                    "camera_tvec_m": None if selected is None else selected["tvec_m"],
                    "yaw_absolute_rad": None if selected is None else selected["observed_yaw_rad"],
                    "reprojection_rms_px": None if selected is None else selected["reprojection_rms_px"],
                    "reprojection_max_px": None if selected is None else selected["reprojection_max_px"],
                    "corner_source": "model" if repair["applied"] else "raw",
                }
            )
        output_frames.append(
            {
                "schema_version": "aim-stack.linux-offline-repair-pnp-frame/1",
                "session_id": frame["session_id"],
                "runtime_instance_id": frame["runtime_instance_id"],
                "producer_epoch": frame["producer_epoch"],
                "frame_seq": frame["frame_seq"],
                "timestamp_ns": frame["timestamp_ns"],
                "image": frame["image"],
                "observation_sink_status": frame["sink_status"],
                "candidate_count": len(output_candidates),
                "candidates": output_candidates,
            }
        )
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for frame in output_frames:
            handle.write(json.dumps(frame, separators=(",", ":")) + "\n")
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1e6
    manifest = {
        "schema_version": "aim-stack.linux-offline-repair-pnp-sidecar-manifest/1",
        "claim": "truth-free offline PyTorch repair plus unchanged OpenCV IPPE replay; not C++/ONNX/TRT integration",
        "frames": len(output_frames),
        "detector_candidates": len(samples),
        "repair_applied": repair_applied,
        "pnp_success": pnp_success,
        "pnp_failure": pnp_failure,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_architecture": checkpoint["architecture"],
        "context_scale_runtime": CONTEXT_SCALE,
        "checkpoint_context_scale_metadata": checkpoint.get("context_scale"),
        "context_metadata_caveat": "runtime source constant is authoritative for this checkpoint; production export remains blocked until manifest correction and numerical parity",
        "camera_calibration": str(calibration_path),
        "camera_calibration_sha256": sha256(calibration_path),
        "camera_calibration_id": calibration_payload["calibration_id"],
        "detector_sidecar": str(detector_sidecar),
        "detector_sidecar_sha256": sha256(detector_sidecar),
        "detector_manifest": str(detector_manifest_path),
        "detector_manifest_sha256": sha256(detector_manifest_path),
        "output": str(output),
        "output_sha256": sha256(output),
        "output_bytes": output.stat().st_size,
        "truth_fields_included": False,
        "elapsed_pnp_serial_ms": elapsed_ms,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "torch": torch.__version__,
        },
        "retention": {
            "classification": "protected_derived_repair_pnp_evidence",
            "deletion_allowed": False,
        },
    }
    write_new_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
