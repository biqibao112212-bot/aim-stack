"""Validate simulator camera/gimbal R/T against independent exact exposure truth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .build_dataset import _load_formal_manifest, discover_canonical_sources
from .dataset import _camera_to_tracker_rotation, load_camera_gimbal_extrinsic
from .schema import ObservationFrame, TruthFrame, iter_json_records


def _quaternion_matrix(values: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = np.asarray(values, dtype=np.float64)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _rotation_error_deg(expected: np.ndarray, actual: np.ndarray) -> float:
    cosine = np.clip((np.trace(expected.T @ actual) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def validate(args: argparse.Namespace) -> dict[str, object]:
    manifest_path = Path(args.manifest).resolve()
    evidence_root = Path(args.evidence_root).resolve()
    raw_root = Path(args.raw_root).resolve()
    extrinsic = load_camera_gimbal_extrinsic(args.extrinsic_yaml)
    records = _load_formal_manifest(manifest_path)
    if args.session_stride > 1:
        records = records[::args.session_stride]
    if args.max_sessions > 0:
        records = records[:args.max_sessions]
    sources = discover_canonical_sources(records, evidence_root, raw_root)
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    sample_count = 0
    for source in sources:
        truths: dict[tuple[str, int, int, int], TruthFrame] = {}
        for record in iter_json_records(source["truth"]):
            if bool(record.get("has_exact_exposure_truth", False)):
                truth = TruthFrame.from_mapping(record)
                truths[(truth.session_id, truth.producer_epoch, truth.frame_seq, truth.timestamp_ns)] = truth
        accepted = 0
        for record in iter_json_records(source["observations"]):
            if accepted >= args.frames_per_session:
                break
            observation = ObservationFrame.from_mapping(record)
            truth = truths.get(observation.key)
            if truth is None:
                continue
            R_world_tracker = _quaternion_matrix(truth.chassis_quaternion_world_wxyz)
            R_world_gimbal = _quaternion_matrix(truth.gimbal_quaternion_world_wxyz)
            R_tracker_gimbal = R_world_tracker.T @ R_world_gimbal
            R_tracker_camera = _camera_to_tracker_rotation(
                observation.gimbal_pitch_deg, observation.gimbal_yaw_deg
            )
            derived_rotation = R_tracker_gimbal.T @ R_tracker_camera
            derived_translation = R_world_gimbal.T @ (
                np.asarray(truth.camera_origin_world_m) - np.asarray(truth.gimbal_origin_world_m)
            )
            rotation_errors.append(_rotation_error_deg(extrinsic.rotation, derived_rotation))
            translation_errors.append(float(np.linalg.norm(extrinsic.translation_m - derived_translation)))
            accepted += 1
            sample_count += 1
    if sample_count == 0:
        raise ValueError("no exact observation/truth exposure pairs were validated")
    r = np.asarray(rotation_errors)
    t = np.asarray(translation_errors)
    report: dict[str, object] = {
        "schema_version": "stage3-extrinsic-validation-v1",
        "status": "pass" if r.max() <= args.max_rotation_error_deg and t.max() <= args.max_translation_error_m else "fail",
        "session_count": len(sources),
        "sample_count": sample_count,
        "R_camera2gimbal": extrinsic.rotation.tolist(),
        "t_camera2gimbal_m": extrinsic.translation_m.tolist(),
        "rotation_error_deg": {
            "median": float(np.quantile(r, 0.5)), "p95": float(np.quantile(r, 0.95)), "max": float(r.max()),
        },
        "translation_error_m": {
            "mean": float(t.mean()), "p95": float(np.quantile(t, 0.95)), "max": float(t.max()),
        },
        "thresholds": {
            "max_rotation_error_deg": args.max_rotation_error_deg,
            "max_translation_error_m": args.max_translation_error_m,
        },
        "method": "exact exposure chassis/gimbal/camera pose; optical pose composed independently",
    }
    if args.report:
        output = Path(args.report).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--extrinsic-yaml", required=True)
    parser.add_argument("--report")
    parser.add_argument("--session-stride", type=int, default=5)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--frames-per-session", type=int, default=80)
    parser.add_argument("--max-rotation-error-deg", type=float, default=1e-4)
    parser.add_argument("--max-translation-error-m", type=float, default=1e-5)
    args = parser.parse_args()
    report = validate(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
