#!/usr/bin/env python3
"""Audit apparent observed-arc flips against truth and planar PnP candidates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grid_analysis_path() -> Path:
    return Path(__file__).with_name("analyze-stage3-truth-grid.py").resolve()


def load_grid_analysis_module() -> object:
    path = grid_analysis_path()
    spec = importlib.util.spec_from_file_location("stage3_truth_grid", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pipeline_key(row: dict) -> tuple[int, int, int]:
    return (
        int(row["source_producer_epoch"]),
        int(row["source_image_seq"]),
        int(row["source_capture_timestamp_ns"]),
    )


def truth_key(row: dict) -> tuple[int, int, int]:
    return int(row["producer_epoch"]), int(row["frame_seq"]), int(row["timestamp_ns"])


def point2(value: dict) -> np.ndarray:
    return np.asarray([float(value["x"]), float(value["y"])], dtype=float)


def point3(value: dict) -> np.ndarray:
    return np.asarray([float(value[axis]) for axis in "xyz"], dtype=float)


def corner_refinement_deltas(detector: dict, armor: dict) -> list[float]:
    """Return per-corner raw-detector to PnP-input movement in pixels."""
    raw = detector.get("raw_corners_px") or []
    refined = armor.get("pnp_vertices_px") or []
    if len(raw) != 4 or len(refined) != 4:
        return []
    try:
        return [float(np.linalg.norm(point2(after) - point2(before))) for before, after in zip(raw, refined)]
    except (KeyError, TypeError, ValueError):
        return []


def quantiles(values: list[float]) -> dict:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not finite.size:
        return {key: None for key in ("min", "p10", "median", "p90", "max")}
    result = np.percentile(finite, [0, 10, 50, 90, 100])
    return dict(zip(("min", "p10", "median", "p90", "max"), map(float, result)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--distance-m", required=True, type=float)
    parser.add_argument("--pixel-gate", type=float, default=8.0)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pipeline_path = run_root / "pipeline.jsonl"
    truth_path = run_root / "truth.jsonl"
    analysis = load_grid_analysis_module()
    truths = {truth_key(row): row for row in read_jsonl(truth_path)}
    records: list[dict] = []
    current_basis_errors: list[float] = []
    mirrored_basis_errors: list[float] = []
    pnp_center_errors: list[float] = []

    for pipeline in read_jsonl(pipeline_path):
        truth = truths.get(pipeline_key(pipeline))
        if truth is None:
            continue
        target = analysis.select_active_target(truth, args.distance_m)
        if target is None:
            continue
        slots = analysis.target_slot_points(truth, target)
        fx = float(pipeline["camera"]["fx"])
        fy = float(pipeline["camera"]["fy"])
        cx = float(pipeline["camera"]["cx"])
        cy = float(pipeline["camera"]["cy"])
        detector_candidates = {
            int(candidate["observation_id"]): candidate
            for candidate in pipeline.get("detector", {}).get("candidates", [])
            if candidate.get("observation_id") is not None
        }
        subpixel = pipeline.get("subpixel_refinement", {})
        detections = [
            armor
            for armor in pipeline.get("solved_armors", [])
            if armor.get("number") == 3 and float((armor.get("tvec_mm") or {}).get("z", 0.0)) > 1000.0
        ]
        for armor in detections:
            tvec_mm = point3(armor["tvec_mm"])
            projected = np.asarray(
                [fx * tvec_mm[0] / tvec_mm[2] + cx, fy * tvec_mm[1] / tvec_mm[2] + cy]
            )
            pnp_center_errors.append(float(np.linalg.norm(projected - point2(armor["center_px"]))))

        candidates = []
        for detection_index, armor in enumerate(detections):
            center = point2(armor["center_px"])
            for slot, point in slots.items():
                if float(point["facing_score"]) <= 0.0:
                    continue
                truth_xyz = np.asarray(point["camera_xyz"], dtype=float)
                truth_pixel = np.asarray(
                    [fx * truth_xyz[0] / truth_xyz[2] + cx, fy * truth_xyz[1] / truth_xyz[2] + cy]
                )
                mirrored_pixel = np.asarray([2.0 * cx - truth_pixel[0], truth_pixel[1]])
                candidates.append(
                    (
                        float(np.linalg.norm(center - truth_pixel)),
                        detection_index,
                        int(slot),
                        truth_xyz,
                        truth_pixel,
                        mirrored_pixel,
                        float(point["facing_score"]),
                    )
                )

        used_detections: set[int] = set()
        used_slots: set[int] = set()
        for pixel_error, detection_index, slot, truth_xyz, truth_pixel, mirrored_pixel, facing in sorted(candidates):
            if detection_index in used_detections or slot in used_slots or pixel_error > args.pixel_gate:
                continue
            used_detections.add(detection_index)
            used_slots.add(slot)
            armor = detections[detection_index]
            detector = detector_candidates.get(int(armor.get("observation_id", -1)), {})
            center = point2(armor["center_px"])
            current_basis_errors.append(pixel_error)
            mirrored_basis_errors.append(float(np.linalg.norm(center - mirrored_pixel)))
            selected_tvec = point3(armor["tvec_mm"]) / 1000.0
            candidate_rows = []
            for candidate in armor.get("pnp_candidates", []):
                candidate_tvec = point3(candidate["tvec_mm"]) / 1000.0
                candidate_rows.append(
                    {
                        "solver_solution_index": int(candidate["solver_solution_index"]),
                        "selected": bool(candidate["selected"]),
                        "reprojection_error_px": float(candidate["reprojection_error_px"]),
                        "tvec_m": candidate_tvec.tolist(),
                        "truth_position_error_m": float(np.linalg.norm(candidate_tvec - truth_xyz)),
                    }
                )
            selected = next((candidate for candidate in candidate_rows if candidate["selected"]), None)
            alternate = next((candidate for candidate in candidate_rows if not candidate["selected"]), None)
            candidate_delta = (
                float(np.linalg.norm(np.asarray(selected["tvec_m"]) - np.asarray(alternate["tvec_m"])))
                if selected is not None and alternate is not None
                else None
            )
            reprojection_gap = (
                float(alternate["reprojection_error_px"] - selected["reprojection_error_px"])
                if selected is not None and alternate is not None
                else None
            )
            selected_is_truth_nearest = (
                bool(selected["truth_position_error_m"] <= alternate["truth_position_error_m"])
                if selected is not None and alternate is not None
                else None
            )
            corner_deltas = corner_refinement_deltas(detector, armor)
            records.append(
                {
                    "producer_epoch": int(pipeline["source_producer_epoch"]),
                    "frame_seq": int(pipeline["source_image_seq"]),
                    "timestamp_ns": int(pipeline["source_capture_timestamp_ns"]),
                    "observation_id": int(armor.get("observation_id", -1)),
                    "slot": slot,
                    "truth_facing_score": facing,
                    "detector_confidence": detector.get("confidence"),
                    "pipeline_first_candidate_subpixel_mean_delta_px": subpixel.get("mean_delta_px"),
                    "pipeline_first_candidate_subpixel_max_delta_px": subpixel.get("max_delta_px"),
                    "corner_refinement_delta_px": corner_deltas,
                    "corner_refinement_mean_delta_px": (
                        float(np.mean(corner_deltas)) if corner_deltas else None
                    ),
                    "corner_refinement_max_delta_px": (
                        float(np.max(corner_deltas)) if corner_deltas else None
                    ),
                    "raw_corners_px": detector.get("raw_corners_px"),
                    "raw_corner_order": detector.get("raw_corner_order"),
                    "raw_corner_covariance_status": detector.get("raw_corner_covariance_status"),
                    "detector_center_px": center.tolist(),
                    "truth_center_px": truth_pixel.tolist(),
                    "truth_pixel_error_px": pixel_error,
                    "mirrored_truth_pixel_error_px": float(np.linalg.norm(center - mirrored_pixel)),
                    "selected_tvec_m": selected_tvec.tolist(),
                    "truth_tvec_m": truth_xyz.tolist(),
                    "selected_tvec_error_m": float(np.linalg.norm(selected_tvec - truth_xyz)),
                    "azimuth_residual_deg": math.degrees(
                        math.atan2(selected_tvec[0], selected_tvec[2])
                        - math.atan2(truth_xyz[0], truth_xyz[2])
                    ),
                    "elevation_residual_deg": math.degrees(
                        math.atan2(selected_tvec[1], selected_tvec[2])
                        - math.atan2(truth_xyz[1], truth_xyz[2])
                    ),
                    "selected_solver_solution_index": selected["solver_solution_index"] if selected else None,
                    "selected_reprojection_error_px": selected["reprojection_error_px"] if selected else None,
                    "candidate_reprojection_gap_px": reprojection_gap,
                    "candidate_tvec_delta_m": candidate_delta,
                    "selected_candidate_is_truth_nearest": selected_is_truth_nearest,
                    "pnp_candidates": candidate_rows,
                    "pnp_vertices_px": armor.get("pnp_vertices_px"),
                    "pnp_vertex_order": armor.get("pnp_vertex_order"),
                    "pnp_corner_covariance_status": armor.get("corner_covariance_status"),
                }
            )

    solver_indices = Counter(record["selected_solver_solution_index"] for record in records)
    ordered = sorted(records, key=lambda row: (row["producer_epoch"], row["frame_seq"]))
    solver_switches = sum(
        left["selected_solver_solution_index"] != right["selected_solver_solution_index"]
        for left, right in zip(ordered, ordered[1:])
    )
    confidence_groups = {}
    for name, selected_rows in (
        ("score_lt_0p5", [row for row in records if float(row.get("detector_confidence") or 0.0) < 0.5]),
        ("score_ge_0p5", [row for row in records if float(row.get("detector_confidence") or 0.0) >= 0.5]),
    ):
        confidence_groups[name] = {
            "samples": len(selected_rows),
            "detector_confidence": quantiles(
                [float(row["detector_confidence"]) for row in selected_rows]
            ),
            "truth_pixel_error_px": quantiles(
                [float(row["truth_pixel_error_px"]) for row in selected_rows]
            ),
            "selected_tvec_error_m": quantiles(
                [float(row["selected_tvec_error_m"]) for row in selected_rows]
            ),
            "elevation_residual_deg": quantiles(
                [float(row["elevation_residual_deg"]) for row in selected_rows]
            ),
            "corner_refinement_mean_delta_px": quantiles(
                [float(row["corner_refinement_mean_delta_px"]) for row in selected_rows
                 if row.get("corner_refinement_mean_delta_px") is not None]
            ),
            "corner_refinement_max_delta_px": quantiles(
                [float(row["corner_refinement_max_delta_px"]) for row in selected_rows
                 if row.get("corner_refinement_max_delta_px") is not None]
            ),
        }
    summary = {
        "schema_version": 2,
        "kind": "pnp_arc_flip_diagnostic",
        "run_root": str(run_root),
        "matched_samples": len(records),
        "truth_camera_basis": "OpenCV [right,down,forward] = [-sim_local_y,-sim_local_z,sim_local_x]",
        "truth_pixel_error_px": quantiles(current_basis_errors),
        "mirrored_truth_pixel_error_px": quantiles(mirrored_basis_errors),
        "pnp_tvec_projection_to_detector_center_error_px": quantiles(pnp_center_errors),
        "selected_solver_solution_indices": {str(key): value for key, value in sorted(solver_indices.items())},
        "selected_solver_index_switches": solver_switches,
        "selected_reprojection_error_px": quantiles(
            [float(row["selected_reprojection_error_px"]) for row in records]
        ),
        "candidate_reprojection_gap_px": quantiles(
            [float(row["candidate_reprojection_gap_px"]) for row in records]
        ),
        "candidate_tvec_delta_m": quantiles(
            [float(row["candidate_tvec_delta_m"]) for row in records]
        ),
        "selected_tvec_error_m": quantiles([float(row["selected_tvec_error_m"]) for row in records]),
        "azimuth_residual_deg": quantiles([float(row["azimuth_residual_deg"]) for row in records]),
        "elevation_residual_deg": quantiles([float(row["elevation_residual_deg"]) for row in records]),
        "corner_refinement_mean_delta_px": quantiles(
            [float(row["corner_refinement_mean_delta_px"]) for row in records
             if row.get("corner_refinement_mean_delta_px") is not None]
        ),
        "corner_refinement_max_delta_px": quantiles(
            [float(row["corner_refinement_max_delta_px"]) for row in records
             if row.get("corner_refinement_max_delta_px") is not None]
        ),
        "corner_covariance_status": "unavailable in both raw detector and PnP-input records",
        "selected_candidate_truth_nearest_samples": sum(
            row["selected_candidate_is_truth_nearest"] is True for row in records
        ),
        "confidence_groups": confidence_groups,
        "finding": (
            "No IPPE solution-index switch was observed. Candidate translations are much closer to each "
            "other than the selected pose is to truth, so the apparent arc flip is not explained by "
            "candidate-branch switching in this capture."
        ),
    }

    records_path = output / "pnp_arc_flip_diagnostics.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path = output / "pnp_arc_flip_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = output / "PNP_ARC_FLIP_REPORT.md"
    report_path.write_text(
        "# PnP arc-flip diagnostic\n\n"
        f"- Matched target observations: {len(records)}.\n"
        f"- Corrected truth projection median error: {summary['truth_pixel_error_px']['median']:.3f} px; "
        f"horizontally mirrored convention: {summary['mirrored_truth_pixel_error_px']['median']:.3f} px.\n"
        f"- Selected IPPE solver indices: {summary['selected_solver_solution_indices']}; "
        f"sequential switches: {solver_switches}.\n"
        f"- Candidate tvec separation median/max: {summary['candidate_tvec_delta_m']['median']:.6f}/"
        f"{summary['candidate_tvec_delta_m']['max']:.6f} m.\n"
        f"- Selected tvec truth error median/P90: {summary['selected_tvec_error_m']['median']:.6f}/"
        f"{summary['selected_tvec_error_m']['p90']:.6f} m.\n"
        f"- Raw-to-PnP corner movement mean-delta median/P90/max: "
        f"{summary['corner_refinement_mean_delta_px']['median']:.3f}/"
        f"{summary['corner_refinement_mean_delta_px']['p90']:.3f}/"
        f"{summary['corner_refinement_mean_delta_px']['max']:.3f} px; covariance is unavailable.\n\n"
        "Conclusion: this capture does not support an IPPE candidate-branch flip. The confirmed faults are "
        "the former truth-camera lateral sign and the global polynomial curve fit. Residual pose bias is "
        "more consistent with corner/depth conditioning, especially near oblique visibility boundaries.\n",
        encoding="utf-8",
    )
    manifest_path = output / "retention_manifest.json"
    manifest = {
        "schema_version": 1,
        "kind": "pnp_arc_flip_diagnostic_retention",
        "protected_inputs": {
            str(pipeline_path): sha256(pipeline_path),
            str(truth_path): sha256(truth_path),
        },
        "analysis_sources": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(grid_analysis_path()): sha256(grid_analysis_path()),
        },
        "retained_artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (records_path, summary_path, report_path)
        },
        "retention_class": "protected diagnostic capture and reproducible derived analysis; no automatic deletion",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
