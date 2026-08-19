#!/usr/bin/env python3
"""Evaluate one frozen corner-repair checkpoint on an authorized sealed test manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.stage3.train_image_corner_repair_formal import (
    ContextSpatialResidualNet,
    ContextSpatialReliabilityNet,
    CornerHeatmapReliabilityNet,
    group_metrics,
    context_patch,
    normalized_context_predictions_to_full,
)
from training.stage3.train_image_corner_repair_pilot import (
    ImageGeometryResidualNet,
    digest,
    load_session_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--test-dataset-manifest", required=True, type=Path)
    parser.add_argument("--result-output", required=True, type=Path)
    parser.add_argument("--errors-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve(strict=True)
    manifest_path = args.test_dataset_manifest.resolve(strict=True)
    result_path = args.result_output.resolve()
    errors_path = args.errors_output.resolve()
    if result_path.exists() or errors_path.exists():
        raise FileExistsError("refusing to overwrite sealed-test evidence")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "aim-stack.corner-repair-detector-dataset/1":
        raise ValueError("unsupported test dataset manifest")
    if manifest.get("test_accessed") is not True or set(manifest.get("splits", ())) != {"test"}:
        raise PermissionError("dataset is not an authorized test-only manifest")
    checkpoint_sha256 = digest(checkpoint_path)
    authorization = manifest.get("test_authorization") or {}
    if authorization.get("repair_checkpoint_sha256") != checkpoint_sha256:
        raise PermissionError("test manifest was not authorized for this repair checkpoint")
    if digest(Path(str(manifest["plan"]))) != manifest["plan_sha256"]:
        raise ValueError("test plan changed after detector dataset build")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("detector_model_sha256") != manifest.get("detector_model_sha256"):
        raise ValueError("repair checkpoint and test dataset use different frozen detectors")
    checkpoint_association = checkpoint.get("association_match_rms_px")
    if checkpoint_association is None:
        recorded_gates = {
            float(json.loads(Path(str(item["path"])).read_text(encoding="utf-8"))["match_rms_px"])
            for item in checkpoint.get("dataset_manifests", ())
        }
        if len(recorded_gates) != 1:
            raise ValueError("legacy repair checkpoint cannot prove one association gate")
        checkpoint_association = next(iter(recorded_gates))
    if float(checkpoint_association) != float(manifest.get("match_rms_px", -2.0)):
        raise ValueError("repair checkpoint and test dataset use different association gates")
    architecture = checkpoint.get("architecture", "v1-global")
    if architecture == "v4-corner-heatmap-reliability":
        model = CornerHeatmapReliabilityNet()
        patch_fn = context_patch
    elif architecture == "v3-context-spatial-reliability":
        model = ContextSpatialReliabilityNet()
        patch_fn = context_patch
    elif architecture == "v2-context-spatial":
        model = ContextSpatialResidualNet()
        patch_fn = context_patch
    elif architecture == "v1-global":
        model = ImageGeometryResidualNet()
        patch_fn = None
    else:
        raise ValueError(f"unsupported checkpoint architecture: {architecture}")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    images: list[np.ndarray] = []
    geometry: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    raw_corners: list[np.ndarray] = []
    scores: list[float] = []
    records: list[dict[str, object]] = []
    applicability: list[dict[str, object]] = []
    for entry in manifest["sessions"]:
        rows_path = Path(str(entry["rows"]["path"])).resolve(strict=True)
        result_in = Path(str(entry["session_result"]["path"])).resolve(strict=True)
        if digest(rows_path) != entry["rows"]["sha256"] or digest(result_in) != entry["session_result"]["sha256"]:
            raise ValueError(f"sealed test evidence changed: {entry['session_id']}")
        if int(entry["rows"].get("uniform_rows", 0)) == 0:
            applicability.append({
                "session_id": entry["session_id"], "mode": entry["mode"],
                "matched_exposures": entry["rows"]["matched_exposures"],
                "qualified_exposures": entry.get("qualified_exposures"),
                "detector_exposure_coverage_fraction": entry.get("detector_exposure_coverage_fraction"),
                "uniform_repair_rows": 0,
            })
            continue
        loaded, _ = load_session_rows(rows_path, result_in.parent, **({"patch_fn": patch_fn} if patch_fn else {}))
        with rows_path.open(encoding="utf-8", newline="") as handle:
            score_rows = [row for row in csv.DictReader(handle) if row["motion_uniform"] == "True"]
        if len(score_rows) != len(loaded["targets"]):
            raise ValueError(f"test detector score alignment failed: {entry['session_id']}")
        offset = sum(len(value) for value in targets)
        images.append(loaded["images"])
        geometry.append(loaded["geometry"])
        targets.append(loaded["targets"])
        raw_corners.append(loaded["raw_corners"])
        scores.extend(float(row["detector_score"]) for row in score_rows)
        records.extend({
            "index": offset + local, "session_id": entry["session_id"], "mode": entry["mode"],
        } for local in range(len(loaded["targets"])))
        applicability.append({
            "session_id": entry["session_id"], "mode": entry["mode"],
            "matched_exposures": entry["rows"]["matched_exposures"],
            "qualified_exposures": entry.get("qualified_exposures"),
            "detector_exposure_coverage_fraction": entry.get("detector_exposure_coverage_fraction"),
            "uniform_repair_rows": len(loaded["targets"]),
        })
    values = {
        "images": np.concatenate(images), "geometry": np.concatenate(geometry),
        "targets": np.concatenate(targets),
        "raw_corners": np.concatenate(raw_corners),
    }
    standardized_geometry = (values["geometry"] - checkpoint["geometry_mean"]) / checkpoint["geometry_std"]
    with torch.no_grad():
        if architecture in {"v3-context-spatial-reliability", "v4-corner-heatmap-reliability"}:
            output = model(
                torch.from_numpy(values["images"]), torch.from_numpy(standardized_geometry),
                torch.from_numpy(np.asarray(scores, dtype=np.float32)),
            ).numpy()
            predicted = output[:, :8]
            reliability_probability = 1.0 / (1.0 + np.exp(-output[:, 8]))
        else:
            predicted = model(torch.from_numpy(values["images"]), torch.from_numpy(standardized_geometry)).numpy()
            reliability_probability = np.ones(len(predicted), dtype=np.float32)
    target_mean = checkpoint.get("target_mean", checkpoint.get("target_mean_px"))
    target_std = checkpoint.get("target_std", checkpoint.get("target_std_px"))
    if checkpoint.get("output_standardized", True):
        predicted = predicted * target_std + target_mean
    if checkpoint.get("target_space", "full-pixel-residual") == "context-normalized-residual":
        predicted = normalized_context_predictions_to_full(values["raw_corners"], predicted)
    apply_repair = np.ones(len(predicted), dtype=bool)
    if architecture in {"v3-context-spatial-reliability", "v4-corner-heatmap-reliability"}:
        apply_repair &= reliability_probability >= float(checkpoint["reliability"]["application_probability_threshold"])
    minimum_score = checkpoint.get("minimum_detector_score")
    if minimum_score is not None:
        apply_repair &= np.asarray(scores) >= float(minimum_score)
    minimum_correction = checkpoint.get("minimum_predicted_correction_rms_px")
    if minimum_correction is not None:
        apply_repair &= np.sqrt(np.mean(np.square(predicted), axis=1)) >= float(minimum_correction)
    predicted[~apply_repair] = 0.0
    repaired = values["targets"] - predicted

    all_indices = list(range(len(records)))
    aggregate = group_metrics(values["targets"], repaired, all_indices)
    by_mode_indices: dict[str, list[int]] = defaultdict(list)
    by_session_indices: dict[str, list[int]] = defaultdict(list)
    for record in records:
        by_mode_indices[str(record["mode"])].append(int(record["index"]))
        by_session_indices[str(record["session_id"])].append(int(record["index"]))
    by_mode = {name: group_metrics(values["targets"], repaired, indices) for name, indices in sorted(by_mode_indices.items())}
    by_session = {name: group_metrics(values["targets"], repaired, indices) for name, indices in sorted(by_session_indices.items())}
    plan = json.loads(Path(str(manifest["plan"])).read_text(encoding="utf-8"))
    gate = plan["split_policy"]["test_gate"]
    aggregate_passed = aggregate["rms_improvement_fraction"] >= float(gate["minimum_aggregate_rms_improvement_fraction"])
    per_mode_passed = all(
        value["rms_improvement_fraction"] >= -float(gate["maximum_per_mode_rms_regression_fraction"])
        for value in by_mode.values()
    )

    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with errors_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("session_id", "mode", "corner_index", "raw_error_px", "repaired_error_px"))
        for record in records:
            index = int(record["index"])
            raw_corner = np.linalg.norm(values["targets"][index].reshape(4, 2), axis=1)
            repaired_corner = np.linalg.norm(repaired[index].reshape(4, 2), axis=1)
            for corner_index, (raw_error, repaired_error) in enumerate(zip(raw_corner, repaired_corner)):
                writer.writerow((record["session_id"], record["mode"], corner_index, float(raw_error), float(repaired_error)))
    result = {
        "schema_version": "aim-stack.corner-repair-test-result/1",
        "scope": "single untouched session-disjoint sealed test; no model update performed",
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha256,
        "test_dataset_manifest": str(manifest_path), "test_dataset_manifest_sha256": digest(manifest_path),
        "model_family": checkpoint["model_family"], "architecture": architecture,
        "association_match_rms_px": float(checkpoint_association),
        "application_policy": {
            "minimum_detector_score": minimum_score,
            "minimum_predicted_correction_rms_px": minimum_correction,
            "rejected_behavior": "return raw detector corners unchanged",
        },
        "detector_applicability": applicability,
        "test_metrics_px": {"aggregate": aggregate, "by_mode": by_mode, "by_session": by_session},
        "test_gate": gate,
        "test_gate_components": {"aggregate_passed": aggregate_passed, "per_mode_passed": per_mode_passed},
        "test_gate_passed": aggregate_passed and per_mode_passed,
        "repair_applied_rows": int(apply_repair.sum()), "test_rows": len(records),
        "test_errors": {"path": str(errors_path), "sha256": digest(errors_path)},
        "training_or_parameter_update_performed": False,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "test_gate_passed": result["test_gate_passed"], "aggregate": aggregate,
        "by_mode": by_mode, "repair_applied_rows": int(apply_repair.sum()), "test_rows": len(records),
    }, indent=2))


if __name__ == "__main__":
    main()
