#!/usr/bin/env python3
"""Train the frozen-design corner repairer on session-disjoint formal data."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import cv2
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.stage3.train_image_corner_repair_pilot import (
    ImageGeometryResidualNet,
    PATCH_HEIGHT,
    PATCH_WIDTH,
    digest,
    load_session_rows,
)


SCHEMA = "aim-stack.corner-repair-validation-result/1"
EPOCHS = 36
BATCH_SIZE = 96
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
PATIENCE = 8
SEED = 1701
CONTEXT_SCALE = 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=("v1-global", "v2-context-spatial"), default="v1-global")
    parser.add_argument("--minimum-detector-score", type=float)
    parser.add_argument("--minimum-predicted-correction-rms-px", type=float)
    parser.add_argument("--target-space", choices=("full-pixel-residual", "context-normalized-residual"), default="full-pixel-residual")
    return parser.parse_args()


def context_patch(image: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Warp an expanded raw quadrilateral so endpoint context remains visible."""
    if image is None or image.shape != (1080, 1440, 4):
        raise ValueError("stored image is not a contract-compatible RGBA32 frame")
    raw_quad = np.asarray([raw[1], raw[0], raw[3], raw[2]], dtype=np.float32)
    center = raw_quad.mean(axis=0, keepdims=True)
    source = center + (raw_quad - center) * CONTEXT_SCALE
    destination = np.asarray(
        [[0, 0], [0, PATCH_HEIGHT - 1], [PATCH_WIDTH - 1, PATCH_HEIGHT - 1], [PATCH_WIDTH - 1, 0]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    warped = cv2.warpPerspective(bgr, transform, (PATCH_WIDTH, PATCH_HEIGHT), flags=cv2.INTER_LINEAR)
    return np.transpose(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), (2, 0, 1)).astype(np.float32) / 255.0


def context_transform(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_quad = np.asarray([raw[1], raw[0], raw[3], raw[2]], dtype=np.float32)
    center = raw_quad.mean(axis=0, keepdims=True)
    source = center + (raw_quad - center) * CONTEXT_SCALE
    destination = np.asarray(
        [[0, 0], [0, PATCH_HEIGHT - 1], [PATCH_WIDTH - 1, PATCH_HEIGHT - 1], [PATCH_WIDTH - 1, 0]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return transform, np.linalg.inv(transform)


def normalized_context_targets(raw: np.ndarray, exact: np.ndarray) -> np.ndarray:
    scale = np.asarray([PATCH_WIDTH, PATCH_HEIGHT], dtype=np.float32)
    output = []
    for raw_item, exact_item in zip(raw, exact):
        transform, _ = context_transform(raw_item)
        raw_patch = cv2.perspectiveTransform(raw_item[None], transform)[0]
        exact_patch = cv2.perspectiveTransform(exact_item[None], transform)[0]
        output.append(((exact_patch - raw_patch) / scale).reshape(-1))
    return np.asarray(output, dtype=np.float32)


def normalized_context_predictions_to_full(raw: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    scale = np.asarray([PATCH_WIDTH, PATCH_HEIGHT], dtype=np.float32)
    output = []
    for raw_item, predicted_item in zip(raw, predicted.reshape(-1, 4, 2)):
        transform, inverse = context_transform(raw_item)
        raw_patch = cv2.perspectiveTransform(raw_item[None], transform)[0]
        predicted_patch = raw_patch + predicted_item * scale
        predicted_full = cv2.perspectiveTransform(predicted_patch[None], inverse)[0]
        output.append((predicted_full - raw_item).reshape(-1))
    return np.asarray(output, dtype=np.float32)


class ContextSpatialResidualNet(nn.Module):
    family = "image-context-patch-plus-raw-geometry-spatial-corner-residual-v2"

    def __init__(self) -> None:
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 4 * 8 + 15, 256), nn.SiLU(), nn.Dropout(0.05),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 8),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, image: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.image(image), geometry), dim=1))


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def improvement(raw: float, repaired: float) -> float:
    return (raw - repaired) / raw if raw > 0 else 0.0


def verified_manifest(path: Path) -> tuple[dict[str, object], dict[str, float]]:
    resolved = path.resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if value.get("schema_version") != "aim-stack.corner-repair-detector-dataset/1":
        raise ValueError(f"unsupported detector dataset: {resolved}")
    if value.get("test_accessed") is not False or set(value.get("splits", ())) - {"train", "validation"}:
        raise PermissionError(f"formal training input accesses the sealed test split: {resolved}")
    if not set(value.get("splits", ())) or set(value.get("splits", ())) - {"train", "validation"}:
        raise ValueError(f"formal dataset must contain only nonempty train/validation splits: {resolved}")
    if digest(Path(str(value["plan"]))) != value["plan_sha256"]:
        raise ValueError(f"formal plan changed after dataset build: {resolved}")
    if digest(Path(str(value["detector_model"]))) != value["detector_model_sha256"]:
        raise ValueError(f"detector model changed after dataset build: {resolved}")
    plan = json.loads(Path(str(value["plan"])).read_text(encoding="utf-8"))
    gate = plan["split_policy"]["validation_gate"]
    return value, {
        "minimum_aggregate_rms_improvement_fraction": float(gate["minimum_aggregate_rms_improvement_fraction"]),
        "maximum_per_mode_rms_regression_fraction": float(gate["maximum_per_mode_rms_regression_fraction"]),
    }


def group_metrics(targets: np.ndarray, repaired: np.ndarray, indices: list[int]) -> dict[str, object]:
    selected = np.asarray(indices, dtype=np.int64)
    raw_rms = rms(targets[selected])
    repaired_rms = rms(repaired[selected])
    corner_raw = np.linalg.norm(targets[selected].reshape(-1, 4, 2), axis=2)
    corner_repaired = np.linalg.norm(repaired[selected].reshape(-1, 4, 2), axis=2)
    return {
        "rows": len(indices),
        "raw_coordinate_rms_px": raw_rms,
        "repaired_coordinate_rms_px": repaired_rms,
        "rms_improvement_fraction": improvement(raw_rms, repaired_rms),
        "raw_corner_error_p50_px": float(np.percentile(corner_raw, 50)),
        "raw_corner_error_p95_px": float(np.percentile(corner_raw, 95)),
        "repaired_corner_error_p50_px": float(np.percentile(corner_repaired, 50)),
        "repaired_corner_error_p95_px": float(np.percentile(corner_repaired, 95)),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite formal model evidence: {output}")
    output.mkdir(parents=True)
    checkpoint = output / "corner-repair.pt"
    result_path = output / "validation-result.json"
    samples_path = output / "validation-errors.csv"

    manifests: list[tuple[Path, dict[str, object]]] = []
    gates: list[dict[str, float]] = []
    detector_hashes: set[str] = set()
    association_gates: set[float] = set()
    for supplied in args.dataset_manifest:
        path = supplied.resolve(strict=True)
        value, gate = verified_manifest(path)
        manifests.append((path, value))
        gates.append(gate)
        detector_hashes.add(str(value["detector_model_sha256"]))
        if "match_rms_px" not in value:
            raise ValueError(f"formal dataset does not declare detector-to-truth association gate: {path}")
        association_gates.add(float(value["match_rms_px"]))
    if len(detector_hashes) != 1:
        raise ValueError("all formal datasets must use the same frozen detector")
    if len(association_gates) != 1:
        raise ValueError("all formal datasets must use the same detector-to-truth association gate")
    if any(gate != gates[0] for gate in gates[1:]):
        raise ValueError("all formal plans must predeclare identical validation gates")

    images: list[np.ndarray] = []
    geometry: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    raw_corners: list[np.ndarray] = []
    exact_corners: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    detector_scores: list[float] = []
    seen_sessions: set[str] = set()
    applicability: list[dict[str, object]] = []
    for manifest_path, manifest in manifests:
        for entry in manifest["sessions"]:
            session_id = str(entry["session_id"])
            if session_id in seen_sessions:
                raise ValueError(f"duplicate complete session across manifests: {session_id}")
            seen_sessions.add(session_id)
            rows_path = Path(str(entry["rows"]["path"])).resolve(strict=True)
            result_path_in = Path(str(entry["session_result"]["path"])).resolve(strict=True)
            if digest(rows_path) != entry["rows"]["sha256"] or digest(result_path_in) != entry["session_result"]["sha256"]:
                raise ValueError(f"formal session evidence changed: {session_id}")
            session_dir = result_path_in.parent
            patch_fn = context_patch if args.architecture == "v2-context-spatial" else None
            loaded, source = load_session_rows(rows_path, session_dir, **({"patch_fn": patch_fn} if patch_fn else {}))
            with rows_path.open(encoding="utf-8", newline="") as handle:
                score_rows = [row for row in csv.DictReader(handle) if row["motion_uniform"] == "True"]
            if len(score_rows) != len(loaded["targets"]):
                raise ValueError(f"detector score alignment failed: {session_id}")
            detector_scores.extend(float(row["detector_score"]) for row in score_rows)
            source.pop("keys")
            offset = sum(len(value) for value in targets)
            images.append(loaded["images"])
            geometry.append(loaded["geometry"])
            targets.append(loaded["targets"])
            raw_corners.append(loaded["raw_corners"])
            exact_corners.append(loaded["exact_corners"])
            records.extend({
                "index": offset + local,
                "session_id": session_id,
                "split": entry["split"],
                "mode": entry["mode"],
            } for local in range(len(loaded["targets"])))
            qualified = entry.get("qualified_exposures")
            coverage = entry.get("detector_exposure_coverage_fraction")
            applicability.append({
                "dataset_manifest": str(manifest_path), "session_id": session_id,
                "split": entry["split"], "mode": entry["mode"],
                "matched_exposures": entry["rows"]["matched_exposures"],
                "qualified_exposures": qualified,
                "detector_exposure_coverage_fraction": coverage,
                "uniform_repair_rows": source["rows_uniform"],
            })
    values = {
        "images": np.concatenate(images),
        "geometry": np.concatenate(geometry),
        "targets": np.concatenate(targets),
        "raw_corners": np.concatenate(raw_corners),
        "exact_corners": np.concatenate(exact_corners),
    }
    model_targets = (
        normalized_context_targets(values["raw_corners"], values["exact_corners"])
        if args.target_space == "context-normalized-residual" else values["targets"]
    )
    split = np.asarray([record["split"] for record in records])
    train_mask, validation_mask = split == "train", split == "validation"
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("formal train or validation split is empty")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    target_mean = model_targets[train_mask].mean(0)
    target_std = np.maximum(model_targets[train_mask].std(0), 1.0e-6)
    geometry_mean = values["geometry"][train_mask].mean(0)
    geometry_std = np.maximum(values["geometry"][train_mask].std(0), 1.0e-6)
    standardized_geometry = (values["geometry"] - geometry_mean) / geometry_std
    standardized_targets = (model_targets - target_mean) / target_std
    train_dataset = TensorDataset(
        torch.from_numpy(values["images"][train_mask]),
        torch.from_numpy(standardized_geometry[train_mask]),
        torch.from_numpy(standardized_targets[train_mask]),
    )
    loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    validation_images = torch.from_numpy(values["images"][validation_mask])
    validation_geometry = torch.from_numpy(standardized_geometry[validation_mask])
    validation_targets = torch.from_numpy(standardized_targets[validation_mask])
    model = ContextSpatialResidualNet() if args.architecture == "v2-context-spatial" else ImageGeometryResidualNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation, stale = float("inf"), 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        training_losses: list[float] = []
        for batch_image, batch_geometry, batch_target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean(torch.square(model(batch_image, batch_geometry) - batch_target))
            loss.backward()
            optimizer.step()
            training_losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(torch.mean(torch.square(model(validation_images, validation_geometry) - validation_targets)).item())
        history.append({"epoch": epoch, "training_normalized_mse": float(np.mean(training_losses)), "validation_normalized_mse": validation_loss})
        if validation_loss < best_validation:
            best_validation, stale = validation_loss, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted = model(torch.from_numpy(values["images"]), torch.from_numpy(standardized_geometry)).numpy()
    predicted = predicted * target_std + target_mean
    predicted_full = (
        normalized_context_predictions_to_full(values["raw_corners"], predicted)
        if args.target_space == "context-normalized-residual" else predicted
    )
    predicted_correction_rms = np.sqrt(np.mean(np.square(predicted_full), axis=1))
    apply_repair = np.ones(len(predicted), dtype=bool)
    if args.minimum_detector_score is not None:
        if not 0.0 <= args.minimum_detector_score <= 1.0:
            raise ValueError("--minimum-detector-score must be in [0, 1]")
        apply_repair &= np.asarray(detector_scores) >= args.minimum_detector_score
    if args.minimum_predicted_correction_rms_px is not None:
        if args.minimum_predicted_correction_rms_px < 0.0:
            raise ValueError("--minimum-predicted-correction-rms-px must be nonnegative")
        apply_repair &= predicted_correction_rms >= args.minimum_predicted_correction_rms_px
    predicted_full[~apply_repair] = 0.0
    repaired = values["targets"] - predicted_full

    validation_indices = np.flatnonzero(validation_mask).tolist()
    aggregate = group_metrics(values["targets"], repaired, validation_indices)
    by_session_indices: dict[str, list[int]] = defaultdict(list)
    by_mode_indices: dict[str, list[int]] = defaultdict(list)
    for record in records:
        if record["split"] == "validation":
            by_session_indices[str(record["session_id"])].append(int(record["index"]))
            by_mode_indices[str(record["mode"])].append(int(record["index"]))
    by_session = {name: group_metrics(values["targets"], repaired, indices) for name, indices in sorted(by_session_indices.items())}
    by_mode = {name: group_metrics(values["targets"], repaired, indices) for name, indices in sorted(by_mode_indices.items())}
    gate = gates[0]
    aggregate_passed = aggregate["rms_improvement_fraction"] >= gate["minimum_aggregate_rms_improvement_fraction"]
    per_mode_passed = all(
        value["rms_improvement_fraction"] >= -gate["maximum_per_mode_rms_regression_fraction"]
        for value in by_mode.values()
    )

    torch.save({
        "model_family": model.family, "state_dict": model.state_dict(),
        "geometry_mean": geometry_mean, "geometry_std": geometry_std,
        "target_mean_px": target_mean, "target_std_px": target_std,
        "patch_size": [128, 64], "context_scale": CONTEXT_SCALE if args.architecture == "v2-context-spatial" else 1.0,
        "architecture": args.architecture, "training_schema": SCHEMA,
        "minimum_detector_score": args.minimum_detector_score,
        "minimum_predicted_correction_rms_px": args.minimum_predicted_correction_rms_px,
        "target_space": args.target_space, "target_mean": target_mean, "target_std": target_std,
        "detector_model_sha256": next(iter(detector_hashes)),
        "association_match_rms_px": next(iter(association_gates)),
        "dataset_manifests": [{"path": str(path), "sha256": digest(path)} for path, _ in manifests],
    }, checkpoint)
    checkpoint_sha256 = digest(checkpoint)
    result = {
        "schema_version": SCHEMA,
        "scope": "session-disjoint formal validation; sealed test not accessed",
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha256,
        "detector_model_sha256": next(iter(detector_hashes)),
        "association_match_rms_px": next(iter(association_gates)),
        "dataset_manifests": [{"path": str(path), "sha256": digest(path)} for path, _ in manifests],
        "model": {
            "family": model.family,
            "image_input": "1.5x-expanded raw-detector quadrilateral warped to rgb-128x64",
            "architecture": args.architecture,
            "target_space": args.target_space,
            "context_scale": CONTEXT_SCALE if args.architecture == "v2-context-spatial" else 1.0,
            "application_policy": {
                "minimum_detector_score": args.minimum_detector_score,
                "minimum_predicted_correction_rms_px": args.minimum_predicted_correction_rms_px,
                "rejected_behavior": "return raw detector corners unchanged",
                "truth_input": False, "motion_input": False,
            },
            "geometry_input": "raw-detector-only-15d",
            "truth_input": False, "motion_input": False, "identity_input": False,
            "range_input": False, "future_input": False,
        },
        "training": {
            "seed": SEED, "device": "cpu", "epochs_completed": epoch,
            "best_validation_normalized_mse": best_validation,
            "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "early_stopping_patience": PATIENCE,
            "training_rows": int(train_mask.sum()), "validation_rows": int(validation_mask.sum()),
            "repair_applied_training_rows": int(np.sum(apply_repair & train_mask)),
            "repair_applied_validation_rows": int(np.sum(apply_repair & validation_mask)),
            "history": history,
        },
        "detector_applicability": applicability,
        "validation_metrics_px": {"aggregate": aggregate, "by_mode": by_mode, "by_session": by_session},
        "validation_gate": gate,
        "validation_gate_components": {"aggregate_passed": aggregate_passed, "per_mode_passed": per_mode_passed},
        "validation_gate_passed": aggregate_passed and per_mode_passed,
        "test_accessed": False,
    }
    with samples_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("session_id", "mode", "corner_index", "raw_error_px", "repaired_error_px"))
        for record in records:
            if record["split"] != "validation":
                continue
            index = int(record["index"])
            raw_corner = np.linalg.norm(values["targets"][index].reshape(4, 2), axis=1)
            repaired_corner = np.linalg.norm(repaired[index].reshape(4, 2), axis=1)
            for corner_index, (raw_error, repaired_error) in enumerate(zip(raw_corner, repaired_corner)):
                writer.writerow((record["session_id"], record["mode"], corner_index, float(raw_error), float(repaired_error)))
    result["validation_errors"] = {"path": str(samples_path), "sha256": digest(samples_path)}
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output), "checkpoint_sha256": checkpoint_sha256,
        "validation_gate_passed": result["validation_gate_passed"],
        "aggregate": aggregate, "by_mode": by_mode,
    }, indent=2))


if __name__ == "__main__":
    main()
