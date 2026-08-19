#!/usr/bin/env python3
"""Train an image-conditioned, detector-only four-corner repair pilot.

This is deliberately a *single-session feasibility run*.  A sample contains a
warped RGB patch made from its raw detector quadrilateral plus the frozen 15-D
raw-corner geometry feature.  Simulator exact corners are read only as the
supervised residual target.  No truth, range, motion, identity, PnP or future
field enters the model input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.stage3.corner_residual_network import observable_features


ORDER = ("bl", "tl", "tr", "br")
PATCH_WIDTH, PATCH_HEIGHT = 128, 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--validation-rows", type=Path, help="optional detector rows from a distinct complete session")
    parser.add_argument("--validation-session", type=Path, help="session owning --validation-rows; required with it")
    parser.add_argument("--metrics-output", required=True, type=Path)
    parser.add_argument("--checkpoint-output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--validation-modulo", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def columns(prefix: str) -> list[str]:
    return [f"{prefix}_{corner}_{axis}_px" for corner in ORDER for axis in ("x", "y")]


def corners(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray([float(row[key]) for key in columns(prefix)], dtype=np.float32).reshape(4, 2)


def frame_key(row: dict[str, str]) -> str:
    return "|".join(row[key] for key in ("producer_epoch", "frame_seq", "timestamp_ns"))


def validation_frame(key: str, modulo: int) -> bool:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % modulo == 0


def patch(image: np.ndarray, raw: np.ndarray) -> np.ndarray:
    if image is None or image.shape != (1080, 1440, 4):
        raise ValueError("stored image is not a contract-compatible RGBA32 frame")
    source = np.asarray([raw[1], raw[0], raw[3], raw[2]], dtype=np.float32)
    destination = np.asarray(
        [[0, 0], [0, PATCH_HEIGHT - 1], [PATCH_WIDTH - 1, PATCH_HEIGHT - 1], [PATCH_WIDTH - 1, 0]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    # `image` comes from the Release's literal RGBA32 frame export.  It is not
    # a PNG decoded by OpenCV, so BGRA conversion would swap red and blue.
    bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    warped = cv2.warpPerspective(bgr, transform, (PATCH_WIDTH, PATCH_HEIGHT), flags=cv2.INTER_LINEAR)
    return np.transpose(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), (2, 0, 1)).astype(np.float32) / 255.0


def load_release_ledger(session: Path) -> dict[tuple[int, int, int], dict[str, object]]:
    ledger: dict[tuple[int, int, int], dict[str, object]] = {}
    with (session / "tcp-identities.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            key = tuple(int(item[name]) for name in ("producer_epoch", "frame_seq", "timestamp_ns"))
            if key in ledger:
                raise ValueError(f"duplicate Release identity: {key}")
            ledger[key] = item
    return ledger


def load_release_rgba(session: Path, identity: dict[str, object]) -> np.ndarray:
    required = ("raw_rgba_file", "raw_rgba_sha256", "payload_sha256", "payload_bytes")
    if any(field not in identity for field in required):
        raise ValueError("Release ledger lacks required full-frame export metadata")
    relative = Path(str(identity["raw_rgba_file"]))
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("frames",):
        raise ValueError(f"unsafe Release raw-frame path: {relative}")
    payload = (session / relative).read_bytes()
    if len(payload) != int(identity["payload_bytes"]) or len(payload) != 1440 * 1080 * 4:
        raise ValueError(f"stored image is not a contract-compatible raw RGBA32 frame: {relative}")
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != identity["payload_sha256"] or actual_sha256 != identity["raw_rgba_sha256"]:
        raise ValueError(f"Release raw-frame hash does not match ledger: {relative}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(1080, 1440, 4).copy()


def load_session_rows(rows_path: Path, session: Path, patch_fn=patch) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with rows_path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError("no detector-matched rows")
    if any(record.get("future_truth_included") != "False" for record in records):
        raise ValueError("training rows violate the no-future-truth input contract")
    kept = [record for record in records if record["motion_uniform"] == "True"]
    if not kept:
        raise ValueError("no uniform-motion rows")
    ledger = load_release_ledger(session)
    images, geometry, targets, raw_corners, exact_corners, keys = [], [], [], [], [], []
    for record in kept:
        raw = corners(record, "raw")
        exact = corners(record, "exact")
        frame = frame_key(record)
        identity = tuple(int(record[name]) for name in ("producer_epoch", "frame_seq", "timestamp_ns"))
        if identity not in ledger:
            raise ValueError(f"detector row lacks a Release identity: {identity}")
        if record["image_file"] != str(ledger[identity]["raw_rgba_file"]):
            raise ValueError(f"detector row raw-frame path does not match Release ledger: {identity}")
        rgba = load_release_rgba(session, ledger[identity])
        images.append(patch_fn(rgba, raw))
        geometry.append(observable_features(raw))
        targets.append((exact - raw).reshape(-1))
        raw_corners.append(raw)
        exact_corners.append(exact)
        keys.append(frame)
    tensors = {
        "images": np.asarray(images, dtype=np.float32),
        "geometry": np.asarray(geometry, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
        "raw_corners": np.asarray(raw_corners, dtype=np.float32),
        "exact_corners": np.asarray(exact_corners, dtype=np.float32),
    }
    return tensors, {"rows_total": len(records), "rows_uniform": len(kept), "unique_frames": len(set(keys)), "keys": keys}


def load(rows_path: Path, session: Path, modulo: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    tensors, source = load_session_rows(rows_path, session)
    validation = np.asarray([validation_frame(key, modulo) for key in source.pop("keys")], dtype=bool)
    tensors["validation"] = validation
    if not tensors["validation"].any() or tensors["validation"].all():
        raise ValueError("frame-group validation split is empty")
    manifest = {
        **source,
        "validation_frames": int(validation.sum()),
        "training_frames": int((~validation).sum()),
    }
    return tensors, manifest


def load_session_split(
    training_rows: Path, training_session: Path, validation_rows: Path, validation_session: Path
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if training_session == validation_session:
        raise ValueError("training and validation must be different complete session directories")
    train, train_source = load_session_rows(training_rows, training_session)
    validation, validation_source = load_session_rows(validation_rows, validation_session)
    train_source.pop("keys")
    validation_source.pop("keys")
    values = {
        key: np.concatenate((train[key], validation[key]), axis=0)
        for key in ("images", "geometry", "targets")
    }
    values["validation"] = np.concatenate(
        (np.zeros(len(train["targets"]), dtype=bool), np.ones(len(validation["targets"]), dtype=bool))
    )
    return values, {
        "training_session": train_source,
        "validation_session": validation_source,
        "training_frames": len(train["targets"]),
        "validation_frames": len(validation["targets"]),
    }


class ImageGeometryResidualNet(nn.Module):
    family = "image-patch-plus-raw-geometry-corner-residual-pilot-v1"

    def __init__(self) -> None:
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(48 + 15, 96), nn.SiLU(), nn.Dropout(0.05),
            nn.Linear(96, 8),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, image: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.image(image), geometry), dim=1))


def rms(values: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(torch.square(values))).item())


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.validation_modulo) <= 0:
        raise ValueError("epochs, batch size, patience and validation modulo must be positive")
    rows, session = args.rows.resolve(strict=True), args.session.resolve(strict=True)
    if (args.validation_rows is None) != (args.validation_session is None):
        raise ValueError("--validation-rows and --validation-session must be supplied together")
    validation_rows = args.validation_rows.resolve(strict=True) if args.validation_rows else None
    validation_session = args.validation_session.resolve(strict=True) if args.validation_session else None
    metrics, checkpoint = args.metrics_output.resolve(), args.checkpoint_output.resolve()
    if metrics.exists() or checkpoint.exists():
        raise FileExistsError("refusing to overwrite pilot evidence or protected checkpoint")
    if args.device == "cuda":
        raise ValueError("the current pilot is explicitly CPU-only; do not silently change its runtime contract")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if validation_rows is None:
        values, split = load(rows, session, args.validation_modulo)
        scope = "single-session feasibility only; frame-group holdout is not an independent generalization test"
        provenance = {
            "rows_csv": str(rows), "rows_sha256": digest(rows), "session": str(session),
            "capture_manifest_sha256": digest(session / "capture-manifest.json"), "checkpoint": str(checkpoint),
        }
        split_metadata = {**split, "unit": "TCP frame identity", "validation_selector": f"sha256(frame_identity) mod {args.validation_modulo} == 0"}
    else:
        values, split = load_session_split(rows, session, validation_rows, validation_session)
        scope = "two-session disjoint smoke only; not a complete motion-coverage, model-selection, or deployment result"
        provenance = {
            "training_rows_csv": str(rows), "training_rows_sha256": digest(rows), "training_session": str(session),
            "training_capture_manifest_sha256": digest(session / "capture-manifest.json"),
            "validation_rows_csv": str(validation_rows), "validation_rows_sha256": digest(validation_rows),
            "validation_session": str(validation_session),
            "validation_capture_manifest_sha256": digest(validation_session / "capture-manifest.json"), "checkpoint": str(checkpoint),
        }
        split_metadata = {**split, "unit": "complete session directory", "validation_selector": "explicit distinct validation session"}
    train = ~values["validation"]
    target_mean, target_std = values["targets"][train].mean(0), np.maximum(values["targets"][train].std(0), 1.0e-6)
    geo_mean, geo_std = values["geometry"][train].mean(0), np.maximum(values["geometry"][train].std(0), 1.0e-6)
    geometry = (values["geometry"] - geo_mean) / geo_std
    targets = (values["targets"] - target_mean) / target_std
    dataset = TensorDataset(torch.from_numpy(values["images"][train]), torch.from_numpy(geometry[train]), torch.from_numpy(targets[train]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    model = ImageGeometryResidualNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state, best_validation, stale = None, float("inf"), 0
    validation_images = torch.from_numpy(values["images"][~train])
    validation_geometry = torch.from_numpy(geometry[~train])
    validation_targets = torch.from_numpy(values["targets"][~train])
    for epoch in range(1, args.epochs + 1):
        model.train()
        for image, feature, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean(torch.square(model(image, feature) - target))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(torch.mean(torch.square(model(validation_images, validation_geometry) - validation_targets)).item())
        if validation_loss < best_validation:
            best_validation, stale = validation_loss, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted = model(torch.from_numpy(values["images"]), torch.from_numpy(geometry)).numpy() * target_std + target_mean
    residual = values["targets"] - predicted
    summary = {
        "schema_version": "image-corner-repair-pilot-v1",
        "scope": scope,
        "model": {"family": model.family, "image_input": "raw-detector-quadrilateral-warped-rgb-128x64", "geometry_input": "raw-detector-only-15d", "truth_input": False, "motion_input": False, "identity_input": False, "range_input": False, "future_input": False},
        "provenance": provenance,
        "split": split_metadata,
        "training": {"seed": args.seed, "device": args.device, "epochs_completed": epoch, "best_validation_normalized_mse": best_validation, "batch_size": args.batch_size, "learning_rate": args.learning_rate},
        "metrics_px": {
            "raw_coordinate_rms_all": rms(torch.from_numpy(values["targets"])),
            "repaired_coordinate_rms_all": rms(torch.from_numpy(residual)),
            "raw_coordinate_rms_validation": rms(torch.from_numpy(values["targets"][~train])),
            "repaired_coordinate_rms_validation": rms(torch.from_numpy(residual[~train])),
        },
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_family": model.family, "state_dict": model.state_dict(), "geometry_mean": geo_mean, "geometry_std": geo_std, "target_mean_px": target_mean, "target_std_px": target_std, "patch_size": [PATCH_WIDTH, PATCH_HEIGHT], "provenance": summary["provenance"]}, checkpoint)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["metrics_px"], indent=2))


if __name__ == "__main__":
    main()
