#!/usr/bin/env python3
"""Train cross-fitted joint four-corner residual networks on simulation data.

The outer held groups are never used for early stopping or normalization.  The
script emits a complete out-of-fold table for the frozen coordinate-only pilot;
PnP propagation is performed by a separate evaluator so training cannot select
itself on pose metrics.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.stage3.corner_residual_network import (
    CORNER_ORDER,
    FEATURE_DIM,
    JointCornerResidualMLP,
    Standardization,
    deterministic_seed,
    observable_features,
)


SCHEMA_VERSION = "stage3-sim-corner-residual-network-oof-v1"
SEEDS = (17, 29, 43)
CORRECTION_ARMS = ("raw", "mean", "ridge", "current_refined", "network", "exact")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--atlas", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--model-output", required=True, type=Path)
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--epochs", type=int, default=240)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--patience", type=int, default=30)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--learning-rate", type=float, default=1.0e-3)
    result.add_argument("--weight-decay", type=float, default=1.0e-4)
    result.add_argument("--gradient-clip", type=float, default=1.0)
    result.add_argument("--correction-cap-px", type=float, default=6.0)
    result.add_argument("--ridge-lambda", type=float, default=1.0e-3)
    result.add_argument(
        "--schemes", default="leave_segment_out,leave_session_out",
        help="comma-separated outer split schemes",
    )
    result.add_argument(
        "--seeds", default=",".join(str(value) for value in SEEDS),
        help="comma-separated ensemble seeds",
    )
    result.add_argument("--fold-limit", type=int, default=0)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(repo: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=repo, text=True
    )
    return {"commit": commit, "dirty": bool(status.strip())}


def read_csv(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return pd.read_csv(handle)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def corner_array(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    fields = []
    for corner in CORNER_ORDER:
        fields.extend((f"{prefix}_{corner}_x_px", f"{prefix}_{corner}_y_px"))
    values = np.asarray(frame[fields].values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{prefix} corner table contains non-finite values")
    return values.reshape(-1, 4, 2)


def split_groups(frame: pd.DataFrame, scheme: str) -> np.ndarray:
    if scheme == "leave_segment_out":
        return np.asarray(
            [f"{session}|segment={segment}" for session, segment in zip(
                frame.session_id.astype(str), frame.segment_index.astype(int)
            )], dtype=object,
        )
    if scheme == "leave_session_out":
        return frame.session_id.astype(str).values.astype(object)
    raise ValueError(f"unknown split scheme: {scheme}")


def validation_groups(frame: pd.DataFrame, scheme: str) -> np.ndarray:
    """Return complete groups available for inner early-stopping validation.

    The two-session stress split leaves only one session for development, so
    its inner validation unit must be a complete segment inside that remaining
    session.  This preserves temporal grouping without emptying the train set.
    """
    if scheme in ("leave_segment_out", "leave_session_out"):
        return np.asarray(
            [f"{session}|segment={segment}" for session, segment in zip(
                frame.session_id.astype(str), frame.segment_index.astype(int)
            )], dtype=object,
        )
    raise ValueError(f"unknown split scheme: {scheme}")


def inner_group(scheme: str, outer_group: str, available: list[str]) -> str:
    if not available:
        raise ValueError("outer fold has no group available for validation")
    token = hashlib.sha256(f"{scheme}|{outer_group}".encode("utf-8")).digest()
    return sorted(available)[int.from_bytes(token[:8], "big") % len(available)]


def ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    ridge_lambda: float,
) -> np.ndarray:
    design = np.column_stack((np.ones(len(train_x)), train_x)).astype(np.float64)
    test_design = np.column_stack((np.ones(len(test_x)), test_x)).astype(np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_lambda
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty, design.T @ train_y.astype(np.float64)
    )
    return (test_design @ coefficients).astype(np.float32)


class CornerDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        *,
        batch_size: int,
        workers: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("train_x", "train_y", "val_x", "val_y"))
        self._train = (
            torch.from_numpy(np.ascontiguousarray(train_x)),
            torch.from_numpy(np.ascontiguousarray(train_y)),
        )
        self._val = (
            torch.from_numpy(np.ascontiguousarray(val_x)),
            torch.from_numpy(np.ascontiguousarray(val_y)),
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = TensorDataset(*self._train)
            self.val_dataset = TensorDataset(*self._val)

    def train_dataloader(self) -> DataLoader:
        generator = torch.Generator().manual_seed(int(self.hparams.seed))
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.hparams.batch_size),
            shuffle=True,
            num_workers=int(self.hparams.workers),
            persistent_workers=int(self.hparams.workers) > 0,
            generator=generator,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.hparams.batch_size),
            shuffle=False,
            num_workers=int(self.hparams.workers),
            persistent_workers=int(self.hparams.workers) > 0,
        )


class CornerLightningModule(L.LightningModule):
    def __init__(
        self,
        *,
        feature_mean: list[float],
        feature_std: list[float],
        target_mean: list[float],
        target_std: list[float],
        learning_rate: float,
        weight_decay: float,
        epochs: int,
        hidden: int = 64,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = JointCornerResidualMLP(hidden=hidden, dropout=dropout)
        self.register_buffer(
            "target_std_px", torch.tensor(target_std, dtype=torch.float32)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], prefix: str) -> torch.Tensor:
        features, target = batch
        prediction = self(features)
        loss = F.smooth_l1_loss(prediction, target, beta=1.0)
        pixel_error = (prediction - target) * self.target_std_px
        coordinate_rmse = torch.sqrt(torch.mean(torch.square(pixel_error)))
        self.log(
            f"{prefix}/loss", loss, on_step=False, on_epoch=True,
            prog_bar=prefix == "val", batch_size=len(features),
        )
        self.log(
            f"{prefix}/coordinate_rmse_px", coordinate_rmse,
            on_step=False, on_epoch=True, prog_bar=prefix == "val",
            batch_size=len(features),
        )
        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        return self._step(batch, "train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        del batch_idx
        self._step(batch, "val")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(self.hparams.epochs), eta_min=1.0e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def train_seed(
    *,
    standardization: Standardization,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    test_features: np.ndarray,
    output: Path,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    deterministic_seed(seed)
    L.seed_everything(seed, workers=True, verbose=False)
    output.mkdir(parents=True, exist_ok=False)
    data = CornerDataModule(
        standardization.normalize_features(train_features),
        standardization.normalize_targets(train_targets),
        standardization.normalize_features(val_features),
        standardization.normalize_targets(val_targets),
        batch_size=args.batch_size,
        workers=args.workers,
        seed=seed,
    )
    model = CornerLightningModule(
        feature_mean=standardization.feature_mean.tolist(),
        feature_std=standardization.feature_std.tolist(),
        target_mean=standardization.target_mean.tolist(),
        target_std=standardization.target_std.tolist(),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
    )
    checkpoint = ModelCheckpoint(
        dirpath=output,
        filename="best",
        monitor="val/coordinate_rmse_px",
        mode="min",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )
    early_stop = EarlyStopping(
        monitor="val/coordinate_rmse_px",
        mode="min",
        patience=args.patience,
        min_delta=1.0e-4,
    )
    logger = CSVLogger(save_dir=output, name="logs", version="v1")
    trainer = L.Trainer(
        accelerator="gpu" if args.device == "cuda" else "cpu",
        devices=1,
        max_epochs=args.epochs,
        deterministic=True,
        benchmark=False,
        gradient_clip_val=args.gradient_clip,
        gradient_clip_algorithm="norm",
        callbacks=[checkpoint, early_stop],
        logger=logger,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=data)
    if not checkpoint.best_model_path:
        raise RuntimeError("corner training produced no checkpoint")
    best = CornerLightningModule.load_from_checkpoint(
        checkpoint.best_model_path, map_location="cpu"
    )
    best.eval()
    with torch.inference_mode():
        normalized = best(
            torch.from_numpy(
                np.ascontiguousarray(
                    standardization.normalize_features(test_features)
                )
            )
        ).numpy()
    prediction = standardization.denormalize_targets(normalized)
    metadata = {
        "seed": seed,
        "checkpoint": str(Path(checkpoint.best_model_path).resolve()),
        "checkpoint_sha256": sha256(Path(checkpoint.best_model_path)),
        "best_score": float(checkpoint.best_model_score),
        "stopped_epoch": int(trainer.current_epoch),
        "global_step": int(trainer.global_step),
        "model_config": best.model.config,
        "standardization": {
            "feature_mean": standardization.feature_mean.tolist(),
            "feature_std": standardization.feature_std.tolist(),
            "target_mean": standardization.target_mean.tolist(),
            "target_std": standardization.target_std.tolist(),
        },
        "lightning_version": L.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    atomic_json(output / "fold_seed_manifest.json", metadata)
    return prediction, metadata


def main() -> None:
    args = parser().parse_args()
    if args.output.exists() or args.model_output.exists():
        raise FileExistsError("refusing to overwrite evidence or protected model output")
    if min(
        args.epochs, args.batch_size, args.patience, args.learning_rate,
        args.weight_decay, args.gradient_clip, args.correction_cap_px,
        args.ridge_lambda,
    ) <= 0:
        raise ValueError("training arguments must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    schemes = tuple(value.strip() for value in args.schemes.split(",") if value.strip())
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not schemes or not seeds:
        raise ValueError("at least one split scheme and seed are required")

    repo = Path(__file__).resolve().parents[2]
    atlas_path = args.atlas.resolve()
    frame = read_csv(atlas_path)
    if len(frame) != 4280:
        raise ValueError(f"frozen atlas row count changed: {len(frame)}")
    raw = corner_array(frame, "raw")
    refined = corner_array(frame, "refined")
    truth = corner_array(frame, "truth")
    features = observable_features(raw)
    targets = (truth - raw).reshape(len(frame), 8).astype(np.float32)
    refinement = (refined - raw).reshape(len(frame), 8).astype(np.float32)

    args.output.mkdir(parents=True)
    args.model_output.mkdir(parents=True)
    progress_path = args.output / "run_progress.json"
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    completed: list[str] = []

    key_fields = (
        "session", "session_id", "producer_epoch", "frame_seq", "timestamp_ns",
        "armor_index", "truth_slot", "segment_index", "motion_mode",
        "range_m", "view_incidence_cos", "projected_sqrt_area_px",
    )
    for scheme in schemes:
        groups = split_groups(frame, scheme)
        inner_groups = validation_groups(frame, scheme)
        outer_groups = sorted(set(str(value) for value in groups))
        if args.fold_limit > 0:
            outer_groups = outer_groups[: args.fold_limit]
        for outer in outer_groups:
            test_mask = groups == outer
            available = sorted(set(str(value) for value in inner_groups[~test_mask]))
            inner = inner_group(scheme, outer, available)
            val_mask = (inner_groups == inner) & ~test_mask
            train_mask = ~(test_mask | val_mask)
            if np.any(train_mask & val_mask) or np.any(train_mask & test_mask) or np.any(val_mask & test_mask):
                raise AssertionError("corner split leakage")
            if min(np.count_nonzero(train_mask), np.count_nonzero(val_mask), np.count_nonzero(test_mask)) <= 0:
                raise ValueError("empty corner split")
            standardization = Standardization.fit(features[train_mask], targets[train_mask])
            mean_prediction = np.repeat(
                standardization.target_mean[None, :], np.count_nonzero(test_mask), axis=0
            )
            ridge_prediction = ridge_fit_predict(
                standardization.normalize_features(features[train_mask]),
                targets[train_mask],
                standardization.normalize_features(features[test_mask]),
                args.ridge_lambda,
            )
            seed_predictions = []
            fold_slug = hashlib.sha256(outer.encode("utf-8")).hexdigest()[:12]
            for seed in seeds:
                seed_output = (
                    args.model_output / scheme / f"fold-{fold_slug}" / f"seed-{seed}"
                )
                prediction, metadata = train_seed(
                    standardization=standardization,
                    train_features=features[train_mask],
                    train_targets=targets[train_mask],
                    val_features=features[val_mask],
                    val_targets=targets[val_mask],
                    test_features=features[test_mask],
                    output=seed_output,
                    seed=seed,
                    args=args,
                )
                seed_predictions.append(
                    np.clip(prediction, -args.correction_cap_px, args.correction_cap_px)
                )
                checkpoint_records.append(
                    {"scheme": scheme, "outer_group": outer, "inner_group": inner, **metadata}
                )
            ensemble = np.stack(seed_predictions, axis=0)
            network_prediction = ensemble.mean(axis=0)
            network_uncertainty = ensemble.std(axis=0)
            predictions = {
                "raw": np.zeros_like(network_prediction),
                "mean": np.clip(
                    mean_prediction, -args.correction_cap_px, args.correction_cap_px
                ),
                "ridge": np.clip(
                    ridge_prediction, -args.correction_cap_px, args.correction_cap_px
                ),
                "current_refined": refinement[test_mask],
                "network": network_prediction,
                "exact": targets[test_mask],
            }
            test_indices = np.flatnonzero(test_mask)
            for local_index, row_index in enumerate(test_indices):
                source = frame.iloc[int(row_index)]
                common = {
                    "scheme": scheme,
                    "outer_group": outer,
                    "inner_validation_group": inner,
                    "row_index": int(row_index),
                    **{name: source[name] for name in key_fields},
                }
                for arm in CORRECTION_ARMS:
                    correction = predictions[arm][local_index].reshape(4, 2)
                    corrected = raw[row_index] + correction
                    error = corrected - truth[row_index]
                    item = {**common, "arm": arm}
                    for corner_index, corner in enumerate(CORNER_ORDER):
                        item[f"raw_{corner}_x_px"] = float(raw[row_index, corner_index, 0])
                        item[f"raw_{corner}_y_px"] = float(raw[row_index, corner_index, 1])
                        item[f"truth_{corner}_x_px"] = float(truth[row_index, corner_index, 0])
                        item[f"truth_{corner}_y_px"] = float(truth[row_index, corner_index, 1])
                        item[f"correction_{corner}_dx_px"] = float(correction[corner_index, 0])
                        item[f"correction_{corner}_dy_px"] = float(correction[corner_index, 1])
                        item[f"corrected_{corner}_x_px"] = float(corrected[corner_index, 0])
                        item[f"corrected_{corner}_y_px"] = float(corrected[corner_index, 1])
                        item[f"error_{corner}_dx_px"] = float(error[corner_index, 0])
                        item[f"error_{corner}_dy_px"] = float(error[corner_index, 1])
                        item[f"error_{corner}_norm_px"] = float(np.linalg.norm(error[corner_index]))
                        item[f"network_uncertainty_{corner}_px"] = (
                            float(np.linalg.norm(network_uncertainty[local_index, 2 * corner_index:2 * corner_index + 2]))
                            if arm == "network" else float("nan")
                        )
                    item["coordinate_rms_px"] = float(np.sqrt(np.mean(np.square(error))))
                    item["corner_norm_mean_px"] = float(np.linalg.norm(error, axis=1).mean())
                    item["corner_norm_max_px"] = float(np.linalg.norm(error, axis=1).max())
                    prediction_rows.append(item)
            for arm in CORRECTION_ARMS:
                selected = [
                    row for row in prediction_rows
                    if row["scheme"] == scheme and row["outer_group"] == outer and row["arm"] == arm
                ]
                values = np.asarray([row["coordinate_rms_px"] for row in selected])
                fold_rows.append(
                    {
                        "scheme": scheme,
                        "outer_group": outer,
                        "inner_validation_group": inner,
                        "arm": arm,
                        "train_samples": int(np.count_nonzero(train_mask)),
                        "validation_samples": int(np.count_nonzero(val_mask)),
                        "test_samples": int(np.count_nonzero(test_mask)),
                        "mean_px": float(values.mean()),
                        "p50_px": float(np.quantile(values, 0.50)),
                        "p90_px": float(np.quantile(values, 0.90)),
                        "p95_px": float(np.quantile(values, 0.95)),
                        "p99_px": float(np.quantile(values, 0.99)),
                        "maximum_px": float(values.max()),
                    }
                )
            completed.append(f"{scheme}:{outer}")
            atomic_json(
                progress_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "running",
                    "completed_outer_folds": completed,
                    "checkpoint_count": len(checkpoint_records),
                },
            )

    if not prediction_rows:
        raise RuntimeError("corner training produced no OOF predictions")
    prediction_fields = list(prediction_rows[0])
    write_csv_gz(
        args.output / "oof_corner_predictions.csv.gz",
        prediction_rows,
        prediction_fields,
    )
    write_csv(args.output / "fold_metrics.csv", fold_rows, list(fold_rows[0]))
    checkpoint_fields = list(checkpoint_records[0])
    serialized_checkpoint_rows = []
    for row in checkpoint_records:
        serialized_checkpoint_rows.append(
            {
                name: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list)) else value
                for name, value in row.items()
            }
        )
    write_csv(
        args.output / "checkpoint_registry.csv",
        serialized_checkpoint_rows,
        checkpoint_fields,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "outer_test_accessed_during_training": False,
        "outer_test_accessed_for_final_prediction": True,
        "deployable": False,
        "atlas": {"path": str(atlas_path), "sha256": sha256(atlas_path), "rows": len(frame)},
        "source": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "model_module": str(Path(__file__).with_name("corner_residual_network.py").resolve()),
            "model_module_sha256": sha256(Path(__file__).with_name("corner_residual_network.py")),
            "git": git_state(repo),
        },
        "environment": {
            "python": os.sys.version,
            "torch": torch.__version__,
            "lightning": L.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        },
        "arguments": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "schemes": schemes,
        "seeds": seeds,
        "prediction_rows": len(prediction_rows),
        "unique_source_rows": int(len(set((row["scheme"], row["row_index"]) for row in prediction_rows))),
        "checkpoint_count": len(checkpoint_records),
        "artifacts": [
            "oof_corner_predictions.csv.gz",
            "fold_metrics.csv",
            "checkpoint_registry.csv",
        ],
    }
    atomic_json(args.output / "manifest.json", manifest)
    atomic_json(progress_path, {**manifest, "status": "complete"})
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        try:
            failed_args = parser().parse_args()
            failed_progress = failed_args.output / "run_progress.json"
            payload: dict[str, Any] = {}
            if failed_progress.exists():
                payload = json.loads(failed_progress.read_text(encoding="utf-8"))
            atomic_json(
                failed_progress,
                {
                    **payload,
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        except Exception:
            pass
        raise
