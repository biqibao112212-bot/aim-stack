"""Fine-tune only the v13 motion router with factor-aware supervision."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .causal_physical_dataset import CausalPhysicalShardDataset
from .causal_physical_state_model import IndependentMotionExpertSystem
from .router_only_loss import router_only_loss
from .train_causal_physical_ab import (
    _audit_dataset_contract,
    _git_state,
    _load_geometry,
    _seed,
    _sha256,
    _state_dict_sha256,
    _to_device,
    _validate,
    _validate_history_contract,
    _write_json,
)
from .train_factorized_motion_experts import _rotation_augmented_batch
from .train_independent_motion_experts import _factor_audit


V13_COMMIT = "f7aa56c8c317f002dc3b698e87af76e57338d348"
V13_BEST_EPOCH = 297
V13_BEST_SHA256 = (
    "f1619d518a60754504f0422234fa83a5eb6cb764deee6ccf13f1fcddcf3759aa"
)
V13_BASELINE_Q3_P95_M = 0.1766372412443161
V13_BASELINE_FACTOR_Q3_P95_M = {
    "translation": 0.20613759011030197,
    "rotation": 0.16752047538757323,
    "combined": 0.19885674864053726,
}


def _router_selection(
    metrics: dict[str, object], baseline: dict[str, object], *,
    q3_tolerance_m: float,
) -> tuple[float, ...]:
    router = metrics["router_diagnostics"]  # type: ignore[index]
    baseline_factors = baseline["strata"]["motion_factor"]  # type: ignore[index]
    factors = metrics["strata"]["motion_factor"]  # type: ignore[index]
    q3 = float(
        metrics["queries"][3]["trajectory_eligible"]["motion_delta"]["p95_m"]  # type: ignore[index]
    )
    baseline_q3 = float(
        baseline["queries"][3]["trajectory_eligible"]["motion_delta"]["p95_m"]  # type: ignore[index]
    )
    factor_q3 = {
        name: float(factors[name]["q3_trajectory_eligible_motion"]["p95_m"])
        for name in ("translation", "rotation", "combined")
    }
    baseline_factor_q3 = {
        name: float(
            baseline_factors[name]["q3_trajectory_eligible_motion"]["p95_m"]
        )
        for name in ("translation", "rotation", "combined")
    }
    q3_regression = max(
        [0.0, q3 - baseline_q3 - q3_tolerance_m]
        + [
            factor_q3[name] - baseline_factor_q3[name] - q3_tolerance_m
            for name in factor_q3
        ]
    )
    per_class = router["per_class"]
    route_recalls = [
        float(per_class[name]["recall"] or 0.0)
        for name in ("translation", "combined")
    ]
    macro_recall = float(router["macro_recall"] or 0.0)
    rotate_fpr = float(
        router["rotate_from_hard_route"]["negative_false_positive_rate"] or 1.0
    )
    return (
        q3_regression,
        1.0 - min(route_recalls),
        max(0.0, rotate_fpr - 0.02),
        1.0 - macro_recall,
        max(factor_q3.values()),
        q3,
    )


def _checkpoint(
    path: Path, model: IndependentMotionExpertSystem, epoch: int,
    metrics: dict[str, object], selection: tuple[float, ...],
    provenance: dict[str, object], role: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    payload: dict[str, object] = {
        "model": model.state_dict(),
        "model_class": model.__class__.__name__,
        "model_config": model.config(),
        "label": "router_only_independent_motion_expert_system",
        "epoch": epoch,
        "checkpoint_role": role,
        "validation": metrics,
        "selection_tuple": selection,
        "provenance": provenance,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def _load_v13_source(
    source_run: Path, dataset_sha256: str,
) -> tuple[IndependentMotionExpertSystem, dict[str, object], Path]:
    manifest_path = source_run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "stage3-independent-motion-experts-run-v1",
        "status": "complete",
        "stop_reason": "epoch_limit",
        "epochs_completed": 300,
        "test_accessed": False,
        "worktree_dirty": False,
        "git_commit": V13_COMMIT,
        "dataset_manifest_sha256": dataset_sha256,
        "frozen_foundation_verified_unchanged": True,
    }
    mismatch = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items() if manifest.get(key) != value
    }
    best = manifest.get("best", {})
    if (
        best.get("epoch") != V13_BEST_EPOCH
        or best.get("sha256") != V13_BEST_SHA256
    ):
        mismatch["best"] = {
            "expected": {"epoch": V13_BEST_EPOCH, "sha256": V13_BEST_SHA256},
            "actual": best,
        }
    if mismatch:
        raise ValueError(f"registered v13 source contract mismatch: {mismatch}")
    checkpoint_path = source_run / str(best["path"])
    if _sha256(checkpoint_path) != V13_BEST_SHA256:
        raise ValueError("registered v13 best checkpoint hash mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_expected = {
        "model_class": "IndependentMotionExpertSystem",
        "label": "independent_motion_expert_system",
        "epoch": V13_BEST_EPOCH,
        "checkpoint_role": "best",
    }
    checkpoint_mismatch = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in checkpoint_expected.items()
        if payload.get(key) != value
    }
    checkpoint_provenance = payload.get("provenance", {})
    provenance_expected = {
        "git_commit": V13_COMMIT,
        "worktree_dirty": False,
        "test_accessed": False,
        "dataset_manifest_sha256": dataset_sha256,
    }
    checkpoint_mismatch.update({
        f"provenance.{key}": {"expected": value, "actual": checkpoint_provenance.get(key)}
        for key, value in provenance_expected.items()
        if checkpoint_provenance.get(key) != value
    })
    if checkpoint_mismatch:
        raise ValueError(f"v13 best checkpoint contract mismatch: {checkpoint_mismatch}")
    config = payload["model_config"]
    model = IndependentMotionExpertSystem(
        geometry=torch.tensor(config["geometry"], dtype=torch.float32),
        position_mean=torch.tensor(config["position_mean"], dtype=torch.float32),
        position_std=torch.tensor(config["position_std"], dtype=torch.float32),
        input_features=int(config["input_features"]),
        channels=int(config["channels"]),
        dropout=float(config["dropout"]),
        history_events=int(config["history_events"]),
        maximum_speed_mps=float(config["maximum_speed_mps"]),
        maximum_yaw_rate_rad_s=float(config["maximum_yaw_rate_rad_s"]),
    )
    model.load_state_dict(payload["model"], strict=True)
    model.freeze_foundations()
    return model, manifest, checkpoint_path


def _freeze_except_router(model: IndependentMotionExpertSystem) -> None:
    model.requires_grad_(False)
    model.router_encoder.requires_grad_(True)
    model.router_head.requires_grad_(True)


def _non_router_hash(model: IndependentMotionExpertSystem) -> str:
    state = {
        name: value for name, value in model.state_dict().items()
        if not name.startswith(("router_encoder.", "router_head."))
    }
    return _state_dict_sha256(state)


def _train_epoch(
    model: IndependentMotionExpertSystem, loader: DataLoader,
    optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
    device: torch.device, geometry: torch.Tensor,
    position_mean: torch.Tensor, position_std: torch.Tensor,
    epoch: int, args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    model.router_encoder.train()
    model.router_head.train()
    totals: dict[str, float] = {}
    count = 0
    parameters = [value for value in model.parameters() if value.requires_grad]
    for batch_index, raw in enumerate(loader):
        batch = _to_device(raw, device)
        generator = torch.Generator().manual_seed(
            args.seed + epoch * 1_000_003 + batch_index
        )
        batch_size = int(batch["obs"].shape[0])
        angle = (
            2.0 * torch.pi * torch.rand(batch_size, generator=generator)
            - torch.pi
        )
        translation_xy = (
            0.5 * torch.rand(batch_size, 2, generator=generator) - 0.25
        )
        keep_original = (
            torch.rand(batch_size, generator=generator)
            >= args.augmentation_probability
        )
        angle[keep_original] = 0.0
        translation_xy[keep_original] = 0.0
        batch = _rotation_augmented_batch(
            batch, geometry, position_mean, position_std,
            angle.to(device), translation_xy.to(device),
        )
        optimizer.zero_grad(set_to_none=True)
        amp_enabled = device.type == "cuda" and args.amp != "off"
        amp_dtype = torch.float16 if args.amp == "float16" else torch.bfloat16
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled,
        ):
            encoded = model.router_encoder(
                batch["obs"], batch["obs_mask"], batch["event_mask"],
                batch["event_time_s"],
            )
            logits = model.router_head(encoded)
            total, parts = router_only_loss(
                logits, batch["future_position"], batch["tau"],
                batch["rule_query"], geometry,
                four_class_weight=args.four_class_weight,
                move_factor_weight=args.move_factor_weight,
                rotate_factor_weight=args.rotate_factor_weight,
                label_smoothing=args.label_smoothing,
                move_negative_mps=args.move_negative_mps,
                move_positive_mps=args.move_positive_mps,
                rotate_negative_rad_s=args.rotate_negative_rad_s,
                rotate_positive_rad_s=args.rotate_positive_rad_s,
            )
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite router-only objective")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, args.grad_clip if args.grad_clip > 0 else float("inf")
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite router-only gradient")
        scaler.step(optimizer)
        scaler.update()
        values = {
            "objective": float(total.detach().cpu()),
            **{name: float(value.detach().cpu()) for name, value in parts.items()},
            "gradient_norm": float(gradient_norm.detach().cpu()),
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise FloatingPointError("non-finite router-only metric")
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value * batch_size
        count += batch_size
    return {name: value / max(count, 1) for name, value in totals.items()}


def _lr_multiplier(epoch: int, *, epochs: int, warmup_epochs: int) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return 0.2 + 0.8 * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    progress = min(max(progress, 0.0), 1.0)
    return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(args: argparse.Namespace) -> Path:
    _seed(args.seed)
    git_state = _git_state()
    if bool(git_state["worktree_dirty"]) and not args.allow_dirty_worktree:
        raise ValueError("official router-only training requires a clean worktree")
    dataset = Path(args.dataset).resolve()
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "stage3-causal-physical-v1":
        raise ValueError("router-only training requires causal physical v1")
    if not bool(manifest.get("qualification_passed", False)):
        raise ValueError("router-only training requires a qualified dataset")
    if bool(manifest.get("test_accessed", True)):
        raise ValueError("router-only training refuses a test-accessed dataset")
    _validate_history_contract(manifest, args.history_events)
    dataset_sha256 = _sha256(manifest_path)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite router output: {output}")

    source_run = Path(args.source_run).resolve()
    model, source_manifest, source_checkpoint = _load_v13_source(
        source_run, dataset_sha256,
    )
    geometry, geometry_payload, geometry_sha256 = _load_geometry(dataset)
    if not torch.allclose(model.decoder.geometry.cpu(), geometry, atol=0, rtol=0):
        raise ValueError("v13 source geometry differs from the registered dataset")
    if model.history_events != args.history_events:
        raise ValueError("router history length must match the v13 source")

    train_ds = CausalPhysicalShardDataset(
        dataset, "train", seed=args.seed, shuffle=True,
        sample_limit=args.train_sample_limit,
    )
    validation_ds = CausalPhysicalShardDataset(
        dataset, "validation", seed=args.seed, shuffle=False,
        sample_limit=args.validation_sample_limit,
    )
    require_classes = (
        {0, 1, 2, 3}
        if args.train_sample_limit == 0 and args.validation_sample_limit == 0
        else None
    )
    qualification: dict[str, object] = {
        "train": _audit_dataset_contract(
            train_ds, geometry, args.history_events,
            args.minimum_supervision_coverage, require_classes,
        ),
        "validation": _audit_dataset_contract(
            validation_ds, geometry, args.history_events,
            args.minimum_supervision_coverage, require_classes,
        ),
        "minimum_required_coverage": args.minimum_supervision_coverage,
    }
    if require_classes is not None:
        qualification["motion_factor"] = {
            "train": _factor_audit(train_ds, geometry, args),
            "validation": _factor_audit(validation_ds, geometry, args),
            "minimum_router_coverage": args.minimum_router_coverage,
        }

    _freeze_except_router(model)
    frozen_hash = _non_router_hash(model)
    initial_router_hash = _state_dict_sha256({
        name: value for name, value in model.state_dict().items()
        if name.startswith(("router_encoder.", "router_head."))
    })
    total_parameters = sum(value.numel() for value in model.parameters())
    trainable_parameters = sum(
        value.numel() for value in model.parameters() if value.requires_grad
    )
    frozen_parameters = total_parameters - trainable_parameters

    output.mkdir(parents=True)
    loader_options = {
        "batch_size": args.batch_size, "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, **loader_options)
    validation_loader = DataLoader(validation_ds, **loader_options)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    geometry_device = geometry.to(device)
    position_mean = torch.from_numpy(train_ds.mean).to(device)
    position_std = torch.from_numpy(train_ds.std).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.router_encoder.parameters(), "lr": args.encoder_lr},
        {"params": model.router_head.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: _lr_multiplier(
            epoch, epochs=args.epochs, warmup_epochs=args.warmup_epochs,
        ),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.amp == "float16",
    )
    source_root = Path(__file__).resolve().parent
    source_names = (
        "train_router_only.py", "router_only_loss.py",
        "causal_physical_state_model.py", "train_causal_physical_ab.py",
        "train_factorized_motion_experts.py",
        "train_independent_motion_experts.py", "causal_physical_dataset.py",
        "physical_model.py", "physical_metrics.py", "pnp_state_targets.py",
    )
    provenance: dict[str, object] = {
        "schema_version": "stage3-router-only-run-v1",
        "status": "running",
        "dataset": str(dataset),
        "dataset_manifest_sha256": dataset_sha256,
        "geometry_template_sha256": geometry_sha256,
        "geometry_hash": geometry_payload.get("geometry_hash"),
        "dataset_qualification": qualification,
        "test_accessed": False,
        "source_v13": {
            "path": str(source_run),
            "manifest_sha256": _sha256(source_run / "run_manifest.json"),
            "verified_status": source_manifest.get("status"),
            "best_epoch": V13_BEST_EPOCH,
            "best_path": source_checkpoint.name,
            "best_sha256": V13_BEST_SHA256,
        },
        "architecture_contract": {
            "source": "complete integrated v13 best checkpoint",
            "trainable": ["router_encoder", "router_head"],
            "frozen": [
                "pose_q0", "translation_expert", "rotation_expert",
                "combined_expert", "rigid_decoder",
            ],
            "route_order": list(model.route_names),
            "hard_route_unchanged": True,
        },
        "objective_contract": {
            "formula": (
                f"{args.four_class_weight}*balanced_four_class_ce + "
                f"{args.move_factor_weight}*balanced_move_factor_bce + "
                f"{args.rotate_factor_weight}*balanced_rotate_factor_bce"
            ),
            "label_smoothing": args.label_smoothing,
            "move_dead_band_mps": [args.move_negative_mps, args.move_positive_mps],
            "rotate_dead_band_rad_s": [
                args.rotate_negative_rad_s, args.rotate_positive_rad_s,
            ],
            "future_truth_role": "detached loss/evaluation labels only",
            "augmentation": {
                "probability": args.augmentation_probability,
                "yaw_rad": [-math.pi, math.pi],
                "translation_xy_m": [-0.25, 0.25],
                "history_and_future_transformed_together": True,
                "pnp_noise_added": False,
            },
        },
        "selection_contract": (
            "no q3 regression beyond tolerance; maximize minimum translation/"
            "combined recall; meet rotate FPR; maximize macro recall; then q3"
        ),
        "acceptance_goals_not_stop_conditions": {
            "translation_recall_min": 0.97,
            "combined_recall_min": 0.97,
            "router_macro_recall_min": 0.98,
            "rotate_false_positive_rate_max": 0.02,
            "q3_overall_p95_m_max": V13_BASELINE_Q3_P95_M,
            "factor_q3_p95_m_max": V13_BASELINE_FACTOR_Q3_P95_M,
        },
        "parameter_contract": {
            "total": total_parameters,
            "trainable": trainable_parameters,
            "frozen": frozen_parameters,
            "frozen_non_router_state_sha256": frozen_hash,
            "initial_router_state_sha256": initial_router_hash,
            "optimizer_contains_router_only": True,
        },
        "config": vars(args),
        "source_sha256": {
            name: _sha256(source_root / name) for name in source_names
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "amp": args.amp,
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
        },
        **git_state,
    }
    _write_json(output / "run_manifest.json", provenance)

    baseline = _validate(model, validation_loader, device, args)
    baseline_q3 = float(
        baseline["queries"][3]["trajectory_eligible"]["motion_delta"]["p95_m"]  # type: ignore[index]
    )
    if abs(baseline_q3 - V13_BASELINE_Q3_P95_M) > 1e-8:
        raise RuntimeError(
            f"loaded v13 baseline q3 changed: {baseline_q3}"
        )
    selection = _router_selection(
        baseline, baseline, q3_tolerance_m=args.q3_tolerance_m,
    )
    history: list[dict[str, object]] = [{
        "epoch": 0, "validation": baseline, "selection_tuple": selection,
    }]
    history_path = output / "stage3-router-only-history.json"
    _write_json(history_path, history)
    initial_path = output / f"stage3-router-only-seed{args.seed}-initial.pt"
    best_path = output / f"stage3-router-only-seed{args.seed}-best.pt"
    _checkpoint(initial_path, model, 0, baseline, selection, provenance, "initial")
    _checkpoint(best_path, model, 0, baseline, selection, provenance, "best")
    best_selection = selection
    best_epoch = 0
    stale_epochs = 0
    milestone_epochs = {20, 50, 80, 100, args.epochs}
    epochs_completed = 0
    stop_reason = "epoch_limit"
    started = time.monotonic()
    validation = baseline
    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        train_metrics = _train_epoch(
            model, train_loader, optimizer, scaler, device, geometry_device,
            position_mean, position_std, epoch, args,
        )
        scheduler.step()
        validation = _validate(model, validation_loader, device, args)
        if _non_router_hash(model) != frozen_hash:
            raise RuntimeError("frozen non-router state changed during training")
        epochs_completed = epoch
        selection = _router_selection(
            validation, baseline, q3_tolerance_m=args.q3_tolerance_m,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation,
            "selection_tuple": selection,
            "lr": [group["lr"] for group in optimizer.param_groups],
        }
        history.append(record)
        _write_json(history_path, history)
        print(json.dumps(record, sort_keys=True), flush=True)
        if selection < best_selection:
            best_selection = selection
            best_epoch = epoch
            stale_epochs = 0
            _checkpoint(
                best_path, model, epoch, validation, selection, provenance,
                "best", optimizer, scheduler, scaler,
            )
        else:
            stale_epochs += 1
        if epoch in milestone_epochs:
            _checkpoint(
                output / f"stage3-router-only-seed{args.seed}-epoch{epoch:03d}.pt",
                model, epoch, validation, selection, provenance, "milestone",
                optimizer, scheduler, scaler,
            )
        if (
            args.patience > 0 and epoch >= args.early_stopping_warmup
            and stale_epochs >= args.patience
        ):
            stop_reason = "early_stopping"
            break
        if args.max_wall_minutes > 0 and (
            time.monotonic() - started >= args.max_wall_minutes * 60
        ):
            stop_reason = "wall_time_limit"
            break

    last_path = output / f"stage3-router-only-seed{args.seed}-last.pt"
    last_selection = _router_selection(
        validation, baseline, q3_tolerance_m=args.q3_tolerance_m,
    )
    _checkpoint(
        last_path, model, epochs_completed, validation, last_selection,
        provenance, "last", optimizer, scheduler, scaler,
    )
    if _non_router_hash(model) != frozen_hash:
        raise RuntimeError("frozen non-router state changed at completion")
    final = {
        **provenance,
        "status": "complete",
        "stop_reason": stop_reason,
        "epochs_completed": epochs_completed,
        "frozen_non_router_verified_unchanged": True,
        "best": {
            "path": best_path.name,
            "sha256": _sha256(best_path),
            "selection_tuple": best_selection,
            "epoch": best_epoch,
            "trained_checkpoint": best_epoch > 0,
        },
        "last": {"path": last_path.name, "sha256": _sha256(last_path)},
    }
    _write_json(output / "run_manifest.json", final)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--encoder-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--early-stopping-warmup", type=int, default=30)
    parser.add_argument("--history-events", type=int, default=32)
    parser.add_argument("--four-class-weight", type=float, default=1.0)
    parser.add_argument("--move-factor-weight", type=float, default=1.0)
    parser.add_argument("--rotate-factor-weight", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--augmentation-probability", type=float, default=0.75)
    parser.add_argument("--q3-tolerance-m", type=float, default=0.002)
    parser.add_argument("--move-negative-mps", type=float, default=0.01)
    parser.add_argument("--move-positive-mps", type=float, default=0.10)
    parser.add_argument("--rotate-negative-rad-s", type=float, default=0.05)
    parser.add_argument("--rotate-positive-rad-s", type=float, default=0.20)
    parser.add_argument("--minimum-supervision-coverage", type=float, default=0.85)
    parser.add_argument("--minimum-router-coverage", type=float, default=0.85)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--amp", choices=("off", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device", default="")
    parser.add_argument("--max-wall-minutes", type=float, default=0.0)
    parser.add_argument("--train-sample-limit", type=int, default=0)
    parser.add_argument("--validation-sample-limit", type=int, default=0)
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    args = parser.parse_args()
    positive = (
        args.epochs, args.batch_size, args.encoder_lr, args.head_lr,
        args.four_class_weight + args.move_factor_weight + args.rotate_factor_weight,
    )
    if min(positive) <= 0:
        parser.error("epochs, batch size, learning rates and total loss weight must be positive")
    if args.weight_decay < 0 or args.grad_clip < 0 or args.q3_tolerance_m < 0:
        parser.error("nonnegative optimizer/selection arguments cannot be negative")
    if args.warmup_epochs < 0 or args.patience < 0 or args.early_stopping_warmup < 0:
        parser.error("warmup and early-stopping arguments cannot be negative")
    if not 0 <= args.label_smoothing < 1:
        parser.error("label smoothing must be within [0,1)")
    if not 0 <= args.augmentation_probability <= 1:
        parser.error("augmentation probability must be within [0,1]")
    if args.train_sample_limit < 0 or args.validation_sample_limit < 0:
        parser.error("sample limits cannot be negative")
    if not 8 <= args.history_events <= 200:
        parser.error("history events must be within [8,200]")
    if not 0 < args.minimum_supervision_coverage <= 1:
        parser.error("minimum supervision coverage must be within (0,1]")
    if not 0 < args.minimum_router_coverage <= 1:
        parser.error("minimum router coverage must be within (0,1]")
    if not 0 <= args.move_negative_mps < args.move_positive_mps:
        parser.error("move router thresholds are invalid")
    if not 0 <= args.rotate_negative_rad_s < args.rotate_positive_rad_s:
        parser.error("rotation router thresholds are invalid")
    train(args)


if __name__ == "__main__":
    main()
