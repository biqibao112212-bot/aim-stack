"""Train paired factorized motion experts with and without rotation augmentation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .causal_physical_dataset import CausalPhysicalShardDataset
from .causal_physical_state_model import (
    FactorizedExpertPhysicalPredictor,
    trainable_parameter_count,
)
from .factorized_expert_loss import factorized_expert_loss
from .pnp_state_targets import _query_pose_from_fixed_truth
from .train_causal_physical_ab import (
    _audit_dataset_contract,
    _capture_rng,
    _git_state,
    _load_geometry,
    _restore_rng,
    _seed,
    _sha256,
    _state_dict_sha256,
    _to_device,
    _validate,
    _validate_history_contract,
    _write_json,
)


ExpertModel = FactorizedExpertPhysicalPredictor


def _expert_selection_tuple(metrics: dict[str, object]) -> tuple[float, ...]:
    strata = metrics["strata"]["motion_class"]  # type: ignore[index]
    dynamic = [strata[name]["q3_trajectory_eligible_motion"] for name in (
        "linear", "spin", "linear_and_spin",
    )]
    q3 = metrics["queries"][3]["trajectory_eligible"]  # type: ignore[index]
    present = [value for value in dynamic if "p95_m" in value]
    if not present:
        present = [q3["motion_delta"]]
    return (
        max(float(value["p95_m"]) for value in present),
        float(q3["motion_delta"]["p95_m"]),
        float(metrics["trajectory_eligible_state_q0"]["p95_m"]),  # type: ignore[index]
        max(float(value["median_m"]) for value in present),
    )


def _checkpoint(
    path: Path, model: ExpertModel, label: str, epoch: int,
    metrics: dict[str, object], provenance: dict[str, object], role: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    payload: dict[str, object] = {
        "model": model.state_dict(), "model_class": model.__class__.__name__,
        "model_config": model.config(), "label": label, "epoch": epoch,
        "checkpoint_role": role, "validation": metrics,
        "selection_tuple": _expert_selection_tuple(metrics),
        "provenance": provenance,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def _rotate_xy_about_q0(
    position: torch.Tensor, center0: torch.Tensor, angle: torch.Tensor,
) -> torch.Tensor:
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    relative_x = position[..., 0] - center0[:, None, None, 0]
    relative_y = position[..., 1] - center0[:, None, None, 1]
    rotated = position.clone()
    rotated[..., 0] = (
        center0[:, None, None, 0]
        + cosine[:, None, None] * relative_x
        - sine[:, None, None] * relative_y
    )
    rotated[..., 1] = (
        center0[:, None, None, 1]
        + sine[:, None, None] * relative_x
        + cosine[:, None, None] * relative_y
    )
    return rotated


def _rotation_augmented_batch(
    batch: dict[str, torch.Tensor], geometry: torch.Tensor,
    position_mean: torch.Tensor, position_std: torch.Tensor,
    angle: torch.Tensor, translation_xy: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Apply one rigid xy transform using a history-only anchor."""
    event_mask = batch["event_mask"].to(torch.bool)
    event_index = torch.arange(
        event_mask.shape[1], device=event_mask.device,
    ).expand_as(event_mask)
    last_index = torch.where(event_mask, event_index, -1).amax(dim=1)
    if bool((last_index < 0).any()):
        raise ValueError("augmentation requires at least one history event")
    batch_index = torch.arange(event_mask.shape[0], device=event_mask.device)
    history_anchor = batch["history_position_m"][batch_index, last_index]
    center0, _ = _query_pose_from_fixed_truth(
        history_anchor[:, None], geometry,
    )
    center0 = center0[:, 0]
    history = _rotate_xy_about_q0(
        batch["history_position_m"], center0, angle,
    )
    future = _rotate_xy_about_q0(batch["future_position"], center0, angle)
    if translation_xy is not None:
        history[..., :2] += translation_xy[:, None, None]
        future[..., :2] += translation_xy[:, None, None]
    normalized = (history - position_mean) / position_std
    obs = torch.cat((normalized, batch["obs"][..., 3:]), dim=-1)
    obs = torch.where(batch["obs_mask"].unsqueeze(-1), obs, torch.zeros_like(obs))
    augmented = dict(batch)
    augmented["obs"] = obs
    augmented["history_position_m"] = torch.where(
        batch["obs_mask"].unsqueeze(-1), history, torch.zeros_like(history)
    )
    augmented["future_position"] = future
    return augmented


def _train_one(
    label: str, model: ExpertModel, batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
    device: torch.device, args: argparse.Namespace,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = device.type == "cuda" and args.amp != "off"
    amp_dtype = torch.float16 if args.amp == "float16" else torch.bfloat16
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        prediction = model(
            batch["obs"], batch["obs_mask"], batch["event_mask"],
            batch["event_time_s"], batch["tau"],
        )
        total, parts = factorized_expert_loss(
            prediction, batch["future_position"], batch["tau"],
            batch["rule_query"], model.decoder.geometry,
            huber_beta_m=args.huber_beta_m,
            reference_horizon_s=args.reference_horizon_s,
            expert_weight=args.expert_weight, gate_weight=args.gate_weight,
            move_negative_mps=args.move_negative_mps,
            move_positive_mps=args.move_positive_mps,
            rotate_negative_rad_s=args.rotate_negative_rad_s,
            rotate_positive_rad_s=args.rotate_positive_rad_s,
        )
    if float(total.detach().cpu()) <= args.minimum_update_loss:
        return {
            "objective": float(total.detach().cpu()),
            **{name: float(value.detach().cpu()) for name, value in parts.items()},
            "gradient_norm": 0.0,
        }
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf")
    )
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError(f"non-finite gradient norm for {label}")
    scaler.step(optimizer)
    scaler.update()
    return {
        "objective": float(total.detach().cpu()),
        **{name: float(value.detach().cpu()) for name, value in parts.items()},
        "gradient_norm": float(gradient_norm.detach().cpu()),
    }


def _train_epoch(
    models: dict[str, ExpertModel], loader: DataLoader,
    optimizers: dict[str, torch.optim.Optimizer],
    scalers: dict[str, torch.amp.GradScaler], device: torch.device,
    geometry: torch.Tensor, position_mean: torch.Tensor,
    position_std: torch.Tensor, epoch: int, args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    for model in models.values():
        model.train()
    totals = {label: {} for label in models}
    count = 0
    for batch_index, raw in enumerate(loader):
        batch = _to_device(raw, device)
        generator = torch.Generator().manual_seed(
            args.seed + epoch * 1_000_003 + batch_index
        )
        angle = (
            2.0 * torch.pi * torch.rand(batch["obs"].shape[0], generator=generator)
            - torch.pi
        ).to(device)
        translation_xy = (
            0.5 * torch.rand(
                batch["obs"].shape[0], 2, generator=generator,
            ) - 0.25
        ).to(device)
        augmented = _rotation_augmented_batch(
            batch, geometry, position_mean, position_std, angle, translation_xy,
        )
        arm_batches = {
            "E_factorized_original": batch,
            "E_factorized_rot_aug": augmented,
        }
        batch_count = int(batch["obs"].shape[0])
        shared_rng = _capture_rng(device)
        after_first = None
        for index, (label, model) in enumerate(models.items()):
            if index:
                _restore_rng(shared_rng)
            values = _train_one(
                label, model, arm_batches[label], optimizers[label],
                scalers[label], device, args,
            )
            if index == 0:
                after_first = _capture_rng(device)
            for key, value in values.items():
                totals[label][key] = totals[label].get(key, 0.0) + value * batch_count
        if after_first is not None:
            _restore_rng(after_first)
        count += batch_count
    return {
        label: {key: value / max(count, 1) for key, value in values.items()}
        for label, values in totals.items()
    }


def train(args: argparse.Namespace) -> Path:
    _seed(args.seed)
    git_state = _git_state()
    if bool(git_state["worktree_dirty"]) and not args.allow_dirty_worktree:
        raise ValueError(
            "official factorized expert training requires a clean worktree"
        )
    dataset = Path(args.dataset).resolve()
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "stage3-causal-physical-v1":
        raise ValueError("factorized experts require causal physical v1")
    if not bool(manifest.get("qualification_passed", False)):
        raise ValueError("factorized experts require a qualified dataset")
    if bool(manifest.get("test_accessed", True)):
        raise ValueError("factorized experts refuse a test-accessed dataset")
    _validate_history_contract(manifest, args.history_events)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite expert output: {output}")
    geometry, geometry_payload, geometry_sha256 = _load_geometry(dataset)
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
    dataset_qualification = {
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
    output.mkdir(parents=True)
    loader_options = {
        "batch_size": args.batch_size, "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, **loader_options)
    validation_loader = DataLoader(validation_ds, **loader_options)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    common = {
        "geometry": geometry, "input_features": 5,
        "channels": args.channels, "dropout": args.dropout,
        "history_events": args.history_events,
        "position_mean": torch.from_numpy(train_ds.mean),
        "position_std": torch.from_numpy(train_ds.std),
        "moving_prior": args.moving_prior,
        "rotating_prior": args.rotating_prior,
    }
    original = FactorizedExpertPhysicalPredictor(**common)
    augmented = copy.deepcopy(original)
    initial_sha256 = _state_dict_sha256(original.state_dict())
    if initial_sha256 != _state_dict_sha256(augmented.state_dict()):
        raise RuntimeError("paired expert models must share exact initialization")
    models = {
        "E_factorized_original": original.to(device),
        "E_factorized_rot_aug": augmented.to(device),
    }
    optimizers = {
        label: torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        ) for label, model in models.items()
    }
    schedulers = {
        label: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max(1, args.epochs), eta_min=args.lr * 0.02,
        ) for label, optimizer in optimizers.items()
    }
    scalers = {
        label: torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and args.amp == "float16",
        ) for label in models
    }
    source_root = Path(__file__).resolve().parent
    source_names = (
        "train_factorized_motion_experts.py", "factorized_expert_loss.py",
        "causal_physical_state_model.py", "train_causal_physical_ab.py",
        "causal_physical_dataset.py", "physical_model.py",
        "physical_metrics.py", "pnp_state_targets.py",
    )
    baseline = Path(args.baseline_run).resolve()
    baseline_manifest = baseline / "run_manifest.json"
    if not baseline_manifest.exists():
        raise ValueError("registered v11 baseline manifest is missing")
    baseline_payload = json.loads(baseline_manifest.read_text(encoding="utf-8"))
    baseline_contract = {
        "schema_version": "stage3-causal-physical-state-ab-run-v1",
        "status": "complete", "stop_reason": "epoch_limit",
        "epochs_completed": 300, "test_accessed": False,
        "worktree_dirty": False,
        "dataset_manifest_sha256": _sha256(manifest_path),
    }
    mismatched = {
        key: {"expected": expected, "actual": baseline_payload.get(key)}
        for key, expected in baseline_contract.items()
        if baseline_payload.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"registered v11 baseline contract mismatch: {mismatched}")
    provenance: dict[str, object] = {
        "schema_version": "stage3-factorized-motion-experts-run-v1",
        "dataset": str(dataset), "dataset_manifest_sha256": _sha256(manifest_path),
        "geometry_template_sha256": geometry_sha256,
        "geometry_hash": geometry_payload.get("geometry_hash"),
        "dataset_qualification": dataset_qualification,
        "test_accessed": False,
        "baseline": {
            "path": str(baseline),
            "manifest_sha256": _sha256(baseline_manifest),
            "role": "frozen v11 shared-model comparison",
            "verified_contract": baseline_contract,
        },
        "input_allowlist": [
            "normalized exact xyz", "cyclic slot sin/cos", "event mask",
            "real event time", "tau", "train-only normalization",
        ],
        "forbidden_predictor_inputs": [
            "center", "velocity", "yaw", "yaw_rate", "motion_class",
            "gate truth", "rule_query", "future truth",
        ],
        "paired_contract": {
            "initial_state_sha256": initial_sha256,
            "trainable_parameters": trainable_parameter_count(original),
            "same_batches_dropout_rng_optimizer_scheduler_amp": True,
            "difference": (
                "E_rot_aug applies deterministic loss-side planar rotation "
                "about q0 plus xy translation within +/-0.25 m"
            ),
        },
        "architecture_contract": {
            "shared_encoder": "32-event fixed-slot causal TCN",
            "q0_head": "center0 and phase0",
            "translation_expert": "velocity_expert and supervised soft move gate",
            "rotation_expert": "omega_expert and supervised soft rotate gate",
            "decoder": "frozen constant-twist four-slot rigid geometry",
        },
        "objective_contract": {
            "formula": (
                "q0 center/phase + "
                f"{args.expert_weight}*positive expert state + "
                f"{args.gate_weight}*balanced gate BCE"
            ),
            "move_dead_band_mps": [args.move_negative_mps, args.move_positive_mps],
            "rotate_dead_band_rad_s": [
                args.rotate_negative_rad_s, args.rotate_positive_rad_s,
            ],
            "future_truth_role": "detached loss/evaluation labels only",
            "sampling": (
                "natural r4 sample order; each gate BCE is positive/negative "
                "group-balanced; experts are positive-only"
            ),
        },
        "selection_contract": (
            "worst dynamic-class q3 motion P95, overall eligible q3 P95, "
            "eligible q0 P95, worst dynamic-class q3 median"
        ),
        "acceptance_contract": {
            "v11_A_q0_p95_m_max": 0.111,
            "q3_motion_p95_m_max": 0.864,
            "moving_speed_ratio_median": [0.85, 1.15],
            "rotating_sign_accuracy_min": 0.99,
            "gate_positive_recall_min": 0.98,
            "gate_negative_false_positive_rate_max": 0.02,
        },
        "config": vars(args),
        "source_sha256": {name: _sha256(source_root / name) for name in source_names},
        "environment": {
            "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "device": str(device), "amp": args.amp,
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
        },
        **git_state,
    }
    _write_json(output / "run_manifest.json", provenance)

    validation = {
        label: _validate(model, validation_loader, device, args)
        for label, model in models.items()
    }
    history: list[dict[str, object]] = [{
        "epoch": 0, "validation": validation,
        "selection_tuple": {
            label: _expert_selection_tuple(value) for label, value in validation.items()
        },
    }]
    history_path = output / "stage3-factorized-experts-history.json"
    _write_json(history_path, history)
    best = {label: (float("inf"),) * 4 for label in models}
    best_epoch = {label: -1 for label in models}
    best_paths = {
        label: output / f"stage3-{label}-seed{args.seed}-best.pt"
        for label in models
    }
    milestone_epochs = {20, 50, 100, 150, 200, 250, args.epochs}
    for label, model in models.items():
        _checkpoint(
            output / f"stage3-{label}-seed{args.seed}-initial.pt",
            model, label, 0, validation[label], provenance, "initial",
        )
    started = time.monotonic()
    epochs_completed = 0
    stop_reason = "epoch_limit"
    position_mean = torch.from_numpy(train_ds.mean).to(device)
    position_std = torch.from_numpy(train_ds.std).to(device)
    geometry_device = geometry.to(device)
    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        train_metrics = _train_epoch(
            models, train_loader, optimizers, scalers, device,
            geometry_device, position_mean, position_std, epoch, args,
        )
        for scheduler in schedulers.values():
            scheduler.step()
        validation = {
            label: _validate(model, validation_loader, device, args)
            for label, model in models.items()
        }
        epochs_completed = epoch
        record = {
            "epoch": epoch, "train": train_metrics, "validation": validation,
            "selection_tuple": {
                label: _expert_selection_tuple(value)
                for label, value in validation.items()
            },
            "lr": {
                label: optimizer.param_groups[0]["lr"]
                for label, optimizer in optimizers.items()
            },
        }
        history.append(record)
        _write_json(history_path, history)
        print(json.dumps(record, sort_keys=True), flush=True)
        for label, model in models.items():
            selection = _expert_selection_tuple(validation[label])
            if selection < best[label]:
                best[label] = selection
                best_epoch[label] = epoch
                _checkpoint(
                    best_paths[label], model, label, epoch, validation[label],
                    provenance, "best", optimizers[label], schedulers[label],
                    scalers[label],
                )
            if epoch in milestone_epochs:
                _checkpoint(
                    output / f"stage3-{label}-seed{args.seed}-epoch{epoch:03d}.pt",
                    model, label, epoch, validation[label], provenance,
                    "milestone", optimizers[label], schedulers[label],
                    scalers[label],
                )
        if args.max_wall_minutes > 0 and (
            time.monotonic() - started >= args.max_wall_minutes * 60
        ):
            stop_reason = "wall_time_limit"
            break
    for label, model in models.items():
        _checkpoint(
            output / f"stage3-{label}-seed{args.seed}-last.pt",
            model, label, epochs_completed, validation[label], provenance, "last",
            optimizers[label], schedulers[label], scalers[label],
        )
    final = {
        **provenance, "status": "complete", "stop_reason": stop_reason,
        "epochs_completed": epochs_completed,
        "best": {
            label: {
                "path": path.name, "sha256": _sha256(path),
                "selection_tuple": best[label], "epoch": best_epoch[label],
                "trained_checkpoint": best_epoch[label] > 0,
            } for label, path in best_paths.items()
        },
    }
    _write_json(output / "run_manifest.json", final)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--history-events", type=int, default=32)
    parser.add_argument("--huber-beta-m", type=float, default=0.005)
    parser.add_argument("--reference-horizon-s", type=float, default=0.5)
    parser.add_argument("--expert-weight", type=float, default=1.0)
    parser.add_argument("--gate-weight", type=float, default=0.1)
    parser.add_argument("--moving-prior", type=float, default=0.464)
    parser.add_argument("--rotating-prior", type=float, default=0.544)
    parser.add_argument("--move-negative-mps", type=float, default=0.01)
    parser.add_argument("--move-positive-mps", type=float, default=0.10)
    parser.add_argument("--rotate-negative-rad-s", type=float, default=0.05)
    parser.add_argument("--rotate-positive-rad-s", type=float, default=0.20)
    parser.add_argument("--minimum-supervision-coverage", type=float, default=0.85)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--minimum-update-loss", type=float, default=1e-10)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--amp", choices=("off", "bfloat16", "float16"), default="bfloat16",
    )
    parser.add_argument("--train-sample-limit", type=int, default=0)
    parser.add_argument("--validation-sample-limit", type=int, default=0)
    parser.add_argument("--max-wall-minutes", type=float, default=0.0)
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    args = parser.parse_args()
    positive = (
        args.epochs, args.batch_size, args.lr, args.channels,
        args.huber_beta_m, args.reference_horizon_s,
        args.expert_weight, args.gate_weight,
    )
    if any(value <= 0 for value in positive):
        parser.error("factorized expert arguments must be positive")
    if args.channels % 4:
        parser.error("channels must be divisible by four")
    if args.weight_decay < 0 or args.grad_clip < 0 or args.minimum_update_loss < 0:
        parser.error("nonnegative optimizer arguments cannot be negative")
    if not 8 <= args.history_events <= 200:
        parser.error("history-events must be within [8,200]")
    if not 0 < args.minimum_supervision_coverage <= 1:
        parser.error("minimum-supervision-coverage must be within (0,1]")
    if args.train_sample_limit < 0 or args.validation_sample_limit < 0:
        parser.error("sample limits cannot be negative")
    if not 0 < args.moving_prior < 1 or not 0 < args.rotating_prior < 1:
        parser.error("gate priors must lie within (0,1)")
    if not 0 <= args.move_negative_mps < args.move_positive_mps:
        parser.error("move gate thresholds are invalid")
    if not 0 <= args.rotate_negative_rad_s < args.rotate_positive_rad_s:
        parser.error("rotate gate thresholds are invalid")
    print(train(args))


if __name__ == "__main__":
    main()
