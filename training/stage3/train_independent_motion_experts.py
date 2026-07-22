"""Train independent rotation/combined experts and a four-class router."""

from __future__ import annotations

import argparse
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
    IndependentMotionExpertSystem,
    trainable_parameter_count,
)
from .independent_motion_expert_loss import independent_motion_expert_loss
from .pnp_state_targets import truth_trajectory_targets
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


V12_COMMIT = "49ebc36df51900bd59f80435f718a1e79bc9317f"
V12_ORIGINAL_SHA256 = (
    "5437d6c48fb57805a4e2a02a8814a7d5681c1821a6137c78d02c7326b954aafa"
)
V12_AUGMENTED_SHA256 = (
    "e3b93c708d174e476af2badc99620a9904dfb669da6545a9610a75507b896b49"
)


def _selection_tuple(metrics: dict[str, object]) -> tuple[float, ...]:
    factors = metrics["strata"]["motion_factor"]  # type: ignore[index]
    dynamic = [
        factors[name]["q3_trajectory_eligible_motion"]
        for name in ("translation", "rotation", "combined")
    ]
    rotation_and_combined = dynamic[1:]
    q3 = metrics["queries"][3]["trajectory_eligible"]["motion_delta"]  # type: ignore[index]
    def metric(value: dict[str, object], name: str) -> float:
        # Bounded smoke selections may omit a factor.  Formal full-data runs
        # require all factors in preflight, so this fallback is diagnostic-only.
        return float(value.get(name, q3[name]))
    router = metrics.get("router_diagnostics", {})
    macro_recall = router.get("macro_recall") if isinstance(router, dict) else None
    router_penalty = 1.0 if macro_recall is None else 1.0 - float(macro_recall)
    return (
        max(metric(value, "p95_m") for value in rotation_and_combined),
        max(metric(value, "p95_m") for value in dynamic),
        float(q3["p95_m"]),
        router_penalty,
        float(metrics["trajectory_eligible_state_q0"]["p95_m"]),  # type: ignore[index]
        max(metric(value, "median_m") for value in rotation_and_combined),
    )


def _checkpoint(
    path: Path, model: IndependentMotionExpertSystem, epoch: int,
    metrics: dict[str, object], provenance: dict[str, object], role: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    payload: dict[str, object] = {
        "model": model.state_dict(),
        "model_class": model.__class__.__name__,
        "model_config": model.config(),
        "label": "independent_motion_expert_system",
        "epoch": epoch, "checkpoint_role": role,
        "validation": metrics,
        "selection_tuple": _selection_tuple(metrics),
        "provenance": provenance,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def _factorized_from_checkpoint(
    path: Path, expected_sha256: str, expected_label: str,
    expected_epoch: int, dataset_manifest_sha256: str,
) -> tuple[FactorizedExpertPhysicalPredictor, dict[str, object]]:
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"source checkpoint hash mismatch for {path.name}: {actual_sha256}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "model_class": "FactorizedExpertPhysicalPredictor",
        "label": expected_label, "epoch": expected_epoch,
        "checkpoint_role": "best",
    }
    mismatch = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items() if payload.get(key) != value
    }
    provenance = payload.get("provenance", {})
    source_contract = {
        "test_accessed": False, "worktree_dirty": False,
        "git_commit": V12_COMMIT,
        "dataset_manifest_sha256": dataset_manifest_sha256,
    }
    mismatch.update({
        f"provenance.{key}": {"expected": value, "actual": provenance.get(key)}
        for key, value in source_contract.items()
        if provenance.get(key) != value
    })
    if mismatch:
        raise ValueError(f"source checkpoint contract mismatch: {mismatch}")
    config = payload["model_config"]
    model = FactorizedExpertPhysicalPredictor(
        geometry=torch.tensor(config["geometry"], dtype=torch.float32),
        position_mean=torch.tensor(config["position_mean"], dtype=torch.float32),
        position_std=torch.tensor(config["position_std"], dtype=torch.float32),
        input_features=int(config["input_features"]),
        channels=int(config["channels"]), dropout=float(config["dropout"]),
        history_events=int(config["history_events"]),
        maximum_speed_mps=float(config["maximum_speed_mps"]),
        maximum_yaw_rate_rad_s=float(config["maximum_yaw_rate_rad_s"]),
        moving_prior=float(config["moving_prior"]),
        rotating_prior=float(config["rotating_prior"]),
    )
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload


def _factor_audit(
    dataset: CausalPhysicalShardDataset, geometry: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, object]:
    counts = np.zeros(4, dtype=np.int64)
    eligible_total = 0
    valid_total = 0
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    with torch.no_grad():
        for batch in loader:
            truth = truth_trajectory_targets(
                batch["future_position"][:, :4], batch["tau"][:, :4],
                geometry, rule_queries=4,
            )
            eligible = batch["rule_query"][:, :4].all(dim=1) & truth["constant_motion"]
            speed = torch.linalg.vector_norm(truth["velocity"], dim=-1)
            abs_omega = truth["omega"].abs()
            move_positive = speed >= args.move_positive_mps
            move_negative = speed <= args.move_negative_mps
            rotate_positive = abs_omega >= args.rotate_positive_rad_s
            rotate_negative = abs_omega <= args.rotate_negative_rad_s
            valid = eligible & (move_positive | move_negative) & (
                rotate_positive | rotate_negative
            )
            labels = move_positive.long() + 2 * rotate_positive.long()
            eligible_total += int(eligible.sum())
            valid_total += int(valid.sum())
            for index in range(4):
                counts[index] += int((valid & (labels == index)).sum())
    if np.any(counts == 0):
        raise ValueError(f"all four motion factors are required: {counts.tolist()}")
    coverage = valid_total / max(eligible_total, 1)
    if coverage < args.minimum_router_coverage:
        raise ValueError(
            f"router supervision coverage {coverage:.4f} is below minimum"
        )
    return {
        "labels": ["stationary", "translation", "rotation", "combined"],
        "counts": counts.tolist(), "eligible_count": eligible_total,
        "valid_count": valid_total, "valid_over_eligible": coverage,
    }


def _foundation_hashes(model: IndependentMotionExpertSystem) -> dict[str, str]:
    return {
        "pose_encoder": _state_dict_sha256(model.pose_encoder.state_dict()),
        "q0_head": _state_dict_sha256(model.q0_head.state_dict()),
        "translation_encoder": _state_dict_sha256(
            model.translation_encoder.state_dict()
        ),
        "translation_head": _state_dict_sha256(
            model.translation_head.state_dict()
        ),
    }


def _train_epoch(
    model: IndependentMotionExpertSystem, loader: DataLoader,
    optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
    device: torch.device, args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    parameters = [value for value in model.parameters() if value.requires_grad]
    for raw in loader:
        batch = _to_device(raw, device)
        optimizer.zero_grad(set_to_none=True)
        amp_enabled = device.type == "cuda" and args.amp != "off"
        amp_dtype = torch.float16 if args.amp == "float16" else torch.bfloat16
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled,
        ):
            prediction = model.forward_trainable_experts(
                batch["obs"], batch["obs_mask"], batch["event_mask"],
                batch["event_time_s"],
            )
            total, parts = independent_motion_expert_loss(
                prediction, batch["future_position"], batch["tau"],
                batch["rule_query"], model.decoder.geometry,
                huber_beta_m=args.huber_beta_m,
                reference_horizon_s=args.reference_horizon_s,
                rotation_weight=args.rotation_weight,
                combined_velocity_weight=args.combo_velocity_weight,
                combined_rotation_weight=args.combo_omega_weight,
                router_weight=args.router_weight,
                move_negative_mps=args.move_negative_mps,
                move_positive_mps=args.move_positive_mps,
                rotate_negative_rad_s=args.rotate_negative_rad_s,
                rotate_positive_rad_s=args.rotate_positive_rad_s,
            )
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite independent expert objective")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, args.grad_clip if args.grad_clip > 0 else float("inf")
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite independent expert gradient")
        scaler.step(optimizer)
        scaler.update()
        batch_count = int(batch["obs"].shape[0])
        values = {
            "objective": float(total.detach().cpu()),
            **{name: float(value.detach().cpu()) for name, value in parts.items()},
            "gradient_norm": float(gradient_norm.detach().cpu()),
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise FloatingPointError("non-finite independent expert metric")
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value * batch_count
        count += batch_count
    return {name: value / max(count, 1) for name, value in totals.items()}


def train(args: argparse.Namespace) -> Path:
    _seed(args.seed)
    git_state = _git_state()
    if bool(git_state["worktree_dirty"]) and not args.allow_dirty_worktree:
        raise ValueError("official independent expert training requires a clean worktree")
    dataset = Path(args.dataset).resolve()
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "stage3-causal-physical-v1":
        raise ValueError("independent experts require causal physical v1")
    if not bool(manifest.get("qualification_passed", False)):
        raise ValueError("independent experts require a qualified dataset")
    if bool(manifest.get("test_accessed", True)):
        raise ValueError("independent experts refuse a test-accessed dataset")
    _validate_history_contract(manifest, args.history_events)
    dataset_sha256 = _sha256(manifest_path)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite independent output: {output}")

    source_run = Path(args.source_run).resolve()
    source_manifest_path = source_run / "run_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_expected = {
        "schema_version": "stage3-factorized-motion-experts-run-v1",
        "status": "complete", "stop_reason": "epoch_limit",
        "epochs_completed": 300, "test_accessed": False,
        "worktree_dirty": False, "git_commit": V12_COMMIT,
        "dataset_manifest_sha256": dataset_sha256,
    }
    mismatch = {
        key: {"expected": value, "actual": source_manifest.get(key)}
        for key, value in source_expected.items()
        if source_manifest.get(key) != value
    }
    if mismatch:
        raise ValueError(f"registered v12 run contract mismatch: {mismatch}")
    source_best = source_manifest.get("best", {})
    expected_best = {
        "E_factorized_original": (266, V12_ORIGINAL_SHA256),
        "E_factorized_rot_aug": (283, V12_AUGMENTED_SHA256),
    }
    for label, (epoch, sha256) in expected_best.items():
        record = source_best.get(label, {})
        if record.get("epoch") != epoch or record.get("sha256") != sha256:
            raise ValueError(f"registered v12 best mismatch for {label}: {record}")
    original_path = source_run / source_best["E_factorized_original"]["path"]
    augmented_path = source_run / source_best["E_factorized_rot_aug"]["path"]
    original, _ = _factorized_from_checkpoint(
        original_path, V12_ORIGINAL_SHA256, "E_factorized_original", 266,
        dataset_sha256,
    )
    augmented, _ = _factorized_from_checkpoint(
        augmented_path, V12_AUGMENTED_SHA256, "E_factorized_rot_aug", 283,
        dataset_sha256,
    )

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
    qualification = {
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

    config = original.config()
    if int(config["channels"]) != args.channels or float(config["dropout"]) != args.dropout:
        raise ValueError("v13 architecture must match registered v12 channels/dropout")
    model = IndependentMotionExpertSystem(
        geometry=geometry, position_mean=torch.from_numpy(train_ds.mean),
        position_std=torch.from_numpy(train_ds.std), input_features=5,
        channels=args.channels, dropout=args.dropout,
        history_events=args.history_events,
        maximum_speed_mps=float(config["maximum_speed_mps"]),
        maximum_yaw_rate_rad_s=float(config["maximum_yaw_rate_rad_s"]),
    )
    model.initialize_from_factorized(augmented, original)
    model.freeze_foundations()
    foundation_hashes = _foundation_hashes(model)
    initial_state_sha256 = _state_dict_sha256(model.state_dict())
    total_parameters = sum(value.numel() for value in model.parameters())
    frozen_parameters = sum(
        value.numel() for value in model.parameters() if not value.requires_grad
    )
    trainable_parameters = trainable_parameter_count(model)
    if trainable_parameters + frozen_parameters != total_parameters:
        raise RuntimeError("parameter accounting mismatch")

    output.mkdir(parents=True)
    loader_options = {
        "batch_size": args.batch_size, "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, **loader_options)
    validation_loader = DataLoader(validation_ds, **loader_options)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(1, args.epochs), eta_min=args.lr * 0.02,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.amp == "float16",
    )
    source_root = Path(__file__).resolve().parent
    source_names = (
        "train_independent_motion_experts.py",
        "independent_motion_expert_loss.py", "causal_physical_state_model.py",
        "train_causal_physical_ab.py", "causal_physical_dataset.py",
        "physical_model.py", "physical_metrics.py", "pnp_state_targets.py",
    )
    provenance: dict[str, object] = {
        "schema_version": "stage3-independent-motion-experts-run-v1",
        "dataset": str(dataset), "dataset_manifest_sha256": dataset_sha256,
        "geometry_template_sha256": geometry_sha256,
        "geometry_hash": geometry_payload.get("geometry_hash"),
        "dataset_qualification": qualification, "test_accessed": False,
        "source_v12": {
            "path": str(source_run),
            "manifest_sha256": _sha256(source_manifest_path),
            "verified_contract": source_expected,
            "pose_q0": {
                "label": "E_factorized_rot_aug", "epoch": 283,
                "path": augmented_path.name, "sha256": V12_AUGMENTED_SHA256,
                "frozen": True,
            },
            "translation": {
                "label": "E_factorized_original", "epoch": 266,
                "path": original_path.name, "sha256": V12_ORIGINAL_SHA256,
                "frozen": True,
            },
        },
        "input_allowlist": [
            "normalized exact xyz", "cyclic slot sin/cos", "event mask",
            "real event time", "tau", "train-only normalization",
        ],
        "forbidden_predictor_inputs": [
            "center", "velocity", "yaw", "yaw_rate", "motion_class",
            "router truth", "rule_query", "future truth",
        ],
        "architecture_contract": {
            "pose_q0": "frozen augmented-v12 encoder and q0 head",
            "translation": "frozen original-v12 encoder and velocity head",
            "rotation": "independent trainable encoder and omega head; pure rotation only",
            "combined": "independent trainable encoder jointly predicting velocity and omega",
            "router": "independent trainable encoder with four-class hard routing",
            "route_order": list(model.route_names),
            "decoder": "frozen constant-twist four-slot rigid geometry",
            "combined_is_not_specialist_trajectory_addition": True,
        },
        "objective_contract": {
            "formula": (
                f"{args.rotation_weight}*pure_rotation_omega + "
                f"{args.combo_velocity_weight}*combined_velocity + "
                f"{args.combo_omega_weight}*combined_omega + "
                f"{args.router_weight}*balanced_four_class_router_ce"
            ),
            "move_dead_band_mps": [args.move_negative_mps, args.move_positive_mps],
            "rotate_dead_band_rad_s": [
                args.rotate_negative_rad_s, args.rotate_positive_rad_s,
            ],
            "future_truth_role": "detached loss/evaluation labels only",
            "sampling": "natural r4 order; per-present-factor group-balanced CE",
        },
        "selection_contract": (
            "factor rotation/combo worst q3 P95, factor dynamic worst q3 P95, "
            "overall eligible q3 P95, router macro-recall penalty, q0 P95, "
            "factor rotation/combo worst q3 median"
        ),
        "acceptance_goals_not_stop_conditions": {
            "router_per_class_recall_min": 0.98,
            "router_macro_recall_min": 0.98,
            "rotation_and_combined_sign_accuracy_min": 0.99,
            "moving_speed_ratio_median": [0.85, 1.15],
            "q0_p95_m_reference": 0.04241919293999669,
            "v12_spin_q3_p95_m_reference": 0.10570980,
            "v12_combined_q3_p95_m_reference": 0.17404612898826596,
        },
        "parameter_contract": {
            "total": total_parameters, "trainable": trainable_parameters,
            "frozen": frozen_parameters,
            "foundation_state_sha256": foundation_hashes,
            "initial_integrated_state_sha256": initial_state_sha256,
            "optimizer_contains_only_trainable_parameters": True,
        },
        "config": vars(args),
        "source_sha256": {
            name: _sha256(source_root / name) for name in source_names
        },
        "environment": {
            "python": sys.version, "numpy": np.__version__,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "device": str(device), "amp": args.amp,
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
        },
        **git_state,
    }
    _write_json(output / "run_manifest.json", provenance)

    validation = _validate(model, validation_loader, device, args)
    history: list[dict[str, object]] = [{
        "epoch": 0, "validation": validation,
        "selection_tuple": _selection_tuple(validation),
    }]
    history_path = output / "stage3-independent-motion-experts-history.json"
    _write_json(history_path, history)
    best = (float("inf"),) * 6
    best_epoch = -1
    best_path = output / f"stage3-independent-motion-experts-seed{args.seed}-best.pt"
    _checkpoint(
        output / f"stage3-independent-motion-experts-seed{args.seed}-initial.pt",
        model, 0, validation, provenance, "initial",
    )
    milestone_epochs = {20, 50, 100, 150, 200, 250, args.epochs}
    started = time.monotonic()
    epochs_completed = 0
    stop_reason = "epoch_limit"
    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        train_metrics = _train_epoch(
            model, train_loader, optimizer, scaler, device, args,
        )
        scheduler.step()
        validation = _validate(model, validation_loader, device, args)
        if epoch == 1 and _foundation_hashes(model) != foundation_hashes:
            raise RuntimeError("frozen v12 foundation changed during epoch 1")
        epochs_completed = epoch
        selection = _selection_tuple(validation)
        record = {
            "epoch": epoch, "train": train_metrics,
            "validation": validation, "selection_tuple": selection,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        _write_json(history_path, history)
        print(json.dumps(record, sort_keys=True), flush=True)
        if selection < best:
            best = selection
            best_epoch = epoch
            _checkpoint(
                best_path, model, epoch, validation, provenance, "best",
                optimizer, scheduler, scaler,
            )
        if epoch in milestone_epochs:
            _checkpoint(
                output / (
                    f"stage3-independent-motion-experts-seed{args.seed}-"
                    f"epoch{epoch:03d}.pt"
                ),
                model, epoch, validation, provenance, "milestone",
                optimizer, scheduler, scaler,
            )
        if args.max_wall_minutes > 0 and (
            time.monotonic() - started >= args.max_wall_minutes * 60
        ):
            stop_reason = "wall_time_limit"
            break
    last_path = output / f"stage3-independent-motion-experts-seed{args.seed}-last.pt"
    _checkpoint(
        last_path, model, epochs_completed, validation, provenance, "last",
        optimizer, scheduler, scaler,
    )
    final_foundation_hashes = _foundation_hashes(model)
    if final_foundation_hashes != foundation_hashes:
        raise RuntimeError("frozen v12 foundation changed during training")
    final = {
        **provenance, "status": "complete", "stop_reason": stop_reason,
        "epochs_completed": epochs_completed,
        "frozen_foundation_verified_unchanged": True,
        "best": {
            "path": best_path.name, "sha256": _sha256(best_path),
            "selection_tuple": best, "epoch": best_epoch,
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
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--history-events", type=int, default=32)
    parser.add_argument("--huber-beta-m", type=float, default=0.005)
    parser.add_argument("--reference-horizon-s", type=float, default=0.5)
    parser.add_argument("--rotation-weight", type=float, default=1.0)
    parser.add_argument("--combo-velocity-weight", type=float, default=1.0)
    parser.add_argument("--combo-omega-weight", type=float, default=1.0)
    parser.add_argument("--router-weight", type=float, default=0.1)
    parser.add_argument("--move-negative-mps", type=float, default=0.01)
    parser.add_argument("--move-positive-mps", type=float, default=0.10)
    parser.add_argument("--rotate-negative-rad-s", type=float, default=0.05)
    parser.add_argument("--rotate-positive-rad-s", type=float, default=0.20)
    parser.add_argument("--minimum-supervision-coverage", type=float, default=0.85)
    parser.add_argument("--minimum-router-coverage", type=float, default=0.85)
    parser.add_argument("--grad-clip", type=float, default=0.0)
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
        args.huber_beta_m, args.reference_horizon_s, args.rotation_weight,
        args.combo_velocity_weight, args.combo_omega_weight, args.router_weight,
    )
    if any(value <= 0 for value in positive):
        parser.error("independent expert arguments must be positive")
    if args.channels % 4:
        parser.error("channels must be divisible by four")
    if args.weight_decay < 0 or args.grad_clip < 0:
        parser.error("nonnegative optimizer arguments cannot be negative")
    if not 8 <= args.history_events <= 200:
        parser.error("history-events must be within [8,200]")
    if not 0 < args.minimum_supervision_coverage <= 1:
        parser.error("minimum-supervision-coverage must be within (0,1]")
    if not 0 < args.minimum_router_coverage <= 1:
        parser.error("minimum-router-coverage must be within (0,1]")
    if args.train_sample_limit < 0 or args.validation_sample_limit < 0:
        parser.error("sample limits cannot be negative")
    if not 0 <= args.move_negative_mps < args.move_positive_mps:
        parser.error("move router thresholds are invalid")
    if not 0 <= args.rotate_negative_rad_s < args.rotate_positive_rad_s:
        parser.error("rotate router thresholds are invalid")
    print(train(args))


if __name__ == "__main__":
    main()
