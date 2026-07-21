"""Train two physical-only Stage-3 predictors on the existing v3 truth labels.

A is a direct anchored displacement network.  B decodes a learned rigid
constant-twist latent through the recorded four-armor geometry.  The test split
is never constructed or opened by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .physical_baseline import ExactStateRigidRollout
from .physical_loss import physical_loss
from .physical_metrics import physical_batch_errors, summary
from .physical_model import AnchoredDeltaPredictor, RigidMotionPredictor
from .shard_dataset import Stage3ShardDataset
from .truth_history_dataset import TruthHistoryShardDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_geometry(dataset: Path) -> tuple[torch.Tensor, dict[str, object], str]:
    path = dataset / "geometry_template.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    armor = sorted(payload["armors"], key=lambda item: int(item["relative_slot"]))
    geometry = torch.tensor(
        [item["relative_position_m"] for item in armor], dtype=torch.float32
    )
    if geometry.shape != (4, 3):
        raise ValueError("geometry template must contain four xyz armor positions")
    return geometry, payload, _sha256(path)


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "train_physical_ab.py", "physical_model.py", "physical_loss.py",
        "physical_metrics.py", "physical_baseline.py", "shard_dataset.py",
        "truth_history_dataset.py", "model.py",
    )
    return {name: _sha256(root / name) for name in names}


def _git_state() -> dict[str, object]:
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(repo), "status", "--short"], text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    return {"git_commit": commit, "worktree_dirty": dirty}


def _to_device(raw: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in raw.items()}


def _capture_rng(device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    )


def _restore_rng(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def _train_one(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = device.type == "cuda" and args.amp != "off"
    amp_dtype = torch.float16 if args.amp == "float16" else torch.bfloat16
    with torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
    ):
        prediction = model(
            batch["obs"], batch["obs_mask"], batch["event_mask"],
            batch["event_time_s"], batch["tau"],
        )["position_mean"]
        loss, metrics = physical_loss(
            prediction, batch["future_position"], batch["tau"],
            query_mask=batch.get("rule_query"),
            huber_beta_m=args.huber_beta_m,
            state_weight=args.state_weight,
            motion_weight=args.motion_weight,
            absolute_weight=args.absolute_weight,
            rigidity_weight=args.rigidity_weight,
        )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    metrics["gradient_norm"] = float(gradient_norm.detach().cpu())
    return metrics


def _train_epoch(
    models: dict[str, torch.nn.Module],
    loader: DataLoader,
    optimizers: dict[str, torch.optim.Optimizer],
    scalers: dict[str, torch.amp.GradScaler],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    for model in models.values():
        model.train()
    totals: dict[str, dict[str, float]] = {label: {} for label in models}
    count = 0
    for raw in loader:
        batch = _to_device(raw, device)
        batch_count = int(batch["obs"].shape[0])
        shared_rng = _capture_rng(device)
        for index, (label, model) in enumerate(models.items()):
            if index:
                _restore_rng(shared_rng)
            values = _train_one(
                model, batch, optimizers[label], scalers[label], device, args
            )
            for key, value in values.items():
                totals[label][key] = totals[label].get(key, 0.0) + value * batch_count
        count += batch_count
    return {
        label: {key: value / max(count, 1) for key, value in values.items()}
        for label, values in totals.items()
    }


def _validate_model(
    model: torch.nn.Module, loader: DataLoader, device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    model.eval()
    arrays: dict[str, list[np.ndarray]] = {}
    tau_parts: list[np.ndarray] = []
    rule_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    motion_class_parts: list[np.ndarray] = []
    losses: list[float] = []
    counts: list[int] = []
    amp_enabled = device.type == "cuda" and args.amp != "off"
    amp_dtype = torch.float16 if args.amp == "float16" else torch.bfloat16
    with torch.no_grad():
        for raw in loader:
            batch = _to_device(raw, device)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                if isinstance(model, ExactStateRigidRollout):
                    prediction = model(
                        batch["future_position"][:, 0],
                        batch["anchor_center_position_m"],
                        batch["anchor_velocity_mps"],
                        batch["anchor_yaw_rate_rad_s"],
                        batch["tau"],
                    )["position_mean"]
                else:
                    prediction = model(
                        batch["obs"], batch["obs_mask"], batch["event_mask"],
                        batch["event_time_s"], batch["tau"],
                    )["position_mean"]
                loss, _ = physical_loss(
                    prediction, batch["future_position"], batch["tau"],
                    query_mask=batch.get("rule_query"),
                    huber_beta_m=args.huber_beta_m,
                    state_weight=args.state_weight,
                    motion_weight=args.motion_weight,
                    absolute_weight=args.absolute_weight,
                    rigidity_weight=args.rigidity_weight,
                )
            error = physical_batch_errors(prediction.float(), batch["future_position"])
            for name in (
                "state_q0_m", "absolute_pg_m", "motion_delta_m", "centroid_q0_m",
                "centered_shape_q0_m", "rigid_residual_q0_m", "permutation_gap_m",
            ):
                arrays.setdefault(name, []).append(error[name].detach().cpu().numpy())
            tau_parts.append(batch["tau"].detach().cpu().numpy())
            rule_parts.append(
                batch.get(
                    "rule_query", torch.ones_like(batch["tau"], dtype=torch.bool)
                ).detach().cpu().numpy()
            )
            distance_parts.append(
                batch.get(
                    "distance_m",
                    torch.linalg.vector_norm(
                        batch["future_position"][:, 0].mean(dim=1), dim=-1
                    ),
                ).detach().cpu().numpy()
            )
            motion_class_parts.append(batch["motion_class"].detach().cpu().numpy())
            counts.append(int(batch["obs"].shape[0]))
            losses.append(float(loss.detach().cpu()))
    merged = {name: np.concatenate(parts, axis=0) for name, parts in arrays.items()}
    tau = np.concatenate(tau_parts, axis=0)
    rule = np.concatenate(rule_parts, axis=0).astype(np.bool_, copy=False)
    distance = np.concatenate(distance_parts, axis=0)
    motion_class = np.concatenate(motion_class_parts, axis=0)
    query_metrics: list[dict[str, object]] = []
    for query in range(tau.shape[1]):
        active_rule = rule[:, query]
        query_metrics.append({
            "query_index": query,
            "tau_s": {
                "median": float(np.median(tau[:, query])),
                "minimum": float(np.min(tau[:, query])),
                "maximum": float(np.max(tau[:, query])),
            },
            "absolute_pg": summary(merged["absolute_pg_m"][:, query]),
            "motion_delta": summary(merged["motion_delta_m"][:, query]),
            "rule": {
                "absolute_pg": summary(merged["absolute_pg_m"][active_rule, query]),
                "motion_delta": summary(merged["motion_delta_m"][active_rule, query]),
            },
            "future_event": {
                "absolute_pg": summary(merged["absolute_pg_m"][~active_rule, query]),
                "motion_delta": summary(merged["motion_delta_m"][~active_rule, query]),
            },
        })

    def stratum(mask: np.ndarray) -> dict[str, object]:
        return {
            "sample_count": int(mask.sum()),
            "state_q0": summary(merged["state_q0_m"][mask]),
            "headline_queries": [{
                "query_index": query,
                "absolute_pg": summary(merged["absolute_pg_m"][mask, query]),
                "motion_delta": summary(merged["motion_delta_m"][mask, query]),
                "rule": {
                    "absolute_pg": summary(
                        merged["absolute_pg_m"][mask & rule[:, query], query]
                    ),
                    "motion_delta": summary(
                        merged["motion_delta_m"][mask & rule[:, query], query]
                    ),
                },
                "future_event": {
                    "absolute_pg": summary(
                        merged["absolute_pg_m"][mask & ~rule[:, query], query]
                    ),
                    "motion_delta": summary(
                        merged["motion_delta_m"][mask & ~rule[:, query], query]
                    ),
                },
            } for query in range(min(4, tau.shape[1]))],
        }

    motion_labels = {
        0: "stationary", 1: "linear", 2: "spin", 3: "linear_and_spin",
    }
    strata = {
        "motion_class": {
            label: stratum(motion_class == index)
            for index, label in motion_labels.items()
        },
        "distance": {
            "near_lt_3m": stratum(distance < 3.0),
            "mid_3_to_5m": stratum((distance >= 3.0) & (distance < 5.0)),
            "far_ge_5m": stratum(distance >= 5.0),
        },
    }
    return {
        "sample_count": int(tau.shape[0]),
        "loss": float(np.average(losses, weights=counts)),
        "state_q0": summary(merged["state_q0_m"]),
        "centroid_q0": summary(merged["centroid_q0_m"]),
        "centered_shape_q0": summary(merged["centered_shape_q0_m"]),
        "rigid_residual_q0": summary(merged["rigid_residual_q0_m"]),
        "permutation_gap": summary(merged["permutation_gap_m"]),
        "queries": query_metrics,
        "rule_query_fraction": float(rule.mean()),
        "strata": strata,
    }


def _validate_pair(
    models: dict[str, torch.nn.Module], loader: DataLoader,
    device: torch.device, args: argparse.Namespace,
) -> dict[str, dict[str, object]]:
    return {
        label: _validate_model(model, loader, device, args)
        for label, model in models.items()
    }


def _selection_tuple(metrics: dict[str, object]) -> tuple[float, ...]:
    queries = metrics["queries"]  # type: ignore[assignment]
    headline = queries[1:4]  # fixed nominal 0.1, 0.2 and 0.5 second queries
    return (
        float(metrics["state_q0"]["p95_m"]),  # type: ignore[index]
        max(float(item["rule"]["motion_delta"]["p95_m"]) for item in headline),
        max(float(item["rule"]["absolute_pg"]["p95_m"]) for item in headline),
        float(metrics["state_q0"]["median_m"]),  # type: ignore[index]
        max(float(item["rule"]["motion_delta"]["median_m"]) for item in headline),
        float(metrics["loss"]),
    )


def _checkpoint(
    path: Path, model: torch.nn.Module, label: str, epoch: int,
    metrics: dict[str, object], provenance: dict[str, object], role: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    payload: dict[str, object] = {
        "model": model.state_dict(),
        "model_class": model.__class__.__name__,
        "model_config": model.config(),  # type: ignore[attr-defined]
        "objective": "physical_truth_state_motion_absolute_rigidity_v1",
        "label": label,
        "epoch": epoch,
        "checkpoint_role": role,
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


def train(args: argparse.Namespace) -> Path:
    _seed(args.seed)
    dataset = Path(args.dataset).resolve()
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = str(manifest.get("schema_version"))
    if schema_version not in {"stage3-dataset-v3", "stage3-truth-history-v1"} or not bool(
        manifest.get("qualification_passed", False)
    ):
        raise ValueError("physical A/B training requires a qualified physical dataset")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training output: {output}")
    output.mkdir(parents=True)
    geometry, geometry_payload, geometry_sha256 = _load_geometry(dataset)

    truth_history = schema_version == "stage3-truth-history-v1"
    dataset_type = TruthHistoryShardDataset if truth_history else Stage3ShardDataset
    train_options: dict[str, object] = {
        "seed": args.seed,
        # Tiny-fit is a memorization diagnostic: both loaders must expose the
        # exact same bounded sample set on every epoch.  Formal training keeps
        # the normal epoch-wise shuffle.
        "shuffle": not args.validation_on_train,
        "sample_limit": args.train_sample_limit,
    }
    if not truth_history:
        train_options["augment"] = not args.no_augment
    train_ds = dataset_type(dataset, "train", **train_options)
    validation_split = "train" if args.validation_on_train else "validation"
    validation_options: dict[str, object] = {
        "seed": args.seed, "shuffle": False,
        "sample_limit": args.validation_sample_limit,
    }
    if not truth_history:
        validation_options["augment"] = False
    validation_ds = dataset_type(dataset, validation_split, **validation_options)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, **loader_options)
    validation_loader = DataLoader(validation_ds, **loader_options)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    input_features = 3 if truth_history else 5
    model_a = AnchoredDeltaPredictor(
        input_features=input_features, channels=args.channels, dropout=args.dropout
    )
    model_b = RigidMotionPredictor(
        geometry, input_features=input_features, channels=args.channels, dropout=args.dropout
    )
    model_b.encoder.load_state_dict(model_a.encoder.state_dict())
    shared_encoder_initial_sha256 = _state_dict_sha256(model_a.encoder.state_dict())
    if shared_encoder_initial_sha256 != _state_dict_sha256(model_b.encoder.state_dict()):
        raise RuntimeError("A/B encoders do not share the exact initialization")
    models: dict[str, torch.nn.Module] = {
        "A_anchored_direct": model_a.to(device),
        "B_rigid_latent": model_b.to(device),
    }
    optimizers = {
        label: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for label, model in models.items()
    }
    schedulers = {
        label: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max(1, args.epochs), eta_min=args.lr * 0.02
        )
        for label, optimizer in optimizers.items()
    }
    scalers = {
        label: torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and args.amp == "float16"
        )
        for label in models
    }
    provenance: dict[str, object] = {
        "schema_version": "stage3-physical-ab-training-run-v1",
        "dataset": str(dataset),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "geometry_template_sha256": geometry_sha256,
        "geometry_hash": geometry_payload.get("geometry_hash"),
        "test_accessed": False,
        "input_source": "exact_truth_history" if truth_history else "pnp_observation_history",
        "validation_split": validation_split,
        "config": vars(args),
        "initialization": "scratch; identical copied temporal-set encoder",
        "shared_encoder_initial_sha256": shared_encoder_initial_sha256,
        "source_sha256": _source_hashes(),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "amp": args.amp,
        },
        **_git_state(),
    }
    _write_json(output / "run_manifest.json", provenance)

    initial_validation = _validate_pair(models, validation_loader, device, args)
    history: list[dict[str, object]] = [{
        "epoch": 0,
        "validation": initial_validation,
        "selection_tuple": {
            label: _selection_tuple(value) for label, value in initial_validation.items()
        },
    }]
    history_path = output / "stage3-physical-ab-history.json"
    _write_json(history_path, history)
    best: dict[str, tuple[float, ...] | None] = {label: None for label in models}
    stale = {label: 0 for label in models}
    best_paths = {label: output / f"stage3-physical-{label}-seed{args.seed}-best.pt" for label in models}
    started = time.monotonic()
    stop_reason = "epochs_completed"
    epochs_completed = 0
    validation = initial_validation
    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        train_metrics = _train_epoch(
            models, train_loader, optimizers, scalers, device, args
        )
        validation = _validate_pair(models, validation_loader, device, args)
        for scheduler in schedulers.values():
            scheduler.step()
        record: dict[str, object] = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": validation,
            "selection_tuple": {
                label: _selection_tuple(value) for label, value in validation.items()
            },
            "lr": {label: scheduler.get_last_lr()[0] for label, scheduler in schedulers.items()},
        }
        history.append(record)
        _write_json(history_path, history)
        compact = {
            "epoch": epoch + 1,
            "train_loss": {
                label: round(float(value["loss"]), 7)
                for label, value in train_metrics.items()
            },
            "selection_tuple": record["selection_tuple"],
        }
        print(json.dumps(compact, sort_keys=True), flush=True)
        epochs_completed = epoch + 1
        for label, model in models.items():
            score = _selection_tuple(validation[label])
            if not all(np.isfinite(score)):
                raise FloatingPointError(f"{label} produced non-finite validation metrics")
            if best[label] is None or score < best[label]:
                best[label] = score
                stale[label] = 0
                _checkpoint(
                    best_paths[label], model, label, epoch + 1,
                    validation[label], provenance, "best",
                )
            else:
                stale[label] += 1
        if all(value >= args.patience for value in stale.values()):
            stop_reason = "both_models_early_stopping"
            break
        if args.max_wall_minutes > 0 and time.monotonic() - started >= args.max_wall_minutes * 60:
            stop_reason = "wall_time_limit"
            break

    for label, model in models.items():
        _checkpoint(
            output / f"stage3-physical-{label}-seed{args.seed}-last.pt",
            model, label, epochs_completed, validation[label], provenance, "last",
            optimizers[label], schedulers[label], scalers[label],
        )
    final = {
        **provenance,
        "status": "complete",
        "stop_reason": stop_reason,
        "epochs_completed": epochs_completed,
        "best": {
            label: {
                "path": path.name,
                "sha256": _sha256(path),
                "selection_tuple": best[label],
            }
            for label, path in best_paths.items()
        },
    }
    _write_json(output / "run_manifest.json", final)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--huber-beta-m", type=float, default=0.05)
    parser.add_argument("--state-weight", type=float, default=2.0)
    parser.add_argument("--motion-weight", type=float, default=1.0)
    parser.add_argument("--absolute-weight", type=float, default=1.0)
    parser.add_argument("--rigidity-weight", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--device", default="")
    parser.add_argument("--amp", choices=("off", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--train-sample-limit", type=int, default=0)
    parser.add_argument("--validation-sample-limit", type=int, default=0)
    parser.add_argument("--validation-on-train", action="store_true")
    parser.add_argument("--max-wall-minutes", type=float, default=0.0)
    args = parser.parse_args()
    positive = (
        args.epochs, args.patience, args.batch_size, args.lr, args.weight_decay,
        args.channels, args.huber_beta_m, args.state_weight, args.motion_weight,
        args.absolute_weight, args.grad_clip,
    )
    if any(value <= 0 for value in positive) or args.rigidity_weight < 0:
        parser.error("invalid non-positive physical A/B training argument")
    if args.channels % 4:
        parser.error("channels must be divisible by four")
    if args.train_sample_limit < 0 or args.validation_sample_limit < 0:
        parser.error("sample limits cannot be negative")
    if args.validation_on_train and (
        args.train_sample_limit <= 0 or args.validation_sample_limit <= 0
    ):
        parser.error("validation-on-train is only allowed for a bounded tiny-fit run")
    print(train(args))


if __name__ == "__main__":
    main()
