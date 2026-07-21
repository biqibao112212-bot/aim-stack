"""Train paired causal clean-physics A/B models with one shared architecture.

A uses fixed-slot q0/future/motion supervision. B uses the same base objective
and adds same-segment history reconstruction plus shared constant-motion
consistency. Test is neither constructed nor opened.
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

from .causal_physical_dataset import CausalPhysicalShardDataset
from .physical_loss import (
    causal_physical_base_loss,
    causal_physical_history_regularizers,
)
from .physical_metrics import fixed_slot_physical_batch_errors, summary
from .physical_model import RigidPoseDeltaPredictor


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


def _capture_rng(device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    )


def _restore_rng(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def _to_device(raw: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in raw.items()}


def _train_one(
    label: str, model: RigidPoseDeltaPredictor, batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
    device: torch.device, radius_m: float, args: argparse.Namespace,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = device.type == "cuda" and args.amp != "off"
    amp_dtype = torch.float16 if args.amp == "float16" else torch.bfloat16
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        future = model(
            batch["obs"], batch["obs_mask"], batch["event_mask"],
            batch["event_time_s"], batch["tau"],
        )
        base, base_metrics = causal_physical_base_loss(
            future, batch["future_position"], batch["tau"], batch["rule_query"],
            huber_beta_m=args.huber_beta_m,
        )
        total = base
        history_value = torch.zeros((), device=device)
        shared_value = torch.zeros((), device=device)
        active_value = torch.zeros((), device=device)
        if label == "B_history_shared_rigid":
            history_prediction = model(
                batch["obs"], batch["obs_mask"], batch["event_mask"],
                batch["event_time_s"], batch["event_time_s"],
            )
            _, extra = causal_physical_history_regularizers(
                future, history_prediction, batch["history_position_m"],
                batch["obs_mask"], batch["event_mask"], batch["event_time_s"],
                batch["tau"], batch["rule_query"],
                geometry_rms_radius_m=radius_m,
                huber_beta_m=args.huber_beta_m,
                constant_history_s=args.constant_history_s,
                constant_history_events=args.constant_history_events,
            )
            history_value = extra["history"]
            shared_value = extra["shared"]
            active_value = extra["shared_active_fraction"]
            total = (
                base + args.history_weight * history_value
                + args.shared_weight * shared_value
            )
    if float(total.detach().cpu()) <= args.minimum_update_loss:
        return {
            "objective": float(total.detach().cpu()),
            "q0": float(base_metrics["q0"].detach().cpu()),
            "absolute": float(base_metrics["absolute"].detach().cpu()),
            "motion": float(base_metrics["motion"].detach().cpu()),
            "history": float(history_value.detach().cpu()),
            "shared": float(shared_value.detach().cpu()),
            "shared_active_fraction": float(active_value.detach().cpu()),
            "gradient_norm": 0.0,
        }
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError(f"non-finite gradient norm for {label}")
    scaler.step(optimizer)
    scaler.update()
    return {
        "objective": float(total.detach().cpu()),
        "q0": float(base_metrics["q0"].detach().cpu()),
        "absolute": float(base_metrics["absolute"].detach().cpu()),
        "motion": float(base_metrics["motion"].detach().cpu()),
        "history": float(history_value.detach().cpu()),
        "shared": float(shared_value.detach().cpu()),
        "shared_active_fraction": float(active_value.detach().cpu()),
        "gradient_norm": float(gradient_norm.detach().cpu()),
    }


def _train_epoch(
    models: dict[str, RigidPoseDeltaPredictor], loader: DataLoader,
    optimizers: dict[str, torch.optim.Optimizer],
    scalers: dict[str, torch.amp.GradScaler], device: torch.device,
    radius_m: float, args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    for model in models.values():
        model.train()
    totals = {label: {} for label in models}
    count = 0
    for raw in loader:
        batch = _to_device(raw, device)
        batch_count = int(batch["obs"].shape[0])
        shared_rng = _capture_rng(device)
        after_a = None
        for index, (label, model) in enumerate(models.items()):
            if index:
                _restore_rng(shared_rng)
            values = _train_one(
                label, model, batch, optimizers[label], scalers[label],
                device, radius_m, args,
            )
            if index == 0:
                after_a = _capture_rng(device)
            for key, value in values.items():
                totals[label][key] = totals[label].get(key, 0.0) + value * batch_count
        if after_a is not None:
            _restore_rng(after_a)
        count += batch_count
    return {
        label: {key: value / max(count, 1) for key, value in values.items()}
        for label, values in totals.items()
    }


def _validate(
    model: RigidPoseDeltaPredictor, loader: DataLoader,
    device: torch.device, args: argparse.Namespace,
) -> dict[str, object]:
    model.eval()
    arrays: dict[str, list[np.ndarray]] = {}
    tau_parts: list[np.ndarray] = []
    rule_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    motion_parts: list[np.ndarray] = []
    amp_enabled = device.type == "cuda" and args.amp != "off"
    amp_dtype = torch.float16 if args.amp == "float16" else torch.bfloat16
    with torch.no_grad():
        for raw in loader:
            batch = _to_device(raw, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                prediction = model(
                    batch["obs"], batch["obs_mask"], batch["event_mask"],
                    batch["event_time_s"], batch["tau"],
                )["position_mean"]
            errors = fixed_slot_physical_batch_errors(
                prediction.float(), batch["future_position"]
            )
            for name, value in errors.items():
                arrays.setdefault(name, []).append(value.detach().cpu().numpy())
            tau_parts.append(batch["tau"].detach().cpu().numpy())
            rule_parts.append(batch["rule_query"].detach().cpu().numpy())
            distance_parts.append(batch["distance_m"].detach().cpu().numpy())
            motion_parts.append(batch["motion_class"].detach().cpu().numpy())
    merged = {name: np.concatenate(parts, axis=0) for name, parts in arrays.items()}
    tau = np.concatenate(tau_parts, axis=0)
    rule = np.concatenate(rule_parts, axis=0).astype(np.bool_, copy=False)
    distance = np.concatenate(distance_parts, axis=0)
    motion_class = np.concatenate(motion_parts, axis=0)

    queries: list[dict[str, object]] = []
    for query in range(tau.shape[1]):
        active = rule[:, query]
        queries.append({
            "query_index": query,
            "tau_s": {"median": float(np.median(tau[:, query]))},
            "absolute": summary(merged["absolute_pg_m"][:, query]),
            "motion_delta": summary(merged["motion_delta_m"][:, query]),
            "rule": {
                "absolute": summary(merged["absolute_pg_m"][active, query]),
                "motion_delta": summary(merged["motion_delta_m"][active, query]),
            },
            "future_event": {
                "absolute": summary(merged["absolute_pg_m"][~active, query]),
                "motion_delta": summary(merged["motion_delta_m"][~active, query]),
            },
        })

    def stratum(mask: np.ndarray) -> dict[str, object]:
        return {
            "sample_count": int(mask.sum()),
            "state_q0": summary(merged["state_q0_m"][mask]),
            "q3_rule_motion": summary(
                merged["motion_delta_m"][mask & rule[:, 3], 3]
            ),
        }

    motion_labels = {0: "stationary", 1: "linear", 2: "spin", 3: "linear_and_spin"}
    return {
        "sample_count": int(tau.shape[0]),
        "slot_policy": "fixed-causal-slots; no permutation search",
        "state_q0": summary(merged["state_q0_m"]),
        "rigid_residual": summary(merged["rigid_residual_m"].reshape(-1)),
        "queries": queries,
        "rule_query_fraction": float(rule.mean()),
        "strata": {
            "motion_class": {
                label: stratum(motion_class == index)
                for index, label in motion_labels.items()
            },
            "distance": {
                "near_lt_3m": stratum(distance < 3.0),
                "mid_3_to_5m": stratum((distance >= 3.0) & (distance < 5.0)),
                "far_ge_5m": stratum(distance >= 5.0),
            },
        },
    }


def _selection_tuple(metrics: dict[str, object]) -> tuple[float, ...]:
    queries = metrics["queries"]  # type: ignore[assignment]
    headline = queries[1:4]
    return (
        max(float(item["rule"]["motion_delta"]["p95_m"]) for item in headline),
        float(metrics["state_q0"]["p95_m"]),  # type: ignore[index]
        max(float(item["rule"]["absolute"]["p95_m"]) for item in headline),
        max(float(item["rule"]["motion_delta"]["median_m"]) for item in headline),
    )


def _checkpoint(
    path: Path, model: RigidPoseDeltaPredictor, label: str, epoch: int,
    metrics: dict[str, object], provenance: dict[str, object], role: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    payload: dict[str, object] = {
        "model": model.state_dict(), "model_class": model.__class__.__name__,
        "model_config": model.config(), "label": label, "epoch": epoch,
        "checkpoint_role": role, "validation": metrics,
        "selection_tuple": _selection_tuple(metrics), "provenance": provenance,
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
    if manifest.get("schema_version") != "stage3-causal-physical-v1":
        raise ValueError("causal physical A/B requires stage3-causal-physical-v1")
    if not bool(manifest.get("qualification_passed", False)):
        raise ValueError("causal physical A/B requires a qualified dataset")
    if bool(manifest.get("test_accessed", True)):
        raise ValueError("causal physical A/B refuses a test-accessed dataset")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training output: {output}")
    output.mkdir(parents=True)
    geometry, geometry_payload, geometry_sha256 = _load_geometry(dataset)
    radius_m = float(torch.sqrt(torch.mean(torch.sum(geometry[:, :2].square(), dim=1))))

    train_ds = CausalPhysicalShardDataset(
        dataset, "train", seed=args.seed, shuffle=not args.validation_on_train,
        sample_limit=args.train_sample_limit,
    )
    validation_split = "train" if args.validation_on_train else "validation"
    validation_ds = CausalPhysicalShardDataset(
        dataset, validation_split, seed=args.seed, shuffle=False,
        sample_limit=args.validation_sample_limit,
    )
    loader_options = {
        "batch_size": args.batch_size, "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, **loader_options)
    validation_loader = DataLoader(validation_ds, **loader_options)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_a = RigidPoseDeltaPredictor(
        geometry, input_features=5, channels=args.channels, dropout=args.dropout,
        position_mean=torch.from_numpy(train_ds.mean),
        position_std=torch.from_numpy(train_ds.std),
    )
    model_b = RigidPoseDeltaPredictor(
        geometry, input_features=5, channels=args.channels, dropout=args.dropout,
        position_mean=torch.from_numpy(train_ds.mean),
        position_std=torch.from_numpy(train_ds.std),
    )
    model_b.load_state_dict(model_a.state_dict())
    initial_sha256 = _state_dict_sha256(model_a.state_dict())
    if initial_sha256 != _state_dict_sha256(model_b.state_dict()):
        raise RuntimeError("paired A/B models do not share the full initialization")
    models = {
        "A_future_rigid": model_a.to(device),
        "B_history_shared_rigid": model_b.to(device),
    }
    optimizers = {
        label: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for label, model in models.items()
    }
    schedulers = {
        label: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max(1, args.epochs), eta_min=args.lr * 0.02
        ) for label, optimizer in optimizers.items()
    }
    scalers = {
        label: torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and args.amp == "float16"
        ) for label in models
    }
    source_names = (
        "train_causal_physical_ab.py", "causal_physical_dataset.py",
        "physical_model.py", "physical_loss.py", "physical_metrics.py",
    )
    source_root = Path(__file__).resolve().parent
    provenance: dict[str, object] = {
        "schema_version": "stage3-causal-physical-ab-run-v1",
        "dataset": str(dataset), "dataset_manifest_sha256": _sha256(manifest_path),
        "geometry_template_sha256": geometry_sha256,
        "geometry_hash": geometry_payload.get("geometry_hash"),
        "test_accessed": False, "validation_split": validation_split,
        "slot_policy": "fixed causal cyclic slots; no permutation search",
        "input_allowlist": ["normalized exact xyz", "cyclic slot sin/cos", "event mask", "real event time", "tau", "train-only normalization"],
        "forbidden_predictor_inputs": ["center", "velocity", "yaw", "yaw_rate", "motion_class", "rule_query", "future truth"],
        "paired_initial_state_sha256": initial_sha256,
        "objectives": {
            "A_future_rigid": "2*q0 + absolute + motion_delta",
            "B_history_shared_rigid": "A + history_weight*history + shared_weight*constant_motion",
        },
        "q0_anchor": "causal last-4 fixed-slot rigid-pose least squares plus learned residual; no future label",
        "config": vars(args),
        "source_sha256": {name: _sha256(source_root / name) for name in source_names},
        "environment": {
            "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "device": str(device), "amp": args.amp,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        **_git_state(),
    }
    _write_json(output / "run_manifest.json", provenance)

    validation = {label: _validate(model, validation_loader, device, args) for label, model in models.items()}
    history: list[dict[str, object]] = [{
        "epoch": 0, "validation": validation,
        "selection_tuple": {label: _selection_tuple(value) for label, value in validation.items()},
    }]
    history_path = output / "stage3-causal-physical-ab-history.json"
    _write_json(history_path, history)
    best = {label: _selection_tuple(value) for label, value in validation.items()}
    best_paths = {
        label: output / f"stage3-causal-physical-{label}-seed{args.seed}-best.pt"
        for label in models
    }
    for label, model in models.items():
        _checkpoint(best_paths[label], model, label, 0, validation[label], provenance, "best")
    stale = {label: 0 for label in models}
    stop_reason = "epoch_limit"
    started = time.monotonic()
    epochs_completed = 0
    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        train_metrics = _train_epoch(
            models, train_loader, optimizers, scalers, device, radius_m, args
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
            "selection_tuple": {label: _selection_tuple(value) for label, value in validation.items()},
            "lr": {label: optimizer.param_groups[0]["lr"] for label, optimizer in optimizers.items()},
        }
        history.append(record)
        _write_json(history_path, history)
        print(json.dumps(record, sort_keys=True), flush=True)
        for label, model in models.items():
            selection = _selection_tuple(validation[label])
            if selection < best[label]:
                best[label] = selection
                stale[label] = 0
                _checkpoint(
                    best_paths[label], model, label, epoch, validation[label], provenance,
                    "best", optimizers[label], schedulers[label], scalers[label],
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
            output / f"stage3-causal-physical-{label}-seed{args.seed}-last.pt",
            model, label, epochs_completed, validation[label], provenance, "last",
            optimizers[label], schedulers[label], scalers[label],
        )
    final = {
        **provenance, "status": "complete", "stop_reason": stop_reason,
        "epochs_completed": epochs_completed,
        "best": {
            label: {"path": path.name, "sha256": _sha256(path), "selection_tuple": best[label]}
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
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--huber-beta-m", type=float, default=0.005)
    parser.add_argument("--history-weight", type=float, default=0.5)
    parser.add_argument("--shared-weight", type=float, default=0.25)
    parser.add_argument("--constant-history-s", type=float, default=0.2)
    parser.add_argument("--constant-history-events", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--minimum-update-loss", type=float, default=1e-10,
        help="skip numerically exact batches so Adam cannot push a perfect zero-motion prior away",
    )
    parser.add_argument("--device", default="")
    parser.add_argument("--amp", choices=("off", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--train-sample-limit", type=int, default=0)
    parser.add_argument("--validation-sample-limit", type=int, default=0)
    parser.add_argument("--validation-on-train", action="store_true")
    parser.add_argument("--max-wall-minutes", type=float, default=0.0)
    args = parser.parse_args()
    positive = (
        args.epochs, args.patience, args.batch_size, args.lr, args.weight_decay,
        args.channels, args.huber_beta_m, args.history_weight, args.shared_weight,
        args.constant_history_s, args.grad_clip,
    )
    if any(value <= 0 for value in positive):
        parser.error("causal physical A/B arguments must be positive")
    if args.channels % 4:
        parser.error("channels must be divisible by four")
    if args.train_sample_limit < 0 or args.validation_sample_limit < 0:
        parser.error("sample limits cannot be negative")
    if args.minimum_update_loss < 0:
        parser.error("minimum-update-loss cannot be negative")
    if args.constant_history_events < 2:
        parser.error("constant-history-events must be at least two")
    if args.validation_on_train and (
        args.train_sample_limit <= 0 or args.validation_sample_limit <= 0
    ):
        parser.error("validation-on-train requires bounded sample limits")
    print(train(args))


if __name__ == "__main__":
    main()
