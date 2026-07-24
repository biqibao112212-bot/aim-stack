"""Train the independent clean-physics cyclic q0 state restorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cyclic_state_loss import cyclic_state_loss
from .cyclic_state_model import CyclicStateRestorer, current_track_support
from .cyclic_track_dataset import CyclicTrackPhysicalDataset
from .cyclic_track_model import ROUTE_NAMES
from .train_causal_physical_ab import (
    _git_state,
    _seed,
    _sha256,
    _state_dict_sha256,
    _to_device,
    _write_json,
)


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0, "mean_m": None, "median_m": None,
            "p90_m": None, "p95_m": None, "p99_m": None, "max_m": None,
        }
    return {
        "count": int(values.size),
        "mean_m": float(values.mean()),
        "median_m": float(np.median(values)),
        "p90_m": float(np.quantile(values, 0.90)),
        "p95_m": float(np.quantile(values, 0.95)),
        "p99_m": float(np.quantile(values, 0.99)),
        "max_m": float(values.max()),
    }


def _p95(item: object) -> float:
    if not isinstance(item, dict) or item.get("p95_m") is None:
        return float("inf")
    return float(item["p95_m"])


def _selection_tuple(metrics: dict[str, object]) -> tuple[float, ...]:
    by_motion = metrics["by_motion"]  # type: ignore[index]
    rotation = by_motion["rotation"]["warm_adjacent"]  # type: ignore[index]
    combined = by_motion["combined"]["warm_adjacent"]  # type: ignore[index]
    visible = metrics["current_visible"]  # type: ignore[index]
    calibration = metrics["warm_adjacent_sigma_absolute_error"]  # type: ignore[index]
    edge = metrics["dynamic_relevant_edge"]  # type: ignore[index]
    return (
        max(_p95(rotation), _p95(combined)),
        _p95(edge),
        _p95(combined),
        _p95(rotation),
        _p95(calibration),
        _p95(visible),
    )


def _equivariance_audit(
    model: CyclicStateRestorer, batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    reference = model(
        batch["obs"], batch["obs_mask"], batch["primary_mask"],
        batch["event_mask"], batch["event_time_s"], batch["switch_step"],
    )
    q0_max = sigma_max = confidence_max = 0.0
    for shift in (1, 2, 3):
        shifted = dict(batch)
        for name in ("obs", "obs_mask", "primary_mask"):
            shifted[name] = torch.roll(batch[name], shifts=shift, dims=2)
        output = model(
            shifted["obs"], shifted["obs_mask"], shifted["primary_mask"],
            shifted["event_mask"], shifted["event_time_s"],
            shifted["switch_step"],
        )
        q0_max = max(q0_max, float((
            output["q0_m"] - torch.roll(reference["q0_m"], shift, dims=1)
        ).abs().max().detach().cpu()))
        sigma_max = max(sigma_max, float((
            output["q0_sigma_m"]
            - torch.roll(reference["q0_sigma_m"], shift, dims=1)
        ).abs().max().detach().cpu()))
        confidence_max = max(confidence_max, float((
            output["confidence"]
            - torch.roll(reference["confidence"], shift, dims=1)
        ).abs().max().detach().cpu()))
        if not torch.equal(
            output["q0_valid"], torch.roll(reference["q0_valid"], shift, dims=1)
        ):
            raise RuntimeError("q0 validity is not C4 roll-equivariant")
    return {
        "q0_max_abs_m": q0_max,
        "sigma_max_abs_m": sigma_max,
        "confidence_max_abs": confidence_max,
    }


def _validate(
    model: CyclicStateRestorer,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    model.eval()
    parts: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "error", "sigma", "motion", "visible", "warm", "cold",
            "self_warm", "edge_warm", "adjacent", "clockwise",
            "counterclockwise", "age", "edge_error", "pair_seen",
            "relevant_edge", "q0_observed",
        )
    }
    equivariance: dict[str, float] | None = None
    amp_enabled = device.type == "cuda" and args.amp != "off"
    amp_dtype = torch.float16 if args.amp == "float16" else torch.bfloat16
    with torch.no_grad():
        for raw in loader:
            batch = _to_device(raw, device)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled,
            ):
                output = model(
                    batch["obs"], batch["obs_mask"], batch["primary_mask"],
                    batch["event_mask"], batch["event_time_s"],
                    batch["switch_step"],
                )
            target = batch["future_position"][:, 0].float()
            error = torch.linalg.vector_norm(output["q0_m"].float() - target, dim=-1)
            target_edge = torch.roll(target, shifts=-1, dims=1) - target
            edge_error = torch.linalg.vector_norm(
                output["edge0_m"].float() - target_edge, dim=-1
            )
            values = {
                "error": error,
                "sigma": output["q0_sigma_m"].squeeze(-1).float(),
                "motion": batch["motion_class"],
                "visible": output["current_visible"],
                "q0_observed": output["q0_observed"],
                "warm": output["warm_hidden"],
                "cold": output["cold"],
                "self_warm": output["self_warm"],
                "edge_warm": output["edge_warm"],
                "adjacent": output["adjacent"],
                "clockwise": output["clockwise"],
                "counterclockwise": output["counterclockwise"],
                "age": output["age_s"].float(),
                "edge_error": edge_error,
                "pair_seen": output["pair_seen"],
                "relevant_edge": output["relevant_edge"],
            }
            for name, value in values.items():
                parts[name].append(value.detach().cpu().numpy())
            if bool(output["q0_valid"][output["cold"]].any()):
                raise RuntimeError("cold tracks must remain invalid")
            if not bool(output["q0_valid"][output["current_visible"]].all()):
                raise RuntimeError("every current visible track must be valid")
            if equivariance is None:
                audit = {
                    name: value[: min(8, value.shape[0])]
                    for name, value in batch.items()
                    if isinstance(value, torch.Tensor)
                }
                with torch.autocast(device_type=device.type, enabled=False):
                    equivariance = _equivariance_audit(model, audit)

    merged = {name: np.concatenate(values) for name, values in parts.items()}
    error = merged["error"]
    sigma = merged["sigma"]
    motion = merged["motion"].astype(np.int64)
    visible = merged["visible"].astype(np.bool_)
    q0_observed = merged["q0_observed"].astype(np.bool_)
    warm = merged["warm"].astype(np.bool_)
    cold = merged["cold"].astype(np.bool_)
    self_warm = merged["self_warm"].astype(np.bool_)
    edge_warm = merged["edge_warm"].astype(np.bool_)
    adjacent = merged["adjacent"].astype(np.bool_)
    clockwise = merged["clockwise"].astype(np.bool_)
    counterclockwise = merged["counterclockwise"].astype(np.bool_)
    age = merged["age"]
    edge_error = merged["edge_error"]
    pair_seen = merged["pair_seen"].astype(np.bool_)
    relevant_edge = merged["relevant_edge"].astype(np.bool_)
    dynamic = (motion == 2) | (motion == 3)
    selected = warm & adjacent & dynamic[:, None]
    recent = age <= args.recent_age_s

    by_motion: dict[str, object] = {}
    for route, name in enumerate(ROUTE_NAMES):
        group = motion == route
        warm_adjacent = group[:, None] & warm & adjacent
        by_motion[name] = {
            "current_visible": _summary(error[group[:, None] & visible]),
            "warm_adjacent": (
                _summary(error[warm_adjacent]) if route in (2, 3)
                else {"count": int(warm_adjacent.sum()), "excluded_by_contract": True}
            ),
            "warm_adjacent_recent": (
                _summary(error[warm_adjacent & recent]) if route in (2, 3)
                else {"count": int((warm_adjacent & recent).sum()),
                      "excluded_by_contract": True}
            ),
            "warm_adjacent_stale": (
                _summary(error[warm_adjacent & ~recent]) if route in (2, 3)
                else {"count": int((warm_adjacent & ~recent).sum()),
                      "excluded_by_contract": True}
            ),
            "self_warm_adjacent": (
                _summary(error[group[:, None] & self_warm]) if route in (2, 3)
                else {"count": int((group[:, None] & self_warm).sum()),
                      "excluded_by_contract": True}
            ),
            "edge_warm_adjacent": (
                _summary(error[group[:, None] & edge_warm]) if route in (2, 3)
                else {"count": int((group[:, None] & edge_warm).sum()),
                      "excluded_by_contract": True}
            ),
            "cold_adjacent_count": int((group[:, None] & cold & adjacent).sum()),
        }

    result: dict[str, object] = {
        "current_visible": _summary(error[visible]),
        "q0_observed_identity": _summary(error[q0_observed]),
        "visible_propagated_to_q0": _summary(error[visible & ~q0_observed]),
        "dynamic_warm_adjacent": _summary(error[selected]),
        "clockwise_dynamic_warm_hidden": _summary(
            error[warm & clockwise & dynamic[:, None]]
        ),
        "counterclockwise_dynamic_warm_hidden": _summary(
            error[warm & counterclockwise & dynamic[:, None]]
        ),
        "warm_adjacent_sigma_absolute_error": _summary(
            np.abs(sigma[selected] - error[selected])
        ),
        "dynamic_relevant_edge": _summary(
            edge_error[pair_seen & relevant_edge & dynamic[:, None]]
        ),
        "by_motion": by_motion,
        "support": {
            "current_visible_count": int(visible.sum()),
            "q0_observed_count": int(q0_observed.sum()),
            "visible_propagated_to_q0_count": int((visible & ~q0_observed).sum()),
            "warm_hidden_count": int(warm.sum()),
            "self_warm_adjacent_count": int(self_warm.sum()),
            "edge_warm_adjacent_count": int(edge_warm.sum()),
            "dynamic_warm_adjacent_count": int(selected.sum()),
            "cold_count": int(cold.sum()),
            "cold_position_excluded_from_final_metrics": True,
            "stationary_translation_hidden_excluded_from_final_metrics": True,
        },
        "equivariance": equivariance or {},
    }
    identity = result["q0_observed_identity"]
    if not isinstance(identity, dict) or identity.get("max_m") is None:
        raise RuntimeError("validation requires q0-observed identity support")
    if float(identity["max_m"]) > 2e-6:
        raise RuntimeError("q0-observed identity bypass exceeded 2e-6 m")
    audit = result["equivariance"]
    if not isinstance(audit, dict) or any(
        float(audit.get(name, float("inf"))) > 2e-6
        for name in ("q0_max_abs_m", "sigma_max_abs_m", "confidence_max_abs")
    ):
        raise RuntimeError("C4 equivariance audit exceeded 2e-6")
    selection = _selection_tuple(result)
    if not all(np.isfinite(value) for value in selection):
        raise RuntimeError("validation selection requires finite non-empty support")
    return result


def _audit_dataset(
    dataset: CyclicTrackPhysicalDataset,
    batch_size: int,
    history_events: int,
) -> dict[str, object]:
    class_count = np.zeros(4, dtype=np.int64)
    visible_count = np.zeros(4, dtype=np.int64)
    warm_adjacent_count = np.zeros(4, dtype=np.int64)
    cold_adjacent_count = np.zeros(4, dtype=np.int64)
    sample_count = 0
    for batch in DataLoader(dataset, batch_size=batch_size, num_workers=0):
        support = current_track_support(
            batch["obs_mask"], batch["primary_mask"], batch["event_mask"],
            batch["event_time_s"], history_events=history_events,
        )
        labels = batch["motion_class"]
        for route in range(4):
            group = labels == route
            class_count[route] += int(group.sum())
            visible_count[route] += int(
                (group[:, None] & support["current_visible"]).sum()
            )
            warm_adjacent_count[route] += int(
                (group[:, None] & support["warm_hidden"] & support["adjacent"]).sum()
            )
            cold_adjacent_count[route] += int(
                (group[:, None] & support["cold"] & support["adjacent"]).sum()
            )
        sample_count += int(labels.shape[0])
    if np.any(class_count == 0) or np.any(visible_count == 0):
        raise ValueError("all motion classes require current visible supervision")
    if np.any(warm_adjacent_count[2:] == 0):
        raise ValueError("rotation and combined require warm adjacent supervision")
    return {
        "sample_count": sample_count,
        "motion_class_count": dict(zip(ROUTE_NAMES, class_count.tolist())),
        "current_visible_track_count": dict(zip(
            ROUTE_NAMES, visible_count.tolist()
        )),
        "warm_adjacent_track_count": dict(zip(
            ROUTE_NAMES, warm_adjacent_count.tolist()
        )),
        "cold_adjacent_track_count": dict(zip(
            ROUTE_NAMES, cold_adjacent_count.tolist()
        )),
        "cold_position_excluded": True,
        "nonrotating_hidden_position_excluded": True,
        "virtual_contract": dataset.virtual_contract,
    }


def _train_epoch(
    model: CyclicStateRestorer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    amp_enabled = device.type == "cuda" and args.amp != "off"
    amp_dtype = torch.float16 if args.amp == "float16" else torch.bfloat16
    for raw in loader:
        batch = _to_device(raw, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled,
        ):
            output = model(
                batch["obs"], batch["obs_mask"], batch["primary_mask"],
                batch["event_mask"], batch["event_time_s"],
                batch["switch_step"],
            )
            total, loss_parts = cyclic_state_loss(
                output, batch["future_position"][:, 0], batch["motion_class"],
                huber_beta_m=args.huber_beta_m,
                sigma_weight=args.sigma_weight,
                edge_weight=args.edge_weight,
            )
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite cyclic-state objective")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite cyclic-state gradient")
        scaler.step(optimizer)
        scaler.update()
        batch_count = int(batch["obs"].shape[0])
        values = {
            "objective": float(total.detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            **{
                name: float(value.detach().cpu())
                for name, value in loss_parts.items()
            },
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise FloatingPointError("non-finite cyclic-state train metric")
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value * batch_count
        count += batch_count
    return {name: value / max(count, 1) for name, value in totals.items()}


def _checkpoint(
    path: Path,
    model: CyclicStateRestorer,
    epoch: int,
    metrics: dict[str, object],
    provenance: dict[str, object],
    role: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    payload: dict[str, object] = {
        "model": model.state_dict(),
        "model_class": model.__class__.__name__,
        "model_config": model.config(),
        "label": "cyclic_q0_state_restorer",
        "epoch": epoch,
        "checkpoint_role": role,
        "validation": metrics,
        "selection_tuple": _selection_tuple(metrics),
        "provenance": provenance,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if next(model.parameters()).device.type == "cuda" else None
        ),
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
    git_state = _git_state()
    if bool(git_state["worktree_dirty"]) and not args.allow_dirty_worktree:
        raise ValueError("official cyclic-state training requires a clean worktree")
    dataset_path = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite cyclic-state output: {output}")

    train_ds = CyclicTrackPhysicalDataset(
        dataset_path, "train", seed=args.seed, shuffle=True,
        sample_limit=args.train_sample_limit,
        secondary_gap_ratio=args.secondary_gap_ratio,
        augment_cyclic_origin=True, augment_direction=True,
    )
    validation_ds = CyclicTrackPhysicalDataset(
        dataset_path, "validation", seed=args.seed,
        shuffle=args.validation_sample_limit > 0,
        sample_limit=args.validation_sample_limit,
        secondary_gap_ratio=args.secondary_gap_ratio,
    )
    qualification = {
        "train": _audit_dataset(
            train_ds, args.audit_batch_size, args.history_events
        ),
        "validation": _audit_dataset(
            validation_ds, args.audit_batch_size, args.history_events
        ),
    }
    train_ds.set_epoch(0)
    output.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CyclicStateRestorer(
        torch.from_numpy(train_ds.mean), torch.from_numpy(train_ds.std),
        channels=args.channels, dropout=args.dropout,
        history_events=args.history_events,
    ).to(device)
    initial_state_sha256 = _state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.2, end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(1, args.epochs - args.warmup_epochs),
        eta_min=args.lr * 0.02,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=(warmup, cosine),
        milestones=(args.warmup_epochs,),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.amp == "float16"
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, **loader_options)
    validation_loader = DataLoader(validation_ds, **loader_options)
    source_root = Path(__file__).resolve().parent
    source_names = (
        "train_cyclic_state_restorer.py", "cyclic_state_model.py",
        "cyclic_state_loss.py", "cyclic_track_dataset.py",
        "cyclic_track_model.py", "model.py",
    )
    provenance: dict[str, object] = {
        "schema_version": "stage3-cyclic-state-restorer-run-v1",
        "status": "training",
        "dataset": str(dataset_path),
        "dataset_manifest_sha256": train_ds.manifest_sha256,
        "dataset_qualification": qualification,
        "test_accessed": False,
        "input_allowlist": [
            "masked normalized physical xyz", "visibility mask",
            "primary mask", "event mask", "real causal event time",
            "tracker switch step",
        ],
        "forbidden_predictor_inputs": [
            "future truth", "motion class", "PnP", "slot identity feature",
            "center", "phase", "fixed radius", "fixed height",
            "geometry template", "query tau",
        ],
        "architecture_contract": {
            "purpose": "q0 current state recovery only",
            "track_labels": "temporary cyclic state handles only",
            "current_visible_update": "all one or two current visible tracks",
            "q0_observed_identity_bypass": True,
            "stale_visible_update": "learned propagation from last event to q0",
            "warm_hidden_update": "causal previously observed tracks",
            "cold_update": "invalid with zero confidence",
            "equivariance": "C4 roll-equivariant",
            "future_prediction": False,
        },
        "objective_contract": {
            "formula": (
                "group-balanced stale-visible all-class and rotation/combined "
                "self/edge-warm q0 SmoothL1 + "
                f"{args.edge_weight}*observed-adjacent-edge SmoothL1 + "
                f"{args.sigma_weight}*detached-error sigma calibration"
            ),
            "current_visible": (
                "exact identity only when observed at q0; otherwise propagated"
            ),
            "cold_position": "excluded",
            "stationary_translation_hidden_position": "excluded",
            "future_truth_role": "q0 label and validation only",
        },
        "selection_contract": (
            "worst rotation/combined warm-adjacent q0 P95, dynamic observed "
            "edge P95, then combined, rotation, sigma calibration and visible P95"
        ),
        "config": vars(args),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_state_sha256": initial_state_sha256,
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
    manifest_path = output / "run_manifest.json"
    history_path = output / "stage3-cyclic-state-restorer-history.json"
    initial_path = output / f"stage3-cyclic-state-restorer-seed{args.seed}-initial.pt"
    last_path = output / f"stage3-cyclic-state-restorer-seed{args.seed}-last.pt"

    start_epoch = 1
    history: list[dict[str, object]]
    best: tuple[float, ...] = (float("inf"),) * 6
    best_epoch = -1
    best_path: Path | None = None
    latest_path: Path | None = None
    if args.resume:
        if not manifest_path.exists() or not history_path.exists():
            raise ValueError("resume output is missing manifest/history")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") != "training" or bool(existing.get("test_accessed", True)):
            raise ValueError("only an unsealed interrupted training run may resume")
        if existing.get("git_commit") != provenance.get("git_commit"):
            raise ValueError("resume requires the original committed source")
        locked_config_keys = (
            "seed", "epochs", "batch_size", "lr", "warmup_epochs",
            "weight_decay", "dropout", "channels", "history_events",
            "secondary_gap_ratio", "huber_beta_m", "sigma_weight",
            "edge_weight", "recent_age_s", "validation_interval",
            "grad_clip", "amp", "train_sample_limit",
            "validation_sample_limit",
        )
        existing_config = existing.get("config", {})
        if not isinstance(existing_config, dict) or any(
            existing_config.get(name) != getattr(args, name)
            for name in locked_config_keys
        ):
            raise ValueError("resume training configuration differs from the run")
        for name in (
            "dataset", "dataset_manifest_sha256", "source_sha256",
            "architecture_contract", "objective_contract", "selection_contract",
        ):
            if existing.get(name) != provenance.get(name):
                raise ValueError(f"resume provenance differs for {name}")
        latest_record = existing.get("latest_checkpoint", {})
        if not isinstance(latest_record, dict) or not latest_record.get("path"):
            raise ValueError("resume manifest is missing latest_checkpoint")
        latest_path = output / str(latest_record["path"])
        if not latest_path.exists():
            raise ValueError("resume latest checkpoint does not exist")
        if latest_record.get("sha256") != _sha256(latest_path):
            raise ValueError("resume latest checkpoint hash mismatch")
        payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        embedded = payload.get("provenance", {})
        if not isinstance(embedded, dict) or any(
            embedded.get(name) != provenance.get(name)
            for name in (
                "git_commit", "dataset", "dataset_manifest_sha256",
                "source_sha256", "architecture_contract", "objective_contract",
            )
        ):
            raise ValueError("resume checkpoint embedded provenance mismatch")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        if "scaler" in payload:
            scaler.load_state_dict(payload["scaler"])
        if "torch_rng_state" not in payload:
            raise ValueError("resume checkpoint is missing RNG state")
        torch.set_rng_state(payload["torch_rng_state"])
        if payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        completed = int(payload["epoch"])
        start_epoch = completed + 1
        history = json.loads(history_path.read_text(encoding="utf-8"))
        history = [item for item in history if int(item["epoch"]) <= completed]
        trained = [
            item for item in history
            if int(item["epoch"]) > 0 and item.get("selection_tuple") is not None
        ]
        if trained:
            record = min(trained, key=lambda item: tuple(item["selection_tuple"]))
            best = tuple(record["selection_tuple"])
            best_epoch = int(record["epoch"])
            best_path = output / (
                f"stage3-cyclic-state-restorer-seed{args.seed}-"
                f"epoch{best_epoch:03d}.pt"
            )
            if not best_path.exists():
                raise ValueError("resume best checkpoint recorded in history is missing")
        provenance["resumed_from_epoch"] = completed
    else:
        _write_json(manifest_path, provenance)
        validation = _validate(model, validation_loader, device, args)
        history = [{
            "epoch": 0,
            "validation": validation,
            "selection_tuple": _selection_tuple(validation),
        }]
        _write_json(history_path, history)
        _checkpoint(initial_path, model, 0, validation, provenance, "initial")

    milestones = {20, 50, 100, args.epochs}
    validation = history[-1]["validation"] if "validation" in history[-1] else {}
    started = time.monotonic()
    epochs_completed = start_epoch - 1
    stop_reason = "epoch_limit"
    for epoch in range(start_epoch, args.epochs + 1):
        train_ds.set_epoch(epoch)
        train_metrics = _train_epoch(
            model, train_loader, optimizer, scaler, device, args
        )
        scheduler.step()
        validate_now = (
            epoch == 1 or epoch % args.validation_interval == 0
            or epoch in milestones or epoch == args.epochs
        )
        record: dict[str, object] = {
            "epoch": epoch,
            "train": train_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if validate_now:
            validation = _validate(model, validation_loader, device, args)
            selection = _selection_tuple(validation)
            record["validation"] = validation
            record["selection_tuple"] = selection
            epoch_path = output / (
                f"stage3-cyclic-state-restorer-seed{args.seed}-epoch{epoch:03d}.pt"
            )
            if epoch_path.exists():
                raise FileExistsError(f"refusing to overwrite checkpoint: {epoch_path}")
            _checkpoint(
                epoch_path, model, epoch, validation, provenance, "validation",
                optimizer, scheduler, scaler,
            )
            latest_path = epoch_path
            if selection < best:
                best = selection
                best_epoch = epoch
                best_path = epoch_path
        history.append(record)
        _write_json(history_path, history)
        if validate_now and latest_path is not None:
            _write_json(manifest_path, {
                **provenance,
                "status": "training",
                "epochs_completed": epoch,
                "latest_checkpoint": {
                    "path": latest_path.name,
                    "sha256": _sha256(latest_path),
                },
                "best_so_far": {
                    "path": best_path.name if best_path is not None else None,
                    "epoch": best_epoch,
                    "selection_tuple": list(best),
                },
            })
        concise = {
            "epoch": epoch,
            "objective": train_metrics["objective"],
            "warm_adjacent_position": train_metrics["warm_adjacent_position"],
            "validated": validate_now,
        }
        if validate_now:
            concise["selection_tuple"] = record["selection_tuple"]
        print(json.dumps(concise, sort_keys=True), flush=True)
        epochs_completed = epoch
        if args.max_wall_minutes > 0 and (
            time.monotonic() - started >= args.max_wall_minutes * 60
        ):
            stop_reason = "wall_time_limit"
            break

    if last_path.exists():
        raise FileExistsError(f"refusing to overwrite final checkpoint: {last_path}")
    _checkpoint(
        last_path, model, epochs_completed, validation, provenance, "last",
        optimizer, scheduler, scaler,
    )
    if best_epoch < 1 or best_path is None or not best_path.exists():
        raise RuntimeError("training produced no trained best checkpoint")
    if latest_path is None or not latest_path.exists():
        raise RuntimeError("training produced no recoverable validation checkpoint")
    final = {
        **provenance,
        "status": "complete",
        "stop_reason": stop_reason,
        "epochs_completed": epochs_completed,
        "best": {
            "path": best_path.name,
            "sha256": _sha256(best_path),
            "epoch": best_epoch,
            "selection_tuple": list(best),
            "trained_checkpoint": True,
        },
        "last": {"path": last_path.name, "sha256": _sha256(last_path)},
        "latest_checkpoint": {
            "path": latest_path.name, "sha256": _sha256(latest_path)
        },
    }
    _write_json(manifest_path, final)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--audit-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--history-events", type=int, default=32)
    parser.add_argument("--secondary-gap-ratio", type=float, default=0.25)
    parser.add_argument("--huber-beta-m", type=float, default=0.01)
    parser.add_argument("--sigma-weight", type=float, default=0.05)
    parser.add_argument("--edge-weight", type=float, default=0.25)
    parser.add_argument("--recent-age-s", type=float, default=0.2)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--amp", choices=("off", "bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--train-sample-limit", type=int, default=0)
    parser.add_argument("--validation-sample-limit", type=int, default=0)
    parser.add_argument("--max-wall-minutes", type=float, default=0.0)
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    positive = (
        args.epochs, args.batch_size, args.audit_batch_size, args.lr,
        args.channels, args.huber_beta_m, args.recent_age_s,
        args.validation_interval, args.grad_clip,
    )
    if any(value <= 0 for value in positive):
        parser.error("cyclic-state training arguments must be positive")
    if args.channels % 4:
        parser.error("channels must be divisible by four")
    if not 8 <= args.history_events <= 200:
        parser.error("history-events must be within [8,200]")
    if not 1 <= args.warmup_epochs < args.epochs:
        parser.error("warmup-epochs must be within [1,epochs)")
    if not 0 <= args.secondary_gap_ratio <= 1:
        parser.error("secondary-gap-ratio must be within [0,1]")
    if args.weight_decay < 0 or args.sigma_weight < 0 or args.edge_weight < 0:
        parser.error("optimizer/loss weights cannot be negative")
    if args.train_sample_limit < 0 or args.validation_sample_limit < 0:
        parser.error("sample limits cannot be negative")
    print(train(args))


if __name__ == "__main__":
    main()
