"""Train one independent center-free future motion expert after frozen V19-S."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cyclic_future_dataset import CyclicFutureExpertDataset
from .cyclic_future_foundation import (
    FrozenV19Adapter,
    load_frozen_v19,
    sha256_file,
    state_dict_sha256,
)
from .cyclic_future_loss import (
    cyclic_future_expert_loss,
    truth_omega_from_future,
)
from .cyclic_future_model import CyclicFutureMotionExpert, DYNAMIC_EXPERTS
from .train_causal_physical_ab import _git_state, _seed, _to_device, _write_json


SELECTION_QUERY_INDEX = 3
SOURCE_FILES = (
    "cyclic_track_dataset.py",
    "cyclic_track_model.py",
    "cyclic_state_model.py",
    "cyclic_anchor_edge_model.py",
    "cyclic_future_dataset.py",
    "cyclic_future_foundation.py",
    "cyclic_future_model.py",
    "cyclic_future_loss.py",
    "train_cyclic_future_expert.py",
)


def _json_sha256(value: object) -> str:
    serialized = json.dumps(value, indent=2, sort_keys=True)
    encoded = serialized.replace("\n", os.linesep).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _model_forward(
    model: CyclicFutureMotionExpert,
    batch: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    *,
    q0_override_m: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    return model(
        batch["obs"], batch["obs_mask"], batch["primary_mask"],
        batch["event_mask"], batch["event_time_s"], batch["switch_step"],
        batch["tau"], state, q0_override_m=q0_override_m,
    )


def _foundation_forward(
    adapter: FrozenV19Adapter,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return adapter(
        batch["obs"], batch["obs_mask"], batch["primary_mask"],
        batch["event_mask"], batch["event_time_s"], batch["switch_step"],
    )


def _roll_state(
    state: dict[str, torch.Tensor], shift: int,
) -> dict[str, torch.Tensor]:
    rolled: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if value.ndim >= 2 and value.shape[1] == 4:
            rolled[name] = torch.roll(value, shifts=shift, dims=1)
        else:
            rolled[name] = value
    if "primary_index" in rolled:
        rolled["primary_index"] = (state["primary_index"] + shift) % 4
    return rolled


def _equivariance_audit(
    model: CyclicFutureMotionExpert,
    adapter: FrozenV19Adapter,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    state = _foundation_forward(adapter, batch)
    reference = _model_forward(model, batch, state)
    position_max = delta_max = omega_max = 0.0
    for shift in (1, 2, 3):
        shifted = dict(batch)
        for name in ("obs", "obs_mask", "primary_mask"):
            shifted[name] = torch.roll(batch[name], shifts=shift, dims=2)
        shifted_state = _foundation_forward(adapter, shifted)
        output = _model_forward(model, shifted, shifted_state)
        position_max = max(position_max, float((
            output["position_m"]
            - torch.roll(reference["position_m"], shifts=shift, dims=2)
        ).abs().max().cpu()))
        delta_max = max(delta_max, float((
            output["delta_m"]
            - torch.roll(reference["delta_m"], shifts=shift, dims=2)
        ).abs().max().cpu()))
        omega_max = max(omega_max, float((
            output["omega_rad_s"] - reference["omega_rad_s"]
        ).abs().max().cpu()))
    return {
        "position_max_abs_m": position_max,
        "delta_max_abs_m": delta_max,
        "omega_max_abs_rad_s": omega_max,
    }


def _pair_distance_drift(position: torch.Tensor) -> torch.Tensor:
    distance = torch.cdist(position, position)
    return (distance - distance[:, :1]).abs().amax(dim=(-1, -2, -3))


def _query_role_metrics(
    error: np.ndarray,
    delta_error: np.ndarray,
    truth_q0_error: np.ndarray,
    rule: np.ndarray,
    valid: np.ndarray,
    visible: np.ndarray,
    warm: np.ndarray,
    clockwise: np.ndarray,
    counterclockwise: np.ndarray,
    primary: np.ndarray,
    query: int,
) -> dict[str, object]:
    eligible = rule[:, query, None] & valid
    row = np.arange(primary.shape[0])
    primary_mask = np.zeros_like(valid)
    primary_mask[row, primary] = True
    roles = {
        "all_task_tracks": valid,
        "current_visible": visible & valid,
        "warm_adjacent": warm & valid,
        "clockwise_warm": clockwise & warm & valid,
        "counterclockwise_warm": counterclockwise & warm & valid,
        "primary": primary_mask & valid,
    }
    result: dict[str, object] = {}
    for name, role in roles.items():
        mask = eligible & role
        result[name] = {
            "cascade_absolute": _summary(error[:, query][mask]),
            "motion_delta": _summary(delta_error[:, query][mask]),
            "truth_q0_anchor": _summary(truth_q0_error[:, query][mask]),
        }
    return result


def _validate(
    model: CyclicFutureMotionExpert,
    adapter: FrozenV19Adapter,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    model.eval()
    adapter.eval()
    cascade_parts: list[np.ndarray] = []
    delta_parts: list[np.ndarray] = []
    truth_q0_parts: list[np.ndarray] = []
    q0_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    visible_parts: list[np.ndarray] = []
    warm_parts: list[np.ndarray] = []
    clockwise_parts: list[np.ndarray] = []
    counterclockwise_parts: list[np.ndarray] = []
    primary_parts: list[np.ndarray] = []
    rule_parts: list[np.ndarray] = []
    tau_parts: list[np.ndarray] = []
    rigid_parts: list[np.ndarray] = []
    predicted_omega_parts: list[np.ndarray] = []
    truth_omega_parts: list[np.ndarray] = []
    omega_support_parts: list[np.ndarray] = []
    velocity_error_parts: list[np.ndarray] = []
    velocity_ratio_parts: list[np.ndarray] = []
    velocity_cosine_parts: list[np.ndarray] = []
    equivariance: dict[str, float] | None = None
    amp_enabled = device.type == "cuda" and args.amp != "off"
    amp_dtype = torch.float16 if args.amp == "float16" else torch.bfloat16
    with torch.no_grad():
        for raw in loader:
            batch = _to_device(raw, device)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled,
            ):
                state = _foundation_forward(adapter, batch)
                output = _model_forward(model, batch, state)
                truth_output = _model_forward(
                    model, batch, state,
                    q0_override_m=batch["future_position"][:, 0],
                )
            target = batch["future_position"].float()
            target_delta = target - target[:, :1]
            cascade = torch.linalg.vector_norm(
                output["position_m"].float() - target, dim=-1
            )
            delta = torch.linalg.vector_norm(
                output["delta_m"].float() - target_delta, dim=-1
            )
            truth_q0 = torch.linalg.vector_norm(
                truth_output["position_m"].float() - target, dim=-1
            )
            q0_error = torch.linalg.vector_norm(
                state["q0_m"].float() - target[:, 0], dim=-1
            )
            cascade_parts.append(cascade.cpu().numpy())
            delta_parts.append(delta.cpu().numpy())
            truth_q0_parts.append(truth_q0.cpu().numpy())
            q0_parts.append(q0_error.cpu().numpy())
            valid_parts.append(output["future_valid"].cpu().numpy())
            visible_parts.append(state["current_visible"].cpu().numpy())
            warm_parts.append(state["anchor_composed"].cpu().numpy())
            clockwise_parts.append(state["clockwise"].cpu().numpy())
            counterclockwise_parts.append(state["counterclockwise"].cpu().numpy())
            primary_parts.append(state["primary_index"].cpu().numpy())
            rule_parts.append(batch["rule_query"].cpu().numpy())
            tau_parts.append(batch["tau"].float().cpu().numpy())
            rigid_parts.append(_pair_distance_drift(output["position_m"].float()).cpu().numpy())
            if args.expert in {"rotation", "combined"}:
                truth_omega, support = truth_omega_from_future(
                    target, batch["tau"], batch["rule_query"],
                    output["future_valid"],
                )
                predicted_omega_parts.append(output["omega_rad_s"].float().cpu().numpy())
                truth_omega_parts.append(truth_omega.cpu().numpy())
                omega_support_parts.append(support.cpu().numpy())
            if args.expert == "translation":
                row = torch.arange(target.shape[0], device=device)
                primary = state["primary_index"]
                dt = batch["tau"][:, 1].float()
                truth_velocity = (
                    target[:, 1][row, primary] - target[:, 0][row, primary]
                ) / dt[:, None].clamp_min(1e-4)
                predicted_velocity = output["velocity_mps"].float()
                velocity_error_parts.append(torch.linalg.vector_norm(
                    predicted_velocity - truth_velocity, dim=-1
                ).cpu().numpy())
                truth_speed = torch.linalg.vector_norm(truth_velocity, dim=-1)
                predicted_speed = torch.linalg.vector_norm(predicted_velocity, dim=-1)
                velocity_ratio_parts.append((
                    predicted_speed / truth_speed.clamp_min(1e-6)
                ).cpu().numpy())
                velocity_cosine_parts.append((
                    (predicted_velocity * truth_velocity).sum(dim=-1)
                    / (predicted_speed * truth_speed).clamp_min(1e-6)
                ).cpu().numpy())
            if equivariance is None:
                equivariance = _equivariance_audit(model, adapter, batch)

    if not cascade_parts or equivariance is None:
        raise RuntimeError("validation produced no expert samples")
    cascade = np.concatenate(cascade_parts)
    delta = np.concatenate(delta_parts)
    truth_q0 = np.concatenate(truth_q0_parts)
    q0_error = np.concatenate(q0_parts)
    valid = np.concatenate(valid_parts).astype(np.bool_)
    visible = np.concatenate(visible_parts).astype(np.bool_)
    warm = np.concatenate(warm_parts).astype(np.bool_)
    clockwise = np.concatenate(clockwise_parts).astype(np.bool_)
    counterclockwise = np.concatenate(counterclockwise_parts).astype(np.bool_)
    primary = np.concatenate(primary_parts).astype(np.int64)
    rule = np.concatenate(rule_parts).astype(np.bool_)
    tau = np.concatenate(tau_parts)
    queries = []
    for query in range(cascade.shape[1]):
        queries.append({
            "query_index": query,
            "tau_s_median": float(np.median(tau[:, query])),
            "eligible_sample_count": int(rule[:, query].sum()),
            "roles": _query_role_metrics(
                cascade, delta, truth_q0, rule, valid, visible, warm,
                clockwise, counterclockwise, primary, query,
            ),
        })
    selected = queries[SELECTION_QUERY_INDEX]
    q0_task = _summary(q0_error[valid])
    result: dict[str, object] = {
        "sample_count": int(cascade.shape[0]),
        "expert": args.expert,
        "queries": queries,
        "selection_query": selected,
        "s_q0_task_tracks": q0_task,
        "rigid_pair_distance_drift": _summary(np.concatenate(rigid_parts)),
        "cyclic_equivariance": equivariance,
        "support": {
            "task_track_count": int(valid.sum()),
            "current_visible_count": int((valid & visible).sum()),
            "warm_adjacent_count": int((valid & warm).sum()),
        },
    }
    if predicted_omega_parts:
        predicted_omega = np.concatenate(predicted_omega_parts)
        truth_omega = np.concatenate(truth_omega_parts)
        support = np.concatenate(omega_support_parts).astype(np.bool_)
        if bool(support.any()):
            omega_error = np.abs(predicted_omega[support] - truth_omega[support])
            ratio = np.abs(predicted_omega[support]) / np.maximum(np.abs(truth_omega[support]), 1e-6)
            result["omega"] = {
                "error_rad_s": _summary(omega_error),
                "abs_ratio_median": float(np.median(ratio)),
                "sign_accuracy": float(np.mean(
                    np.sign(predicted_omega[support]) == np.sign(truth_omega[support])
                )),
                "support_count": int(support.sum()),
            }
        else:
            result["omega"] = {
                "error_rad_s": _summary(np.array([], dtype=np.float64)),
                "abs_ratio_median": None,
                "sign_accuracy": None,
                "support_count": 0,
            }
    if velocity_error_parts:
        result["translation_velocity"] = {
            "error_mps": _summary(np.concatenate(velocity_error_parts)),
            "speed_ratio_median": float(np.median(np.concatenate(velocity_ratio_parts))),
            "direction_cosine_median": float(np.median(np.concatenate(velocity_cosine_parts))),
        }
    return result


def _selection_tuple(metrics: dict[str, object]) -> tuple[float, ...]:
    roles = metrics["selection_query"]["roles"]  # type: ignore[index]
    if metrics["expert"] == "translation":
        candidates = (roles["current_visible"],)
    else:
        candidates = (roles["current_visible"], roles["warm_adjacent"])
    cascade_p95 = max(
        float(item["cascade_absolute"]["p95_m"] or float("inf"))
        for item in candidates
    )
    motion_p95 = max(
        float(item["motion_delta"]["p95_m"] or float("inf"))
        for item in candidates
    )
    truth_p95 = max(
        float(item["truth_q0_anchor"]["p95_m"] or float("inf"))
        for item in candidates
    )
    cascade_median = max(
        float(item["cascade_absolute"]["median_m"] or float("inf"))
        for item in candidates
    )
    omega_penalty = 0.0
    if "omega" in metrics:
        accuracy = metrics["omega"]["sign_accuracy"]  # type: ignore[index]
        omega_penalty = 1.0 if accuracy is None else 1.0 - float(accuracy)
    return cascade_p95, motion_p95, truth_p95, omega_penalty, cascade_median


def _checkpoint(
    path: Path,
    model: CyclicFutureMotionExpert,
    epoch: int,
    metrics: dict[str, object],
    provenance: dict[str, object],
    role: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    record: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "model": model.state_dict(),
        "model_class": model.__class__.__name__,
        "model_config": model.config(),
        "epoch": int(epoch),
        "checkpoint_role": role,
        "validation": metrics,
        "selection_tuple": _selection_tuple(metrics),
        "provenance": provenance,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if next(model.parameters()).device.type == "cuda" else None
        ),
        "record": record,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    temporary = path.with_name(
        f".{path.name}.pending-{os.getpid()}-{time.time_ns()}"
    )
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


RESUME_PROVENANCE_FIELDS = (
    "schema_version", "expert", "test_accessed", "dataset",
    "dataset_manifest_sha256", "dataset_qualification", "foundation",
    "frozen_foundation_initial_state_sha256",
    "git_commit", "worktree_dirty", "config", "model_config",
    "input_allowlist", "forbidden_predictor_inputs",
    "architecture_contract", "objective_contract", "selection_contract",
    "source_sha256",
)


def _validate_resume_checkpoint(
    payload: dict[str, object],
    *,
    expected_epoch: int,
    expected_role: str,
    provenance: dict[str, object],
    model: CyclicFutureMotionExpert,
    device: torch.device,
) -> dict[str, object]:
    if payload.get("model_class") != model.__class__.__name__:
        raise ValueError("resume checkpoint model class mismatch")
    if payload.get("model_config") != model.config():
        raise ValueError("resume checkpoint model config mismatch")
    if int(payload.get("epoch", -1)) != expected_epoch:
        raise ValueError("resume checkpoint epoch mismatch")
    if payload.get("checkpoint_role") != expected_role:
        raise ValueError("resume checkpoint role mismatch")
    embedded = payload.get("provenance")
    if not isinstance(embedded, dict) or any(
        embedded.get(name) != provenance.get(name)
        for name in RESUME_PROVENANCE_FIELDS
    ):
        raise ValueError("resume checkpoint embedded provenance mismatch")
    record = payload.get("record")
    if not isinstance(record, dict) or int(record.get("epoch", -1)) != expected_epoch:
        raise ValueError("resume checkpoint record is missing or malformed")
    if tuple(payload.get("selection_tuple", ())) != tuple(
        record.get("selection_tuple", ())
    ) or payload.get("validation") != record.get("validation"):
        raise ValueError("resume checkpoint record differs from validation payload")
    for name in ("model", "optimizer", "scheduler", "scaler", "torch_rng_state"):
        if name not in payload:
            raise ValueError(f"resume checkpoint is missing {name}")
    rng = payload["torch_rng_state"]
    if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8:
        raise ValueError("resume checkpoint CPU RNG state is malformed")
    if device.type == "cuda":
        cuda_rng = payload.get("cuda_rng_state_all")
        if not isinstance(cuda_rng, list) or not cuda_rng:
            raise ValueError("resume checkpoint CUDA RNG state is missing")
    return record


def _train_epoch(
    model: CyclicFutureMotionExpert,
    adapter: FrozenV19Adapter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    adapter.eval()
    totals: dict[str, float] = {}
    sample_count = 0
    amp_enabled = device.type == "cuda" and args.amp != "off"
    amp_dtype = torch.float16 if args.amp == "float16" else torch.bfloat16
    for raw in loader:
        batch = _to_device(raw, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled,
        ):
            state = _foundation_forward(adapter, batch)
            output = _model_forward(model, batch, state)
            total, parts = cyclic_future_expert_loss(
                args.expert, output, state, batch["future_position"],
                batch["tau"], batch["rule_query"],
                huber_beta_m=args.huber_beta_m,
                omega_weight=args.omega_weight,
                omega_sign_weight=args.omega_sign_weight,
            )
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("non-finite future expert loss")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip,
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("non-finite future expert gradient")
        scaler.step(optimizer)
        scaler.update()
        count = int(batch["obs"].shape[0])
        values = {
            **{name: float(value.detach().cpu()) for name, value in parts.items()},
            "gradient_norm": float(gradient_norm.detach().cpu()),
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise FloatingPointError("non-finite future expert train metric")
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value * count
        sample_count += count
    if sample_count == 0:
        raise RuntimeError("training produced no expert samples")
    return {name: value / sample_count for name, value in totals.items()}


def _lr_lambda(epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return float(epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def train(args: argparse.Namespace) -> Path:
    _seed(args.seed)
    git_state = _git_state()
    if bool(git_state["worktree_dirty"]) and not args.allow_dirty_worktree:
        raise ValueError("official future expert training requires a clean worktree")
    output = Path(args.output).resolve()
    if args.resume:
        if not output.is_dir():
            raise FileNotFoundError(f"resume output directory is missing: {output}")
    elif output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_ds = CyclicFutureExpertDataset(
        args.dataset, "train", expert=args.expert, seed=args.seed,
        shuffle=True, sample_limit=args.train_sample_limit,
        secondary_gap_ratio=args.secondary_gap_ratio,
        augment_cyclic_origin=True, augment_direction=True,
    )
    validation_ds = CyclicFutureExpertDataset(
        args.dataset, "validation", expert=args.expert, seed=args.seed,
        shuffle=False, sample_limit=args.validation_sample_limit,
        secondary_gap_ratio=args.secondary_gap_ratio,
    )
    source_test_accessed = bool(train_ds.manifest.get("test_accessed", True))
    if source_test_accessed:
        raise ValueError("future expert source dataset has accessed test")
    foundation, foundation_info = load_frozen_v19(
        args.foundation_checkpoint,
        expected_dataset_manifest_sha256=train_ds.manifest_sha256,
    )
    adapter = FrozenV19Adapter(foundation).to(device)
    frozen_initial_sha = state_dict_sha256(adapter.foundation.state_dict())
    model = CyclicFutureMotionExpert(
        args.expert,
        torch.from_numpy(train_ds.mean), torch.from_numpy(train_ds.std),
        channels=args.channels, dropout=args.dropout,
        history_events=args.history_events,
        max_speed_mps=args.max_speed_mps,
        max_acceleration_mps2=args.max_acceleration_mps2,
        max_omega_rad_s=args.max_omega_rad_s,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: _lr_lambda(epoch, args.warmup_epochs, args.epochs),
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=(device.type == "cuda" and args.amp == "float16"),
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, **loader_options)
    validation_loader = DataLoader(validation_ds, **loader_options)
    source_dir = Path(__file__).resolve().parent
    source_sha = {
        name: sha256_file(source_dir / name) for name in SOURCE_FILES
    }
    config = {
        key: value for key, value in vars(args).items()
        if key != "resume"
    }
    config["dataset"] = str(Path(args.dataset).resolve())
    config["foundation_checkpoint"] = str(Path(args.foundation_checkpoint).resolve())
    config["output"] = str(output)
    provenance: dict[str, object] = {
        "schema_version": "stage3-cyclic-future-expert-run-v1",
        "expert": args.expert,
        "status": "running",
        "test_accessed": False,
        "dataset": str(Path(args.dataset).resolve()),
        "dataset_manifest_sha256": train_ds.manifest_sha256,
        "dataset_qualification": {
            "qualification_passed": train_ds.manifest.get("qualification_passed"),
            "test_accessed": source_test_accessed,
            "train_expert_samples": len(train_ds),
            "validation_expert_samples": len(validation_ds),
            "virtual_contract": train_ds.virtual_contract,
        },
        "foundation": foundation_info,
        "frozen_foundation_initial_state_sha256": frozen_initial_sha,
        "git_commit": git_state["git_commit"],
        "worktree_dirty": git_state["worktree_dirty"],
        "config": config,
        "model_config": model.config(),
        "input_allowlist": [
            "causal normalized visible xyz", "visibility/primary/event masks",
            "causal real event time", "switch step", "future query tau",
            "frozen V19 q0/edge validity, age, sigma and cyclic role state",
        ],
        "forbidden_predictor_inputs": [
            "future truth", "motion class", "rule_query", "truth velocity",
            "truth omega", "center", "phase", "radius", "height template",
            "physical slot identity", "PnP", "test",
        ],
        "architecture_contract": {
            "s_layer": "frozen V19 q0 current-state restorer",
            "future_layer": "independent center-free continuous motion expert",
            "q0_head": False,
            "tau_zero_identity": True,
            "fixed_geometry": False,
            "combined_is_independent": True,
        },
        "objective_contract": {
            "primary": "group-balanced future delta SmoothL1",
            "omega_magnitude_weight": args.omega_weight,
            "omega_sign_weight": args.omega_sign_weight,
            "future_truth_role": "loss and validation only",
            "confidence_controls_eligibility": False,
        },
        "selection_contract": (
            "actual query index 3 tau: worst task-role cascade P95, motion-delta "
            "P95, truth-q0 P95, omega-sign penalty, cascade median"
        ),
        "source_sha256": source_sha,
    }
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "run_manifest.json"
    history_path = output / f"stage3-cyclic-future-{args.expert}-history.json"
    history: list[dict[str, object]] = []
    start_epoch = 1
    best_epoch = -1
    best_tuple: tuple[float, ...] | None = None
    best_path: Path | None = None
    resume_chain: list[dict[str, object]] = []
    resume_count = 0

    if args.resume:
        if not manifest_path.is_file() or not history_path.is_file():
            raise ValueError("resume output is missing manifest/history")
        source_manifest_sha = sha256_file(manifest_path)
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") not in {"running", "interrupted"}:
            raise ValueError("only a running or interrupted run may resume")
        if bool(existing.get("test_accessed", True)):
            raise ValueError("test-accessed run cannot resume")
        for name in RESUME_PROVENANCE_FIELDS:
            if existing.get(name) != provenance.get(name):
                raise ValueError(f"resume provenance mismatch: {name}")
        if existing.get("history") != history_path.name:
            raise ValueError("resume history path mismatch")
        committed_epoch = int(existing.get("epochs_completed", -1))
        if not 0 <= committed_epoch <= args.epochs:
            raise ValueError("resume manifest epoch is outside the run budget")
        loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_history, list):
            raise ValueError("resume history is malformed")
        committed_history = [
            item for item in loaded_history
            if isinstance(item, dict) and int(item.get("epoch", -1)) <= committed_epoch
        ]
        if (
            not committed_history
            or int(committed_history[0].get("epoch", -1)) != 0
            or int(committed_history[-1].get("epoch", -1)) != committed_epoch
            or any(
                int(left.get("epoch", -1)) >= int(right.get("epoch", -1))
                for left, right in zip(committed_history, committed_history[1:])
            )
            or _json_sha256(committed_history) != existing.get("history_sha256")
        ):
            raise ValueError("resume committed history differs from manifest")
        latest = existing.get("latest_checkpoint", {})
        if not isinstance(latest, dict) or not latest.get("path"):
            raise ValueError("resume manifest is missing latest checkpoint")
        latest_path = output / str(latest.get("path", ""))
        if not latest_path.is_file() or sha256_file(latest_path) != latest.get("sha256"):
            raise ValueError("resume latest checkpoint missing or changed")
        payload = torch.load(latest_path, map_location=device, weights_only=False)
        _validate_resume_checkpoint(
            payload,
            expected_epoch=committed_epoch,
            expected_role="initial" if committed_epoch == 0 else "validation",
            provenance=provenance,
            model=model,
            device=device,
        )

        best = existing.get("best", {})
        if committed_epoch > 0:
            if not isinstance(best, dict) or not bool(best.get("trained_checkpoint")):
                raise ValueError("resume manifest is missing trained best checkpoint")
            best_epoch = int(best.get("epoch", -1))
            best_path = output / str(best.get("path", ""))
            if (
                not 0 < best_epoch <= committed_epoch
                or not best_path.is_file()
                or sha256_file(best_path) != best.get("sha256")
            ):
                raise ValueError("resume best checkpoint missing or changed")
            matching_best = [
                item for item in committed_history
                if int(item.get("epoch", -1)) == best_epoch
            ]
            if (
                len(matching_best) != 1
                or matching_best[0].get("selection_tuple")
                != best.get("selection_tuple")
            ):
                raise ValueError("resume best checkpoint differs from history")
            best_tuple = tuple(float(value) for value in best["selection_tuple"])
        elif isinstance(best, dict) and bool(best.get("trained_checkpoint", False)):
            raise ValueError("epoch-zero resume cannot have a trained best")

        checkpoint_pattern = re.compile(
            rf"^stage3-cyclic-future-{re.escape(args.expert)}-"
            rf"seed{args.seed}-epoch(\d+)\.pt$"
        )
        orphan_candidates: list[tuple[int, Path]] = []
        for candidate in output.glob(
            f"stage3-cyclic-future-{args.expert}-seed{args.seed}-epoch*.pt"
        ):
            match = checkpoint_pattern.match(candidate.name)
            if match and int(match.group(1)) > committed_epoch:
                orphan_candidates.append((int(match.group(1)), candidate))
        if len(orphan_candidates) > 1:
            raise ValueError("resume found multiple uncommitted checkpoints")
        adopted_orphan = False
        if orphan_candidates:
            orphan_epoch, orphan_path = orphan_candidates[0]
            if orphan_epoch > args.epochs:
                raise ValueError("orphan checkpoint exceeds configured epoch budget")
            orphan_payload = torch.load(
                orphan_path, map_location=device, weights_only=False,
            )
            orphan_record = _validate_resume_checkpoint(
                orphan_payload,
                expected_epoch=orphan_epoch,
                expected_role="validation",
                provenance=provenance,
                model=model,
                device=device,
            )
            uncommitted_history = [
                item for item in loaded_history
                if isinstance(item, dict) and int(item.get("epoch", -1)) > committed_epoch
            ]
            if uncommitted_history and uncommitted_history != [orphan_record]:
                raise ValueError("orphan checkpoint differs from uncommitted history")
            history = [*committed_history, orphan_record]
            _write_json(history_path, history)
            selection = tuple(float(value) for value in orphan_record["selection_tuple"])
            if best_tuple is None or selection < best_tuple:
                best_tuple = selection
                best_epoch = orphan_epoch
                best_path = orphan_path
            latest_path = orphan_path
            payload = orphan_payload
            committed_epoch = orphan_epoch
            adopted_orphan = True
            existing = {
                **existing,
                "status": (
                    "interrupted" if bool(orphan_record.get("wall_stop"))
                    else "running"
                ),
                "stop_reason": (
                    "wall_time_resumable"
                    if bool(orphan_record.get("wall_stop")) else None
                ),
                "epochs_completed": committed_epoch,
                "history_sha256": sha256_file(history_path),
                "latest_checkpoint": {
                    "path": latest_path.name,
                    "sha256": sha256_file(latest_path),
                },
                "best": {
                    "epoch": best_epoch,
                    "path": best_path.name if best_path else None,
                    "sha256": sha256_file(best_path) if best_path else None,
                    "selection_tuple": list(best_tuple) if best_tuple else None,
                    "trained_checkpoint": best_epoch > 0,
                },
            }
            _write_json(manifest_path, existing)
        else:
            history = committed_history
            if loaded_history != committed_history:
                _write_json(history_path, committed_history)

        resume_chain = list(existing.get("resume_chain", []))
        resume_count = int(existing.get("resume_count", 0)) + 1
        resume_chain.append({
            "resume_index": resume_count,
            "from_epoch": committed_epoch,
            "source_manifest_sha256": source_manifest_sha,
            "latest_checkpoint_path": latest_path.name,
            "latest_checkpoint_sha256": sha256_file(latest_path),
            "adopted_orphan_checkpoint": adopted_orphan,
        })
        existing = {
            **existing,
            "status": "running",
            "stop_reason": None,
            "resume_count": resume_count,
            "resume_chain": resume_chain,
        }
        _write_json(manifest_path, existing)

        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        # ``map_location=device`` also moves the serialized CPU RNG tensor to
        # CUDA.  The default CPU generator requires a CPU ByteTensor.
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if device.type == "cuda" and payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all([
                state.cpu() for state in payload["cuda_rng_state_all"]
            ])
        start_epoch = committed_epoch + 1
    else:
        initial_metrics = _validate(
            model, adapter, validation_loader, device, args,
        )
        initial_path = output / f"stage3-cyclic-future-{args.expert}-seed{args.seed}-initial.pt"
        initial_record = {
            "epoch": 0, "lr": float(optimizer.param_groups[0]["lr"]),
            "selection_tuple": list(_selection_tuple(initial_metrics)),
            "validation": initial_metrics,
        }
        _checkpoint(
            initial_path, model, 0, initial_metrics, provenance, "initial",
            optimizer, scheduler, scaler, record=initial_record,
        )
        history.append(initial_record)
        _write_json(history_path, history)
        _write_json(manifest_path, {
            **provenance,
            "status": "running",
            "stop_reason": None,
            "epochs_completed": 0,
            "history": history_path.name,
            "history_sha256": sha256_file(history_path),
            "latest_checkpoint": {
                "path": initial_path.name,
                "sha256": sha256_file(initial_path),
            },
            "best": {
                "epoch": -1, "path": None, "sha256": None,
                "selection_tuple": None, "trained_checkpoint": False,
            },
            "resume_count": 0,
            "resume_chain": [],
        })

    started = time.monotonic()
    stop_reason = "epoch_limit"
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        train_ds.set_epoch(epoch)
        train_metrics = _train_epoch(
            model, adapter, train_loader, optimizer, scaler, device, args,
        )
        scheduler.step()
        last_epoch = epoch
        validate_now = (
            epoch % args.validation_interval == 0 or epoch == args.epochs
        )
        wall_stop = (
            epoch < args.epochs
            and
            args.max_wall_minutes > 0
            and (time.monotonic() - started) / 60.0 >= args.max_wall_minutes
        )
        if wall_stop:
            validate_now = True
            stop_reason = "wall_time_resumable"
        if not validate_now:
            continue
        validation = _validate(model, adapter, validation_loader, device, args)
        selection = _selection_tuple(validation)
        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "selection_tuple": list(selection),
            "train": train_metrics,
            "validation": validation,
            "wall_stop": wall_stop,
        }
        epoch_path = output / f"stage3-cyclic-future-{args.expert}-seed{args.seed}-epoch{epoch:03d}.pt"
        _checkpoint(
            epoch_path, model, epoch, validation, provenance, "validation",
            optimizer, scheduler, scaler, record=record,
        )
        history.append(record)
        _write_json(history_path, history)
        epoch_sha = sha256_file(epoch_path)
        if best_tuple is None or selection < best_tuple:
            best_tuple = selection
            best_epoch = epoch
            best_path = epoch_path
        running = {
            **provenance,
            "status": "running" if not wall_stop else "interrupted",
            "stop_reason": None if not wall_stop else stop_reason,
            "epochs_completed": epoch,
            "history": history_path.name,
            "history_sha256": sha256_file(history_path),
            "latest_checkpoint": {"path": epoch_path.name, "sha256": epoch_sha},
            "best": {
                "epoch": best_epoch,
                "path": best_path.name if best_path else None,
                "sha256": sha256_file(best_path) if best_path else None,
                "selection_tuple": list(best_tuple) if best_tuple else None,
                "trained_checkpoint": best_epoch > 0,
            },
            "resume_count": resume_count,
            "resume_chain": resume_chain,
        }
        _write_json(manifest_path, running)
        print(json.dumps({
            "epoch": epoch, "expert": args.expert, "train": train_metrics,
            "selection_tuple": selection,
        }, sort_keys=True), flush=True)
        if wall_stop:
            return manifest_path

    if last_epoch != args.epochs or best_path is None or best_epoch <= 0:
        raise RuntimeError("future expert training did not produce a trained best")
    frozen_final_sha = state_dict_sha256(adapter.foundation.state_dict())
    frozen_unchanged = frozen_final_sha == frozen_initial_sha
    if not frozen_unchanged:
        raise RuntimeError("frozen V19 foundation changed during future training")
    last_metrics = history[-1]["validation"]
    last_path = output / f"stage3-cyclic-future-{args.expert}-seed{args.seed}-last.pt"
    if last_path.exists():
        raise FileExistsError(f"refusing to overwrite final checkpoint: {last_path}")
    _checkpoint(last_path, model, last_epoch, last_metrics, provenance, "last")
    completed = {
        **provenance,
        "status": "complete",
        "stop_reason": "epoch_limit",
        "epochs_completed": last_epoch,
        "history": history_path.name,
        "history_sha256": sha256_file(history_path),
        "latest_checkpoint": {
            "path": (
                f"stage3-cyclic-future-{args.expert}-seed{args.seed}-epoch{last_epoch:03d}.pt"
            ),
            "sha256": sha256_file(
                output / f"stage3-cyclic-future-{args.expert}-seed{args.seed}-epoch{last_epoch:03d}.pt"
            ),
        },
        "best": {
            "epoch": best_epoch, "path": best_path.name,
            "sha256": sha256_file(best_path),
            "selection_tuple": list(best_tuple), "trained_checkpoint": True,
        },
        "last": {"path": last_path.name, "sha256": sha256_file(last_path)},
        "frozen_foundation_final_state_sha256": frozen_final_sha,
        "frozen_foundation_verified_unchanged": True,
        "resume_count": resume_count,
        "resume_chain": resume_chain,
    }
    _write_json(manifest_path, completed)
    return manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--foundation-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expert", choices=DYNAMIC_EXPERTS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--history-events", type=int, default=32)
    parser.add_argument("--secondary-gap-ratio", type=float, default=0.25)
    parser.add_argument("--huber-beta-m", type=float, default=0.01)
    parser.add_argument("--omega-weight", type=float, default=0.10)
    parser.add_argument("--omega-sign-weight", type=float, default=0.05)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--amp", choices=("off", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="")
    parser.add_argument("--max-speed-mps", type=float, default=7.0)
    parser.add_argument("--max-acceleration-mps2", type=float, default=100.0)
    parser.add_argument("--max-omega-rad-s", type=float, default=20.0)
    parser.add_argument("--train-sample-limit", type=int, default=0)
    parser.add_argument("--validation-sample-limit", type=int, default=0)
    parser.add_argument("--max-wall-minutes", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.validation_interval <= 0:
        parser.error("epochs, batch-size and validation-interval must be positive")
    if not 0 <= args.warmup_epochs < args.epochs:
        parser.error("warmup-epochs must be within [0,epochs)")
    if min(
        args.lr, args.huber_beta_m, args.grad_clip,
        args.max_speed_mps, args.max_acceleration_mps2, args.max_omega_rad_s,
    ) <= 0:
        parser.error("optimizer, loss and motion bounds must be positive")
    if min(args.omega_weight, args.omega_sign_weight, args.weight_decay) < 0:
        parser.error("nonnegative weights are required")
    if args.train_sample_limit < 0 or args.validation_sample_limit < 0:
        parser.error("sample limits must be nonnegative")
    try:
        path = train(args)
    except Exception as exc:  # pragma: no cover - surfaced to runtime stderr
        print(f"future expert training failed: {exc}", file=sys.stderr)
        raise
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
