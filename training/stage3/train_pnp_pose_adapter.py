"""Train Module A: causal PnP history to the current canonical rigid pose.

Only train/validation PnP histories and the same-time q0 physical truth are
admitted. Future queries, motion state, motion class, test data, and the frozen
downstream motion model are unavailable to the predictor and objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .pnp_pose_adapter_loss import current_pose_loss
from .pnp_pose_adapter_model import CurrentPnPPoseAdapter, trainable_parameter_count
from .pnp_state_targets import _query_pose_from_fixed_truth, geometry_c4_asymmetry_m
from .shard_dataset import Stage3ShardDataset


MOTION_NAMES = ("stationary", "translation", "rotation", "combined")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_state() -> dict[str, object]:
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    ).strip())
    return {"git_commit": commit, "worktree_dirty": dirty}


def _load_contract(
    dataset: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, str]]:
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "stage3-dataset-v4-observation":
        raise ValueError("Module A requires stage3-dataset-v4-observation")
    if not bool(manifest.get("qualification_passed", False)):
        raise ValueError("Module A refuses an unqualified dataset")
    expected_features = [
        "x", "y", "z", "sin_yaw", "cos_yaw",
        "reprojection_rms_px", "valid_candidate_fraction",
    ]
    if manifest.get("tensor_contract", {}).get("history_features") != expected_features:
        raise ValueError("Module A history feature order does not match its contract")
    geometry_path = dataset / "geometry_template.json"
    geometry_payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    slots = [int(item["relative_slot"]) for item in geometry_payload["armors"]]
    if sorted(slots) != [0, 1, 2, 3] or len(set(slots)) != 4:
        raise ValueError("geometry template must define unique relative_slot 0..3")
    armor = sorted(geometry_payload["armors"], key=lambda item: int(item["relative_slot"]))
    geometry = torch.tensor([item["relative_position_m"] for item in armor], dtype=torch.float32)
    normalization_path = dataset / str(manifest["normalization"])
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    observation_mean = torch.tensor(normalization["obs_xyz"]["mean"], dtype=torch.float32)
    observation_std = torch.tensor(normalization["obs_xyz"]["std"], dtype=torch.float32)
    if tuple(geometry.shape) != (4, 3):
        raise ValueError("geometry template must define four fixed slots")
    if not bool(torch.isfinite(observation_mean).all()) or not bool(
        torch.isfinite(observation_std).all()
    ) or bool(torch.any(observation_std <= 0)):
        raise ValueError("observation normalization must be finite with positive std")
    if geometry_c4_asymmetry_m(geometry) <= 0.005:
        raise ValueError("geometry cannot identify full canonical phase above 5 mm")
    return geometry, observation_mean, observation_std, {
        "dataset_manifest_sha256": _sha256(manifest_path),
        "geometry_template_sha256": _sha256(geometry_path),
        "normalization_sha256": _sha256(normalization_path),
    }


def _downstream_contract(run_value: str) -> dict[str, object] | None:
    if not run_value:
        return None
    run = Path(run_value).resolve()
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "stage3-moving-refinement-run-v1":
        raise ValueError("downstream contract is not a v15 moving-refinement run")
    if manifest.get("status") != "complete" or bool(manifest.get("test_accessed", True)):
        raise ValueError("downstream v15 contract must be complete with test sealed")
    if bool(manifest.get("worktree_dirty", True)):
        raise ValueError("downstream v15 contract must come from a clean worktree")
    if not bool(manifest.get("frozen_base_verified_unchanged", False)):
        raise ValueError("downstream v15 frozen foundation was not verified")
    best = manifest.get("best", {})
    if not bool(best.get("trained_checkpoint", False)) or int(best.get("epoch", 0)) <= 0:
        raise ValueError("downstream v15 best checkpoint must be trained")
    checkpoint = run / str(best.get("path", ""))
    if not checkpoint.is_file() or _sha256(checkpoint) != str(best.get("sha256", "")):
        raise ValueError("downstream v15 best checkpoint hash mismatch")
    return {
        "role": "frozen_not_loaded_not_optimized",
        "run": str(run),
        "manifest_sha256": _sha256(manifest_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "best_epoch": int(best.get("epoch", 0)),
        "source_git_commit": manifest.get("git_commit"),
    }


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "train_pnp_pose_adapter.py", "pnp_pose_adapter_model.py",
        "pnp_pose_adapter_loss.py", "pnp_state_targets.py",
        "causal_physical_state_model.py", "physical_model.py", "shard_dataset.py",
    )
    return {name: _sha256(root / name) for name in names}


def _anchor_mask(raw: dict[str, torch.Tensor], tolerance_s: float) -> torch.Tensor:
    event = raw["event_mask"].to(torch.bool) & torch.isfinite(raw["event_time_s"])
    index = torch.arange(event.shape[1]).view(1, -1)
    last = torch.where(event, index, torch.full_like(index, -1)).amax(dim=1)
    safe = last.clamp_min(0)
    last_time = raw["event_time_s"].gather(1, safe[:, None]).squeeze(1)
    return (last >= 0) & (last_time <= 0.0) & (last_time >= -tolerance_s)


def _prepare_batch(
    raw: dict[str, torch.Tensor], device: torch.device, tolerance_s: float,
) -> dict[str, torch.Tensor] | None:
    active = _anchor_mask(raw, tolerance_s)
    if not bool(active.any()):
        return None
    if not torch.equal(
        raw["tau"][active, 0], torch.zeros_like(raw["tau"][active, 0])
    ):
        raise ValueError("Module A requires exact same-time tau[0]=0 labels")
    # q1..q7 remain on CPU and never enter the optimizer path.
    return {
        "obs": raw["obs"][active].to(device, non_blocking=True),
        "obs_mask": raw["obs_mask"][active].to(device, non_blocking=True),
        "event_mask": raw["event_mask"][active].to(device, non_blocking=True),
        "event_time_s": raw["event_time_s"][active].to(device, non_blocking=True),
        "target_q0": raw["future_position"][active, 0].to(device, non_blocking=True),
        "motion_class": raw["motion_class"][active].to(device, non_blocking=True),
    }


def _stats(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return {"count": 0}
    if not np.isfinite(array).all():
        raise ValueError("validation metric contains non-finite values")
    return {
        "count": int(len(array)), "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)), "max": float(array.max()),
    }


def _batch_errors(
    output: dict[str, torch.Tensor], target_q0: torch.Tensor, geometry: torch.Tensor,
) -> dict[str, torch.Tensor]:
    truth_center, truth_phase = _query_pose_from_fixed_truth(target_q0[:, None], geometry)
    truth_center, truth_phase = truth_center[:, 0], truth_phase[:, 0]
    prediction = output["position_mean"][:, 0].float()
    fixed = torch.linalg.vector_norm(prediction - target_q0.float(), dim=-1).mean(dim=1)
    pair = torch.linalg.vector_norm(
        prediction[:, :, None] - target_q0.float()[:, None, :], dim=-1
    )
    unordered = 0.5 * (pair.amin(dim=2).mean(dim=1) + pair.amin(dim=1).mean(dim=1))
    center = torch.linalg.vector_norm(output["center"].float() - truth_center, dim=-1)
    predicted_phase = output["phase"].float()
    delta = torch.atan2(
        predicted_phase[:, 0] * truth_phase[:, 1] - predicted_phase[:, 1] * truth_phase[:, 0],
        predicted_phase[:, 0] * truth_phase[:, 0] + predicted_phase[:, 1] * truth_phase[:, 1],
    )
    absolute_degrees = delta.abs() * (180.0 / torch.pi)
    modulo = torch.remainder(delta + torch.pi / 4.0, torch.pi / 2.0) - torch.pi / 4.0
    modulo_degrees = modulo.abs() * (180.0 / torch.pi)
    alias_index = torch.round(delta / (torch.pi / 2.0)).abs().to(torch.int64)
    pair_i, pair_j = torch.triu_indices(4, 4, offset=1, device=prediction.device)
    rigid = torch.abs(
        torch.linalg.vector_norm(prediction[:, pair_i] - prediction[:, pair_j], dim=-1)
        - torch.linalg.vector_norm(geometry[pair_i] - geometry[pair_j], dim=-1)
    ).mean(dim=1)
    return {
        "fixed_slot_position_m": fixed,
        "unordered_set_position_m": unordered,
        "center_m": center,
        "phase_abs_deg": absolute_degrees,
        "phase_modulo_90_deg": modulo_degrees,
        "phase_alias_index": alias_index,
        "rigid_residual_m": rigid,
        "truth_center": truth_center,
    }


def _latest_raw_pnp_error(
    batch: dict[str, torch.Tensor], target_q0: torch.Tensor,
    observation_mean: torch.Tensor, observation_std: torch.Tensor,
) -> torch.Tensor:
    event = batch["event_mask"].to(torch.bool)
    index = torch.arange(event.shape[1], device=event.device).view(1, -1)
    last = torch.where(event, index, torch.full_like(index, -1)).amax(dim=1)
    gather_xyz = last.view(-1, 1, 1, 1).expand(-1, 1, 4, 3)
    gather_mask = last.view(-1, 1, 1).expand(-1, 1, 4)
    xyz = batch["obs"][..., :3].float().gather(1, gather_xyz).squeeze(1)
    xyz = xyz * observation_std + observation_mean
    visible = batch["obs_mask"].gather(1, gather_mask).squeeze(1)
    pair = torch.linalg.vector_norm(xyz[:, :, None] - target_q0[:, None], dim=-1)
    nearest = pair.amin(dim=2)
    return (nearest * visible).sum(dim=1) / visible.sum(dim=1).clamp_min(1)


def _metric_bundle(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, object]:
    alias = arrays["phase_alias_index"][mask]
    return {
        "sample_count": int(mask.sum()),
        "fixed_slot_position_m": _stats(arrays["fixed_slot_position_m"][mask]),
        "unordered_set_position_m": _stats(arrays["unordered_set_position_m"][mask]),
        "center_m": _stats(arrays["center_m"][mask]),
        "phase_abs_deg": _stats(arrays["phase_abs_deg"][mask]),
        "phase_modulo_90_deg": _stats(arrays["phase_modulo_90_deg"][mask]),
        "phase_alias_fraction": float(np.mean(alias > 0)) if len(alias) else 0.0,
        "phase_180_fraction": float(np.mean(alias >= 2)) if len(alias) else 0.0,
    }


def _validate(
    model: CurrentPnPPoseAdapter, loader: DataLoader, device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    model.eval()
    parts: dict[str, list[np.ndarray]] = {}
    input_count = 0
    amp_enabled = device.type == "cuda" and args.amp != "off"
    amp_dtype = torch.float16 if args.amp == "float16" else torch.bfloat16
    with torch.no_grad():
        for raw in loader:
            input_count += int(raw["obs"].shape[0])
            batch = _prepare_batch(raw, device, args.anchor_tolerance_s)
            if batch is None:
                continue
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(
                    batch["obs"], batch["obs_mask"], batch["event_mask"],
                    batch["event_time_s"],
                )
            errors = _batch_errors(output, batch["target_q0"], model.decoder.geometry)
            raw_error = _latest_raw_pnp_error(
                batch, batch["target_q0"], model.observation_mean, model.observation_std
            )
            event = batch["event_mask"].to(torch.bool)
            index = torch.arange(event.shape[1], device=device).view(1, -1)
            last = torch.where(event, index, torch.full_like(index, -1)).amax(dim=1)
            visible = batch["obs_mask"].sum(dim=2).gather(1, last[:, None]).squeeze(1)
            values = {
                **{name: value for name, value in errors.items() if name != "truth_center"},
                "raw_latest_pnp_nearest_truth_m": raw_error,
                "motion_class": batch["motion_class"],
                "distance_m": torch.linalg.vector_norm(errors["truth_center"], dim=-1),
                "latest_visible_count": visible,
            }
            for name, value in values.items():
                parts.setdefault(name, []).append(value.detach().cpu().numpy())
    if not parts:
        raise ValueError("validation has no samples with an observation at q0")
    arrays = {name: np.concatenate(value) for name, value in parts.items()}
    mask = np.ones(len(arrays["center_m"]), dtype=bool)
    overall = _metric_bundle(arrays, mask)
    overall.update({
        "input_sample_count": input_count,
        "anchor_qualified_fraction": float(mask.sum() / max(input_count, 1)),
        "raw_latest_pnp_nearest_truth_m": _stats(arrays["raw_latest_pnp_nearest_truth_m"]),
        "rigid_residual_m": _stats(arrays["rigid_residual_m"]),
    })
    return {
        "overall": overall,
        "strata": {
            "motion_class": {
                name: _metric_bundle(arrays, arrays["motion_class"] == index)
                for index, name in enumerate(MOTION_NAMES)
            },
            "distance": {
                "near_lt_3m": _metric_bundle(arrays, arrays["distance_m"] < 3.0),
                "mid_3_to_5m": _metric_bundle(
                    arrays, (arrays["distance_m"] >= 3.0) & (arrays["distance_m"] < 5.0)
                ),
                "far_ge_5m": _metric_bundle(arrays, arrays["distance_m"] >= 5.0),
            },
            "latest_visible_count": {
                str(count): _metric_bundle(arrays, arrays["latest_visible_count"] == count)
                for count in range(1, 5)
            },
        },
    }


def _selection_tuple(metrics: dict[str, object]) -> tuple[float, ...]:
    overall = metrics["overall"]
    return (
        float(overall["fixed_slot_position_m"]["p95"]),
        float(overall["center_m"]["p95"]),
        float(overall["phase_abs_deg"]["p95"]),
        float(overall["fixed_slot_position_m"]["p99"]),
    )


def _checkpoint(
    path: Path, model: CurrentPnPPoseAdapter, epoch: int,
    metrics: dict[str, object], provenance: dict[str, object], role: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    payload: dict[str, object] = {
        "model": model.state_dict(), "model_class": model.__class__.__name__,
        "model_config": model.config(), "epoch": epoch, "checkpoint_role": role,
        "validation": metrics, "selection_tuple": _selection_tuple(metrics),
        "provenance": provenance,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def _train_epoch(
    model: CurrentPnPPoseAdapter, loader: DataLoader,
    optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
    device: torch.device, args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    totals = {"objective": 0.0, "center": 0.0, "phase": 0.0, "gradient_norm": 0.0}
    count = 0
    input_count = 0
    amp_enabled = device.type == "cuda" and args.amp != "off"
    amp_dtype = torch.float16 if args.amp == "float16" else torch.bfloat16
    for raw in loader:
        input_count += int(raw["obs"].shape[0])
        batch = _prepare_batch(raw, device, args.anchor_tolerance_s)
        if batch is None:
            continue
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            output = model(
                batch["obs"], batch["obs_mask"], batch["event_mask"],
                batch["event_time_s"],
            )
            loss, parts = current_pose_loss(
                output, batch["target_q0"], model.decoder.geometry,
                huber_beta_m=args.huber_beta_m,
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Module A produced a non-finite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf")
        )
        scaler.step(optimizer)
        scaler.update()
        batch_count = int(batch["obs"].shape[0])
        values = {
            "objective": float(loss.detach().cpu()),
            "center": float(parts["center"].detach().cpu()),
            "phase": float(parts["phase"].detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
        }
        for name, value in values.items():
            totals[name] += value * batch_count
        count += batch_count
    if not count:
        raise ValueError("training has no samples with an observation at q0")
    return {
        **{name: value / count for name, value in totals.items()},
        "qualified_sample_count": count,
        "input_sample_count": input_count,
        "anchor_qualified_fraction": count / max(input_count, 1),
    }


def train(args: argparse.Namespace) -> Path:
    _seed(args.seed)
    git_state = _git_state()
    if bool(git_state["worktree_dirty"]) and not args.allow_dirty_worktree:
        raise ValueError("formal Module A training requires a clean worktree")
    dataset = Path(args.dataset).resolve()
    geometry, observation_mean, observation_std, contract_hashes = _load_contract(dataset)
    downstream = _downstream_contract(args.downstream_run)
    if downstream is None and not args.allow_dirty_worktree:
        raise ValueError("formal Module A training requires a frozen downstream v15 contract")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training output: {output}")
    output.mkdir(parents=True)
    train_ds = Stage3ShardDataset(
        dataset, "train", augment=not args.no_augment, seed=args.seed,
        shuffle=True, sample_limit=args.train_sample_limit,
    )
    validation_ds = Stage3ShardDataset(
        dataset, "validation", augment=False, seed=args.seed,
        shuffle=False, sample_limit=args.validation_sample_limit,
    )
    loader_options = {
        "batch_size": args.batch_size, "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, **loader_options)
    validation_loader = DataLoader(validation_ds, **loader_options)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CurrentPnPPoseAdapter(
        geometry, observation_mean, observation_std,
        channels=args.channels, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def lr_factor(epoch: int) -> float:
        if epoch < args.warmup_epochs:
            return max((epoch + 1) / max(args.warmup_epochs, 1), 0.02)
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.amp == "float16"
    )
    provenance: dict[str, object] = {
        "schema_version": "stage3-pnp-pose-adapter-run-v1",
        "status": "running", "pid_at_launch": os.getpid(),
        "dataset": str(dataset), "dataset_schema": "stage3-dataset-v4-observation",
        **contract_hashes,
        "test_accessed": False, "validation_split": "validation",
        "diagnostic_only": bool(args.allow_dirty_worktree),
        "same_time_label_contract": {
            "target": "future_position[:,0] only, where tau[0]=0 by dataset contract",
            "q1_to_q7_moved_to_device": False,
            "last_observation_must_equal_q0": True,
            "anchor_tolerance_s": args.anchor_tolerance_s,
        },
        "predictor_input_allowlist": [
            "normalized PnP xyz", "PnP sin/cos yaw", "candidate/event masks",
            "causal real event time",
        ],
        "ignored_because_unqualified": [
            "zero-filled reprojection RMS", "broken candidate-fraction channel",
        ],
        "forbidden_predictor_inputs": [
            "q0 physical truth", "q1-q7 future truth", "future PnP", "motion class",
            "exact velocity", "exact yaw rate", "tracked identity", "test shard",
        ],
        "objective": "q0 center SmoothL1 + geometry-radius-scaled full-phase SmoothL1",
        "checkpoint_selection": (
            "lexicographic fixed-slot q0 P95, center P95, full-phase P95, fixed-slot q0 P99"
        ),
        "canonical_slot_gate": {
            "fixed_relative_slot_loss": True,
            "unordered_set_metric_is_auxiliary_only": True,
            "reports_full_phase_and_modulo_90_phase": True,
            "reports_quarter_turn_alias_fraction": True,
        },
        "downstream_motion_model": downstream,
        "downstream_loaded_or_optimized": False,
        "config": vars(args), "trainable_parameters": trainable_parameter_count(model),
        "source_sha256": _source_hashes(),
        "environment": {
            "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "amp": args.amp,
        },
        **git_state,
    }
    manifest_path = output / "run_manifest.json"
    history_path = output / "stage3-pnp-pose-adapter-history.json"
    _write_json(manifest_path, provenance)
    initial_validation = _validate(model, validation_loader, device, args)
    history: list[dict[str, object]] = [{
        "epoch": 0, "validation": initial_validation,
        "selection_tuple": _selection_tuple(initial_validation),
        "eligible_for_best": False,
    }]
    _write_json(history_path, history)
    _checkpoint(
        output / f"stage3-pnp-pose-adapter-seed{args.seed}-initial.pt",
        model, 0, initial_validation, provenance, "untrained_initial_baseline",
    )
    best = (float("inf"),) * 4
    best_epoch = 0
    best_path = output / f"stage3-pnp-pose-adapter-seed{args.seed}-best.pt"
    stale = 0
    started = time.monotonic()
    epochs_completed = 0
    stop_reason = "epoch_limit"
    validation = initial_validation
    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        train_metrics = _train_epoch(model, train_loader, optimizer, scaler, device, args)
        validation = _validate(model, validation_loader, device, args)
        score = _selection_tuple(validation)
        if not np.isfinite(score).all():
            raise FloatingPointError("Module A produced non-finite validation metrics")
        scheduler.step()
        elapsed = time.monotonic() - started
        record = {
            "epoch": epoch + 1, "train": train_metrics, "validation": validation,
            "selection_tuple": score, "lr": scheduler.get_last_lr()[0],
            "elapsed_seconds": elapsed,
        }
        history.append(record)
        _write_json(history_path, history)
        print(json.dumps({
            "epoch": epoch + 1, "train_objective": train_metrics["objective"],
            "selection_tuple": score,
            "fixed_slot_p95_m": validation["overall"]["fixed_slot_position_m"]["p95"],
            "center_p95_m": validation["overall"]["center_m"]["p95"],
            "phase_p95_deg": validation["overall"]["phase_abs_deg"]["p95"],
            "phase_alias_fraction": validation["overall"]["phase_alias_fraction"],
            "elapsed_seconds": elapsed,
        }, sort_keys=True), flush=True)
        epochs_completed = epoch + 1
        if score < best:
            best, best_epoch, stale = score, epoch + 1, 0
            _checkpoint(best_path, model, epoch + 1, validation, provenance, "best")
        elif epoch + 1 >= args.early_stopping_warmup:
            stale += 1
        if epoch + 1 >= args.early_stopping_warmup and stale >= args.patience:
            stop_reason = "early_stopping"
            break
        if args.max_wall_minutes > 0 and elapsed >= args.max_wall_minutes * 60:
            stop_reason = "wall_time_limit"
            break
    last_path = output / f"stage3-pnp-pose-adapter-seed{args.seed}-last.pt"
    _checkpoint(
        last_path, model, epochs_completed, validation, provenance, "last",
        optimizer, scheduler,
    )
    if epochs_completed == args.epochs and stop_reason == "epoch_limit":
        epoch_path = output / f"stage3-pnp-pose-adapter-seed{args.seed}-epoch{args.epochs}.pt"
        _checkpoint(epoch_path, model, epochs_completed, validation, provenance, "epoch_limit")
    final = {
        **provenance, "status": "complete", "stop_reason": stop_reason,
        "epochs_completed": epochs_completed,
        "best": {
            "path": best_path.name, "epoch": best_epoch,
            "selection_tuple": list(best), "sha256": _sha256(best_path),
            "trained_checkpoint": best_epoch > 0,
        },
        "last": {"path": last_path.name, "sha256": _sha256(last_path)},
        "procedure_complete": bool(best_epoch > 0 and stop_reason != "wall_time_limit"),
        "qualified_training_candidate": False,
        "requires_manual_acceptance": True,
        "manual_acceptance_gates": [
            "fixed-slot and center validation improve over registered baselines",
            "full canonical phase and quarter-turn alias are acceptable",
            "major distance and visibility strata do not regress",
            "causal rolling-cache validation passes before v15 integration",
        ],
        "history": history_path.name, "history_sha256": _sha256(history_path),
    }
    _write_json(manifest_path, final)
    print(str(output), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--downstream-run", default="")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--early-stopping-warmup", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta-m", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--anchor-tolerance-s", type=float, default=1e-6)
    parser.add_argument("--amp", choices=("off", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--train-sample-limit", type=int, default=0)
    parser.add_argument("--validation-sample-limit", type=int, default=0)
    parser.add_argument("--max-wall-minutes", type=float, default=0.0)
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    args = parser.parse_args()
    positive = (
        args.epochs, args.patience, args.early_stopping_warmup,
        args.batch_size, args.channels,
    )
    if any(value < 1 for value in positive):
        parser.error("epoch, patience, batch, and channel settings must be positive")
    if args.anchor_tolerance_s < 0 or args.huber_beta_m <= 0:
        parser.error("anchor tolerance must be non-negative and Huber beta positive")
    train(args)


if __name__ == "__main__":
    main()
