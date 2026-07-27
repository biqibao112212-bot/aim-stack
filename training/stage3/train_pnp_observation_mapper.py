"""Train the small PnP-to-clean observation mapper with frozen S provenance."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .cyclic_future_foundation import load_frozen_v19
from .observable_future_pnp_ab import sha256_file, state_dict_sha256
from .pnp_observation_mapper import (
    AlignedAnchoredWindowPnPObservationMapper,
    AnchoredWindowPnPObservationMapper,
    CausalPnPObservationMapper,
    PnPObservationMappingDataset,
    WindowPnPObservationMapper,
)
from .pnp_q0_hypothesis_adapter import load_frozen_pnp_mapper
from .train_causal_physical_ab import _git_state, _seed, _to_device


RUN_SCHEMA = "stage3-pnp-observation-mapper-run-v1"


def _atomic_json(path: Path, payload: object) -> None:
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}-{time.time_ns()}")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pending.replace(path)


def _atomic_checkpoint(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite PnP mapper checkpoint: {path}")
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}-{time.time_ns()}")
    try:
        torch.save(payload, pending)
        pending.replace(path)
    finally:
        if pending.exists():
            pending.unlink()


def _stats(values: list[np.ndarray]) -> dict[str, float | int]:
    merged = np.concatenate(values).astype(np.float64, copy=False)
    if not merged.size or not np.isfinite(merged).all():
        raise ValueError("PnP mapper metric vector is empty or non-finite")
    return {
        "count": int(merged.size),
        "mean_m": float(merged.mean()),
        "p50_m": float(np.quantile(merged, 0.50)),
        "p95_m": float(np.quantile(merged, 0.95)),
        "p99_m": float(np.quantile(merged, 0.99)),
        "max_m": float(merged.max()),
    }


def mapping_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    huber_beta_m: float,
    mse_weight: float,
    tail_weight: float,
    tail_fraction: float,
    q0_primary_weight: float,
    primary_history_weight: float,
    primary_relative_weight: float,
    primary_increment_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = batch["pnp_s_obs_mask"].to(torch.bool)
    difference = prediction["corrected_obs_m"] - batch["clean_s_obs_m"]
    selected = difference[mask]
    if selected.numel() == 0:
        raise ValueError("PnP mapper batch has no paired observations")
    huber = F.smooth_l1_loss(
        selected, torch.zeros_like(selected), beta=huber_beta_m
    )
    squared_distance = selected.square().sum(dim=-1)
    mse = squared_distance.mean()
    count = max(1, int(math.ceil(squared_distance.numel() * tail_fraction)))
    tail = torch.topk(squared_distance, count, sorted=False).values.mean()
    q0_primary_mask = (
        mask & batch["pnp_s_primary_mask"].to(torch.bool)
        & batch["pnp_s_event_time_s"].abs().le(1e-6).unsqueeze(-1)
    )
    if bool(q0_primary_mask.any()):
        q0_primary_difference = difference[q0_primary_mask]
        q0_primary_huber = F.smooth_l1_loss(
            q0_primary_difference, torch.zeros_like(q0_primary_difference),
            beta=huber_beta_m,
        )
    else:
        q0_primary_huber = huber.new_zeros(())
    primary_history_mask = mask & batch["pnp_s_primary_mask"].to(torch.bool)
    primary_difference = difference[primary_history_mask]
    primary_history_huber = F.smooth_l1_loss(
        primary_difference, torch.zeros_like(primary_difference),
        beta=huber_beta_m,
    )
    primary_event_count = primary_history_mask.sum(dim=2)
    primary_event_valid = (
        batch["pnp_s_event_mask"].to(torch.bool)
        & primary_event_count.eq(1)
    )
    predicted_primary = torch.where(
        primary_history_mask.unsqueeze(-1), prediction["corrected_obs_m"],
        torch.zeros_like(prediction["corrected_obs_m"]),
    ).sum(dim=2)
    target_primary = torch.where(
        primary_history_mask.unsqueeze(-1), batch["clean_s_obs_m"],
        torch.zeros_like(batch["clean_s_obs_m"]),
    ).sum(dim=2)
    q0_event = (
        primary_event_valid
        & batch["pnp_s_event_time_s"].abs().le(1e-6)
    )
    if bool(torch.any(q0_event.sum(dim=1) != 1)):
        raise ValueError("PnP mapper primary trajectory requires one q0 event")
    predicted_q0 = (
        predicted_primary * q0_event.unsqueeze(-1).to(difference.dtype)
    ).sum(dim=1)
    target_q0 = (
        target_primary * q0_event.unsqueeze(-1).to(difference.dtype)
    ).sum(dim=1)
    relative_difference = (
        predicted_primary - predicted_q0.detach()[:, None, :]
        - target_primary + target_q0[:, None, :]
    )
    relative_mask = primary_event_valid & ~q0_event
    if bool(torch.any(relative_mask.sum(dim=1) < 1)):
        raise ValueError("PnP mapper trajectory requires a non-q0 history event")
    primary_relative_huber = F.smooth_l1_loss(
        relative_difference[relative_mask],
        torch.zeros_like(relative_difference[relative_mask]),
        beta=huber_beta_m,
    )
    same_primary_handle = (
        primary_history_mask[:, 1:] & primary_history_mask[:, :-1]
    ).any(dim=2)
    increment_mask = (
        primary_event_valid[:, 1:] & primary_event_valid[:, :-1]
        & same_primary_handle
    )
    increment_difference = (
        predicted_primary[:, 1:] - predicted_primary[:, :-1]
        - target_primary[:, 1:] + target_primary[:, :-1]
    )
    if bool(increment_mask.any()):
        selected_increment = increment_difference[increment_mask]
        primary_increment_huber = F.smooth_l1_loss(
            selected_increment, torch.zeros_like(selected_increment),
            beta=huber_beta_m,
        )
    else:
        primary_increment_huber = huber.new_zeros(())
    total = (
        huber + mse_weight * mse + tail_weight * tail
        + q0_primary_weight * q0_primary_huber
        + primary_history_weight * primary_history_huber
        + primary_relative_weight * primary_relative_huber
        + primary_increment_weight * primary_increment_huber
    )
    return total, {
        "huber": huber, "mse": mse, "tail_mse": tail,
        "q0_primary_huber": q0_primary_huber,
        "primary_history_huber": primary_history_huber,
        "primary_relative_huber": primary_relative_huber,
        "primary_increment_huber": primary_increment_huber,
    }


@torch.no_grad()
def evaluate(
    model: CausalPnPObservationMapper,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    raw: list[np.ndarray] = []
    corrected: list[np.ndarray] = []
    raw_q0: list[np.ndarray] = []
    corrected_q0: list[np.ndarray] = []
    raw_q0_primary: list[np.ndarray] = []
    corrected_q0_primary: list[np.ndarray] = []
    raw_q0_nonprimary: list[np.ndarray] = []
    corrected_q0_nonprimary: list[np.ndarray] = []
    raw_primary: list[np.ndarray] = []
    corrected_primary: list[np.ndarray] = []
    raw_nonprimary: list[np.ndarray] = []
    corrected_nonprimary: list[np.ndarray] = []
    raw_primary_relative: list[np.ndarray] = []
    corrected_primary_relative: list[np.ndarray] = []
    raw_primary_increment: list[np.ndarray] = []
    corrected_primary_increment: list[np.ndarray] = []
    residual_xyz: list[np.ndarray] = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        prediction = model(
            batch["pnp_s_obs_m"], batch["pnp_s_obs_mask"],
            batch["pnp_s_event_time_s"], batch["pnp_s_event_mask"],
        )
        mask = batch["pnp_s_obs_mask"].to(torch.bool)
        target = batch["clean_s_obs_m"]
        raw_error = torch.linalg.vector_norm(
            batch["pnp_s_obs_m"] - target, dim=-1
        )
        corrected_error = torch.linalg.vector_norm(
            prediction["corrected_obs_m"] - target, dim=-1
        )
        raw.append(raw_error[mask].cpu().numpy())
        corrected.append(corrected_error[mask].cpu().numpy())
        primary_mask = mask & batch["pnp_s_primary_mask"].to(torch.bool)
        nonprimary_mask = mask & ~batch["pnp_s_primary_mask"].to(torch.bool)
        raw_primary.append(raw_error[primary_mask].cpu().numpy())
        corrected_primary.append(corrected_error[primary_mask].cpu().numpy())
        if bool(nonprimary_mask.any()):
            raw_nonprimary.append(raw_error[nonprimary_mask].cpu().numpy())
            corrected_nonprimary.append(corrected_error[nonprimary_mask].cpu().numpy())
        primary_event_valid = (
            batch["pnp_s_event_mask"].to(torch.bool)
            & primary_mask.sum(dim=2).eq(1)
        )
        raw_track = torch.where(
            primary_mask.unsqueeze(-1), batch["pnp_s_obs_m"],
            torch.zeros_like(batch["pnp_s_obs_m"]),
        ).sum(dim=2)
        corrected_track = torch.where(
            primary_mask.unsqueeze(-1), prediction["corrected_obs_m"],
            torch.zeros_like(prediction["corrected_obs_m"]),
        ).sum(dim=2)
        clean_track = torch.where(
            primary_mask.unsqueeze(-1), target, torch.zeros_like(target)
        ).sum(dim=2)
        q0_event = (
            primary_event_valid
            & batch["pnp_s_event_time_s"].abs().le(1e-6)
        )
        if bool(torch.any(q0_event.sum(dim=1) != 1)):
            raise ValueError("PnP mapper evaluation requires one primary q0 event")
        q0_weight = q0_event.unsqueeze(-1).to(target.dtype)
        raw_q0_track = (raw_track * q0_weight).sum(dim=1)
        corrected_q0_track = (corrected_track * q0_weight).sum(dim=1)
        clean_q0_track = (clean_track * q0_weight).sum(dim=1)
        raw_relative_error = torch.linalg.vector_norm(
            raw_track - raw_q0_track[:, None, :]
            - clean_track + clean_q0_track[:, None, :], dim=-1,
        )
        corrected_relative_error = torch.linalg.vector_norm(
            corrected_track - corrected_q0_track[:, None, :]
            - clean_track + clean_q0_track[:, None, :], dim=-1,
        )
        relative_mask = primary_event_valid & ~q0_event
        if bool(torch.any(relative_mask.sum(dim=1) < 1)):
            raise ValueError("PnP mapper evaluation has no non-q0 history")
        raw_primary_relative.append(
            raw_relative_error[relative_mask].cpu().numpy()
        )
        corrected_primary_relative.append(
            corrected_relative_error[relative_mask].cpu().numpy()
        )
        same_primary_handle = (
            primary_mask[:, 1:] & primary_mask[:, :-1]
        ).any(dim=2)
        increment_mask = (
            primary_event_valid[:, 1:] & primary_event_valid[:, :-1]
            & same_primary_handle
        )
        raw_increment_error = torch.linalg.vector_norm(
            raw_track[:, 1:] - raw_track[:, :-1]
            - clean_track[:, 1:] + clean_track[:, :-1], dim=-1,
        )
        corrected_increment_error = torch.linalg.vector_norm(
            corrected_track[:, 1:] - corrected_track[:, :-1]
            - clean_track[:, 1:] + clean_track[:, :-1], dim=-1,
        )
        if bool(increment_mask.any()):
            raw_primary_increment.append(
                raw_increment_error[increment_mask].cpu().numpy()
            )
            corrected_primary_increment.append(
                corrected_increment_error[increment_mask].cpu().numpy()
            )
        residual_xyz.append(
            (prediction["corrected_obs_m"] - target)[mask].cpu().numpy()
        )
        q0_mask = mask & batch["pnp_s_event_time_s"].abs().le(1e-6).unsqueeze(-1)
        q0_primary = q0_mask & batch["pnp_s_primary_mask"].to(torch.bool)
        q0_nonprimary = q0_mask & ~batch["pnp_s_primary_mask"].to(torch.bool)
        if bool(q0_mask.any()):
            raw_q0.append(raw_error[q0_mask].cpu().numpy())
            corrected_q0.append(corrected_error[q0_mask].cpu().numpy())
        if bool(q0_primary.any()):
            raw_q0_primary.append(raw_error[q0_primary].cpu().numpy())
            corrected_q0_primary.append(corrected_error[q0_primary].cpu().numpy())
        if bool(q0_nonprimary.any()):
            raw_q0_nonprimary.append(raw_error[q0_nonprimary].cpu().numpy())
            corrected_q0_nonprimary.append(corrected_error[q0_nonprimary].cpu().numpy())
    xyz = np.concatenate(residual_xyz).astype(np.float64, copy=False)
    return {
        "raw_all_observation": _stats(raw),
        "corrected_all_observation": _stats(corrected),
        "raw_primary_history_observation": _stats(raw_primary),
        "corrected_primary_history_observation": _stats(corrected_primary),
        "raw_nonprimary_history_observation": _stats(raw_nonprimary),
        "corrected_nonprimary_history_observation": _stats(corrected_nonprimary),
        "raw_primary_relative_history": _stats(raw_primary_relative),
        "corrected_primary_relative_history": _stats(corrected_primary_relative),
        "raw_primary_increment": _stats(raw_primary_increment),
        "corrected_primary_increment": _stats(corrected_primary_increment),
        "raw_q0_observation": _stats(raw_q0),
        "corrected_q0_observation": _stats(corrected_q0),
        "raw_q0_primary_observation": _stats(raw_q0_primary),
        "corrected_q0_primary_observation": _stats(corrected_q0_primary),
        "raw_q0_nonprimary_observation": _stats(raw_q0_nonprimary),
        "corrected_q0_nonprimary_observation": _stats(corrected_q0_nonprimary),
        "corrected_residual_xyz_mean_m": xyz.mean(axis=0).tolist(),
        "corrected_residual_xyz_std_m": xyz.std(axis=0).tolist(),
    }


def _learning_rate(
    base: float, update: int, total_updates: int, warmup_updates: int,
    minimum: float,
) -> float:
    if update <= warmup_updates:
        return base * update / max(warmup_updates, 1)
    progress = (update - warmup_updates) / max(total_updates - warmup_updates, 1)
    floor = minimum / base
    return base * (
        floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    )


def train(args: argparse.Namespace) -> Path:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    frozen_s, s_provenance = load_frozen_v19(args.s_checkpoint)
    frozen_s_state = state_dict_sha256(frozen_s.state_dict())
    train_dataset = PnPObservationMappingDataset(
        args.dataset, "train", sample_limit=args.train_limit,
        motion_class=(args.motion_class if args.motion_class >= 0 else None),
        require_common=args.require_common,
    )
    validation_dataset = PnPObservationMappingDataset(
        args.dataset, "validation", sample_limit=args.validation_limit,
        motion_class=(args.motion_class if args.motion_class >= 0 else None),
        require_common=args.require_common,
    )
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        generator=train_generator, num_workers=args.workers,
        pin_memory=device.type == "cuda", drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    initial_mapper_provenance: dict[str, Any] | None = None
    initial_window_mapper_provenance: dict[str, Any] | None = None
    initial_mapper: torch.nn.Module | None = None
    if args.initial_checkpoint:
        initial_mapper, initial_mapper_provenance = load_frozen_pnp_mapper(
            args.initial_checkpoint
        )
        if (
            initial_mapper_provenance["provenance"]["dataset_manifest_sha256"]
            != train_dataset.manifest_sha256
        ):
            raise ValueError("initial PnP mapper dataset provenance differs")
        if (
            initial_mapper_provenance["provenance"]["frozen_s"][
                "state_dict_sha256"
            ] != s_provenance["state_dict_sha256"]
        ):
            raise ValueError("initial PnP mapper frozen S provenance differs")
    initial_window_mapper: torch.nn.Module | None = None
    if args.window_initial_checkpoint:
        initial_window_mapper, initial_window_mapper_provenance = (
            load_frozen_pnp_mapper(args.window_initial_checkpoint)
        )
        if not isinstance(initial_window_mapper, WindowPnPObservationMapper):
            raise ValueError("window initialization requires a window mapper")
        if (
            initial_window_mapper_provenance["provenance"][
                "dataset_manifest_sha256"
            ] != train_dataset.manifest_sha256
        ):
            raise ValueError("initial window mapper dataset provenance differs")
        if (
            initial_window_mapper_provenance["provenance"]["frozen_s"][
                "state_dict_sha256"
            ] != s_provenance["state_dict_sha256"]
        ):
            raise ValueError("initial window mapper frozen S provenance differs")
    if args.mapper_family in {"anchored-window", "aligned-anchored-window"}:
        if not isinstance(initial_mapper, CausalPnPObservationMapper):
            raise ValueError("anchored window mapper requires a causal anchor")
        window = WindowPnPObservationMapper(
            frozen_s.position_mean.detach().cpu(),
            frozen_s.position_std.detach().cpu(),
            channels=args.channels, dropout=args.dropout,
            history_events=32, history_scale_s=args.history_scale_s,
        )
        if initial_window_mapper is not None:
            if initial_window_mapper.config != window.config:
                raise ValueError(
                    "initial window mapper config differs from requested model"
                )
            window.load_state_dict(initial_window_mapper.state_dict(), strict=True)
        wrapper_class = (
            AlignedAnchoredWindowPnPObservationMapper
            if args.mapper_family == "aligned-anchored-window"
            else AnchoredWindowPnPObservationMapper
        )
        model = wrapper_class(initial_mapper, window)
    else:
        if initial_window_mapper is not None:
            raise ValueError(
                "window initialization is only valid for an anchored window mapper"
            )
        mapper_class = (
            CausalPnPObservationMapper
            if args.mapper_family == "causal" else WindowPnPObservationMapper
        )
        model = mapper_class(
            frozen_s.position_mean.detach().cpu(),
            frozen_s.position_std.detach().cpu(),
            channels=args.channels, dropout=args.dropout,
            history_events=32, history_scale_s=args.history_scale_s,
        )
        if initial_mapper is not None:
            if initial_mapper.config != model.config:
                raise ValueError(
                    "initial PnP mapper config differs from requested model"
                )
            model.load_state_dict(initial_mapper.state_dict(), strict=True)
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    epoch_updates = len(train_loader)
    planned_updates = args.epochs * epoch_updates
    total_updates = (
        min(planned_updates, args.max_updates)
        if args.max_updates > 0 else planned_updates
    )
    if total_updates <= 0:
        raise ValueError("PnP mapper training requires at least one update")

    git = _git_state()
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    update = 0
    started = time.time()
    stop = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {
            "objective": 0.0, "huber": 0.0, "mse": 0.0,
            "tail_mse": 0.0, "q0_primary_huber": 0.0,
            "primary_history_huber": 0.0,
            "primary_relative_huber": 0.0,
            "primary_increment_huber": 0.0,
        }
        batches = 0
        for raw_batch in train_loader:
            update += 1
            lr = _learning_rate(
                args.learning_rate, update, total_updates,
                args.warmup_updates, args.minimum_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch["pnp_s_obs_m"], batch["pnp_s_obs_mask"],
                batch["pnp_s_event_time_s"], batch["pnp_s_event_mask"],
            )
            objective, components = mapping_loss(
                prediction, batch, huber_beta_m=args.huber_beta_m,
                mse_weight=args.mse_weight, tail_weight=args.tail_weight,
                tail_fraction=args.tail_fraction,
                q0_primary_weight=args.q0_primary_weight,
                primary_history_weight=args.primary_history_weight,
                primary_relative_weight=args.primary_relative_weight,
                primary_increment_weight=args.primary_increment_weight,
            )
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            sums["objective"] += float(objective.detach())
            for name, value in components.items():
                sums[name] += float(value.detach())
            batches += 1
            if args.max_updates > 0 and update >= args.max_updates:
                stop = True
                break

        validate_now = (
            epoch == 1 or epoch % args.validation_interval == 0 or stop
            or epoch == args.epochs
        )
        if validate_now:
            metrics = evaluate(model, validation_loader, device)
            if args.primary_relative_weight > 0:
                relative = metrics["corrected_primary_relative_history"]
                q0_primary = metrics["corrected_q0_primary_observation"]
                increment = metrics["corrected_primary_increment"]
                selection = (
                    float(relative["p95_m"]), float(relative["p99_m"]),
                    float(q0_primary["p95_m"]), float(increment["p95_m"]),
                )
            else:
                corrected = (
                    metrics["corrected_primary_history_observation"]
                    if args.primary_history_weight > 0
                    else metrics["corrected_all_observation"]
                )
                selection = (
                    float(corrected["p95_m"]), float(corrected["p99_m"]),
                    float(corrected["mean_m"]),
                )
            checkpoint_name = f"epoch-{epoch:04d}-update-{update:06d}.pt"
            checkpoint_path = output / checkpoint_name
            checkpoint = {
                "schema_version": RUN_SCHEMA,
                "model_class": type(model).__name__,
                "epoch": epoch, "update": update,
                "model_config": model.config,
                "model": model.state_dict(),
                "validation": metrics, "selection": selection,
                "provenance": {
                    "dataset_manifest_path": str(
                        Path(args.dataset).resolve() / "dataset_manifest.json"
                    ),
                    "dataset_manifest_sha256": train_dataset.manifest_sha256,
                    "frozen_s": s_provenance,
                    "git": git,
                    "test_accessed": False,
                    "deployable_pipeline": False,
                    "oracle_association": True,
                    "initial_mapper": initial_mapper_provenance,
                    "initial_window_mapper": initial_window_mapper_provenance,
                },
            }
            _atomic_checkpoint(checkpoint_path, checkpoint)
            item = {
                "epoch": epoch, "update": update,
                "learning_rate": lr,
                "train": {name: value / max(batches, 1) for name, value in sums.items()},
                "validation": metrics,
                "checkpoint": checkpoint_name,
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
            history.append(item)
            if best is None or selection < tuple(best["selection"]):
                best = {
                    "epoch": epoch, "update": update,
                    "path": checkpoint_name,
                    "sha256": item["checkpoint_sha256"],
                    "selection": selection,
                    "validation": metrics,
                }
            progress = {
                "schema_version": RUN_SCHEMA,
                "status": "running" if not stop else "complete",
                "epoch": epoch, "update": update,
                "best": best, "history": history,
                "elapsed_s": time.time() - started,
                "train_sample_count": len(train_dataset),
                "validation_sample_count": len(validation_dataset),
                "dataset_manifest_sha256": train_dataset.manifest_sha256,
                "frozen_s": s_provenance,
                "frozen_s_verified_unchanged": (
                    state_dict_sha256(frozen_s.state_dict()) == frozen_s_state
                ),
                "model_config": model.config,
                "training_arguments": vars(args),
                "git": git,
                "test_accessed": False,
                "deployable_pipeline": False,
                "oracle_association": True,
                "initial_mapper": initial_mapper_provenance,
                "initial_window_mapper": initial_window_mapper_provenance,
            }
            _atomic_json(output / "run_progress.json", progress)
        if stop:
            break

    if best is None:
        raise RuntimeError("PnP mapper training produced no validation checkpoint")
    manifest = json.loads((output / "run_progress.json").read_text(encoding="utf-8"))
    manifest.update({
        "status": "complete",
        "stop_reason": "max_updates" if stop else "epoch_limit",
        "elapsed_s": time.time() - started,
        "frozen_s_verified_unchanged": (
            state_dict_sha256(frozen_s.state_dict()) == frozen_s_state
        ),
    })
    _atomic_json(output / "run_manifest.json", manifest)
    return output / "run_manifest.json"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", required=True)
    result.add_argument("--s-checkpoint", required=True)
    result.add_argument("--initial-checkpoint", default="")
    result.add_argument("--window-initial-checkpoint", default="")
    result.add_argument("--output", required=True)
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int, default=20260727)
    result.add_argument("--channels", type=int, default=48)
    result.add_argument(
        "--mapper-family", choices=(
            "causal", "window", "anchored-window", "aligned-anchored-window",
        ),
        default="causal",
    )
    result.add_argument("--dropout", type=float, default=0.05)
    result.add_argument("--history-scale-s", type=float, default=0.32)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--epochs", type=int, default=60)
    result.add_argument("--max-updates", type=int, default=0)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--minimum-learning-rate", type=float, default=3e-6)
    result.add_argument("--warmup-updates", type=int, default=200)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--gradient-clip-norm", type=float, default=1.0)
    result.add_argument("--huber-beta-m", type=float, default=0.01)
    result.add_argument("--mse-weight", type=float, default=0.25)
    result.add_argument("--tail-weight", type=float, default=0.05)
    result.add_argument("--tail-fraction", type=float, default=0.1)
    result.add_argument("--q0-primary-weight", type=float, default=2.0)
    result.add_argument("--primary-history-weight", type=float, default=0.0)
    result.add_argument("--primary-relative-weight", type=float, default=0.0)
    result.add_argument("--primary-increment-weight", type=float, default=0.0)
    result.add_argument("--validation-interval", type=int, default=2)
    result.add_argument("--train-limit", type=int, default=0)
    result.add_argument("--validation-limit", type=int, default=0)
    result.add_argument(
        "--motion-class", type=int, default=-1,
        help="optional selection-only motion class 0..3; never exposed to the model",
    )
    result.add_argument(
        "--require-common", action="store_true",
        help="selection-only strict S/F common coverage filter",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if not 0 < args.tail_fraction <= 1:
        raise ValueError("tail fraction must be within (0,1]")
    if min(
        args.learning_rate, args.minimum_learning_rate, args.huber_beta_m,
        args.gradient_clip_norm,
    ) <= 0:
        raise ValueError("positive PnP mapper optimization arguments required")
    if min(
        args.mse_weight, args.tail_weight, args.q0_primary_weight,
        args.primary_history_weight, args.primary_relative_weight,
        args.primary_increment_weight,
    ) < 0:
        raise ValueError("PnP mapper loss weights cannot be negative")
    if args.motion_class not in {-1, 0, 1, 2, 3}:
        raise ValueError("motion class must be -1 or 0..3")
    print(train(args))


if __name__ == "__main__":
    main()
