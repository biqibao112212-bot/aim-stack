"""Train the paired real-PnP adapter arm or the joint S+F retraining arm.

Both arms use the same oracle-associated, window-anonymized common-coverage
dataset and the same combined-motion F parent.  Conditional true-branch error
is the selection metric; hard routing remains diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as TF
from torch.utils.data import DataLoader

from .cyclic_anchor_edge_loss import cyclic_anchor_edge_loss
from .cyclic_future_foundation import load_frozen_v19
from .observable_future_loss import (
    _target_candidate_row,
    observable_future_batch_metrics,
    observable_future_loss,
)
from .observable_future_pnp_ab import (
    CausalSelectedPnPAdapter,
    ObservableFuturePnPSFDataset,
    f_forward,
    load_observable_f_checkpoint,
    sf_compose,
    sha256_file,
    state_dict_sha256,
)
from .train_causal_physical_ab import _git_state, _seed, _to_device


RUN_SCHEMA = "stage3-observable-future-pnp-ab-run-v1"
SOURCE_FILES = (
    "observable_future_model.py",
    "observable_future_pnp_ab.py",
    "build_observable_future_pnp_sf_upper_bound_dataset.py",
    "train_observable_future_pnp_ab.py",
)


def _atomic_json(path: Path, payload: object) -> None:
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}-{time.time_ns()}")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for attempt in range(20):
        try:
            pending.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def _atomic_checkpoint(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite PnP A/B checkpoint: {path}")
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}-{time.time_ns()}")
    try:
        torch.save(payload, pending)
        pending.replace(path)
    finally:
        if pending.exists():
            pending.unlink()


def _percentiles(values: list[np.ndarray]) -> dict[str, float]:
    merged = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
    if merged.size == 0:
        raise ValueError("PnP A/B evaluation metric has no values")
    return {
        "count": int(merged.size),
        "mean_m": float(merged.mean()),
        "p50_m": float(np.percentile(merged, 50)),
        "p95_m": float(np.percentile(merged, 95)),
        "p99_m": float(np.percentile(merged, 99)),
        "max_m": float(merged.max()),
    }


def _absolute_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    absolute_view = dict(prediction)
    absolute_view["conditional_delta_m"] = (
        prediction["conditional_position_m"]
        - batch["current_position_m"][:, None, None, :]
    )
    return observable_future_loss(
        absolute_view,
        batch["candidate_step"], batch["candidate_mask"], batch["tau_s"],
        batch["target_switch_count"], batch["target_visible_delta_m"],
        batch["target_query_mask"],
        huber_beta_m=args.huber_beta_m,
        switch_weight=args.switch_weight,
        position_weight=args.position_weight,
        position_mse_weight=args.position_mse_weight,
        rate_weight=args.rate_weight,
        rate_huber_beta_mps=args.rate_huber_beta_mps,
        rate_tau_floor_s=args.rate_tau_floor_s,
        position_tail_weight=args.position_tail_weight,
        position_tail_fraction=args.position_tail_fraction,
        macro_balance_weight=args.macro_balance_weight,
        position_macro_balance_weight=args.position_macro_balance_weight,
        switch_focal_gamma=args.switch_focal_gamma,
    )


def _normalized_s_obs(
    model: torch.nn.Module,
    position_m: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mean = model.position_mean.to(position_m.dtype)
    std = model.position_std.to(position_m.dtype)
    normalized = (position_m - mean) / std
    return torch.where(mask.unsqueeze(-1), normalized, torch.zeros_like(normalized))


def _s_forward(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    mask = batch[f"{prefix}obs_mask"].to(torch.bool)
    return model(
        _normalized_s_obs(model, batch[f"{prefix}obs_m"], mask),
        mask, batch[f"{prefix}primary_mask"].to(torch.bool),
        batch[f"{prefix}event_mask"].to(torch.bool),
        batch[f"{prefix}event_time_s"], batch[f"{prefix}switch_step"],
    )


def _adapter_forward(
    adapter: CausalSelectedPnPAdapter,
    batch: dict[str, torch.Tensor],
    *,
    clean: bool,
    bypass: bool = False,
) -> dict[str, torch.Tensor]:
    prefix = "" if clean else "pnp_"
    return adapter(
        batch[f"{prefix}history_position_rel_m"],
        batch[f"{prefix}history_time_s"], batch[f"{prefix}history_dt_s"],
        batch[f"{prefix}history_switch_step"], batch[f"{prefix}history_mask"],
        batch[f"{prefix}current_position_m"],
        batch[f"{prefix}candidate_relation_m"], batch[f"{prefix}candidate_step"],
        batch[f"{prefix}candidate_confidence"], bypass=bypass,
    )


def _forward_arm_a(
    adapter: CausalSelectedPnPAdapter,
    frozen_f: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    clean: bool = False,
    bypass: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    corrected = _adapter_forward(adapter, batch, clean=clean, bypass=bypass)
    prefix = "" if clean else "pnp_"
    prediction = f_forward(
        frozen_f, batch, prefix=prefix,
        history_position_rel_m=corrected["history_position_rel_m"],
        current_position_m=corrected["current_position_m"],
        candidate_relation_m=corrected["candidate_relation_m"],
        candidate_confidence=corrected["candidate_confidence"],
        detach_observation_inputs=False,
    )
    return prediction, corrected


def _forward_arm_b(
    s_model: torch.nn.Module,
    f_model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    s_output = _s_forward(s_model, batch, "pnp_s_")
    composed = sf_compose(s_output, batch)
    prediction = f_forward(
        f_model, batch, prefix="pnp_",
        history_position_rel_m=composed["history_position_rel_m"],
        current_position_m=composed["current_position_m"],
        candidate_relation_m=composed["candidate_relation_m"],
        candidate_confidence=composed["candidate_confidence"],
        detach_observation_inputs=False,
    )
    return prediction, s_output, composed


def _eval_add(
    storage: dict[str, Any],
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    candidate_supported: torch.Tensor | None = None,
) -> None:
    view = dict(prediction)
    view["conditional_delta_m"] = (
        prediction["conditional_position_m"]
        - batch["current_position_m"][:, None, None, :]
    )
    metrics = observable_future_batch_metrics(
        view, batch["candidate_step"], batch["candidate_mask"],
        batch["target_switch_count"], batch["target_visible_delta_m"],
        batch["target_query_mask"],
    )
    storage["eligible"] += int(metrics["eligible_count"])
    storage["correct"] += int(metrics["correct_count"])
    storage["conditional"].append(metrics["conditional_error_m"].cpu().numpy())
    storage["hard"].append(metrics["hard_error_m"].cpu().numpy())
    if candidate_supported is not None:
        true_row, query_mask = _target_candidate_row(
            batch["candidate_step"], batch["candidate_mask"],
            batch["target_switch_count"], batch["target_query_mask"],
        )
        support = candidate_supported.gather(1, true_row).to(torch.bool)
        conditional = torch.linalg.vector_norm(
            view["conditional_delta_m"].gather(
                2, true_row[:, :, None, None].expand(-1, -1, 1, 3)
            ).squeeze(2) - batch["target_visible_delta_m"], dim=-1,
        )
        supported = query_mask & support
        unsupported = query_mask & ~support
        storage["supported_count"] += int(supported.sum())
        storage["unsupported_count"] += int(unsupported.sum())
        if bool(supported.any()):
            storage["supported"].append(conditional[supported].cpu().numpy())
        if bool(unsupported.any()):
            storage["unsupported"].append(conditional[unsupported].cpu().numpy())


def _new_eval_storage() -> dict[str, Any]:
    return {
        "eligible": 0, "correct": 0, "conditional": [], "hard": [],
        "supported_count": 0, "unsupported_count": 0,
        "supported": [], "unsupported": [],
    }


def _finish_eval(storage: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "eligible_query_count": storage["eligible"],
        "switch_accuracy": storage["correct"] / storage["eligible"],
        "conditional_position": _percentiles(storage["conditional"]),
        "hard_routed_position": _percentiles(storage["hard"]),
    }
    total_support = storage["supported_count"] + storage["unsupported_count"]
    if total_support:
        result["true_role_support"] = {
            "supported_count": storage["supported_count"],
            "unsupported_count": storage["unsupported_count"],
            "supported_fraction": storage["supported_count"] / total_support,
            "supported_conditional_position": (
                _percentiles(storage["supported"]) if storage["supported"] else None
            ),
            "unsupported_conditional_position": (
                _percentiles(storage["unsupported"]) if storage["unsupported"] else None
            ),
        }
    return result


@torch.no_grad()
def evaluate(
    arm: str,
    models: dict[str, torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    for model in models.values():
        model.eval()
    pnp_storage = _new_eval_storage()
    clean_storage = _new_eval_storage()
    clean_s_storage = _new_eval_storage()
    current_error: list[np.ndarray] = []
    for raw in loader:
        batch = _to_device(raw, device)
        if arm == "adapter":
            prediction, corrected = _forward_arm_a(
                models["adapter"], models["frozen_f"], batch
            )
            _eval_add(pnp_storage, prediction, batch)
            current_error.append(torch.linalg.vector_norm(
                corrected["current_position_m"] - batch["current_position_m"], dim=-1
            ).cpu().numpy())
            clean_prediction, _ = _forward_arm_a(
                models["adapter"], models["frozen_f"], batch, clean=True
            )
            _eval_add(clean_storage, clean_prediction, batch)
        else:
            prediction, _, composed = _forward_arm_b(
                models["s"], models["f"], batch
            )
            _eval_add(
                pnp_storage, prediction, batch,
                candidate_supported=composed["candidate_supported"],
            )
            current_error.append(torch.linalg.vector_norm(
                composed["current_position_m"] - batch["current_position_m"], dim=-1
            ).cpu().numpy())
            clean_prediction = f_forward(models["f"], batch)
            _eval_add(clean_storage, clean_prediction, batch)
            clean_s = _s_forward(models["s"], batch, "clean_s_")
            clean_composed = sf_compose(clean_s, {
                **batch,
                "pnp_current_position_m": batch["current_position_m"],
                "pnp_history_position_rel_m": batch["history_position_rel_m"],
            })
            clean_s_prediction = f_forward(
                models["f"], batch,
                history_position_rel_m=clean_composed["history_position_rel_m"],
                current_position_m=clean_composed["current_position_m"],
                candidate_relation_m=clean_composed["candidate_relation_m"],
                candidate_confidence=clean_composed["candidate_confidence"],
                detach_observation_inputs=False,
            )
            _eval_add(
                clean_s_storage, clean_s_prediction, batch,
                candidate_supported=clean_composed["candidate_supported"],
            )
    result = {
        "pnp": _finish_eval(pnp_storage),
        "clean_truth_s": _finish_eval(clean_storage),
        "current_position_error": _percentiles(current_error),
    }
    if arm == "adapter":
        result["clean_bypass_bit_exact_by_construction"] = True
    else:
        result["clean_s_f"] = _finish_eval(clean_s_storage)
    return result


def _learning_rate(base: float, update: int, args: argparse.Namespace) -> float:
    if update <= args.warmup_updates:
        return base * update / max(args.warmup_updates, 1)
    progress = (update - args.warmup_updates) / max(
        args.max_updates - args.warmup_updates, 1
    )
    floor = args.minimum_learning_rate / args.learning_rate
    return base * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def _train_batch_a(
    models: dict[str, torch.nn.Module],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction, corrected = _forward_arm_a(
        models["adapter"], models["frozen_f"], batch
    )
    future_loss, _ = _absolute_loss(prediction, batch, args)
    corrected_absolute = (
        corrected["current_position_m"][:, None, :]
        + corrected["history_position_rel_m"]
    )
    clean_absolute = (
        batch["current_position_m"][:, None, :]
        + batch["history_position_rel_m"]
    )
    history_reconstruction = TF.smooth_l1_loss(
        corrected_absolute, clean_absolute, beta=args.huber_beta_m
    )
    current_reconstruction = TF.smooth_l1_loss(
        corrected["current_position_m"], batch["current_position_m"],
        beta=args.huber_beta_m,
    )
    candidate_reconstruction = TF.smooth_l1_loss(
        corrected["candidate_relation_m"], batch["candidate_relation_m"],
        beta=args.huber_beta_m,
    )
    identity = _adapter_forward(models["adapter"], batch, clean=True)
    identity_loss = (
        identity["history_residual_m"].square().mean()
        + identity["current_residual_m"].square().mean()
        + (identity["candidate_relation_m"] - batch["candidate_relation_m"]).square().mean()
    )
    reconstruction = history_reconstruction + current_reconstruction + candidate_reconstruction
    objective = (
        future_loss + args.reconstruction_weight * reconstruction
        + args.clean_replay_weight * identity_loss
    )
    return objective, {
        "future": float(future_loss.detach()),
        "reconstruction": float(reconstruction.detach()),
        "clean_identity": float(identity_loss.detach()),
    }


def _train_batch_b(
    models: dict[str, torch.nn.Module],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction, s_output, _ = _forward_arm_b(models["s"], models["f"], batch)
    future_loss, _ = _absolute_loss(prediction, batch, args)
    anchor_loss, _ = cyclic_anchor_edge_loss(
        s_output, batch["pnp_s_truth_q0_m"], batch["motion_class"],
        huber_beta_m=args.huber_beta_m,
    )
    all_q0_loss = TF.smooth_l1_loss(
        s_output["q0_m"], batch["pnp_s_truth_q0_m"], beta=args.huber_beta_m
    )
    clean_prediction = f_forward(models["f"], batch)
    clean_f_loss, _ = _absolute_loss(clean_prediction, batch, args)
    clean_s = _s_forward(models["s"], batch, "clean_s_")
    with torch.no_grad():
        clean_s_parent = _s_forward(models["frozen_s"], batch, "clean_s_")
    clean_s_distill = (
        TF.smooth_l1_loss(clean_s["q0_m"], clean_s_parent["q0_m"], beta=args.huber_beta_m)
        + TF.smooth_l1_loss(
            clean_s["edge0_m"], clean_s_parent["edge0_m"], beta=args.huber_beta_m
        )
        + TF.mse_loss(clean_s["confidence"], clean_s_parent["confidence"])
    )
    s_loss = anchor_loss + args.all_q0_weight * all_q0_loss
    clean_loss = clean_f_loss + clean_s_distill
    objective = (
        future_loss + args.s_aux_weight * s_loss
        + args.clean_replay_weight * clean_loss
    )
    return objective, {
        "future": float(future_loss.detach()),
        "s_aux": float(s_loss.detach()),
        "clean_replay": float(clean_loss.detach()),
    }


def train(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    lock_path = output_dir / "run.lock"
    with lock_path.open("x", encoding="utf-8") as handle:
        json.dump({
            "pid": os.getpid(), "host": socket.gethostname(),
            "started_unix_s": time.time(), "arm": args.arm,
        }, handle, sort_keys=True)
    _seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    train_dataset = ObservableFuturePnPSFDataset(
        args.dataset, "train", motion_class=3, sample_limit=args.train_limit,
        allow_diagnostic=args.allow_diagnostic_dataset,
    )
    validation_dataset = ObservableFuturePnPSFDataset(
        args.dataset, "validation", motion_class=3,
        sample_limit=args.validation_limit,
        allow_diagnostic=args.allow_diagnostic_dataset,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )

    frozen_f, f_parent = load_observable_f_checkpoint(args.f_checkpoint)
    frozen_f.requires_grad_(False).eval().to(device)
    models: dict[str, torch.nn.Module]
    if args.arm == "adapter":
        adapter = CausalSelectedPnPAdapter(
            channels=args.adapter_channels, dropout=args.adapter_dropout
        ).to(device)
        models = {"adapter": adapter, "frozen_f": frozen_f}
        optimizer = torch.optim.AdamW(
            adapter.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        trainable_names = [f"adapter.{name}" for name, _ in adapter.named_parameters()]
        foundation = {"f": f_parent}
    else:
        s_model, s_parent_info = load_frozen_v19(args.s_checkpoint)
        frozen_s, frozen_s_info = load_frozen_v19(args.s_checkpoint)
        s_model.requires_grad_(True).train().to(device)
        frozen_s.requires_grad_(False).eval().to(device)
        f_model, f_train_parent = load_observable_f_checkpoint(args.f_checkpoint)
        f_model.requires_grad_(True).train().to(device)
        models = {
            "s": s_model, "f": f_model,
            "frozen_s": frozen_s, "frozen_f": frozen_f,
        }
        optimizer = torch.optim.AdamW([
            {"params": s_model.parameters(), "lr": args.s_learning_rate,
             "base_lr": args.s_learning_rate},
            {"params": f_model.parameters(), "lr": args.learning_rate,
             "base_lr": args.learning_rate},
        ], weight_decay=args.weight_decay)
        trainable_names = (
            [f"s.{name}" for name, _ in s_model.named_parameters()]
            + [f"f.{name}" for name, _ in f_model.named_parameters()]
        )
        foundation = {
            "s": s_parent_info, "s_frozen_copy": frozen_s_info,
            "f": f_train_parent,
        }

    source_dir = Path(__file__).resolve().parent
    dataset_manifest_path = Path(args.dataset).resolve() / "dataset_manifest.json"
    provenance = {
        "schema_version": RUN_SCHEMA,
        "arm": args.arm,
        "experiment_scope": "combined motion common-coverage paired PnP A/B",
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "train_sample_count": len(train_dataset),
        "validation_sample_count": len(validation_dataset),
        "test_accessed": False,
        "deployable_pipeline": False,
        "oracle_association": True,
        "conditional_metric_primary": True,
        "hard_metric_diagnostic_only": True,
        "differentiable_s_f_boundary": args.arm == "sf_joint",
        "trainable_parameter_names": trainable_names,
        "foundation": foundation,
        "git": _git_state(),
        "source_sha256": {
            name: sha256_file(source_dir / name) for name in SOURCE_FILES
        },
        "training_arguments": dict(vars(args)),
    }
    started = time.time()
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    update = 0
    initial = {
        "status": "running", "pid": os.getpid(), "epoch": 0, "update": 0,
        "heartbeat_unix_s": time.time(), "history": history, "best": best,
        **provenance,
    }
    _atomic_json(output_dir / "run_progress.json", initial)
    _atomic_json(output_dir / "run_manifest.json", initial)
    try:
        for epoch in range(1, args.epochs + 1):
            for name, model in models.items():
                if name.startswith("frozen"):
                    model.eval()
                else:
                    model.train()
            component_totals: dict[str, list[float]] = {}
            objectives: list[float] = []
            for raw in train_loader:
                batch = _to_device(raw, device)
                next_update = update + 1
                for group in optimizer.param_groups:
                    base = float(group.get("base_lr", args.learning_rate))
                    group["lr"] = _learning_rate(base, next_update, args)
                optimizer.zero_grad(set_to_none=True)
                if args.arm == "adapter":
                    objective, parts = _train_batch_a(models, batch, args)
                else:
                    objective, parts = _train_batch_b(models, batch, args)
                objective.backward()
                if args.gradient_clip_norm > 0:
                    parameters = [
                        parameter for group in optimizer.param_groups
                        for parameter in group["params"]
                    ]
                    torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip_norm)
                optimizer.step()
                update = next_update
                objectives.append(float(objective.detach()))
                for name, value in parts.items():
                    component_totals.setdefault(name, []).append(value)
                if update % args.heartbeat_updates == 0:
                    _atomic_json(output_dir / "run_progress.json", {
                        "status": "running", "pid": os.getpid(),
                        "epoch": epoch, "update": update,
                        "heartbeat_unix_s": time.time(), "history": history,
                        "best": best,
                        "cuda_max_memory_allocated_bytes": (
                            int(torch.cuda.max_memory_allocated(device))
                            if device.type == "cuda" else 0
                        ),
                        "cuda_max_memory_reserved_bytes": (
                            int(torch.cuda.max_memory_reserved(device))
                            if device.type == "cuda" else 0
                        ),
                        **provenance,
                    })
                if update >= args.max_updates:
                    break
            reached_limit = update >= args.max_updates
            validate_now = (
                epoch <= 3 or epoch % args.validation_interval == 0
                or reached_limit or epoch == args.epochs
            )
            if validate_now:
                validation = evaluate(args.arm, models, validation_loader, device)
                record = {
                    "epoch": epoch, "update": update,
                    "elapsed_s": time.time() - started,
                    "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                    "train_objective": float(np.mean(objectives)),
                    "train_components": {
                        name: float(np.mean(values))
                        for name, values in component_totals.items()
                    },
                    "validation": validation,
                }
                history.append(record)
                selection = (
                    float(validation["pnp"]["conditional_position"]["p95_m"]),
                    float(validation["current_position_error"]["p95_m"]),
                    float(validation["pnp"]["hard_routed_position"]["p95_m"]),
                )
                checkpoint_path = output_dir / (
                    f"epoch-{epoch:04d}-update-{update:06d}.pt"
                )
                payload: dict[str, Any] = {
                    "schema_version": RUN_SCHEMA,
                    "arm": args.arm, "epoch": epoch, "update": update,
                    "validation": validation, "selection": selection,
                    "optimizer": optimizer.state_dict(), "provenance": provenance,
                    "torch_rng_state": torch.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                }
                if device.type == "cuda":
                    payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
                if args.arm == "adapter":
                    payload["adapter_config"] = models["adapter"].config
                    payload["adapter"] = models["adapter"].state_dict()
                    if state_dict_sha256(models["frozen_f"].state_dict()) != f_parent["state_dict_sha256"]:
                        raise RuntimeError("adapter arm mutated frozen F")
                else:
                    payload["s_config"] = models["s"].config()
                    payload["s"] = models["s"].state_dict()
                    payload["f_config"] = models["f"].config
                    payload["f"] = models["f"].state_dict()
                    if state_dict_sha256(models["frozen_s"].state_dict()) != foundation["s_frozen_copy"]["state_dict_sha256"]:
                        raise RuntimeError("joint arm mutated frozen clean-S teacher")
                    if state_dict_sha256(models["frozen_f"].state_dict()) != f_parent["state_dict_sha256"]:
                        raise RuntimeError("joint arm mutated frozen clean-F reference")
                _atomic_checkpoint(checkpoint_path, payload)
                checkpoint = {
                    "path": checkpoint_path.name,
                    "sha256": sha256_file(checkpoint_path),
                    "epoch": epoch, "update": update,
                    "selection": list(selection), "validation": validation,
                }
                if best is None or tuple(checkpoint["selection"]) < tuple(best["selection"]):
                    best = checkpoint
                progress = {
                    "status": "running", "pid": os.getpid(),
                    "epoch": epoch, "update": update,
                    "heartbeat_unix_s": time.time(), "history": history,
                    "latest": checkpoint, "best": best,
                    "elapsed_s": time.time() - started,
                    "cuda_max_memory_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda" else 0
                    ),
                    "cuda_max_memory_reserved_bytes": (
                        int(torch.cuda.max_memory_reserved(device))
                        if device.type == "cuda" else 0
                    ),
                    **provenance,
                }
                _atomic_json(output_dir / "run_progress.json", progress)
                _atomic_json(output_dir / "run_manifest.json", progress)
                print(json.dumps(record, sort_keys=True), flush=True)
            if reached_limit:
                break
        if best is None:
            raise RuntimeError("PnP A/B training produced no validation checkpoint")
        final = {
            "status": "complete", "stop_reason": (
                "max_updates" if update >= args.max_updates else "epoch_limit"
            ),
            "acceptance_status": "comparison_pending_peer_arm_and_review",
            "pid": os.getpid(), "epoch": epoch, "update": update,
            "history": history, "best": best,
            "elapsed_s": time.time() - started,
            "cuda_max_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else 0
            ),
            "cuda_max_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda" else 0
            ),
            **provenance,
        }
        _atomic_json(output_dir / "run_progress.json", final)
        _atomic_json(output_dir / "run_manifest.json", final)
        return output_dir
    except Exception as error:
        _atomic_json(output_dir / "run_failed.json", {
            "status": "failed", "pid": os.getpid(), "epoch": locals().get("epoch", 0),
            "update": update, "error_type": type(error).__name__,
            "error": str(error), "elapsed_s": time.time() - started,
            **provenance,
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("adapter", "sf_joint"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--f-checkpoint", required=True)
    parser.add_argument("--s-checkpoint", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max-updates", type=int, default=10000)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--heartbeat-updates", type=int, default=50)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--allow-diagnostic-dataset", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--s-learning-rate", type=float, default=3e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-updates", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--adapter-channels", type=int, default=96)
    parser.add_argument("--adapter-dropout", type=float, default=0.05)
    parser.add_argument("--reconstruction-weight", type=float, default=10.0)
    parser.add_argument("--s-aux-weight", type=float, default=2.0)
    parser.add_argument("--all-q0-weight", type=float, default=1.0)
    parser.add_argument("--clean-replay-weight", type=float, default=0.2)
    parser.add_argument("--huber-beta-m", type=float, default=0.01)
    parser.add_argument("--switch-weight", type=float, default=1.0)
    parser.add_argument("--position-weight", type=float, default=50.0)
    parser.add_argument("--position-mse-weight", type=float, default=200.0)
    parser.add_argument("--rate-weight", type=float, default=0.005)
    parser.add_argument("--rate-huber-beta-mps", type=float, default=0.02)
    parser.add_argument("--rate-tau-floor-s", type=float, default=0.05)
    parser.add_argument("--position-tail-weight", type=float, default=0.2)
    parser.add_argument("--position-tail-fraction", type=float, default=0.1)
    parser.add_argument("--macro-balance-weight", type=float, default=0.25)
    parser.add_argument("--position-macro-balance-weight", type=float, default=0.25)
    parser.add_argument("--switch-focal-gamma", type=float, default=2.0)
    args = parser.parse_args()
    if args.arm == "sf_joint" and not args.s_checkpoint:
        parser.error("sf_joint requires --s-checkpoint")
    if (
        args.batch_size < 1 or args.workers < 0 or args.epochs < 1
        or args.max_updates < 1 or args.validation_interval < 1
        or args.heartbeat_updates < 1 or args.train_limit < 0
        or args.validation_limit < 0 or args.adapter_channels < 16
        or not 0 <= args.adapter_dropout < 1
        or min(
            args.learning_rate, args.s_learning_rate,
            args.minimum_learning_rate, args.huber_beta_m,
            args.rate_huber_beta_mps, args.rate_tau_floor_s,
        ) <= 0
        or args.minimum_learning_rate > args.learning_rate
        or min(
            args.weight_decay, args.gradient_clip_norm,
            args.reconstruction_weight, args.s_aux_weight,
            args.all_q0_weight, args.clean_replay_weight,
            args.switch_weight, args.position_weight,
            args.position_mse_weight, args.rate_weight,
            args.position_tail_weight, args.switch_focal_gamma,
        ) < 0
        or not 0 < args.position_tail_fraction <= 1
        or not 0 <= args.macro_balance_weight <= 1
        or not 0 <= args.position_macro_balance_weight <= 1
    ):
        parser.error("PnP A/B training arguments are invalid")
    print(train(args))


if __name__ == "__main__":
    main()
