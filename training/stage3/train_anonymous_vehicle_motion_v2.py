"""Train the v2 visibility-aware anonymous vehicle motion pilot.

This remains an oracle-associated diagnostic experiment: Mapper, S and H are
frozen and the test split stays sealed.  Unlike v1, the primary selector target
is a modulo-four physical role and exact signed crossing count is auxiliary.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import inspect
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from .anonymous_vehicle_motion import FORWARD_FIELDS
from .anonymous_vehicle_motion_v2 import (
    VisibilityAwareAnonymousVehicleFutureModel,
    target_roles,
    visibility_aware_future_loss,
)
from .cyclic_future_foundation import load_frozen_v19
from .observable_future_pnp_ab import sha256_file, state_dict_sha256
from .pnp_q0_hypothesis_adapter import (
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .train_anonymous_vehicle_motion import (
    CHECKPOINT_INTERVAL,
    ERROR_THRESHOLDS_MM,
    BalancedMotionHistorySampler,
    _cuda_amp_dtype,
    _dataset,
    _forward_only,
    _json_sha256,
    _require_runtime,
    _restore_rng_state,
    _rng_state,
    _validate_bindings,
    apply_prefix_dropout,
    frozen_upstream_batch,
)
from .train_causal_physical_ab import _git_state, _seed, _to_device
from .train_pnp_window_mapper_distillation import _atomic_checkpoint, _atomic_json


RUN_SCHEMA = "stage3-visibility-aware-anonymous-motion-oracle-pilot-v2"
MOTION_CLASSES = (2, 3)
TRAJECTORY_MODULES = (
    "context", "handle_encoder", "time_basis",
    "trajectory_coefficient_head", "motion_regime_gate",
)
SELECTOR_MODULES = (
    "role_coefficient_head", "exact_selector_context",
    "direction_score_head", "crossing_interval_head", "temperature_head",
)
STAGES = ("trajectory", "selector", "joint", "recalibration")


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {"count": 0}
    if not np.isfinite(values).all():
        raise ValueError("validation produced non-finite errors")
    return {
        "count": int(values.size),
        "mean_m": float(values.mean()),
        "p50_m": float(np.percentile(values, 50)),
        "p95_m": float(np.percentile(values, 95)),
        "p99_m": float(np.percentile(values, 99)),
    }


def _coverage(values: np.ndarray) -> dict[str, float]:
    return {
        f"le_{threshold}mm": float(np.mean(values <= threshold / 1000.0))
        for threshold in ERROR_THRESHOLDS_MM
    }


def _window_values(value: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    count = query.sum(dim=1)
    keep = count > 0
    total = torch.where(query, value, torch.zeros_like(value)).sum(dim=1)
    return total[keep] / count[keep].to(value.dtype)


@torch.no_grad()
def evaluate(
    model: VisibilityAwareAnonymousVehicleFutureModel,
    loader: DataLoader,
    mapper: torch.nn.Module,
    s_model: torch.nn.Module,
    h_model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    names = ("overall", "rotation", "combined", "history_8_15", "switch_3_plus")
    storage: dict[str, dict[str, Any]] = {
        name: {
            "conditional": [], "hard": [], "anchor_relative": [],
            "conditional_window": [], "hard_window": [], "q0": [],
            "role_correct": 0, "exact_correct": 0, "count": 0,
        }
        for name in names
    }
    amp_enabled = device.type == "cuda"
    for raw_cpu in loader:
        raw = _to_device(raw_cpu, device)
        amp = (
            torch.autocast("cuda", dtype=_cuda_amp_dtype())
            if amp_enabled else nullcontext()
        )
        with amp:
            batch = frozen_upstream_batch(mapper, s_model, h_model, raw)
            prediction = model(_forward_only(batch))
        positive = batch["target_query_mask"] & (batch["tau_s"] > 0)
        role = target_roles(batch["target_switch_count"], positive)
        conditional = prediction["role_position_m"].gather(
            2, role[:, :, None, None].expand(-1, -1, 1, 3),
        ).squeeze(2)
        target = batch["truth_current_position_m"][:, None] + batch["target_visible_delta_m"]
        conditional_error = torch.linalg.vector_norm(conditional - target, dim=-1)
        hard_error = torch.linalg.vector_norm(prediction["position_m"] - target, dim=-1)
        relative_error = torch.linalg.vector_norm(
            (conditional - batch["current_position_m"][:, None])
            - batch["target_visible_delta_m"],
            dim=-1,
        )
        q0_error = torch.linalg.vector_norm(
            batch["current_position_m"] - batch["truth_current_position_m"], dim=-1,
        )
        role_correct = prediction["selected_role"] == role
        exact_correct = (
            prediction["selected_switch_step_aux"]
            == batch["target_switch_count"].to(torch.long)
        )
        active_count = prediction["history_active_count"]
        switch_count = (
            batch["history_switch_step"].abs()
            * batch["history_event_mask"].to(batch["history_switch_step"].dtype)
        ).sum(dim=1)
        sample_masks = {
            "overall": torch.ones_like(active_count, dtype=torch.bool),
            "rotation": batch["motion_class"] == 2,
            "combined": batch["motion_class"] == 3,
            "history_8_15": (active_count >= 8) & (active_count <= 15),
            "switch_3_plus": switch_count >= 3,
        }
        for name, sample_mask in sample_masks.items():
            query = positive & sample_mask[:, None]
            if not bool(query.any()):
                continue
            store = storage[name]
            store["conditional"].append(conditional_error[query].float().cpu().numpy())
            store["hard"].append(hard_error[query].float().cpu().numpy())
            store["anchor_relative"].append(relative_error[query].float().cpu().numpy())
            store["conditional_window"].append(
                _window_values(conditional_error, query).float().cpu().numpy()
            )
            store["hard_window"].append(
                _window_values(hard_error, query).float().cpu().numpy()
            )
            store["q0"].append(q0_error[sample_mask].float().cpu().numpy())
            store["role_correct"] += int((role_correct & query).sum())
            store["exact_correct"] += int((exact_correct & query).sum())
            store["count"] += int(query.sum())
    result: dict[str, Any] = {}
    for name, store in storage.items():
        arrays: dict[str, np.ndarray] = {}
        for metric_name in (
            "conditional", "hard", "anchor_relative",
            "conditional_window", "hard_window", "q0",
        ):
            parts = store[metric_name]
            arrays[metric_name] = (
                np.concatenate(parts).astype(np.float64, copy=False)
                if parts else np.empty(0, dtype=np.float64)
            )
        count = int(store["count"])
        result[name] = {
            "conditional_query_weighted": _distribution(arrays["conditional"]),
            "hard_query_weighted": _distribution(arrays["hard"]),
            "conditional_window_weighted": _distribution(arrays["conditional_window"]),
            "hard_window_weighted": _distribution(arrays["hard_window"]),
            "anchor_relative_trajectory": _distribution(arrays["anchor_relative"]),
            "q0_absolute": _distribution(arrays["q0"]),
            "role_accuracy_mod4": store["role_correct"] / count if count else None,
            "exact_signed_crossing_accuracy_aux": (
                store["exact_correct"] / count if count else None
            ),
            "conditional_coverage": _coverage(arrays["conditional"]) if count else {},
            "hard_coverage": _coverage(arrays["hard"]) if count else {},
        }
    return result


def _module_parameters(
    model: VisibilityAwareAnonymousVehicleFutureModel,
    names: tuple[str, ...],
) -> list[torch.nn.Parameter]:
    result: list[torch.nn.Parameter] = []
    for name in names:
        result.extend(getattr(model, name).parameters())
    return result


def stage_trainable_parameter_ids(
    model: VisibilityAwareAnonymousVehicleFutureModel, stage: str,
) -> set[int]:
    if stage == "trajectory":
        names = TRAJECTORY_MODULES
    elif stage in {"selector", "recalibration"}:
        names = SELECTOR_MODULES
    elif stage == "joint":
        names = TRAJECTORY_MODULES + SELECTOR_MODULES
    else:
        raise ValueError(f"unknown stage: {stage}")
    return {id(parameter) for parameter in _module_parameters(model, names)}


def configure_stage(
    model: VisibilityAwareAnonymousVehicleFutureModel, stage: str,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    allowed = stage_trainable_parameter_ids(model, stage)
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in allowed)
    active_modules = (
        TRAJECTORY_MODULES if stage == "trajectory"
        else SELECTOR_MODULES if stage in {"selector", "recalibration"}
        else TRAJECTORY_MODULES + SELECTOR_MODULES
    )
    for name in active_modules:
        getattr(model, name).train()


def stage_for_update(
    args: argparse.Namespace, global_update: int,
) -> tuple[str, int, int]:
    lengths = (
        ("trajectory", args.trajectory_updates),
        ("selector", args.selector_updates),
        ("joint", args.joint_updates),
        ("recalibration", args.recalibration_updates),
    )
    previous = 0
    for name, length in lengths:
        endpoint = previous + length
        if global_update <= endpoint:
            return name, global_update - previous, length
        previous = endpoint
    raise ValueError("global update exceeds fixed endpoint")


def _set_phase_lr(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    stage_update: int,
    stage_total: int,
) -> float:
    progress = (stage_update - 1) / max(stage_total - 1, 1)
    value = base_lr * (
        0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    )
    for group in optimizer.param_groups:
        group["lr"] = value
    return value


def stage_loss_weights(stage: str) -> dict[str, float]:
    if stage == "trajectory":
        return {
            "trajectory_weight": 1.0, "trend_weight": 0.25,
            "role_weight": 0.0, "exact_crossing_weight": 0.0,
            "distance_risk_weight": 0.0, "joint_position_weight": 0.0,
            "regime_balance_weight": 0.01, "regime_entropy_weight": 0.002,
        }
    if stage == "selector":
        return {
            "trajectory_weight": 0.0, "trend_weight": 0.0,
            "role_weight": 1.0, "exact_crossing_weight": 0.15,
            "distance_risk_weight": 0.25, "joint_position_weight": 0.0,
            "regime_balance_weight": 0.0, "regime_entropy_weight": 0.0,
        }
    if stage == "joint":
        return {
            "trajectory_weight": 1.0, "trend_weight": 0.25,
            "role_weight": 1.0, "exact_crossing_weight": 0.15,
            "distance_risk_weight": 0.25, "joint_position_weight": 0.1,
            "regime_balance_weight": 0.01, "regime_entropy_weight": 0.002,
        }
    if stage == "recalibration":
        return {
            "trajectory_weight": 0.0, "trend_weight": 0.0,
            "role_weight": 1.0, "exact_crossing_weight": 0.15,
            "distance_risk_weight": 0.25, "joint_position_weight": 0.1,
            "regime_balance_weight": 0.0, "regime_entropy_weight": 0.0,
        }
    raise ValueError(f"unknown stage: {stage}")


def _checkpoint_payload(
    *,
    model: VisibilityAwareAnonymousVehicleFutureModel,
    trajectory_optimizer: torch.optim.Optimizer,
    selector_optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    provenance: dict[str, Any],
    contract_sha256: str,
    global_update: int,
    stage: str,
    stage_update: int,
    validation_history: list[dict[str, Any]],
    sampler: BalancedMotionHistorySampler,
    prefix_generator: torch.Generator,
    stage_endpoint: bool,
    fixed_endpoint: bool,
    run_id: str,
    gradient_isolation_verified: bool,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA,
        "model_class": type(model).__name__,
        "model_config": model.config,
        "model": model.state_dict(),
        "model_state_dict_sha256": state_dict_sha256(model.state_dict()),
        "trajectory_optimizer": trajectory_optimizer.state_dict(),
        "selector_optimizer": selector_optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "progress": {
            "global_update": global_update,
            "stage": stage,
            "stage_update": stage_update,
        },
        "validation_history": validation_history,
        "provenance": provenance,
        "contract_sha256": contract_sha256,
        "rng": _rng_state(sampler, prefix_generator),
        "checkpoint_role": (
            "fixed_final_endpoint" if fixed_endpoint
            else "fixed_stage_endpoint" if stage_endpoint
            else "periodic_recovery"
        ),
        "fixed_endpoint": fixed_endpoint,
        "stage_endpoint": stage_endpoint,
        "checkpoint_selected_by_validation": False,
        "run_id": run_id,
        "gradient_isolation_verified": gradient_isolation_verified,
    }


def train(args: argparse.Namespace) -> Path:
    if not args.diagnostic_oracle_association:
        raise ValueError("requires explicit --diagnostic-oracle-association")
    if not args.allow_mapper_h_mismatch:
        raise ValueError("requires explicit --allow-mapper-h-mismatch")
    stage_lengths = (
        args.trajectory_updates, args.selector_updates,
        args.joint_updates, args.recalibration_updates,
    )
    if min(stage_lengths) <= 0:
        raise ValueError("all four fixed stages need positive updates")
    if CHECKPOINT_INTERVAL != 150:
        raise RuntimeError("immutable recovery checkpoint interval changed")
    if "detach_selector_context" not in inspect.signature(
        VisibilityAwareAnonymousVehicleFutureModel.forward
    ).parameters:
        raise RuntimeError("v2 joint isolation contract is missing")

    output = Path(args.output).resolve()
    resume = Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None
    if resume is None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing existing non-resume output: {output}")
        output.mkdir(parents=True, exist_ok=True)
    else:
        if not output.is_dir() or not resume.is_file():
            raise ValueError("resume requires an existing output and checkpoint")
        if resume.parent != output / "checkpoints":
            raise ValueError("resume checkpoint must be inside output/checkpoints")
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    _seed(args.seed)
    device = _require_runtime(args)
    dataset_path = Path(args.dataset).resolve()
    manifest_path = dataset_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("test_accessed", True)):
        raise ValueError("v2 pilot refuses a dataset that accessed test")
    if manifest.get("oracle_association") is not True:
        raise ValueError("v2 pilot requires oracle-associated data")
    if manifest.get("deployable_pipeline") is not False:
        raise ValueError("oracle-associated data cannot be deployable")
    dataset_manifest_sha256 = sha256_file(manifest_path)
    train_dataset = _dataset(
        dataset_path, "train", sample_limit=args.train_limit_per_class,
    )
    validation_dataset = _dataset(
        dataset_path, "validation", sample_limit=args.validation_limit_per_class,
    )
    sampler = BalancedMotionHistorySampler(train_dataset.strata(), seed=args.seed + 1)
    prefix_generator = torch.Generator().manual_seed(args.seed + 2)
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.validation_batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )

    mapper, mapper_info = load_frozen_pnp_mapper(args.mapper_checkpoint)
    s_model, s_info = load_frozen_v19(args.s_checkpoint)
    h_model, h_info = load_frozen_hypothesis_adapter(
        args.h_checkpoint, allow_diagnostic=True,
    )
    mismatch = _validate_bindings(
        dataset_manifest_sha256, mapper_info, s_info, h_info,
    )
    for frozen in (mapper, s_model, h_model):
        frozen.to(device).eval().requires_grad_(False)
    frozen_initial = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    }

    model = VisibilityAwareAnonymousVehicleFutureModel(
        channels=args.channels, dropout=args.dropout,
        message_layers=args.message_layers, basis_count=args.basis_count,
        latent_experts=args.latent_experts,
    ).to(device)
    trajectory_parameters = _module_parameters(model, TRAJECTORY_MODULES)
    selector_parameters = _module_parameters(model, SELECTOR_MODULES)
    trajectory_optimizer = torch.optim.AdamW(
        trajectory_parameters, lr=args.trajectory_learning_rate,
        weight_decay=args.weight_decay,
    )
    selector_optimizer = torch.optim.AdamW(
        selector_parameters, lr=args.selector_learning_rate,
        weight_decay=args.weight_decay,
    )
    amp_dtype = _cuda_amp_dtype()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    total_updates = sum(stage_lengths)
    source_paths = {
        "trainer_v2": Path(__file__).resolve(),
        "model_v2": Path(__file__).with_name("anonymous_vehicle_motion_v2.py").resolve(),
        "shared_runner_v1": Path(__file__).with_name("train_anonymous_vehicle_motion.py").resolve(),
    }
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    provenance = {
        "diagnostic_only": True,
        "oracle_association": True,
        "deployable_pipeline": False,
        "test_accessed": False,
        "fixed_endpoint": True,
        "checkpoint_selection": False,
        "oracle_lane_semantics": "window-local handles from truth association",
        "physical_id_forward_input": False,
        "motion_class_forward_input": False,
        "primary_selector_target": "relative physical role modulo four",
        "exact_signed_crossing": "auxiliary only; never controls final XYZ",
        "loss_reduction": "equal query mean per window; balanced motion/history sampler",
        "prefix_dropout_before_frozen_mapper_s_h": True,
        "positive_time_learned_loss_only": True,
        "motion_classes": list(MOTION_CLASSES),
        "dataset": {
            "path": str(dataset_path), "manifest_path": str(manifest_path),
            "manifest_sha256": dataset_manifest_sha256,
            "train": train_dataset.audit, "validation": validation_dataset.audit,
        },
        "mapper": mapper_info, "s": s_info, "h": h_info,
        "mapper_h_compatibility": mismatch["mapper_h"],
        "dataset_provenance_compatibility": mismatch["dataset_manifest"],
        "frozen_initial_state_dict_sha256": frozen_initial,
        "sampler": {
            "strategy": "equal_motion_x_history_bin_with_replacement_v1",
            "support": sampler.support,
        },
        "runtime": {
            "platform": platform.platform(), "python_executable": sys.executable,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device), "amp": True,
            "amp_dtype": str(amp_dtype), "gradient_finite_gate": True,
            "tf32": True, "num_workers": 0,
        },
        "git": _git_state(),
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_sha256": source_hashes,
    }
    contract = {
        "schema_version": RUN_SCHEMA,
        "args": {name: value for name, value in vars(args).items() if name != "resume_checkpoint"},
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "mapper_state_dict_sha256": mapper_info["state_dict_sha256"],
        "s_state_dict_sha256": s_info["state_dict_sha256"],
        "h_state_dict_sha256": h_info["state_dict_sha256"],
        "source_sha256": source_hashes,
        "fixed_total_updates": total_updates,
        "fixed_stage_order": list(STAGES),
    }
    contract_sha256 = _json_sha256(contract)
    run_id = str(uuid.uuid4())
    global_update = 0
    gradient_isolation_verified = False
    validation_history: list[dict[str, Any]] = []
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != RUN_SCHEMA:
            raise ValueError("resume checkpoint schema differs")
        if payload.get("contract_sha256") != contract_sha256:
            raise ValueError("resume checkpoint contract differs")
        global_update = int(payload["progress"]["global_update"])
        if global_update >= total_updates:
            raise ValueError("fixed endpoint is already complete")
        later = [
            path for path in checkpoint_dir.glob("checkpoint-update-*.pt")
            if int(path.stem.rsplit("-", 1)[-1]) > global_update
        ]
        if later:
            raise ValueError("resume must use the latest immutable checkpoint")
        model.load_state_dict(payload["model"], strict=True)
        trajectory_optimizer.load_state_dict(payload["trajectory_optimizer"])
        selector_optimizer.load_state_dict(payload["selector_optimizer"])
        scaler.load_state_dict(payload["scaler"])
        validation_history = list(payload.get("validation_history", []))
        provenance = payload["provenance"]
        run_id = str(payload.get("run_id", run_id))
        gradient_isolation_verified = bool(
            payload.get("gradient_isolation_verified", False)
        )
        _restore_rng_state(payload["rng"], sampler, prefix_generator)
    else:
        validation_history.append({
            "global_update": 0, "stage": "initial",
            "metrics": evaluate(model, validation_loader, mapper, s_model, h_model, device),
        })

    manifest_payload: dict[str, Any] = {
        "schema_version": RUN_SCHEMA, "status": "running", "run_id": run_id,
        "provenance": provenance, "contract": contract,
        "contract_sha256": contract_sha256, "model_config": model.config,
        "fixed_schedule": {
            "trajectory_updates": args.trajectory_updates,
            "selector_updates": args.selector_updates,
            "joint_updates": args.joint_updates,
            "recalibration_updates": args.recalibration_updates,
            "total_updates": total_updates,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "endpoint_selected_by_validation": False,
        },
        "progress": {"global_update": global_update},
        "validation_history": validation_history,
    }
    _atomic_json(output / "run_manifest.json", manifest_payload)
    started = time.time()
    previous_stage: str | None = None
    while global_update < total_updates:
        next_update = global_update + 1
        stage, stage_update, stage_total = stage_for_update(args, next_update)
        if stage != previous_stage:
            configure_stage(model, stage)
            previous_stage = stage
        indices = sampler.draw(args.batch_size)
        raw_cpu = default_collate([train_dataset[index] for index in indices])
        raw_cpu = apply_prefix_dropout(
            raw_cpu, probability=args.prefix_dropout_probability,
            minimum_events=args.minimum_history_events,
            generator=prefix_generator,
        )
        raw = _to_device(raw_cpu, device)
        with torch.autocast("cuda", dtype=amp_dtype):
            batch = frozen_upstream_batch(mapper, s_model, h_model, raw)
            prediction = model(
                _forward_only(batch), detach_selector_context=(stage == "joint"),
            )
            weights = stage_loss_weights(stage)
            loss, components = visibility_aware_future_loss(
                prediction, batch, **weights,
            )

        if stage == "joint" and not gradient_isolation_verified:
            selector_weights = stage_loss_weights("recalibration")
            with torch.autocast("cuda", dtype=amp_dtype):
                selector_only, _ = visibility_aware_future_loss(
                    prediction, batch, **selector_weights,
                )
            isolated = torch.autograd.grad(
                selector_only, trajectory_parameters,
                retain_graph=True, allow_unused=True,
            )
            if any(
                value is not None and bool(torch.any(value.detach() != 0))
                for value in isolated
            ):
                raise RuntimeError("joint selector loss leaked into trajectory parameters")
            gradient_isolation_verified = True

        trajectory_optimizer.zero_grad(set_to_none=True)
        selector_optimizer.zero_grad(set_to_none=True)
        active_optimizers: list[torch.optim.Optimizer] = []
        if stage in {"trajectory", "joint"}:
            lr_trajectory = _set_phase_lr(
                trajectory_optimizer,
                base_lr=(
                    args.trajectory_learning_rate if stage == "trajectory"
                    else args.joint_trajectory_learning_rate
                ),
                stage_update=stage_update, stage_total=stage_total,
            )
            active_optimizers.append(trajectory_optimizer)
        else:
            lr_trajectory = 0.0
        if stage in {"selector", "joint", "recalibration"}:
            selector_lr = (
                args.selector_learning_rate if stage == "selector"
                else args.joint_selector_learning_rate if stage == "joint"
                else args.recalibration_learning_rate
            )
            lr_selector = _set_phase_lr(
                selector_optimizer, base_lr=selector_lr,
                stage_update=stage_update, stage_total=stage_total,
            )
            active_optimizers.append(selector_optimizer)
        else:
            lr_selector = 0.0
        scaler.scale(loss).backward()
        for optimizer in active_optimizers:
            scaler.unscale_(optimizer)
        nonfinite_gradients = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
        ]
        if nonfinite_gradients:
            raise RuntimeError(
                "non-finite unscaled gradients: " + ", ".join(nonfinite_gradients[:8])
            )
        if stage in {"trajectory", "joint"}:
            torch.nn.utils.clip_grad_norm_(trajectory_parameters, args.gradient_clip_norm)
        if stage in {"selector", "joint", "recalibration"}:
            torch.nn.utils.clip_grad_norm_(selector_parameters, args.gradient_clip_norm)
        for optimizer in active_optimizers:
            scaler.step(optimizer)
        scaler.update()
        nonfinite_parameters = [
            name for name, parameter in model.named_parameters()
            if not bool(torch.isfinite(parameter).all())
        ]
        if nonfinite_parameters:
            raise RuntimeError(
                "optimizer produced non-finite parameters: "
                + ", ".join(nonfinite_parameters[:8])
            )
        global_update = next_update

        if global_update % args.log_interval == 0 or stage_update == stage_total:
            print(json.dumps({
                "global_update": global_update, "stage": stage,
                "stage_update": stage_update,
                "objective": float(components["objective"].detach()),
                "trajectory": float(components["trajectory"].detach()),
                "role": float(components["role"].detach()),
                "exact_crossing_aux": float(components["exact_crossing"].detach()),
                "lr_trajectory": lr_trajectory, "lr_selector": lr_selector,
                "elapsed_s": time.time() - started,
            }, sort_keys=True), flush=True)

        stage_endpoint = stage_update == stage_total
        fixed_endpoint = global_update == total_updates
        if stage_endpoint:
            validation_history.append({
                "global_update": global_update, "stage": stage,
                "metrics": evaluate(
                    model, validation_loader, mapper, s_model, h_model, device,
                ),
            })
            configure_stage(model, stage)
        if global_update % CHECKPOINT_INTERVAL == 0 or stage_endpoint or fixed_endpoint:
            checkpoint_path = checkpoint_dir / f"checkpoint-update-{global_update:06d}.pt"
            _atomic_checkpoint(checkpoint_path, _checkpoint_payload(
                model=model, trajectory_optimizer=trajectory_optimizer,
                selector_optimizer=selector_optimizer, scaler=scaler,
                provenance=provenance, contract_sha256=contract_sha256,
                global_update=global_update, stage=stage, stage_update=stage_update,
                validation_history=validation_history, sampler=sampler,
                prefix_generator=prefix_generator, stage_endpoint=stage_endpoint,
                fixed_endpoint=fixed_endpoint, run_id=run_id,
                gradient_isolation_verified=gradient_isolation_verified,
            ))
            manifest_payload["progress"] = {
                "global_update": global_update, "stage": stage,
                "stage_update": stage_update, "latest_checkpoint": str(checkpoint_path),
            }
            manifest_payload["validation_history"] = validation_history
            manifest_payload["gradient_isolation_verified"] = gradient_isolation_verified
            _atomic_json(output / "run_manifest.json", manifest_payload)

    frozen_final = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    }
    if frozen_final != frozen_initial:
        raise RuntimeError("training changed frozen Mapper/S/H state")
    final_checkpoint = checkpoint_dir / f"checkpoint-update-{total_updates:06d}.pt"
    if not final_checkpoint.is_file():
        raise RuntimeError("fixed final endpoint checkpoint is missing")
    final_metrics = validation_history[-1]["metrics"]
    _atomic_json(output / "final_metrics.json", final_metrics)
    manifest_payload.update({
        "status": "complete", "stop_reason": "fixed_update_endpoint",
        "progress": {"global_update": total_updates, "latest_checkpoint": str(final_checkpoint)},
        "fixed_final_checkpoint": {
            "path": str(final_checkpoint), "sha256": sha256_file(final_checkpoint),
            "update": total_updates, "selected_by_validation": False,
        },
        "validation_history": validation_history,
        "final_validation": final_metrics,
        "frozen_final_state_dict_sha256": frozen_final,
        "frozen_state_hashes_unchanged": True,
        "gradient_isolation_verified": gradient_isolation_verified,
        "elapsed_s": time.time() - started,
    })
    _atomic_json(output / "run_manifest.json", manifest_payload)
    return final_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="v2 visibility-aware anonymous vehicle motion diagnostic pilot"
    )
    parser.add_argument("--diagnostic-oracle-association", action="store_true")
    parser.add_argument("--allow-mapper-h-mismatch", action="store_true")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mapper-checkpoint", required=True)
    parser.add_argument("--s-checkpoint", required=True)
    parser.add_argument("--h-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-batch-size", type=int, default=96)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--message-layers", type=int, default=3)
    parser.add_argument("--basis-count", type=int, default=8)
    parser.add_argument("--latent-experts", type=int, default=3)
    parser.add_argument("--trajectory-updates", type=int, default=1200)
    parser.add_argument("--selector-updates", type=int, default=600)
    parser.add_argument("--joint-updates", type=int, default=300)
    parser.add_argument("--recalibration-updates", type=int, default=300)
    parser.add_argument("--trajectory-learning-rate", type=float, default=3e-4)
    parser.add_argument("--selector-learning-rate", type=float, default=3e-4)
    parser.add_argument("--joint-trajectory-learning-rate", type=float, default=1e-4)
    parser.add_argument("--joint-selector-learning-rate", type=float, default=1e-4)
    parser.add_argument("--recalibration-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--prefix-dropout-probability", type=float, default=0.75)
    parser.add_argument("--minimum-history-events", type=int, default=8)
    parser.add_argument("--train-limit-per-class", type=int, default=0)
    parser.add_argument("--validation-limit-per-class", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=25)
    return parser


def main() -> None:
    checkpoint = train(build_parser().parse_args())
    print(json.dumps({
        "status": "complete", "fixed_final_checkpoint": str(checkpoint),
        "selected_by_validation": False, "oracle_association": True,
        "deployable_pipeline": False, "test_accessed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
