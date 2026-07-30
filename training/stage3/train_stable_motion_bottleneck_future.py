"""Train the supervised stable-motion-bottleneck future predictor (v5)."""

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

from .anonymous_vehicle_motion_v2 import target_roles
from .cyclic_future_foundation import load_frozen_v19
from .motion_truth_supervision import (
    MOTION_TARGET_FIELD,
    NORMALIZED_MOTION_TARGET_FIELD,
    MotionTruthIndex,
    fit_motion_scales,
    normalize_attached_motion,
)
from .observable_future_pnp_ab import sha256_file, state_dict_sha256
from .pnp_q0_hypothesis_adapter import (
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .stable_motion_bottleneck_future import (
    StableMotionBottleneckAnonymousFutureModel,
    stable_motion_future_loss,
)
from .train_anonymous_vehicle_motion import (
    CHECKPOINT_INTERVAL,
    _cuda_amp_dtype,
    _distribution,
    _forward_only,
    _json_sha256,
    _restore_rng_state,
    _rng_state,
    _validate_bindings,
    frozen_upstream_batch,
)
from .train_causal_physical_ab import _git_state, _seed, _to_device
from .train_increment_invariant_anonymous_future import (
    HierarchicalSessionHistorySampler,
    _dataset,
    _history_label,
    _sequential_session_ids,
    _set_phase_lr,
    apply_bin_preserving_prefix_dropout,
    assert_frozen_upstream_unchanged,
    evaluate as evaluate_position,
)
from .train_pnp_window_mapper_distillation import _atomic_checkpoint, _atomic_json


RUN_SCHEMA = "stage3-stable-motion-bottleneck-future-oracle-pilot-v5"
STAGES = ("motion_state", "trajectory", "selector", "decoder_joint")
STATE_MODULES = ("context", "motion_state_head")
DECODER_MODULES = (
    "motion_state_encoder", "handle_encoder", "time_basis",
    "trajectory_coefficient_head",
)
SELECTOR_MODULES = ("role_coefficient_head",)
ALL_TRAINABLE_MODULES = STATE_MODULES + DECODER_MODULES + SELECTOR_MODULES


def motion_state_cells(dataset: Any) -> dict[tuple[int, str, str], list[int]]:
    """Balance motion, session, history support and stationary/active state."""
    result: dict[tuple[int, str, str], list[int]] = {}
    for part_index, part in enumerate(dataset.parts):
        active_count = part.tensors["pnp_s_event_mask"].to(torch.bool).sum(dim=1)
        target = part.tensors[MOTION_TARGET_FIELD]
        moving = target.abs().amax(dim=1) > 1e-5
        offset = dataset.offsets[part_index]
        for local, (session, count, is_moving) in enumerate(zip(
            part.session_ids, active_count.tolist(), moving.tolist(), strict=True,
        )):
            state = "active" if bool(is_moving) else "stationary"
            cell = f"{_history_label(int(count))}/{state}"
            key = (int(part.motion_class), str(session), cell)
            result.setdefault(key, []).append(offset + local)
    if not result or any(not values for values in result.values()):
        raise ValueError("motion-state sampler has an empty cell")
    return result


def _prepare_batch(
    mapper: torch.nn.Module,
    s_model: torch.nn.Module,
    h_model: torch.nn.Module,
    raw: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    batch = frozen_upstream_batch(mapper, s_model, h_model, raw)
    for name in (MOTION_TARGET_FIELD, NORMALIZED_MOTION_TARGET_FIELD):
        if name not in raw:
            raise ValueError(f"joined motion target missing: {name}")
        batch[name] = raw[name].detach()
    return batch


def stage_for_update(
    args: argparse.Namespace, global_update: int,
) -> tuple[str, int, int]:
    lengths = (
        ("motion_state", args.motion_state_updates),
        ("trajectory", args.trajectory_updates),
        ("selector", args.selector_updates),
        ("decoder_joint", args.decoder_joint_updates),
    )
    previous = 0
    for name, length in lengths:
        endpoint = previous + length
        if global_update <= endpoint:
            return name, global_update - previous, length
        previous = endpoint
    raise ValueError("global update exceeds fixed endpoint")


def stage_modules(stage: str) -> tuple[str, ...]:
    if stage == "motion_state":
        return STATE_MODULES
    if stage == "trajectory":
        return DECODER_MODULES
    if stage == "selector":
        return SELECTOR_MODULES
    if stage == "decoder_joint":
        return DECODER_MODULES + SELECTOR_MODULES
    raise ValueError(f"unknown stage: {stage}")


def stage_loss_weights(stage: str) -> dict[str, float]:
    if stage == "motion_state":
        return {"motion_weight": 1.0}
    if stage == "trajectory":
        return {
            "trajectory_weight": 1.0, "trend_weight": 0.25,
        }
    if stage == "selector":
        return {
            "role_weight": 1.0, "distance_risk_weight": 0.25,
        }
    if stage == "decoder_joint":
        return {
            "trajectory_weight": 1.0, "trend_weight": 0.25,
            "role_weight": 1.0, "distance_risk_weight": 0.25,
        }
    raise ValueError(f"unknown stage: {stage}")


def _module_parameters(
    model: StableMotionBottleneckAnonymousFutureModel,
    names: tuple[str, ...],
) -> list[torch.nn.Parameter]:
    return [
        parameter
        for name in names
        for parameter in getattr(model, name).parameters()
    ]


def configure_stage(
    model: StableMotionBottleneckAnonymousFutureModel, stage: str,
) -> None:
    allowed = {id(value) for value in _module_parameters(model, stage_modules(stage))}
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in allowed)
    for name in stage_modules(stage):
        getattr(model, name).train()


def inactive_module_hashes(
    model: StableMotionBottleneckAnonymousFutureModel, stage: str,
) -> dict[str, str]:
    active = set(stage_modules(stage))
    return {
        name: state_dict_sha256(getattr(model, name).state_dict())
        for name in ALL_TRAINABLE_MODULES if name not in active
    }


def assert_inactive_modules_unchanged(
    model: StableMotionBottleneckAnonymousFutureModel,
    expected: dict[str, str],
) -> None:
    changed = [
        name for name, digest in expected.items()
        if state_dict_sha256(getattr(model, name).state_dict()) != digest
    ]
    if changed:
        raise RuntimeError("inactive stage modules changed: " + ", ".join(changed))


def _require_runtime(args: argparse.Namespace) -> torch.device:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v5 pilot requires a CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return device


def _motion_distribution(values: list[np.ndarray]) -> dict[str, float | int]:
    return _distribution(
        np.concatenate(values).astype(np.float64, copy=False)
        if values else np.empty(0, dtype=np.float64)
    )


@torch.no_grad()
def evaluate_motion(
    model: StableMotionBottleneckAnonymousFutureModel,
    loader: DataLoader,
    mapper: torch.nn.Module,
    s_model: torch.nn.Module,
    h_model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    group_names = ("overall", "rotation", "combined", "stationary", "active")
    storage = {
        name: {
            "velocity": [], "yaw": [], "normalized": [], "oracle": [],
            "yaw_sign_correct": 0, "yaw_sign_count": 0,
            "velocity_cosine": [], "count": 0,
        }
        for name in group_names
    }
    sessions: dict[str, dict[str, Any]] = {}
    offset = 0
    for raw_cpu in loader:
        raw = _to_device(raw_cpu, device)
        amp = (
            torch.autocast("cuda", dtype=_cuda_amp_dtype())
            if device.type == "cuda" else nullcontext()
        )
        with amp:
            batch = _prepare_batch(mapper, s_model, h_model, raw)
            prediction = model(_forward_only(batch))
        target_physical = batch[MOTION_TARGET_FIELD]
        target_normalized = batch[NORMALIZED_MOTION_TARGET_FIELD]
        predicted_physical = prediction["motion_state_physical"]
        velocity_error = torch.linalg.vector_norm(
            predicted_physical[:, :3] - target_physical[:, :3], dim=-1,
        )
        yaw_error = (predicted_physical[:, 3] - target_physical[:, 3]).abs()
        normalized_error = (
            prediction["motion_state_normalized"] - target_normalized
        ).abs().mean(dim=-1)
        active = target_physical.abs().amax(dim=1) > 1e-5
        yaw_valid = target_physical[:, 3].abs() > 0.5
        velocity_valid = torch.linalg.vector_norm(target_physical[:, :2], dim=-1) > 0.1
        velocity_cosine = torch.nn.functional.cosine_similarity(
            predicted_physical[:, :2], target_physical[:, :2], dim=-1, eps=1e-8,
        )

        primary = prediction["primary_index"]
        relative_role = torch.arange(4, device=device)[None]
        ordered_handle = torch.remainder(primary[:, None] + relative_role, 4)
        ordered_relation = batch["q0_relation_m"].gather(
            1, ordered_handle.unsqueeze(-1).expand(-1, -1, 3),
        ).clone()
        ordered_relation[:, 0] = 0.0
        ordered_supported = batch["q0_supported"].gather(1, ordered_handle)
        oracle = model.decode_ordered(
            current_position_m=batch["current_position_m"],
            tau_s=batch["tau_s"],
            ordered_q0_relation_m=ordered_relation,
            ordered_q0_supported=ordered_supported,
            motion_state_normalized=target_normalized,
        )
        positive = batch["target_query_mask"] & (batch["tau_s"] > 0)
        role = target_roles(batch["target_switch_count"], positive)
        oracle_conditional = oracle["role_position_m"].gather(
            2, role[:, :, None, None].expand(-1, -1, 1, 3),
        ).squeeze(2)
        target_position = (
            batch["truth_current_position_m"][:, None]
            + batch["target_visible_delta_m"]
        )
        oracle_error = torch.linalg.vector_norm(
            oracle_conditional - target_position, dim=-1,
        )
        oracle_window = []
        for row in range(positive.shape[0]):
            oracle_window.append(oracle_error[row, positive[row]].mean())
        oracle_window_t = torch.stack(oracle_window)

        sample_groups = {
            "overall": torch.ones_like(active),
            "rotation": batch["motion_class"] == 2,
            "combined": batch["motion_class"] == 3,
            "stationary": ~active,
            "active": active,
        }
        for name, mask in sample_groups.items():
            if not bool(mask.any()):
                continue
            item = storage[name]
            item["velocity"].append(velocity_error[mask].float().cpu().numpy())
            item["yaw"].append(yaw_error[mask].float().cpu().numpy())
            item["normalized"].append(normalized_error[mask].float().cpu().numpy())
            item["oracle"].append(oracle_window_t[mask].float().cpu().numpy())
            sign_mask = mask & yaw_valid
            item["yaw_sign_correct"] += int((
                torch.sign(predicted_physical[:, 3])
                == torch.sign(target_physical[:, 3])
            )[sign_mask].sum())
            item["yaw_sign_count"] += int(sign_mask.sum())
            cosine_mask = mask & velocity_valid
            if bool(cosine_mask.any()):
                item["velocity_cosine"].append(
                    velocity_cosine[cosine_mask].float().cpu().numpy()
                )
            item["count"] += int(mask.sum())

        batch_sessions = _sequential_session_ids(loader.dataset, offset, len(active))
        offset += len(active)
        for row, session in enumerate(batch_sessions):
            item = sessions.setdefault(session, {
                "motion_class": int(batch["motion_class"][row]),
                "velocity": [], "yaw": [], "normalized": [],
            })
            item["velocity"].append(float(velocity_error[row]))
            item["yaw"].append(float(yaw_error[row]))
            item["normalized"].append(float(normalized_error[row]))

    result: dict[str, Any] = {}
    for name, item in storage.items():
        cosine = (
            np.concatenate(item["velocity_cosine"]).astype(np.float64, copy=False)
            if item["velocity_cosine"] else np.empty(0, dtype=np.float64)
        )
        result[name] = {
            "sample_count": item["count"],
            "velocity_vector_error_mps": _motion_distribution(item["velocity"]),
            "yaw_absolute_error_rad_s": _motion_distribution(item["yaw"]),
            "normalized_state_mae": _motion_distribution(item["normalized"]),
            "truth_state_oracle_conditional_window_error_m": _motion_distribution(
                item["oracle"],
            ),
            "yaw_sign_accuracy_abs_truth_gt_0_5": (
                item["yaw_sign_correct"] / item["yaw_sign_count"]
                if item["yaw_sign_count"] else None
            ),
            "planar_velocity_cosine": _distribution(cosine),
        }
    result["per_session"] = {
        session: {
            "motion_class": item["motion_class"],
            "velocity_vector_error_mps": _distribution(np.asarray(item["velocity"])),
            "yaw_absolute_error_rad_s": _distribution(np.asarray(item["yaw"])),
            "normalized_state_mae": _distribution(np.asarray(item["normalized"])),
        }
        for session, item in sorted(sessions.items())
    }
    if sessions:
        result["session_macro"] = {
            "session_count": len(sessions),
            "velocity_mean_mps": float(np.mean([
                np.mean(item["velocity"]) for item in sessions.values()
            ])),
            "yaw_mean_rad_s": float(np.mean([
                np.mean(item["yaw"]) for item in sessions.values()
            ])),
            "normalized_state_mae": float(np.mean([
                np.mean(item["normalized"]) for item in sessions.values()
            ])),
        }
    return result


def evaluate(
    model: StableMotionBottleneckAnonymousFutureModel,
    loader: DataLoader,
    mapper: torch.nn.Module,
    s_model: torch.nn.Module,
    h_model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "future_position": evaluate_position(
            model, loader, mapper, s_model, h_model, device,
        ),
        "motion_state": evaluate_motion(
            model, loader, mapper, s_model, h_model, device,
        ),
    }


def _checkpoint_payload(
    *,
    model: StableMotionBottleneckAnonymousFutureModel,
    optimizers: dict[str, torch.optim.Optimizer],
    scaler: torch.amp.GradScaler,
    provenance: dict[str, Any],
    contract_sha256: str,
    global_update: int,
    stage: str,
    stage_update: int,
    validation_history: list[dict[str, Any]],
    sampler: HierarchicalSessionHistorySampler,
    prefix_generator: torch.Generator,
    stage_endpoint: bool,
    fixed_endpoint: bool,
    run_id: str,
    gradient_isolation_verified: dict[str, bool],
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA,
        "model_class": type(model).__name__,
        "model_config": model.config,
        "model": model.state_dict(),
        "model_state_dict_sha256": state_dict_sha256(model.state_dict()),
        "optimizers": {name: value.state_dict() for name, value in optimizers.items()},
        "scaler": scaler.state_dict(),
        "progress": {
            "global_update": global_update, "stage": stage,
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
        args.motion_state_updates, args.trajectory_updates,
        args.selector_updates, args.decoder_joint_updates,
    )
    if min(stage_lengths) <= 0:
        raise ValueError("all four fixed stages need positive updates")
    if CHECKPOINT_INTERVAL != 150:
        raise RuntimeError("immutable recovery checkpoint interval changed")
    signature = inspect.signature(StableMotionBottleneckAnonymousFutureModel.forward)
    if "detach_motion_code" not in signature.parameters:
        raise RuntimeError("v5 forward lacks motion-gradient isolation")

    output = Path(args.output).resolve()
    resume = Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None
    if resume is None:
        if output.exists():
            raise FileExistsError(f"refusing existing non-resume output: {output}")
        output.mkdir(parents=True)
    elif not output.is_dir() or not resume.is_file():
        raise ValueError("resume requires existing output and checkpoint")
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    _seed(args.seed)
    device = _require_runtime(args)
    dataset_path = Path(args.dataset).resolve()
    manifest_path = dataset_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("test_accessed", True)):
        raise ValueError("v5 pilot refuses a dataset that accessed test")
    if manifest.get("oracle_association") is not True:
        raise ValueError("v5 pilot requires oracle-associated data")
    expected_truth_sha = manifest.get("truth_history_manifest_sha256")
    if not isinstance(expected_truth_sha, str):
        raise ValueError("paired dataset lacks its truth-history binding")
    dataset_manifest_sha256 = sha256_file(manifest_path)
    truth_index = MotionTruthIndex(
        args.truth_history, expected_manifest_sha256=expected_truth_sha,
    )
    train_dataset = _dataset(
        dataset_path, "train", sample_limit=args.train_limit_per_class,
    )
    validation_dataset = _dataset(
        dataset_path, "validation", sample_limit=args.validation_limit_per_class,
    )
    train_join = truth_index.attach(train_dataset, "train")
    validation_join = truth_index.attach(validation_dataset, "validation")
    motion_scale = fit_motion_scales(train_dataset)
    normalize_attached_motion(train_dataset, motion_scale)
    normalize_attached_motion(validation_dataset, motion_scale)
    sampler = HierarchicalSessionHistorySampler(
        motion_state_cells(train_dataset), seed=args.seed + 1,
    )
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

    model = StableMotionBottleneckAnonymousFutureModel(
        velocity_scale_mps=tuple(float(value) for value in motion_scale[:3]),
        yaw_rate_scale_rad_s=float(motion_scale[3]),
        channels=args.channels, dropout=args.dropout,
        message_layers=args.message_layers, basis_count=args.basis_count,
    ).to(device)
    optimizers = {
        "state": torch.optim.AdamW(
            _module_parameters(model, STATE_MODULES),
            lr=args.motion_state_learning_rate, weight_decay=args.weight_decay,
        ),
        "decoder": torch.optim.AdamW(
            _module_parameters(model, DECODER_MODULES),
            lr=args.trajectory_learning_rate, weight_decay=args.weight_decay,
        ),
        "selector": torch.optim.AdamW(
            _module_parameters(model, SELECTOR_MODULES),
            lr=args.selector_learning_rate, weight_decay=args.weight_decay,
        ),
    }
    amp_dtype = _cuda_amp_dtype()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    total_updates = sum(stage_lengths)
    source_paths = {
        "trainer_v5": Path(__file__).resolve(),
        "model_v5": Path(__file__).with_name("stable_motion_bottleneck_future.py").resolve(),
        "truth_join": Path(__file__).with_name("motion_truth_supervision.py").resolve(),
        "model_v4": Path(__file__).with_name("increment_invariant_anonymous_future.py").resolve(),
        "shared_runner": Path(__file__).with_name("train_anonymous_vehicle_motion.py").resolve(),
    }
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    provenance = {
        "diagnostic_only": True,
        "oracle_association": True,
        "deployable_pipeline": False,
        "test_accessed": False,
        "truth_used_as_forward_input": False,
        "truth_used_as_loss_target_only": True,
        "motion_state_semantics": "tracker/chassis vx,vy,vz,physical yaw_rate",
        "anonymous_reflection_changes_motion_label": False,
        "motion_state_scale": motion_scale.tolist(),
        "future_gradient_to_motion_encoder": False,
        "decoder_reads_high_dimensional_context": False,
        "physics_decoder": False,
        "truth_history": truth_index.provenance,
        "truth_join": {"train": train_join, "validation": validation_join},
        "dataset": {
            "path": str(dataset_path), "manifest_sha256": dataset_manifest_sha256,
            "train": train_dataset.audit, "validation": validation_dataset.audit,
        },
        "mapper": mapper_info, "s": s_info, "h": h_info,
        "mapper_h_compatibility": mismatch["mapper_h"],
        "dataset_provenance_compatibility": mismatch["dataset_manifest"],
        "frozen_initial_state_dict_sha256": frozen_initial,
        "sampler": {
            "strategy": (
                "equal motion/session/history/stationary-active hierarchical v1"
            ),
            "support": sampler.support,
        },
        "runtime": {
            "platform": platform.platform(), "python_executable": sys.executable,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device), "amp_dtype": str(amp_dtype),
            "tf32": True, "num_workers": 0,
        },
        "git": _git_state(),
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_sha256": source_hashes,
    }
    contract = {
        "schema_version": RUN_SCHEMA,
        "args": {
            name: value for name, value in vars(args).items()
            if name != "resume_checkpoint"
        },
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "truth_manifest_sha256": truth_index.manifest_sha256,
        "truth_key_set_sha256": truth_index.key_set_sha256,
        "truth_label_sha256": truth_index.label_sha256,
        "train_join_key_set_sha256": train_join["joined_key_set_sha256"],
        "validation_join_key_set_sha256": validation_join["joined_key_set_sha256"],
        "motion_state_scale": motion_scale.tolist(),
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
    validation_history: list[dict[str, Any]] = []
    isolation = {"future_to_state": False, "selector_to_decoder": False}
    if resume is not None:
        if resume.parent != checkpoint_dir:
            raise ValueError("resume checkpoint must be inside output/checkpoints")
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != RUN_SCHEMA:
            raise ValueError("resume checkpoint schema differs")
        if payload.get("contract_sha256") != contract_sha256:
            raise ValueError("resume checkpoint contract differs")
        global_update = int(payload["progress"]["global_update"])
        later = [
            path for path in checkpoint_dir.glob("checkpoint-update-*.pt")
            if int(path.stem.rsplit("-", 1)[-1]) > global_update
        ]
        if later:
            raise ValueError("resume must use the latest immutable checkpoint")
        model.load_state_dict(payload["model"], strict=True)
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][name])
        scaler.load_state_dict(payload["scaler"])
        validation_history = list(payload.get("validation_history", []))
        provenance = payload["provenance"]
        run_id = str(payload.get("run_id", run_id))
        isolation = dict(payload.get("gradient_isolation_verified", isolation))
        _restore_rng_state(payload["rng"], sampler, prefix_generator)
    else:
        validation_history.append({
            "global_update": 0, "stage": "initial",
            "metrics": evaluate(
                model, validation_loader, mapper, s_model, h_model, device,
            ),
        })

    manifest_payload: dict[str, Any] = {
        "schema_version": RUN_SCHEMA, "status": "running", "run_id": run_id,
        "provenance": provenance, "contract": contract,
        "contract_sha256": contract_sha256, "model_config": model.config,
        "fixed_schedule": dict(zip(STAGES, stage_lengths, strict=True)),
        "progress": {"global_update": global_update},
        "validation_history": validation_history,
    }
    _atomic_json(output / "run_manifest.json", manifest_payload)
    started = time.time()
    previous_stage: str | None = None
    inactive_hashes: dict[str, str] = {}
    while global_update < total_updates:
        next_update = global_update + 1
        stage, stage_update, stage_total = stage_for_update(args, next_update)
        if stage != previous_stage:
            configure_stage(model, stage)
            previous_stage = stage
            inactive_hashes = inactive_module_hashes(model, stage)
        indices = sampler.draw(args.batch_size)
        raw_cpu = default_collate([train_dataset[index] for index in indices])
        raw_cpu = apply_bin_preserving_prefix_dropout(
            raw_cpu, probability=args.prefix_dropout_probability,
            generator=prefix_generator,
        )
        raw = _to_device(raw_cpu, device)
        with torch.autocast("cuda", dtype=amp_dtype):
            batch = _prepare_batch(mapper, s_model, h_model, raw)
            prediction = model(
                _forward_only(batch), detach_motion_code=True,
                detach_selector_context=(stage == "decoder_joint"),
            )
            weights = stage_loss_weights(stage)
            loss, components = stable_motion_future_loss(
                prediction, batch, **weights,
            )

        if stage == "trajectory" and not isolation["future_to_state"]:
            gradient = torch.autograd.grad(
                loss, _module_parameters(model, STATE_MODULES),
                retain_graph=True, allow_unused=True,
            )
            if any(value is not None and bool(torch.any(value != 0)) for value in gradient):
                raise RuntimeError("future loss leaked into physical state estimator")
            isolation["future_to_state"] = True
        if stage == "decoder_joint" and not isolation["selector_to_decoder"]:
            with torch.autocast("cuda", dtype=amp_dtype):
                selector_only, _ = stable_motion_future_loss(
                    prediction, batch, **stage_loss_weights("selector"),
                )
            gradient = torch.autograd.grad(
                selector_only, _module_parameters(model, DECODER_MODULES),
                retain_graph=True, allow_unused=True,
            )
            if any(value is not None and bool(torch.any(value != 0)) for value in gradient):
                raise RuntimeError("selector loss leaked into trajectory decoder")
            isolation["selector_to_decoder"] = True

        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        if stage == "motion_state":
            active_optimizers = (optimizers["state"],)
            lr_state = _set_phase_lr(
                optimizers["state"], base_lr=args.motion_state_learning_rate,
                stage_update=stage_update, stage_total=stage_total,
            )
            lr_decoder = lr_selector = 0.0
        elif stage == "trajectory":
            active_optimizers = (optimizers["decoder"],)
            lr_decoder = _set_phase_lr(
                optimizers["decoder"], base_lr=args.trajectory_learning_rate,
                stage_update=stage_update, stage_total=stage_total,
            )
            lr_state = lr_selector = 0.0
        elif stage == "selector":
            active_optimizers = (optimizers["selector"],)
            lr_selector = _set_phase_lr(
                optimizers["selector"], base_lr=args.selector_learning_rate,
                stage_update=stage_update, stage_total=stage_total,
            )
            lr_state = lr_decoder = 0.0
        else:
            active_optimizers = (optimizers["decoder"], optimizers["selector"])
            lr_decoder = _set_phase_lr(
                optimizers["decoder"], base_lr=args.joint_trajectory_learning_rate,
                stage_update=stage_update, stage_total=stage_total,
            )
            lr_selector = _set_phase_lr(
                optimizers["selector"], base_lr=args.joint_selector_learning_rate,
                stage_update=stage_update, stage_total=stage_total,
            )
            lr_state = 0.0
        scaler.scale(loss).backward()
        for optimizer in active_optimizers:
            scaler.unscale_(optimizer)
        nonfinite = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
        ]
        if nonfinite:
            raise RuntimeError("non-finite gradients: " + ", ".join(nonfinite[:8]))
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            args.gradient_clip_norm,
        )
        for optimizer in active_optimizers:
            scaler.step(optimizer)
        scaler.update()
        if any(not bool(torch.isfinite(value).all()) for value in model.parameters()):
            raise RuntimeError("optimizer produced non-finite parameters")
        global_update = next_update

        if global_update % args.log_interval == 0 or stage_update == stage_total:
            print(json.dumps({
                "global_update": global_update, "stage": stage,
                "stage_update": stage_update,
                "objective": float(components["objective"].detach()),
                "motion": float(components["motion"].detach()),
                "trajectory": float(components["trajectory"].detach()),
                "role": float(components["role"].detach()),
                "lr_state": lr_state, "lr_decoder": lr_decoder,
                "lr_selector": lr_selector, "elapsed_s": time.time() - started,
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
            assert_frozen_upstream_unchanged(
                mapper, s_model, h_model, frozen_initial,
            )
            assert_inactive_modules_unchanged(model, inactive_hashes)
            checkpoint_path = checkpoint_dir / f"checkpoint-update-{global_update:06d}.pt"
            _atomic_checkpoint(checkpoint_path, _checkpoint_payload(
                model=model, optimizers=optimizers, scaler=scaler,
                provenance=provenance, contract_sha256=contract_sha256,
                global_update=global_update, stage=stage, stage_update=stage_update,
                validation_history=validation_history, sampler=sampler,
                prefix_generator=prefix_generator, stage_endpoint=stage_endpoint,
                fixed_endpoint=fixed_endpoint, run_id=run_id,
                gradient_isolation_verified=isolation,
            ))
            manifest_payload["progress"] = {
                "global_update": global_update, "stage": stage,
                "stage_update": stage_update, "latest_checkpoint": str(checkpoint_path),
            }
            manifest_payload["validation_history"] = validation_history
            manifest_payload["gradient_isolation_verified"] = isolation
            _atomic_json(output / "run_manifest.json", manifest_payload)

    assert_frozen_upstream_unchanged(mapper, s_model, h_model, frozen_initial)
    final_checkpoint = checkpoint_dir / f"checkpoint-update-{total_updates:06d}.pt"
    if not final_checkpoint.is_file():
        raise RuntimeError("fixed final endpoint checkpoint is missing")
    final_metrics = validation_history[-1]["metrics"]
    _atomic_json(output / "final_metrics.json", final_metrics)
    manifest_payload.update({
        "status": "complete", "stop_reason": "fixed_update_endpoint",
        "progress": {
            "global_update": total_updates,
            "latest_checkpoint": str(final_checkpoint),
        },
        "fixed_final_checkpoint": {
            "path": str(final_checkpoint), "sha256": sha256_file(final_checkpoint),
            "update": total_updates, "selected_by_validation": False,
        },
        "validation_history": validation_history,
        "final_validation": final_metrics,
        "frozen_final_state_dict_sha256": frozen_initial,
        "frozen_state_hashes_unchanged": True,
        "gradient_isolation_verified": isolation,
        "elapsed_s": time.time() - started,
    })
    _atomic_json(output / "run_manifest.json", manifest_payload)
    return final_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="v5 supervised stable-motion-bottleneck future pilot"
    )
    parser.add_argument("--diagnostic-oracle-association", action="store_true")
    parser.add_argument("--allow-mapper-h-mismatch", action="store_true")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--truth-history", required=True)
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
    parser.add_argument("--motion-state-updates", type=int, default=800)
    parser.add_argument("--trajectory-updates", type=int, default=1200)
    parser.add_argument("--selector-updates", type=int, default=600)
    parser.add_argument("--decoder-joint-updates", type=int, default=300)
    parser.add_argument("--motion-state-learning-rate", type=float, default=3e-4)
    parser.add_argument("--trajectory-learning-rate", type=float, default=3e-4)
    parser.add_argument("--selector-learning-rate", type=float, default=3e-4)
    parser.add_argument("--joint-trajectory-learning-rate", type=float, default=1e-4)
    parser.add_argument("--joint-selector-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--prefix-dropout-probability", type=float, default=0.75)
    parser.add_argument("--train-limit-per-class", type=int, default=0)
    parser.add_argument("--validation-limit-per-class", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=25)
    return parser


def main() -> None:
    checkpoint = train(build_parser().parse_args())
    print(checkpoint)


if __name__ == "__main__":
    main()
