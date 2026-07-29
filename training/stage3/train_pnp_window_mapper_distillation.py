"""Task-distill a PnP past-window mapper through frozen S, H, and F.

The mapper still receives only PnP XYZ, observation mask, event time, and
event mask.  Clean observations and future targets exist solely on the loss
side.  This is an oracle-associated, diagnostic upper bound rather than a
deployable association pipeline.
"""

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
from .observable_future_pnp_ab import (
    ObservableFuturePnPSFDataset,
    canonicalize_direction_keep_c4,
    f_forward,
    load_observable_f_checkpoint,
    sha256_file,
    state_dict_sha256,
)
from .pnp_observation_mapper import AlignedAnchoredWindowPnPObservationMapper
from .pnp_q0_hypothesis_adapter import (
    compose_hypothesis_for_f,
    hypothesis_forward,
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .train_causal_physical_ab import _git_state, _seed, _to_device
from .train_pnp_q0_hypothesis_adapter import _s_forward


RUN_SCHEMA = "stage3-pnp-window-mapper-task-distillation-v1"
MAPPER_CHECKPOINT_SCHEMA = "stage3-pnp-observation-mapper-run-v1"
MAPPER_INPUT_FIELDS = (
    "pnp_s_obs_m",
    "pnp_s_obs_mask",
    "pnp_s_event_time_s",
    "pnp_s_event_mask",
)


def _atomic_json(path: Path, payload: object) -> None:
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}-{time.time_ns()}")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pending.replace(path)


def _atomic_checkpoint(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite distilled mapper: {path}")
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}-{time.time_ns()}")
    try:
        torch.save(payload, pending)
        pending.replace(path)
    finally:
        if pending.exists():
            pending.unlink()


def _stats(parts: list[np.ndarray]) -> dict[str, float | int]:
    if not parts:
        raise ValueError("task-distillation metric is empty")
    values = np.concatenate(parts).astype(np.float64, copy=False)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("task-distillation metric is empty or non-finite")
    return {
        "count": int(values.size),
        "mean_m": float(values.mean()),
        "p50_m": float(np.quantile(values, 0.50)),
        "p95_m": float(np.quantile(values, 0.95)),
        "p99_m": float(np.quantile(values, 0.99)),
        "max_m": float(values.max()),
    }


def _validate_batch_contract(batch: dict[str, torch.Tensor]) -> None:
    if not torch.equal(batch["candidate_step"], batch["pnp_candidate_step"]):
        raise ValueError("clean/PnP candidate steps differ")
    if not torch.equal(batch["candidate_mask"], batch["pnp_candidate_mask"]):
        raise ValueError("clean/PnP candidate masks differ")
    if not torch.equal(batch["tau_s"], batch["pnp_tau_s"]):
        raise ValueError("clean/PnP query times differ")


def gather_true_branch(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather the anonymous true candidate only after F has predicted all rows."""
    step = batch["candidate_step"].to(torch.long)
    target_step = batch["target_switch_count"].to(torch.long)
    candidate_mask = batch["candidate_mask"].to(torch.bool)
    eligible = batch["target_query_mask"].to(torch.bool) & (batch["tau_s"] > 0)
    matches = (
        step[:, None, :] == target_step[:, :, None]
    ) & candidate_mask[:, None, :]
    if bool(torch.any(eligible & (matches.sum(dim=-1) != 1))):
        raise ValueError("eligible query does not have exactly one valid true branch")
    true_row = matches.to(torch.long).argmax(dim=-1)
    gather = true_row[:, :, None, None].expand(-1, -1, 1, 3)
    true_position = prediction["conditional_position_m"].gather(
        2, gather
    ).squeeze(2)
    truth_position = (
        batch["current_position_m"][:, None, :]
        + batch["target_visible_delta_m"]
    )
    return true_position, truth_position, eligible, true_row


def _student_pipeline(
    mapper: AlignedAnchoredWindowPnPObservationMapper,
    s_model: torch.nn.Module,
    h_model: torch.nn.Module,
    f_model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[
    dict[str, torch.Tensor], dict[str, torch.Tensor],
    dict[str, torch.Tensor], dict[str, torch.Tensor],
]:
    mapped = mapper(*(batch[name] for name in MAPPER_INPUT_FIELDS))
    s_output = _s_forward(
        s_model, mapped["corrected_obs_m"], batch["pnp_s_obs_mask"],
        batch["pnp_s_primary_mask"], batch["pnp_s_event_mask"],
        batch["pnp_s_event_time_s"], batch["pnp_s_switch_step"],
    )
    h_output = hypothesis_forward(h_model, s_output)
    composed = compose_hypothesis_for_f(
        h_output, s_output["primary_index"], mapped["corrected_obs_m"],
        batch["pnp_s_obs_mask"], batch["pnp_s_primary_mask"],
        batch["pnp_candidate_step"], batch["pnp_s_event_mask"],
    )
    prediction = f_forward(
        f_model, batch, prefix="pnp_",
        history_position_rel_m=composed["history_position_rel_m"],
        current_position_m=composed["current_position_m"],
        candidate_relation_m=composed["candidate_relation_m"],
        candidate_confidence=composed["candidate_confidence"],
        detach_observation_inputs=False,
    )
    return mapped, s_output, composed, prediction


def _clean_teacher_pipeline(
    s_model: torch.nn.Module,
    h_model: torch.nn.Module,
    f_model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    s_output = _s_forward(
        s_model, batch["clean_s_obs_m"], batch["clean_s_obs_mask"],
        batch["clean_s_primary_mask"], batch["clean_s_event_mask"],
        batch["clean_s_event_time_s"], batch["clean_s_switch_step"],
    )
    h_output = hypothesis_forward(h_model, s_output)
    composed = compose_hypothesis_for_f(
        h_output, s_output["primary_index"], batch["clean_s_obs_m"],
        batch["clean_s_obs_mask"], batch["clean_s_primary_mask"],
        batch["candidate_step"], batch["clean_s_event_mask"],
    )
    prediction = f_forward(
        f_model, batch,
        history_position_rel_m=composed["history_position_rel_m"],
        current_position_m=composed["current_position_m"],
        candidate_relation_m=composed["candidate_relation_m"],
        candidate_confidence=composed["candidate_confidence"],
        detach_observation_inputs=True,
    )
    return composed, prediction


def task_distillation_loss(
    mapped: dict[str, torch.Tensor],
    student_composed: dict[str, torch.Tensor],
    student_prediction: dict[str, torch.Tensor],
    teacher_composed: dict[str, torch.Tensor],
    teacher_prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    huber_beta_m: float,
    future_weight: float,
    history_weight: float,
    current_weight: float,
    candidate_weight: float,
    observation_weight: float,
    future_target: str = "clean-teacher",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    student_true, truth_position, eligible, true_row = gather_true_branch(
        student_prediction, batch
    )
    teacher_true, _, teacher_eligible, teacher_row = gather_true_branch(
        teacher_prediction, batch
    )
    if not torch.equal(eligible, teacher_eligible) or not torch.equal(
        true_row, teacher_row
    ):
        raise RuntimeError("student and teacher true-branch contracts differ")
    if not bool(eligible.any()):
        raise ValueError("task-distillation batch has no positive-time query")
    if future_target == "clean-teacher":
        future_target_position = teacher_true
    elif future_target == "physical-truth":
        future_target_position = truth_position
    else:
        raise ValueError(f"unsupported future target: {future_target}")
    future_huber = F.smooth_l1_loss(
        student_true[eligible], future_target_position[eligible],
        beta=huber_beta_m,
    )

    history_mask = (
        batch["pnp_history_mask"].to(torch.bool)
        & batch["history_mask"].to(torch.bool)
    )
    if not bool(history_mask.any()):
        raise ValueError("task-distillation batch has no common history")
    history_huber = F.smooth_l1_loss(
        student_composed["history_position_rel_m"][history_mask],
        teacher_composed["history_position_rel_m"][history_mask],
        beta=huber_beta_m,
    )
    current_huber = F.smooth_l1_loss(
        student_composed["current_position_m"],
        teacher_composed["current_position_m"],
        beta=huber_beta_m,
    )
    candidate_mask = (
        batch["pnp_candidate_mask"].to(torch.bool)
        & batch["candidate_mask"].to(torch.bool)
    )
    candidate_huber = F.smooth_l1_loss(
        student_composed["candidate_relation_m"][candidate_mask],
        teacher_composed["candidate_relation_m"][candidate_mask],
        beta=huber_beta_m,
    )
    observation_mask = batch["pnp_s_obs_mask"].to(torch.bool)
    observation_huber = F.smooth_l1_loss(
        mapped["corrected_obs_m"][observation_mask],
        batch["clean_s_obs_m"][observation_mask],
        beta=huber_beta_m,
    )
    objective = (
        future_weight * future_huber
        + history_weight * history_huber
        + current_weight * current_huber
        + candidate_weight * candidate_huber
        + observation_weight * observation_huber
    )
    return objective, {
        "future_target_huber": future_huber,
        "history_teacher_huber": history_huber,
        "current_teacher_huber": current_huber,
        "candidate_teacher_huber": candidate_huber,
        "observation_clean_huber": observation_huber,
    }


@torch.no_grad()
def evaluate(
    mapper: AlignedAnchoredWindowPnPObservationMapper,
    s_model: torch.nn.Module,
    h_model: torch.nn.Module,
    f_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    mapper.eval()
    for model in (s_model, h_model, f_model):
        model.eval()
    conditional: list[np.ndarray] = []
    teacher_conditional: list[np.ndarray] = []
    teacher_gap: list[np.ndarray] = []
    hard: list[np.ndarray] = []
    correct: list[np.ndarray] = []
    current: list[np.ndarray] = []
    history_gap: list[np.ndarray] = []
    observation: list[np.ndarray] = []
    q0_identity_max = 0.0
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        _validate_batch_contract(batch)
        mapped, _, student_composed, student_prediction = _student_pipeline(
            mapper, s_model, h_model, f_model, batch
        )
        teacher_composed, teacher_prediction = _clean_teacher_pipeline(
            s_model, h_model, f_model, batch
        )
        student_true, truth, eligible, true_row = gather_true_branch(
            student_prediction, batch
        )
        teacher_true, _, _, _ = gather_true_branch(teacher_prediction, batch)
        conditional.append(torch.linalg.vector_norm(
            student_true[eligible] - truth[eligible], dim=-1
        ).cpu().numpy())
        teacher_conditional.append(torch.linalg.vector_norm(
            teacher_true[eligible] - truth[eligible], dim=-1
        ).cpu().numpy())
        teacher_gap.append(torch.linalg.vector_norm(
            student_true[eligible] - teacher_true[eligible], dim=-1
        ).cpu().numpy())
        hard_row = student_prediction["selected_candidate_row"]
        hard_gather = hard_row[:, :, None, None].expand(-1, -1, 1, 3)
        hard_position = student_prediction["conditional_position_m"].gather(
            2, hard_gather
        ).squeeze(2)
        hard.append(torch.linalg.vector_norm(
            hard_position[eligible] - truth[eligible], dim=-1
        ).cpu().numpy())
        selected_step = batch["candidate_step"].to(torch.long).gather(1, hard_row)
        correct.append((
            selected_step[eligible]
            == batch["target_switch_count"].to(torch.long)[eligible]
        ).cpu().numpy())
        current.append(torch.linalg.vector_norm(
            student_composed["current_position_m"]
            - batch["current_position_m"], dim=-1
        ).cpu().numpy())
        history_mask = (
            batch["pnp_history_mask"].to(torch.bool)
            & batch["history_mask"].to(torch.bool)
        )
        history_gap.append(torch.linalg.vector_norm(
            student_composed["history_position_rel_m"][history_mask]
            - teacher_composed["history_position_rel_m"][history_mask], dim=-1
        ).cpu().numpy())
        obs_mask = batch["pnp_s_obs_mask"].to(torch.bool)
        observation.append(torch.linalg.vector_norm(
            mapped["corrected_obs_m"][obs_mask]
            - batch["clean_s_obs_m"][obs_mask], dim=-1
        ).cpu().numpy())
        q0_event = (
            batch["pnp_s_event_mask"].to(torch.bool)
            & batch["pnp_s_event_time_s"].abs().le(1e-6)
        )
        anchor = mapper.anchor_mapper(*(
            batch[name] for name in MAPPER_INPUT_FIELDS
        ))["corrected_obs_m"]
        q0_mask = q0_event[:, :, None, None].expand_as(anchor)
        q0_identity_max = max(q0_identity_max, float((
            mapped["corrected_obs_m"][q0_mask] - anchor[q0_mask]
        ).abs().max()))
    correctness = np.concatenate(correct).astype(np.float64, copy=False)
    return {
        "conditional_position": _stats(conditional),
        "clean_teacher_conditional_position": _stats(teacher_conditional),
        "student_teacher_true_branch_gap": _stats(teacher_gap),
        "hard_position": _stats(hard),
        "switch_accuracy": float(correctness.mean()),
        "current_position": _stats(current),
        "history_relative_teacher_gap": _stats(history_gap),
        "mapped_observation": _stats(observation),
        "q0_anchor_identity_max_abs_m": q0_identity_max,
    }


def _learning_rate(
    base: float, update: int, total_updates: int, warmup_updates: int,
    minimum: float,
) -> float:
    if update <= warmup_updates:
        return base * update / max(warmup_updates, 1)
    progress = (update - warmup_updates) / max(
        total_updates - warmup_updates, 1
    )
    floor = minimum / base
    return base * (
        floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    )


def train(args: argparse.Namespace) -> Path:
    if not (
        args.diagnostic_only
        and args.allow_diagnostic_h
        and args.allow_mapper_h_mismatch
    ):
        raise ValueError(
            "task distillation requires explicit diagnostic H/mismatch opt-ins"
        )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_dataset = ObservableFuturePnPSFDataset(
        args.dataset, "train", motion_class=3, sample_limit=args.train_limit
    )
    validation_dataset = ObservableFuturePnPSFDataset(
        args.dataset, "validation", motion_class=3,
        sample_limit=args.validation_limit,
    )
    canonicalize_direction_keep_c4(train_dataset.tensors, train_dataset.pair_ids)
    canonicalize_direction_keep_c4(
        validation_dataset.tensors, validation_dataset.pair_ids
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=args.workers,
        pin_memory=device.type == "cuda", drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )

    mapper, mapper_provenance = load_frozen_pnp_mapper(args.mapper_checkpoint)
    if not isinstance(mapper, AlignedAnchoredWindowPnPObservationMapper):
        raise ValueError("task distillation requires the aligned anchored mapper")
    s_model, s_provenance = load_frozen_v19(args.s_checkpoint)
    h_model, h_provenance = load_frozen_hypothesis_adapter(
        args.h_checkpoint, allow_diagnostic=True
    )
    f_model, f_provenance = load_observable_f_checkpoint(args.f_checkpoint)
    if mapper_provenance["provenance"]["dataset_manifest_sha256"] != train_dataset.manifest_sha256:
        raise ValueError("initial mapper and task-distillation datasets differ")
    if h_provenance["provenance"]["dataset_manifest_sha256"] != train_dataset.manifest_sha256:
        raise ValueError("H and task-distillation datasets differ")
    if mapper_provenance["provenance"]["frozen_s"]["state_dict_sha256"] != s_provenance["state_dict_sha256"]:
        raise ValueError("initial mapper and supplied frozen S differ")
    h_frozen = h_provenance["provenance"]
    for name, loaded_hash in (
        ("frozen_s", s_provenance["state_dict_sha256"]),
        ("frozen_f", f_provenance["state_dict_sha256"]),
    ):
        if h_frozen[name]["state_dict_sha256"] != loaded_hash:
            raise ValueError(f"H and supplied {name} differ")
    mapper_h_mismatch = {
        "h_expected": h_frozen["frozen_mapper"]["state_dict_sha256"],
        "initial_mapper": mapper_provenance["state_dict_sha256"],
    }
    if mapper_h_mismatch["h_expected"] == mapper_h_mismatch["initial_mapper"]:
        raise ValueError("diagnostic mapper/H mismatch opt-in was unnecessary")

    mapper.to(device).eval().requires_grad_(False)
    mapper.window_smoother.requires_grad_(True).train()
    for model in (s_model, h_model, f_model):
        model.to(device).eval().requires_grad_(False)
    trainable = [
        parameter for parameter in mapper.window_smoother.parameters()
        if parameter.requires_grad
    ]
    if not trainable or any(
        parameter.requires_grad for parameter in mapper.anchor_mapper.parameters()
    ):
        raise RuntimeError("task distillation must optimize only the window smoother")
    frozen_before = {
        "anchor": state_dict_sha256(mapper.anchor_mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
        "f": state_dict_sha256(f_model.state_dict()),
    }
    initial_window_hash = state_dict_sha256(mapper.window_smoother.state_dict())
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    epoch_updates = len(train_loader)
    planned_updates = args.epochs * epoch_updates
    total_updates = min(planned_updates, args.max_updates) if args.max_updates > 0 else planned_updates
    if total_updates <= 0:
        raise ValueError("task distillation requires at least one update")

    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    update = 0
    stop = False
    started = time.time()
    git = _git_state()
    source_path = Path(__file__).resolve()
    for epoch in range(1, args.epochs + 1):
        mapper.train()
        sums = {
            "objective": 0.0,
            "future_target_huber": 0.0,
            "history_teacher_huber": 0.0,
            "current_teacher_huber": 0.0,
            "candidate_teacher_huber": 0.0,
            "observation_clean_huber": 0.0,
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
            _validate_batch_contract(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_composed, teacher_prediction = _clean_teacher_pipeline(
                    s_model, h_model, f_model, batch
                )
            mapped, _, student_composed, student_prediction = _student_pipeline(
                mapper, s_model, h_model, f_model, batch
            )
            objective, components = task_distillation_loss(
                mapped, student_composed, student_prediction,
                teacher_composed, teacher_prediction, batch,
                huber_beta_m=args.huber_beta_m,
                future_weight=args.future_weight,
                history_weight=args.history_weight,
                current_weight=args.current_weight,
                candidate_weight=args.candidate_weight,
                observation_weight=args.observation_weight,
                future_target=args.future_target,
            )
            objective.backward()
            if not any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                and bool(torch.any(parameter.grad != 0))
                for parameter in trainable
            ):
                raise RuntimeError("task loss produced no finite nonzero mapper gradient")
            if any(parameter.grad is not None for model in (s_model, h_model, f_model) for parameter in model.parameters()):
                raise RuntimeError("frozen S/H/F unexpectedly accumulated gradients")
            torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip_norm)
            optimizer.step()
            sums["objective"] += float(objective.detach())
            for name, value in components.items():
                sums[name] += float(value.detach())
            batches += 1
            if args.max_updates > 0 and update >= args.max_updates:
                stop = True
                break

        validate_now = (
            epoch == 1 or epoch % args.validation_interval == 0
            or stop or epoch == args.epochs
        )
        if validate_now:
            metrics = evaluate(
                mapper, s_model, h_model, f_model, validation_loader, device
            )
            selection = (
                float(metrics["conditional_position"]["p95_m"]),
                float(metrics["conditional_position"]["p99_m"]),
                float(metrics["student_teacher_true_branch_gap"]["p95_m"]),
                float(metrics["current_position"]["p95_m"]),
            )
            checkpoint_name = f"epoch-{epoch:04d}-update-{update:06d}.pt"
            checkpoint_path = output / checkpoint_name
            provenance = {
                "dataset_manifest_path": str(
                    Path(args.dataset).resolve() / "dataset_manifest.json"
                ),
                "dataset_manifest_sha256": train_dataset.manifest_sha256,
                "initial_mapper": mapper_provenance,
                "frozen_s": s_provenance,
                "frozen_h": h_provenance,
                "frozen_f": f_provenance,
                "mapper_h_compatibility_mismatch": mapper_h_mismatch,
                "optimizer_only_mapper_window": True,
                "future_target_contract": args.future_target,
                "teacher_future_only_in_loss": args.future_target == "clean-teacher",
                "future_truth_only_in_loss": args.future_target == "physical-truth",
                "future_truth_only_in_validation": args.future_target == "clean-teacher",
                "mapper_input_fields": list(MAPPER_INPUT_FIELDS),
                "canonical_direction_reflection_removed": True,
                "window_local_c4_origin_retained": True,
                "diagnostic_only": True,
                "diagnostic_reasons": [
                    "oracle_association",
                    "legacy_h_diagnostic_provenance",
                    "mapper_h_provenance_mismatch",
                    "dirty_training_source",
                ],
                "oracle_association": True,
                "deployable_pipeline": False,
                "test_accessed": False,
                "source_path": str(source_path),
                "source_sha256": sha256_file(source_path),
                "git": git,
            }
            checkpoint = {
                "schema_version": MAPPER_CHECKPOINT_SCHEMA,
                "training_schema": RUN_SCHEMA,
                "model_class": type(mapper).__name__,
                "model_config": mapper.config,
                "model": mapper.state_dict(),
                "epoch": epoch,
                "update": update,
                "validation": metrics,
                "selection": selection,
                "provenance": provenance,
            }
            _atomic_checkpoint(checkpoint_path, checkpoint)
            item = {
                "epoch": epoch,
                "update": update,
                "learning_rate": lr,
                "train": {
                    name: value / max(batches, 1) for name, value in sums.items()
                },
                "validation": metrics,
                "selection": selection,
                "checkpoint": checkpoint_name,
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
            history.append(item)
            if best is None or selection < tuple(best["selection"]):
                best = {
                    "epoch": epoch,
                    "update": update,
                    "path": checkpoint_name,
                    "sha256": item["checkpoint_sha256"],
                    "selection": selection,
                    "validation": metrics,
                }
            progress = {
                "schema_version": RUN_SCHEMA,
                "status": "running" if not stop else "complete",
                "epoch": epoch,
                "update": update,
                "best": best,
                "history": history,
                "elapsed_s": time.time() - started,
                "train_sample_count": len(train_dataset),
                "validation_sample_count": len(validation_dataset),
                "training_arguments": vars(args),
                "model_config": mapper.config,
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in trainable
                ),
                "initial_window_state_dict_sha256": initial_window_hash,
                "provenance": provenance,
            }
            _atomic_json(output / "run_progress.json", progress)
        if stop:
            break

    if best is None:
        raise RuntimeError("task distillation produced no checkpoint")
    frozen_unchanged = {
        "anchor": state_dict_sha256(mapper.anchor_mapper.state_dict()) == frozen_before["anchor"],
        "s": state_dict_sha256(s_model.state_dict()) == frozen_before["s"],
        "h": state_dict_sha256(h_model.state_dict()) == frozen_before["h"],
        "f": state_dict_sha256(f_model.state_dict()) == frozen_before["f"],
    }
    if not all(frozen_unchanged.values()):
        raise RuntimeError("task distillation changed a frozen state hash")
    manifest = json.loads(
        (output / "run_progress.json").read_text(encoding="utf-8")
    )
    manifest.update({
        "status": "complete",
        "stop_reason": "max_updates" if stop else "epoch_limit",
        "elapsed_s": time.time() - started,
        "frozen_state_hashes_unchanged": frozen_unchanged,
        "gate": {
            "conditional_p95_lt_350mm": (
                float(best["validation"]["conditional_position"]["p95_m"])
                < 0.350
            ),
            "current_p95_le_150mm": (
                float(best["validation"]["current_position"]["p95_m"])
                <= 0.150
            ),
            "q0_anchor_bit_exact": (
                float(best["validation"]["q0_anchor_identity_max_abs_m"]) == 0.0
            ),
        },
    })
    if args.future_target == "clean-teacher":
        manifest["gate"]["teacher_gap_p95_lt_150mm"] = (
            float(best["validation"]["student_teacher_true_branch_gap"]["p95_m"])
            < 0.150
        )
    manifest["gate_passed"] = all(manifest["gate"].values())
    _atomic_json(output / "run_manifest.json", manifest)
    return output / "run_manifest.json"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", required=True)
    result.add_argument("--mapper-checkpoint", required=True)
    result.add_argument("--s-checkpoint", required=True)
    result.add_argument("--h-checkpoint", required=True)
    result.add_argument("--f-checkpoint", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--diagnostic-only", action="store_true")
    result.add_argument("--allow-diagnostic-h", action="store_true")
    result.add_argument("--allow-mapper-h-mismatch", action="store_true")
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int, default=20260727)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--epochs", type=int, default=20)
    result.add_argument("--max-updates", type=int, default=1200)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    result.add_argument("--warmup-updates", type=int, default=100)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--gradient-clip-norm", type=float, default=1.0)
    result.add_argument("--huber-beta-m", type=float, default=0.05)
    result.add_argument("--future-weight", type=float, default=1.0)
    result.add_argument("--history-weight", type=float, default=0.5)
    result.add_argument("--current-weight", type=float, default=0.25)
    result.add_argument("--candidate-weight", type=float, default=0.1)
    result.add_argument("--observation-weight", type=float, default=0.1)
    result.add_argument(
        "--future-target",
        choices=("clean-teacher", "physical-truth"),
        default="clean-teacher",
    )
    result.add_argument("--validation-interval", type=int, default=1)
    result.add_argument("--train-limit", type=int, default=0)
    result.add_argument("--validation-limit", type=int, default=0)
    return result


def main() -> None:
    args = parser().parse_args()
    if min(
        args.learning_rate, args.minimum_learning_rate, args.gradient_clip_norm,
        args.huber_beta_m,
    ) <= 0:
        raise ValueError("positive task-distillation optimization values required")
    if min(
        args.future_weight, args.history_weight, args.current_weight,
        args.candidate_weight, args.observation_weight,
    ) < 0:
        raise ValueError("task-distillation weights cannot be negative")
    print(train(args))


if __name__ == "__main__":
    main()
