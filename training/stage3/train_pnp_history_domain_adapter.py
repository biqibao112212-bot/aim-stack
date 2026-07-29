"""Train an anonymous PnP history adapter immediately before frozen F."""

from __future__ import annotations

import argparse
import json
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
from .pnp_history_domain_adapter import PnPHistoryDomainAdapter
from .pnp_q0_hypothesis_adapter import (
    compose_hypothesis_for_f,
    hypothesis_forward,
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .train_causal_physical_ab import _git_state, _seed, _to_device
from .train_pnp_q0_hypothesis_adapter import _s_forward
from .train_pnp_window_mapper_distillation import (
    _atomic_checkpoint,
    _atomic_json,
    _clean_teacher_pipeline,
    _learning_rate,
    _stats,
    _validate_batch_contract,
    gather_true_branch,
)


RUN_SCHEMA = "stage3-pnp-anonymous-history-domain-adapter-v1"
HISTORY_INPUT_FIELDS = (
    "history_position_rel_m",
    "pnp_history_time_s",
    "pnp_history_dt_s",
    "pnp_history_switch_step",
    "pnp_history_mask",
)


def _mapped_composed(
    mapper: torch.nn.Module,
    s_model: torch.nn.Module,
    h_model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    mapped = mapper(
        batch["pnp_s_obs_m"], batch["pnp_s_obs_mask"],
        batch["pnp_s_event_time_s"], batch["pnp_s_event_mask"],
    )
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
    return mapped, composed


def _adapted_prediction(
    adapter: PnPHistoryDomainAdapter,
    f_model: torch.nn.Module,
    composed: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    adapted = adapter(
        composed["history_position_rel_m"],
        batch["pnp_history_time_s"], batch["pnp_history_dt_s"],
        batch["pnp_history_switch_step"], batch["pnp_history_mask"],
    )
    prediction = f_forward(
        f_model, batch, prefix="pnp_",
        history_position_rel_m=adapted["corrected_history_position_rel_m"],
        current_position_m=composed["current_position_m"],
        candidate_relation_m=composed["candidate_relation_m"],
        candidate_confidence=composed["candidate_confidence"],
        detach_observation_inputs=False,
    )
    return adapted, prediction


def _hybrid_history_teacher_prediction(
    f_model: torch.nn.Module,
    student_composed: dict[str, torch.Tensor],
    clean_composed: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Change only history so the teacher is reachable by this adapter."""
    return f_forward(
        f_model, batch, prefix="pnp_",
        history_position_rel_m=clean_composed["history_position_rel_m"],
        current_position_m=student_composed["current_position_m"],
        candidate_relation_m=student_composed["candidate_relation_m"],
        candidate_confidence=student_composed["candidate_confidence"],
        detach_observation_inputs=True,
    )


def history_domain_loss(
    adapted: dict[str, torch.Tensor],
    student_prediction: dict[str, torch.Tensor],
    teacher_composed: dict[str, torch.Tensor],
    teacher_prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    huber_beta_m: float,
    future_weight: float,
    history_weight: float,
    residual_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    query_mask = batch["tau_s"] > 0
    dense_mask = (
        query_mask[:, :, None]
        & batch["candidate_mask"].to(torch.bool)[:, None, :]
    )
    if not bool(dense_mask.any()):
        raise ValueError("history adapter batch has no positive-time candidate")
    future_huber = F.smooth_l1_loss(
        student_prediction["conditional_position_m"][dense_mask],
        teacher_prediction["conditional_position_m"][dense_mask],
        beta=huber_beta_m,
    )
    history_mask = (
        batch["pnp_history_mask"].to(torch.bool)
        & batch["history_mask"].to(torch.bool)
    )
    if not bool(history_mask.any()):
        raise ValueError("history adapter batch has no common history")
    history_huber = F.smooth_l1_loss(
        adapted["corrected_history_position_rel_m"][history_mask],
        teacher_composed["history_position_rel_m"][history_mask],
        beta=huber_beta_m,
    )
    residual_huber = F.smooth_l1_loss(
        adapted["residual_m"][history_mask],
        torch.zeros_like(adapted["residual_m"][history_mask]),
        beta=huber_beta_m,
    )
    objective = (
        future_weight * future_huber
        + history_weight * history_huber
        + residual_weight * residual_huber
    )
    return objective, {
        "dense_candidate_teacher_huber": future_huber,
        "history_teacher_huber": history_huber,
        "residual_huber": residual_huber,
    }


@torch.no_grad()
def evaluate(
    adapter: PnPHistoryDomainAdapter,
    mapper: torch.nn.Module,
    s_model: torch.nn.Module,
    h_model: torch.nn.Module,
    f_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    for model in (adapter, mapper, s_model, h_model, f_model):
        model.eval()
    conditional: list[np.ndarray] = []
    teacher_conditional: list[np.ndarray] = []
    teacher_gap: list[np.ndarray] = []
    hard: list[np.ndarray] = []
    correct: list[np.ndarray] = []
    current: list[np.ndarray] = []
    raw_history_gap: list[np.ndarray] = []
    adapted_history_gap: list[np.ndarray] = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        _validate_batch_contract(batch)
        _, composed = _mapped_composed(mapper, s_model, h_model, batch)
        adapted, prediction = _adapted_prediction(
            adapter, f_model, composed, batch
        )
        teacher_composed, _ = _clean_teacher_pipeline(
            s_model, h_model, f_model, batch
        )
        teacher_prediction = _hybrid_history_teacher_prediction(
            f_model, composed, teacher_composed, batch
        )
        student_true, truth, eligible, _ = gather_true_branch(prediction, batch)
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
        hard_row = prediction["selected_candidate_row"]
        hard_gather = hard_row[:, :, None, None].expand(-1, -1, 1, 3)
        hard_position = prediction["conditional_position_m"].gather(
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
            composed["current_position_m"] - batch["current_position_m"], dim=-1
        ).cpu().numpy())
        history_mask = (
            batch["pnp_history_mask"].to(torch.bool)
            & batch["history_mask"].to(torch.bool)
        )
        target_history = teacher_composed["history_position_rel_m"]
        raw_history_gap.append(torch.linalg.vector_norm(
            composed["history_position_rel_m"][history_mask]
            - target_history[history_mask], dim=-1
        ).cpu().numpy())
        adapted_history_gap.append(torch.linalg.vector_norm(
            adapted["corrected_history_position_rel_m"][history_mask]
            - target_history[history_mask], dim=-1
        ).cpu().numpy())
    correctness = np.concatenate(correct).astype(np.float64, copy=False)
    return {
        "conditional_position": _stats(conditional),
        "hybrid_history_teacher_conditional_position": _stats(teacher_conditional),
        "student_teacher_true_branch_gap": _stats(teacher_gap),
        "hard_position": _stats(hard),
        "switch_accuracy": float(correctness.mean()),
        "current_position": _stats(current),
        "raw_history_teacher_gap": _stats(raw_history_gap),
        "adapted_history_teacher_gap": _stats(adapted_history_gap),
    }


def train(args: argparse.Namespace) -> Path:
    if not (
        args.diagnostic_only
        and args.allow_diagnostic_h
        and args.allow_mapper_h_mismatch
    ):
        raise ValueError("history adapter requires explicit diagnostic opt-ins")
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
    s_model, s_provenance = load_frozen_v19(args.s_checkpoint)
    h_model, h_provenance = load_frozen_hypothesis_adapter(
        args.h_checkpoint, allow_diagnostic=True
    )
    f_model, f_provenance = load_observable_f_checkpoint(args.f_checkpoint)
    if mapper_provenance["provenance"]["dataset_manifest_sha256"] != train_dataset.manifest_sha256:
        raise ValueError("mapper and history-adapter datasets differ")
    if h_provenance["provenance"]["dataset_manifest_sha256"] != train_dataset.manifest_sha256:
        raise ValueError("H and history-adapter datasets differ")
    if mapper_provenance["provenance"]["frozen_s"]["state_dict_sha256"] != s_provenance["state_dict_sha256"]:
        raise ValueError("mapper and supplied frozen S differ")
    h_frozen = h_provenance["provenance"]
    if h_frozen["frozen_s"]["state_dict_sha256"] != s_provenance["state_dict_sha256"]:
        raise ValueError("H and supplied frozen S differ")
    if h_frozen["frozen_f"]["state_dict_sha256"] != f_provenance["state_dict_sha256"]:
        raise ValueError("H and supplied frozen F differ")
    mapper_h_mismatch = {
        "h_expected": h_frozen["frozen_mapper"]["state_dict_sha256"],
        "loaded": mapper_provenance["state_dict_sha256"],
    }
    if mapper_h_mismatch["h_expected"] == mapper_h_mismatch["loaded"]:
        raise ValueError("diagnostic mapper/H mismatch opt-in was unnecessary")

    for model in (mapper, s_model, h_model, f_model):
        model.to(device).eval().requires_grad_(False)
    adapter = PnPHistoryDomainAdapter(
        channels=args.channels, dropout=args.dropout,
        history_events=32, position_scale_m=args.position_scale_m,
        history_scale_s=args.history_scale_s,
        switch_scale=args.switch_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    frozen_before = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
        "f": state_dict_sha256(f_model.state_dict()),
    }
    epoch_updates = len(train_loader)
    planned_updates = args.epochs * epoch_updates
    total_updates = min(planned_updates, args.max_updates) if args.max_updates > 0 else planned_updates
    if total_updates <= 0:
        raise ValueError("history adapter requires at least one update")
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    update = 0
    stop = False
    started = time.time()
    git = _git_state()
    source_path = Path(__file__).resolve()
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        sums = {
            "objective": 0.0, "dense_candidate_teacher_huber": 0.0,
            "history_teacher_huber": 0.0, "residual_huber": 0.0,
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
                _, composed = _mapped_composed(mapper, s_model, h_model, batch)
                teacher_composed, _ = _clean_teacher_pipeline(
                    s_model, h_model, f_model, batch
                )
                teacher_prediction = _hybrid_history_teacher_prediction(
                    f_model, composed, teacher_composed, batch
                )
            adapted, prediction = _adapted_prediction(
                adapter, f_model, composed, batch
            )
            objective, components = history_domain_loss(
                adapted, prediction, teacher_composed, teacher_prediction, batch,
                huber_beta_m=args.huber_beta_m,
                future_weight=args.future_weight,
                history_weight=args.history_weight,
                residual_weight=args.residual_weight,
            )
            objective.backward()
            if not any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                and bool(torch.any(parameter.grad != 0))
                for parameter in adapter.parameters()
            ):
                raise RuntimeError("history adapter received no finite nonzero gradient")
            if any(parameter.grad is not None for model in (mapper, s_model, h_model, f_model) for parameter in model.parameters()):
                raise RuntimeError("frozen mapper/S/H/F accumulated gradients")
            torch.nn.utils.clip_grad_norm_(
                adapter.parameters(), args.gradient_clip_norm
            )
            optimizer.step()
            sums["objective"] += float(objective.detach())
            for name, value in components.items():
                sums[name] += float(value.detach())
            batches += 1
            if args.max_updates > 0 and update >= args.max_updates:
                stop = True
                break
        if epoch == 1 or epoch % args.validation_interval == 0 or stop or epoch == args.epochs:
            metrics = evaluate(
                adapter, mapper, s_model, h_model, f_model,
                validation_loader, device,
            )
            selection = (
                float(metrics["conditional_position"]["p95_m"]),
                float(metrics["conditional_position"]["p99_m"]),
                float(metrics["adapted_history_teacher_gap"]["p95_m"]),
            )
            checkpoint_name = f"epoch-{epoch:04d}-update-{update:06d}.pt"
            checkpoint_path = output / checkpoint_name
            provenance = {
                "dataset_manifest_path": str(
                    Path(args.dataset).resolve() / "dataset_manifest.json"
                ),
                "dataset_manifest_sha256": train_dataset.manifest_sha256,
                "frozen_mapper": mapper_provenance,
                "frozen_s": s_provenance,
                "frozen_h": h_provenance,
                "frozen_f": f_provenance,
                "mapper_h_compatibility_mismatch": mapper_h_mismatch,
                "adapter_input_fields": list(HISTORY_INPUT_FIELDS),
                "clean_and_future_fields_loss_side_only": True,
                "hybrid_teacher_changes_history_only": True,
                "anonymous_selected_history": True,
                "diagnostic_only": True,
                "diagnostic_reasons": [
                    "oracle_association", "legacy_h_diagnostic_provenance",
                    "mapper_h_provenance_mismatch", "dirty_training_source",
                ],
                "oracle_association": True,
                "deployable_pipeline": False,
                "test_accessed": False,
                "source_path": str(source_path),
                "source_sha256": sha256_file(source_path),
                "git": git,
            }
            checkpoint = {
                "schema_version": RUN_SCHEMA,
                "model_class": type(adapter).__name__,
                "model_config": adapter.config,
                "model": adapter.state_dict(),
                "epoch": epoch, "update": update,
                "validation": metrics, "selection": selection,
                "provenance": provenance,
            }
            _atomic_checkpoint(checkpoint_path, checkpoint)
            item = {
                "epoch": epoch, "update": update, "learning_rate": lr,
                "train": {
                    name: value / max(batches, 1) for name, value in sums.items()
                },
                "validation": metrics, "selection": selection,
                "checkpoint": checkpoint_name,
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
            history.append(item)
            if best is None or selection < tuple(best["selection"]):
                best = {
                    "epoch": epoch, "update": update,
                    "path": checkpoint_name,
                    "sha256": item["checkpoint_sha256"],
                    "selection": selection, "validation": metrics,
                }
            progress = {
                "schema_version": RUN_SCHEMA,
                "status": "running" if not stop else "complete",
                "epoch": epoch, "update": update, "best": best,
                "history": history, "elapsed_s": time.time() - started,
                "train_sample_count": len(train_dataset),
                "validation_sample_count": len(validation_dataset),
                "training_arguments": vars(args),
                "model_config": adapter.config,
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in adapter.parameters()
                ),
                "provenance": provenance,
            }
            _atomic_json(output / "run_progress.json", progress)
        if stop:
            break
    if best is None:
        raise RuntimeError("history adapter produced no checkpoint")
    frozen_unchanged = {
        "mapper": state_dict_sha256(mapper.state_dict()) == frozen_before["mapper"],
        "s": state_dict_sha256(s_model.state_dict()) == frozen_before["s"],
        "h": state_dict_sha256(h_model.state_dict()) == frozen_before["h"],
        "f": state_dict_sha256(f_model.state_dict()) == frozen_before["f"],
    }
    if not all(frozen_unchanged.values()):
        raise RuntimeError("history adapter changed a frozen state hash")
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
            "teacher_gap_p95_lt_150mm": (
                float(best["validation"]["student_teacher_true_branch_gap"]["p95_m"])
                < 0.150
            ),
            "adapted_history_p95_lt_30mm": (
                float(best["validation"]["adapted_history_teacher_gap"]["p95_m"])
                < 0.030
            ),
            "current_p95_le_150mm": (
                float(best["validation"]["current_position"]["p95_m"])
                <= 0.150
            ),
        },
    })
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
    result.add_argument("--channels", type=int, default=64)
    result.add_argument("--dropout", type=float, default=0.05)
    result.add_argument("--position-scale-m", type=float, default=1.0)
    result.add_argument("--history-scale-s", type=float, default=0.32)
    result.add_argument("--switch-scale", type=float, default=6.0)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--epochs", type=int, default=20)
    result.add_argument("--max-updates", type=int, default=1200)
    result.add_argument("--learning-rate", type=float, default=2e-4)
    result.add_argument("--minimum-learning-rate", type=float, default=2e-6)
    result.add_argument("--warmup-updates", type=int, default=100)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--gradient-clip-norm", type=float, default=1.0)
    result.add_argument("--huber-beta-m", type=float, default=0.05)
    result.add_argument("--future-weight", type=float, default=1.0)
    result.add_argument("--history-weight", type=float, default=1.0)
    result.add_argument("--residual-weight", type=float, default=0.05)
    result.add_argument("--validation-interval", type=int, default=1)
    result.add_argument("--train-limit", type=int, default=0)
    result.add_argument("--validation-limit", type=int, default=0)
    return result


def main() -> None:
    args = parser().parse_args()
    if min(
        args.position_scale_m, args.history_scale_s, args.switch_scale,
        args.learning_rate, args.minimum_learning_rate,
        args.gradient_clip_norm, args.huber_beta_m,
    ) <= 0:
        raise ValueError("positive history-adapter configuration required")
    if min(args.future_weight, args.history_weight, args.residual_weight) < 0:
        raise ValueError("history-adapter weights cannot be negative")
    print(train(args))


if __name__ == "__main__":
    main()
