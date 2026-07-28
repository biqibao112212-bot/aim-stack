"""Diagnostic local training for joint PnP-domain trajectories and visibility."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .cyclic_future_foundation import load_frozen_v19
from .joint_visible_future import (
    JointVisibleFutureModel,
    LearnedVisibleStateSelector,
    joint_visible_future_loss,
)
from .observable_future_pnp_ab import (
    ObservableFuturePnPSFDataset,
    canonicalize_direction_keep_c4,
    load_observable_f_checkpoint,
    sha256_file,
    state_dict_sha256,
)
from .pnp_q0_hypothesis_adapter import (
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .train_causal_physical_ab import _git_state, _seed, _to_device
from .train_dual_domain_pnp_f import _pnp_composed
from .train_observable_future_pnp_ab import (
    _eval_add,
    _finish_eval,
    _new_eval_storage,
)
from .train_pnp_window_mapper_distillation import (
    _atomic_checkpoint,
    _atomic_json,
    _learning_rate,
)


RUN_SCHEMA = "stage3-joint-visible-future-diagnostic-v1"
TRAJECTORY_SELECTOR_PREFIXES = ("switch_candidate_head.", "switch_logit.")
CACHE_FIELDS = (
    "history_position_rel_m", "history_time_s", "history_dt_s",
    "history_switch_step", "history_mask", "current_position_m",
    "candidate_relation_m", "candidate_step", "candidate_mask",
    "candidate_confidence", "candidate_supported", "tau_s",
    "target_switch_count", "target_visible_delta_m", "target_query_mask",
    "truth_current_position_m", "motion_class",
)


class CachedJointFutureDataset(Dataset):
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        if set(tensors) != set(CACHE_FIELDS):
            raise ValueError("joint cache fields differ from the fixed schema")
        counts = {int(value.shape[0]) for value in tensors.values()}
        if len(counts) != 1:
            raise ValueError("joint cache tensors have inconsistent lengths")
        self.tensors = tensors
        self.count = counts.pop()

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: value[index] for name, value in self.tensors.items()}


def _tensor_dict_sha256(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _build_cache(
    dataset: ObservableFuturePnPSFDataset,
    mapper: torch.nn.Module,
    s_model: torch.nn.Module,
    h_model: torch.nn.Module,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[CachedJointFutureDataset, dict[str, Any]]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    parts: dict[str, list[torch.Tensor]] = {name: [] for name in CACHE_FIELDS}
    started = time.time()
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        composed = _pnp_composed(mapper, s_model, h_model, batch)
        cache_batch = {
            "history_position_rel_m": composed["history_position_rel_m"],
            "history_time_s": batch["pnp_history_time_s"],
            "history_dt_s": batch["pnp_history_dt_s"],
            "history_switch_step": batch["pnp_history_switch_step"],
            "history_mask": batch["pnp_history_mask"],
            "current_position_m": composed["current_position_m"],
            "candidate_relation_m": composed["candidate_relation_m"],
            "candidate_step": batch["pnp_candidate_step"],
            "candidate_mask": batch["pnp_candidate_mask"],
            "candidate_confidence": composed["candidate_confidence"],
            "candidate_supported": composed["candidate_supported"],
            "tau_s": batch["pnp_tau_s"],
            "target_switch_count": batch["target_switch_count"],
            "target_visible_delta_m": batch["target_visible_delta_m"],
            "target_query_mask": batch["target_query_mask"],
            "truth_current_position_m": batch["current_position_m"],
            "motion_class": batch["motion_class"],
        }
        for name, value in cache_batch.items():
            parts[name].append(value.detach().cpu().contiguous())
    tensors = {name: torch.cat(values, dim=0) for name, values in parts.items()}
    cache = CachedJointFutureDataset(tensors)
    return cache, {
        "schema_version": "joint-visible-frozen-upstream-cache-v1",
        "sample_count": len(cache),
        "sha256": _tensor_dict_sha256(tensors),
        "elapsed_s": time.time() - started,
        "fields": {
            name: {
                "shape": list(value.shape), "dtype": str(value.dtype),
                "bytes": value.numel() * value.element_size(),
            }
            for name, value in tensors.items()
        },
    }


def _loss_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = dict(batch)
    # Evaluation/loss helpers define targets relative to truth current, while
    # the model consumes the frozen mapped/H current position.
    result["model_current_position_m"] = batch["current_position_m"]
    result["current_position_m"] = batch["truth_current_position_m"]
    return result


def _model_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = dict(batch)
    result["current_position_m"] = batch["model_current_position_m"]
    return result


@torch.no_grad()
def evaluate(
    model: JointVisibleFutureModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    storage = _new_eval_storage()
    conditional_hasher = hashlib.sha256()
    minimum_count = 0
    minimum_correct = 0
    confidence: list[np.ndarray] = []
    for raw_batch in loader:
        cached = _to_device(raw_batch, device)
        loss_batch = _loss_batch(cached)
        prediction = model(_model_batch(loss_batch))
        _eval_add(
            storage, prediction, loss_batch,
            candidate_supported=cached["candidate_supported"],
        )
        value = prediction["conditional_position_m"].detach().cpu().contiguous().numpy()
        conditional_hasher.update(value.tobytes())
        row = prediction["selected_candidate_row"]
        step = cached["candidate_step"].gather(1, row)
        mask = (
            cached["target_query_mask"].to(torch.bool)
            & (cached["target_switch_count"].abs() == 1)
        )
        minimum_count += int(mask.sum())
        minimum_correct += int((mask & (step == cached["target_switch_count"])).sum())
        selected_probability = prediction["switch_probability"].gather(
            2, row.unsqueeze(-1),
        ).squeeze(-1)
        confidence.append(selected_probability[cached["target_query_mask"].to(torch.bool)].cpu().numpy())
    metrics = _finish_eval(storage)
    metrics["minimum_step_switch_count"] = minimum_count
    metrics["minimum_step_switch_recall"] = (
        minimum_correct / minimum_count if minimum_count else None
    )
    merged_confidence = np.concatenate(confidence).astype(np.float64, copy=False)
    metrics["selected_probability"] = {
        "mean": float(merged_confidence.mean()),
        "p50": float(np.percentile(merged_confidence, 50)),
        "p90": float(np.percentile(merged_confidence, 90)),
    }
    metrics["conditional_output_sha256"] = conditional_hasher.hexdigest()
    return metrics


def _configure_parameters(
    model: JointVisibleFutureModel,
    *,
    trajectory_enabled: bool,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], list[str]]:
    selector_parameters = list(model.selector.parameters())
    for parameter in selector_parameters:
        parameter.requires_grad_(True)
    trajectory_parameters: list[torch.nn.Parameter] = []
    frozen_names: list[str] = []
    for name, parameter in model.trajectory.named_parameters():
        enabled = not name.startswith(TRAJECTORY_SELECTOR_PREFIXES)
        parameter.requires_grad_(enabled and trajectory_enabled)
        if enabled:
            trajectory_parameters.append(parameter)
        else:
            frozen_names.append(name)
    if not selector_parameters or not trajectory_parameters or not frozen_names:
        raise RuntimeError("joint model parameter partition is empty")
    return selector_parameters, trajectory_parameters, frozen_names


def _set_train_mode(
    model: JointVisibleFutureModel, *, trajectory_enabled: bool,
) -> None:
    model.train()
    if trajectory_enabled:
        model.trajectory.train()
        model.trajectory.switch_candidate_head.eval()
        model.trajectory.switch_logit.eval()
    else:
        model.trajectory.eval()
    model.selector.train()


def _checkpoint_payload(
    model: JointVisibleFutureModel,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    update: int,
    validation: dict[str, Any],
    provenance: dict[str, Any],
    trajectory_started: bool,
    data_loader_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA,
        "model_class": type(model).__name__,
        "model_config": model.config,
        "trajectory": model.trajectory.state_dict(),
        "selector": model.selector.state_dict(),
        "trajectory_state_dict_sha256": state_dict_sha256(model.trajectory.state_dict()),
        "selector_state_dict_sha256": state_dict_sha256(model.selector.state_dict()),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "update": update,
        "trajectory_joint_training_started": trajectory_started,
        "validation": validation,
        "provenance": provenance,
        "rng": {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "data_loader_generator": data_loader_generator.get_state(),
        },
    }


def train(args: argparse.Namespace) -> Path:
    if not (
        args.diagnostic_only
        and args.allow_diagnostic_h
        and args.allow_mapper_h_mismatch
    ):
        raise ValueError("joint visible training requires explicit diagnostic opt-ins")
    output = Path(args.output).resolve()
    resume_path = (
        Path(args.resume_checkpoint).resolve()
        if args.resume_checkpoint else None
    )
    if resume_path is None:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite joint run: {output}")
        output.mkdir(parents=True)
    else:
        if not output.is_dir() or resume_path.parent != output:
            raise ValueError("resume checkpoint must belong to the existing output")
    _seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("joint visible training requires the requested CUDA device")

    dataset_path = Path(args.dataset).resolve()
    manifest_path = dataset_path / "dataset_manifest.json"
    dataset_manifest_sha256 = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("test_accessed", True)):
        raise ValueError("joint visible training cannot use an accessed test split")

    train_dataset = ObservableFuturePnPSFDataset(
        dataset_path, "train", motion_class=3,
        sample_limit=args.train_limit, allow_diagnostic=False,
    )
    validation_dataset = ObservableFuturePnPSFDataset(
        dataset_path, "validation", motion_class=3,
        sample_limit=args.validation_limit, allow_diagnostic=False,
    )
    canonicalize_direction_keep_c4(train_dataset.tensors, train_dataset.pair_ids)
    canonicalize_direction_keep_c4(
        validation_dataset.tensors, validation_dataset.pair_ids,
    )
    mapper, mapper_provenance = load_frozen_pnp_mapper(args.mapper_checkpoint)
    s_model, s_provenance = load_frozen_v19(args.s_checkpoint)
    h_model, h_provenance = load_frozen_hypothesis_adapter(
        args.h_checkpoint, allow_diagnostic=True,
    )
    trajectory, trajectory_provenance = load_observable_f_checkpoint(
        args.trajectory_checkpoint,
    )
    trajectory_payload = torch.load(
        Path(args.trajectory_checkpoint).resolve(), map_location="cpu",
        weights_only=False,
    )
    parent_provenance = trajectory_payload.get("provenance", {})
    if parent_provenance.get("training_stage") != "trajectory":
        raise ValueError("joint training requires a trajectory-stage parent")
    if parent_provenance.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("joint trajectory parent dataset differs")
    expected_bindings = {
        "mapper": (
            parent_provenance.get("frozen_mapper", {}).get("state_dict_sha256"),
            mapper_provenance["state_dict_sha256"],
        ),
        "s": (
            parent_provenance.get("frozen_s", {}).get("state_dict_sha256"),
            s_provenance["state_dict_sha256"],
        ),
        "h": (
            parent_provenance.get("frozen_h", {}).get("state_dict_sha256"),
            h_provenance["state_dict_sha256"],
        ),
    }
    mismatched = [name for name, values in expected_bindings.items() if values[0] != values[1]]
    if mismatched:
        raise ValueError("joint frozen parent mismatch: " + ", ".join(mismatched))

    for frozen in (mapper, s_model, h_model):
        frozen.to(device).eval().requires_grad_(False)
    train_cache, train_cache_manifest = _build_cache(
        train_dataset, mapper, s_model, h_model, device=device,
        batch_size=args.cache_batch_size,
    )
    validation_cache, validation_cache_manifest = _build_cache(
        validation_dataset, mapper, s_model, h_model, device=device,
        batch_size=args.cache_batch_size,
    )
    cache_manifest = {
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "mapper_state_dict_sha256": mapper_provenance["state_dict_sha256"],
        "s_state_dict_sha256": s_provenance["state_dict_sha256"],
        "h_state_dict_sha256": h_provenance["state_dict_sha256"],
        "train": train_cache_manifest,
        "validation": validation_cache_manifest,
    }
    _atomic_json(output / "frozen_feature_cache_manifest.json", cache_manifest)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_cache, batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=0, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_cache, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    selector = LearnedVisibleStateSelector(
        channels=args.selector_channels, dropout=args.selector_dropout,
        position_scale_m=trajectory.position_scale_m,
        history_scale_s=trajectory.history_scale_s,
        trained_horizon_s=trajectory.trained_horizon_s,
        maximum_absolute_step=trajectory.maximum_absolute_step,
    )
    model = JointVisibleFutureModel(trajectory, selector).to(device)
    resume_payload: dict[str, Any] | None = None
    trajectory_started = False
    if resume_path is not None:
        resume_payload = torch.load(
            resume_path, map_location="cpu", weights_only=False,
        )
        if resume_payload.get("schema_version") != RUN_SCHEMA:
            raise ValueError("joint resume checkpoint schema mismatch")
        if resume_payload.get("model_config") != model.config:
            raise ValueError("joint resume model configuration mismatch")
        resume_provenance = resume_payload.get("provenance", {})
        if (
            resume_provenance.get("dataset_manifest_sha256")
            != dataset_manifest_sha256
            or resume_provenance.get("trajectory_parent", {}).get("sha256")
            != trajectory_provenance["sha256"]
        ):
            raise ValueError("joint resume asset binding mismatch")
        model.trajectory.load_state_dict(resume_payload["trajectory"], strict=True)
        model.selector.load_state_dict(resume_payload["selector"], strict=True)
        trajectory_started = bool(
            resume_payload["trajectory_joint_training_started"]
        )
    selector_parameters, trajectory_parameters, frozen_trajectory_names = (
        _configure_parameters(
            model, trajectory_enabled=trajectory_started,
        )
    )
    optimizer = torch.optim.AdamW([
        {
            "params": selector_parameters,
            "lr": args.learning_rate,
            "name": "selector",
        },
        {
            "params": trajectory_parameters,
            "lr": args.trajectory_learning_rate,
            "name": "trajectory",
        },
    ], weight_decay=args.weight_decay)
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
    initial_old_selector_hash = state_dict_sha256({
        name: value for name, value in model.trajectory.state_dict().items()
        if name.startswith(TRAJECTORY_SELECTOR_PREFIXES)
    })
    total_updates = min(args.max_updates, args.epochs * len(train_loader))
    provenance = {
        "diagnostic_only": True,
        "oracle_association": True,
        "deployable_pipeline": False,
        "test_accessed": False,
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "mapper": mapper_provenance,
        "s": s_provenance,
        "h": h_provenance,
        "trajectory_parent": {
            **trajectory_provenance,
            "provenance": parent_provenance,
        },
        "frozen_feature_cache": cache_manifest,
        "selector_definition": model.selector.config,
        "position_and_visibility_joint_after_update": args.selector_warmup_updates,
        "old_trajectory_selector_frozen": True,
        "old_trajectory_selector_parameter_names": frozen_trajectory_names,
        "physical_id_input": False,
        "motion_class_forward_input": False,
        "git": _git_state(),
        "source_path": str(Path(__file__).resolve()),
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    if resume_payload is None:
        initial_validation = evaluate(model, validation_loader, device)
        history: list[dict[str, Any]] = [{
            "epoch": 0, "update": 0, "phase": "random-selector-baseline",
            "validation": initial_validation,
        }]
        update = 0
        first_epoch = 1
    else:
        progress_path = output / "run_progress.json"
        if not progress_path.is_file():
            raise ValueError("joint resume requires run_progress.json")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        history = list(progress["history"])
        update = int(resume_payload["update"])
        first_epoch = int(resume_payload["epoch"]) + 1
        rng = resume_payload["rng"]
        torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and rng["torch_cuda"]:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        generator.set_state(rng["data_loader_generator"])
    started = time.time()
    stop = False
    for epoch in range(first_epoch, args.epochs + 1):
        sums: dict[str, float] = {}
        batches = 0
        for raw_batch in train_loader:
            update += 1
            if (
                not trajectory_started
                and update > args.selector_warmup_updates
            ):
                trajectory_started = True
                for parameter in trajectory_parameters:
                    parameter.requires_grad_(True)
            _set_train_mode(model, trajectory_enabled=trajectory_started)
            selector_lr = _learning_rate(
                args.learning_rate, update, total_updates,
                args.learning_rate_warmup_updates,
                args.minimum_learning_rate,
            )
            joint_update = max(1, update - args.selector_warmup_updates)
            joint_total = max(1, total_updates - args.selector_warmup_updates)
            trajectory_lr = _learning_rate(
                args.trajectory_learning_rate, joint_update, joint_total,
                args.trajectory_learning_rate_warmup_updates,
                args.minimum_trajectory_learning_rate,
            )
            optimizer.param_groups[0]["lr"] = selector_lr
            optimizer.param_groups[1]["lr"] = trajectory_lr
            cached = _to_device(raw_batch, device)
            loss_batch = _loss_batch(cached)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(_model_batch(loss_batch))
            objective, components = joint_visible_future_loss(
                prediction, loss_batch,
                switch_weight=args.switch_weight,
                conditional_position_weight=(
                    args.conditional_position_weight if trajectory_started else 0.0
                ),
                mixture_weight=args.mixture_weight,
                expected_cost_weight=args.expected_cost_weight,
                mixture_sigma_m=args.mixture_sigma_m,
                huber_beta_m=args.huber_beta_m,
                macro_balance_weight=args.macro_balance_weight,
                focal_gamma=args.focal_gamma,
            )
            objective.backward()
            if not any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                and bool(torch.any(parameter.grad != 0))
                for parameter in selector_parameters
            ):
                raise RuntimeError("joint selector received no finite gradient")
            if trajectory_started and not any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                and bool(torch.any(parameter.grad != 0))
                for parameter in trajectory_parameters
            ):
                raise RuntimeError("joint trajectory received no finite gradient")
            torch.nn.utils.clip_grad_norm_(
                selector_parameters + trajectory_parameters,
                args.gradient_clip_norm,
            )
            optimizer.step()
            for name, value in components.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach())
            batches += 1
            if update >= total_updates:
                stop = True
                break

        validate_now = (
            epoch <= 2 or epoch % args.validation_interval == 0 or stop
        )
        if validate_now:
            metrics = evaluate(model, validation_loader, device)
            record = {
                "epoch": epoch,
                "update": update,
                "phase": "joint" if trajectory_started else "selector-warmup",
                "learning_rate": selector_lr,
                "trajectory_learning_rate": trajectory_lr,
                "train": {
                    name: value / max(batches, 1) for name, value in sums.items()
                },
                "validation": metrics,
                "elapsed_s": time.time() - started,
            }
            history.append(record)
            checkpoint_path = output / f"epoch-{epoch:04d}-update-{update:06d}.pt"
            payload = _checkpoint_payload(
                model, optimizer, epoch=epoch, update=update,
                validation=metrics, provenance=provenance,
                trajectory_started=trajectory_started,
                data_loader_generator=generator,
            )
            _atomic_checkpoint(checkpoint_path, payload)
            _atomic_json(output / "run_progress.json", {
                "schema_version": RUN_SCHEMA,
                "status": "running" if not stop else "complete",
                "epoch": epoch,
                "update": update,
                "latest_checkpoint": checkpoint_path.name,
                "history": history,
                "provenance": provenance,
            })
            print(json.dumps({
                "epoch": epoch,
                "update": update,
                "phase": record["phase"],
                "switch_accuracy": metrics["switch_accuracy"],
                "minimum_step_recall": metrics["minimum_step_switch_recall"],
                "conditional_p95_m": metrics["conditional_position"]["p95_m"],
                "hard_p95_m": metrics["hard_routed_position"]["p95_m"],
                "elapsed_s": record["elapsed_s"],
            }), flush=True)
        if stop:
            break

    old_selector_hash = state_dict_sha256({
        name: value for name, value in model.trajectory.state_dict().items()
        if name.startswith(TRAJECTORY_SELECTOR_PREFIXES)
    })
    if old_selector_hash != initial_old_selector_hash:
        raise RuntimeError("joint training changed the obsolete trajectory selector")
    final_checkpoint = history[-1]
    manifest_payload = {
        "schema_version": RUN_SCHEMA,
        "status": "complete",
        "stop_reason": "fixed_max_updates" if stop else "epoch_limit",
        "epoch": final_checkpoint["epoch"],
        "update": final_checkpoint["update"],
        "elapsed_s": time.time() - started,
        "training_arguments": vars(args),
        "history": history,
        "final": final_checkpoint,
        "provenance": provenance,
        "old_trajectory_selector_unchanged": True,
    }
    _atomic_json(output / "run_manifest.json", manifest_payload)
    return output / "run_manifest.json"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", required=True)
    result.add_argument("--mapper-checkpoint", required=True)
    result.add_argument("--s-checkpoint", required=True)
    result.add_argument("--h-checkpoint", required=True)
    result.add_argument("--trajectory-checkpoint", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--resume-checkpoint", default="")
    result.add_argument("--diagnostic-only", action="store_true")
    result.add_argument("--allow-diagnostic-h", action="store_true")
    result.add_argument("--allow-mapper-h-mismatch", action="store_true")
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int, default=20260729)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--cache-batch-size", type=int, default=256)
    result.add_argument("--epochs", type=int, default=60)
    result.add_argument("--max-updates", type=int, default=3000)
    result.add_argument("--selector-warmup-updates", type=int, default=500)
    result.add_argument("--selector-channels", type=int, default=128)
    result.add_argument("--selector-dropout", type=float, default=0.05)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    result.add_argument("--learning-rate-warmup-updates", type=int, default=200)
    result.add_argument("--trajectory-learning-rate", type=float, default=2e-5)
    result.add_argument("--minimum-trajectory-learning-rate", type=float, default=2e-7)
    result.add_argument("--trajectory-learning-rate-warmup-updates", type=int, default=100)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--gradient-clip-norm", type=float, default=1.0)
    result.add_argument("--switch-weight", type=float, default=1.0)
    result.add_argument("--conditional-position-weight", type=float, default=50.0)
    result.add_argument("--mixture-weight", type=float, default=0.05)
    result.add_argument("--expected-cost-weight", type=float, default=5.0)
    result.add_argument("--mixture-sigma-m", type=float, default=0.15)
    result.add_argument("--huber-beta-m", type=float, default=0.01)
    result.add_argument("--macro-balance-weight", type=float, default=0.5)
    result.add_argument("--focal-gamma", type=float, default=2.0)
    result.add_argument("--validation-interval", type=int, default=5)
    result.add_argument("--train-limit", type=int, default=0)
    result.add_argument("--validation-limit", type=int, default=0)
    return result


def main() -> None:
    args = parser().parse_args()
    if min(
        args.batch_size, args.cache_batch_size, args.epochs, args.max_updates,
        args.selector_channels,
    ) <= 0:
        raise ValueError("joint visible integer configuration must be positive")
    print(train(args))


if __name__ == "__main__":
    main()
