"""Train the diagnostic oracle-associated anonymous vehicle motion pilot.

This runner deliberately does not claim a deployable end-to-end pipeline.  Its
four history lanes come from the paired dataset's oracle association.  The
script never opens a test split and always trains/evaluates rotation and
combined motion together.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Iterable
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from .anonymous_vehicle_motion import (
    FORWARD_FIELDS,
    AnonymousVehicleFutureModel,
    anonymous_vehicle_future_loss,
    target_candidate_rows,
)
from .cyclic_future_foundation import load_frozen_v19
from .observable_future_pnp_ab import (
    ObservableFuturePnPSFDataset,
    canonicalize_direction_keep_c4,
    sha256_file,
    state_dict_sha256,
)
from .pnp_q0_hypothesis_adapter import (
    hypothesis_forward,
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .train_causal_physical_ab import _git_state, _seed, _to_device
from .train_pnp_q0_hypothesis_adapter import _s_forward
from .train_pnp_window_mapper_distillation import _atomic_checkpoint, _atomic_json


RUN_SCHEMA = "stage3-anonymous-vehicle-motion-oracle-pilot-v1"
MOTION_CLASSES = (2, 3)
TRAJECTORY_MODULES = ("context", "candidate_encoder", "trajectory_head")
SELECTOR_MODULES = (
    "selector_context", "direction_score_head", "crossing_interval_head",
    "temperature_head",
)
CHECKPOINT_INTERVAL = 150
HISTORY_BINS = ((8, 15), (16, 23), (24, 1_000_000))
ERROR_THRESHOLDS_MM = (20, 50, 100, 150, 200, 300)


class CombinedMotionDataset(Dataset):
    """One index space over explicit rotation and combined-motion datasets."""

    def __init__(self, parts: Iterable[ObservableFuturePnPSFDataset]) -> None:
        self.parts = tuple(parts)
        if tuple(part.motion_class for part in self.parts) != MOTION_CLASSES:
            raise ValueError("combined dataset must contain motion classes 2 then 3")
        self.offsets: list[int] = [0]
        for part in self.parts:
            self.offsets.append(self.offsets[-1] + len(part))
        if not self.offsets[-1]:
            raise ValueError("combined motion dataset is empty")

    def __len__(self) -> int:
        return self.offsets[-1]

    def _locate(self, index: int) -> tuple[ObservableFuturePnPSFDataset, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        part_index = 0 if index < self.offsets[1] else 1
        return self.parts[part_index], index - self.offsets[part_index]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        part, local = self._locate(index)
        return part[local]

    def strata(self) -> dict[tuple[int, str], list[int]]:
        result: dict[tuple[int, str], list[int]] = defaultdict(list)
        for part_index, part in enumerate(self.parts):
            active_count = part.tensors["pnp_s_event_mask"].to(torch.bool).sum(dim=1)
            for local_index, count in enumerate(active_count.tolist()):
                label = None
                for lower, upper in HISTORY_BINS:
                    if lower <= int(count) <= upper:
                        label = f"{lower}-{upper if upper < 1_000_000 else 'plus'}"
                        break
                if label is None:
                    raise ValueError(f"history shorter than eight events: {count}")
                result[(part.motion_class, label)].append(
                    self.offsets[part_index] + local_index
                )
        if not result or any(not values for values in result.values()):
            raise ValueError("balanced sampler has an empty stratum")
        return dict(result)

    @property
    def audit(self) -> dict[str, Any]:
        return {
            str(part.motion_class): {
                "sample_count": len(part),
                "split_audit": part.split_audit,
                "session_count": len(part.session_set),
            }
            for part in self.parts
        }


class BalancedMotionHistorySampler:
    """Stateful exact-near-balanced sampler with resumable RNG state."""

    def __init__(
        self, strata: dict[tuple[int, str], list[int]], *, seed: int,
    ) -> None:
        self.labels = tuple(sorted(strata))
        self.indices = {
            label: torch.tensor(strata[label], dtype=torch.long)
            for label in self.labels
        }
        self.generator = torch.Generator().manual_seed(int(seed))
        self.calls = 0

    def draw(self, batch_size: int) -> list[int]:
        if batch_size < len(self.labels):
            raise ValueError(
                "batch size must be at least the number of motion/history strata"
            )
        result: list[int] = []
        while len(result) < batch_size:
            order = torch.randperm(len(self.labels), generator=self.generator)
            for label_index in order.tolist():
                values = self.indices[self.labels[label_index]]
                selected = int(torch.randint(
                    len(values), (1,), generator=self.generator,
                ).item())
                result.append(int(values[selected]))
                if len(result) == batch_size:
                    break
        self.calls += 1
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "labels": [list(value) for value in self.labels],
            "generator_state": self.generator.get_state(),
            "calls": self.calls,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        labels = tuple((int(value[0]), str(value[1])) for value in state["labels"])
        if labels != self.labels:
            raise ValueError("resume sampler strata differ from the dataset")
        self.generator.set_state(state["generator_state"])
        self.calls = int(state["calls"])

    @property
    def support(self) -> dict[str, int]:
        return {
            f"motion_{motion}_history_{history}": int(len(self.indices[(motion, history)]))
            for motion, history in self.labels
        }


def _tensor_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    return state_dict_sha256(state)


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


def _history_dt(time_s: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(time_s)
    pair = active[:, 1:] & active[:, :-1]
    result[:, 1:] = torch.where(
        pair, time_s[:, 1:] - time_s[:, :-1], torch.zeros_like(time_s[:, 1:]),
    )
    if bool(torch.any(result < -1e-7)):
        raise ValueError("history event time must be monotone")
    return result.clamp_min(0.0)


def apply_prefix_dropout(
    batch: dict[str, torch.Tensor],
    *,
    probability: float,
    minimum_events: int,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Crop a causal prefix before any frozen Mapper/S/H computation."""
    if not 0.0 <= probability <= 1.0 or minimum_events < 8:
        raise ValueError("invalid prefix-dropout contract")
    result = dict(batch)
    active = batch["pnp_s_event_mask"].to(torch.bool).clone()
    retained = torch.zeros_like(active)
    for row in range(active.shape[0]):
        indices = torch.nonzero(active[row], as_tuple=False).flatten()
        if indices.numel() < minimum_events:
            raise ValueError("prefix dropout received a short history")
        keep = int(indices.numel())
        if float(torch.rand((), generator=generator)) < probability:
            keep = int(torch.randint(
                minimum_events, int(indices.numel()) + 1, (1,), generator=generator,
            ).item())
        retained[row, indices[-keep:]] = True
    obs_mask = batch["pnp_s_obs_mask"].to(torch.bool) & retained.unsqueeze(-1)
    primary_mask = (
        batch["pnp_s_primary_mask"].to(torch.bool) & retained.unsqueeze(-1)
    )
    obs = torch.where(
        obs_mask.unsqueeze(-1), batch["pnp_s_obs_m"],
        torch.zeros_like(batch["pnp_s_obs_m"]),
    )
    event_time = torch.where(
        retained, batch["pnp_s_event_time_s"],
        torch.zeros_like(batch["pnp_s_event_time_s"]),
    )
    switch = torch.where(
        retained, batch["pnp_s_switch_step"],
        torch.zeros_like(batch["pnp_s_switch_step"]),
    ).clone()
    for row in range(retained.shape[0]):
        first = int(torch.nonzero(retained[row], as_tuple=False).flatten()[0])
        switch[row, first] = 0
    result.update({
        "pnp_s_obs_m": obs,
        "pnp_s_obs_mask": obs_mask,
        "pnp_s_primary_mask": primary_mask,
        "pnp_s_event_mask": retained,
        "pnp_s_event_time_s": event_time,
        "pnp_s_switch_step": switch,
    })
    return result


@torch.no_grad()
def frozen_upstream_batch(
    mapper: torch.nn.Module,
    s_model: torch.nn.Module,
    h_model: torch.nn.Module,
    raw_batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Create the MotionContext contract from frozen oracle-associated lanes."""
    mapper.eval()
    s_model.eval()
    h_model.eval()
    mapped = mapper(
        raw_batch["pnp_s_obs_m"], raw_batch["pnp_s_obs_mask"],
        raw_batch["pnp_s_event_time_s"], raw_batch["pnp_s_event_mask"],
    )
    s_output = _s_forward(
        s_model, mapped["corrected_obs_m"], raw_batch["pnp_s_obs_mask"],
        raw_batch["pnp_s_primary_mask"], raw_batch["pnp_s_event_mask"],
        raw_batch["pnp_s_event_time_s"], raw_batch["pnp_s_switch_step"],
    )
    h_output = hypothesis_forward(h_model, s_output)
    primary = s_output["primary_index"].to(torch.long)
    rows = torch.arange(primary.shape[0], device=primary.device)
    q0 = h_output["q0_m"]
    current = q0[rows, primary]
    active = raw_batch["pnp_s_event_mask"].to(torch.bool)
    visible = raw_batch["pnp_s_obs_mask"].to(torch.bool) & active.unsqueeze(-1)
    corrected = mapped["corrected_obs_m"]
    history_relative = corrected - current[:, None, None]
    history_relative = torch.where(
        visible.unsqueeze(-1), history_relative, torch.zeros_like(history_relative),
    )
    q0_relation = q0 - current[:, None]
    q0_relation[rows, primary] = 0

    step = raw_batch["pnp_candidate_step"].to(torch.long)
    candidate_handle = torch.remainder(primary[:, None] + step, 4)
    gather3 = candidate_handle.unsqueeze(-1).expand(-1, -1, 3)
    relation = q0.gather(1, gather3) - current[:, None]
    relation = torch.where(
        (torch.remainder(step, 4) == 0).unsqueeze(-1),
        torch.zeros_like(relation), relation,
    )
    candidate_confidence = h_output["confidence_for_f"].gather(
        1, candidate_handle,
    )
    candidate_supported = h_output["evidence_supported"].gather(
        1, candidate_handle,
    )
    time_s = raw_batch["pnp_s_event_time_s"]
    prepared = {
        "history_obs_rel_m": history_relative,
        "history_obs_mask": visible,
        "history_primary_mask": raw_batch["pnp_s_primary_mask"].to(torch.bool),
        "history_event_mask": active,
        "history_time_s": time_s,
        "history_dt_s": _history_dt(time_s, active),
        "history_switch_step": raw_batch["pnp_s_switch_step"],
        "q0_relation_m": q0_relation,
        "q0_sigma_m": h_output["hypothesis_sigma_m"],
        "q0_confidence": h_output["confidence_for_f"],
        "q0_age_s": s_output["age_s"],
        "q0_support_class": h_output["support_class"],
        "q0_supported": h_output["evidence_supported"],
        "current_position_m": current,
        "candidate_relation_m": relation,
        "candidate_step": step,
        "candidate_mask": raw_batch["pnp_candidate_mask"].to(torch.bool),
        "candidate_confidence": candidate_confidence,
        "candidate_supported": candidate_supported,
        "tau_s": raw_batch["pnp_tau_s"],
        "target_switch_count": raw_batch["target_switch_count"],
        "target_visible_delta_m": raw_batch["target_visible_delta_m"],
        "target_query_mask": raw_batch["target_query_mask"].to(torch.bool),
        "truth_current_position_m": raw_batch["current_position_m"],
        "motion_class": raw_batch["motion_class"],
    }
    return {name: value.detach() for name, value in prepared.items()}


def _forward_only(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: batch[name] for name in FORWARD_FIELDS}


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


@torch.no_grad()
def evaluate(
    model: AnonymousVehicleFutureModel,
    loader: DataLoader,
    mapper: torch.nn.Module,
    s_model: torch.nn.Module,
    h_model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    storage: dict[str, dict[str, list[np.ndarray] | int]] = {
        name: {"conditional": [], "hard": [], "correct": 0, "count": 0}
        for name in ("overall", "rotation", "combined", "history_8_15", "switch_3_plus")
    }
    use_amp = device.type == "cuda"
    for raw_cpu in loader:
        raw = _to_device(raw_cpu, device)
        amp = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
        with amp:
            batch = frozen_upstream_batch(mapper, s_model, h_model, raw)
            prediction = model(_forward_only(batch))
        positive = batch["target_query_mask"] & (batch["tau_s"] > 0)
        row = target_candidate_rows(
            batch["candidate_step"], batch["candidate_mask"],
            batch["target_switch_count"], positive,
        )
        conditional = prediction["conditional_position_m"].gather(
            2, row[:, :, None, None].expand(-1, -1, 1, 3),
        ).squeeze(2)
        target = (
            batch["truth_current_position_m"][:, None]
            + batch["target_visible_delta_m"]
        )
        conditional_error = torch.linalg.vector_norm(conditional - target, dim=-1)
        hard_error = torch.linalg.vector_norm(prediction["position_m"] - target, dim=-1)
        selected_step = prediction["selected_switch_step"]
        correct = selected_step == batch["target_switch_count"]
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
            storage[name]["conditional"].append(
                conditional_error[query].float().cpu().numpy()
            )
            storage[name]["hard"].append(hard_error[query].float().cpu().numpy())
            storage[name]["correct"] = int(storage[name]["correct"]) + int(
                (correct & query).sum()
            )
            storage[name]["count"] = int(storage[name]["count"]) + int(query.sum())
    result: dict[str, Any] = {}
    for name, values in storage.items():
        conditional_parts = values["conditional"]
        hard_parts = values["hard"]
        assert isinstance(conditional_parts, list) and isinstance(hard_parts, list)
        conditional = (
            np.concatenate(conditional_parts).astype(np.float64, copy=False)
            if conditional_parts else np.empty(0, dtype=np.float64)
        )
        hard = (
            np.concatenate(hard_parts).astype(np.float64, copy=False)
            if hard_parts else np.empty(0, dtype=np.float64)
        )
        count = int(values["count"])
        result[name] = {
            "conditional": _distribution(conditional),
            "hard": _distribution(hard),
            "selection_accuracy": (
                int(values["correct"]) / count if count else None
            ),
            "conditional_coverage": _coverage(conditional) if count else {},
            "hard_coverage": _coverage(hard) if count else {},
        }
    return result


def _module_parameters(
    model: AnonymousVehicleFutureModel, names: tuple[str, ...],
) -> list[torch.nn.Parameter]:
    result: list[torch.nn.Parameter] = []
    for name in names:
        result.extend(getattr(model, name).parameters())
    return result


def _configure_stage(model: AnonymousVehicleFutureModel, stage: str) -> None:
    trajectory = _module_parameters(model, TRAJECTORY_MODULES)
    selector = _module_parameters(model, SELECTOR_MODULES)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == "trajectory":
        model.train()
        for parameter in trajectory:
            parameter.requires_grad_(True)
        for name in SELECTOR_MODULES:
            getattr(model, name).eval()
    elif stage == "selector":
        model.eval()
        for parameter in selector:
            parameter.requires_grad_(True)
        for name in SELECTOR_MODULES:
            getattr(model, name).train()
    elif stage == "joint":
        model.train()
        for parameter in trajectory + selector:
            parameter.requires_grad_(True)
    else:
        raise ValueError(f"unknown stage: {stage}")


def _stage_for_update(args: argparse.Namespace, global_update: int) -> tuple[str, int, int]:
    boundaries = (
        ("trajectory", args.trajectory_updates),
        ("selector", args.trajectory_updates + args.selector_updates),
        ("joint", args.trajectory_updates + args.selector_updates + args.joint_updates),
    )
    previous = 0
    for name, endpoint in boundaries:
        if global_update <= endpoint:
            return name, global_update - previous, endpoint - previous
        previous = endpoint
    raise ValueError("global update exceeds the fixed endpoint")


def _set_phase_lr(
    optimizer: torch.optim.Optimizer,
    *, base_lr: float, stage_update: int, stage_total: int,
) -> float:
    progress = (stage_update - 1) / max(stage_total - 1, 1)
    multiplier = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    value = base_lr * multiplier
    for group in optimizer.param_groups:
        group["lr"] = value
    return value


def _rng_state(
    sampler: BalancedMotionHistorySampler,
    prefix_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "sampler": sampler.state_dict(),
        "prefix_dropout_generator": prefix_generator.get_state(),
    }


def _restore_rng_state(
    state: dict[str, Any],
    sampler: BalancedMotionHistorySampler,
    prefix_generator: torch.Generator,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    sampler.load_state_dict(state["sampler"])
    prefix_generator.set_state(state["prefix_dropout_generator"])


def _checkpoint_payload(
    *,
    model: AnonymousVehicleFutureModel,
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
    }


def _require_runtime(args: argparse.Namespace) -> torch.device:
    environment = os.environ.get("CONDA_DEFAULT_ENV", "").lower()
    if "yolov8" not in environment and "yolov8" not in sys.executable.lower():
        raise RuntimeError(
            "this pilot must run in the yolov8 virtual environment on Windows or Linux"
        )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this pilot requires a CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return device


def _dataset(
    path: Path, split: str, *, sample_limit: int,
) -> CombinedMotionDataset:
    parts: list[ObservableFuturePnPSFDataset] = []
    for motion_class in MOTION_CLASSES:
        part = ObservableFuturePnPSFDataset(
            path, split, motion_class=motion_class,
            sample_limit=sample_limit, allow_diagnostic=False,
        )
        # Remove only the pre-baked reflection while retaining the anonymous
        # pair-id-derived C4 origin.  This must precede every Mapper/S/H call.
        canonicalize_direction_keep_c4(part.tensors, part.pair_ids)
        parts.append(part)
    return CombinedMotionDataset(parts)


def _validate_bindings(
    dataset_manifest_sha256: str,
    mapper: dict[str, Any],
    s_model: dict[str, Any],
    h_model: dict[str, Any],
) -> dict[str, Any]:
    mapper_parent = mapper["provenance"]
    h_parent = h_model["provenance"]
    mapper_dataset = mapper_parent.get("dataset_manifest_sha256")
    h_dataset = h_parent.get("dataset_manifest_sha256")
    if mapper_dataset != h_dataset:
        raise ValueError("Mapper and H parent datasets differ")
    if (
        mapper_parent.get("frozen_s", {}).get("state_dict_sha256")
        != s_model["state_dict_sha256"]
    ):
        raise ValueError("Mapper and supplied S differ")
    if (
        h_parent.get("frozen_s", {}).get("state_dict_sha256")
        != s_model["state_dict_sha256"]
    ):
        raise ValueError("H and supplied S differ")
    expected = h_parent.get("frozen_mapper", {}).get("state_dict_sha256")
    actual = mapper["state_dict_sha256"]
    if not isinstance(expected, str):
        raise ValueError("H does not record its expected Mapper state")
    return {
        "mapper_h": {
            "expected_by_explicit_opt_in": True,
            "h_expected_state_dict_sha256": expected,
            "loaded_state_dict_sha256": actual,
            "actual_mismatch": expected != actual,
            "allowed_for_diagnostic_pilot": True,
        },
        # The r2 SF artifact was rebuilt after the frozen Mapper/H pair.  This
        # pilot records that manifest drift instead of silently claiming an
        # exact parent-dataset binding; the run remains diagnostic-only.
        "dataset_manifest": {
            "pilot_dataset_sha256": dataset_manifest_sha256,
            "mapper_h_parent_dataset_sha256": mapper_dataset,
            "actual_mismatch": mapper_dataset != dataset_manifest_sha256,
            "allowed_for_diagnostic_pilot": True,
        },
    }


def train(args: argparse.Namespace) -> Path:
    if not args.diagnostic_oracle_association:
        raise ValueError("requires explicit --diagnostic-oracle-association")
    if not args.allow_mapper_h_mismatch:
        raise ValueError("requires explicit --allow-mapper-h-mismatch")
    if min(args.trajectory_updates, args.selector_updates, args.joint_updates) <= 0:
        raise ValueError("all three fixed training stages need positive updates")
    if CHECKPOINT_INTERVAL != 150:
        raise RuntimeError("immutable recovery checkpoint interval changed")
    forward_signature = inspect.signature(AnonymousVehicleFutureModel.forward)
    if "detach_selector_context" not in forward_signature.parameters:
        raise RuntimeError(
            "joint stage is blocked: AnonymousVehicleFutureModel.forward must "
            "provide detach_selector_context without changing this runner"
        )

    output = Path(args.output).resolve()
    resume_path = (
        Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None
    )
    if resume_path is None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing existing non-resume output: {output}")
        output.mkdir(parents=True, exist_ok=True)
    else:
        if not output.is_dir() or not resume_path.is_file():
            raise ValueError("resume requires an existing output and checkpoint")
        if resume_path.parent != output / "checkpoints":
            raise ValueError("resume checkpoint must be in output/checkpoints")
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    _seed(args.seed)
    device = _require_runtime(args)
    dataset_path = Path(args.dataset).resolve()
    manifest_path = dataset_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("test_accessed", True)):
        raise ValueError("motion pilot refuses a dataset that accessed test")
    if manifest.get("oracle_association") is not True:
        raise ValueError("motion pilot requires oracle-associated data")
    if manifest.get("deployable_pipeline") is not False:
        raise ValueError("oracle-associated data cannot be marked deployable")
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

    model = AnonymousVehicleFutureModel(
        channels=args.channels, dropout=args.dropout,
        message_layers=args.message_layers,
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
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    total_updates = args.trajectory_updates + args.selector_updates + args.joint_updates
    source_path = Path(__file__).resolve()
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
        "prefix_dropout_before_frozen_mapper_s_h": True,
        "positive_time_learned_loss_only": True,
        "motion_classes": list(MOTION_CLASSES),
        "dataset": {
            "path": str(dataset_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": dataset_manifest_sha256,
            "train": train_dataset.audit,
            "validation": validation_dataset.audit,
        },
        "mapper": mapper_info,
        "s": s_info,
        "h": h_info,
        "mapper_h_compatibility": mismatch["mapper_h"],
        "dataset_provenance_compatibility": mismatch["dataset_manifest"],
        "frozen_initial_state_dict_sha256": frozen_initial,
        "sampler": {
            "strategy": "equal_motion_x_history_bin_with_replacement_v1",
            "support": sampler.support,
        },
        "runtime": {
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "amp": True,
            "tf32": True,
            "num_workers": 0,
        },
        "git": _git_state(),
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
    }
    contract = {
        "schema_version": RUN_SCHEMA,
        "args": {
            name: value for name, value in vars(args).items()
            if name != "resume_checkpoint"
        },
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "mapper_state_dict_sha256": mapper_info["state_dict_sha256"],
        "s_state_dict_sha256": s_info["state_dict_sha256"],
        "h_state_dict_sha256": h_info["state_dict_sha256"],
        "source_sha256": provenance["source_sha256"],
        "fixed_total_updates": total_updates,
    }
    contract_sha256 = _json_sha256(contract)
    run_id = str(uuid.uuid4())
    global_update = 0
    validation_history: list[dict[str, Any]] = []

    if resume_path is not None:
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != RUN_SCHEMA:
            raise ValueError("resume checkpoint schema differs")
        if payload.get("contract_sha256") != contract_sha256:
            raise ValueError("resume checkpoint training contract differs")
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
        _restore_rng_state(payload["rng"], sampler, prefix_generator)
    else:
        initial_metrics = evaluate(
            model, validation_loader, mapper, s_model, h_model, device,
        )
        validation_history.append({
            "global_update": 0, "stage": "initial", "metrics": initial_metrics,
        })

    manifest_payload: dict[str, Any] = {
        "schema_version": RUN_SCHEMA,
        "status": "running",
        "run_id": run_id,
        "provenance": provenance,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "model_config": model.config,
        "fixed_schedule": {
            "trajectory_updates": args.trajectory_updates,
            "selector_updates": args.selector_updates,
            "joint_updates": args.joint_updates,
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
    gradient_isolation_verified = False

    while global_update < total_updates:
        next_update = global_update + 1
        stage, stage_update, stage_total = _stage_for_update(args, next_update)
        if stage != previous_stage:
            _configure_stage(model, stage)
            previous_stage = stage
        indices = sampler.draw(args.batch_size)
        raw_cpu = default_collate([train_dataset[index] for index in indices])
        raw_cpu = apply_prefix_dropout(
            raw_cpu, probability=args.prefix_dropout_probability,
            minimum_events=args.minimum_history_events,
            generator=prefix_generator,
        )
        raw = _to_device(raw_cpu, device)
        with torch.autocast("cuda", dtype=torch.float16):
            batch = frozen_upstream_batch(mapper, s_model, h_model, raw)
            prediction = model(
                _forward_only(batch),
                detach_selector_context=(stage == "joint"),
            )
            if stage == "trajectory":
                weights = (1.0, 0.25, 0.0, 0.0, 0.0)
            elif stage == "selector":
                weights = (0.0, 0.0, 1.0, 0.25, 0.0)
            else:
                weights = (1.0, 0.25, 1.0, 0.25, 0.0)
            loss, components = anonymous_vehicle_future_loss(
                prediction, batch,
                trajectory_weight=weights[0], trend_weight=weights[1],
                switch_weight=weights[2], distance_risk_weight=weights[3],
                joint_position_weight=weights[4],
            )

        if stage == "joint" and not gradient_isolation_verified:
            with torch.autocast("cuda", dtype=torch.float16):
                selector_only, _ = anonymous_vehicle_future_loss(
                    prediction, batch, trajectory_weight=0.0, trend_weight=0.0,
                    switch_weight=1.0, distance_risk_weight=0.25,
                    joint_position_weight=0.0,
                )
            isolated = torch.autograd.grad(
                selector_only, trajectory_parameters, retain_graph=True,
                allow_unused=True,
            )
            if any(
                value is not None and bool(torch.any(value.detach() != 0))
                for value in isolated
            ):
                raise RuntimeError(
                    "joint selector loss leaked into context/trajectory parameters"
                )
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
        if stage in {"selector", "joint"}:
            lr_selector = _set_phase_lr(
                selector_optimizer,
                base_lr=(
                    args.selector_learning_rate if stage == "selector"
                    else args.joint_selector_learning_rate
                ),
                stage_update=stage_update, stage_total=stage_total,
            )
            active_optimizers.append(selector_optimizer)
        else:
            lr_selector = 0.0
        scaler.scale(loss).backward()
        for optimizer in active_optimizers:
            scaler.unscale_(optimizer)
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        torch.nn.utils.clip_grad_norm_(trainable_parameters, args.gradient_clip_norm)
        for optimizer in active_optimizers:
            scaler.step(optimizer)
        scaler.update()
        global_update = next_update

        if global_update % args.log_interval == 0 or stage_update == stage_total:
            print(json.dumps({
                "global_update": global_update,
                "stage": stage,
                "stage_update": stage_update,
                "objective": float(components["objective"].detach()),
                "trajectory": float(components["trajectory"].detach()),
                "switch": float(components["switch"].detach()),
                "lr_trajectory": lr_trajectory,
                "lr_selector": lr_selector,
                "elapsed_s": time.time() - started,
            }, sort_keys=True), flush=True)

        stage_endpoint = stage_update == stage_total
        fixed_endpoint = global_update == total_updates
        if stage_endpoint:
            metrics = evaluate(
                model, validation_loader, mapper, s_model, h_model, device,
            )
            validation_history.append({
                "global_update": global_update,
                "stage": stage,
                "metrics": metrics,
            })
            _configure_stage(model, stage)
        if (
            global_update % CHECKPOINT_INTERVAL == 0
            or stage_endpoint or fixed_endpoint
        ):
            checkpoint_path = checkpoint_dir / f"checkpoint-update-{global_update:06d}.pt"
            payload = _checkpoint_payload(
                model=model,
                trajectory_optimizer=trajectory_optimizer,
                selector_optimizer=selector_optimizer,
                scaler=scaler,
                provenance=provenance,
                contract_sha256=contract_sha256,
                global_update=global_update,
                stage=stage,
                stage_update=stage_update,
                validation_history=validation_history,
                sampler=sampler,
                prefix_generator=prefix_generator,
                stage_endpoint=stage_endpoint,
                fixed_endpoint=fixed_endpoint,
            )
            payload["run_id"] = run_id
            _atomic_checkpoint(checkpoint_path, payload)
            manifest_payload["progress"] = {
                "global_update": global_update,
                "stage": stage,
                "stage_update": stage_update,
                "latest_checkpoint": str(checkpoint_path),
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
        "status": "complete",
        "stop_reason": "fixed_update_endpoint",
        "progress": {
            "global_update": total_updates,
            "latest_checkpoint": str(final_checkpoint),
        },
        "fixed_final_checkpoint": {
            "path": str(final_checkpoint),
            "sha256": sha256_file(final_checkpoint),
            "update": total_updates,
            "selected_by_validation": False,
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
        description=(
            "Diagnostic oracle-associated anonymous vehicle motion pilot; "
            "not a deployable pipeline"
        )
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
    parser.add_argument("--trajectory-updates", type=int, default=1200)
    parser.add_argument("--selector-updates", type=int, default=600)
    parser.add_argument("--joint-updates", type=int, default=300)
    parser.add_argument("--trajectory-learning-rate", type=float, default=3e-4)
    parser.add_argument("--selector-learning-rate", type=float, default=3e-4)
    parser.add_argument("--joint-trajectory-learning-rate", type=float, default=1e-4)
    parser.add_argument("--joint-selector-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--prefix-dropout-probability", type=float, default=0.75)
    parser.add_argument("--minimum-history-events", type=int, default=8)
    parser.add_argument("--train-limit-per-class", type=int, default=0)
    parser.add_argument("--validation-limit-per-class", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = train(args)
    print(json.dumps({
        "status": "complete",
        "fixed_final_checkpoint": str(checkpoint),
        "selected_by_validation": False,
        "oracle_association": True,
        "deployable_pipeline": False,
        "test_accessed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
