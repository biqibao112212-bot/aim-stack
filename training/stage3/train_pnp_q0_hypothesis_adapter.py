"""Train A3 H while the PnP mapper, V19 S and accepted F stay frozen."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import secrets
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .cyclic_future_foundation import load_frozen_v19
from .formal_run_contract import (
    capture_formal_contract,
    configure_formal_runtime,
    load_protocol,
    require_asset_binding,
    require_compatible_contracts,
    require_exact_protocol_arguments,
    require_fixed_final_state,
    require_formal_checkpoint_manifest,
    resolve_formal_schedule,
    verify_formal_contract,
)
from .observable_future_pnp_ab import (
    load_observable_f_checkpoint,
    sha256_file,
    state_dict_sha256,
)
from .pnp_q0_hypothesis_adapter import (
    C4Q0HypothesisAdapter,
    PnPQ0HypothesisDataset,
    hypothesis_forward,
    load_frozen_pnp_mapper,
    roll_s_output_c4,
)
from .split_audit import require_formal_split_isolation
from .train_causal_physical_ab import _git_state, _seed, _to_device


RUN_SCHEMA = "stage3-pnp-q0-hypothesis-adapter-run-v1"
RUN_LOCK_SCHEMA = "stage3-formal-h-run-lock-v1"
FORMAL_SOURCE_BUNDLE = (
    "training/stage3/formalization_protocol.json",
    "training/stage3/formal_run_contract.py",
    "training/stage3/train_pnp_q0_hypothesis_adapter.py",
    "training/stage3/pnp_q0_hypothesis_adapter.py",
    "training/stage3/pnp_observation_mapper.py",
    "training/stage3/observable_future_pnp_ab.py",
    "training/stage3/build_observable_future_pnp_sf_upper_bound_dataset.py",
    "training/stage3/split_audit.py",
    "training/stage3/schema.py",
    "training/stage3/cyclic_future_foundation.py",
    "training/stage3/cyclic_anchor_edge_model.py",
    "training/stage3/cyclic_track_model.py",
    "training/stage3/train_causal_physical_ab.py",
)


def _atomic_json(path: Path, payload: object) -> None:
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}-{time.time_ns()}")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pending.replace(path)


def _atomic_checkpoint(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite H checkpoint: {path}")
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}-{time.time_ns()}")
    try:
        torch.save(payload, pending)
        pending.replace(path)
    finally:
        if pending.exists():
            pending.unlink()


def _atomic_recovery_checkpoint(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite H recovery state: {path}")
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}-{time.time_ns()}")
    try:
        torch.save(payload, pending)
        pending.replace(path)
    finally:
        if pending.exists():
            pending.unlink()


def _read_run_lock(path: Path) -> dict[str, Any]:
    owner_path = path.with_name(".formal_h_training.owner.json")
    try:
        payload = json.loads(owner_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema_version": "unreadable", "pid": -1}
    return payload if isinstance(payload, dict) else {
        "schema_version": "invalid", "pid": -1,
    }

def _lock_stream_nonblocking(stream: Any) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\n")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_stream(stream: Any) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _acquire_run_lock(output: Path) -> tuple[Path, str, Any]:
    path = output / ".formal_h_training.lock"
    owner_path = output / ".formal_h_training.owner.json"
    token = secrets.token_hex(16)
    payload = {
        "schema_version": RUN_LOCK_SCHEMA,
        "pid": os.getpid(),
        "token": token,
        "process_started_unix_ns": time.time_ns(),
        "command": sys.argv,
        "released": False,
    }
    stream = path.open("a+b")
    try:
        _lock_stream_nonblocking(stream)
    except OSError as error:
        stream.close()
        owner = _read_run_lock(path)
        raise RuntimeError(
            "H output directory is owned by a live training process: "
            f"pid={owner.get('pid', 'unknown')} lock={path}"
        ) from error
    try:
        previous = _read_run_lock(path)
        if previous.get("released") is False:
            previous_pid = int(previous.get("pid", -1))
            _atomic_json(
                output / (
                    f".formal_h_training.stale-pid-{previous_pid}"
                    f"-{time.time_ns()}.json"
                ),
                previous,
            )
        _atomic_json(owner_path, payload)
    except BaseException:
        _unlock_stream(stream)
        stream.close()
        raise
    return path, token, stream


def _release_run_lock(path: Path, token: str, stream: Any) -> None:
    try:
        owner = _read_run_lock(path)
        if owner.get("pid") == os.getpid() and owner.get("token") == token:
            owner["released"] = True
            owner["released_unix_ns"] = time.time_ns()
            _atomic_json(path.with_name(".formal_h_training.owner.json"), owner)
    finally:
        _unlock_stream(stream)
        stream.close()


def _recovery_training_arguments(args: argparse.Namespace) -> dict[str, Any]:
    values = dict(vars(args))
    values.pop("resume", None)
    return values


def _capture_rng_state(train_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "train_generator": train_generator.get_state(),
    }


def _restore_rng_state(
    state: dict[str, Any], train_generator: torch.Generator,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in state["torch_cuda"]]
        )
    train_generator.set_state(state["train_generator"].cpu())


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device)


def _write_recovery_state(
    output: Path,
    *,
    epoch: int,
    update: int,
    elapsed_s: float,
    model: C4Q0HypothesisAdapter,
    optimizer: torch.optim.Optimizer,
    train_generator: torch.Generator,
    arguments: argparse.Namespace,
    formal_contract: dict[str, Any],
    dataset_manifest_sha256: str,
    mapper_state_sha256: str,
    s_state_sha256: str,
    f_state_sha256: str,
) -> Path:
    name = (
        f"recovery-epoch-{epoch:04d}-update-{update:06d}"
        f"-{time.time_ns()}.pt"
    )
    path = output / name
    payload = {
        "schema_version": "stage3-formal-h-recovery-v1",
        "completed_epoch": int(epoch),
        "update": int(update),
        "elapsed_s": float(elapsed_s),
        "model_config": model.config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": _capture_rng_state(train_generator),
        "training_arguments": _recovery_training_arguments(arguments),
        "formal_source_contract": formal_contract,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "frozen_state_dict_sha256": {
            "mapper": mapper_state_sha256,
            "s": s_state_sha256,
            "f": f_state_sha256,
        },
        "validation_accessed": False,
        "model_selection_performed": False,
        "test_accessed": False,
    }
    _atomic_recovery_checkpoint(path, payload)
    _atomic_json(output / "recovery_latest.json", {
        "schema_version": "stage3-formal-h-recovery-pointer-v1",
        "path": name,
        "sha256": sha256_file(path),
        "completed_epoch": int(epoch),
        "update": int(update),
    })
    return path


def _load_recovery_state(output: Path) -> tuple[Path, dict[str, Any]]:
    pointer_path = output / "recovery_latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("schema_version") != "stage3-formal-h-recovery-pointer-v1":
        raise ValueError("formal H recovery pointer schema mismatch")
    checkpoint = (output / str(pointer["path"])).resolve()
    if checkpoint.parent != output or not checkpoint.is_file():
        raise ValueError("formal H recovery checkpoint escapes its output directory")
    if sha256_file(checkpoint) != pointer.get("sha256"):
        raise ValueError("formal H recovery checkpoint hash mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "stage3-formal-h-recovery-v1":
        raise ValueError("formal H recovery checkpoint schema mismatch")
    if (
        int(payload.get("completed_epoch", -1))
        != int(pointer.get("completed_epoch", -2))
        or int(payload.get("update", -1)) != int(pointer.get("update", -2))
    ):
        raise ValueError("formal H recovery pointer metadata mismatch")
    return checkpoint, payload


def _s_forward(
    model: torch.nn.Module,
    obs_m: torch.Tensor,
    obs_mask: torch.Tensor,
    primary_mask: torch.Tensor,
    event_mask: torch.Tensor,
    event_time_s: torch.Tensor,
    switch_step: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mask = obs_mask.to(torch.bool)
    normalized = (obs_m - model.position_mean) / model.position_std
    normalized = torch.where(mask.unsqueeze(-1), normalized, torch.zeros_like(normalized))
    return model(
        normalized, mask, primary_mask.to(torch.bool), event_mask.to(torch.bool),
        event_time_s, switch_step,
    )


def frozen_mapper_s_forward(
    mapper: torch.nn.Module,
    s_model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    mapper.eval()
    s_model.eval()
    with torch.no_grad():
        mapped = mapper(
            batch["pnp_s_obs_m"], batch["pnp_s_obs_mask"],
            batch["pnp_s_event_time_s"], batch["pnp_s_event_mask"],
        )
        s_output = _s_forward(
            s_model, mapped["corrected_obs_m"], batch["pnp_s_obs_mask"],
            batch["pnp_s_primary_mask"], batch["pnp_s_event_mask"],
            batch["pnp_s_event_time_s"], batch["pnp_s_switch_step"],
        )
    return (
        {name: value.detach() for name, value in mapped.items()},
        {name: value.detach() for name, value in s_output.items()},
    )


def frozen_clean_s_forward(
    s_model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    s_model.eval()
    with torch.no_grad():
        s_output = _s_forward(
            s_model, batch["clean_s_obs_m"], batch["clean_s_obs_mask"],
            batch["clean_s_primary_mask"], batch["clean_s_event_mask"],
            batch["clean_s_event_time_s"], batch["clean_s_switch_step"],
        )
    return {name: value.detach() for name, value in s_output.items()}


def _masked_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor | None:
    selected = mask.to(torch.bool)
    if not bool(selected.any()):
        return None
    # Index before subtraction: a poisoned true-cold label is never consumed.
    difference = prediction[selected] - target[selected]
    return F.smooth_l1_loss(difference, torch.zeros_like(difference), beta=beta)


def hypothesis_loss(
    prediction: dict[str, torch.Tensor],
    target_q0_m: torch.Tensor,
    *,
    huber_beta_m: float,
    q0_weight: float = 1.0,
    edge_weight: float = 0.5,
    opposite_weight: float = 0.5,
    support_weight: float = 0.2,
    sigma_weight: float = 0.2,
    c4_loss: torch.Tensor | None = None,
    c4_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    roles = (
        prediction["observed"].to(torch.bool),
        prediction["warm_adjacent"].to(torch.bool),
        prediction["warm_opposite"].to(torch.bool),
    )
    role_losses = [
        loss for mask in roles
        if (loss := _masked_huber(
            prediction["q0_m"], target_q0_m, mask, beta=huber_beta_m
        )) is not None
    ]
    if not role_losses:
        raise ValueError("H batch has no coordinate-supported handle")
    q0_role_macro = torch.stack(role_losses).mean()
    supported = prediction["evidence_supported"].to(torch.bool)
    edge_mask = supported & torch.roll(supported, shifts=-1, dims=1)
    if bool(edge_mask.any()):
        target_edge = (
            torch.roll(target_q0_m, shifts=-1, dims=1)[edge_mask]
            - target_q0_m[edge_mask]
        )
        edge_difference = prediction["edge0_m"][edge_mask] - target_edge
        directed_edge = F.smooth_l1_loss(
            edge_difference, torch.zeros_like(edge_difference), beta=huber_beta_m
        )
    else:
        directed_edge = q0_role_macro.new_zeros(())
    opposite = _masked_huber(
        prediction["q0_m"], target_q0_m,
        prediction["warm_opposite"], beta=huber_beta_m,
    )
    if opposite is None:
        opposite = q0_role_macro.new_zeros(())
    support = F.binary_cross_entropy_with_logits(
        prediction["support_logits"], supported.to(prediction["support_logits"].dtype)
    )
    if bool(supported.any()):
        sigma = prediction["hypothesis_sigma_m"][supported].clamp_min(1e-5)
        absolute_error = (
            prediction["q0_m"][supported].detach()
            - target_q0_m[supported]
        ).abs()
        sigma_calibration = (
            absolute_error / sigma + torch.log(sigma / 0.1)
        ).mean()
    else:
        sigma_calibration = q0_role_macro.new_zeros(())
    equivariance = (
        q0_role_macro.new_zeros(()) if c4_loss is None else c4_loss
    )
    total = (
        q0_weight * q0_role_macro
        + edge_weight * directed_edge
        + opposite_weight * opposite
        + support_weight * support
        + sigma_weight * sigma_calibration
        + c4_weight * equivariance
    )
    return total, {
        "q0_role_macro": q0_role_macro,
        "directed_edge": directed_edge,
        "warm_opposite": opposite,
        "support": support,
        "sigma_calibration": sigma_calibration,
        "c4": equivariance,
    }


def c4_equivariance_loss(
    model: C4Q0HypothesisAdapter,
    s_output: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    shift: int,
) -> torch.Tensor:
    rolled = hypothesis_forward(model, roll_s_output_c4(s_output, shift))
    pieces = []
    for name in ("q0_m", "edge0_m", "hypothesis_sigma_m", "support_probability"):
        expected = torch.roll(reference[name], shifts=shift, dims=1)
        pieces.append(F.mse_loss(rolled[name], expected))
    return torch.stack(pieces).mean()


def _stats(parts: list[np.ndarray]) -> dict[str, float | int | None]:
    if not parts:
        return {
            "count": 0, "mean_m": None, "p50_m": None,
            "p95_m": None, "p99_m": None, "max_m": None,
        }
    merged = np.concatenate(parts).astype(np.float64, copy=False)
    if not merged.size or not np.isfinite(merged).all():
        raise ValueError("H metric contains no finite values")
    return {
        "count": int(merged.size), "mean_m": float(merged.mean()),
        "p50_m": float(np.quantile(merged, 0.50)),
        "p95_m": float(np.quantile(merged, 0.95)),
        "p99_m": float(np.quantile(merged, 0.99)),
        "max_m": float(merged.max()),
    }


@torch.no_grad()
def evaluate(
    model: C4Q0HypothesisAdapter,
    mapper: torch.nn.Module,
    s_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    domain: str = "mapped",
) -> dict[str, Any]:
    if domain not in {"mapped", "clean"}:
        raise ValueError("H evaluation domain must be mapped or clean")
    model.eval()
    names = ("observed", "warm_adjacent", "warm_opposite")
    before: dict[str, list[np.ndarray]] = {name: [] for name in names}
    after: dict[str, list[np.ndarray]] = {name: [] for name in names}
    before_all: list[np.ndarray] = []
    after_all: list[np.ndarray] = []
    edge_before: list[np.ndarray] = []
    edge_after: list[np.ndarray] = []
    support_correct: list[np.ndarray] = []
    support_brier: list[np.ndarray] = []
    role_counts = {name: 0 for name in (*names, "true_cold")}
    c4_max = 0.0
    for batch_index, raw_batch in enumerate(loader):
        batch = _to_device(raw_batch, device)
        if domain == "mapped":
            _, s_output = frozen_mapper_s_forward(mapper, s_model, batch)
        else:
            s_output = frozen_clean_s_forward(s_model, batch)
        prediction = hypothesis_forward(model, s_output)
        target = batch["pnp_s_truth_q0_m"]
        s_error = torch.linalg.vector_norm(s_output["q0_m"] - target, dim=-1)
        h_error = torch.linalg.vector_norm(prediction["q0_m"] - target, dim=-1)
        supported = prediction["evidence_supported"].to(torch.bool)
        before_all.append(s_error[supported].cpu().numpy())
        after_all.append(h_error[supported].cpu().numpy())
        for name in names:
            mask = prediction[name].to(torch.bool)
            role_counts[name] += int(mask.sum())
            if bool(mask.any()):
                before[name].append(s_error[mask].cpu().numpy())
                after[name].append(h_error[mask].cpu().numpy())
        cold = prediction["true_cold"].to(torch.bool)
        role_counts["true_cold"] += int(cold.sum())
        edge_mask = supported & torch.roll(supported, shifts=-1, dims=1)
        if bool(edge_mask.any()):
            truth_edge = torch.roll(target, shifts=-1, dims=1)[edge_mask] - target[edge_mask]
            edge_before.append(torch.linalg.vector_norm(
                s_output["edge0_m"][edge_mask] - truth_edge, dim=-1
            ).cpu().numpy())
            edge_after.append(torch.linalg.vector_norm(
                prediction["edge0_m"][edge_mask] - truth_edge, dim=-1
            ).cpu().numpy())
        probability = prediction["support_probability"]
        support_correct.append(
            ((probability >= 0.5) == supported).cpu().numpy().reshape(-1)
        )
        support_brier.append(
            (probability - supported.to(probability.dtype)).square().cpu().numpy().reshape(-1)
        )
        if batch_index == 0:
            for shift in (1, 2, 3):
                rolled = hypothesis_forward(model, roll_s_output_c4(s_output, shift))
                for name in ("q0_m", "edge0_m", "hypothesis_sigma_m", "support_probability"):
                    difference = rolled[name] - torch.roll(
                        prediction[name], shifts=shift, dims=1
                    )
                    c4_max = max(c4_max, float(difference.abs().max()))
    support_correct_np = np.concatenate(support_correct)
    support_brier_np = np.concatenate(support_brier)
    return {
        "s_q0_supported": _stats(before_all),
        "h_q0_supported": _stats(after_all),
        "s_q0_by_role": {name: _stats(before[name]) for name in names},
        "h_q0_by_role": {name: _stats(after[name]) for name in names},
        "s_directed_edge": _stats(edge_before),
        "h_directed_edge": _stats(edge_after),
        "role_counts": role_counts,
        "true_cold_coordinate_metric_count": 0,
        "support_accuracy": float(support_correct_np.mean()),
        "support_brier": float(support_brier_np.mean()),
        "c4_max_abs": c4_max,
        "domain": domain,
    }


def _learning_rate(
    base: float, update: int, total: int, warmup: int, minimum: float
) -> float:
    if update <= warmup:
        return base * update / max(warmup, 1)
    progress = (update - warmup) / max(total - warmup, 1)
    floor = minimum / base
    return base * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def build_h_optimizer(
    model: C4Q0HypothesisAdapter,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate, weight_decay=weight_decay,
    )
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    model_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if optimizer_ids != model_ids:
        raise RuntimeError("H optimizer does not contain exactly H parameters")
    return optimizer


def _train_locked(args: argparse.Namespace, output: Path) -> Path:
    formal_contract: dict[str, Any] | None = None
    formal_protocol: dict[str, Any] | None = None
    formal_root_protocol: dict[str, Any] | None = None
    formal_assets: dict[str, Any] | None = None
    if args.resume and not args.formal_oracle:
        raise ValueError("H recovery is available only in formal-oracle mode")
    if args.formal_oracle:
        protocol_path, protocol = load_protocol(
            args.formal_protocol if args.formal_protocol else None
        )
        formal_protocol = protocol["hypothesis"]
        formal_root_protocol = protocol
        formal_assets = protocol["assets"]
        require_exact_protocol_arguments(args, formal_protocol, (
            "seed", "batch_size", "device", "workers", "epochs",
            "channels",
            "message_layers", "dropout", "age_scale_s", "learning_rate",
            "minimum_learning_rate", "warmup_updates", "weight_decay",
            "gradient_clip_norm", "huber_beta_m", "clean_weight",
            "q0_weight", "edge_weight", "opposite_weight", "support_weight",
            "sigma_weight", "c4_weight", "maximum_parameters",
            "motion_class", "validation_from_train", "patience",
        ))
        if args.max_updates != int(formal_protocol["schedule_total_updates"]):
            raise ValueError("formal H LR schedule differs from protocol")
        if int(formal_protocol.get("recovery_epoch_interval", 0)) != 1:
            raise ValueError("formal H requires one recovery state per full epoch")
        if args.train_limit or args.validation_limit:
            raise ValueError("formal H cannot limit train or validation")
        configure_formal_runtime(args.device, args.workers)
        formal_contract = capture_formal_contract(
            FORMAL_SOURCE_BUNDLE, protocol_path=protocol_path,
            requested_device=args.device, workers=args.workers,
        )
    _seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    mapper, mapper_provenance = load_frozen_pnp_mapper(args.mapper_checkpoint)
    s_model, s_provenance = load_frozen_v19(args.s_checkpoint)
    f_model, f_provenance = load_observable_f_checkpoint(args.f_checkpoint)
    mapper_state = state_dict_sha256(mapper.state_dict())
    s_state = state_dict_sha256(s_model.state_dict())
    f_state = state_dict_sha256(f_model.state_dict())
    for frozen in (mapper, s_model, f_model):
        frozen.eval().requires_grad_(False)
    mapper.to(device)
    s_model.to(device)

    motion_class = None if args.motion_class < 0 else args.motion_class
    train_dataset = PnPQ0HypothesisDataset(
        args.dataset, "train", sample_limit=args.train_limit,
        motion_class=motion_class,
    )
    if args.validation_from_train:
        validation_dataset = train_dataset
    else:
        validation_dataset = PnPQ0HypothesisDataset(
            args.dataset, "validation", sample_limit=args.validation_limit,
            motion_class=motion_class,
        )
        if set(train_dataset.session_ids) & set(validation_dataset.session_ids):
            raise ValueError("H train/validation sessions overlap")
    if mapper_provenance["provenance"]["dataset_manifest_sha256"] != train_dataset.manifest_sha256:
        raise ValueError("H mapper and paired dataset manifests differ")
    if args.formal_oracle:
        assert (
            formal_contract is not None
            and formal_assets is not None
            and formal_root_protocol is not None
        )
        mapper_parent = mapper_provenance["provenance"]
        if mapper_parent.get("formal_oracle_evaluation") is not True:
            raise ValueError("formal H requires a formal-oracle mapper replay")
        if not isinstance(mapper_parent.get("formal_source_contract"), dict):
            raise ValueError("formal H mapper lacks a source contract")
        require_compatible_contracts(
            "mapper", mapper_parent["formal_source_contract"], formal_contract
        )
        if train_dataset.manifest_sha256 != formal_assets["dataset_manifest_sha256"]:
            raise ValueError("formal H dataset asset mismatch")
        require_asset_binding("frozen_s", s_provenance, formal_assets["frozen_s"])
        require_asset_binding("clean_f", f_provenance, formal_assets["clean_f"])
        mapper_checkpoint_path = Path(mapper_provenance["path"])
        mapper_manifest_path = mapper_checkpoint_path.parent / "run_manifest.json"
        mapper_manifest = json.loads(
            mapper_manifest_path.read_text(encoding="utf-8")
        )
        require_formal_checkpoint_manifest(
            "formal H mapper parent",
            mapper_checkpoint_path,
            mapper_manifest,
            expected_update=int(
                formal_root_protocol["mapper"]["fixed_final_update"]
            ),
            checkpoint_update=int(mapper_provenance["update"]),
        )
    if (
        mapper_provenance["provenance"]["frozen_s"]["state_dict_sha256"]
        != s_provenance["state_dict_sha256"]
    ):
        raise ValueError("H mapper and supplied frozen S checkpoints differ")
    session_overlap = len(train_dataset.session_set & validation_dataset.session_set)
    sample_overlap = len(
        train_dataset.sample_key_set & validation_dataset.sample_key_set
    )
    if not args.validation_from_train and session_overlap:
        raise ValueError("formal H train/validation sessions overlap")
    if args.formal_oracle:
        split_isolation = require_formal_split_isolation(
            train_dataset, validation_dataset
        )
    else:
        split_isolation = None
    source_names = (
        Path(__file__).name,
        "pnp_q0_hypothesis_adapter.py",
        "pnp_observation_mapper.py",
        "observable_future_pnp_ab.py",
        "cyclic_future_foundation.py",
        "cyclic_anchor_edge_model.py",
        "cyclic_track_model.py",
    )
    source_sha256 = {
        name: sha256_file(Path(__file__).with_name(name))
        for name in source_names
    }
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        generator=train_generator,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    model = C4Q0HypothesisAdapter(
        s_model.position_mean.detach().cpu(), s_model.position_std.detach().cpu(),
        channels=args.channels, dropout=args.dropout,
        message_layers=args.message_layers, age_scale_s=args.age_scale_s,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > args.maximum_parameters:
        raise ValueError(f"H parameter count {parameter_count} exceeds cap")
    optimizer = build_h_optimizer(
        model, learning_rate=args.learning_rate, weight_decay=args.weight_decay
    )
    frozen_ids = {
        id(parameter)
        for frozen in (mapper, s_model, f_model)
        for parameter in frozen.parameters()
    }
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    if frozen_ids & optimizer_ids:
        raise RuntimeError("H optimizer contains frozen mapper/S/F parameters")

    planned_updates = args.epochs * len(train_loader)
    total_updates = min(planned_updates, args.max_updates) if args.max_updates > 0 else planned_updates
    stop_update = args.max_updates
    if args.formal_oracle:
        assert formal_protocol is not None
        total_updates, stop_update = resolve_formal_schedule(
            args.max_updates, planned_updates, formal_protocol
        )
    if total_updates <= 0:
        raise ValueError("H training needs at least one update")
    git = _git_state()
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    update = 0
    start_epoch = 1
    elapsed_before_s = 0.0
    recovery_resume: dict[str, Any] | None = None
    if args.resume:
        assert formal_contract is not None
        recovery_path, recovery = _load_recovery_state(output)
        if recovery.get("formal_source_contract") != formal_contract:
            raise ValueError("formal H recovery source contract mismatch")
        if recovery.get("training_arguments") != _recovery_training_arguments(args):
            raise ValueError("formal H recovery training arguments mismatch")
        if recovery.get("dataset_manifest_sha256") != train_dataset.manifest_sha256:
            raise ValueError("formal H recovery dataset manifest mismatch")
        expected_frozen = {"mapper": mapper_state, "s": s_state, "f": f_state}
        if recovery.get("frozen_state_dict_sha256") != expected_frozen:
            raise ValueError("formal H recovery frozen state mismatch")
        if recovery.get("model_config") != model.config:
            raise ValueError("formal H recovery model configuration mismatch")
        if (
            recovery.get("validation_accessed") is not False
            or recovery.get("model_selection_performed") is not False
            or recovery.get("test_accessed") is not False
        ):
            raise ValueError("formal H recovery state crossed an evaluation boundary")
        completed_epoch = int(recovery.get("completed_epoch", -1))
        recovered_update = int(recovery.get("update", -1))
        if (
            completed_epoch < 1
            or completed_epoch >= args.epochs
            or recovered_update != completed_epoch * len(train_loader)
            or recovered_update >= stop_update
        ):
            raise ValueError("formal H recovery epoch/update boundary is invalid")
        elapsed_before_s = float(recovery.get("elapsed_s", -1.0))
        if not math.isfinite(elapsed_before_s) or elapsed_before_s < 0.0:
            raise ValueError("formal H recovery elapsed time is invalid")
        model.load_state_dict(recovery["model"], strict=True)
        optimizer.load_state_dict(recovery["optimizer"])
        _move_optimizer_state_to_device(optimizer, device)
        _restore_rng_state(recovery["rng_state"], train_generator)
        update = recovered_update
        start_epoch = completed_epoch + 1
        recovery_resume = {
            "path": str(recovery_path),
            "sha256": sha256_file(recovery_path),
            "completed_epoch": completed_epoch,
            "update": recovered_update,
        }
    bad_rounds = 0
    stop_reason = "epoch_limit"
    started = time.time()
    elapsed_now = lambda: elapsed_before_s + time.time() - started
    stop = False
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        sums = {
            "objective": 0.0, "q0_role_macro": 0.0, "directed_edge": 0.0,
            "warm_opposite": 0.0, "support": 0.0,
            "sigma_calibration": 0.0, "c4": 0.0,
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
            _, s_output = frozen_mapper_s_forward(mapper, s_model, batch)
            clean_s_output = frozen_clean_s_forward(s_model, batch)
            optimizer.zero_grad(set_to_none=True)
            prediction = hypothesis_forward(model, s_output)
            equivariance = c4_equivariance_loss(
                model, s_output, prediction, shift=1 + (update % 3)
            )
            mapped_objective, mapped_components = hypothesis_loss(
                prediction, batch["pnp_s_truth_q0_m"],
                huber_beta_m=args.huber_beta_m,
                q0_weight=args.q0_weight, edge_weight=args.edge_weight,
                opposite_weight=args.opposite_weight,
                support_weight=args.support_weight, sigma_weight=args.sigma_weight,
                c4_loss=equivariance, c4_weight=args.c4_weight,
            )
            clean_prediction = hypothesis_forward(model, clean_s_output)
            clean_objective, clean_components = hypothesis_loss(
                clean_prediction, batch["pnp_s_truth_q0_m"],
                huber_beta_m=args.huber_beta_m,
                q0_weight=args.q0_weight, edge_weight=args.edge_weight,
                opposite_weight=args.opposite_weight,
                support_weight=args.support_weight, sigma_weight=args.sigma_weight,
                c4_loss=None, c4_weight=0.0,
            )
            objective = mapped_objective + args.clean_weight * clean_objective
            components = {
                name: mapped_components[name] + args.clean_weight * clean_components[name]
                for name in mapped_components
            }
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            sums["objective"] += float(objective.detach())
            for name, value in components.items():
                sums[name] += float(value.detach())
            batches += 1
            if stop_update > 0 and update >= stop_update:
                stop = True
                stop_reason = (
                    "fixed_final_update" if args.formal_oracle
                    else "max_updates"
                )
                break

        validate_now = (
            (stop or epoch == args.epochs)
            if args.formal_oracle else
            (epoch == 1 or epoch % args.validation_interval == 0
             or stop or epoch == args.epochs)
        )
        if validate_now:
            metrics = evaluate(
                model, mapper, s_model, validation_loader, device, domain="mapped"
            )
            clean_metrics = evaluate(
                model, mapper, s_model, validation_loader, device, domain="clean"
            )
            metrics["clean_domain"] = clean_metrics
            supported = metrics["h_q0_supported"]
            clean_supported = clean_metrics["h_q0_supported"]
            warm_opposite = metrics["h_q0_by_role"]["warm_opposite"]
            warm_adjacent = metrics["h_q0_by_role"]["warm_adjacent"]
            selection = (
                float(supported["p95_m"]),
                float(clean_supported["p95_m"]),
                float(warm_opposite["p95_m"] or math.inf),
                float(warm_adjacent["p95_m"] or math.inf),
            )
            checkpoint_name = f"epoch-{epoch:04d}-update-{update:06d}.pt"
            checkpoint_path = output / checkpoint_name
            if checkpoint_path.exists() and args.formal_oracle:
                checkpoint_name = (
                    f"epoch-{epoch:04d}-update-{update:06d}"
                    f"-recovered-{time.time_ns()}.pt"
                )
                checkpoint_path = output / checkpoint_name
            checkpoint = {
                "schema_version": RUN_SCHEMA,
                "model_class": "C4Q0HypothesisAdapter",
                "epoch": epoch, "update": update,
                "model_config": model.config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "validation": metrics, "selection": selection,
                "provenance": {
                    "dataset_manifest_path": str(
                        Path(args.dataset).resolve() / "dataset_manifest.json"
                    ),
                    "dataset_manifest_sha256": train_dataset.manifest_sha256,
                    "frozen_mapper": mapper_provenance,
                    "frozen_s": s_provenance,
                    "frozen_f": f_provenance,
                    "git": git,
                    "test_accessed": False,
                    "oracle_association": True,
                    "deployable_pipeline": False,
                    "optimizer_only_h": True,
                    "canonical_direction_reflection_removed": True,
                    "window_local_c4_origin_retained": True,
                    "cold_coordinate_supervision": False,
                    "validation_from_train": bool(args.validation_from_train),
                    "diagnostic_only": (
                        bool(args.validation_from_train)
                        if not args.formal_oracle else False
                    ),
                    "formal_oracle_evaluation": bool(args.formal_oracle),
                    "fixed_final_checkpoint": bool(args.formal_oracle),
                    "full_chain_provenance_clean": False,
                    "formal_source_contract": formal_contract,
                    "recovery_resume": recovery_resume,
                    "support_metric_diagnostic_only": True,
                    "support_output_consumed_by_f": False,
                    "train_split_audit": train_dataset.split_audit,
                    "validation_split_audit": validation_dataset.split_audit,
                    "train_validation_session_overlap_count": session_overlap,
                    "train_validation_sample_key_overlap_count": sample_overlap,
                    "split_isolation": split_isolation,
                    "source_sha256": source_sha256,
                },
            }
            _atomic_checkpoint(checkpoint_path, checkpoint)
            item = {
                "epoch": epoch, "update": update, "learning_rate": lr,
                "train": {name: value / max(batches, 1) for name, value in sums.items()},
                "validation": metrics, "selection": selection,
                "checkpoint": checkpoint_name,
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
            history.append(item)
            if args.formal_oracle or best is None or selection < tuple(best["selection"]):
                best = {
                    "epoch": epoch, "update": update, "path": checkpoint_name,
                    "sha256": item["checkpoint_sha256"],
                    "selection": selection, "validation": metrics,
                }
                bad_rounds = 0
            else:
                bad_rounds += 1
            elapsed = elapsed_now()
            rate = update / max(elapsed, 1e-6)
            print(json.dumps({
                "epoch": epoch, "update": update,
                "h_supported_p95_m": supported["p95_m"],
                "h_warm_adjacent_p95_m": warm_adjacent["p95_m"],
                "h_warm_opposite_p95_m": warm_opposite["p95_m"],
                "clean_h_supported_p95_m": clean_supported["p95_m"],
                "support_accuracy": metrics["support_accuracy"],
                "c4_max_abs": metrics["c4_max_abs"],
                "elapsed_s": elapsed,
                "eta_s": max(total_updates - update, 0) / max(rate, 1e-9),
            }, sort_keys=True), flush=True)
            progress = {
                "schema_version": RUN_SCHEMA,
                "status": "running", "epoch": epoch, "update": update,
                "best": best, "history": history,
                "elapsed_s": elapsed,
                "train_sample_count": len(train_dataset),
                "validation_sample_count": len(validation_dataset),
                "validation_from_train": bool(args.validation_from_train),
                "dataset_manifest_sha256": train_dataset.manifest_sha256,
                "source_direction_counts": train_dataset.source_direction_counts,
                "c4_origin_counts": train_dataset.c4_origin_counts,
                "sample_strategy": train_dataset.sample_strategy,
                "parameter_count": parameter_count,
                "maximum_parameters": args.maximum_parameters,
                "model_config": model.config,
                "training_arguments": vars(args),
                "frozen_mapper": mapper_provenance,
                "frozen_s": s_provenance,
                "frozen_f": f_provenance,
                "frozen_mapper_verified_unchanged": state_dict_sha256(mapper.state_dict()) == mapper_state,
                "frozen_s_verified_unchanged": state_dict_sha256(s_model.state_dict()) == s_state,
                "frozen_f_verified_unchanged": state_dict_sha256(f_model.state_dict()) == f_state,
                "optimizer_only_h": True,
                "test_accessed": False,
                "oracle_association": True,
                "deployable_pipeline": False,
                "formal_oracle_evaluation": bool(args.formal_oracle),
                "fixed_final_checkpoint": bool(args.formal_oracle),
                "full_chain_provenance_clean": False,
                "formal_source_contract": formal_contract,
                "recovery_resume": recovery_resume,
                "cold_coordinate_supervision": False,
                "source_sha256": source_sha256,
                "train_split_audit": train_dataset.split_audit,
                "validation_split_audit": validation_dataset.split_audit,
                "train_validation_session_overlap_count": session_overlap,
                "train_validation_sample_key_overlap_count": sample_overlap,
                "split_isolation": split_isolation,
                "git": git,
            }
            _atomic_json(output / "run_progress.json", progress)
            if (
                not args.formal_oracle
                and args.patience > 0 and bad_rounds >= args.patience
            ):
                stop = True
                stop_reason = "early_stopping"
        epoch_completed = batches == len(train_loader) and not stop
        if (
            args.formal_oracle
            and epoch_completed
            and epoch % int(formal_protocol["recovery_epoch_interval"]) == 0
        ):
            recovery_path = _write_recovery_state(
                output,
                epoch=epoch,
                update=update,
                elapsed_s=elapsed_now(),
                model=model,
                optimizer=optimizer,
                train_generator=train_generator,
                arguments=args,
                formal_contract=formal_contract,
                dataset_manifest_sha256=train_dataset.manifest_sha256,
                mapper_state_sha256=mapper_state,
                s_state_sha256=s_state,
                f_state_sha256=f_state,
            )
            print(json.dumps({
                "epoch": epoch,
                "update": update,
                "recovery_checkpoint": str(recovery_path),
                "validation_accessed": False,
                "model_selection_performed": False,
                "elapsed_s": elapsed_now(),
            }, sort_keys=True), flush=True)
        if stop:
            break
    if best is None:
        raise RuntimeError("H training produced no validation checkpoint")
    for frozen in (mapper, s_model, f_model):
        if any(parameter.grad is not None for parameter in frozen.parameters()):
            raise RuntimeError("frozen mapper/S/F received gradients")
    final_frozen_hashes = {
        "mapper": state_dict_sha256(mapper.state_dict()) == mapper_state,
        "s": state_dict_sha256(s_model.state_dict()) == s_state,
        "f": state_dict_sha256(f_model.state_dict()) == f_state,
    }
    if not all(final_frozen_hashes.values()):
        raise RuntimeError("H training changed a frozen mapper/S/F state hash")
    final_source_sha256 = {
        name: sha256_file(Path(__file__).with_name(name))
        for name in source_names
    }
    if final_source_sha256 != source_sha256:
        raise RuntimeError("H training source bundle changed during the run")
    if args.formal_oracle:
        assert formal_contract is not None and formal_protocol is not None
        require_fixed_final_state(
            "formal H", formal_protocol,
            update=update, stop_reason=stop_reason,
            history=history, best=best,
        )
        verify_formal_contract(formal_contract)
    manifest = json.loads((output / "run_progress.json").read_text(encoding="utf-8"))
    manifest.update({
        "status": "complete", "stop_reason": stop_reason,
        "elapsed_s": elapsed_now(),
        "frozen_mapper_verified_unchanged": final_frozen_hashes["mapper"],
        "frozen_s_verified_unchanged": final_frozen_hashes["s"],
        "frozen_f_verified_unchanged": final_frozen_hashes["f"],
    })
    if args.formal_oracle:
        metrics = best["validation"]
        clean = metrics["clean_domain"]
        gates = formal_protocol["gates"]
        manifest["formal_gates"] = {
            "mapped_supported_q0_p95": (
                float(metrics["h_q0_supported"]["p95_m"])
                <= gates["mapped_supported_q0_p95_m_max"]
            ),
            "mapped_warm_adjacent_p95": (
                float(metrics["h_q0_by_role"]["warm_adjacent"]["p95_m"])
                <= gates["mapped_warm_adjacent_p95_m_max"]
            ),
            "mapped_warm_opposite_p95": (
                float(metrics["h_q0_by_role"]["warm_opposite"]["p95_m"])
                <= gates["mapped_warm_opposite_p95_m_max"]
            ),
            "clean_supported_q0_p95": (
                float(clean["h_q0_supported"]["p95_m"])
                <= gates["clean_supported_q0_p95_m_max"]
            ),
            "c4": float(metrics["c4_max_abs"]) <= gates["c4_max_abs_max"],
            "support_accuracy": (
                float(metrics["support_accuracy"])
                >= gates["support_accuracy_min"]
            ),
            "cold_coordinate_unsupervised": (
                int(metrics["true_cold_coordinate_metric_count"])
                == gates["true_cold_coordinate_metric_count"]
            ),
            "no_session_overlap": session_overlap == 0,
            "no_sample_overlap": sample_overlap == 0,
            "frozen_components_unchanged": all(final_frozen_hashes.values()),
        }
        manifest["formal_gate_passed"] = all(manifest["formal_gates"].values())
    _atomic_json(output / "run_manifest.json", manifest)
    return output / "run_manifest.json"


def train(args: argparse.Namespace) -> Path:
    output = Path(args.output).resolve()
    if args.resume:
        if not output.is_dir():
            raise FileNotFoundError("formal H resume output directory is missing")
        if (output / "run_manifest.json").exists():
            raise RuntimeError("formal H run is already complete")
    else:
        output.mkdir(parents=True, exist_ok=False)
    lock_path, lock_token, lock_stream = _acquire_run_lock(output)
    try:
        return _train_locked(args, output)
    finally:
        _release_run_lock(lock_path, lock_token, lock_stream)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", required=True)
    result.add_argument("--mapper-checkpoint", required=True)
    result.add_argument("--s-checkpoint", required=True)
    result.add_argument("--f-checkpoint", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--formal-oracle", action="store_true")
    result.add_argument(
        "--resume", action="store_true",
        help="resume a formal-oracle run from its latest full-epoch state",
    )
    result.add_argument("--formal-protocol", default="")
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int, default=20260727)
    result.add_argument("--channels", type=int, default=64)
    result.add_argument("--dropout", type=float, default=0.05)
    result.add_argument("--message-layers", type=int, default=2)
    result.add_argument("--age-scale-s", type=float, default=0.32)
    result.add_argument("--maximum-parameters", type=int, default=150000)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--epochs", type=int, default=80)
    result.add_argument("--max-updates", type=int, default=5000)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--minimum-learning-rate", type=float, default=3e-6)
    result.add_argument("--warmup-updates", type=int, default=200)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--gradient-clip-norm", type=float, default=1.0)
    result.add_argument("--huber-beta-m", type=float, default=0.01)
    result.add_argument("--q0-weight", type=float, default=1.0)
    result.add_argument("--edge-weight", type=float, default=0.5)
    result.add_argument("--opposite-weight", type=float, default=0.5)
    result.add_argument("--support-weight", type=float, default=0.2)
    result.add_argument("--sigma-weight", type=float, default=0.2)
    result.add_argument("--c4-weight", type=float, default=0.1)
    result.add_argument("--clean-weight", type=float, default=0.5)
    result.add_argument("--validation-interval", type=int, default=1)
    result.add_argument("--patience", type=int, default=8)
    result.add_argument("--train-limit", type=int, default=0)
    result.add_argument("--validation-limit", type=int, default=0)
    result.add_argument("--validation-from-train", action="store_true")
    result.add_argument(
        "--motion-class", type=int, default=-1,
        help="selection-only optional class; never passed to mapper/S/H",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    positive = (
        args.learning_rate, args.minimum_learning_rate, args.gradient_clip_norm,
        args.huber_beta_m, args.maximum_parameters,
    )
    if min(positive) <= 0:
        raise ValueError("positive H optimization arguments required")
    if min(
        args.q0_weight, args.edge_weight, args.opposite_weight,
        args.support_weight, args.sigma_weight, args.c4_weight,
        args.clean_weight,
    ) < 0:
        raise ValueError("H loss weights cannot be negative")
    if args.motion_class not in {-1, 0, 1, 2, 3}:
        raise ValueError("H motion class must be -1 or 0..3")
    print(train(args))


if __name__ == "__main__":
    main()
