"""Train only frozen-F switch heads on the accepted A3 PnP hypothesis stream."""

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
from torch.utils.data import DataLoader

from .cyclic_future_foundation import load_frozen_v19
from .evaluate_pnp_q0_hypothesis_sf import (
    _add_prediction,
    _collector,
    _finish,
    canonicalize_direction_keep_c4,
)
from .observable_future_pnp_ab import (
    ObservableFuturePnPSFDataset,
    load_observable_f_checkpoint,
    sha256_file,
    state_dict_sha256,
)
from .pnp_q0_hypothesis_adapter import (
    compose_hypothesis_for_f,
    hypothesis_forward,
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .train_causal_physical_ab import _git_state, _seed, _to_device
from .train_observable_future_dedicated_selector import (
    FrozenTrajectoryDedicatedSelector,
    _module_state_sha256,
)
from .train_observable_future_selector import (
    _scheduled_learning_rate,
    _selector_loss,
)
from .train_pnp_q0_hypothesis_adapter import _s_forward


RUN_SCHEMA = "stage3-pnp-q0-hypothesis-selector-run-v1"
PNP_SELECTOR_TRAINABLE_PREFIXES = (
    "switch_candidate_head.",
    "switch_logit.",
)


def _atomic_json(path: Path, payload: object) -> None:
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}-{time.time_ns()}")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pending.replace(path)


def _atomic_checkpoint(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite PnP selector checkpoint: {path}")
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}-{time.time_ns()}")
    try:
        torch.save(payload, pending)
        pending.replace(path)
    finally:
        if pending.exists():
            pending.unlink()


def freeze_pnp_selector(
    system: FrozenTrajectoryDedicatedSelector,
) -> tuple[list[str], list[str]]:
    system.trajectory_model.eval().requires_grad_(False)
    trainable: list[str] = []
    frozen: list[str] = []
    for name, parameter in system.selector_model.named_parameters():
        selected = name.startswith(PNP_SELECTOR_TRAINABLE_PREFIXES)
        parameter.requires_grad_(selected)
        (trainable if selected else frozen).append(name)
    if not trainable or not frozen:
        raise ValueError("PnP selector freeze did not partition F")
    return trainable, frozen


def set_pnp_selector_train_mode(system: FrozenTrajectoryDedicatedSelector) -> None:
    system.eval()
    system.selector_model.switch_candidate_head.train()
    system.selector_model.switch_logit.train()


def _verify_loaded_record(
    name: str,
    recorded: dict[str, Any],
    loaded: dict[str, Any],
) -> None:
    for key in ("sha256", "state_dict_sha256"):
        if recorded.get(key) != loaded.get(key):
            raise ValueError(f"PnP selector {name} {key} provenance mismatch")


def _validate_selector_run_manifest(
    path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("PnP selector requires final run_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    best = manifest.get("best", {})
    if (
        manifest.get("schema_version") != RUN_SCHEMA
        or manifest.get("status") != "complete"
        or not all(bool(value) for value in manifest.get("gates", {}).values())
        or best.get("path") != path.name
        or best.get("sha256") != sha256_file(path)
        or best.get("epoch") != payload.get("epoch")
        or best.get("update") != payload.get("update")
        or best.get("selection") != list(payload.get("selection", ()))
        or best.get("validation") != payload.get("validation")
    ):
        raise ValueError("PnP selector is not the accepted manifest-declared best")
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "status": manifest["status"],
        "gates": manifest["gates"],
    }


def load_frozen_pnp_selector(
    checkpoint_path: str | Path,
) -> tuple[
    FrozenTrajectoryDedicatedSelector,
    torch.nn.Module,
    torch.nn.Module,
    torch.nn.Module,
    dict[str, Any],
]:
    """Strictly reconstruct a selector and its complete frozen A3 input stack."""
    path = Path(checkpoint_path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != RUN_SCHEMA
        or payload.get("model_class")
        != "PnPQ0HypothesisFrozenTrajectorySelector"
    ):
        raise ValueError("PnP selector checkpoint schema mismatch")
    run_manifest = _validate_selector_run_manifest(path, payload)
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("PnP selector checkpoint provenance is missing")
    required = (
        "mapper", "frozen_s", "hypothesis_adapter", "frozen_f",
        "selector_source",
    )
    if any(not isinstance(provenance.get(name), dict) for name in required):
        raise ValueError("PnP selector checkpoint frozen provenance is incomplete")
    expected_contract = {
        "test_accessed": False,
        "oracle_association": True,
        "deployable_pipeline": False,
        "optimizer_only_switch_heads": True,
        "conditional_trajectory_trainable": False,
    }
    for name, expected in expected_contract.items():
        if provenance.get(name) is not expected:
            raise ValueError(f"PnP selector contract mismatch: {name}")

    mapper, mapper_info = load_frozen_pnp_mapper(provenance["mapper"]["path"])
    s_model, s_info = load_frozen_v19(provenance["frozen_s"]["checkpoint_path"])
    h_model, h_info = load_frozen_hypothesis_adapter(
        provenance["hypothesis_adapter"]["path"]
    )
    trajectory_model, f_info = load_observable_f_checkpoint(
        provenance["frozen_f"]["path"]
    )
    selector_model, selector_source_info = load_observable_f_checkpoint(
        provenance["selector_source"]["path"]
    )
    _verify_loaded_record("mapper", provenance["mapper"], mapper_info)
    _verify_loaded_record("H", provenance["hypothesis_adapter"], h_info)
    _verify_loaded_record("F", provenance["frozen_f"], f_info)
    _verify_loaded_record(
        "selector source", provenance["selector_source"], selector_source_info
    )
    if (
        provenance["frozen_s"].get("state_dict_sha256")
        != s_info.get("state_dict_sha256")
    ):
        raise ValueError("PnP selector S state provenance mismatch")
    if (
        provenance["frozen_s"].get("checkpoint_sha256")
        != s_info.get("checkpoint_sha256")
    ):
        raise ValueError("PnP selector S file provenance mismatch")
    if (
        mapper_info["provenance"]["frozen_s"]["state_dict_sha256"]
        != s_info["state_dict_sha256"]
    ):
        raise ValueError("PnP selector mapper/S provenance chain is broken")
    h_frozen = h_info["provenance"]
    for name, actual in (
        ("frozen_mapper", mapper_info["state_dict_sha256"]),
        ("frozen_s", s_info["state_dict_sha256"]),
        ("frozen_f", f_info["state_dict_sha256"]),
    ):
        if h_frozen[name]["state_dict_sha256"] != actual:
            raise ValueError(f"PnP selector H/{name} provenance chain is broken")
    if trajectory_model.config != payload.get("model_config"):
        raise ValueError("PnP selector trajectory model config mismatch")
    if selector_model.config != trajectory_model.config:
        raise ValueError("PnP selector source/trajectory configs differ")
    if (
        provenance.get("dataset_manifest_sha256")
        != h_info["provenance"]["dataset_manifest_sha256"]
    ):
        raise ValueError("PnP selector H/dataset provenance chain is broken")
    if (
        provenance.get("dataset_manifest_sha256")
        != mapper_info["provenance"]["dataset_manifest_sha256"]
    ):
        raise ValueError("PnP selector mapper/dataset provenance chain is broken")

    system = FrozenTrajectoryDedicatedSelector(trajectory_model, selector_model)
    expected_names = {
        name for name in system.selector_model.state_dict()
        if name.startswith(PNP_SELECTOR_TRAINABLE_PREFIXES)
    }
    selector_state = payload.get("selector_state")
    if not isinstance(selector_state, dict) or set(selector_state) != expected_names:
        raise ValueError("PnP selector state is not exactly the switch-head subset")
    restored = system.selector_model.state_dict()
    restored.update(selector_state)
    system.selector_model.load_state_dict(restored, strict=True)
    trainable_names, frozen_names = freeze_pnp_selector(system)
    selector_sha = _module_state_sha256(
        system.selector_model, prefixes=PNP_SELECTOR_TRAINABLE_PREFIXES
    )
    if selector_sha != payload.get("selector_state_sha256"):
        raise ValueError("PnP selector switch-head state hash mismatch")
    frozen_selector_sha = _module_state_sha256(
        system.selector_model,
        prefixes=PNP_SELECTOR_TRAINABLE_PREFIXES,
        invert=True,
    )
    source_frozen_sha = _module_state_sha256(
        trajectory_model,
        prefixes=PNP_SELECTOR_TRAINABLE_PREFIXES,
        invert=True,
    )
    if frozen_selector_sha != source_frozen_sha:
        raise ValueError("PnP selector changed a frozen F parameter")
    system.requires_grad_(False)
    for model in (mapper, s_model, h_model, system):
        model.eval()
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("frozen PnP selector loader returned trainable state")
    return system, mapper, s_model, h_model, {
        "path": str(path),
        "sha256": sha256_file(path),
        "epoch": int(payload["epoch"]),
        "update": int(payload["update"]),
        "selector_state_sha256": selector_sha,
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
        "provenance": provenance,
        "validation": payload.get("validation"),
        "run_manifest": run_manifest,
    }


@torch.no_grad()
def compose_a3_batch(
    mapper: torch.nn.Module,
    s_model: torch.nn.Module,
    h_model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
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
    return compose_hypothesis_for_f(
        h_output, s_output["primary_index"], mapped["corrected_obs_m"],
        batch["pnp_s_obs_mask"], batch["pnp_s_primary_mask"],
        batch["pnp_candidate_step"],
    )


def _system_forward(
    system: FrozenTrajectoryDedicatedSelector,
    batch: dict[str, torch.Tensor],
    composed: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return system(
        composed["history_position_rel_m"],
        batch["pnp_history_time_s"], batch["pnp_history_dt_s"],
        batch["pnp_history_switch_step"], batch["pnp_history_mask"],
        composed["current_position_m"], composed["candidate_relation_m"],
        batch["pnp_candidate_step"], batch["pnp_candidate_mask"],
        composed["candidate_confidence"], batch["pnp_tau_s"],
    )


def _metric_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = dict(batch)
    for suffix in (
        "candidate_step", "candidate_mask", "candidate_confidence", "tau_s"
    ):
        result[suffix] = batch[f"pnp_{suffix}"]
    return result


def _selector_loss_prediction(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    result = dict(prediction)
    result["conditional_delta_m"] = (
        prediction["conditional_position_m"]
        - batch["current_position_m"][:, None, None, :]
    )
    return result


@torch.no_grad()
def evaluate_selector(
    system: FrozenTrajectoryDedicatedSelector,
    mapper: torch.nn.Module,
    s_model: torch.nn.Module,
    h_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    system.eval()
    collector = _collector()
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        composed = compose_a3_batch(mapper, s_model, h_model, batch)
        prediction = _system_forward(system, batch, composed)
        metric_batch = _metric_batch(batch)
        _add_prediction(
            collector, prediction, composed["current_position_m"], metric_batch,
            composed["candidate_support_class"],
        )
    result = _finish(collector)
    step_accuracies = [
        float(value["switch_accuracy"])
        for value in result["by_signed_step"].values()
        if value["switch_accuracy"] is not None
    ]
    result["switch_macro_recall"] = float(np.mean(step_accuracies))
    result["switch_minimum_step_recall"] = float(np.min(step_accuracies))
    return result


def _selector_checkpoint_state(
    system: FrozenTrajectoryDedicatedSelector,
) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in system.selector_model.state_dict().items()
        if name.startswith(PNP_SELECTOR_TRAINABLE_PREFIXES)
    }


def train(args: argparse.Namespace) -> Path:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dataset = {
        split: ObservableFuturePnPSFDataset(args.dataset, split, motion_class=3)
        for split in ("train", "validation")
    }
    if set(dataset["train"].pair_ids) & set(dataset["validation"].pair_ids):
        raise ValueError("PnP selector train/validation pairs overlap")
    for split in ("train", "validation"):
        canonicalize_direction_keep_c4(
            dataset[split].tensors, dataset[split].pair_ids
        )
    mapper, mapper_provenance = load_frozen_pnp_mapper(args.mapper_checkpoint)
    s_model, s_provenance = load_frozen_v19(args.s_checkpoint)
    h_model, h_provenance = load_frozen_hypothesis_adapter(args.h_checkpoint)
    trajectory_model, f_provenance = load_observable_f_checkpoint(args.f_checkpoint)
    selector_model, selector_source = load_observable_f_checkpoint(args.f_checkpoint)
    if dataset["train"].manifest_sha256 != h_provenance["provenance"]["dataset_manifest_sha256"]:
        raise ValueError("PnP selector and H dataset manifests differ")
    if h_provenance["provenance"]["frozen_mapper"]["state_dict_sha256"] != mapper_provenance["state_dict_sha256"]:
        raise ValueError("PnP selector mapper differs from H provenance")
    if h_provenance["provenance"]["frozen_s"]["state_dict_sha256"] != s_provenance["state_dict_sha256"]:
        raise ValueError("PnP selector S differs from H provenance")
    if mapper_provenance["provenance"]["frozen_s"]["state_dict_sha256"] != s_provenance["state_dict_sha256"]:
        raise ValueError("PnP selector mapper differs from supplied S")
    if h_provenance["provenance"]["frozen_f"]["state_dict_sha256"] != f_provenance["state_dict_sha256"]:
        raise ValueError("PnP selector F differs from H provenance")
    system = FrozenTrajectoryDedicatedSelector(trajectory_model, selector_model)
    trainable_names, frozen_names = freeze_pnp_selector(system)
    for model in (mapper, s_model, h_model):
        model.eval().requires_grad_(False)
    mapper.to(device)
    s_model.to(device)
    h_model.to(device)
    system.to(device)
    initial_hashes = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
        "trajectory_f": _module_state_sha256(system.trajectory_model),
        "selector_frozen": _module_state_sha256(
            system.selector_model, prefixes=PNP_SELECTOR_TRAINABLE_PREFIXES,
            invert=True,
        ),
    }
    optimizer = torch.optim.AdamW(
        (parameter for parameter in system.selector_model.parameters()
         if parameter.requires_grad),
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    trainable_ids = {
        id(parameter) for parameter in system.selector_model.parameters()
        if parameter.requires_grad
    }
    if optimizer_ids != trainable_ids:
        raise RuntimeError("PnP selector optimizer is not exactly the switch heads")
    train_loader = DataLoader(
        dataset["train"], batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        dataset["validation"], batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    baseline = evaluate_selector(
        system, mapper, s_model, h_model, validation_loader, device
    )
    probe_batch = _to_device(next(iter(validation_loader)), device)
    with torch.no_grad():
        probe_composed = compose_a3_batch(mapper, s_model, h_model, probe_batch)
        probe_reference = _system_forward(system, probe_batch, probe_composed)
        probe_conditional = probe_reference["conditional_delta_m"].cpu()
    provenance = {
        "schema_version": RUN_SCHEMA,
        "dataset_manifest_sha256": dataset["train"].manifest_sha256,
        "mapper": mapper_provenance, "frozen_s": s_provenance,
        "hypothesis_adapter": h_provenance, "frozen_f": f_provenance,
        "selector_source": selector_source,
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
        "optimizer_only_switch_heads": True,
        "conditional_trajectory_trainable": False,
        "direction_contract": "reflection removed; window-local C4 retained",
        "test_accessed": False, "oracle_association": True,
        "deployable_pipeline": False, "git": _git_state(),
        "training_arguments": vars(args),
    }
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    update = 0
    bad_rounds = 0
    stop = False
    stop_reason = "epoch_limit"
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        set_pnp_selector_train_mode(system)
        objective_sum = 0.0
        batches = 0
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            next_update = update + 1
            lr = _scheduled_learning_rate(
                next_update, base_learning_rate=args.learning_rate,
                minimum_learning_rate=args.minimum_learning_rate,
                warmup_updates=args.warmup_updates, total_updates=args.max_updates,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            composed = compose_a3_batch(mapper, s_model, h_model, batch)
            optimizer.zero_grad(set_to_none=True)
            prediction = _system_forward(system, batch, composed)
            objective, _ = _selector_loss(
                _selector_loss_prediction(prediction, batch),
                _metric_batch(batch), args,
            )
            objective.backward()
            for model in (mapper, s_model, h_model, system.trajectory_model):
                if any(parameter.grad is not None for parameter in model.parameters()):
                    raise RuntimeError("PnP selector sent gradients into a frozen model")
            frozen_named = dict(system.selector_model.named_parameters())
            if any(frozen_named[name].grad is not None for name in frozen_names):
                raise RuntimeError("PnP selector sent gradients into frozen F paths")
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in system.selector_model.parameters()
                 if parameter.requires_grad), args.gradient_clip_norm,
            )
            optimizer.step()
            update = next_update
            objective_sum += float(objective.detach())
            batches += 1
            if update >= args.max_updates:
                stop = True
                stop_reason = "max_updates"
                break
        validate_now = (
            epoch == 1 or epoch % args.validation_interval == 0
            or stop or epoch == args.epochs
        )
        if validate_now:
            metrics = evaluate_selector(
                system, mapper, s_model, h_model, validation_loader, device
            )
            if metrics["conditional_position"] != baseline["conditional_position"]:
                raise RuntimeError("PnP selector changed frozen conditional metrics")
            with torch.no_grad():
                current_probe = _system_forward(system, probe_batch, probe_composed)
            if not torch.equal(current_probe["conditional_delta_m"].cpu(), probe_conditional):
                raise RuntimeError("PnP selector changed conditional trajectory tensors")
            current_hashes = {
                "mapper": state_dict_sha256(mapper.state_dict()),
                "s": state_dict_sha256(s_model.state_dict()),
                "h": state_dict_sha256(h_model.state_dict()),
                "trajectory_f": _module_state_sha256(system.trajectory_model),
                "selector_frozen": _module_state_sha256(
                    system.selector_model, prefixes=PNP_SELECTOR_TRAINABLE_PREFIXES,
                    invert=True,
                ),
            }
            if current_hashes != initial_hashes:
                raise RuntimeError("PnP selector changed a frozen state hash")
            metrics["conditional_probe_bit_exact"] = True
            hard = metrics["hard_position"]
            selection = (
                float(hard["p95_m"]), float(hard["p99_m"]),
                -float(metrics["switch_accuracy"]),
                -float(metrics["switch_macro_recall"]),
            )
            checkpoint_name = f"epoch-{epoch:04d}-update-{update:06d}.pt"
            checkpoint_path = output / checkpoint_name
            checkpoint = {
                "schema_version": RUN_SCHEMA,
                "model_class": "PnPQ0HypothesisFrozenTrajectorySelector",
                "epoch": epoch, "update": update,
                "model_config": system.trajectory_model.config,
                "selector_state": _selector_checkpoint_state(system),
                "selector_state_sha256": _module_state_sha256(
                    system.selector_model, prefixes=PNP_SELECTOR_TRAINABLE_PREFIXES
                ),
                "optimizer": optimizer.state_dict(),
                "validation": metrics, "selection": selection,
                "provenance": provenance,
            }
            _atomic_checkpoint(checkpoint_path, checkpoint)
            record = {
                "epoch": epoch, "update": update, "learning_rate": lr,
                "train_objective": objective_sum / max(batches, 1),
                "validation": metrics, "selection": selection,
                "checkpoint": checkpoint_name,
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
            history.append(record)
            if best is None or selection < tuple(best["selection"]):
                best = {
                    "epoch": epoch, "update": update, "path": checkpoint_name,
                    "sha256": record["checkpoint_sha256"],
                    "selection": selection, "validation": metrics,
                }
                bad_rounds = 0
            else:
                bad_rounds += 1
            elapsed = time.time() - started
            print(json.dumps({
                "epoch": epoch, "update": update,
                "hard_p95_m": hard["p95_m"], "hard_p99_m": hard["p99_m"],
                "switch_accuracy": metrics["switch_accuracy"],
                "switch_macro_recall": metrics["switch_macro_recall"],
                "elapsed_s": elapsed,
            }, sort_keys=True), flush=True)
            _atomic_json(output / "run_progress.json", {
                "status": "running", "baseline_validation": baseline,
                "best": best, "history": history,
                "initial_hashes": initial_hashes,
                "elapsed_s": elapsed, **provenance,
            })
            if args.patience > 0 and bad_rounds >= args.patience:
                stop = True
                stop_reason = "early_stopping"
        if stop:
            break
    if best is None:
        raise RuntimeError("PnP selector produced no validation checkpoint")
    best_metrics = best["validation"]
    gates = {
        "conditional_probe_bit_exact": bool(best_metrics["conditional_probe_bit_exact"]),
        "minimum_switch_accuracy": float(best_metrics["switch_accuracy"]) >= args.minimum_switch_accuracy,
        "maximum_hard_p95_m": float(best_metrics["hard_position"]["p95_m"]) <= args.maximum_hard_p95_m,
        "frozen_hashes_unchanged": True,
    }
    progress = json.loads((output / "run_progress.json").read_text(encoding="utf-8"))
    progress.update({
        "status": "complete" if all(gates.values()) else "gate_failed",
        "stop_reason": stop_reason, "gates": gates,
        "elapsed_s": time.time() - started,
    })
    _atomic_json(output / "run_manifest.json", progress)
    return output / "run_manifest.json"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", required=True)
    result.add_argument("--mapper-checkpoint", required=True)
    result.add_argument("--s-checkpoint", required=True)
    result.add_argument("--h-checkpoint", required=True)
    result.add_argument("--f-checkpoint", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int, default=20260727)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--epochs", type=int, default=80)
    result.add_argument("--max-updates", type=int, default=5000)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--minimum-learning-rate", type=float, default=3e-6)
    result.add_argument("--warmup-updates", type=int, default=200)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--gradient-clip-norm", type=float, default=1.0)
    result.add_argument("--switch-weight", type=float, default=1.0)
    result.add_argument("--macro-balance-weight", type=float, default=0.5)
    result.add_argument("--switch-focal-gamma", type=float, default=2.0)
    result.add_argument("--distance-cost-weight", type=float, default=1.0)
    result.add_argument("--distance-cost-scale-m", type=float, default=0.3)
    result.add_argument("--distance-cost-cap", type=float, default=2.0)
    result.add_argument("--validation-interval", type=int, default=2)
    result.add_argument("--patience", type=int, default=8)
    result.add_argument("--minimum-switch-accuracy", type=float, default=0.79)
    result.add_argument("--maximum-hard-p95-m", type=float, default=0.40173)
    return result


def main() -> None:
    args = parser().parse_args()
    if (
        min(args.batch_size, args.epochs, args.max_updates, args.validation_interval) < 1
        or min(args.learning_rate, args.minimum_learning_rate, args.gradient_clip_norm) <= 0
        or args.minimum_learning_rate > args.learning_rate
        or min(
            args.weight_decay, args.switch_weight, args.macro_balance_weight,
            args.switch_focal_gamma, args.distance_cost_weight,
        ) < 0
        or not 0 <= args.minimum_switch_accuracy <= 1
        or args.maximum_hard_p95_m <= 0
    ):
        raise ValueError("invalid PnP selector arguments")
    print(train(args))


if __name__ == "__main__":
    main()
