"""Train one anonymous monotone switch-progress selector over frozen PnP V50."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .cyclic_future_foundation import load_frozen_v19
from .observable_future_pnp_ab import (
    ObservableFuturePnPSFDataset,
    canonicalize_direction_keep_c4,
    load_observable_f_checkpoint,
    sha256_file,
    state_dict_sha256,
)
from .ordinal_visible_selector import (
    OrdinalVisibleFutureModel,
    OrdinalVisibleProgressSelector,
    ordinal_visible_selector_loss,
)
from .pnp_q0_hypothesis_adapter import (
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .train_causal_physical_ab import _git_state, _seed, _to_device
from .train_joint_visible_future import (
    _build_cache,
    _loss_batch,
    _model_batch,
)
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


RUN_SCHEMA = "stage3-ordinal-visible-selector-diagnostic-v1"
BASELINE_V52 = {
    "conditional_p95_m": 0.21585246920585632,
    "hard_p95_m": 0.36353662610054016,
    "hard_p99_m": 0.5603015303611752,
    "switch_accuracy": 0.8325464070888041,
    "minimum_step_switch_recall": 0.6902102496714849,
}
STRONG_GATE = {
    "conditional_p95_tolerance_m": 1e-5,
    "hard_p95_max_m": 0.350,
    "hard_p99_max_m": 0.550,
    "switch_accuracy_min": 0.840,
    "minimum_step_switch_recall_min": 0.720,
    "predicted_nonmonotone_fraction_max": 0.010,
}


def reproducible_cache_contract(cache_manifest: dict[str, Any]) -> dict[str, Any]:
    """Strip wall-clock evidence while retaining every cache identity field."""
    result: dict[str, Any] = {}
    for name, value in cache_manifest.items():
        if name in ("train", "validation") and isinstance(value, dict):
            result[name] = {
                key: item for key, item in value.items() if key != "elapsed_s"
            }
        else:
            result[name] = value
    return result


def stratified_capacity_subset(
    dataset: ObservableFuturePnPSFDataset,
    limit: int,
    *,
    seed: int,
) -> tuple[Dataset, dict[str, Any]]:
    """Balance direction, maximum step and session for a capacity diagnostic."""
    if limit <= 0 or limit > len(dataset):
        raise ValueError("capacity subset limit must be within the dataset")
    target = dataset.tensors["target_switch_count"]
    query_mask = dataset.tensors["target_query_mask"].to(torch.bool)
    positive = torch.any(query_mask & (target > 0), dim=1)
    negative = torch.any(query_mask & (target < 0), dim=1)
    if bool(torch.any(positive & negative)):
        raise ValueError("capacity source contains a reversing target sequence")
    direction = torch.where(
        negative, torch.full_like(negative, -1, dtype=torch.long),
        torch.where(
            positive, torch.ones_like(positive, dtype=torch.long),
            torch.zeros_like(positive, dtype=torch.long),
        ),
    )
    magnitude = torch.where(
        query_mask, target.abs(), torch.zeros_like(target),
    ).amax(dim=1)
    rng = np.random.default_rng(seed)
    grouped: dict[tuple[int, int], dict[str, list[int]]] = {}
    for index, session in enumerate(dataset.session_ids):
        stratum = (int(direction[index]), int(magnitude[index]))
        grouped.setdefault(stratum, {}).setdefault(session, []).append(index)
    queues: dict[tuple[int, int], list[int]] = {}
    for stratum, session_groups in grouped.items():
        for values in session_groups.values():
            rng.shuffle(values)
        sessions = sorted(session_groups)
        rng.shuffle(sessions)
        queue: list[int] = []
        while any(session_groups[value] for value in sessions):
            for session in sessions:
                if session_groups[session]:
                    queue.append(session_groups[session].pop())
        queues[stratum] = queue
    strata = sorted(queues)
    selected: list[int] = []
    while len(selected) < limit:
        progressed = False
        for stratum in strata:
            if queues[stratum] and len(selected) < limit:
                selected.append(queues[stratum].pop())
                progressed = True
        if not progressed:
            break
    if len(selected) != limit:
        raise RuntimeError("capacity sampler did not reach its fixed size")
    selected.sort()
    support: dict[str, int] = {}
    session_support: set[str] = set()
    for index in selected:
        key = f"direction_{int(direction[index]):+d}_max_step_{int(magnitude[index])}"
        support[key] = support.get(key, 0) + 1
        session_support.add(dataset.session_ids[index])
    digest = hashlib.sha256(
        np.asarray(selected, dtype=np.int64).tobytes()
    ).hexdigest()
    return Subset(dataset, selected), {
        "strategy": "round_robin_direction_max_step_session_v1",
        "seed": seed,
        "sample_count": len(selected),
        "selected_indices_sha256": digest,
        "stratum_support": support,
        "session_count": len(session_support),
    }


def _loss_prediction(
    prediction: dict[str, torch.Tensor],
    truth_current_position_m: torch.Tensor,
) -> dict[str, torch.Tensor]:
    result = dict(prediction)
    result["conditional_delta_m"] = (
        prediction["conditional_position_m"]
        - truth_current_position_m[:, None, None]
    )
    return result


@torch.no_grad()
def evaluate(
    model: OrdinalVisibleFutureModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    storage = _new_eval_storage()
    conditional_hasher = hashlib.sha256()
    one_step_count = 0
    one_step_correct = 0
    confidence: list[np.ndarray] = []
    hard_errors: list[np.ndarray] = []
    nonmonotone = 0
    sequence_count = 0
    for raw_batch in loader:
        cached = _to_device(raw_batch, device)
        loss_batch = _loss_batch(cached)
        prediction = model(_model_batch(loss_batch))
        _eval_add(
            storage, prediction, loss_batch,
            candidate_supported=cached["candidate_supported"],
        )
        conditional_hasher.update(
            prediction["conditional_position_m"].detach().cpu().contiguous().numpy().tobytes()
        )
        row = prediction["selected_candidate_row"]
        selected_step = cached["candidate_step"].gather(1, row)
        query_mask = cached["target_query_mask"].to(torch.bool)
        one_step = query_mask & (cached["target_switch_count"].abs() == 1)
        one_step_count += int(one_step.sum())
        one_step_correct += int(
            (one_step & (selected_step == cached["target_switch_count"])).sum()
        )
        selected_probability = prediction["switch_probability"].gather(
            2, row.unsqueeze(-1),
        ).squeeze(-1)
        confidence.append(selected_probability[query_mask].cpu().numpy())
        truth_delta = loss_batch["target_visible_delta_m"]
        hard_delta = (
            prediction["position_m"]
            - loss_batch["current_position_m"][:, None]
        )
        hard_errors.append(torch.linalg.vector_norm(
            hard_delta - truth_delta, dim=-1,
        )[query_mask].cpu().numpy())
        tau = cached["tau_s"]
        if tau.ndim == 1:
            tau = tau.unsqueeze(0).expand(selected_step.shape[0], -1)
        for sample in range(selected_step.shape[0]):
            valid = query_mask[sample]
            if int(valid.sum()) < 2:
                continue
            order = tau[sample, valid].argsort()
            sequence = selected_step[sample, valid][order]
            difference = sequence[1:] - sequence[:-1]
            if bool(torch.any(difference > 0) and torch.any(difference < 0)):
                nonmonotone += 1
            sequence_count += 1
    metrics = _finish_eval(storage)
    metrics["minimum_step_switch_count"] = one_step_count
    metrics["minimum_step_switch_recall"] = (
        one_step_correct / one_step_count if one_step_count else None
    )
    merged_confidence = np.concatenate(confidence).astype(np.float64, copy=False)
    metrics["selected_probability"] = {
        "mean": float(merged_confidence.mean()),
        "p50": float(np.percentile(merged_confidence, 50)),
        "p90": float(np.percentile(merged_confidence, 90)),
    }
    merged_error = np.concatenate(hard_errors).astype(np.float64, copy=False)
    metrics["hard_coverage"] = {
        f"le_{millimetres}mm": float(np.mean(merged_error <= millimetres / 1000.0))
        for millimetres in (20, 50, 100, 150, 200, 300)
    }
    metrics["predicted_sequence_monotonicity"] = {
        "sequence_count": sequence_count,
        "nonmonotone_count": nonmonotone,
        "nonmonotone_fraction": nonmonotone / sequence_count if sequence_count else None,
    }
    metrics["conditional_output_sha256"] = conditional_hasher.hexdigest()
    return metrics


def _checkpoint_payload(
    model: OrdinalVisibleFutureModel,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    update: int,
    validation: dict[str, Any] | None,
    provenance: dict[str, Any],
    trajectory_hash: str,
    generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA,
        "model_class": type(model).__name__,
        "model_config": model.config,
        "selector": model.selector.state_dict(),
        "selector_state_dict_sha256": state_dict_sha256(model.selector.state_dict()),
        "frozen_trajectory_state_dict_sha256": trajectory_hash,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "update": update,
        "validation": validation,
        "provenance": provenance,
        "rng": {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "data_loader_generator": generator.get_state(),
        },
    }


def _gate(
    initial: dict[str, Any], final: dict[str, Any], *, capacity_gate: bool,
) -> dict[str, Any]:
    conditional_bit_exact = (
        final["conditional_output_sha256"] == initial["conditional_output_sha256"]
    )
    if capacity_gate:
        checks = {
            "conditional_bit_exact": conditional_bit_exact,
            "train_switch_accuracy_ge_95pct": final["switch_accuracy"] >= 0.95,
            "train_one_step_recall_ge_90pct": (
                final["minimum_step_switch_recall"] is not None
                and final["minimum_step_switch_recall"] >= 0.90
            ),
        }
    else:
        checks = {
            "conditional_bit_exact": conditional_bit_exact,
            "conditional_p95_preserved": abs(
                final["conditional_position"]["p95_m"]
                - initial["conditional_position"]["p95_m"]
            ) <= STRONG_GATE["conditional_p95_tolerance_m"],
            "hard_p95_le_350mm": (
                final["hard_routed_position"]["p95_m"]
                <= STRONG_GATE["hard_p95_max_m"]
            ),
            "hard_p99_le_550mm": (
                final["hard_routed_position"]["p99_m"]
                <= STRONG_GATE["hard_p99_max_m"]
            ),
            "switch_accuracy_ge_84pct": (
                final["switch_accuracy"] >= STRONG_GATE["switch_accuracy_min"]
            ),
            "one_step_recall_ge_72pct": (
                final["minimum_step_switch_recall"] is not None
                and final["minimum_step_switch_recall"]
                >= STRONG_GATE["minimum_step_switch_recall_min"]
            ),
            "nonmonotone_fraction_lt_1pct": (
                final["predicted_sequence_monotonicity"]["nonmonotone_fraction"]
                < STRONG_GATE["predicted_nonmonotone_fraction_max"]
            ),
        }
    return {
        "mode": "capacity" if capacity_gate else "full_validation",
        "checks": checks,
        "passed": all(checks.values()),
        "baseline_v52": BASELINE_V52,
        "strong_thresholds": STRONG_GATE,
    }


def train(args: argparse.Namespace) -> Path:
    if not (
        args.diagnostic_only
        and args.allow_diagnostic_h
        and args.allow_mapper_h_mismatch
    ):
        raise ValueError("ordinal training requires explicit diagnostic opt-ins")
    output = Path(args.output).resolve()
    resume_path = Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None
    if resume_path is None:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite ordinal run: {output}")
        output.mkdir(parents=True)
    elif not output.is_dir() or resume_path.parent != output:
        raise ValueError("resume checkpoint must belong to the existing output")

    _seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    dataset_path = Path(args.dataset).resolve()
    manifest_path = dataset_path / "dataset_manifest.json"
    dataset_manifest_sha256 = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("test_accessed", True)):
        raise ValueError("ordinal training cannot use an accessed test split")

    train_dataset_full = ObservableFuturePnPSFDataset(
        dataset_path, "train", motion_class=3,
        sample_limit=0 if args.capacity_gate else args.train_limit,
        allow_diagnostic=False,
    )
    validation_dataset_full = ObservableFuturePnPSFDataset(
        dataset_path, "validation", motion_class=3,
        sample_limit=0 if args.capacity_gate else args.validation_limit,
        allow_diagnostic=False,
    )
    canonicalize_direction_keep_c4(
        train_dataset_full.tensors, train_dataset_full.pair_ids,
    )
    canonicalize_direction_keep_c4(
        validation_dataset_full.tensors, validation_dataset_full.pair_ids,
    )
    capacity_selection: dict[str, Any] | None = None
    if args.capacity_gate:
        train_dataset, train_selection = stratified_capacity_subset(
            train_dataset_full, args.train_limit, seed=args.seed,
        )
        validation_dataset, validation_selection = stratified_capacity_subset(
            validation_dataset_full, args.validation_limit, seed=args.seed,
        )
        capacity_selection = {
            "train": train_selection, "validation": validation_selection,
        }
    else:
        train_dataset = train_dataset_full
        validation_dataset = validation_dataset_full
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
        raise ValueError("ordinal training requires a trajectory-stage parent")
    if parent_provenance.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("ordinal trajectory parent dataset differs")
    bindings = {
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
    mismatch = [name for name, values in bindings.items() if values[0] != values[1]]
    if mismatch:
        raise ValueError("ordinal frozen parent mismatch: " + ", ".join(mismatch))
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
        "capacity_selection": capacity_selection,
        "train": train_cache_manifest,
        "validation": validation_cache_manifest,
    }
    _atomic_json(output / "frozen_feature_cache_manifest.json", cache_manifest)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_cache, batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=0, pin_memory=device.type == "cuda",
    )
    train_eval_loader = DataLoader(
        train_cache, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_cache, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    selector = OrdinalVisibleProgressSelector(
        frozen_context_features=4 * trajectory.channels + 15,
        channels=args.selector_channels, dropout=args.selector_dropout,
        trained_horizon_s=trajectory.trained_horizon_s,
        maximum_absolute_step=trajectory.maximum_absolute_step,
    )
    model = OrdinalVisibleFutureModel(trajectory, selector).to(device)
    trajectory_hash = state_dict_sha256(model.trajectory.state_dict())
    optimizer = torch.optim.AdamW(
        model.selector.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
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
        "trajectory_parent": {**trajectory_provenance, "provenance": parent_provenance},
        "frozen_feature_cache": cache_manifest,
        "selector_definition": model.selector.config,
        "trajectory_and_upstream_frozen": True,
        "physical_id_input": False,
        "motion_class_forward_input": False,
        "fixed_switch_time": False,
        "git": _git_state(),
        "source_path": str(Path(__file__).resolve()),
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    resume_payload: dict[str, Any] | None = None
    if resume_path is not None:
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_payload.get("schema_version") != RUN_SCHEMA:
            raise ValueError("ordinal resume checkpoint schema mismatch")
        if resume_payload.get("model_config") != model.config:
            raise ValueError("ordinal resume model configuration mismatch")
        if resume_payload.get("frozen_trajectory_state_dict_sha256") != trajectory_hash:
            raise ValueError("ordinal resume trajectory binding mismatch")
        resume_provenance = resume_payload.get("provenance", {})
        if (
            resume_provenance.get("dataset_manifest_sha256") != dataset_manifest_sha256
            or reproducible_cache_contract(
                resume_provenance.get("frozen_feature_cache", {})
            ) != reproducible_cache_contract(cache_manifest)
        ):
            raise ValueError("ordinal resume data binding mismatch")
        model.selector.load_state_dict(resume_payload["selector"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"])

    if resume_payload is None:
        initial_validation = evaluate(model, validation_loader, device)
        initial_gate_metrics = (
            evaluate(model, train_eval_loader, device)
            if args.capacity_gate else initial_validation
        )
        history: list[dict[str, Any]] = [{
            "epoch": 0, "update": 0, "phase": "untrained-ordinal-baseline",
            "validation": initial_validation,
        }]
        update = 0
        first_epoch = 1
    else:
        progress_path = output / "run_progress.json"
        if not progress_path.is_file():
            raise ValueError("ordinal resume requires run_progress.json")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        history = list(progress["history"])
        initial_validation = history[0]["validation"]
        initial_gate_metrics = progress["initial_gate_metrics"]
        update = int(resume_payload["update"])
        first_epoch = int(resume_payload["epoch"]) + 1
        rng = resume_payload["rng"]
        torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and rng["torch_cuda"]:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        generator.set_state(rng["data_loader_generator"])

    total_updates = min(args.max_updates, args.epochs * len(train_loader))
    started = time.time()
    stop = update >= total_updates
    last_validation = history[-1].get("validation")
    for epoch in range(first_epoch, args.epochs + 1):
        if stop:
            break
        model.train()
        sums: dict[str, float] = {}
        batches = 0
        for raw_batch in train_loader:
            update += 1
            learning_rate = _learning_rate(
                args.learning_rate, update, total_updates,
                args.learning_rate_warmup_updates,
                args.minimum_learning_rate,
            )
            optimizer.param_groups[0]["lr"] = learning_rate
            cached = _to_device(raw_batch, device)
            loss_batch = _loss_batch(cached)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(_model_batch(loss_batch))
            objective, components = ordinal_visible_selector_loss(
                _loss_prediction(
                    prediction, loss_batch["current_position_m"],
                ),
                loss_batch,
                switch_weight=args.switch_weight,
                macro_balance_weight=args.macro_balance_weight,
                switch_focal_gamma=args.switch_focal_gamma,
                distance_cost_weight=args.distance_cost_weight,
                distance_cost_scale_m=args.distance_cost_scale_m,
                distance_cost_cap=args.distance_cost_cap,
                direction_weight=args.direction_weight,
            )
            objective.backward()
            finite_gradient = [
                parameter.grad for parameter in model.selector.parameters()
                if parameter.grad is not None
            ]
            if not finite_gradient or not all(
                bool(torch.isfinite(value).all()) for value in finite_gradient
            ) or not any(bool(torch.any(value != 0)) for value in finite_gradient):
                raise RuntimeError("ordinal selector received no finite nonzero gradient")
            torch.nn.utils.clip_grad_norm_(
                model.selector.parameters(), args.gradient_clip_norm,
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
            last_validation = evaluate(model, validation_loader, device)
            record = {
                "epoch": epoch,
                "update": update,
                "learning_rate": learning_rate,
                "train": {
                    name: value / max(batches, 1) for name, value in sums.items()
                },
                "validation": last_validation,
                "elapsed_s": time.time() - started,
            }
            history.append(record)
            checkpoint_path = output / f"epoch-{epoch:04d}-update-{update:06d}.pt"
            _atomic_checkpoint(checkpoint_path, _checkpoint_payload(
                model, optimizer, epoch=epoch, update=update,
                validation=last_validation, provenance=provenance,
                trajectory_hash=trajectory_hash, generator=generator,
            ))
            print(json.dumps({
                "epoch": epoch,
                "update": update,
                "switch_accuracy": last_validation["switch_accuracy"],
                "minimum_step_recall": last_validation["minimum_step_switch_recall"],
                "conditional_p95_m": last_validation["conditional_position"]["p95_m"],
                "hard_p95_m": last_validation["hard_routed_position"]["p95_m"],
                "nonmonotone_fraction": last_validation[
                    "predicted_sequence_monotonicity"
                ]["nonmonotone_fraction"],
                "elapsed_s": record["elapsed_s"],
            }), flush=True)

        recovery = output / f"recovery-epoch-{epoch:04d}-update-{update:06d}.pt"
        _atomic_checkpoint(recovery, _checkpoint_payload(
            model, optimizer, epoch=epoch, update=update,
            validation=last_validation if validate_now else None,
            provenance=provenance, trajectory_hash=trajectory_hash,
            generator=generator,
        ))
        _atomic_json(output / "run_progress.json", {
            "schema_version": RUN_SCHEMA,
            "status": "complete" if stop else "running",
            "epoch": epoch,
            "update": update,
            "latest_recovery_checkpoint": recovery.name,
            "initial_gate_metrics": initial_gate_metrics,
            "history": history,
            "provenance": provenance,
        })

    if last_validation is None:
        last_validation = evaluate(model, validation_loader, device)
    final_train = evaluate(model, train_eval_loader, device)
    final_trajectory_hash = state_dict_sha256(model.trajectory.state_dict())
    if final_trajectory_hash != trajectory_hash:
        raise RuntimeError("ordinal training changed the frozen trajectory")
    gate_source = final_train if args.capacity_gate else last_validation
    gate = _gate(initial_gate_metrics, gate_source, capacity_gate=args.capacity_gate)
    manifest_payload = {
        "schema_version": RUN_SCHEMA,
        "status": "complete",
        "stop_reason": "fixed_max_updates" if stop else "epoch_limit",
        "epoch": history[-1]["epoch"],
        "update": update,
        "elapsed_s": time.time() - started,
        "training_arguments": vars(args),
        "history": history,
        "initial_validation": initial_validation,
        "initial_gate_metrics": initial_gate_metrics,
        "final_validation": last_validation,
        "final_train": final_train,
        "gate": gate,
        "gate_passed": gate["passed"],
        "provenance": provenance,
        "frozen_trajectory_initial_state_dict_sha256": trajectory_hash,
        "frozen_trajectory_final_state_dict_sha256": final_trajectory_hash,
        "frozen_trajectory_verified_unchanged": True,
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
    result.add_argument("--capacity-gate", action="store_true")
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int, default=20260729)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--cache-batch-size", type=int, default=256)
    result.add_argument("--epochs", type=int, default=80)
    result.add_argument("--max-updates", type=int, default=3000)
    result.add_argument("--selector-channels", type=int, default=128)
    result.add_argument("--selector-dropout", type=float, default=0.05)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    result.add_argument("--learning-rate-warmup-updates", type=int, default=200)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--gradient-clip-norm", type=float, default=1.0)
    result.add_argument("--switch-weight", type=float, default=1.0)
    result.add_argument("--macro-balance-weight", type=float, default=0.25)
    result.add_argument("--switch-focal-gamma", type=float, default=2.0)
    result.add_argument("--distance-cost-weight", type=float, default=1.0)
    result.add_argument("--distance-cost-scale-m", type=float, default=0.3)
    result.add_argument("--distance-cost-cap", type=float, default=2.0)
    result.add_argument("--direction-weight", type=float, default=0.1)
    result.add_argument("--validation-interval", type=int, default=5)
    result.add_argument("--train-limit", type=int, default=0)
    result.add_argument("--validation-limit", type=int, default=0)
    return result


def main() -> None:
    args = parser().parse_args()
    if min(
        args.batch_size, args.cache_batch_size, args.epochs,
        args.max_updates, args.selector_channels,
        args.learning_rate,
        args.gradient_clip_norm, args.validation_interval,
    ) <= 0:
        raise ValueError("ordinal training configuration must be positive")
    print(train(args))


if __name__ == "__main__":
    main()
