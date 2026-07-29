"""Fine-tune only the final future position over a frozen PnP/V50/V66 stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cyclic_future_foundation import load_frozen_v19
from .final_visible_position_refiner import (
    FinalVisiblePositionRefiner,
    FrozenVisibleSystemFinalRefiner,
    final_visible_position_loss,
)
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
)
from .pnp_q0_hypothesis_adapter import (
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .train_causal_physical_ab import _git_state, _seed, _to_device
from .train_joint_visible_future import _build_cache, _loss_batch, _model_batch
from .train_ordinal_visible_selector import reproducible_cache_contract
from .train_pnp_window_mapper_distillation import (
    _atomic_checkpoint,
    _atomic_json,
    _learning_rate,
)


RUN_SCHEMA = "stage3-final-visible-position-refiner-diagnostic-v1"
V66_SCHEMA = "stage3-ordinal-visible-selector-diagnostic-v1"
MILLIMETRE_THRESHOLDS = (20, 50, 100, 150, 200, 300, 500)


def _position_metrics(error_m: np.ndarray) -> dict[str, Any]:
    values = error_m.astype(np.float64, copy=False)
    return {
        "count": int(values.size),
        "mean_m": float(values.mean()),
        "p50_m": float(np.percentile(values, 50)),
        "p95_m": float(np.percentile(values, 95)),
        "p99_m": float(np.percentile(values, 99)),
        "max_m": float(values.max()),
        "coverage": {
            f"le_{millimetres}mm": float(
                np.mean(values <= millimetres / 1000.0)
            )
            for millimetres in MILLIMETRE_THRESHOLDS
        },
    }


@torch.no_grad()
def evaluate(
    model: FrozenVisibleSystemFinalRefiner,
    loader: DataLoader,
    device: torch.device,
    *,
    collect_queries: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    model.eval()
    refined_errors: list[np.ndarray] = []
    baseline_errors: list[np.ndarray] = []
    residual_norms: list[np.ndarray] = []
    query_times: list[np.ndarray] = []
    switch_correct = 0
    switch_count = 0
    for raw_batch in loader:
        cached = _to_device(raw_batch, device)
        loss_batch = _loss_batch(cached)
        prediction = model(_model_batch(loss_batch))
        query_mask = loss_batch["target_query_mask"].to(torch.bool)
        target_position = (
            loss_batch["current_position_m"][:, None]
            + loss_batch["target_visible_delta_m"]
        )
        refined = torch.linalg.vector_norm(
            prediction["position_m"] - target_position, dim=-1,
        )
        baseline = torch.linalg.vector_norm(
            prediction["unrefined_position_m"] - target_position, dim=-1,
        )
        residual = torch.linalg.vector_norm(
            prediction["position_residual_m"], dim=-1,
        )
        refined_errors.append(refined[query_mask].cpu().numpy())
        baseline_errors.append(baseline[query_mask].cpu().numpy())
        residual_norms.append(residual[query_mask].cpu().numpy())

        row = prediction["selected_candidate_row"]
        selected_step = cached["candidate_step"].gather(1, row)
        switch_correct += int(
            (
                query_mask
                & (selected_step == loss_batch["target_switch_count"])
            ).sum()
        )
        switch_count += int(query_mask.sum())
        if collect_queries:
            tau = loss_batch["tau_s"]
            if tau.ndim == 1:
                tau = tau.unsqueeze(0).expand(query_mask.shape[0], -1)
            query_times.append(tau[query_mask].cpu().numpy())

    refined_m = np.concatenate(refined_errors)
    baseline_m = np.concatenate(baseline_errors)
    residual_m = np.concatenate(residual_norms)
    delta_m = refined_m - baseline_m
    metrics = {
        "final_position": _position_metrics(refined_m),
        "frozen_v66_baseline": _position_metrics(baseline_m),
        "per_query_change": {
            "mean_error_delta_m": float(delta_m.mean()),
            "median_error_delta_m": float(np.median(delta_m)),
            "improved_fraction": float(np.mean(delta_m < 0.0)),
            "unchanged_fraction": float(np.mean(delta_m == 0.0)),
        },
        "position_residual": _position_metrics(residual_m),
        "frozen_switch_accuracy": switch_correct / switch_count,
    }
    queries = None
    if collect_queries:
        queries = {
            "tau_s": np.concatenate(query_times).astype(np.float32),
            "final_error_m": refined_m.astype(np.float32),
            "frozen_v66_error_m": baseline_m.astype(np.float32),
            "position_residual_m": residual_m.astype(np.float32),
        }
    return metrics, queries


def load_frozen_v66(
    trajectory_checkpoint: str | Path,
    selector_checkpoint: str | Path,
    *,
    dataset_manifest_sha256: str,
) -> tuple[OrdinalVisibleFutureModel, dict[str, Any]]:
    trajectory, trajectory_provenance = load_observable_f_checkpoint(
        trajectory_checkpoint,
    )
    path = Path(selector_checkpoint).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != V66_SCHEMA:
        raise ValueError("final refiner requires a V66 ordinal checkpoint")
    provenance = payload.get("provenance", {})
    if provenance.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("V66 checkpoint dataset binding differs")
    selector_config = payload.get("model_config", {}).get("selector", {})
    selector = OrdinalVisibleProgressSelector(
        frozen_context_features=int(selector_config["frozen_context_features"]),
        channels=int(selector_config["channels"]),
        dropout=float(selector_config["dropout"]),
        trained_horizon_s=float(selector_config["trained_horizon_s"]),
        maximum_absolute_step=int(selector_config["maximum_absolute_step"]),
    )
    base = OrdinalVisibleFutureModel(trajectory, selector)
    if base.config != payload.get("model_config"):
        raise ValueError("V66 checkpoint config is not exactly reproducible")
    if (
        state_dict_sha256(base.trajectory.state_dict())
        != payload.get("frozen_trajectory_state_dict_sha256")
    ):
        raise ValueError("V66 checkpoint trajectory binding differs")
    base.selector.load_state_dict(payload["selector"], strict=True)
    selector_hash = state_dict_sha256(base.selector.state_dict())
    if selector_hash != payload.get("selector_state_dict_sha256"):
        raise ValueError("V66 checkpoint selector hash differs")
    base.eval().requires_grad_(False)
    return base, {
        "path": str(path),
        "sha256": sha256_file(path),
        "epoch": int(payload["epoch"]),
        "update": int(payload["update"]),
        "state_dict_sha256": state_dict_sha256(base.state_dict()),
        "trajectory": trajectory_provenance,
        "selector_state_dict_sha256": selector_hash,
        "provenance": provenance,
    }


def _checkpoint_payload(
    model: FrozenVisibleSystemFinalRefiner,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    update: int,
    validation: dict[str, Any] | None,
    provenance: dict[str, Any],
    frozen_base_hash: str,
    generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA,
        "model_class": type(model.refiner).__name__,
        "model_config": model.config,
        "refiner": model.refiner.state_dict(),
        "refiner_state_dict_sha256": state_dict_sha256(
            model.refiner.state_dict(),
        ),
        "frozen_base_state_dict_sha256": frozen_base_hash,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "update": update,
        "validation": validation,
        "provenance": provenance,
        "rng": {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else []
            ),
            "data_loader_generator": generator.get_state(),
        },
    }


def _gate(
    initial: dict[str, Any], final: dict[str, Any], *, frozen_unchanged: bool,
) -> dict[str, Any]:
    baseline = initial["frozen_v66_baseline"]
    refined = final["final_position"]
    checks = {
        "frozen_stack_bit_exact": frozen_unchanged,
        "mean_error_improved": refined["mean_m"] < baseline["mean_m"],
        "median_error_improved": refined["p50_m"] < baseline["p50_m"],
        "p95_regression_le_5mm": refined["p95_m"] <= baseline["p95_m"] + 0.005,
        "p99_regression_le_10mm": refined["p99_m"] <= baseline["p99_m"] + 0.010,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "interpretation": (
            "fixed endpoint acceptance; validation was not used to select a checkpoint"
        ),
    }


def train(args: argparse.Namespace) -> Path:
    if not (
        args.diagnostic_only
        and args.allow_diagnostic_h
        and args.allow_mapper_h_mismatch
    ):
        raise ValueError("final refiner training requires diagnostic opt-ins")
    output = Path(args.output).resolve()
    resume_path = (
        Path(args.resume_checkpoint).resolve()
        if args.resume_checkpoint else None
    )
    if resume_path is None:
        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite final refiner run: {output}"
            )
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
        raise ValueError("final refiner training cannot use an accessed test split")

    train_dataset = ObservableFuturePnPSFDataset(
        dataset_path, "train", motion_class=3,
        sample_limit=args.train_limit, allow_diagnostic=False,
    )
    validation_dataset = ObservableFuturePnPSFDataset(
        dataset_path, "validation", motion_class=3,
        sample_limit=args.validation_limit, allow_diagnostic=False,
    )
    canonicalize_direction_keep_c4(
        train_dataset.tensors, train_dataset.pair_ids,
    )
    canonicalize_direction_keep_c4(
        validation_dataset.tensors, validation_dataset.pair_ids,
    )
    mapper, mapper_provenance = load_frozen_pnp_mapper(
        args.mapper_checkpoint,
    )
    s_model, s_provenance = load_frozen_v19(args.s_checkpoint)
    h_model, h_provenance = load_frozen_hypothesis_adapter(
        args.h_checkpoint, allow_diagnostic=True,
    )
    base, base_provenance = load_frozen_v66(
        args.trajectory_checkpoint, args.selector_checkpoint,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    parent_provenance = base_provenance["provenance"]
    bindings = {
        "mapper": (
            parent_provenance.get("mapper", {}).get("state_dict_sha256"),
            mapper_provenance["state_dict_sha256"],
        ),
        "s": (
            parent_provenance.get("s", {}).get("state_dict_sha256"),
            s_provenance["state_dict_sha256"],
        ),
        "h": (
            parent_provenance.get("h", {}).get("state_dict_sha256"),
            h_provenance["state_dict_sha256"],
        ),
    }
    mismatch = [name for name, values in bindings.items() if values[0] != values[1]]
    if mismatch:
        raise ValueError("final refiner frozen parent mismatch: " + ", ".join(mismatch))
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
        generator=generator, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_cache, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )

    refiner = FinalVisiblePositionRefiner(
        frozen_context_features=4 * base.trajectory.channels + 15,
        channels=args.refiner_channels, dropout=args.refiner_dropout,
        position_scale_m=base.trajectory.position_scale_m,
        trained_horizon_s=base.trajectory.trained_horizon_s,
        maximum_absolute_step=base.trajectory.maximum_absolute_step,
        maximum_residual_m=args.maximum_residual_m,
    )
    model = FrozenVisibleSystemFinalRefiner(base, refiner).to(device)
    frozen_base_hash = state_dict_sha256(model.base.state_dict())
    optimizer = torch.optim.AdamW(
        model.refiner.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    provenance = {
        "diagnostic_only": True,
        "oracle_association": True,
        "deployable_pipeline": False,
        "test_accessed": False,
        "training_stage": "final_future_position_residual",
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "mapper": mapper_provenance,
        "s": s_provenance,
        "h": h_provenance,
        "frozen_v66_parent": base_provenance,
        "frozen_feature_cache": cache_manifest,
        "refiner_definition": model.refiner.config,
        "all_upstream_models_frozen": True,
        "checkpoint_selection": "none_fixed_endpoint_only",
        "physical_id_input": False,
        "motion_class_forward_input": False,
        "future_truth_forward_input": False,
        "git": _git_state(),
        "source_path": str(Path(__file__).resolve()),
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }

    resume_payload: dict[str, Any] | None = None
    if resume_path is not None:
        resume_payload = torch.load(
            resume_path, map_location="cpu", weights_only=False,
        )
        if resume_payload.get("schema_version") != RUN_SCHEMA:
            raise ValueError("final refiner resume checkpoint schema mismatch")
        if resume_payload.get("model_config") != model.config:
            raise ValueError("final refiner resume model configuration mismatch")
        if (
            resume_payload.get("frozen_base_state_dict_sha256")
            != frozen_base_hash
        ):
            raise ValueError("final refiner resume base binding mismatch")
        resume_provenance = resume_payload.get("provenance", {})
        if (
            resume_provenance.get("dataset_manifest_sha256")
            != dataset_manifest_sha256
            or reproducible_cache_contract(
                resume_provenance.get("frozen_feature_cache", {})
            ) != reproducible_cache_contract(cache_manifest)
        ):
            raise ValueError("final refiner resume data binding mismatch")
        model.refiner.load_state_dict(resume_payload["refiner"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"])

    if resume_payload is None:
        initial_validation, _ = evaluate(model, validation_loader, device)
        baseline = initial_validation["frozen_v66_baseline"]
        initial_refined = initial_validation["final_position"]
        if baseline != initial_refined:
            raise RuntimeError("zero refiner is not exactly the frozen V66 baseline")
        history: list[dict[str, Any]] = [{
            "epoch": 0,
            "update": 0,
            "phase": "zero_residual_frozen_v66_baseline",
            "validation": initial_validation,
        }]
        update = 0
        first_epoch = 1
    else:
        progress_path = output / "run_progress.json"
        if not progress_path.is_file():
            raise ValueError("final refiner resume requires run_progress.json")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        history = list(progress["history"])
        initial_validation = history[0]["validation"]
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
    learning_rate = args.learning_rate
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
            objective, components = final_visible_position_loss(
                prediction, loss_batch,
                huber_beta_m=args.huber_beta_m,
                euclidean_weight=args.euclidean_weight,
                macro_balance_weight=args.macro_balance_weight,
            )
            objective.backward()
            gradients = [
                parameter.grad for parameter in model.refiner.parameters()
                if parameter.grad is not None
            ]
            if not gradients or not all(
                bool(torch.isfinite(value).all()) for value in gradients
            ) or not any(bool(torch.any(value != 0)) for value in gradients):
                raise RuntimeError("final refiner received no finite nonzero gradient")
            torch.nn.utils.clip_grad_norm_(
                model.refiner.parameters(), args.gradient_clip_norm,
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
            last_validation, _ = evaluate(model, validation_loader, device)
            record = {
                "epoch": epoch,
                "update": update,
                "learning_rate": learning_rate,
                "train": {
                    name: value / max(batches, 1)
                    for name, value in sums.items()
                },
                "validation": last_validation,
                "elapsed_s": time.time() - started,
            }
            history.append(record)
            checkpoint = output / f"epoch-{epoch:04d}-update-{update:06d}.pt"
            _atomic_checkpoint(checkpoint, _checkpoint_payload(
                model, optimizer, epoch=epoch, update=update,
                validation=last_validation, provenance=provenance,
                frozen_base_hash=frozen_base_hash, generator=generator,
            ))
            final_metrics = last_validation["final_position"]
            print(json.dumps({
                "epoch": epoch,
                "update": update,
                "mean_mm": 1000.0 * final_metrics["mean_m"],
                "p50_mm": 1000.0 * final_metrics["p50_m"],
                "p95_mm": 1000.0 * final_metrics["p95_m"],
                "p99_mm": 1000.0 * final_metrics["p99_m"],
                "elapsed_s": record["elapsed_s"],
            }), flush=True)

        recovery = output / f"recovery-epoch-{epoch:04d}-update-{update:06d}.pt"
        _atomic_checkpoint(recovery, _checkpoint_payload(
            model, optimizer, epoch=epoch, update=update,
            validation=last_validation if validate_now else None,
            provenance=provenance, frozen_base_hash=frozen_base_hash,
            generator=generator,
        ))
        _atomic_json(output / "run_progress.json", {
            "schema_version": RUN_SCHEMA,
            "status": "complete" if stop else "running",
            "epoch": epoch,
            "update": update,
            "latest_recovery_checkpoint": recovery.name,
            "history": history,
            "provenance": provenance,
        })

    final_validation, queries = evaluate(
        model, validation_loader, device, collect_queries=True,
    )
    assert queries is not None
    np.savez_compressed(output / "validation_queries.npz", **queries)
    final_base_hash = state_dict_sha256(model.base.state_dict())
    frozen_unchanged = final_base_hash == frozen_base_hash
    if not frozen_unchanged:
        raise RuntimeError("final position training changed the frozen stack")
    gate = _gate(
        initial_validation, final_validation,
        frozen_unchanged=frozen_unchanged,
    )
    final_checkpoint = output / f"final-epoch-{history[-1]['epoch']:04d}-update-{update:06d}.pt"
    _atomic_checkpoint(final_checkpoint, _checkpoint_payload(
        model, optimizer, epoch=int(history[-1]["epoch"]), update=update,
        validation=final_validation, provenance=provenance,
        frozen_base_hash=frozen_base_hash, generator=generator,
    ))
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
        "final_validation": final_validation,
        "gate": gate,
        "gate_passed": gate["passed"],
        "provenance": provenance,
        "final_checkpoint": str(final_checkpoint),
        "validation_queries": str(output / "validation_queries.npz"),
        "frozen_base_initial_state_dict_sha256": frozen_base_hash,
        "frozen_base_final_state_dict_sha256": final_base_hash,
        "frozen_base_verified_unchanged": frozen_unchanged,
    }
    _atomic_json(output / "run_manifest.json", manifest_payload)
    _atomic_json(output / "run_progress.json", {
        "schema_version": RUN_SCHEMA,
        "status": "complete",
        "epoch": history[-1]["epoch"],
        "update": update,
        "latest_recovery_checkpoint": (
            f"recovery-epoch-{history[-1]['epoch']:04d}-update-{update:06d}.pt"
        ),
        "history": history,
        "provenance": provenance,
    })
    return output / "run_manifest.json"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", required=True)
    result.add_argument("--mapper-checkpoint", required=True)
    result.add_argument("--s-checkpoint", required=True)
    result.add_argument("--h-checkpoint", required=True)
    result.add_argument("--trajectory-checkpoint", required=True)
    result.add_argument("--selector-checkpoint", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--resume-checkpoint", default="")
    result.add_argument("--diagnostic-only", action="store_true")
    result.add_argument("--allow-diagnostic-h", action="store_true")
    result.add_argument("--allow-mapper-h-mismatch", action="store_true")
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int, default=20260729)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--cache-batch-size", type=int, default=256)
    result.add_argument("--epochs", type=int, default=80)
    result.add_argument("--max-updates", type=int, default=1000)
    result.add_argument("--refiner-channels", type=int, default=128)
    result.add_argument("--refiner-dropout", type=float, default=0.05)
    result.add_argument("--maximum-residual-m", type=float, default=0.75)
    result.add_argument("--learning-rate", type=float, default=2e-4)
    result.add_argument("--minimum-learning-rate", type=float, default=2e-6)
    result.add_argument("--learning-rate-warmup-updates", type=int, default=100)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--gradient-clip-norm", type=float, default=1.0)
    result.add_argument("--huber-beta-m", type=float, default=0.02)
    result.add_argument("--euclidean-weight", type=float, default=0.1)
    result.add_argument("--macro-balance-weight", type=float, default=0.1)
    result.add_argument("--validation-interval", type=int, default=5)
    result.add_argument("--train-limit", type=int, default=0)
    result.add_argument("--validation-limit", type=int, default=0)
    return result


def main() -> None:
    args = parser().parse_args()
    if min(
        args.batch_size, args.cache_batch_size, args.epochs,
        args.max_updates, args.refiner_channels, args.maximum_residual_m,
        args.learning_rate, args.gradient_clip_norm,
        args.validation_interval, args.huber_beta_m,
    ) <= 0:
        raise ValueError("final refiner training configuration must be positive")
    print(train(args))


if __name__ == "__main__":
    main()
