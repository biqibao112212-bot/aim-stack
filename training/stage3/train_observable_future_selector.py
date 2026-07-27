"""Train anonymous armor selection while keeping F trajectories bit-exact frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .observable_future_dataset import ObservableFutureDataset
from .observable_future_loss import observable_future_batch_metrics
from .observable_future_model import AnonymousCandidateFutureExpert, DYNAMIC_EXPERTS
from .observable_future_selector_loss import observable_future_selector_loss
from .train_causal_physical_ab import _git_state, _seed, _to_device, _write_json


SELECTOR_PARAMETER_PREFIXES = ("switch_candidate_head.", "switch_logit.")
SOURCE_FILES = (
    "observable_future_dataset.py",
    "build_observable_future_dataset.py",
    "observable_future_model.py",
    "observable_future_loss.py",
    "observable_future_selector_loss.py",
    "audit_observable_future_dataset.py",
    "train_observable_future_selector.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_state_sha256(
    model: AnonymousCandidateFutureExpert, *, selector: bool
) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        is_selector = name.startswith(SELECTOR_PARAMETER_PREFIXES)
        if is_selector != selector:
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def freeze_for_selector_only(
    model: AnonymousCandidateFutureExpert,
) -> tuple[list[str], list[str]]:
    """Freeze every trajectory-affecting parameter and return both name sets."""
    trainable: list[str] = []
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        selected = name.startswith(SELECTOR_PARAMETER_PREFIXES)
        parameter.requires_grad_(selected)
        (trainable if selected else frozen).append(name)
    if not trainable or not frozen:
        raise ValueError("selector-only freeze did not partition model parameters")
    return trainable, frozen


def set_selector_train_mode(model: AnonymousCandidateFutureExpert) -> None:
    """Keep frozen/dropout trajectory features deterministic during training."""
    model.eval()
    model.switch_candidate_head.train()
    model.switch_logit.train()


def _model_forward(
    model: AnonymousCandidateFutureExpert,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return model(
        batch["history_position_rel_m"], batch["history_time_s"],
        batch["history_dt_s"], batch["history_switch_step"],
        batch["history_mask"], batch["current_position_m"],
        batch["candidate_relation_m"], batch["candidate_step"],
        batch["candidate_mask"], batch["candidate_confidence"],
        batch["tau_s"],
    )


def _selector_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return observable_future_selector_loss(
        prediction,
        batch["candidate_step"], batch["candidate_mask"], batch["tau_s"],
        batch["target_switch_count"], batch["target_visible_delta_m"],
        batch["target_query_mask"],
        switch_weight=args.switch_weight,
        macro_balance_weight=args.macro_balance_weight,
        switch_focal_gamma=args.switch_focal_gamma,
        distance_cost_weight=args.distance_cost_weight,
        distance_cost_scale_m=args.distance_cost_scale_m,
        distance_cost_cap=args.distance_cost_cap,
    )


def _percentiles(values: list[np.ndarray]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0, "mean_m": None, "p50_m": None, "p95_m": None,
            "p99_m": None, "max_m": None,
        }
    merged = np.concatenate(values).astype(np.float64, copy=False)
    return {
        "count": int(merged.size),
        "mean_m": float(merged.mean()),
        "p50_m": float(np.quantile(merged, 0.50)),
        "p95_m": float(np.quantile(merged, 0.95)),
        "p99_m": float(np.quantile(merged, 0.99)),
        "max_m": float(merged.max(initial=0.0)),
    }


def _scheduled_learning_rate(
    update: int, *, base_learning_rate: float, minimum_learning_rate: float,
    warmup_updates: int, total_updates: int,
) -> float:
    if total_updates <= 0:
        return float(base_learning_rate)
    if warmup_updates > 0 and update <= warmup_updates:
        return float(max(
            minimum_learning_rate,
            base_learning_rate * float(update) / float(warmup_updates),
        ))
    decay_updates = max(total_updates - warmup_updates, 1)
    progress = min(max(update - warmup_updates, 0) / decay_updates, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(
        minimum_learning_rate
        + (base_learning_rate - minimum_learning_rate) * cosine
    )


def _instantiate_from_checkpoint(
    checkpoint: dict[str, object], expert: str,
) -> AnonymousCandidateFutureExpert:
    if checkpoint.get("model_class") != "AnonymousCandidateFutureExpert":
        raise ValueError("selector foundation model class mismatch")
    config = checkpoint.get("model_config")
    if not isinstance(config, dict) or config.get("family") != "anonymous-observable-future-expert-v9":
        raise ValueError("selector foundation must be observable F v9")
    if config.get("expert") != expert:
        raise ValueError("selector foundation expert mismatch")
    model = AnonymousCandidateFutureExpert(
        expert,
        channels=int(config["channels"]),
        dropout=float(config["dropout"]),
        position_scale_m=float(config["position_scale_m"]),
        history_scale_s=float(config["history_scale_s"]),
        trained_horizon_s=float(config["trained_horizon_s"]),
        maximum_absolute_step=int(config["maximum_absolute_step"]),
        trajectory_rank=int(config["trajectory_rank"]),
    )
    if model.config != config:
        raise ValueError("selector foundation config does not round-trip")
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("selector foundation has no model state")
    model.load_state_dict(state, strict=True)
    return model


def _validate_foundation_provenance(
    checkpoint: dict[str, object], dataset_manifest_sha256: str
) -> None:
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("selector foundation has no provenance")
    if provenance.get("test_accessed") is not False:
        raise ValueError("selector foundation did not keep test sealed")
    if provenance.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("selector foundation dataset manifest mismatch")


@torch.no_grad()
def evaluate(
    model: AnonymousCandidateFutureExpert,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    model.eval()
    objectives: list[float] = []
    switch_losses: list[float] = []
    distance_costs: list[float] = []
    conditional_errors: list[np.ndarray] = []
    hard_errors: list[np.ndarray] = []
    correct_hard_errors: list[np.ndarray] = []
    wrong_hard_errors: list[np.ndarray] = []
    eligible = correct = switched = switched_correct = 0
    predicted_switched = false_switched = 0
    step_total: dict[int, int] = {}
    step_correct: dict[int, int] = {}
    confusion: dict[int, dict[int, int]] = {}
    for raw in loader:
        batch = _to_device(raw, device)
        prediction = _model_forward(model, batch)
        objective, parts = _selector_loss(prediction, batch, args)
        objectives.append(float(objective.cpu()))
        switch_losses.append(float(parts["switch"].cpu()))
        distance_costs.append(float(parts["distance_cost"].cpu()))
        tau = batch["tau_s"]
        if tau.ndim == 1:
            tau = tau.unsqueeze(0).expand_as(batch["target_switch_count"])
        learned_mask = batch["target_query_mask"].to(torch.bool) & (tau > 0)
        metric = observable_future_batch_metrics(
            prediction, batch["candidate_step"], batch["candidate_mask"],
            batch["target_switch_count"], batch["target_visible_delta_m"],
            learned_mask,
        )
        eligible += int(metric["eligible_count"])
        correct += int(metric["correct_count"])
        switched += int(metric["switched_count"])
        switched_correct += int(metric["switched_correct_count"])
        conditional_errors.append(metric["conditional_error_m"].cpu().numpy())
        hard = metric["hard_error_m"]
        hard_errors.append(hard.cpu().numpy())
        selected = prediction["selected_switch_step"]
        correct_mask = learned_mask & (selected == batch["target_switch_count"])
        flat_correct = correct_mask[learned_mask]
        correct_hard_errors.append(hard[flat_correct].cpu().numpy())
        wrong_hard_errors.append(hard[~flat_correct].cpu().numpy())
        predicted_role = learned_mask & (selected != 0)
        predicted_switched += int(predicted_role.sum())
        false_switched += int(
            (predicted_role & (batch["target_switch_count"] == 0)).sum()
        )
        target_values = batch["target_switch_count"][learned_mask].to(torch.long)
        selected_values = selected[learned_mask].to(torch.long)
        for target, chosen in zip(target_values.tolist(), selected_values.tolist()):
            target = int(target)
            chosen = int(chosen)
            step_total[target] = step_total.get(target, 0) + 1
            if target == chosen:
                step_correct[target] = step_correct.get(target, 0) + 1
            confusion.setdefault(target, {})[chosen] = (
                confusion.setdefault(target, {}).get(chosen, 0) + 1
            )
    if eligible == 0:
        raise ValueError("selector validation contains no eligible query")
    recalls = {
        str(step): step_correct.get(step, 0) / count
        for step, count in sorted(step_total.items())
    }
    return {
        "objective": float(np.mean(objectives)),
        "switch_loss": float(np.mean(switch_losses)),
        "distance_cost_loss": float(np.mean(distance_costs)),
        "eligible_query_count": eligible,
        "switch_accuracy": correct / eligible,
        "switch_error_rate": 1.0 - correct / eligible,
        "switched_query_count": switched,
        "switched_recall": switched_correct / switched if switched else None,
        "predicted_switched_query_count": predicted_switched,
        "false_switched_query_count": false_switched,
        "switched_precision": (
            (predicted_switched - false_switched) / predicted_switched
            if predicted_switched else None
        ),
        "switch_recall_by_step": recalls,
        "switch_macro_recall": float(np.mean(list(recalls.values()))),
        "switch_minimum_step_recall": min(recalls.values()),
        "switch_confusion": {
            str(target): {str(chosen): count for chosen, count in sorted(row.items())}
            for target, row in sorted(confusion.items())
        },
        "conditional_position": _percentiles(conditional_errors),
        "hard_routed_position": _percentiles(hard_errors),
        "correct_route_position": _percentiles(correct_hard_errors),
        "wrong_route_position": _percentiles(wrong_hard_errors),
    }


def _checkpoint(
    path: Path, model: AnonymousCandidateFutureExpert,
    optimizer: torch.optim.Optimizer, epoch: int, update: int,
    metrics: dict[str, object], provenance: dict[str, object],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite selector checkpoint: {path}")
    torch.save({
        "model_class": "AnonymousCandidateFutureExpert",
        "model_config": model.config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_phase": "selector-only",
        "epoch": epoch,
        "update": update,
        "validation": metrics,
        "provenance": provenance,
    }, path)


def train(args: argparse.Namespace) -> Path:
    dataset_dir = Path(args.dataset).resolve()
    output_dir = Path(args.output).resolve()
    initial_path = Path(args.initial_checkpoint).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite selector run: {output_dir}")
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("test_accessed", True)):
        raise ValueError("selector dataset accessed test")
    dataset_manifest_sha256 = _sha256(manifest_path)
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    _validate_foundation_provenance(initial, dataset_manifest_sha256)
    model = _instantiate_from_checkpoint(initial, args.expert)
    if int(model.maximum_absolute_step) != max(abs(int(v)) for v in manifest["candidate_steps"]):
        raise ValueError("selector foundation candidate range disagrees with dataset")
    initial_checkpoint_sha256 = _sha256(initial_path)
    trainable_names, frozen_names = freeze_for_selector_only(model)
    initial_frozen_sha256 = _tensor_state_sha256(model, selector=False)
    initial_selector_sha256 = _tensor_state_sha256(model, selector=True)

    _seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    model = model.to(device)
    train_dataset = ObservableFutureDataset(
        dataset_dir, "train", args.expert, seed=args.seed, shuffle=True,
        sample_limit=args.train_limit,
    )
    validation_dataset = ObservableFutureDataset(
        dataset_dir, "validation", args.expert, seed=args.seed + 1,
        shuffle=False, sample_limit=args.validation_limit,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    probe_raw = next(iter(validation_loader))
    probe = _to_device(probe_raw, device)
    model.eval()
    with torch.no_grad():
        conditional_probe = _model_forward(model, probe)["conditional_delta_m"].detach().cpu()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    output_dir.mkdir(parents=True)
    source_dir = Path(__file__).resolve().parent
    provenance = {
        "schema_version": "stage3-observable-future-selector-run-v1",
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "dataset_qualification": True,
        "test_accessed": False,
        "phase": "truth-S combined selector-only",
        "evaluation_split": "validation",
        "initial_checkpoint": str(initial_path),
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "initial_frozen_state_sha256": initial_frozen_sha256,
        "initial_selector_state_sha256": initial_selector_sha256,
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
        "conditional_trajectory_trainable": False,
        "physical_identity_input": False,
        "frozen_S_retrained": False,
        "git": _git_state(),
        "source_sha256": {name: _sha256(source_dir / name) for name in SOURCE_FILES},
        "training_arguments": dict(vars(args)),
    }
    baseline = evaluate(model, validation_loader, device, args)
    history: list[dict[str, object]] = []
    best_tuple: tuple[float, ...] | None = None
    best_record: dict[str, object] | None = None
    update = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        set_selector_train_mode(model)
        train_objectives: list[float] = []
        for raw in train_loader:
            batch = _to_device(raw, device)
            next_update = update + 1
            learning_rate = _scheduled_learning_rate(
                next_update, base_learning_rate=args.learning_rate,
                minimum_learning_rate=args.minimum_learning_rate,
                warmup_updates=args.warmup_updates,
                total_updates=args.max_updates,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            prediction = _model_forward(model, batch)
            objective, _ = _selector_loss(prediction, batch, args)
            objective.backward()
            for name, parameter in model.named_parameters():
                if name in frozen_names and parameter.grad is not None:
                    raise RuntimeError(f"frozen trajectory parameter received gradient: {name}")
            if args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.gradient_clip_norm,
                )
            optimizer.step()
            update = next_update
            train_objectives.append(float(objective.detach().cpu()))
            if args.max_updates > 0 and update >= args.max_updates:
                break
        reached_limit = args.max_updates > 0 and update >= args.max_updates
        validate_now = (
            epoch % args.validation_interval == 0
            or reached_limit or epoch == args.epochs
        )
        record: dict[str, object] = {
            "epoch": epoch, "update": update,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_objective": float(np.mean(train_objectives)),
            "validation": None,
        }
        history.append(record)
        if not validate_now:
            if reached_limit:
                break
            continue
        frozen_sha256 = _tensor_state_sha256(model, selector=False)
        if frozen_sha256 != initial_frozen_sha256:
            raise RuntimeError("selector training changed frozen trajectory state")
        model.eval()
        with torch.no_grad():
            current_probe = _model_forward(model, probe)["conditional_delta_m"].detach().cpu()
        if not torch.equal(current_probe, conditional_probe):
            raise RuntimeError("selector training changed conditional trajectory output")
        validation = evaluate(model, validation_loader, device, args)
        if validation["conditional_position"] != baseline["conditional_position"]:
            raise RuntimeError("full validation conditional metrics changed")
        validation["conditional_probe_bit_exact"] = True
        validation["frozen_state_sha256"] = frozen_sha256
        record["validation"] = validation
        hard = validation["hard_routed_position"]
        selection = (
            float(hard["p95_m"]), float(hard["p99_m"]),
            -float(validation["switch_accuracy"]),
            -float(validation["switch_macro_recall"]),
            -float(validation["switch_minimum_step_recall"]),
        )
        checkpoint_path = output_dir / f"selector-epoch-{epoch:04d}.pt"
        _checkpoint(
            checkpoint_path, model, optimizer, epoch, update, validation,
            provenance,
        )
        checkpoint_sha256 = _sha256(checkpoint_path)
        if best_tuple is None or selection < best_tuple:
            best_tuple = selection
            best_record = {
                "path": checkpoint_path.name,
                "sha256": checkpoint_sha256,
                "epoch": epoch,
                "update": update,
                "selection_tuple": list(selection),
                "validation": validation,
            }
        _write_json(output_dir / "run_progress.json", {
            "status": "running", "baseline_validation": baseline,
            "history": history, "best": best_record,
            "elapsed_s": time.time() - started, **provenance,
        })
        print(json.dumps(record), flush=True)
        if reached_limit:
            break
    if best_record is None:
        raise RuntimeError("selector training produced no validation checkpoint")
    best_validation = best_record["validation"]
    gates = {
        "conditional_probe_bit_exact": bool(best_validation["conditional_probe_bit_exact"]),
        "frozen_state_unchanged": best_validation["frozen_state_sha256"] == initial_frozen_sha256,
        "minimum_switch_accuracy": float(best_validation["switch_accuracy"]) >= args.minimum_switch_accuracy,
        "minimum_macro_recall": float(best_validation["switch_macro_recall"]) >= args.minimum_macro_recall,
        "minimum_step_recall": float(best_validation["switch_minimum_step_recall"]) >= args.minimum_step_recall,
        "maximum_hard_p95_m": float(best_validation["hard_routed_position"]["p95_m"]) <= args.maximum_hard_p95_m,
    }
    final = {
        "status": "complete" if all(gates.values()) else "gate_failed",
        "stop_reason": "max_updates" if args.max_updates > 0 and update >= args.max_updates else "epoch_limit",
        "expert": args.expert,
        "model_config": model.config,
        "baseline_validation": baseline,
        "best": best_record,
        "history": history,
        "final_frozen_state_sha256": _tensor_state_sha256(model, selector=False),
        "final_selector_state_sha256": _tensor_state_sha256(model, selector=True),
        "gates": gates,
        "gate_thresholds": {
            "minimum_switch_accuracy": args.minimum_switch_accuracy,
            "minimum_macro_recall": args.minimum_macro_recall,
            "minimum_step_recall": args.minimum_step_recall,
            "maximum_hard_p95_m": args.maximum_hard_p95_m,
        },
        "elapsed_s": time.time() - started,
        **provenance,
    }
    _write_json(output_dir / "run_manifest.json", final)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--expert", choices=DYNAMIC_EXPERTS, default="combined")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-updates", type=int, default=6000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-updates", type=int, default=300)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--switch-weight", type=float, default=1.0)
    parser.add_argument("--macro-balance-weight", type=float, default=0.25)
    parser.add_argument("--switch-focal-gamma", type=float, default=2.0)
    parser.add_argument("--distance-cost-weight", type=float, default=1.0)
    parser.add_argument("--distance-cost-scale-m", type=float, default=0.3)
    parser.add_argument("--distance-cost-cap", type=float, default=2.0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--minimum-switch-accuracy", type=float, default=0.95)
    parser.add_argument("--minimum-macro-recall", type=float, default=0.92)
    parser.add_argument("--minimum-step-recall", type=float, default=0.80)
    parser.add_argument("--maximum-hard-p95-m", type=float, default=0.10)
    args = parser.parse_args()
    if (
        args.batch_size < 1 or args.workers < 0 or args.epochs < 1
        or args.max_updates < 1 or args.validation_interval < 1
        or args.warmup_updates < 0
        or args.minimum_learning_rate > args.learning_rate
        or min(args.learning_rate, args.minimum_learning_rate) <= 0
        or min(args.switch_weight, args.distance_cost_weight,
               args.switch_focal_gamma, args.weight_decay,
               args.gradient_clip_norm) < 0
        or args.switch_weight + args.distance_cost_weight <= 0
        or min(args.distance_cost_scale_m, args.distance_cost_cap,
               args.maximum_hard_p95_m) <= 0
        or not 0 <= args.macro_balance_weight <= 1
        or not 0 <= args.minimum_switch_accuracy <= 1
        or not 0 <= args.minimum_macro_recall <= 1
        or not 0 <= args.minimum_step_recall <= 1
    ):
        parser.error("observable F selector training arguments are invalid")
    print(train(args))


if __name__ == "__main__":
    main()
