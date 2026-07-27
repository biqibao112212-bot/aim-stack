"""Train one independent anonymous observable-target F expert.

This first executable gate uses truth-S candidate relations from the qualified
anonymous derivative.  Only after it passes is the identical F checkpoint
eligible for the frozen-S A/B adapter gate.  Test data, PnP, router, export, and
online integration remain sealed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .observable_future_dataset import ObservableFutureDataset
from .observable_future_loss import (
    observable_future_batch_metrics,
    observable_future_loss,
)
from .observable_future_model import AnonymousCandidateFutureExpert, DYNAMIC_EXPERTS
from .train_causal_physical_ab import _git_state, _seed, _to_device, _write_json


SOURCE_FILES = (
    "observable_future_dataset.py",
    "build_observable_future_dataset.py",
    "observable_future_model.py",
    "observable_future_loss.py",
    "audit_observable_future_dataset.py",
    "train_observable_future_expert.py",
)


class _CachedSamples(Dataset):
    """Materialize a bounded tiny-fit subset once; never used for full runs."""

    def __init__(self, source: ObservableFutureDataset) -> None:
        self.samples = list(source)
        if not self.samples:
            raise ValueError("tiny-fit cache is empty")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


def _sample_query_roles(sample: dict[str, torch.Tensor]) -> tuple[tuple[int, int], ...]:
    tau = sample["tau_s"]
    mask = sample["target_query_mask"].to(torch.bool) & (tau > 0)
    step = sample["target_switch_count"][mask].to(torch.long)
    tau_bin = torch.clamp(torch.floor(tau[mask] / 0.1).to(torch.long), 0, 5)
    return tuple(sorted({(int(s), int(t)) for s, t in zip(step, tau_bin)}))


def _samples_sha256(
    samples: list[dict[str, torch.Tensor]], source_indices: list[int]
) -> str:
    digest = hashlib.sha256()
    for source_index, sample in zip(source_indices, samples):
        digest.update(int(source_index).to_bytes(8, "little", signed=False))
        for name in sorted(sample):
            value = sample[name].detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class _BalancedCachedSamples(Dataset):
    """Deterministically balance a tiny-fit cache by signed-step/time roles."""

    def __init__(
        self, source: ObservableFutureDataset, *, limit: int, seed: int
    ) -> None:
        all_samples = list(source)
        if limit < 1 or len(all_samples) < limit:
            raise ValueError("balanced tiny-fit source is smaller than its limit")
        indexed = [
            (index, sample, _sample_query_roles(sample))
            for index, sample in enumerate(all_samples)
        ]
        indexed = [item for item in indexed if item[2]]
        if len(indexed) < limit:
            raise ValueError("balanced tiny-fit has too few learned-query samples")
        source_indices = [item[0] for item in indexed]
        eligible_samples = [item[1] for item in indexed]
        roles = [item[2] for item in indexed]
        support = Counter(role for values in roles for role in values)
        if not support:
            raise ValueError("balanced tiny-fit source has no query roles")
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, values in enumerate(roles):
            assigned = min(values, key=lambda role: (support[role], role))
            buckets[assigned].append(index)
        rng = np.random.default_rng(int(seed))
        for values in buckets.values():
            rng.shuffle(values)
        ordered_roles = sorted(buckets, key=lambda role: (support[role], role))
        selected: list[int] = []
        while len(selected) < limit:
            progress = False
            for role in ordered_roles:
                if buckets[role]:
                    selected.append(buckets[role].pop())
                    progress = True
                    if len(selected) == limit:
                        break
            if not progress:
                raise RuntimeError("balanced tiny-fit selection exhausted early")
        self.source_indices = [source_indices[index] for index in selected]
        self.samples = [eligible_samples[index] for index in selected]
        self.selection_sha256 = _samples_sha256(self.samples, self.source_indices)
        query_support: Counter[tuple[int, int]] = Counter()
        for sample in self.samples:
            tau = sample["tau_s"]
            mask = sample["target_query_mask"].to(torch.bool) & (tau > 0)
            for step, time in zip(
                sample["target_switch_count"][mask].to(torch.long), tau[mask]
            ):
                query_support[(int(step), min(int(float(time) // 0.1), 5))] += 1
        self.query_role_support = {
            f"step={step},tau_bin={tau_bin}": count
            for (step, tau_bin), count in sorted(query_support.items())
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return observable_future_loss(
        prediction,
        batch["candidate_step"], batch["candidate_mask"], batch["tau_s"],
        batch["target_switch_count"], batch["target_visible_delta_m"],
        batch["target_query_mask"],
        huber_beta_m=args.huber_beta_m,
        switch_weight=args.switch_weight,
        position_weight=args.position_weight,
        position_mse_weight=args.position_mse_weight,
        rate_weight=args.rate_weight,
        rate_huber_beta_mps=args.rate_huber_beta_mps,
        rate_tau_floor_s=args.rate_tau_floor_s,
        position_tail_weight=args.position_tail_weight,
        position_tail_fraction=args.position_tail_fraction,
        trend_weight=args.trend_weight,
        macro_balance_weight=args.macro_balance_weight,
        position_macro_balance_weight=args.position_macro_balance_weight,
        switch_focal_gamma=args.switch_focal_gamma,
    )


def _percentiles(values: list[np.ndarray]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50_m": None, "p95_m": None, "p99_m": None, "max_m": None}
    merged = np.concatenate(values).astype(np.float64, copy=False)
    return {
        "count": int(merged.size),
        "p50_m": float(np.quantile(merged, 0.50)),
        "p95_m": float(np.quantile(merged, 0.95)),
        "p99_m": float(np.quantile(merged, 0.99)),
        "max_m": float(merged.max(initial=0.0)),
    }


def _scheduled_learning_rate(
    update: int,
    *,
    base_learning_rate: float,
    minimum_learning_rate: float,
    warmup_updates: int,
    total_updates: int,
) -> float:
    """Deterministic per-update warmup/cosine schedule.

    ``total_updates == 0`` deliberately keeps the old constant-rate behaviour;
    formal runs must either provide an explicit update budget or use a future
    epoch-counted scheduler once expert sample counts are manifest-bound.
    """
    if total_updates <= 0:
        return float(base_learning_rate)
    if warmup_updates > 0 and update <= warmup_updates:
        fraction = float(update) / float(warmup_updates)
        return float(max(minimum_learning_rate, base_learning_rate * fraction))
    decay_updates = max(total_updates - warmup_updates, 1)
    progress = min(max(update - warmup_updates, 0) / decay_updates, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(
        minimum_learning_rate
        + (base_learning_rate - minimum_learning_rate) * cosine
    )


@torch.no_grad()
def evaluate(
    model: AnonymousCandidateFutureExpert,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    model.eval()
    objectives: list[float] = []
    conditional_errors: list[np.ndarray] = []
    hard_errors: list[np.ndarray] = []
    eligible = correct = switched = switched_correct = 0
    predicted_switched = false_switched = 0
    step_total: dict[int, int] = {}
    step_correct: dict[int, int] = {}
    structural_zero_queries = 0
    for raw in loader:
        batch = _to_device(raw, device)
        prediction = _model_forward(model, batch)
        objective, _ = _loss(prediction, batch, args)
        objectives.append(float(objective.cpu()))
        tau = batch["tau_s"]
        if tau.ndim == 1:
            tau = tau.unsqueeze(0).expand_as(batch["target_switch_count"])
        structural_mask = batch["target_query_mask"].to(torch.bool) & (tau == 0)
        if bool(structural_mask.any()):
            if not bool(torch.all(prediction["selected_switch_step"][structural_mask] == 0)):
                raise ValueError("tau-zero structural switch identity failed")
            if not torch.equal(
                prediction["delta_m"][structural_mask],
                torch.zeros_like(prediction["delta_m"][structural_mask]),
            ):
                raise ValueError("tau-zero structural position identity failed")
        structural_zero_queries += int(structural_mask.sum())
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
        hard_errors.append(metric["hard_error_m"].cpu().numpy())
        selected = prediction["selected_switch_step"]
        mask = learned_mask
        predicted_role = mask & (selected != 0)
        predicted_switched += int(predicted_role.sum())
        false_switched += int(
            (predicted_role & (batch["target_switch_count"] == 0)).sum()
        )
        for step in batch["candidate_step"].unique().tolist():
            step_value = int(step)
            role = mask & (batch["target_switch_count"] == step)
            count = int(role.sum())
            if count:
                step_total[step_value] = step_total.get(step_value, 0) + count
                step_correct[step_value] = step_correct.get(step_value, 0) + int(
                    (role & (selected == step)).sum()
                )
    if eligible == 0:
        raise ValueError("validation contains no eligible observable queries")
    recalls = {
        str(step): step_correct.get(step, 0) / count
        for step, count in sorted(step_total.items())
    }
    minimum_recall = min(recalls.values())
    return {
        "objective": float(np.mean(objectives)),
        "structural_tau_zero_query_count": structural_zero_queries,
        "eligible_query_count": eligible,
        "switch_accuracy": correct / eligible,
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
        "switch_minimum_step_recall": minimum_recall,
        "conditional_position": _percentiles(conditional_errors),
        "hard_routed_position": _percentiles(hard_errors),
    }


def _checkpoint(
    path: Path,
    model: AnonymousCandidateFutureExpert,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    update: int,
    metrics: dict[str, object],
    provenance: dict[str, object],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite F checkpoint: {path}")
    torch.save({
        "model_class": "AnonymousCandidateFutureExpert",
        "model_config": model.config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "update": update,
        "validation": metrics,
        "provenance": provenance,
    }, path)


def train(args: argparse.Namespace) -> Path:
    dataset_dir = Path(args.dataset).resolve()
    output_dir = Path(args.output).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite F run: {output_dir}")
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("test_accessed", True)):
        raise ValueError("observable F training dataset accessed test")
    _seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    balanced_tiny_fit = args.tiny_fit and args.tiny_fit_selection == "balanced"
    train_dataset = ObservableFutureDataset(
        dataset_dir, "train", args.expert, seed=args.seed,
        shuffle=not args.tiny_fit,
        sample_limit=0 if balanced_tiny_fit else args.train_limit,
    )
    validation_split = "train" if args.tiny_fit else "validation"
    if args.tiny_fit:
        cached_train = (
            _BalancedCachedSamples(
                train_dataset, limit=args.train_limit, seed=args.seed,
            )
            if balanced_tiny_fit else _CachedSamples(train_dataset)
        )
        if args.train_limit > 0 and len(cached_train) != args.train_limit:
            raise ValueError("tiny-fit cache did not reach the requested fixed size")
        train_source: Dataset = cached_train
        validation_source: Dataset = cached_train
    else:
        train_source = train_dataset
        validation_source = ObservableFutureDataset(
            dataset_dir, validation_split, args.expert, seed=args.seed + 1,
            shuffle=False, sample_limit=args.validation_limit,
        )
    train_loader = DataLoader(
        train_source, batch_size=args.batch_size, num_workers=args.workers,
        shuffle=bool(args.tiny_fit),
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_source, batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    model = AnonymousCandidateFutureExpert(
        args.expert, channels=args.channels, dropout=args.dropout,
        position_scale_m=args.position_scale_m,
        history_scale_s=args.history_scale_s,
        trained_horizon_s=args.trained_horizon_s,
        maximum_absolute_step=max(abs(value) for value in manifest["candidate_steps"]),
        trajectory_rank=args.trajectory_rank,
    ).to(device)
    initial_checkpoint_sha256 = None
    if args.initial_checkpoint:
        initial_path = Path(args.initial_checkpoint).resolve()
        initial = torch.load(initial_path, map_location="cpu", weights_only=False)
        if initial.get("model_class") != "AnonymousCandidateFutureExpert":
            raise ValueError("initial F checkpoint model class mismatch")
        if initial.get("model_config") != model.config:
            raise ValueError("initial F checkpoint model config mismatch")
        model.load_state_dict(initial["model"], strict=True)
        initial_checkpoint_sha256 = _sha256(initial_path)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    output_dir.mkdir(parents=True)
    source_dir = Path(__file__).resolve().parent
    provenance = {
        "schema_version": "stage3-observable-future-expert-run-v1",
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "dataset_qualification": True,
        "test_accessed": False,
        "phase": "truth-S relation gate",
        "evaluation_split": validation_split,
        "tiny_fit_same_window_evaluation": bool(args.tiny_fit),
        "tiny_fit_cached_in_memory": bool(args.tiny_fit),
        "tiny_fit_selection_sha256": (
            getattr(cached_train, "selection_sha256", None)
            if args.tiny_fit else None
        ),
        "tiny_fit_query_role_support": (
            getattr(cached_train, "query_role_support", None)
            if args.tiny_fit else None
        ),
        "frozen_S_retrained": False,
        "initial_checkpoint": (
            str(Path(args.initial_checkpoint).resolve())
            if args.initial_checkpoint else None
        ),
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "physics_decoder_at_inference": False,
        "git": _git_state(),
        "source_sha256": {name: _sha256(source_dir / name) for name in SOURCE_FILES},
        "training_arguments": dict(vars(args)),
    }
    history: list[dict[str, object]] = []
    best_tuple: tuple[float, ...] | None = None
    best_record: dict[str, object] | None = None
    update = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        if not args.tiny_fit:
            train_dataset.set_epoch(epoch)
        model.train()
        train_objective: list[float] = []
        for raw in train_loader:
            batch = _to_device(raw, device)
            next_update = update + 1
            learning_rate = _scheduled_learning_rate(
                next_update,
                base_learning_rate=args.learning_rate,
                minimum_learning_rate=args.minimum_learning_rate,
                warmup_updates=args.warmup_updates,
                total_updates=args.max_updates,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            prediction = _model_forward(model, batch)
            objective, _ = _loss(prediction, batch, args)
            objective.backward()
            if args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            update = next_update
            train_objective.append(float(objective.detach().cpu()))
            if args.max_updates > 0 and update >= args.max_updates:
                break
        reached_update_limit = args.max_updates > 0 and update >= args.max_updates
        validate_now = (
            epoch % args.validation_interval == 0
            or reached_update_limit or epoch == args.epochs
        )
        record = {
            "epoch": epoch,
            "update": update,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_objective": float(np.mean(train_objective)),
            "validation": None,
        }
        history.append(record)
        if not validate_now:
            if reached_update_limit:
                break
            continue
        validation = evaluate(model, validation_loader, device, args)
        record["validation"] = validation
        selection = (
            -float(validation["switch_minimum_step_recall"]),
            -float(validation["switch_macro_recall"]),
            float(validation["hard_routed_position"]["p99_m"]),
            float(validation["hard_routed_position"]["p95_m"]),
            float(validation["conditional_position"]["p95_m"]),
        )
        checkpoint_path = output_dir / f"epoch-{epoch:04d}.pt"
        _checkpoint(checkpoint_path, model, optimizer, epoch, update, validation, provenance)
        checkpoint_sha = _sha256(checkpoint_path)
        if best_tuple is None or selection < best_tuple:
            best_tuple = selection
            best_record = {
                "path": checkpoint_path.name,
                "sha256": checkpoint_sha,
                "epoch": epoch,
                "update": update,
                "selection_tuple": list(selection),
                "validation": validation,
            }
        _write_json(output_dir / "run_progress.json", {
            "status": "running", "history": history, "best": best_record,
            "elapsed_s": time.time() - started, **provenance,
        })
        print(json.dumps(record), flush=True)
        if reached_update_limit:
            break
    if best_record is None:
        raise RuntimeError("F training produced no validation checkpoint")
    best_validation = best_record["validation"]
    gates = {
        "switch_macro_recall": float(best_validation["switch_macro_recall"]) >= args.minimum_macro_recall,
        "switch_minimum_step_recall": float(best_validation["switch_minimum_step_recall"]) >= args.minimum_step_recall,
        "conditional_p95_m": float(best_validation["conditional_position"]["p95_m"]) <= args.maximum_conditional_p95_m,
        "hard_p95_m": float(best_validation["hard_routed_position"]["p95_m"]) <= args.maximum_hard_p95_m,
        "hard_p99_m": float(best_validation["hard_routed_position"]["p99_m"]) <= args.maximum_hard_p99_m,
    }
    final = {
        "status": "complete" if all(gates.values()) else "gate_failed",
        "stop_reason": "max_updates" if args.max_updates > 0 and update >= args.max_updates else "epoch_limit",
        "expert": args.expert,
        "model_config": model.config,
        "best": best_record,
        "history": history,
        "gates": gates,
        "gate_thresholds": {
            "minimum_macro_recall": args.minimum_macro_recall,
            "minimum_step_recall": args.minimum_step_recall,
            "maximum_conditional_p95_m": args.maximum_conditional_p95_m,
            "maximum_hard_p95_m": args.maximum_hard_p95_m,
            "maximum_hard_p99_m": args.maximum_hard_p99_m,
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
    parser.add_argument("--expert", choices=DYNAMIC_EXPERTS, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--position-scale-m", type=float, default=1.0)
    parser.add_argument("--history-scale-s", type=float, default=1.0)
    parser.add_argument("--trained-horizon-s", type=float, default=0.55)
    parser.add_argument("--trajectory-rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-updates", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=0.0)
    parser.add_argument("--huber-beta-m", type=float, default=0.01)
    parser.add_argument("--switch-weight", type=float, default=1.0)
    parser.add_argument("--position-weight", type=float, default=50.0)
    parser.add_argument("--position-mse-weight", type=float, default=0.0)
    parser.add_argument("--rate-weight", type=float, default=0.0)
    parser.add_argument("--rate-huber-beta-mps", type=float, default=0.02)
    parser.add_argument("--rate-tau-floor-s", type=float, default=0.05)
    parser.add_argument("--position-tail-weight", type=float, default=0.0)
    parser.add_argument("--position-tail-fraction", type=float, default=0.2)
    parser.add_argument("--trend-weight", type=float, default=0.0)
    parser.add_argument("--macro-balance-weight", type=float, default=0.25)
    parser.add_argument("--position-macro-balance-weight", type=float, default=0.0)
    parser.add_argument("--switch-focal-gamma", type=float, default=0.0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--tiny-fit", action="store_true")
    parser.add_argument("--initial-checkpoint", default="")
    parser.add_argument(
        "--tiny-fit-selection", choices=("balanced", "sequential"),
        default="balanced",
    )
    parser.add_argument("--minimum-macro-recall", type=float, default=0.995)
    parser.add_argument("--minimum-step-recall", type=float, default=0.99)
    parser.add_argument("--maximum-conditional-p95-m", type=float, default=0.001)
    parser.add_argument("--maximum-hard-p95-m", type=float, default=0.002)
    parser.add_argument("--maximum-hard-p99-m", type=float, default=0.005)
    args = parser.parse_args()
    if args.tiny_fit:
        if args.train_limit == 0:
            args.train_limit = 512
        args.dropout = 0.0
        args.weight_decay = 0.0
        if args.max_updates == 0:
            args.max_updates = 5000
        if args.warmup_updates == 0:
            args.warmup_updates = 250
        if args.validation_limit == 0:
            args.validation_limit = args.train_limit
        if args.validation_interval == 1:
            args.validation_interval = 25
    if (
        args.channels < 16 or args.trajectory_rank < 2
        or not 0 <= args.dropout < 1 or args.batch_size < 1
        or args.workers < 0 or args.epochs < 1 or args.max_updates < 0
        or args.validation_interval < 1 or args.warmup_updates < 0
        or args.minimum_learning_rate > args.learning_rate
        or min(args.learning_rate, args.minimum_learning_rate,
               args.huber_beta_m, args.position_scale_m,
               args.history_scale_s, args.trained_horizon_s) <= 0
        or min(args.switch_weight, args.position_weight,
               args.position_mse_weight, args.rate_weight,
               args.position_tail_weight, args.trend_weight) < 0
        or min(args.rate_huber_beta_mps, args.rate_tau_floor_s) <= 0
        or not 0 <= args.macro_balance_weight <= 1
        or not 0 <= args.position_macro_balance_weight <= 1
        or args.switch_focal_gamma < 0
        or not 0 < args.position_tail_fraction <= 1
        or not 0 <= args.minimum_macro_recall <= 1
        or not 0 <= args.minimum_step_recall <= 1
        or min(args.maximum_conditional_p95_m, args.maximum_hard_p95_m,
               args.maximum_hard_p99_m) <= 0
    ):
        parser.error("observable F training arguments are invalid")
    print(train(args))


if __name__ == "__main__":
    main()
