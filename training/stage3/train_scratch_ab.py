"""Train controlled scratch A/B future-observation models on the v4 shards.

Model A uses only masked direct future-observation Huber loss.  Model B is
architecturally identical and adds a low-weight physical-position auxiliary
loss.  Both models start from the exact same random state and see the exact
same batches and dropout masks.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import stage3_direct_observation_loss
from .model import Stage3TCN
from .shard_dataset import Stage3ShardDataset


PERMUTATIONS = tuple(itertools.permutations(range(4)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = ("train_scratch_ab.py", "model.py", "losses.py", "shard_dataset.py")
    return {name: _sha256(root / name) for name in names}


def _git_state(dataset: Path) -> dict[str, object]:
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(repo), "status", "--short"], text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    return {
        "git_commit": commit,
        "worktree_dirty": dirty,
        "dataset_manifest_sha256": _sha256(dataset / "dataset_manifest.json"),
    }


def _model_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "channels": args.channels,
        "dropout": args.dropout,
        "input_features": 7,
        "observation_heads": False,
        "direct_observation_heads": True,
    }


def _all_permuted(value: torch.Tensor) -> torch.Tensor:
    permutation = torch.tensor(PERMUTATIONS, dtype=torch.long, device=value.device)
    index = permutation.view(1, len(PERMUTATIONS), 1, 4, 1).expand(
        value.shape[0], -1, value.shape[1], -1, value.shape[3]
    )
    return value.unsqueeze(1).expand(
        -1, len(PERMUTATIONS), -1, -1, -1
    ).gather(3, index)


def _aligned_outputs(
    prediction: dict[str, torch.Tensor], target_position: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    observation_candidates = _all_permuted(prediction["observation_mean"])
    score = torch.linalg.vector_norm(
        observation_candidates - target_position.unsqueeze(1), dim=-1
    ).mean(dim=(2, 3))
    best = score.argmin(dim=1)
    gather = best.view(-1, 1, 1, 1, 1).expand(
        -1, 1, observation_candidates.shape[2], 4, 3
    )
    observation = observation_candidates.gather(1, gather).squeeze(1)
    physical_candidates = _all_permuted(prediction["position_mean"])
    physical = physical_candidates.gather(1, gather).squeeze(1)
    return observation, physical


def _validation_values(
    prediction: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[np.ndarray, np.ndarray]:
    observation, physical = _aligned_outputs(prediction, batch["future_position"])
    active = (
        batch["future_observation_mask"]
        & batch["future_observation_frame_available"].unsqueeze(-1)
        & ~batch["future_observation_ambiguous"].unsqueeze(-1)
    )
    usable = active.any(dim=-1)
    denominator = active.sum(dim=-1).clamp_min(1)
    observation_point = torch.linalg.vector_norm(
        observation - batch["future_observation_position"], dim=-1
    )
    observation_query = (observation_point * active).sum(dim=-1) / denominator
    physical_query = torch.linalg.vector_norm(
        physical - batch["future_position"], dim=-1
    ).mean(dim=-1)
    return (
        observation_query[usable].detach().cpu().numpy(),
        physical_query.detach().cpu().numpy().reshape(-1),
    )


def _summary(parts: list[np.ndarray]) -> dict[str, float | int]:
    values = np.concatenate(parts).astype(np.float64, copy=False)
    if values.size == 0:
        raise ValueError("validation has no usable future observations")
    return {
        "count": int(values.size),
        "mean_m": float(values.mean()),
        "median_m": float(np.quantile(values, 0.5)),
        "p90_m": float(np.quantile(values, 0.9)),
        "p95_m": float(np.quantile(values, 0.95)),
        "p99_m": float(np.quantile(values, 0.99)),
    }


def _validate_pair(
    model_a: Stage3TCN,
    model_b: Stage3TCN,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, dict[str, float | int]]]:
    model_a.eval()
    model_b.eval()
    observation_values: dict[str, list[np.ndarray]] = {"A": [], "B": []}
    physical_values: dict[str, list[np.ndarray]] = {"A": [], "B": []}
    with torch.no_grad():
        for raw in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in raw.items()}
            for label, model in (("A", model_a), ("B", model_b)):
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    prediction = model(
                        batch["obs"], batch["obs_mask"], batch["event_mask"],
                        batch["event_time_s"], batch["tau"],
                    )
                observation, physical = _validation_values(prediction, batch)
                observation_values[label].append(observation)
                physical_values[label].append(physical)
    return {
        label: {
            "prediction_vs_future_observation": _summary(observation_values[label]),
            "prediction_vs_truth": _summary(physical_values[label]),
        }
        for label in ("A", "B")
    }


def _capture_rng(device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    return cpu, cuda


def _restore_rng(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    cpu, cuda = state
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


def _train_one(
    model: Stage3TCN,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    physical_aux_weight: float,
    huber_beta_m: float,
) -> tuple[float, float, float]:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        prediction = model(
            batch["obs"], batch["obs_mask"], batch["event_mask"],
            batch["event_time_s"], batch["tau"],
        )
        loss, metrics = stage3_direct_observation_loss(
            prediction,
            batch["future_position"],
            batch["future_observation_position"],
            batch["future_observation_mask"],
            batch["future_observation_frame_available"],
            batch["future_observation_ambiguous"],
            physical_aux_weight=physical_aux_weight,
            huber_beta_m=huber_beta_m,
        )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return (
        float(loss.detach().cpu()),
        metrics["observation_huber"],
        metrics["physical_huber"],
    )


def _train_epoch_pair(
    model_a: Stage3TCN,
    model_b: Stage3TCN,
    loader: DataLoader,
    optimizer_a: torch.optim.Optimizer,
    optimizer_b: torch.optim.Optimizer,
    scaler_a: torch.amp.GradScaler,
    scaler_b: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    model_a.train()
    model_b.train()
    totals = {
        "A": {"loss": 0.0, "observation_huber": 0.0, "physical_huber": 0.0},
        "B": {"loss": 0.0, "observation_huber": 0.0, "physical_huber": 0.0},
    }
    count = 0
    for raw in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in raw.items()}
        batch_count = int(batch["obs"].shape[0])
        shared_rng = _capture_rng(device)
        values_a = _train_one(
            model_a, batch, optimizer_a, scaler_a, device,
            physical_aux_weight=0.0, huber_beta_m=args.huber_beta_m,
        )
        _restore_rng(shared_rng)
        values_b = _train_one(
            model_b, batch, optimizer_b, scaler_b, device,
            physical_aux_weight=args.physical_aux_weight,
            huber_beta_m=args.huber_beta_m,
        )
        for label, values in (("A", values_a), ("B", values_b)):
            for key, value in zip(("loss", "observation_huber", "physical_huber"), values):
                totals[label][key] += value * batch_count
        count += batch_count
    return {
        label: {key: value / max(count, 1) for key, value in totals[label].items()}
        for label in ("A", "B")
    }


def _selection_score(metrics: dict[str, dict[str, float | int]]) -> float:
    observation = metrics["prediction_vs_future_observation"]
    return float(observation["median_m"]) + 0.25 * float(observation["p95_m"])


def _save_checkpoint(
    path: Path,
    model: Stage3TCN,
    label: str,
    epoch: int,
    validation: dict[str, dict[str, float | int]],
    provenance: dict[str, object],
    args: argparse.Namespace,
    role: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    payload: dict[str, object] = {
        "model": model.state_dict(),
        "model_class": "Stage3TCN-direct-observation-v1",
        "model_config": _model_config(args),
        "objective": "observation_only" if label == "A" else "observation_plus_physical_aux",
        "physical_aux_weight": 0.0 if label == "A" else args.physical_aux_weight,
        "checkpoint_role": role,
        "epoch": epoch,
        "validation": validation,
        "selection_score": _selection_score(validation),
        "seed": args.seed,
        "provenance": provenance,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def train(args: argparse.Namespace) -> Path:
    _seed(args.seed)
    dataset = Path(args.dataset).resolve()
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "stage3-dataset-v4-observation"
        or not manifest.get("qualification_passed")
    ):
        raise ValueError("scratch A/B training requires a qualified v4 dataset")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training output: {output}")
    output.mkdir(parents=True)

    train_ds = Stage3ShardDataset(
        dataset, "train", augment=not args.no_augment, seed=args.seed,
        shuffle=True, sample_limit=args.train_sample_limit,
    )
    validation_ds = Stage3ShardDataset(
        dataset, "validation", augment=False, seed=args.seed, shuffle=False,
        sample_limit=args.validation_sample_limit,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_ds, batch_size=args.batch_size, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    initial = Stage3TCN(**_model_config(args))
    model_a = copy.deepcopy(initial).to(device)
    model_b = copy.deepcopy(initial).to(device)
    del initial
    if not all(
        torch.equal(a, b)
        for a, b in zip(model_a.state_dict().values(), model_b.state_dict().values())
    ):
        raise RuntimeError("scratch A/B models did not receive identical initialization")

    optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_a, max(1, args.epochs), eta_min=args.lr * 0.01
    )
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_b, max(1, args.epochs), eta_min=args.lr * 0.01
    )
    scaler_a = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scaler_b = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    provenance: dict[str, object] = {
        "schema_version": "stage3-scratch-ab-training-run-v1",
        "dataset": str(dataset),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "config": vars(args),
        "initialization": "identical_random_state_no_checkpoint",
        "test_accessed": False,
        "training_source_sha256": _source_hashes(),
        "environment": {
            "python": __import__("sys").version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        **_git_state(dataset),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )

    initial_validation = _validate_pair(
        model_a, model_b, validation_loader, device
    )
    history: list[dict[str, object]] = [{
        "epoch": 0,
        "validation": initial_validation,
        "selection_score": {
            label: _selection_score(initial_validation[label]) for label in ("A", "B")
        },
    }]
    history_path = output / "stage3-scratch-ab-history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    best = {"A": float("inf"), "B": float("inf")}
    stale = {"A": 0, "B": 0}
    best_paths = {
        "A": output / f"stage3-scratch-a-seed{args.seed}-best.pt",
        "B": output / f"stage3-scratch-b-seed{args.seed}-best.pt",
    }
    started = time.monotonic()
    stop_reason = "epochs_completed"
    epochs_completed = 0
    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        train_metrics = _train_epoch_pair(
            model_a, model_b, train_loader, optimizer_a, optimizer_b,
            scaler_a, scaler_b, device, args,
        )
        validation = _validate_pair(model_a, model_b, validation_loader, device)
        scheduler_a.step()
        scheduler_b.step()
        record: dict[str, object] = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": validation,
            "selection_score": {
                label: _selection_score(validation[label]) for label in ("A", "B")
            },
            "lr": {"A": scheduler_a.get_last_lr()[0], "B": scheduler_b.get_last_lr()[0]},
        }
        history.append(record)
        # Persist every completed epoch so a detached/long-running Windows
        # process remains externally auditable even when stdout is buffered.
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        epochs_completed = epoch + 1
        print(json.dumps(record, sort_keys=True), flush=True)
        for label, model in (("A", model_a), ("B", model_b)):
            score = _selection_score(validation[label])
            if not np.isfinite(score):
                raise FloatingPointError(f"scratch model {label} produced a non-finite score")
            if score < best[label]:
                best[label] = score
                stale[label] = 0
                _save_checkpoint(
                    best_paths[label], model, label, epoch + 1,
                    validation[label], provenance, args, "best",
                )
            else:
                stale[label] += 1
        if all(value >= args.patience for value in stale.values()):
            stop_reason = "both_models_early_stopping"
            break
        if (
            args.max_wall_minutes > 0
            and time.monotonic() - started >= args.max_wall_minutes * 60.0
        ):
            stop_reason = "wall_time_limit"
            break

    for label, model, optimizer, scheduler, scaler in (
        ("A", model_a, optimizer_a, scheduler_a, scaler_a),
        ("B", model_b, optimizer_b, scheduler_b, scaler_b),
    ):
        _save_checkpoint(
            output / f"stage3-scratch-{label.lower()}-seed{args.seed}-last.pt",
            model, label, epochs_completed, validation[label], provenance, args,
            "last", optimizer, scheduler, scaler,
        )
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    final = {
        **provenance,
        "status": "complete",
        "stop_reason": stop_reason,
        "epochs_completed": epochs_completed,
        "best": {
            label: {
                "path": best_paths[label].name,
                "sha256": _sha256(best_paths[label]),
                "selection_score": best[label],
            }
            for label in ("A", "B")
        },
    }
    (output / "run_manifest.json").write_text(
        json.dumps(final, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--physical-aux-weight", type=float, default=0.2)
    parser.add_argument("--huber-beta-m", type=float, default=0.1)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--device", default="")
    parser.add_argument("--train-sample-limit", type=int, default=0)
    parser.add_argument("--validation-sample-limit", type=int, default=0)
    parser.add_argument("--max-wall-minutes", type=float, default=0.0)
    args = parser.parse_args()
    if (
        args.epochs < 1 or args.patience < 1 or args.batch_size < 1
        or args.lr <= 0 or args.channels < 1 or args.physical_aux_weight < 0
        or args.huber_beta_m <= 0 or args.train_sample_limit < 0
        or args.validation_sample_limit < 0
    ):
        parser.error("invalid non-positive scratch A/B training argument")
    print(train(args))


if __name__ == "__main__":
    main()
