"""Train the v4 future-observation predictor with masked exact-frame labels."""

from __future__ import annotations

import argparse
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

from .losses import stage3_observation_loss
from .model import Stage3TCN
from .shard_dataset import Stage3ShardDataset
from .train import _position_set_l2


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
    return {name: _sha256(root / name) for name in ("train_observation.py", "model.py", "losses.py", "shard_dataset.py")}


def _git_state(dataset: Path) -> dict[str, object]:
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    return {"git_commit": commit, "worktree_dirty": dirty, "dataset_manifest_sha256": _sha256(dataset / "dataset_manifest.json")}


def _load_initial(model: Stage3TCN, path: Path) -> dict[str, list[str]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model", checkpoint)
    destination = model.state_dict()
    accepted: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in source.items():
        if key not in destination:
            skipped.append(key)
            continue
        if key == "armor_mlp.0.weight" and tuple(value.shape) == (32, 5) and tuple(destination[key].shape) == (32, 7):
            expanded = destination[key].clone()
            expanded[:, 5:] = 0.0
            expanded[:, :5] = value
            accepted[key] = expanded
        elif tuple(value.shape) == tuple(destination[key].shape):
            accepted[key] = value
        else:
            skipped.append(key)
    model.load_state_dict(accepted, strict=False)
    return {"accepted": sorted(accepted), "skipped": sorted(skipped)}


def _paired_observation_l2(prediction: torch.Tensor, target_position: torch.Tensor, target_observation: torch.Tensor, target_mask: torch.Tensor, frame_available: torch.Tensor, ambiguous: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    predicted_observation = prediction["position_mean"] + prediction["observation_residual_mean"]
    physical_scores = []
    aligned_observations = []
    for permutation in PERMUTATIONS:
        index = torch.tensor(permutation, dtype=torch.long, device=prediction["position_mean"].device)
        aligned_physical = prediction["position_mean"].index_select(2, index)
        aligned_observation = predicted_observation.index_select(2, index)
        physical_scores.append(torch.linalg.vector_norm(aligned_physical - target_position, dim=-1).mean(dim=(1, 2)))
        aligned_observations.append(aligned_observation)
    best = torch.stack(physical_scores, dim=1).argmin(dim=1)
    selected = torch.stack(aligned_observations, dim=1)
    gather = best.view(-1, 1, 1, 1, 1).expand(-1, 1, selected.shape[2], selected.shape[3], selected.shape[4])
    selected = torch.gather(selected, 1, gather).squeeze(1)
    active = target_mask & frame_available.unsqueeze(-1) & ~ambiguous.unsqueeze(-1)
    errors = torch.linalg.vector_norm(selected - target_observation, dim=-1)
    active_f = active.to(errors.dtype)
    return (errors * active_f).sum(), active_f.sum()


def _run_epoch(model: Stage3TCN, loader: DataLoader, device: torch.device, optimizer=None, scaler=None, accumulation: int = 1, physical_weight: float = 1.0) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "physical_set_l2_m": 0.0, "observation_set_l2_m": 0.0}
    observation_error_sum = 0.0
    observation_error_count = 0.0
    count = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        tensors = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(tensors["obs"], tensors["obs_mask"], tensors["event_mask"], tensors["event_time_s"], tensors["tau"])
                loss, _ = stage3_observation_loss(
                    prediction, tensors["future_position"], tensors["future_normal"], tensors["motion_class"],
                    tensors["future_observation_position"], tensors["future_observation_mask"],
                    tensors["future_observation_frame_available"], tensors["future_observation_ambiguous"],
                    physical_weight=physical_weight,
                )
                scaled = loss / accumulation if training else loss
            if training:
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                if (batch_index + 1) % accumulation == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        batch_count = int(tensors["obs"].shape[0])
        totals["loss"] += float(loss.detach().cpu()) * batch_count
        totals["physical_set_l2_m"] += float(_position_set_l2(prediction["position_mean"].detach(), tensors["future_position"]).sum().cpu())
        observation_sum, observation_count = _paired_observation_l2(
            prediction, tensors["future_position"], tensors["future_observation_position"],
            tensors["future_observation_mask"], tensors["future_observation_frame_available"],
            tensors["future_observation_ambiguous"],
        )
        observation_error_sum += float(observation_sum.detach().cpu())
        observation_error_count += float(observation_count.detach().cpu())
        count += batch_count
    if training and len(loader) % accumulation:
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    metrics["observation_set_l2_m"] = observation_error_sum / max(observation_error_count, 1.0)
    return metrics


def train(args: argparse.Namespace) -> Path:
    _seed(args.seed)
    dataset = Path(args.dataset).resolve()
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "stage3-dataset-v4-observation" or not manifest.get("qualification_passed"):
        raise ValueError("observation training requires a qualified v4 dataset")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training output: {output}")
    train_ds = Stage3ShardDataset(dataset, "train", augment=not args.no_augment, seed=args.seed, shuffle=True)
    val_ds = Stage3ShardDataset(dataset, "validation", augment=False, seed=args.seed, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0, pin_memory=torch.cuda.is_available())
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = Stage3TCN(channels=args.channels, dropout=args.dropout, input_features=7, observation_heads=True).to(device)
    initialization = None
    if args.init_checkpoint:
        initialization = _load_initial(model, Path(args.init_checkpoint).resolve())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, args.epochs), eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output.mkdir(parents=True)
    provenance = {
        "schema_version": "stage3-observation-training-run-v1",
        "dataset": str(dataset),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "config": vars(args),
        "test_accessed": False,
        "training_source_sha256": _source_hashes(),
        "environment": {"python": __import__("sys").version, "numpy": np.__version__, "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available()},
        **_git_state(dataset),
        "initialization": initialization,
    }
    (output / "run_manifest.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    best_path = output / f"stage3-observation-seed{args.seed}-best.pt"
    best = float("inf")
    stale = 0
    history: list[dict[str, float | int]] = []
    started = time.monotonic()
    stop_reason = "epochs_completed"
    for epoch in range(args.epochs):
        warmup = epoch < args.warmup_epochs
        for name, parameter in model.named_parameters():
            parameter.requires_grad = warmup == ("observation_residual" in name or "visibility_logits" in name)
        if not warmup:
            for parameter in model.parameters():
                parameter.requires_grad = True
        train_ds.set_epoch(epoch)
        train_metrics = _run_epoch(model, train_loader, device, optimizer, scaler, args.accumulation, args.physical_weight)
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, device, physical_weight=args.physical_weight)
        scheduler.step()
        record = {"epoch": epoch + 1, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"validation_{k}": v for k, v in val_metrics.items()}, "lr": scheduler.get_last_lr()[0]}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        monitor = val_metrics["observation_set_l2_m"]
        if not np.isfinite(monitor) or not all(np.isfinite(value) for value in (*train_metrics.values(), *val_metrics.values())):
            raise FloatingPointError("observation training produced a non-finite metric")
        if monitor < best:
            best = monitor
            stale = 0
            torch.save({
                "model": model.state_dict(), "model_class": "Stage3TCN-observation-v1", "checkpoint_role": "best",
                "model_config": {"channels": args.channels, "dropout": args.dropout, "input_features": 7, "observation_heads": True},
                "config": vars(args), "epoch": epoch + 1, "monitor": "validation_observation_set_l2_m", "monitor_value": best,
                "seed": args.seed, "provenance": provenance,
            }, best_path)
        else:
            stale += 1
            if stale >= args.patience:
                stop_reason = "early_stopping"
                break
        if args.max_wall_minutes > 0 and time.monotonic() - started >= args.max_wall_minutes * 60.0:
            stop_reason = "wall_time_limit"
            break
    last_path = output / f"stage3-observation-seed{args.seed}-last.pt"
    torch.save({"model": model.state_dict(), "model_class": "Stage3TCN-observation-v1", "checkpoint_role": "last", "model_config": {"channels": args.channels, "dropout": args.dropout, "input_features": 7, "observation_heads": True}, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": len(history), "stop_reason": stop_reason, "seed": args.seed, "provenance": provenance}, last_path)
    (output / f"stage3-observation-seed{args.seed}-history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    final = {**provenance, "status": "complete", "stop_reason": stop_reason, "epochs_completed": len(history), "best_checkpoint": best_path.name, "best_checkpoint_sha256": _sha256(best_path), "last_checkpoint": last_path.name, "last_checkpoint_sha256": _sha256(last_path)}
    (output / "run_manifest.json").write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--max-wall-minutes", type=float, default=0.0)
    parser.add_argument("--device", default="")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--physical-weight", type=float, default=2.0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.accumulation < 1 or args.channels < 1 or args.lr <= 0 or args.warmup_epochs < 0 or args.physical_weight <= 0:
        parser.error("epochs, batch-size, accumulation, channels and lr must be positive")
    print(train(args))


if __name__ == "__main__":
    main()
