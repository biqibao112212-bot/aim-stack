"""CUDA-only exploratory trainer for the pre-registered V19 sparse/dense branches."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.corner_pnp.data import sha256, write_json_new

from .data import ArmorPoseDataset, load_development_pack, move_tensor_tree
from .dense_correspondence_head import build_dense_correspondence_net
from .losses import LossOutput, dense_loss, sparse_loss
from .sparse_prob_head import ProbabilisticCornerNet


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _mean_epoch(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                loss_function: Callable[[torch.nn.Module, dict[str, object]], LossOutput],
                optimizer: torch.optim.Optimizer | None) -> dict[str, float]:
    model.train(optimizer is not None)
    totals: dict[str, float] = {}
    samples = 0
    for batch in loader:
        moved = move_tensor_tree(batch, device)
        assert isinstance(moved, dict)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            output = loss_function(model, moved)
            if not torch.isfinite(output.total):
                raise FloatingPointError("non-finite armor-pose loss")
            if optimizer is not None:
                output.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        count = int(moved["online"]["patch_rgb"].shape[0])
        samples += count
        values = {"loss": output.total.detach()} | {key: value.detach() for key, value in output.diagnostics.items()}
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + count * float(value)
    return {key: value / samples for key, value in totals.items()} | {"samples": samples}


def train(*, plan_path: Path, train_pack_path: Path, validation_pack_path: Path,
          output_dir: Path, branch: str, epochs: int, batch_size: int, learning_rate: float,
          seed: int, maximum_samples: int | None = None, dense_warmup_epochs: int = 2,
          initial_checkpoint: Path | None = None,
          dense_architecture: str | None = None) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("armor-pose training requires CUDA and refuses CPU fallback")
    if branch not in {"sparse", "dense"}:
        raise ValueError("branch must be sparse or dense")
    plan_path = plan_path.resolve(strict=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "aim-stack.armor-pose-experiment-plan/1":
        raise ValueError("unsupported armor-pose plan")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite protected run: {output_dir}")
    train_pack = load_development_pack(train_pack_path, expected_split="train")
    validation_pack = load_development_pack(
        validation_pack_path, expected_split="validation",
        feature_mean=train_pack.feature_mean, feature_std=train_pack.feature_std,
    )
    if set(train_pack.values["session_id"]) & set(validation_pack.values["session_id"]):
        raise PermissionError("whole-session train/validation overlap")
    _seed(seed)
    device = torch.device("cuda")
    initial_checkpoint_hash: str | None = None
    loaded: dict[str, object] | None = None
    if initial_checkpoint is not None:
        initial_checkpoint = initial_checkpoint.resolve(strict=True)
        loaded = torch.load(initial_checkpoint, map_location=device, weights_only=False)
        if loaded.get("schema_version") != "aim-stack.armor-pose-checkpoint/1" or loaded.get("branch") != branch:
            raise ValueError("initial checkpoint branch/schema mismatch")
    if branch == "sparse":
        model: torch.nn.Module = ProbabilisticCornerNet()
    elif loaded is not None:
        model_config = loaded.get("model")
        if not isinstance(model_config, dict):
            raise ValueError("dense checkpoint omitted its reconstructible model config")
        model = build_dense_correspondence_net(model_config=model_config)
        if dense_architecture is not None and model.config.get("architecture") != dense_architecture:
            raise ValueError("requested dense architecture disagrees with initial checkpoint")
    else:
        if dense_architecture is None:
            raise ValueError("new dense training requires an explicit dense architecture")
        model = build_dense_correspondence_net(architecture=dense_architecture)
    model = model.to(device)
    if loaded is not None:
        model.load_state_dict(loaded["state_dict"], strict=True)
        assert initial_checkpoint is not None
        initial_checkpoint_hash = sha256(initial_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    train_loader = DataLoader(ArmorPoseDataset(train_pack, maximum_samples=maximum_samples),
                              batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              generator=torch.Generator().manual_seed(seed))
    validation_loader = DataLoader(ArmorPoseDataset(validation_pack, maximum_samples=maximum_samples),
                                   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    output_dir.mkdir(parents=True)
    (output_dir / "checkpoints").mkdir()
    write_json_new(output_dir / "run-manifest.json", {
        "schema_version": "aim-stack.armor-pose-training-run/1",
        "branch": branch, "plan": str(plan_path), "plan_sha256": sha256(plan_path),
        "train_pack": str(train_pack_path.resolve()), "train_pack_sha256": sha256(train_pack_path.resolve()),
        "validation_pack": str(validation_pack_path.resolve()), "validation_pack_sha256": sha256(validation_pack_path.resolve()),
        "train_sessions": sorted(map(str, set(train_pack.values["session_id"]))),
        "validation_sessions": sorted(map(str, set(validation_pack.values["session_id"]))),
        "exploratory_only": True, "test_accessed": False, "online_truth_input": False,
        "device": "cuda", "gpu": torch.cuda.get_device_name(0), "cpu_fallback": False,
        "seed": seed, "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "maximum_samples": maximum_samples, "model": model.config,
        "dense_architecture": model.config.get("architecture") if branch == "dense" else None,
        "dense_warmup_epochs": dense_warmup_epochs if branch == "dense" else None,
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint is not None else None,
        "initial_checkpoint_sha256": initial_checkpoint_hash,
    })
    history: list[dict[str, object]] = []
    best = float("inf")
    selected: Path | None = None
    for epoch in range(1, epochs + 1):
        if branch == "sparse":
            training_loss_function = sparse_loss
            validation_loss_function = sparse_loss
        else:
            pose_weight = 0.0 if epoch <= dense_warmup_epochs else 1.0
            training_loss_function = lambda current, batch, weight=pose_weight: dense_loss(
                current, batch, correspondence_count=64, pose_weight=weight
            )
            validation_loss_function = lambda current, batch, weight=pose_weight: dense_loss(
                current, batch, correspondence_count=64, pose_weight=weight
            )
        training = _mean_epoch(model, train_loader, device, training_loss_function, optimizer)
        validation = _mean_epoch(model, validation_loader, device, validation_loss_function, None)
        history.append({"epoch": epoch, "train": training, "validation": validation})
        if validation["loss"] < best:
            best = validation["loss"]
            selected = output_dir / "checkpoints" / f"best-epoch-{epoch:03d}.pt"
            torch.save({
                "schema_version": "aim-stack.armor-pose-checkpoint/1", "branch": branch,
                "state_dict": model.state_dict(), "model": model.config,
                "feature_mean": train_pack.feature_mean, "feature_std": train_pack.feature_std,
                "plan_sha256": sha256(plan_path), "epoch": epoch, "validation": validation,
                "online_truth_input": False, "test_accessed": False, "device": "cuda",
            }, selected)
    if selected is None:
        raise RuntimeError("no checkpoint selected")
    result: dict[str, object] = {
        "schema_version": "aim-stack.armor-pose-training-result/1", "branch": branch,
        "selected_checkpoint": str(selected), "selected_checkpoint_sha256": sha256(selected),
        "best_validation_loss": best, "history": history, "exploratory_only": True,
        "test_accessed": False, "test_used_for_selection": False,
    }
    write_json_new(output_dir / "training-result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--train-pack", type=Path, required=True)
    parser.add_argument("--validation-pack", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branch", choices=("sparse", "dense"), required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--maximum-samples", type=int)
    parser.add_argument("--dense-warmup-epochs", type=int, default=2)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--dense-architecture", choices=("legacy_global_average_v1", "spatial_projective_v2"))
    args = parser.parse_args()
    print(json.dumps(train(plan_path=args.plan, train_pack_path=args.train_pack,
                           validation_pack_path=args.validation_pack, output_dir=args.output_dir,
                           branch=args.branch, epochs=args.epochs, batch_size=args.batch_size,
                           learning_rate=args.learning_rate, seed=args.seed,
                           maximum_samples=args.maximum_samples,
                           dense_warmup_epochs=args.dense_warmup_epochs,
                           initial_checkpoint=args.initial_checkpoint,
                           dense_architecture=args.dense_architecture), indent=2))


if __name__ == "__main__":
    main()
