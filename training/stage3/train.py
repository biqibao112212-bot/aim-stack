"""Train the causal Stage-3 TCN in the Windows ``yolov8`` environment."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .losses import stage3_loss
from .model import Stage3TCN
from .shard_dataset import Stage3ShardDataset


class Stage3ArrayDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray, augment: bool, seed: int) -> None:
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source = int(self.indices[index])
        obs = self.arrays["obs"][source].copy()
        obs_mask = self.arrays["obs_mask"][source].copy()
        event_mask = self.arrays["event_mask"][source].copy()
        event_time_s = self.arrays["event_time_s"][source].copy()
        if self.augment:
            rng = np.random.default_rng(self.seed + self.epoch * 100003 + source)
            for t in range(obs_mask.shape[0]):
                for armor in range(obs_mask.shape[1]):
                    if obs_mask[t, armor] and rng.random() < 0.05:
                        obs_mask[t, armor] = False
                if event_mask[t] and rng.random() < 0.30:
                    length = int(rng.integers(2, 11))
                    event_mask[t : min(event_mask.shape[0], t + length)] = False
                    obs_mask[t : min(obs_mask.shape[0], t + length)] = False
        return {
            "obs": torch.from_numpy(obs),
            "obs_mask": torch.from_numpy(obs_mask),
            "event_mask": torch.from_numpy(event_mask),
            "event_time_s": torch.from_numpy(event_time_s),
            "tau": torch.from_numpy(self.arrays["tau"][source].astype(np.float32, copy=False)),
            "future_position": torch.from_numpy(self.arrays["future_position"][source]),
            "future_normal": torch.from_numpy(self.arrays["future_normal"][source]),
            "motion_class": torch.tensor(int(self.arrays["motion_class"][source]), dtype=torch.long),
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_split(dataset_dir: Path, split_name: str) -> np.ndarray:
    arrays = np.load(dataset_dir / "samples.npz", allow_pickle=False)
    session_ids = np.asarray(arrays["session_id"]).astype(str)
    splits = json.loads((dataset_dir / "splits.json").read_text(encoding="utf-8"))
    wanted = set(str(value) for value in splits[split_name])
    return np.flatnonzero(np.asarray([value in wanted for value in session_ids]))


def _load_selection(path: str, split: str, *, required: bool = False) -> list[str] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get(split)
    if values is None and not required:
        return None
    if not isinstance(values, list) or not values:
        raise ValueError(f"selection has no non-empty {split} session list")
    result = [str(value) for value in values]
    if len(result) != len(set(result)):
        raise ValueError(f"selection contains duplicate {split} sessions")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(dataset_dir: Path) -> dict[str, object]:
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
    dataset_manifest = dataset_dir / "dataset_manifest.json"
    return {
        "git_commit": commit,
        "worktree_dirty": dirty,
        "dataset_manifest_sha256": _file_sha256(dataset_manifest),
    }


def _training_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        name: _file_sha256(root / name)
        for name in ("train.py", "model.py", "losses.py", "shard_dataset.py")
    }


def _position_set_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    candidates = []
    for permutation in itertools.permutations(range(4)):
        index = torch.tensor(permutation, dtype=torch.long, device=prediction.device)
        aligned = prediction.index_select(2, index)
        candidates.append(torch.linalg.vector_norm(aligned - target, dim=-1).mean(dim=(1, 2)))
    return torch.stack(candidates, dim=1).min(dim=1).values


def _run_epoch(model: Stage3TCN, loader: DataLoader, device: torch.device, optimizer=None, scaler=None, accumulation=1, clip=1.0) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    position_total = 0.0
    count = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        tensors = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.set_grad_enabled(training):
            use_amp = device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction = model(
                    tensors["obs"], tensors["obs_mask"], tensors["event_mask"],
                    tensors["event_time_s"], tensors["tau"]
                )
                loss, _ = stage3_loss(prediction, tensors["future_position"], tensors["future_normal"], tensors["motion_class"])
                scaled_loss = loss / accumulation if training else loss
            if training:
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                if (batch_index + 1) % accumulation == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach().cpu()) * tensors["obs"].shape[0]
        position_total += float(_position_set_l2(
            prediction["position_mean"].detach(), tensors["future_position"]
        ).sum().cpu())
        count += tensors["obs"].shape[0]
    if training and len(loader) % accumulation:
        remainder = len(loader) % accumulation
        if scaler is not None:
            scaler.unscale_(optimizer)
        correction = accumulation / remainder
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {
        "loss": total / max(count, 1),
        "position_set_l2_m": position_total / max(count, 1),
    }


def train_one(args: argparse.Namespace) -> Path:
    _seed_everything(args.seed)
    dataset_dir = Path(args.dataset)
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    if manifest.get("schema_version") == "stage3-dataset-v3":
        if not bool(manifest.get("qualification_passed", False)):
            raise ValueError("formal v3 dataset qualification did not pass")
        selection_payload = (
            json.loads(Path(args.selection).read_text(encoding="utf-8")) if args.selection else {}
        )
        if args.selection:
            if selection_payload.get("dataset_manifest_sha256") != _file_sha256(manifest_path):
                raise ValueError("selection does not match dataset manifest")
            if selection_payload.get("test") != []:
                raise ValueError("training selection must keep test empty")
            purpose = str(selection_payload.get("purpose", ""))
            selected_train = [str(value) for value in selection_payload.get("train", ())]
            selected_validation = [str(value) for value in selection_payload.get("validation", ())]
            validation_source = str(selection_payload.get("validation_source_split", "validation"))
            expected = {"overfit1": (1, 1), "overfit4": (4, 4), "pilot24": (16, 8)}
            if purpose not in expected or (len(selected_train), len(selected_validation)) != expected[purpose]:
                raise ValueError("selection purpose/cardinality contract mismatch")
            if purpose.startswith("overfit"):
                if validation_source != "train" or selected_train != selected_validation:
                    raise ValueError("overfit validation must reuse the exact train sessions")
                if not args.no_augment or args.limit != 0:
                    raise ValueError("overfit runs require --no-augment and --limit 0")
            elif validation_source != "validation" or set(selected_train) & set(selected_validation):
                raise ValueError("pilot validation must be disjoint and sourced from validation")
            if purpose == "pilot24":
                expected_quotas = {
                    "train": {"stationary": 2, "linear": 4, "spin": 4, "linear_and_spin": 6},
                    "validation": {"stationary": 1, "linear": 2, "spin": 2, "linear_and_spin": 3},
                }
                if selection_payload.get("mode_quotas") != expected_quotas:
                    raise ValueError("pilot selection mode quota declaration mismatch")
                selector_path = Path(__file__).with_name("select_sessions.py")
                if selection_payload.get("selector_source_sha256") != _file_sha256(selector_path):
                    raise ValueError("pilot selection was not produced by the current selector")
                qualification_path = dataset_dir / str(manifest["qualification_report"])
                if _file_sha256(qualification_path) != manifest.get("artifact_sha256", {}).get("qualification_report"):
                    raise ValueError("qualification report hash mismatch")
                qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
                report_by_id = {str(item["session_id"]): item for item in qualification["sessions"]}
                for split_name, selected in (("train", selected_train), ("validation", selected_validation)):
                    counts: dict[str, int] = {}
                    for session_id in selected:
                        report = report_by_id.get(session_id)
                        if report is None or report.get("split") != split_name:
                            raise ValueError(f"pilot session is absent or in the wrong split: {session_id}")
                        mode = str(report["manifest"]["mode"])
                        counts[mode] = counts.get(mode, 0) + 1
                    if counts != expected_quotas[split_name]:
                        raise ValueError(f"pilot {split_name} mode quotas are not satisfied: {counts}")
        train_sessions = _load_selection(args.selection, "train", required=bool(args.selection))
        validation_sessions = _load_selection(args.selection, "validation")
        validation_split = str(selection_payload.get("validation_source_split", "validation"))
        if validation_split not in {"train", "validation"}:
            raise ValueError("validation_source_split must be train or validation")
        if args.selection and validation_sessions is None and validation_split == "train":
            validation_sessions = train_sessions
        train_ds = Stage3ShardDataset(
            dataset_dir, "train", augment=not args.no_augment, shuffle=True, seed=args.seed,
            session_ids=train_sessions, sample_limit=args.limit,
        )
        val_ds = Stage3ShardDataset(
            dataset_dir, validation_split, augment=False, shuffle=False, seed=args.seed,
            session_ids=validation_sessions,
            sample_limit=(max(1, args.limit // 3) if args.limit > 0 else 0),
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0, pin_memory=torch.cuda.is_available())
    elif args.allow_legacy_v1:
        arrays_loaded = np.load(dataset_dir / "samples.npz", allow_pickle=False)
        arrays = {key: arrays_loaded[key] for key in arrays_loaded.files}
        train_indices = _load_split(dataset_dir, "train")
        val_indices = _load_split(dataset_dir, "validation")
        if len(train_indices) == 0 or len(val_indices) == 0:
            raise ValueError("session split must contain non-empty train and validation sets")
        if args.limit > 0:
            train_indices = train_indices[: args.limit]
            val_indices = val_indices[: max(1, args.limit // 3)]
        train_ds = Stage3ArrayDataset(arrays, train_indices, augment=not args.no_augment, seed=args.seed)
        val_ds = Stage3ArrayDataset(arrays, val_indices, augment=False, seed=args.seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
        selection_payload = {}
    else:
        raise ValueError("training requires qualified stage3-dataset-v3")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = Stage3TCN(channels=args.channels, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, args.epochs), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = float("inf")
    best_loss = float("inf")
    stale = 0
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training output: {output}")
    output.mkdir(parents=True)
    provenance = {
        "schema_version": "stage3-training-run-v1",
        "dataset": str(dataset_dir.resolve()),
        "selection": (None if not args.selection else str(Path(args.selection).resolve())),
        "config": vars(args),
        **(_git_state(dataset_dir) if manifest.get("schema_version") == "stage3-dataset-v3" else {}),
        "test_accessed": False,
        "training_source_sha256": _training_source_hashes(),
        "selection_sha256": (None if not args.selection else _file_sha256(Path(args.selection))),
        "selection_payload": (None if not args.selection else selection_payload),
        "environment": {
            "python": __import__("sys").version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "normalization_sha256": (
            manifest.get("artifact_sha256", {}).get("normalization")
            if manifest.get("schema_version") == "stage3-dataset-v3" else None
        ),
    }
    normalization_payload = (
        json.loads((dataset_dir / str(manifest["normalization"])).read_text(encoding="utf-8"))
        if manifest.get("schema_version") == "stage3-dataset-v3" else None
    )
    (output / "run_manifest.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    best_path = output / f"stage3-seed{args.seed}-best.pt"
    history = []
    started = time.monotonic()
    stop_reason = "epochs_completed"
    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        train_metrics = _run_epoch(model, train_loader, device, optimizer, scaler, args.accumulation)
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, device)
        if not all(np.isfinite(value) for value in (*train_metrics.values(), *val_metrics.values())):
            raise FloatingPointError("training produced a non-finite loss or position monitor")
        scheduler.step()
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_position_set_l2_m": train_metrics["position_set_l2_m"],
            "validation_loss": val_metrics["loss"],
            "validation_position_set_l2_m": val_metrics["position_set_l2_m"],
            "lr": scheduler.get_last_lr()[0],
        })
        print(json.dumps(history[-1], sort_keys=True))
        monitor = val_metrics["position_set_l2_m"]
        if monitor < best or (monitor == best and val_metrics["loss"] < best_loss):
            best = monitor
            best_loss = val_metrics["loss"]
            stale = 0
            torch.save({
                "model": model.state_dict(),
                "model_class": "Stage3TCN",
                "checkpoint_role": "best",
                "model_config": {"channels": args.channels, "dropout": args.dropout},
                "config": vars(args),
                "epoch": epoch + 1,
                "monitor": "validation_position_set_l2_m",
                "monitor_value": best,
                "validation_loss": best_loss,
                "seed": args.seed,
                "provenance": provenance,
                "normalization": normalization_payload,
            }, best_path)
        else:
            stale += 1
            if stale >= args.patience:
                stop_reason = "early_stopping"
                break
        if args.max_wall_minutes > 0 and time.monotonic() - started >= args.max_wall_minutes * 60.0:
            stop_reason = "wall_time_limit"
            break
    last_path = output / f"stage3-seed{args.seed}-last.pt"
    torch.save({
        "model": model.state_dict(),
        "model_class": "Stage3TCN",
        "checkpoint_role": "last",
        "model_config": {"channels": args.channels, "dropout": args.dropout},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": len(history),
        "stop_reason": stop_reason,
        "seed": args.seed,
        "provenance": provenance,
        "normalization": normalization_payload,
    }, last_path)
    (output / f"stage3-seed{args.seed}-history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    final_provenance = {
        **provenance,
        "status": "complete",
        "stop_reason": stop_reason,
        "epochs_completed": len(history),
        "best_checkpoint": best_path.name,
        "best_checkpoint_sha256": _file_sha256(best_path),
        "last_checkpoint": last_path.name,
        "last_checkpoint_sha256": _file_sha256(last_path),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(final_provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--selection", default="")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--max-wall-minutes", type=float, default=0.0)
    parser.add_argument("--device", default="")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--allow-legacy-v1", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least one")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.accumulation < 1:
        parser.error("--accumulation must be positive")
    if args.channels < 1:
        parser.error("--channels must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    print(train_one(args))


if __name__ == "__main__":
    main()
