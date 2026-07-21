"""Evaluate a scratch direct-observation checkpoint on qualified v4 shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import Stage3TCN
from .shard_dataset import Stage3ShardDataset
from .train_scratch_ab import _summary, _validation_values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(args: argparse.Namespace) -> Path:
    if args.split == "test" and not args.allow_test:
        raise ValueError("test evaluation requires explicit --allow-test")
    dataset = Path(args.dataset).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "stage3-dataset-v4-observation"
        or not manifest.get("qualification_passed")
    ):
        raise ValueError("evaluation requires a qualified v4 observation dataset")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    provenance = checkpoint.get("provenance", {})
    if provenance.get("dataset_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("checkpoint and dataset manifest do not match")
    config = checkpoint.get("model_config", {})
    if not config.get("direct_observation_heads"):
        raise ValueError("checkpoint has no direct observation head")
    model = Stage3TCN(**config)
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    shard_dataset = Stage3ShardDataset(
        dataset, args.split, augment=False, seed=int(checkpoint.get("seed", 0)),
        shuffle=False,
    )
    loader = DataLoader(
        shard_dataset, batch_size=args.batch_size, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    observation_parts: list[np.ndarray] = []
    physical_parts: list[np.ndarray] = []
    coverage = {
        "query_count": 0,
        "exact_frame": 0,
        "missing_exact_frame": 0,
        "ambiguous": 0,
        "usable_observation_query": 0,
    }
    with torch.no_grad():
        for raw in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in raw.items()}
            with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                prediction = model(
                    batch["obs"], batch["obs_mask"], batch["event_mask"],
                    batch["event_time_s"], batch["tau"],
                )
            observation, physical = _validation_values(prediction, batch)
            observation_parts.append(observation)
            physical_parts.append(physical)
            available = batch["future_observation_frame_available"]
            ambiguous = batch["future_observation_ambiguous"]
            active = available & ~ambiguous
            usable = active & batch["future_observation_mask"].any(dim=-1)
            coverage["query_count"] += int(available.numel())
            coverage["missing_exact_frame"] += int((~available).sum().cpu())
            coverage["ambiguous"] += int((available & ambiguous).sum().cpu())
            coverage["exact_frame"] += int(active.sum().cpu())
            coverage["usable_observation_query"] += int(usable.sum().cpu())
    report = {
        "schema_version": "stage3-scratch-observation-evaluation-v1",
        "split": args.split,
        "objective": checkpoint.get("objective"),
        "physical_aux_weight": checkpoint.get("physical_aux_weight"),
        "sample_count": len(shard_dataset),
        "coverage": coverage,
        "metrics": {
            "prediction_vs_future_observation": _summary(observation_parts),
            "physical_head_vs_truth": _summary(physical_parts),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "provenance": provenance,
        },
        "dataset_manifest_sha256": _sha256(manifest_path),
        "test_accessed": args.split == "test",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    print(evaluate(args))


if __name__ == "__main__":
    main()
