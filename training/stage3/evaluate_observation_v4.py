"""Evaluate the v4 future-observation model on exact validation shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .model import Stage3TCN
from .shard_dataset import _validate_shard_arrays


PERMUTATIONS = tuple(itertools.permutations(range(4)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    a = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean_m": float(a.mean()), "median_m": float(np.quantile(a, .5)), "p90_m": float(np.quantile(a, .9)), "p95_m": float(np.quantile(a, .95)), "p99_m": float(np.quantile(a, .99))}


def _align(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = []
    per_query = []
    aligned = []
    for permutation in PERMUTATIONS:
        index = list(permutation)
        value = prediction[:, :, index, :]
        aligned.append(value)
        errors = np.linalg.norm(value - target, axis=-1).mean(axis=2)
        per_query.append(errors)
        scores.append(errors.mean(axis=1))
    scores_array = np.stack(scores, axis=1)
    best = scores_array.argmin(axis=1)
    selected = np.stack(aligned, axis=1)[np.arange(len(prediction)), best]
    per_query_array = np.stack(per_query, axis=1)
    return selected, per_query_array[np.arange(len(prediction)), best], best


def evaluate(args: argparse.Namespace) -> Path:
    dataset = Path(args.dataset).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "stage3-dataset-v4-observation" or not manifest.get("qualification_passed"):
        raise ValueError("evaluation requires qualified stage3-dataset-v4-observation")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    provenance = checkpoint.get("provenance", {})
    if provenance.get("dataset_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("checkpoint does not match v4 dataset manifest")
    config = checkpoint.get("model_config", {})
    model = Stage3TCN(channels=int(config.get("channels", 64)), dropout=float(config.get("dropout", .1)), input_features=int(config.get("input_features", 7)), observation_heads=bool(config.get("observation_heads", True)))
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    normalization = json.loads((dataset / str(manifest["normalization"])).read_text(encoding="utf-8"))
    obs_mean = np.asarray(normalization["obs_xyz"]["mean"], dtype=np.float32)
    obs_std = np.asarray(normalization["obs_xyz"]["std"], dtype=np.float32)
    quality_mean = np.asarray(normalization.get("obs_quality", {}).get("mean", [0.0, 0.0]), dtype=np.float32)
    quality_std = np.asarray(normalization.get("obs_quality", {}).get("std", [1.0, 1.0]), dtype=np.float32)
    values: dict[str, list[float]] = defaultdict(list)
    visibility_true: list[float] = []
    visibility_pred: list[float] = []
    coverage = defaultdict(int)
    shards = [item for item in manifest["shards"] if item["split"] == args.split]
    sample_count = 0
    query_count = 0
    for shard in shards:
        shard_path = dataset / str(shard["path"])
        if _sha256(shard_path) != str(shard["sha256"]):
            raise ValueError(f"shard hash mismatch: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        _validate_shard_arrays(arrays, int(shard["sample_count"]))
        for start in range(0, len(arrays["session_id"]), args.batch_size):
            indices = np.arange(start, min(start + args.batch_size, len(arrays["session_id"])))
            obs = arrays["obs"][indices].astype(np.float32, copy=True)
            obs[..., :3] = (obs[..., :3] - obs_mean) / obs_std
            if obs.shape[-1] >= 7:
                obs[..., 5:7] = (obs[..., 5:7] - quality_mean) / quality_std
            tensors = {
                "obs": torch.from_numpy(obs).to(device),
                "obs_mask": torch.from_numpy(arrays["obs_mask"][indices]).to(device),
                "event_mask": torch.from_numpy(arrays["event_mask"][indices]).to(device),
                "event_time_s": torch.from_numpy(arrays["event_time_s"][indices]).to(device),
                "tau": torch.from_numpy(arrays["tau"][indices].astype(np.float32)).to(device),
            }
            with torch.no_grad():
                prediction = model(**tensors)
            physical = prediction["position_mean"].cpu().numpy()
            observed_prediction = (prediction["position_mean"] + prediction["observation_residual_mean"]).cpu().numpy()
            target_physical = arrays["future_position"][indices].astype(np.float64)
            target_observation = arrays["future_observation_position"][indices].astype(np.float64)
            target_mask = arrays["future_observation_mask"][indices]
            frame_available = arrays["future_observation_frame_available"][indices]
            ambiguous = arrays["future_observation_ambiguous"][indices]
            aligned_physical, physical_errors, best_permutation = _align(physical.astype(np.float64), target_physical)
            observed_candidates = np.stack([observed_prediction[:, :, list(permutation), :] for permutation in PERMUTATIONS], axis=1)
            aligned_observation = observed_candidates[np.arange(len(indices)), best_permutation]
            del aligned_physical
            active = frame_available & ~ambiguous
            observation_active = target_mask & active[..., None]
            observation_error = np.linalg.norm(aligned_observation - target_observation, axis=-1)
            future_observation_error = np.linalg.norm(target_observation - target_physical, axis=-1)
            for local in range(len(indices)):
                for query in range(target_physical.shape[1]):
                    query_count += 1
                    values["prediction_vs_truth"].append(float(physical_errors[local, query]))
                    if not frame_available[local, query]:
                        coverage["missing_exact_frame"] += 1
                    elif ambiguous[local, query]:
                        coverage["ambiguous"] += 1
                    else:
                        coverage["exact_frame"] += 1
                    mask = observation_active[local, query]
                    if not np.any(mask):
                        continue
                    values["prediction_vs_future_observation"].append(float(observation_error[local, query][mask].mean()))
                    values["future_observation_vs_truth"].append(float(future_observation_error[local, query][mask].mean()))
                    coverage["usable_observation_query"] += 1
            visible_logits = prediction["visibility_logits"].sigmoid().cpu().numpy()
            visibility_true.extend(target_mask[active].astype(np.float32).reshape(-1).tolist())
            visibility_pred.extend((visible_logits[active] >= .5).astype(np.float32).reshape(-1).tolist())
            sample_count += len(indices)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "stage3-observation-evaluation-v1",
        "split": args.split,
        "sample_count": sample_count,
        "query_count": query_count,
        "metrics": {name: _summary(values[name]) for name in ("prediction_vs_truth", "future_observation_vs_truth", "prediction_vs_future_observation")},
        "coverage": dict(coverage),
        "visibility": {
            "count": len(visibility_true),
            "accuracy": float(np.mean(np.asarray(visibility_true) == np.asarray(visibility_pred))) if visibility_true else None,
            "positive_rate_true": float(np.mean(visibility_true)) if visibility_true else None,
            "positive_rate_pred": float(np.mean(visibility_pred)) if visibility_pred else None,
        },
        "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path), "provenance": provenance},
        "dataset_manifest_sha256": _sha256(manifest_path),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=256)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
