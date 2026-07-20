"""Convert raw Stage-3 JSONL streams into session-disjoint tensor shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .dataset import build_samples, samples_to_arrays
from .schema import schema_fingerprint


def _split_sessions(session_ids: list[str], seed: int) -> dict[str, list[str]]:
    unique = sorted(set(session_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_train = int(round(n * 0.60))
    n_val = int(round(n * 0.20))
    return {"train": sorted(unique[:n_train]), "validation": sorted(unique[n_train:n_train + n_val]), "test": sorted(unique[n_train + n_val:])}


def _subset(arrays: dict[str, np.ndarray], session_ids: set[str]) -> dict[str, np.ndarray]:
    mask = np.asarray([str(value) in session_ids for value in arrays["session_id"]])
    return {key: value[mask] if value.ndim > 0 and value.shape[0] == mask.shape[0] else value for key, value in arrays.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_paths = [Path(value) for value in args.manifest]
    manifests = {}
    if manifest_paths:
        from .dataset import load_manifests
        manifests = load_manifests(manifest_paths)
    samples = build_samples(args.observations, args.truth, manifests, augment=False, seed=args.seed)
    arrays = samples_to_arrays(samples)
    splits = _split_sessions([str(value) for value in arrays["session_id"]], args.seed)
    if any(len(ids) == 0 for ids in splits.values()):
        raise ValueError("at least three sessions are required for non-empty train/validation/test splits")
    npz_path = output / "samples.npz"
    np.savez_compressed(npz_path, **arrays)
    (output / "splits.json").write_text(json.dumps(splits, indent=2, sort_keys=True), encoding="utf-8")
    dataset_manifest = {
        "schema_version": "stage3-dataset-v1",
        "observation_schema": "stage3-observation-v1",
        "truth_schema": "stage3-truth-v1",
        "schema_fingerprint": schema_fingerprint(),
        "seed": args.seed,
        "sample_count": len(samples),
        "session_count": len(set(str(value) for value in arrays["session_id"])),
        "shape": {key: list(value.shape) for key, value in arrays.items() if hasattr(value, "shape")},
        "source_observations": str(Path(args.observations).resolve()),
        "source_truth": str(Path(args.truth).resolve()),
        "samples_sha256": _sha256(npz_path),
        "split_counts": {name: len(ids) for name, ids in splits.items()},
    }
    (output / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(dataset_manifest, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    convert(parser.parse_args())


if __name__ == "__main__":
    main()
