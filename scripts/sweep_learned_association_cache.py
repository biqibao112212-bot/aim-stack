#!/usr/bin/env python3
"""Fast threshold sweep over a hash-bound prepared association cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from trajectory_uv_association import LearnedCausalAssociator, PairLogisticModel


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluator = load_script(
        repo / "scripts" / "evaluate_learned_observation_association.py",
        "learned_association_evaluator",
    )
    baseline_module = load_script(
        repo / "scripts" / "evaluate-observation-association.py",
        "association_baseline",
    )
    cache_path = source / "prepared_association_events.jsonl"
    prepared = [
        json.loads(line)
        for line in cache_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(prepared) != 120:
        raise ValueError(f"cache must account for 120 runs, found {len(prepared)}")
    rows = []
    variants = [
        (birth, drop)
        for birth in (0.1, 0.3, 0.5, 0.7)
        for drop in (1e-6, 1e-3, 1e-2)
        if drop < birth
    ]
    for item in prepared:
        model = PairLogisticModel.load(
            source / "models" / f"repeat-{int(item['repeat']):02d}.json"
        )
        baseline = baseline_module.CausalAssociator(25.0, False)
        metrics = evaluator.evaluate_associator(item, baseline, baseline_module)
        rows.append({"variant": "baseline_cv_gate25", "birth_probability": "", "drop_probability": "", "run": item["stable_run"], **metrics})
        for birth, drop in variants:
            associator = LearnedCausalAssociator(
                model,
                birth_probability=birth,
                drop_probability=drop,
                geometry_gate_deg=25.0,
            )
            metrics = evaluator.evaluate_associator(item, associator, baseline_module)
            rows.append(
                {
                    "variant": f"learned_b{birth:g}_d{drop:g}",
                    "birth_probability": birth,
                    "drop_probability": drop,
                    "run": item["stable_run"],
                    **metrics,
                }
            )
    write_csv(output / "association_threshold_runs.csv", rows)
    summary_rows = []
    for variant in sorted({row["variant"] for row in rows}):
        selected = [
            row for row in rows if row["variant"] == variant and int(row["scored_detections"]) > 0
        ]
        accuracy = np.asarray([float(row["global_mapping_accuracy"]) for row in selected])
        summary_rows.append(
            {
                "variant": variant,
                "scored_runs": len(selected),
                "global_mapping_accuracy_mean": float(np.mean(accuracy)),
                "global_mapping_accuracy_p05": float(np.percentile(accuracy, 5)),
                "global_mapping_accuracy_p50": float(np.percentile(accuracy, 50)),
                "global_mapping_accuracy_p95": float(np.percentile(accuracy, 95)),
                "associated_fraction": float(
                    sum(int(row["associated_detections"]) for row in selected)
                    / max(sum(int(row["scored_detections"]) for row in selected), 1)
                ),
                "mean_track_purity": float(
                    np.mean([float(row["mean_track_purity"]) for row in selected])
                ),
                "latency_p99_us_mean": float(
                    np.mean([float(row["latency_p99_us"]) for row in selected])
                ),
            }
        )
    write_csv(output / "association_threshold_summary.csv", summary_rows)
    best = max(summary_rows, key=lambda row: float(row["global_mapping_accuracy_mean"]))
    summary = {
        "schema_version": 1,
        "kind": "learned_pair_association_threshold_sweep",
        "truth_is_runtime_input": False,
        "cache_runs": len(prepared),
        "cache_sha256": sha256_file(cache_path),
        "best_variant": best,
    }
    (output / "threshold_sweep_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "retention_manifest.json"
    }
    (output / "retention_manifest.json").write_text(
        json.dumps(
            {
                "classification": "long_term_private_evidence",
                "deletion_allowed": False,
                "source_cache": {"path": str(cache_path), "sha256": sha256_file(cache_path)},
                "artifacts": artifacts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
