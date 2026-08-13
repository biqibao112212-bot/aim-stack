#!/usr/bin/env python3
"""Evaluate cyclic segment association variants on the accepted cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from trajectory_uv_association import CyclicSegmentAssociator


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
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    cache = args.cache.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluator = load_script(
        repo / "scripts" / "evaluate_learned_observation_association.py", "association_eval"
    )
    baseline_module = load_script(
        repo / "scripts" / "evaluate-observation-association.py", "association_baseline"
    )
    prepared = [
        json.loads(line)
        for line in cache.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(prepared) != 120:
        raise ValueError(f"expected 120 cached runs, found {len(prepared)}")
    rows = []
    variants = [
        (gate, timeout, descending)
        for gate in (0.5, 1.0, 2.0, 4.0)
        for timeout in (0.25, 0.5, 1.0, 2.0)
        for descending in (False, True)
    ]
    for item in prepared:
        baseline = baseline_module.CausalAssociator(25.0, False)
        metrics = evaluator.evaluate_associator(item, baseline, baseline_module)
        rows.append(
            {
                "variant": "baseline_cv_gate25",
                "gate_deg": "",
                "timeout_s": "",
                "descending_u": "",
                "run": item["stable_run"],
                **metrics,
            }
        )
        for gate, timeout, descending in variants:
            associator = CyclicSegmentAssociator(gate, timeout, birth_descending_u=descending)
            metrics = evaluator.evaluate_associator(item, associator, baseline_module)
            rows.append(
                {
                    "variant": f"cyclic_g{gate:g}_t{timeout:g}_{'desc' if descending else 'asc'}",
                    "gate_deg": gate,
                    "timeout_s": timeout,
                    "descending_u": descending,
                    "run": item["stable_run"],
                    **metrics,
                }
            )
    write_csv(output / "cyclic_association_runs.csv", rows)
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
                "mean_track_purity": float(np.mean([float(row["mean_track_purity"]) for row in selected])),
                "latency_p99_us_mean": float(np.mean([float(row["latency_p99_us"]) for row in selected])),
            }
        )
    write_csv(output / "cyclic_association_summary.csv", summary_rows)
    best = max(summary_rows, key=lambda row: float(row["global_mapping_accuracy_mean"]))
    summary = {
        "schema_version": 1,
        "kind": "cyclic_segment_association_sweep",
        "truth_is_runtime_input": False,
        "cache_runs": len(prepared),
        "cache_sha256": sha256_file(cache),
        "best_variant": best,
    }
    (output / "cyclic_sweep_summary.json").write_text(
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
                "source_cache": {"path": str(cache), "sha256": sha256_file(cache)},
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
