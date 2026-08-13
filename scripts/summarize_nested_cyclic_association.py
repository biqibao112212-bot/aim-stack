#!/usr/bin/env python3
"""Nested-repeat selection and paired bootstrap for cyclic association."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


RUN_PATTERN = re.compile(r"^(?P<motion>[^:]+):r(?P<scale>\d+p\d+)-d(?P<distance>\d+p\d+)-rep(?P<repeat>\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_run(run: str) -> dict:
    match = RUN_PATTERN.match(run)
    if not match:
        raise ValueError(run)
    values = match.groupdict()
    return {
        "motion": values["motion"],
        "scale": float(values["scale"].replace("p", ".")),
        "distance_m": float(values["distance"].replace("p", ".")),
        "repeat": int(values["repeat"]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_bootstrap(rows: list[dict], samples: int = 10000) -> dict:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        key = (row["motion"], row["scale"], row["distance_m"])
        grouped[key].append(float(row["cyclic_accuracy"]) - float(row["baseline_accuracy"]))
    keys = sorted(grouped)
    rng = np.random.RandomState(20260809)
    draws = []
    for _ in range(samples):
        selected_keys = rng.choice(len(keys), len(keys), replace=True)
        condition_values = []
        for key_index in selected_keys:
            values = grouped[keys[int(key_index)]]
            selected_runs = rng.choice(values, len(values), replace=True)
            condition_values.append(float(np.mean(selected_runs)))
        draws.append(float(np.mean(condition_values)))
    observed = float(np.mean([np.mean(grouped[key]) for key in keys]))
    return {
        "condition_equal_improvement_fraction": observed,
        "bootstrap_ci95_low": float(np.percentile(draws, 2.5)),
        "bootstrap_ci95_high": float(np.percentile(draws, 97.5)),
        "conditions": len(keys),
        "paired_runs": len(rows),
        "bootstrap_samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    runs_path = args.runs.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with runs_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update(parse_run(row["run"]))
    variants = sorted({row["variant"] for row in rows if row["variant"].startswith("cyclic_")})
    repeats = sorted({int(row["repeat"]) for row in rows})
    selections = []
    selected_rows = []
    for repeat in repeats:
        training = [row for row in rows if int(row["repeat"]) != repeat]
        scores = {
            variant: float(
                np.mean(
                    [
                        float(row["global_mapping_accuracy"])
                        for row in training
                        if row["variant"] == variant and int(row["scored_detections"]) > 0
                    ]
                )
            )
            for variant in variants
        }
        selected_variant = max(scores, key=scores.get)
        selections.append(
            {
                "test_repeat": repeat,
                "selected_variant": selected_variant,
                "training_accuracy_mean": scores[selected_variant],
            }
        )
        cyclic_test = {
            row["run"]: row
            for row in rows
            if int(row["repeat"]) == repeat
            and row["variant"] == selected_variant
            and int(row["scored_detections"]) > 0
        }
        baseline_test = {
            row["run"]: row
            for row in rows
            if int(row["repeat"]) == repeat
            and row["variant"] == "baseline_cv_gate25"
            and int(row["scored_detections"]) > 0
        }
        if set(cyclic_test) != set(baseline_test):
            raise ValueError(f"paired run mismatch for repeat {repeat}")
        for run in sorted(cyclic_test):
            cyclic = cyclic_test[run]
            baseline = baseline_test[run]
            selected_rows.append(
                {
                    "run": run,
                    "motion": cyclic["motion"],
                    "scale": cyclic["scale"],
                    "distance_m": cyclic["distance_m"],
                    "repeat": repeat,
                    "selected_variant": selected_variant,
                    "cyclic_accuracy": float(cyclic["global_mapping_accuracy"]),
                    "baseline_accuracy": float(baseline["global_mapping_accuracy"]),
                    "cyclic_associated_fraction": int(cyclic["associated_detections"]) / max(int(cyclic["scored_detections"]), 1),
                    "baseline_associated_fraction": int(baseline["associated_detections"]) / max(int(baseline["scored_detections"]), 1),
                    "cyclic_track_purity": float(cyclic["mean_track_purity"]),
                    "baseline_track_purity": float(baseline["mean_track_purity"]),
                    "cyclic_latency_p99_us": float(cyclic["latency_p99_us"]),
                    "baseline_latency_p99_us": float(baseline["latency_p99_us"]),
                }
            )
    write_csv(output / "nested_fold_selections.csv", selections)
    write_csv(output / "nested_paired_runs.csv", selected_rows)
    bootstrap = paired_bootstrap(selected_rows)
    summary = {
        "schema_version": 1,
        "kind": "nested_repeat_cyclic_association_selection",
        "truth_is_runtime_input": False,
        "parameter_selection_contract": "each test repeat uses the variant selected only on the other four repeats",
        "scored_runs": len(selected_rows),
        "cyclic_accuracy_mean": float(np.mean([row["cyclic_accuracy"] for row in selected_rows])),
        "baseline_accuracy_mean": float(np.mean([row["baseline_accuracy"] for row in selected_rows])),
        "cyclic_associated_fraction": float(np.mean([row["cyclic_associated_fraction"] for row in selected_rows])),
        "baseline_associated_fraction": float(np.mean([row["baseline_associated_fraction"] for row in selected_rows])),
        "cyclic_track_purity_mean": float(np.mean([row["cyclic_track_purity"] for row in selected_rows])),
        "baseline_track_purity_mean": float(np.mean([row["baseline_track_purity"] for row in selected_rows])),
        "cyclic_latency_p99_us_mean": float(np.mean([row["cyclic_latency_p99_us"] for row in selected_rows])),
        "baseline_latency_p99_us_mean": float(np.mean([row["baseline_latency_p99_us"] for row in selected_rows])),
        "paired_bootstrap": bootstrap,
        "fold_selections": selections,
        "source_runs_sha256": sha256_file(runs_path),
    }
    (output / "nested_cyclic_summary.json").write_text(
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
