#!/usr/bin/env python3
"""Grouped evaluation of a learned runtime-only same-armor pair cost."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from trajectory_uv_association import (
    LearnedCausalAssociator,
    PairLogisticModel,
    build_pair_training_rows,
)


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_labeled_paths(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    seen = set()
    for value in values:
        label, separator, raw = value.partition("=")
        path = Path(raw).resolve()
        key = (label, path)
        if not separator or not label or not path.is_dir() or key in seen:
            raise ValueError(value)
        seen.add(key)
        result.append(key)
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_training_by_repeat(
    analyses: list[tuple[str, Path]], method_module
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], set[tuple[str, str]], dict[str, str]]:
    rows_by_repeat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    accepted_runs: set[tuple[str, str]] = set()
    source_hashes: dict[str, str] = {}
    for label, analysis in analyses:
        method_module.validate_analysis_source(label, analysis)
        path = analysis / "observed_points.jsonl"
        rows = read_jsonl(path)
        for row in rows:
            rows_by_repeat[int(row["repeat"])].append(row)
        source_hashes[f"{label}:{analysis.name}:observed_points.jsonl"] = sha256_file(path)
        truth_path = analysis / "truth_points.jsonl"
        for row in read_jsonl(truth_path):
            accepted_runs.add((label, str(row["run"])))
        source_hashes[f"{label}:{analysis.name}:truth_points.jsonl"] = sha256_file(truth_path)
        summary_path = analysis / "analysis_summary.json"
        source_hashes[f"{label}:{analysis.name}:analysis_summary.json"] = sha256_file(summary_path)
    result = {}
    for repeat, rows in sorted(rows_by_repeat.items()):
        result[repeat] = build_pair_training_rows(rows)
    return result, accepted_runs, source_hashes


def authorized_roots(analyses: list[tuple[str, Path]]) -> dict[str, set[Path]]:
    result: dict[str, set[Path]] = defaultdict(set)
    for label, analysis in analyses:
        summary = json.loads((analysis / "analysis_summary.json").read_text(encoding="utf-8"))
        result[label].update(Path(raw).resolve() for raw in summary["roots"])
    return result


def prepare_runs(
    roots: list[tuple[str, Path]],
    analyses: list[tuple[str, Path]],
    accepted_runs: set[tuple[str, str]],
    association_module,
    grid_module,
) -> list[dict[str, Any]]:
    allowed = authorized_roots(analyses)
    result = []
    seen = set()
    for label, root in roots:
        if root not in allowed.get(label, set()):
            raise ValueError(f"unauthorized raw root for {label}: {root}")
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if (label, run_dir.name) not in accepted_runs:
                continue
            if not (run_dir / "stage3_observations.jsonl").exists():
                continue
            stable = f"{label}:{run_dir.name}"
            if stable in seen:
                raise ValueError(f"duplicate run: {stable}")
            seen.add(stable)
            item = association_module.prepare_run(run_dir, root, grid_module)
            item.update({"motion": label, "run_name": run_dir.name, "stable_run": stable})
            result.append(item)
    return result


def evaluate_associator(prepared: dict[str, Any], associator, association_module) -> dict[str, Any]:
    rows = []
    latency = []
    for detections in prepared["events"]:
        runtime = [
            {key: value for key, value in detection.items() if key != "truth_slot"}
            for detection in detections
        ]
        truth_slots = [detection.get("truth_slot") for detection in detections]
        start = time.perf_counter_ns()
        assignments = associator.update(runtime)
        latency.append((time.perf_counter_ns() - start) / 1000.0)
        for (_detection, track_id), truth_slot in zip(assignments, truth_slots):
            rows.append({"track_id": track_id, "truth_slot": truth_slot})
    scored = [row for row in rows if row["truth_slot"] is not None]
    mapping, correct = association_module.best_slot_mapping(scored)
    associated = [row for row in scored if row["track_id"] is not None]
    purity_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for row in associated:
        purity_counts[int(row["track_id"])][int(row["truth_slot"])] += 1
    purities = [max(counts.values()) / sum(counts.values()) for counts in purity_counts.values()]
    return {
        "valid_detections": prepared["valid_detections"],
        "scored_detections": len(scored),
        "associated_detections": len(associated),
        "global_mapping_accuracy": correct / len(scored) if scored else float("nan"),
        "mean_track_purity": float(np.mean(purities)) if purities else float("nan"),
        "latency_p50_us": float(np.percentile(latency, 50)) if latency else float("nan"),
        "latency_p99_us": float(np.percentile(latency, 99)) if latency else float("nan"),
        "mapped_tracks": len(mapping),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--root", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-runs", type=int, default=120)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    analyses = parse_labeled_paths(args.analysis)
    roots = parse_labeled_paths(args.root)
    method_module = load_script(
        repo / "scripts" / "evaluate-trajectory-processing-methods.py", "method_eval"
    )
    association_module = load_script(
        repo / "scripts" / "evaluate-observation-association.py", "association_baseline"
    )
    grid_module = association_module.load_grid_analysis(repo)
    training, accepted_runs, source_hashes = load_training_by_repeat(analyses, method_module)
    prepared = prepare_runs(
        roots, analyses, accepted_runs, association_module, grid_module
    )
    if len(prepared) != args.expected_runs:
        raise ValueError(f"expected {args.expected_runs} runs, found {len(prepared)}")
    cache_path = output / "prepared_association_events.jsonl"
    with cache_path.open("w", encoding="utf-8") as handle:
        for item in prepared:
            cached = {
                key: value
                for key, value in item.items()
                if key in {"motion", "run_name", "stable_run", "scale", "distance_m", "repeat", "events", "valid_detections"}
            }
            handle.write(json.dumps(cached, separators=(",", ":")) + "\n")
    repeats = sorted(training)
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    result_rows = []
    variants = (("learned_birth_0p3", 0.3), ("learned_birth_0p5", 0.5), ("learned_birth_0p7", 0.7))
    for repeat in repeats:
        train_x = np.vstack([training[value][0] for value in repeats if value != repeat])
        train_y = np.concatenate([training[value][1] for value in repeats if value != repeat])
        test_runs = [item for item in prepared if int(item["repeat"]) == repeat]
        test_ids = sorted(str(item["stable_run"]) for item in test_runs)
        model = PairLogisticModel.fit(
            train_x,
            train_y,
            {
                "split": "repeat_holdout",
                "test_repeat": repeat,
                "test_run_hash": hashlib.sha256("\n".join(test_ids).encode("utf-8")).hexdigest(),
            },
        )
        model.save(model_dir / f"repeat-{repeat:02d}.json")
        for item in test_runs:
            baseline = association_module.CausalAssociator(25.0, False)
            metrics = evaluate_associator(item, baseline, association_module)
            result_rows.append(
                {
                    "variant": "baseline_cv_gate25",
                    "motion": item["motion"],
                    "run": item["stable_run"],
                    "distance_m": item["distance_m"],
                    "scale": item["scale"],
                    "repeat": item["repeat"],
                    **metrics,
                }
            )
            for name, birth_probability in variants:
                learned = LearnedCausalAssociator(
                    model,
                    birth_probability=birth_probability,
                    drop_probability=0.05,
                    geometry_gate_deg=25.0,
                )
                metrics = evaluate_associator(item, learned, association_module)
                result_rows.append(
                    {
                        "variant": name,
                        "motion": item["motion"],
                        "run": item["stable_run"],
                        "distance_m": item["distance_m"],
                        "scale": item["scale"],
                        "repeat": item["repeat"],
                        **metrics,
                    }
                )
    all_x = np.vstack([training[value][0] for value in repeats])
    all_y = np.concatenate([training[value][1] for value in repeats])
    final_model = PairLogisticModel.fit(
        all_x,
        all_y,
        {"split": "all_accepted_runs", "source_hashes": source_hashes},
    )
    final_model.save(output / "uv_pair_association_model.json")
    write_csv(output / "association_variant_runs.csv", result_rows)
    summary_rows = []
    for variant in sorted({row["variant"] for row in result_rows}):
        selected = [
            row
            for row in result_rows
            if row["variant"] == variant and int(row["scored_detections"]) > 0
        ]
        accuracy = np.asarray([float(row["global_mapping_accuracy"]) for row in selected])
        purity = np.asarray([float(row["mean_track_purity"]) for row in selected])
        summary_rows.append(
            {
                "variant": variant,
                "scored_runs": len(selected),
                "all_runs": len(result_rows) // 4,
                "global_mapping_accuracy_mean": float(np.mean(accuracy)),
                "global_mapping_accuracy_p05": float(np.percentile(accuracy, 5)),
                "global_mapping_accuracy_p50": float(np.percentile(accuracy, 50)),
                "global_mapping_accuracy_p95": float(np.percentile(accuracy, 95)),
                "mean_track_purity": float(np.mean(purity)),
                "associated_fraction": float(
                    sum(int(row["associated_detections"]) for row in selected)
                    / max(sum(int(row["scored_detections"]) for row in selected), 1)
                ),
                "latency_p99_us_mean": float(np.mean([float(row["latency_p99_us"]) for row in selected])),
            }
        )
    write_csv(output / "association_variant_summary.csv", summary_rows)
    best = max(summary_rows, key=lambda row: float(row["global_mapping_accuracy_mean"]))
    summary = {
        "schema_version": 1,
        "kind": "grouped_learned_observation_association",
        "runtime_input_contract": ["timestamp_s", "camera_tvec_m converted to u/v"],
        "truth_is_runtime_input": False,
        "training_contract": "truth supplies same/different pair labels; test repeats are complete-run held out",
        "runs": len(prepared),
        "runs_with_scored_detections": len(
            {row["run"] for row in result_rows if int(row["scored_detections"]) > 0}
        ),
        "best_variant": best,
        "source_hashes": source_hashes,
    }
    (output / "association_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    artifacts = {
        str(path.relative_to(output)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "retention_manifest.json"
    }
    (output / "retention_manifest.json").write_text(
        json.dumps(
            {
                "classification": "protected_derived_model_and_long_term_private_evidence",
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
