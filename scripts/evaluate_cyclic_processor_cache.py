#!/usr/bin/env python3
"""End-to-end cyclic association plus safe CV/Ridge replay from cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

from trajectory_uv_association import CyclicSegmentAssociator
from trajectory_uv_processor import UvRidgeModel


VARIANT_PATTERN = re.compile(
    r"^cyclic_g(?P<gate>[\d.]+)_t(?P<timeout>[\d.]+)_(?P<direction>asc|desc)$"
)


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
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "mean": float(np.mean(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--analysis", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--selections", required=True, type=Path)
    parser.add_argument("--processor-models", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    replay = load_script(
        repo / "scripts" / "evaluate_trajectory_processor_replay.py", "processor_replay"
    )
    method_module = load_script(
        repo / "scripts" / "evaluate-trajectory-processing-methods.py", "method_eval"
    )
    association_module = load_script(
        repo / "scripts" / "evaluate-observation-association.py", "association_baseline"
    )
    analyses = replay.parse_labeled_paths(args.analysis)
    _examples, truth, source_hashes = replay.load_examples(analyses, method_module)
    cache_path = args.cache.resolve()
    prepared = [
        json.loads(line)
        for line in cache_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(prepared) != 120:
        raise ValueError(f"expected 120 runs, found {len(prepared)}")
    with args.selections.resolve().open("r", encoding="utf-8") as handle:
        selections = {
            int(row["test_repeat"]): row["selected_variant"]
            for row in csv.DictReader(handle)
        }
    predictions = []
    run_rows = []
    latency_rows = []
    for item in prepared:
        repeat = int(item["repeat"])
        variant = selections[repeat]
        match = VARIANT_PATTERN.match(variant)
        if not match:
            raise ValueError(variant)
        model = UvRidgeModel.load(
            args.processor_models.resolve() / f"repeat-{repeat:02d}.json"
        )
        if model.config.enable_kalman_fallback:
            raise ValueError("cyclic end-to-end replay requires hold-safe processor models")
        associator = CyclicSegmentAssociator(
            continuity_gate_deg=float(match.group("gate")),
            reacquire_timeout_s=float(match.group("timeout")),
            birth_descending_u=match.group("direction") == "desc",
        )
        run_predictions, run_row, run_latency = replay.process_run(
            item,
            model,
            (0.05, 0.10, 0.20),
            25.0,
            association_module,
            truth,
            0.1,
            associator_override=associator,
        )
        run_row["association_variant"] = variant
        predictions.extend(run_predictions)
        run_rows.append(run_row)
        latency_rows.extend(run_latency)
    if not predictions:
        raise ValueError("no cyclic end-to-end predictions")
    method_rows = replay.grouped_metric_rows(predictions, ("horizon_s", "method"), "method")
    gap_rows = replay.grouped_metric_rows(
        predictions, ("horizon_s", "gap_category", "method"), "gap_method"
    )
    slot_rows = replay.grouped_metric_rows(
        predictions, ("horizon_s", "motion", "mapped_slot"), "slot"
    )
    condition_rows = replay.grouped_metric_rows(
        predictions, ("horizon_s", "motion", "distance_m", "scale"), "condition"
    )
    confidence_rows = replay.confidence_rows(predictions)
    write_csv(output / "replay_predictions.csv", predictions)
    write_csv(output / "run_availability.csv", run_rows)
    write_csv(output / "method_metrics.csv", method_rows)
    write_csv(output / "gap_metrics.csv", gap_rows)
    write_csv(output / "slot_metrics.csv", slot_rows)
    write_csv(output / "condition_metrics.csv", condition_rows)
    write_csv(output / "confidence_metrics.csv", confidence_rows)
    write_csv(output / "latency_samples.csv", latency_rows)
    replay.plot_summary(output, gap_rows, confidence_rows)
    chain = quantiles([float(row["chain_error_deg"]) for row in predictions])
    identity_correct = quantiles(
        [float(row["chain_error_deg"]) for row in predictions if bool(row["identity_correct"])]
    )
    scored_runs = [row for row in run_rows if int(row["scored_associations"]) > 0]
    summary = {
        "schema_version": 1,
        "kind": "nested_cyclic_association_plus_hold_safe_cv_ridge",
        "truth_is_runtime_input": False,
        "parameter_selection_contract": "cyclic parameters for each repeat were selected only on the other four repeats",
        "collection_runs": len(run_rows),
        "runs_with_observation": sum(bool(row["has_observation"]) for row in run_rows),
        "runs_with_prediction": sum(bool(row["has_prediction"]) for row in run_rows),
        "prediction_rows": len(predictions),
        "association_mapping_accuracy_mean": float(np.mean([float(row["global_mapping_accuracy"]) for row in scored_runs])),
        "identity_correct_prediction_rate": float(np.mean([bool(row["identity_correct"]) for row in predictions])),
        "all_chain_error_deg": chain,
        "identity_correct_chain_error_deg": identity_correct,
        "processor_predict_latency_us": quantiles([float(row["predict_latency_us"]) for row in predictions]),
        "association_event_latency_us": quantiles([float(row["association_event_us"]) for row in latency_rows]),
        "source_cache": {"path": str(cache_path), "sha256": sha256_file(cache_path)},
        "source_hashes": source_hashes,
        "fold_selections": selections,
    }
    (output / "cyclic_processor_summary.json").write_text(
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
