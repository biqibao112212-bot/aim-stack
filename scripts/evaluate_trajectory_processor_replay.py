#!/usr/bin/env python3
"""End-to-end offline replay for observation association plus CV/Ridge processing.

The runtime path receives detections stripped of truth labels.  Truth is joined
only after a run has completed, first to score the run-global track mapping and
then to evaluate future angular error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trajectory_uv_processor import (
    CausalUvRidgeProcessor,
    ProcessorConfig,
    UvRidgeModel,
    angular_error_deg,
)


DEFAULT_HORIZONS = (0.05, 0.10, 0.20)


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_labeled_paths(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    seen: set[tuple[str, Path]] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected LABEL=PATH: {value}")
        label, raw = value.split("=", 1)
        path = Path(raw).resolve()
        key = (label, path)
        if not label or not path.is_dir() or key in seen:
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
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(array):
        return {key: float("nan") for key in ("p50", "p90", "p95", "p99", "mean")}
    return {
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "mean": float(np.mean(array)),
    }


def load_examples(
    analyses: list[tuple[str, Path]], method_module
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, str]]:
    examples: list[dict[str, Any]] = []
    truth_series: dict[
        tuple[str, str, int], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    source_hashes: dict[str, str] = {}
    for label, analysis in analyses:
        method_module.validate_analysis_source(label, analysis)
        local = method_module.build_examples(label, analysis)
        examples.extend(local)
        truth_rows = read_jsonl(analysis / "truth_points.jsonl")
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in truth_rows:
            grouped[(str(row["run"]), int(row["slot"]))].append(row)
        for (run, slot), rows in grouped.items():
            # Association replay is run-relative.  The accepted analysis
            # already persists the exact relative t_s derived from each run's
            # first truth timestamp, so do not mix it with Unix epoch seconds.
            ordered = sorted(rows, key=lambda row: float(row["t_s"]))
            times = np.asarray([float(row["t_s"]) for row in ordered], dtype=float)
            u = method_module.unwrap_degrees(
                np.asarray([float(row["u_deg"]) for row in ordered], dtype=float)
            )
            v = np.asarray([float(row["v_deg"]) for row in ordered], dtype=float)
            if len(times) > 1 and np.any(np.diff(times) <= 0.0):
                raise ValueError(f"truth relative times regress: {label}/{run}/slot={slot}")
            truth_series[(label, run, slot)] = (times, u, v)
        for filename in ("analysis_summary.json", "observed_points.jsonl", "truth_points.jsonl"):
            path = analysis / filename
            source_hashes[f"{label}:{analysis.name}:{filename}"] = sha256_file(path)
    ids = [str(item["example_id"]) for item in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate oracle training example IDs")
    return examples, truth_series, source_hashes


def authorized_roots(analyses: list[tuple[str, Path]]) -> dict[str, set[Path]]:
    result: dict[str, set[Path]] = defaultdict(set)
    for label, analysis in analyses:
        summary = json.loads((analysis / "analysis_summary.json").read_text(encoding="utf-8"))
        result[label].update(Path(raw).resolve() for raw in summary["roots"])
    return result


def prepare_runs(
    roots: list[tuple[str, Path]],
    analyses: list[tuple[str, Path]],
    association_module,
    grid_module,
    accepted_run_names: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    allowed = authorized_roots(analyses)
    prepared: list[dict[str, Any]] = []
    stable_ids: set[str] = set()
    for label, root in roots:
        if root not in allowed.get(label, set()):
            raise ValueError(f"raw root is not bound by {label} analysis: {root}")
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if not (run_dir / "stage3_observations.jsonl").exists():
                continue
            # A source root can contain diagnostic radii that the accepted
            # analysis intentionally excludes.  The truth-point authority,
            # not directory presence, defines the replay cohort.
            if (label, run_dir.name) not in accepted_run_names:
                continue
            item = association_module.prepare_run(run_dir, root, grid_module)
            stable_id = f"{label}:{run_dir.name}"
            if stable_id in stable_ids:
                raise ValueError(f"duplicate stable run ID: {stable_id}")
            stable_ids.add(stable_id)
            item.update(
                {
                    "motion": label,
                    "run_name": run_dir.name,
                    "stable_run": stable_id,
                    "raw_root": str(root),
                }
            )
            prepared.append(item)
    return prepared


def interpolate_truth(
    series: tuple[np.ndarray, np.ndarray, np.ndarray] | None, timestamp_s: float
) -> np.ndarray | None:
    if series is None:
        return None
    times, u, v = series
    if timestamp_s < times[0] or timestamp_s > times[-1]:
        return None
    return np.asarray([np.interp(timestamp_s, times, u), np.interp(timestamp_s, times, v)])


def error_one(predicted: tuple[float, float], actual: np.ndarray) -> float:
    return float(
        angular_error_deg(
            np.asarray([[predicted[0], predicted[1]]], dtype=float),
            np.asarray([actual], dtype=float),
        )[0]
    )


def gap_category(preceding_gap_s: float) -> str:
    if not math.isfinite(preceding_gap_s):
        return "new_track"
    if preceding_gap_s > 0.12:
        return "reacquisition_gt120ms"
    if preceding_gap_s > 0.04:
        return "gap_40_120ms"
    return "continuous_le40ms"


def process_run(
    prepared: dict[str, Any],
    model: UvRidgeModel,
    horizons: tuple[float, ...],
    gate_deg: float,
    association_module,
    truth: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    evaluation_interval_s: float,
    associator_override=None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, float]]]:
    associator = associator_override or association_module.CausalAssociator(gate_deg, False)
    processors: dict[int, CausalUvRidgeProcessor] = {}
    last_evaluation: dict[int, float] = defaultdict(lambda: -float("inf"))
    association_rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    latency_rows: list[dict[str, float]] = []
    associated_count = 0
    for detections in prepared["events"]:
        runtime_detections = [
            {key: value for key, value in detection.items() if key != "truth_slot"}
            for detection in detections
        ]
        score_slots = [detection.get("truth_slot") for detection in detections]
        association_start = time.perf_counter_ns()
        assignments = associator.update(runtime_detections)
        association_us = (time.perf_counter_ns() - association_start) / 1000.0
        latency_rows.append(
            {
                "association_event_us": association_us,
                "event_detections": float(len(runtime_detections)),
            }
        )
        for (detection, track_id), truth_slot in zip(assignments, score_slots):
            association_rows.append(
                {
                    "track_id": track_id,
                    "truth_slot": truth_slot,
                    "t_s": detection["t_s"],
                }
            )
            if track_id is None:
                continue
            associated_count += 1
            processor = processors.setdefault(track_id, CausalUvRidgeProcessor(model))
            preceding_gap = processor.update(
                float(detection["t_s"]),
                math.degrees(float(detection["u"])),
                math.degrees(float(detection["v"])),
            )
            if float(detection["t_s"]) - last_evaluation[track_id] < evaluation_interval_s:
                continue
            last_evaluation[track_id] = float(detection["t_s"])
            for horizon_s in horizons:
                prediction_start = time.perf_counter_ns()
                prediction = processor.predict(horizon_s)
                predict_us = (time.perf_counter_ns() - prediction_start) / 1000.0
                row = {
                    "motion": prepared["motion"],
                    "run": prepared["stable_run"],
                    "run_name": prepared["run_name"],
                    "distance_m": prepared["distance_m"],
                    "scale": prepared["scale"],
                    "repeat": prepared["repeat"],
                    "track_id": track_id,
                    "anchor_truth_slot": truth_slot,
                    "timestamp_s": float(detection["t_s"]),
                    "horizon_s": horizon_s,
                    "preceding_gap_s": preceding_gap,
                    "gap_category": gap_category(preceding_gap),
                    "predict_latency_us": predict_us,
                    "association_mode": detection.get("association_mode", "unspecified"),
                    "association_confidence": float(
                        detection.get("association_confidence", 0.0)
                    ),
                }
                row.update(asdict(prediction))
                pending.append(row)
    scored_associations = [row for row in association_rows if row["truth_slot"] is not None]
    mapping, correct = association_module.best_slot_mapping(scored_associations)
    predictions: list[dict[str, Any]] = []
    for row in pending:
        mapped_slot = mapping.get(int(row["track_id"]))
        if mapped_slot is None:
            continue
        future_s = float(row["timestamp_s"] + row["horizon_s"])
        actual = interpolate_truth(
            truth.get((prepared["motion"], prepared["run_name"], mapped_slot)), future_s
        )
        if actual is None:
            continue
        predicted = (float(row["u_deg"]), float(row["v_deg"]))
        row["mapped_slot"] = mapped_slot
        row["identity_correct"] = row["anchor_truth_slot"] == mapped_slot
        row["truth_u_deg"] = float(actual[0])
        row["truth_v_deg"] = float(actual[1])
        row["chain_error_deg"] = error_one(predicted, actual)
        local_actual = (
            None
            if row["anchor_truth_slot"] is None
            else interpolate_truth(
                truth.get(
                    (
                        prepared["motion"],
                        prepared["run_name"],
                        int(row["anchor_truth_slot"]),
                    )
                ),
                future_s,
            )
        )
        row["conditional_error_deg"] = (
            float("nan") if local_actual is None else error_one(predicted, local_actual)
        )
        row["covered_by_uncertainty_p90"] = (
            row["chain_error_deg"] <= row["uncertainty_p90_deg"]
        )
        predictions.append(row)
    run_row = {
        "motion": prepared["motion"],
        "run": prepared["stable_run"],
        "distance_m": prepared["distance_m"],
        "scale": prepared["scale"],
        "repeat": prepared["repeat"],
        "valid_detections": prepared["valid_detections"],
        "associated_detections": associated_count,
        "scored_associations": len(scored_associations),
        "global_mapping_accuracy": (
            correct / len(scored_associations) if scored_associations else float("nan")
        ),
        "prediction_rows": len(predictions),
        "has_observation": prepared["valid_detections"] > 0,
        "has_prediction": bool(predictions),
    }
    return predictions, run_row, latency_rows


def grouped_metric_rows(
    predictions: list[dict[str, Any]], fields: tuple[str, ...], output_name: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[tuple(row[field] for field in fields)].append(row)
    result: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        chain = quantiles([float(row["chain_error_deg"]) for row in rows])
        conditional = quantiles([float(row["conditional_error_deg"]) for row in rows])
        result.append(
            {
                **dict(zip(fields, key)),
                "group": output_name,
                "samples": len(rows),
                "chain_error_p50_deg": chain["p50"],
                "chain_error_p90_deg": chain["p90"],
                "chain_error_p95_deg": chain["p95"],
                "chain_error_p99_deg": chain["p99"],
                "conditional_error_p95_deg": conditional["p95"],
                "identity_correct_rate": float(
                    np.mean([bool(row["identity_correct"]) for row in rows])
                ),
                "uncertainty_coverage": float(
                    np.mean([bool(row["covered_by_uncertainty_p90"]) for row in rows])
                ),
                "confidence_p50": float(
                    np.percentile([float(row["confidence"]) for row in rows], 50)
                ),
            }
        )
    return result


def confidence_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, 6)
    for horizon in sorted({float(row["horizon_s"]) for row in predictions}):
        selected_horizon = [row for row in predictions if float(row["horizon_s"]) == horizon]
        for low, high in zip(edges[:-1], edges[1:]):
            selected = [
                row
                for row in selected_horizon
                if low <= float(row["confidence"]) < high
                or (high == 1.0 and float(row["confidence"]) == 1.0)
            ]
            if not selected:
                continue
            errors = quantiles([float(row["chain_error_deg"]) for row in selected])
            result.append(
                {
                    "horizon_s": horizon,
                    "confidence_low": low,
                    "confidence_high": high,
                    "samples": len(selected),
                    "chain_error_p50_deg": errors["p50"],
                    "chain_error_p95_deg": errors["p95"],
                    "identity_correct_rate": float(
                        np.mean([bool(row["identity_correct"]) for row in selected])
                    ),
                    "uncertainty_coverage": float(
                        np.mean(
                            [bool(row["covered_by_uncertainty_p90"]) for row in selected]
                        )
                    ),
                }
            )
    return result


def plot_summary(
    output: Path,
    gap_rows: list[dict[str, Any]],
    confidence: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    gap_order = ["new_track", "continuous_le40ms", "gap_40_120ms", "reacquisition_gt120ms"]
    methods = ["hold_fallback", "kalman_fallback", "ridge_uv_residual"]
    selected_gap = [row for row in gap_rows if abs(float(row["horizon_s"]) - 0.1) < 1e-9]
    x = np.arange(len(gap_order), dtype=float)
    width = 0.24
    for method_index, method in enumerate(methods):
        values = []
        for gap in gap_order:
            match = next(
                (
                    row
                    for row in selected_gap
                    if row["gap_category"] == gap and row["method"] == method
                ),
                None,
            )
            values.append(float("nan") if match is None else float(match["chain_error_p95_deg"]))
        axes[0].bar(x + (method_index - 1) * width, values, width, label=method)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        ["new", "continuous", "40–120 ms", ">120 ms"], rotation=15
    )
    axes[0].set_ylabel("end-to-end angular error P95 (deg)")
    axes[0].set_title("100 ms replay by gap and fallback")
    axes[0].legend(fontsize=8)
    for horizon in sorted({float(row["horizon_s"]) for row in confidence}):
        rows = [row for row in confidence if float(row["horizon_s"]) == horizon]
        centers = [
            (float(row["confidence_low"]) + float(row["confidence_high"])) * 0.5
            for row in rows
        ]
        values = [float(row["chain_error_p95_deg"]) for row in rows]
        axes[1].plot(centers, values, marker="o", label=f"{int(horizon * 1000)} ms")
    axes[1].set_xlabel("reported confidence bin center")
    axes[1].set_ylabel("end-to-end angular error P95 (deg)")
    axes[1].set_title("Confidence diagnostic (not calibrated probability)")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "replay_gap_confidence.png", dpi=180)
    fig.savefig(output / "replay_gap_confidence.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--root", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gate-deg", type=float, default=25.0)
    parser.add_argument("--evaluation-rate-hz", type=float, default=10.0)
    parser.add_argument("--expected-runs", type=int, default=120)
    parser.add_argument("--max-train-examples", type=int, default=12000)
    parser.add_argument(
        "--reuse-models",
        action="store_true",
        help="reuse fold models only when their test-repeat and run hash match",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    analyses = parse_labeled_paths(args.analysis)
    roots = parse_labeled_paths(args.root)
    method_module = load_script(
        repo / "scripts" / "evaluate-trajectory-processing-methods.py",
        "trajectory_method_evaluation",
    )
    association_module = load_script(
        repo / "scripts" / "evaluate-observation-association.py",
        "observation_association",
    )
    grid_module = association_module.load_grid_analysis(repo)
    examples, truth, source_hashes = load_examples(analyses, method_module)
    accepted_run_names = {(label, run) for label, run, _slot in truth}
    prepared = prepare_runs(
        roots,
        analyses,
        association_module,
        grid_module,
        accepted_run_names,
    )
    if len(prepared) != args.expected_runs:
        raise ValueError(f"expected {args.expected_runs} runs, found {len(prepared)}")
    repeats = sorted({int(item["repeat"]) for item in prepared})
    horizons = DEFAULT_HORIZONS
    config = ProcessorConfig(max_train_examples_per_horizon=args.max_train_examples)
    predictions: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, float]] = []
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    for repeat in repeats:
        test_runs = [item for item in prepared if int(item["repeat"]) == repeat]
        test_ids = {str(item["stable_run"]) for item in test_runs}
        train = [item for item in examples if int(item["repeat"]) != repeat]
        train_ids = {str(item["run"]) for item in train}
        overlap = train_ids.intersection(test_ids)
        if overlap:
            raise ValueError(f"train/test run leakage for repeat {repeat}: {sorted(overlap)[:3]}")
        test_run_hash = hashlib.sha256(
            "\n".join(sorted(test_ids)).encode("utf-8")
        ).hexdigest()
        model_path = model_dir / f"repeat-{repeat:02d}.json"
        if args.reuse_models and model_path.exists():
            model = UvRidgeModel.load(model_path)
            if (
                int(model.metadata.get("test_repeat", -1)) != repeat
                or model.metadata.get("test_run_hash") != test_run_hash
            ):
                raise ValueError(f"reused model cohort mismatch: {model_path}")
        else:
            model = UvRidgeModel.fit(
                train,
                config,
                {
                    "split": "repeat_holdout",
                    "test_repeat": repeat,
                    "train_run_count": len(train_ids),
                    "test_run_count": len(test_ids),
                    "test_run_hash": test_run_hash,
                },
            )
            model.save(model_path)
        for prepared_run in test_runs:
            run_predictions, run_row, run_latency = process_run(
                prepared_run,
                model,
                horizons,
                args.gate_deg,
                association_module,
                truth,
                1.0 / args.evaluation_rate_hz,
            )
            predictions.extend(run_predictions)
            run_rows.append(run_row)
            latency_rows.extend(run_latency)
    final_model_path = output / "cv_ridge_uv_model.json"
    if args.reuse_models and final_model_path.exists():
        final_model = UvRidgeModel.load(final_model_path)
        if final_model.metadata.get("source_hashes") != source_hashes:
            raise ValueError(f"reused final model source mismatch: {final_model_path}")
    else:
        final_model = UvRidgeModel.fit(
            examples,
            config,
            {
                "split": "all_accepted_runs",
                "training_run_count": len({str(item["run"]) for item in examples}),
                "source_hashes": source_hashes,
            },
        )
        final_model.save(final_model_path)
    if not predictions:
        raise ValueError("end-to-end replay produced no scored predictions")
    method_rows = grouped_metric_rows(predictions, ("horizon_s", "method"), "method")
    gap_rows = grouped_metric_rows(
        predictions, ("horizon_s", "gap_category", "method"), "gap_method"
    )
    slot_rows = grouped_metric_rows(
        predictions, ("horizon_s", "motion", "mapped_slot"), "slot"
    )
    condition_rows = grouped_metric_rows(
        predictions,
        ("horizon_s", "motion", "distance_m", "scale"),
        "condition",
    )
    confidence = confidence_rows(predictions)
    per_run = grouped_metric_rows(predictions, ("horizon_s", "run"), "run")
    write_csv(output / "replay_predictions.csv", predictions)
    write_csv(output / "run_availability.csv", run_rows)
    write_csv(output / "method_metrics.csv", method_rows)
    write_csv(output / "gap_metrics.csv", gap_rows)
    write_csv(output / "slot_metrics.csv", slot_rows)
    write_csv(output / "condition_metrics.csv", condition_rows)
    write_csv(output / "confidence_metrics.csv", confidence)
    write_csv(output / "per_run_metrics.csv", per_run)
    write_csv(output / "latency_samples.csv", latency_rows)
    plot_summary(output, gap_rows, confidence)
    predict_latency = quantiles([float(row["predict_latency_us"]) for row in predictions])
    association_latency = quantiles(
        [float(row["association_event_us"]) for row in latency_rows]
    )
    scored_runs = [row for row in run_rows if int(row["scored_associations"]) > 0]
    zero_observation = [row["run"] for row in run_rows if not row["has_observation"]]
    mapping_accuracy = quantiles(
        [float(row["global_mapping_accuracy"]) for row in scored_runs]
    )
    all_chain = quantiles([float(row["chain_error_deg"]) for row in predictions])
    correct_chain = quantiles(
        [
            float(row["chain_error_deg"])
            for row in predictions
            if bool(row["identity_correct"])
        ]
    )
    summary = {
        "schema_version": 1,
        "kind": "observation_association_plus_cv_ridge_replay",
        "decision_boundary": "offline Python replay; not an online or production latency claim",
        "runtime_input_contract": ["timestamp_s", "camera_tvec_m converted to u/v"],
        "truth_is_runtime_input": False,
        "training_contract": "truth labels Ridge residual targets; complete repeat-held-out runs are never used for model fitting",
        "association_contract": "truth_slot is stripped before CausalAssociator.update and joined only after each run for scoring",
        "gate_deg": args.gate_deg,
        "evaluation_rate_hz": args.evaluation_rate_hz,
        "reused_hash_matched_models": bool(args.reuse_models),
        "collection_runs": len(run_rows),
        "runs_with_observation": sum(bool(row["has_observation"]) for row in run_rows),
        "runs_with_prediction": sum(bool(row["has_prediction"]) for row in run_rows),
        "zero_observation_runs": zero_observation,
        "prediction_rows": len(predictions),
        "association_global_mapping_accuracy": mapping_accuracy,
        "all_chain_error_deg": all_chain,
        "identity_correct_chain_error_deg": correct_chain,
        "identity_correct_prediction_rate": float(
            np.mean([bool(row["identity_correct"]) for row in predictions])
        ),
        "processor_predict_latency_us": predict_latency,
        "association_event_latency_us": association_latency,
        "artifacts": [
            "cv_ridge_uv_model.json",
            "replay_predictions.csv",
            "run_availability.csv",
            "method_metrics.csv",
            "gap_metrics.csv",
            "slot_metrics.csv",
            "condition_metrics.csv",
            "confidence_metrics.csv",
            "per_run_metrics.csv",
            "latency_samples.csv",
            "replay_gap_confidence.png",
            "replay_gap_confidence.svg",
        ],
        "source_hashes": source_hashes,
    }
    (output / "replay_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    artifact_hashes = {
        str(path.relative_to(output)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "retention_manifest.json"
    }
    retention = {
        "classification": "protected_derived_model_and_long_term_private_evidence",
        "deletion_allowed": False,
        "artifacts": artifact_hashes,
    }
    (output / "retention_manifest.json").write_text(
        json.dumps(retention, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
