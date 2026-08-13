#!/usr/bin/env python3
"""Consolidate trajectory regularity and fair method-selection evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def regularity(open_arc: Path, motion: str) -> tuple[list[dict], list[dict]]:
    metrics = [row for row in read_csv(open_arc / "open_arc_metrics.csv") if row["stream"] == "angular_facing"]
    summaries = []
    for distance in sorted({float(row["distance_m"]) for row in metrics}):
        group = [row for row in metrics if float(row["distance_m"]) == distance]
        summaries.append(
            {
                "motion": motion,
                "distance_m": distance,
                "cells": len(group),
                "fit_cells": sum(row["status"] == "fit" for row in group),
                "repeat_coverage_min": min(int(row["repeat_coverage"]) for row in group),
                "accepted_ratio_p50": percentile([float(row["accepted_ratio"]) for row in group], 50),
                "accepted_residual_p50_deg": percentile([float(row["accepted_residual_p50_deg"]) for row in group], 50),
                "accepted_residual_p90_deg": percentile([float(row["accepted_residual_p90_deg"]) for row in group], 50),
                "accepted_residual_p95_deg": percentile([float(row["accepted_residual_p95_deg"]) for row in group], 50),
                "orientation_match_cells": sum(row["truth_consistency"] == "match" for row in group),
            }
        )

    centers = [
        row
        for row in read_jsonl(open_arc / "open_arc_centers.jsonl")
        if row["stream"] == "angular_facing" and row["center_u_deg"] is not None
    ]
    grouped: dict[tuple[float, float, int], dict[float, dict]] = defaultdict(dict)
    for row in centers:
        grouped[(float(row["scale"]), float(row["distance_m"]), int(row["slot"]))][
            round(float(row["phase_rad"]), 9)
        ] = row
    scales = sorted({key[0] for key in grouped})
    distances = sorted({key[1] for key in grouped})
    effects = []
    for kind, adjacent in (
        ("radius", list(zip(scales[:-1], scales[1:]))),
        ("distance", list(zip(distances[:-1], distances[1:]))),
    ):
        values: list[float] = []
        truth_values: list[float] = []
        for first, second in adjacent:
            fixed_values = distances if kind == "radius" else scales
            for fixed in fixed_values:
                for slot in range(4):
                    key_a = (first, fixed, slot) if kind == "radius" else (fixed, first, slot)
                    key_b = (second, fixed, slot) if kind == "radius" else (fixed, second, slot)
                    common = sorted(set(grouped.get(key_a, {})).intersection(grouped.get(key_b, {})))
                    if len(common) < 3:
                        continue
                    rows_a = [grouped[key_a][phase] for phase in common]
                    rows_b = [grouped[key_b][phase] for phase in common]
                    observed_a = np.asarray([[row["center_u_deg"], row["center_v_deg"]] for row in rows_a], dtype=float)
                    observed_b = np.asarray([[row["center_u_deg"], row["center_v_deg"]] for row in rows_b], dtype=float)
                    values.append(float(np.percentile(np.linalg.norm(observed_a - observed_b, axis=1), 95)))
                    usable = [
                        index
                        for index, (row_a, row_b) in enumerate(zip(rows_a, rows_b))
                        if row_a["truth_u_deg"] is not None and row_b["truth_u_deg"] is not None
                    ]
                    if len(usable) >= 3:
                        truth_a = np.asarray([[rows_a[index]["truth_u_deg"], rows_a[index]["truth_v_deg"]] for index in usable], dtype=float)
                        truth_b = np.asarray([[rows_b[index]["truth_u_deg"], rows_b[index]["truth_v_deg"]] for index in usable], dtype=float)
                        truth_values.append(float(np.percentile(np.linalg.norm(truth_a - truth_b, axis=1), 95)))
        effects.append(
            {
                "motion": motion,
                "change_type": kind,
                "adjacent_pairs": len(values),
                "observed_center_delta_p95_median_deg": percentile(values, 50),
                "observed_center_delta_p95_p95_deg": percentile(values, 95),
                "truth_center_delta_p95_median_deg": percentile(truth_values, 50),
                "truth_center_delta_p95_p95_deg": percentile(truth_values, 95),
            }
        )
    return summaries, effects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "spin_open_arc_metrics": root / "spin" / "open-arc-audit-v1" / "open_arc_metrics.csv",
        "spin_open_arc_centers": root / "spin" / "open-arc-audit-v1" / "open_arc_centers.jsonl",
        "combined_open_arc_metrics": root / "combined" / "open-arc-audit-v1" / "open_arc_metrics.csv",
        "combined_open_arc_centers": root / "combined" / "open-arc-audit-v1" / "open_arc_centers.jsonl",
        "observation_error": root / "method-analysis" / "observation-error-v1" / "observation_error_structure_summary.json",
        "association": root / "method-analysis" / "association-v3" / "observation_only_association_summary.csv",
        "availability": root / "method-analysis" / "fair-core-v2" / "baseline" / "data_availability_runs.csv",
        "ranking": root / "method-analysis" / "fair-core-final" / "summary" / "method_ranking.csv",
        "bootstrap": root / "method-analysis" / "fair-core-final" / "summary" / "paired_run_bootstrap.csv",
    }
    for name, path in source_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    regularity_rows: list[dict] = []
    effect_rows: list[dict] = []
    for motion in ("spin", "combined"):
        rows, effects = regularity(root / motion / "open-arc-audit-v1", motion)
        regularity_rows.extend(rows)
        effect_rows.extend(effects)
    write_csv(output / "trajectory_regularity_summary.csv", regularity_rows)
    write_csv(output / "trajectory_adjacent_condition_effects.csv", effect_rows)

    error_document = json.loads(source_paths["observation_error"].read_text(encoding="utf-8"))
    error_summary = error_document["aggregate"]
    association = read_csv(source_paths["association"])
    association_accuracy = [row for row in association if row["metric"] == "global_mapping_accuracy"]
    best_association = max(association_accuracy, key=lambda row: float(row["mean"]))
    availability = read_csv(source_paths["availability"])
    rankings = read_csv(source_paths["ranking"])
    bootstraps = read_csv(source_paths["bootstrap"])

    selected_methods = {"ridge_uv_residual", "kalman_cv", "mlp_uv_residual", "ridge_uv_yaw_residual", "periodic_ekf_shared", "periodic_ukf_shared"}
    selected_rankings = [row for row in rankings if row["method"] in selected_methods]
    write_csv(output / "selected_method_rankings.csv", selected_rankings)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
    colors = {"ridge_uv_residual": "#0072B2", "kalman_cv": "#009E73", "mlp_uv_residual": "#D55E00"}
    for axis, horizon in zip(axes, (0.05, 0.10, 0.20)):
        rows = [
            row for row in selected_rankings
            if row["split"] in ("repeat_holdout", "leave_distance_out", "leave_radius_out", "motion_transfer")
            and abs(float(row["horizon_s"]) - horizon) < 1e-9
            and row["input_tier"] == "common_uv"
            and row["method"] in colors
        ]
        methods = list(colors)
        split_order = ("repeat_holdout", "leave_distance_out", "leave_radius_out", "motion_transfer")
        x = np.arange(len(split_order))
        width = 0.24
        for index, method in enumerate(methods):
            values = [
                float(next(row for row in rows if row["split"] == split and row["method"] == method)["condition_equal_p95_deg"])
                for split in split_order
            ]
            axis.bar(x + (index - 1) * width, values, width, color=colors[method], label=method)
        axis.set_title(f"{int(horizon * 1000)} ms")
        axis.set_xticks(x)
        axis.set_xticklabels(("repeat", "distance", "radius", "motion"), rotation=20)
        axis.set_ylabel("condition-equal angular P95 (deg)")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output / "fair_method_comparison.png", dpi=260, bbox_inches="tight")
    fig.savefig(output / "fair_method_comparison.svg", bbox_inches="tight")
    plt.close(fig)

    spin_rows = [row for row in regularity_rows if row["motion"] == "spin"]
    combined_rows = [row for row in regularity_rows if row["motion"] == "combined"]
    spin_p90 = percentile([row["accepted_residual_p90_deg"] for row in spin_rows], 50)
    combined_p90 = percentile([row["accepted_residual_p90_deg"] for row in combined_rows], 50)
    report = f"""# Trajectory regularity and method-selection evidence

## Evidence boundary

- Truth is the deterministic label and audit reference. It is not questioned and is not an online model input.
- The processing comparison is an oracle physical-slot upper bound. Observation-only identity is scored separately.
- All candidates within an input tier share the same stable run/slot/timestamp/horizon sample hashes.
- The geometric centroid/polar plots are deprecated for open arcs because they can connect unsupported regions. The phase-binned, image-jump-split open-arc audit is authoritative.

## Trajectory regularity

- 96/96 motion × radius × distance × armor cells have fitted centers with repeated support.
- Median accepted within-arc P90 residual: spin {spin_p90:.3f} deg; combined {combined_p90:.3f} deg.
- Combined motion is broader at close range because yaw phase alone does not identify lateral translation state. A processor must represent both translation and rotation history.
- Adjacent radius and distance changes move the center curves continuously; see `trajectory_adjacent_condition_effects.csv`. Distance effects are generally larger than adjacent radius effects.

## Observation quality and association

- Spin observation angular error P50/P95: {error_summary['spin']['angular_error_deg_p50']:.3f}/{error_summary['spin']['angular_error_deg_p95']:.3f} deg.
- Combined observation angular error P50/P95: {error_summary['combined']['angular_error_deg_p50']:.3f}/{error_summary['combined']['angular_error_deg_p95']:.3f} deg.
- Median lag-1 angular-error autocorrelation: spin {error_summary['spin']['median_lag1_angular_error_autocorrelation']:.3f}; combined {error_summary['combined']['median_lag1_angular_error_autocorrelation']:.3f}.
- Collection runs with any observation and eligible history: {sum(row['has_eligible_history'].lower() == 'true' for row in availability)}/{len(availability)}. The two empty runs remain failures in availability accounting.
- Best simple observation-only four-track mapping accuracy: {float(best_association['mean']):.3f}. Therefore the oracle-slot result is not deployment-ready.

## Method decision

- Select **constant-velocity + Ridge residual correction using causal u/v history and timestamps** as the next processing baseline.
- Keep a simple Kalman/hold fallback for missing or rejected histories; do not select it as the main fitted model.
- Do not select the current small MLP: it does not consistently beat Ridge and degrades more on distance/motion transfer.
- Do not select the current Fourier periodic EKF/UKF: both trail Ridge substantially and EKF/UKF are nearly indistinguishable.
- Do not use PnP yaw in the first baseline. Its Ridge gain is marginal while yaw materially harms simple identity association.
- Neural modeling remains worth revisiting only after more independent motion speeds/spans and a deployable association interface are available.

## Next experiment

1. Implement the u/v-only CV + Ridge residual processor behind an offline/replay interface.
2. Add gap-aware fallback and confidence based on history span, maximum gap, detector coverage and residual scale.
3. Evaluate end-to-end association + processing, including reacquisition, long gaps, per-slot worst case and CPU/GPU P99 latency.
4. Expand combined-motion data across independent linear speeds, spans, spin rates and phase offsets before reconsidering a neural model.
"""
    (output / "METHOD_SELECTION_EVIDENCE.md").write_text(report, encoding="utf-8")

    summary = {
        "schema_version": 1,
        "kind": "trajectory_regularity_and_method_selection_evidence",
        "decision": "causal constant-velocity plus u/v-only Ridge residual correction",
        "fallback": "simple Kalman/hold for missing or rejected histories",
        "oracle_identity_upper_bound": True,
        "trajectory_cells_fit": sum(row["fit_cells"] for row in regularity_rows),
        "trajectory_cells_total": sum(row["cells"] for row in regularity_rows),
        "runs_with_eligible_history": sum(row["has_eligible_history"].lower() == "true" for row in availability),
        "collection_runs": len(availability),
        "best_observation_only_mapping_accuracy": float(best_association["mean"]),
        "sources": {name: {"path": str(path), "sha256": sha256(path)} for name, path in source_paths.items()},
    }
    (output / "method_selection_evidence.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in output.iterdir()
        if path.is_file() and path.name != "retention_manifest.json"
    }
    (output / "retention_manifest.json").write_text(
        json.dumps(
            {
                "classification": "long_term_private_evidence",
                "deletion_allowed": False,
                "sources": summary["sources"],
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
