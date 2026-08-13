#!/usr/bin/env python3
"""Quantify observation-vs-truth structure for trajectory method selection.

The analysis is descriptive. Truth fields are used only as labels, audit
covariates, and scoring targets. The PnP-yaw harmonic model is the only model
in this file whose phase input is available from the observation stream; the
truth-phase model is explicitly reported as an oracle upper bound.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LAGS = (1, 2, 4, 8, 16)
HARMONICS = 4
COLOR = {"spin": "#0072B2", "combined": "#D55E00"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="angular-facing analysis directory; repeat for each motion mode",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def parse_inputs(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"analysis input must be LABEL=DIR: {value}")
        label, raw_path = value.split("=", 1)
        path = Path(raw_path).resolve()
        if not label or not path.is_dir():
            raise ValueError(f"invalid analysis input: {value}")
        result.append((label, path))
    if len({label for label, _ in result}) != len(result):
        raise ValueError("analysis labels must be unique")
    return result


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def finite(value: Any, default: float = float("nan")) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) else default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentiles(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {f"{prefix}_n": 0}
    return {
        f"{prefix}_n": int(values.size),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_p99": float(np.percentile(values, 99)),
        f"{prefix}_max": float(np.max(values)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def harmonic_design(phase: np.ndarray) -> np.ndarray:
    columns = [np.ones(len(phase), dtype=float)]
    for harmonic in range(1, HARMONICS + 1):
        columns.extend((np.sin(harmonic * phase), np.cos(harmonic * phase)))
    return np.column_stack(columns)


def fit_harmonic(train_phase: np.ndarray, train_error: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(harmonic_design(train_phase), train_error, rcond=None)[0]


def angular_error(e_u: np.ndarray, e_v: np.ndarray) -> np.ndarray:
    return np.hypot(e_u, e_v)


def load_dataset(label: str, path: Path) -> tuple[list[dict], list[dict], list[dict], dict]:
    summary = read_json(path / "analysis_summary.json")
    truth = read_jsonl(path / "truth_points.jsonl")
    observed = read_jsonl(path / "observed_points.jsonl")
    with (path / "run_metrics.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        run_metrics = list(csv.DictReader(handle))
    for row in truth:
        row["motion"] = label
    for row in observed:
        row["motion"] = label
        row["error_u_deg"] = float(row["u_deg"]) - float(row["truth_u_deg"])
        row["error_v_deg"] = float(row["v_deg"]) - float(row["truth_v_deg"])
        row["angular_error_deg"] = math.hypot(row["error_u_deg"], row["error_v_deg"])
        row["signed_depth_error_m"] = float(row["pnp_camera_z_m"]) - float(row["truth_camera_z_m"])
    for row in run_metrics:
        row["motion"] = label
    return truth, observed, run_metrics, summary


def condition_metrics(observed: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in observed:
        actual_scale = round(finite(row.get("actual_radial_scale"), float(row["scale"])), 3)
        grouped[(row["motion"], float(row["distance_m"]), actual_scale, int(row["slot"]))].append(row)
    result = []
    for (motion, distance_m, scale, slot), rows in sorted(grouped.items()):
        angular = np.asarray([row["angular_error_deg"] for row in rows])
        depth = np.abs(np.asarray([row["signed_depth_error_m"] for row in rows]))
        position = np.asarray([finite(row.get("pnp_position_error_m")) for row in rows])
        facing = np.asarray([finite(row.get("truth_facing_score")) for row in rows])
        robust_center = np.median(angular)
        mad = np.median(np.abs(angular - robust_center))
        threshold = robust_center + max(6.0 * 1.4826 * mad, 0.05)
        record: dict[str, Any] = {
            "motion": motion,
            "distance_m": distance_m,
            "scale": scale,
            "actual_distance_camera_m_p50": float(
                np.median([finite(row.get("target_distance_camera_m")) for row in rows])
            ),
            "actual_distance_gimbal_m_p50": float(
                np.median([finite(row.get("target_distance_gimbal_m")) for row in rows])
            ),
            "actual_radius_even_m_p50": float(
                np.median([finite(row.get("target_radius_even_m")) for row in rows])
            ),
            "actual_radius_odd_m_p50": float(
                np.median([finite(row.get("target_radius_odd_m")) for row in rows])
            ),
            "slot": slot,
            "repeats": len({int(row["repeat"]) for row in rows}),
            "robust_outlier_threshold_deg": float(threshold),
            "robust_outlier_rate": float(np.mean(angular > threshold)),
            "error_u_mean_deg": float(np.mean([row["error_u_deg"] for row in rows])),
            "error_v_mean_deg": float(np.mean([row["error_v_deg"] for row in rows])),
            "error_uv_correlation": float(
                np.corrcoef(
                    [row["error_u_deg"] for row in rows], [row["error_v_deg"] for row in rows]
                )[0, 1]
            ),
            "angular_error_skew": float(stats.skew(angular)),
            "angular_error_excess_kurtosis": float(stats.kurtosis(angular)),
            "truth_facing_p50": float(np.nanmedian(facing)),
        }
        record.update(percentiles(angular, "angular_error_deg"))
        record.update(percentiles(depth, "abs_depth_error_m"))
        record.update(percentiles(position, "position_error_m"))
        result.append(record)
    return result


def dependence_metrics(observed: list[dict]) -> list[dict]:
    result = []
    for motion in sorted({row["motion"] for row in observed}):
        rows = [row for row in observed if row["motion"] == motion]
        error = np.asarray([row["angular_error_deg"] for row in rows])
        variables = {
            "distance_m": [float(row["distance_m"]) for row in rows],
            "actual_radial_scale": [finite(row.get("actual_radial_scale"), float(row["scale"])) for row in rows],
            "truth_facing_score": [finite(row.get("truth_facing_score")) for row in rows],
            "pnp_depth_m": [finite(row.get("pnp_camera_z_m")) for row in rows],
            "abs_signed_depth_error_m": [abs(finite(row.get("signed_depth_error_m"))) for row in rows],
            "target_speed_mps": [finite(row.get("target_speed_mps")) for row in rows],
            "observation_armor_count": [finite(row.get("observation_armor_count")) for row in rows],
        }
        for variable, raw in variables.items():
            values = np.asarray(raw, dtype=float)
            valid = np.isfinite(values) & np.isfinite(error)
            if valid.sum() < 20 or np.std(values[valid]) < 1e-12:
                rho, pvalue = float("nan"), float("nan")
            else:
                rho, pvalue = stats.spearmanr(values[valid], error[valid])
            result.append(
                {
                    "motion": motion,
                    "variable": variable,
                    "samples": int(valid.sum()),
                    "spearman_rho_with_angular_error": float(rho),
                    "p_value": float(pvalue),
                }
            )
    return result


def autocorrelation_metrics(observed: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in observed:
        grouped[(row["motion"], row["run"], int(row["slot"]))].append(row)
    result = []
    for (motion, run, slot), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["timestamp_ns"]))
        if len(ordered) < 40:
            continue
        times = np.asarray([int(row["timestamp_ns"]) * 1e-9 for row in ordered])
        dt = np.diff(times)
        for component in ("error_u_deg", "error_v_deg", "signed_depth_error_m"):
            values = np.asarray([float(row[component]) for row in ordered])
            values = values - np.mean(values)
            variance = float(np.dot(values, values))
            for lag in LAGS:
                if len(values) <= lag or variance <= 1e-15:
                    correlation = float("nan")
                else:
                    correlation = float(np.dot(values[:-lag], values[lag:]) / variance)
                result.append(
                    {
                        "motion": motion,
                        "run": run,
                        "slot": slot,
                        "component": component,
                        "lag_frames": lag,
                        "median_dt_ms": float(np.median(dt) * 1000.0) if dt.size else float("nan"),
                        "autocorrelation": correlation,
                        "samples": len(values),
                    }
                )
    return result


def missingness_metrics(truth: list[dict], observed: list[dict]) -> list[dict]:
    observed_keys = {
        (row["motion"], row["run"], int(row["frame_seq"]), int(row["timestamp_ns"]), int(row["slot"]))
        for row in observed
    }
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in truth:
        if finite(row.get("truth_facing_score")) > 0.0:
            grouped[(row["motion"], row["run"], int(row["slot"]))].append(row)
    result = []
    for (motion, run, slot), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["timestamp_ns"]))
        present = np.asarray(
            [
                (motion, run, int(row["frame_seq"]), int(row["timestamp_ns"]), slot) in observed_keys
                for row in ordered
            ],
            dtype=bool,
        )
        timestamps = np.asarray([int(row["timestamp_ns"]) * 1e-9 for row in ordered])
        gaps: list[float] = []
        start = None
        for index, is_present in enumerate(present):
            if not is_present and start is None:
                start = index
            if is_present and start is not None:
                gaps.append(float(timestamps[index] - timestamps[start]))
                start = None
        if start is not None and len(timestamps):
            typical_dt = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.0
            gaps.append(float(timestamps[-1] - timestamps[start] + typical_dt))
        gap_values = np.asarray(gaps, dtype=float)
        result.append(
            {
                "motion": motion,
                "run": run,
                "slot": slot,
                "front_facing_truth_frames": int(len(ordered)),
                "observed_frames": int(present.sum()),
                "coverage": float(present.mean()) if present.size else float("nan"),
                "missing_streaks": int(len(gaps)),
                "missing_streak_p50_ms": float(np.percentile(gap_values, 50) * 1000.0)
                if gap_values.size
                else 0.0,
                "missing_streak_p90_ms": float(np.percentile(gap_values, 90) * 1000.0)
                if gap_values.size
                else 0.0,
                "missing_streak_max_ms": float(np.max(gap_values) * 1000.0) if gap_values.size else 0.0,
            }
        )
    return result


def harmonic_holdout_metrics(observed: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in observed:
        actual_scale = round(finite(row.get("actual_radial_scale"), float(row["scale"])), 3)
        grouped[(row["motion"], float(row["distance_m"]), actual_scale, int(row["slot"]))].append(row)
    result = []
    for (motion, distance_m, scale, slot), rows in sorted(grouped.items()):
        repeats = sorted({int(row["repeat"]) for row in rows})
        if len(repeats) < 3:
            continue
        held_repeat = repeats[-1]
        train = [row for row in rows if int(row["repeat"]) != held_repeat]
        test = [row for row in rows if int(row["repeat"]) == held_repeat]
        if len(train) < 50 or len(test) < 20:
            continue
        train_error = np.asarray([[row["error_u_deg"], row["error_v_deg"]] for row in train])
        test_error = np.asarray([[row["error_u_deg"], row["error_v_deg"]] for row in test])
        for phase_source in ("phase_rad", "pnp_yaw_absolute_rad"):
            train_phase = np.asarray([finite(row.get(phase_source)) for row in train])
            test_phase = np.asarray([finite(row.get(phase_source)) for row in test])
            valid_train = np.isfinite(train_phase)
            valid_test = np.isfinite(test_phase)
            if valid_train.sum() < 50 or valid_test.sum() < 20:
                continue
            weights = fit_harmonic(train_phase[valid_train], train_error[valid_train])
            predicted_bias = harmonic_design(test_phase[valid_test]) @ weights
            raw = angular_error(test_error[valid_test, 0], test_error[valid_test, 1])
            corrected = angular_error(
                test_error[valid_test, 0] - predicted_bias[:, 0],
                test_error[valid_test, 1] - predicted_bias[:, 1],
            )
            result.append(
                {
                    "motion": motion,
                    "distance_m": distance_m,
                    "scale": scale,
                    "slot": slot,
                    "held_repeat": held_repeat,
                    "phase_source": "truth_phase_oracle" if phase_source == "phase_rad" else "observed_pnp_yaw",
                    "train_samples": int(valid_train.sum()),
                    "test_samples": int(valid_test.sum()),
                    "raw_p50_deg": float(np.percentile(raw, 50)),
                    "raw_p95_deg": float(np.percentile(raw, 95)),
                    "corrected_p50_deg": float(np.percentile(corrected, 50)),
                    "corrected_p95_deg": float(np.percentile(corrected, 95)),
                    "p95_improvement_fraction": float(
                        1.0 - np.percentile(corrected, 95) / max(np.percentile(raw, 95), 1e-12)
                    ),
                }
            )
    return result


def plot_error_by_distance(output: Path, condition: list[dict]) -> None:
    motions = sorted({row["motion"] for row in condition})
    fig, axes = plt.subplots(1, len(motions), figsize=(6.0 * len(motions), 4.8), squeeze=False, constrained_layout=True)
    for column, motion in enumerate(motions):
        ax = axes[0, column]
        rows = [row for row in condition if row["motion"] == motion]
        for scale, marker in zip(sorted({row["scale"] for row in rows}), ("o", "s", "^")):
            values = []
            for distance in sorted({row["distance_m"] for row in rows}):
                cell = [row["angular_error_deg_p95"] for row in rows if row["scale"] == scale and row["distance_m"] == distance]
                values.append((distance, float(np.median(cell))))
            ax.plot([x for x, _ in values], [y for _, y in values], marker=marker, linewidth=1.8, label=f"radius {scale:g}")
        ax.set_title(motion)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Observation–truth angular error P95 (deg)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Observation error across distance and radius")
    fig.savefig(output / "error_by_distance_radius.png", dpi=260, bbox_inches="tight")
    fig.savefig(output / "error_by_distance_radius.svg", bbox_inches="tight")
    plt.close(fig)


def plot_autocorrelation(output: Path, rows: list[dict]) -> None:
    motions = sorted({row["motion"] for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    components = ("error_u_deg", "error_v_deg", "signed_depth_error_m")
    labels = ("Horizontal angular error", "Vertical angular error", "Signed depth error")
    for ax, component, label in zip(axes, components, labels):
        for motion in motions:
            color = COLOR.get(motion, None)
            medians = []
            p10 = []
            p90 = []
            for lag in LAGS:
                values = np.asarray(
                    [row["autocorrelation"] for row in rows if row["motion"] == motion and row["component"] == component and row["lag_frames"] == lag],
                    dtype=float,
                )
                values = values[np.isfinite(values)]
                medians.append(float(np.median(values)))
                p10.append(float(np.percentile(values, 10)))
                p90.append(float(np.percentile(values, 90)))
            ax.plot(LAGS, medians, marker="o", color=color, label=motion)
            ax.fill_between(LAGS, p10, p90, color=color, alpha=0.14)
        ax.axhline(0.0, color="#666666", linewidth=0.8)
        # Matplotlib 2.2 uses the legacy ``basex`` keyword.
        ax.set_xscale("log", basex=2)
        ax.set_xticks(LAGS)
        ax.set_xticklabels([str(lag) for lag in LAGS])
        ax.set_xlabel("Lag (observed frames)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(output / "error_autocorrelation.png", dpi=260, bbox_inches="tight")
    fig.savefig(output / "error_autocorrelation.svg", bbox_inches="tight")
    plt.close(fig)


def plot_harmonic_improvement(output: Path, rows: list[dict]) -> None:
    motions = sorted({row["motion"] for row in rows})
    sources = ("observed_pnp_yaw", "truth_phase_oracle")
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    positions = np.arange(len(motions), dtype=float)
    width = 0.34
    for index, source in enumerate(sources):
        values = [
            [row["p95_improvement_fraction"] for row in rows if row["motion"] == motion and row["phase_source"] == source]
            for motion in motions
        ]
        medians = [float(np.median(value)) for value in values]
        low = [float(np.percentile(value, 10)) for value in values]
        high = [float(np.percentile(value, 90)) for value in values]
        x = positions + (index - 0.5) * width
        ax.bar(x, medians, width=width, color=("#56B4E9", "#E69F00")[index], label=source.replace("_", " "))
        ax.errorbar(x, medians, yerr=[np.asarray(medians) - low, np.asarray(high) - medians], fmt="none", color="#222222", capsize=3)
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(motions)
    ax.set_ylabel("Held-repeat P95 error reduction fraction")
    ax.set_title("Learnable periodic observation bias (median; P10–P90 across cells)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output / "harmonic_bias_learnability.png", dpi=260, bbox_inches="tight")
    fig.savefig(output / "harmonic_bias_learnability.svg", bbox_inches="tight")
    plt.close(fig)


def build_report(
    output: Path,
    observed: list[dict],
    condition: list[dict],
    dependence: list[dict],
    autocorrelation: list[dict],
    missingness: list[dict],
    harmonic: list[dict],
) -> dict:
    aggregate: dict[str, dict] = {}
    for motion in sorted({row["motion"] for row in observed}):
        rows = [row for row in observed if row["motion"] == motion]
        angular = np.asarray([row["angular_error_deg"] for row in rows])
        depth = np.abs(np.asarray([row["signed_depth_error_m"] for row in rows]))
        coverage = np.asarray([row["coverage"] for row in missingness if row["motion"] == motion])
        gap = np.asarray([row["missing_streak_p90_ms"] for row in missingness if row["motion"] == motion])
        lag1 = np.asarray(
            [row["autocorrelation"] for row in autocorrelation if row["motion"] == motion and row["component"] in ("error_u_deg", "error_v_deg") and row["lag_frames"] == 1]
        )
        observed_phase = np.asarray(
            [row["p95_improvement_fraction"] for row in harmonic if row["motion"] == motion and row["phase_source"] == "observed_pnp_yaw"]
        )
        oracle_phase = np.asarray(
            [row["p95_improvement_fraction"] for row in harmonic if row["motion"] == motion and row["phase_source"] == "truth_phase_oracle"]
        )
        aggregate[motion] = {
            **percentiles(angular, "angular_error_deg"),
            **percentiles(depth, "abs_depth_error_m"),
            "median_front_facing_coverage": float(np.nanmedian(coverage)),
            "median_missing_streak_p90_ms": float(np.nanmedian(gap)),
            "median_lag1_angular_error_autocorrelation": float(np.nanmedian(lag1)),
            "median_observed_yaw_harmonic_p95_improvement_fraction": float(np.nanmedian(observed_phase)),
            "median_truth_phase_oracle_p95_improvement_fraction": float(np.nanmedian(oracle_phase)),
        }
    summary = {
        "kind": "observation_truth_error_structure",
        "prediction": False,
        "truth_policy": "truth is label/audit only; never an online model input",
        "aggregate": aggregate,
        "condition_cells": len(condition),
        "dependence_rows": len(dependence),
        "autocorrelation_rows": len(autocorrelation),
        "missingness_rows": len(missingness),
        "harmonic_holdout_rows": len(harmonic),
    }
    report_lines = [
        "# Observation–truth error structure",
        "",
        "Truth is treated as the deterministic label and acceptance reference. No result in this report questions truth fidelity.",
        "",
        "## Aggregate evidence",
        "",
    ]
    for motion, values in aggregate.items():
        report_lines.extend(
            [
                f"### {motion}",
                "",
                f"- Samples: {values['angular_error_deg_n']:,}",
                f"- Angular error P50/P90/P95: {values['angular_error_deg_p50']:.4f}/{values['angular_error_deg_p90']:.4f}/{values['angular_error_deg_p95']:.4f} deg",
                f"- Absolute depth error P50/P95: {values['abs_depth_error_m_p50']:.4f}/{values['abs_depth_error_m_p95']:.4f} m",
                f"- Median front-facing observation coverage: {values['median_front_facing_coverage']:.3f}",
                f"- Median P90 missing streak: {values['median_missing_streak_p90_ms']:.1f} ms",
                f"- Median one-frame angular-error autocorrelation: {values['median_lag1_angular_error_autocorrelation']:.3f}",
                f"- Held-repeat P95 reduction from observed-PnP-yaw harmonics: {values['median_observed_yaw_harmonic_p95_improvement_fraction']:.3f}",
                f"- Oracle truth-phase harmonic upper bound: {values['median_truth_phase_oracle_p95_improvement_fraction']:.3f}",
                "",
            ]
        )
    report_lines.extend(
        [
            "## Interpretation boundary",
            "",
            "- Strong autocorrelation or repeat-held harmonic improvement demonstrates learnable systematic structure, not deployment readiness.",
            "- Heavy tails, distance/facing dependence, and long missing streaks violate a fixed-Gaussian, fixed-rate Kalman assumption.",
            "- The truth-phase harmonic result is an oracle upper bound. The observed-PnP-yaw result is the corresponding observation-domain probe.",
            "- Physical slot labels are assigned offline for analysis; runtime association remains a separate interface problem.",
        ]
    )
    (output / "OBSERVATION_ERROR_STRUCTURE_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    inputs = parse_inputs(args.analysis)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_truth: list[dict] = []
    all_observed: list[dict] = []
    all_run_metrics: list[dict] = []
    sources = {}
    for label, path in inputs:
        truth, observed, run_metrics, summary = load_dataset(label, path)
        all_truth.extend(truth)
        all_observed.extend(observed)
        all_run_metrics.extend(run_metrics)
        source_files = [path / "analysis_summary.json", path / "truth_points.jsonl", path / "observed_points.jsonl", path / "run_metrics.csv"]
        sources[label] = {
            "analysis_dir": str(path),
            "analysis_summary": summary,
            "files": {str(file): sha256_file(file) for file in source_files},
        }
    condition = condition_metrics(all_observed)
    dependence = dependence_metrics(all_observed)
    autocorrelation = autocorrelation_metrics(all_observed)
    missingness = missingness_metrics(all_truth, all_observed)
    harmonic = harmonic_holdout_metrics(all_observed)
    write_csv(output / "condition_error_metrics.csv", condition)
    write_csv(output / "error_dependence_metrics.csv", dependence)
    write_csv(output / "temporal_autocorrelation_metrics.csv", autocorrelation)
    write_csv(output / "missingness_metrics.csv", missingness)
    write_csv(output / "harmonic_holdout_metrics.csv", harmonic)
    write_csv(output / "source_run_metrics.csv", all_run_metrics)
    plot_error_by_distance(output, condition)
    plot_autocorrelation(output, autocorrelation)
    plot_harmonic_improvement(output, harmonic)
    summary = build_report(output, all_observed, condition, dependence, autocorrelation, missingness, harmonic)
    summary["sources"] = sources
    script = Path(__file__).resolve()
    summary["script"] = {"path": str(script), "sha256": sha256_file(script)}
    with (output / "observation_error_structure_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    retention = {
        "classification": "long_term_private_evidence",
        "deletion_allowed": False,
        "reason": "Method-selection evidence derived from protected high-rate captures.",
        "script": summary["script"],
        "source_labels": [label for label, _ in inputs],
        "artifacts": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    with (output / "retention_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(retention, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
