#!/usr/bin/env python3
"""Summarize and visualize frozen large-scale combined-motion evidence."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GATE_MM = 55.0
COLORS = {
    "hold": "#7F7F7F",
    "same_slot_cv_400ms": "#0072B2",
    "direct_joint": "#D55E00",
    "direct_oracle_omega": "#009E73",
    "clean": "#009E73",
    "pnp": "#D55E00",
    "wide_6mm": "#0072B2",
    "precision_16mm": "#CC79A7",
}
LINESTYLES = {0.05: "-", 0.10: "--", 0.20: "-."}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_gzip_csv(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return pd.read_csv(handle)


def read_csv(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        return pd.read_csv(handle)


def qstats(values: Iterable[float], scale: float = 1.0) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64) * scale
    return {
        "n": int(array.size),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def ecdf(ax: Any, values_mm: np.ndarray, *, label: str, color: str, linestyle: str = "-") -> None:
    values = np.sort(np.maximum(np.asarray(values_mm, dtype=np.float64), 0.01))
    probability = np.arange(1, len(values) + 1, dtype=np.float64) / len(values)
    ax.step(values, probability, where="post", label=label, color=color,
            linestyle=linestyle, linewidth=1.5)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: Any, label: str) -> None:
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top")


def add_gate(ax: Any) -> None:
    ax.axvline(GATE_MM, color="#000000", linestyle=":", linewidth=1.0,
               label="55 mm diagnostic gate")


def current_measurement_errors(prediction: pd.DataFrame) -> pd.DataFrame:
    keys = ["session_id", "timestamp_ns", "sample_stratum", "horizon_s"]
    hold = prediction[
        (prediction.method == "hold")
        & (prediction.sample_stratum == "regular_grid")
        & (prediction.horizon_s == 0.05)
    ]
    metadata = [
        "camera_profile_id",
        "distance_bin",
        "linear_speed_bin",
        "absolute_yaw_rate_bin",
        "primary_slot",
    ]
    components = ["error_tracker_x_m", "error_tracker_y_m", "error_tracker_z_m"]
    pnp = hold[hold.input_domain == "pnp"][keys + metadata + components]
    clean = hold[hold.input_domain == "clean"][keys + components]
    merged = pnp.merge(clean, on=keys, suffixes=("_pnp", "_clean"))
    for axis in "xyz":
        merged[f"measurement_{axis}_m"] = (
            merged[f"error_tracker_{axis}_m_pnp"]
            - merged[f"error_tracker_{axis}_m_clean"]
        )
    merged["measurement_cross_depth_m"] = np.hypot(
        merged.measurement_y_m, merged.measurement_z_m
    )
    merged["measurement_depth_abs_m"] = merged.measurement_x_m.abs()
    merged["measurement_3d_m"] = np.sqrt(
        merged.measurement_x_m**2
        + merged.measurement_y_m**2
        + merged.measurement_z_m**2
    )
    return merged


def prediction_group_summary(
    frame: pd.DataFrame, keys: list[str]
) -> list[dict[str, Any]]:
    rows = []
    for key, group in frame.groupby(keys):
        if not isinstance(key, tuple):
            key = (key,)
        item = dict(zip(keys, key))
        item.update(
            {
                f"cross_depth_mm_{name}": value
                for name, value in qstats(group.error_cross_depth_m, 1000.0).items()
            }
        )
        item["within_55mm_rate"] = float(
            np.mean(group.error_cross_depth_m <= GATE_MM / 1000.0)
        )
        item["session_count"] = int(group.session_id.nunique())
        rows.append(item)
    return rows


def make_distribution_figure(
    prediction: pd.DataFrame, measurement: pd.DataFrame, output: Path
) -> list[Path]:
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))

    regular_pnp = prediction[
        (prediction.sample_stratum == "regular_grid")
        & (prediction.future_regime == "constant_twist")
        & (prediction.input_domain == "pnp")
        & (prediction.horizon_s == 0.20)
    ]
    for method, label in (
        ("hold", "Hold"),
        ("same_slot_cv_400ms", "Same-slot CV (400 ms)"),
        ("direct_joint", "Direct joint model"),
    ):
        values = regular_pnp[regular_pnp.method == method].error_cross_depth_m * 1000
        ecdf(axes[0, 0], values.values, label=label, color=COLORS[method])
    add_gate(axes[0, 0])
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlim(1, 20000)
    axes[0, 0].set_title("Regular constant-twist, PnP input, 200 ms")
    axes[0, 0].set_xlabel("Cross-depth miss (mm)")
    axes[0, 0].set_ylabel("Empirical cumulative probability")
    axes[0, 0].legend(frameon=False, loc="lower right")
    panel_label(axes[0, 0], "A")

    clean = prediction[
        (prediction.sample_stratum == "regular_grid")
        & (prediction.future_regime == "constant_twist")
        & (prediction.input_domain == "clean")
        & (prediction.method == "direct_joint")
    ]
    for horizon in (0.05, 0.10, 0.20):
        values = clean[clean.horizon_s == horizon].error_cross_depth_m * 1000
        ecdf(
            axes[0, 1],
            values.values,
            label=f"{int(horizon * 1000)} ms",
            color="#0072B2",
            linestyle=LINESTYLES[horizon],
        )
    add_gate(axes[0, 1])
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlim(0.01, 1000)
    axes[0, 1].set_title("Regular constant-twist, clean history")
    axes[0, 1].set_xlabel("Cross-depth miss (mm)")
    axes[0, 1].set_ylabel("Empirical cumulative probability")
    axes[0, 1].legend(frameon=False, loc="lower right")
    panel_label(axes[0, 1], "B")

    for profile, label in (
        ("wide_6mm", "Wide 6 mm"),
        ("precision_16mm", "Precision 16 mm"),
    ):
        values = (
            measurement[measurement.camera_profile_id == profile]
            .measurement_cross_depth_m
            * 1000
        )
        ecdf(
            axes[1, 0], values.values, label=f"{label} (n={len(values)})",
            color=COLORS[profile]
        )
    add_gate(axes[1, 0])
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlim(1, 20000)
    axes[1, 0].set_title("Current PnP observation error at regular anchors")
    axes[1, 0].set_xlabel("Cross-depth observation error (mm)")
    axes[1, 0].set_ylabel("Empirical cumulative probability")
    axes[1, 0].legend(frameon=False, loc="lower right")
    panel_label(axes[1, 0], "C")

    stress = prediction[
        (prediction.sample_stratum == "reversal_stress")
        & (prediction.future_regime == "cross_reversal")
        & (prediction.method == "direct_joint")
        & (prediction.horizon_s == 0.20)
    ]
    for domain, label in (("clean", "Clean history"), ("pnp", "PnP history")):
        values = stress[stress.input_domain == domain].error_cross_depth_m * 1000
        ecdf(
            axes[1, 1], values.values, label=f"{label} (n={len(values)})",
            color=COLORS[domain]
        )
    add_gate(axes[1, 1])
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlim(1, 3000)
    axes[1, 1].set_title("Reversal stress set, 200 ms")
    axes[1, 1].set_xlabel("Cross-depth miss (mm)")
    axes[1, 1].set_ylabel("Empirical cumulative probability")
    axes[1, 1].legend(frameon=False, loc="lower right")
    panel_label(axes[1, 1], "D")

    for ax in axes.ravel():
        ax.set_ylim(0, 1.01)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    paths = [output / "combined_motion_generalization_distributions.png",
             output / "combined_motion_generalization_distributions.pdf"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def heatmap(
    ax: Any,
    matrix: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    *,
    title: str,
    colorbar_label: str,
    vmax: float | None = None,
    fmt: str = ".0f",
) -> None:
    image = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                fraction = value / (vmax if vmax else np.nanmax(matrix))
                color = "white" if fraction > 0.55 else "black"
                ax.text(column, row, format(value, fmt), ha="center", va="center",
                        color=color, fontsize=7)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)


def make_condition_figure(
    prediction: pd.DataFrame, anchors: pd.DataFrame, output: Path
) -> list[Path]:
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    horizons = [0.05, 0.10, 0.20]
    horizon_labels = ["50", "100", "200"]
    main = prediction[
        (prediction.sample_stratum == "regular_grid")
        & (prediction.future_regime == "constant_twist")
        & (prediction.input_domain == "pnp")
        & (prediction.method == "direct_joint")
    ]
    distance_matrix = np.full((3, 3), np.nan)
    for distance in range(3):
        for column, horizon in enumerate(horizons):
            values = main[
                (main.distance_bin == distance) & (main.horizon_s == horizon)
            ].error_cross_depth_m
            distance_matrix[distance, column] = np.quantile(values, 0.95) * 1000
    heatmap(
        axes[0, 0], distance_matrix, horizon_labels,
        ["Near", "Medium", "Far"],
        title="PnP direct model by distance bin",
        colorbar_label="Cross-depth P95 (mm)", vmax=650,
    )
    axes[0, 0].set_xlabel("Prediction horizon (ms)")
    panel_label(axes[0, 0], "A")

    direction_matrix = np.full((8, 3), np.nan)
    for direction in range(8):
        for column, horizon in enumerate(horizons):
            values = main[
                (main.direction_sector == direction) & (main.horizon_s == horizon)
            ].error_cross_depth_m
            direction_matrix[direction, column] = np.quantile(values, 0.95) * 1000
    heatmap(
        axes[0, 1], direction_matrix, horizon_labels,
        [str(value) for value in range(8)],
        title="PnP direct model by direction sector",
        colorbar_label="Cross-depth P95 (mm)", vmax=650,
    )
    axes[0, 1].set_xlabel("Prediction horizon (ms)")
    axes[0, 1].set_ylabel("Direction sector")
    panel_label(axes[0, 1], "B")

    clean = prediction[
        (prediction.sample_stratum == "regular_grid")
        & (prediction.input_domain == "clean")
        & (prediction.method == "direct_joint")
    ]
    coupled = [("[0,1)", "0–1 / 0–5"), ("[1,2)", "1–2 / 5–10"),
               ("[2,3.1]", "2–3 / 10–15")]
    pass_matrix = np.full((3, 3), np.nan)
    for row, (speed, _) in enumerate(coupled):
        for column, horizon in enumerate(horizons):
            values = clean[
                (clean.linear_speed_bin == speed) & (clean.horizon_s == horizon)
            ].error_cross_depth_m
            pass_matrix[row, column] = np.mean(values <= GATE_MM / 1000.0) * 100
    heatmap(
        axes[1, 0], pass_matrix, horizon_labels,
        [label for _, label in coupled],
        title="Clean direct model: 55 mm point-level pass rate",
        colorbar_label="Pass rate (%)", vmax=100, fmt=".1f",
    )
    axes[1, 0].set_xlabel("Prediction horizon (ms)")
    axes[1, 0].set_ylabel("Speed (m/s) / |yaw rate| (rad/s)")
    panel_label(axes[1, 0], "C")

    coverage = (
        anchors.groupby(["sample_stratum", "status"]).size().unstack(fill_value=0)
    )
    status_order = [
        "evaluated_common_anchor",
        "no_current_observation",
        "insufficient_direct_history",
    ]
    labels = ["Evaluated", "No current PnP", "<4 s observed history"]
    bottom = np.zeros(len(coverage), dtype=np.float64)
    x = np.arange(len(coverage))
    for status, label, color in zip(
        status_order,
        labels,
        ["#009E73", "#D55E00", "#E69F00"],
    ):
        values = coverage.get(status, pd.Series(0, index=coverage.index)).values
        rates = values / coverage.sum(axis=1).values * 100
        axes[1, 1].bar(x, rates, bottom=bottom, label=label, color=color, width=0.65)
        bottom += rates
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(["Regular grid", "Reversal stress"])
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].set_ylabel("Planned anchors (%)")
    axes[1, 1].set_title("Estimator availability is part of the result")
    axes[1, 1].legend(frameon=False, loc="upper center", ncol=1)
    panel_label(axes[1, 1], "D")

    fig.tight_layout()
    paths = [output / "combined_motion_conditions_and_coverage.png",
             output / "combined_motion_conditions_and_coverage.pdf"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    evidence = args.evidence.resolve()
    output = evidence / "analysis"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {output}")
    output.mkdir()
    manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("formal_complete_run", False):
        raise ValueError("analysis requires a complete formal run")
    prediction = read_gzip_csv(evidence / "prediction_distribution.csv.gz")
    fit = read_gzip_csv(evidence / "fit_distribution.csv.gz")
    anchors = read_csv(evidence / "anchor_coverage.csv")
    audit = read_csv(evidence / "session_data_audit.csv")
    measurement = current_measurement_errors(prediction)

    candidate_counts: Counter[int] = Counter()
    profile_counts: Counter[str] = Counter()
    for value in audit.candidate_count_histogram_json:
        candidate_counts.update({int(k): int(v) for k, v in json.loads(value).items()})
    for value in audit.camera_profile_histogram_json:
        profile_counts.update({str(k): int(v) for k, v in json.loads(value).items()})

    regular_constant = prediction[
        (prediction.sample_stratum == "regular_grid")
        & (prediction.future_regime == "constant_twist")
    ]
    reversal_cross = prediction[
        (prediction.sample_stratum == "reversal_stress")
        & (prediction.future_regime == "cross_reversal")
    ]
    micro_rows = prediction_group_summary(
        pd.concat([regular_constant, reversal_cross]),
        ["sample_stratum", "future_regime", "input_domain", "method", "horizon_s"],
    )
    with (output / "primary_result_table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(micro_rows[0]))
        writer.writeheader()
        writer.writerows(micro_rows)

    session_pass = []
    regular = prediction[prediction.sample_stratum == "regular_grid"]
    for key, group in regular.groupby(["input_domain", "method", "horizon_s"]):
        domain, method, horizon = key
        p95 = group.groupby("session_id").error_cross_depth_m.quantile(0.95)
        session_pass.append(
            {
                "input_domain": domain,
                "method": method,
                "horizon_s": float(horizon),
                "sessions_with_rows": int(len(p95)),
                "session_p95_within_55mm_count": int(np.sum(p95 <= GATE_MM / 1000.0)),
                "session_p95_within_55mm_rate": float(np.mean(p95 <= GATE_MM / 1000.0)),
                "session_p95_mm_distribution": qstats(p95, 1000.0),
            }
        )

    omega_rows = []
    for domain, group in fit[fit.method == "direct_joint"].groupby("input_domain"):
        omega_rows.append(
            {
                "input_domain": domain,
                "absolute_omega_error_rad_s": qstats(group.absolute_omega_error_rad_s),
                "sign_correct_rate": float(np.mean(group.omega_sign_correct)),
            }
        )
    measurement_rows = []
    for profile, group in measurement.groupby("camera_profile_id"):
        measurement_rows.append(
            {
                "camera_profile_id": profile,
                "cross_depth_error_mm": qstats(group.measurement_cross_depth_m, 1000.0),
                "absolute_depth_error_mm": qstats(group.measurement_depth_abs_m, 1000.0),
                "cross_depth_within_55mm_rate": float(
                    np.mean(group.measurement_cross_depth_m <= GATE_MM / 1000.0)
                ),
            }
        )
    coverage_rows = []
    for (stratum, status), count in anchors.groupby(
        ["sample_stratum", "status"]
    ).size().items():
        total = int(np.sum(anchors.sample_stratum == stratum))
        coverage_rows.append(
            {
                "sample_stratum": stratum,
                "status": status,
                "count": int(count),
                "planned": total,
                "rate": float(count / total),
            }
        )

    summary = {
        "schema_version": "combined-motion-large-scale-analysis-v1",
        "source_manifest_sha256": sha256(evidence / "manifest.json"),
        "data_audit": {
            "session_count": int(audit.session_id.nunique()),
            "truth_frames_post_start": int(audit.truth_frames_post_start.sum()),
            "truth_frames_with_observation": int(audit.truth_frames_with_observation.sum()),
            "truth_frames_without_observation": int(audit.truth_frames_without_observation.sum()),
            "associated_observation_event_count": int(
                audit.associated_observation_event_count.sum()
            ),
            "candidate_count_histogram": dict(sorted(candidate_counts.items())),
            "camera_profile_histogram": dict(sorted(profile_counts.items())),
            "session_observation_coverage_distribution": qstats(
                audit.observation_frame_coverage
            ),
            "over_four_candidate_frame_count": int(
                audit.observation_over_four_valid.sum()
            ),
            "ambiguous_assignment_count": int(
                audit.observation_ambiguous_assignment.sum()
            ),
            "duplicate_observation_timestamp_count": int(
                audit.observation_duplicate_timestamp.sum()
            ),
            "unmatched_observation_timestamp_count": int(
                audit.observation_unmatched_timestamp.sum()
            ),
            "transform_unmatched_count": int(
                audit.observation_transform_unmatched.sum()
            ),
            "zero_evaluated_regular_session_count": int(
                np.sum(audit.evaluated_regular_anchor_count == 0)
            ),
            "zero_evaluated_reversal_stress_session_count": int(
                np.sum(audit.evaluated_reversal_stress_anchor_count == 0)
            ),
        },
        "anchor_coverage": coverage_rows,
        "current_pnp_measurement_error_by_profile": measurement_rows,
        "direct_joint_omega_fit": omega_rows,
        "regular_session_macro": session_pass,
        "design_warning": {
            "absolute_yaw_rate_and_linear_speed_bins_are_perfectly_confounded": True,
            "speed_to_abs_yaw_rate_bin_mapping": {
                "[0,1) m/s": "[0,5) rad/s",
                "[1,2) m/s": "[5,10) rad/s",
                "[2,3.1] m/s": "[10,15.1] rad/s",
            },
            "separate_speed_and_yaw_rate_causal_effects_identifiable": False,
        },
        "primary_result_table": "primary_result_table.csv",
        "complete_distribution_source": "../prediction_distribution.csv.gz",
    }
    write_json(output / "analysis_summary.json", summary)

    figure_paths = []
    figure_paths.extend(make_distribution_figure(prediction, measurement, output))
    figure_paths.extend(make_condition_figure(prediction, anchors, output))
    artifacts = []
    for path in sorted(output.iterdir()):
        if path.name == "analysis_manifest.json":
            continue
        artifacts.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    write_json(
        output / "analysis_manifest.json",
        {
            "schema_version": "combined-motion-large-scale-analysis-manifest-v1",
            "source_evidence": str(evidence),
            "source_manifest_sha256": sha256(evidence / "manifest.json"),
            "complete_prediction_distribution_retained": True,
            "figures_are_distribution_and_condition_summaries_not_replacements_for_rows": True,
            "artifacts": artifacts,
        },
    )


if __name__ == "__main__":
    main()
