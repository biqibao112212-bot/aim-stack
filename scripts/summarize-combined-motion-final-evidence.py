#!/usr/bin/env python3
"""Create final tables and publication-quality figures from frozen evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = (
    "hold",
    "same_slot_world_cv",
    "v1_isotropic_single_window",
    "los_raw_omega",
    "los_memory31",
    "direct_joint_omega_phase",
    "direct_oracle_omega",
)
METHOD_LABELS = {
    "hold": "Hold",
    "same_slot_world_cv": "Same-slot CV",
    "v1_isotropic_single_window": "V1 isotropic",
    "los_raw_omega": "LOS + raw ω",
    "los_memory31": "LOS + ω memory",
    "direct_joint_omega_phase": "Direct joint",
    "direct_oracle_omega": "Direct + oracle ω",
}
COLORS = {
    "hold": "#7F7F7F",
    "same_slot_world_cv": "#0072B2",
    "v1_isotropic_single_window": "#D55E00",
    "los_raw_omega": "#E69F00",
    "los_memory31": "#009E73",
    "direct_joint_omega_phase": "#CC79A7",
    "direct_oracle_omega": "#000000",
}
LINESTYLES = {
    "hold": ":",
    "same_slot_world_cv": "--",
    "v1_isotropic_single_window": "-.",
    "los_raw_omega": ":",
    "los_memory31": "--",
    "direct_joint_omega_phase": "-",
    "direct_oracle_omega": "-.",
}


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


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, evidence: Path, name: str) -> list[dict[str, Any]]:
    result = []
    for suffix, kwargs in (
        ("png", {"dpi": 300}),
        ("svg", {}),
        ("pdf", {}),
    ):
        path = evidence / f"{name}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        result.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    plt.close(fig)
    return result


def empirical_cdf(
    values: pd.Series, *, scale: float = 1000.0
) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values.values.astype(np.float64) * scale)
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    return x, y


def plot_p95(data: pd.DataFrame, evidence: Path) -> list[dict[str, Any]]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for panel, (axis, regime) in enumerate(
        zip(axes, ("constant_twist", "cross_reversal"))
    ):
        subset = data[(data.input_domain == "pnp") & (data.future_regime == regime)]
        for method in METHOD_ORDER:
            active = subset[subset.method == method]
            if active.empty:
                continue
            values = [
                active[active.horizon_s == horizon].error_cross_depth_m.quantile(0.95)
                * 1000.0
                for horizon in (0.05, 0.10, 0.20)
            ]
            axis.plot(
                [50, 100, 200],
                values,
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                marker="o",
                ms=4,
                lw=1.5,
                label=METHOD_LABELS[method],
            )
        axis.axhline(55.0, color="#333333", lw=1.0, linestyle="--", label="55 mm gate")
        axis.set_yscale("log")
        axis.set_xticks([50, 100, 200])
        axis.set_xlabel("Prediction horizon (ms)")
        axis.set_title("Constant twist" if regime == "constant_twist" else "Cross reversal")
        axis.grid(axis="y", alpha=0.18)
        axis.text(-0.12, 1.04, chr(ord("A") + panel), transform=axis.transAxes, fontweight="bold")
    axes[0].set_ylabel("Cross-depth error P95 (mm, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return save_figure(fig, evidence, "final_test_cross_depth_p95")


def plot_ecdf(data: pd.DataFrame, evidence: Path) -> list[dict[str, Any]]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    shown = (
        "same_slot_world_cv",
        "v1_isotropic_single_window",
        "los_memory31",
        "direct_joint_omega_phase",
        "direct_oracle_omega",
    )
    for panel, (axis, regime) in enumerate(
        zip(axes, ("constant_twist", "cross_reversal"))
    ):
        subset = data[
            (data.input_domain == "pnp")
            & (data.future_regime == regime)
            & np.isclose(data.horizon_s, 0.20)
        ]
        for method in shown:
            values = subset[subset.method == method].error_cross_depth_m
            if values.empty:
                continue
            x, y = empirical_cdf(values)
            axis.step(
                x,
                y,
                where="post",
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                lw=1.5,
                label=f"{METHOD_LABELS[method]} (n={len(values)})",
            )
        axis.axvline(55.0, color="#333333", lw=1.0, linestyle="--")
        axis.set_xscale("log")
        axis.set_xlabel("Cross-depth error at 200 ms (mm, log scale)")
        axis.set_title("Constant twist" if regime == "constant_twist" else "Cross reversal")
        axis.grid(alpha=0.18)
        axis.legend(frameon=False, loc="lower right")
        axis.text(-0.12, 1.04, chr(ord("A") + panel), transform=axis.transAxes, fontweight="bold")
    axes[0].set_ylabel("Empirical cumulative probability")
    fig.tight_layout()
    return save_figure(fig, evidence, "final_test_cross_depth_ecdf_200ms")


def plot_trajectory(data: pd.DataFrame, evidence: Path) -> list[dict[str, Any]]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    shown = (
        "same_slot_world_cv",
        "los_memory31",
        "direct_joint_omega_phase",
    )
    for panel, (axis, regime) in enumerate(
        zip(axes, ("constant_twist", "cross_reversal"))
    ):
        subset = data[
            (data.input_domain == "pnp")
            & (data.future_regime == regime)
            & np.isclose(data.horizon_s, 0.20)
        ]
        truth = subset[subset.method == "direct_joint_omega_phase"]
        axis.plot(
            truth.truth_x_m,
            truth.truth_y_m,
            color="black",
            marker="o",
            ms=3,
            lw=1.0,
            label=f"Future truth (n={len(truth)})",
        )
        for method in shown:
            active = subset[subset.method == method]
            axis.scatter(
                active.prediction_x_m,
                active.prediction_y_m,
                color=COLORS[method],
                marker={
                    "same_slot_world_cv": "x",
                    "los_memory31": "s",
                    "direct_joint_omega_phase": "^",
                }[method],
                s=22,
                alpha=0.85,
                label=METHOD_LABELS[method],
            )
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel("World x (m)")
        axis.set_ylabel("World y (m)")
        axis.set_title("Constant twist" if regime == "constant_twist" else "Cross reversal")
        axis.grid(alpha=0.18)
        axis.legend(frameon=False, loc="best")
        axis.text(-0.12, 1.04, chr(ord("A") + panel), transform=axis.transAxes, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, evidence, "final_test_prediction_truth_overlay_200ms")


def plot_omega(fit: pd.DataFrame, evidence: Path) -> list[dict[str, Any]]:
    columns = (
        ("v1_isotropic_absolute_error_rad_s", "V1 isotropic", "#D55E00", "-."),
        ("los_raw_absolute_error_rad_s", "LOS raw", "#E69F00", ":"),
        ("los_memory31_absolute_error_rad_s", "LOS memory", "#009E73", "--"),
        ("direct_joint_absolute_error_rad_s", "Direct joint", "#CC79A7", "-"),
    )
    fig, axis = plt.subplots(figsize=(3.6, 3.0))
    for column, label, color, linestyle in columns:
        x, y = empirical_cdf(fit[column], scale=1.0)
        axis.step(x, y, where="post", color=color, linestyle=linestyle, lw=1.6, label=label)
    axis.set_xscale("log")
    axis.set_xlabel("Absolute angular-speed error (rad/s, log scale)")
    axis.set_ylabel("Empirical cumulative probability")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return save_figure(fig, evidence, "final_test_omega_error_ecdf")


def main() -> None:
    args = parse_args()
    evidence = args.evidence.resolve()
    if not (evidence / "manifest.json").exists():
        raise FileNotFoundError(f"missing frozen evidence manifest in {evidence}")
    with gzip.open(evidence / "prediction_distribution.csv.gz", "rt", encoding="utf-8") as handle:
        data = pd.read_csv(handle)
    with gzip.open(evidence / "fit_distribution.csv.gz", "rt", encoding="utf-8") as handle:
        fit = pd.read_csv(handle)
    configure_style()
    artifacts = []
    artifacts.extend(plot_p95(data, evidence))
    artifacts.extend(plot_ecdf(data, evidence))
    artifacts.extend(plot_trajectory(data, evidence))
    artifacts.extend(plot_omega(fit, evidence))
    write_json(
        evidence / "postprocess_manifest.json",
        {
            "schema_version": "combined-motion-final-evidence-postprocess-v1",
            "source_manifest_sha256": sha256(evidence / "manifest.json"),
            "prediction_distribution_sha256": sha256(
                evidence / "prediction_distribution.csv.gz"
            ),
            "fit_distribution_sha256": sha256(evidence / "fit_distribution.csv.gz"),
            "figures": artifacts,
            "style": {
                "palette": "Okabe-Ito compatible",
                "redundant_line_styles": True,
                "formats": ["png-300dpi", "svg", "pdf"],
            },
        },
    )


if __name__ == "__main__":
    main()
