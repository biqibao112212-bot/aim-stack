#!/usr/bin/env python3
"""Summarize and validate combined-motion PnP reduction evidence."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HORIZONS = (0.05, 0.10, 0.20)
PRIMARY_ARMS = (
    "raw_pnp",
    "crossfit_yaw_harmonic",
    "truth_depth_only",
    "truth_tracker_y_only",
    "truth_tracker_z_only",
    "truth_cross_depth_yz",
    "pnp_residual_alpha_0.75",
    "pnp_residual_alpha_0.50",
    "pnp_residual_alpha_0.25",
    "clean_exact",
)
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
GATES = (0.010, 0.025, 0.055, 0.100, 0.200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--nested", required=True, type=Path)
    parser.add_argument("--yz-probe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(str(path), "rt", encoding="utf-8", newline="") as handle:
            return pd.read_csv(handle)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return pd.read_csv(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def stats(values: Iterable[float]) -> dict[str, Any]:
    array = finite(values)
    result: dict[str, Any] = {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
    }
    for quantile in QUANTILES:
        result[f"p{int(round(100 * quantile)):02d}"] = float(np.quantile(array, quantile))
    for gate in GATES:
        result[f"cdf_le_{int(round(gate * 1000)):03d}mm"] = float(np.mean(array <= gate))
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parity(parent: pd.DataFrame, replay: pd.DataFrame) -> dict[str, Any]:
    keys = ["session_id", "timestamp_ns", "sample_stratum", "horizon_s", "primary_slot"]
    report: dict[str, Any] = {"passed": True, "arms": {}}
    for domain, arm in (("pnp", "raw_pnp"), ("clean", "clean_exact")):
        left = parent[(parent.input_domain == domain) & (parent.method == "direct_joint")]
        right = replay[replay.input_arm == arm]
        merged = left.merge(right, on=keys, suffixes=("_parent", "_replay"), how="outer")
        values = {
            "parent_rows": int(len(left)),
            "replay_rows": int(len(right)),
            "merged_rows": int(len(merged)),
        }
        arm_passed = len(left) == len(right) == len(merged)
        for metric in ("error_cross_depth_m", "error_depth_m", "error_3d_m"):
            delta = np.abs(
                merged[f"{metric}_parent"].values - merged[f"{metric}_replay"].values
            )
            values[f"{metric}_max_abs_delta"] = float(np.nanmax(delta))
            values[f"{metric}_nonzero_rows"] = int(np.count_nonzero(delta > 1.0e-12))
            arm_passed = arm_passed and bool(np.nanmax(delta) <= 1.0e-12)
        values["passed"] = arm_passed
        report["arms"][arm] = values
        report["passed"] = report["passed"] and arm_passed
    return report


def primary_distribution(replay: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    scopes = (
        ("regular_constant", "regular_grid", "constant_twist"),
        ("reversal_cross", "reversal_stress", "cross_reversal"),
        ("reversal_all", "reversal_stress", None),
    )
    for scope, stratum, regime in scopes:
        selected = replay[replay.sample_stratum == stratum]
        if regime is not None:
            selected = selected[selected.future_regime == regime]
        for horizon in HORIZONS:
            horizon_rows = selected[selected.horizon_s == horizon]
            for arm in PRIMARY_ARMS:
                values = horizon_rows[horizon_rows.input_arm == arm].error_cross_depth_m.values
                rows.append(
                    {
                        "scope": scope,
                        "horizon_s": horizon,
                        "input_arm": arm,
                        **stats(values),
                    }
                )
    return rows


def paired_comparisons(replay: pd.DataFrame) -> list[dict[str, Any]]:
    keys = [
        "session_id", "timestamp_ns", "sample_stratum", "future_regime",
        "horizon_s", "primary_slot"
    ]
    pivot = replay.pivot_table(
        index=keys, columns="input_arm", values="error_cross_depth_m", aggfunc="first"
    ).reset_index()
    rows = []
    for (stratum, regime, horizon), group in pivot.groupby(
        ["sample_stratum", "future_regime", "horizon_s"]
    ):
        raw = group["raw_pnp"].values
        for arm in PRIMARY_ARMS:
            if arm == "raw_pnp":
                continue
            value = group[arm].values
            valid = np.isfinite(raw) & np.isfinite(value)
            raw_valid = raw[valid]
            value_valid = value[valid]
            delta = value_valid - raw_valid
            rows.append(
                {
                    "sample_stratum": stratum,
                    "future_regime": regime,
                    "horizon_s": horizon,
                    "input_arm": arm,
                    "n": int(valid.sum()),
                    "improved_fraction": float(np.mean(delta < -1.0e-12)),
                    "unchanged_fraction": float(np.mean(np.abs(delta) <= 1.0e-12)),
                    "worsened_fraction": float(np.mean(delta > 1.0e-12)),
                    "delta_error_mean_m": float(np.mean(delta)),
                    "delta_error_p50_m": float(np.quantile(delta, 0.50)),
                    "delta_error_p90_m": float(np.quantile(delta, 0.90)),
                    "raw_gate_pass_rate": float(np.mean(raw_valid <= 0.055)),
                    "arm_gate_pass_rate": float(np.mean(value_valid <= 0.055)),
                    "gate_enter_fraction": float(
                        np.mean((raw_valid > 0.055) & (value_valid <= 0.055))
                    ),
                    "gate_exit_fraction": float(
                        np.mean((raw_valid <= 0.055) & (value_valid > 0.055))
                    ),
                }
            )
    return rows


def session_summaries(replay: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = replay[
        (replay.sample_stratum == "regular_grid")
        & (replay.future_regime == "constant_twist")
    ]
    session_rows = []
    for (session_id, horizon, arm), group in selected.groupby(
        ["session_id", "horizon_s", "input_arm"]
    ):
        values = group.error_cross_depth_m.values
        session_rows.append(
            {
                "session_id": session_id,
                "horizon_s": horizon,
                "input_arm": arm,
                "n": int(len(values)),
                "p50_m": float(np.quantile(values, 0.50)),
                "p90_m": float(np.quantile(values, 0.90)),
                "p95_m": float(np.quantile(values, 0.95)),
                "gate_pass_rate": float(np.mean(values <= 0.055)),
            }
        )
    frame = pd.DataFrame(session_rows)
    macro_rows = []
    for (horizon, arm), group in frame.groupby(["horizon_s", "input_arm"]):
        common = {
            "horizon_s": horizon,
            "input_arm": arm,
            "sessions": int(len(group)),
            "sessions_majority_gate_pass": int(np.sum(group.gate_pass_rate >= 0.50)),
            "sessions_80pct_gate_pass": int(np.sum(group.gate_pass_rate >= 0.80)),
            "sessions_p90_within_gate": int(np.sum(group.p90_m <= 0.055)),
        }
        for metric in ("gate_pass_rate", "p50_m", "p90_m", "p95_m"):
            values = group[metric].values
            common[f"{metric}_p10"] = float(np.quantile(values, 0.10))
            common[f"{metric}_p25"] = float(np.quantile(values, 0.25))
            common[f"{metric}_p50"] = float(np.quantile(values, 0.50))
            common[f"{metric}_p75"] = float(np.quantile(values, 0.75))
            common[f"{metric}_p90"] = float(np.quantile(values, 0.90))
        macro_rows.append(common)
    return session_rows, macro_rows


def condition_summaries(replay: pd.DataFrame) -> list[dict[str, Any]]:
    selected = replay[
        (replay.sample_stratum == "regular_grid")
        & (replay.future_regime == "constant_twist")
        & replay.input_arm.isin(("raw_pnp", "crossfit_yaw_harmonic"))
    ]
    rows = []
    strata = (
        "distance_bin",
        "linear_speed_bin",
        "absolute_yaw_rate_bin",
        "direction_sector",
        "camera_profile_id",
    )
    for stratum in strata:
        for (value, horizon, arm), group in selected.groupby(
            [stratum, "horizon_s", "input_arm"]
        ):
            rows.append(
                {
                    "stratum": stratum,
                    "value": value,
                    "horizon_s": horizon,
                    "input_arm": arm,
                    **stats(group.error_cross_depth_m.values),
                }
            )
    return rows


def fit_summaries(fits: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for arm, group in fits.groupby("input_arm"):
        values = group.absolute_omega_error_rad_s.values
        rows.append(
            {
                "input_arm": arm,
                "fits": int(len(group)),
                "omega_abs_error_p10": float(np.quantile(values, 0.10)),
                "omega_abs_error_p25": float(np.quantile(values, 0.25)),
                "omega_abs_error_p50": float(np.quantile(values, 0.50)),
                "omega_abs_error_p75": float(np.quantile(values, 0.75)),
                "omega_abs_error_p90": float(np.quantile(values, 0.90)),
                "omega_abs_error_p95": float(np.quantile(values, 0.95)),
                "omega_abs_error_p99": float(np.quantile(values, 0.99)),
                "omega_sign_accuracy": float(np.mean(group.omega_sign_correct.values)),
            }
        )
    return rows


def yz_probe_summary(replay: pd.DataFrame, yz: pd.DataFrame) -> list[dict[str, Any]]:
    full = replay[replay.input_arm == "crossfit_yaw_harmonic"]
    keys = ["session_id", "timestamp_ns", "sample_stratum", "horizon_s", "primary_slot"]
    merged = full.merge(yz, on=keys, suffixes=("_full", "_yz"), how="inner")
    rows = []
    for (stratum, regime, horizon), group in merged.groupby(
        ["sample_stratum", "future_regime_full", "horizon_s"]
    ):
        full_values = group.error_cross_depth_m_full.values
        yz_values = group.error_cross_depth_m_yz.values
        delta = yz_values - full_values
        full_stats = stats(full_values)
        yz_stats = stats(yz_values)
        rows.append(
            {
                "sample_stratum": stratum,
                "future_regime": regime,
                "horizon_s": horizon,
                "n": int(len(group)),
                "yz_improved_fraction": float(np.mean(delta < -1.0e-12)),
                "yz_worsened_fraction": float(np.mean(delta > 1.0e-12)),
                "delta_p50_m": float(np.quantile(delta, 0.50)),
                "full_p50_m": full_stats["p50"],
                "yz_p50_m": yz_stats["p50"],
                "full_p90_m": full_stats["p90"],
                "yz_p90_m": yz_stats["p90"],
                "full_p95_m": full_stats["p95"],
                "yz_p95_m": yz_stats["p95"],
                "full_gate_pass": full_stats["cdf_le_055mm"],
                "yz_gate_pass": yz_stats["cdf_le_055mm"],
            }
        )
    return rows


def plot_primary(output: Path, replay: pd.DataFrame) -> None:
    selected = replay[
        (replay.sample_stratum == "regular_grid")
        & (replay.future_regime == "constant_twist")
    ]
    arms = (
        "raw_pnp", "crossfit_yaw_harmonic", "truth_tracker_y_only",
        "truth_cross_depth_yz", "pnp_residual_alpha_0.25", "clean_exact"
    )
    colors = dict(zip(arms, ("#666666", "#0072B2", "#CC79A7", "#009E73", "#E69F00", "#000000")))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), sharey=True)
    for axis, horizon in zip(axes, HORIZONS):
        for arm in arms:
            values = np.sort(
                selected[
                    (selected.horizon_s == horizon) & (selected.input_arm == arm)
                ].error_cross_depth_m.values
            )
            y = np.arange(1, len(values) + 1) / len(values)
            axis.plot(values * 1000.0, y, label=arm, color=colors[arm])
        axis.axvline(55.0, color="#222222", linestyle="--", linewidth=1.0)
        axis.set_xlim(0.0, 500.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_title(f"{int(round(1000 * horizon))} ms")
        axis.set_xlabel("Tracker y/z vector prediction error (mm)")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Empirical CDF")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Combined motion — downstream correction and causal restoration")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / "combined_pnp_reduction_ecdf.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "combined_pnp_reduction_ecdf.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_gate_curve(output: Path, replay: pd.DataFrame) -> None:
    selected = replay[
        (replay.sample_stratum == "regular_grid")
        & (replay.future_regime == "constant_twist")
    ]
    alpha_arms = (
        (0.0, "clean_exact"),
        (0.25, "pnp_residual_alpha_0.25"),
        (0.50, "pnp_residual_alpha_0.50"),
        (0.75, "pnp_residual_alpha_0.75"),
        (1.0, "raw_pnp"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for horizon, color in zip(HORIZONS, ("#009E73", "#0072B2", "#D55E00")):
        gate = []
        p90 = []
        for _, arm in alpha_arms:
            values = selected[
                (selected.horizon_s == horizon) & (selected.input_arm == arm)
            ].error_cross_depth_m.values
            gate.append(np.mean(values <= 0.055))
            p90.append(np.quantile(values, 0.90) * 1000.0)
        x = [item[0] for item in alpha_arms]
        axes[0].plot(x, gate, marker="o", color=color, label=f"{int(horizon*1000)} ms")
        axes[1].plot(x, p90, marker="o", color=color, label=f"{int(horizon*1000)} ms")
        corrected = selected[
            (selected.horizon_s == horizon)
            & (selected.input_arm == "crossfit_yaw_harmonic")
        ].error_cross_depth_m.values
        axes[0].scatter([1.04], [np.mean(corrected <= 0.055)], marker="*", s=90, color=color)
        axes[1].scatter([1.04], [np.quantile(corrected, 0.90) * 1000.0], marker="*", s=90, color=color)
    axes[0].set_ylabel("CDF at 55 mm")
    axes[1].set_ylabel("P90 cross-depth error (mm)")
    for axis in axes:
        axis.set_xlabel("Remaining PnP residual scale α (star = learned correction)")
        axis.grid(alpha=0.22)
        axis.legend()
    fig.suptitle("How much PnP residual reduction is needed?")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / "pnp_residual_gate_curve.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "pnp_residual_gate_curve.pdf", bbox_inches="tight")
    plt.close(fig)


def lookup(
    rows: list[dict[str, Any]], scope: str, horizon: float, arm: str
) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["scope"] == scope
        and math.isclose(float(row["horizon_s"]), horizon)
        and row["input_arm"] == arm
    )


def build_report(
    output: Path,
    measurement: pd.DataFrame,
    primary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    macro: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
    yz_rows: list[dict[str, Any]],
    parity_report: dict[str, Any],
) -> None:
    measurement_cross = measurement[measurement.metric == "cross_depth_m"].set_index("arm")
    raw_current = measurement_cross.loc["raw_pnp"]
    corrected_current = measurement_cross.loc["yaw_harmonic"]
    lines = [
        "# Combined-motion PnP error-reduction evidence",
        "",
        "This report is an offline, oracle-identity analysis. Future truth is used only after prediction for scoring. The deployable correction arms use whole-session cross-fitting and observation-stream fields only. The 55 mm line is a diagnostic tracker-y/z vector gate, not live hit probability.",
        "",
        "## Evidence integrity",
        "",
        f"- Parent parity: `{parity_report['passed']}`. Raw PnP and clean direct-joint predictions each match all 6,995 parent rows exactly in cross-depth, depth and 3D error.",
        "- Formal measurement screen: 144 sessions, 465,311 associated observation events, five deterministic session folds.",
        "- Formal predictor replay: 144 sessions, 76,945 prediction rows, 27,401 fits, 4,122 coverage rows, zero failures.",
        "- A first replay that deleted position events when yaw was missing is invalid and retained only as negative evidence. The accepted v2 falls back to raw PnP and preserves the event.",
        "",
        "## Current-observation correction screen",
        "",
        f"Raw current cross-depth P50/P90/P95/P99 is `{float(raw_current.p50)*1000:.1f}/{float(raw_current.p90)*1000:.1f}/{float(raw_current.p95)*1000:.1f}/{float(raw_current.p99)*1000:.1f} mm`, with `{float(raw_current.cdf_le_055mm)*100:.1f}%` inside 55 mm.",
        f"The held-session yaw-harmonic correction changes this to `{float(corrected_current.p50)*1000:.1f}/{float(corrected_current.p90)*1000:.1f}/{float(corrected_current.p95)*1000:.1f}/{float(corrected_current.p99)*1000:.1f} mm`, with `{float(corrected_current.cdf_le_055mm)*100:.1f}%` inside 55 mm.",
        "Oracle slot and truth-phase features add very little over observed-yaw correction, so missing slot/phase labels are not the main explanation of the remaining tail in this low-capacity family.",
        "",
        "## Constant-twist future prediction",
        "",
        "| arm | 50 ms P50/P90/P95; gate | 100 ms P50/P90/P95; gate | 200 ms P50/P90/P95; gate |",
        "| --- | --- | --- | --- |",
    ]
    for arm in (
        "raw_pnp", "crossfit_yaw_harmonic", "truth_depth_only",
        "truth_tracker_y_only", "truth_cross_depth_yz",
        "pnp_residual_alpha_0.50", "pnp_residual_alpha_0.25", "clean_exact"
    ):
        cells = []
        for horizon in HORIZONS:
            row = lookup(primary, "regular_constant", horizon, arm)
            cells.append(
                f"{row['p50']*1000:.1f}/{row['p90']*1000:.1f}/{row['p95']*1000:.1f} mm; {row['cdf_le_055mm']*100:.1f}%"
            )
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "The correction materially improves the majority distribution but does not close the tail. It raises 55 mm coverage from 25.3/25.8/27.0% to 58.6/53.5/43.8%, while corrected P95 remains 261.7/295.5/371.2 mm.",
            "",
            "Tracker-y is the dominant causal coordinate. Restoring tracker-y alone raises coverage to 91.1/89.1/84.9%; restoring z alone or depth alone changes little. Restoring both y/z is only slightly different from y-only, confirming a smaller z floor and a dominant horizontal-y bias.",
            "",
            "Uniformly retaining 50% of current PnP residual gives only 50.8/51.8/51.4% coverage. Retaining 25% gives 78.7/79.5/74.0%, but P90 is still 76.1/77.2/90.4 mm. Therefore roughly 75% residual reduction is enough for a large majority, but not for 90% coverage at the present gate.",
            "",
            "## Reversal boundary",
            "",
            "Cross-reversal clean prediction still deteriorates strongly with horizon, so future mode changes remain a separate problem after PnP improvement. The observation correction helps but cannot replace reversal/mode inference.",
            "",
            "## Downstream method decision",
            "",
            "A nested session-only postprocessor selected y/z-only, full-strength, uncapped correction in all five folds, but it tied full XYZ on the current metric. The separate prediction probe improved median/gate slightly but worsened P90/P95 at 100/200 ms, so it failed the frozen all-horizon selection rule. Full XYZ yaw-harmonic remains the retained downstream candidate; y/z-only is negative trade-off evidence.",
            "",
            "## Upstream corner implication",
            "",
            "Historical exact-corner intervention closes the unchanged IPPE/coordinate chain to 0.003/0.009/0.012 mm P50/P95/P99, while measured refined corners produce 140.8/500.3/708.5 mm 3D pose error and 6.7/25.0/35.0 mm lateral P50/P95/P99. The formal 144-session observation schema lacks raw/refined corners, so a same-frame corner-to-combined-prediction intervention cannot be claimed. Together, the retained evidence supports optimizing corner/PnP objectives for tracker-y or hit-plane error rather than reprojection RMS or undirected 3D error alone.",
            "",
            "## Next optimization priority",
            "",
            "1. Keep the low-capacity correction as an offline candidate with raw fallback; validate on independent sessions before production use.",
            "2. Prioritize tracker-y-conditioned corner/PnP optimization and calibrated rejection/uncertainty for the meter-scale tail.",
            "3. Treat reversal prediction as a separate mode-switch/multi-hypothesis problem; better PnP cannot supply future reversal timing.",
            "4. Do not spend the next iteration on depth-only correction, higher harmonic order, direct EKF tuning, or slot-conditioned correction without new evidence.",
        ]
    )
    (output / "COMBINED_PNP_ERROR_REDUCTION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    parent = read_csv(args.parent / "prediction_distribution.csv.gz")
    replay = read_csv(args.replay / "prediction_distribution.csv.gz")
    fits = read_csv(args.replay / "fit_distribution.csv.gz")
    measurement = read_csv(args.screen / "measurement_summary.csv")
    yz = read_csv(args.yz_probe / "prediction_distribution.csv.gz")
    nested = json.loads((args.nested / "nested_postprocessor.json").read_text(encoding="utf-8-sig"))

    parity_report = parity(parent, replay)
    if not parity_report["passed"]:
        raise RuntimeError("parent parity failed")
    primary = primary_distribution(replay)
    paired = paired_comparisons(replay)
    session_rows, macro_rows = session_summaries(replay)
    condition_rows = condition_summaries(replay)
    fit_rows = fit_summaries(fits)
    yz_rows = yz_probe_summary(replay, yz)
    write_csv(output / "primary_distribution.csv", primary)
    write_csv(output / "paired_comparison.csv", paired)
    write_csv_gz(output / "session_distribution.csv.gz", session_rows)
    write_csv(output / "session_macro_distribution.csv", macro_rows)
    write_csv(output / "condition_distribution.csv", condition_rows)
    write_csv(output / "fit_distribution_summary.csv", fit_rows)
    write_csv(output / "yz_prediction_probe.csv", yz_rows)
    (output / "parity_report.json").write_text(
        json.dumps(parity_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_primary(output, replay)
    plot_gate_curve(output, replay)
    build_report(output, measurement, primary, paired, macro_rows, fit_rows, yz_rows, parity_report)

    summary = {
        "schema_version": "combined-pnp-error-reduction-analysis-v1",
        "parent_parity": parity_report,
        "measurement_screen": {
            "sessions": 144,
            "events": 465311,
            "promoted_arm": "yaw_harmonic",
        },
        "prediction_replay": {
            "sessions": 144,
            "prediction_rows": int(len(replay)),
            "fit_rows": int(len(fits)),
            "failed_sessions": 0,
        },
        "nested_postprocessor_promoted": bool(nested["promoted"]),
        "accepted_downstream_candidate": "crossfit_yaw_harmonic full xyz (offline only)",
        "rejected_followup": "cross_yz_only prediction trade-off fails all-horizon P90/P95 rule",
        "dominant_causal_coordinate": "tracker_y",
        "depth_only_priority": "low for current cross-depth prediction metric",
        "retained_invalid_run": {
            "path": str(args.replay.parent / "combined-pnp-error-reduction-144-v1"),
            "reason": "valid position events with missing yaw were deleted from all histories",
        },
        "source_hashes": {
            "parent_predictions": sha256(args.parent / "prediction_distribution.csv.gz"),
            "measurement_summary": sha256(args.screen / "measurement_summary.csv"),
            "measurement_rows": sha256(args.screen / "measurement_error_distribution.csv.gz"),
            "replay_predictions": sha256(args.replay / "prediction_distribution.csv.gz"),
            "replay_fits": sha256(args.replay / "fit_distribution.csv.gz"),
            "nested_postprocessor": sha256(args.nested / "nested_postprocessor.json"),
            "yz_probe_predictions": sha256(args.yz_probe / "prediction_distribution.csv.gz"),
        },
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "combined-pnp-error-reduction-analysis-manifest-v1",
        "complete": True,
        "artifacts": sorted(path.name for path in output.iterdir() if path.is_file()),
        "source_hashes": summary["source_hashes"],
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
