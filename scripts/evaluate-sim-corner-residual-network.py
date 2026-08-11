#!/usr/bin/env python3
"""Propagate cross-fitted corner corrections through unchanged free-IPPE PnP.

This evaluator is intentionally separate from training.  It verifies the raw
arm against retained production-matched evidence, emits per-detection pose
rows, summarizes the full distributions, and produces publication-style
figures without using pose error to select network checkpoints.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CORNER_ORDER = ("bl", "tl", "tr", "br")
ARM_ORDER = ("raw", "mean", "ridge", "current_refined", "network", "exact")
DISPLAY_ARMS = ("raw", "current_refined", "ridge", "network", "exact")
COLORS = {
    "raw": "#000000",
    "mean": "#56B4E9",
    "ridge": "#E69F00",
    "current_refined": "#CC79A7",
    "network": "#009E73",
    "exact": "#0072B2",
}
LINESTYLES = {
    "raw": "-",
    "mean": ":",
    "ridge": "-.",
    "current_refined": "--",
    "network": "-",
    "exact": ":",
}
OBJECT_POINTS_MM = np.asarray(
    [
        [-67.5, 27.5, 0.0],
        [-67.5, -27.5, 0.0],
        [67.5, -27.5, 0.0],
        [67.5, 27.5, 0.0],
    ],
    dtype=np.float32,
)
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--predictions", required=True, type=Path)
    result.add_argument("--retained-arm-rows", required=True, type=Path)
    result.add_argument("--observations", required=True, type=Path, action="append")
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--raw-position-tolerance-mm", type=float, default=0.10)
    result.add_argument("--raw-reprojection-tolerance-px", type=float, default=1.0e-3)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def git_state(repo: Path) -> dict[str, Any]:
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "status", "--short"], cwd=repo, text=True
            ).strip()
        ),
    }


def record_key(record: Any) -> tuple[str, int, int, int, int]:
    return (
        str(record["session_id"]),
        int(record["producer_epoch"]),
        int(record["frame_seq"]),
        int(record["timestamp_ns"]),
        int(record["armor_index"]),
    )


def load_camera_lookup(
    paths: Iterable[Path],
) -> tuple[dict[tuple[str, int, int, int], tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    lookup: dict[tuple[str, int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    profiles: set[str] = set()
    image_sizes: set[tuple[int, int]] = set()
    exposure_mismatch = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if not item.get("gimbal_pose_exposure_matched", False):
                    exposure_mismatch += 1
                key = (
                    str(item["session_id"]),
                    int(item["producer_epoch"]),
                    int(item["frame_seq"]),
                    int(item["timestamp_ns"]),
                )
                matrix = np.asarray(item["camera_matrix"], dtype=np.float64).reshape(3, 3)
                distortion = np.asarray(item["distortion_coeffs"], dtype=np.float64)
                lookup[key] = (matrix, distortion)
                profiles.add(str(item["camera_profile_id"]))
                image_sizes.add((int(item["image_width"]), int(item["image_height"])))
    return lookup, {
        "frames": len(lookup),
        "camera_profiles": sorted(profiles),
        "image_sizes": sorted([list(value) for value in image_sizes]),
        "exposure_mismatch_frames": exposure_mismatch,
    }


def load_retained_raw(path: Path) -> pd.DataFrame:
    columns = [
        "session_id", "producer_epoch", "frame_seq", "timestamp_ns", "armor_index",
        "arm", "replicate", "truth_camera_x_m", "truth_camera_y_m", "truth_camera_z_m",
        "selected_position_error_m", "selected_lateral_error_m", "selected_depth_error_m",
        "selected_reprojection_rms_px", "pnp_failed",
    ]
    selected = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=100_000):
        mask = (chunk["arm"] == "actual_raw") & (chunk["replicate"] == 0)
        if mask.any():
            selected.append(chunk.loc[mask].copy())
    if not selected:
        raise ValueError("retained arm table contains no actual_raw replicate-0 rows")
    result = pd.concat(selected, ignore_index=True)
    if result.duplicated(
        ["session_id", "producer_epoch", "frame_seq", "timestamp_ns", "armor_index"]
    ).any():
        raise ValueError("retained raw pose rows are not unique")
    return result


def corrected_corners(row: Any) -> np.ndarray:
    return np.asarray(
        [[row[f"corrected_{corner}_x_px"], row[f"corrected_{corner}_y_px"]]
         for corner in CORNER_ORDER],
        dtype=np.float32,
    )


def solve_ippe(
    corners: np.ndarray, matrix: np.ndarray, distortion: np.ndarray
) -> dict[str, Any]:
    result = cv2.solvePnPGeneric(
        OBJECT_POINTS_MM,
        corners,
        matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE,
    )
    success, rvecs, tvecs = bool(result[0]), result[1], result[2]
    if not success or not rvecs:
        return {"pnp_failed": True, "candidate_count": 0}
    candidates = []
    for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        projected, _ = cv2.projectPoints(
            OBJECT_POINTS_MM, rvec, tvec, matrix, distortion
        )
        difference = projected.reshape(4, 2) - corners.astype(np.float64)
        reprojection = float(np.sqrt(np.mean(np.sum(np.square(difference), axis=1))))
        candidates.append((reprojection, index, rvec.reshape(3), tvec.reshape(3)))
    reprojection, index, rvec, tvec = min(candidates, key=lambda value: value[0])
    return {
        "pnp_failed": False,
        "candidate_count": len(candidates),
        "selected_solution_index": index,
        "selected_reprojection_rms_px": reprojection,
        "rvec_x": float(rvec[0]),
        "rvec_y": float(rvec[1]),
        "rvec_z": float(rvec[2]),
        "camera_tvec_x_m": float(tvec[0] / 1000.0),
        "camera_tvec_y_m": float(tvec[1] / 1000.0),
        "camera_tvec_z_m": float(tvec[2] / 1000.0),
    }


def describe(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0}
    result: dict[str, float | int] = {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "minimum": float(np.min(finite)),
    }
    for quantile in QUANTILES:
        result[f"p{int(round(quantile * 100)):02d}"] = float(np.quantile(finite, quantile))
    result["maximum"] = float(np.max(finite))
    return result


def distribution_table(
    frame: pd.DataFrame,
    metrics: dict[str, str],
    *,
    unit: str,
) -> pd.DataFrame:
    rows = []
    for scheme in sorted(frame["scheme"].unique()):
        for arm in ARM_ORDER:
            subset = frame[(frame["scheme"] == scheme) & (frame["arm"] == arm)]
            if subset.empty:
                continue
            for metric, column in metrics.items():
                rows.append(
                    {
                        "scheme": scheme,
                        "arm": arm,
                        "metric": metric,
                        "unit": unit,
                        **describe(subset[column].to_numpy()),
                    }
                )
    return pd.DataFrame(rows)


def corner_distribution(predictions: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "coordinate_rms": "coordinate_rms_px",
        "corner_norm_mean": "corner_norm_mean_px",
        "corner_norm_max": "corner_norm_max_px",
    }
    for corner in CORNER_ORDER:
        metrics[f"{corner}_error_dx"] = f"error_{corner}_dx_px"
        metrics[f"{corner}_error_dy"] = f"error_{corner}_dy_px"
        metrics[f"{corner}_error_norm"] = f"error_{corner}_norm_px"
        metrics[f"{corner}_correction_dx"] = f"correction_{corner}_dx_px"
        metrics[f"{corner}_correction_dy"] = f"correction_{corner}_dy_px"
    return distribution_table(predictions, metrics, unit="px")


def pnp_distribution(pose: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "position_error": "position_error_mm",
        "camera_lateral_error": "lateral_error_mm",
        "absolute_depth_error": "absolute_depth_error_mm",
        "signed_depth_error": "signed_depth_error_mm",
        "reprojection_rms": "selected_reprojection_rms_px",
    }
    result = distribution_table(pose, metrics, unit="mixed")
    result.loc[result["metric"] != "reprojection_rms", "unit"] = "mm"
    result.loc[result["metric"] == "reprojection_rms", "unit"] = "px"
    return result


def held_group_distribution(
    frame: pd.DataFrame, metrics: dict[str, str], *, unit_by_metric: dict[str, str]
) -> pd.DataFrame:
    rows = []
    for (scheme, outer_group, arm), subset in frame.groupby(
        ["scheme", "outer_group", "arm"], sort=True
    ):
        for metric, column in metrics.items():
            rows.append({
                "scheme": scheme,
                "outer_group": outer_group,
                "arm": arm,
                "metric": metric,
                "unit": unit_by_metric[metric],
                **describe(subset[column].to_numpy()),
            })
    return pd.DataFrame(rows)


def rank_correlation(left: pd.Series, right: pd.Series) -> float:
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return float("nan")
    return float(left.rank().corr(right.rank()))


def uncertainty_diagnostic(predictions: pd.DataFrame) -> pd.DataFrame:
    key = ["scheme", "outer_group", "row_index"]
    raw = predictions[predictions.arm == "raw"][key + ["coordinate_rms_px"]].rename(
        columns={"coordinate_rms_px": "raw_coordinate_rms_px"}
    )
    network = predictions[predictions.arm == "network"].merge(
        raw, on=key, how="left", validate="one_to_one"
    )
    uncertainty_columns = [f"network_uncertainty_{corner}_px" for corner in CORNER_ORDER]
    network["ensemble_uncertainty_px"] = network[uncertainty_columns].mean(axis=1)
    network["network_minus_raw_coordinate_rms_px"] = (
        network["coordinate_rms_px"] - network["raw_coordinate_rms_px"]
    )
    rows = []
    for (scheme, outer_group), subset in network.groupby(
        ["scheme", "outer_group"], sort=True
    ):
        rows.append({
            "scheme": scheme,
            "outer_group": outer_group,
            "count": len(subset),
            "uncertainty_mean_px": float(subset["ensemble_uncertainty_px"].mean()),
            "uncertainty_p95_px": float(subset["ensemble_uncertainty_px"].quantile(0.95)),
            "network_improved_fraction": float(
                np.mean(subset["network_minus_raw_coordinate_rms_px"] < 0)
            ),
            "network_minus_raw_mean_px": float(
                subset["network_minus_raw_coordinate_rms_px"].mean()
            ),
            "spearman_uncertainty_vs_network_error": float(
                rank_correlation(
                    subset["ensemble_uncertainty_px"], subset["coordinate_rms_px"]
                )
            ),
            "spearman_uncertainty_vs_network_minus_raw": float(
                rank_correlation(
                    subset["ensemble_uncertainty_px"],
                    subset["network_minus_raw_coordinate_rms_px"],
                )
            ),
        })
    return pd.DataFrame(rows)


def paired_table(pose: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = [
        "scheme", "session_id", "producer_epoch", "frame_seq", "timestamp_ns",
        "armor_index",
    ]
    metric_columns = (
        "position_error_mm", "lateral_error_mm", "absolute_depth_error_mm",
        "selected_reprojection_rms_px",
    )
    raw = pose[pose["arm"] == "raw"][key + list(metric_columns)].copy()
    raw = raw.rename(columns={name: f"raw_{name}" for name in metric_columns})
    merged = pose.merge(raw, on=key, how="left", validate="many_to_one")
    long_rows = []
    summary_rows = []
    for arm in ARM_ORDER:
        if arm == "raw":
            continue
        subset = merged[merged["arm"] == arm].copy()
        for metric in metric_columns:
            delta = subset[metric].to_numpy() - subset[f"raw_{metric}"].to_numpy()
            for source, value in zip(subset.itertuples(index=False), delta):
                long_rows.append(
                    {
                        "scheme": source.scheme,
                        "session_id": source.session_id,
                        "producer_epoch": source.producer_epoch,
                        "frame_seq": source.frame_seq,
                        "timestamp_ns": source.timestamp_ns,
                        "armor_index": source.armor_index,
                        "arm": arm,
                        "metric": metric,
                        "delta_arm_minus_raw": float(value),
                    }
                )
        for scheme in sorted(subset["scheme"].unique()):
            scheme_subset = subset[subset["scheme"] == scheme]
            for metric in metric_columns:
                delta = (
                    scheme_subset[metric].to_numpy()
                    - scheme_subset[f"raw_{metric}"].to_numpy()
                )
                tolerance = 1.0e-9
                summary_rows.append(
                    {
                        "scheme": scheme,
                        "arm": arm,
                        "metric": metric,
                        "improved_fraction": float(np.mean(delta < -tolerance)),
                        "unchanged_fraction": float(np.mean(np.abs(delta) <= tolerance)),
                        "worsened_fraction": float(np.mean(delta > tolerance)),
                        **describe(delta),
                    }
                )
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(np.asarray(values, dtype=np.float64))
    return values, np.arange(1, len(values) + 1, dtype=np.float64) / len(values)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.18, linewidth=0.6)


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_corner_ecdf(predictions: pd.DataFrame, output: Path) -> None:
    schemes = sorted(predictions["scheme"].unique())
    fig, axes = plt.subplots(1, len(schemes), figsize=(6.4 * len(schemes), 4.5), squeeze=False)
    for ax, scheme in zip(axes[0], schemes):
        for arm in DISPLAY_ARMS:
            values = predictions.loc[
                (predictions["scheme"] == scheme) & (predictions["arm"] == arm),
                "coordinate_rms_px",
            ].to_numpy()
            x, y = ecdf(values)
            ax.plot(x, y, label=f"{arm} (n={len(values)})", color=COLORS[arm],
                    linestyle=LINESTYLES[arm], linewidth=1.7)
        ax.set_title(scheme.replace("_", " "))
        ax.set_xlabel("Joint 8-D coordinate RMS (px)")
        ax.set_ylabel("Empirical cumulative fraction")
        ax.set_ylim(0, 1.005)
        style_axes(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Complete out-of-fold corner-error distributions")
    fig.tight_layout()
    save_figure(fig, output, "corner_error_ecdf")


def plot_pnp_ecdf(pose: pd.DataFrame, output: Path) -> None:
    schemes = sorted(pose["scheme"].unique())
    metrics = (
        ("position_error_mm", "3-D position error (mm)"),
        ("lateral_error_mm", "Camera-lateral error (mm)"),
        ("absolute_depth_error_mm", "Absolute camera-depth error (mm)"),
    )
    fig, axes = plt.subplots(len(schemes), 3, figsize=(15, 4.2 * len(schemes)), squeeze=False)
    for row_index, scheme in enumerate(schemes):
        for column_index, (metric, label) in enumerate(metrics):
            ax = axes[row_index, column_index]
            for arm in DISPLAY_ARMS:
                values = pose.loc[
                    (pose["scheme"] == scheme) & (pose["arm"] == arm), metric
                ].to_numpy()
                x, y = ecdf(values)
                ax.plot(x, y, label=f"{arm} (n={len(values)})", color=COLORS[arm],
                        linestyle=LINESTYLES[arm], linewidth=1.6)
            ax.set_xscale("symlog", linthresh=0.1)
            ax.set_xlabel(label)
            ax.set_ylabel("Empirical cumulative fraction")
            ax.set_title(scheme.replace("_", " "))
            ax.set_ylim(0, 1.005)
            style_axes(ax)
            if column_index == 0:
                ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Unchanged free-IPPE pose-error distributions (all detections)")
    fig.tight_layout()
    save_figure(fig, output, "pnp_error_ecdf")


def plot_signed_corner_residuals(predictions: pd.DataFrame, output: Path) -> None:
    subset = predictions[
        (predictions["scheme"] == "leave_segment_out")
        & predictions["arm"].isin(("raw", "network"))
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
    all_values = []
    for corner in CORNER_ORDER:
        all_values.extend(subset[f"error_{corner}_dx_px"].to_numpy())
        all_values.extend(subset[f"error_{corner}_dy_px"].to_numpy())
    limit = max(1.0, float(np.max(np.abs(np.asarray(all_values)))))
    for row_index, arm in enumerate(("raw", "network")):
        arm_rows = subset[subset["arm"] == arm]
        for column_index, corner in enumerate(CORNER_ORDER):
            ax = axes[row_index, column_index]
            x = arm_rows[f"error_{corner}_dx_px"].to_numpy()
            y = arm_rows[f"error_{corner}_dy_px"].to_numpy()
            ax.hexbin(x, y, gridsize=45, mincnt=1, cmap="viridis", bins="log")
            ax.axhline(0, color="#777777", linewidth=0.6)
            ax.axvline(0, color="#777777", linewidth=0.6)
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{arm} · {corner} (n={len(x)})")
            ax.set_xlabel("Signed x residual (px)")
            ax.set_ylabel("Signed y residual (px)")
            style_axes(ax)
    fig.suptitle("Per-corner signed residual support, primary OOF")
    fig.tight_layout()
    save_figure(fig, output, "corner_signed_residuals")


def plot_pose_vs_range(pose: pd.DataFrame, output: Path) -> None:
    subset = pose[
        (pose["scheme"] == "leave_segment_out")
        & pose["arm"].isin(("raw", "network"))
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for row_index, arm in enumerate(("raw", "network")):
        arm_rows = subset[subset["arm"] == arm]
        for column_index, (metric, label) in enumerate((
            ("lateral_error_mm", "Camera-lateral error (mm)"),
            ("absolute_depth_error_mm", "Absolute depth error (mm)"),
        )):
            ax = axes[row_index, column_index]
            image = ax.hexbin(
                arm_rows["range_m"], arm_rows[metric], gridsize=48,
                mincnt=1, cmap="viridis", bins="log",
            )
            ax.set_title(f"{arm} (n={len(arm_rows)})")
            ax.set_xlabel("Truth camera range (m)")
            ax.set_ylabel(label)
            style_axes(ax)
            fig.colorbar(image, ax=ax, label="log10 count")
    fig.suptitle("Pose error versus range, primary OOF")
    fig.tight_layout()
    save_figure(fig, output, "pnp_error_vs_range")


def plot_stress_sessions(
    predictions: pd.DataFrame, pose: pd.DataFrame, output: Path
) -> None:
    session_groups = sorted(
        predictions.loc[predictions.scheme == "leave_session_out", "outer_group"].unique()
    )
    fig, axes = plt.subplots(len(session_groups), 2, figsize=(12, 4.2 * len(session_groups)), squeeze=False)
    for row_index, outer_group in enumerate(session_groups):
        short_name = str(outer_group).split("-r1-")[-1]
        for column_index, (source, metric, xlabel) in enumerate((
            (predictions, "coordinate_rms_px", "Joint 8-D coordinate RMS (px)"),
            (pose, "lateral_error_mm", "Camera-lateral error (mm)"),
        )):
            ax = axes[row_index, column_index]
            for arm in ("raw", "mean", "network"):
                values = source.loc[
                    (source.scheme == "leave_session_out")
                    & (source.outer_group == outer_group)
                    & (source.arm == arm), metric
                ].to_numpy()
                x, y = ecdf(values)
                ax.plot(
                    x, y, label=f"{arm} (n={len(values)})", color=COLORS[arm],
                    linestyle=LINESTYLES[arm], linewidth=1.7,
                )
            ax.set_title(f"Held out: {short_name}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Empirical cumulative fraction")
            ax.set_ylim(0, 1.005)
            style_axes(ax)
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Cross-session stress test disaggregated by held session")
    fig.tight_layout()
    save_figure(fig, output, "stress_session_ecdf")


def acceptance(corner_summary: pd.DataFrame, pose_summary: pd.DataFrame) -> dict[str, Any]:
    checks = []
    scheme = "leave_segment_out"
    for quantile in ("p50", "p90", "p95"):
        for baseline in ("raw", "current_refined"):
            network = corner_summary.loc[
                (corner_summary.scheme == scheme) & (corner_summary.arm == "network")
                & (corner_summary.metric == "coordinate_rms"), quantile
            ].iloc[0]
            reference = corner_summary.loc[
                (corner_summary.scheme == scheme) & (corner_summary.arm == baseline)
                & (corner_summary.metric == "coordinate_rms"), quantile
            ].iloc[0]
            checks.append({
                "name": f"corner_{quantile}_network_better_than_{baseline}",
                "passed": bool(network < reference),
                "network_px": float(network),
                "reference_px": float(reference),
            })
    for metric in ("position_error", "camera_lateral_error"):
        for quantile in ("p50", "p90", "p95"):
            network = pose_summary.loc[
                (pose_summary.scheme == scheme) & (pose_summary.arm == "network")
                & (pose_summary.metric == metric), quantile
            ].iloc[0]
            reference = pose_summary.loc[
                (pose_summary.scheme == scheme) & (pose_summary.arm == "raw")
                & (pose_summary.metric == metric), quantile
            ].iloc[0]
            checks.append({
                "name": f"pnp_{metric}_{quantile}_network_no_worse_than_raw",
                "passed": bool(network <= reference),
                "network_mm": float(network),
                "reference_mm": float(reference),
            })
    return {"passed": all(item["passed"] for item in checks), "checks": checks}


def report_markdown(
    corner_summary: pd.DataFrame,
    pose_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    held_corner: pd.DataFrame,
    held_pose: pd.DataFrame,
    uncertainty: pd.DataFrame,
    parity: dict[str, Any],
    accepted: dict[str, Any],
) -> str:
    def metric_lines(summary: pd.DataFrame, scheme: str, metric: str) -> list[str]:
        rows = summary[(summary.scheme == scheme) & (summary.metric == metric)]
        lines = []
        for arm in ARM_ORDER:
            selected = rows[rows.arm == arm]
            if selected.empty:
                continue
            row = selected.iloc[0]
            lines.append(
                f"| {arm} | {int(row['count'])} | {row['mean']:.4f} | "
                f"{row['p25']:.4f} | {row['p50']:.4f} | {row['p75']:.4f} | "
                f"{row['p90']:.4f} | {row['p95']:.4f} | {row['p99']:.4f} | "
                f"{row['maximum']:.4f} |"
            )
        return lines

    lines = [
        "# Simulation joint four-corner repair: OOF evidence", "",
        "This is an offline simulation result. It is not deployable and does not alter the detector, PnP, tracker, or fire-control path.", "",
        f"- Raw free-IPPE parity passed: `{parity['passed']}`",
        f"- Frozen pilot acceptance passed: `{accepted['passed']}`", "",
    ]
    for scheme in ("leave_segment_out", "leave_session_out"):
        lines.extend([f"## {scheme}", "", "### Joint corner coordinate RMS (px)", "",
                      "| arm | n | mean | P25 | P50 | P75 | P90 | P95 | P99 | max |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        lines.extend(metric_lines(corner_summary, scheme, "coordinate_rms"))
        for metric, title in (("position_error", "3-D PnP error (mm)"),
                              ("camera_lateral_error", "Camera-lateral PnP error (mm)"),
                              ("absolute_depth_error", "Absolute depth PnP error (mm)")):
            lines.extend(["", f"### {title}", "",
                          "| arm | n | mean | P25 | P50 | P75 | P90 | P95 | P99 | max |",
                          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
            lines.extend(metric_lines(pose_summary, scheme, metric))
        lines.append("")
    network_pairs = paired_summary[paired_summary.arm == "network"]
    lines.extend(["## Network paired outcome against raw", "",
                  "Negative delta means the network reduced error for the same detection.", "",
                  "| scheme | metric | improved | unchanged | worsened | median delta | P90 delta |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in network_pairs.itertuples(index=False):
        lines.append(
            f"| {row.scheme} | {row.metric} | {row.improved_fraction:.3f} | "
            f"{row.unchanged_fraction:.3f} | {row.worsened_fraction:.3f} | "
            f"{row.p50:.4f} | {row.p90:.4f} |"
        )
    lines.extend([
        "", "## Cross-session asymmetry (must not be pooled away)", "",
        "| held session | arm | corner P50 (px) | corner P95 (px) | 3-D P50 (mm) | 3-D P95 (mm) | lateral P50 (mm) | lateral P95 (mm) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    stress_corner = held_corner[held_corner.scheme == "leave_session_out"]
    stress_pose = held_pose[held_pose.scheme == "leave_session_out"]
    for outer_group in sorted(stress_corner.outer_group.unique()):
        for arm in ("raw", "mean", "network"):
            corner = stress_corner[
                (stress_corner.outer_group == outer_group) & (stress_corner.arm == arm)
                & (stress_corner.metric == "coordinate_rms")
            ].iloc[0]
            position = stress_pose[
                (stress_pose.outer_group == outer_group) & (stress_pose.arm == arm)
                & (stress_pose.metric == "position_error")
            ].iloc[0]
            lateral = stress_pose[
                (stress_pose.outer_group == outer_group) & (stress_pose.arm == arm)
                & (stress_pose.metric == "camera_lateral_error")
            ].iloc[0]
            lines.append(
                f"| {str(outer_group).split('-r1-')[-1]} | {arm} | "
                f"{corner.p50:.4f} | {corner.p95:.4f} | {position.p50:.3f} | "
                f"{position.p95:.3f} | {lateral.p50:.3f} | {lateral.p95:.3f} |"
            )
    lines.extend(["", "## Ensemble uncertainty diagnostic", "",
                  "The three-seed spread is retained as evidence but is not a validated rejection score.", "",
                  "| scheme | held group | n | uncertainty mean (px) | correction improved | mean corner delta (px) | rho(uncertainty, error) | rho(uncertainty, delta) |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in uncertainty.itertuples(index=False):
        lines.append(
            f"| {row.scheme} | {str(row.outer_group).split('-r1-')[-1]} | {row.count} | "
            f"{row.uncertainty_mean_px:.4f} | {row.network_improved_fraction:.3f} | "
            f"{row.network_minus_raw_mean_px:.4f} | "
            f"{row.spearman_uncertainty_vs_network_error:.3f} | "
            f"{row.spearman_uncertainty_vs_network_minus_raw:.3f} |"
        )
    lines.extend(["", "## Interpretation boundary", "",
                  "Passing the frozen pilot only supports collecting image patches/heatmaps for a stronger simulation experiment. It does not establish sim-to-real transfer.", ""])
    return "\n".join(lines)


def main() -> None:
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {args.output}")
    args.output.mkdir(parents=True)
    predictions = pd.read_csv(args.predictions)
    required_schemes = {"leave_segment_out", "leave_session_out"}
    if set(predictions["scheme"].unique()) != required_schemes:
        raise ValueError("predictions do not contain both frozen OOF schemes")
    expected = 4280 * len(ARM_ORDER) * len(required_schemes)
    if len(predictions) != expected:
        raise ValueError(f"unexpected prediction row count: {len(predictions)} != {expected}")

    retained = load_retained_raw(args.retained_arm_rows)
    truth_lookup = {record_key(row): np.asarray(
        [row["truth_camera_x_m"], row["truth_camera_y_m"], row["truth_camera_z_m"]],
        dtype=np.float64,
    ) for _, row in retained.iterrows()}
    retained_lookup = {record_key(row): row for _, row in retained.iterrows()}
    camera_lookup, camera_audit = load_camera_lookup(args.observations)
    if camera_audit["exposure_mismatch_frames"]:
        raise ValueError("non-exposure-matched camera metadata found")

    pose_rows = []
    failed = 0
    missing = 0
    for row in predictions.to_dict(orient="records"):
        key = record_key(row)
        frame_key = key[:4]
        if key not in truth_lookup or frame_key not in camera_lookup:
            missing += 1
            continue
        matrix, distortion = camera_lookup[frame_key]
        solved = solve_ippe(corrected_corners(row), matrix, distortion)
        if solved["pnp_failed"]:
            failed += 1
            continue
        truth = truth_lookup[key]
        tvec = np.asarray(
            [solved["camera_tvec_x_m"], solved["camera_tvec_y_m"], solved["camera_tvec_z_m"]]
        )
        error = tvec - truth
        pose_rows.append({
            "scheme": row["scheme"],
            "outer_group": row["outer_group"],
            "row_index": row["row_index"],
            "session_id": key[0],
            "producer_epoch": key[1],
            "frame_seq": key[2],
            "timestamp_ns": key[3],
            "armor_index": key[4],
            "segment_index": row["segment_index"],
            "motion_mode": row["motion_mode"],
            "range_m": row["range_m"],
            "view_incidence_cos": row["view_incidence_cos"],
            "projected_sqrt_area_px": row["projected_sqrt_area_px"],
            "arm": row["arm"],
            **solved,
            "error_x_mm": float(error[0] * 1000.0),
            "error_y_mm": float(error[1] * 1000.0),
            "signed_depth_error_mm": float(error[2] * 1000.0),
            "absolute_depth_error_mm": float(abs(error[2]) * 1000.0),
            "lateral_error_mm": float(np.linalg.norm(error[:2]) * 1000.0),
            "position_error_mm": float(np.linalg.norm(error) * 1000.0),
        })
    if missing or failed:
        raise RuntimeError(f"PnP evaluation incomplete: missing={missing}, failed={failed}")
    pose = pd.DataFrame(pose_rows)

    primary_raw = pose[
        (pose["scheme"] == "leave_segment_out") & (pose["arm"] == "raw")
    ]
    parity_deltas = []
    for row in primary_raw.to_dict(orient="records"):
        retained_row = retained_lookup[record_key(row)]
        parity_deltas.append({
            "position_mm": abs(row["position_error_mm"] - retained_row["selected_position_error_m"] * 1000.0),
            "lateral_mm": abs(row["lateral_error_mm"] - retained_row["selected_lateral_error_m"] * 1000.0),
            "depth_mm": abs(row["signed_depth_error_mm"] - retained_row["selected_depth_error_m"] * 1000.0),
            "reprojection_px": abs(row["selected_reprojection_rms_px"] - retained_row["selected_reprojection_rms_px"]),
        })
    parity_frame = pd.DataFrame(parity_deltas)
    parity = {
        "rows": len(parity_frame),
        "maximum_absolute_delta": {
            name: float(parity_frame[name].max()) for name in parity_frame.columns
        },
    }
    parity["passed"] = bool(
        parity_frame[["position_mm", "lateral_mm", "depth_mm"]].to_numpy().max()
        <= args.raw_position_tolerance_mm
        and parity_frame["reprojection_px"].max() <= args.raw_reprojection_tolerance_px
    )
    if not parity["passed"]:
        atomic_json(args.output / "raw_parity_failure.json", parity)
        raise RuntimeError("recomputed raw IPPE does not match retained evidence")

    corner_summary = corner_distribution(predictions)
    pose_summary = pnp_distribution(pose)
    paired_rows, paired_summary = paired_table(pose)
    held_corner = held_group_distribution(
        predictions,
        {"coordinate_rms": "coordinate_rms_px"},
        unit_by_metric={"coordinate_rms": "px"},
    )
    held_pose = held_group_distribution(
        pose,
        {
            "position_error": "position_error_mm",
            "camera_lateral_error": "lateral_error_mm",
            "absolute_depth_error": "absolute_depth_error_mm",
        },
        unit_by_metric={
            "position_error": "mm",
            "camera_lateral_error": "mm",
            "absolute_depth_error": "mm",
        },
    )
    uncertainty = uncertainty_diagnostic(predictions)
    accepted = acceptance(corner_summary, pose_summary)

    predictions.to_csv(args.output / "oof_corner_predictions_copy.csv.gz", index=False, compression="gzip")
    with gzip.open(args.output / "oof_pnp_rows.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        pose.to_csv(handle, index=False)
    with gzip.open(args.output / "paired_deltas_vs_raw.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        paired_rows.to_csv(handle, index=False)
    corner_summary.to_csv(args.output / "corner_distribution.csv", index=False)
    pose_summary.to_csv(args.output / "pnp_distribution.csv", index=False)
    paired_summary.to_csv(args.output / "paired_summary_vs_raw.csv", index=False)
    held_corner.to_csv(args.output / "held_group_corner_distribution.csv", index=False)
    held_pose.to_csv(args.output / "held_group_pnp_distribution.csv", index=False)
    uncertainty.to_csv(args.output / "network_uncertainty_diagnostic.csv", index=False)
    atomic_json(args.output / "raw_pnp_parity.json", parity)
    atomic_json(args.output / "acceptance.json", accepted)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
    })
    plot_corner_ecdf(predictions, args.output)
    plot_pnp_ecdf(pose, args.output)
    plot_signed_corner_residuals(predictions, args.output)
    plot_pose_vs_range(pose, args.output)
    plot_stress_sessions(predictions, pose, args.output)

    report = report_markdown(
        corner_summary, pose_summary, paired_summary, held_corner, held_pose,
        uncertainty, parity, accepted,
    )
    (args.output / "report.md").write_text(report, encoding="utf-8")
    repo = Path(__file__).resolve().parents[1]
    manifest = {
        "schema_version": "stage3-sim-corner-residual-network-pnp-evidence-v1",
        "status": "complete",
        "deployable": False,
        "training_selected_on_pnp": False,
        "inputs": {
            "predictions": {"path": str(args.predictions.resolve()), "sha256": sha256(args.predictions)},
            "retained_arm_rows": {"path": str(args.retained_arm_rows.resolve()), "sha256": sha256(args.retained_arm_rows)},
            "observations": [
                {"path": str(path.resolve()), "sha256": sha256(path)} for path in args.observations
            ],
        },
        "source": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git": git_state(repo),
        },
        "environment": {"python": os.sys.version, "opencv": cv2.__version__, "numpy": np.__version__},
        "camera_audit": camera_audit,
        "rows": {"corner": len(predictions), "pose": len(pose), "paired": len(paired_rows)},
        "raw_parity": parity,
        "acceptance": accepted,
        "artifacts": sorted(path.name for path in args.output.iterdir()),
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
