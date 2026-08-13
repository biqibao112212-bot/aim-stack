#!/usr/bin/env python3
"""Describe repeatability and condition-dependent structure of four-plate paths.

This is deliberately a descriptive trajectory-regularity analysis. It merges
repeated runs by condition and target phase, fits a periodic center curve, and
reports empirical residual bands. It does not construct a future-prediction
dataset or evaluate any forecast horizon.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SLOT_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
REPEAT_COLORS = ("#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00")
TWO_PI = 2.0 * math.pi


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    return (values + 180.0) % 360.0 - 180.0


def phase_mod(values: np.ndarray) -> np.ndarray:
    return np.mod(values, TWO_PI)


def periodic_interpolate(query: np.ndarray, grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.full_like(query, np.nan, dtype=float)
    if np.count_nonzero(valid) == 1:
        return np.full_like(query, float(values[valid][0]), dtype=float)
    x = grid[valid]
    y = values[valid]
    xp = np.concatenate(([x[-1] - TWO_PI], x, [x[0] + TWO_PI]))
    yp = np.concatenate(([y[-1]], y, [y[0]]))
    return np.interp(phase_mod(query), xp, yp)


def fill_short_circular_gaps(values: np.ndarray, max_gap_bins: int = 2) -> np.ndarray:
    """Fill only tiny supported gaps; large unobserved arcs remain NaN."""
    result = values.copy()
    count = len(values)
    valid = np.isfinite(values)
    if not np.any(valid):
        return result
    for index in np.flatnonzero(~valid):
        previous = next((step for step in range(1, max_gap_bins + 2) if valid[(index - step) % count]), None)
        following = next((step for step in range(1, max_gap_bins + 2) if valid[(index + step) % count]), None)
        if previous is None or following is None or previous + following - 1 > max_gap_bins:
            continue
        left = values[(index - previous) % count]
        right = values[(index + following) % count]
        result[index] = float(left + previous / (previous + following) * (right - left))
    return result


def fit_periodic_curve(rows: list[dict], bins: int = 128) -> dict:
    # Use the physical yaw in the simulator/world frame.  ``phase_rad`` in the
    # upstream analysis is intentionally zeroed at the beginning of each run;
    # merging repeats on that field confounds different absolute target poses
    # and can manufacture broad or flipped-looking curves.
    phase = phase_mod(
        np.asarray(
            [float(row.get("target_yaw_rad", row["phase_rad"])) for row in rows],
            dtype=float,
        )
    )
    u = np.asarray([float(row["u_deg"]) for row in rows], dtype=float)
    v = np.asarray([float(row["v_deg"]) for row in rows], dtype=float)
    grid = (np.arange(bins, dtype=float) + 0.5) * TWO_PI / bins
    indices = np.minimum((phase / TWO_PI * bins).astype(int), bins - 1)
    center_u = np.full(bins, np.nan, dtype=float)
    center_v = np.full(bins, np.nan, dtype=float)
    for index in range(bins):
        mask = indices == index
        if np.any(mask):
            center_u[index] = float(np.median(u[mask]))
            center_v[index] = float(np.median(v[mask]))
    center_u_grid = fill_short_circular_gaps(center_u)
    center_v_grid = fill_short_circular_gaps(center_v)
    fitted_u = center_u_grid[indices]
    fitted_v = center_v_grid[indices]
    residual = np.hypot(u - fitted_u, v - fitted_v)
    q50 = np.full(bins, np.nan, dtype=float)
    q90 = np.full(bins, np.nan, dtype=float)
    q95 = np.full(bins, np.nan, dtype=float)
    for index in range(bins):
        mask = indices == index
        if np.any(mask):
            q50[index], q90[index], q95[index] = np.percentile(residual[mask], [50, 90, 95])
    return {
        "grid": grid,
        "center_u": center_u_grid,
        "center_v": center_v_grid,
        "residual": residual,
        "phase": phase,
        "q50": q50,
        "q90": q90,
        "q95": q95,
        "phase_coverage": float(np.count_nonzero(np.isfinite(center_u)) / bins),
    }


def curve_distance(first: dict, second: dict) -> np.ndarray:
    valid = (
        np.isfinite(first["center_u"])
        & np.isfinite(first["center_v"])
        & np.isfinite(second["center_u"])
        & np.isfinite(second["center_v"])
    )
    return np.hypot(
        first["center_u"][valid] - second["center_u"][valid],
        first["center_v"][valid] - second["center_v"][valid],
    )


def robust_outlier_rate(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = median + 3.0 * 1.4826 * mad
    if threshold <= 1e-9:
        threshold = float(np.percentile(values, 95))
    return float(np.mean(values > threshold)), threshold


def curve_descriptors(curve: dict) -> dict:
    u = curve["center_u"]
    v = curve["center_v"]
    valid = np.isfinite(u) & np.isfinite(v)
    if np.count_nonzero(valid) < 2:
        return {
            "u_span_deg": float("nan"),
            "v_span_deg": float("nan"),
            "curve_length_deg": float("nan"),
        }
    unwrapped_u = np.degrees(np.unwrap(np.radians(u[valid])))
    unwrapped_v = np.degrees(np.unwrap(np.radians(v[valid])))
    length = float(np.sum(np.hypot(np.diff(unwrapped_u), np.diff(unwrapped_v))))
    return {
        "u_span_deg": float(np.percentile(unwrapped_u, 99) - np.percentile(unwrapped_u, 1)),
        "v_span_deg": float(np.percentile(unwrapped_v, 99) - np.percentile(unwrapped_v, 1)),
        "curve_length_deg": length,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_values(values: list[float]) -> dict:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return {"n": 0}
    return {
        "n": len(finite),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def group_rows(rows: list[dict]) -> dict[tuple[float, float, int], list[dict]]:
    grouped: dict[tuple[float, float, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["scale"]), float(row["distance_m"]), int(row["slot"]))].append(row)
    return grouped


def supported_plot_segments(u: np.ndarray, v: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return finite, image-space-continuous curve segments for drawing only.

    Phase adjacency is necessary but not sufficient: a PnP family switch can
    put adjacent phase bins far apart in the image.  Connecting those bins is
    the source of the apparent spikes/flips in the old overview figures.
    """
    valid = np.isfinite(u) & np.isfinite(v)
    starts = np.flatnonzero(valid & np.r_[True, ~valid[:-1]])
    stops = np.flatnonzero(valid & np.r_[~valid[1:], True]) + 1
    runs = [np.arange(start, stop) for start, stop in zip(starts, stops)]
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for run in runs:
        if run.size < 2:
            continue
        points = np.column_stack((u[run], v[run]))
        jumps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        finite_jumps = jumps[np.isfinite(jumps)]
        if finite_jumps.size:
            median = float(np.median(finite_jumps))
            mad = float(np.median(np.abs(finite_jumps - median)))
            threshold = max(0.25, 4.0 * median, median + 6.0 * 1.4826 * mad)
            pieces = np.split(run, np.flatnonzero(jumps > threshold) + 1)
        else:
            pieces = [run]
        result.extend((u[piece], v[piece]) for piece in pieces if piece.size >= 2)
    return result


def plot_trajectory_grid(
    output: Path,
    grouped: dict[tuple[float, float, int], list[dict]],
    stream_name: str,
    scales: list[float],
    distances: list[float],
) -> None:
    fig, axes = plt.subplots(
        len(scales), len(distances),
        figsize=(4.0 * len(distances), 3.2 * len(scales)),
        squeeze=False,
    )
    for row_index, scale in enumerate(scales):
        for col_index, distance_m in enumerate(distances):
            axis = axes[row_index][col_index]
            for slot, color in enumerate(SLOT_COLORS):
                points = grouped.get((scale, distance_m, slot), [])
                if not points:
                    continue
                merged = fit_periodic_curve(points)
                repeats = sorted({int(point["repeat"]) for point in points})
                for repeat in repeats:
                    repeat_points = [point for point in points if int(point["repeat"]) == repeat]
                    if len(repeat_points) < 8:
                        continue
                    repeat_curve = fit_periodic_curve(repeat_points)
                    for segment_u, segment_v in supported_plot_segments(
                        repeat_curve["center_u"], repeat_curve["center_v"]
                    ):
                        axis.plot(segment_u, segment_v, color=color, alpha=0.18, linewidth=0.7)
                if stream_name == "observed":
                    axis.scatter(
                        [point["u_deg"] for point in points],
                        [point["v_deg"] for point in points],
                        s=1.2,
                        alpha=0.08,
                        color=color,
                    )
                segments = supported_plot_segments(merged["center_u"], merged["center_v"])
                for segment_index, (segment_u, segment_v) in enumerate(segments):
                    axis.plot(
                        segment_u,
                        segment_v,
                        color=color,
                        linewidth=1.8,
                        label=f"slot {slot}" if segment_index == 0 else None,
                    )
            axis.set_title(f"scale={scale:g}, d={distance_m:g} m")
            axis.set_xlabel("camera azimuth (deg)")
            axis.set_ylabel("camera elevation (deg)")
            axis.grid(alpha=0.2)
            if row_index == 0 and col_index == 0:
                axis.legend(fontsize=8, loc="best")
    fig.suptitle(
        f"{stream_name.capitalize()} repeated paths and merged center curves",
        y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output / f"trajectory_regularity_{stream_name}.png", dpi=220)
    plt.close(fig)


def plot_probability_bands(
    output: Path,
    grouped: dict[tuple[float, float, int], list[dict]],
    stream_name: str,
    scales: list[float],
    distances: list[float],
) -> None:
    fig, axes = plt.subplots(
        len(scales), len(distances),
        figsize=(4.0 * len(distances), 3.0 * len(scales)),
        squeeze=False,
    )
    for row_index, scale in enumerate(scales):
        for col_index, distance_m in enumerate(distances):
            axis = axes[row_index][col_index]
            for slot, color in enumerate(SLOT_COLORS):
                points = grouped.get((scale, distance_m, slot), [])
                if not points:
                    continue
                curve = fit_periodic_curve(points)
                phase_deg = np.degrees(curve["grid"])
                axis.fill_between(
                    phase_deg,
                    curve["q50"],
                    curve["q95"],
                    color=color,
                    alpha=0.13,
                )
                axis.plot(
                    phase_deg,
                    curve["q50"],
                    color=color,
                    linewidth=1.0,
                    label=f"slot {slot} P50",
                )
                axis.plot(
                    phase_deg,
                    curve["q95"],
                    color=color,
                    linewidth=0.8,
                    linestyle="--",
                )
            axis.set_title(f"scale={scale:g}, d={distance_m:g} m")
            axis.set_xlabel("target phase (deg)")
            axis.set_ylabel("radial residual (deg)")
            axis.set_xlim(0, 360)
            axis.grid(alpha=0.2)
            if row_index == 0 and col_index == 0:
                axis.legend(fontsize=7, loc="upper right", ncol=2)
    fig.suptitle(
        f"{stream_name.capitalize()} empirical radial residual bands around merged trajectories",
        y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    filename = "trajectory_regularity_probability.png" if stream_name == "observed" else f"trajectory_regularity_probability_{stream_name}.png"
    fig.savefig(output / filename, dpi=220)
    plt.close(fig)


def plot_summary_heatmaps(output: Path, metric_rows: list[dict], scales: list[float], distances: list[float]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), squeeze=False)
    for slot in range(4):
        for row_index, (metric, title) in enumerate(
            (("residual_p95_deg", "observed P95 residual (deg)"), ("repeat_center_p95_deg", "repeat-center P95 (deg)"))
        ):
            axis = axes[row_index][slot]
            values = np.full((len(scales), len(distances)), np.nan)
            for item in metric_rows:
                if item["stream"] != "observed" or int(item["slot"]) != slot:
                    continue
                values[scales.index(float(item["scale"])), distances.index(float(item["distance_m"]))] = float(item[metric])
            image = axis.imshow(values, cmap="viridis", aspect="auto", vmin=0.0)
            axis.set_xticks(range(len(distances)), [f"{distance:g}" for distance in distances])
            axis.set_yticks(range(len(scales)), [f"{scale:g}" for scale in scales])
            axis.set_xlabel("distance (m)")
            axis.set_ylabel("radius scale")
            axis.set_title(f"slot {slot}: {title}")
            for i in range(len(scales)):
                for j in range(len(distances)):
                    if np.isfinite(values[i, j]):
                        color = "white" if values[i, j] > np.nanmax(values) * 0.55 else "black"
                        axis.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Observed trajectory regularity across radius and distance", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output / "trajectory_regularity_summary.png", dpi=220)
    plt.close(fig)


def fit_geometric_curve(rows: list[dict], bins: int = 128, center: np.ndarray | None = None) -> dict:
    """Fit only the supported camera-plane locus without assuming time phase.

    The polar parameter is measured around the robust point-cloud center. This
    treats a different starting yaw as a reparameterization of the same
    geometric path, which is the correct object for the current descriptive
    question.
    """
    u = np.asarray([float(row["u_deg"]) for row in rows], dtype=float)
    v = np.asarray([float(row["v_deg"]) for row in rows], dtype=float)
    if center is None:
        center = np.asarray([np.median(u), np.median(v)], dtype=float)
    theta = np.mod(np.arctan2(v - center[1], u - center[0]), TWO_PI)
    radius = np.hypot(u - center[0], v - center[1])
    grid = (np.arange(bins, dtype=float) + 0.5) * TWO_PI / bins
    indices = np.minimum((theta / TWO_PI * bins).astype(int), bins - 1)
    center_radius = np.full(bins, np.nan, dtype=float)
    for index in range(bins):
        mask = indices == index
        if np.any(mask):
            center_radius[index] = float(np.median(radius[mask]))
    center_radius = fill_short_circular_gaps(center_radius)
    fitted_radius = center_radius[indices]
    residual = np.abs(radius - fitted_radius)
    q50 = np.full(bins, np.nan, dtype=float)
    q90 = np.full(bins, np.nan, dtype=float)
    q95 = np.full(bins, np.nan, dtype=float)
    for index in range(bins):
        mask = indices == index
        if np.any(mask):
            q50[index], q90[index], q95[index] = np.percentile(residual[mask], [50, 90, 95])
    valid = np.isfinite(center_radius)
    return {
        "center": center,
        "grid": grid,
        "center_radius": center_radius,
        "curve_u": center[0] + fitted_radius * np.cos(theta),
        "curve_v": center[1] + fitted_radius * np.sin(theta),
        "grid_curve_u": center[0] + center_radius * np.cos(grid),
        "grid_curve_v": center[1] + center_radius * np.sin(grid),
        "theta": theta,
        "residual": residual,
        "q50": q50,
        "q90": q90,
        "q95": q95,
        "theta_coverage": float(np.count_nonzero(valid) / bins),
    }


def geometric_curve_distance(first: dict, second: dict) -> np.ndarray:
    valid = (
        np.isfinite(first["grid_curve_u"])
        & np.isfinite(first["grid_curve_v"])
        & np.isfinite(second["grid_curve_u"])
        & np.isfinite(second["grid_curve_v"])
    )
    return np.hypot(
        first["grid_curve_u"][valid] - second["grid_curve_u"][valid],
        first["grid_curve_v"][valid] - second["grid_curve_v"][valid],
    )


def geometric_shape_distance(first: dict, second: dict) -> np.ndarray:
    """Compare curve shape after removing each curve's camera-plane position."""
    first_u = first["grid_curve_u"] - first["center"][0]
    first_v = first["grid_curve_v"] - first["center"][1]
    second_u = second["grid_curve_u"] - second["center"][0]
    second_v = second["grid_curve_v"] - second["center"][1]
    valid = np.isfinite(first_u) & np.isfinite(first_v) & np.isfinite(second_u) & np.isfinite(second_v)
    return np.hypot(first_u[valid] - second_u[valid], first_v[valid] - second_v[valid])


def geometric_curve_descriptors(curve: dict) -> dict:
    valid = np.isfinite(curve["grid_curve_u"]) & np.isfinite(curve["grid_curve_v"])
    if np.count_nonzero(valid) < 2:
        return {"u_span_deg": float("nan"), "v_span_deg": float("nan"), "curve_length_deg": float("nan")}
    u = curve["grid_curve_u"][valid]
    v = curve["grid_curve_v"][valid]
    return {
        "center_u_deg": float(curve["center"][0]),
        "center_v_deg": float(curve["center"][1]),
        "u_span_deg": float(np.percentile(u, 99) - np.percentile(u, 1)),
        "v_span_deg": float(np.percentile(v, 99) - np.percentile(v, 1)),
        "curve_length_deg": float(np.sum(np.hypot(np.diff(u), np.diff(v)))),
    }


def plot_geometric_trajectory_grid(
    output: Path,
    grouped: dict[tuple[float, float, int], list[dict]],
    stream_name: str,
    scales: list[float],
    distances: list[float],
) -> None:
    fig, axes = plt.subplots(
        len(scales), len(distances), figsize=(4.0 * len(distances), 3.2 * len(scales)), squeeze=False
    )
    for row_index, scale in enumerate(scales):
        for col_index, distance_m in enumerate(distances):
            axis = axes[row_index][col_index]
            for slot, color in enumerate(SLOT_COLORS):
                points = grouped.get((scale, distance_m, slot), [])
                if not points:
                    continue
                merged = fit_geometric_curve(points)
                repeats = sorted({int(point["repeat"]) for point in points})
                for repeat in repeats:
                    repeat_points = [point for point in points if int(point["repeat"]) == repeat]
                    if len(repeat_points) < 8:
                        continue
                    repeat_curve = fit_geometric_curve(repeat_points, center=merged["center"])
                    axis.plot(
                        repeat_curve["grid_curve_u"],
                        repeat_curve["grid_curve_v"],
                        color=color,
                        alpha=0.20,
                        linewidth=0.8,
                    )
                if stream_name == "observed":
                    axis.scatter(
                        [point["u_deg"] for point in points],
                        [point["v_deg"] for point in points],
                        s=1.2,
                        alpha=0.08,
                        color=color,
                    )
                axis.plot(
                    merged["grid_curve_u"],
                    merged["grid_curve_v"],
                    color=color,
                    linewidth=1.8,
                    label=f"slot {slot}",
                )
            axis.set_title(f"scale={scale:g}, d={distance_m:g} m")
            axis.set_xlabel("camera azimuth (deg)")
            axis.set_ylabel("camera elevation (deg)")
            axis.grid(alpha=0.2)
            if row_index == 0 and col_index == 0:
                axis.legend(fontsize=8, loc="best")
    fig.suptitle(f"{stream_name.capitalize()} geometric loci and merged center curves", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output / f"geometric_trajectory_{stream_name}.png", dpi=220)
    plt.close(fig)


def plot_geometric_detail_grids(
    output: Path,
    grouped: dict[tuple[float, float, int], list[dict]],
    stream_name: str,
    scales: list[float],
    distances: list[float],
) -> None:
    """Show raw cloud, every repeat locus, and merged 2-D density per slot."""
    for slot in range(4):
        fig, axes = plt.subplots(
            len(scales), len(distances), figsize=(4.2 * len(distances), 3.4 * len(scales)), squeeze=False
        )
        for row_index, scale in enumerate(scales):
            for col_index, distance_m in enumerate(distances):
                axis = axes[row_index][col_index]
                points = grouped.get((scale, distance_m, slot), [])
                if not points:
                    axis.set_visible(False)
                    continue
                u = np.asarray([float(point["u_deg"]) for point in points])
                v = np.asarray([float(point["v_deg"]) for point in points])
                axis.hexbin(
                    u,
                    v,
                    gridsize=28,
                    mincnt=1,
                    bins="log",
                    cmap="Greys",
                    alpha=0.38,
                    linewidths=0.0,
                )
                repeats = sorted({int(point["repeat"]) for point in points})
                for repeat_index, repeat in enumerate(repeats):
                    repeat_points = [point for point in points if int(point["repeat"]) == repeat]
                    repeat_color = REPEAT_COLORS[repeat_index % len(REPEAT_COLORS)]
                    axis.scatter(
                        [point["u_deg"] for point in repeat_points],
                        [point["v_deg"] for point in repeat_points],
                        s=1.0,
                        alpha=0.045,
                        color=repeat_color,
                    )
                    if len(repeat_points) >= 8:
                        merged = fit_geometric_curve(points)
                        repeat_curve = fit_geometric_curve(repeat_points, center=merged["center"])
                        axis.plot(
                            repeat_curve["grid_curve_u"],
                            repeat_curve["grid_curve_v"],
                            color=repeat_color,
                            alpha=0.78,
                            linewidth=0.85,
                            label=f"repeat {repeat:g}",
                        )
                merged = fit_geometric_curve(points)
                axis.plot(
                    merged["grid_curve_u"],
                    merged["grid_curve_v"],
                    color="#000000",
                    linewidth=2.0,
                    label="merged center",
                )
                axis.set_title(f"r={scale:g}, d={distance_m:g} m, n={len(points)}")
                axis.set_xlabel("camera azimuth (deg)")
                axis.set_ylabel("camera elevation (deg)")
                axis.grid(alpha=0.18)
                if row_index == 0 and col_index == 0:
                    axis.legend(fontsize=7, loc="best", frameon=True)
        fig.suptitle(
            f"{stream_name.capitalize()} slot {slot}: raw cloud, repeats and merged density",
            y=1.01,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(output / f"geometric_detail_{stream_name}_slot{slot}.png", dpi=240)
        plt.close(fig)


def plot_geometric_probability_bands(
    output: Path,
    grouped: dict[tuple[float, float, int], list[dict]],
    stream_name: str,
    scales: list[float],
    distances: list[float],
) -> None:
    fig, axes = plt.subplots(
        len(scales), len(distances), figsize=(4.0 * len(distances), 3.0 * len(scales)), squeeze=False
    )
    for row_index, scale in enumerate(scales):
        for col_index, distance_m in enumerate(distances):
            axis = axes[row_index][col_index]
            for slot, color in enumerate(SLOT_COLORS):
                points = grouped.get((scale, distance_m, slot), [])
                if not points:
                    continue
                curve = fit_geometric_curve(points)
                theta_deg = np.degrees(curve["grid"])
                axis.fill_between(theta_deg, curve["q50"], curve["q95"], color=color, alpha=0.13)
                axis.plot(theta_deg, curve["q50"], color=color, linewidth=1.0, label=f"slot {slot} P50")
                axis.plot(theta_deg, curve["q95"], color=color, linewidth=0.8, linestyle="--")
            axis.set_title(f"scale={scale:g}, d={distance_m:g} m")
            axis.set_xlabel("geometric polar angle (deg)")
            axis.set_ylabel("radial residual (deg)")
            axis.set_xlim(0, 360)
            axis.grid(alpha=0.2)
            if row_index == 0 and col_index == 0:
                axis.legend(fontsize=7, loc="upper right", ncol=2)
    fig.suptitle(f"{stream_name.capitalize()} empirical bands around geometric loci", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output / f"geometric_probability_{stream_name}.png", dpi=220)
    plt.close(fig)


def plot_geometric_summary_heatmaps(output: Path, metric_rows: list[dict], stream_name: str, scales: list[float], distances: list[float]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), squeeze=False)
    for slot in range(4):
        for row_index, (metric, title) in enumerate(
            (("residual_p95_deg", "merged residual P95 (deg)"), ("repeat_curve_p95_deg", "repeat-locus P95 (deg)"))
        ):
            axis = axes[row_index][slot]
            values = np.full((len(scales), len(distances)), np.nan)
            for item in metric_rows:
                if item["stream"] != stream_name or int(item["slot"]) != slot:
                    continue
                values[scales.index(float(item["scale"])), distances.index(float(item["distance_m"]))] = float(item[metric])
            image = axis.imshow(values, cmap="viridis", aspect="auto", vmin=0.0)
            axis.set_xticks(range(len(distances)), [f"{distance:g}" for distance in distances])
            axis.set_yticks(range(len(scales)), [f"{scale:g}" for scale in scales])
            axis.set_xlabel("distance (m)")
            axis.set_ylabel("radius scale")
            axis.set_title(f"slot {slot}: {title}")
            for i in range(len(scales)):
                for j in range(len(distances)):
                    if np.isfinite(values[i, j]):
                        color = "white" if values[i, j] > np.nanmax(values) * 0.55 else "black"
                        axis.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(f"{stream_name.capitalize()} geometric trajectory regularity", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output / f"geometric_summary_{stream_name}.png", dpi=220)
    plt.close(fig)


def run_geometric_analysis(analysis_dir: Path, output: Path) -> None:
    truth = read_jsonl(analysis_dir / "truth_points.jsonl")
    observed = read_jsonl(analysis_dir / "observed_points.jsonl")
    streams = {"truth": truth, "observed": observed}
    grouped_streams = {name: group_rows(rows) for name, rows in streams.items()}
    scales = sorted({float(row["scale"]) for row in truth})
    distances = sorted({float(row["distance_m"]) for row in truth})
    metric_rows: list[dict] = []
    repeat_rows: list[dict] = []
    curve_cache: dict[tuple[str, float, float, int], dict] = {}
    for stream_name, grouped in grouped_streams.items():
        for (scale, distance_m, slot), points in sorted(grouped.items()):
            if len(points) < 8:
                continue
            merged = fit_geometric_curve(points)
            curve_cache[(stream_name, scale, distance_m, slot)] = merged
            residual = merged["residual"]
            outlier_rate, outlier_threshold = robust_outlier_rate(residual)
            repeat_mean_errors: list[float] = []
            repeat_p95_errors: list[float] = []
            for repeat in sorted({int(point["repeat"]) for point in points}):
                repeat_points = [point for point in points if int(point["repeat"]) == repeat]
                if len(repeat_points) < 8:
                    continue
                repeat_curve = fit_geometric_curve(repeat_points, center=merged["center"])
                distances_to_merged = geometric_curve_distance(repeat_curve, merged)
                repeat_mean = float(np.mean(distances_to_merged)) if distances_to_merged.size else float("nan")
                repeat_p95 = float(np.percentile(distances_to_merged, 95)) if distances_to_merged.size else float("nan")
                repeat_mean_errors.append(repeat_mean)
                repeat_p95_errors.append(repeat_p95)
                repeat_rows.append(
                    {
                        "stream": stream_name,
                        "scale": scale,
                        "distance_m": distance_m,
                        "slot": slot,
                        "repeat": repeat,
                        "samples": len(repeat_points),
                        "theta_coverage": repeat_curve["theta_coverage"],
                        "repeat_curve_mean_deg": repeat_mean,
                        "repeat_curve_p95_deg": repeat_p95,
                        "repeat_residual_p50_deg": float(np.percentile(repeat_curve["residual"], 50)),
                        "repeat_residual_p90_deg": float(np.percentile(repeat_curve["residual"], 90)),
                        "repeat_residual_p95_deg": float(np.percentile(repeat_curve["residual"], 95)),
                    }
                )
            metric_rows.append(
                {
                    "stream": stream_name,
                    "scale": scale,
                    "distance_m": distance_m,
                    "slot": slot,
                    "samples": len(points),
                    "runs": len({str(point["run"]) for point in points}),
                    "theta_coverage": merged["theta_coverage"],
                    "residual_mean_deg": float(np.mean(residual)),
                    "residual_median_deg": float(np.median(residual)),
                    "residual_p50_deg": float(np.percentile(residual, 50)),
                    "residual_p90_deg": float(np.percentile(residual, 90)),
                    "residual_p95_deg": float(np.percentile(residual, 95)),
                    "trajectory_width_p50_deg": float(np.percentile(residual, 50)),
                    "trajectory_width_p90_deg": float(np.percentile(residual, 90)),
                    "trajectory_width_p95_deg": float(np.percentile(residual, 95)),
                    "outlier_rate": outlier_rate,
                    "outlier_threshold_deg": outlier_threshold,
                    "repeat_curve_mean_deg": float(np.mean(repeat_mean_errors)) if repeat_mean_errors else float("nan"),
                    "repeat_curve_p95_deg": float(np.percentile(repeat_p95_errors, 95)) if repeat_p95_errors else float("nan"),
                    **geometric_curve_descriptors(merged),
                }
            )

    effect_rows: list[dict] = []
    for stream_name in streams:
        for slot in range(4):
            curves = [
                (scale, distance_m, curve_cache[(stream_name, scale, distance_m, slot)])
                for scale in scales
                for distance_m in distances
                if (stream_name, scale, distance_m, slot) in curve_cache
            ]
            for index_a, (scale_a, distance_a, curve_a) in enumerate(curves):
                for scale_b, distance_b, curve_b in curves[index_a + 1 :]:
                    same_scale = scale_a == scale_b
                    same_distance = distance_a == distance_b
                    if not (same_scale or same_distance) or (same_scale and same_distance):
                        continue
                    full_delta = geometric_curve_distance(curve_a, curve_b)
                    shape_delta = geometric_shape_distance(curve_a, curve_b)
                    if full_delta.size and shape_delta.size:
                        descriptor_a = geometric_curve_descriptors(curve_a)
                        descriptor_b = geometric_curve_descriptors(curve_b)
                        center_delta = float(np.linalg.norm(curve_a["center"] - curve_b["center"]))
                        adjacent = (
                            same_scale
                            and abs(distances.index(distance_a) - distances.index(distance_b)) == 1
                        ) or (
                            same_distance
                            and abs(scales.index(scale_a) - scales.index(scale_b)) == 1
                        )
                        effect_rows.append(
                            {
                                "stream": stream_name,
                                "slot": slot,
                                "change_type": "distance" if same_scale else "radius",
                                "adjacent": adjacent,
                                "from_scale": scale_a,
                                "from_distance_m": distance_a,
                                "to_scale": scale_b,
                                "to_distance_m": distance_b,
                                "center_delta_deg": center_delta,
                                "shape_mean_distance_deg": float(np.mean(shape_delta)),
                                "shape_p95_distance_deg": float(np.percentile(shape_delta, 95)),
                                "full_curve_mean_distance_deg": float(np.mean(full_delta)),
                                "full_curve_p95_distance_deg": float(np.percentile(full_delta, 95)),
                                "u_span_delta_deg": float(descriptor_b["u_span_deg"] - descriptor_a["u_span_deg"]),
                                "v_span_delta_deg": float(descriptor_b["v_span_deg"] - descriptor_a["v_span_deg"]),
                                "curve_length_delta_deg": float(descriptor_b["curve_length_deg"] - descriptor_a["curve_length_deg"]),
                                "curve_mean_distance_deg": float(np.mean(full_delta)),
                                "curve_p95_distance_deg": float(np.percentile(full_delta, 95)),
                            }
                        )

    write_csv(output / "geometric_trajectory_metrics.csv", metric_rows)
    write_csv(output / "geometric_trajectory_repeats.csv", repeat_rows)
    write_csv(output / "geometric_trajectory_condition_effects.csv", effect_rows)
    plot_geometric_trajectory_grid(output, grouped_streams["truth"], "truth", scales, distances)
    plot_geometric_trajectory_grid(output, grouped_streams["observed"], "observed", scales, distances)
    plot_geometric_detail_grids(output, grouped_streams["truth"], "truth", scales, distances)
    plot_geometric_detail_grids(output, grouped_streams["observed"], "observed", scales, distances)
    plot_geometric_probability_bands(output, grouped_streams["truth"], "truth", scales, distances)
    plot_geometric_probability_bands(output, grouped_streams["observed"], "observed", scales, distances)
    plot_geometric_summary_heatmaps(output, metric_rows, "truth", scales, distances)
    plot_geometric_summary_heatmaps(output, metric_rows, "observed", scales, distances)

    aggregate: dict[str, dict] = {}
    for stream_name in streams:
        stream_metrics = [row for row in metric_rows if row["stream"] == stream_name]
        stream_effects = [row for row in effect_rows if row["stream"] == stream_name]
        effect_summary: dict[str, dict] = {}
        for change_type in ("distance", "radius"):
            selected = [row for row in stream_effects if row["change_type"] == change_type]
            adjacent = [row for row in selected if row["adjacent"]]
            effect_summary[change_type] = {
                "all": {
                    "full_curve_p95_deg": summarize_values([row["full_curve_p95_distance_deg"] for row in selected]),
                    "shape_p95_deg": summarize_values([row["shape_p95_distance_deg"] for row in selected]),
                    "center_delta_deg": summarize_values([row["center_delta_deg"] for row in selected]),
                    "abs_u_span_delta_deg": summarize_values([abs(row["u_span_delta_deg"]) for row in selected]),
                    "abs_v_span_delta_deg": summarize_values([abs(row["v_span_delta_deg"]) for row in selected]),
                    "abs_curve_length_delta_deg": summarize_values([abs(row["curve_length_delta_deg"]) for row in selected]),
                },
                "adjacent": {
                    "full_curve_p95_deg": summarize_values([row["full_curve_p95_distance_deg"] for row in adjacent]),
                    "shape_p95_deg": summarize_values([row["shape_p95_distance_deg"] for row in adjacent]),
                    "center_delta_deg": summarize_values([row["center_delta_deg"] for row in adjacent]),
                    "abs_u_span_delta_deg": summarize_values([abs(row["u_span_delta_deg"]) for row in adjacent]),
                    "abs_v_span_delta_deg": summarize_values([abs(row["v_span_delta_deg"]) for row in adjacent]),
                    "abs_curve_length_delta_deg": summarize_values([abs(row["curve_length_delta_deg"]) for row in adjacent]),
                },
            }
        aggregate[stream_name] = {
            "merged_residual_p95_deg": summarize_values([row["residual_p95_deg"] for row in stream_metrics]),
            "trajectory_width_p50_deg": summarize_values([row["trajectory_width_p50_deg"] for row in stream_metrics]),
            "trajectory_width_p90_deg": summarize_values([row["trajectory_width_p90_deg"] for row in stream_metrics]),
            "trajectory_width_p95_deg": summarize_values([row["trajectory_width_p95_deg"] for row in stream_metrics]),
            "repeat_curve_p95_deg": summarize_values([row["repeat_curve_p95_deg"] for row in stream_metrics]),
            "outlier_rate": summarize_values([row["outlier_rate"] for row in stream_metrics]),
            "theta_coverage": summarize_values([row["theta_coverage"] for row in stream_metrics]),
            "effect_summary": effect_summary,
        }

    summary = {
        "schema_version": 1,
        "kind": "geometric_trajectory_regularity",
        "scope": "descriptive closed-locus fitting; no phase-based future prediction or forecast horizon",
        "parameterization": "camera azimuth/elevation point-cloud median center plus circular polar angle around that center",
        "curve_contract": "per-condition/per-slot polar-bin median radius on supported arcs only; gaps over two bins remain NaN; residual bands are empirical radial P50/P90/P95 bands",
        "conditions": len(scales) * len(distances),
        "scales": scales,
        "distances_m": distances,
        "truth_samples": len(truth),
        "observed_samples": len(observed),
        "truth_metric_rows": sum(row["stream"] == "truth" for row in metric_rows),
        "observed_metric_rows": sum(row["stream"] == "observed" for row in metric_rows),
        "aggregate_metrics": aggregate,
        "artifacts": [
            "geometric_trajectory_metrics.csv",
            "geometric_trajectory_repeats.csv",
            "geometric_trajectory_condition_effects.csv",
            "geometric_trajectory_truth.png",
            "geometric_trajectory_observed.png",
            "geometric_detail_truth_slot0.png",
            "geometric_detail_truth_slot1.png",
            "geometric_detail_truth_slot2.png",
            "geometric_detail_truth_slot3.png",
            "geometric_detail_observed_slot0.png",
            "geometric_detail_observed_slot1.png",
            "geometric_detail_observed_slot2.png",
            "geometric_detail_observed_slot3.png",
            "geometric_probability_truth.png",
            "geometric_probability_observed.png",
            "geometric_summary_truth.png",
            "geometric_summary_observed.png",
        ],
    }
    (output / "geometric_trajectory_regularity_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-geometric-diagnostic",
        action="store_true",
        help=(
            "also emit the legacy centroid/polar diagnostic; it is not valid "
            "as an authoritative fit for open or non-star-shaped arcs"
        ),
    )
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.resolve()
    output = (args.output or analysis_dir / "regularity").resolve()
    output.mkdir(parents=True, exist_ok=True)

    truth = read_jsonl(analysis_dir / "truth_points.jsonl")
    observed = read_jsonl(analysis_dir / "observed_points.jsonl")
    streams = {"truth": truth, "observed": observed}
    grouped_streams = {name: group_rows(rows) for name, rows in streams.items()}
    scales = sorted({float(row["scale"]) for row in truth})
    distances = sorted({float(row["distance_m"]) for row in truth})
    metric_rows: list[dict] = []
    repeat_rows: list[dict] = []
    curve_cache: dict[tuple[str, float, float, int], dict] = {}

    for stream_name, grouped in grouped_streams.items():
        for (scale, distance_m, slot), points in sorted(grouped.items()):
            if len(points) < 8:
                continue
            merged = fit_periodic_curve(points)
            curve_cache[(stream_name, scale, distance_m, slot)] = merged
            residual = merged["residual"]
            outlier_rate, outlier_threshold = robust_outlier_rate(residual)
            descriptors = curve_descriptors(merged)
            repeat_center_errors: list[float] = []
            repeat_p95_errors: list[float] = []
            for repeat in sorted({int(point["repeat"]) for point in points}):
                repeat_points = [point for point in points if int(point["repeat"]) == repeat]
                if len(repeat_points) < 8:
                    continue
                repeat_curve = fit_periodic_curve(repeat_points)
                distances_to_merged = curve_distance(repeat_curve, merged)
                if distances_to_merged.size:
                    repeat_center_errors.append(float(np.mean(distances_to_merged)))
                    repeat_p95_errors.append(float(np.percentile(distances_to_merged, 95)))
                repeat_rows.append(
                    {
                        "stream": stream_name,
                        "scale": scale,
                        "distance_m": distance_m,
                        "slot": slot,
                        "repeat": repeat,
                        "samples": len(repeat_points),
                        "phase_coverage": repeat_curve["phase_coverage"],
                        "repeat_center_mean_deg": repeat_center_errors[-1] if repeat_center_errors else float("nan"),
                        "repeat_center_p95_deg": repeat_p95_errors[-1] if repeat_p95_errors else float("nan"),
                    }
                )
            metric_rows.append(
                {
                    "stream": stream_name,
                    "scale": scale,
                    "distance_m": distance_m,
                    "slot": slot,
                    "samples": len(points),
                    "runs": len({str(point["run"]) for point in points}),
                    "phase_coverage": merged["phase_coverage"],
                    "residual_mean_deg": float(np.mean(residual)),
                    "residual_median_deg": float(np.median(residual)),
                    "residual_p90_deg": float(np.percentile(residual, 90)),
                    "residual_p95_deg": float(np.percentile(residual, 95)),
                    "outlier_rate": outlier_rate,
                    "outlier_threshold_deg": outlier_threshold,
                    "repeat_center_mean_deg": float(np.mean(repeat_center_errors)) if repeat_center_errors else float("nan"),
                    "repeat_center_p95_deg": float(np.percentile(repeat_p95_errors, 95)) if repeat_p95_errors else float("nan"),
                    **descriptors,
                }
            )

    condition_effect_rows: list[dict] = []
    for stream_name in streams:
        for slot in range(4):
            curves = [
                (scale, distance_m, curve_cache[(stream_name, scale, distance_m, slot)])
                for scale in scales
                for distance_m in distances
                if (stream_name, scale, distance_m, slot) in curve_cache
            ]
            for index_a, (scale_a, distance_a, curve_a) in enumerate(curves):
                for scale_b, distance_b, curve_b in curves[index_a + 1 :]:
                    same_scale = scale_a == scale_b
                    same_distance = distance_a == distance_b
                    if not (same_scale or same_distance) or (same_scale and same_distance):
                        continue
                    delta = curve_distance(curve_a, curve_b)
                    if delta.size:
                        condition_effect_rows.append(
                            {
                                "stream": stream_name,
                                "slot": slot,
                                "change_type": "distance" if same_scale else "radius",
                                "from_scale": scale_a,
                                "from_distance_m": distance_a,
                                "to_scale": scale_b,
                                "to_distance_m": distance_b,
                                "curve_mean_distance_deg": float(np.mean(delta)),
                                "curve_p95_distance_deg": float(np.percentile(delta, 95)),
                            }
                        )

    write_csv(output / "trajectory_regularity_metrics.csv", metric_rows)
    write_csv(output / "trajectory_regularity_repeats.csv", repeat_rows)
    write_csv(output / "trajectory_regularity_condition_effects.csv", condition_effect_rows)
    plot_trajectory_grid(output, grouped_streams["truth"], "truth", scales, distances)
    plot_trajectory_grid(output, grouped_streams["observed"], "observed", scales, distances)
    plot_probability_bands(output, grouped_streams["truth"], "truth", scales, distances)
    plot_probability_bands(output, grouped_streams["observed"], "observed", scales, distances)
    plot_summary_heatmaps(output, metric_rows, scales, distances)

    observed_metrics = [row for row in metric_rows if row["stream"] == "observed"]
    truth_metrics = [row for row in metric_rows if row["stream"] == "truth"]
    summary = {
        "schema_version": 1,
        "kind": "descriptive_trajectory_regularity",
        "scope": "phase-aligned repeated trajectory fitting; no future prediction or forecast horizon",
        "phase_contract": "absolute simulator target yaw modulo 2*pi; upstream per-run-zeroed phase is forbidden when repeats are merged",
        "curve_contract": "per-condition/per-slot phase-bin median center curve on supported arcs only; gaps over two bins remain NaN; residual bands are empirical radial P50/P90/P95 bands",
        "outlier_contract": "residual > median + 3*1.4826*MAD, with empirical P95 fallback when MAD is zero",
        "conditions": len(scales) * len(distances),
        "scales": scales,
        "distances_m": distances,
        "truth_samples": len(truth),
        "observed_samples": len(observed),
        "truth_metric_rows": len(truth_metrics),
        "observed_metric_rows": len(observed_metrics),
        "artifacts": [
            "trajectory_regularity_metrics.csv",
            "trajectory_regularity_repeats.csv",
            "trajectory_regularity_condition_effects.csv",
            "trajectory_regularity_truth.png",
            "trajectory_regularity_observed.png",
            "trajectory_regularity_probability_truth.png",
            "trajectory_regularity_probability.png",
            "trajectory_regularity_summary.png",
        ],
    }
    (output / "trajectory_regularity_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.include_geometric_diagnostic:
        geometric_output = output / "geometric"
        geometric_output.mkdir(parents=True, exist_ok=True)
        run_geometric_analysis(analysis_dir, geometric_output)
    sources = {
        str(analysis_dir / "truth_points.jsonl"): sha256(analysis_dir / "truth_points.jsonl"),
        str(analysis_dir / "observed_points.jsonl"): sha256(analysis_dir / "observed_points.jsonl"),
        str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
    }
    artifacts = {}
    for path in sorted(candidate for candidate in output.rglob("*") if candidate.is_file()):
        if path.name == "retention_manifest.json":
            continue
        artifacts[str(path.relative_to(output))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    retention = {
        "schema_version": 1,
        "kind": "trajectory_regularity_retention_manifest",
        "classification": "long_term_private_evidence",
        "deletion_allowed": False,
        "scope": "descriptive truth/observation trajectory regularity; no future predictor",
        "sources": sources,
        "artifacts": artifacts,
    }
    (output / "retention_manifest.json").write_text(
        json.dumps(retention, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
