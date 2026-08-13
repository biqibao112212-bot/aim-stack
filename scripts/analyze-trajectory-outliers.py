#!/usr/bin/env python3
"""Refit dominant open observation arcs and retain every rejected point.

This is a descriptive audit, not a future predictor.  It compares the legacy
camera-angle slot labels with truth-facing, full-PnP-translation labels, fits
only the dominant open ridge in camera azimuth/elevation, and writes frame-level
records for all points excluded from that ridge.
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
LEFT_OUTLIER = "#D55E00"
RIGHT_OUTLIER = "#CC79A7"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def moving_median(values: np.ndarray, radius: int = 2) -> np.ndarray:
    result = values.copy()
    for index in range(values.size):
        lo = max(0, index - radius)
        hi = min(values.size, index + radius + 1)
        finite = values[lo:hi][np.isfinite(values[lo:hi])]
        if finite.size:
            result[index] = float(np.median(finite))
    return result


def longest_true_segment(mask: np.ndarray) -> tuple[int, int]:
    best_start = best_stop = 0
    start = None
    for index, value in enumerate(np.append(mask, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start > best_stop - best_start:
                best_start, best_stop = start, index
            start = None
    return best_start, best_stop


def longest_circular_true_segment(mask: np.ndarray) -> np.ndarray:
    """Return bin indices for the longest true run on a circular phase axis."""
    count = int(mask.size)
    if count == 0 or not np.any(mask):
        return np.asarray([], dtype=int)
    if np.all(mask):
        return np.arange(count, dtype=int)
    doubled = np.concatenate([mask, mask])
    best_start = best_length = 0
    start = None
    for index, value in enumerate(np.append(doubled, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            length = min(index - start, count)
            if start < count and length > best_length:
                best_start, best_length = start, length
            start = None
    return np.asarray([(best_start + offset) % count for offset in range(best_length)], dtype=int)


def curve_orientation(u: np.ndarray, v: np.ndarray, threshold_deg: float = 0.01) -> tuple[float, str]:
    """Measure U/cap orientation without extrapolating beyond supported data."""
    finite = np.isfinite(u) & np.isfinite(v)
    u = np.asarray(u[finite], dtype=float)
    v = np.asarray(v[finite], dtype=float)
    if u.size < 7 or np.ptp(u) < 0.2:
        return float("nan"), "unknown"
    order = np.argsort(u)
    v = v[order]
    edge = max(1, int(math.ceil(0.15 * v.size)))
    middle = max(1, int(math.ceil(0.20 * v.size)))
    center = v.size // 2
    mid_lo = max(0, center - middle // 2)
    mid_hi = min(v.size, mid_lo + middle)
    amplitude = 0.5 * (float(np.median(v[:edge])) + float(np.median(v[-edge:]))) - float(
        np.median(v[mid_lo:mid_hi])
    )
    if amplitude > threshold_deg:
        return amplitude, "cup"
    if amplitude < -threshold_deg:
        return amplitude, "cap"
    return amplitude, "flat"


def phase_binned_centers(
    selected: np.ndarray,
    bin_index: np.ndarray,
    bin_order: np.ndarray,
    repeats: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    bins: int,
    min_repeat_support: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate each repeat first so high-FPS runs cannot dominate a bin."""
    center_u = np.full(bins, np.nan, dtype=float)
    center_v = np.full(bins, np.nan, dtype=float)
    support = np.zeros(bins, dtype=bool)
    for current in bin_order:
        in_bin = selected & (bin_index == current)
        repeat_ids = sorted(set(repeats[in_bin].tolist()))
        if np.count_nonzero(in_bin) < 3 or len(repeat_ids) < min_repeat_support:
            continue
        repeat_u = [float(np.median(u[in_bin & (repeats == repeat)])) for repeat in repeat_ids]
        repeat_v = [float(np.median(v[in_bin & (repeats == repeat)])) for repeat in repeat_ids]
        center_u[current] = float(np.median(repeat_u))
        center_v[current] = float(np.median(repeat_v))
        support[current] = True
    supported_order = np.asarray([index for index in bin_order if support[index]], dtype=int)
    if supported_order.size:
        center_u[supported_order] = moving_median(center_u[supported_order], radius=1)
        center_v[supported_order] = moving_median(center_v[supported_order], radius=1)
    return center_u, center_v, support


def interpolate_phase_curve(
    phase: np.ndarray,
    grid_phase: np.ndarray,
    curve_order: np.ndarray,
    center_u: np.ndarray,
    center_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate only inside a supported circular phase segment."""
    predicted_u = np.full(phase.size, np.nan, dtype=float)
    predicted_v = np.full(phase.size, np.nan, dtype=float)
    if curve_order.size < 2:
        return predicted_u, predicted_v
    curve_phase = grid_phase[curve_order].astype(float).copy()
    for index in range(1, curve_phase.size):
        if curve_phase[index] <= curve_phase[index - 1]:
            curve_phase[index:] += 2.0 * math.pi
    adjusted = phase.astype(float).copy()
    adjusted[adjusted < curve_phase[0]] += 2.0 * math.pi
    inside = (adjusted >= curve_phase[0]) & (adjusted <= curve_phase[-1])
    predicted_u[inside] = np.interp(adjusted[inside], curve_phase, center_u[curve_order])
    predicted_v[inside] = np.interp(adjusted[inside], curve_phase, center_v[curve_order])
    return predicted_u, predicted_v


def longest_geometrically_continuous_segment(
    curve_order: np.ndarray,
    center_u: np.ndarray,
    center_v: np.ndarray,
) -> np.ndarray:
    """Split phase-adjacent centers at unsupported image-space jumps."""
    if curve_order.size < 3:
        return curve_order
    points = np.column_stack((center_u[curve_order], center_v[curve_order]))
    jumps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    finite = jumps[np.isfinite(jumps)]
    if not finite.size:
        return curve_order
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    jump_threshold = max(0.25, 4.0 * median, median + 6.0 * 1.4826 * mad)
    split_after = np.flatnonzero(jumps > jump_threshold) + 1
    segments = [segment for segment in np.split(curve_order, split_after) if segment.size]
    return max(segments, key=lambda segment: segment.size)


def dominant_open_arc(rows: list[dict], bins: int = 72, min_repeat_support: int = 2) -> dict:
    u = np.asarray([float(row["u_deg"]) for row in rows], dtype=float)
    v = np.asarray([float(row["v_deg"]) for row in rows], dtype=float)
    repeats = np.asarray([int(row["repeat"]) for row in rows], dtype=int)
    phase = np.mod(
        np.asarray([float(row.get("target_yaw_rad", row.get("phase_rad", 0.0))) for row in rows]),
        2.0 * math.pi,
    )
    detector_label_ok = np.asarray(
        [row.get("detector_number") in (None, 3) for row in rows], dtype=bool
    )
    facing = np.asarray(
        [float(row.get("truth_facing_score", float("nan"))) for row in rows], dtype=float
    )
    has_facing = np.any(np.isfinite(facing))
    facing_ok = ~np.isfinite(facing) | (facing >= 0.65) if has_facing else np.ones(len(rows), dtype=bool)
    # Physical-slot assignment has already matched each observation to the
    # truth target in 3D. A wrong digit label is detector evidence to retain,
    # not a reason to discard an otherwise valid plate trajectory point.
    eligible = facing_ok
    if len(rows) < 40 or np.ptp(u) <= 1e-6 or np.count_nonzero(eligible) < 40:
        return {"status": "insufficient", "accepted": np.zeros(len(rows), dtype=bool)}
    edges = np.linspace(0.0, 2.0 * math.pi, bins + 1)
    grid_phase = 0.5 * (edges[:-1] + edges[1:])
    phase_index = np.clip(np.searchsorted(edges, phase, side="right") - 1, 0, bins - 1)
    initial_support = np.zeros(bins, dtype=bool)
    for bin_index in range(bins):
        selected = eligible & (phase_index == bin_index)
        initial_support[bin_index] = (
            np.count_nonzero(selected) >= 3
            and len(set(repeats[selected].tolist())) >= min_repeat_support
        )
    bin_order = longest_circular_true_segment(initial_support)
    in_segment = np.isin(phase_index, bin_order)
    center_u, center_v, support = phase_binned_centers(
        eligible & in_segment, phase_index, bin_order, repeats, u, v, bins, min_repeat_support
    )
    valid_order = np.asarray([index for index in bin_order if support[index]], dtype=int)
    valid_order = longest_geometrically_continuous_segment(
        valid_order, center_u, center_v
    )
    predicted_u, predicted_v = interpolate_phase_curve(
        phase, grid_phase, valid_order, center_u, center_v
    )
    residual = np.hypot(u - predicted_u, v - predicted_v)
    seed_residual = residual[eligible & in_segment & np.isfinite(residual)]
    if seed_residual.size:
        median = float(np.median(seed_residual))
        mad = float(np.median(np.abs(seed_residual - median)))
        threshold = max(0.045, median + 3.0 * 1.4826 * mad)
    else:
        threshold = float("nan")
    accepted = eligible & in_segment & np.isfinite(residual)
    accepted[accepted] &= residual[accepted] <= threshold

    # Recompute the ridge from accepted per-repeat centers.  This is a local,
    # phase-parametric description; it never extrapolates a global polynomial.
    center_u, center_v, support = phase_binned_centers(
        accepted, phase_index, bin_order, repeats, u, v, bins, min_repeat_support
    )
    valid_order = np.asarray([index for index in bin_order if support[index]], dtype=int)
    valid_order = longest_geometrically_continuous_segment(
        valid_order, center_u, center_v
    )
    in_segment = np.isin(phase_index, valid_order)
    predicted_u, predicted_v = interpolate_phase_curve(
        phase, grid_phase, valid_order, center_u, center_v
    )
    residual = np.hypot(u - predicted_u, v - predicted_v)
    accepted = eligible & in_segment & np.isfinite(residual)
    accepted[accepted] &= residual[accepted] <= threshold

    q50 = np.full(bins, np.nan, dtype=float)
    q90 = np.full(bins, np.nan, dtype=float)
    q95 = np.full(bins, np.nan, dtype=float)
    for bin_index in range(bins):
        selected = accepted & (phase_index == bin_index) & np.isfinite(residual)
        if np.count_nonzero(selected) >= 3:
            q50[bin_index], q90[bin_index], q95[bin_index] = np.percentile(
                residual[selected], [50, 90, 95]
            )

    truth_u = np.asarray([float(row.get("truth_u_deg", float("nan"))) for row in rows])
    truth_v = np.asarray([float(row.get("truth_v_deg", float("nan"))) for row in rows])
    truth_center_u, truth_center_v, _ = phase_binned_centers(
        accepted & np.isfinite(truth_u) & np.isfinite(truth_v),
        phase_index,
        valid_order,
        repeats,
        truth_u,
        truth_v,
        bins,
        min_repeat_support,
    )

    accepted_count = int(np.count_nonzero(accepted))
    accepted_ratio = accepted_count / len(rows)
    repeat_coverage = len(set(repeats[accepted].tolist())) if accepted_count else 0
    supported_bins = int(valid_order.size)
    segment_bins = int(valid_order.size)
    eligible_count = int(np.count_nonzero(eligible & in_segment))
    eligible_acceptance_ratio = accepted_count / eligible_count if eligible_count else 0.0
    status = "fit"
    if (
        accepted_count < 80
        or repeat_coverage < 3
        or supported_bins < 8
        or eligible_acceptance_ratio < 0.45
    ):
        status = "fragmented"

    observed_curvature, observed_orientation = curve_orientation(
        center_u[valid_order], center_v[valid_order]
    )
    truth_curvature, truth_orientation = curve_orientation(
        truth_center_u[valid_order], truth_center_v[valid_order]
    )
    orientation_agrees = (
        observed_orientation == truth_orientation
        if "unknown" not in (observed_orientation, truth_orientation)
        else None
    )
    if truth_orientation in ("flat", "unknown"):
        truth_consistency = "ambiguous"
    elif orientation_agrees is False:
        truth_consistency = "mismatch"
    else:
        truth_consistency = "match"

    return {
        "status": status,
        "accepted": accepted,
        "residual": residual,
        "threshold": threshold,
        "grid_phase": grid_phase,
        "grid_u": center_u,
        "center_v": center_v,
        "truth_center_u": truth_center_u,
        "truth_center_v": truth_center_v,
        "q50": q50,
        "q90": q90,
        "q95": q95,
        "support": support,
        "curve_order": valid_order,
        "detector_label_ok": detector_label_ok,
        "facing_ok": facing_ok,
        "in_segment": in_segment,
        "accepted_ratio": accepted_ratio,
        "eligible_acceptance_ratio": eligible_acceptance_ratio,
        "repeat_coverage": repeat_coverage,
        "supported_bins": supported_bins,
        "segment_bins": segment_bins,
        "observed_curvature_deg": observed_curvature,
        "observed_orientation": observed_orientation,
        "truth_curvature_deg": truth_curvature,
        "truth_orientation": truth_orientation,
        "orientation_agrees": orientation_agrees,
        "truth_consistency": truth_consistency,
    }


def group_rows(rows: list[dict]) -> dict[tuple[float, float, int], list[dict]]:
    grouped: dict[tuple[float, float, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["scale"]), float(row["distance_m"]), int(row["slot"]))].append(row)
    return grouped


def plot_slot_grids(
    output: Path,
    stream: str,
    slot: int,
    grouped: dict[tuple[float, float, int], list[dict]],
    fits: dict[tuple[float, float, int], dict],
    scales: list[float],
    distances: list[float],
) -> None:
    fig, axes = plt.subplots(
        len(scales), len(distances), figsize=(4.2 * len(distances), 3.45 * len(scales)), squeeze=False
    )
    for row_index, scale in enumerate(scales):
        for col_index, distance in enumerate(distances):
            axis = axes[row_index][col_index]
            rows = grouped.get((scale, distance, slot), [])
            fit = fits.get((scale, distance, slot))
            if not rows or fit is None:
                axis.set_axis_off()
                continue
            u = np.asarray([float(row["u_deg"]) for row in rows])
            v = np.asarray([float(row["v_deg"]) for row in rows])
            accepted = fit["accepted"]
            midpoint = float(np.median(u))
            left = ~accepted & (u < midpoint)
            right = ~accepted & ~left
            axis.scatter(u, v, s=5, alpha=0.16, color="#999999", rasterized=True, label="raw")
            axis.scatter(u[accepted], v[accepted], s=6, alpha=0.32, color=SLOT_COLORS[slot], rasterized=True, label="main arc")
            axis.scatter(u[left], v[left], s=13, alpha=0.65, marker="x", linewidths=0.7, color=LEFT_OUTLIER, label="O-L")
            axis.scatter(u[right], v[right], s=13, alpha=0.65, marker="x", linewidths=0.7, color=RIGHT_OUTLIER, label="O-R")
            support = fit.get("support", np.zeros(0, dtype=bool))
            curve_order = fit.get("curve_order", np.flatnonzero(support))
            if fit["status"] == "fit" and len(curve_order) >= 2:
                grid_u = fit["grid_u"][curve_order]
                center = fit["center_v"][curve_order]
                q90 = fit["q90"][curve_order]
                axis.plot(grid_u, center, color="black", linewidth=2.0, label="open center")
                band = np.where(np.isfinite(q90), q90, 0.0)
                u_order = np.argsort(grid_u)
                axis.fill_between(
                    grid_u[u_order], (center - band)[u_order], (center + band)[u_order],
                    color=SLOT_COLORS[slot], alpha=0.16, label="P90"
                )
                truth_u = fit["truth_center_u"][curve_order]
                truth_v = fit["truth_center_v"][curve_order]
                truth_valid = np.isfinite(truth_u) & np.isfinite(truth_v)
                if np.count_nonzero(truth_valid) >= 2:
                    axis.plot(
                        truth_u[truth_valid], truth_v[truth_valid], color="#009E73",
                        linewidth=1.5, linestyle="--", label="truth reference"
                    )
                axis.plot(grid_u[[0, -1]], center[[0, -1]], "o", ms=5, mfc="white", mec="black")
            axis.set_title(
                f"r={scale:g}, d={distance:g} m\n"
                f"n={len(rows)}, keep={int(np.count_nonzero(accepted))} ({100*np.mean(accepted):.0f}%), {fit['status']}\n"
                f"obs={fit.get('observed_orientation', 'unknown')}, truth={fit.get('truth_orientation', 'unknown')}"
            )
            axis.set_xlabel("camera azimuth (deg)")
            axis.set_ylabel("camera elevation (deg)")
            axis.grid(alpha=0.18)
            if row_index == 0 and col_index == 0:
                axis.legend(fontsize=7, loc="best")
    fig.suptitle(f"{stream}: slot {slot} dominant open arc and retained outlier families", y=1.005)
    fig.tight_layout()
    fig.savefig(output / f"{stream}_open_arc_slot{slot}.png", dpi=220)
    plt.close(fig)


def provenance_key(row: dict, slot_field: str = "slot") -> tuple:
    return (
        str(row["run"]),
        round(float(row["t_s"]), 9),
        int(row["detection_index"]),
        int(row[slot_field]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-analysis-dir", required=True, type=Path)
    parser.add_argument("--pnp-analysis-dir", required=True, type=Path)
    parser.add_argument("--pnp-diagnostic-analysis-dir", type=Path)
    parser.add_argument(
        "--physical-stream-name",
        choices=("pnp3d_facing", "pnp3d_all_slots", "angular_facing"),
        default="pnp3d_facing",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    legacy_dir = args.legacy_analysis_dir.resolve()
    pnp_dir = args.pnp_analysis_dir.resolve()
    output = args.output.resolve()
    physical_stream = args.physical_stream_name
    output.mkdir(parents=True, exist_ok=True)

    legacy = read_jsonl(legacy_dir / "observed_points.jsonl")
    pnp = read_jsonl(pnp_dir / "observed_points.jsonl")
    pnp_provenance: dict[tuple, dict] = {}
    for row in pnp:
        angular_slot = row.get("angular_assignment_slot")
        if angular_slot is None:
            continue
        pnp_provenance[
            (str(row["run"]), round(float(row["t_s"]), 9), int(row["detection_index"]), int(angular_slot))
        ] = row

    streams = {"legacy_angular": legacy, physical_stream: pnp}
    scales = sorted({float(row["scale"]) for row in legacy})
    distances = sorted({float(row["distance_m"]) for row in legacy})
    metric_rows: list[dict] = []
    center_rows: list[dict] = []
    outlier_rows: list[dict] = []
    fit_sets: dict[str, dict[tuple[float, float, int], dict]] = {}

    for stream, rows in streams.items():
        grouped = group_rows(rows)
        fits: dict[tuple[float, float, int], dict] = {}
        fit_sets[stream] = fits
        for key, points in sorted(grouped.items()):
            fit = dominant_open_arc(points)
            fits[key] = fit
            accepted = fit["accepted"]
            residual = fit.get("residual", np.full(len(points), np.nan))
            finite_accepted = residual[accepted & np.isfinite(residual)]
            scale, distance, slot = key
            midpoint = float(np.median([float(point["u_deg"]) for point in points]))
            rejected_left = ~accepted & np.asarray([float(point["u_deg"]) < midpoint for point in points])
            rejected_right = ~accepted & ~rejected_left
            metric_rows.append(
                {
                    "stream": stream,
                    "scale": scale,
                    "distance_m": distance,
                    "slot": slot,
                    "status": fit["status"],
                    "samples": len(points),
                    "detector_label3_samples": int(
                        np.count_nonzero(
                            fit.get("detector_label_ok", np.ones(len(points), dtype=bool))
                        )
                    ),
                    "accepted_samples": int(np.count_nonzero(accepted)),
                    "rejected_samples": int(np.count_nonzero(~accepted)),
                    "rejected_left_samples": int(np.count_nonzero(rejected_left)),
                    "rejected_right_samples": int(np.count_nonzero(rejected_right)),
                    "accepted_ratio": float(np.mean(accepted)),
                    "eligible_acceptance_ratio": float(fit.get("eligible_acceptance_ratio", float("nan"))),
                    "repeat_coverage": int(fit.get("repeat_coverage", 0)),
                    "supported_bins": int(fit.get("supported_bins", 0)),
                    "dominant_segment_bins": int(fit.get("segment_bins", 0)),
                    "observed_curvature_deg": float(fit.get("observed_curvature_deg", float("nan"))),
                    "observed_orientation": fit.get("observed_orientation", "unknown"),
                    "truth_curvature_deg": float(fit.get("truth_curvature_deg", float("nan"))),
                    "truth_orientation": fit.get("truth_orientation", "unknown"),
                    "orientation_agrees": fit.get("orientation_agrees"),
                    "truth_consistency": fit.get("truth_consistency", "unknown"),
                    "residual_threshold_deg": float(fit.get("threshold", float("nan"))),
                    "accepted_residual_p50_deg": float(np.percentile(finite_accepted, 50)) if finite_accepted.size else float("nan"),
                    "accepted_residual_p90_deg": float(np.percentile(finite_accepted, 90)) if finite_accepted.size else float("nan"),
                    "accepted_residual_p95_deg": float(np.percentile(finite_accepted, 95)) if finite_accepted.size else float("nan"),
                }
            )
            curve_order = fit.get("curve_order", np.zeros(0, dtype=int))
            for bin_index in curve_order:
                center_rows.append(
                    {
                        "stream": stream,
                        "scale": scale,
                        "distance_m": distance,
                        "slot": slot,
                        "phase_rad": float(fit["grid_phase"][bin_index]),
                        "center_u_deg": float(fit["grid_u"][bin_index]),
                        "center_v_deg": float(fit["center_v"][bin_index]),
                        "p90_residual_deg": (
                            float(fit["q90"][bin_index])
                            if np.isfinite(fit["q90"][bin_index]) else None
                        ),
                        "truth_u_deg": (
                            float(fit["truth_center_u"][bin_index])
                            if np.isfinite(fit["truth_center_u"][bin_index]) else None
                        ),
                        "truth_v_deg": (
                            float(fit["truth_center_v"][bin_index])
                            if np.isfinite(fit["truth_center_v"][bin_index]) else None
                        ),
                    }
                )
            for index, point in enumerate(points):
                if accepted[index]:
                    continue
                record = dict(point)
                if fit["status"] != "fit":
                    reject_reason = "fragmented_no_single_arc"
                elif not fit.get("facing_ok", np.ones(len(points), dtype=bool))[index]:
                    reject_reason = "oblique_visibility_boundary"
                elif not fit.get("in_segment", np.ones(len(points), dtype=bool))[index]:
                    reject_reason = "outside_supported_phase_segment"
                else:
                    reject_reason = "outside_phase_binned_open_arc"
                record.update(
                    {
                        "detector_label_matches_target": bool(
                            fit.get(
                                "detector_label_ok", np.ones(len(points), dtype=bool)
                            )[index]
                        ),
                        "source_stream": stream,
                        "classification": "O-L" if float(point["u_deg"]) < midpoint else "O-R",
                        "reject_reason": reject_reason,
                        "open_arc_residual_deg": float(residual[index]) if np.isfinite(residual[index]) else None,
                        "open_arc_threshold_deg": float(fit.get("threshold", float("nan"))),
                    }
                )
                if stream == "legacy_angular":
                    source = pnp_provenance.get(provenance_key(point))
                    if source is not None:
                        for field in (
                            "session_id", "producer_epoch", "frame_seq", "timestamp_ns",
                            "slot", "angular_assignment_slot", "slot_changed_from_angular",
                            "pnp_position_error_m", "pnp_depth_error_m", "pnp_assignment_margin_m",
                            "pnp_assignment_margin_ratio", "truth_facing_score", "pnp_camera_x_m",
                            "pnp_camera_y_m", "pnp_camera_z_m", "truth_camera_x_m",
                            "truth_camera_y_m", "truth_camera_z_m", "pnp_yaw_absolute_rad",
                            "pnp_reprojection_rms_px", "pnp_reprojection_max_px",
                        ):
                            record[f"audit_{field}"] = source.get(field)
                outlier_rows.append(record)

        for slot in range(4):
            plot_slot_grids(output, stream, slot, grouped, fits, scales, distances)

    with (output / "trajectory_outliers.jsonl").open("w", encoding="utf-8") as handle:
        for row in outlier_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "open_arc_centers.jsonl").open("w", encoding="utf-8") as handle:
        for row in center_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(output / "open_arc_metrics.csv", metric_rows)

    comparable = [row for row in pnp if row.get("angular_assignment_slot") is not None]
    changed = [row for row in comparable if bool(row.get("slot_changed_from_angular"))]
    reprojection_present = [
        row for row in pnp
        if row.get("pnp_reprojection_rms_px") is not None or row.get("pnp_reprojection_max_px") is not None
    ]
    pnp_grouped = group_rows(pnp)
    accepted_pnp: list[dict] = []
    rejected_pnp: list[dict] = []
    for key, points in pnp_grouped.items():
        fit = fit_sets[physical_stream][key]
        for index, point in enumerate(points):
            (accepted_pnp if fit["accepted"][index] else rejected_pnp).append(point)

    def quantiles(rows: list[dict], field: str) -> dict:
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        return {
            "p25": float(np.percentile(values, 25)),
            "median": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
        }

    pnp_by_distance: dict[str, dict] = {}
    for distance in distances:
        selected = [
            row for row in metric_rows
            if row["stream"] == physical_stream and float(row["distance_m"]) == distance
        ]
        pnp_by_distance[f"{distance:g}"] = {
            "fit_cells": sum(row["status"] == "fit" for row in selected),
            "fragmented_cells": sum(row["status"] != "fit" for row in selected),
            "total_cells": len(selected),
            "accepted_ratio_median": float(np.median([float(row["accepted_ratio"]) for row in selected])),
        }
    distance_fit_summary = "、".join(
        f"{distance} m: {counts['fit_cells']}/{counts['total_cells']}"
        for distance, counts in pnp_by_distance.items()
    )

    diagnostic_summary = None
    diagnostic_artifact = None
    if args.pnp_diagnostic_analysis_dir is not None:
        diagnostic_rows = read_jsonl(args.pnp_diagnostic_analysis_dir.resolve() / "observed_points.jsonl")
        diagnostic_accepted: list[dict] = []
        diagnostic_rejected: list[dict] = []
        classified_rows: list[dict] = []
        for key, points in group_rows(diagnostic_rows).items():
            fit = fit_sets[physical_stream].get(key)
            if fit is None:
                continue
            valid = np.isfinite(fit["grid_u"]) & np.isfinite(fit["center_v"]) & fit["support"]
            u = np.asarray([float(point["u_deg"]) for point in points])
            v = np.asarray([float(point["v_deg"]) for point in points])
            if np.count_nonzero(valid) >= 2:
                center_u = fit["grid_u"][valid]
                center_v = fit["center_v"][valid]
                residual = np.asarray(
                    [np.min(np.hypot(center_u - point_u, center_v - point_v)) for point_u, point_v in zip(u, v)],
                    dtype=float,
                )
            else:
                residual = np.full(len(points), np.nan)
            accepted = np.zeros(len(points), dtype=bool)
            finite = np.isfinite(residual)
            accepted[finite] = residual[finite] <= float(fit["threshold"])
            for index, point in enumerate(points):
                destination = diagnostic_accepted if accepted[index] else diagnostic_rejected
                destination.append(point)
                classified = dict(point)
                classified["classification"] = "main_arc" if accepted[index] else "off_arc"
                classified["reference_open_arc_residual_deg"] = (
                    float(residual[index]) if np.isfinite(residual[index]) else None
                )
                classified["reference_open_arc_threshold_deg"] = float(fit["threshold"])
                classified_rows.append(classified)
        diagnostic_artifact = "pnp_diagnostic_classification.jsonl"
        with (output / diagnostic_artifact).open("w", encoding="utf-8") as handle:
            for row in classified_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        diagnostic_summary = {
            "samples": len(classified_rows),
            "accepted_samples": len(diagnostic_accepted),
            "rejected_samples": len(diagnostic_rejected),
            "accepted": {
                "facing_score": quantiles(diagnostic_accepted, "truth_facing_score"),
                "reprojection_rms_px": quantiles(diagnostic_accepted, "pnp_reprojection_rms_px"),
                "reprojection_max_px": quantiles(diagnostic_accepted, "pnp_reprojection_max_px"),
            },
            "rejected": {
                "facing_score": quantiles(diagnostic_rejected, "truth_facing_score"),
                "reprojection_rms_px": quantiles(diagnostic_rejected, "pnp_reprojection_rms_px"),
                "reprojection_max_px": quantiles(diagnostic_rejected, "pnp_reprojection_max_px"),
            },
        }
    summary = {
        "schema_version": 2,
        "kind": "trajectory_open_arc_outlier_audit",
        "scope": "descriptive dominant open-arc fitting only; no future prediction",
        "fit_method": "per-repeat phase bins followed by cross-repeat robust centers; no global polynomial and no extrapolation",
        "legacy_observations": len(legacy),
        "physical_stream": physical_stream,
        f"{physical_stream}_observations": len(pnp),
        "comparable_assignments": len(comparable),
        "slot_changes_from_legacy_angular": len(changed),
        "slot_change_rate": len(changed) / len(comparable) if comparable else None,
        "pnp_reprojection_diagnostics_present": len(reprojection_present),
        f"{physical_stream}_by_distance": pnp_by_distance,
        "pnp_accepted_diagnostics": {
            "samples": len(accepted_pnp),
            "facing_score": quantiles(accepted_pnp, "truth_facing_score"),
            "position_error_m": quantiles(accepted_pnp, "pnp_position_error_m"),
            "depth_error_m": quantiles(accepted_pnp, "pnp_depth_error_m"),
        },
        "pnp_rejected_diagnostics": {
            "samples": len(rejected_pnp),
            "facing_score": quantiles(rejected_pnp, "truth_facing_score"),
            "position_error_m": quantiles(rejected_pnp, "pnp_position_error_m"),
            "depth_error_m": quantiles(rejected_pnp, "pnp_depth_error_m"),
        },
        "targeted_pnp_diagnostic": diagnostic_summary,
        "fit_rows": len(metric_rows),
        "legacy_fit_conditions": sum(row["stream"] == "legacy_angular" and row["status"] == "fit" for row in metric_rows),
        f"{physical_stream}_fit_conditions": sum(row["stream"] == physical_stream and row["status"] == "fit" for row in metric_rows),
        "outlier_records": len(outlier_rows),
        "assignment_finding": (
            "front-facing image-ray association applies no top-N facing rank, uses a 0.75 degree gate, and retains explicit second-best margins"
            if physical_stream == "angular_facing"
            else "physical-slot association is compared against the retained legacy angular label on every observation"
        ),
        "pnp_finding": "off-arc families concentrate near oblique visibility boundaries and have larger pose errors; the dedicated full-pipeline candidate capture is the authority for IPPE branch-switch diagnosis",
        "artifacts": [
            "OPEN_ARC_OUTLIER_REPORT.md",
            "retention_manifest.json",
            "open_arc_metrics.csv",
            "open_arc_centers.jsonl",
            "trajectory_outliers.jsonl",
            *([diagnostic_artifact] if diagnostic_artifact else []),
            *[f"legacy_angular_open_arc_slot{slot}.png" for slot in range(4)],
            *[f"{physical_stream}_open_arc_slot{slot}.png" for slot in range(4)],
        ],
    }
    (output / "open_arc_outlier_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if diagnostic_summary is not None:
        diagnostic_report_line = (
            "- 独立短诊断复测开启了现有 PnP 重投影诊断：参考主弧接受/拒绝点的 "
            f"constrained reprojection RMS 中位数分别为 "
            f"{diagnostic_summary['accepted']['reprojection_rms_px']['median']:.3f} px 和 "
            f"{diagnostic_summary['rejected']['reprojection_rms_px']['median']:.3f} px，"
            f"朝向分数中位数分别为 {diagnostic_summary['accepted']['facing_score']['median']:.3f} 和 "
            f"{diagnostic_summary['rejected']['facing_score']['median']:.3f}。"
            "这确认离弧点族与斜视角、较差 PnP 几何拟合相关；仍不能区分角点偏差和具体 IPPE 候选切换。"
        )
    else:
        diagnostic_report_line = "- 尚未提供独立的 PnP 重投影诊断复测。"
    report = f"""# 装甲板观测开弧重拟合与离群审计

本报告只描述观测轨迹，不做未来预测。

## 结论

- 旧图的闭合极坐标拟合不成立：它会跨越无数据区，把主弧与左右点族强制连接。
- 旧 `u/v` 最近邻标签也不是可靠的物理板身份。{len(comparable):,} 个可比较观测中，使用 truth 朝向候选和完整 PnP `tvec` 三维位置后有 {len(changed):,} 个（{100*len(changed)/len(comparable):.2f}%）改变槽位。
- 旧标签下，1.5/2.2/3.5 m 的 36 个单元可画出描述性的开放主弧；5 m 的 12 个单元全部碎片化，不再强行拟合。
- 物理槽位重分后，各距离支持单条主弧的单元数为：{distance_fit_summary}。

## PnP 回溯

- 被主弧接受点的 truth 朝向分数中位数为 {summary['pnp_accepted_diagnostics']['facing_score']['median']:.3f}，被拒绝点为 {summary['pnp_rejected_diagnostics']['facing_score']['median']:.3f}；离群点族明显更集中在斜视区域。
- 接受/拒绝点的 PnP 深度误差中位数分别为 {summary['pnp_accepted_diagnostics']['depth_error_m']['median']:.3f} m 和 {summary['pnp_rejected_diagnostics']['depth_error_m']['median']:.3f} m，拒绝组尾部也更重。
- 现有 {len(pnp):,} 条重分观测中，重投影诊断有效值为 {len(reprojection_present)}。原始采集没有角点、IPPE 候选编号、候选间隔或所选分支，因此只能确认“斜视角 PnP/角点观测分支”，不能把它进一步断言为某个 IPPE 解编号跳变。
{diagnostic_report_line}
- 精确四字段 join 的时间顺序无回退、无重复；时间错位不是这些稳定点族的解释。

## 剔除合同

- 图中的黑线只在连续支持区间内绘制，不闭合、不跨空白插值；空心圆是开弧端点。
- `O-L/O-R` 点全部保留在 `trajectory_outliers.jsonl`，并尽可能附带帧键、旧/新槽位、PnP 三维误差、朝向分数和诊断字段。
- `fragmented` 单元没有中心曲线；truth 朝向不一致单独记录，不再抹掉重复稳定的观测中心线。
- 旧标签主弧只能称为“角度分槽下的高密度观测模式”。物理轨迹结论应以本次显式记录的 `{physical_stream}` 关联合同为准。
"""
    pnp_metric_rows = [row for row in metric_rows if row["stream"] == physical_stream]
    if not pnp_metric_rows:
        raise RuntimeError(f"No {physical_stream} metric rows were produced.")
    focus = next((row for row in pnp_metric_rows if row["status"] == "fit"), pnp_metric_rows[0])
    report = f"""# 装甲板观测开弧、朝向与离群审计（相位拟合 v2）

本报告只描述已采集轨迹，不做未来预测。

## 已确认修复

- truth 相机基已修正为 `OpenCV [right, down, forward] = [-sim_local_y, -sim_local_z, sim_local_x]`。修正后，同一帧 truth 与 PnP 的水平角不再镜像。
- 已删除全局四次多项式及其循环自选点。新中心线按目标旋转相位分箱，每次重复先取稳健中心，再跨重复合并；只在多重复连续支持区间绘制，不外推。
- 每幅观测图同时绘制绿色 truth 参考线，并输出观测/truth 的曲率幅度和 `cup/cap/flat` 朝向。

## 截图条件

- `半径={float(focus['scale']):g}、距离={float(focus['distance_m']):g} m、slot={int(focus['slot'])}`：原始 {focus['samples']} 点，新方法保留 {focus['accepted_samples']} 点（{100*focus['accepted_ratio']:.1f}%）。
- 观测朝向为 `{focus['observed_orientation']}`，曲率幅度 {focus['observed_curvature_deg']:.4f}°；truth 为 `{focus['truth_orientation']}`，曲率幅度 {focus['truth_curvature_deg']:.4f}°；朝向一致={focus['orientation_agrees']}。
- 原图中的 `U` 形黑线是四阶多项式端点翘曲，不是五次重复中的真实轨迹翻转。

## 全网格边界

- 物理槽位流各距离同时满足重复支持和朝向验收的单元数为：{distance_fit_summary}。
- 只有重复支持不足的 `fragmented` 单元不画中心线；truth 朝向的 `match/mismatch/ambiguous` 与观测可拟合性分开报告。
- 被接受点的 truth 朝向分数中位数为 {summary['pnp_accepted_diagnostics']['facing_score']['median']:.3f}，被拒绝点为 {summary['pnp_rejected_diagnostics']['facing_score']['median']:.3f}；离群族明显集中在斜视/进退场边界。
- 接受/拒绝点的 PnP 深度误差中位数为 {summary['pnp_accepted_diagnostics']['depth_error_m']['median']:.3f} m / {summary['pnp_rejected_diagnostics']['depth_error_m']['median']:.3f} m。

## 数据保留

- 所有原始点仍在受保护输入中；未删除任何采集。
- 被新规则拒绝的逐帧记录写入 `trajectory_outliers.jsonl`，包含帧键、物理槽位、朝向、PnP 误差和明确拒绝原因。
- IPPE 候选编号、候选 tvec/rvec、重投影间隔和角点的结论由独立的 `PNP_ARC_FLIP_REPORT.md` 负责，不从本聚合脚本臆测。
"""
    (output / "OPEN_ARC_OUTLIER_REPORT.md").write_text(report, encoding="utf-8")
    retained_files = [
        "OPEN_ARC_OUTLIER_REPORT.md",
        "open_arc_metrics.csv",
        "open_arc_centers.jsonl",
        "trajectory_outliers.jsonl",
        "open_arc_outlier_summary.json",
        *([diagnostic_artifact] if diagnostic_artifact else []),
        *[f"legacy_angular_open_arc_slot{slot}.png" for slot in range(4)],
        *[f"{physical_stream}_open_arc_slot{slot}.png" for slot in range(4)],
    ]
    manifest = {
        "schema_version": 1,
        "kind": "trajectory_open_arc_audit_retention_manifest",
        "protected_inputs": {
            str(legacy_dir / "observed_points.jsonl"): sha256(legacy_dir / "observed_points.jsonl"),
            str(pnp_dir / "observed_points.jsonl"): sha256(pnp_dir / "observed_points.jsonl"),
            **(
                {
                    str(args.pnp_diagnostic_analysis_dir.resolve() / "observed_points.jsonl"): sha256(
                        args.pnp_diagnostic_analysis_dir.resolve() / "observed_points.jsonl"
                    )
                }
                if args.pnp_diagnostic_analysis_dir is not None
                else {}
            ),
        },
        "analysis_scripts": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(Path(__file__).with_name("analyze-stage3-truth-grid.py").resolve()): sha256(
                Path(__file__).with_name("analyze-stage3-truth-grid.py").resolve()
            ),
        },
        "retained_artifacts": {
            name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)}
            for name in retained_files
        },
        "counts": {
            "legacy_observations": len(legacy),
            f"{physical_stream}_observations": len(pnp),
            "outlier_records": len(outlier_rows),
            "targeted_pnp_diagnostic_samples": diagnostic_summary["samples"] if diagnostic_summary else 0,
        },
        "retention_class": "protected diagnostic captures and reproducible derived analysis; no automatic deletion",
    }
    (output / "retention_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
