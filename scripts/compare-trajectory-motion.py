#!/usr/bin/env python3
"""Compare descriptive center trajectories for spin-only and combined motion.

This script consumes the persisted center trajectories produced by
``analyze-trajectory-outliers.py``.  It does not predict future motion.  The two
conditions are aligned by phase bin, then compared per physical armor slot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


STREAM = "angular_facing"
PHASE_KEY_DIGITS = 6
COLORS = {"spin": "#0072B2", "combined": "#D55E00"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare phase-aligned observed trajectories for spin and combined motion."
    )
    parser.add_argument("--spin-analysis-dir", type=Path, required=True)
    parser.add_argument("--combined-analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream", default=STREAM)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_centers(path: Path, stream: str) -> dict[int, list[dict[str, float]]]:
    slots: dict[int, list[dict[str, float]]] = {slot: [] for slot in range(4)}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("stream") != stream:
                continue
            slot = int(row["slot"])
            if slot not in slots:
                continue
            slots[slot].append(
                {
                    "phase_rad": float(row["phase_rad"]),
                    "u_deg": float(row["center_u_deg"]),
                    "v_deg": float(row["center_v_deg"]),
                    "p90_deg": (
                        float(row["p90_residual_deg"])
                        if row.get("p90_residual_deg") is not None
                        else float("nan")
                    ),
                    "truth_u_deg": (
                        float(row["truth_u_deg"])
                        if row.get("truth_u_deg") is not None
                        else float("nan")
                    ),
                    "truth_v_deg": (
                        float(row["truth_v_deg"])
                        if row.get("truth_v_deg") is not None
                        else float("nan")
                    ),
                }
            )
    for rows in slots.values():
        rows.sort(key=lambda item: item["phase_rad"])
    return slots


def load_metrics(path: Path, stream: str) -> dict[int, dict[str, Any]]:
    metrics: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("stream") != stream or row.get("status") != "fit":
                continue
            slot = int(row["slot"])
            metrics[slot] = {
                "accepted_ratio": float(row["accepted_ratio"]),
                "eligible_acceptance_ratio": float(row["eligible_acceptance_ratio"]),
                "accepted_residual_p90_deg": float(row["accepted_residual_p90_deg"]),
                "observed_curvature_deg": float(row["observed_curvature_deg"]),
                "observed_orientation": row["observed_orientation"],
                "samples": int(row["samples"]),
                "accepted_samples": int(row["accepted_samples"]),
                "repeat_coverage": int(row["repeat_coverage"]),
            }
    return metrics


def phase_map(rows: list[dict[str, float]]) -> dict[float, dict[str, float]]:
    return {round(row["phase_rad"], PHASE_KEY_DIGITS): row for row in rows}


def aligned_rows(
    spin_rows: list[dict[str, float]], combined_rows: list[dict[str, float]]
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    spin_by_phase = phase_map(spin_rows)
    combined_by_phase = phase_map(combined_rows)
    common = sorted(set(spin_by_phase) & set(combined_by_phase))
    return [spin_by_phase[key] for key in common], [combined_by_phase[key] for key in common]


def band_polygon(rows: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray([[row["u_deg"], row["v_deg"]] for row in rows], dtype=float)
    widths = np.asarray([row["p90_deg"] for row in rows], dtype=float)
    if len(points) < 2:
        return points, points
    finite = np.isfinite(widths)
    if not finite.any():
        return points, points
    if not finite.all():
        indices = np.arange(len(widths))
        widths = np.interp(indices, indices[finite], widths[finite])
    tangent = np.gradient(points, axis=0)
    norm = np.linalg.norm(tangent, axis=1)
    norm[norm < 1e-12] = 1.0
    normal = np.column_stack((-tangent[:, 1] / norm, tangent[:, 0] / norm))
    return points + normal * widths[:, None], points - normal * widths[:, None]


def add_probability_band(ax: plt.Axes, rows: list[dict[str, float]], color: str) -> None:
    if len(rows) < 2:
        return
    upper, lower = band_polygon(rows)
    polygon = np.vstack((upper, lower[::-1]))
    ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.13, linewidth=0)


def summarize_slot(
    slot: int,
    spin_rows: list[dict[str, float]],
    combined_rows: list[dict[str, float]],
    spin_metric: dict[str, Any],
    combined_metric: dict[str, Any],
) -> dict[str, Any]:
    spin_aligned, combined_aligned = aligned_rows(spin_rows, combined_rows)
    if not spin_aligned:
        raise RuntimeError(f"slot {slot}: no common phase bins")
    spin_xy = np.asarray([[row["u_deg"], row["v_deg"]] for row in spin_aligned])
    combined_xy = np.asarray([[row["u_deg"], row["v_deg"]] for row in combined_aligned])
    displacement = np.linalg.norm(combined_xy - spin_xy, axis=1)
    spin_truth = np.asarray(
        [[row["truth_u_deg"], row["truth_v_deg"]] for row in spin_aligned]
    )
    combined_truth = np.asarray(
        [[row["truth_u_deg"], row["truth_v_deg"]] for row in combined_aligned]
    )
    truth_finite = np.isfinite(spin_truth).all(axis=1) & np.isfinite(combined_truth).all(axis=1)
    truth_displacement = np.linalg.norm(
        combined_truth[truth_finite] - spin_truth[truth_finite], axis=1
    )
    spin_measurement_bias = np.linalg.norm(spin_xy[truth_finite] - spin_truth[truth_finite], axis=1)
    combined_measurement_bias = np.linalg.norm(
        combined_xy[truth_finite] - combined_truth[truth_finite], axis=1
    )
    if not truth_displacement.size:
        raise RuntimeError(f"slot {slot}: no common finite truth bins")
    spin_width = np.asarray([row["p90_deg"] for row in spin_aligned])
    combined_width = np.asarray([row["p90_deg"] for row in combined_aligned])
    spin_width = spin_width[np.isfinite(spin_width)]
    combined_width = combined_width[np.isfinite(combined_width)]
    if not spin_width.size or not combined_width.size:
        raise RuntimeError(f"slot {slot}: no finite phase-band widths")
    return {
        "slot": slot,
        "common_phase_bins": int(displacement.size),
        "center_displacement_p50_deg": float(np.median(displacement)),
        "center_displacement_p90_deg": float(np.percentile(displacement, 90)),
        "center_displacement_max_deg": float(np.max(displacement)),
        "truth_center_displacement_p50_deg": float(np.median(truth_displacement)),
        "truth_center_displacement_p90_deg": float(np.percentile(truth_displacement, 90)),
        "spin_observation_truth_bias_p50_deg": float(np.median(spin_measurement_bias)),
        "combined_observation_truth_bias_p50_deg": float(np.median(combined_measurement_bias)),
        "spin_phase_band_p50_deg": float(np.median(spin_width)),
        "combined_phase_band_p50_deg": float(np.median(combined_width)),
        "phase_band_p50_ratio": float(np.median(combined_width) / np.median(spin_width)),
        "spin_accepted_ratio": spin_metric["accepted_ratio"],
        "combined_accepted_ratio": combined_metric["accepted_ratio"],
        "accepted_ratio_delta": combined_metric["accepted_ratio"] - spin_metric["accepted_ratio"],
        "spin_fit_residual_p90_deg": spin_metric["accepted_residual_p90_deg"],
        "combined_fit_residual_p90_deg": combined_metric["accepted_residual_p90_deg"],
        "fit_residual_p90_ratio": combined_metric["accepted_residual_p90_deg"]
        / spin_metric["accepted_residual_p90_deg"],
        "spin_curvature_deg": spin_metric["observed_curvature_deg"],
        "combined_curvature_deg": combined_metric["observed_curvature_deg"],
        "spin_orientation": spin_metric["observed_orientation"],
        "combined_orientation": combined_metric["observed_orientation"],
        "spin_samples": spin_metric["samples"],
        "combined_samples": combined_metric["samples"],
        "spin_repeat_coverage": spin_metric["repeat_coverage"],
        "combined_repeat_coverage": combined_metric["repeat_coverage"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(
    output: Path,
    spin_centers: dict[int, list[dict[str, float]]],
    combined_centers: dict[int, list[dict[str, float]]],
    summaries: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.3), constrained_layout=True)
    for slot, ax in enumerate(axes.flat):
        spin_rows = spin_centers[slot]
        combined_rows = combined_centers[slot]
        add_probability_band(ax, spin_rows, COLORS["spin"])
        add_probability_band(ax, combined_rows, COLORS["combined"])
        for name, rows in (("Spin observed", spin_rows), ("Combined observed", combined_rows)):
            xy = np.asarray([[row["u_deg"], row["v_deg"]] for row in rows])
            color = COLORS["spin" if name == "Spin observed" else "combined"]
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                marker="o",
                markersize=3.2,
                linewidth=1.7,
                color=color,
                label=name,
            )
            truth = np.asarray([[row["truth_u_deg"], row["truth_v_deg"]] for row in rows])
            finite = np.isfinite(truth).all(axis=1)
            ax.plot(
                truth[finite, 0],
                truth[finite, 1],
                linestyle="--",
                linewidth=1.25,
                color=color,
                alpha=0.78,
                label=name.replace("observed", "truth"),
            )
        summary = summaries[slot]
        ax.set_title(
            f"Armor slot {slot}  |  observed shift P50 {summary['center_displacement_p50_deg']:.3f}°"
        )
        ax.set_xlabel("Horizontal observation angle (deg)")
        ax.set_ylabel("Vertical observation angle (deg)")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")
        if slot == 0:
            ax.legend(frameon=False, loc="best")
    fig.suptitle(
        "Observed armor trajectories at 5 m, radius 1.0, 30°/s\n"
        "Solid: observed; dashed: truth; shaded: observed phase-bin P90 band",
        fontsize=13,
    )
    fig.savefig(output / "spin_vs_combined_centers.png", dpi=260, bbox_inches="tight")
    fig.savefig(output / "spin_vs_combined_centers.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    spin_dir = args.spin_analysis_dir.resolve()
    combined_dir = args.combined_analysis_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    spin_centers_path = spin_dir / "open_arc_centers.jsonl"
    combined_centers_path = combined_dir / "open_arc_centers.jsonl"
    spin_metrics_path = spin_dir / "open_arc_metrics.csv"
    combined_metrics_path = combined_dir / "open_arc_metrics.csv"
    for required in (
        spin_centers_path,
        combined_centers_path,
        spin_metrics_path,
        combined_metrics_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    spin_centers = load_centers(spin_centers_path, args.stream)
    combined_centers = load_centers(combined_centers_path, args.stream)
    spin_metrics = load_metrics(spin_metrics_path, args.stream)
    combined_metrics = load_metrics(combined_metrics_path, args.stream)
    missing = [slot for slot in range(4) if not spin_centers[slot] or not combined_centers[slot]]
    if missing:
        raise RuntimeError(f"missing center trajectories for slots: {missing}")
    missing_metrics = [slot for slot in range(4) if slot not in spin_metrics or slot not in combined_metrics]
    if missing_metrics:
        raise RuntimeError(f"missing fit metrics for slots: {missing_metrics}")

    summaries = [
        summarize_slot(
            slot,
            spin_centers[slot],
            combined_centers[slot],
            spin_metrics[slot],
            combined_metrics[slot],
        )
        for slot in range(4)
    ]
    write_csv(output / "motion_comparison_metrics.csv", summaries)
    plot_comparison(output, spin_centers, combined_centers, summaries)

    all_displacements = [row["center_displacement_p50_deg"] for row in summaries]
    summary = {
        "kind": "descriptive_trajectory_motion_comparison",
        "prediction": False,
        "stream": args.stream,
        "conditions": {
            "shared": {"distance_m": 5.0, "radius_scale": 1.0, "spin_deg_s": 30.0, "repeats": 5},
            "spin": {"linear_speed_mps": 0.0, "linear_span_m": 0.0},
            "combined": {"linear_speed_mps": 1.0, "linear_span_m": 2.0, "direction_deg": 90.0},
        },
        "slot_metrics": summaries,
        "aggregate": {
            "median_of_slot_center_displacement_p50_deg": float(np.median(all_displacements)),
            "median_of_slot_truth_displacement_p50_deg": float(
                np.median([row["truth_center_displacement_p50_deg"] for row in summaries])
            ),
            "median_spin_observation_truth_bias_p50_deg": float(
                np.median([row["spin_observation_truth_bias_p50_deg"] for row in summaries])
            ),
            "median_combined_observation_truth_bias_p50_deg": float(
                np.median([row["combined_observation_truth_bias_p50_deg"] for row in summaries])
            ),
            "median_phase_band_p50_ratio": float(
                np.median([row["phase_band_p50_ratio"] for row in summaries])
            ),
            "median_spin_accepted_ratio": float(
                np.median([row["spin_accepted_ratio"] for row in summaries])
            ),
            "median_combined_accepted_ratio": float(
                np.median([row["combined_accepted_ratio"] for row in summaries])
            ),
            "median_fit_residual_p90_ratio": float(
                np.median([row["fit_residual_p90_ratio"] for row in summaries])
            ),
        },
        "sources": {
            "spin_analysis_dir": str(spin_dir),
            "combined_analysis_dir": str(combined_dir),
            "files": {
                str(spin_centers_path): sha256_file(spin_centers_path),
                str(combined_centers_path): sha256_file(combined_centers_path),
                str(spin_metrics_path): sha256_file(spin_metrics_path),
                str(combined_metrics_path): sha256_file(combined_metrics_path),
            },
        },
    }
    with (output / "motion_comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    script_path = Path(__file__).resolve()
    retention = {
        "classification": "long_term_private_evidence",
        "deletion_allowed": False,
        "reason": "Matched truth-gated baseline for descriptive trajectory research.",
        "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
        "artifacts": [
            "motion_comparison_metrics.csv",
            "motion_comparison_summary.json",
            "spin_vs_combined_centers.png",
            "spin_vs_combined_centers.svg",
        ],
    }
    with (output / "retention_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(retention, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
