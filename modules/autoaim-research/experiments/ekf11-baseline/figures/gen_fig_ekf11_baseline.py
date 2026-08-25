#!/usr/bin/env python3
"""Plot the locked EKF11 baseline collection and calculate frame rates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "estimate": "#D55E00",
    "truth": "#0072B2",
    "observation": "#009E73",
    "secondary": "#CC79A7",
    "neutral": "#5B6573",
}

SCENARIO_TITLES = {
    "spin_8": r"Stationary spin: $\omega=8\,\mathrm{rad/s}$",
    "translate_1p5": r"Translation: $v=1.5\,\mathrm{m/s}$",
    "translate_1_spin_6":
        r"Translation + spin: $v=1\,\mathrm{m/s},\ \omega=6\,\mathrm{rad/s}$",
}


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "lines.linewidth": 1.6,
    }
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"{path}:{line_number}: {error}") from error
    if len(rows) < 2:
        raise RuntimeError(f"not enough records: {path}")
    return rows


def vec(rows: list[dict], branch: str, key: str, width: int) -> np.ndarray:
    result = np.full((len(rows), width), np.nan)
    for index, row in enumerate(rows):
        value = row.get(branch)
        if isinstance(value, dict):
            item = value.get(key)
            if isinstance(item, list) and len(item) == width:
                result[index] = np.asarray(item, dtype=float)
    return result


def scalar(rows: list[dict], branch: str, key: str) -> np.ndarray:
    result = np.full(len(rows), np.nan)
    for index, row in enumerate(rows):
        value = row.get(branch)
        if isinstance(value, dict) and value.get(key) is not None:
            result[index] = float(value[key])
    return result


def rmse(estimate: np.ndarray, truth: np.ndarray) -> float | None:
    mask = np.isfinite(estimate) & np.isfinite(truth)
    if not np.any(mask):
        return None
    return float(np.sqrt(np.mean(np.square(estimate[mask] - truth[mask]))))


def metrics(rows: list[dict]) -> dict:
    timestamps = np.asarray([row["timestamp_ns"] for row in rows], dtype=np.int64)
    sequences = np.asarray([row["frame_seq"] for row in rows], dtype=np.int64)
    duration = float((timestamps[-1] - timestamps[0]) * 1e-9)
    dt_s = np.diff(timestamps).astype(float) * 1e-9
    estimates = np.asarray([row.get("ekf_estimate") is not None for row in rows])
    detected = np.asarray([row.get("detection_count", 0) > 0 for row in rows])
    omega_est = scalar(rows, "ekf_estimate", "omega_rad_s")
    omega_truth = scalar(rows, "truth", "omega_rad_s")
    velocity_est = vec(rows, "ekf_estimate", "velocity_mps", 3)
    velocity_truth = vec(rows, "truth", "velocity_mps", 3)
    speed_est = np.linalg.norm(velocity_est, axis=1)
    speed_truth = np.linalg.norm(velocity_truth, axis=1)
    return {
        "records": len(rows),
        "duration_s": duration,
        "processed_event_fps": (len(rows) - 1) / duration,
        "source_sequence_rate_hz": float(sequences[-1] - sequences[0]) / duration,
        "detected_event_fps": int(detected.sum()) / duration,
        "ekf_estimate_fps": int(estimates.sum()) / duration,
        "source_frames_skipped": int((sequences[-1] - sequences[0]) - (len(rows) - 1)),
        "inter_event_ms_median": float(np.median(dt_s) * 1e3),
        "inter_event_ms_p95": float(np.percentile(dt_s, 95) * 1e3),
        "truth_speed_mps_median": float(np.nanmedian(speed_truth)),
        "truth_abs_omega_rad_s_median": float(np.nanmedian(np.abs(omega_truth))),
        "omega_rmse_rad_s": rmse(omega_est, omega_truth),
        "center_speed_rmse_mps": rmse(speed_est, speed_truth),
        "radius_even_rmse_m": rmse(
            scalar(rows, "ekf_estimate", "radius_even_m"),
            scalar(rows, "truth", "radius_even_m"),
        ),
        "radius_odd_rmse_m": rmse(
            scalar(rows, "ekf_estimate", "radius_odd_m"),
            scalar(rows, "truth", "radius_odd_m"),
        ),
    }


def plot_scenario(scenario_id: str, rows: list[dict], output_dir: Path) -> dict:
    timestamps = np.asarray([row["timestamp_ns"] for row in rows], dtype=np.int64)
    t_s = (timestamps - timestamps[0]).astype(float) * 1e-9
    omega_est = scalar(rows, "ekf_estimate", "omega_rad_s")
    omega_truth = scalar(rows, "truth", "omega_rad_s")
    velocity_est = vec(rows, "ekf_estimate", "velocity_mps", 3)
    velocity_truth = vec(rows, "truth", "velocity_mps", 3)
    horizontal_truth_scale = np.nanmax(np.abs(velocity_truth[:, :2]), axis=0)
    dominant_axis = int(np.argmax(horizontal_truth_scale))
    if not np.isfinite(horizontal_truth_scale[dominant_axis]) or \
            horizontal_truth_scale[dominant_axis] < 1e-6:
        dominant_axis = 0
    transverse_axis = 1 - dominant_axis
    axis_name = ("x", "y")[dominant_axis]
    transverse_name = ("x", "y")[transverse_axis]
    radius_even_est = scalar(rows, "ekf_estimate", "radius_even_m")
    radius_odd_est = scalar(rows, "ekf_estimate", "radius_odd_m")
    radius_even_truth = scalar(rows, "truth", "radius_even_m")
    radius_odd_truth = scalar(rows, "truth", "radius_odd_m")
    pnp = vec(rows, "primary_pnp", "xyz_m", 3)
    center_est = vec(rows, "ekf_estimate", "center_m", 3)
    center_truth = vec(rows, "truth", "center_m", 3)
    armor_truth = vec(rows, "truth", "armor_m", 3)
    summary = metrics(rows)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(t_s, omega_truth, color=COLORS["truth"], linestyle="--", label="Truth")
    ax.plot(t_s, omega_est, color=COLORS["estimate"], label="11D EKF")
    ax.set(xlabel="Exposure time (s)", ylabel="Angular velocity (rad/s)",
           title="A  Angular velocity")
    ax.legend(ncol=2, loc="best")

    ax = axes[0, 1]
    ax.plot(t_s, velocity_truth[:, dominant_axis], color=COLORS["truth"],
            linestyle="--", label=rf"Truth $v_{axis_name}$")
    ax.plot(t_s, velocity_est[:, dominant_axis], color=COLORS["estimate"],
            label=rf"EKF $v_{axis_name}$")
    ax.plot(t_s, velocity_est[:, transverse_axis], color=COLORS["secondary"],
            alpha=0.8, label=rf"EKF transverse $v_{transverse_name}$")
    ax.set(xlabel="Exposure time (s)", ylabel="Center velocity (m/s)",
           title="B  Center linear velocity")
    ax.legend(ncol=3, loc="best")

    ax = axes[1, 0]
    ax.plot(t_s, radius_even_truth, color=COLORS["truth"], linestyle="--",
            label="Truth even")
    ax.plot(t_s, radius_odd_truth, color=COLORS["secondary"], linestyle="--",
            label="Truth odd")
    ax.plot(t_s, radius_even_est, color=COLORS["estimate"], label="EKF even")
    ax.plot(t_s, radius_odd_est, color=COLORS["neutral"], label="EKF odd")
    ax.set(xlabel="Exposure time (s)", ylabel="Radius (m)",
           title="C  Reconstructed radii")
    ax.legend(ncol=2, loc="best")

    ax = axes[1, 1]
    pnp_mask = np.isfinite(pnp[:, 0]) & np.isfinite(pnp[:, 1])
    armor_mask = np.isfinite(armor_truth[:, 0]) & np.isfinite(armor_truth[:, 1])
    estimate_mask = np.isfinite(center_est[:, 0]) & np.isfinite(center_est[:, 1])
    truth_mask = np.isfinite(center_truth[:, 0]) & np.isfinite(center_truth[:, 1])
    ax.scatter(pnp[pnp_mask, 0], pnp[pnp_mask, 1], s=8, alpha=0.28,
               color=COLORS["observation"], label="PnP observation", rasterized=True)
    ax.scatter(armor_truth[armor_mask, 0], armor_truth[armor_mask, 1],
               s=7, color=COLORS["truth"], alpha=0.65,
               label="True observed armor", rasterized=True)
    ax.plot(center_est[estimate_mask, 0], center_est[estimate_mask, 1],
            color=COLORS["estimate"], linewidth=1.5, label="EKF center")
    ax.plot(center_truth[truth_mask, 0], center_truth[truth_mask, 1],
            color="#111111", linestyle="--", linewidth=1.3, label="True center")
    if np.any(truth_mask):
        ax.scatter(center_truth[truth_mask][0, 0], center_truth[truth_mask][0, 1],
                   marker="x", s=36, color="#111111", linewidths=1.3,
                   label="True center start")
    ax.set(xlabel="x forward (m)", ylabel="y left (m)",
           title="D  Top view: observation and reconstructed motion")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(ncol=2, loc="best")

    fig.suptitle(
        SCENARIO_TITLES[scenario_id]
        + "\n"
        + f"20 s timestamp window | processed {summary['processed_event_fps']:.1f} FPS | "
        + f"source sequence {summary['source_sequence_rate_hz']:.1f} Hz",
        fontsize=13,
        fontweight="bold",
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"fig_{scenario_id}.{suffix}")
    plt.close(fig)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "aim-stack.ekf11-baseline-summary/v1",
        "collection_manifest": str(args.manifest),
        "scenarios": {},
    }
    for scenario in manifest["scenarios"]:
        scenario_id = scenario["id"]
        rows = load_jsonl(Path(scenario["jsonl"]))
        summary["scenarios"][scenario_id] = plot_scenario(
            scenario_id, rows, args.output_dir
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
