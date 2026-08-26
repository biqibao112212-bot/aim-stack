#!/usr/bin/env python3
"""Plot the locked EKF11 baseline collection and calculate frame rates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


COLORS = {
    "estimate": "#D55E00",
    "truth": "#0072B2",
    "observation": "#009E73",
    "secondary": "#CC79A7",
    "neutral": "#5B6573",
}

SLOT_COLORS = ("#E69F00", "#56B4E9", "#009E73", "#CC79A7")

SCENARIO_TITLES = {
    "spin_8": r"Stationary spin: $\omega=8\,\mathrm{rad/s}$",
    "translate_1p5": r"Translation: $v=1.5\,\mathrm{m/s}$",
    "translate_1_spin_6":
        r"Translation + spin: $v=1\,\mathrm{m/s},\ \omega=6\,\mathrm{rad/s}$",
}

SCENARIO_PREDICTION_TITLES = {
    "spin_8": "A  原地旋转 8 rad/s",
    "translate_1p5": "B  往复平移 1.5 m/s",
    "translate_1_spin_6": "C  平移 1 m/s + 旋转 6 rad/s",
}

PREDICTION_STANDARD_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "prediction_evaluation_standard_v1.json"
)
PREDICTION_STANDARD = json.loads(
    PREDICTION_STANDARD_PATH.read_text(encoding="utf-8")
)
REQUIRED_HORIZONS_MS = tuple(PREDICTION_STANDARD["required_horizons_ms"])
CURVE_STEP_MS = float(PREDICTION_STANDARD["curve_step_ms"])
MAX_HORIZON_MS = float(max(REQUIRED_HORIZONS_MS))
PREDICTION_HORIZONS_S = (
    np.arange(0.0, MAX_HORIZON_MS + CURVE_STEP_MS, CURVE_STEP_MS) * 1e-3
)
MAX_TRUTH_INTERPOLATION_GAP_S = (
    float(PREDICTION_STANDARD["max_truth_interpolation_gap_ms"]) * 1e-3
)
SMALL_ARMOR_WIDTH_M = float(
    PREDICTION_STANDARD["armor_geometry_m"]["small"]["width"]
)
BIG_ARMOR_WIDTH_M = float(
    PREDICTION_STANDARD["armor_geometry_m"]["big"]["width"]
)
ARMOR_HEIGHT_M = float(
    PREDICTION_STANDARD["armor_geometry_m"]["small"]["height"]
)


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


def rotate_z(points: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Rotate one or more xyz points around +z."""
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    result = np.array(points, dtype=float, copy=True)
    result[..., 0] = cosine * points[..., 0] - sine * points[..., 1]
    result[..., 1] = sine * points[..., 0] + cosine * points[..., 1]
    return result


def truth_local_armor_offsets(rows: list[dict]) -> np.ndarray:
    """Recover the four fixed target-frame armor offsets from saved truth."""
    samples: dict[int, list[np.ndarray]] = {slot: [] for slot in range(4)}
    for row in rows:
        truth = row.get("truth")
        if not isinstance(truth, dict) or truth.get("armor_m") is None:
            continue
        slot = truth.get("matched_armor_slot")
        if not isinstance(slot, int) or slot not in samples:
            continue
        center = np.asarray(truth["center_m"], dtype=float)
        armor = np.asarray(truth["armor_m"], dtype=float)
        samples[slot].append(rotate_z(armor - center, -float(truth["yaw_rad"])))

    offsets = []
    for slot in range(4):
        if not samples[slot]:
            raise RuntimeError(f"truth armor slot {slot} has no saved samples")
        values = np.asarray(samples[slot], dtype=float)
        median = np.median(values, axis=0)
        if float(np.max(np.linalg.norm(values - median, axis=1))) > 1e-4:
            raise RuntimeError(f"truth armor slot {slot} is not rigid")
        offsets.append(median)
    return np.asarray(offsets, dtype=float)


def decode_ekf_armors(estimate: dict, horizon_s: float) -> np.ndarray:
    """Apply the vendored Tongji 11D transition and four-armor decoder."""
    center = np.asarray(estimate["center_m"], dtype=float)
    velocity = np.asarray(estimate["velocity_mps"], dtype=float)
    center = center + velocity * horizon_s
    yaw = float(estimate["yaw_rad"]) + float(estimate["omega_rad_s"]) * horizon_s

    armors = np.empty((4, 3), dtype=float)
    for slot in range(4):
        odd = slot % 2 == 1
        radius = float(estimate["radius_odd_m" if odd else "radius_even_m"])
        height = float(estimate["height_odd_m" if odd else "height_even_m"])
        angle = yaw + slot * math.pi / 2.0
        armors[slot] = (
            center[0] - radius * math.cos(angle),
            center[1] - radius * math.sin(angle),
            height,
        )
    return armors


def interpolate_truth_pose(
    times_s: np.ndarray,
    centers_m: np.ndarray,
    unwrapped_yaw_rad: np.ndarray,
    query_s: float,
) -> tuple[np.ndarray, float] | None:
    """Interpolate truth only across a bounded pair of recorded exposures."""
    if query_s < times_s[0] or query_s > times_s[-1]:
        return None
    right = int(np.searchsorted(times_s, query_s, side="left"))
    if right < len(times_s) and abs(float(times_s[right] - query_s)) < 1e-9:
        return centers_m[right].copy(), float(unwrapped_yaw_rad[right])
    if right == 0 or right >= len(times_s):
        return None
    left = right - 1
    gap_s = float(times_s[right] - times_s[left])
    if gap_s <= 0.0 or gap_s > MAX_TRUTH_INTERPOLATION_GAP_S:
        return None
    alpha = float((query_s - times_s[left]) / gap_s)
    center = centers_m[left] + alpha * (centers_m[right] - centers_m[left])
    yaw = unwrapped_yaw_rad[left] + alpha * (
        unwrapped_yaw_rad[right] - unwrapped_yaw_rad[left]
    )
    return center, float(yaw)


def prediction_horizon_metrics(rows: list[dict]) -> dict:
    """Roll each tracking posterior forward and compare one physical armor."""
    timestamp_ns = np.asarray([row["timestamp_ns"] for row in rows], dtype=np.int64)
    times_s = (timestamp_ns - timestamp_ns[0]).astype(float) * 1e-9
    if np.any(np.diff(times_s) <= 0.0):
        raise RuntimeError("prediction evaluation requires increasing timestamps")
    centers_m = vec(rows, "truth", "center_m", 3)
    yaw_rad = np.unwrap(scalar(rows, "truth", "yaw_rad"))
    if not np.all(np.isfinite(centers_m)) or not np.all(np.isfinite(yaw_rad)):
        raise RuntimeError("prediction evaluation requires complete target truth")
    local_offsets_m = truth_local_armor_offsets(rows)

    horizons_s = [float(round(value, 6)) for value in PREDICTION_HORIZONS_S]
    samples = {
        horizon_s: {
            "normal_radial_abs_m": [],
            "tangential_abs_m": [],
            "vertical_abs_m": [],
            "error_3d_m": [],
            "small_armor_window": [],
            "big_armor_window": [],
        }
        for horizon_s in horizons_s
    }
    eligible_time_anchors = {horizon_s: 0 for horizon_s in horizons_s}
    evaluated_posteriors = 0

    for index, row in enumerate(rows):
        estimate = row.get("ekf_estimate")
        truth = row.get("truth")
        if row.get("tracker_state") != "tracking" or not isinstance(estimate, dict):
            continue
        if not isinstance(truth, dict) or truth.get("armor_m") is None:
            continue
        physical_slot = truth.get("matched_armor_slot")
        if not isinstance(physical_slot, int) or not 0 <= physical_slot < 4:
            continue

        current_truth_armor = np.asarray(truth["armor_m"], dtype=float)
        current_candidates = decode_ekf_armors(estimate, 0.0)
        estimator_slot = int(
            np.argmin(np.linalg.norm(current_candidates - current_truth_armor, axis=1))
        )
        evaluated_posteriors += 1

        for horizon_s in horizons_s:
            query_s = float(times_s[index] + horizon_s)
            if query_s < times_s[0] or query_s > times_s[-1]:
                continue
            eligible_time_anchors[horizon_s] += 1
            future_pose = interpolate_truth_pose(
                times_s,
                centers_m,
                yaw_rad,
                query_s,
            )
            if future_pose is None:
                continue
            future_center, future_yaw = future_pose
            future_truth_armor = future_center + rotate_z(
                local_offsets_m[physical_slot], future_yaw
            )
            predicted_armor = decode_ekf_armors(estimate, horizon_s)[estimator_slot]
            error = predicted_armor - future_truth_armor

            normal_radial = future_truth_armor - future_center
            normal_radial[2] = 0.0
            normal_radial_norm = float(np.linalg.norm(normal_radial))
            if normal_radial_norm <= 1e-9:
                continue
            normal_radial /= normal_radial_norm
            tangential = np.asarray(
                [-normal_radial[1], normal_radial[0], 0.0], dtype=float
            )
            tangential_abs_m = abs(float(error @ tangential))
            vertical_abs_m = abs(float(error[2]))
            bucket = samples[horizon_s]
            bucket["normal_radial_abs_m"].append(abs(float(error @ normal_radial)))
            bucket["tangential_abs_m"].append(tangential_abs_m)
            bucket["vertical_abs_m"].append(vertical_abs_m)
            bucket["error_3d_m"].append(float(np.linalg.norm(error)))
            bucket["small_armor_window"].append(
                tangential_abs_m <= SMALL_ARMOR_WIDTH_M / 2.0
                and vertical_abs_m <= ARMOR_HEIGHT_M / 2.0
            )
            bucket["big_armor_window"].append(
                tangential_abs_m <= BIG_ARMOR_WIDTH_M / 2.0
                and vertical_abs_m <= ARMOR_HEIGHT_M / 2.0
            )

    result = {
        "evaluated_tracking_posteriors": evaluated_posteriors,
        "horizons_ms": [round(value * 1000.0, 6) for value in horizons_s],
        "eligible_time_anchor_count": [],
        "sample_count": [],
        "availability": [],
        "normal_radial_abs_m": {"p50": [], "p95": []},
        "tangential_abs_m": {"p50": [], "p95": []},
        "vertical_abs_m": {"p50": [], "p95": []},
        "error_3d_m": {"p50": [], "p95": []},
        "small_armor_window_coverage": [],
        "big_armor_window_coverage": [],
        "truth_local_armor_offsets_m": local_offsets_m.tolist(),
    }
    for horizon_s in horizons_s:
        bucket = samples[horizon_s]
        eligible_count = eligible_time_anchors[horizon_s]
        sample_count = len(bucket["error_3d_m"])
        result["eligible_time_anchor_count"].append(eligible_count)
        result["sample_count"].append(sample_count)
        result["availability"].append(
            float(sample_count / eligible_count) if eligible_count else 0.0
        )
        for metric in (
            "normal_radial_abs_m",
            "tangential_abs_m",
            "vertical_abs_m",
            "error_3d_m",
        ):
            values = np.asarray(bucket[metric], dtype=float)
            if not values.size:
                raise RuntimeError(f"no prediction samples at horizon {horizon_s}")
            result[metric]["p50"].append(float(np.percentile(values, 50)))
            result[metric]["p95"].append(float(np.percentile(values, 95)))
        result["small_armor_window_coverage"].append(
            float(np.mean(np.asarray(bucket["small_armor_window"], dtype=float)))
        )
        result["big_armor_window_coverage"].append(
            float(np.mean(np.asarray(bucket["big_armor_window"], dtype=float)))
        )
    return result


def save_deterministic_figure(figure: plt.Figure, output_base: Path) -> None:
    figure.savefig(
        output_base.with_suffix(".png"),
        metadata={"Software": "aim-stack ekf11 baseline"},
    )
    figure.savefig(
        output_base.with_suffix(".pdf"),
        metadata={
            "Creator": "aim-stack ekf11 baseline",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    svg_path = output_base.with_suffix(".svg")
    figure.savefig(
        svg_path,
        metadata={"Creator": "aim-stack ekf11 baseline", "Date": None},
    )
    # Matplotlib writes path-data lines with trailing spaces. Normalize them so
    # the tracked vector artifact passes Git whitespace checks.
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_lines) + "\n",
        encoding="utf-8",
    )


def plot_prediction_horizons(predictions: dict[str, dict], output_dir: Path) -> None:
    scenario_order = ("spin_8", "translate_1p5", "translate_1_spin_6")
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            # Matplotlib exposes the system Noto CJK collection under its JP
            # family name; the collection still contains Simplified Chinese.
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "svg.hashsalt": "aim-stack-ekf11-prediction-v1",
        }
    ):
        figure, axes = plt.subplots(
            1, 3, figsize=(12.6, 4.25), sharex=True, sharey=True,
            constrained_layout=True,
        )
        plot_specs = (
            ("normal_radial_abs_m", "#D55E00", "o", "法向/径向"),
            ("tangential_abs_m", "#0072B2", "s", "切向"),
        )
        for axis, scenario_id in zip(axes, scenario_order):
            result = predictions[scenario_id]
            horizons_ms = np.asarray(result["horizons_ms"], dtype=float)
            for metric, color, marker, label in plot_specs:
                p50_cm = np.asarray(result[metric]["p50"], dtype=float) * 100.0
                p95_cm = np.asarray(result[metric]["p95"], dtype=float) * 100.0
                axis.plot(
                    horizons_ms, p50_cm, color=color, marker=marker,
                    markevery=10, markersize=4.0, linewidth=2.1,
                    label=f"{label} p50",
                )
                axis.plot(
                    horizons_ms, p95_cm, color=color, linestyle=":",
                    linewidth=1.9, label=f"{label} p95",
                )
            axis.set_title(SCENARIO_PREDICTION_TITLES[scenario_id], loc="left")
            axis.set_xlabel("预测时域 τ（ms）")
            axis.set_xlim(0.0, MAX_HORIZON_MS)
            axis.set_xticks((0, 100, 200, 300, 400, 500))
            axis.grid(True, alpha=0.22)
        axes[0].set_ylabel("绝对位置误差（cm）")
        axes[0].set_ylim(bottom=0.0)
        axes[0].legend(loc="upper left", ncol=2, fontsize=8)
        figure.suptitle(
            "11 维 EKF 的未来装甲板位置误差",
            fontsize=14, fontweight="bold",
        )
        figure.text(
            0.5, -0.015,
            "从每个 tracking 后验状态外推同一块物理装甲板；实线为 p50，点线为 p95。",
            ha="center", va="top", fontsize=9,
        )
        save_deterministic_figure(figure, output_dir / "fig_prediction_horizon")
        plt.close(figure)


def plot_small_armor_window_coverage(
    predictions: dict[str, dict], output_dir: Path
) -> None:
    scenario_order = ("spin_8", "translate_1p5", "translate_1_spin_6")
    scenario_styles = {
        "spin_8": ("#D55E00", "o"),
        "translate_1p5": ("#0072B2", "s"),
        "translate_1_spin_6": ("#009E73", "^"),
    }
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "svg.hashsalt": "aim-stack-ekf11-window-v1",
        }
    ):
        figure, axis = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
        for scenario_id in scenario_order:
            result = predictions[scenario_id]
            color, marker = scenario_styles[scenario_id]
            axis.plot(
                result["horizons_ms"],
                np.asarray(result["small_armor_window_coverage"], dtype=float) * 100.0,
                color=color,
                marker=marker,
                markevery=10,
                markersize=5.0,
                linewidth=2.2,
                label=SCENARIO_PREDICTION_TITLES[scenario_id].split("  ", 1)[1],
            )
        axis.set(
            xlim=(0.0, MAX_HORIZON_MS),
            ylim=(0.0, 100.0),
            xlabel="预测时域 τ（ms）",
            ylabel="落入板面窗口的样本比例（%）",
            title="未来预测点的小装甲板窗口覆盖率",
        )
        axis.set_xticks((0, 100, 200, 300, 400, 500))
        axis.set_yticks((0, 20, 40, 60, 80, 100))
        axis.legend(loc="upper right")
        axis.grid(True, alpha=0.22)
        figure.text(
            0.5,
            -0.01,
            "小装甲板 135 × 55 mm：|切向误差| ≤ 67.5 mm 且 |竖直误差| ≤ 27.5 mm。",
            ha="center",
            va="top",
            fontsize=9,
        )
        save_deterministic_figure(
            figure, output_dir / "fig_small_armor_window_coverage"
        )
        plt.close(figure)


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
    pnp = vec(rows, "primary_pnp", "xyz_m", 3)
    armor_truth = vec(rows, "truth", "armor_m", 3)
    slots = scalar(rows, "truth", "matched_armor_slot")
    observation_mask = (
        np.all(np.isfinite(pnp), axis=1)
        & np.all(np.isfinite(armor_truth), axis=1)
        & np.isfinite(slots)
    )
    radial_absolute = np.asarray([], dtype=float)
    transverse = np.asarray([], dtype=float)
    slot_switches = 0
    if np.any(observation_mask):
        error = pnp[observation_mask] - armor_truth[observation_mask]
        truth_range = np.linalg.norm(armor_truth[observation_mask], axis=1)
        valid_range = truth_range > 1e-9
        line_of_sight = (
            armor_truth[observation_mask][valid_range]
            / truth_range[valid_range, np.newaxis]
        )
        error = error[valid_range]
        radial = np.sum(error * line_of_sight, axis=1)
        radial_absolute = np.abs(radial)
        transverse = np.linalg.norm(error - radial[:, np.newaxis] * line_of_sight, axis=1)
        observed_slots = slots[observation_mask].astype(int)
        slot_switches = int(np.sum(observed_slots[1:] != observed_slots[:-1]))
    result = {
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
        "matched_armor_slot_switches": slot_switches,
    }
    if radial_absolute.size:
        result.update(
            {
                "pnp_abs_radial_error_m_median": float(np.median(radial_absolute)),
                "pnp_abs_radial_error_m_p95": float(np.percentile(radial_absolute, 95)),
                "pnp_transverse_error_m_median": float(np.median(transverse)),
                "pnp_transverse_error_m_p95": float(np.percentile(transverse, 95)),
            }
        )
    return result


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
    slots = scalar(rows, "truth", "matched_armor_slot")
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
    for slot, color in enumerate(SLOT_COLORS):
        slot_mask = slots == slot
        observed_slot = pnp_mask & slot_mask
        truth_slot = armor_mask & slot_mask
        ax.scatter(
            pnp[observed_slot, 1], pnp[observed_slot, 0],
            s=9, alpha=0.25, color=color, linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            armor_truth[truth_slot, 1], armor_truth[truth_slot, 0],
            s=11, alpha=0.72, color=color, marker="x", linewidths=0.75,
            rasterized=True,
        )
    ax.plot(center_est[estimate_mask, 1], center_est[estimate_mask, 0],
            color=COLORS["estimate"], linewidth=1.5, label="EKF center")
    ax.plot(center_truth[truth_mask, 1], center_truth[truth_mask, 0],
            color="#111111", linestyle="--", linewidth=1.3, label="True center")
    if np.any(truth_mask):
        ax.scatter(center_truth[truth_mask][0, 1], center_truth[truth_mask][0, 0],
                   marker="x", s=36, color="#111111", linewidths=1.3,
                   label="True center start")
    top_view_title = "D  Top view (x-forward points upward; color = armor slot)"
    if summary.get("pnp_abs_radial_error_m_median") is not None:
        top_view_title += (
            "\n"
            f"{summary['matched_armor_slot_switches']} slot switches | "
            f"median |radial| {summary['pnp_abs_radial_error_m_median']:.3f} m | "
            f"transverse {summary['pnp_transverse_error_m_median']:.3f} m"
        )
    ax.set(xlabel="y left (m)", ylabel="x forward (m)",
           title=top_view_title)
    ax.set_aspect("equal", adjustable="box")
    trace_handles = [
        Line2D([], [], marker="o", linestyle="None", markersize=4,
               markerfacecolor="#555555", markeredgewidth=0,
               label="PnP observation"),
        Line2D([], [], marker="x", linestyle="None", markersize=5,
               color="#555555", label="Same-exposure truth armor"),
        Line2D([], [], color=COLORS["estimate"], label="EKF center"),
        Line2D([], [], color="#111111", linestyle="--", label="True center"),
    ]
    trace_legend = ax.legend(handles=trace_handles, ncol=2, loc="upper left")
    ax.add_artist(trace_legend)

    fig.suptitle(
        SCENARIO_TITLES[scenario_id]
        + "\n"
        + f"{summary['duration_s']:.1f} s timestamp window | "
        + f"processed {summary['processed_event_fps']:.1f} FPS | "
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
    parser.add_argument(
        "--prediction-only",
        action="store_true",
        help="generate only the prediction-horizon figure and summary",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "aim-stack.ekf11-baseline-summary/v1",
        "collection_manifest": str(args.manifest),
        "scenarios": {},
    }
    scenario_rows = {}
    for scenario in manifest["scenarios"]:
        scenario_id = scenario["id"]
        rows = load_jsonl(Path(scenario["jsonl"]))
        scenario_rows[scenario_id] = rows
        if not args.prediction_only:
            summary["scenarios"][scenario_id] = plot_scenario(
                scenario_id, rows, args.output_dir
            )
    if not args.prediction_only:
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    predictions = {
        scenario_id: prediction_horizon_metrics(rows)
        for scenario_id, rows in scenario_rows.items()
    }
    plot_prediction_horizons(predictions, args.output_dir)
    plot_small_armor_window_coverage(predictions, args.output_dir)
    prediction_summary = {
        "schema": "aim-stack.ekf11-prediction-horizon/v1",
        "collection_manifest": str(args.manifest),
        "evaluation_standard": {
            "id": PREDICTION_STANDARD["id"],
            "schema": PREDICTION_STANDARD["schema"],
            "path": str(PREDICTION_STANDARD_PATH),
        },
        "method": {
            "source_state": "post-update Tongji 11D EKF state while tracker_state=tracking",
            "rollout": "constant center velocity, constant yaw rate, fixed radii and heights",
            "armor_correspondence": (
                "nearest decoded EKF candidate to the same-exposure truth-matched physical armor; "
                "the correspondence is held for the future horizon"
            ),
            "future_truth": (
                "later recorded exposure truth, linearly interpolated only when the bracketing gap "
                f"is <= {MAX_TRUTH_INTERPOLATION_GAP_S * 1000.0:.0f} ms"
            ),
            "normal_radial_definition": (
                "horizontal target-center-to-armor axis; this is also the armor normal in the Z4 model"
            ),
            "tangential_definition": "horizontal axis perpendicular to normal_radial",
            "vertical_definition": "tracker +z, lying in the armor plane",
            "window_definition": (
                "valid-sample geometric coverage: abs(tangential) <= half width and "
                "abs(vertical) <= half height"
            ),
        },
        "scenarios": predictions,
    }
    (args.output_dir / "prediction_horizon_summary.json").write_text(
        json.dumps(prediction_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
