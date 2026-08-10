#!/usr/bin/env python3
"""Build hit-oriented corner -> PnP -> state/prediction ablation evidence.

The analysis is deliberately offline and oracle-identity.  Truth is used only
to join/score samples and to construct controlled PnP-residual scaling arms.
It does not implement or authorize a production predictor.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
HORIZONS_S = (0.0, 0.05, 0.10, 0.20)
HISTORY_SIZE = 16
MAX_HISTORY_SPAN_S = 0.75
MAX_HISTORY_GAP_S = 0.12
MAX_FUTURE_BRACKET_S = 0.04
EVALUATION_INTERVAL_S = 0.10
FIRE_YAW_MISS_TOLERANCE_M = 0.055
SMALL_ARMOR_HALF_WIDTH_M = 0.0675
SMALL_ARMOR_HALF_HEIGHT_M = 0.0275

MOTION_ORDER = ("stationary", "linear", "spin", "linear_and_spin")
MOTION_LABEL = {
    "stationary": "Stationary",
    "linear": "Translation",
    "spin": "Rotation",
    "linear_and_spin": "Combined",
}
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#222222",
    "gray": "#777777",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corner-arm-rows", required=True, type=Path)
    parser.add_argument("--paired-trajectory-rows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Support old pandas builds that cannot open Unicode Windows paths."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return pd.read_csv(handle, **kwargs)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write through a Python handle for Unicode-path compatibility."""
    if path.suffix == ".gz":
        with gzip.open(str(path), "wt", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
    else:
        with path.open("w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)


def finite_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def percentile(values: Iterable[float], q: float) -> float:
    array = finite_array(values)
    return float(np.percentile(array, q)) if len(array) else float("nan")


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(valid) < 3:
        return float("nan")
    aa, bb = a[valid], b[valid]
    if np.std(aa) <= 1e-12 or np.std(bb) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.8,
        }
    )


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def add_aim_metrics(frame: pd.DataFrame, prefix: str, truth_prefix: str) -> pd.DataFrame:
    pred = np.asarray(
        frame[[f"{prefix}_x_m", f"{prefix}_y_m", f"{prefix}_z_m"]].values,
        dtype=float,
    )
    truth = np.asarray(
        frame[
            [f"{truth_prefix}_x_m", f"{truth_prefix}_y_m", f"{truth_prefix}_z_m"]
        ].values,
        dtype=float,
    )
    error = pred - truth
    truth_norm = np.linalg.norm(truth, axis=1)
    rhat = truth / np.maximum(truth_norm[:, None], 1e-12)
    radial = np.sum(error * rhat, axis=1)
    transverse = np.linalg.norm(error - radial[:, None] * rhat, axis=1)
    valid_depth = (pred[:, 2] > 1e-6) & (truth[:, 2] > 1e-6)
    yaw_miss = np.full(len(frame), np.nan)
    pitch_miss = np.full(len(frame), np.nan)
    yaw_miss[valid_depth] = (
        truth[valid_depth, 2]
        * pred[valid_depth, 0]
        / pred[valid_depth, 2]
        - truth[valid_depth, 0]
    )
    pitch_miss[valid_depth] = (
        truth[valid_depth, 2]
        * pred[valid_depth, 1]
        / pred[valid_depth, 2]
        - truth[valid_depth, 1]
    )
    pred_ray = pred / np.maximum(np.linalg.norm(pred, axis=1)[:, None], 1e-12)
    truth_ray = truth / np.maximum(truth_norm[:, None], 1e-12)
    angular = np.degrees(
        np.arccos(np.clip(np.sum(pred_ray * truth_ray, axis=1), -1.0, 1.0))
    )
    result = frame.copy()
    result["position_error_m"] = np.linalg.norm(error, axis=1)
    result["radial_error_signed_m"] = radial
    result["radial_error_abs_m"] = np.abs(radial)
    result["transverse_error_m"] = transverse
    result["yaw_plane_miss_signed_m"] = yaw_miss
    result["yaw_plane_miss_abs_m"] = np.abs(yaw_miss)
    result["pitch_plane_miss_signed_m"] = pitch_miss
    result["pitch_plane_miss_abs_m"] = np.abs(pitch_miss)
    result["ray_angular_error_deg"] = angular
    result["fire_yaw_55mm_pass"] = np.abs(yaw_miss) <= FIRE_YAW_MISS_TOLERANCE_M
    result["small_armor_center_rectangle_proxy_pass"] = (
        (np.abs(yaw_miss) <= SMALL_ARMOR_HALF_WIDTH_M)
        & (np.abs(pitch_miss) <= SMALL_ARMOR_HALF_HEIGHT_M)
    )
    result["small_armor_yaw_half_width_units"] = (
        np.abs(yaw_miss) / SMALL_ARMOR_HALF_WIDTH_M
    )
    result["small_armor_pitch_half_height_units"] = (
        np.abs(pitch_miss) / SMALL_ARMOR_HALF_HEIGHT_M
    )
    return result


def metric_summary(rows: pd.DataFrame, group_cols: list[str], scope: str) -> pd.DataFrame:
    metrics = (
        "position_error_m",
        "radial_error_abs_m",
        "transverse_error_m",
        "yaw_plane_miss_abs_m",
        "pitch_plane_miss_abs_m",
        "ray_angular_error_deg",
    )
    output: list[dict] = []
    grouped = rows.groupby(group_cols)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        common = {name: value for name, value in zip(group_cols, keys)}
        for metric in metrics:
            values = finite_array(group[metric].values)
            if not len(values):
                continue
            output.append(
                {
                    "scope": scope,
                    **common,
                    "metric": metric,
                    "samples": int(len(values)),
                    "mean": float(np.mean(values)),
                    "p50": float(np.percentile(values, 50)),
                    "p90": float(np.percentile(values, 90)),
                    "p95": float(np.percentile(values, 95)),
                    "p99": float(np.percentile(values, 99)),
                    "maximum": float(np.max(values)),
                }
            )
        output.append(
            {
                "scope": scope,
                **common,
                "metric": "fire_yaw_55mm_pass_rate",
                "samples": int(len(group)),
                "mean": float(np.mean(group["fire_yaw_55mm_pass"].astype(float))),
            }
        )
        output.append(
            {
                "scope": scope,
                **common,
                "metric": "small_armor_center_rectangle_proxy_pass_rate",
                "samples": int(len(group)),
                "mean": float(
                    np.mean(group["small_armor_center_rectangle_proxy_pass"].astype(float))
                ),
            }
        )
    return pd.DataFrame(output)


def export_corner_evidence(source: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = read_csv(source)
    selected = frame[frame["arm"].isin(("exact", "actual_raw", "actual_refined"))].copy()
    keep = [
        "session_id",
        "producer_epoch",
        "frame_seq",
        "timestamp_ns",
        "armor_index",
        "truth_slot",
        "motion_mode",
        "range_m",
        "view_incidence_cos",
        "projected_sqrt_area_px",
        "arm",
        "input_corner_coordinate_rms_px",
        "selected_position_error_m",
        "selected_lateral_error_m",
        "selected_depth_error_m",
        "selected_reprojection_rms_px",
        "selection_regret_m",
    ]
    selected = selected[keep].copy()
    selected["selected_abs_depth_error_m"] = np.abs(selected["selected_depth_error_m"])
    write_csv(selected, output / "corner_to_pnp_samples.csv.gz")

    metrics: list[dict] = []
    for arm, group in selected.groupby("arm"):
        for name in (
            "input_corner_coordinate_rms_px",
            "selected_position_error_m",
            "selected_lateral_error_m",
            "selected_abs_depth_error_m",
            "selected_reprojection_rms_px",
        ):
            values = finite_array(group[name].values)
            metrics.append(
                {
                    "arm": arm,
                    "metric": name,
                    "samples": len(values),
                    "mean": float(np.mean(values)),
                    "p50": float(np.percentile(values, 50)),
                    "p90": float(np.percentile(values, 90)),
                    "p95": float(np.percentile(values, 95)),
                    "p99": float(np.percentile(values, 99)),
                    "maximum": float(np.max(values)),
                }
            )
    summary = pd.DataFrame(metrics)
    write_csv(summary, output / "corner_to_pnp_metrics.csv")
    return selected, summary


def plot_corner_ecdf(rows: pd.DataFrame, output: Path) -> None:
    names = {
        "exact": ("Exact corners", COLORS["green"]),
        "actual_raw": ("Raw corners", COLORS["blue"]),
        "actual_refined": ("Refined corners", COLORS["orange"]),
    }
    panels = (
        ("selected_position_error_m", "3D position error (m)"),
        ("selected_lateral_error_m", "Camera-lateral error (m)"),
        ("selected_abs_depth_error_m", "Absolute depth error (m)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
    for axis, (metric, label) in zip(axes, panels):
        for arm in ("exact", "actual_raw", "actual_refined"):
            values = np.sort(finite_array(rows.loc[rows.arm == arm, metric].values))
            y = np.arange(1, len(values) + 1) / len(values)
            axis.step(values, y, where="post", label=f"{names[arm][0]} (n={len(values):,})", color=names[arm][1])
        axis.set_xscale("symlog", linthresh=1e-5)
        axis.set_xlabel(label)
        axis.set_ylabel("Empirical CDF")
        axis.set_ylim(0, 1.005)
        axis.legend(loc="lower right")
    fig.suptitle("Exact-corner intervention closes PnP; measured corners leave depth-dominated tails")
    save_figure(fig, output, "corner_exact_raw_refined_ecdf")


def build_pnp_directional(source: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [
        "dataset",
        "session_id",
        "motion_mode",
        "configured_distance_m",
        "configured_yaw_deg",
        "configured_linear_speed_mps",
        "configured_spin_rad_s",
        "time_s",
        "truth_slot",
        "is_visible_slot",
        "truth_camera_x_m",
        "truth_camera_y_m",
        "truth_camera_z_m",
        "pnp_camera_x_m",
        "pnp_camera_y_m",
        "pnp_camera_z_m",
    ]
    frame = read_csv(source, usecols=columns)
    finite = np.isfinite(
        np.asarray(
            frame[
                [
                    "truth_camera_x_m",
                    "truth_camera_y_m",
                    "truth_camera_z_m",
                    "pnp_camera_x_m",
                    "pnp_camera_y_m",
                    "pnp_camera_z_m",
                ]
            ].values,
            dtype=float,
        )
    ).all(axis=1)
    frame = frame.loc[finite].copy()
    directional = add_aim_metrics(frame, "pnp_camera", "truth_camera")
    write_csv(directional, output / "pnp_directional_samples.csv.gz")

    visible = directional[directional["is_visible_slot"]].copy()
    summaries = [
        metric_summary(visible, ["motion_mode"], "visible_by_motion"),
        metric_summary(
            visible,
            ["motion_mode", "configured_distance_m"],
            "visible_by_motion_distance",
        ),
        metric_summary(
            visible,
            ["motion_mode", "configured_linear_speed_mps", "configured_spin_rad_s"],
            "visible_by_motion_rate",
        ),
    ]
    summary = pd.concat(summaries, ignore_index=True, sort=False)
    write_csv(summary, output / "pnp_directional_metrics.csv")

    fits: list[dict] = []
    for (session_id, slot), group in visible.groupby(["session_id", "truth_slot"]):
        ordered = group.sort_values("time_s")
        truth = np.asarray(
            ordered[["truth_camera_x_m", "truth_camera_y_m", "truth_camera_z_m"]].values,
            dtype=float,
        )
        pnp = np.asarray(
            ordered[["pnp_camera_x_m", "pnp_camera_y_m", "pnp_camera_z_m"]].values,
            dtype=float,
        )
        truth_uv = truth[:, :2] / truth[:, 2:3]
        pnp_uv = pnp[:, :2] / pnp[:, 2:3]
        median_range = float(np.median(np.linalg.norm(truth, axis=1)))
        pointwise = (pnp_uv - truth_uv) * median_range
        truth_steps = np.linalg.norm(np.diff(truth_uv, axis=0), axis=1)
        pnp_steps = np.linalg.norm(np.diff(pnp_uv, axis=0), axis=1)
        fits.append(
            {
                "session_id": session_id,
                "truth_slot": int(slot),
                "motion_mode": str(ordered["motion_mode"].iloc[0]),
                "configured_distance_m": float(ordered["configured_distance_m"].iloc[0]),
                "configured_linear_speed_mps": float(ordered["configured_linear_speed_mps"].iloc[0]),
                "configured_spin_rad_s": float(ordered["configured_spin_rad_s"].iloc[0]),
                "samples": int(len(ordered)),
                "time_span_s": float(ordered["time_s"].iloc[-1] - ordered["time_s"].iloc[0]),
                "yaw_ray_correlation": safe_corr(truth_uv[:, 0], pnp_uv[:, 0]),
                "pitch_ray_correlation": safe_corr(truth_uv[:, 1], pnp_uv[:, 1]),
                "ray_plane_pointwise_rmse_m": float(np.sqrt(np.mean(np.sum(pointwise**2, axis=1)))),
                "ray_path_length_ratio": float(np.sum(pnp_steps) / max(np.sum(truth_steps), 1e-12)),
                "radial_bias_m": float(np.mean(ordered["radial_error_signed_m"])),
                "radial_abs_p95_m": percentile(ordered["radial_error_abs_m"], 95),
                "transverse_p95_m": percentile(ordered["transverse_error_m"], 95),
                "yaw_plane_miss_p95_m": percentile(ordered["yaw_plane_miss_abs_m"], 95),
            }
        )
    fit_frame = pd.DataFrame(fits)
    write_csv(fit_frame, output / "trajectory_fit_metrics.csv")
    return directional, summary, fit_frame


def plot_pnp_directional_ecdf(rows: pd.DataFrame, output: Path) -> None:
    visible = rows[rows["is_visible_slot"]]
    distances = sorted(float(value) for value in visible["configured_distance_m"].unique())
    colors = (COLORS["blue"], COLORS["orange"], COLORS["green"])
    fig, axes = plt.subplots(4, 2, figsize=(10.5, 12.0), sharey=True)
    for row_index, motion in enumerate(MOTION_ORDER):
        local = visible[visible.motion_mode == motion]
        for col_index, (metric, label) in enumerate(
            (("radial_error_abs_m", "Absolute LOS-radial error (m)"), ("transverse_error_m", "LOS-transverse error (m)"))
        ):
            axis = axes[row_index, col_index]
            for distance, color in zip(distances, colors):
                values = np.sort(
                    finite_array(local.loc[np.isclose(local.configured_distance_m, distance), metric].values)
                )
                if not len(values):
                    continue
                axis.step(
                    values,
                    np.arange(1, len(values) + 1) / len(values),
                    where="post",
                    color=color,
                    label=f"{distance:g} m (n={len(values):,})",
                )
            axis.set_xscale("symlog", linthresh=1e-3)
            axis.set_xlabel(label)
            axis.set_ylabel("Empirical CDF")
            axis.set_ylim(0, 1.005)
            axis.set_title(MOTION_LABEL[motion])
            if row_index == 0:
                axis.legend(loc="lower right")
    fig.suptitle("Current PnP error direction: depth/radial and hit-relevant transverse components")
    save_figure(fig, output, "pnp_radial_transverse_ecdf_by_motion_distance")


def aim_metrics_one(pred: np.ndarray, truth: np.ndarray) -> dict[str, float | bool]:
    error = pred - truth
    truth_norm = float(np.linalg.norm(truth))
    rhat = truth / max(truth_norm, 1e-12)
    radial = float(np.dot(error, rhat))
    transverse = float(np.linalg.norm(error - radial * rhat))
    if pred[2] <= 1e-6 or truth[2] <= 1e-6:
        yaw_miss = pitch_miss = angular = float("nan")
    else:
        yaw_miss = float(truth[2] * pred[0] / pred[2] - truth[0])
        pitch_miss = float(truth[2] * pred[1] / pred[2] - truth[1])
        dot = float(np.dot(pred / np.linalg.norm(pred), truth / truth_norm))
        angular = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
    return {
        "position_error_m": float(np.linalg.norm(error)),
        "radial_error_signed_m": radial,
        "radial_error_abs_m": abs(radial),
        "transverse_error_m": transverse,
        "yaw_plane_miss_signed_m": yaw_miss,
        "yaw_plane_miss_abs_m": abs(yaw_miss),
        "pitch_plane_miss_signed_m": pitch_miss,
        "pitch_plane_miss_abs_m": abs(pitch_miss),
        "ray_angular_error_deg": angular,
        "fire_yaw_55mm_pass": abs(yaw_miss) <= FIRE_YAW_MISS_TOLERANCE_M,
        "small_armor_center_rectangle_proxy_pass": (
            abs(yaw_miss) <= SMALL_ARMOR_HALF_WIDTH_M
            and abs(pitch_miss) <= SMALL_ARMOR_HALF_HEIGHT_M
        ),
        "small_armor_yaw_half_width_units": abs(yaw_miss) / SMALL_ARMOR_HALF_WIDTH_M,
        "small_armor_pitch_half_height_units": abs(pitch_miss) / SMALL_ARMOR_HALF_HEIGHT_M,
    }


def interpolate_bracket(times: np.ndarray, truth: np.ndarray, target_s: float) -> np.ndarray | None:
    index = int(np.searchsorted(times, target_s))
    if index == 0 or index >= len(times):
        return None
    before, after = float(times[index - 1]), float(times[index])
    if after - before > MAX_FUTURE_BRACKET_S:
        return None
    weight = (target_s - before) / max(after - before, 1e-12)
    return truth[index - 1] * (1.0 - weight) + truth[index] * weight


def build_prediction_ablation(
    directional: pd.DataFrame, output: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    visible = directional[directional["is_visible_slot"]].copy()
    prediction_rows: list[dict] = []
    for (session_id, slot), group in visible.groupby(["session_id", "truth_slot"]):
        ordered = group.sort_values("time_s")
        times = np.asarray(ordered["time_s"].values, dtype=float)
        truth = np.asarray(
            ordered[["truth_camera_x_m", "truth_camera_y_m", "truth_camera_z_m"]].values,
            dtype=float,
        )
        pnp = np.asarray(
            ordered[["pnp_camera_x_m", "pnp_camera_y_m", "pnp_camera_z_m"]].values,
            dtype=float,
        )
        meta = ordered.iloc[0]
        last_evaluation_s = -float("inf")
        for index in range(HISTORY_SIZE - 1, len(times)):
            history_times = times[index - HISTORY_SIZE + 1 : index + 1]
            gaps = np.diff(history_times)
            if (
                history_times[-1] - history_times[0] > MAX_HISTORY_SPAN_S
                or np.max(gaps) > MAX_HISTORY_GAP_S
                or history_times[-1] - last_evaluation_s < EVALUATION_INTERVAL_S
            ):
                continue
            last_evaluation_s = float(history_times[-1])
            local_times = history_times - history_times[-1]
            design = np.column_stack([np.ones(HISTORY_SIZE), local_times])
            history_truth = truth[index - HISTORY_SIZE + 1 : index + 1]
            history_residual = pnp[index - HISTORY_SIZE + 1 : index + 1] - history_truth
            truth_coef = np.linalg.lstsq(design, history_truth, rcond=None)[0]
            residual_coef = np.linalg.lstsq(design, history_residual, rcond=None)[0]
            for horizon_s in HORIZONS_S:
                future_s = float(history_times[-1] + horizon_s)
                future_truth = (
                    truth[index]
                    if horizon_s == 0.0
                    else interpolate_bracket(times, truth, future_s)
                )
                if future_truth is None:
                    continue
                for alpha in ALPHAS:
                    anchor_observation = truth[index] + alpha * (pnp[index] - truth[index])
                    cv_prediction = (
                        truth_coef[0]
                        + truth_coef[1] * horizon_s
                        + alpha * (residual_coef[0] + residual_coef[1] * horizon_s)
                    )
                    for method, prediction in (
                        ("hold", anchor_observation),
                        ("cv_ols_16", cv_prediction),
                    ):
                        metrics = aim_metrics_one(prediction, future_truth)
                        observation_yaw_at_future = float(
                            future_truth[2] * anchor_observation[0] / anchor_observation[2]
                        )
                        prediction_yaw_at_future = float(
                            future_truth[2] * prediction[0] / prediction[2]
                        )
                        observation_pitch_at_future = float(
                            future_truth[2] * anchor_observation[1] / anchor_observation[2]
                        )
                        prediction_pitch_at_future = float(
                            future_truth[2] * prediction[1] / prediction[2]
                        )
                        prediction_rows.append(
                            {
                                "session_id": session_id,
                                "truth_slot": int(slot),
                                "motion_mode": str(meta["motion_mode"]),
                                "configured_distance_m": float(meta["configured_distance_m"]),
                                "configured_yaw_deg": float(meta["configured_yaw_deg"]),
                                "configured_linear_speed_mps": float(meta["configured_linear_speed_mps"]),
                                "configured_spin_rad_s": float(meta["configured_spin_rad_s"]),
                                "anchor_time_s": float(history_times[-1]),
                                "future_time_s": future_s,
                                "horizon_s": horizon_s,
                                "pnp_residual_scale_alpha": alpha,
                                "pnp_error_reduction_fraction": 1.0 - alpha,
                                "method": method,
                                "history_samples": HISTORY_SIZE,
                                "history_span_s": float(history_times[-1] - history_times[0]),
                                "history_max_gap_s": float(np.max(gaps)),
                                "anchor_observation_x_m": float(anchor_observation[0]),
                                "anchor_observation_y_m": float(anchor_observation[1]),
                                "anchor_observation_z_m": float(anchor_observation[2]),
                                "prediction_x_m": float(prediction[0]),
                                "prediction_y_m": float(prediction[1]),
                                "prediction_z_m": float(prediction[2]),
                                "future_truth_x_m": float(future_truth[0]),
                                "future_truth_y_m": float(future_truth[1]),
                                "future_truth_z_m": float(future_truth[2]),
                                "observation_ray_x_at_future_truth_z_m": observation_yaw_at_future,
                                "observation_ray_y_at_future_truth_z_m": observation_pitch_at_future,
                                "prediction_ray_x_at_future_truth_z_m": prediction_yaw_at_future,
                                "prediction_ray_y_at_future_truth_z_m": prediction_pitch_at_future,
                                **metrics,
                            }
                        )
    predictions = pd.DataFrame(prediction_rows)
    write_csv(predictions, output / "prediction_ablation_samples.csv.gz")

    summaries = [
        metric_summary(
            predictions,
            ["motion_mode", "method", "pnp_residual_scale_alpha", "horizon_s"],
            "motion",
        ),
        metric_summary(
            predictions,
            [
                "motion_mode",
                "configured_linear_speed_mps",
                "configured_spin_rad_s",
                "method",
                "pnp_residual_scale_alpha",
                "horizon_s",
            ],
            "motion_rate",
        ),
        metric_summary(
            predictions,
            [
                "motion_mode",
                "configured_distance_m",
                "method",
                "pnp_residual_scale_alpha",
                "horizon_s",
            ],
            "motion_distance",
        ),
    ]
    summary = pd.concat(summaries, ignore_index=True, sort=False)
    write_csv(summary, output / "prediction_ablation_metrics.csv")

    threshold_rows: list[dict] = []
    cv = summary[
        (summary["scope"] == "motion")
        & (summary["method"] == "cv_ols_16")
        & (summary["metric"] == "yaw_plane_miss_abs_m")
    ]
    for (motion, horizon), group in cv.groupby(["motion_mode", "horizon_s"]):
        passing = group[group["p95"] <= FIRE_YAW_MISS_TOLERANCE_M]
        exact = group[np.isclose(group["pnp_residual_scale_alpha"], 0.0)]
        current = group[np.isclose(group["pnp_residual_scale_alpha"], 1.0)]
        maximum_alpha = (
            float(passing["pnp_residual_scale_alpha"].max()) if len(passing) else float("nan")
        )
        threshold_rows.append(
            {
                "motion_mode": motion,
                "horizon_s": float(horizon),
                "criterion": "P95 absolute yaw-plane miss <= 0.055 m",
                "current_alpha_1_p95_m": float(current["p95"].iloc[0]) if len(current) else float("nan"),
                "exact_pnp_alpha_0_floor_p95_m": float(exact["p95"].iloc[0]) if len(exact) else float("nan"),
                "largest_tested_pnp_residual_scale_passing": maximum_alpha,
                "minimum_tested_pnp_error_reduction_fraction": (
                    1.0 - maximum_alpha if math.isfinite(maximum_alpha) else float("nan")
                ),
                "exact_pnp_can_pass": bool(len(exact) and float(exact["p95"].iloc[0]) <= FIRE_YAW_MISS_TOLERANCE_M),
                "current_pnp_can_pass": bool(len(current) and float(current["p95"].iloc[0]) <= FIRE_YAW_MISS_TOLERANCE_M),
                "samples_current": int(current["samples"].iloc[0]) if len(current) else 0,
            }
        )
    thresholds = pd.DataFrame(threshold_rows)
    write_csv(thresholds, output / "pnp_improvement_thresholds.csv")

    representative_rows: list[pd.DataFrame] = []
    for motion in MOTION_ORDER:
        representative_horizon = 0.05 if motion == "linear_and_spin" else 0.10
        candidates = predictions[
            (predictions["method"] == "cv_ols_16")
            & np.isclose(predictions["pnp_residual_scale_alpha"], 1.0)
            & np.isclose(predictions["horizon_s"], representative_horizon)
        ]
        local = candidates[candidates["motion_mode"] == motion]
        if not len(local):
            continue
        counts = local.groupby(["session_id", "truth_slot"]).size().sort_values(ascending=False)
        session_id, slot = counts.index[0]
        chosen = local[(local["session_id"] == session_id) & (local["truth_slot"] == slot)].copy()
        chosen["representative_reason"] = (
            "largest current-PnP CV trajectory within motion at selected display horizon"
        )
        representative_rows.append(chosen)
    representatives = pd.concat(representative_rows, ignore_index=True, sort=False)
    write_csv(representatives, output / "representative_prediction_trajectories.csv")
    return predictions, summary, thresholds, representatives


def plot_trajectory_overlays(
    directional: pd.DataFrame, representatives: pd.DataFrame, output: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    for axis, motion in zip(axes.flat, MOTION_ORDER):
        rep = representatives[representatives.motion_mode == motion]
        if not len(rep):
            axis.set_visible(False)
            continue
        session = rep.session_id.iloc[0]
        slot = int(rep.truth_slot.iloc[0])
        local = directional[
            (directional.session_id == session)
            & (directional.truth_slot == slot)
            & directional.is_visible_slot
        ].sort_values("time_s")
        truth = np.asarray(local[["truth_camera_x_m", "truth_camera_z_m"]].values, dtype=float)
        pnp = np.asarray(local[["pnp_camera_x_m", "pnp_camera_z_m"]].values, dtype=float)
        axis.plot(truth[:, 0], truth[:, 1], color=COLORS["black"], label="Truth", zorder=3)
        stride = max(1, len(truth) // 80)
        axis.scatter(
            truth[::stride, 0],
            truth[::stride, 1],
            s=9,
            color=COLORS["black"],
            zorder=4,
        )
        axis.scatter(pnp[:, 0], pnp[:, 1], s=5, alpha=0.28, color=COLORS["orange"], label="Observed PnP")
        axis.set_xlabel("Camera right x (m)")
        axis.set_ylabel("Camera forward z / depth (m)")
        meta = local.iloc[0]
        axis.set_title(
            f"{MOTION_LABEL[motion]} — d={float(meta.configured_distance_m):g} m, "
            f"v={float(meta.configured_linear_speed_mps):g} m/s, "
            f"ω={float(meta.configured_spin_rad_s):g} rad/s"
        )
        axis.legend()
    fig.suptitle("Observed PnP trajectory versus exact trajectory (camera x–depth plane)")
    save_figure(fig, output, "trajectory_overlay_observed_truth_camera_xz")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for axis, motion in zip(axes.flat, MOTION_ORDER):
        local = representatives[representatives.motion_mode == motion].sort_values("future_time_s")
        if not len(local):
            axis.set_visible(False)
            continue
        start = float(local.future_time_s.min())
        local = local[local.future_time_s <= start + 3.0]
        time = np.asarray(local.future_time_s.values, dtype=float) - start
        axis.plot(time, np.asarray(local.future_truth_x_m.values, dtype=float) * 1000.0, color=COLORS["black"], label="Future truth")
        axis.plot(time, np.asarray(local.observation_ray_x_at_future_truth_z_m.values, dtype=float) * 1000.0, color=COLORS["orange"], linestyle=":", label="Hold current observation")
        axis.plot(time, np.asarray(local.prediction_ray_x_at_future_truth_z_m.values, dtype=float) * 1000.0, color=COLORS["blue"], linestyle="--", label="Simple CV prediction")
        axis.set_xlabel("Future evaluation time within shown arc (s)")
        axis.set_ylabel("Horizontal aim coordinate at truth depth (mm)")
        horizon_ms = int(round(float(local.horizon_s.iloc[0]) * 1000.0))
        axis.set_title(f"{MOTION_LABEL[motion]} — {horizon_ms} ms horizon")
        axis.legend()
    fig.suptitle("Prediction, held observation, and truth on the same hit-relevant plane")
    save_figure(fig, output, "trajectory_overlay_prediction_observation_truth")


def plot_prediction_ablation(summary: pd.DataFrame, output: Path) -> None:
    base = summary[
        (summary.scope == "motion")
        & (summary.method == "cv_ols_16")
        & (summary.metric == "yaw_plane_miss_abs_m")
    ]
    horizon_colors = {
        0.0: COLORS["gray"],
        0.05: COLORS["green"],
        0.10: COLORS["blue"],
        0.20: COLORS["red"],
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), sharex=True, sharey=True)
    for axis, motion in zip(axes.flat, MOTION_ORDER):
        local = base[base.motion_mode == motion]
        for horizon in HORIZONS_S:
            line = local[np.isclose(local.horizon_s, horizon)].sort_values("pnp_residual_scale_alpha")
            axis.plot(
                line.pnp_residual_scale_alpha,
                line.p95 * 1000.0,
                marker="o",
                color=horizon_colors[horizon],
                label=f"{int(round(horizon * 1000))} ms",
            )
        axis.axhline(FIRE_YAW_MISS_TOLERANCE_M * 1000.0, color=COLORS["black"], linestyle="--", linewidth=1.2, label="55 mm yaw gate")
        axis.set_title(MOTION_LABEL[motion])
        axis.set_xlabel("Remaining PnP residual scale α (0=exact, 1=current)")
        axis.set_ylabel("P95 horizontal miss at truth depth (mm)")
        axis.legend(ncol=2)
    fig.suptitle("How much PnP improvement can reduce simple-CV future aiming error?")
    save_figure(fig, output, "prediction_p95_vs_pnp_residual_scale")

    current_exact = base[base.pnp_residual_scale_alpha.isin((0.0, 1.0)) & (base.horizon_s > 0)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    width = 0.34
    x = np.arange(len(MOTION_ORDER))
    for axis, horizon in zip(axes, (0.05, 0.10, 0.20)):
        local = current_exact[np.isclose(current_exact.horizon_s, horizon)]
        exact_values, current_values = [], []
        for motion in MOTION_ORDER:
            cell = local[local.motion_mode == motion]
            exact_values.append(float(cell.loc[np.isclose(cell.pnp_residual_scale_alpha, 0.0), "p95"].iloc[0]) * 1000.0 if len(cell.loc[np.isclose(cell.pnp_residual_scale_alpha, 0.0)]) else np.nan)
            current_values.append(float(cell.loc[np.isclose(cell.pnp_residual_scale_alpha, 1.0), "p95"].iloc[0]) * 1000.0 if len(cell.loc[np.isclose(cell.pnp_residual_scale_alpha, 1.0)]) else np.nan)
        axis.bar(x - width / 2, exact_values, width, color=COLORS["green"], label="Exact PnP input")
        axis.bar(x + width / 2, current_values, width, color=COLORS["orange"], label="Current PnP input")
        axis.axhline(55.0, color=COLORS["black"], linestyle="--", linewidth=1.2)
        axis.set_xticks(x)
        axis.set_xticklabels([MOTION_LABEL[m] for m in MOTION_ORDER], rotation=25, ha="right")
        axis.set_title(f"{int(horizon * 1000)} ms")
        axis.set_ylabel("P95 horizontal miss (mm)")
        axis.legend()
    fig.suptitle("Measurement floor versus motion-model floor for the same simple CV estimator")
    save_figure(fig, output, "prediction_motion_current_vs_exact_pnp")


def write_report(
    output: Path,
    corner_summary: pd.DataFrame,
    pnp_summary: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    trajectory_fits: pd.DataFrame,
) -> None:
    def lookup(frame: pd.DataFrame, **conditions) -> pd.DataFrame:
        result = frame
        for key, value in conditions.items():
            if isinstance(value, float):
                result = result[np.isclose(result[key].astype(float), value)]
            else:
                result = result[result[key] == value]
        return result

    corner_lines = []
    for arm in ("exact", "actual_raw", "actual_refined"):
        p = lookup(corner_summary, arm=arm, metric="selected_position_error_m")
        l = lookup(corner_summary, arm=arm, metric="selected_lateral_error_m")
        corner_lines.append(
            f"- {arm}: 3D P50/P95/P99 = {p.p50.iloc[0]*1000:.3f}/{p.p95.iloc[0]*1000:.3f}/{p.p99.iloc[0]*1000:.3f} mm; lateral P95 = {l.p95.iloc[0]*1000:.3f} mm."
        )

    pnp_lines = []
    for motion in MOTION_ORDER:
        radial = lookup(pnp_summary, scope="visible_by_motion", motion_mode=motion, metric="radial_error_abs_m")
        transverse = lookup(pnp_summary, scope="visible_by_motion", motion_mode=motion, metric="transverse_error_m")
        yaw = lookup(pnp_summary, scope="visible_by_motion", motion_mode=motion, metric="yaw_plane_miss_abs_m")
        pnp_lines.append(
            f"- {MOTION_LABEL[motion]}: |radial| P50/P95 = {radial.p50.iloc[0]*1000:.1f}/{radial.p95.iloc[0]*1000:.1f} mm; transverse P50/P95 = {transverse.p50.iloc[0]*1000:.1f}/{transverse.p95.iloc[0]*1000:.1f} mm; horizontal hit-plane P95 = {yaw.p95.iloc[0]*1000:.1f} mm."
        )

    threshold_lines = []
    for _, row in thresholds.iterrows():
        if float(row.horizon_s) == 0.0:
            continue
        if bool(row.current_pnp_can_pass):
            verdict = "current PnP already passes this offline P95 yaw gate"
        elif bool(row.exact_pnp_can_pass):
            reduction = float(row.minimum_tested_pnp_error_reduction_fraction) * 100.0
            verdict = f"requires at least {reduction:.0f}% PnP-residual reduction on the tested grid"
        else:
            verdict = "even exact PnP fails; PnP improvement alone cannot meet the gate"
        threshold_lines.append(
            f"- {MOTION_LABEL[str(row.motion_mode)]}, {int(round(float(row.horizon_s)*1000))} ms: current/exact-input P95 = {float(row.current_alpha_1_p95_m)*1000:.1f}/{float(row.exact_pnp_alpha_0_floor_p95_m)*1000:.1f} mm (n={int(row.samples_current)}); {verdict}."
        )

    fit_lines = []
    for motion in MOTION_ORDER:
        local = trajectory_fits[trajectory_fits.motion_mode == motion]
        fit_lines.append(
            f"- {MOTION_LABEL[motion]}: per-session/slot yaw-ray correlation median = {percentile(local.yaw_ray_correlation, 50):.3f}; ray-path inflation median = {percentile(local.ray_path_length_ratio, 50):.2f}x; pointwise ray-plane RMSE median = {percentile(local.ray_plane_pointwise_rmse_m, 50)*1000:.1f} mm."
        )

    text = f"""# Hit-oriented ablation evidence report

## Scope and causal boundary

This is an offline, oracle-identity ablation over retained evidence. Truth is used only after collection to pair/score rows and to synthesize `p_alpha = truth + alpha * (PnP - truth)`. The sweep keeps timestamps, visible-arc coverage, gaps, and slot histories fixed. It is not an online predictor, a fire-control acceptance result, or a live-hit probability estimate.

The hit-oriented plane is placed at the future truth depth and perpendicular to the camera optical axis. Horizontal miss is compared with the current configured yaw miss tolerance of 55 mm. The small-armor rectangle proxy uses half-width/half-height 67.5/27.5 mm. The rectangle proxy is optimistic because it omits plate obliquity, projectile flight, latency, dispersion, and mechanical/control errors.

## 1. Exact-corner intervention into the existing IPPE solver

{chr(10).join(corner_lines)}

The exact-corner arm uses the same planar IPPE solver and coordinate contract as the measured arms. Therefore its micrometre-scale closure is direct evidence that the large measured PnP tails originate upstream in corner geometry/planar conditioning, not in an unavoidable solver-coordinate mismatch.

## 2. Current PnP: depth and hit-relevant direction are different problems

{chr(10).join(pnp_lines)}

The full per-sample table is retained. Large 3D errors can be predominantly radial/depth while the camera-ray miss remains much smaller; conversely, a modest transverse error can already consume most of a 55 mm aiming corridor.

## 3. Observed trajectory versus exact trajectory

{chr(10).join(fit_lines)}

Pointwise correspondence and curve-shape agreement are both reported because a visually smooth/repeatable observed arc can still be geometrically biased, and high-frequency PnP jitter can greatly inflate apparent path length.

## 4. Truth-restoration and simple-filter ablation

The estimator is intentionally simple: 16 recent exposure-matched samples, ordinary least-squares constant velocity in camera XYZ, actual timestamps, maximum 120 ms history gap, and 50/100/200 ms forecasts. `alpha=1` is current PnP and `alpha=0` is exact position input. Physical slot identity is oracle-only, so these numbers isolate measurement/model error and do not solve association.

{chr(10).join(threshold_lines)}

The translation result should be read as a baseline, not as proof of deployment: its current-PnP P95 horizontal miss remains inside 55 mm across the tested horizons, while combined high-rate motion is limited primarily by model curvature/visibility at 100–200 ms. Rotation is locally predictable by CV over the supported visible arcs, but high-rate/long-horizon coverage becomes sparse; the rate-stratified table must be checked before generalizing.

## 5. Retained complete distributions

- `corner_to_pnp_samples.csv.gz`: every exact/raw/refined corner-to-pose sample used here.
- `pnp_directional_samples.csv.gz`: every paired PnP/truth row with radial, transverse, hit-plane, angular, and proxy-pass fields.
- `prediction_ablation_samples.csv.gz`: every estimator sample for both hold/CV, all six residual scales, and all four horizons.
- Summary CSV files are navigation aids only; they do not replace the sample tables.

## 6. Decision boundary

1. Corner work can dramatically improve 3D depth because exact corners close the current solver, but refinement must be optimized by directional/hit-plane effects and tails, not reprojection RMS alone.
2. PnP acceptance must separate radial depth from transverse miss. A single 3D norm or angle is insufficient.
3. PnP improvement can reduce the observation floor, but if the `alpha=0` arm fails, the remaining error is motion-model/history/coverage error and cannot be repaired by corners alone.
4. Translation is an appropriate simple-filter baseline. Rotation and especially combined motion require rate-aware nonlinear/periodic or multiple-model hypotheses, plus non-oracle association and coverage validation.
"""
    (output / "report.md").write_text(text, encoding="utf-8")


def write_manifest(output: Path, sources: list[Path], metadata: dict) -> None:
    artifacts = []
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.name == "retention_manifest.json" or not path.is_file():
            continue
        artifacts.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "autoaim-hit-oriented-ablation-v1",
        "protected": True,
        "truth_usage": "offline pairing, scoring, and controlled residual scaling only",
        "identity_boundary": "oracle physical slot; not deployable association",
        "metric_contract": {
            "fire_yaw_miss_tolerance_m": FIRE_YAW_MISS_TOLERANCE_M,
            "small_armor_half_width_m": SMALL_ARMOR_HALF_WIDTH_M,
            "small_armor_half_height_m": SMALL_ARMOR_HALF_HEIGHT_M,
            "hit_plane": "fronto-parallel plane at future truth camera depth; optimistic geometry proxy",
        },
        "prediction_contract": {
            "alphas": list(ALPHAS),
            "horizons_s": list(HORIZONS_S),
            "history_size": HISTORY_SIZE,
            "max_history_span_s": MAX_HISTORY_SPAN_S,
            "max_history_gap_s": MAX_HISTORY_GAP_S,
            "max_future_truth_bracket_s": MAX_FUTURE_BRACKET_S,
            "evaluation_interval_s": EVALUATION_INTERVAL_S,
            "methods": ["hold", "cv_ols_16"],
        },
        "sources": [
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sources
        ],
        "metadata": metadata,
        "artifacts": artifacts,
    }
    (output / "retention_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    sources = [args.corner_arm_rows.resolve(), args.paired_trajectory_rows.resolve()]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    configure_plotting()

    corner_rows, corner_summary = export_corner_evidence(sources[0], output)
    plot_corner_ecdf(corner_rows, output)

    directional, pnp_summary, trajectory_fits = build_pnp_directional(sources[1], output)
    plot_pnp_directional_ecdf(directional, output)

    predictions, prediction_summary, thresholds, representatives = build_prediction_ablation(
        directional, output
    )
    plot_trajectory_overlays(directional, representatives, output)
    plot_prediction_ablation(prediction_summary, output)
    write_report(
        output,
        corner_summary,
        pnp_summary,
        prediction_summary,
        thresholds,
        trajectory_fits,
    )
    metadata = {
        "corner_samples": int(len(corner_rows)),
        "paired_pnp_samples": int(len(directional)),
        "visible_pnp_samples": int(np.count_nonzero(directional.is_visible_slot)),
        "trajectory_fit_groups": int(len(trajectory_fits)),
        "prediction_samples": int(len(predictions)),
        "representative_prediction_samples": int(len(representatives)),
        "motions": sorted(str(value) for value in directional.motion_mode.unique()),
    }
    (output / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_manifest(output, sources, metadata)
    print(json.dumps({"output": str(output), **metadata}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
