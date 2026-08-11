#!/usr/bin/env python3
"""Propagate OOF corner repair through the frozen local combined-motion expert.

The corner network itself is frame-local.  Motion labels are used only to
construct leakage-resistant OOF domains and to score the downstream causal
400 ms LOS-aware rigid predictor at fixed future horizons.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HISTORY_S = 0.40
HORIZONS_S = (0.05, 0.10, 0.20)
ANCHOR_STRIDE = 24
DEPTH_WEIGHT = 0.1
HUBER_DELTA_M = 0.02
OMEGA_MEMORY_COUNT = 31
OMEGA_GRID_STEP = 1.0
MAX_OMEGA = 16.0
DOMAINS = (
    "current_refined",
    "raw",
    "network_segment_oof",
    "network_session_oof",
    "exact",
)
METHODS = (
    "hold",
    "same_slot_world_cv",
    "los_raw_omega",
    "los_memory31",
    "los_oracle_omega",
)
COLORS = {
    "current_refined": "#CC79A7",
    "raw": "#000000",
    "network_segment_oof": "#009E73",
    "network_session_oof": "#E69F00",
    "exact": "#0072B2",
}
LINESTYLES = {
    "current_refined": "--",
    "raw": "-",
    "network_segment_oof": "-",
    "network_session_oof": "-.",
    "exact": ":",
}
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pnp-rows", required=True, type=Path)
    result.add_argument("--corner-rows", required=True, type=Path)
    result.add_argument("--atlas", required=True, type=Path)
    result.add_argument("--observations", required=True, type=Path)
    result.add_argument("--truth", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    return result


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    def strict_json(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: strict_json(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [strict_json(child) for child in item]
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            return None
        if isinstance(item, np.integer):
            return int(item)
        return item

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(strict_json(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
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


def write_csv_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def camera_to_tracker_rotation(pitch_rad: float, yaw_rad: float) -> np.ndarray:
    cp, sp = math.cos(-pitch_rad), math.sin(-pitch_rad)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    pitch = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64
    )
    yaw = np.asarray(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64
    )
    camera_to_tracker_convention = np.asarray(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    return camera_to_tracker_convention @ yaw @ pitch


def camera_point_to_tracker(camera_point_m: np.ndarray, observation: dict[str, Any]) -> np.ndarray:
    camera_to_gimbal = np.asarray(
        observation["R_camera2gimbal"], dtype=np.float64
    ).reshape(3, 3)
    camera_to_gimbal_translation = np.asarray(
        observation["t_camera2gimbal_m"], dtype=np.float64
    )
    tracker_to_camera = camera_to_tracker_rotation(
        math.radians(float(observation["gimbal_pitch_deg"])),
        math.radians(float(observation["gimbal_yaw_deg"])),
    )
    tracker_to_gimbal = tracker_to_camera @ camera_to_gimbal.T
    return tracker_to_gimbal @ (
        camera_to_gimbal @ camera_point_m + camera_to_gimbal_translation
    )


def segment_ranges(atlas: pd.DataFrame) -> list[tuple[int, int, int]]:
    combined = atlas[atlas["session"] == "combined"]
    ranges = []
    for segment, rows in combined.groupby("segment_index", sort=True):
        ranges.append(
            (int(segment), int(rows["timestamp_ns"].min()), int(rows["timestamp_ns"].max()))
        )
    if len(ranges) != 6:
        raise ValueError(f"expected six frozen combined segments, got {len(ranges)}")
    return ranges


def segment_for_timestamp(timestamp_ns: int, ranges: list[tuple[int, int, int]]) -> int | None:
    for segment, start, end in ranges:
        if start <= timestamp_ns <= end:
            return segment
    return None


def load_truth_frames(base, truth_path: Path, ranges: list[tuple[int, int, int]]):
    frames = []
    keys = []
    with truth_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not bool(record.get("has_exact_exposure_truth", False)):
                continue
            timestamp_ns = int(record["timestamp_ns"])
            segment = segment_for_timestamp(timestamp_ns, ranges)
            if segment is None:
                continue
            target = base.selected_target(record)
            if target is None:
                continue
            local = np.asarray(
                [armor["chassis_local_position_m"] for armor in target["armors"]],
                dtype=np.float64,
            )
            center = np.asarray(target["world_position_m"], dtype=np.float64)
            target_rotation = base.quaternion_matrix(target["world_quaternion_wxyz"])
            armor_world = center[None, :] + local @ target_rotation.T
            exposure = record["exposure_state"]
            frame = base.Frame(
                timestamp_ns=timestamp_ns,
                center_world_m=center,
                velocity_world_mps=np.asarray(
                    target["world_velocity_mps"], dtype=np.float64
                ),
                yaw_rate_rad_s=float(target["world_vyaw_rad_s"]),
                armor_world_m=armor_world,
                armor_local_m=local,
                tracker_origin_world_m=np.asarray(
                    exposure["gimbal_position_world_m"], dtype=np.float64
                ),
                tracker_to_world=base.quaternion_matrix(
                    exposure["chassis_quaternion_world_wxyz"]
                ),
                observed_world_m=np.full((4, 3), np.nan, dtype=np.float64),
                observed_mask=np.zeros(4, dtype=np.bool_),
                observed_range_m=np.full(4, np.inf, dtype=np.float64),
                association_ambiguous=False,
                segment_id=segment,
            )
            frames.append(frame)
            keys.append(
                (
                    str(record["session_id"]),
                    int(record["producer_epoch"]),
                    int(record["frame_seq"]),
                    timestamp_ns,
                )
            )
    if not frames:
        raise ValueError("no exact truth frames inside atlas segments")
    order = np.argsort([frame.timestamp_ns for frame in frames])
    frames = [frames[int(index)] for index in order]
    keys = [keys[int(index)] for index in order]
    return frames, {key: index for index, key in enumerate(keys)}


def load_observations(path: Path) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            key = (
                str(record["session_id"]),
                int(record["producer_epoch"]),
                int(record["frame_seq"]),
                int(record["timestamp_ns"]),
            )
            result[key] = record
    return result


def select_domain_rows(pnp: pd.DataFrame, domain: str) -> pd.DataFrame:
    if domain == "network_segment_oof":
        scheme, arm = "leave_segment_out", "network"
    elif domain == "network_session_oof":
        scheme, arm = "leave_session_out", "network"
    else:
        scheme, arm = "leave_segment_out", domain
    return pnp[(pnp["scheme"] == scheme) & (pnp["arm"] == arm)].copy()


def populate_domain(
    base_frames,
    frame_lookup: dict[tuple[str, int, int, int], int],
    observation_lookup: dict[tuple[str, int, int, int], dict[str, Any]],
    rows: pd.DataFrame,
) -> tuple[list[Any], dict[str, float]]:
    frames = copy.deepcopy(base_frames)
    parity = []
    exact_error = []
    duplicates = 0
    for row in rows.itertuples(index=False):
        key = (
            str(row.session_id), int(row.producer_epoch), int(row.frame_seq),
            int(row.timestamp_ns),
        )
        frame_index = frame_lookup.get(key)
        observation = observation_lookup.get(key)
        if frame_index is None or observation is None:
            continue
        frame = frames[frame_index]
        slot = int(row.truth_slot)
        if frame.observed_mask[slot]:
            duplicates += 1
            continue
        camera_point = np.asarray(
            [row.camera_tvec_x_m, row.camera_tvec_y_m, row.camera_tvec_z_m],
            dtype=np.float64,
        )
        tracker_point = camera_point_to_tracker(camera_point, observation)
        world_point = (
            frame.tracker_origin_world_m + frame.tracker_to_world @ tracker_point
        )
        frame.observed_world_m[slot] = world_point
        frame.observed_mask[slot] = True
        frame.observed_range_m[slot] = float(np.linalg.norm(tracker_point[:2]))
        armor_index = int(row.armor_index)
        if str(row.arm) == "current_refined":
            expected = np.asarray(
                observation["armors"][armor_index]["position_m"], dtype=np.float64
            )
            parity.append(float(np.linalg.norm(tracker_point - expected)))
        if str(row.arm) == "exact":
            exact_error.append(float(np.linalg.norm(world_point - frame.armor_world_m[slot])))
    return frames, {
        "duplicates": duplicates,
        "observed_events": int(sum(frame.observed_mask.sum() for frame in frames)),
        "current_refined_tracker_parity_max_m": max(parity) if parity else math.nan,
        "exact_world_error_p95_m": float(np.quantile(exact_error, 0.95)) if exact_error else math.nan,
        "exact_world_error_max_m": max(exact_error) if exact_error else math.nan,
    }


def evaluate(base, los, frames_by_domain: dict[str, list[Any]]):
    reference = frames_by_domain["current_refined"]
    geometry = reference[0].armor_local_m
    omega_memory: dict[str, dict[int, list[float]]] = {
        domain: {} for domain in DOMAINS
    }
    prediction_rows = []
    omega_rows = []
    eligible_anchors = 0
    for anchor_index in range(0, len(reference), ANCHOR_STRIDE):
        reference_anchor = reference[anchor_index]
        current_slots = np.flatnonzero(reference_anchor.observed_mask)
        if current_slots.size == 0:
            continue
        primary = int(
            current_slots[
                np.argmin(reference_anchor.observed_range_m[current_slots])
            ]
        )
        histories = {
            domain: base.history_observations(
                frames, anchor_index, HISTORY_S, "pnp"
            )
            for domain, frames in frames_by_domain.items()
        }
        if any(history is None for history in histories.values()):
            continue
        if any(not frames[anchor_index].observed_mask[primary] for frames in frames_by_domain.values()):
            continue
        fitted = {}
        for domain, frames in frames_by_domain.items():
            anchor = frames[anchor_index]
            times, slots, positions = histories[domain]
            raw_omega, raw_coefficient, raw_loss = los.estimate_weighted_omega(
                base,
                times,
                slots,
                positions,
                geometry,
                anchor.tracker_to_world,
                depth_weight=DEPTH_WEIGHT,
                huber_delta_m=HUBER_DELTA_M,
                max_omega=MAX_OMEGA,
                grid_step=OMEGA_GRID_STEP,
            )
            memory = omega_memory[domain].setdefault(anchor.segment_id, [])
            memory.append(float(raw_omega))
            memory_omega = float(np.median(memory[-OMEGA_MEMORY_COUNT:]))
            memory_coefficient, memory_loss = los.fit_weighted_rigid(
                base,
                times,
                slots,
                positions,
                geometry,
                memory_omega,
                anchor.tracker_to_world,
                depth_weight=DEPTH_WEIGHT,
                huber_delta_m=HUBER_DELTA_M,
                robust=True,
            )
            oracle_coefficient, oracle_loss = los.fit_weighted_rigid(
                base,
                times,
                slots,
                positions,
                geometry,
                anchor.yaw_rate_rad_s,
                anchor.tracker_to_world,
                depth_weight=DEPTH_WEIGHT,
                huber_delta_m=HUBER_DELTA_M,
                robust=True,
            )
            active = slots == primary
            cv_coefficient = base.fit_cv(times[active], positions[active])
            fitted[domain] = {
                "hold": anchor.observed_world_m[primary],
                "same_slot_world_cv": cv_coefficient,
                "los_raw_omega": (raw_omega, raw_coefficient),
                "los_memory31": (memory_omega, memory_coefficient),
                "los_oracle_omega": (anchor.yaw_rate_rad_s, oracle_coefficient),
            }
            omega_rows.append(
                {
                    "timestamp_ns": anchor.timestamp_ns,
                    "segment_index": anchor.segment_id,
                    "input_domain": domain,
                    "truth_omega_rad_s": anchor.yaw_rate_rad_s,
                    "raw_omega_rad_s": raw_omega,
                    "memory_omega_rad_s": memory_omega,
                    "raw_absolute_error_rad_s": abs(raw_omega - anchor.yaw_rate_rad_s),
                    "memory_absolute_error_rad_s": abs(memory_omega - anchor.yaw_rate_rad_s),
                    "memory_count": min(len(memory), OMEGA_MEMORY_COUNT),
                    "history_event_count": len(times),
                    "raw_fit_loss": raw_loss,
                    "memory_fit_loss": memory_loss,
                    "oracle_fit_loss": oracle_loss,
                }
            )
        eligible_anchors += 1
        for horizon_s in HORIZONS_S:
            future = base.nearest_future(reference, anchor_index, horizon_s)
            if future is None or future.segment_id != reference_anchor.segment_id:
                continue
            effective_horizon = (
                future.timestamp_ns - reference_anchor.timestamp_ns
            ) / 1e9
            truth = future.armor_world_m[primary]
            for domain, frames in frames_by_domain.items():
                anchor = frames[anchor_index]
                for method in METHODS:
                    model = fitted[domain][method]
                    if model is None:
                        continue
                    if method == "hold":
                        prediction = np.asarray(model, dtype=np.float64)
                        model_omega = math.nan
                    elif method == "same_slot_world_cv":
                        prediction = np.asarray([1.0, effective_horizon]) @ np.asarray(model)
                        model_omega = math.nan
                    else:
                        model_omega, coefficient = model
                        prediction = base.predict_rigid(
                            np.asarray(coefficient), geometry, primary,
                            float(model_omega), effective_horizon,
                        )
                    item = {
                        "timestamp_ns": anchor.timestamp_ns,
                        "segment_index": anchor.segment_id,
                        "input_domain": domain,
                        "method": method,
                        "history_window_s": HISTORY_S,
                        "horizon_s": horizon_s,
                        "effective_horizon_s": effective_horizon,
                        "primary_slot": primary,
                        "visible_count_at_anchor": int(current_slots.size),
                        "truth_omega_rad_s": anchor.yaw_rate_rad_s,
                        "model_omega_rad_s": model_omega,
                        "prediction_x_m": float(prediction[0]),
                        "prediction_y_m": float(prediction[1]),
                        "prediction_z_m": float(prediction[2]),
                        "truth_x_m": float(truth[0]),
                        "truth_y_m": float(truth[1]),
                        "truth_z_m": float(truth[2]),
                    }
                    item.update(
                        base.error_components(
                            prediction, truth, reference_anchor.tracker_to_world
                        )
                    )
                    prediction_rows.append(item)
    return prediction_rows, omega_rows, eligible_anchors


def describe(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    result: dict[str, float | int] = {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "minimum": float(np.min(array)),
    }
    for quantile in QUANTILES:
        result[f"p{int(round(100 * quantile)):02d}"] = float(
            np.quantile(array, quantile)
        )
    result["maximum"] = float(np.max(array))
    return result


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = {
        "cross_depth_error": "error_cross_depth_m",
        "absolute_depth_error": "error_depth_m",
        "3d_error": "error_3d_m",
        "absolute_tracker_y_error": "error_tracker_y_m",
        "absolute_tracker_z_error": "error_tracker_z_m",
    }
    for (domain, method, horizon), group in predictions.groupby(
        ["input_domain", "method", "horizon_s"], sort=True
    ):
        for metric, column in metrics.items():
            values = np.abs(group[column].to_numpy()) * 1000.0
            rows.append(
                {
                    "input_domain": domain,
                    "method": method,
                    "horizon_s": horizon,
                    "metric": metric,
                    "unit": "mm",
                    "within_55mm_fraction": float(np.mean(values <= 55.0)),
                    **describe(values),
                }
            )
    return pd.DataFrame(rows)


def paired_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    key = ["timestamp_ns", "segment_index", "method", "horizon_s", "primary_slot"]
    baseline = predictions[predictions.input_domain == "current_refined"][
        key + ["error_cross_depth_m"]
    ].rename(columns={"error_cross_depth_m": "baseline_error_m"})
    rows = []
    for domain in ("raw", "network_segment_oof", "network_session_oof", "exact"):
        merged = predictions[predictions.input_domain == domain].merge(
            baseline, on=key, how="inner", validate="one_to_one"
        )
        for (method, horizon), group in merged.groupby(["method", "horizon_s"], sort=True):
            delta = (
                group["error_cross_depth_m"].to_numpy()
                - group["baseline_error_m"].to_numpy()
            ) * 1000.0
            rows.append(
                {
                    "input_domain": domain,
                    "baseline": "current_refined",
                    "method": method,
                    "horizon_s": horizon,
                    "improved_fraction": float(np.mean(delta < 0)),
                    "worsened_fraction": float(np.mean(delta > 0)),
                    **describe(delta),
                }
            )
    return pd.DataFrame(rows)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(np.asarray(values, dtype=np.float64))
    return values, np.arange(1, len(values) + 1) / len(values)


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_final_ecdf(predictions: pd.DataFrame, output: Path) -> None:
    selected = predictions[predictions.method == "los_memory31"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, horizon in zip(axes, HORIZONS_S):
        for domain in DOMAINS:
            values = selected.loc[
                (selected.input_domain == domain) & (selected.horizon_s == horizon),
                "error_cross_depth_m",
            ].to_numpy() * 1000.0
            x, y = ecdf(values)
            ax.plot(
                x, y, color=COLORS[domain], linestyle=LINESTYLES[domain],
                linewidth=1.7, label=f"{domain} (n={len(values)})",
            )
        ax.axvline(55.0, color="#777777", linestyle="--", linewidth=1.0)
        ax.set_title(f"{int(horizon * 1000)} ms")
        ax.set_xlabel("Future cross-depth error (mm)")
        ax.set_ylabel("Empirical cumulative fraction")
        ax.set_ylim(0, 1.005)
        ax.grid(True, alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if horizon == HORIZONS_S[0]:
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Frozen 400 ms LOS combined-motion expert: full OOF distributions")
    fig.tight_layout()
    save_figure(fig, output, "combined_prediction_cross_depth_ecdf")


def plot_quantiles(summary: pd.DataFrame, output: Path) -> None:
    selected = summary[
        (summary.method == "los_memory31")
        & (summary.metric == "cross_depth_error")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, quantile in zip(axes, ("p50", "p90", "p95")):
        for domain in DOMAINS:
            rows = selected[selected.input_domain == domain].sort_values("horizon_s")
            ax.plot(
                rows.horizon_s * 1000.0, rows[quantile], marker="o",
                color=COLORS[domain], linestyle=LINESTYLES[domain], label=domain,
            )
        ax.axhline(55.0, color="#777777", linestyle="--", linewidth=1.0)
        ax.set_title(quantile.upper())
        ax.set_xlabel("Prediction horizon (ms)")
        ax.set_ylabel("Future cross-depth error (mm)")
        ax.grid(True, alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if quantile == "p50":
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Center and tail of future error (not only P95)")
    fig.tight_layout()
    save_figure(fig, output, "combined_prediction_quantiles")


def report(summary: pd.DataFrame, paired: pd.DataFrame, audit: dict[str, Any]) -> str:
    selected = summary[
        (summary.method == "los_memory31")
        & (summary.metric == "cross_depth_error")
    ]
    lines = [
        "# Corner repair to combined-motion prediction", "",
        "This evaluates only the frozen 400 ms local LOS-aware rigid expert on six independent approximately 2.2 s configuration segments. The 4 s reversal-phase expert is not evaluable from this corner atlas and is not silently approximated.", "",
        "## Future cross-depth distribution (mm)", "",
        "| input | horizon | n | mean | P25 | P50 | P75 | P90 | P95 | P99 | max | <=55 mm |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selected.sort_values(["input_domain", "horizon_s"]).itertuples(index=False):
        lines.append(
            f"| {row.input_domain} | {int(row.horizon_s * 1000)} ms | {row.count} | "
            f"{row.mean:.2f} | {row.p25:.2f} | {row.p50:.2f} | {row.p75:.2f} | "
            f"{row.p90:.2f} | {row.p95:.2f} | {row.p99:.2f} | {row.maximum:.2f} | "
            f"{row.within_55mm_fraction:.3f} |"
        )
    lines.extend(["", "## Paired against current refined PnP", "",
                  "Negative delta means lower future cross-depth error on the same anchor.", "",
                  "| input | horizon | improved | worsened | mean delta (mm) | P50 delta | P90 delta | P95 delta |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    pair = paired[paired.method == "los_memory31"]
    for row in pair.sort_values(["input_domain", "horizon_s"]).itertuples(index=False):
        lines.append(
            f"| {row.input_domain} | {int(row.horizon_s * 1000)} ms | "
            f"{row.improved_fraction:.3f} | {row.worsened_fraction:.3f} | "
            f"{row.mean:.2f} | {row.p50:.2f} | {row.p90:.2f} | {row.p95:.2f} |"
        )
    lines.extend([
        "", "## Audit and boundary", "",
        f"- Current-refined camera-to-tracker parity maximum: `{audit['current_refined_tracker_parity_max_m']:.3e} m`.",
        f"- Exact-corner world-position P95/maximum: `{audit['exact_world_error_p95_m']:.3e}/{audit['exact_world_error_max_m']:.3e} m`.",
        f"- Eligible causal anchors: `{audit['eligible_anchors']}`; maximum available omega-memory count: `{audit['maximum_memory_count']}` of the configured 31.",
        "- All network positions are out-of-fold. Motion mode, truth omega and future truth are never corner-network inputs; truth omega appears only in the explicit oracle predictor arm.",
        "- Identity/slot is an offline truth association handle. This experiment is not deployable and is not a hit-probability test.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    base = load_module(
        "combined_factorization_corner_repair",
        repo / "scripts" / "evaluate-combined-motion-factorization.py",
    )
    los = load_module(
        "combined_los_corner_repair",
        repo / "scripts" / "evaluate-combined-motion-los-fit.py",
    )

    atlas = pd.read_csv(args.atlas)
    ranges = segment_ranges(atlas)
    base_frames, frame_lookup = load_truth_frames(base, args.truth, ranges)
    observation_lookup = load_observations(args.observations)
    pnp = pd.read_csv(args.pnp_rows)
    pnp = pnp[pnp.session_id.str.contains("combined")].copy()
    corners = pd.read_csv(
        args.corner_rows,
        usecols=["scheme", "row_index", "session_id", "truth_slot"],
    )
    slots = corners[corners.session_id.str.contains("combined")][
        ["row_index", "truth_slot"]
    ].drop_duplicates()
    if slots.duplicated("row_index").any():
        raise ValueError("row_index maps to multiple truth slots")
    pnp = pnp.merge(slots, on="row_index", how="left", validate="many_to_one")
    if pnp.truth_slot.isna().any():
        raise ValueError("PnP rows are missing truth-slot provenance")

    frames_by_domain = {}
    domain_audits = {}
    for domain in DOMAINS:
        rows = select_domain_rows(pnp, domain)
        if len(rows) != 2199:
            raise ValueError(f"{domain} expected 2199 OOF detections, got {len(rows)}")
        frames, domain_audit = populate_domain(
            base_frames, frame_lookup, observation_lookup, rows
        )
        frames_by_domain[domain] = frames
        domain_audits[domain] = domain_audit
    masks = [
        np.stack([frame.observed_mask for frame in frames])
        for frames in frames_by_domain.values()
    ]
    if any(not np.array_equal(masks[0], mask) for mask in masks[1:]):
        raise AssertionError("OOF domains do not share the same availability mask")
    parity = domain_audits["current_refined"]["current_refined_tracker_parity_max_m"]
    if not parity < 1.0e-6:
        raise AssertionError(f"camera-to-tracker parity failed: {parity} m")

    prediction_rows, omega_rows, eligible_anchors = evaluate(base, los, frames_by_domain)
    predictions = pd.DataFrame(prediction_rows)
    omega = pd.DataFrame(omega_rows)
    summary = summarize(predictions)
    paired = paired_summary(predictions)
    maximum_memory = int(omega.memory_count.max())
    audit = {
        "truth_frames": len(base_frames),
        "eligible_anchors": eligible_anchors,
        "maximum_memory_count": maximum_memory,
        "current_refined_tracker_parity_max_m": parity,
        "exact_world_error_p95_m": domain_audits["exact"]["exact_world_error_p95_m"],
        "exact_world_error_max_m": domain_audits["exact"]["exact_world_error_max_m"],
        "domain_audits": domain_audits,
    }

    write_csv_gz(args.output / "prediction_rows.csv.gz", prediction_rows)
    write_csv_gz(args.output / "omega_rows.csv.gz", omega_rows)
    summary.to_csv(args.output / "summary.csv", index=False)
    paired.to_csv(args.output / "paired_vs_current_refined.csv", index=False)
    atomic_json(args.output / "audit.json", audit)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9, "pdf.fonttype": 42,
    })
    plot_final_ecdf(predictions, args.output)
    plot_quantiles(summary, args.output)
    (args.output / "report.md").write_text(
        report(summary, paired, audit), encoding="utf-8"
    )
    manifest = {
        "schema_version": "corner-repair-combined-local-prediction-v1",
        "status": "complete",
        "deployable": False,
        "scope": "frozen 400 ms local LOS expert only; six independent short segments; no 4 s reversal-phase claim",
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in {
                "pnp_rows": args.pnp_rows,
                "corner_rows": args.corner_rows,
                "atlas": args.atlas,
                "observations": args.observations,
                "truth": args.truth,
            }.items()
        },
        "source": {
            "script": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
            "git": git_state(repo),
        },
        "contract": {
            "history_s": HISTORY_S,
            "horizons_s": HORIZONS_S,
            "anchor_stride": ANCHOR_STRIDE,
            "depth_weight": DEPTH_WEIGHT,
            "huber_delta_m": HUBER_DELTA_M,
            "omega_memory_count": OMEGA_MEMORY_COUNT,
            "omega_grid_step": OMEGA_GRID_STEP,
            "max_omega": MAX_OMEGA,
            "future_truth_used_as_input": False,
            "truth_slot_identity": "offline analysis only",
        },
        "rows": {
            "prediction": len(prediction_rows),
            "omega": len(omega_rows),
        },
        "audit": audit,
        "artifacts": sorted(path.name for path in args.output.iterdir()),
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
