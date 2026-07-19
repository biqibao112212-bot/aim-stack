#!/usr/bin/env python3
"""Quantify legacy versus diagnostic joint PnP on cyclic target-3 observations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


SPIN_DEG_S = 30.0
METHODS = ("legacy", "corrected", "joint_refined")
METHOD_LABELS = {
    "legacy": "legacy fixed-tvec yaw grid",
    "corrected": "production chassis-fixed +15 deg",
    "joint_refined": "joint yaw+tvec (refined corners)",
    "joint_raw": "joint yaw+tvec (raw corners)",
}
ID_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def percentile(values: List[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), quantile))


def stats(values: List[float]) -> Dict[str, object]:
    return {
        "count": len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else None,
    }


def wrap180(value: float) -> float:
    return (value + 90.0) % 180.0 - 90.0


def selected(candidates: object) -> Optional[dict]:
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("selected"):
            return candidate
    return candidates[0] if candidates and isinstance(candidates[0], dict) else None


def load_run(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            timestamp = record.get("source_capture_timestamp_ns")
            if not finite(timestamp):
                continue
            for armor in record.get("observations") or []:
                if not isinstance(armor, dict):
                    continue
                if not armor.get("target_candidate") or armor.get("number") != 3:
                    continue
                plate_id = armor.get("canonical_plate_id")
                ab = armor.get("pnp_ab")
                if not finite(plate_id) or not isinstance(ab, dict):
                    continue
                legacy = ab.get("legacy") or {}
                legacy_yaw = legacy.get(
                    "yaw_absolute_deg", armor.get("armor_yaw_absolute_deg")
                )
                if not finite(legacy_yaw):
                    continue
                refined_group = ab.get("joint_refined") or {}
                raw_group = ab.get("joint_raw") or {}
                refined = selected(refined_group.get("candidates"))
                raw = selected(raw_group.get("candidates"))
                if refined is None or raw is None:
                    continue
                corrected_yaw = ab.get("corrected_chassis_yaw_deg")
                if not finite(corrected_yaw):
                    corrected_yaw = armor.get("armor_yaw_deg")
                if not finite(corrected_yaw):
                    continue
                row = {
                    "seq": record.get("source_image_seq"),
                    "t": float(timestamp) * 1e-9,
                    "id": int(plate_id),
                    "legacy": float(legacy_yaw),
                    "corrected": float(corrected_yaw),
                    "joint_refined": float(refined["yaw_deg"]),
                    "joint_raw": float(raw["yaw_deg"]),
                    "legacy_reprojection": legacy.get("reprojection_rms_px"),
                    "corrected_reprojection": refined_group.get(
                        "constrained_reprojection_rms_px"
                    ),
                    "joint_refined_reprojection": refined.get("reprojection_rms_px"),
                    "joint_raw_reprojection": raw.get("reprojection_rms_px"),
                    "joint_refined_sensitivity": refined.get("yaw_sensitivity_deg_per_px"),
                    "joint_raw_sensitivity": raw.get("yaw_sensitivity_deg_per_px"),
                    "joint_refined_sensitivity_valid": bool(refined.get("yaw_sensitivity_valid")),
                    "joint_raw_sensitivity_valid": bool(raw.get("yaw_sensitivity_valid")),
                    "joint_refined_condition": refined.get("translation_information_condition"),
                    "joint_raw_condition": raw.get("translation_information_condition"),
                    "joint_refined_converged": bool(refined.get("converged")),
                    "joint_raw_converged": bool(raw.get("converged")),
                    "joint_refined_improved": bool(refined.get("improved")),
                    "joint_raw_improved": bool(raw.get("improved")),
                    "joint_refined_bound": bool(refined.get("search_bound_hit")),
                    "joint_raw_bound": bool(raw.get("search_bound_hit")),
                    "joint_refined_iterations": refined.get("iterations"),
                    "joint_raw_iterations": raw.get("iterations"),
                    "joint_refined_solve_us": refined_group.get("solve_us"),
                    "joint_raw_solve_us": raw_group.get("solve_us"),
                }
                rows.append(row)
    rows.sort(key=lambda item: (float(item["t"]), int(item["id"])))
    return rows


def temporal_increment_errors(rows: List[dict], method: str) -> List[float]:
    errors: List[float] = []
    for plate_id in range(4):
        plate = [row for row in rows if int(row["id"]) == plate_id]
        plate.sort(key=lambda item: float(item["t"]))
        for left, right in zip(plate, plate[1:]):
            dt = float(right["t"]) - float(left["t"])
            if dt <= 0.0 or dt > 0.2:
                continue
            observed = wrap180(float(right[method]) - float(left[method]))
            expected = SPIN_DEG_S * dt
            errors.append(abs(wrap180(observed - expected)))
    return errors


def fixed_frequency_fit(rows: List[dict], method: str) -> Dict[str, object]:
    if not rows:
        return {"samples": 0}
    t0 = float(rows[0]["t"])
    t = np.asarray([float(row["t"]) - t0 for row in rows], dtype=float)
    y = np.asarray([float(row[method]) for row in rows], dtype=float)
    ids = np.asarray([int(row["id"]) for row in rows], dtype=int)
    phase = np.radians(SPIN_DEG_S * t)
    columns = [(ids == plate_id).astype(float) for plate_id in range(4)]
    columns.extend((np.sin(phase), np.cos(phase)))
    design = np.column_stack(columns)
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficients
    return {
        "samples": len(rows),
        "fixed_frequency_deg_s": SPIN_DEG_S,
        "amplitude_deg": float(math.hypot(coefficients[-2], coefficients[-1])),
        "rmse_deg": float(np.sqrt(np.mean(residual * residual))),
        "residual_abs_deg": stats([abs(float(value)) for value in residual]),
    }


def read_simulator_stats(run_path: Path) -> Optional[dict]:
    path = run_path.parent / "simulator.stats.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(label: str, path: Path, rows: List[dict]) -> Dict[str, object]:
    result: Dict[str, object] = {
        "label": label,
        "source": str(path),
        "observations": len(rows),
        "duration_s": float(rows[-1]["t"] - rows[0]["t"]) if rows else 0.0,
        "canonical_id_counts": {
            str(plate_id): sum(int(row["id"]) == plate_id for row in rows)
            for plate_id in range(4)
        },
        "methods": {},
    }
    for method in METHODS:
        reprojection = [
            float(row[method + "_reprojection"])
            for row in rows
            if finite(row.get(method + "_reprojection"))
        ]
        result["methods"][method] = {
            "yaw_deg": stats([float(row[method]) for row in rows]),
            "reprojection_rms_px": stats(reprojection),
            "adjacent_same_id_increment_abs_error_deg": stats(
                temporal_increment_errors(rows, method)
            ),
            "fixed_30_deg_s_sine_fit": fixed_frequency_fit(rows, method),
        }
    corrected_delta = [
        abs(wrap180(float(row["corrected"]) - float(row["legacy"])))
        for row in rows
    ]
    refined_delta = [
        abs(wrap180(float(row["joint_refined"]) - float(row["legacy"])))
        for row in rows
    ]
    raw_delta = [
        abs(wrap180(float(row["joint_raw"]) - float(row["legacy"])))
        for row in rows
    ]
    result["legacy_to_joint_abs_yaw_delta_deg"] = {
        "corrected_production": stats(corrected_delta),
        "refined": stats(refined_delta),
        "raw": stats(raw_delta),
    }
    for method in ("joint_refined", "joint_raw"):
        valid_sensitivity = [
            float(row[method + "_sensitivity"])
            for row in rows
            if row[method + "_sensitivity_valid"]
            and finite(row.get(method + "_sensitivity"))
        ]
        result[method + "_diagnostics"] = {
            "candidate_count": len(rows),
            "converged_count": sum(bool(row[method + "_converged"]) for row in rows),
            "improved_count": sum(bool(row[method + "_improved"]) for row in rows),
            "bound_hit_count": sum(bool(row[method + "_bound"]) for row in rows),
            "sensitivity_valid_count": len(valid_sensitivity),
            "yaw_sensitivity_deg_per_px": stats(valid_sensitivity),
            "translation_information_condition": stats(
                [
                    float(row[method + "_condition"])
                    for row in rows
                    if finite(row.get(method + "_condition"))
                ]
            ),
            "iterations": stats(
                [
                    float(row[method + "_iterations"])
                    for row in rows
                    if finite(row.get(method + "_iterations"))
                ]
            ),
            "solve_us": stats(
                [
                    float(row[method + "_solve_us"])
                    for row in rows
                    if finite(row.get(method + "_solve_us"))
                ]
            ),
        }
    simulator = read_simulator_stats(path)
    if simulator is not None:
        result["simulator"] = {
            key: simulator.get(key)
            for key in (
                "elapsed_s",
                "main_update_hz",
                "capture_copy_submit_hz",
                "talos_tcp_image_sent_hz",
                "capture_processing_complete_total",
                "capture_queue_drop_total",
                "capture_fast_map_error_total",
                "talos_tcp_image_connect_total",
                "talos_tcp_image_sent_total",
                "talos_tcp_image_bind_fail_total",
            )
        }
    return result


def draw_yaw(runs: List[Tuple[str, List[dict]]], output: Path) -> None:
    figure, axes = plt.subplots(len(runs), len(METHODS), figsize=(19, 4.5 * len(runs)), squeeze=False)
    for row_index, (label, rows) in enumerate(runs):
        if not rows:
            continue
        t0 = float(rows[0]["t"])
        for column, method in enumerate(METHODS):
            axis = axes[row_index][column]
            for plate_id in range(4):
                selected_rows = [row for row in rows if int(row["id"]) == plate_id]
                # Connect high-rate observations, but leave detector gaps
                # disconnected so the figure does not invent motion.
                segments: List[List[dict]] = []
                segment: List[dict] = []
                for observation in selected_rows:
                    if segment and float(observation["t"]) - float(segment[-1]["t"]) > 0.2:
                        segments.append(segment)
                        segment = []
                    segment.append(observation)
                if segment:
                    segments.append(segment)
                for segment_index, segment in enumerate(segments):
                    axis.plot(
                        [float(item["t"]) - t0 for item in segment],
                        [float(item[method]) for item in segment],
                        color=ID_COLORS[plate_id],
                        linewidth=0.75,
                        marker=".",
                        markersize=1.8,
                        alpha=0.78,
                        label="plate %d" % plate_id if segment_index == 0 else None,
                    )
            increment = stats(temporal_increment_errors(rows, method))
            axis.set_title(
                "%s — %s\nN=%d, adjacent error p50=%.2f°, p95=%.2f°"
                % (
                    label,
                    METHOD_LABELS[method],
                    len(rows),
                    increment["p50"] or 0.0,
                    increment["p95"] or 0.0,
                )
            )
            axis.set_ylim(-90, 90)
            axis.grid(True, alpha=0.22)
            if column == 0:
                axis.set_ylabel("yaw (deg), connected observations")
            if row_index == len(runs) - 1:
                axis.set_xlabel("time since first observation (s)")
            if row_index == 0 and column == 2:
                axis.legend(loc="upper right", markerscale=2, fontsize=8)
    figure.suptitle(
        "Target 3, 30 deg/s spin, zero linear speed: continuous per-plate yaw curves",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(str(output), dpi=180)


def draw_metrics(runs: List[Tuple[str, List[dict]]], output: Path) -> None:
    labels = [label for label, _ in runs]
    positions = np.arange(len(runs), dtype=float)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    width = 0.24
    for index, method in enumerate(METHODS):
        residual_sets = [
            [
                float(row[method + "_reprojection"])
                for row in rows
                if finite(row.get(method + "_reprojection"))
            ]
            for _, rows in runs
        ]
        axes[0][0].boxplot(
            residual_sets,
            positions=positions + (index - 1) * width,
            widths=width * 0.85,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": ("#999999", "#4c78a8", "#f58518")[index], "alpha": 0.65},
            medianprops={"color": "black"},
        )
    axes[0][0].set_title("Constrained-model reprojection RMS")
    axes[0][0].set_ylabel("pixels")
    axes[0][0].grid(True, axis="y", alpha=0.2)

    for index, method in enumerate(("joint_refined", "joint_raw")):
        sensitivity_sets = [
            [
                float(row[method + "_sensitivity"])
                for row in rows
                if row[method + "_sensitivity_valid"]
                and finite(row.get(method + "_sensitivity"))
            ]
            for _, rows in runs
        ]
        axes[0][1].boxplot(
            sensitivity_sets,
            positions=positions + (index - 0.5) * 0.3,
            widths=0.25,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": ("#4c78a8", "#f58518")[index], "alpha": 0.65},
            medianprops={"color": "black"},
        )
    axes[0][1].set_title("Local yaw sensitivity after marginalizing translation")
    axes[0][1].set_ylabel("degrees per 1-pixel residual perturbation")
    axes[0][1].grid(True, axis="y", alpha=0.2)

    delta_sets = [
        [abs(wrap180(float(row["corrected"]) - float(row["legacy"]))) for row in rows]
        for _, rows in runs
    ]
    axes[1][0].boxplot(delta_sets, positions=positions, widths=0.45, showfliers=False)
    axes[1][0].set_title("Production chassis-fixed yaw change from legacy")
    axes[1][0].set_ylabel("absolute wrapped difference (deg)")
    axes[1][0].grid(True, axis="y", alpha=0.2)

    solve_sets = [
        [float(row["joint_refined_solve_us"]) / 1000.0 for row in rows]
        for _, rows in runs
    ]
    axes[1][1].boxplot(solve_sets, positions=positions, widths=0.45, showfliers=False)
    axes[1][1].set_title("Joint-refined solve time per armor")
    axes[1][1].set_ylabel("milliseconds")
    axes[1][1].grid(True, axis="y", alpha=0.2)

    for axis_row in axes:
        for axis in axis_row:
            axis.set_xticks(positions)
            axis.set_xticklabels(labels)
    axes[0][0].legend(
        [plt.Line2D([0], [0], color=color, linewidth=8, alpha=0.65) for color in ("#999999", "#4c78a8", "#f58518")],
        [METHOD_LABELS[method] for method in METHODS],
        fontsize=8,
    )
    axes[0][1].legend(
        [plt.Line2D([0], [0], color=color, linewidth=8, alpha=0.65) for color in ("#4c78a8", "#f58518")],
        [METHOD_LABELS["joint_refined"], METHOD_LABELS["joint_raw"]],
        fontsize=8,
    )
    figure.suptitle(
        "PnP A/B diagnostics: algorithm gain versus distance-limited conditioning",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    figure.savefig(str(output), dpi=180)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=JSONL")
    parser.add_argument("--yaw-output", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    loaded: List[Tuple[str, Path, List[dict]]] = []
    summaries: Dict[str, object] = {}
    for raw in args.run:
        label, separator, path_text = raw.partition("=")
        if not separator:
            parser.error("invalid --run: %r" % raw)
        path = Path(path_text)
        rows = load_run(path)
        loaded.append((label, path, rows))
        summaries[label] = summarize(label, path, rows)
    draw_yaw([(label, rows) for label, _, rows in loaded], Path(args.yaw_output))
    draw_metrics([(label, rows) for label, _, rows in loaded], Path(args.metrics_output))
    Path(args.summary_output).write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
