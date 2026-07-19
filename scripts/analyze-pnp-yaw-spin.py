#!/usr/bin/env python3
"""Quantify target-3 PnP yaw during constant-speed spin runs.

The analysis keeps the raw observations visible.  It separates valid active
tracker samples from lost/reacquired samples, marks ID changes and jump flags,
and fits only a descriptive fixed-frequency sinusoid with one constant offset
per tracked ID.  The fit is diagnostic evidence, not a pass/fail gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


OMEGA_DEG_S = 30.0
ACTIVE_STATES = {"tracking", "detecting"}
ID_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            tracker = record.get("tracker") or {}
            armor = tracker.get("tracked_armor")
            if not isinstance(armor, dict):
                continue
            timestamp = record.get("source_capture_timestamp_ns")
            yaw = armor.get("armor_yaw_deg")
            tracked_id = tracker.get("tracked_id")
            if not finite(timestamp) or not finite(yaw) or not finite(tracked_id):
                continue
            selected_reprojection = None
            for candidate in armor.get("pnp_candidates") or []:
                if isinstance(candidate, dict) and candidate.get("selected") and finite(candidate.get("reprojection_error_px")):
                    selected_reprojection = float(candidate["reprojection_error_px"])
                    break
            distance_m = armor.get("distance_mm")
            current_match_ids = tracker.get("current_match_ids")
            live = (
                bool(tracker.get("detected", False))
                and
                isinstance(current_match_ids, list)
                and bool(current_match_ids)
                and int(tracker.get("primary_observation_index", -1)) >= 0
            )
            rows.append(
                {
                    "t": float(timestamp) * 1e-9,
                    "yaw": float(yaw),
                    "id": int(tracked_id),
                    "number": int(armor.get("number", -1)),
                    "state": str(tracker.get("tracker_state", "")),
                    "jump": int(tracker.get("jump_flag", 0)),
                    "live": live,
                    "reprojection_error_px": selected_reprojection,
                    "distance_m": float(distance_m) / 1000.0 if finite(distance_m) else None,
                }
            )
    rows.sort(key=lambda row: float(row["t"]))
    return rows


def primary(row: dict[str, object]) -> bool:
    return bool(row["live"]) and str(row["state"]) in ACTIVE_STATES and int(row["number"]) == 3


def split_segments(rows: list[dict[str, object]], gap_s: float = 0.2) -> list[list[dict[str, object]]]:
    segments: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for row in rows:
        if not current:
            current = [row]
            continue
        same_id = int(row["id"]) == int(current[-1]["id"])
        close = float(row["t"]) - float(current[-1]["t"]) <= gap_s
        if same_id and close:
            current.append(row)
        else:
            segments.append(current)
            current = [row]
    if current:
        segments.append(current)
    return segments


def fit_fixed_frequency(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"samples": 0}
    t0 = float(rows[0]["t"])
    t = np.asarray([float(row["t"]) - t0 for row in rows], dtype=float)
    y = np.asarray([float(row["yaw"]) for row in rows], dtype=float)
    ids = np.asarray([int(row["id"]) for row in rows], dtype=int)
    omega = math.radians(OMEGA_DEG_S)
    columns = [np.ones(len(t)), np.sin(omega * t), np.cos(omega * t)]
    for tracked_id in (1, 2, 3):
        columns.append((ids == tracked_id).astype(float))
    design = np.column_stack(columns)
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ coefficients
    residual = y - predicted
    absolute = np.abs(residual)
    amplitude = math.hypot(float(coefficients[1]), float(coefficients[2]))
    phase_deg = math.degrees(math.atan2(float(coefficients[2]), float(coefficients[1])))
    return {
        "samples": int(len(rows)),
        "frequency_deg_s_fixed": OMEGA_DEG_S,
        "amplitude_deg": amplitude,
        "phase_deg": phase_deg,
        "offset_deg": float(coefficients[0]),
        "id_offsets_deg": {str(i): float(coefficients[i + 2]) for i in (1, 2, 3)},
        "rmse_deg": float(np.sqrt(np.mean(residual * residual))),
        "mae_deg": float(np.mean(absolute)),
        "residual_p50_deg": float(np.percentile(absolute, 50)),
        "residual_p90_deg": float(np.percentile(absolute, 90)),
        "residual_p95_deg": float(np.percentile(absolute, 95)),
        "residual_p99_deg": float(np.percentile(absolute, 99)),
        "residual_max_deg": float(np.max(absolute)),
        "coefficients": [float(value) for value in coefficients],
    }


def summarize(label: str, rows: list[dict[str, object]]) -> dict[str, object]:
    active = [row for row in rows if bool(row["live"]) and str(row["state"]) in ACTIVE_STATES]
    primary_rows = [row for row in active if int(row["number"]) == 3]
    segments = split_segments(primary_rows)
    segment_sizes = [len(segment) for segment in segments]
    segment_durations = [float(segment[-1]["t"]) - float(segment[0]["t"]) for segment in segments]
    id_changes = sum(
        int(left["id"]) != int(right["id"])
        for left, right in zip(primary_rows, primary_rows[1:])
        if float(right["t"]) - float(left["t"]) <= 0.2
    )
    transition_deltas: list[float] = []
    transition_intervals: list[float] = []
    transition_slot_diffs: list[int] = []
    for left, right in zip(primary_rows, primary_rows[1:]):
        if int(left["id"]) == int(right["id"]):
            continue
        if float(right["t"]) - float(left["t"]) > 0.2:
            continue
        delta = float(right["yaw"]) - float(left["yaw"])
        delta = (delta + 180.0) % 360.0 - 180.0
        transition_deltas.append(abs(delta))
        transition_intervals.append(float(right["t"]) - float(left["t"]))
        transition_slot_diffs.append((int(right["id"]) - int(left["id"])) % 4)
    reprojection_errors: list[float] = []
    primary_distances: list[float] = []
    for row in primary_rows:
        # The pipeline JSONL intentionally retains selected PnP-candidate
        # diagnostics, so detector/PnP conditioning can be reported beside
        # the yaw fit without reopening images.
        # These fields are attached by load_rows when available.
        if finite(row.get("reprojection_error_px")):
            reprojection_errors.append(float(row["reprojection_error_px"]))
        if finite(row.get("distance_m")):
            primary_distances.append(float(row["distance_m"]))

    def percentiles(values: list[float]) -> dict[str, object]:
        return {
            "count": len(values),
            "p50": float(np.percentile(values, 50)) if values else None,
            "p90": float(np.percentile(values, 90)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
            "max": float(max(values)) if values else None,
        }

    return {
        "label": label,
        "all_tracked_samples": len(rows),
        "active_samples": len(active),
        "primary_samples_state_active_number_3": len(primary_rows),
        "active_number_not_3": sum(int(row["number"]) != 3 for row in active),
        "active_jump_flag_samples": sum(int(row["jump"]) != 0 for row in active),
        "tracked_ids": sorted({int(row["id"]) for row in rows}),
        "active_id_counts": {
            str(tracked_id): sum(int(row["id"]) == tracked_id for row in active)
            for tracked_id in (0, 1, 2, 3)
        },
        "active_id_changes_within_gap": id_changes,
        "id_transition_abs_yaw_delta_deg": {
            "count": len(transition_deltas),
            "p50": float(np.percentile(transition_deltas, 50)) if transition_deltas else None,
            "p90": float(np.percentile(transition_deltas, 90)) if transition_deltas else None,
            "p95": float(np.percentile(transition_deltas, 95)) if transition_deltas else None,
        },
        "id_transition_interval_s": percentiles(transition_intervals),
        "id_transition_slot_diff_mod4": {
            str(diff): transition_slot_diffs.count(diff) for diff in range(4)
        },
        "primary_segment_count": len(segments),
        "primary_segment_samples_p50": float(np.percentile(segment_sizes, 50)) if segment_sizes else None,
        "primary_segment_samples_p90": float(np.percentile(segment_sizes, 90)) if segment_sizes else None,
        "primary_segment_duration_s_p50": float(np.percentile(segment_durations, 50)) if segment_durations else None,
        "primary_segment_duration_s_p90": float(np.percentile(segment_durations, 90)) if segment_durations else None,
        "primary_pnp_reprojection_error_px": percentiles(reprojection_errors),
        "primary_distance_m": percentiles(primary_distances),
        "primary_fit": fit_fixed_frequency(primary_rows),
    }


def draw(label: str, rows: list[dict[str, object]], axis_time, axis_phase, fit: dict[str, object]) -> None:
    if not rows:
        axis_time.text(0.5, 0.5, "no tracked_armor samples", ha="center", va="center")
        axis_phase.text(0.5, 0.5, "no tracked_armor samples", ha="center", va="center")
        return
    t0 = float(rows[0]["t"])
    active = [row for row in rows if bool(row["live"]) and str(row["state"]) in ACTIVE_STATES]
    for tracked_id in (0, 1, 2, 3):
        selected = [row for row in active if int(row["id"]) == tracked_id]
        if not selected:
            continue
        times = [float(row["t"]) - t0 for row in selected]
        yaws = [float(row["yaw"]) for row in selected]
        phases = [(OMEGA_DEG_S * time) % 360.0 for time in times]
        color = ID_COLORS[tracked_id]
        axis_time.scatter(times, yaws, s=5, alpha=0.45, color=color, label=f"ID {tracked_id}")
        axis_phase.scatter(phases, yaws, s=5, alpha=0.4, color=color, label=f"ID {tracked_id}")
    for left, right in zip(rows, rows[1:]):
        if int(left["id"]) != int(right["id"]):
            axis_time.axvline(float(right["t"]) - t0, color="#777777", linewidth=0.3, alpha=0.4)
    axis_time.set_title(f"{label}: raw active tracked yaw; ID changes marked")
    axis_time.set_ylabel("yaw (deg)")
    axis_time.grid(True, alpha=0.2)
    axis_time.legend(loc="upper right", ncol=2, fontsize=8)
    axis_phase.set_title(f"{label}: phase-folded yaw, phase speed fixed at {OMEGA_DEG_S:g} deg/s")
    axis_phase.set_xlabel("spin phase (deg)")
    axis_phase.set_ylabel("yaw (deg)")
    axis_phase.set_xlim(0, 360)
    axis_phase.grid(True, alpha=0.2)
    axis_phase.legend(loc="upper right", ncol=2, fontsize=8)
    if fit.get("samples"):
        phase = np.linspace(0.0, 360.0, 361)
        coefficients = np.asarray(fit["coefficients"], dtype=float)
        model = coefficients[0] + coefficients[1] * np.sin(np.radians(phase)) + coefficients[2] * np.cos(np.radians(phase))
        axis_phase.plot(phase, model, color="#111111", linewidth=1.2, label="fixed-frequency diagnostic fit")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    loaded: list[tuple[str, list[dict[str, object]]]] = []
    summaries: dict[str, object] = {}
    for raw in args.run:
        label, separator, path = raw.partition("=")
        if not separator or not label or not path:
            parser.error(f"invalid --run value: {raw!r}")
        rows = load_rows(Path(path))
        loaded.append((label, rows))
        summaries[label] = summarize(label, rows)

    figure, axes = plt.subplots(len(loaded), 2, figsize=(16, 4.8 * len(loaded)), squeeze=False)
    for index, (label, rows) in enumerate(loaded):
        draw(label, rows, axes[index][0], axes[index][1], summaries[label]["primary_fit"])
    axes[-1][0].set_xlabel("time since first tracked sample (s)")
    figure.suptitle("Target-3 slow-spin PnP yaw: segmented raw samples and phase fold")
    figure.tight_layout()
    figure.savefig(args.output, dpi=180)
    Path(args.summary_output).write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
