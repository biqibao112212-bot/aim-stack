#!/usr/bin/env python3
"""Evaluate causal future prediction using only historical detector observations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wrap_rad(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def error_deg(actual_u: float, actual_v: float, predicted_u: float, predicted_v: float) -> float:
    return math.degrees(math.hypot(wrap_rad(actual_u - predicted_u), actual_v - predicted_v))


def make_truth_series(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = sorted(rows, key=lambda row: row["t_s"])
    times = np.asarray([float(row["t_s"]) for row in ordered], dtype=float)
    u = np.unwrap(np.radians(np.asarray([float(row["u_deg"]) for row in ordered], dtype=float)))
    v = np.radians(np.asarray([float(row["v_deg"]) for row in ordered], dtype=float))
    return times, u, v


def make_observation_examples(
    observations: list[dict],
    truth_series: tuple[np.ndarray, np.ndarray, np.ndarray],
    horizon_s: float,
    history_size: int = 5,
) -> list[dict]:
    if len(observations) < history_size:
        return []
    ordered = sorted(observations, key=lambda row: row["t_s"])
    truth_times, truth_u, truth_v = truth_series
    obs_times = np.asarray([float(row["t_s"]) for row in ordered], dtype=float)
    obs_u = np.unwrap(np.radians(np.asarray([float(row["u_deg"]) for row in ordered], dtype=float)))
    obs_v = np.radians(np.asarray([float(row["v_deg"]) for row in ordered], dtype=float))
    examples: list[dict] = []
    for index in range(history_size - 1, len(ordered)):
        current_time = obs_times[index]
        future_time = current_time + horizon_s
        if future_time > truth_times[-1]:
            break
        start = index - history_size + 1
        local_t = obs_times[start : index + 1] - current_time
        design = np.column_stack([local_t, np.ones_like(local_t)])
        u_velocity, u_intercept = np.linalg.lstsq(design, obs_u[start : index + 1], rcond=None)[0]
        v_velocity, v_intercept = np.linalg.lstsq(design, obs_v[start : index + 1], rcond=None)[0]
        actual_u = float(np.interp(future_time, truth_times, truth_u))
        actual_v = float(np.interp(future_time, truth_times, truth_v))
        current_u = float(obs_u[index])
        current_v = float(obs_v[index])
        baseline_u = current_u + float(u_velocity) * horizon_s
        baseline_v = current_v + float(v_velocity) * horizon_s
        examples.append(
            {
                "features": np.asarray([current_u, current_v, u_velocity, v_velocity, horizon_s], dtype=float),
                "actual": np.asarray([actual_u, actual_v], dtype=float),
                "baseline_error_deg": error_deg(actual_u, actual_v, baseline_u, baseline_v),
                "run": ordered[index]["run"],
                "scale": float(ordered[index]["scale"]),
                "distance_m": float(ordered[index]["distance_m"]),
                "slot": int(ordered[index]["slot"]),
            }
        )
    return examples


def fit_linear(train: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.vstack([example["features"] for example in train])
    y = np.vstack([example["actual"] for example in train])
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-9] = 1.0
    normalized = (x - mean) / scale
    design = np.column_stack([normalized, np.ones(len(normalized))])
    weights = np.linalg.lstsq(design, y, rcond=None)[0]
    return weights, mean, scale


def predict_linear(model: tuple[np.ndarray, np.ndarray, np.ndarray], examples: list[dict]) -> np.ndarray:
    weights, mean, scale = model
    x = np.vstack([example["features"] for example in examples])
    normalized = (x - mean) / scale
    design = np.column_stack([normalized, np.ones(len(normalized))])
    return design @ weights


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(errors: list[float], prefix: str) -> dict:
    values = np.asarray(errors, dtype=float)
    return {
        f"{prefix}_samples": int(values.size),
        f"{prefix}_mean_error_deg": float(np.mean(values)),
        f"{prefix}_median_error_deg": float(np.median(values)),
        f"{prefix}_p95_error_deg": float(np.percentile(values, 95)),
    }


def plot_error_heatmaps(
    output: Path,
    rows_by_kind: dict[str, list[dict]],
    horizons: tuple[float, ...],
    distances: list[float],
    scales: list[float],
) -> None:
    figure, axes = plt.subplots(2, len(horizons), figsize=(15, 7), squeeze=False)
    for row_index, kind in enumerate(("baseline", "learned")):
        rows = rows_by_kind[kind]
        for col_index, horizon_s in enumerate(horizons):
            axis = axes[row_index][col_index]
            condition_values: dict[tuple[float, float], list[float]] = defaultdict(list)
            for item in rows:
                if abs(float(item["horizon_s"]) - horizon_s) > 1e-9:
                    continue
                condition_values[(float(item["scale"]), float(item["distance_m"]))].append(
                    float(item[f"{kind}_p95_error_deg"])
                )
            values = np.full((len(scales), len(distances)), np.nan, dtype=float)
            for (scale, distance), errors in condition_values.items():
                scale_index = scales.index(scale)
                distance_index = distances.index(distance)
                values[scale_index, distance_index] = float(np.mean(errors))
            image = axis.imshow(values, cmap="viridis", aspect="auto", vmin=0.0)
            axis.set_xticks(range(len(distances)))
            axis.set_xticklabels([f"{distance:g}" for distance in distances])
            axis.set_yticks(range(len(scales)))
            axis.set_yticklabels([f"{scale:g}" for scale in scales])
            axis.set_xlabel("distance (m)")
            axis.set_ylabel("radius scale")
            axis.set_title(f"{kind} {int(round(horizon_s * 1000))} ms mean-slot P95 (deg)")
            for y_index in range(len(scales)):
                for x_index in range(len(distances)):
                    value = values[y_index, x_index]
                    if np.isfinite(value):
                        text_color = "white" if value > np.nanmax(values) * 0.55 else "black"
                        axis.text(x_index, y_index, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Held-out future prediction error; heatmaps average the four physical slots", y=1.01)
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(output / "observation_only_prediction_error.png", dpi=180)
    plt.close(figure)


def plot_slot_summary(output: Path, rows: list[dict], horizons: tuple[float, ...]) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for slot, color in enumerate(colors):
        slot_rows = [row for row in rows if int(row["slot"]) == slot]
        x = [int(round(float(row["horizon_s"]) * 1000)) for row in slot_rows]
        y = [float(row["learned_p95_error_deg"]) for row in slot_rows]
        axis.plot(x, y, marker="o", color=color, label=f"slot {slot}")
    axis.set_xlabel("prediction horizon (ms)")
    axis.set_ylabel("pooled learned P95 angular error (deg)")
    axis.set_title("Held-out observation-only prediction by physical slot")
    axis.set_xticks([int(round(horizon * 1000)) for horizon in horizons])
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "observation_only_slot_error.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.resolve()
    output = (args.output or analysis_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    truth = read_jsonl(analysis_dir / "truth_points.jsonl")
    observed = read_jsonl(analysis_dir / "observed_points.jsonl")
    truth_by_run_slot: dict[tuple[str, int], list[dict]] = defaultdict(list)
    observed_by_run_slot: dict[tuple[str, int], list[dict]] = defaultdict(list)
    condition_runs: dict[tuple[float, float], set[str]] = defaultdict(set)
    run_repeat: dict[str, int] = {}
    for row in truth:
        truth_by_run_slot[(row["run"], int(row["slot"]))].append(row)
        condition_runs[(float(row["scale"]), float(row["distance_m"]))].add(row["run"])
        run_repeat[row["run"]] = int(row["repeat"])
    for row in observed:
        observed_by_run_slot[(row["run"], int(row["slot"]))].append(row)

    baseline_rows: list[dict] = []
    learned_rows: list[dict] = []
    aggregate_errors: dict[tuple[str, float], list[float]] = defaultdict(list)
    slot_errors: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    horizons = (0.05, 0.10, 0.20)
    for condition, runs_set in sorted(condition_runs.items()):
        scale, distance_m = condition
        runs = sorted(runs_set, key=lambda run: (run_repeat.get(run, 10**9), run))
        test_run = runs[-1]
        train_runs = [run for run in runs if run != test_run]
        for horizon_s in horizons:
            for slot in range(4):
                train_examples: list[dict] = []
                test_examples: list[dict] = []
                for run in runs:
                    truth_series = make_truth_series(truth_by_run_slot[(run, slot)])
                    examples = make_observation_examples(
                        observed_by_run_slot[(run, slot)], truth_series, horizon_s
                    )
                    if run == test_run:
                        test_examples.extend(examples)
                    else:
                        train_examples.extend(examples)
                if not test_examples or not train_examples:
                    continue
                baseline_errors = [example["baseline_error_deg"] for example in test_examples]
                baseline_row = {
                    "scale": scale,
                    "distance_m": distance_m,
                    "slot": slot,
                    "horizon_s": horizon_s,
                    "train_runs": ",".join(train_runs),
                    "test_run": test_run,
                }
                baseline_row.update(summarize(baseline_errors, "baseline"))
                baseline_rows.append(baseline_row)
                aggregate_errors[("baseline", horizon_s)].extend(baseline_errors)
                slot_errors[("baseline", slot, horizon_s)].extend(baseline_errors)

                model = fit_linear(train_examples)
                predictions = predict_linear(model, test_examples)
                learned_errors = [
                    error_deg(float(example["actual"][0]), float(example["actual"][1]), float(prediction[0]), float(prediction[1]))
                    for example, prediction in zip(test_examples, predictions)
                ]
                learned_row = {
                    "scale": scale,
                    "distance_m": distance_m,
                    "slot": slot,
                    "horizon_s": horizon_s,
                    "train_runs": ",".join(train_runs),
                    "test_run": test_run,
                }
                learned_row.update(summarize(learned_errors, "learned"))
                learned_rows.append(learned_row)
                aggregate_errors[("learned", horizon_s)].extend(learned_errors)
                slot_errors[("learned", slot, horizon_s)].extend(learned_errors)

    write_csv(output / "observation_only_baseline.csv", baseline_rows)
    write_csv(output / "observation_only_learned_linear.csv", learned_rows)
    slot_summary_rows: list[dict] = []
    for slot in range(4):
        for horizon_s in horizons:
            row = {"slot": slot, "horizon_s": horizon_s}
            row.update(summarize(slot_errors[("baseline", slot, horizon_s)], "baseline"))
            row.update(summarize(slot_errors[("learned", slot, horizon_s)], "learned"))
            slot_summary_rows.append(row)
    write_csv(output / "observation_only_slot_summary.csv", slot_summary_rows)
    plot_error_heatmaps(
        output,
        {"baseline": baseline_rows, "learned": learned_rows},
        horizons,
        sorted({float(condition[1]) for condition in condition_runs}),
        sorted({float(condition[0]) for condition in condition_runs}),
    )
    plot_slot_summary(output, slot_summary_rows, horizons)
    aggregate_summary = {}
    for horizon_s in horizons:
        aggregate_summary[str(horizon_s)] = {}
        for kind in ("baseline", "learned"):
            aggregate_summary[str(horizon_s)][kind] = summarize(
                aggregate_errors[(kind, horizon_s)], kind
            )
    summary = {
        "schema_version": 3,
        "kind": "observation_only_future_prediction",
        "input_contract": "historical detector camera angles and finite-difference velocities only; truth is used only for supervised training labels and held-out evaluation",
        "identity_contract": "relative_slot labels are assigned offline for evaluation; deployment identity association is not evaluated here",
        "model_contract": "one standardized linear model is trained independently for each physical relative_slot, condition, and horizon; all available historical repeats except the numerically highest repeat are used for training and the highest repeat is held out",
        "horizons_s": list(horizons),
        "conditions": len(condition_runs),
        "run_counts_by_condition": {
            f"{scale:g}@{distance:g}m": len(runs)
            for (scale, distance), runs in sorted(condition_runs.items())
        },
        "baseline_rows": len(baseline_rows),
        "learned_rows": len(learned_rows),
        "aggregate_held_out_metrics": aggregate_summary,
        "artifacts": [
            "observation_only_baseline.csv",
            "observation_only_learned_linear.csv",
            "observation_only_slot_summary.csv",
            "observation_only_prediction_error.png",
            "observation_only_slot_error.png",
        ],
    }
    (output / "observation_prediction_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
