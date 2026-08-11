#!/usr/bin/env python3
"""Cross-fitted downstream PnP correction screen for combined motion.

This is an offline analysis.  Deployable arms use only fields present in the
observation stream and are fitted with whole-session holdouts.  Truth slot and
truth radial phase are explicit oracle probes.  Truth is never a runtime input
to a deployable arm.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODE = "linear_and_spin"
FOLDS = 5
RIDGE_LAMBDA = 1.0e-3
IRLS_ITERATIONS = 5
HUBER_DELTA_M = 0.10
DEPTH_WEIGHT = 0.10
GATES_M = (0.010, 0.025, 0.055, 0.100, 0.200)
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

MODEL_SPECS = (
    "affine",
    "yaw_harmonic",
    "yaw_slot_oracle",
    "truth_phase_oracle",
    "truth_phase_slot_oracle",
)
DEPLOYABLE_SPECS = ("affine", "yaw_harmonic")
ORACLE_SPECS = (
    "yaw_slot_oracle",
    "truth_phase_oracle",
    "truth_phase_slot_oracle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )
    parser.add_argument("--session-limit", type=int, default=None)
    return parser.parse_args()


def load_large(repo: Path):
    import importlib.util
    import sys

    path = repo / "scripts" / "evaluate-combined-motion-large-scale.py"
    spec = importlib.util.spec_from_file_location("combined_pnp_screen_large", path)
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


def session_fold(session_id: str) -> int:
    return int(hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:8], 16) % FOLDS


def read_conditions(manifest_path: Path) -> list[dict[str, Any]]:
    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("mode") == MODE:
                    rows.append(row)
    return rows


def extract_session(task: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    repo_raw, workspace_raw, condition = task
    repo = Path(repo_raw)
    workspace = Path(workspace_raw)
    large = load_large(repo)
    base, _, _ = large.modules(repo)
    session_id = str(condition["session_id"])
    try:
        frames, audit, sources = large.load_formal_session(repo, workspace, condition)
        frame_by_timestamp = {frame.timestamp_ns: frame for frame in frames}
        runtime_root = workspace / "runtime" / "stage3-formal-20260720-v2"
        truth_path, observation_path, _, result_path = large.select_run_paths(
            runtime_root, condition
        )
        start_ns = base.scene_motion_start_ns(result_path)
        rows: list[dict[str, Any]] = []
        with observation_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                timestamp_ns = int(record["timestamp_ns"])
                if timestamp_ns < start_ns:
                    continue
                frame = frame_by_timestamp.get(timestamp_ns)
                if frame is None:
                    continue
                valid = []
                for armor in record.get("armors", []):
                    if not bool(armor.get("valid", False)):
                        continue
                    position = np.asarray(armor.get("position_m", []), dtype=np.float64)
                    yaw = armor.get("yaw_absolute_rad", armor.get("yaw_rad"))
                    if position.shape != (3,) or not np.isfinite(position).all():
                        continue
                    if yaw is None or not math.isfinite(float(yaw)):
                        continue
                    valid.append((position, float(yaw)))
                if not valid or len(valid) > 4:
                    continue
                local = np.stack([item[0] for item in valid])
                world = frame.tracker_origin_world_m[None, :] + local @ frame.tracker_to_world.T
                slots, ambiguous = base.best_assignment(world, frame.armor_world_m)
                if ambiguous:
                    continue
                for observation_index, slot_raw in enumerate(slots):
                    slot = int(slot_raw)
                    truth_tracker = (
                        frame.armor_world_m[slot] - frame.tracker_origin_world_m
                    ) @ frame.tracker_to_world
                    radial_tracker = (
                        frame.armor_world_m[slot] - frame.center_world_m
                    ) @ frame.tracker_to_world
                    truth_phase = math.atan2(
                        float(radial_tracker[1]), float(radial_tracker[0])
                    )
                    observed = local[observation_index]
                    rows.append(
                        {
                            "session_id": session_id,
                            "fold": session_fold(session_id),
                            "timestamp_ns": timestamp_ns,
                            "slot": slot,
                            "camera_profile_id": str(
                                record.get("camera_profile_id", "missing")
                            ),
                            "candidate_count": len(valid),
                            "distance_m": float(condition["distance_m"]),
                            "linear_speed_mps": float(condition["linear_speed_mps"]),
                            "spin_rad_s": float(condition["spin_rad_s"]),
                            "direction_sector": int(condition["direction_sector"]),
                            "obs_x_m": float(observed[0]),
                            "obs_y_m": float(observed[1]),
                            "obs_z_m": float(observed[2]),
                            "pnp_yaw_rad": valid[observation_index][1],
                            "truth_phase_rad": truth_phase,
                            "truth_x_m": float(truth_tracker[0]),
                            "truth_y_m": float(truth_tracker[1]),
                            "truth_z_m": float(truth_tracker[2]),
                        }
                    )
        return {
            "session_id": session_id,
            "status": "ok",
            "rows": rows,
            "audit": audit,
            "sources": sources,
            "truth_path": str(truth_path),
            "observation_path": str(observation_path),
        }
    except Exception as error:
        return {
            "session_id": session_id,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "rows": [],
        }


def feature_matrix(rows: list[dict[str, Any]], spec: str) -> tuple[np.ndarray, list[str]]:
    obs = np.asarray(
        [[row["obs_x_m"], row["obs_y_m"], row["obs_z_m"]] for row in rows],
        dtype=np.float64,
    )
    radius = np.linalg.norm(obs, axis=1)
    cross = np.linalg.norm(obs[:, 1:3], axis=1)
    profile = np.asarray(
        [1.0 if row["camera_profile_id"] == "precision_16mm" else 0.0 for row in rows]
    )
    values = [obs[:, 0], obs[:, 1], obs[:, 2], radius, cross, radius * radius, profile]
    names = ["obs_x", "obs_y", "obs_z", "range", "obs_cross", "range_sq", "precision16"]

    if spec != "affine":
        phase_key = "truth_phase_rad" if spec.startswith("truth_phase") else "pnp_yaw_rad"
        phase = np.asarray([row[phase_key] for row in rows], dtype=np.float64)
        for harmonic in range(1, 5):
            sin_value = np.sin(harmonic * phase)
            cos_value = np.cos(harmonic * phase)
            values.extend((sin_value, cos_value, radius * sin_value, radius * cos_value))
            names.extend(
                (
                    f"sin{harmonic}",
                    f"cos{harmonic}",
                    f"range_sin{harmonic}",
                    f"range_cos{harmonic}",
                )
            )

    if "slot" in spec:
        slots = np.asarray([int(row["slot"]) for row in rows], dtype=np.int64)
        for slot in range(4):
            indicator = (slots == slot).astype(np.float64)
            values.append(indicator)
            names.append(f"slot_{slot}")
            if spec != "affine":
                phase_key = (
                    "truth_phase_rad" if spec.startswith("truth_phase") else "pnp_yaw_rad"
                )
                phase = np.asarray([row[phase_key] for row in rows], dtype=np.float64)
                values.extend((indicator * np.sin(phase), indicator * np.cos(phase)))
                names.extend((f"slot_{slot}_sin1", f"slot_{slot}_cos1"))
    return np.column_stack(values), names


def targets(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(
        [[row["obs_x_m"], row["obs_y_m"], row["obs_z_m"]] for row in rows],
        dtype=np.float64,
    )
    truth = np.asarray(
        [[row["truth_x_m"], row["truth_y_m"], row["truth_z_m"]] for row in rows],
        dtype=np.float64,
    )
    return observed, truth - observed


def fit_model(rows: list[dict[str, Any]], spec: str) -> dict[str, Any]:
    raw, names = feature_matrix(rows, spec)
    _, target = targets(rows)
    mean = np.mean(raw, axis=0)
    scale = np.std(raw, axis=0)
    scale = np.where(scale > 1.0e-9, scale, 1.0)
    standardized = (raw - mean) / scale
    design = np.column_stack((np.ones(len(rows)), standardized))
    penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE_LAMBDA
    penalty[0, 0] = 0.0
    weights = np.ones(len(rows), dtype=np.float64)
    coefficient = np.zeros((design.shape[1], 3), dtype=np.float64)
    metric = np.asarray([math.sqrt(DEPTH_WEIGHT), 1.0, 1.0], dtype=np.float64)
    for _ in range(IRLS_ITERATIONS):
        root = np.sqrt(weights)
        weighted_design = design * root[:, None]
        weighted_target = target * root[:, None]
        coefficient = np.linalg.solve(
            weighted_design.T @ weighted_design + penalty,
            weighted_design.T @ weighted_target,
        )
        residual = (design @ coefficient - target) * metric[None, :]
        norm = np.linalg.norm(residual, axis=1)
        weights = np.minimum(1.0, HUBER_DELTA_M / np.maximum(norm, 1.0e-12))
    return {
        "spec": spec,
        "feature_names": names,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": coefficient.tolist(),
        "train_rows": len(rows),
        "train_sessions": len({row["session_id"] for row in rows}),
    }


def predict_model(
    rows: list[dict[str, Any]], model: dict[str, Any]
) -> np.ndarray:
    raw, names = feature_matrix(rows, str(model["spec"]))
    if names != model["feature_names"]:
        raise ValueError("feature contract mismatch")
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    design = np.column_stack((np.ones(len(rows)), (raw - mean) / scale))
    observed, _ = targets(rows)
    return observed + design @ coefficient


def metric_values(prediction: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    error = prediction - truth
    return {
        "depth_abs_m": np.abs(error[:, 0]),
        "tracker_y_abs_m": np.abs(error[:, 1]),
        "tracker_z_abs_m": np.abs(error[:, 2]),
        "cross_depth_m": np.linalg.norm(error[:, 1:3], axis=1),
        "position_3d_m": np.linalg.norm(error, axis=1),
    }


def summarize_metric(values: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
    }
    for quantile in QUANTILES:
        result[f"p{int(round(quantile * 100)):02d}"] = float(np.quantile(values, quantile))
    for gate in GATES_M:
        result[f"cdf_le_{int(round(gate * 1000)):03d}mm"] = float(np.mean(values <= gate))
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_ecdf(path: Path, arm_values: dict[str, np.ndarray]) -> None:
    colors = {
        "raw_pnp": "#666666",
        "affine": "#0072B2",
        "yaw_harmonic": "#009E73",
        "yaw_slot_oracle": "#CC79A7",
        "truth_phase_oracle": "#E69F00",
        "truth_phase_slot_oracle": "#D55E00",
        "clean_exact": "#000000",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    for axis, limit_m in zip(axes, (0.25, 1.0)):
        for arm, values in arm_values.items():
            ordered = np.sort(values)
            y = np.arange(1, len(ordered) + 1) / len(ordered)
            axis.plot(ordered * 1000.0, y, label=arm, color=colors.get(arm))
        axis.axvline(55.0, color="#222222", linestyle="--", linewidth=1.0)
        axis.set_xlim(0.0, limit_m * 1000.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Current-observation cross-depth error (mm)")
        axis.set_ylabel("Empirical CDF")
        axis.grid(alpha=0.22)
    axes[0].set_title("Central distribution")
    axes[1].set_title("Full 0–1 m view")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Held-session PnP correction screen — combined motion")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    workspace = args.workspace.resolve() if args.workspace else repo.parents[1]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    manifest_path = (
        workspace
        / "dataset"
        / "autoaim-stage3-v1"
        / "stage3-20260719-v1"
        / "session_manifest.jsonl"
    )
    conditions = read_conditions(manifest_path)
    if args.session_limit is not None:
        conditions = conditions[: args.session_limit]
    tasks = [(str(repo), str(workspace), condition) for condition in conditions]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(extract_session, tasks))
    failures = [result for result in results if result["status"] != "ok"]
    if failures:
        (output / "failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(f"{len(failures)} session extraction failures")
    rows = [row for result in results for row in result["rows"]]
    rows.sort(key=lambda row: (row["session_id"], row["timestamp_ns"], row["slot"]))

    fold_counts = []
    for fold in range(FOLDS):
        fold_rows = [row for row in rows if row["fold"] == fold]
        fold_counts.append(
            {
                "fold": fold,
                "sessions": len({row["session_id"] for row in fold_rows}),
                "events": len(fold_rows),
            }
        )
    write_csv(output / "fold_counts.csv", fold_counts)

    models: dict[str, dict[str, Any]] = defaultdict(dict)
    cross_fitted: dict[str, np.ndarray] = {
        spec: np.full((len(rows), 3), np.nan, dtype=np.float64) for spec in MODEL_SPECS
    }
    indices_by_fold = {
        fold: np.asarray([index for index, row in enumerate(rows) if row["fold"] == fold])
        for fold in range(FOLDS)
    }
    for fold in range(FOLDS):
        train = [row for row in rows if row["fold"] != fold]
        test_indices = indices_by_fold[fold]
        test = [rows[int(index)] for index in test_indices]
        for spec in MODEL_SPECS:
            model = fit_model(train, spec)
            models[str(fold)][spec] = model
            cross_fitted[spec][test_indices] = predict_model(test, model)

    observed, residual = targets(rows)
    truth = observed + residual
    arms: dict[str, np.ndarray] = {"raw_pnp": observed}
    arms.update(cross_fitted)
    arms["truth_depth_only"] = np.column_stack((truth[:, 0], observed[:, 1], observed[:, 2]))
    arms["truth_tracker_y_only"] = np.column_stack((observed[:, 0], truth[:, 1], observed[:, 2]))
    arms["truth_tracker_z_only"] = np.column_stack((observed[:, 0], observed[:, 1], truth[:, 2]))
    arms["truth_cross_depth_yz"] = np.column_stack((observed[:, 0], truth[:, 1], truth[:, 2]))
    arms["clean_exact"] = truth.copy()

    summaries: list[dict[str, Any]] = []
    metrics_by_arm = {arm: metric_values(prediction, truth) for arm, prediction in arms.items()}
    for arm, metrics in metrics_by_arm.items():
        for metric, values in metrics.items():
            summaries.append({"arm": arm, "metric": metric, **summarize_metric(values)})
    write_csv(output / "measurement_summary.csv", summaries)

    raw_cross = metrics_by_arm["raw_pnp"]["cross_depth_m"]
    promotion_rows = []
    passing = []
    raw_p50 = float(np.quantile(raw_cross, 0.50))
    raw_p90 = float(np.quantile(raw_cross, 0.90))
    raw_gate = float(np.mean(raw_cross <= 0.055))
    for spec in DEPLOYABLE_SPECS:
        values = metrics_by_arm[spec]["cross_depth_m"]
        p50 = float(np.quantile(values, 0.50))
        p90 = float(np.quantile(values, 0.90))
        gate = float(np.mean(values <= 0.055))
        passed = p50 <= raw_p50 and p90 <= raw_p90 and gate > raw_gate
        score = 0.5 * ((raw_p50 - p50) / raw_p50 + (raw_p90 - p90) / raw_p90)
        promotion_rows.append(
            {
                "arm": spec,
                "p50_m": p50,
                "p90_m": p90,
                "cdf_le_055m": gate,
                "promotion_pass": int(passed),
                "selection_score": score,
            }
        )
        if passed:
            passing.append((score, 1 if spec == "affine" else 0, spec))
    passing.sort(reverse=True)
    promoted = passing[0][2] if passing else None

    oracle_rank = []
    for spec in ORACLE_SPECS:
        values = metrics_by_arm[spec]["cross_depth_m"]
        oracle_rank.append((float(np.quantile(values, 0.90)), spec))
    oracle_rank.sort()
    best_oracle = oracle_rank[0][1]
    write_csv(output / "promotion_decision.csv", promotion_rows)

    long_fields = [
        "session_id",
        "fold",
        "timestamp_ns",
        "slot",
        "camera_profile_id",
        "distance_m",
        "linear_speed_mps",
        "spin_rad_s",
        "direction_sector",
        "arm",
        "depth_abs_m",
        "tracker_y_abs_m",
        "tracker_z_abs_m",
        "cross_depth_m",
        "position_3d_m",
    ]

    def long_rows():
        for arm, metrics in metrics_by_arm.items():
            for index, row in enumerate(rows):
                yield {
                    "session_id": row["session_id"],
                    "fold": row["fold"],
                    "timestamp_ns": row["timestamp_ns"],
                    "slot": row["slot"],
                    "camera_profile_id": row["camera_profile_id"],
                    "distance_m": row["distance_m"],
                    "linear_speed_mps": row["linear_speed_mps"],
                    "spin_rad_s": row["spin_rad_s"],
                    "direction_sector": row["direction_sector"],
                    "arm": arm,
                    **{name: float(values[index]) for name, values in metrics.items()},
                }

    write_csv_gz(output / "measurement_error_distribution.csv.gz", long_rows(), long_fields)
    model_payload = {
        "schema_version": "combined-pnp-cross-fitted-correction-models-v1",
        "folds": FOLDS,
        "fold_function": "sha256(session_id) first 32 bits modulo 5",
        "ridge_lambda": RIDGE_LAMBDA,
        "irls_iterations": IRLS_ITERATIONS,
        "huber_delta_m": HUBER_DELTA_M,
        "models": models,
        "promoted_deployable_arm": promoted,
        "best_oracle_arm": best_oracle,
    }
    (output / "cross_fitted_models.json").write_text(
        json.dumps(model_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_ecdf(
        output / "measurement_cross_depth_ecdf",
        {
            arm: metrics_by_arm[arm]["cross_depth_m"]
            for arm in ("raw_pnp", *MODEL_SPECS)
        },
    )

    manifest = {
        "schema_version": "combined-pnp-downstream-correction-screen-v1",
        "formal_complete": args.session_limit is None and len(conditions) == 144,
        "sessions": len(conditions),
        "events": len(rows),
        "fold_counts": fold_counts,
        "promoted_deployable_arm": promoted,
        "best_oracle_arm": best_oracle,
        "contract": str(
            repo / "modules" / "autoaim" / "docs" / "combined_motion_pnp_error_reduction_contract.json"
        ),
        "sources": {
            "dataset_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        },
        "artifacts": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
