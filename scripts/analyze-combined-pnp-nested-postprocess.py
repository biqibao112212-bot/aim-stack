#!/usr/bin/env python3
"""Nested held-session selection of bounded coordinate PnP correction."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np


MASKS = {
    "full_xyz": np.asarray([1.0, 1.0, 1.0]),
    "cross_yz_only": np.asarray([0.0, 1.0, 1.0]),
    "tracker_y_only": np.asarray([0.0, 1.0, 0.0]),
}
BLENDS = (0.50, 0.75, 1.00)
CAPS = (0.10, 0.20, 0.40, math.inf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    return parser.parse_args()


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def extract_task(task: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    repo = Path(task[0])
    screen = load_script(
        f"nested_pnp_screen_worker_{os.getpid()}",
        repo / "scripts" / "analyze-combined-pnp-downstream-correction.py",
    )
    return screen.extract_session(task)


def apply_config(
    observed: np.ndarray,
    corrected: np.ndarray,
    mask_name: str,
    blend: float,
    cap: float,
) -> np.ndarray:
    delta = (corrected - observed) * MASKS[mask_name][None, :]
    weighted_norm = np.sqrt(0.1 * delta[:, 0] ** 2 + delta[:, 1] ** 2 + delta[:, 2] ** 2)
    if math.isfinite(cap):
        scale = np.minimum(1.0, cap / np.maximum(weighted_norm, 1.0e-12))
        delta = delta * scale[:, None]
    return observed + blend * delta


def cross_error(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.linalg.norm((prediction - truth)[:, 1:3], axis=1)


def metrics(values: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
        "cdf_le_055m": float(np.mean(values <= 0.055)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    workspace = args.workspace.resolve() if args.workspace else repo.parents[1]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    screen = load_script(
        "nested_pnp_screen", repo / "scripts" / "analyze-combined-pnp-downstream-correction.py"
    )
    models_payload = json.loads(args.models.resolve().read_text(encoding="utf-8-sig"))
    manifest_path = (
        workspace / "dataset" / "autoaim-stage3-v1" / "stage3-20260719-v1" / "session_manifest.jsonl"
    )
    conditions = screen.read_conditions(manifest_path)
    tasks = [(str(repo), str(workspace), condition) for condition in conditions]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(extract_task, tasks))
    failures = [result for result in results if result["status"] != "ok"]
    if failures:
        (output / "failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(f"{len(failures)} extraction failures")
    rows = [row for result in results for row in result["rows"]]
    rows.sort(key=lambda row: (row["session_id"], row["timestamp_ns"], row["slot"]))
    observed, residual = screen.targets(rows)
    truth = observed + residual
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    outer_prediction = np.full_like(observed, np.nan)
    selections = []
    inner_candidates = []

    for outer in range(screen.FOLDS):
        inner_prediction = np.full_like(observed, np.nan)
        inner_mask = folds != outer
        for held_inner in range(screen.FOLDS):
            if held_inner == outer:
                continue
            train_rows = [
                row for row in rows if row["fold"] != outer and row["fold"] != held_inner
            ]
            test_indices = np.flatnonzero(folds == held_inner)
            test_rows = [rows[int(index)] for index in test_indices]
            model = screen.fit_model(train_rows, "yaw_harmonic")
            inner_prediction[test_indices] = screen.predict_model(test_rows, model)
        raw_metrics = metrics(cross_error(observed[inner_mask], truth[inner_mask]))
        eligible = []
        for mask_name in MASKS:
            for blend in BLENDS:
                for cap in CAPS:
                    candidate = apply_config(
                        observed[inner_mask], inner_prediction[inner_mask], mask_name, blend, cap
                    )
                    value = metrics(cross_error(candidate, truth[inner_mask]))
                    passed = (
                        value["p50"] <= raw_metrics["p50"]
                        and value["p90"] <= raw_metrics["p90"]
                        and value["p95"] <= raw_metrics["p95"]
                        and value["cdf_le_055m"] > raw_metrics["cdf_le_055m"]
                    )
                    record = {
                        "outer_fold": outer,
                        "mask": mask_name,
                        "blend": blend,
                        "cap_m": "infinite" if not math.isfinite(cap) else cap,
                        "eligible": int(passed),
                        **value,
                    }
                    inner_candidates.append(record)
                    if passed:
                        coordinate_count = int(np.sum(MASKS[mask_name]))
                        cap_rank = -cap if math.isfinite(cap) else -1.0e9
                        rank = (
                            value["cdf_le_055m"],
                            -value["p90"],
                            -value["p50"],
                            -value["p95"],
                            -coordinate_count,
                            -blend,
                            cap_rank,
                        )
                        eligible.append((rank, mask_name, blend, cap, value))
        if not eligible:
            raise RuntimeError(f"no eligible nested postprocessor for outer fold {outer}")
        eligible.sort(key=lambda item: item[0], reverse=True)
        _, mask_name, blend, cap, inner_value = eligible[0]
        outer_indices = np.flatnonzero(folds == outer)
        outer_rows = [rows[int(index)] for index in outer_indices]
        original_model = models_payload["models"][str(outer)]["yaw_harmonic"]
        original_prediction = screen.predict_model(outer_rows, original_model)
        outer_prediction[outer_indices] = apply_config(
            observed[outer_indices], original_prediction, mask_name, blend, cap
        )
        selections.append(
            {
                "outer_fold": outer,
                "mask": mask_name,
                "blend": blend,
                "cap_m": "infinite" if not math.isfinite(cap) else cap,
                "inner_metrics": inner_value,
            }
        )

    original = np.full_like(observed, np.nan)
    for fold in range(screen.FOLDS):
        indices = np.flatnonzero(folds == fold)
        fold_rows = [rows[int(index)] for index in indices]
        model = models_payload["models"][str(fold)]["yaw_harmonic"]
        original[indices] = screen.predict_model(fold_rows, model)
    summaries = []
    for arm, prediction in (
        ("raw_pnp", observed),
        ("yaw_harmonic", original),
        ("nested_postprocessed", outer_prediction),
    ):
        summaries.append({"arm": arm, **metrics(cross_error(prediction, truth))})
    summary_by_arm = {row["arm"]: row for row in summaries}
    base = summary_by_arm["yaw_harmonic"]
    nested = summary_by_arm["nested_postprocessed"]
    promoted = (
        nested["cdf_le_055m"] > base["cdf_le_055m"]
        and nested["p50"] <= base["p50"]
        and nested["p90"] <= base["p90"]
        and nested["p95"] <= base["p95"]
    )
    write_csv(output / "measurement_summary.csv", summaries)
    write_csv(output / "inner_candidate_metrics.csv", inner_candidates)
    long_fields = [
        "session_id", "fold", "timestamp_ns", "slot", "camera_profile_id",
        "distance_m", "linear_speed_mps", "spin_rad_s", "direction_sector",
        "raw_cross_depth_m", "yaw_harmonic_cross_depth_m", "nested_cross_depth_m"
    ]
    with gzip.open(output / "outer_held_distribution.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fields)
        writer.writeheader()
        raw_error = cross_error(observed, truth)
        original_error = cross_error(original, truth)
        nested_error = cross_error(outer_prediction, truth)
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    **{key: row[key] for key in long_fields[:9]},
                    "raw_cross_depth_m": float(raw_error[index]),
                    "yaw_harmonic_cross_depth_m": float(original_error[index]),
                    "nested_cross_depth_m": float(nested_error[index]),
                }
            )
    payload = {
        "schema_version": "combined-pnp-nested-postprocessor-v1",
        "promoted": bool(promoted),
        "selections": selections,
        "outer_held_summary": summaries,
        "parent_models": str(args.models.resolve()),
        "complete": len(conditions) == 144 and len(rows) == 465311,
    }
    (output / "nested_postprocessor.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
