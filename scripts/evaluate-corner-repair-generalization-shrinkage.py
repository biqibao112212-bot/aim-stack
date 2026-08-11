#!/usr/bin/env python3
"""Select conservative corner-network shrinkage without outer-session leakage.

For every leave-session-out fold, alpha is selected only on that fold's inner
validation segment.  The frozen outer predictions are then combined as

    mean_correction + alpha * (network_correction - mean_correction).

The script preserves every empirical row and emits full ECDF/histogram figures;
quantiles are navigation summaries, not substitutes for the distribution.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.stage3.corner_residual_network import (  # noqa: E402
    CORNER_ORDER,
    Standardization,
    observable_features,
)
from training.stage3.train_corner_residual_network import (  # noqa: E402
    CornerLightningModule,
    corner_array,
)


SCHEMA_VERSION = "stage3-corner-repair-generalization-shrinkage-v1"
ARMS = ("raw", "mean", "network", "nested_shrinkage")
COLORS = {
    "raw": "#6B7280",
    "mean": "#009E73",
    "network": "#D55E00",
    "nested_shrinkage": "#0072B2",
}
LABELS = {
    "raw": "Raw detector",
    "mean": "Train-mean correction",
    "network": "Full network",
    "nested_shrinkage": "Nested-selected shrinkage",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--atlas", required=True, type=Path)
    result.add_argument("--training-evidence", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--alpha-step", type=float, default=0.05)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prediction_from_checkpoints(
    checkpoint_rows: pd.DataFrame,
    features: np.ndarray,
    correction_cap_px: float,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for row in checkpoint_rows.itertuples(index=False):
        payload = json.loads(row.standardization)
        standardization = Standardization(
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float32),
            feature_std=np.asarray(payload["feature_std"], dtype=np.float32),
            target_mean=np.asarray(payload["target_mean"], dtype=np.float32),
            target_std=np.asarray(payload["target_std"], dtype=np.float32),
        )
        model = CornerLightningModule.load_from_checkpoint(
            str(Path(row.checkpoint)), map_location="cpu"
        )
        model.eval()
        normalized_features = standardization.normalize_features(features)
        with torch.inference_mode():
            normalized_prediction = model(
                torch.from_numpy(np.ascontiguousarray(normalized_features))
            ).numpy()
        prediction = standardization.denormalize_targets(normalized_prediction)
        predictions.append(
            np.clip(prediction, -correction_cap_px, correction_cap_px)
        )
    if not predictions:
        raise ValueError("fold has no checkpoint predictions")
    return np.stack(predictions, axis=0).mean(axis=0)


def coordinate_rms(
    raw: np.ndarray, truth: np.ndarray, correction: np.ndarray
) -> np.ndarray:
    residual = raw + correction.reshape(-1, 4, 2) - truth
    return np.sqrt(np.mean(np.square(residual), axis=(1, 2)))


def select_alpha(
    raw: np.ndarray,
    truth: np.ndarray,
    mean_correction: np.ndarray,
    network_correction: np.ndarray,
    alpha_grid: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    best_key: tuple[float, float] | None = None
    selected = 0.0
    for alpha in alpha_grid:
        correction = mean_correction + float(alpha) * (
            network_correction - mean_correction
        )
        values = coordinate_rms(raw, truth, correction)
        row = {
            "alpha": float(alpha),
            "validation_mean_px": float(values.mean()),
            "validation_p50_px": float(np.quantile(values, 0.50)),
            "validation_p90_px": float(np.quantile(values, 0.90)),
            "validation_p95_px": float(np.quantile(values, 0.95)),
            "validation_p99_px": float(np.quantile(values, 0.99)),
        }
        rows.append(row)
        # Primary criterion is the mean over the complete validation
        # distribution.  Ties prefer the smaller, safer intervention.
        key = (row["validation_mean_px"], float(alpha))
        if best_key is None or key < best_key:
            best_key = key
            selected = float(alpha)
    return selected, rows


def distribution_metrics(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    quantiles = {
        f"p{int(round(probability * 100)):02d}_px": float(
            np.quantile(values, probability)
        )
        for probability in (0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00)
    }
    return {
        "samples": int(len(values)),
        "mean_px": float(values.mean()),
        "std_px": float(values.std()),
        **quantiles,
        "fraction_le_1px": float(np.mean(values <= 1.0)),
        "fraction_le_2px": float(np.mean(values <= 2.0)),
    }


def make_figures(rows: pd.DataFrame, output: Path) -> None:
    groups = sorted(rows.outer_group.unique())
    fig, axes = plt.subplots(
        1, len(groups), figsize=(7.0 * len(groups), 5.0),
        constrained_layout=True, squeeze=False,
    )
    for axis, group in zip(axes[0], groups):
        selected = rows[rows.outer_group == group]
        for arm in ARMS:
            values = np.sort(
                selected[selected.arm == arm].coordinate_rms_px.to_numpy(float)
            )
            probability = np.arange(1, len(values) + 1, dtype=float) / len(values)
            axis.step(
                values, probability, where="post", color=COLORS[arm],
                linewidth=2.0, label=LABELS[arm],
            )
        axis.set_xlabel("Per-detection coordinate RMS error (px)")
        axis.set_ylabel("Empirical cumulative probability")
        axis.set_ylim(0.0, 1.005)
        axis.set_title(f"Held session: {group.split('-')[-2]}")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="lower right")
    fig.suptitle("Complete held-session corner-error distributions")
    fig.savefig(output / "held_session_error_ecdf.png", dpi=220, facecolor="white")
    fig.savefig(output / "held_session_error_ecdf.svg", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(
        len(groups), 1, figsize=(12.0, 4.5 * len(groups)),
        constrained_layout=True, squeeze=False,
    )
    for axis, group in zip(axes[:, 0], groups):
        selected = rows[rows.outer_group == group]
        maximum = float(selected.coordinate_rms_px.max())
        bins = np.linspace(0.0, maximum, 61)
        for arm in ARMS:
            values = selected[selected.arm == arm].coordinate_rms_px.to_numpy(float)
            axis.hist(
                values, bins=bins, density=True, histtype="step",
                linewidth=1.8, color=COLORS[arm], label=LABELS[arm],
            )
        axis.set_xlabel("Per-detection coordinate RMS error (px)")
        axis.set_ylabel("Density")
        axis.set_title(f"Held session: {group.split('-')[-2]}")
        axis.grid(True, alpha=0.20)
        axis.legend()
    fig.suptitle("Held-session error histograms (shared bins within each panel)")
    fig.savefig(output / "held_session_error_histogram.png", dpi=220, facecolor="white")
    fig.savefig(output / "held_session_error_histogram.svg", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {args.output}")
    if not 0.0 < args.alpha_step <= 1.0:
        raise ValueError("alpha-step must be in (0,1]")

    atlas_path = args.atlas.resolve(strict=True)
    evidence = args.training_evidence.resolve(strict=True)
    manifest_path = evidence / "manifest.json"
    checkpoint_path = evidence / "checkpoint_registry.csv"
    prediction_path = evidence / "oof_corner_predictions.csv.gz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    correction_cap_px = float(manifest["arguments"]["correction_cap_px"])

    atlas = pd.read_csv(atlas_path)
    raw = corner_array(atlas, "raw")
    truth = corner_array(atlas, "truth")
    features = observable_features(raw)
    targets = (truth - raw).reshape(len(atlas), 8).astype(np.float32)
    inner_groups = np.asarray(
        [f"{session}|segment={int(segment)}" for session, segment in zip(
            atlas.session_id.astype(str), atlas.segment_index
        )], dtype=object,
    )

    checkpoints = pd.read_csv(checkpoint_path)
    checkpoints = checkpoints[checkpoints.scheme == "leave_session_out"].copy()
    outer_predictions = pd.read_csv(prediction_path)
    outer_predictions = outer_predictions[
        (outer_predictions.scheme == "leave_session_out")
        & outer_predictions.arm.isin(("raw", "mean", "network"))
    ].copy()
    alpha_grid = np.unique(
        np.append(np.arange(0.0, 1.0 + 0.5 * args.alpha_step, args.alpha_step), 1.0)
    )
    alpha_grid = np.clip(alpha_grid, 0.0, 1.0)

    args.output.mkdir(parents=True)
    alpha_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    empirical_rows: list[dict[str, Any]] = []
    for outer in sorted(checkpoints.outer_group.unique()):
        fold_checkpoints = checkpoints[checkpoints.outer_group == outer]
        inner_values = fold_checkpoints.inner_group.unique()
        if len(inner_values) != 1:
            raise ValueError(f"{outer}: inconsistent inner validation group")
        inner = str(inner_values[0])
        outer_mask = atlas.session_id.astype(str).to_numpy() == outer
        val_mask = (inner_groups == inner) & ~outer_mask
        train_mask = ~(outer_mask | val_mask)
        standardization_payload = json.loads(fold_checkpoints.iloc[0].standardization)
        mean_correction = np.repeat(
            np.asarray(standardization_payload["target_mean"], dtype=np.float32)[None],
            int(np.count_nonzero(val_mask)), axis=0,
        )
        val_network = prediction_from_checkpoints(
            fold_checkpoints, features[val_mask], correction_cap_px
        )
        selected_alpha, curve = select_alpha(
            raw[val_mask], truth[val_mask], mean_correction, val_network, alpha_grid
        )
        for row in curve:
            alpha_rows.append(
                {
                    "outer_group": outer,
                    "inner_validation_group": inner,
                    "train_samples": int(np.count_nonzero(train_mask)),
                    "validation_samples": int(np.count_nonzero(val_mask)),
                    **row,
                    "selected": bool(np.isclose(row["alpha"], selected_alpha)),
                }
            )
        selected_curve = next(
            row for row in curve if np.isclose(row["alpha"], selected_alpha)
        )
        selection_rows.append(
            {
                "outer_group": outer,
                "inner_validation_group": inner,
                "selected_alpha": selected_alpha,
                "selection_metric": "complete_validation_distribution_mean_coordinate_rms_px",
                **{name: value for name, value in selected_curve.items() if name != "alpha"},
            }
        )

        tables: dict[str, pd.DataFrame] = {}
        for arm in ("raw", "mean", "network"):
            table = outer_predictions[
                (outer_predictions.outer_group == outer)
                & (outer_predictions.arm == arm)
            ].sort_values("row_index").reset_index(drop=True)
            tables[arm] = table
        keys = tables["raw"][["row_index", "session_id", "frame_seq"]].to_numpy()
        if any(
            not np.array_equal(
                keys, tables[arm][["row_index", "session_id", "frame_seq"]].to_numpy()
            )
            for arm in ("mean", "network")
        ):
            raise AssertionError(f"{outer}: outer arm rows are not aligned")
        mean_outer = np.column_stack([
            tables["mean"][[
                f"correction_{corner}_dx_px", f"correction_{corner}_dy_px"
            ]].to_numpy(float) for corner in CORNER_ORDER
        ]).reshape(-1, 8)
        network_outer = np.column_stack([
            tables["network"][[
                f"correction_{corner}_dx_px", f"correction_{corner}_dy_px"
            ]].to_numpy(float) for corner in CORNER_ORDER
        ]).reshape(-1, 8)
        correction_by_arm = {
            "raw": np.zeros_like(mean_outer),
            "mean": mean_outer,
            "network": network_outer,
            "nested_shrinkage": mean_outer + selected_alpha * (
                network_outer - mean_outer
            ),
        }
        row_indices = tables["raw"].row_index.to_numpy(int)
        for arm, correction in correction_by_arm.items():
            residual = raw[row_indices] + correction.reshape(-1, 4, 2) - truth[row_indices]
            rms = np.sqrt(np.mean(np.square(residual), axis=(1, 2)))
            norms = np.linalg.norm(residual, axis=2)
            for local_index, row_index in enumerate(row_indices):
                item: dict[str, Any] = {
                    "outer_group": outer,
                    "inner_validation_group": inner,
                    "selected_alpha": selected_alpha if arm == "nested_shrinkage" else "",
                    "row_index": int(row_index),
                    "session_id": str(atlas.iloc[row_index].session_id),
                    "segment_index": int(atlas.iloc[row_index].segment_index),
                    "motion_mode": str(atlas.iloc[row_index].motion_mode),
                    "frame_seq": int(atlas.iloc[row_index].frame_seq),
                    "timestamp_ns": int(atlas.iloc[row_index].timestamp_ns),
                    "truth_slot": int(atlas.iloc[row_index].truth_slot),
                    "arm": arm,
                    "coordinate_rms_px": float(rms[local_index]),
                    "corner_norm_mean_px": float(norms[local_index].mean()),
                    "corner_norm_max_px": float(norms[local_index].max()),
                }
                for corner_index, corner in enumerate(CORNER_ORDER):
                    item[f"error_{corner}_dx_px"] = float(residual[local_index, corner_index, 0])
                    item[f"error_{corner}_dy_px"] = float(residual[local_index, corner_index, 1])
                    item[f"error_{corner}_norm_px"] = float(norms[local_index, corner_index])
                empirical_rows.append(item)

    empirical = pd.DataFrame(empirical_rows)
    metric_rows: list[dict[str, Any]] = []
    for (outer, arm), group in empirical.groupby(["outer_group", "arm"], sort=True):
        metric_rows.append(
            {
                "outer_group": outer,
                "arm": arm,
                **distribution_metrics(group.coordinate_rms_px.to_numpy(float)),
            }
        )
    for arm, group in empirical.groupby("arm", sort=True):
        metric_rows.append(
            {
                "outer_group": "ALL_HELD_SESSIONS",
                "arm": arm,
                **distribution_metrics(group.coordinate_rms_px.to_numpy(float)),
            }
        )

    write_csv(args.output / "alpha_validation_curve.csv", alpha_rows)
    write_csv(args.output / "selected_alpha_by_outer_session.csv", selection_rows)
    write_csv(args.output / "distribution_metrics.csv", metric_rows)
    write_csv_gz(args.output / "empirical_error_rows.csv.gz", empirical_rows)
    make_figures(empirical, args.output)

    report_lines = [
        "# Conservative corner-repair generalization audit",
        "",
        "Each held session remains untouched while alpha is chosen on a complete inner validation segment from the training session. Selection minimizes the mean over the complete validation distribution; P95 is reported but is not the selection objective.",
        "",
        "## Selected intervention strength",
        "",
        "| held session | inner validation group | alpha | validation mean / P50 / P90 / P95 / P99 (px) |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in selection_rows:
        report_lines.append(
            f"| {row['outer_group']} | {row['inner_validation_group']} | "
            f"{row['selected_alpha']:.2f} | {row['validation_mean_px']:.3f} / "
            f"{row['validation_p50_px']:.3f} / {row['validation_p90_px']:.3f} / "
            f"{row['validation_p95_px']:.3f} / {row['validation_p99_px']:.3f} |"
        )
    report_lines.extend([
        "",
        "## Held-session distributions",
        "",
        "| held session | arm | mean / P50 / P90 / P95 / P99 (px) | <=1 px | <=2 px |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for row in metric_rows:
        if row["outer_group"] == "ALL_HELD_SESSIONS":
            continue
        report_lines.append(
            f"| {row['outer_group']} | {LABELS[row['arm']]} | "
            f"{row['mean_px']:.3f} / {row['p50_px']:.3f} / {row['p90_px']:.3f} / "
            f"{row['p95_px']:.3f} / {row['p99_px']:.3f} | "
            f"{row['fraction_le_1px']:.1%} | {row['fraction_le_2px']:.1%} |"
        )
    report_lines.extend([
        "",
        "The compressed empirical table retains every detection, four signed corner residual vectors, and all corner norms. The ECDF and histogram use every retained row.",
        "",
        "This remains simulation-only and non-deployable: only two exact-truth sessions exist, so complete-session cross-fitting can expose negative transfer but cannot establish broad scene/domain generalization.",
    ])
    (args.output / "report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    atomic_json(
        args.output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "selection": {
                "outer_test_accessed_for_alpha_selection": False,
                "criterion": "mean coordinate RMS over complete inner validation distribution",
                "tie_break": "smaller alpha",
                "alpha_grid": alpha_grid.tolist(),
            },
            "scope": "simulation-only coordinate repair; no motion/time/session inputs",
            "deployable": False,
            "uniform_motion_only_downstream": True,
            "reversal_or_endpoint_evaluation": False,
            "inputs": {
                "atlas": {"path": str(atlas_path), "sha256": sha256(atlas_path)},
                "training_manifest": {
                    "path": str(manifest_path), "sha256": sha256(manifest_path)
                },
                "checkpoints": {
                    "path": str(checkpoint_path), "sha256": sha256(checkpoint_path)
                },
                "outer_predictions": {
                    "path": str(prediction_path), "sha256": sha256(prediction_path)
                },
            },
            "outputs": [
                "alpha_validation_curve.csv",
                "selected_alpha_by_outer_session.csv",
                "distribution_metrics.csv",
                "empirical_error_rows.csv.gz",
                "held_session_error_ecdf.png",
                "held_session_error_ecdf.svg",
                "held_session_error_histogram.png",
                "held_session_error_histogram.svg",
                "report.md",
            ],
        },
    )


if __name__ == "__main__":
    main()
