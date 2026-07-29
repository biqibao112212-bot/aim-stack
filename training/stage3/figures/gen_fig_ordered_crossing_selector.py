"""Re-evaluate the fixed ordered-crossing selector and draw diagnostic figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .gen_fig_joint_visible_future import (
    COLORS,
    _extract_queries,
    _load_truth_context,
    _load_validation_identity,
    _percentiles,
    _plot_distribution,
    _plot_switch_analysis,
    _plot_trends,
    _save,
    _style,
    _summary,
)
from ..cyclic_future_foundation import load_frozen_v19
from ..observable_future_pnp_ab import (
    ObservableFuturePnPSFDataset,
    canonicalize_direction_keep_c4,
    load_observable_f_checkpoint,
    sha256_file,
    state_dict_sha256,
)
from ..ordinal_visible_selector import (
    OrdinalVisibleFutureModel,
    OrdinalVisibleProgressSelector,
)
from ..pnp_q0_hypothesis_adapter import (
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from ..train_joint_visible_future import _build_cache


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _construct_model(
    checkpoint: dict[str, Any], trajectory_parent: Path,
    device: torch.device,
) -> OrdinalVisibleFutureModel:
    trajectory, _ = load_observable_f_checkpoint(trajectory_parent)
    config = checkpoint["model_config"]["selector"]
    selector = OrdinalVisibleProgressSelector(
        frozen_context_features=int(config["frozen_context_features"]),
        channels=int(config["channels"]),
        dropout=float(config["dropout"]),
        trained_horizon_s=float(config["trained_horizon_s"]),
        maximum_absolute_step=int(config["maximum_absolute_step"]),
    )
    model = OrdinalVisibleFutureModel(trajectory, selector)
    model.selector.load_state_dict(checkpoint["selector"], strict=True)
    if state_dict_sha256(model.trajectory.state_dict()) != checkpoint[
        "frozen_trajectory_state_dict_sha256"
    ]:
        raise ValueError("ordered figure trajectory state hash differs")
    if state_dict_sha256(model.selector.state_dict()) != checkpoint[
        "selector_state_dict_sha256"
    ]:
        raise ValueError("ordered figure selector state hash differs")
    return model.to(device).eval()


def _plot_training(run_manifest: dict[str, Any], output: Path) -> None:
    history = run_manifest["history"]
    updates = np.asarray([item["update"] for item in history])
    accuracy = np.asarray([
        item["validation"]["switch_accuracy"] for item in history
    ]) * 100
    recall = np.asarray([
        item["validation"]["minimum_step_switch_recall"] for item in history
    ]) * 100
    conditional = np.asarray([
        item["validation"]["conditional_position"]["p95_m"] for item in history
    ]) * 1000
    hard = np.asarray([
        item["validation"]["hard_routed_position"]["p95_m"] for item in history
    ]) * 1000
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    axes[0].plot(updates, accuracy, "o-", color=COLORS["coral"], label="All-step accuracy")
    axes[0].plot(updates, recall, "s--", color=COLORS["blue"], label="One-step recall")
    axes[0].axhline(83.25464070888041, color=COLORS["gray"], linestyle=":", label="V52 accuracy")
    axes[0].set(xlabel="Optimizer updates", ylabel="Selection metric (%)", title="Selection learning")
    axes[0].legend(loc="lower right")
    axes[1].plot(updates, conditional, "o-", color=COLORS["teal"], label="Conditional P95")
    axes[1].plot(updates, hard, "s-", color=COLORS["coral"], label="Hard-routed P95")
    axes[1].axhline(363.5366, color=COLORS["gray"], linestyle=":", label="V52 hard P95")
    axes[1].set(xlabel="Optimizer updates", ylabel="Position error (mm)", title="Validation error")
    axes[1].legend(loc="upper right")
    fig.suptitle("Ordered crossing-time selector: fixed-endpoint training", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, output, "fig_training_dynamics")


def _metric_view(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        metrics["conditional_position"]["p95_m"] * 1000,
        metrics["hard_routed_position"]["p95_m"] * 1000,
        metrics["hard_routed_position"]["p99_m"] * 1000,
        metrics["switch_accuracy"] * 100,
        metrics["minimum_step_switch_recall"] * 100,
    )


def _plot_comparison(
    trajectory_checkpoint: dict[str, Any], selector_manifest: dict[str, Any],
    joint_manifest: dict[str, Any], ordered_checkpoint: dict[str, Any],
    output: Path,
) -> dict[str, dict[str, float]]:
    methods = [
        ("V50", "Trajectory", trajectory_checkpoint["validation"]["pnp_domain"]),
        ("V52", "Flat selector", selector_manifest["best"]["validation"]["pnp_domain"]),
        ("V64", "Joint", joint_manifest["final"]["validation"]),
        ("V66", "Ordered times", ordered_checkpoint["validation"]),
    ]
    names = [item[0] for item in methods]
    values = np.asarray([_metric_view(item[2]) for item in methods])
    titles = ["Conditional P95", "Hard P95", "Hard P99", "Accuracy", "One-step recall"]
    units = ["Error (mm)", "Error (mm)", "Error (mm)", "Accuracy (%)", "Recall (%)"]
    fig, axes = plt.subplots(1, 5, figsize=(13.0, 2.8))
    colors = [COLORS["gray"], COLORS["teal"], COLORS["orange"], COLORS["coral"]]
    for column, ax in enumerate(axes):
        bars = ax.bar(np.arange(len(names)), values[:, column], color=colors, width=0.7)
        ax.set_xticks(np.arange(len(names)), names)
        ax.set_ylabel(units[column])
        ax.set_title(titles[column])
        for bar, value in zip(bars, values[:, column]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}",
                    ha="center", va="bottom", fontsize=6.5)
    fig.suptitle("Same combined-motion diagnostic validation", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, output, "fig_baseline_comparison")
    return {
        f"{name} {description}": {
            "conditional_p95_mm": float(row[0]),
            "hard_p95_mm": float(row[1]),
            "hard_p99_mm": float(row[2]),
            "switch_accuracy_pct": float(row[3]),
            "one_step_recall_pct": float(row[4]),
        }
        for (name, description, _), row in zip(methods, values)
    }


def generate(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir).resolve()
    run_manifest = _read_json(run_dir / "run_manifest.json")
    if run_manifest.get("status") != "complete" or run_manifest.get("update") != 2125:
        raise ValueError("ordered run is not the fixed 2125-update run")
    final_epoch = int(run_manifest["epoch"])
    checkpoint_path = run_dir / f"epoch-{final_epoch:04d}-update-002125.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("update") != 2125:
        raise ValueError("ordered final checkpoint update differs")
    training_args = run_manifest["training_arguments"]
    dataset_root = Path(training_args["dataset"]).resolve()
    dataset_manifest = _read_json(dataset_root / "dataset_manifest.json")
    if dataset_manifest.get("test_accessed", True):
        raise ValueError("ordered figure dataset reports test access")
    truth_root = Path(dataset_manifest["truth_history_dataset"]).resolve()
    pair_ids, session_ids, t0_values = _load_validation_identity(dataset_root)
    dataset = ObservableFuturePnPSFDataset(
        dataset_root, "validation", motion_class=3, allow_diagnostic=False,
    )
    if tuple(str(value) for value in pair_ids) != dataset.pair_ids:
        raise ValueError("ordered validation identity order differs")
    canonicalize_direction_keep_c4(dataset.tensors, dataset.pair_ids)
    distance_m, yaw_rate_rad_s = _load_truth_context(
        truth_root, session_ids, t0_values,
    )
    raw_mask = dataset.tensors["pnp_s_obs_mask"].to(torch.bool)
    raw_error = torch.linalg.vector_norm(
        dataset.tensors["pnp_s_obs_m"] - dataset.tensors["clean_s_obs_m"], dim=-1,
    )
    raw_pnp_rms_m = np.asarray([
        float(torch.sqrt(raw_error[index][raw_mask[index]].square().mean()))
        for index in range(len(dataset))
    ])

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    mapper, _ = load_frozen_pnp_mapper(training_args["mapper_checkpoint"])
    s_model, _ = load_frozen_v19(training_args["s_checkpoint"])
    h_model, _ = load_frozen_hypothesis_adapter(
        training_args["h_checkpoint"], allow_diagnostic=True,
    )
    for frozen in (mapper, s_model, h_model):
        frozen.to(device).eval().requires_grad_(False)
    cache, _ = _build_cache(
        dataset, mapper, s_model, h_model, device=device,
        batch_size=int(training_args["cache_batch_size"]),
    )
    loader = DataLoader(
        cache, batch_size=int(training_args["batch_size"]), shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    trajectory_parent = Path(training_args["trajectory_checkpoint"]).resolve()
    model = _construct_model(checkpoint, trajectory_parent, device)
    queries, conditional_hash = _extract_queries(
        model, loader, device,
        sample_distance_m=distance_m,
        sample_yaw_rate_rad_s=yaw_rate_rad_s,
        sample_raw_pnp_rms_m=raw_pnp_rms_m,
        session_ids=session_ids,
    )
    expected = checkpoint["validation"]
    conditional_metrics = _percentiles(queries["conditional_error_m"])
    hard_metrics = _percentiles(queries["hard_error_m"])
    if abs(conditional_metrics["p95_m"] - expected["conditional_position"]["p95_m"]) > 1e-7:
        raise ValueError("ordered conditional P95 replay differs")
    if abs(hard_metrics["p95_m"] - expected["hard_routed_position"]["p95_m"]) > 1e-7:
        raise ValueError("ordered hard P95 replay differs")
    if conditional_hash != expected["conditional_output_sha256"]:
        raise ValueError("ordered conditional tensor hash differs")

    output = Path(args.output_dir).resolve() if args.output_dir else run_dir / "figures-r1"
    output.mkdir(exist_ok=False)
    _style()
    _plot_training(run_manifest, output)
    _plot_distribution(queries, output)
    _plot_trends(queries, output)
    _plot_switch_analysis(queries, output)
    trajectory_checkpoint = torch.load(
        trajectory_parent, map_location="cpu", weights_only=False,
    )
    comparison = _plot_comparison(
        trajectory_checkpoint,
        _read_json(Path(args.selector_baseline_manifest).resolve()),
        _read_json(Path(args.joint_baseline_manifest).resolve()),
        checkpoint, output,
    )
    np.savez_compressed(output / "validation_queries.npz", **queries)
    summary = _summary(
        queries, comparison, checkpoint_path, run_manifest, truth_root,
    )
    summary["schema_version"] = "ordered-crossing-selector-figures-v1"
    summary["fixed_endpoint_update"] = 2125
    summary["automatic_gate"] = run_manifest["gate"]
    summary["predicted_nonmonotone_fraction"] = expected[
        "predicted_sequence_monotonicity"
    ]["nonmonotone_fraction"]
    (output / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-dir", required=True)
    result.add_argument("--output-dir", default="")
    result.add_argument("--selector-baseline-manifest", required=True)
    result.add_argument("--joint-baseline-manifest", required=True)
    result.add_argument("--device", default="cuda")
    return result


def main() -> None:
    print(generate(parser().parse_args()))


if __name__ == "__main__":
    main()
