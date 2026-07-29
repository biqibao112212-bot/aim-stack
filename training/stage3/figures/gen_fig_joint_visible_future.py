"""Re-evaluate a completed joint visible-future run and draw diagnostic figures.

The script only opens the validation split. Physical distance and yaw rate are
joined after inference for stratified analysis; neither is passed to the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..cyclic_future_foundation import load_frozen_v19
from ..joint_visible_future import JointVisibleFutureModel, LearnedVisibleStateSelector
from ..observable_future_loss import _target_candidate_row
from ..observable_future_pnp_ab import (
    ObservableFuturePnPSFDataset,
    canonicalize_direction_keep_c4,
    load_observable_f_checkpoint,
    sha256_file,
    state_dict_sha256,
)
from ..pnp_q0_hypothesis_adapter import (
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from ..train_causal_physical_ab import _to_device
from ..train_joint_visible_future import _build_cache, _loss_batch, _model_batch


COLORS = {
    "teal": "#264653",
    "cyan": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "coral": "#E76F51",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "gray": "#8C8C8C",
}
THRESHOLDS_MM = np.asarray([20, 50, 100, 150, 200, 300, 500], dtype=np.float64)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.8,
    })


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf")
    fig.savefig(output / f"{stem}.png", dpi=300)
    plt.close(fig)


def _percentiles(values_m: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values_m, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean_m": float(values.mean()),
        "p50_m": float(np.percentile(values, 50)),
        "p95_m": float(np.percentile(values, 95)),
        "p99_m": float(np.percentile(values, 99)),
        "max_m": float(values.max()),
    }


def _load_validation_identity(
    dataset_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifest = _read_json(dataset_root / "dataset_manifest.json")
    pair_ids: list[np.ndarray] = []
    session_ids: list[np.ndarray] = []
    t0_values: list[np.ndarray] = []
    for item in manifest["shards"]:
        if item["split"] != "validation":
            continue
        path = dataset_root / Path(str(item["path"]).replace("\\", "/"))
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"validation identity shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as loaded:
            keep = (
                (loaded["motion_class"] == 3)
                & loaded["pnp_sf_common_usable"].astype(np.bool_)
            )
            pair_ids.append(loaded["pair_id"][keep].copy())
            session_ids.append(loaded["session_id"][keep].copy())
            t0_values.append(loaded["t0_ns"][keep].copy())
    return (
        np.concatenate(pair_ids),
        np.concatenate(session_ids),
        np.concatenate(t0_values),
    )


def _load_truth_context(
    truth_root: Path,
    session_ids: np.ndarray,
    t0_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    manifest_path = truth_root / "dataset_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("test_accessed", True):
        raise ValueError("truth-history manifest reports test access")
    lookup: dict[tuple[str, int], tuple[float, float]] = {}
    for item in manifest["shards"]:
        if item["split"] != "validation":
            continue
        path = truth_root / Path(str(item["path"]).replace("\\", "/"))
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"truth-history shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as loaded:
            for row, (session_id, t0_ns) in enumerate(zip(
                loaded["session_id"], loaded["t0_ns"]
            )):
                key = (str(session_id), int(t0_ns))
                if key in lookup:
                    raise ValueError(f"duplicate truth-history key: {key}")
                lookup[key] = (
                    float(loaded["distance_m"][row]),
                    float(loaded["anchor_yaw_rate_rad_s"][row]),
                )
    distance: list[float] = []
    yaw_rate: list[float] = []
    for session_id, t0_ns in zip(session_ids, t0_values):
        key = (str(session_id), int(t0_ns))
        if key not in lookup:
            raise ValueError(f"truth-history key missing: {key}")
        item = lookup[key]
        distance.append(item[0])
        yaw_rate.append(item[1])
    return np.asarray(distance), np.asarray(yaw_rate)


def _construct_model(
    checkpoint: dict[str, Any], trajectory_parent: Path, device: torch.device,
) -> JointVisibleFutureModel:
    trajectory, _ = load_observable_f_checkpoint(trajectory_parent)
    selector_config = checkpoint["model_config"]["selector"]
    selector = LearnedVisibleStateSelector(
        channels=int(selector_config["channels"]),
        dropout=float(selector_config["dropout"]),
        position_scale_m=float(selector_config["position_scale_m"]),
        history_scale_s=float(selector_config["history_scale_s"]),
        trained_horizon_s=float(selector_config["trained_horizon_s"]),
        maximum_absolute_step=int(selector_config["maximum_absolute_step"]),
    )
    model = JointVisibleFutureModel(trajectory, selector)
    model.trajectory.load_state_dict(checkpoint["trajectory"], strict=True)
    model.selector.load_state_dict(checkpoint["selector"], strict=True)
    if state_dict_sha256(model.trajectory.state_dict()) != checkpoint[
        "trajectory_state_dict_sha256"
    ]:
        raise ValueError("trajectory state hash differs from final checkpoint")
    if state_dict_sha256(model.selector.state_dict()) != checkpoint[
        "selector_state_dict_sha256"
    ]:
        raise ValueError("selector state hash differs from final checkpoint")
    return model.to(device).eval()


@torch.no_grad()
def _extract_queries(
    model: JointVisibleFutureModel,
    loader: DataLoader,
    device: torch.device,
    *,
    sample_distance_m: np.ndarray,
    sample_yaw_rate_rad_s: np.ndarray,
    sample_raw_pnp_rms_m: np.ndarray,
    session_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], str]:
    parts: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "sample_index", "session_id", "tau_s", "target_switch_count",
            "selected_switch_step", "selection_correct", "selected_probability",
            "true_probability", "true_role_supported", "conditional_error_m",
            "hard_error_m", "distance_m", "yaw_rate_rad_s",
            "upstream_current_error_m", "raw_pnp_rms_m",
        )
    }
    conditional_hasher = hashlib.sha256()
    offset = 0
    for raw_batch in loader:
        cached = _to_device(raw_batch, device)
        batch = _loss_batch(cached)
        prediction = model(_model_batch(batch))
        conditional_hasher.update(
            prediction["conditional_position_m"].detach().cpu().contiguous().numpy().tobytes()
        )
        true_row, query_mask = _target_candidate_row(
            cached["candidate_step"], cached["candidate_mask"],
            cached["target_switch_count"], cached["target_query_mask"],
        )
        selected_row = prediction["selected_candidate_row"]
        gather_true = true_row[:, :, None, None].expand(-1, -1, 1, 3)
        gather_hard = selected_row[:, :, None, None].expand(-1, -1, 1, 3)
        conditional_position = prediction["conditional_position_m"].gather(
            2, gather_true
        ).squeeze(2)
        hard_position = prediction["conditional_position_m"].gather(
            2, gather_hard
        ).squeeze(2)
        target_position = (
            cached["truth_current_position_m"][:, None]
            + cached["target_visible_delta_m"]
        )
        conditional_error = torch.linalg.vector_norm(
            conditional_position - target_position, dim=-1
        )
        hard_error = torch.linalg.vector_norm(hard_position - target_position, dim=-1)
        selected_step = cached["candidate_step"].gather(1, selected_row)
        selected_probability = prediction["switch_probability"].gather(
            2, selected_row.unsqueeze(-1)
        ).squeeze(-1)
        true_probability = prediction["switch_probability"].gather(
            2, true_row.unsqueeze(-1)
        ).squeeze(-1)
        true_support = cached["candidate_supported"].gather(1, true_row)
        current_error = torch.linalg.vector_norm(
            cached["current_position_m"] - cached["truth_current_position_m"], dim=-1
        )
        count = int(query_mask.shape[0])
        sample_index = torch.arange(
            offset, offset + count, device=device, dtype=torch.long
        )[:, None].expand_as(query_mask)
        mask = query_mask.to(torch.bool)

        def add(name: str, value: torch.Tensor) -> None:
            parts[name].append(value[mask].detach().cpu().numpy())

        add("sample_index", sample_index)
        add("tau_s", cached["tau_s"])
        add("target_switch_count", cached["target_switch_count"])
        add("selected_switch_step", selected_step)
        add("selection_correct", selected_step == cached["target_switch_count"])
        add("selected_probability", selected_probability)
        add("true_probability", true_probability)
        add("true_role_supported", true_support)
        add("conditional_error_m", conditional_error)
        add("hard_error_m", hard_error)
        repeated_current = current_error[:, None].expand_as(cached["tau_s"])
        add("upstream_current_error_m", repeated_current)
        sample_cpu = sample_index[mask].detach().cpu().numpy()
        parts["session_id"].append(session_ids[sample_cpu])
        parts["distance_m"].append(sample_distance_m[sample_cpu])
        parts["yaw_rate_rad_s"].append(sample_yaw_rate_rad_s[sample_cpu])
        parts["raw_pnp_rms_m"].append(sample_raw_pnp_rms_m[sample_cpu])
        offset += count
    return (
        {name: np.concatenate(values) for name, values in parts.items()},
        conditional_hasher.hexdigest(),
    )


def _trend_groups(
    x: np.ndarray, *, discrete: bool, bins: int = 10,
) -> list[np.ndarray]:
    if discrete:
        return [np.flatnonzero(x == value) for value in np.unique(x)]
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bins + 1)))
    groups: list[np.ndarray] = []
    for index in range(len(edges) - 1):
        upper = x <= edges[index + 1] if index == len(edges) - 2 else x < edges[index + 1]
        group = np.flatnonzero((x >= edges[index]) & upper)
        if group.size:
            groups.append(group)
    return groups


def _trend_stats(
    x: np.ndarray, y_mm: np.ndarray, *, discrete: bool,
) -> tuple[np.ndarray, ...]:
    groups = _trend_groups(x, discrete=discrete)
    centers = np.asarray([np.median(x[group]) for group in groups])
    quantiles = np.asarray([
        np.percentile(y_mm[group], [10, 25, 50, 75, 90]) for group in groups
    ])
    return (centers, *[quantiles[:, index] for index in range(5)])


def _plot_training(run_manifest: dict[str, Any], output: Path) -> None:
    history = run_manifest["history"]
    updates = np.asarray([item["update"] for item in history])
    accuracy = np.asarray([item["validation"]["switch_accuracy"] for item in history]) * 100
    recall = np.asarray([
        item["validation"]["minimum_step_switch_recall"] for item in history
    ]) * 100
    conditional = np.asarray([
        item["validation"]["conditional_position"]["p95_m"] for item in history
    ]) * 1000
    hard = np.asarray([
        item["validation"]["hard_routed_position"]["p95_m"] for item in history
    ]) * 1000
    warmup = int(run_manifest["training_arguments"]["selector_warmup_updates"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    axes[0].plot(updates, accuracy, "o-", color=COLORS["coral"], label="All-step accuracy")
    axes[0].plot(updates, recall, "s--", color=COLORS["blue"], label="One-step recall")
    axes[0].set(xlabel="Optimizer updates", ylabel="Selection metric (%)", title="Selection learning")
    axes[0].legend(loc="lower right")
    axes[1].plot(updates, conditional, "o-", color=COLORS["teal"], label="Conditional P95")
    axes[1].plot(updates, hard, "s-", color=COLORS["coral"], label="Hard-routed P95")
    axes[1].set(xlabel="Optimizer updates", ylabel="Position error (mm)", title="Validation error")
    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.axvline(warmup, color=COLORS["gray"], linestyle=":", linewidth=1.2)
        ax.text(warmup, 0.03, "joint update starts", rotation=90,
                transform=ax.get_xaxis_transform(), va="bottom", ha="right", fontsize=7)
    fig.suptitle("V64 training dynamics — combined-motion diagnostic validation", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, output, "fig_training_dynamics")


def _ecdf(values_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.maximum(values_mm, 0.05))
    return x, np.arange(1, x.size + 1, dtype=np.float64) / x.size


def _plot_distribution(data: dict[str, np.ndarray], output: Path) -> None:
    conditional = data["conditional_error_m"] * 1000
    hard = data["hard_error_m"] * 1000
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.8))
    for values, color, label in (
        (conditional, COLORS["teal"], "Conditional (correct branch)"),
        (hard, COLORS["coral"], "Hard-routed output"),
    ):
        x, y = _ecdf(values)
        axes[0].plot(x, y * 100, color=color, label=label)
    axes[0].set_xscale("log")
    axes[0].set(xlabel="Position error (mm, log scale)", ylabel="Queries covered (%)", title="Empirical CDF")
    axes[0].legend(loc="lower right")
    for values, color, label, marker in (
        (conditional, COLORS["teal"], "Conditional", "o"),
        (hard, COLORS["coral"], "Hard-routed", "s"),
    ):
        coverage = [(values <= threshold).mean() * 100 for threshold in THRESHOLDS_MM]
        axes[1].plot(THRESHOLDS_MM, coverage, marker=marker, color=color, label=label)
    axes[1].set(xlabel="Allowed error (mm)", ylabel="Coverage (%)", title="Operational coverage")
    axes[1].set_xticks(THRESHOLDS_MM)
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend(loc="lower right")
    upper = max(np.percentile(conditional, 99.9), np.percentile(hard, 99.9))
    bins = np.geomspace(0.1, max(upper, 1.0), 55)
    axes[2].hist(conditional.clip(0.1), bins=bins, histtype="step", density=True,
                 linewidth=1.8, color=COLORS["teal"], label="Conditional")
    axes[2].hist(hard.clip(0.1), bins=bins, histtype="step", density=True,
                 linewidth=1.8, color=COLORS["coral"], label="Hard-routed")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set(xlabel="Position error (mm, log scale)", ylabel="Density", title="Full distribution")
    axes[2].legend(loc="upper right")
    fig.suptitle("Final error distribution — tails retained, not used alone", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, output, "fig_error_distribution")


def _plot_trends(data: dict[str, np.ndarray], output: Path) -> None:
    fields = [
        (data["tau_s"], "Future time (s)", False),
        (np.abs(data["yaw_rate_rad_s"]) * 180 / np.pi, "Physical |yaw rate| (deg/s)", False),
        (data["distance_m"], "Physical distance (m)", False),
        (np.abs(data["target_switch_count"]), "Future switch count |step|", True),
        (data["upstream_current_error_m"] * 1000, "Mapper/S/H current error (mm)", False),
        (data["raw_pnp_rms_m"] * 1000, "Raw PnP observed-point RMS (mm)", False),
    ]
    conditional_mm = data["conditional_error_m"] * 1000
    hard_mm = data["hard_error_m"] * 1000
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.7), sharey=True)
    rng = np.random.default_rng(20260729)
    sample = rng.choice(len(hard_mm), size=min(5000, len(hard_mm)), replace=False)
    floor = 0.1
    for ax, (x, label, discrete) in zip(axes.flat, fields):
        ax.scatter(x[sample], np.maximum(hard_mm[sample], floor), s=4,
                   alpha=0.055, color=COLORS["gray"], rasterized=True)
        centers, q10, q25, q50, q75, q90 = _trend_stats(x, hard_mm, discrete=discrete)
        ax.fill_between(centers, np.maximum(q10, floor), np.maximum(q90, floor),
                        color=COLORS["gold"], alpha=0.18, linewidth=0, label="Hard 10–90%")
        ax.fill_between(centers, np.maximum(q25, floor), np.maximum(q75, floor),
                        color=COLORS["orange"], alpha=0.25, linewidth=0, label="Hard 25–75%")
        ax.plot(centers, np.maximum(q50, floor), color=COLORS["coral"], marker="o",
                markersize=3, label="Hard median")
        c_centers, _, _, c50, _, _ = _trend_stats(x, conditional_mm, discrete=discrete)
        ax.plot(c_centers, np.maximum(c50, floor), color=COLORS["teal"], linestyle="--",
                marker="s", markersize=2.5, label="Conditional median")
        ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("Position error (mm, log scale)")
    axes[0, 0].legend(loc="upper left", fontsize=7)
    fig.suptitle(
        "Error trends and empirical quantile bands — combined-motion validation",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output, "fig_error_trends")


def _plot_switch_analysis(data: dict[str, np.ndarray], output: Path) -> None:
    true_step = data["target_switch_count"].astype(np.int64)
    selected_step = data["selected_switch_step"].astype(np.int64)
    correct = data["selection_correct"].astype(np.bool_)
    hard_mm = data["hard_error_m"] * 1000
    steps = np.arange(min(true_step.min(), selected_step.min()), max(true_step.max(), selected_step.max()) + 1)
    matrix = np.zeros((len(steps), len(steps)), dtype=np.float64)
    for row, target in enumerate(steps):
        role = true_step == target
        if role.any():
            for column, selected in enumerate(steps):
                matrix[row, column] = np.mean(selected_step[role] == selected)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2))
    true_steps = np.unique(true_step)
    role_accuracy = np.asarray([correct[true_step == step].mean() * 100 for step in true_steps])
    role_count = np.asarray([(true_step == step).sum() for step in true_steps])
    axes[0, 0].bar(true_steps, role_accuracy, color=COLORS["coral"], width=0.72)
    axes[0, 0].set(xlabel="True signed switch step", ylabel="Selection accuracy (%)", title="Recall by future role")
    axes[0, 0].set_xticks(true_steps)
    for x, y, count in zip(true_steps, role_accuracy, role_count):
        axes[0, 0].text(x, min(y + 2, 103), str(count), ha="center", fontsize=6, color="#555555")
    image = axes[0, 1].imshow(matrix * 100, origin="lower", cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    axes[0, 1].set_xticks(np.arange(len(steps)), steps)
    axes[0, 1].set_yticks(np.arange(len(steps)), steps)
    axes[0, 1].set(xlabel="Selected step", ylabel="True step", title="Row-normalized confusion (%)")
    fig.colorbar(image, ax=axes[0, 1], shrink=0.78)
    absolute_steps = np.unique(np.abs(true_step))
    p50 = [np.percentile(hard_mm[np.abs(true_step) == step], 50) for step in absolute_steps]
    p90 = [np.percentile(hard_mm[np.abs(true_step) == step], 90) for step in absolute_steps]
    p95 = [np.percentile(hard_mm[np.abs(true_step) == step], 95) for step in absolute_steps]
    axes[1, 0].plot(absolute_steps, p50, "o-", color=COLORS["teal"], label="P50")
    axes[1, 0].plot(absolute_steps, p90, "s-", color=COLORS["orange"], label="P90")
    axes[1, 0].plot(absolute_steps, p95, "^-", color=COLORS["coral"], label="P95")
    axes[1, 0].set(xlabel="Absolute future switch count", ylabel="Hard error (mm)", title="Error growth with switching")
    axes[1, 0].legend()
    probability = data["selected_probability"]
    probability_edges = np.linspace(0, 1, 11)
    centers: list[float] = []
    observed: list[float] = []
    counts: list[int] = []
    for low, high in zip(probability_edges[:-1], probability_edges[1:]):
        role = (probability >= low) & ((probability <= high) if high == 1 else (probability < high))
        if int(role.sum()) >= 30:
            centers.append(float(probability[role].mean()))
            observed.append(float(correct[role].mean()))
            counts.append(int(role.sum()))
    axes[1, 1].plot([0, 1], [0, 1], color=COLORS["gray"], linestyle=":", label="Perfect calibration")
    axes[1, 1].plot(centers, observed, "o-", color=COLORS["blue"], label="Observed accuracy")
    axes[1, 1].set(
        xlabel="Selected-candidate probability", ylabel="Empirical accuracy",
        title="Selector calibration (bins n≥30)",
    )
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    for x, y, count in zip(centers, observed, counts):
        axes[1, 1].text(x, y + 0.035, str(count), ha="center", fontsize=6, color="#555555")
    axes[1, 1].legend(loc="upper left", fontsize=7)
    fig.suptitle("Selector structure — counts label each bin", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, output, "fig_switch_analysis")


def _metric_view(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        metrics["conditional_position"]["p95_m"] * 1000,
        metrics["hard_routed_position"]["p95_m"] * 1000,
        metrics["switch_accuracy"] * 100,
        metrics["minimum_step_switch_recall"] * 100,
    )


def _plot_comparison(
    trajectory_checkpoint: dict[str, Any], selector_manifest: dict[str, Any] | None,
    final_checkpoint: dict[str, Any], output: Path,
) -> dict[str, dict[str, float]]:
    methods: list[tuple[str, dict[str, Any]]] = [
        ("V50\nTrajectory", trajectory_checkpoint["validation"]["pnp_domain"]),
    ]
    if selector_manifest is not None:
        methods.append(("V52\nStaged selector", selector_manifest["best"]["validation"]["pnp_domain"]))
    methods.append(("V64\nJoint", final_checkpoint["validation"]))
    names = [item[0] for item in methods]
    values = np.asarray([_metric_view(item[1]) for item in methods])
    titles = ["Conditional P95", "Hard-routed P95", "Selection accuracy", "One-step recall"]
    units = ["Error (mm)", "Error (mm)", "Accuracy (%)", "Recall (%)"]
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.8))
    colors = [COLORS["gray"], COLORS["teal"], COLORS["coral"]][-len(names):]
    for column, ax in enumerate(axes):
        bars = ax.bar(np.arange(len(names)), values[:, column], color=colors, width=0.68)
        ax.set_xticks(np.arange(len(names)), names)
        ax.set_ylabel(units[column])
        ax.set_title(titles[column])
        for bar, value in zip(bars, values[:, column]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}",
                    ha="center", va="bottom", fontsize=7)
    fig.suptitle("Same diagnostic validation: staged baseline versus joint training", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, output, "fig_baseline_comparison")
    return {
        name.replace("\n", " "): {
            "conditional_p95_mm": float(row[0]), "hard_p95_mm": float(row[1]),
            "switch_accuracy_pct": float(row[2]), "one_step_recall_pct": float(row[3]),
        }
        for name, row in zip(names, values)
    }


def _bootstrap_ci(
    values: np.ndarray,
    sessions: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    repetitions: int = 500,
) -> list[float]:
    unique = np.unique(sessions)
    groups = [values[sessions == session] for session in unique]
    rng = np.random.default_rng(20260729)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        selected = rng.integers(0, len(groups), size=len(groups))
        estimates[index] = statistic(np.concatenate([groups[item] for item in selected]))
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def _summary(
    data: dict[str, np.ndarray], comparison: dict[str, dict[str, float]],
    checkpoint_path: Path, run_manifest: dict[str, Any], truth_root: Path,
) -> dict[str, Any]:
    conditional = data["conditional_error_m"]
    hard = data["hard_error_m"]
    correct = data["selection_correct"].astype(np.bool_)
    sessions = data["session_id"]
    coverage = {
        str(int(threshold)): {
            "conditional": float(np.mean(conditional * 1000 <= threshold)),
            "hard": float(np.mean(hard * 1000 <= threshold)),
        }
        for threshold in THRESHOLDS_MM
    }
    per_step: dict[str, Any] = {}
    for step in np.unique(data["target_switch_count"]):
        role = data["target_switch_count"] == step
        per_step[str(int(step))] = {
            "count": int(role.sum()),
            "accuracy": float(correct[role].mean()),
            "conditional": _percentiles(conditional[role]),
            "hard": _percentiles(hard[role]),
        }
    return {
        "schema_version": "joint-visible-future-figures-v1",
        "scope": "combined-motion diagnostic validation only",
        "oracle_association": True,
        "untouched_test": False,
        "test_accessed": False,
        "query_count": int(len(hard)),
        "window_count": int(len(np.unique(data["sample_index"]))),
        "session_count": int(len(np.unique(sessions))),
        "final_checkpoint": str(checkpoint_path),
        "final_checkpoint_sha256": sha256_file(checkpoint_path),
        "truth_history": str(truth_root),
        "truth_context_forward_input": False,
        "conditional": _percentiles(conditional),
        "hard": _percentiles(hard),
        "selection_accuracy": float(correct.mean()),
        "one_step_recall": float(correct[np.abs(data["target_switch_count"]) == 1].mean()),
        "cluster_bootstrap_95pct": {
            "selection_accuracy": _bootstrap_ci(correct.astype(np.float64), sessions, np.mean),
            "conditional_p95_m": _bootstrap_ci(conditional, sessions, lambda value: float(np.percentile(value, 95))),
            "hard_p95_m": _bootstrap_ci(hard, sessions, lambda value: float(np.percentile(value, 95))),
        },
        "coverage_by_allowed_error_mm": coverage,
        "correct_selection": {
            "count": int(correct.sum()), "hard": _percentiles(hard[correct]),
        },
        "wrong_selection": {
            "count": int((~correct).sum()), "hard": _percentiles(hard[~correct]),
        },
        "per_signed_switch_step": per_step,
        "comparison": comparison,
        "training_update": int(run_manifest["update"]),
        "notes": [
            "Empirical quantile bands describe the validation distribution; they are not IID confidence intervals.",
            "Cluster bootstrap intervals resample the 23 validation sessions.",
            "Distance and yaw rate are exact physical truth joined after inference and never passed to the model.",
            "The validation split was repeatedly observed during development and is not untouched acceptance evidence.",
            "Translation, rotation-only and stationary comparisons are unavailable because this run filters motion_class=3.",
        ],
    }


def generate(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir).resolve()
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = _read_json(run_manifest_path)
    if run_manifest.get("status") != "complete" or run_manifest.get("update") != 3000:
        raise ValueError("joint run is not the completed fixed 3000-update run")
    progress = _read_json(run_dir / "run_progress.json")
    checkpoint_path = run_dir / progress["latest_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["update"] != run_manifest["update"]:
        raise ValueError("latest checkpoint and run manifest differ")
    training_args = run_manifest["training_arguments"]
    dataset_root = Path(training_args["dataset"]).resolve()
    dataset_manifest = _read_json(dataset_root / "dataset_manifest.json")
    if dataset_manifest.get("test_accessed", True):
        raise ValueError("prediction dataset reports test access")
    truth_root = Path(dataset_manifest["truth_history_dataset"]).resolve()
    pair_ids, session_ids, t0_values = _load_validation_identity(dataset_root)
    dataset = ObservableFuturePnPSFDataset(
        dataset_root, "validation", motion_class=3, allow_diagnostic=False,
    )
    if tuple(str(value) for value in pair_ids) != dataset.pair_ids:
        raise ValueError("validation identity order differs from materialized dataset")
    canonicalize_direction_keep_c4(dataset.tensors, dataset.pair_ids)
    distance_m, yaw_rate_rad_s = _load_truth_context(truth_root, session_ids, t0_values)
    raw_mask = dataset.tensors["pnp_s_obs_mask"].to(torch.bool)
    raw_error = torch.linalg.vector_norm(
        dataset.tensors["pnp_s_obs_m"] - dataset.tensors["clean_s_obs_m"], dim=-1
    )
    raw_pnp_rms_m = np.empty(len(dataset), dtype=np.float64)
    for index in range(len(dataset)):
        values = raw_error[index][raw_mask[index]].numpy().astype(np.float64)
        raw_pnp_rms_m[index] = float(np.sqrt(np.mean(values ** 2)))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for figure evaluation but is unavailable")
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
        raise ValueError("re-evaluated conditional P95 differs from training")
    if abs(hard_metrics["p95_m"] - expected["hard_routed_position"]["p95_m"]) > 1e-7:
        raise ValueError("re-evaluated hard P95 differs from training")
    if conditional_hash != expected["conditional_output_sha256"]:
        raise ValueError("re-evaluated conditional tensor hash differs from training")

    output = (
        Path(args.output_dir).resolve()
        if args.output_dir else run_dir / "figures"
    )
    output.mkdir(exist_ok=False)
    _style()
    _plot_training(run_manifest, output)
    _plot_distribution(queries, output)
    _plot_trends(queries, output)
    _plot_switch_analysis(queries, output)
    trajectory_checkpoint = torch.load(
        trajectory_parent, map_location="cpu", weights_only=False,
    )
    selector_manifest = (
        _read_json(Path(args.selector_baseline_manifest).resolve())
        if args.selector_baseline_manifest else None
    )
    comparison = _plot_comparison(
        trajectory_checkpoint, selector_manifest, checkpoint, output,
    )
    np.savez_compressed(output / "validation_queries.npz", **queries)
    summary = _summary(
        queries, comparison, checkpoint_path, run_manifest, truth_root,
    )
    (output / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-dir", required=True)
    result.add_argument("--output-dir", default="")
    result.add_argument("--selector-baseline-manifest", default="")
    result.add_argument("--device", default="cuda")
    return result


def main() -> None:
    print(generate(parser().parse_args()))


if __name__ == "__main__":
    main()
