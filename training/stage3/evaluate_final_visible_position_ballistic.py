"""Evaluate frozen V67 at a causal range-derived projectile flight time."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .build_observable_future_dataset import _dense_times, _rollout_dense_truth
from .cyclic_future_foundation import load_frozen_v19
from .evaluate_final_visible_position_generalization import (
    MOTION_CLASSES,
    _datasets_for_motion,
    _load_system,
    _manifest_sessions,
    _source_coverage,
)
from .observable_future_dataset import construct_observable_future_sample
from .observable_future_pnp_ab import (
    ObservableFuturePnPSFDataset,
    sha256_file,
    state_dict_sha256,
)
from .pnp_q0_hypothesis_adapter import (
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .train_causal_physical_ab import _git_state, _to_device
from .train_final_visible_position_refiner import _position_metrics
from .train_joint_visible_future import (
    CachedJointFutureDataset,
    _build_cache,
    _loss_batch,
    _model_batch,
)
from .train_pnp_window_mapper_distillation import _atomic_json


EVALUATION_SCHEMA = "stage3-final-position-ballistic-generalization-v1"
FLIGHT_TIME_FORMULA = "norm(frozen_upstream_current_position_m)/bullet_speed_mps"
DISTANCE_EDGES_M = tuple(float(value) for value in range(1, 8))
DISPLAY_BODY_PERCENTILE = 95.0


@dataclass(frozen=True)
class TruthState:
    history_position_m: np.ndarray
    event_mask: np.ndarray
    event_time_s: np.ndarray
    q0_position_m: np.ndarray
    center_m: np.ndarray
    velocity_mps: np.ndarray
    yaw_rate_rad_s: float


def _resolve_shard(root: Path, value: object) -> Path:
    return root / Path(str(value).replace("\\", "/"))


def _load_truth_states(
    sf_dataset: Path,
    required_keys: set[tuple[str, int]],
) -> tuple[dict[tuple[str, int], TruthState], dict[str, Any]]:
    sf_manifest = json.loads(
        (sf_dataset / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    causal_dir = Path(str(sf_manifest["causal_physical_dataset"])).resolve()
    causal_manifest = json.loads(
        (causal_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    truth_dir = Path(str(causal_manifest["source_truth_history"])).resolve()
    truth_manifest_path = truth_dir / "dataset_manifest.json"
    truth_manifest = json.loads(truth_manifest_path.read_text(encoding="utf-8"))
    if bool(truth_manifest.get("test_accessed", True)):
        raise ValueError("ballistic evaluation refuses truth history that accessed test")

    states: dict[tuple[str, int], TruthState] = {}
    for shard in truth_manifest["shards"]:
        if str(shard["split"]) not in {"train", "validation"}:
            continue
        path = _resolve_shard(truth_dir, shard["path"])
        if sha256_file(path) != str(shard["sha256"]):
            raise ValueError(f"truth-history shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as loaded:
            for row, (session_id, t0_ns) in enumerate(
                zip(loaded["session_id"], loaded["t0_ns"])
            ):
                key = (str(session_id), int(t0_ns))
                if key not in required_keys:
                    continue
                if key in states:
                    raise ValueError(f"duplicate truth-history state: {key}")
                tau = loaded["tau"][row]
                zero = np.flatnonzero(tau == 0.0)
                if zero.size < 1:
                    raise ValueError(f"truth-history state has no q0 query: {key}")
                states[key] = TruthState(
                    history_position_m=loaded["truth_obs"][row].copy(),
                    event_mask=loaded["event_mask"][row].copy(),
                    event_time_s=loaded["event_time_s"][row].copy(),
                    q0_position_m=loaded["future_position"][row, int(zero[0])].copy(),
                    center_m=loaded["anchor_center_position_m"][row].copy(),
                    velocity_mps=loaded["anchor_velocity_mps"][row].copy(),
                    yaw_rate_rad_s=float(loaded["anchor_yaw_rate_rad_s"][row]),
                )
    missing = sorted(required_keys - set(states))
    if missing:
        raise ValueError(f"missing {len(missing)} ballistic truth states; first={missing[0]}")
    return states, {
        "path": str(truth_dir),
        "manifest_path": str(truth_manifest_path),
        "manifest_sha256": sha256_file(truth_manifest_path),
        "test_accessed": False,
        "state_count": len(states),
    }


def _dataset_metadata(
    datasets: Iterable[Dataset],
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]]:
    sessions: list[str] = []
    t0_values: list[int] = []
    pair_ids: list[str] = []
    for dataset in datasets:
        if not isinstance(dataset, ObservableFuturePnPSFDataset):
            raise TypeError("ballistic metadata requires the paired PnP S/F dataset")
        sessions.extend(dataset.session_ids)
        t0_values.extend(dataset.t0_ns)
        pair_ids.extend(dataset.pair_ids)
    if not (len(sessions) == len(t0_values) == len(pair_ids)):
        raise RuntimeError("ballistic metadata arrays are misaligned")
    return tuple(sessions), tuple(t0_values), tuple(pair_ids)


def _ballistic_label(
    state: TruthState,
    model_current_position_m: np.ndarray,
    *,
    reverse_direction: bool,
    bullet_speed_mps: float,
    dense_step_s: float,
) -> dict[str, Any]:
    estimated_distance_m = float(np.linalg.norm(model_current_position_m))
    flight_time_s = estimated_distance_m / bullet_speed_mps
    query = np.asarray([0.0, flight_time_s], dtype=np.float32)
    dense_time = _dense_times(query, dense_step_s)
    dense_position = _rollout_dense_truth(
        state.q0_position_m,
        state.center_m,
        state.velocity_mps,
        state.yaw_rate_rad_s,
        dense_time,
    )
    sample = construct_observable_future_sample(
        state.history_position_m,
        state.event_mask,
        state.event_time_s,
        dense_position,
        dense_time,
        query,
        np.ones(2, dtype=np.bool_),
        query_match_tolerance_s=1e-6,
    )
    switch_count = int(sample["target_switch_count"][1])
    if reverse_direction:
        switch_count *= -1
    truth_current = sample["current_position_m"].astype(np.float32, copy=False)
    delta = sample["target_visible_delta_m"][1].astype(np.float32, copy=False)
    return {
        "estimated_distance_m": estimated_distance_m,
        "flight_time_s": flight_time_s,
        "truth_current_position_m": truth_current,
        "truth_distance_m": float(np.linalg.norm(truth_current)),
        "target_visible_delta_m": delta,
        "future_truth_distance_m": float(np.linalg.norm(truth_current + delta)),
        "target_switch_count": switch_count,
    }


def _dynamic_cache(
    cache: CachedJointFutureDataset,
    sessions: tuple[str, ...],
    t0_values: tuple[int, ...],
    pair_ids: tuple[str, ...],
    truth_states: dict[tuple[str, int], TruthState],
    *,
    bullet_speed_mps: float,
    dense_step_s: float,
) -> tuple[CachedJointFutureDataset, dict[str, np.ndarray], dict[str, Any]]:
    count = len(cache)
    if not (count == len(sessions) == len(t0_values) == len(pair_ids)):
        raise ValueError("cache and ballistic metadata counts differ")
    model_current = cache.tensors["current_position_m"].numpy()
    cached_truth_current = cache.tensors["truth_current_position_m"].numpy()
    labels = [
        _ballistic_label(
            truth_states[(sessions[row], t0_values[row])],
            model_current[row],
            reverse_direction=bool((int(pair_ids[row][:16], 16) >> 2) & 1),
            bullet_speed_mps=bullet_speed_mps,
            dense_step_s=dense_step_s,
        )
        for row in range(count)
    ]
    rebuilt_truth_current = np.stack(
        [label["truth_current_position_m"] for label in labels]
    )
    q0_error = np.linalg.norm(rebuilt_truth_current - cached_truth_current, axis=1)
    if float(q0_error.max(initial=0.0)) > 5e-5:
        raise ValueError(
            f"ballistic truth and frozen cache q0 differ: max={q0_error.max()} m"
        )

    tensors = dict(cache.tensors)
    tensors["tau_s"] = torch.from_numpy(
        np.asarray([label["flight_time_s"] for label in labels], dtype=np.float32)[:, None]
    )
    tensors["target_switch_count"] = torch.from_numpy(
        np.asarray([label["target_switch_count"] for label in labels], dtype=np.int64)[:, None]
    )
    tensors["target_visible_delta_m"] = torch.from_numpy(
        np.stack([label["target_visible_delta_m"] for label in labels])[:, None, :]
    )
    tensors["target_query_mask"] = torch.ones((count, 1), dtype=torch.bool)
    dynamic = CachedJointFutureDataset(tensors)
    metadata = {
        "session_id": np.asarray(sessions),
        "t0_ns": np.asarray(t0_values, dtype=np.int64),
        "estimated_distance_m": np.asarray(
            [label["estimated_distance_m"] for label in labels], dtype=np.float32,
        ),
        "truth_distance_m": np.asarray(
            [label["truth_distance_m"] for label in labels], dtype=np.float32,
        ),
        "future_truth_distance_m": np.asarray(
            [label["future_truth_distance_m"] for label in labels], dtype=np.float32,
        ),
        "flight_time_s": np.asarray(
            [label["flight_time_s"] for label in labels], dtype=np.float32,
        ),
        "target_switch_count": np.asarray(
            [label["target_switch_count"] for label in labels], dtype=np.int64,
        ),
    }
    audit = {
        "sample_count": count,
        "q0_reconstruction_max_m": float(q0_error.max(initial=0.0)),
        "estimated_distance_min_m": float(metadata["estimated_distance_m"].min()),
        "estimated_distance_max_m": float(metadata["estimated_distance_m"].max()),
        "truth_distance_min_m": float(metadata["truth_distance_m"].min()),
        "truth_distance_max_m": float(metadata["truth_distance_m"].max()),
        "flight_time_min_s": float(metadata["flight_time_s"].min()),
        "flight_time_max_s": float(metadata["flight_time_s"].max()),
    }
    return dynamic, metadata, audit


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    dataset: CachedJointFutureDataset,
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    refined_errors: list[np.ndarray] = []
    baseline_errors: list[np.ndarray] = []
    residual_norms: list[np.ndarray] = []
    predicted_switch: list[np.ndarray] = []
    selected_probability: list[np.ndarray] = []
    model.eval()
    for raw_batch in loader:
        cached = _to_device(raw_batch, device)
        loss_batch = _loss_batch(cached)
        prediction = model(_model_batch(loss_batch))
        query_mask = loss_batch["target_query_mask"].to(torch.bool)
        target_position = (
            loss_batch["current_position_m"][:, None]
            + loss_batch["target_visible_delta_m"]
        )
        refined = torch.linalg.vector_norm(
            prediction["position_m"] - target_position, dim=-1,
        )
        baseline = torch.linalg.vector_norm(
            prediction["unrefined_position_m"] - target_position, dim=-1,
        )
        residual = torch.linalg.vector_norm(
            prediction["position_residual_m"], dim=-1,
        )
        selected_row = prediction["selected_candidate_row"]
        step = cached["candidate_step"].gather(1, selected_row)
        probability = prediction["switch_probability"].gather(
            2, selected_row.unsqueeze(-1),
        ).squeeze(-1)
        refined_errors.append(refined[query_mask].cpu().numpy())
        baseline_errors.append(baseline[query_mask].cpu().numpy())
        residual_norms.append(residual[query_mask].cpu().numpy())
        predicted_switch.append(step[query_mask].cpu().numpy())
        selected_probability.append(probability[query_mask].cpu().numpy())

    refined_m = np.concatenate(refined_errors).astype(np.float32)
    baseline_m = np.concatenate(baseline_errors).astype(np.float32)
    residual_m = np.concatenate(residual_norms).astype(np.float32)
    predicted_step = np.concatenate(predicted_switch).astype(np.int64)
    probability = np.concatenate(selected_probability).astype(np.float32)
    truth_step = dataset.tensors["target_switch_count"].numpy().reshape(-1)
    if refined_m.size != len(dataset):
        raise RuntimeError("ballistic evaluation must produce one query per window")
    correct_selection = predicted_step == truth_step
    metrics = {
        "final_position": _position_metrics(refined_m),
        "frozen_v66_baseline": _position_metrics(baseline_m),
        "position_residual": _position_metrics(residual_m),
        "frozen_switch_accuracy": float(np.mean(correct_selection)),
        "selection_stratified_final_position": {
            "correct": _position_metrics(refined_m[correct_selection]),
            "wrong": (
                _position_metrics(refined_m[~correct_selection])
                if bool((~correct_selection).any()) else None
            ),
        },
        "selected_probability": {
            "mean": float(probability.mean()),
            "p50": float(np.percentile(probability, 50)),
            "p90": float(np.percentile(probability, 90)),
        },
    }
    return metrics, {
        "final_error_m": refined_m,
        "frozen_v66_error_m": baseline_m,
        "position_residual_m": residual_m,
        "predicted_switch_count": predicted_step,
        "selected_probability": probability,
    }


def _table_rows(queries: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    distance = queries["truth_distance_m"].astype(np.float64)
    estimated = queries["estimated_distance_m"].astype(np.float64)
    error_mm = queries["final_error_m"].astype(np.float64) * 1000.0
    flight_time = queries["flight_time_s"].astype(np.float64)
    rows: list[dict[str, Any]] = []
    bins = [
        (f"[{left:.0f},{right:.0f})", left, right)
        for left, right in zip(DISTANCE_EDGES_M[:-1], DISTANCE_EDGES_M[1:])
    ]
    bins.append(("[1,7) overall", 1.0, 7.0))
    for label, left, right in bins:
        mask = (distance >= left) & (distance < right)
        values = error_mm[mask]
        row: dict[str, Any] = {
            "distance_bin_m": label,
            "count": int(mask.sum()),
            "descriptive_only": bool(mask.sum() < 100),
        }
        if values.size:
            row.update({
                "mean_error_mm": float(values.mean()),
                "p50_error_mm": float(np.percentile(values, 50)),
                "p90_error_mm": float(np.percentile(values, 90)),
                "p95_error_mm": float(np.percentile(values, 95)),
                "p99_error_mm": float(np.percentile(values, 99)),
                "coverage_le_50mm": float(np.mean(values <= 50.0)),
                "coverage_le_100mm": float(np.mean(values <= 100.0)),
                "coverage_le_200mm": float(np.mean(values <= 200.0)),
                "coverage_le_300mm": float(np.mean(values <= 300.0)),
                "mean_flight_time_s": float(flight_time[mask].mean()),
                "mean_abs_range_error_mm": float(
                    np.abs(estimated[mask] - distance[mask]).mean() * 1000.0
                ),
            })
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "distance_bin_m", "count", "descriptive_only", "mean_error_mm",
        "p50_error_mm", "p90_error_mm", "p95_error_mm", "p99_error_mm",
        "coverage_le_50mm", "coverage_le_100mm", "coverage_le_200mm",
        "coverage_le_300mm", "mean_flight_time_s", "mean_abs_range_error_mm",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, motion_name: str, queries: dict[str, np.ndarray]) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distance = queries["truth_distance_m"].astype(np.float64)
    error_mm = queries["final_error_m"].astype(np.float64) * 1000.0
    cap_mm = max(50.0, float(np.percentile(error_mm, DISPLAY_BODY_PERCENTILE)))
    overflow = error_mm > cap_mm
    plotted = np.minimum(error_mm, cap_mm)
    figure, axis = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    axis.scatter(
        distance[~overflow], plotted[~overflow], s=12, alpha=0.30,
        color="#2878B5", linewidths=0, rasterized=True, label="V67 query",
    )
    if bool(overflow.any()):
        axis.scatter(
            distance[overflow], plotted[overflow], s=28, alpha=0.9,
            marker="^", color="#D62728", linewidths=0,
            label=f">P{DISPLAY_BODY_PERCENTILE:g} display cap (n={int(overflow.sum())})",
        )
    centers: list[float] = []
    medians: list[float] = []
    for left, right in zip(DISTANCE_EDGES_M[:-1], DISTANCE_EDGES_M[1:]):
        mask = (distance >= left) & (distance < right)
        if bool(mask.any()):
            centers.append(0.5 * (left + right))
            medians.append(float(np.median(error_mm[mask])))
    axis.scatter(
        centers, medians, color="#F28E2B", marker="D", s=42,
        linewidths=0, label="1 m-bin median",
    )
    axis.set_xlim(1.0, 7.0)
    axis.set_ylim(0.0, cap_mm * 1.04)
    axis.set_xticks(DISTANCE_EDGES_M)
    axis.set_xlabel("Current target truth distance (m)")
    axis.set_ylabel("Future position error at flight time (mm)")
    axis.set_title(
        f"{motion_name.capitalize()} motion | range-derived flight time | n={error_mm.size}"
    )
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper left", frameon=True)
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "display_cap_mm": cap_mm,
        "display_body_percentile": DISPLAY_BODY_PERCENTILE,
        "overflow_count": int(overflow.sum()),
    }


def _write_markdown(
    path: Path,
    tables: dict[str, list[dict[str, Any]]],
    bullet_speed_mps: float,
) -> None:
    lines = [
        "# Fixed-6-mm ballistic-time frozen V67 evaluation",
        "",
        f"Flight-time query: `frozen upstream current range / {bullet_speed_mps:g} m/s`.",
        "The network receives only the resulting continuous query time; truth is label/analysis only.",
        "",
    ]
    for motion_name, rows in tables.items():
        lines.extend([
            f"## {motion_name.capitalize()}", "",
            "| Distance (m) | N | Mean (mm) | P50 | P90 | P95 | <=100 mm | <=300 mm |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            if int(row["count"]) == 0:
                lines.append(f"| {row['distance_bin_m']} | 0 | - | - | - | - | - | - |")
                continue
            lines.append(
                "| {distance_bin_m} | {count} | {mean_error_mm:.2f} | "
                "{p50_error_mm:.2f} | {p90_error_mm:.2f} | {p95_error_mm:.2f} | "
                "{coverage_le_100mm:.2%} | {coverage_le_300mm:.2%} |".format(**row)
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    if args.bullet_speed_mps <= 0.0 or args.dense_step_ms <= 0.0:
        raise ValueError("bullet speed and dense step must be positive")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite ballistic evaluation: {output}")
    output.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    git = _git_state()
    if bool(git["worktree_dirty"]):
        raise RuntimeError("ballistic evaluation requires a clean committed worktree")

    dataset_path = Path(args.dataset).resolve()
    dataset_manifest_path = dataset_path / "dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if not bool(dataset_manifest.get("qualification_passed", False)):
        raise ValueError("ballistic evaluation requires a qualified dataset")
    if bool(dataset_manifest.get("test_accessed", True)):
        raise ValueError("ballistic evaluation refuses a dataset that accessed test")

    system, system_provenance = _load_system(
        args.refiner_checkpoint,
        args.trajectory_checkpoint,
        args.selector_checkpoint,
    )
    original_manifest_path = Path(
        str(system_provenance["provenance"]["dataset_manifest_path"])
    ).resolve()
    original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    overlap = sorted(
        _manifest_sessions(original_manifest) & _manifest_sessions(dataset_manifest)
    )
    if overlap:
        raise ValueError("ballistic generalization sessions overlap V67 training data")

    mapper, mapper_provenance = load_frozen_pnp_mapper(args.mapper_checkpoint)
    s_model, s_provenance = load_frozen_v19(args.s_checkpoint)
    h_model, h_provenance = load_frozen_hypothesis_adapter(
        args.h_checkpoint, allow_diagnostic=True,
    )
    expected = system_provenance["provenance"]
    bindings = {
        "mapper": (
            expected.get("mapper", {}).get("state_dict_sha256"),
            mapper_provenance["state_dict_sha256"],
        ),
        "s": (
            expected.get("s", {}).get("state_dict_sha256"),
            s_provenance["state_dict_sha256"],
        ),
        "h": (
            expected.get("h", {}).get("state_dict_sha256"),
            h_provenance["state_dict_sha256"],
        ),
    }
    mismatch = [name for name, values in bindings.items() if values[0] != values[1]]
    if mismatch:
        raise ValueError("ballistic upstream binding differs: " + ", ".join(mismatch))
    for frozen in (mapper, s_model, h_model, system):
        frozen.to(device).eval().requires_grad_(False)

    frozen_initial = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
        "system": state_dict_sha256(system.state_dict()),
    }
    started = time.time()
    datasets_by_motion: dict[str, list[Dataset]] = {}
    audits_by_motion: dict[str, Any] = {}
    required_keys: set[tuple[str, int]] = set()
    for motion_class, motion_name in MOTION_CLASSES.items():
        datasets, dataset_audit = _datasets_for_motion(dataset_path, motion_class)
        datasets_by_motion[motion_name] = datasets
        audits_by_motion[motion_name] = dataset_audit
        sessions, t0_values, _ = _dataset_metadata(datasets)
        required_keys.update(zip(sessions, t0_values))
    truth_states, truth_provenance = _load_truth_states(dataset_path, required_keys)

    per_motion: dict[str, Any] = {}
    all_queries: dict[str, list[np.ndarray]] = {}
    tables: dict[str, list[dict[str, Any]]] = {}
    dense_step_s = args.dense_step_ms / 1000.0
    for motion_class, motion_name in MOTION_CLASSES.items():
        datasets = datasets_by_motion[motion_name]
        source: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        cache, cache_manifest = _build_cache(
            source, mapper, s_model, h_model,
            device=device, batch_size=args.cache_batch_size,
        )
        sessions, t0_values, pair_ids = _dataset_metadata(datasets)
        dynamic, metadata, label_audit = _dynamic_cache(
            cache, sessions, t0_values, pair_ids, truth_states,
            bullet_speed_mps=args.bullet_speed_mps,
            dense_step_s=dense_step_s,
        )
        metrics, predictions = _evaluate(
            system, dynamic, device, batch_size=args.batch_size,
        )
        queries = {**metadata, **predictions}
        queries["motion_class"] = np.full(
            len(dynamic), motion_class, dtype=np.int64,
        )
        np.savez_compressed(output / f"{motion_name}_queries.npz", **queries)
        table = _table_rows(queries)
        table_path = output / f"{motion_name}_distance_table.csv"
        _write_csv(table_path, table)
        plot = _plot(output / f"{motion_name}_distance_error.png", motion_name, queries)
        tables[motion_name] = table
        per_motion[motion_name] = {
            "dataset": audits_by_motion[motion_name],
            "cache": cache_manifest,
            "ballistic_label_audit": label_audit,
            "metrics": metrics,
            "queries": str(output / f"{motion_name}_queries.npz"),
            "queries_sha256": sha256_file(output / f"{motion_name}_queries.npz"),
            "distance_table": str(table_path),
            "distance_table_sha256": sha256_file(table_path),
            "plot": plot,
        }
        for name, value in queries.items():
            all_queries.setdefault(name, []).append(value)

    merged_queries = {
        name: np.concatenate(values, axis=0) for name, values in all_queries.items()
    }
    np.savez_compressed(output / "all_ballistic_queries.npz", **merged_queries)
    _write_markdown(
        output / "ballistic_summary.md", tables, args.bullet_speed_mps,
    )
    raw_manifest_path = Path(args.raw_manifest).resolve()
    source_coverage = _source_coverage(dataset_manifest, raw_manifest_path)
    frozen_final = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
        "system": state_dict_sha256(system.state_dict()),
    }
    result = {
        "schema_version": EVALUATION_SCHEMA,
        "status": "complete",
        "evaluation_only": True,
        "model_updated": False,
        "test_accessed": False,
        "elapsed_s": time.time() - started,
        "ballistic_query": {
            "bullet_speed_mps": args.bullet_speed_mps,
            "flight_time_formula": FLIGHT_TIME_FORMULA,
            "range_source": "frozen_mapper_s_h_current_visible_armor_position",
            "neural_input_change": "continuous_tau_query_only",
            "truth_used_as_network_input": False,
            "plot_x_axis": "exact_q0_current_visible_armor_distance_m",
            "dense_truth_step_s": dense_step_s,
        },
        "new_dataset": {
            "path": str(dataset_path),
            "manifest_path": str(dataset_manifest_path),
            "manifest_sha256": sha256_file(dataset_manifest_path),
            "raw_manifest_path": str(raw_manifest_path),
            "raw_manifest_sha256": sha256_file(raw_manifest_path),
            "session_overlap_count": len(overlap),
            "source_coverage": source_coverage,
        },
        "truth_labels": truth_provenance,
        "model": system_provenance,
        "upstream": {
            "mapper": mapper_provenance,
            "s": s_provenance,
            "h": h_provenance,
        },
        "per_motion": per_motion,
        "overall": {
            "sample_count": int(merged_queries["final_error_m"].size),
            "final_position": _position_metrics(merged_queries["final_error_m"]),
            "frozen_v66_baseline": _position_metrics(
                merged_queries["frozen_v66_error_m"]
            ),
        },
        "all_queries": str(output / "all_ballistic_queries.npz"),
        "all_queries_sha256": sha256_file(output / "all_ballistic_queries.npz"),
        "summary_markdown": str(output / "ballistic_summary.md"),
        "summary_markdown_sha256": sha256_file(output / "ballistic_summary.md"),
        "frozen_initial_state_dict_sha256": frozen_initial,
        "frozen_final_state_dict_sha256": frozen_final,
        "all_frozen_states_verified_unchanged": frozen_initial == frozen_final,
        "git": git,
        "source_path": str(Path(__file__).resolve()),
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    manifest_path = output / "ballistic_evaluation_manifest.json"
    _atomic_json(manifest_path, result)
    return manifest_path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", required=True)
    result.add_argument("--raw-manifest", required=True)
    result.add_argument("--mapper-checkpoint", required=True)
    result.add_argument("--s-checkpoint", required=True)
    result.add_argument("--h-checkpoint", required=True)
    result.add_argument("--trajectory-checkpoint", required=True)
    result.add_argument("--selector-checkpoint", required=True)
    result.add_argument("--refiner-checkpoint", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--bullet-speed-mps", type=float, default=22.0)
    result.add_argument("--dense-step-ms", type=float, default=1.0)
    result.add_argument("--device", default="cuda")
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--cache-batch-size", type=int, default=256)
    return result


def main() -> None:
    args = parser().parse_args()
    if min(args.batch_size, args.cache_batch_size) <= 0:
        raise ValueError("ballistic evaluation batch sizes must be positive")
    print(run(args))


if __name__ == "__main__":
    main()
