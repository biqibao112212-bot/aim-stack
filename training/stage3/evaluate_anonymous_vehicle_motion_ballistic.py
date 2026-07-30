"""Evaluate the frozen anonymous MotionContext at projectile flight time.

This is a diagnostic, oracle-associated validation pass.  Truth is used only
to construct the future-visible target label and never enters the model
forward contract.  One continuous query is evaluated per validation window at
``norm(frozen_upstream_current_position) / bullet_speed``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .anonymous_vehicle_motion import (
    AnonymousVehicleFutureModel,
    target_candidate_rows,
)
from .cyclic_future_foundation import load_frozen_v19
from .evaluate_final_visible_position_ballistic import (
    TruthState,
    _ballistic_label,
)
from .observable_future_pnp_ab import sha256_file, state_dict_sha256
from .pnp_q0_hypothesis_adapter import (
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .train_anonymous_vehicle_motion import (
    RUN_SCHEMA,
    CombinedMotionDataset,
    _dataset,
    _forward_only,
    frozen_upstream_batch,
)
from .train_causal_physical_ab import _git_state, _to_device
from .train_pnp_window_mapper_distillation import _atomic_json


EVALUATION_SCHEMA = "stage3-anonymous-vehicle-motion-ballistic-v1"
FLIGHT_TIME_FORMULA = "norm(frozen_upstream_current_position_m)/bullet_speed_mps"
MOTION_NAMES = {2: "rotation", 3: "combined"}
DISTANCE_EDGES_M = tuple(float(value) for value in range(1, 8))
DISPLAY_BODY_PERCENTILE = 95.0
EXPECTED_FINAL_UPDATE = 2100
OPPOSITE_SOURCE_FAILURE = "observable target jumped to an opposite source slot"


def _resolve_shard(root: Path, value: object) -> Path:
    return root / Path(str(value).replace("\\", "/"))


def _load_truth_states(
    truth_history: Path,
    required_keys: set[tuple[str, int]],
) -> tuple[dict[tuple[str, int], TruthState], dict[str, Any]]:
    manifest_path = truth_history / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("test_accessed", True)):
        raise ValueError("ballistic evaluation refuses truth history that accessed test")
    if not bool(manifest.get("qualification_passed", False)):
        raise ValueError("ballistic evaluation requires qualified truth history")

    states: dict[tuple[str, int], TruthState] = {}
    for shard in manifest["shards"]:
        if str(shard["split"]) not in {"train", "validation"}:
            continue
        path = _resolve_shard(truth_history, shard["path"])
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
        raise ValueError(
            f"missing {len(missing)} ballistic truth states; first={missing[0]}"
        )
    return states, {
        "path": str(truth_history),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "test_accessed": False,
        "state_count": len(states),
    }


def _metadata(
    dataset: CombinedMotionDataset,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...], np.ndarray]:
    sessions: list[str] = []
    t0_values: list[int] = []
    pair_ids: list[str] = []
    motion: list[int] = []
    for part in dataset.parts:
        sessions.extend(part.session_ids)
        t0_values.extend(part.t0_ns)
        pair_ids.extend(part.pair_ids)
        motion.extend([part.motion_class] * len(part))
    if not (len(dataset) == len(sessions) == len(t0_values) == len(pair_ids)):
        raise RuntimeError("ballistic metadata arrays are misaligned")
    return (
        tuple(sessions), tuple(t0_values), tuple(pair_ids),
        np.asarray(motion, dtype=np.int64),
    )


def _pair_reverse_flag(pair_id: str) -> bool:
    return bool((int(pair_id[:16], 16) >> 2) & 1)


def _canonical_ballistic_label(
    state: TruthState,
    model_current_position_m: np.ndarray,
    truth_current_position_m: np.ndarray,
    *,
    bullet_speed_mps: float,
    dense_step_s: float,
) -> tuple[dict[str, Any] | None, str | None]:
    """Build a label after `_dataset` has already removed reflection.

    Opposite-source jumps have no label under the adjacent visible-stream
    contract.  They are retained in the audit denominator but fail closed from
    prediction metrics.  Every other label error remains fatal.
    """
    try:
        return _ballistic_label(
            state,
            model_current_position_m,
            truth_current_position_m=truth_current_position_m,
            reverse_direction=False,
            bullet_speed_mps=bullet_speed_mps,
            dense_step_s=dense_step_s,
        ), None
    except ValueError as error:
        if str(error) == OPPOSITE_SOURCE_FAILURE:
            return None, "opposite_source_jump"
        raise


def _load_motion_checkpoint(path: Path) -> tuple[AnonymousVehicleFutureModel, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != RUN_SCHEMA:
        raise ValueError("anonymous motion checkpoint schema mismatch")
    if not bool(payload.get("fixed_endpoint", False)):
        raise ValueError("ballistic evaluation requires the fixed final endpoint")
    if payload.get("checkpoint_role") != "fixed_final_endpoint":
        raise ValueError("anonymous motion checkpoint is not the final endpoint")
    if int(payload.get("progress", {}).get("global_update", -1)) != EXPECTED_FINAL_UPDATE:
        raise ValueError("anonymous motion checkpoint is not update 2100")
    provenance = payload.get("provenance", {})
    if (
        provenance.get("oracle_association") is not True
        or provenance.get("deployable_pipeline") is not False
        or provenance.get("test_accessed") is not False
    ):
        raise ValueError("anonymous motion checkpoint provenance is not sealed")
    config = payload["model_config"]
    context = config["motion_context"]
    model = AnonymousVehicleFutureModel(
        channels=int(config["channels"]),
        dropout=float(config["dropout"]),
        message_layers=int(context["message_layers"]),
        trained_horizon_s=float(config["trained_horizon_s"]),
        maximum_absolute_step=int(config["maximum_absolute_step"]),
        position_scale_m=float(config["position_scale_m"]),
        history_scale_s=float(config["history_scale_s"]),
        residual_scale_m=float(config["residual_scale_m"]),
    )
    model.load_state_dict(payload["model"], strict=True)
    actual_hash = state_dict_sha256(model.state_dict())
    if actual_hash != payload.get("model_state_dict_sha256"):
        raise ValueError("anonymous motion checkpoint state hash mismatch")
    return model, {
        "path": str(path),
        "sha256": sha256_file(path),
        "state_dict_sha256": actual_hash,
        "global_update": int(payload["progress"]["global_update"]),
        "checkpoint_role": payload["checkpoint_role"],
        "fixed_endpoint": True,
        "provenance": provenance,
        "model_config": config,
    }


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    if not np.isfinite(values).all():
        raise ValueError("ballistic evaluation produced non-finite values")
    return {
        "count": int(values.size),
        "mean_m": float(values.mean()),
        "p50_m": float(np.percentile(values, 50)),
        "p90_m": float(np.percentile(values, 90)),
        "p95_m": float(np.percentile(values, 95)),
        "p99_m": float(np.percentile(values, 99)),
    }


def _selection_diagnostics(queries: dict[str, np.ndarray]) -> dict[str, Any]:
    predicted = queries["predicted_switch_count"].astype(np.int64)
    target = queries["target_switch_count"].astype(np.int64)
    exact = predicted == target
    same_role = np.remainder(predicted - target, 4) == 0
    wrong_exact = ~exact
    wrong_role = ~same_role
    excess = (
        queries["hard_error_m"].astype(np.float64)
        - queries["conditional_error_m"].astype(np.float64)
    )
    return {
        "exact_signed_step_accuracy": float(exact.mean()),
        "modulo4_physical_role_accuracy": float(same_role.mean()),
        "exact_wrong_count": int(wrong_exact.sum()),
        "exact_wrong_but_same_role_count": int((wrong_exact & same_role).sum()),
        "exact_wrong_but_same_role_fraction": (
            float((wrong_exact & same_role).sum() / wrong_exact.sum())
            if bool(wrong_exact.any()) else None
        ),
        "wrong_role_count": int(wrong_role.sum()),
        "hard_minus_conditional_m": _distribution(excess),
        "wrong_role_hard_minus_conditional_m": (
            _distribution(excess[wrong_role]) if bool(wrong_role.any())
            else {"count": 0}
        ),
    }


def _metrics(queries: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "sample_count": int(queries["hard_error_m"].size),
        "hard_position": _distribution(queries["hard_error_m"]),
        "conditional_position": _distribution(queries["conditional_error_m"]),
        "hard_anchor_relative_displacement": _distribution(
            queries["hard_displacement_error_m"]
        ),
        "conditional_anchor_relative_displacement": _distribution(
            queries["conditional_displacement_error_m"]
        ),
        "upstream_q0_position": _distribution(queries["q0_error_m"]),
        "selection": _selection_diagnostics(queries),
    }


def _distance_rows(queries: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    distance = queries["truth_distance_m"].astype(np.float64)
    hard_mm = queries["hard_error_m"].astype(np.float64) * 1000.0
    conditional_mm = queries["conditional_error_m"].astype(np.float64) * 1000.0
    predicted = queries["predicted_switch_count"].astype(np.int64)
    target = queries["target_switch_count"].astype(np.int64)
    exact = predicted == target
    role = np.remainder(predicted - target, 4) == 0
    bins = [
        (f"[{left:.0f},{right:.0f})", left, right)
        for left, right in zip(DISTANCE_EDGES_M[:-1], DISTANCE_EDGES_M[1:])
    ]
    bins.append(("[1,7) overall", 1.0, 7.0))
    rows: list[dict[str, Any]] = []
    for label, left, right in bins:
        mask = (distance >= left) & (distance < right)
        row: dict[str, Any] = {"distance_bin_m": label, "count": int(mask.sum())}
        if bool(mask.any()):
            row.update({
                "hard_mean_mm": float(hard_mm[mask].mean()),
                "hard_p50_mm": float(np.percentile(hard_mm[mask], 50)),
                "hard_p95_mm": float(np.percentile(hard_mm[mask], 95)),
                "conditional_mean_mm": float(conditional_mm[mask].mean()),
                "conditional_p50_mm": float(np.percentile(conditional_mm[mask], 50)),
                "conditional_p95_mm": float(np.percentile(conditional_mm[mask], 95)),
                "exact_step_accuracy": float(exact[mask].mean()),
                "modulo4_role_accuracy": float(role[mask].mean()),
            })
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "distance_bin_m", "count", "hard_mean_mm", "hard_p50_mm", "hard_p95_mm",
        "conditional_mean_mm", "conditional_p50_mm", "conditional_p95_mm",
        "exact_step_accuracy", "modulo4_role_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    path: Path,
    motion_name: str,
    queries: dict[str, np.ndarray],
    bullet_speed_mps: float,
    raw_count: int,
) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distance = queries["truth_distance_m"].astype(np.float64)
    hard_mm = queries["hard_error_m"].astype(np.float64) * 1000.0
    conditional_mm = queries["conditional_error_m"].astype(np.float64) * 1000.0
    cap_mm = max(
        50.0,
        float(np.percentile(np.concatenate((hard_mm, conditional_mm)), DISPLAY_BODY_PERCENTILE)),
    )
    figure, axis = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    for values, color, marker, label in (
        (conditional_mm, "#F28E2B", "x", "Known future role (conditional)"),
        (hard_mm, "#2878B5", "o", "Model-selected final position (hard)"),
    ):
        overflow = values > cap_mm
        axis.scatter(
            distance[~overflow], values[~overflow], s=16, alpha=0.38,
            color=color, marker=marker, linewidths=0.7 if marker == "x" else 0,
            rasterized=True, label=label,
        )
        if bool(overflow.any()):
            axis.scatter(
                distance[overflow], np.full(int(overflow.sum()), cap_mm),
                s=24, alpha=0.85, color=color, marker="^", linewidths=0,
            )
    axis.set_xlim(max(1.0, float(distance.min()) - 0.15), min(7.0, float(distance.max()) + 0.15))
    axis.set_ylim(0.0, cap_mm * 1.04)
    axis.set_xlabel("Current target truth distance (m)")
    axis.set_ylabel("Future visible-position error at flight time (mm)")
    axis.set_title(
        f"{motion_name.capitalize()} | range / {bullet_speed_mps:g} m/s | "
        f"eligible={hard_mm.size}/{raw_count}"
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
        "hard_overflow_count": int((hard_mm > cap_mm).sum()),
        "conditional_overflow_count": int((conditional_mm > cap_mm).sum()),
    }


def _write_summary(
    path: Path,
    metrics: dict[str, Any],
    bullet_speed_mps: float,
    raw_counts: dict[str, int],
) -> None:
    lines = [
        "# Anonymous MotionContext ballistic-time diagnostic", "",
        f"Query time is frozen upstream range divided by {bullet_speed_mps:g} m/s.",
        "Truth supplies labels and the plot x-axis only; it is not a model input.",
        "Exact signed-step accuracy is reported separately from modulo-4 physical-role accuracy.",
        "",
        "| Motion | Eligible/raw | Hard mean (mm) | Hard P50 | Hard P95 | Conditional mean (mm) | Conditional P95 | Exact step | Mod-4 role |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("rotation", "combined"):
        value = metrics[name]
        hard = value["hard_position"]
        conditional = value["conditional_position"]
        selection = value["selection"]
        lines.append(
            f"| {name} | {value['sample_count']}/{raw_counts[name]} | "
            f"{hard['mean_m'] * 1000:.2f} | "
            f"{hard['p50_m'] * 1000:.2f} | {hard['p95_m'] * 1000:.2f} | "
            f"{conditional['mean_m'] * 1000:.2f} | {conditional['p95_m'] * 1000:.2f} | "
            f"{selection['exact_signed_step_accuracy']:.2%} | "
            f"{selection['modulo4_physical_role_accuracy']:.2%} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def run(args: argparse.Namespace) -> Path:
    if args.bullet_speed_mps <= 0.0 or args.dense_step_ms <= 0.0:
        raise ValueError("bullet speed and dense truth step must be positive")
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
    dataset_manifest_sha256 = sha256_file(dataset_manifest_path)
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    dataset = _dataset(dataset_path, "validation", sample_limit=0)
    sessions, t0_values, pair_ids, motion = _metadata(dataset)
    required_keys = set(zip(sessions, t0_values))
    truth_states, truth_provenance = _load_truth_states(
        Path(args.truth_history).resolve(), required_keys,
    )

    mapper, mapper_info = load_frozen_pnp_mapper(args.mapper_checkpoint)
    s_model, s_info = load_frozen_v19(args.s_checkpoint)
    h_model, h_info = load_frozen_hypothesis_adapter(
        args.h_checkpoint, allow_diagnostic=True,
    )
    model, model_info = _load_motion_checkpoint(Path(args.motion_checkpoint).resolve())
    expected_dataset_sha256 = model_info["provenance"]["dataset"]["manifest_sha256"]
    if dataset_manifest_sha256 != expected_dataset_sha256:
        raise ValueError("ballistic dataset differs from the motion checkpoint")
    expected_truth_sha256 = dataset_manifest.get("truth_history_manifest_sha256")
    if truth_provenance["manifest_sha256"] != expected_truth_sha256:
        raise ValueError("ballistic truth history differs from the paired dataset")
    expected = model_info["provenance"]["frozen_initial_state_dict_sha256"]
    loaded_hashes = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    }
    if loaded_hashes != expected:
        raise ValueError("ballistic upstream states differ from the motion checkpoint")
    for frozen in (mapper, s_model, h_model, model):
        frozen.to(device).eval().requires_grad_(False)
    frozen_initial = {
        **loaded_hashes,
        "motion": state_dict_sha256(model.state_dict()),
    }

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    collected: dict[str, list[np.ndarray]] = {}
    label_failures: list[dict[str, Any]] = []
    cursor = 0
    dense_step_s = args.dense_step_ms / 1000.0
    started = time.time()
    for raw_cpu in loader:
        raw = _to_device(raw_cpu, device)
        batch = frozen_upstream_batch(mapper, s_model, h_model, raw)
        batch_count = int(batch["current_position_m"].shape[0])
        indices = range(cursor, cursor + batch_count)
        estimated_distance = torch.linalg.vector_norm(
            batch["current_position_m"].float(), dim=1,
        ).cpu().numpy()
        truth_current = batch["truth_current_position_m"].float().cpu().numpy()
        attempted = [
            _canonical_ballistic_label(
                truth_states[(sessions[index], t0_values[index])],
                batch["current_position_m"][row].float().cpu().numpy(),
                truth_current[row],
                bullet_speed_mps=args.bullet_speed_mps,
                dense_step_s=dense_step_s,
            )
            for row, index in enumerate(indices)
        ]
        eligible_rows = [row for row, (label, _) in enumerate(attempted) if label is not None]
        for row, (label, reason) in enumerate(attempted):
            if label is not None:
                continue
            index = cursor + row
            label_failures.append({
                "dataset_index": index,
                "session_id": sessions[index],
                "t0_ns": t0_values[index],
                "pair_id": pair_ids[index],
                "motion_class": int(motion[index]),
                "source_reverse_bit": _pair_reverse_flag(pair_ids[index]),
                "reason": reason,
            })
        if not eligible_rows:
            cursor += batch_count
            continue
        select = torch.tensor(eligible_rows, dtype=torch.long, device=device)
        dynamic = {
            name: value.index_select(0, select) for name, value in batch.items()
        }
        labels = [attempted[row][0] for row in eligible_rows]
        if any(label is None for label in labels):
            raise RuntimeError("eligible ballistic label unexpectedly missing")
        labels = [label for label in labels if label is not None]
        selected_indices = [cursor + row for row in eligible_rows]
        flight_time = np.asarray(
            [label["flight_time_s"] for label in labels], dtype=np.float32,
        )
        if bool(np.any(flight_time <= 0.0)) or bool(
            np.any(flight_time > model.trained_horizon_s + 1e-6)
        ):
            raise ValueError("ballistic query is outside the trained horizon")
        target_step = np.asarray(
            [label["target_switch_count"] for label in labels], dtype=np.int64,
        )
        if bool(np.any(np.abs(target_step) > model.maximum_absolute_step)):
            raise ValueError("ballistic label exceeds the model candidate range")
        target_delta = np.stack(
            [label["target_visible_delta_m"] for label in labels]
        ).astype(np.float32, copy=False)
        dynamic["tau_s"] = torch.from_numpy(flight_time[:, None]).to(device)
        dynamic["target_switch_count"] = torch.from_numpy(target_step[:, None]).to(device)
        dynamic["target_visible_delta_m"] = torch.from_numpy(target_delta[:, None]).to(device)
        dynamic["target_query_mask"] = torch.ones(
            (len(eligible_rows), 1), dtype=torch.bool, device=device,
        )
        prediction = model(_forward_only(dynamic))
        row = target_candidate_rows(
            dynamic["candidate_step"], dynamic["candidate_mask"],
            dynamic["target_switch_count"], dynamic["target_query_mask"],
        )
        conditional = prediction["conditional_position_m"].gather(
            2, row[:, :, None, None].expand(-1, -1, 1, 3),
        ).squeeze(2).squeeze(1)
        hard = prediction["position_m"].squeeze(1)
        target = dynamic["truth_current_position_m"] + dynamic["target_visible_delta_m"].squeeze(1)
        current = dynamic["current_position_m"]
        conditional_error = torch.linalg.vector_norm(conditional - target, dim=-1)
        hard_error = torch.linalg.vector_norm(hard - target, dim=-1)
        conditional_displacement = torch.linalg.vector_norm(
            (conditional - current) - dynamic["target_visible_delta_m"].squeeze(1), dim=-1,
        )
        hard_displacement = torch.linalg.vector_norm(
            (hard - current) - dynamic["target_visible_delta_m"].squeeze(1), dim=-1,
        )
        q0_error = torch.linalg.vector_norm(
            current - dynamic["truth_current_position_m"], dim=-1,
        )
        predicted_step = prediction["selected_switch_step"].squeeze(1)
        values = {
            "dataset_index": np.asarray(selected_indices, dtype=np.int64),
            "session_id": np.asarray([sessions[index] for index in selected_indices]),
            "t0_ns": np.asarray([t0_values[index] for index in selected_indices], dtype=np.int64),
            "pair_id": np.asarray([pair_ids[index] for index in selected_indices]),
            "source_reverse_bit": np.asarray(
                [_pair_reverse_flag(pair_ids[index]) for index in selected_indices],
                dtype=np.bool_,
            ),
            "motion_class": motion[selected_indices],
            "estimated_distance_m": estimated_distance[eligible_rows].astype(np.float32),
            "truth_distance_m": np.asarray(
                [label["truth_distance_m"] for label in labels], dtype=np.float32,
            ),
            "flight_time_s": flight_time,
            "tau_s": flight_time.copy(),
            "target_switch_count": target_step,
            "predicted_switch_count": predicted_step.cpu().numpy().astype(np.int64),
            "conditional_error_m": conditional_error.float().cpu().numpy(),
            "hard_error_m": hard_error.float().cpu().numpy(),
            "conditional_displacement_error_m": conditional_displacement.float().cpu().numpy(),
            "hard_displacement_error_m": hard_displacement.float().cpu().numpy(),
            "q0_error_m": q0_error.float().cpu().numpy(),
        }
        for name, value in values.items():
            collected.setdefault(name, []).append(value)
        cursor += batch_count
    if cursor != len(dataset):
        raise RuntimeError("ballistic evaluation did not consume the full validation set")
    queries = {name: np.concatenate(parts) for name, parts in collected.items()}
    evaluated_count = len(dataset) - len(label_failures)
    if not all(len(value) == evaluated_count for value in queries.values()):
        raise RuntimeError("ballistic query arrays have inconsistent lengths")

    frozen_final = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
        "motion": state_dict_sha256(model.state_dict()),
    }
    if frozen_initial != frozen_final:
        raise RuntimeError("ballistic evaluation changed a frozen model state")

    per_motion: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    raw_motion_counts = {
        name: int((motion == motion_class).sum())
        for motion_class, name in MOTION_NAMES.items()
    }
    for motion_class, motion_name in MOTION_NAMES.items():
        keep = queries["motion_class"] == motion_class
        subset = {name: value[keep] for name, value in queries.items()}
        query_path = output / f"{motion_name}_queries.npz"
        np.savez_compressed(query_path, **subset)
        rows = _distance_rows(subset)
        table_path = output / f"{motion_name}_distance_table.csv"
        _write_csv(table_path, rows)
        plot = _plot(
            output / f"{motion_name}_distance_error.png",
            motion_name,
            subset,
            args.bullet_speed_mps,
            raw_motion_counts[motion_name],
        )
        metrics[motion_name] = _metrics(subset)
        per_motion[motion_name] = {
            "metrics": metrics[motion_name],
            "queries": str(query_path),
            "queries_sha256": sha256_file(query_path),
            "distance_table": str(table_path),
            "distance_table_sha256": sha256_file(table_path),
            "plot": plot,
        }
    metrics["overall"] = _metrics(queries)
    all_queries = output / "all_ballistic_queries.npz"
    np.savez_compressed(all_queries, **queries)
    summary = output / "ballistic_summary.md"
    _write_summary(summary, metrics, args.bullet_speed_mps, raw_motion_counts)

    failure_reason_counts: dict[str, int] = {}
    for failure in label_failures:
        reason = str(failure["reason"])
        failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1
    result = {
        "schema_version": EVALUATION_SCHEMA,
        "status": "complete",
        "evaluation_only": True,
        "model_updated": False,
        "oracle_association": True,
        "deployable_pipeline": False,
        "test_accessed": False,
        "elapsed_s": time.time() - started,
        "ballistic_query": {
            "bullet_speed_mps": args.bullet_speed_mps,
            "flight_time_formula": FLIGHT_TIME_FORMULA,
            "range_source": "frozen_mapper_s_h_current_visible_armor_position",
            "plot_x_axis": "exact_q0_current_visible_armor_distance_m",
            "truth_used_as_network_input": False,
            "dense_truth_step_s": dense_step_s,
        },
        "dataset": {
            "path": str(dataset_path),
            "manifest_sha256": dataset_manifest_sha256,
            "validation_sample_count": len(dataset),
            "evaluated_sample_count": evaluated_count,
            "label_coverage": evaluated_count / len(dataset),
            "motion_counts": raw_motion_counts,
            "evaluated_motion_counts": {
                name: int((queries["motion_class"] == motion_class).sum())
                for motion_class, name in MOTION_NAMES.items()
            },
            "label_failure_count": len(label_failures),
            "label_failure_reason_counts": failure_reason_counts,
            "label_failures": label_failures,
        },
        "truth_labels": truth_provenance,
        "upstream": {"mapper": mapper_info, "s": s_info, "h": h_info},
        "motion_model": model_info,
        "per_motion": per_motion,
        "overall": metrics["overall"],
        "all_queries": str(all_queries),
        "all_queries_sha256": sha256_file(all_queries),
        "summary": str(summary),
        "summary_sha256": sha256_file(summary),
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
    result.add_argument("--truth-history", required=True)
    result.add_argument("--mapper-checkpoint", required=True)
    result.add_argument("--s-checkpoint", required=True)
    result.add_argument("--h-checkpoint", required=True)
    result.add_argument("--motion-checkpoint", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--bullet-speed-mps", type=float, default=22.0)
    result.add_argument("--dense-step-ms", type=float, default=1.0)
    result.add_argument("--device", default="cuda")
    result.add_argument("--batch-size", type=int, default=128)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    print(run(args))


if __name__ == "__main__":
    main()
