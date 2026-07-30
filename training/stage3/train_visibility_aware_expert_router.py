"""Calibrate the v2 history-only latent-expert router.

The v70 trajectory bank is frozen. Future truth is used only to derive the
lowest-error expert label in the supervised loss; the gate forward pass still
receives causal anonymous history through MotionContext and nothing else.
This bounded diagnostic asks whether expert choice is identifiable from the
existing causal state. It does not promote an oracle best-of-experts result.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any
import uuid

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate

from .anonymous_vehicle_motion_v2 import (
    VisibilityAwareAnonymousVehicleFutureModel,
    target_roles,
)
from .cyclic_future_foundation import load_frozen_v19
from .observable_future_pnp_ab import sha256_file, state_dict_sha256
from .pnp_q0_hypothesis_adapter import (
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .train_anonymous_vehicle_motion import (
    CHECKPOINT_INTERVAL,
    BalancedMotionHistorySampler,
    _cuda_amp_dtype,
    _dataset,
    _forward_only,
    _json_sha256,
    _require_runtime,
    _validate_bindings,
    apply_prefix_dropout,
    frozen_upstream_batch,
)
from .train_anonymous_vehicle_motion_v2 import (
    RUN_SCHEMA as PARENT_RUN_SCHEMA,
    evaluate as evaluate_future,
)
from .train_causal_physical_ab import _git_state, _seed, _to_device
from .train_pnp_window_mapper_distillation import _atomic_checkpoint, _atomic_json


RUN_SCHEMA = "stage3-visibility-aware-expert-router-diagnostic-v1"
EXPECTED_PARENT_UPDATE = 2400
EXPECTED_PARENT_SHA256 = (
    "2b1af57f0cac21ed564b4a9031634dc6dcd7e04ebcf6b3e9ad197d3e497e68cb"
)
MOTION_CLASSES = (2, 3)
HISTORY_BINS = ((8, 15), (16, 23), (24, 1_000_000))
RELIABLE_MARGIN_M = 0.02
TIE_MARGIN_M = 1e-6


def _history_label(count: int) -> str:
    for lower, upper in HISTORY_BINS:
        if lower <= count <= upper:
            return f"{lower}-{upper if upper < 1_000_000 else 'plus'}"
    raise ValueError(f"history count outside router contract: {count}")


def expert_counterfactual_role_position(
    model: VisibilityAwareAnonymousVehicleFutureModel,
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Evaluate each frozen expert separately as [B,Q,4,E,3]."""
    coefficient = prediction["trajectory_coefficient"].float()
    basis = prediction["time_basis"].float()
    dynamic = torch.einsum(
        "bqr,bherc->bqhec", basis, coefficient,
    ) / float(model.basis_count) ** 0.5
    residual = torch.tanh(dynamic) * model.residual_scale_m
    primary = prediction["primary_index"].to(torch.long)
    role = torch.arange(4, device=primary.device)[None]
    handle = torch.remainder(primary[:, None] + role, 4)
    q0_relation = batch["q0_relation_m"].gather(
        1, handle.unsqueeze(-1).expand(-1, -1, 3),
    ).float()
    q0_relation[:, 0] = 0.0
    tau_scale = (
        batch["tau_s"].float() / model.trained_horizon_s
    )[:, :, None, None, None]
    return (
        batch["current_position_m"].float()[:, None, None, None]
        + q0_relation[:, None, :, None]
        + tau_scale * residual
    )


def best_expert_targets(
    expert_position: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return detached best expert, per-window error [B,E], and valid windows."""
    if expert_position.ndim != 5 or expert_position.shape[2] != 4:
        raise ValueError("expert positions must have shape [B,Q,4,E,3]")
    mask = batch["target_query_mask"].to(torch.bool) & (batch["tau_s"] > 0)
    if mask.shape != expert_position.shape[:2]:
        raise ValueError("expert positions and target query mask disagree")
    role = target_roles(batch["target_switch_count"], mask)
    selected = expert_position.gather(
        2,
        role[:, :, None, None, None].expand(
            -1, -1, 1, expert_position.shape[3], 3,
        ),
    ).squeeze(2)
    target = (
        batch["truth_current_position_m"].float()[:, None]
        + batch["target_visible_delta_m"].float()
    )
    error = torch.linalg.vector_norm(selected - target[:, :, None], dim=-1)
    count = mask.sum(dim=1)
    valid_window = count > 0
    if not bool(valid_window.all()):
        raise ValueError("every router training window needs a positive query")
    window_error = torch.where(
        mask[:, :, None], error, torch.zeros_like(error),
    ).sum(dim=1) / count[:, None].to(error.dtype)
    target_expert = window_error.detach().argmin(dim=1)
    return target_expert, window_error.detach(), valid_window


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean_m": float(array.mean()),
        "p50_m": float(np.percentile(array, 50)),
        "p95_m": float(np.percentile(array, 95)),
    }


def _hard_expert_role_position(
    expert_position: torch.Tensor,
    expert_index: torch.Tensor,
) -> torch.Tensor:
    """Gather one history-selected expert for every role as [B,Q,4,3]."""
    if expert_index.shape != expert_position.shape[:1]:
        raise ValueError("expert index must contain one choice per window")
    return expert_position.gather(
        3,
        expert_index[:, None, None, None, None].expand(
            -1, expert_position.shape[1], 4, 1, 3,
        ),
    ).squeeze(3)


def _masked_window_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.sum(dim=1)
    if not bool((count > 0).all()):
        raise ValueError("every router validation window needs a positive query")
    return torch.where(mask, value, torch.zeros_like(value)).sum(dim=1) / count.to(
        value.dtype
    )


def _macro_recall(confusion: list[list[int]]) -> float | None:
    recalls = [
        row[index] / sum(row)
        for index, row in enumerate(confusion)
        if sum(row) > 0
    ]
    return float(np.mean(recalls)) if recalls else None


def _load_parent(
    path: Path,
    *,
    expected_sha256: str = EXPECTED_PARENT_SHA256,
) -> tuple[VisibilityAwareAnonymousVehicleFutureModel, dict[str, Any]]:
    checkpoint_sha256 = sha256_file(path)
    if checkpoint_sha256 != expected_sha256:
        raise ValueError("router parent is not the hash-locked v70 checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != PARENT_RUN_SCHEMA:
        raise ValueError("router parent must be the v2 motion run")
    if (
        payload.get("checkpoint_role") != "fixed_final_endpoint"
        or payload.get("fixed_endpoint") is not True
        or int(payload.get("progress", {}).get("global_update", -1))
        != EXPECTED_PARENT_UPDATE
    ):
        raise ValueError("router parent must be fixed update 2400")
    provenance = payload.get("provenance", {})
    if (
        provenance.get("oracle_association") is not True
        or provenance.get("deployable_pipeline") is not False
        or provenance.get("test_accessed") is not False
    ):
        raise ValueError("router parent provenance is not sealed")
    config = payload["model_config"]
    context = config["motion_context"]
    model = VisibilityAwareAnonymousVehicleFutureModel(
        channels=int(config["channels"]), dropout=float(config["dropout"]),
        message_layers=int(context["message_layers"]),
        trained_horizon_s=float(config["trained_horizon_s"]),
        maximum_absolute_step=int(config["maximum_absolute_step"]),
        position_scale_m=float(config["position_scale_m"]),
        history_scale_s=float(config["history_scale_s"]),
        residual_scale_m=float(config["residual_scale_m"]),
        basis_count=int(config["basis_count"]),
        latent_experts=int(config["latent_experts"]),
    )
    model.load_state_dict(payload["model"], strict=True)
    actual = state_dict_sha256(model.state_dict())
    if actual != payload.get("model_state_dict_sha256"):
        raise ValueError("router parent state hash mismatch")
    return model, {
        "path": str(path), "sha256": checkpoint_sha256,
        "state_dict_sha256": actual, "schema_version": PARENT_RUN_SCHEMA,
        "global_update": EXPECTED_PARENT_UPDATE,
        "provenance": provenance, "model_config": config,
    }


def _frozen_motion_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("motion_regime_gate.")
    }


def _assert_frozen_states(
    model: nn.Module,
    mapper: nn.Module,
    s_model: nn.Module,
    h_model: nn.Module,
    *,
    frozen_motion_sha256: str,
    upstream_sha256: dict[str, str],
) -> None:
    actual_motion = state_dict_sha256(_frozen_motion_state(model))
    actual_upstream = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    }
    if actual_motion != frozen_motion_sha256 or actual_upstream != upstream_sha256:
        raise RuntimeError("router training changed a frozen state")


def _comparison_to_initial(
    initial: dict[str, Any],
    final: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    initial_groups = initial["routing_diagnostic"]
    final_groups = final["routing_diagnostic"]
    for name, current in final_groups.items():
        old = initial_groups[name]
        old_mean = old["selected_expert_window_error"].get("mean_m")
        current_mean = current["selected_expert_window_error"].get("mean_m")
        oracle_mean = current["oracle_best_window_error_noncausal_bound"].get("mean_m")
        denominator = (
            old_mean - oracle_mean
            if old_mean is not None and oracle_mean is not None else None
        )
        result[name] = {
            "initial_v70_history_gate_mean_m": old_mean,
            "trained_history_gate_mean_m": current_mean,
            "oracle_best_of_three_mean_m_noncausal_bound": oracle_mean,
            "mean_regret_reduction_fraction": (
                (old_mean - current_mean) / denominator
                if denominator is not None and denominator > 1e-12 else None
            ),
            "initial_best_expert_accuracy": old[
                "best_expert_classification_accuracy"
            ],
            "trained_best_expert_accuracy": current[
                "best_expert_classification_accuracy"
            ],
        }
    return result


def _train_validation_generalization(
    train: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, validation_group in validation["routing_diagnostic"].items():
        train_group = train["routing_diagnostic"][name]
        train_accuracy = train_group["best_expert_classification_accuracy"]
        validation_accuracy = validation_group["best_expert_classification_accuracy"]
        train_macro = train_group["macro_recall"]
        validation_macro = validation_group["macro_recall"]
        result[name] = {
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
            "accuracy_gap_fraction": (
                train_accuracy - validation_accuracy
                if train_accuracy is not None and validation_accuracy is not None
                else None
            ),
            "train_macro_recall": train_macro,
            "validation_macro_recall": validation_macro,
            "macro_recall_gap_fraction": (
                train_macro - validation_macro
                if train_macro is not None and validation_macro is not None
                else None
            ),
        }
    return result


def _attach_train_selected_fixed_expert(
    train: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    train_overall = train["routing_diagnostic"]["overall"]
    expert_index = int(
        train_overall["slice_oracle_best_fixed_single_expert_noncausal"]
    )
    for group in validation["routing_diagnostic"].values():
        fixed = group["fixed_single_expert_window_error"][expert_index]
        selected = group["selected_expert_window_error"]
        group["train_overall_selected_fixed_expert"] = expert_index
        group["train_selected_fixed_expert_window_error"] = fixed
        group["improvement_over_train_selected_fixed_mean_m"] = (
            fixed["mean_m"] - selected["mean_m"]
            if "mean_m" in fixed and "mean_m" in selected else None
        )


def _predeclared_acceptance(metrics: dict[str, Any]) -> dict[str, Any]:
    comparison = metrics["comparison_to_initial_v70_gate"]
    diagnostic = metrics["routing_diagnostic"]
    generalization = metrics["train_validation_generalization"]
    overall = diagnostic["overall"]
    reliable_confusion = overall[
        "reliable_margin_confusion_matrix_target_rows"
    ]
    reliable_recalls = [
        row[index] / sum(row) if sum(row) else None
        for index, row in enumerate(reliable_confusion)
    ]
    combined_old = comparison["combined"]["initial_v70_history_gate_mean_m"]
    combined_new = comparison["combined"]["trained_history_gate_mean_m"]
    train_validation_gap = generalization["overall"][
        "macro_recall_gap_fraction"
    ]
    checks = {
        "overall_oracle_gap_closed_at_least_50pct": (
            comparison["overall"]["mean_regret_reduction_fraction"] is not None
            and comparison["overall"]["mean_regret_reduction_fraction"] >= 0.50
        ),
        "short_rotation_oracle_gap_closed_at_least_50pct": (
            comparison["motion_2_history_8-15"][
                "mean_regret_reduction_fraction"
            ] is not None
            and comparison["motion_2_history_8-15"][
                "mean_regret_reduction_fraction"
            ] >= 0.50
        ),
        "beats_train_selected_fixed_single_expert": (
            overall["improvement_over_train_selected_fixed_mean_m"] is not None
            and overall["improvement_over_train_selected_fixed_mean_m"] > 0.0
        ),
        "reliable_margin_macro_recall_at_least_65pct": (
            overall["reliable_margin_macro_recall"] is not None
            and overall["reliable_margin_macro_recall"] >= 0.65
        ),
        "each_reliable_expert_recall_at_least_50pct": all(
            recall is not None and recall >= 0.50 for recall in reliable_recalls
        ),
        "combined_mean_regression_no_more_than_5pct": (
            combined_old is not None and combined_new is not None
            and combined_new <= 1.05 * combined_old
        ),
        "train_validation_macro_recall_gap_no_more_than_15pp": (
            train_validation_gap is not None
            and abs(train_validation_gap) <= 0.15
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "all_checks_pass": all(checks.values()),
        "checks": checks,
        "reliable_margin_per_expert_recall": reliable_recalls,
        "thresholds_predeclared_before_full_run": True,
        "oracle_bound_is_not_deployable_performance": True,
    }


@torch.no_grad()
def evaluate_router(
    model: VisibilityAwareAnonymousVehicleFutureModel,
    loader: DataLoader,
    mapper: nn.Module,
    s_model: nn.Module,
    h_model: nn.Module,
    device: torch.device,
    *,
    include_soft_mixture_metrics: bool = True,
) -> dict[str, Any]:
    future = (
        evaluate_future(model, loader, mapper, s_model, h_model, device)
        if include_soft_mixture_metrics else None
    )
    groups: dict[str, dict[str, Any]] = {}
    for name in ("overall", "rotation", "combined"):
        groups[name] = {
            "correct": 0, "count": 0, "selected_error": [],
            "oracle_error": [], "regret": [], "target_histogram": [0, 0, 0],
            "predicted_histogram": [0, 0, 0], "margin": [],
            "conditional_query_error": [], "final_query_error": [],
            "conditional_window_error": [], "final_window_error": [],
            "fixed_expert_window_error": [[], [], []],
            "confusion": [[0, 0, 0] for _ in range(3)],
            "reliable_confusion": [[0, 0, 0] for _ in range(3)],
            "reliable_count": 0, "reliable_correct": 0,
        }
    for motion in MOTION_CLASSES:
        for lower, upper in HISTORY_BINS:
            groups[f"motion_{motion}_history_{_history_label(lower)}"] = {
                "correct": 0, "count": 0, "selected_error": [],
                "oracle_error": [], "regret": [], "target_histogram": [0, 0, 0],
                "predicted_histogram": [0, 0, 0], "margin": [],
                "conditional_query_error": [], "final_query_error": [],
                "conditional_window_error": [], "final_window_error": [],
                "fixed_expert_window_error": [[], [], []],
                "confusion": [[0, 0, 0] for _ in range(3)],
                "reliable_confusion": [[0, 0, 0] for _ in range(3)],
                "reliable_count": 0, "reliable_correct": 0,
            }
    model.eval()
    amp_enabled = device.type == "cuda"
    for raw_cpu in loader:
        raw = _to_device(raw_cpu, device)
        amp = (
            torch.autocast("cuda", dtype=_cuda_amp_dtype())
            if amp_enabled else nullcontext()
        )
        with amp:
            batch = frozen_upstream_batch(mapper, s_model, h_model, raw)
            prediction = model(_forward_only(batch))
        expert_position = expert_counterfactual_role_position(model, prediction, batch)
        target, error, _ = best_expert_targets(expert_position, batch)
        predicted = prediction["motion_regime_logits"].float().argmax(dim=1)
        selected_error = error.gather(1, predicted[:, None]).squeeze(1)
        oracle_error = error.gather(1, target[:, None]).squeeze(1)
        ordered_error = error.sort(dim=1).values
        margin = ordered_error[:, 1] - ordered_error[:, 0]
        positive = batch["target_query_mask"].to(torch.bool) & (batch["tau_s"] > 0)
        role = target_roles(batch["target_switch_count"], positive)
        hard_role_position = _hard_expert_role_position(expert_position, predicted)
        conditional_position = hard_role_position.gather(
            2, role[:, :, None, None].expand(-1, -1, 1, 3),
        ).squeeze(2)
        final_position = hard_role_position.gather(
            2,
            prediction["selected_role"].to(torch.long)[:, :, None, None].expand(
                -1, -1, 1, 3,
            ),
        ).squeeze(2)
        truth_position = (
            batch["truth_current_position_m"].float()[:, None]
            + batch["target_visible_delta_m"].float()
        )
        conditional_query_error = torch.linalg.vector_norm(
            conditional_position - truth_position, dim=-1,
        )
        final_query_error = torch.linalg.vector_norm(
            final_position - truth_position, dim=-1,
        )
        conditional_window_error = _masked_window_mean(
            conditional_query_error, positive,
        )
        final_window_error = _masked_window_mean(final_query_error, positive)
        history_count = prediction["history_active_count"].to(torch.long)
        motion_class = batch["motion_class"].to(torch.long)
        for row in range(target.shape[0]):
            motion = int(motion_class[row])
            labels = (
                "overall",
                "rotation" if motion == 2 else "combined",
                f"motion_{motion}_history_{_history_label(int(history_count[row]))}",
            )
            for label in labels:
                group = groups[label]
                truth_index = int(target[row])
                predicted_index = int(predicted[row])
                group["count"] += 1
                group["correct"] += int(truth_index == predicted_index)
                group["target_histogram"][truth_index] += 1
                group["predicted_histogram"][predicted_index] += 1
                group["confusion"][truth_index][predicted_index] += 1
                selected_value = float(selected_error[row])
                oracle_value = float(oracle_error[row])
                group["selected_error"].append(selected_value)
                group["oracle_error"].append(oracle_value)
                group["regret"].append(selected_value - oracle_value)
                margin_value = float(margin[row])
                group["margin"].append(margin_value)
                if margin_value >= RELIABLE_MARGIN_M:
                    group["reliable_count"] += 1
                    group["reliable_correct"] += int(truth_index == predicted_index)
                    group["reliable_confusion"][truth_index][predicted_index] += 1
                query_mask = positive[row]
                group["conditional_query_error"].extend(
                    conditional_query_error[row][query_mask].float().cpu().tolist()
                )
                group["final_query_error"].extend(
                    final_query_error[row][query_mask].float().cpu().tolist()
                )
                group["conditional_window_error"].append(
                    float(conditional_window_error[row])
                )
                group["final_window_error"].append(float(final_window_error[row]))
                for expert_index in range(error.shape[1]):
                    group["fixed_expert_window_error"][expert_index].append(
                        float(error[row, expert_index])
                    )
    diagnostic: dict[str, Any] = {}
    for name, group in groups.items():
        count = int(group["count"])
        fixed = [
            _distribution(values)
            for values in group["fixed_expert_window_error"]
        ]
        fixed_means = [
            item.get("mean_m", float("inf"))
            for item in fixed
        ]
        best_fixed = int(np.argmin(fixed_means)) if count else None
        reliable_count = int(group["reliable_count"])
        selected_mean = _distribution(group["selected_error"])
        best_fixed_mean = fixed_means[best_fixed] if best_fixed is not None else None
        diagnostic[name] = {
            "window_count": count,
            "best_expert_classification_accuracy": (
                int(group["correct"]) / count if count else None
            ),
            "selected_expert_window_error": selected_mean,
            "oracle_best_window_error_noncausal_bound": _distribution(group["oracle_error"]),
            "routing_regret": _distribution(group["regret"]),
            "target_expert_histogram": group["target_histogram"],
            "predicted_expert_histogram": group["predicted_histogram"],
            "expert_confusion_matrix_target_rows": group["confusion"],
            "macro_recall": _macro_recall(group["confusion"]),
            "oracle_label_margin": _distribution(group["margin"]),
            "tie_fraction_at_1e-6_m": (
                float(np.mean(np.asarray(group["margin"]) <= TIE_MARGIN_M))
                if count else None
            ),
            "reliable_margin_threshold_m": RELIABLE_MARGIN_M,
            "reliable_margin_window_count": reliable_count,
            "reliable_margin_fraction": reliable_count / count if count else None,
            "reliable_margin_accuracy": (
                int(group["reliable_correct"]) / reliable_count
                if reliable_count else None
            ),
            "reliable_margin_confusion_matrix_target_rows": group["reliable_confusion"],
            "reliable_margin_macro_recall": _macro_recall(group["reliable_confusion"]),
            "achieved_history_only_hard_router": {
                "oracle_role_conditional_query_error": _distribution(
                    group["conditional_query_error"]
                ),
                "frozen_selector_final_query_error": _distribution(
                    group["final_query_error"]
                ),
                "oracle_role_conditional_window_error": _distribution(
                    group["conditional_window_error"]
                ),
                "frozen_selector_final_window_error": _distribution(
                    group["final_window_error"]
                ),
            },
            "fixed_single_expert_window_error": fixed,
            "slice_oracle_best_fixed_single_expert_noncausal": best_fixed,
            "improvement_over_slice_oracle_best_fixed_mean_m": (
                best_fixed_mean - selected_mean["mean_m"]
                if count and best_fixed_mean is not None else None
            ),
        }
    return {
        "primary_metric": "routing_diagnostic.*.achieved_history_only_hard_router",
        "current_soft_mixture_metrics_secondary": future,
        "routing_diagnostic": diagnostic,
    }


def _rng_state(
    sampler: BalancedMotionHistorySampler,
    prefix_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "sampler": sampler.state_dict(),
        "prefix_dropout_generator": prefix_generator.get_state(),
    }


def _restore_rng(
    state: dict[str, Any],
    sampler: BalancedMotionHistorySampler,
    prefix_generator: torch.Generator,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    sampler.load_state_dict(state["sampler"])
    prefix_generator.set_state(state["prefix_dropout_generator"])


def _validate_resume_metadata(
    payload: dict[str, Any], checkpoint: Path,
) -> tuple[int, list[dict[str, Any]]]:
    update = int(payload["progress"]["global_update"])
    try:
        filename_update = int(checkpoint.stem.rsplit("-", 1)[-1])
    except ValueError as error:
        raise ValueError("router resume checkpoint filename has no update") from error
    if filename_update != update:
        raise ValueError("router resume filename and payload update differ")
    history = list(payload.get("validation_history", []))
    history_updates = [int(item["global_update"]) for item in history]
    if (
        not history_updates
        or history_updates[0] != 0
        or history_updates != sorted(history_updates)
        or history_updates[-1] > update
    ):
        raise ValueError("router resume validation history is inconsistent")
    return update, history


def train(args: argparse.Namespace) -> Path:
    if not args.diagnostic_oracle_association:
        raise ValueError("requires explicit --diagnostic-oracle-association")
    if not args.allow_mapper_h_mismatch:
        raise ValueError("requires explicit --allow-mapper-h-mismatch")
    if args.router_updates <= 0:
        raise ValueError("router update count must be positive")
    output = Path(args.output).resolve()
    resume = Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None
    if resume is None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing existing router output: {output}")
        output.mkdir(parents=True, exist_ok=True)
    else:
        if not output.is_dir() or not resume.is_file():
            raise ValueError("router resume requires output and checkpoint")
        if resume.parent != output / "checkpoints":
            raise ValueError("router resume checkpoint is outside output")
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    _seed(args.seed)
    device = _require_runtime(args)
    dataset_path = Path(args.dataset).resolve()
    manifest_path = dataset_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        bool(manifest.get("test_accessed", True))
        or manifest.get("oracle_association") is not True
        or manifest.get("deployable_pipeline") is not False
    ):
        raise ValueError("router dataset provenance is not sealed")
    dataset_sha = sha256_file(manifest_path)
    train_dataset = _dataset(
        dataset_path, "train", sample_limit=args.train_limit_per_class,
    )
    validation_dataset = _dataset(
        dataset_path, "validation", sample_limit=args.validation_limit_per_class,
    )
    sampler = BalancedMotionHistorySampler(train_dataset.strata(), seed=args.seed + 1)
    prefix_generator = torch.Generator().manual_seed(args.seed + 2)
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.validation_batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )

    mapper, mapper_info = load_frozen_pnp_mapper(args.mapper_checkpoint)
    s_model, s_info = load_frozen_v19(args.s_checkpoint)
    h_model, h_info = load_frozen_hypothesis_adapter(
        args.h_checkpoint, allow_diagnostic=True,
    )
    compatibility = _validate_bindings(dataset_sha, mapper_info, s_info, h_info)
    model, parent_info = _load_parent(Path(args.parent_checkpoint).resolve())
    if parent_info["provenance"]["dataset"]["manifest_sha256"] != dataset_sha:
        raise ValueError("router parent dataset differs")
    expected_upstream = parent_info["provenance"]["frozen_initial_state_dict_sha256"]
    upstream_initial = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    }
    if upstream_initial != expected_upstream:
        raise ValueError("router upstream differs from parent")
    for frozen in (mapper, s_model, h_model):
        frozen.to(device).eval().requires_grad_(False)
    model.to(device).eval().requires_grad_(False)
    model.motion_regime_gate.train().requires_grad_(True)
    frozen_motion_initial = state_dict_sha256(_frozen_motion_state(model))
    gate_initial = state_dict_sha256(model.motion_regime_gate.state_dict())
    gate_parameters = list(model.motion_regime_gate.parameters())
    optimizer = torch.optim.AdamW(
        gate_parameters, lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    amp_dtype = _cuda_amp_dtype()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    source_paths = {
        "router_trainer": Path(__file__).resolve(),
        "motion_model_v2": Path(__file__).with_name(
            "anonymous_vehicle_motion_v2.py"
        ).resolve(),
        "shared_trainer_v1": Path(__file__).with_name(
            "train_anonymous_vehicle_motion.py"
        ).resolve(),
        "parent_trainer_v2": Path(__file__).with_name(
            "train_anonymous_vehicle_motion_v2.py"
        ).resolve(),
    }
    source_hashes = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    provenance = {
        "diagnostic_only": True, "oracle_association": True,
        "deployable_pipeline": False, "test_accessed": False,
        "future_truth_forward_input": False,
        "future_truth_usage": "detached best-expert training label only",
        "trainable_module": "motion_regime_gate",
        "frozen_parent": parent_info,
        "dataset": {"path": str(dataset_path), "manifest_sha256": dataset_sha},
        "mapper": mapper_info, "s": s_info, "h": h_info,
        "mapper_h_compatibility": compatibility["mapper_h"],
        "dataset_provenance_compatibility": compatibility["dataset_manifest"],
        "sampler": {
            "strategy": "equal_motion_x_history_bin_with_replacement_v1",
            "support": sampler.support,
        },
        "runtime": {
            "platform": platform.platform(), "python_executable": sys.executable,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device), "amp_dtype": str(amp_dtype),
        },
        "git": _git_state(),
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_sha256": source_hashes,
        "frozen_motion_initial_state_dict_sha256": frozen_motion_initial,
        "gate_initial_state_dict_sha256": gate_initial,
        "upstream_initial_state_dict_sha256": upstream_initial,
    }
    contract = {
        "schema_version": RUN_SCHEMA,
        "args": {name: value for name, value in vars(args).items() if name != "resume_checkpoint"},
        "dataset_manifest_sha256": dataset_sha,
        "parent_checkpoint_sha256": parent_info["sha256"],
        "source_sha256": source_hashes,
        "fixed_router_updates": args.router_updates,
    }
    contract_sha = _json_sha256(contract)
    run_id = str(uuid.uuid4())
    update = 0
    validation_history: list[dict[str, Any]] = []
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != RUN_SCHEMA:
            raise ValueError("router resume schema differs")
        if payload.get("contract_sha256") != contract_sha:
            raise ValueError("router resume contract differs")
        update, validation_history = _validate_resume_metadata(payload, resume)
        if update >= args.router_updates:
            raise ValueError("router endpoint already complete")
        later = [
            path for path in checkpoint_dir.glob("checkpoint-update-*.pt")
            if int(path.stem.rsplit("-", 1)[-1]) > update
        ]
        if later:
            raise ValueError("router resume must use latest checkpoint")
        model.load_state_dict(payload["model"], strict=True)
        if state_dict_sha256(model.state_dict()) != payload.get(
            "model_state_dict_sha256"
        ):
            raise ValueError("router resume model state hash mismatch")
        _assert_frozen_states(
            model, mapper, s_model, h_model,
            frozen_motion_sha256=frozen_motion_initial,
            upstream_sha256=upstream_initial,
        )
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        provenance = payload["provenance"]
        run_id = str(payload["run_id"])
        _restore_rng(payload["rng"], sampler, prefix_generator)
    else:
        validation_history.append({
            "global_update": 0,
            "metrics": evaluate_router(
                model, validation_loader, mapper, s_model, h_model, device,
            ),
        })
    model.eval()
    model.motion_regime_gate.train()

    manifest_payload: dict[str, Any] = {
        "schema_version": RUN_SCHEMA, "status": "running", "run_id": run_id,
        "provenance": provenance, "contract": contract,
        "contract_sha256": contract_sha, "progress": {"global_update": update},
        "validation_history": validation_history,
    }
    _atomic_json(output / "run_manifest.json", manifest_payload)
    started = time.time()
    while update < args.router_updates:
        indices = sampler.draw(args.batch_size)
        raw_cpu = default_collate([train_dataset[index] for index in indices])
        raw_cpu = apply_prefix_dropout(
            raw_cpu, probability=args.prefix_dropout_probability,
            minimum_events=args.minimum_history_events,
            generator=prefix_generator,
        )
        raw = _to_device(raw_cpu, device)
        with torch.autocast("cuda", dtype=amp_dtype):
            batch = frozen_upstream_batch(mapper, s_model, h_model, raw)
            prediction = model(_forward_only(batch))
        expert_position = expert_counterfactual_role_position(model, prediction, batch)
        target_expert, window_error, _ = best_expert_targets(expert_position, batch)
        logits = prediction["motion_regime_logits"].float()
        loss = F.cross_entropy(logits, target_expert)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nonfinite = [
            name for name, parameter in model.motion_regime_gate.named_parameters()
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        ]
        if nonfinite:
            raise RuntimeError("router produced non-finite gradients: " + ", ".join(nonfinite))
        torch.nn.utils.clip_grad_norm_(gate_parameters, args.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        update += 1
        if update % args.log_interval == 0 or update == args.router_updates:
            predicted_expert = logits.argmax(dim=1)
            selected_error = window_error.gather(1, predicted_expert[:, None]).mean()
            oracle_error = window_error.min(dim=1).values.mean()
            print(json.dumps({
                "global_update": update, "router_ce": float(loss.detach()),
                "best_expert_accuracy": float((predicted_expert == target_expert).float().mean()),
                "selected_expert_error_m": float(selected_error),
                "oracle_best_error_m": float(oracle_error),
                "elapsed_s": time.time() - started,
            }, sort_keys=True), flush=True)
        endpoint = update == args.router_updates
        if update % CHECKPOINT_INTERVAL == 0 or endpoint:
            _assert_frozen_states(
                model, mapper, s_model, h_model,
                frozen_motion_sha256=frozen_motion_initial,
                upstream_sha256=upstream_initial,
            )
            if endpoint:
                final_validation = evaluate_router(
                    model, validation_loader, mapper, s_model, h_model, device,
                )
                train_evaluation_loader = DataLoader(
                    train_dataset, batch_size=args.validation_batch_size,
                    shuffle=False, num_workers=0, pin_memory=True,
                )
                final_train = evaluate_router(
                    model, train_evaluation_loader, mapper, s_model, h_model,
                    device, include_soft_mixture_metrics=False,
                )
                final_validation["comparison_to_initial_v70_gate"] = (
                    _comparison_to_initial(
                        validation_history[0]["metrics"], final_validation,
                    )
                )
                final_validation["train_routing_diagnostic"] = final_train[
                    "routing_diagnostic"
                ]
                _attach_train_selected_fixed_expert(final_train, final_validation)
                final_validation["train_validation_generalization"] = (
                    _train_validation_generalization(final_train, final_validation)
                )
                final_validation["predeclared_acceptance"] = (
                    _predeclared_acceptance(final_validation)
                )
                validation_history.append({
                    "global_update": update,
                    "metrics": final_validation,
                })
                model.eval()
                model.motion_regime_gate.train()
            checkpoint = checkpoint_dir / f"checkpoint-update-{update:06d}.pt"
            payload = {
                "schema_version": RUN_SCHEMA,
                "model_class": type(model).__name__, "model_config": model.config,
                "model": model.state_dict(),
                "model_state_dict_sha256": state_dict_sha256(model.state_dict()),
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "progress": {"global_update": update},
                "validation_history": validation_history,
                "provenance": provenance, "contract_sha256": contract_sha,
                "rng": _rng_state(sampler, prefix_generator), "run_id": run_id,
                "checkpoint_role": "fixed_final_endpoint" if endpoint else "periodic_recovery",
                "fixed_endpoint": endpoint,
                "checkpoint_selected_by_validation": False,
            }
            _atomic_checkpoint(checkpoint, payload)
            manifest_payload["progress"] = {
                "global_update": update, "latest_checkpoint": str(checkpoint),
            }
            manifest_payload["validation_history"] = validation_history
            _atomic_json(output / "run_manifest.json", manifest_payload)

    _assert_frozen_states(
        model, mapper, s_model, h_model,
        frozen_motion_sha256=frozen_motion_initial,
        upstream_sha256=upstream_initial,
    )
    frozen_motion_final = state_dict_sha256(_frozen_motion_state(model))
    upstream_final = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    }
    final_checkpoint = checkpoint_dir / f"checkpoint-update-{args.router_updates:06d}.pt"
    final_metrics = validation_history[-1]["metrics"]
    _atomic_json(output / "final_metrics.json", final_metrics)
    manifest_payload.update({
        "status": "complete", "stop_reason": "fixed_update_endpoint",
        "progress": {"global_update": args.router_updates, "latest_checkpoint": str(final_checkpoint)},
        "fixed_final_checkpoint": {
            "path": str(final_checkpoint), "sha256": sha256_file(final_checkpoint),
            "update": args.router_updates, "selected_by_validation": False,
        },
        "validation_history": validation_history, "final_validation": final_metrics,
        "frozen_motion_final_state_dict_sha256": frozen_motion_final,
        "upstream_final_state_dict_sha256": upstream_final,
        "all_frozen_states_unchanged": True,
        "gate_final_state_dict_sha256": state_dict_sha256(model.motion_regime_gate.state_dict()),
        "elapsed_s": time.time() - started,
    })
    _atomic_json(output / "run_manifest.json", manifest_payload)
    return final_checkpoint


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--diagnostic-oracle-association", action="store_true")
    result.add_argument("--allow-mapper-h-mismatch", action="store_true")
    result.add_argument("--dataset", required=True)
    result.add_argument("--mapper-checkpoint", required=True)
    result.add_argument("--s-checkpoint", required=True)
    result.add_argument("--h-checkpoint", required=True)
    result.add_argument("--parent-checkpoint", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--resume-checkpoint")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--seed", type=int, default=20260730)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--validation-batch-size", type=int, default=96)
    result.add_argument("--router-updates", type=int, default=600)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--gradient-clip-norm", type=float, default=1.0)
    result.add_argument("--prefix-dropout-probability", type=float, default=0.75)
    result.add_argument("--minimum-history-events", type=int, default=8)
    result.add_argument("--train-limit-per-class", type=int, default=0)
    result.add_argument("--validation-limit-per-class", type=int, default=0)
    result.add_argument("--log-interval", type=int, default=25)
    return result


def main() -> None:
    checkpoint = train(parser().parse_args())
    print(json.dumps({
        "status": "complete", "fixed_final_checkpoint": str(checkpoint),
        "selected_by_validation": False, "oracle_association": True,
        "deployable_pipeline": False, "test_accessed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
