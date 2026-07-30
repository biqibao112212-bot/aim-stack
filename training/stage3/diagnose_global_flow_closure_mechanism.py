"""Post-hoc mechanism audit for the rejected V11 closure checkpoints.

This module never trains and never reads test.  It distinguishes a real
state-conditioned history closure from an iterative observation reader, and it
replaces V11's contaminated crossed-flow diagnostic with validation-only
physically coherent hybrids whose common velocity comes from truth labels.
Truth is used only to construct the audit intervention and its labels; it is
never passed to the model.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .cyclic_future_foundation import load_frozen_v19
from .global_flow_closure_future import AnonymousGlobalFlowClosureProbe
from .joint_rigid_flow_probe import LOCAL_LAG_SCALES_S
from .motion_truth_supervision import (
    MOTION_TARGET_FIELD,
    MotionTruthIndex,
    fit_motion_scales,
    normalize_attached_motion,
)
from .observable_future_pnp_ab import sha256_file, state_dict_sha256
from .pnp_q0_hypothesis_adapter import (
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .train_anonymous_vehicle_motion import (
    _cuda_amp_dtype,
    _distribution,
    _json_sha256,
    _validate_bindings,
    frozen_upstream_batch,
)
from .train_causal_physical_ab import _git_state, _seed, _to_device
from .train_global_flow_closure_probe import RUN_SCHEMA, _sample_closure_error
from .train_increment_invariant_anonymous_future import _dataset
from .train_paired_twist_set_probe import _validate_args
from .train_pnp_window_mapper_distillation import _atomic_json


AUDIT_SCHEMA = "stage3-v11-global-flow-closure-mechanism-audit-v1"
REFINEMENT_MODES = ("normal", "prediction_blind", "prediction_shuffled")


def _compatible_roll(
    value: torch.Tensor,
    valid: torch.Tensor,
    *,
    streams: int,
    event_count: int,
) -> torch.Tensor:
    """Roll event time only within the same anonymous stream and lag scale."""
    if value.shape[:2] != valid.shape:
        raise ValueError("prediction roll shapes differ")
    scale_count = len(LOCAL_LAG_SCALES_S)
    if value.shape[1] != streams * event_count * scale_count:
        raise ValueError("compatible prediction roll factor count differs")
    grouped_value = value.reshape(
        value.shape[0], streams, event_count, scale_count, value.shape[-1],
    )
    grouped_valid = valid.reshape(
        valid.shape[0], streams, event_count, scale_count,
    )
    return AnonymousGlobalFlowClosureProbe._roll_grouped(
        grouped_value, grouped_valid,
    ).reshape_as(value)


def _refinement_forward(
    model: AnonymousGlobalFlowClosureProbe,
    history: dict[str, torch.Tensor],
    *,
    mode: str,
    refinement_steps: int = 2,
) -> dict[str, torch.Tensor]:
    """Replay V11 head while intervening only on the update prediction path."""
    if mode not in REFINEMENT_MODES:
        raise ValueError(f"unknown refinement mechanism mode: {mode}")
    head = model.motion_state_head
    handle_geometry = history["_handle_geometry_raw"]
    handle_kinematics = history["_handle_kinematics_raw"]
    handle_valid = history["_handle_raw_valid"]
    pair_geometry = history["_pair_geometry_raw"]
    pair_kinematics = history["_pair_kinematics_raw"]
    pair_valid = history["_pair_raw_valid"]
    event_count = pair_valid.shape[1] // len(LOCAL_LAG_SCALES_S)
    handle_factor = head.handle_initial_encoder(torch.cat((
        handle_geometry, handle_kinematics,
    ), dim=-1))
    pair_factor = head.pair_initial_encoder(
        pair_geometry, pair_kinematics[..., :6], pair_kinematics[..., 6:13],
    )
    handle_pool = head._masked_mean(handle_factor, handle_valid)
    pair_pool = head._masked_mean(pair_factor, pair_valid)
    logits = head.handle_initial_state(handle_pool)
    logits = torch.cat((
        logits[:, :3], logits[:, 3:4] + head.pair_initial_yaw(pair_pool),
    ), dim=-1)
    initial = torch.tanh(logits)
    handle_touched = torch.zeros_like(handle_valid)
    pair_touched = torch.zeros_like(pair_valid)
    handle_geometry_prior, handle_time, pair_geometry_prior, pair_time = (
        head._prior_contexts(
            handle_geometry, handle_kinematics, pair_geometry, pair_kinematics,
        )
    )
    handle_prior = torch.cat((handle_geometry_prior, handle_time), dim=-1)
    observed_handle = handle_kinematics[..., :3]
    observed_pair = pair_kinematics[..., :3]
    for _ in range(int(refinement_steps)):
        state = torch.tanh(logits)
        handle_prediction, pair_prediction = head._decode_history(
            state, handle_geometry_prior, handle_time,
            pair_geometry_prior, pair_time,
        )
        if mode == "prediction_blind":
            update_handle_prediction = torch.zeros_like(handle_prediction)
            update_pair_prediction = torch.zeros_like(pair_prediction)
        elif mode == "prediction_shuffled":
            update_handle_prediction = _compatible_roll(
                handle_prediction, handle_valid,
                streams=4, event_count=event_count,
            )
            update_pair_prediction = _compatible_roll(
                pair_prediction, pair_valid,
                streams=1, event_count=event_count,
            )
            handle_touched |= (
                update_handle_prediction != handle_prediction
            ).any(dim=-1) & handle_valid
            pair_touched |= (
                update_pair_prediction != pair_prediction
            ).any(dim=-1) & pair_valid
        else:
            update_handle_prediction = handle_prediction
            update_pair_prediction = pair_prediction
        handle_residual = observed_handle - update_handle_prediction
        pair_residual = observed_pair - update_pair_prediction
        handle_message = head.handle_residual_encoder(torch.cat((
            handle_prior, handle_residual, update_handle_prediction,
        ), dim=-1))
        pair_message = head.pair_residual_encoder(
            pair_geometry_prior, pair_residual, pair_time,
        )
        handle_residual_pool = head._masked_mean(handle_message, handle_valid)
        pair_residual_pool = head._masked_mean(pair_message, pair_valid)
        handle_update = head.handle_state_update(torch.cat((
            handle_residual_pool, state,
        ), dim=-1))
        pair_yaw_update = head.pair_yaw_update(pair_residual_pool)
        update = torch.cat((
            handle_update[:, :3],
            handle_update[:, 3:4] + pair_yaw_update,
        ), dim=-1)
        logits = logits + 0.5 * torch.tanh(update)
    state = torch.tanh(logits)
    handle_prediction, pair_prediction = head._decode_history(
        state, handle_geometry_prior, handle_time,
        pair_geometry_prior, pair_time,
    )
    return {
        "motion_state_normalized": state,
        "initial_motion_state_normalized": initial,
        "handle_closure_residual_normalized": (
            observed_handle - handle_prediction
        ),
        "pair_closure_residual_normalized": observed_pair - pair_prediction,
        "handle_factor_valid": handle_valid,
        "pair_factor_valid": pair_valid,
        "shuffled_handle_touched": handle_touched,
        "shuffled_pair_touched": pair_touched,
    }


def _fixed_state_closure(
    model: AnonymousGlobalFlowClosureProbe,
    history: dict[str, torch.Tensor],
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode history at a supplied intact state without re-estimation."""
    head = model.motion_state_head
    hg, ht, pg, pt = head._prior_contexts(
        history["_handle_geometry_raw"], history["_handle_kinematics_raw"],
        history["_pair_geometry_raw"], history["_pair_kinematics_raw"],
    )
    handle_prediction, pair_prediction = head._decode_history(
        state, hg, ht, pg, pt,
    )
    return (
        history["_handle_kinematics_raw"][..., :3] - handle_prediction,
        history["_pair_kinematics_raw"][..., :3] - pair_prediction,
    )


def _support_signature(
    history: dict[str, torch.Tensor], row: int,
) -> tuple[int, bytes, bytes]:
    event_count = history["_pair_raw_valid"].shape[1] // len(LOCAL_LAG_SCALES_S)
    pair_scale = history["_pair_raw_valid"].reshape(
        -1, event_count, len(LOCAL_LAG_SCALES_S),
    ).any(dim=1)
    return (
        int(history["history_active_count"][row]),
        history["pair_flow_available"][row].to(torch.bool).cpu().numpy().tobytes(),
        pair_scale[row].to(torch.bool).cpu().numpy().tobytes(),
    )


def _physical_donor_index(
    history: dict[str, torch.Tensor],
    target: torch.Tensor,
    combined: torch.Tensor,
    *,
    relation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose same-support donors by yaw sign/magnitude, never by apparent v."""
    if relation not in {"opposite_sign_similar_magnitude", "same_sign_different_magnitude"}:
        raise ValueError("physical donor relation differs")
    target_cpu = target.float().cpu()
    combined_cpu = combined.to(torch.bool).cpu()
    source = torch.arange(target.shape[0], dtype=torch.long)
    selected = torch.zeros(target.shape[0], dtype=torch.bool)
    groups: dict[tuple[int, bytes, bytes], list[int]] = {}
    for row in torch.nonzero(combined_cpu, as_tuple=False).flatten().tolist():
        if abs(float(target_cpu[row, 3])) <= 0.5:
            continue
        groups.setdefault(_support_signature(history, row), []).append(row)
    for rows in groups.values():
        if len(rows) < 2:
            continue
        for row in rows:
            yaw = float(target_cpu[row, 3])
            candidates = [other for other in rows if other != row]
            if relation == "opposite_sign_similar_magnitude":
                candidates = [
                    other for other in candidates
                    if float(target_cpu[other, 3]) * yaw < 0
                ]
                if not candidates:
                    continue
                donor = min(
                    candidates,
                    key=lambda other: abs(
                        abs(float(target_cpu[other, 3])) - abs(yaw)
                    ),
                )
            else:
                candidates = [
                    other for other in candidates
                    if float(target_cpu[other, 3]) * yaw > 0
                ]
                if not candidates:
                    continue
                donor = max(
                    candidates,
                    key=lambda other: abs(
                        abs(float(target_cpu[other, 3])) - abs(yaw)
                    ),
                )
                if abs(
                    abs(float(target_cpu[donor, 3])) - abs(yaw)
                ) < 1.0:
                    continue
            source[row] = donor
            selected[row] = True
    return source, selected


def _truth_common_rotation_hybrid(
    model: AnonymousGlobalFlowClosureProbe,
    history: dict[str, torch.Tensor],
    source: torch.Tensor,
    target_velocity_mps: torch.Tensor,
    *,
    event_count: int,
) -> dict[str, torch.Tensor]:
    """Compose truth common velocity with donor centered rotation on donor time."""
    result = dict(history)
    scales = len(LOCAL_LAG_SCALES_S)
    handle_geometry = history["_handle_geometry_raw"].reshape(
        -1, 4, event_count, scales, 12,
    )
    handle_kinematics = history["_handle_kinematics_raw"].reshape(
        -1, 4, event_count, scales, 14,
    )
    handle_valid = history["_handle_raw_valid"].reshape(
        -1, 4, event_count, scales,
    )
    donor_geometry = handle_geometry.index_select(0, source)
    donor_kinematics = handle_kinematics.index_select(0, source)
    donor_valid = handle_valid.index_select(0, source)
    common_rate = target_velocity_mps * (
        float(model.context.history_scale_s) / float(model.context.position_scale_m)
    )
    elapsed_s = 0.01 * torch.expm1(donor_kinematics[..., 6]).clamp_min(0)
    elapsed_norm = elapsed_s / float(model.context.history_scale_s)
    centered_delta = donor_geometry[..., :3] - donor_geometry[..., 3:6]
    hybrid_delta = (
        common_rate[:, None, None, None] * elapsed_norm.unsqueeze(-1)
        + centered_delta
    )
    hybrid_rate = hybrid_delta / elapsed_norm.clamp_min(1e-7).unsqueeze(-1)

    target_weight = handle_valid.unsqueeze(-1).to(handle_geometry.dtype)
    target_center = handle_geometry[..., 6:9] - handle_geometry[..., :3]
    q0_estimate = target_center - (
        common_rate[:, None, None, None]
        * handle_kinematics[..., 7].unsqueeze(-1)
    )
    q0_center = (q0_estimate * target_weight).sum(dim=(1, 2, 3)) / (
        target_weight.sum(dim=(1, 2, 3)).clamp_min(1)
    )
    current_time = donor_kinematics[..., 7]
    current_center = q0_center[:, None, None, None] + (
        common_rate[:, None, None, None] * current_time.unsqueeze(-1)
    )
    prior_center = current_center - (
        common_rate[:, None, None, None] * elapsed_norm.unsqueeze(-1)
    )
    hybrid_geometry = donor_geometry.clone()
    hybrid_geometry[..., 6:9] = current_center + donor_geometry[..., :3]
    hybrid_geometry[..., 9:12] = prior_center + donor_geometry[..., 3:6]
    hybrid_kinematics = donor_kinematics.clone()
    hybrid_kinematics[..., :3] = hybrid_delta
    hybrid_kinematics[..., 3:6] = hybrid_rate
    result["_handle_geometry_raw"] = hybrid_geometry.reshape_as(
        history["_handle_geometry_raw"]
    )
    result["_handle_kinematics_raw"] = hybrid_kinematics.reshape_as(
        history["_handle_kinematics_raw"]
    )
    result["_handle_raw_valid"] = donor_valid.reshape_as(
        history["_handle_raw_valid"]
    )
    for name in ("_pair_geometry_raw", "_pair_kinematics_raw", "_pair_raw_valid"):
        result[name] = history[name].index_select(0, source)
    return result


def _metric_distribution(values: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    return _distribution(values[mask].float().cpu().numpy())


def _state_error(
    normalized: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    physical = normalized * scale.to(normalized.dtype)
    velocity = torch.linalg.vector_norm(physical[:, :3] - target[:, :3], dim=-1)
    yaw = (physical[:, 3] - target[:, 3]).abs()
    return {
        "sample_count": int(mask.sum()),
        "velocity_error_mps": _metric_distribution(velocity, mask),
        "yaw_error_rad_s": _metric_distribution(yaw, mask),
    }


def _transfer_metrics(
    prediction: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor,
) -> dict[str, Any]:
    valid = mask & (truth.abs() > 0.5)
    p = prediction[valid].float()
    t = truth[valid].float()
    if p.numel() < 2:
        raise ValueError("yaw transfer audit lacks support")
    slope = float((p * t).sum() / t.square().sum().clamp_min(1e-8))
    p_np = p.cpu().numpy().astype(np.float64)
    t_np = t.cpu().numpy().astype(np.float64)
    correlation = float(np.corrcoef(p_np, t_np)[0, 1])
    margin = p.abs() > 0.5
    return {
        "sample_count": int(p.numel()),
        "yaw_mae_rad_s": float((p - t).abs().mean()),
        "zero_intercept_slope": slope,
        "pearson_correlation": correlation,
        "median_absolute_ratio": float(torch.median(p.abs() / t.abs())),
        "prediction_margin_coverage": float(margin.float().mean()),
        "sign_accuracy_with_prediction_margin": float((
            torch.sign(p[margin]) == torch.sign(t[margin])
        ).float().mean()) if bool(margin.any()) else None,
    }


def _load_audit_runtime(run: Path, device: torch.device):
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((run / "probe_result.json").read_text(encoding="utf-8"))
    fixed = manifest.get("fixed_final_checkpoint")
    recorded_git = manifest.get("provenance", {}).get("git", {})
    if (
        manifest.get("schema_version") != RUN_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("progress", {}).get("global_update") != 200
        or not isinstance(fixed, dict)
        or fixed.get("update") != 200
        or fixed.get("selected_by_validation") is not False
        or report.get("schema_version")
        != "stage3-v11-global-flow-closure-probe-result-v1"
        or report.get("test_accessed") is not False
        or report.get("fixed_updates") != 200
        or recorded_git.get("worktree_dirty") is not False
    ):
        raise ValueError("V11 audit run is incomplete")
    values = dict(manifest["contract"]["args"])
    values.setdefault("stop_after_update", 0)
    args = argparse.Namespace(**values)
    _validate_args(args)
    if (
        Path(args.output).resolve() != run.resolve()
        or report.get("seed") != args.seed
        or report.get("source_commit") != recorded_git.get("git_commit")
        or report.get("contract_sha256") != manifest.get("contract_sha256")
    ):
        raise ValueError("V11 audit manifest/report binding differs")
    checkpoint = Path(report["checkpoint"]).resolve()
    if (
        checkpoint != (run / "checkpoints" / "checkpoint-update-000200.pt").resolve()
        or Path(fixed.get("path", "")).resolve() != checkpoint
        or fixed.get("sha256") != report.get("checkpoint_sha256")
        or sha256_file(checkpoint) != report.get("checkpoint_sha256")
    ):
        raise ValueError("V11 audit checkpoint identity differs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != RUN_SCHEMA
        or payload.get("model_class") != "AnonymousGlobalFlowClosureProbe"
        or payload.get("fixed_endpoint") is not True
        or payload.get("checkpoint_selected_by_validation") is not False
        or payload.get("progress") != {
            "global_update": 200, "stage": "motion_state", "stage_update": 200,
        }
        or payload.get("contract_sha256") != manifest.get("contract_sha256")
        or payload.get("run_id") != manifest.get("run_id")
        or payload.get("model_config") != manifest.get("model_config")
        or _json_sha256(payload.get("final_diagnostics"))
        != _json_sha256(manifest.get("final_diagnostics"))
        or state_dict_sha256(payload.get("model", {}))
        != payload.get("model_state_dict_sha256")
    ):
        raise ValueError("V11 audit checkpoint payload differs")

    dataset_path = Path(args.dataset).resolve()
    dataset_manifest_path = dataset_path / "dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("test_accessed") is not False:
        raise ValueError("V11 mechanism audit refuses test-accessed data")
    truth_index = MotionTruthIndex(
        args.truth_history,
        expected_manifest_sha256=dataset_manifest["truth_history_manifest_sha256"],
    )
    train_dataset = _dataset(dataset_path, "train", sample_limit=0)
    validation_dataset = _dataset(dataset_path, "validation", sample_limit=0)
    truth_index.attach(train_dataset, "train")
    truth_index.attach(validation_dataset, "validation")
    motion_scale = fit_motion_scales(train_dataset)
    normalize_attached_motion(train_dataset, motion_scale)
    normalize_attached_motion(validation_dataset, motion_scale)
    expected_scale = torch.tensor(payload["model_config"]["motion_state_scale"])
    torch.testing.assert_close(motion_scale.cpu(), expected_scale, rtol=0, atol=0)
    loader = DataLoader(
        validation_dataset, batch_size=args.validation_batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )
    mapper, mapper_info = load_frozen_pnp_mapper(args.mapper_checkpoint)
    s_model, s_info = load_frozen_v19(args.s_checkpoint)
    h_model, h_info = load_frozen_hypothesis_adapter(
        args.h_checkpoint, allow_diagnostic=True,
    )
    _validate_bindings(
        sha256_file(dataset_manifest_path), mapper_info, s_info, h_info,
    )
    for frozen in (mapper, s_model, h_model):
        frozen.to(device).eval().requires_grad_(False)
    model = AnonymousGlobalFlowClosureProbe(
        velocity_scale_mps=tuple(float(value) for value in motion_scale[:3]),
        yaw_rate_scale_rad_s=float(motion_scale[3]),
        channels=args.channels, dropout=args.dropout,
        message_layers=args.message_layers, basis_count=args.basis_count,
    ).to(device).eval().requires_grad_(False)
    model.load_state_dict(payload["model"], strict=True)
    return args, report, loader, mapper, s_model, h_model, model


@torch.inference_mode()
def run_mechanism_audit(run: Path, device: torch.device) -> dict[str, Any]:
    args, report, loader, mapper, s_model, h_model, model = _load_audit_runtime(
        run, device,
    )
    saved_fields = {name: [] for name in model._field_names()}
    targets: list[torch.Tensor] = []
    classes: list[torch.Tensor] = []
    native_replay = {
        name: {"velocity": [], "yaw": []}
        for name in ("overall", "combined", "high_speed_combined")
    }
    for raw_cpu in loader:
        raw = _to_device(raw_cpu, device)
        amp = (
            torch.autocast("cuda", dtype=_cuda_amp_dtype())
            if device.type == "cuda" else nullcontext()
        )
        with amp:
            batch = frozen_upstream_batch(mapper, s_model, h_model, raw)
            batch_fields = {
                name: batch[name] for name in model._field_names()
            }
            native = model.estimate_motion_state(**batch_fields)
        batch_target = raw[MOTION_TARGET_FIELD]
        native_physical = native["state"]["motion_state_physical"]
        velocity_error = torch.linalg.vector_norm(
            native_physical[:, :3] - batch_target[:, :3], dim=-1,
        )
        yaw_error = (native_physical[:, 3] - batch_target[:, 3]).abs()
        batch_combined = raw["motion_class"] == 3
        batch_speed = torch.linalg.vector_norm(batch_target[:, :2], dim=-1)
        batch_masks = {
            "overall": torch.ones_like(batch_combined),
            "combined": batch_combined,
            "high_speed_combined": batch_combined & (batch_speed > 1.7),
        }
        for name, mask in batch_masks.items():
            if bool(mask.any()):
                native_replay[name]["velocity"].append(
                    velocity_error[mask].float().cpu().numpy()
                )
                native_replay[name]["yaw"].append(
                    yaw_error[mask].float().cpu().numpy()
                )
        for name in saved_fields:
            saved_fields[name].append(batch[name].detach().cpu())
        targets.append(raw[MOTION_TARGET_FIELD].detach().cpu())
        classes.append(raw["motion_class"].detach().cpu())
    fields = {name: torch.cat(values).to(device) for name, values in saved_fields.items()}
    target = torch.cat(targets).to(device)
    motion_class = torch.cat(classes).to(device)
    with (
        torch.autocast("cuda", dtype=_cuda_amp_dtype())
        if device.type == "cuda" else nullcontext()
    ):
        history = model.context(**fields)
        normal_native = model.motion_state_head(
            history["_handle_geometry_raw"], history["_handle_kinematics_raw"],
            history["_handle_raw_valid"], history["_pair_geometry_raw"],
            history["_pair_kinematics_raw"], history["_pair_raw_valid"],
        )
        mode_states = {
            mode: _refinement_forward(model, history, mode=mode)
            for mode in REFINEMENT_MODES
        }
        mode_states["zero_refinement"] = _refinement_forward(
            model, history, mode="normal", refinement_steps=0,
        )
    torch.testing.assert_close(
        mode_states["normal"]["motion_state_normalized"],
        normal_native["motion_state_normalized"], rtol=2e-5, atol=4e-6,
    )
    pair_count = history["pair_flow_available"].sum(dim=1)
    speed = torch.linalg.vector_norm(target[:, :2], dim=-1)
    masks = {
        "overall": torch.ones(target.shape[0], dtype=torch.bool, device=device),
        "combined": motion_class == 3,
        "high_speed_combined": (motion_class == 3) & (speed > 1.7),
        "core": (
            (motion_class == 3) & (speed <= 1.2)
            & (history["history_active_count"] == 32) & (pair_count == 3)
        ),
        "combined_pair1": (motion_class == 3) & (pair_count == 1),
        "combined_pair2": (motion_class == 3) & (pair_count == 2),
        "combined_pair3": (motion_class == 3) & (pair_count == 3),
    }
    refinement = {
        mode: {
            name: _state_error(
                state["motion_state_normalized"], model.motion_state_scale,
                target, mask,
            )
            for name, mask in masks.items()
        }
        for mode, state in mode_states.items()
    }
    shuffled_state = mode_states["prediction_shuffled"]
    refinement["prediction_shuffled"]["intervention_coverage"] = {
        "handle_rows_touched": int(
            shuffled_state["shuffled_handle_touched"].any(dim=1).sum()
        ),
        "handle_factors_touched": int(
            shuffled_state["shuffled_handle_touched"].sum()
        ),
        "handle_valid_factors": int(
            shuffled_state["handle_factor_valid"].sum()
        ),
        "pair_rows_touched": int(
            shuffled_state["shuffled_pair_touched"].any(dim=1).sum()
        ),
        "pair_factors_touched": int(
            shuffled_state["shuffled_pair_touched"].sum()
        ),
        "pair_valid_factors": int(
            shuffled_state["pair_factor_valid"].sum()
        ),
    }
    replay_mapping = {
        "overall": ("overall_velocity_mean_mps", "overall_yaw_mean_rad_s"),
        "combined": ("combined_velocity_mean_mps", "combined_yaw_mean_rad_s"),
        "high_speed_combined": (
            "high_speed_combined_velocity_mean_mps",
            "high_speed_combined_yaw_mean_rad_s",
        ),
    }
    source_replay: dict[str, Any] = {}
    for name, (velocity_key, yaw_key) in replay_mapping.items():
        actual_velocity = _distribution(np.concatenate(
            native_replay[name]["velocity"]
        ).astype(np.float64, copy=False))
        actual_yaw = _distribution(np.concatenate(
            native_replay[name]["yaw"]
        ).astype(np.float64, copy=False))
        metrics: dict[str, Any] = {}
        if velocity_key is not None:
            delta = abs(actual_velocity["mean_m"] - report[velocity_key])
            metrics["velocity_mean_absolute_difference"] = delta
            if delta > 1e-7:
                raise ValueError(f"V11 audit velocity replay differs: {name}")
        delta = abs(actual_yaw["mean_m"] - report[yaw_key])
        metrics["yaw_mean_absolute_difference"] = delta
        if delta > 1e-7:
            raise ValueError(f"V11 audit yaw replay differs: {name}")
        metrics["velocity_error_mps"] = actual_velocity
        metrics["yaw_error_rad_s"] = actual_yaw
        source_replay[name] = metrics

    event_count = fields["history_event_mask"].shape[1]
    fixed_state = mode_states["normal"]["motion_state_normalized"]
    with (
        torch.autocast("cuda", dtype=_cuda_amp_dtype())
        if device.type == "cuda" else nullcontext()
    ):
        fixed_handle, fixed_pair = _fixed_state_closure(
            model, history, fixed_state,
        )
    intervention: dict[str, Any] = {}
    for kind in ("handle", "pair"):
        broken = model._roll_raw_geometry(
            history, handle=kind == "handle", event_count=event_count,
        )
        with (
            torch.autocast("cuda", dtype=_cuda_amp_dtype())
            if device.type == "cuda" else nullcontext()
        ):
            broken_fixed_handle, broken_fixed_pair = _fixed_state_closure(
                model, broken, fixed_state,
            )
        geometry_name = f"_{kind}_geometry_raw"
        valid_name = f"_{kind}_raw_valid"
        if kind == "handle":
            decoder_geometry = history[geometry_name][..., 3:6]
            broken_decoder_geometry = broken[geometry_name][..., 3:6]
        else:
            decoder_geometry = torch.cat((
                history[geometry_name][..., 3:6],
                history[geometry_name][..., 9:12],
            ), dim=-1)
            broken_decoder_geometry = torch.cat((
                broken[geometry_name][..., 3:6],
                broken[geometry_name][..., 9:12],
            ), dim=-1)
        changed_factor = (
            broken_decoder_geometry != decoder_geometry
        ).any(dim=-1) & history[valid_name]
        normal_residual = fixed_handle if kind == "handle" else fixed_pair
        broken_residual = (
            broken_fixed_handle if kind == "handle" else broken_fixed_pair
        )
        normal_error, supported = _sample_closure_error(
            normal_residual, changed_factor,
        )
        broken_error, _ = _sample_closure_error(
            broken_residual, changed_factor,
        )
        with (
            torch.autocast("cuda", dtype=_cuda_amp_dtype())
            if device.type == "cuda" else nullcontext()
        ):
            broken_reestimated = _refinement_forward(
                model, broken, mode="normal",
            )["motion_state_normalized"]
        groups: dict[str, Any] = {}
        for count in range(4):
            group = pair_count == count
            changed_row = changed_factor.any(dim=1) & group
            valid_factors = history[valid_name][group].sum()
            groups[f"pair{count}"] = {
                "sample_count": int(group.sum()),
                "rows_changed": int(changed_row.sum()),
                "valid_factor_count": int(valid_factors),
                "changed_factor_count": int(changed_factor[group].sum()),
                "changed_factor_fraction": float(
                    changed_factor[group].sum().float()
                    / valid_factors.clamp_min(1)
                ),
                "intact_fixed_state_closure": _metric_distribution(
                    normal_error, changed_row & supported,
                ),
                "broken_fixed_state_closure": _metric_distribution(
                    broken_error, changed_row & supported,
                ),
                "intact_state_error_on_changed_rows": _state_error(
                    fixed_state, model.motion_state_scale, target, changed_row,
                ),
                "broken_reestimated_state_error_on_changed_rows": _state_error(
                    broken_reestimated, model.motion_state_scale,
                    target, changed_row,
                ),
            }
        intervention[kind] = {"groups": groups}

    physical_cross: dict[str, Any] = {}
    combined = motion_class == 3
    for relation in (
        "opposite_sign_similar_magnitude", "same_sign_different_magnitude",
    ):
        source_cpu, selected_cpu = _physical_donor_index(
            history, target, combined, relation=relation,
        )
        source = source_cpu.to(device)
        selected = selected_cpu.to(device)
        if int(selected.sum()) < 16:
            raise ValueError(f"physical crossed audit lacks support: {relation}")
        hybrid = _truth_common_rotation_hybrid(
            model, history, source, target[:, :3], event_count=event_count,
        )
        with (
            torch.autocast("cuda", dtype=_cuda_amp_dtype())
            if device.type == "cuda" else nullcontext()
        ):
            state = model._state_from_raw_history(hybrid)["state"][
                "motion_state_physical"
            ]
        donor = target.index_select(0, source)
        physical_cross[relation] = {
            "sample_count": int(selected.sum()),
            "velocity_error_to_injected_truth_mps": _metric_distribution(
                torch.linalg.vector_norm(state[:, :3] - target[:, :3], dim=-1),
                selected,
            ),
            "yaw_transfer": _transfer_metrics(state[:, 3], donor[:, 3], selected),
        }

    return {
        "schema_version": AUDIT_SCHEMA,
        "validation_only": True,
        "test_accessed": False,
        "source_run": str(run),
        "source_commit": report["source_commit"],
        "checkpoint": report["checkpoint"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "audit_commit": _git_state()["git_commit"],
        "seed": args.seed,
        "sample_count": int(target.shape[0]),
        "refinement_mechanism": refinement,
        "source_metric_replay": source_replay,
        "fixed_state_pairing": intervention,
        "physical_truth_constructed_cross": physical_cross,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="audit rejected V11 mechanism")
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"mechanism audit output already exists: {output}")
    git = _git_state()
    if git.get("worktree_dirty") is not False:
        raise RuntimeError("mechanism audit requires a clean checkout")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("mechanism audit requires CUDA")
    _seed(args.seed)
    result = run_mechanism_audit(Path(args.run).resolve(), device)
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "mechanism_audit.json", result)
    print(json.dumps({
        "seed": result["seed"], "sample_count": result["sample_count"],
        "test_accessed": result["test_accessed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
