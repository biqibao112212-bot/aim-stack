"""Run one fixed 200-update omega-first ordered closure structural probe."""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path

import numpy as np
import torch

from .diagnose_global_flow_closure_mechanism import (
    _physical_donor_index,
    _transfer_metrics,
)
from .factorized_common_relative_motion_future import apply_common_velocity_ramp
from .motion_truth_supervision import MOTION_TARGET_FIELD
from .observable_future_pnp_ab import sha256_file, state_dict_sha256
from .omega_first_ordered_closure_future import (
    AnonymousOmegaFirstOrderedClosureProbe,
    omega_first_ordered_state_loss,
    omega_first_ordered_train_step,
)
from .train_anonymous_vehicle_motion import _cuda_amp_dtype, _json_sha256
from .train_causal_physical_ab import _git_state, _to_device
from .train_global_flow_closure_probe import (
    GROUP_NAMES,
    _append,
    _sample_closure_error,
)
from .train_joint_rigid_flow_probe import _state_parameter_count
from .train_paired_twist_set_probe import (
    PROBE_SEEDS,
    V8_JOINT_CONTROL_CHECKPOINTS,
    V8_JOINT_CONTROL_SHA256,
    V8_JOINT_REACHABLE_STATE_PARAMETERS,
    _load_v8_control,
    _motion_distribution,
    _validate_args,
    _validated_mean,
    build_probe_parser,
)
from .train_pnp_window_mapper_distillation import _atomic_json
from .train_robust_multiscale_motion_future import (
    FROZEN_FUTURE_MODULES,
    _preflight_control,
)
from .train_stable_motion_bottleneck_future import (
    ALL_TRAINABLE_MODULES, STATE_MODULES,
    _callable_contract,
    _prepare_batch,
    train,
)


RUN_SCHEMA = "stage3-anonymous-omega-first-ordered-closure-probe-v12"
DIAGNOSTIC_SCHEMA = "stage3-v12-omega-first-ordered-validation-diagnostics-v1"
DIAGNOSTIC_FIELDS = frozenset({
    "schema_version", "validation_only", "test_accessed", "seed",
    "v8_joint_control_checkpoint", "v8_joint_control_checkpoint_sha256",
    "groups", "write_isolation", "common_ramp_equivariance",
    "relative_reversal_equivariance",
    "factor_level_truth_common_donor_relative_cross",
})
WRITE_ISOLATION_FIELDS = frozenset({
    "zero_velocity_max_absolute_yaw_difference_normalized",
})
INTERVENTION_COVERAGE_FIELDS = (
    "handle_rows_touched", "handle_factors_touched", "handle_valid_factors",
    "pair_rows_touched", "pair_factors_touched", "pair_valid_factors",
)


def build_omega_first_probe_parser():
    parser = build_probe_parser()
    parser.description = "fixed 200-update omega-first ordered closure probe"
    return parser


def _state_errors(
    physical: torch.Tensor, target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    velocity = torch.linalg.vector_norm(
        physical[:, :3] - target[:, :3], dim=-1,
    )
    yaw_signed = physical[:, 3] - target[:, 3]
    return velocity, yaw_signed.abs(), yaw_signed


def _cross_sample_geometry_derangement(
    history: dict[str, torch.Tensor], *, handle: bool,
) -> dict[str, torch.Tensor]:
    """Break geometry/differential pairing using equal-support donor rows.

    A within-row roll is an identity when only one factor is available.  This
    deterministic cross-row intervention instead swaps geometry only between
    rows with the exact same valid event/scale-slot mask.  Target kinematics,
    timestamps and validity remain unchanged, so no support is created,
    removed or moved to another slot.
    """
    geometry_name = "_handle_geometry_raw" if handle else "_pair_geometry_raw"
    valid_name = "_handle_raw_valid" if handle else "_pair_raw_valid"
    geometry = history[geometry_name]
    valid = history[valid_name].to(torch.bool)
    if geometry.ndim != 3 or valid.shape != geometry.shape[:2]:
        raise ValueError("cross-sample geometry shapes differ")
    result = dict(history)
    broken = geometry.clone()
    groups: dict[bytes, list[int]] = {}
    valid_cpu = valid.detach().cpu()
    for row in range(valid.shape[0]):
        if not bool(valid_cpu[row].any()):
            continue
        signature = valid_cpu[row].numpy().tobytes()
        groups.setdefault(signature, []).append(row)
    for row_group in groups.values():
        if len(row_group) < 2:
            continue
        rows = torch.tensor(row_group, dtype=torch.long, device=geometry.device)
        donors = torch.roll(rows, shifts=1)
        target_valid = valid.index_select(0, rows)
        donor_geometry = geometry.index_select(0, donors)
        target_geometry = broken.index_select(0, rows)
        broken.index_copy_(
            0, rows,
            torch.where(
                target_valid.unsqueeze(-1), donor_geometry, target_geometry,
            ),
        )
    result[geometry_name] = broken
    return result


def _reflect_relative_history_with_truth_common(
    model,
    history: dict[str, torch.Tensor],
    target_velocity_mps: torch.Tensor,
    *,
    event_count: int,
) -> dict[str, torch.Tensor]:
    """Reflect only relative planar geometry while keeping common truth flow."""
    result = dict(history)
    pair_factors = history["_pair_raw_valid"].shape[1]
    if pair_factors % event_count:
        raise ValueError("relative reflection factor count differs")
    scales = pair_factors // event_count
    (
        handle_geometry, handle_kinematics, handle_valid,
        pair_geometry, pair_kinematics, _,
    ) = model.motion_state_head._reshape_factors(
        history["_handle_geometry_raw"], history["_handle_kinematics_raw"],
        history["_handle_raw_valid"], history["_pair_geometry_raw"],
        history["_pair_kinematics_raw"], history["_pair_raw_valid"],
    )
    (
        relative_geometry, relative_delta, _, elapsed_normalized,
    ) = model.motion_state_head._derived_relative_factors(
        handle_geometry, handle_kinematics, handle_valid,
    )
    reflection = handle_geometry.new_tensor([1.0, -1.0, 1.0])
    current_relative = relative_geometry[..., :3] * reflection
    prior_relative = relative_geometry[..., 3:6] * reflection
    reflected_relative_delta = relative_delta * reflection
    common_rate = target_velocity_mps.to(handle_geometry.dtype) * (
        float(model.context.history_scale_s)
        / float(model.context.position_scale_m)
    )
    weight = handle_valid.unsqueeze(-1).to(handle_geometry.dtype)
    original_center = handle_geometry[..., 6:9] - relative_geometry[..., :3]
    q0 = original_center - (
        common_rate[:, None, None, None]
        * handle_kinematics[..., 7].unsqueeze(-1)
    )
    q0_center = (q0 * weight).sum(dim=(1, 2, 3)) / (
        weight.sum(dim=(1, 2, 3)).clamp_min(1)
    )
    current_time = handle_kinematics[..., 7]
    current_center = q0_center[:, None, None, None] + (
        common_rate[:, None, None, None] * current_time.unsqueeze(-1)
    )
    prior_center = current_center - (
        common_rate[:, None, None, None]
        * elapsed_normalized.unsqueeze(-1)
    )
    reflected_handle_geometry = handle_geometry.clone()
    reflected_handle_geometry[..., :3] = current_relative
    reflected_handle_geometry[..., 3:6] = prior_relative
    reflected_handle_geometry[..., 6:9] = current_center + current_relative
    reflected_handle_geometry[..., 9:12] = prior_center + prior_relative
    reflected_delta = (
        common_rate[:, None, None, None]
        * elapsed_normalized.unsqueeze(-1)
        + reflected_relative_delta
    )
    reflected_handle_kinematics = handle_kinematics.clone()
    reflected_handle_kinematics[..., :3] = reflected_delta
    reflected_handle_kinematics[..., 3:6] = (
        reflected_delta / elapsed_normalized.clamp_min(1e-7).unsqueeze(-1)
    )
    result["_handle_geometry_raw"] = reflected_handle_geometry.reshape_as(
        history["_handle_geometry_raw"]
    )
    result["_handle_kinematics_raw"] = reflected_handle_kinematics.reshape_as(
        history["_handle_kinematics_raw"]
    )

    pair_geometry = pair_geometry.clone()
    pair_kinematics = pair_kinematics.clone()
    for start in (0, 3, 6, 9):
        pair_geometry[..., start:start + 3] *= reflection
    pair_kinematics[..., :3] *= reflection
    pair_kinematics[..., 3:6] *= reflection
    result["_pair_geometry_raw"] = pair_geometry.reshape_as(
        history["_pair_geometry_raw"]
    )
    result["_pair_kinematics_raw"] = pair_kinematics.reshape_as(
        history["_pair_kinematics_raw"]
    )
    return result


def _synthetic_twist_history_on_target_support(
    model,
    history: dict[str, torch.Tensor],
    common_velocity_mps: torch.Tensor,
    yaw_rate_rad_s: torch.Tensor,
    *,
    event_count: int,
) -> dict[str, torch.Tensor]:
    """Synthesize one twist without changing target support or time slots."""
    if (
        event_count < 1
        or history["_pair_raw_valid"].shape[1] % event_count
    ):
        raise ValueError("omega-first synthetic twist event count differs")
    result = dict(history)
    (
        hg, hk, hv, pg, pk, pv,
    ) = model.motion_state_head._reshape_factors(
        history["_handle_geometry_raw"], history["_handle_kinematics_raw"],
        history["_handle_raw_valid"], history["_pair_geometry_raw"],
        history["_pair_kinematics_raw"], history["_pair_raw_valid"],
    )
    relative_geometry, _, _, elapsed_normalized = (
        model.motion_state_head._derived_relative_factors(hg, hk, hv)
    )
    elapsed_s = elapsed_normalized * float(model.context.history_scale_s)
    theta = yaw_rate_rad_s.to(hg.dtype)[:, None, None, None] * elapsed_s

    def rotate(vector: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        x, y = vector[..., 0], vector[..., 1]
        return torch.stack((
            cosine * x - sine * y,
            sine * x + cosine * y,
            vector[..., 2],
        ), dim=-1)

    prior_relative = relative_geometry[..., 3:6]
    rotated_current = rotate(prior_relative, theta)
    raw_relative_delta = rotated_current - prior_relative
    weight = hv.unsqueeze(-1).to(hg.dtype)
    factor_count = weight.sum(dim=(1, 2, 3)).clamp_min(1)
    elapsed_safe = elapsed_normalized.clamp_min(1e-7)
    mean_elapsed = (
        elapsed_normalized.unsqueeze(-1) * weight
    ).sum(dim=(1, 2, 3)) / factor_count
    mean_inverse_elapsed = (
        elapsed_safe.reciprocal().unsqueeze(-1) * weight
    ).sum(dim=(1, 2, 3)) / factor_count
    mean_delta = (raw_relative_delta * weight).sum(dim=(1, 2, 3)) / factor_count
    mean_rate = (
        raw_relative_delta / elapsed_safe.unsqueeze(-1) * weight
    ).sum(dim=(1, 2, 3)) / factor_count
    constraint = torch.stack((
        torch.cat((torch.ones_like(mean_elapsed), mean_elapsed), dim=-1),
        torch.cat((mean_inverse_elapsed, torch.ones_like(mean_elapsed)), dim=-1),
    ), dim=1)
    right_hand_side = torch.stack((mean_delta, mean_rate), dim=1)
    coefficients = torch.linalg.pinv(
        constraint.float(), rcond=1e-5,
    ).matmul(right_hand_side.float()).to(hg.dtype)
    constant_correction = coefficients[:, 0]
    elapsed_correction = coefficients[:, 1]
    relative_delta = raw_relative_delta - (
        constant_correction[:, None, None, None]
        + elapsed_correction[:, None, None, None]
        * elapsed_normalized.unsqueeze(-1)
    )
    relative_delta = torch.where(
        hv.unsqueeze(-1), relative_delta, torch.zeros_like(relative_delta),
    )
    prior_mean = (prior_relative * weight).sum(dim=(1, 2, 3)) / factor_count
    prior_relative = prior_relative - prior_mean[:, None, None, None]
    current_relative = prior_relative + relative_delta

    common_rate = common_velocity_mps.to(hg.dtype) * (
        float(model.context.history_scale_s)
        / float(model.context.position_scale_m)
    )
    original_center = hg[..., 6:9] - relative_geometry[..., :3]
    q0 = original_center - (
        common_rate[:, None, None, None] * hk[..., 7].unsqueeze(-1)
    )
    q0_center = (q0 * weight).sum(dim=(1, 2, 3)) / (
        weight.sum(dim=(1, 2, 3)).clamp_min(1)
    )
    current_center = q0_center[:, None, None, None] + (
        common_rate[:, None, None, None] * hk[..., 7].unsqueeze(-1)
    )
    prior_center = current_center - (
        common_rate[:, None, None, None]
        * elapsed_normalized.unsqueeze(-1)
    )
    new_hg = hg.clone()
    new_hg[..., :3] = current_relative
    new_hg[..., 3:6] = prior_relative
    new_hg[..., 6:9] = current_center + current_relative
    new_hg[..., 9:12] = prior_center + prior_relative
    new_delta = (
        common_rate[:, None, None, None]
        * elapsed_normalized.unsqueeze(-1)
        + relative_delta
    )
    new_hk = hk.clone()
    new_hk[..., :3] = new_delta
    new_hk[..., 3:6] = new_delta / elapsed_normalized.clamp_min(
        1e-7
    ).unsqueeze(-1)
    result["_handle_geometry_raw"] = new_hg.reshape_as(
        history["_handle_geometry_raw"]
    )
    result["_handle_kinematics_raw"] = new_hk.reshape_as(
        history["_handle_kinematics_raw"]
    )

    pair_elapsed_s = 0.01 * torch.expm1(pk[..., 8]).clamp_min(0)
    pair_theta = yaw_rate_rad_s.to(pg.dtype)[:, None, None, None] * pair_elapsed_s
    pair_prior = pg[..., 3:6]
    pair_current = rotate(pair_prior, pair_theta)
    primary_sign = torch.where(
        pk[..., 11:12] > 0.5, -torch.ones_like(pk[..., 11:12]),
        torch.ones_like(pk[..., 11:12]),
    )
    pair_current = pair_current * primary_sign
    pair_delta = pair_current - pair_prior
    pair_elapsed_normalized = pair_elapsed_s / float(model.context.history_scale_s)
    new_pg = pg.clone()
    new_pg[..., :3] = pair_current
    new_pg[..., 3:6] = pair_prior
    new_pg[..., 6:9] = pair_current / pair_current.norm(
        dim=-1, keepdim=True,
    ).clamp_min(1e-6)
    new_pg[..., 9:12] = pair_prior / pair_prior.norm(
        dim=-1, keepdim=True,
    ).clamp_min(1e-6)
    new_pk = pk.clone()
    new_pk[..., :3] = pair_delta
    new_pk[..., 3:6] = pair_delta / pair_elapsed_normalized.clamp_min(
        1e-7
    ).unsqueeze(-1)
    new_pk[..., 6] = pair_current.norm(dim=-1)
    new_pk[..., 7] = pair_prior.norm(dim=-1)
    result["_pair_geometry_raw"] = new_pg.reshape_as(
        history["_pair_geometry_raw"]
    )
    result["_pair_kinematics_raw"] = new_pk.reshape_as(
        history["_pair_kinematics_raw"]
    )
    return result


@torch.inference_mode()
def omega_first_ordered_validation_diagnostics(
    model,
    loader,
    mapper,
    s_model,
    h_model,
    device: torch.device,
) -> dict:
    seed = int(torch.initial_seed())
    if seed not in PROBE_SEEDS:
        raise ValueError("omega-first diagnostic seed differs")
    control = _load_v8_control(seed, model).to(device).eval().requires_grad_(False)
    model.eval()
    metric_keys = (
        "candidate_velocity", "candidate_yaw", "candidate_yaw_signed",
        "control_velocity", "control_yaw", "control_yaw_signed",
        "broken_handle_velocity", "broken_handle_yaw",
        "broken_pair_velocity", "broken_pair_yaw",
        "zero_angular_velocity", "zero_angular_yaw",
        "zero_velocity_velocity", "zero_velocity_yaw",
        "zero_all_velocity", "zero_all_yaw",
        "angular_handle_closure", "common_handle_closure", "pair_closure",
        "fixed_broken_handle_closure", "fixed_intact_handle_closure",
        "fixed_broken_pair_closure", "fixed_intact_pair_closure",
    )
    storage = {
        group: {key: [] for key in metric_keys} | {
            "candidate_sign": 0, "control_sign": 0, "sign_count": 0,
            "count": 0,
            "handle_rows_touched": 0, "handle_factors_touched": 0,
            "handle_valid_factors": 0,
            "pair_rows_touched": 0, "pair_factors_touched": 0,
            "pair_valid_factors": 0,
        }
        for group in GROUP_NAMES
    }
    saved_fields = {name: [] for name in model._field_names()}
    saved_target: list[torch.Tensor] = []
    saved_class: list[torch.Tensor] = []
    ramp_velocity_equivariance: list[np.ndarray] = []
    ramp_yaw_invariance: list[np.ndarray] = []
    zero_velocity_yaw_max = 0.0
    for raw_cpu in loader:
        raw = _to_device(raw_cpu, device)
        amp = (
            torch.autocast("cuda", dtype=_cuda_amp_dtype())
            if device.type == "cuda" else nullcontext()
        )
        with amp:
            batch = _prepare_batch(mapper, s_model, h_model, raw)
            fields = {name: batch[name] for name in model._field_names()}
            candidate = model.estimate_motion_state(**fields)
            row = torch.arange(
                batch[MOTION_TARGET_FIELD].shape[0], device=device,
            )
            ramp_sign = torch.where(row.remainder(2) == 0, 1.0, -1.0)
            ramp = torch.zeros(
                row.shape[0], 3, dtype=batch["history_obs_rel_m"].dtype,
                device=device,
            )
            ramp[:, 0] = 0.45 * ramp_sign
            ramp[:, 1] = -0.35 * ramp_sign
            ramped_batch = apply_common_velocity_ramp(
                batch, ramp, model.motion_state_scale.to(ramp.dtype),
            )
            ramped = model.estimate_motion_state(**{
                name: ramped_batch[name] for name in model._field_names()
            })
            broken_handle = model.estimate_motion_state_broken_handle_geometry(
                **fields
            )
            zero_angular = model.estimate_motion_state_zero_angular_refinement(
                **fields
            )
            zero_velocity = model.estimate_motion_state_zero_velocity_refinement(
                **fields
            )
            zero_all = model.estimate_motion_state_zero_refinement(**fields)
            baseline = control.estimate_motion_state(**fields)
            raw_history = model.context(**fields)
            event_count = fields["history_event_mask"].shape[1]
            broken_handle_history = model._roll_raw_geometry(
                raw_history, handle=True, event_count=event_count,
            )
            broken_pair_history = _cross_sample_geometry_derangement(
                raw_history, handle=False,
            )
            broken_pair = model._state_from_raw_history(broken_pair_history)
            fixed_state = candidate["state"]["motion_state_normalized"]
            fixed = model.motion_state_head.decode_closure_at_state(
                raw_history["_handle_geometry_raw"],
                raw_history["_handle_kinematics_raw"],
                raw_history["_handle_raw_valid"],
                raw_history["_pair_geometry_raw"],
                raw_history["_pair_kinematics_raw"],
                raw_history["_pair_raw_valid"], fixed_state,
            )
            fixed_broken_handle = model.motion_state_head.decode_closure_at_state(
                broken_handle_history["_handle_geometry_raw"],
                broken_handle_history["_handle_kinematics_raw"],
                broken_handle_history["_handle_raw_valid"],
                broken_handle_history["_pair_geometry_raw"],
                broken_handle_history["_pair_kinematics_raw"],
                broken_handle_history["_pair_raw_valid"], fixed_state,
            )
            fixed_broken_pair = model.motion_state_head.decode_closure_at_state(
                broken_pair_history["_handle_geometry_raw"],
                broken_pair_history["_handle_kinematics_raw"],
                broken_pair_history["_handle_raw_valid"],
                broken_pair_history["_pair_geometry_raw"],
                broken_pair_history["_pair_kinematics_raw"],
                broken_pair_history["_pair_raw_valid"], fixed_state,
            )
        target = batch[MOTION_TARGET_FIELD]
        candidate_physical = candidate["state"]["motion_state_physical"]
        ramped_physical = ramped["state"]["motion_state_physical"]
        ramp_velocity_equivariance.append(torch.linalg.vector_norm(
            (ramped_physical[:, :3] - candidate_physical[:, :3]) - ramp,
            dim=-1,
        ).float().cpu().numpy())
        ramp_yaw_invariance.append((
            ramped_physical[:, 3] - candidate_physical[:, 3]
        ).abs().float().cpu().numpy())
        states = {
            "candidate": candidate_physical,
            "control": baseline["state"]["motion_state_physical"],
            "broken_handle": broken_handle["state"]["motion_state_physical"],
            "broken_pair": broken_pair["state"]["motion_state_physical"],
            "zero_angular": zero_angular["state"]["motion_state_physical"],
            "zero_velocity": zero_velocity["state"]["motion_state_physical"],
            "zero_all": zero_all["state"]["motion_state_physical"],
        }
        errors = {name: _state_errors(value, target) for name, value in states.items()}
        zero_velocity_yaw_max = max(
            zero_velocity_yaw_max,
            float((
                zero_velocity["state"]["motion_state_normalized"][:, 3]
                - candidate["state"]["motion_state_normalized"][:, 3]
            ).abs().max()),
        )
        candidate_state = candidate["state"]
        angular_closure, angular_supported = _sample_closure_error(
            candidate_state["angular_handle_closure_residual_normalized"],
            candidate_state["handle_factor_valid"],
        )
        common_closure, common_supported = _sample_closure_error(
            candidate_state["common_handle_closure_residual_normalized"],
            candidate_state["handle_factor_valid"],
        )
        pair_closure, pair_supported = _sample_closure_error(
            candidate_state["pair_closure_residual_normalized"],
            candidate_state["pair_factor_valid"],
        )
        changed_handle = (
            fixed_broken_handle["handle_decoder_geometry"]
            != fixed["handle_decoder_geometry"]
        ).any(dim=-1) & fixed["handle_factor_valid"]
        changed_pair = (
            fixed_broken_pair["pair_decoder_geometry"]
            != fixed["pair_decoder_geometry"]
        ).any(dim=-1) & fixed["pair_factor_valid"]
        fixed_intact_handle, fixed_handle_supported = _sample_closure_error(
            fixed["angular_handle_residual"], changed_handle,
        )
        fixed_broken_handle_error, _ = _sample_closure_error(
            fixed_broken_handle["angular_handle_residual"], changed_handle,
        )
        fixed_intact_pair, fixed_pair_supported = _sample_closure_error(
            fixed["pair_residual"], changed_pair,
        )
        fixed_broken_pair_error, _ = _sample_closure_error(
            fixed_broken_pair["pair_residual"], changed_pair,
        )
        speed = torch.linalg.vector_norm(target[:, :2], dim=-1)
        combined = batch["motion_class"] == 3
        pair_count = candidate["history"]["pair_flow_available"].sum(dim=1)
        history32 = candidate["history"]["history_active_count"] == 32
        masks = {
            "overall": torch.ones_like(combined),
            "combined": combined,
            "combined_speed_gt_1_7": combined & (speed > 1.7),
            "core": combined & (speed <= 1.2) & history32 & (pair_count == 3),
            "pair0": pair_count == 0,
            "combined_pair1": combined & (pair_count == 1),
            "combined_pair2": combined & (pair_count == 2),
            "combined_pair3": combined & (pair_count == 3),
        }
        yaw_valid = target[:, 3].abs() > 0.5
        for group, mask in masks.items():
            if not bool(mask.any()):
                continue
            item = storage[group]
            for prefix in ("candidate", "control"):
                _append(item, f"{prefix}_velocity", errors[prefix][0], mask)
                _append(item, f"{prefix}_yaw", errors[prefix][1], mask)
                _append(item, f"{prefix}_yaw_signed", errors[prefix][2], mask)
            for prefix in (
                "broken_handle", "broken_pair", "zero_angular",
                "zero_velocity", "zero_all",
            ):
                _append(item, f"{prefix}_velocity", errors[prefix][0], mask)
                _append(item, f"{prefix}_yaw", errors[prefix][1], mask)
            _append(item, "angular_handle_closure", angular_closure, mask & angular_supported)
            _append(item, "common_handle_closure", common_closure, mask & common_supported)
            _append(item, "pair_closure", pair_closure, mask & pair_supported)
            _append(
                item, "fixed_intact_handle_closure", fixed_intact_handle,
                mask & fixed_handle_supported,
            )
            _append(
                item, "fixed_broken_handle_closure", fixed_broken_handle_error,
                mask & fixed_handle_supported,
            )
            _append(
                item, "fixed_intact_pair_closure", fixed_intact_pair,
                mask & fixed_pair_supported,
            )
            _append(
                item, "fixed_broken_pair_closure", fixed_broken_pair_error,
                mask & fixed_pair_supported,
            )
            handle_changed_rows = changed_handle.any(dim=1) & mask
            pair_changed_rows = changed_pair.any(dim=1) & mask
            item["handle_rows_touched"] += int(handle_changed_rows.sum())
            item["handle_factors_touched"] += int(changed_handle[mask].sum())
            item["handle_valid_factors"] += int(fixed["handle_factor_valid"][mask].sum())
            item["pair_rows_touched"] += int(pair_changed_rows.sum())
            item["pair_factors_touched"] += int(changed_pair[mask].sum())
            item["pair_valid_factors"] += int(fixed["pair_factor_valid"][mask].sum())
            sign_mask = mask & yaw_valid
            item["candidate_sign"] += int((
                torch.sign(states["candidate"][:, 3]) == torch.sign(target[:, 3])
            )[sign_mask].sum())
            item["control_sign"] += int((
                torch.sign(states["control"][:, 3]) == torch.sign(target[:, 3])
            )[sign_mask].sum())
            item["sign_count"] += int(sign_mask.sum())
            item["count"] += int(mask.sum())
        for name in saved_fields:
            saved_fields[name].append(fields[name].detach().cpu())
        saved_target.append(target.detach().cpu())
        saved_class.append(batch["motion_class"].detach().cpu())

    groups = {}
    for name, item in storage.items():
        if item["count"] < 1:
            raise ValueError(f"omega-first diagnostic group lacks support: {name}")
        metrics = {
            "sample_count": item["count"],
            "candidate_yaw_sign_accuracy": (
                item["candidate_sign"] / item["sign_count"]
                if item["sign_count"] else None
            ),
            "control_yaw_sign_accuracy": (
                item["control_sign"] / item["sign_count"]
                if item["sign_count"] else None
            ),
            "intervention_coverage": {
                key: item[key] for key in INTERVENTION_COVERAGE_FIELDS
            },
        }
        for key, values in item.items():
            if isinstance(values, list) and values:
                metrics[key] = _motion_distribution(values)
        groups[name] = metrics

    all_fields = {
        name: torch.cat(values).to(device) for name, values in saved_fields.items()
    }
    all_target = torch.cat(saved_target).to(device)
    all_class = torch.cat(saved_class).to(device)
    with (
        torch.autocast("cuda", dtype=_cuda_amp_dtype())
        if device.type == "cuda" else nullcontext()
    ):
        all_history = model.context(**all_fields)
        all_candidate = model._state_from_raw_history(all_history)
        full_broken_handle_history = model._roll_raw_geometry(
            all_history, handle=True,
            event_count=all_fields["history_event_mask"].shape[1],
        )
        full_broken_pair_history = _cross_sample_geometry_derangement(
            all_history, handle=False,
        )
        full_broken_pair_state = model._state_from_raw_history(
            full_broken_pair_history
        )["state"]["motion_state_physical"]
        full_broken_handle_state = model._state_from_raw_history(
            full_broken_handle_history
        )["state"]["motion_state_physical"]
        full_state = all_candidate["state"]["motion_state_normalized"]
        full_fixed = model.motion_state_head.decode_closure_at_state(
            all_history["_handle_geometry_raw"],
            all_history["_handle_kinematics_raw"],
            all_history["_handle_raw_valid"],
            all_history["_pair_geometry_raw"],
            all_history["_pair_kinematics_raw"],
            all_history["_pair_raw_valid"], full_state,
        )
        full_fixed_broken_handle = model.motion_state_head.decode_closure_at_state(
            full_broken_handle_history["_handle_geometry_raw"],
            full_broken_handle_history["_handle_kinematics_raw"],
            full_broken_handle_history["_handle_raw_valid"],
            full_broken_handle_history["_pair_geometry_raw"],
            full_broken_handle_history["_pair_kinematics_raw"],
            full_broken_handle_history["_pair_raw_valid"], full_state,
        )
        full_fixed_broken_pair = model.motion_state_head.decode_closure_at_state(
            full_broken_pair_history["_handle_geometry_raw"],
            full_broken_pair_history["_handle_kinematics_raw"],
            full_broken_pair_history["_handle_raw_valid"],
            full_broken_pair_history["_pair_geometry_raw"],
            full_broken_pair_history["_pair_kinematics_raw"],
            full_broken_pair_history["_pair_raw_valid"], full_state,
        )
    full_changed_handle = (
        full_fixed_broken_handle["handle_decoder_geometry"]
        != full_fixed["handle_decoder_geometry"]
    ).any(dim=-1) & full_fixed["handle_factor_valid"]
    full_changed_pair = (
        full_fixed_broken_pair["pair_decoder_geometry"]
        != full_fixed["pair_decoder_geometry"]
    ).any(dim=-1) & full_fixed["pair_factor_valid"]
    full_intact_handle, full_handle_supported = _sample_closure_error(
        full_fixed["angular_handle_residual"], full_changed_handle,
    )
    full_broken_handle_error, _ = _sample_closure_error(
        full_fixed_broken_handle["angular_handle_residual"],
        full_changed_handle,
    )
    full_intact_pair, full_pair_supported = _sample_closure_error(
        full_fixed["pair_residual"], full_changed_pair,
    )
    full_broken_pair_error, _ = _sample_closure_error(
        full_fixed_broken_pair["pair_residual"], full_changed_pair,
    )
    full_pair_count = all_candidate["history"]["pair_flow_available"].sum(dim=1)
    full_speed = torch.linalg.vector_norm(all_target[:, :2], dim=-1)
    full_combined = all_class == 3
    full_history32 = all_candidate["history"]["history_active_count"] == 32
    full_masks = {
        "overall": torch.ones_like(full_combined),
        "combined": full_combined,
        "combined_speed_gt_1_7": full_combined & (full_speed > 1.7),
        "core": (
            full_combined & (full_speed <= 1.2) & full_history32
            & (full_pair_count == 3)
        ),
        "pair0": full_pair_count == 0,
        "combined_pair1": full_combined & (full_pair_count == 1),
        "combined_pair2": full_combined & (full_pair_count == 2),
        "combined_pair3": full_combined & (full_pair_count == 3),
    }
    full_broken_pair_velocity, full_broken_pair_yaw, _ = _state_errors(
        full_broken_pair_state, all_target,
    )
    full_broken_handle_velocity, full_broken_handle_yaw, _ = _state_errors(
        full_broken_handle_state, all_target,
    )
    full_candidate_velocity, full_candidate_yaw, _ = _state_errors(
        all_candidate["state"]["motion_state_physical"], all_target,
    )
    for name, mask in full_masks.items():
        group = groups[name]
        group["broken_pair_velocity"] = _motion_distribution([
            full_broken_pair_velocity[mask].float().cpu().numpy()
        ])
        group["broken_pair_yaw"] = _motion_distribution([
            full_broken_pair_yaw[mask].float().cpu().numpy()
        ])
        handle_changed_rows = full_changed_handle.any(dim=1) & mask
        if bool(handle_changed_rows.any()):
            group["candidate_handle_intervention_velocity"] = _motion_distribution([
                full_candidate_velocity[
                    handle_changed_rows
                ].float().cpu().numpy()
            ])
            group["candidate_handle_intervention_yaw"] = _motion_distribution([
                full_candidate_yaw[handle_changed_rows].float().cpu().numpy()
            ])
            group["broken_handle_intervention_velocity"] = _motion_distribution([
                full_broken_handle_velocity[
                    handle_changed_rows
                ].float().cpu().numpy()
            ])
            group["broken_handle_intervention_yaw"] = _motion_distribution([
                full_broken_handle_yaw[
                    handle_changed_rows
                ].float().cpu().numpy()
            ])
        pair_changed_rows = full_changed_pair.any(dim=1) & mask
        if bool(pair_changed_rows.any()):
            group["candidate_pair_intervention_velocity"] = _motion_distribution([
                full_candidate_velocity[pair_changed_rows].float().cpu().numpy()
            ])
            group["candidate_pair_intervention_yaw"] = _motion_distribution([
                full_candidate_yaw[pair_changed_rows].float().cpu().numpy()
            ])
            group["broken_pair_intervention_velocity"] = _motion_distribution([
                full_broken_pair_velocity[
                    pair_changed_rows
                ].float().cpu().numpy()
            ])
            group["broken_pair_intervention_yaw"] = _motion_distribution([
                full_broken_pair_yaw[pair_changed_rows].float().cpu().numpy()
            ])
        handle_mask = mask & full_handle_supported
        if bool(handle_mask.any()):
            group["fixed_intact_handle_closure"] = _motion_distribution([
                full_intact_handle[handle_mask].float().cpu().numpy()
            ])
            group["fixed_broken_handle_closure"] = _motion_distribution([
                full_broken_handle_error[handle_mask].float().cpu().numpy()
            ])
        pair_mask = mask & full_pair_supported
        if bool(pair_mask.any()):
            group["fixed_intact_pair_closure"] = _motion_distribution([
                full_intact_pair[pair_mask].float().cpu().numpy()
            ])
            group["fixed_broken_pair_closure"] = _motion_distribution([
                full_broken_pair_error[pair_mask].float().cpu().numpy()
            ])
        group["intervention_coverage"] = {
            "handle_rows_touched": int((
                full_changed_handle.any(dim=1) & mask
            ).sum()),
            "handle_factors_touched": int(full_changed_handle[mask].sum()),
            "handle_valid_factors": int(
                full_fixed["handle_factor_valid"][mask].sum()
            ),
            "pair_rows_touched": int((
                full_changed_pair.any(dim=1) & mask
            ).sum()),
            "pair_factors_touched": int(full_changed_pair[mask].sum()),
            "pair_valid_factors": int(
                full_fixed["pair_factor_valid"][mask].sum()
            ),
        }
    physical_cross = {}
    for relation in (
        "opposite_sign_similar_magnitude", "same_sign_different_magnitude",
    ):
        source_cpu, selected_cpu = _physical_donor_index(
            all_history, all_target, all_class == 3, relation=relation,
        )
        source = source_cpu.to(device)
        selected = selected_cpu.to(device)
        if int(selected.sum()) < 32:
            raise ValueError(f"omega-first crossed support differs: {relation}")
        donor = all_target.index_select(0, source)
        event_count = all_fields["history_event_mask"].shape[1]
        hybrid_aa = _synthetic_twist_history_on_target_support(
            model, all_history, all_target[:, :3], all_target[:, 3],
            event_count=event_count,
        )
        hybrid_ab = _synthetic_twist_history_on_target_support(
            model, all_history, all_target[:, :3], donor[:, 3],
            event_count=event_count,
        )
        hybrid_ba = _synthetic_twist_history_on_target_support(
            model, all_history, donor[:, :3], all_target[:, 3],
            event_count=event_count,
        )
        hybrid_bb = _synthetic_twist_history_on_target_support(
            model, all_history, donor[:, :3], donor[:, 3],
            event_count=event_count,
        )
        with (
            torch.autocast("cuda", dtype=_cuda_amp_dtype())
            if device.type == "cuda" else nullcontext()
        ):
            cells = {
                name: model._state_from_raw_history(value)["state"][
                    "motion_state_physical"
                ]
                for name, value in {
                    "aa": hybrid_aa, "ab": hybrid_ab,
                    "ba": hybrid_ba, "bb": hybrid_bb,
                }.items()
            }
        crossed = cells["ab"]
        yaw_gap = (
            donor[:, 3].abs() - all_target[:, 3].abs()
        ).abs()
        if relation == "opposite_sign_similar_magnitude":
            selected = selected & (yaw_gap <= 1.0)
        else:
            selected = selected & (yaw_gap >= 2.0)
        if int(selected.sum()) < 32:
            raise ValueError(f"omega-first crossed relation support differs: {relation}")
        common_delta = donor[:, :3] - all_target[:, :3]
        common_switch_error = torch.cat((
            torch.linalg.vector_norm(
                (cells["ba"][:, :3] - cells["aa"][:, :3]) - common_delta,
                dim=-1,
            ),
            torch.linalg.vector_norm(
                (cells["bb"][:, :3] - cells["ab"][:, :3]) - common_delta,
                dim=-1,
            ),
        ))
        common_switch_yaw_leak = torch.cat((
            (cells["ba"][:, 3] - cells["aa"][:, 3]).abs(),
            (cells["bb"][:, 3] - cells["ab"][:, 3]).abs(),
        ))
        relative_switch_velocity_leak = torch.cat((
            torch.linalg.vector_norm(
                cells["ab"][:, :3] - cells["aa"][:, :3], dim=-1,
            ),
            torch.linalg.vector_norm(
                cells["bb"][:, :3] - cells["ba"][:, :3], dim=-1,
            ),
        ))
        relative_yaw_prediction = torch.cat((
            cells["ab"][:, 3] - cells["aa"][:, 3],
            cells["bb"][:, 3] - cells["ba"][:, 3],
        ))
        relative_yaw_truth = torch.cat((
            donor[:, 3] - all_target[:, 3],
            donor[:, 3] - all_target[:, 3],
        ))
        factorial_selected = torch.cat((selected, selected))
        predicted_common_delta = torch.cat((
            cells["ba"][:, :3] - cells["aa"][:, :3],
            cells["bb"][:, :3] - cells["ab"][:, :3],
        ))
        common_delta_truth = torch.cat((common_delta, common_delta))
        common_axis_transfer = {}
        for axis, axis_name in enumerate(("x", "y")):
            axis_truth = common_delta_truth[:, axis]
            axis_support_count = int((
                factorial_selected & (axis_truth.abs() > 0.5)
            ).sum())
            common_axis_transfer[axis_name] = {
                "support_count": axis_support_count,
                "transfer": _transfer_metrics(
                    predicted_common_delta[:, axis], axis_truth,
                    factorial_selected,
                ),
            }
        selected_common = common_delta[selected]
        selected_planar = selected_common[:, :2]
        direction_supported = torch.linalg.vector_norm(
            selected_planar, dim=-1,
        ) > 0.5
        quadrants = (
            (selected_planar[:, 0] >= 0).to(torch.int64) * 2
            + (selected_planar[:, 1] >= 0).to(torch.int64)
        )
        physical_cross[relation] = {
            "sample_count": int(selected.sum()),
            "unique_donor_count": int(torch.unique(source[selected]).numel()),
            "donor_target_absolute_yaw_gap_rad_s": _motion_distribution([
                yaw_gap[selected].float().cpu().numpy()
            ]),
            "velocity_error_to_injected_truth_mps": _motion_distribution([
                torch.linalg.vector_norm(
                    crossed[:, :3] - all_target[:, :3], dim=-1,
                )[selected].float().cpu().numpy()
            ]),
            "yaw_transfer": _transfer_metrics(
                crossed[:, 3], donor[:, 3], selected,
            ),
            "factorial_aa_ab_ba_bb": {
                "common_switch_truth_delta_magnitude_mps": _motion_distribution([
                    torch.linalg.vector_norm(
                        selected_common, dim=-1,
                    ).float().cpu().numpy()
                ]),
                "common_switch_direction_quadrant_count": int(torch.unique(
                    quadrants[direction_supported]
                ).numel()),
                "common_switch_velocity_axis_transfer": common_axis_transfer,
                "common_switch_velocity_delta_error_mps": _motion_distribution([
                    common_switch_error[factorial_selected].float().cpu().numpy()
                ]),
                "common_switch_yaw_leak_rad_s": _motion_distribution([
                    common_switch_yaw_leak[factorial_selected].float().cpu().numpy()
                ]),
                "relative_switch_velocity_leak_mps": _motion_distribution([
                    relative_switch_velocity_leak[
                        factorial_selected
                    ].float().cpu().numpy()
                ]),
                "relative_switch_yaw_delta_transfer": _transfer_metrics(
                    relative_yaw_prediction, relative_yaw_truth,
                    factorial_selected,
                ),
            },
        }
    reflected_history = _reflect_relative_history_with_truth_common(
        model, all_history, all_target[:, :3],
        event_count=all_fields["history_event_mask"].shape[1],
    )
    with (
        torch.autocast("cuda", dtype=_cuda_amp_dtype())
        if device.type == "cuda" else nullcontext()
    ):
        reflected = model._state_from_raw_history(reflected_history)["state"][
            "motion_state_physical"
        ]
    reversal_selected = (all_class == 3) & (all_target[:, 3].abs() > 0.5)
    if int(reversal_selected.sum()) < 128:
        raise ValueError("omega-first relative reversal support differs")
    return {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "validation_only": True,
        "test_accessed": False,
        "seed": seed,
        "v8_joint_control_checkpoint": str(
            V8_JOINT_CONTROL_CHECKPOINTS[seed].resolve()
        ),
        "v8_joint_control_checkpoint_sha256": V8_JOINT_CONTROL_SHA256[seed],
        "groups": groups,
        "write_isolation": {
            "zero_velocity_max_absolute_yaw_difference_normalized": (
                zero_velocity_yaw_max
            ),
        },
        "common_ramp_equivariance": {
            "velocity_delta_error_mps": _motion_distribution(
                ramp_velocity_equivariance
            ),
            "yaw_invariance_error_rad_s": _motion_distribution(
                ramp_yaw_invariance
            ),
        },
        "relative_reversal_equivariance": {
            "sample_count": int(reversal_selected.sum()),
            "velocity_prediction_invariance_mps": _motion_distribution([
                torch.linalg.vector_norm(
                    reflected[:, :3]
                    - all_candidate["state"]["motion_state_physical"][:, :3],
                    dim=-1,
                )[reversal_selected].float().cpu().numpy()
            ]),
            "yaw_prediction_antisymmetry_rad_s": _motion_distribution([
                (
                    reflected[:, 3]
                    + all_candidate["state"]["motion_state_physical"][:, 3]
                ).abs()[reversal_selected].float().cpu().numpy()
            ]),
            "velocity_error_to_unchanged_truth_mps": _motion_distribution([
                torch.linalg.vector_norm(
                    reflected[:, :3] - all_target[:, :3], dim=-1,
                )[reversal_selected].float().cpu().numpy()
            ]),
            "yaw_transfer_to_negated_truth": _transfer_metrics(
                reflected[:, 3], -all_target[:, 3], reversal_selected,
            ),
        },
        "factor_level_truth_common_donor_relative_cross": physical_cross,
    }


def _validate_completed_omega_first_probe(
    args, checkpoint: Path, parameter_count: int,
) -> dict:
    output = Path(args.output).resolve()
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    fixed = manifest.get("fixed_final_checkpoint")
    contract = manifest.get("contract")
    expected_checkpoint = output / "checkpoints" / "checkpoint-update-000200.pt"
    if (
        manifest.get("schema_version") != RUN_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("progress", {}).get("global_update") != 200
        or manifest.get("state_gate_only") is not True
        or manifest.get("state_gate_future_modules_unchanged") is not True
        or not isinstance(fixed, dict) or fixed.get("update") != 200
        or fixed.get("selected_by_validation") is not False
        or checkpoint.resolve() != expected_checkpoint.resolve()
        or Path(str(fixed.get("path", ""))).resolve() != checkpoint.resolve()
        or fixed.get("sha256") != sha256_file(checkpoint)
        or not isinstance(contract, dict)
        or manifest.get("contract_sha256") != _json_sha256(contract)
    ):
        raise ValueError("omega-first fixed artifact is incomplete")
    recorded_git = manifest.get("provenance", {}).get("git", {})
    current_git = _git_state()
    if (
        recorded_git.get("worktree_dirty") is not False
        or current_git.get("worktree_dirty") is not False
        or recorded_git.get("git_commit") != current_git.get("git_commit")
    ):
        raise ValueError("omega-first source checkout changed")
    expected_callables = {
        "state_loss_function": _callable_contract(omega_first_ordered_state_loss),
        "state_step_function": _callable_contract(omega_first_ordered_train_step),
        "final_diagnostic_function": _callable_contract(
            omega_first_ordered_validation_diagnostics
        ),
    }
    for place in ("contract", "provenance"):
        source = manifest.get(place, {})
        if any(source.get(name) != value for name, value in expected_callables.items()):
            raise ValueError("omega-first callable contracts differ")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != RUN_SCHEMA
        or payload.get("model_class") != "AnonymousOmegaFirstOrderedClosureProbe"
        or payload.get("checkpoint_selected_by_validation") is not False
        or payload.get("fixed_endpoint") is not True
        or payload.get("progress") != {
            "global_update": 200, "stage": "motion_state", "stage_update": 200,
        }
        or state_dict_sha256(payload.get("model", {}))
        != payload.get("model_state_dict_sha256")
        or payload.get("contract_sha256") != manifest.get("contract_sha256")
        or payload.get("run_id") != manifest.get("run_id")
        or payload.get("model_config") != manifest.get("model_config")
        or _json_sha256(payload.get("provenance"))
        != _json_sha256(manifest.get("provenance"))
        or _json_sha256(payload.get("validation_history"))
        != _json_sha256(manifest.get("validation_history"))
        or _json_sha256(payload.get("final_diagnostics"))
        != _json_sha256(manifest.get("final_diagnostics"))
    ):
        raise ValueError("omega-first checkpoint identity differs")
    history = payload.get("validation_history")
    if (
        not isinstance(history, list) or not history
        or _json_sha256(history[-1].get("metrics"))
        != _json_sha256(manifest.get("final_validation"))
        or manifest.get("state_substage_counts")
        != {"omega_first_ordered_closure_structural_probe": 200}
        or manifest.get("state_substage_transitions") != [{
            "global_update": 1,
            "substage": "omega_first_ordered_closure_structural_probe",
        }]
    ):
        raise ValueError("omega-first validation or update schedule differs")
    for name in (
        "gradient_isolation_verified", "state_substage_counts",
        "state_substage_transitions", "state_branch_hash_history",
        "final_diagnostics",
    ):
        if _json_sha256(payload.get(name)) != _json_sha256(manifest.get(name)):
            raise ValueError(f"omega-first checkpoint/manifest {name} differs")
    model_state = payload["model"]
    config = payload["model_config"]
    scale = config.get("motion_state_scale")
    if not isinstance(scale, list) or len(scale) != 4:
        raise ValueError("omega-first checkpoint motion scale differs")
    reconstructed = AnonymousOmegaFirstOrderedClosureProbe(
        velocity_scale_mps=tuple(float(value) for value in scale[:3]),
        yaw_rate_scale_rad_s=float(scale[3]),
        channels=args.channels, dropout=args.dropout,
        message_layers=args.message_layers, basis_count=args.basis_count,
    )
    if reconstructed.config != config:
        raise ValueError("omega-first reconstructed model config differs")
    reconstructed.load_state_dict(model_state, strict=True)
    reconstructed_count = sum(
        parameter.numel() for name in STATE_MODULES
        for parameter in getattr(reconstructed, name).parameters()
    )
    if reconstructed_count != parameter_count:
        raise ValueError("omega-first reconstructed capacity differs")
    actual_hashes = {
        name: state_dict_sha256({
            key[len(name) + 1:]: value for key, value in model_state.items()
            if key.startswith(name + ".")
        })
        for name in ALL_TRAINABLE_MODULES
    }
    if actual_hashes != manifest.get("trainable_final_state_dict_sha256"):
        raise ValueError("omega-first final module hashes differ")
    initial_hashes = manifest.get("trainable_initial_state_dict_sha256")
    if any(
        actual_hashes.get(name) != initial_hashes.get(name)
        for name in FROZEN_FUTURE_MODULES
    ):
        raise ValueError("omega-first frozen future modules changed")
    diagnostics = manifest.get("final_diagnostics")
    if not isinstance(diagnostics, dict) or set(diagnostics) != DIAGNOSTIC_FIELDS:
        raise ValueError("omega-first diagnostic fields differ")
    write_isolation = diagnostics["write_isolation"]
    if (
        not isinstance(write_isolation, dict)
        or set(write_isolation) != WRITE_ISOLATION_FIELDS
    ):
        raise ValueError("omega-first write isolation fields differ")
    write_isolation_value = write_isolation[
        "zero_velocity_max_absolute_yaw_difference_normalized"
    ]
    if (
        isinstance(write_isolation_value, bool)
        or not isinstance(write_isolation_value, (int, float))
        or not math.isfinite(float(write_isolation_value))
        or float(write_isolation_value) != 0.0
    ):
        raise ValueError("omega-first write isolation differs")
    if (
        diagnostics.get("schema_version") != DIAGNOSTIC_SCHEMA
        or diagnostics.get("validation_only") is not True
        or diagnostics.get("test_accessed") is not False
        or diagnostics.get("seed") != args.seed
        or Path(diagnostics.get("v8_joint_control_checkpoint", "")).resolve()
        != V8_JOINT_CONTROL_CHECKPOINTS[args.seed].resolve()
        or diagnostics.get("v8_joint_control_checkpoint_sha256")
        != V8_JOINT_CONTROL_SHA256[args.seed]
        or set(diagnostics.get("groups", {})) != set(GROUP_NAMES)
        or set(diagnostics.get(
            "factor_level_truth_common_donor_relative_cross", {}
        )) != {
            "opposite_sign_similar_magnitude",
            "same_sign_different_magnitude",
        }
        or set(diagnostics.get("common_ramp_equivariance", {})) != {
            "velocity_delta_error_mps", "yaw_invariance_error_rad_s",
        }
        or set(diagnostics.get("relative_reversal_equivariance", {})) != {
            "sample_count", "velocity_prediction_invariance_mps",
            "yaw_prediction_antisymmetry_rad_s",
            "velocity_error_to_unchanged_truth_mps", "yaw_transfer_to_negated_truth",
        }
    ):
        raise ValueError("omega-first diagnostics are incomplete")
    source_manifest = json.loads(
        (Path(args.dataset).resolve() / "dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if source_manifest.get("test_accessed") is not False:
        raise ValueError("omega-first source dataset accessed test")
    result = {
        "schema_version": "stage3-v12-omega-first-ordered-probe-result-v1",
        "seed": args.seed,
        "fixed_updates": 200,
        "test_accessed": False,
        "source_commit": recorded_git["git_commit"],
        "contract_sha256": manifest["contract_sha256"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": fixed["sha256"],
        "total_state_parameter_count": parameter_count,
        "diagnostics": diagnostics,
    }
    groups = diagnostics["groups"]
    standard = manifest["final_validation"]["motion_state"]
    for name in ("overall", "combined", "combined_speed_gt_1_7"):
        if (
            _json_sha256(groups[name]["candidate_velocity"])
            != _json_sha256(standard[name]["velocity_vector_error_mps"])
            or _json_sha256(groups[name]["candidate_yaw"])
            != _json_sha256(standard[name]["yaw_absolute_error_rad_s"])
        ):
            raise ValueError(f"omega-first baseline differs: {name}")
    mapping = {
        "overall_velocity_mean_mps": ("overall", "candidate_velocity"),
        "overall_yaw_mean_rad_s": ("overall", "candidate_yaw"),
        "combined_velocity_mean_mps": ("combined", "candidate_velocity"),
        "combined_yaw_mean_rad_s": ("combined", "candidate_yaw"),
        "high_speed_combined_velocity_mean_mps": (
            "combined_speed_gt_1_7", "candidate_velocity",
        ),
        "high_speed_combined_yaw_mean_rad_s": (
            "combined_speed_gt_1_7", "candidate_yaw",
        ),
        "core_yaw_mean_rad_s": ("core", "candidate_yaw"),
        "control_core_yaw_mean_rad_s": ("core", "control_yaw"),
    }
    for count in (1, 2, 3):
        group = f"combined_pair{count}"
        for metric in ("velocity", "yaw"):
            mapping[f"pair{count}_{metric}_mean"] = (
                group, f"candidate_{metric}",
            )
            mapping[f"control_pair{count}_{metric}_mean"] = (
                group, f"control_{metric}",
            )
    for key, (group, metric) in mapping.items():
        result[key] = _validated_mean(groups[group], metric, name=key)
    sign = standard["overall"]["yaw_sign_accuracy_abs_truth_gt_0_5"]
    if (
        isinstance(sign, bool) or not isinstance(sign, (int, float))
        or not math.isfinite(float(sign))
    ):
        raise ValueError("omega-first yaw sign is invalid")
    result["overall_yaw_sign_accuracy"] = float(sign)
    return result


def main() -> None:
    args = build_omega_first_probe_parser().parse_args()
    _validate_args(args)
    git = _git_state()
    if git.get("worktree_dirty") is not False or git.get("git_commit") in {
        None, "unknown",
    }:
        raise RuntimeError("omega-first probe requires a clean source commit")
    v77 = Path(args.v77_control_checkpoint).resolve()
    _preflight_control(v77)
    parameter_count = _state_parameter_count(
        AnonymousOmegaFirstOrderedClosureProbe, args,
    )
    if (
        abs(parameter_count - V8_JOINT_REACHABLE_STATE_PARAMETERS)
        / V8_JOINT_REACHABLE_STATE_PARAMETERS > 0.05
    ):
        raise ValueError("omega-first capacity differs from V8 by more than 5%")
    checkpoint = train(
        args,
        model_class=AnonymousOmegaFirstOrderedClosureProbe,
        state_loss_function=omega_first_ordered_state_loss,
        state_step_function=omega_first_ordered_train_step,
        final_diagnostic_function=omega_first_ordered_validation_diagnostics,
        run_schema=RUN_SCHEMA,
        extra_source_paths={
            "omega_first_model_and_step": Path(__file__).with_name(
                "omega_first_ordered_closure_future.py"
            ),
            "inherited_global_flow_closure_model": Path(__file__).with_name(
                "global_flow_closure_future.py"
            ),
            "omega_first_runner_and_diagnostics": Path(__file__),
            "v11_mechanism_audit_helpers": Path(__file__).with_name(
                "diagnose_global_flow_closure_mechanism.py"
            ),
            "v9_paired_token_context": Path(__file__).with_name(
                "paired_twist_set_future.py"
            ),
        },
        state_gate_only=True,
        frozen_initialization_checkpoint=v77,
        frozen_initialization_modules=FROZEN_FUTURE_MODULES,
    )
    report = _validate_completed_omega_first_probe(
        args, checkpoint, parameter_count,
    )
    _atomic_json(Path(args.output).resolve() / "probe_result.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
