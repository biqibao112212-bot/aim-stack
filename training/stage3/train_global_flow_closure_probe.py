"""Run one fixed 200-update V11 learned global-history-closure probe."""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path

import numpy as np
import torch

from .global_flow_closure_future import (
    AnonymousGlobalFlowClosureProbe,
    global_flow_closure_state_loss,
    global_flow_closure_train_step,
)
from .motion_truth_supervision import MOTION_TARGET_FIELD
from .observable_future_pnp_ab import sha256_file, state_dict_sha256
from .train_anonymous_vehicle_motion import _cuda_amp_dtype, _json_sha256
from .train_causal_physical_ab import _git_state, _to_device
from .joint_rigid_flow_probe import LOCAL_LAG_SCALES_S
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
    ALL_TRAINABLE_MODULES,
    _callable_contract,
    _prepare_batch,
    train,
)


RUN_SCHEMA = "stage3-anonymous-global-flow-closure-structural-probe-v11"
DIAGNOSTIC_SCHEMA = "stage3-v11-global-flow-closure-validation-diagnostics-v1"
GROUP_NAMES = (
    "overall", "combined", "combined_speed_gt_1_7", "core", "pair0",
    "combined_pair1", "combined_pair2", "combined_pair3",
)


def build_closure_probe_parser():
    parser = build_probe_parser()
    parser.description = "bounded 200-update V11 global history-closure probe"
    return parser


def _sample_closure_error(
    residual: torch.Tensor, valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    distance = torch.linalg.vector_norm(residual.float(), dim=-1)
    supported = valid.any(dim=1)
    value = (
        distance * valid.to(distance.dtype)
    ).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    return value, supported


def _append(storage: dict, key: str, value: torch.Tensor, mask: torch.Tensor) -> None:
    if bool(mask.any()):
        storage[key].append(value[mask].float().cpu().numpy())


def _cross_source_index(
    handle_valid: torch.Tensor,
    pair_valid: torch.Tensor,
    history_active_count: torch.Tensor,
    target: torch.Tensor,
    combined: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose a far donor with the same discrete history/pair support class."""
    handle_cpu = handle_valid.to(torch.bool).cpu()
    pair_cpu = pair_valid.to(torch.bool).cpu()
    active_cpu = history_active_count.to(torch.long).cpu()
    target_cpu = target.float().cpu()
    combined_cpu = combined.to(torch.bool).cpu()
    source = torch.arange(target.shape[0], dtype=torch.long)
    selected = torch.zeros(target.shape[0], dtype=torch.bool)
    groups: dict[bytes, list[int]] = {}
    event_count = pair_cpu.shape[1] // len(LOCAL_LAG_SCALES_S)
    pair_scale_available = pair_cpu.reshape(
        pair_cpu.shape[0], event_count, len(LOCAL_LAG_SCALES_S),
    ).any(dim=1)
    for row in torch.nonzero(combined_cpu, as_tuple=False).flatten().tolist():
        signature = (
            active_cpu[row].numpy().tobytes()
            + pair_scale_available[row].numpy().tobytes()
        )
        groups.setdefault(signature, []).append(row)
    for rows in groups.values():
        if len(rows) < 2:
            continue
        values = target_cpu[rows]
        velocity = torch.cdist(values[:, :3], values[:, :3])
        yaw = (values[:, None, 3] - values[None, :, 3]).abs()
        opposite = (
            torch.sign(values[:, None, 3])
            != torch.sign(values[None, :, 3])
        ).to(values.dtype)
        score = velocity / 1.0 + yaw / 3.0 + 4.0 * opposite
        score.fill_diagonal_(-torch.inf)
        donor = score.argmax(dim=1)
        for local, row in enumerate(rows):
            source[row] = rows[int(donor[local])]
            selected[row] = True
    return source, selected


@torch.inference_mode()
def global_flow_closure_validation_diagnostics(
    model,
    loader,
    mapper,
    s_model,
    h_model,
    device: torch.device,
) -> dict:
    seed = int(torch.initial_seed())
    if seed not in PROBE_SEEDS:
        raise ValueError("v11 diagnostic seed differs")
    control = _load_v8_control(seed, model).to(device).eval().requires_grad_(False)
    model.eval()
    storage = {
        group: {
            key: [] for key in (
                "candidate_velocity", "candidate_yaw", "candidate_yaw_signed",
                "control_velocity", "control_yaw", "control_yaw_signed",
                "broken_handle_velocity", "broken_handle_yaw",
                "broken_pair_velocity", "broken_pair_yaw",
                "zero_refinement_velocity", "zero_refinement_yaw",
                "handle_closure", "pair_closure",
            )
        } | {"candidate_sign": 0, "control_sign": 0, "sign_count": 0, "count": 0}
        for group in GROUP_NAMES
    }
    saved_fields: dict[str, list[torch.Tensor]] = {
        name: [] for name in model._field_names()
    }
    saved_target: list[torch.Tensor] = []
    saved_class: list[torch.Tensor] = []
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
            broken_handle = model.estimate_motion_state_broken_handle_geometry(
                **fields
            )
            broken_pair = model.estimate_motion_state_broken_pair_geometry(**fields)
            zero_refinement = model.estimate_motion_state_zero_refinement(**fields)
            baseline = control.estimate_motion_state(**fields)
        target = batch[MOTION_TARGET_FIELD]
        states = {
            "candidate": candidate["state"]["motion_state_physical"],
            "broken_handle": broken_handle["state"]["motion_state_physical"],
            "broken_pair": broken_pair["state"]["motion_state_physical"],
            "zero_refinement": zero_refinement["state"]["motion_state_physical"],
            "control": baseline["state"]["motion_state_physical"],
        }
        velocity = {
            name: torch.linalg.vector_norm(value[:, :3] - target[:, :3], dim=-1)
            for name, value in states.items()
        }
        yaw_signed = {name: value[:, 3] - target[:, 3] for name, value in states.items()}
        yaw = {name: value.abs() for name, value in yaw_signed.items()}
        handle_closure, handle_supported = _sample_closure_error(
            candidate["state"]["handle_closure_residual_normalized"],
            candidate["state"]["handle_factor_valid"],
        )
        pair_closure, pair_supported = _sample_closure_error(
            candidate["state"]["pair_closure_residual_normalized"],
            candidate["state"]["pair_factor_valid"],
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
        for name, mask in masks.items():
            if not bool(mask.any()):
                continue
            item = storage[name]
            for prefix in ("candidate", "control"):
                _append(item, f"{prefix}_velocity", velocity[prefix], mask)
                _append(item, f"{prefix}_yaw", yaw[prefix], mask)
                _append(item, f"{prefix}_yaw_signed", yaw_signed[prefix], mask)
            for prefix in ("broken_handle", "broken_pair", "zero_refinement"):
                _append(item, f"{prefix}_velocity", velocity[prefix], mask)
                _append(item, f"{prefix}_yaw", yaw[prefix], mask)
            _append(item, "handle_closure", handle_closure, mask & handle_supported)
            _append(item, "pair_closure", pair_closure, mask & pair_supported)
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

    result = {}
    for name, item in storage.items():
        if item["count"] < 1:
            raise ValueError(f"v11 diagnostic group has no support: {name}")
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
        }
        for key, values in item.items():
            if isinstance(values, list) and values:
                metrics[key] = _motion_distribution(values)
        result[name] = metrics

    all_fields = {
        name: torch.cat(values, dim=0).to(device)
        for name, values in saved_fields.items()
    }
    all_target = torch.cat(saved_target, dim=0).to(device)
    all_class = torch.cat(saved_class, dim=0).to(device)
    amp = (
        torch.autocast("cuda", dtype=_cuda_amp_dtype())
        if device.type == "cuda" else nullcontext()
    )
    with amp:
        raw_history = model.context(**all_fields)
        source_cpu, selected_cpu = _cross_source_index(
            raw_history["_handle_raw_valid"], raw_history["_pair_raw_valid"],
            raw_history["history_active_count"],
            all_target, all_class == 3,
        )
        source = source_cpu.to(device)
        selected = selected_cpu.to(device)
        crossed_state = model.estimate_motion_state_crossed_rotation_factors(
            source, **all_fields,
        )["state"]
        broken_crossed_state = (
            model.estimate_motion_state_crossed_rotation_broken_pairing(
                source, **all_fields,
            )["state"]
        )
        crossed = crossed_state["motion_state_physical"]
        broken_crossed = broken_crossed_state["motion_state_physical"]
    donor_target = all_target.index_select(0, source)
    if int(selected.sum()) < 32:
        raise ValueError("v11 crossed rotation diagnostic lacks support")
    crossed_velocity_a = torch.linalg.vector_norm(
        crossed[:, :3] - all_target[:, :3], dim=-1,
    )
    crossed_velocity_b = torch.linalg.vector_norm(
        crossed[:, :3] - donor_target[:, :3], dim=-1,
    )
    crossed_yaw_a = (crossed[:, 3] - all_target[:, 3]).abs()
    crossed_yaw_b = (crossed[:, 3] - donor_target[:, 3]).abs()
    broken_crossed_velocity = torch.linalg.vector_norm(
        broken_crossed[:, :3] - all_target[:, :3], dim=-1,
    )
    broken_crossed_yaw = (broken_crossed[:, 3] - donor_target[:, 3]).abs()
    crossed_handle_closure, _ = _sample_closure_error(
        crossed_state["handle_closure_residual_normalized"],
        crossed_state["handle_factor_valid"],
    )
    crossed_pair_closure, _ = _sample_closure_error(
        crossed_state["pair_closure_residual_normalized"],
        crossed_state["pair_factor_valid"],
    )
    broken_handle_closure, _ = _sample_closure_error(
        broken_crossed_state["handle_closure_residual_normalized"],
        broken_crossed_state["handle_factor_valid"],
    )
    broken_pair_closure, _ = _sample_closure_error(
        broken_crossed_state["pair_closure_residual_normalized"],
        broken_crossed_state["pair_factor_valid"],
    )
    crossed_closure = crossed_handle_closure + crossed_pair_closure
    broken_crossed_closure = broken_handle_closure + broken_pair_closure
    yaw_valid = donor_target[:, 3].abs() > 0.5
    cross_mask = selected
    cross_sign_mask = selected & yaw_valid
    cross = {
        "sample_count": int(cross_mask.sum()),
        "velocity_error_to_translation_source": _motion_distribution([
            crossed_velocity_a[cross_mask].float().cpu().numpy()
        ]),
        "velocity_error_to_rotation_donor": _motion_distribution([
            crossed_velocity_b[cross_mask].float().cpu().numpy()
        ]),
        "yaw_error_to_translation_source": _motion_distribution([
            crossed_yaw_a[cross_mask].float().cpu().numpy()
        ]),
        "yaw_error_to_rotation_donor": _motion_distribution([
            crossed_yaw_b[cross_mask].float().cpu().numpy()
        ]),
        "broken_pairing_velocity_error_to_hybrid": _motion_distribution([
            broken_crossed_velocity[cross_mask].float().cpu().numpy()
        ]),
        "broken_pairing_yaw_error_to_hybrid": _motion_distribution([
            broken_crossed_yaw[cross_mask].float().cpu().numpy()
        ]),
        "intact_history_closure_error": _motion_distribution([
            crossed_closure[cross_mask].float().cpu().numpy()
        ]),
        "broken_pairing_history_closure_error": _motion_distribution([
            broken_crossed_closure[cross_mask].float().cpu().numpy()
        ]),
        "yaw_sign_accuracy_to_rotation_donor": float((
            torch.sign(crossed[:, 3]) == torch.sign(donor_target[:, 3])
        )[cross_sign_mask].float().mean()),
    }
    return {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "validation_only": True,
        "test_accessed": False,
        "seed": seed,
        "v8_joint_control_checkpoint": str(
            V8_JOINT_CONTROL_CHECKPOINTS[seed].resolve()
        ),
        "v8_joint_control_checkpoint_sha256": V8_JOINT_CONTROL_SHA256[seed],
        "groups": result,
        "crossed_rotation_factors": cross,
    }


def _validate_completed_closure_probe(
    args, checkpoint: Path, parameter_count: int,
) -> dict:
    output = Path(args.output).resolve()
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    fixed = manifest.get("fixed_final_checkpoint")
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
        or fixed.get("sha256") != sha256_file(checkpoint)
    ):
        raise ValueError("v11 fixed artifact is incomplete or unbound")
    recorded_git = manifest.get("provenance", {}).get("git", {})
    current_git = _git_state()
    if (
        recorded_git.get("worktree_dirty") is not False
        or current_git.get("worktree_dirty") is not False
        or recorded_git.get("git_commit") != current_git.get("git_commit")
    ):
        raise ValueError("v11 source checkout changed during formal run")
    expected_callables = {
        "state_loss_function": _callable_contract(global_flow_closure_state_loss),
        "state_step_function": _callable_contract(global_flow_closure_train_step),
        "final_diagnostic_function": _callable_contract(
            global_flow_closure_validation_diagnostics
        ),
    }
    for place in ("contract", "provenance"):
        source = manifest.get(place, {})
        if any(source.get(name) != value for name, value in expected_callables.items()):
            raise ValueError("v11 callable contracts differ")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != RUN_SCHEMA
        or payload.get("model_class") != "AnonymousGlobalFlowClosureProbe"
        or payload.get("checkpoint_selected_by_validation") is not False
        or payload.get("fixed_endpoint") is not True
        or state_dict_sha256(payload.get("model", {}))
        != payload.get("model_state_dict_sha256")
        or payload.get("contract_sha256") != manifest.get("contract_sha256")
        or payload.get("run_id") != manifest.get("run_id")
        or _json_sha256(payload.get("final_diagnostics"))
        != _json_sha256(manifest.get("final_diagnostics"))
    ):
        raise ValueError("v11 checkpoint identity differs")
    model_state = payload["model"]
    actual_hashes = {
        name: state_dict_sha256({
            key[len(name) + 1:]: value for key, value in model_state.items()
            if key.startswith(name + ".")
        })
        for name in ALL_TRAINABLE_MODULES
    }
    if actual_hashes != manifest.get("trainable_final_state_dict_sha256"):
        raise ValueError("v11 final module hashes differ")
    initial_hashes = manifest.get("trainable_initial_state_dict_sha256")
    if any(
        actual_hashes.get(name) != initial_hashes.get(name)
        for name in FROZEN_FUTURE_MODULES
    ):
        raise ValueError("v11 frozen future hashes differ")
    diagnostics = manifest.get("final_diagnostics")
    if (
        not isinstance(diagnostics, dict)
        or diagnostics.get("schema_version") != DIAGNOSTIC_SCHEMA
        or diagnostics.get("validation_only") is not True
        or diagnostics.get("test_accessed") is not False
        or diagnostics.get("seed") != args.seed
        or Path(diagnostics.get("v8_joint_control_checkpoint", "")).resolve()
        != V8_JOINT_CONTROL_CHECKPOINTS[args.seed].resolve()
        or diagnostics.get("v8_joint_control_checkpoint_sha256")
        != V8_JOINT_CONTROL_SHA256[args.seed]
        or set(diagnostics.get("groups", {})) != set(GROUP_NAMES)
        or diagnostics.get("crossed_rotation_factors", {}).get("sample_count", 0) < 32
    ):
        raise ValueError("v11 diagnostics are incomplete or unbound")
    source_manifest = json.loads(
        (Path(args.dataset).resolve() / "dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if source_manifest.get("test_accessed") is not False:
        raise ValueError("v11 source dataset accessed test")
    result = {
        "schema_version": "stage3-v11-global-flow-closure-probe-result-v1",
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
            raise ValueError(f"v11 diagnostic baseline differs: {name}")
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
            mapping[f"pair{count}_{metric}_mean"] = (group, f"candidate_{metric}")
            mapping[f"control_pair{count}_{metric}_mean"] = (
                group, f"control_{metric}",
            )
            mapping[f"broken_pair_pair{count}_{metric}_mean"] = (
                group, f"broken_pair_{metric}",
            )
    mapping.update({
        "broken_handle_overall_velocity_mean": ("overall", "broken_handle_velocity"),
        "zero_refinement_combined_velocity_mean": (
            "combined", "zero_refinement_velocity",
        ),
        "zero_refinement_combined_yaw_mean": (
            "combined", "zero_refinement_yaw",
        ),
    })
    for key, (group, metric) in mapping.items():
        result[key] = _validated_mean(groups[group], metric, name=key)
    sign = standard["overall"]["yaw_sign_accuracy_abs_truth_gt_0_5"]
    if (
        isinstance(sign, bool) or not isinstance(sign, (int, float))
        or not math.isfinite(float(sign))
    ):
        raise ValueError("v11 yaw-sign accuracy is invalid")
    result["overall_yaw_sign_accuracy"] = float(sign)
    result["crossed_rotation_factors"] = diagnostics["crossed_rotation_factors"]
    return result


def main() -> None:
    args = build_closure_probe_parser().parse_args()
    _validate_args(args)
    git = _git_state()
    if git.get("worktree_dirty") is not False or git.get("git_commit") in {
        None, "unknown",
    }:
        raise RuntimeError("v11 structural probe requires a clean source commit")
    v77 = Path(args.v77_control_checkpoint).resolve()
    _preflight_control(v77)
    parameter_count = _state_parameter_count(AnonymousGlobalFlowClosureProbe, args)
    if (
        abs(parameter_count - V8_JOINT_REACHABLE_STATE_PARAMETERS)
        / V8_JOINT_REACHABLE_STATE_PARAMETERS > 0.05
    ):
        raise ValueError("v11 reachable capacity differs from V8 by more than 5%")
    checkpoint = train(
        args,
        model_class=AnonymousGlobalFlowClosureProbe,
        state_loss_function=global_flow_closure_state_loss,
        state_step_function=global_flow_closure_train_step,
        final_diagnostic_function=global_flow_closure_validation_diagnostics,
        run_schema=RUN_SCHEMA,
        extra_source_paths={
            "v11_runner_and_diagnostics": Path(__file__),
            "v11_model_and_step": Path(__file__).with_name(
                "global_flow_closure_future.py"
            ),
            "v9_paired_token_context": Path(__file__).with_name(
                "paired_twist_set_future.py"
            ),
            "common_velocity_ramp": Path(__file__).with_name(
                "factorized_common_relative_motion_future.py"
            ),
        },
        state_gate_only=True,
        frozen_initialization_checkpoint=v77,
        frozen_initialization_modules=FROZEN_FUTURE_MODULES,
    )
    report = _validate_completed_closure_probe(args, checkpoint, parameter_count)
    _atomic_json(Path(args.output).resolve() / "probe_result.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
