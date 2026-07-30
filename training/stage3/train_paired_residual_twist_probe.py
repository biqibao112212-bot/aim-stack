"""Run one fixed 200-update V10 paired-residual structural probe."""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path

import numpy as np
import torch

from .observable_future_pnp_ab import sha256_file, state_dict_sha256
from .paired_residual_twist_future import (
    AnonymousPairedResidualTwistProbe,
    paired_residual_probe_train_step,
    paired_residual_state_loss,
)
from .train_anonymous_vehicle_motion import _cuda_amp_dtype, _json_sha256
from .train_causal_physical_ab import _git_state, _to_device
from .train_joint_rigid_flow_probe import _state_parameter_count
from .train_paired_twist_set_probe import (
    PROBE_SEEDS,
    V8_JOINT_CONTROL_CHECKPOINTS,
    V8_JOINT_CONTROL_SHA256,
    V8_JOINT_REACHABLE_STATE_PARAMETERS,
    V8_JOINT_TOTAL_STATE_PARAMETERS,
    _load_v8_control,
    _motion_distribution,
    _validate_args,
    _validated_mean,
    build_probe_parser,
)
from .train_pnp_window_mapper_distillation import _atomic_json
from .train_robust_multiscale_motion_future import FROZEN_FUTURE_MODULES, _preflight_control
from .train_stable_motion_bottleneck_future import (
    ALL_TRAINABLE_MODULES,
    _callable_contract,
    _prepare_batch,
    train,
)
from .motion_truth_supervision import MOTION_TARGET_FIELD


RUN_SCHEMA = "stage3-anonymous-paired-residual-twist-structural-probe-v10"
DIAGNOSTIC_SCHEMA = "stage3-v10-paired-residual-validation-diagnostics-v1"


def build_residual_probe_parser():
    parser = build_probe_parser()
    parser.description = "bounded 200-update V10 paired-residual structural probe"
    return parser


@torch.inference_mode()
def paired_residual_validation_diagnostics(
    model,
    loader,
    mapper,
    s_model,
    h_model,
    device: torch.device,
) -> dict:
    """Replay V8 plus pairing and zero-residual V10 interventions."""
    seed = int(torch.initial_seed())
    if seed not in PROBE_SEEDS:
        raise ValueError("v10 diagnostic seed differs")
    control = _load_v8_control(seed, model).to(device).eval().requires_grad_(False)
    model.eval()
    group_names = (
        "overall", "combined", "combined_speed_gt_1_7", "core",
        "combined_pair1_2",
    )
    storage = {
        group: {
            "candidate_velocity": [], "candidate_yaw": [],
            "control_velocity": [], "control_yaw": [],
            "broken_velocity": [], "broken_yaw": [],
            "zero_residual_velocity": [],
            "candidate_sign": 0, "control_sign": 0,
            "sign_count": 0, "count": 0,
        }
        for group in group_names
    }
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
            broken = model.estimate_motion_state_broken_pairing(**fields)
            zero_residual = model.estimate_motion_state_zero_rotation_residual(
                **fields
            )
            baseline = control.estimate_motion_state(**fields)
        target = batch[MOTION_TARGET_FIELD]
        candidate_state = candidate["state"]["motion_state_physical"]
        broken_state = broken["state"]["motion_state_physical"]
        zero_state = zero_residual["state"]["motion_state_physical"]
        control_state = baseline["state"]["motion_state_physical"]
        velocity_error = lambda value: torch.linalg.vector_norm(
            value[:, :3] - target[:, :3], dim=-1,
        )
        candidate_velocity = velocity_error(candidate_state)
        broken_velocity = velocity_error(broken_state)
        zero_velocity = velocity_error(zero_state)
        control_velocity = velocity_error(control_state)
        candidate_yaw = (candidate_state[:, 3] - target[:, 3]).abs()
        broken_yaw = (broken_state[:, 3] - target[:, 3]).abs()
        control_yaw = (control_state[:, 3] - target[:, 3]).abs()
        speed = torch.linalg.vector_norm(target[:, :2], dim=-1)
        combined = batch["motion_class"] == 3
        pair_count = candidate["history"]["pair_flow_available"].sum(dim=1)
        history32 = candidate["history"]["history_active_count"] == 32
        pair3 = pair_count == 3
        masks = {
            "overall": torch.ones_like(combined),
            "combined": combined,
            "combined_speed_gt_1_7": combined & (speed > 1.7),
            "core": combined & (speed <= 1.2) & history32 & pair3,
            "combined_pair1_2": combined & ((pair_count == 1) | (pair_count == 2)),
        }
        yaw_valid = target[:, 3].abs() > 0.5
        for name, mask in masks.items():
            if not bool(mask.any()):
                continue
            item = storage[name]
            for key, value in (
                ("candidate_velocity", candidate_velocity),
                ("candidate_yaw", candidate_yaw),
                ("control_velocity", control_velocity),
                ("control_yaw", control_yaw),
                ("broken_velocity", broken_velocity),
                ("broken_yaw", broken_yaw),
                ("zero_residual_velocity", zero_velocity),
            ):
                item[key].append(value[mask].float().cpu().numpy())
            sign_mask = mask & yaw_valid
            item["candidate_sign"] += int((
                torch.sign(candidate_state[:, 3]) == torch.sign(target[:, 3])
            )[sign_mask].sum())
            item["control_sign"] += int((
                torch.sign(control_state[:, 3]) == torch.sign(target[:, 3])
            )[sign_mask].sum())
            item["sign_count"] += int(sign_mask.sum())
            item["count"] += int(mask.sum())
    result = {}
    for name, item in storage.items():
        if item["count"] < 1:
            raise ValueError(f"v10 diagnostic group has no support: {name}")
        result[name] = {
            "sample_count": item["count"],
            "candidate_velocity": _motion_distribution(item["candidate_velocity"]),
            "candidate_yaw": _motion_distribution(item["candidate_yaw"]),
            "control_velocity": _motion_distribution(item["control_velocity"]),
            "control_yaw": _motion_distribution(item["control_yaw"]),
            "broken_pairing_candidate_velocity": _motion_distribution(
                item["broken_velocity"],
            ),
            "broken_pairing_candidate_yaw": _motion_distribution(
                item["broken_yaw"],
            ),
            "zero_rotation_residual_candidate_velocity": _motion_distribution(
                item["zero_residual_velocity"],
            ),
            "candidate_yaw_sign_accuracy": (
                item["candidate_sign"] / item["sign_count"]
                if item["sign_count"] else None
            ),
            "control_yaw_sign_accuracy": (
                item["control_sign"] / item["sign_count"]
                if item["sign_count"] else None
            ),
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
    }


def _validate_completed_residual_probe(
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
        or manifest.get("state_gate_only") is not True
        or manifest.get("state_gate_future_modules_unchanged") is not True
        or manifest.get("progress", {}).get("global_update") != 200
        or not isinstance(fixed, dict)
        or fixed.get("update") != 200
        or fixed.get("selected_by_validation") is not False
        or checkpoint.resolve() != expected_checkpoint.resolve()
        or Path(str(fixed.get("path", ""))).resolve() != checkpoint.resolve()
        or fixed.get("sha256") != sha256_file(checkpoint)
        or not isinstance(contract, dict)
        or manifest.get("contract_sha256") != _json_sha256(contract)
    ):
        raise ValueError("v10 fixed artifact is incomplete or unbound")
    recorded_git = manifest.get("provenance", {}).get("git", {})
    current_git = _git_state()
    if (
        recorded_git.get("worktree_dirty") is not False
        or recorded_git.get("git_commit") in {None, "unknown"}
        or current_git.get("worktree_dirty") is not False
        or current_git.get("git_commit") != recorded_git.get("git_commit")
    ):
        raise ValueError("v10 source checkout changed during the formal run")
    source_manifest = json.loads(
        (Path(args.dataset).resolve() / "dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if source_manifest.get("test_accessed") is not False:
        raise ValueError("v10 source dataset accessed test")
    expected = {
        "state_loss_function": _callable_contract(paired_residual_state_loss),
        "state_step_function": _callable_contract(paired_residual_probe_train_step),
        "final_diagnostic_function": _callable_contract(
            paired_residual_validation_diagnostics
        ),
    }
    for place in ("contract", "provenance"):
        source = manifest.get(place, {})
        if any(source.get(name) != value for name, value in expected.items()):
            raise ValueError("v10 callable contracts differ")
    if (
        manifest.get("state_substage_counts")
        != {"paired_residual_structural_probe": 200}
        or manifest.get("state_substage_transitions") != [{
            "global_update": 1, "substage": "paired_residual_structural_probe",
        }]
    ):
        raise ValueError("v10 update schedule differs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != RUN_SCHEMA
        or payload.get("contract_sha256") != manifest.get("contract_sha256")
        or payload.get("run_id") != manifest.get("run_id")
        or payload.get("model_class") != "AnonymousPairedResidualTwistProbe"
        or payload.get("checkpoint_selected_by_validation") is not False
        or payload.get("fixed_endpoint") is not True
        or payload.get("progress") != {
            "global_update": 200, "stage": "motion_state", "stage_update": 200,
        }
        or state_dict_sha256(payload.get("model", {}))
        != payload.get("model_state_dict_sha256")
        or payload.get("model_config") != manifest.get("model_config")
        or _json_sha256(payload.get("provenance"))
        != _json_sha256(manifest.get("provenance"))
        or _json_sha256(payload.get("validation_history"))
        != _json_sha256(manifest.get("validation_history"))
    ):
        raise ValueError("v10 checkpoint identity differs")
    history = payload.get("validation_history")
    if (
        not isinstance(history, list) or not history
        or _json_sha256(history[-1].get("metrics"))
        != _json_sha256(manifest.get("final_validation"))
    ):
        raise ValueError("v10 final validation is not checkpoint bound")
    for name in (
        "gradient_isolation_verified", "state_substage_counts",
        "state_substage_transitions", "state_branch_hash_history",
        "final_diagnostics",
    ):
        if _json_sha256(payload.get(name)) != _json_sha256(manifest.get(name)):
            raise ValueError(f"v10 checkpoint/manifest {name} differs")
    model_state = payload.get("model")
    actual_hashes = {
        name: state_dict_sha256({
            key[len(name) + 1:]: value
            for key, value in model_state.items() if key.startswith(name + ".")
        })
        for name in ALL_TRAINABLE_MODULES
    }
    if actual_hashes != manifest.get("trainable_final_state_dict_sha256"):
        raise ValueError("v10 final module hashes differ")
    initial_hashes = manifest.get("trainable_initial_state_dict_sha256")
    if any(actual_hashes.get(name) != initial_hashes.get(name)
           for name in FROZEN_FUTURE_MODULES):
        raise ValueError("v10 frozen future hashes differ")
    diagnostics = manifest.get("final_diagnostics")
    expected_groups = {
        "overall", "combined", "combined_speed_gt_1_7", "core",
        "combined_pair1_2",
    }
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
        or set(diagnostics.get("groups", {})) != expected_groups
    ):
        raise ValueError("v10 diagnostics are incomplete or unbound")
    state = manifest["final_validation"]["motion_state"]
    groups = diagnostics["groups"]
    for name in ("overall", "combined", "combined_speed_gt_1_7"):
        if (
            _json_sha256(groups[name]["candidate_velocity"])
            != _json_sha256(state[name]["velocity_vector_error_mps"])
            or _json_sha256(groups[name]["candidate_yaw"])
            != _json_sha256(state[name]["yaw_absolute_error_rad_s"])
        ):
            raise ValueError(f"v10 diagnostic baseline differs: {name}")
    result = {
        "schema_version": "stage3-v10-paired-residual-probe-result-v1",
        "seed": args.seed,
        "fixed_updates": 200,
        "test_accessed": source_manifest["test_accessed"],
        "source_commit": manifest["provenance"]["git"]["git_commit"],
        "contract_sha256": manifest["contract_sha256"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": fixed["sha256"],
        "total_state_parameter_count": parameter_count,
        "diagnostics": diagnostics,
    }
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
        "pair1_2_velocity_mean_mps": (
            "combined_pair1_2", "candidate_velocity",
        ),
        "pair1_2_yaw_mean_rad_s": ("combined_pair1_2", "candidate_yaw"),
        "control_pair1_2_velocity_mean_mps": (
            "combined_pair1_2", "control_velocity",
        ),
        "control_pair1_2_yaw_mean_rad_s": (
            "combined_pair1_2", "control_yaw",
        ),
        "broken_pairing_high_speed_velocity_mean_mps": (
            "combined_speed_gt_1_7", "broken_pairing_candidate_velocity",
        ),
        "broken_pairing_pair1_2_velocity_mean_mps": (
            "combined_pair1_2", "broken_pairing_candidate_velocity",
        ),
        "broken_pairing_pair1_2_yaw_mean_rad_s": (
            "combined_pair1_2", "broken_pairing_candidate_yaw",
        ),
        "zero_residual_combined_velocity_mean_mps": (
            "combined", "zero_rotation_residual_candidate_velocity",
        ),
        "zero_residual_high_speed_velocity_mean_mps": (
            "combined_speed_gt_1_7", "zero_rotation_residual_candidate_velocity",
        ),
    }
    for key, (group, metric) in mapping.items():
        result[key] = _validated_mean(groups[group], metric, name=key)
    sign = state["overall"]["yaw_sign_accuracy_abs_truth_gt_0_5"]
    if (
        isinstance(sign, bool) or not isinstance(sign, (int, float))
        or not math.isfinite(float(sign)) or not 0.0 <= float(sign) <= 1.0
    ):
        raise ValueError("v10 yaw-sign accuracy is invalid")
    result["overall_yaw_sign_accuracy"] = sign
    return result


def main() -> None:
    args = build_residual_probe_parser().parse_args()
    _validate_args(args)
    git = _git_state()
    if git.get("worktree_dirty") is not False or git.get("git_commit") in {
        None, "unknown",
    }:
        raise RuntimeError("v10 structural probe requires a clean source commit")
    v77 = Path(args.v77_control_checkpoint).resolve()
    _preflight_control(v77)
    parameter_count = _state_parameter_count(AnonymousPairedResidualTwistProbe, args)
    if (
        abs(parameter_count - V8_JOINT_REACHABLE_STATE_PARAMETERS)
        / V8_JOINT_REACHABLE_STATE_PARAMETERS > 0.05
    ):
        raise ValueError("v10 reachable capacity differs from V8 by more than 5%")
    checkpoint = train(
        args,
        model_class=AnonymousPairedResidualTwistProbe,
        state_loss_function=paired_residual_state_loss,
        state_step_function=paired_residual_probe_train_step,
        final_diagnostic_function=paired_residual_validation_diagnostics,
        run_schema=RUN_SCHEMA,
        extra_source_paths={
            "v10_runner_and_diagnostics": Path(__file__),
            "v10_model_and_step": Path(__file__).with_name(
                "paired_residual_twist_future.py"
            ),
            "v9_paired_token_context": Path(__file__).with_name(
                "paired_twist_set_future.py"
            ),
            "v8_control_model_and_ramp": Path(__file__).with_name(
                "joint_rigid_flow_probe.py"
            ),
            "common_velocity_ramp": Path(__file__).with_name(
                "factorized_common_relative_motion_future.py"
            ),
        },
        state_gate_only=True,
        frozen_initialization_checkpoint=v77,
        frozen_initialization_modules=FROZEN_FUTURE_MODULES,
    )
    report = _validate_completed_residual_probe(args, checkpoint, parameter_count)
    _atomic_json(Path(args.output).resolve() / "probe_result.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
