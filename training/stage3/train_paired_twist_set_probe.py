"""Run one fixed 200-update V9 paired-twist structural probe."""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path

import numpy as np
import torch

from .joint_rigid_flow_probe import AnonymousJointTwistProbe
from .observable_future_pnp_ab import sha256_file, state_dict_sha256
from .paired_twist_set_future import (
    AnonymousPairedTwistSetProbe,
    paired_twist_probe_train_step,
    paired_twist_state_loss,
)
from .train_anonymous_vehicle_motion import _cuda_amp_dtype, _distribution, _json_sha256
from .train_causal_physical_ab import _git_state, _to_device
from .train_joint_rigid_flow_probe import (
    EXPECTED_PATHS as V8_EXPECTED_PATHS,
    LOCKED_VALUES,
    _state_parameter_count,
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
    build_parser,
    train,
)
from .motion_truth_supervision import MOTION_TARGET_FIELD


RUN_SCHEMA = "stage3-anonymous-paired-twist-set-structural-probe-v9"
PROBE_SEEDS = (20260730, 20260731)
V8_JOINT_CONTROL_CHECKPOINTS = {
    20260730: Path(
        r"D:\仿真\models\engines\stage3-training\20260730-v80-v8-probe-joint-seed20260730-r1\checkpoints\checkpoint-update-000200.pt"
    ),
    20260731: Path(
        r"D:\仿真\models\engines\stage3-training\20260730-v80-v8-probe-joint-seed20260731-r1\checkpoints\checkpoint-update-000200.pt"
    ),
}
V8_JOINT_CONTROL_SHA256 = {
    20260730: "758ac8ddb6dc36fa7a07eacf283c617b591324654fab0446057436380228c87f",
    20260731: "c85bd71b1f9dbd32c4683cf61ec1ecc25006b1501716325acc39aab314b77f8d",
}
# Formal channels=96 V8 has 1,898,569 optimizer-owned state parameters, but
# 410,881 live in legacy context modules that cannot reach the joint loss.
# V9 contains no such dead branch, so capacity is matched to the audited
# gradient-reachable V8 count rather than the misleading optimizer total.
V8_JOINT_REACHABLE_STATE_PARAMETERS = 1_487_688
V8_JOINT_TOTAL_STATE_PARAMETERS = 1_898_569


def build_probe_parser():
    parser = build_parser()
    parser.description = "bounded 200-update V9 paired-twist structural probe"
    parser.add_argument("--v77-control-checkpoint", required=True)
    parser.add_argument("--v8-joint-control-checkpoint", required=True)
    parser.set_defaults(
        motion_state_updates=200,
        trajectory_updates=0,
        selector_updates=0,
        decoder_joint_updates=0,
        stop_after_update=0,
    )
    return parser


def _validate_args(args) -> None:
    if (
        args.motion_state_updates != 200
        or any(value != 0 for value in (
            args.trajectory_updates, args.selector_updates, args.decoder_joint_updates,
        ))
        or args.stop_after_update != 0
    ):
        raise ValueError("v9 structural probe is fixed to 200 state-only updates")
    if args.seed not in PROBE_SEEDS:
        raise ValueError(f"v9 probe seed must be one of {PROBE_SEEDS}")
    if args.diagnostic_oracle_association is not True:
        raise ValueError("v9 probe requires diagnostic association")
    if args.allow_mapper_h_mismatch is not True:
        raise ValueError("v9 probe requires the qualified mapper/H mismatch")
    mismatched_values = {
        name: {"expected": expected, "actual": getattr(args, name)}
        for name, expected in LOCKED_VALUES.items()
        if getattr(args, name) != expected
    }
    if mismatched_values:
        raise ValueError(f"v9 probe scalar contract differs: {mismatched_values}")
    expected_paths = dict(V8_EXPECTED_PATHS)
    expected_paths.pop("v77_control_checkpoint")
    expected_paths["v77_control_checkpoint"] = V8_EXPECTED_PATHS[
        "v77_control_checkpoint"
    ]
    expected_paths["v8_joint_control_checkpoint"] = V8_JOINT_CONTROL_CHECKPOINTS[
        args.seed
    ]
    mismatched_paths = {
        name: {"expected": str(expected.resolve()), "actual": str(Path(
            getattr(args, name)
        ).resolve())}
        for name, expected in expected_paths.items()
        if Path(getattr(args, name)).resolve() != expected.resolve()
    }
    if mismatched_paths:
        raise ValueError(f"v9 probe artifact contract differs: {mismatched_paths}")
    control = Path(args.v8_joint_control_checkpoint).resolve()
    if sha256_file(control) != V8_JOINT_CONTROL_SHA256[args.seed]:
        raise ValueError("v9 V8-joint control checkpoint hash differs")


def _motion_distribution(values: list[np.ndarray]) -> dict[str, float | int]:
    return _distribution(
        np.concatenate(values).astype(np.float64, copy=False)
        if values else np.empty(0, dtype=np.float64)
    )


def _load_v8_control(seed: int, candidate) -> AnonymousJointTwistProbe:
    path = V8_JOINT_CONTROL_CHECKPOINTS[seed].resolve()
    if sha256_file(path) != V8_JOINT_CONTROL_SHA256[seed]:
        raise ValueError("v9 diagnostic V8 control hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version")
        != "stage3-anonymous-rigid-flow-structural-probe-v8"
        or payload.get("model_class") != "AnonymousJointTwistProbe"
        or payload.get("progress") != {
            "global_update": 200, "stage": "motion_state", "stage_update": 200,
        }
        or payload.get("fixed_endpoint") is not True
        or payload.get("checkpoint_selected_by_validation") is not False
    ):
        raise ValueError("v9 diagnostic V8 control identity differs")
    control = AnonymousJointTwistProbe(
        velocity_scale_mps=tuple(float(value) for value in candidate.motion_state_scale[:3]),
        yaw_rate_scale_rad_s=float(candidate.motion_state_scale[3]),
        channels=candidate.channels,
        dropout=candidate.dropout,
        message_layers=candidate.message_layers,
        basis_count=candidate.basis_count,
    )
    control.load_state_dict(payload["model"], strict=True)
    if state_dict_sha256(control.state_dict()) != payload.get("model_state_dict_sha256"):
        raise ValueError("v9 diagnostic V8 control model hash differs")
    return control


@torch.inference_mode()
def paired_twist_validation_diagnostics(
    model,
    loader,
    mapper,
    s_model,
    h_model,
    device: torch.device,
) -> dict:
    """Checkpoint-bound V8 control and broken-pairing validation comparison."""
    seed = int(torch.initial_seed())
    if seed not in PROBE_SEEDS:
        raise ValueError("v9 diagnostic seed differs")
    control = _load_v8_control(seed, model).to(device).eval().requires_grad_(False)
    model.eval()
    group_names = ("overall", "combined", "combined_speed_gt_1_7", "core")
    storage = {
        group: {
            "candidate_velocity": [], "candidate_yaw": [],
            "control_velocity": [], "control_yaw": [],
            "broken_velocity": [], "candidate_sign": 0, "control_sign": 0,
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
            fields = {
                name: batch[name] for name in model._field_names()
            }
            candidate = model.estimate_motion_state(**fields)
            broken = model.estimate_motion_state_broken_pairing(**fields)
            baseline = control.estimate_motion_state(**fields)
        target = batch[MOTION_TARGET_FIELD]
        candidate_state = candidate["state"]["motion_state_physical"]
        broken_state = broken["state"]["motion_state_physical"]
        control_state = baseline["state"]["motion_state_physical"]
        candidate_velocity = torch.linalg.vector_norm(
            candidate_state[:, :3] - target[:, :3], dim=-1,
        )
        broken_velocity = torch.linalg.vector_norm(
            broken_state[:, :3] - target[:, :3], dim=-1,
        )
        control_velocity = torch.linalg.vector_norm(
            control_state[:, :3] - target[:, :3], dim=-1,
        )
        candidate_yaw = (candidate_state[:, 3] - target[:, 3]).abs()
        control_yaw = (control_state[:, 3] - target[:, 3]).abs()
        speed = torch.linalg.vector_norm(target[:, :2], dim=-1)
        combined = batch["motion_class"] == 3
        pair3 = candidate["history"]["pair_flow_available"].all(dim=1)
        history32 = candidate["history"]["history_active_count"] == 32
        masks = {
            "overall": torch.ones_like(combined),
            "combined": combined,
            "combined_speed_gt_1_7": combined & (speed > 1.7),
            "core": combined & (speed <= 1.2) & history32 & pair3,
        }
        yaw_valid = target[:, 3].abs() > 0.5
        for name, mask in masks.items():
            if not bool(mask.any()):
                continue
            item = storage[name]
            item["candidate_velocity"].append(
                candidate_velocity[mask].float().cpu().numpy()
            )
            item["candidate_yaw"].append(candidate_yaw[mask].float().cpu().numpy())
            item["control_velocity"].append(
                control_velocity[mask].float().cpu().numpy()
            )
            item["control_yaw"].append(control_yaw[mask].float().cpu().numpy())
            item["broken_velocity"].append(
                broken_velocity[mask].float().cpu().numpy()
            )
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
            raise ValueError(f"v9 diagnostic group has no support: {name}")
        result[name] = {
            "sample_count": item["count"],
            "candidate_velocity": _motion_distribution(item["candidate_velocity"]),
            "candidate_yaw": _motion_distribution(item["candidate_yaw"]),
            "control_velocity": _motion_distribution(item["control_velocity"]),
            "control_yaw": _motion_distribution(item["control_yaw"]),
            "broken_pairing_candidate_velocity": _motion_distribution(
                item["broken_velocity"],
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
        "schema_version": "stage3-v9-paired-twist-validation-diagnostics-v1",
        "validation_only": True,
        "test_accessed": False,
        "seed": seed,
        "v8_joint_control_checkpoint": str(
            V8_JOINT_CONTROL_CHECKPOINTS[seed].resolve()
        ),
        "v8_joint_control_checkpoint_sha256": V8_JOINT_CONTROL_SHA256[seed],
        "groups": result,
    }


def _validated_mean(group: dict, metric: str, *, name: str) -> float:
    value = group.get(metric, {}).get("mean_m")
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value)) or float(value) < 0.0
    ):
        raise ValueError(f"v9 probe {name} is invalid")
    return float(value)


def _validate_completed_probe(args, checkpoint: Path, parameter_count: int) -> dict:
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
        raise ValueError("v9 probe fixed artifact is incomplete or unbound")
    recorded_git = manifest.get("provenance", {}).get("git", {})
    current_git = _git_state()
    if (
        recorded_git.get("worktree_dirty") is not False
        or recorded_git.get("git_commit") in {None, "unknown"}
        or current_git.get("worktree_dirty") is not False
        or current_git.get("git_commit") != recorded_git.get("git_commit")
    ):
        raise ValueError("v9 source checkout changed during the formal run")
    source_manifest = json.loads(
        (Path(args.dataset).resolve() / "dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if source_manifest.get("test_accessed") is not False:
        raise ValueError("v9 source dataset accessed test")
    expected_loss = _callable_contract(paired_twist_state_loss)
    expected_step = _callable_contract(paired_twist_probe_train_step)
    expected_diagnostic = _callable_contract(paired_twist_validation_diagnostics)
    for place in ("contract", "provenance"):
        source = manifest.get(place, {})
        if (
            source.get("state_loss_function") != expected_loss
            or source.get("state_step_function") != expected_step
            or source.get("final_diagnostic_function") != expected_diagnostic
        ):
            raise ValueError("v9 probe callable contracts differ")
    if (
        manifest.get("state_substage_counts")
        != {"paired_twist_structural_probe": 200}
        or manifest.get("state_substage_transitions") != [{
            "global_update": 1, "substage": "paired_twist_structural_probe",
        }]
    ):
        raise ValueError("v9 probe update schedule differs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != RUN_SCHEMA
        or payload.get("contract_sha256") != manifest.get("contract_sha256")
        or payload.get("run_id") != manifest.get("run_id")
        or payload.get("model_class") != "AnonymousPairedTwistSetProbe"
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
        raise ValueError("v9 probe checkpoint identity differs")
    history = payload.get("validation_history")
    if (
        not isinstance(history, list) or not history
        or _json_sha256(history[-1].get("metrics"))
        != _json_sha256(manifest.get("final_validation"))
    ):
        raise ValueError("v9 final validation is not checkpoint bound")
    for name in (
        "gradient_isolation_verified", "state_substage_counts",
        "state_substage_transitions", "state_branch_hash_history",
        "final_diagnostics",
    ):
        if _json_sha256(payload.get(name)) != _json_sha256(manifest.get(name)):
            raise ValueError(f"v9 checkpoint/manifest {name} differs")
    model_state = payload.get("model")
    actual_hashes = {
        name: state_dict_sha256({
            key[len(name) + 1:]: value
            for key, value in model_state.items() if key.startswith(name + ".")
        })
        for name in ALL_TRAINABLE_MODULES
    }
    if actual_hashes != manifest.get("trainable_final_state_dict_sha256"):
        raise ValueError("v9 final module hashes differ")
    initial_hashes = manifest.get("trainable_initial_state_dict_sha256")
    if any(actual_hashes.get(name) != initial_hashes.get(name)
           for name in FROZEN_FUTURE_MODULES):
        raise ValueError("v9 frozen future hashes differ")
    diagnostics = manifest.get("final_diagnostics")
    if (
        not isinstance(diagnostics, dict)
        or diagnostics.get("schema_version")
        != "stage3-v9-paired-twist-validation-diagnostics-v1"
        or diagnostics.get("validation_only") is not True
        or diagnostics.get("test_accessed") is not False
        or diagnostics.get("seed") != args.seed
        or Path(diagnostics.get("v8_joint_control_checkpoint", "")).resolve()
        != V8_JOINT_CONTROL_CHECKPOINTS[args.seed].resolve()
        or diagnostics.get("v8_joint_control_checkpoint_sha256")
        != V8_JOINT_CONTROL_SHA256[args.seed]
        or set(diagnostics.get("groups", {}))
        != {"overall", "combined", "combined_speed_gt_1_7", "core"}
    ):
        raise ValueError("v9 validation diagnostics are incomplete or unbound")
    state = manifest["final_validation"]["motion_state"]
    diagnostic_groups = diagnostics["groups"]
    for name in ("overall", "combined", "combined_speed_gt_1_7"):
        if (
            _json_sha256(diagnostic_groups[name]["candidate_velocity"])
            != _json_sha256(state[name]["velocity_vector_error_mps"])
            or _json_sha256(diagnostic_groups[name]["candidate_yaw"])
            != _json_sha256(state[name]["yaw_absolute_error_rad_s"])
        ):
            raise ValueError(f"v9 diagnostic baseline differs: {name}")
    result = {
        "schema_version": "stage3-v9-paired-twist-probe-result-v1",
        "seed": args.seed,
        "fixed_updates": 200,
        "test_accessed": source_manifest["test_accessed"],
        "source_commit": manifest["provenance"]["git"]["git_commit"],
        "contract_sha256": manifest["contract_sha256"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": fixed["sha256"],
        "total_state_parameter_count": parameter_count,
        "overall_velocity_mean_mps": _validated_mean(
            state["overall"], "velocity_vector_error_mps", name="overall velocity",
        ),
        "overall_yaw_mean_rad_s": _validated_mean(
            state["overall"], "yaw_absolute_error_rad_s", name="overall yaw",
        ),
        "overall_yaw_sign_accuracy": state["overall"][
            "yaw_sign_accuracy_abs_truth_gt_0_5"
        ],
        "combined_velocity_mean_mps": _validated_mean(
            state["combined"], "velocity_vector_error_mps", name="combined velocity",
        ),
        "combined_yaw_mean_rad_s": _validated_mean(
            state["combined"], "yaw_absolute_error_rad_s", name="combined yaw",
        ),
        "high_speed_combined_velocity_mean_mps": _validated_mean(
            state["combined_speed_gt_1_7"], "velocity_vector_error_mps",
            name="high-speed combined velocity",
        ),
        "high_speed_combined_yaw_mean_rad_s": _validated_mean(
            state["combined_speed_gt_1_7"], "yaw_absolute_error_rad_s",
            name="high-speed combined yaw",
        ),
        "core_yaw_mean_rad_s": _validated_mean(
            diagnostic_groups["core"], "candidate_yaw", name="core yaw",
        ),
        "control_core_yaw_mean_rad_s": _validated_mean(
            diagnostic_groups["core"], "control_yaw", name="control core yaw",
        ),
        "broken_pairing_high_speed_velocity_mean_mps": _validated_mean(
            diagnostic_groups["combined_speed_gt_1_7"],
            "broken_pairing_candidate_velocity", name="broken pairing high speed",
        ),
        "diagnostics": diagnostics,
    }
    sign = result["overall_yaw_sign_accuracy"]
    if not isinstance(sign, (int, float)) or not 0.0 <= float(sign) <= 1.0:
        raise ValueError("v9 yaw-sign accuracy is invalid")
    return result


def main() -> None:
    args = build_probe_parser().parse_args()
    _validate_args(args)
    git = _git_state()
    if git.get("worktree_dirty") is not False or git.get("git_commit") in {None, "unknown"}:
        raise RuntimeError("v9 structural probe requires a clean source commit")
    v77 = Path(args.v77_control_checkpoint).resolve()
    _preflight_control(v77)
    parameter_count = _state_parameter_count(AnonymousPairedTwistSetProbe, args)
    v8_count = _state_parameter_count(AnonymousJointTwistProbe, args)
    if v8_count != V8_JOINT_TOTAL_STATE_PARAMETERS:
        raise ValueError("v9 V8 control architecture capacity differs")
    if (
        abs(parameter_count - V8_JOINT_REACHABLE_STATE_PARAMETERS)
        / V8_JOINT_REACHABLE_STATE_PARAMETERS > 0.05
    ):
        raise ValueError("v9 reachable state capacity differs from V8 by more than 5%")
    checkpoint = train(
        args,
        model_class=AnonymousPairedTwistSetProbe,
        state_loss_function=paired_twist_state_loss,
        state_step_function=paired_twist_probe_train_step,
        final_diagnostic_function=paired_twist_validation_diagnostics,
        run_schema=RUN_SCHEMA,
        extra_source_paths={
            "v9_runner_and_diagnostics": Path(__file__),
            "v9_model_and_step": Path(__file__).with_name(
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
    report = _validate_completed_probe(args, checkpoint, parameter_count)
    _atomic_json(Path(args.output).resolve() / "probe_result.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
