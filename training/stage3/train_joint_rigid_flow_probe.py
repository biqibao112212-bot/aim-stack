"""Run one fixed 200-update V8 rigid-flow structural probe."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .factorized_common_relative_motion_future import (
    FactorizedCommonRelativeMotionStateV7,
)
from .joint_rigid_flow_probe import (
    AnonymousJointTwistProbe,
    AnonymousSeparatedRigidFlowProbe,
    rigid_flow_probe_train_step,
)
from .robust_multiscale_motion_future import robust_multiscale_motion_state_loss
from .observable_future_pnp_ab import sha256_file, state_dict_sha256
from .train_anonymous_vehicle_motion import _json_sha256
from .train_causal_physical_ab import _git_state
from .train_robust_multiscale_motion_future import (
    FROZEN_FUTURE_MODULES,
    _preflight_control,
)
from .train_stable_motion_bottleneck_future import (
    ALL_TRAINABLE_MODULES,
    STATE_MODULES,
    _callable_contract,
    build_parser,
    train,
)
from .train_pnp_window_mapper_distillation import _atomic_json


RUN_SCHEMA = "stage3-anonymous-rigid-flow-structural-probe-v8"
PROBE_SEEDS = (20260730, 20260731)
EXPECTED_PATHS = {
    "dataset": Path(
        r"D:\仿真\dataset\autoaim-stage3-v1\derived\stage3-multistate-fixed6mm-v2-pnp-sf-20260730-r2"
    ),
    "truth_history": Path(
        r"D:\仿真\dataset\autoaim-stage3-v1\derived\stage3-multistate-fixed6mm-v2-truth-history-20260730-r2"
    ),
    "mapper_checkpoint": Path(
        r"D:\仿真\models\engines\stage3-training\20260727-v41-a4-aligned-anchored-window-capacity-r1\epoch-0002-update-000264.pt"
    ),
    "s_checkpoint": Path(
        r"D:\仿真\models\engines\stage3-training\20260724-v19-anchor-edge-state-restorer-120ep-seed0-r2\stage3-cyclic-anchor-edge-restorer-seed0-epoch110.pt"
    ),
    "h_checkpoint": Path(
        r"D:\仿真\models\engines\stage3-training\20260727-v35-a3-h0-warm-only-full-r1\diagnostic-sealed-epoch-0032-update-004224.pt"
    ),
    "v77_control_checkpoint": Path(
        r"D:\仿真\models\engines\stage3-training\20260730-v77-v5-multistate-latest32-full-r1\checkpoints\checkpoint-update-000800.pt"
    ),
}
LOCKED_VALUES = {
    "device": "cuda:0",
    "batch_size": 64,
    "validation_batch_size": 96,
    "channels": 96,
    "dropout": 0.05,
    "message_layers": 3,
    "basis_count": 8,
    "motion_state_learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "gradient_clip_norm": 1.0,
    "prefix_dropout_probability": 0.75,
    "train_limit_per_class": 0,
    "validation_limit_per_class": 0,
    "log_interval": 25,
}


def build_probe_parser():
    parser = build_parser()
    parser.description = "bounded 200-update post-v7 rigid-flow structural probe"
    parser.add_argument(
        "--variant", choices=("v7_factorized", "separated", "joint"), required=True,
    )
    parser.add_argument("--v77-control-checkpoint", required=True)
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
        raise ValueError("v8 structural probe is fixed to 200 state-only updates")
    if args.seed not in PROBE_SEEDS:
        raise ValueError(f"v8 structural probe seed must be one of {PROBE_SEEDS}")
    if args.diagnostic_oracle_association is not True:
        raise ValueError("v8 structural probe requires explicit diagnostic association")
    if args.allow_mapper_h_mismatch is not True:
        raise ValueError("v8 structural probe requires the qualified mapper/H mismatch")
    mismatched_values = {
        name: {"expected": expected, "actual": getattr(args, name)}
        for name, expected in LOCKED_VALUES.items()
        if getattr(args, name) != expected
    }
    if mismatched_values:
        raise ValueError(f"v8 structural probe scalar contract differs: {mismatched_values}")
    mismatched_paths = {
        name: {"expected": str(expected.resolve()), "actual": str(Path(getattr(args, name)).resolve())}
        for name, expected in EXPECTED_PATHS.items()
        if Path(getattr(args, name)).resolve() != expected.resolve()
    }
    if mismatched_paths:
        raise ValueError(f"v8 structural probe artifact contract differs: {mismatched_paths}")


def _state_parameter_count(model_class, args) -> int:
    model = model_class(
        velocity_scale_mps=(3.0, 3.2, 0.25),
        yaw_rate_scale_rad_s=17.25,
        channels=args.channels,
        dropout=args.dropout,
        message_layers=args.message_layers,
        basis_count=args.basis_count,
    )
    return sum(
        parameter.numel()
        for name in STATE_MODULES
        for parameter in getattr(model, name).parameters()
    )


def _validated_mean(group: dict, metric: str, *, name: str) -> float:
    value = group.get(metric, {}).get("mean_m")
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value)) or float(value) < 0.0
    ):
        raise ValueError(f"v8 probe {name} is not finite and nonnegative")
    return float(value)


def _validate_completed_probe(args, model_class, checkpoint: Path, parameter_count: int) -> dict:
    output = Path(args.output).resolve()
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    fixed = manifest.get("fixed_final_checkpoint")
    contract = manifest.get("contract")
    contract_sha = manifest.get("contract_sha256")
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
        or Path(str(fixed.get("path", ""))).resolve() != checkpoint.resolve()
        or checkpoint.resolve() != expected_checkpoint.resolve()
        or fixed.get("sha256") != sha256_file(checkpoint)
        or not isinstance(contract, dict)
        or contract_sha != _json_sha256(contract)
    ):
        raise ValueError("v8 probe fixed artifact is incomplete or unbound")
    expected_loss = _callable_contract(robust_multiscale_motion_state_loss)
    expected_step = _callable_contract(rigid_flow_probe_train_step)
    ramp_source = Path(__file__).with_name(
        "factorized_common_relative_motion_future.py"
    ).resolve()
    probe_source = Path(__file__).with_name("joint_rigid_flow_probe.py").resolve()
    if (
        manifest.get("contract", {}).get("state_loss_function") != expected_loss
        or manifest.get("provenance", {}).get("state_loss_function") != expected_loss
        or manifest.get("contract", {}).get("state_step_function") != expected_step
        or manifest.get("provenance", {}).get("state_step_function") != expected_step
        or Path(manifest.get("provenance", {}).get("source_paths", {}).get(
            "common_velocity_ramp", ""
        )).resolve() != ramp_source
        or manifest.get("provenance", {}).get("source_sha256", {}).get(
            "common_velocity_ramp"
        ) != sha256_file(ramp_source)
        or Path(manifest.get("provenance", {}).get("source_paths", {}).get(
            "probe_model_and_step", ""
        )).resolve() != probe_source
        or manifest.get("provenance", {}).get("source_sha256", {}).get(
            "probe_model_and_step"
        ) != sha256_file(probe_source)
    ):
        raise ValueError("v8 probe state loss/step dependencies are not source bound")
    if (
        manifest.get("state_substage_counts") != {"joint_structural_probe": 200}
        or manifest.get("state_substage_transitions") != [{
            "global_update": 1, "substage": "joint_structural_probe",
        }]
    ):
        raise ValueError("v8 probe actual update schedule differs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != RUN_SCHEMA
        or payload.get("contract_sha256") != contract_sha
        or payload.get("run_id") != manifest.get("run_id")
        or payload.get("model_class") != model_class.__name__
        or payload.get("checkpoint_selected_by_validation") is not False
        or payload.get("fixed_endpoint") is not True
        or payload.get("progress") != {
            "global_update": 200, "stage": "motion_state", "stage_update": 200,
        }
    ):
        raise ValueError("v8 probe checkpoint identity differs")
    if (
        payload.get("model_config") != manifest.get("model_config")
        or state_dict_sha256(payload.get("model", {}))
        != payload.get("model_state_dict_sha256")
        or _json_sha256(payload.get("provenance"))
        != _json_sha256(manifest.get("provenance"))
        or _json_sha256(payload.get("validation_history"))
        != _json_sha256(manifest.get("validation_history"))
    ):
        raise ValueError("v8 probe checkpoint/manifest model or provenance differs")
    history = payload.get("validation_history")
    if (
        not isinstance(history, list) or not history
        or _json_sha256(history[-1].get("metrics"))
        != _json_sha256(manifest.get("final_validation"))
    ):
        raise ValueError("v8 probe final validation is not checkpoint bound")
    for name in (
        "gradient_isolation_verified", "state_substage_counts",
        "state_substage_transitions", "state_branch_hash_history",
        "final_diagnostics",
    ):
        if _json_sha256(payload.get(name)) != _json_sha256(manifest.get(name)):
            raise ValueError(f"v8 probe checkpoint/manifest {name} differs")
    model_state = payload.get("model")
    if not isinstance(model_state, dict):
        raise ValueError("v8 probe checkpoint model state is missing")
    actual_module_hashes = {
        name: state_dict_sha256({
            key[len(name) + 1:]: value
            for key, value in model_state.items() if key.startswith(name + ".")
        })
        for name in ALL_TRAINABLE_MODULES
    }
    if actual_module_hashes != manifest.get("trainable_final_state_dict_sha256"):
        raise ValueError("v8 probe final module hashes differ")
    trainable_initial = manifest.get("trainable_initial_state_dict_sha256")
    if (
        not isinstance(trainable_initial, dict)
        or any(actual_module_hashes.get(name) != trainable_initial.get(name)
               for name in FROZEN_FUTURE_MODULES)
    ):
        raise ValueError("v8 probe frozen future hashes are not checkpoint bound")
    if manifest.get("frozen_final_state_dict_sha256") != manifest.get(
        "provenance", {}
    ).get("frozen_initial_state_dict_sha256"):
        raise ValueError("v8 probe frozen Mapper/S/H hashes differ")
    source_manifest = json.loads(
        (Path(args.dataset).resolve() / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("test_accessed") is not False:
        raise ValueError("v8 probe source dataset accessed test")
    state = manifest["final_validation"]["motion_state"]
    result = {
        "schema_version": "stage3-v8-rigid-flow-probe-result-v1",
        "variant": args.variant,
        "model_class": model_class.__name__,
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
    }
    sign = result["overall_yaw_sign_accuracy"]
    if (
        isinstance(sign, bool) or not isinstance(sign, (int, float))
        or not math.isfinite(float(sign)) or not 0.0 <= float(sign) <= 1.0
    ):
        raise ValueError("v8 probe yaw-sign accuracy is invalid")
    return result


def main() -> None:
    args = build_probe_parser().parse_args()
    _validate_args(args)
    git = _git_state()
    if git.get("worktree_dirty") is not False or git.get("git_commit") in {None, "unknown"}:
        raise RuntimeError("v8 structural probe requires a clean source commit")
    control = Path(args.v77_control_checkpoint).resolve()
    _preflight_control(control)
    model_class = {
        "v7_factorized": FactorizedCommonRelativeMotionStateV7,
        "separated": AnonymousSeparatedRigidFlowProbe,
        "joint": AnonymousJointTwistProbe,
    }[args.variant]
    parameter_count = _state_parameter_count(model_class, args)
    checkpoint = train(
        args,
        model_class=model_class,
        state_loss_function=robust_multiscale_motion_state_loss,
        state_step_function=rigid_flow_probe_train_step,
        run_schema=RUN_SCHEMA,
        extra_source_paths={
            "probe_runner": Path(__file__),
            "probe_model_and_step": Path(__file__).with_name(
                "joint_rigid_flow_probe.py"
            ),
            # rigid_flow_probe_train_step imports this augmentation helper; a
            # callable source hash is not recursive, so bind its owner too.
            "common_velocity_ramp": Path(__file__).with_name(
                "factorized_common_relative_motion_future.py"
            ),
        },
        state_gate_only=True,
        frozen_initialization_checkpoint=control,
        frozen_initialization_modules=FROZEN_FUTURE_MODULES,
    )
    output = Path(args.output).resolve()
    report = _validate_completed_probe(args, model_class, checkpoint, parameter_count)
    _atomic_json(output / "probe_result.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
