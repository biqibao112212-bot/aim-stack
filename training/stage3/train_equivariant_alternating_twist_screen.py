"""Run one fixed 100-update V13 typed-alternating structural screen."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .equivariant_alternating_twist_future import (
    AnonymousEquivariantAlternatingTwistProbe,
    equivariant_alternating_train_step,
)
from .observable_future_pnp_ab import sha256_file, state_dict_sha256
from .motion_truth_supervision import MOTION_TARGET_FIELD, MotionTruthIndex
from .paired_twist_set_future import LOCAL_LAG_SCALES_S
from .train_causal_physical_ab import _git_state
from .train_joint_rigid_flow_probe import (
    EXPECTED_PATHS as V8_EXPECTED_PATHS,
    LOCKED_VALUES,
    _state_parameter_count,
)
from .train_omega_first_ordered_closure_probe import (
    DIAGNOSTIC_FIELDS,
    DIAGNOSTIC_SCHEMA,
    GROUP_NAMES,
    WRITE_ISOLATION_FIELDS,
    omega_first_ordered_validation_diagnostics,
)
from .finalize_omega_first_ordered_closure_probe import (
    _assert_finite_tree,
    _validate_diagnostic_binding,
)
from .train_paired_twist_set_probe import (
    PROBE_SEEDS,
    V8_JOINT_CONTROL_CHECKPOINTS,
    V8_JOINT_CONTROL_SHA256,
    V8_JOINT_REACHABLE_STATE_PARAMETERS,
    build_probe_parser,
)
from .train_pnp_window_mapper_distillation import _atomic_json
from .train_robust_multiscale_motion_future import (
    FROZEN_FUTURE_MODULES,
    _preflight_control,
)
from .train_increment_invariant_anonymous_future import _history_label
from .train_anonymous_vehicle_motion import _json_sha256
from .train_stable_motion_bottleneck_future import (
    ALL_TRAINABLE_MODULES,
    STATE_MODULES,
    _callable_contract,
    _dataset,
    train,
)


RUN_SCHEMA = "stage3-equivariant-alternating-twist-screen-v13"
RESULT_SCHEMA = "stage3-equivariant-alternating-twist-screen-result-v1"
V12_RESULT_ROOTS = {
    20260730: Path(
        r"D:\仿真\models\engines\stage3-training\20260731-v85-v12-omega-first-ordered-seed20260730-r2"
    ),
    20260731: Path(
        r"D:\仿真\models\engines\stage3-training\20260731-v85-v12-omega-first-ordered-seed20260731-r2"
    ),
}
V12_RESULT_SHA256 = {
    20260730: "dd3a46c4af3836ce14165cde708f3389939b37427a21e2d209ea27d7741b4ba5",
    20260731: "b98b8e092d1e33cf6831daea20270cdc0854551d2013aae3b8227e818c4c0676",
}
V12_CHECKPOINT_SHA256 = {
    20260730: "e5e5556bc7c600ea85c7d15c1b29bcbcc3cc1340536a74a68a13ebcd90226bb6",
    20260731: "638b76b3493b64c4088fe65e1c3ea03bdc6ab3526ac3d382268d7dcf0052c466",
}
EXPECTED_SUBSTAGE_COUNTS = {
    "typed_alternating_omega0": 35,
    "typed_alternating_velocity0": 20,
    "typed_alternating_omega1": 25,
    "typed_alternating_velocity1": 20,
}
EXPECTED_SUBSTAGE_TRANSITIONS = [
    {"global_update": 1, "substage": "typed_alternating_omega0"},
    {"global_update": 36, "substage": "typed_alternating_velocity0"},
    {"global_update": 56, "substage": "typed_alternating_omega1"},
    {"global_update": 81, "substage": "typed_alternating_velocity1"},
]
YAW_ALIAS_LIMIT_RAD_S = (
    AnonymousEquivariantAlternatingTwistProbe.max_abs_yaw_rate_rad_s
)


def _is_lower_sha256(value) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def alternating_motion_state_cells(
    dataset,
) -> dict[tuple[int, str, str], list[int]]:
    """Balance exact observable pair-scale support as well as motion/history."""
    result: dict[tuple[int, str, str], list[int]] = {}
    scales = torch.tensor(LOCAL_LAG_SCALES_S, dtype=torch.float64)
    middle = torch.sqrt(scales[:-1] * scales[1:])
    lower = torch.cat((scales[:1] * 0.5, middle))
    upper = torch.cat((middle, scales[-1:] * 1.5))
    for part_index, part in enumerate(dataset.parts):
        active = part.tensors["pnp_s_event_mask"].to(torch.bool)
        visible = part.tensors["pnp_s_obs_mask"].to(torch.bool) & active.unsqueeze(-1)
        time_s = part.tensors["pnp_s_event_time_s"].to(torch.float64)
        pair_event = visible.sum(dim=2) == 2
        elapsed = time_s[:, :, None] - time_s[:, None, :]
        events = time_s.shape[1]
        current_index = torch.arange(events)[None, :, None]
        prior_index = torch.arange(events)[None, None, :]
        same_set = (visible[:, :, None] == visible[:, None, :]).all(dim=-1)
        causal = (
            pair_event[:, :, None] & pair_event[:, None, :] & same_set
            & (prior_index < current_index) & (elapsed > 1e-7)
        )
        scale_supported = (
            causal.unsqueeze(-1)
            & (elapsed.unsqueeze(-1) >= lower)
            & (elapsed.unsqueeze(-1) < upper)
        ).any(dim=(1, 2))
        pair_support = scale_supported.sum(dim=1).to(torch.long)
        active_count = active.sum(dim=1)
        moving = part.tensors[MOTION_TARGET_FIELD].abs().amax(dim=1) > 1e-5
        offset = dataset.offsets[part_index]
        for local, (session, count, support, is_moving) in enumerate(zip(
            part.session_ids, active_count.tolist(), pair_support.tolist(),
            moving.tolist(), strict=True,
        )):
            state = "active" if bool(is_moving) else "stationary"
            cell = f"{_history_label(int(count))}/{state}/pair{int(support)}"
            key = (int(part.motion_class), str(session), cell)
            result.setdefault(key, []).append(offset + local)
    if not result or any(not values for values in result.values()):
        raise ValueError("alternating sampler has an empty support cell")
    return result


def alternating_dataset_preflight(train_dataset, validation_dataset) -> dict:
    """Bind the physical yaw envelope required by signed atan2 factors."""
    split_report = {}
    for name, dataset in (
        ("train", train_dataset), ("validation", validation_dataset),
    ):
        values = torch.cat([
            part.tensors[MOTION_TARGET_FIELD][:, 3].to(torch.float64)
            for part in dataset.parts
        ])
        if values.numel() < 1 or not bool(torch.isfinite(values).all()):
            raise ValueError(f"alternating {name} yaw targets are invalid")
        maximum = float(values.abs().max())
        if maximum > YAW_ALIAS_LIMIT_RAD_S:
            raise ValueError(
                f"alternating {name} yaw exceeds alias envelope: {maximum}"
            )
        split_report[name] = {
            "sample_count": int(values.numel()),
            "max_abs_yaw_rate_rad_s": maximum,
        }
    return {
        "schema_version": "stage3-v13-yaw-alias-preflight-v1",
        "max_abs_yaw_rate_limit_rad_s": YAW_ALIAS_LIMIT_RAD_S,
        "max_factor_elapsed_s": 0.105,
        "phase_upper_bound_rad": YAW_ALIAS_LIMIT_RAD_S * 0.105,
        "splits": split_report,
    }


def _validate_yaw_preflight_report(
    report: object, *, recomputed: dict | None = None,
) -> None:
    """Reject structurally valid but source-unbound alias-envelope evidence."""
    if not isinstance(report, dict):
        raise ValueError("alternating yaw alias preflight differs")
    phase = report.get("phase_upper_bound_rad")
    limit = report.get("max_abs_yaw_rate_limit_rad_s")
    elapsed = report.get("max_factor_elapsed_s")
    if (
        report.get("schema_version")
        != "stage3-v13-yaw-alias-preflight-v1"
        or limit != YAW_ALIAS_LIMIT_RAD_S
        or elapsed != 0.105
        or isinstance(phase, bool)
        or not isinstance(phase, (int, float))
        or not math.isfinite(float(phase))
        or float(phase) != float(limit) * float(elapsed)
        or float(phase) >= math.pi
        or set(report.get("splits", {})) != {"train", "validation"}
    ):
        raise ValueError("alternating yaw alias preflight differs")
    for name, split in report["splits"].items():
        if (
            not isinstance(split, dict)
            or set(split) != {"sample_count", "max_abs_yaw_rate_rad_s"}
            or type(split.get("sample_count")) is not int
            or split["sample_count"] < 1
            or isinstance(split.get("max_abs_yaw_rate_rad_s"), bool)
            or not isinstance(split.get("max_abs_yaw_rate_rad_s"), (int, float))
            or not math.isfinite(float(split["max_abs_yaw_rate_rad_s"]))
            or not 0.0 <= split["max_abs_yaw_rate_rad_s"] <= YAW_ALIAS_LIMIT_RAD_S
        ):
            raise ValueError(f"alternating {name} yaw alias report differs")
    if recomputed is not None and report != recomputed:
        raise ValueError("alternating yaw alias preflight is not bound to truth")


def build_alternating_screen_parser():
    parser = build_probe_parser()
    parser.description = "fixed 100-update V13 typed-alternating screen"
    parser.set_defaults(
        motion_state_updates=100,
        trajectory_updates=0,
        selector_updates=0,
        decoder_joint_updates=0,
        stop_after_update=0,
    )
    return parser


def _validate_args(args) -> None:
    if (
        args.motion_state_updates != 100
        or any(value != 0 for value in (
            args.trajectory_updates, args.selector_updates,
            args.decoder_joint_updates,
        ))
        or args.stop_after_update != 0
    ):
        raise ValueError("alternating screen is fixed to 100 state-only updates")
    if args.seed not in PROBE_SEEDS:
        raise ValueError("alternating screen seed differs")
    if args.diagnostic_oracle_association is not True:
        raise ValueError("alternating screen requires diagnostic association")
    if args.allow_mapper_h_mismatch is not True:
        raise ValueError("alternating screen requires qualified mapper/H mismatch")
    mismatched_values = {
        name: {"expected": expected, "actual": getattr(args, name)}
        for name, expected in LOCKED_VALUES.items()
        if getattr(args, name) != expected
    }
    if mismatched_values:
        raise ValueError(
            f"alternating screen scalar contract differs: {mismatched_values}"
        )
    expected_paths = dict(V8_EXPECTED_PATHS)
    expected_paths["v8_joint_control_checkpoint"] = (
        V8_JOINT_CONTROL_CHECKPOINTS[args.seed]
    )
    mismatched_paths = {
        name: {
            "expected": str(expected.resolve()),
            "actual": str(Path(getattr(args, name)).resolve()),
        }
        for name, expected in expected_paths.items()
        if Path(getattr(args, name)).resolve() != expected.resolve()
    }
    if mismatched_paths:
        raise ValueError(
            f"alternating screen artifact contract differs: {mismatched_paths}"
        )
    control = Path(args.v8_joint_control_checkpoint).resolve()
    if sha256_file(control) != V8_JOINT_CONTROL_SHA256[args.seed]:
        raise ValueError("alternating screen V8 control hash differs")


def _worsens(after: float, before: float, *, relative: float, absolute: float) -> bool:
    return (
        (before > 0.0 and after > before and after >= before * (1.0 + relative))
        or after >= before + absolute
    )


def _screen_checks(diagnostics: dict, v12: dict) -> dict[str, dict[str, bool]]:
    _assert_finite_tree(diagnostics, path="v13.diagnostics")
    _assert_finite_tree(v12, path="v12")
    overall_sign = diagnostics["groups"]["overall"][
        "candidate_yaw_sign_accuracy"
    ]
    v12_sign = v12["diagnostics"]["groups"]["overall"][
        "candidate_yaw_sign_accuracy"
    ]
    _validate_diagnostic_binding(
        {"overall_yaw_sign_accuracy": overall_sign, "diagnostics": diagnostics},
        {"overall_yaw_sign_accuracy": v12_sign},
    )
    groups = diagnostics["groups"]
    checks: dict[str, bool] = {}
    for group_name, minimum_improvement in (
        ("pair0", 0.25), ("combined_pair1", 0.20),
        ("combined_pair2", 0.20),
    ):
        candidate = groups[group_name]["candidate_yaw"]
        baseline = v12["diagnostics"]["groups"][group_name]["candidate_yaw"]
        short = group_name.replace("combined_", "")
        checks[f"{short}_yaw_mean_improves"] = (
            candidate["mean_m"] <= baseline["mean_m"] * (1.0 - minimum_improvement)
        )
        checks[f"{short}_yaw_p50_improves"] = (
            candidate["p50_m"] <= baseline["p50_m"] * (1.0 - minimum_improvement)
        )
        checks[f"{short}_yaw_sign_at_least_0_75"] = (
            groups[group_name]["candidate_yaw_sign_accuracy"] >= 0.75
        )
    for metric in ("mean_m", "p50_m"):
        checks[f"pair3_yaw_{metric}_does_not_regress"] = (
            groups["combined_pair3"]["candidate_yaw"][metric]
            <= v12["diagnostics"]["groups"]["combined_pair3"][
                "candidate_yaw"
            ][metric] * 1.10
        )
        checks[f"overall_yaw_{metric}_improves"] = (
            groups["overall"]["candidate_yaw"][metric]
            <= v12["diagnostics"]["groups"]["overall"][
                "candidate_yaw"
            ][metric] * 0.80
        )
    for group_name in (
        "overall", "pair0", "combined_pair1", "combined_pair2",
        "combined_pair3",
    ):
        short = group_name.replace("combined_", "")
        for field in ("candidate_velocity", "candidate_yaw"):
            metric = "velocity" if field.endswith("velocity") else "yaw"
            checks[f"{short}_{metric}_p95_no_catastrophic_regression"] = (
                groups[group_name][field]["p95_m"]
                <= v12["diagnostics"]["groups"][group_name][field]["p95_m"]
                * 1.10
            )
    checks["overall_velocity_mean_does_not_regress"] = (
        groups["overall"]["candidate_velocity"]["mean_m"]
        <= v12["diagnostics"]["groups"]["overall"][
            "candidate_velocity"
        ]["mean_m"] * 1.10
    )
    checks["overall_velocity_p50_does_not_regress"] = (
        groups["overall"]["candidate_velocity"]["p50_m"]
        <= v12["diagnostics"]["groups"]["overall"][
            "candidate_velocity"
        ]["p50_m"] * 1.10
    )
    pair0 = groups["pair0"]
    checks["pair0_handle_break_worsens_yaw"] = _worsens(
        pair0["broken_handle_intervention_yaw"]["mean_m"],
        pair0["candidate_handle_intervention_yaw"]["mean_m"],
        relative=0.10, absolute=0.50,
    )
    checks["pair0_handle_break_worsens_closure"] = _worsens(
        pair0["fixed_broken_handle_closure"]["mean_m"],
        pair0["fixed_intact_handle_closure"]["mean_m"],
        relative=0.05, absolute=0.002,
    )
    for pair in (1, 2, 3):
        group = groups[f"combined_pair{pair}"]
        checks[f"pair{pair}_break_worsens_yaw"] = _worsens(
            group["broken_pair_intervention_yaw"]["mean_m"],
            group["candidate_pair_intervention_yaw"]["mean_m"],
            relative=0.10, absolute=0.30,
        )
        checks[f"pair{pair}_break_worsens_closure"] = _worsens(
            group["fixed_broken_pair_closure"]["mean_m"],
            group["fixed_intact_pair_closure"]["mean_m"],
            relative=0.05, absolute=0.002,
        )
    ramp = diagnostics["common_ramp_equivariance"]
    checks["common_ramp_velocity_mean_at_most_0_15"] = (
        ramp["velocity_delta_error_mps"]["mean_m"] <= 0.15
    )
    checks["common_ramp_yaw_p99_at_most_0_02"] = (
        ramp["yaw_invariance_error_rad_s"]["p99_m"] <= 0.02
    )
    reversal = diagnostics["relative_reversal_equivariance"]
    checks["reflection_velocity_invariance_at_most_0_15"] = (
        reversal["velocity_prediction_invariance_mps"]["mean_m"] <= 0.15
    )
    checks["reflection_yaw_antisymmetry_at_most_0_50"] = (
        reversal["yaw_prediction_antisymmetry_rad_s"]["mean_m"] <= 0.50
    )
    isolation_value = diagnostics["write_isolation"][
        "zero_velocity_max_absolute_yaw_difference_normalized"
    ]
    checks["write_isolation_exact_zero"] = (
        not isinstance(isolation_value, bool)
        and isinstance(isolation_value, (int, float))
        and math.isfinite(float(isolation_value))
        and float(isolation_value) == 0.0
    )
    return {
        name: {"required": True, "passed": bool(passed)}
        for name, passed in checks.items()
    }


def _validated_v12_baseline(seed: int) -> tuple[Path, dict]:
    path = (V12_RESULT_ROOTS[seed] / "probe_result.json").resolve()
    if sha256_file(path) != V12_RESULT_SHA256[seed]:
        raise ValueError("alternating fixed V12 result hash differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("seed") != seed
        or value.get("checkpoint_sha256") != V12_CHECKPOINT_SHA256[seed]
        or value.get("test_accessed") is not False
        or not isinstance(value.get("diagnostics"), dict)
    ):
        raise ValueError("alternating fixed V12 result identity differs")
    return path, value


def _validate_completed_alternating_screen(
    args, checkpoint: Path, parameter_count: int,
) -> dict:
    output = Path(args.output).resolve()
    manifest = json.loads(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    fixed = manifest.get("fixed_final_checkpoint")
    contract = manifest.get("contract")
    expected_checkpoint = output / "checkpoints" / "checkpoint-update-000100.pt"
    if (
        manifest.get("schema_version") != RUN_SCHEMA
        or manifest.get("status") != "complete"
        or type(manifest.get("progress", {}).get("global_update")) is not int
        or manifest.get("progress", {}).get("global_update") != 100
        or manifest.get("state_gate_only") is not True
        or manifest.get("state_gate_future_modules_unchanged") is not True
        or manifest.get("fixed_schedule") != {
            "motion_state": 100, "trajectory": 0,
            "selector": 0, "decoder_joint": 0,
        }
        or not isinstance(fixed, dict)
        or type(fixed.get("update")) is not int
        or fixed.get("update") != 100
        or fixed.get("selected_by_validation") is not False
        or checkpoint.resolve() != expected_checkpoint.resolve()
        or Path(str(fixed.get("path", ""))).resolve() != checkpoint.resolve()
        or fixed.get("sha256") != sha256_file(checkpoint)
        or not isinstance(contract, dict)
        or manifest.get("contract_sha256") != _json_sha256(contract)
        or type(contract.get("fixed_total_updates")) is not int
        or contract.get("fixed_total_updates") != 100
        or contract.get("fixed_stage_order") != ["motion_state"]
    ):
        raise ValueError("alternating fixed artifact is incomplete")
    recorded_git = manifest.get("provenance", {}).get("git", {})
    current_git = _git_state()
    if (
        recorded_git.get("worktree_dirty") is not False
        or current_git.get("worktree_dirty") is not False
        or recorded_git.get("git_commit") in {None, "unknown"}
        or not isinstance(recorded_git.get("git_commit"), str)
        or recorded_git.get("git_commit") != current_git.get("git_commit")
    ):
        raise ValueError("alternating source checkout changed")
    expected_callables = {
        "state_loss_function": None,
        "state_step_function": _callable_contract(
            equivariant_alternating_train_step
        ),
        "motion_state_cell_function": _callable_contract(
            alternating_motion_state_cells
        ),
        "dataset_preflight_function": _callable_contract(
            alternating_dataset_preflight
        ),
        "final_diagnostic_function": _callable_contract(
            omega_first_ordered_validation_diagnostics
        ),
    }
    for place in ("contract", "provenance"):
        source = manifest.get(place, {})
        if any(source.get(name) != value for name, value in expected_callables.items()):
            raise ValueError("alternating callable contracts differ")
    preflight = contract.get("dataset_preflight_report")
    if (
        preflight != manifest.get("provenance", {}).get(
            "dataset_preflight_report"
        )
    ):
        raise ValueError("alternating yaw alias preflight differs")
    _validate_yaw_preflight_report(preflight)
    provenance = manifest.get("provenance", {})
    source_paths = provenance.get("source_paths")
    source_hashes = provenance.get("source_sha256")
    if (
        not isinstance(source_paths, dict) or not isinstance(source_hashes, dict)
        or set(source_paths) != set(source_hashes)
        or any(
            sha256_file(Path(source_paths[name]).resolve()) != source_hashes[name]
            for name in source_paths
        )
    ):
        raise ValueError("alternating source hashes differ")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != RUN_SCHEMA
        or payload.get("model_class")
        != "AnonymousEquivariantAlternatingTwistProbe"
        or payload.get("checkpoint_selected_by_validation") is not False
        or payload.get("fixed_endpoint") is not True
        or payload.get("checkpoint_role") != "fixed_final_endpoint"
        or payload.get("progress") != {
            "global_update": 100, "stage": "motion_state", "stage_update": 100,
        }
        or any(
            type(payload.get("progress", {}).get(name)) is not int
            for name in ("global_update", "stage_update")
        )
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
        raise ValueError("alternating checkpoint identity differs")
    history = payload.get("validation_history")
    if (
        not isinstance(history, list) or len(history) != 2
        or history[0].get("global_update") != 0
        or history[-1].get("global_update") != 100
        or _json_sha256(history[-1].get("metrics"))
        != _json_sha256(manifest.get("final_validation"))
        or manifest.get("state_substage_counts") != EXPECTED_SUBSTAGE_COUNTS
        or any(
            type(value) is not int
            for value in manifest.get("state_substage_counts", {}).values()
        )
        or manifest.get("state_substage_transitions")
        != EXPECTED_SUBSTAGE_TRANSITIONS
        or any(
            type(entry.get("global_update")) is not int
            for entry in manifest.get("state_substage_transitions", [])
        )
    ):
        raise ValueError("alternating validation or typed schedule differs")
    for name in (
        "gradient_isolation_verified", "state_substage_counts",
        "state_substage_transitions", "state_branch_hash_history",
        "final_diagnostics",
    ):
        if _json_sha256(payload.get(name)) != _json_sha256(manifest.get(name)):
            raise ValueError(f"alternating checkpoint/manifest {name} differs")

    branch_history = manifest.get("state_branch_hash_history")
    expected_updates = [0, 35, 55, 80, 100]
    expected_branches = [None, "omega0", "velocity0", "omega1", "velocity1"]
    if not isinstance(branch_history, list) or len(branch_history) != 5:
        raise ValueError("alternating branch history length differs")
    previous_hashes = None
    for entry, update, changed_branch in zip(
        branch_history, expected_updates, expected_branches, strict=True,
    ):
        hashes = entry.get("hashes")
        if (
            type(entry.get("global_update")) is not int
            or entry.get("global_update") != update
            or set(hashes or {})
            != {"context", "omega0", "velocity0", "omega1", "velocity1"}
            or any(
                not _is_lower_sha256(value)
                for value in (hashes or {}).values()
            )
        ):
            raise ValueError("alternating branch hash entry differs")
        if changed_branch is None:
            if entry.get("reason") != "initial":
                raise ValueError("alternating initial branch hash differs")
        else:
            if (
                entry.get("reason") != "state_substage_endpoint"
                or entry.get("substage") != f"typed_alternating_{changed_branch}"
                or hashes[changed_branch] == previous_hashes[changed_branch]
                or any(
                    hashes[name] != previous_hashes[name]
                    for name in hashes if name != changed_branch
                )
            ):
                raise ValueError("alternating typed branch write history differs")
        previous_hashes = hashes

    model_state = payload["model"]
    config = payload["model_config"]
    scale = config.get("motion_state_scale")
    if not isinstance(scale, list) or len(scale) != 4:
        raise ValueError("alternating checkpoint motion scale differs")
    reconstructed = AnonymousEquivariantAlternatingTwistProbe(
        velocity_scale_mps=tuple(float(value) for value in scale[:3]),
        yaw_rate_scale_rad_s=float(scale[3]), channels=args.channels,
        dropout=args.dropout, message_layers=args.message_layers,
        basis_count=args.basis_count,
    )
    if reconstructed.config != config:
        raise ValueError("alternating reconstructed model config differs")
    reconstructed.load_state_dict(model_state, strict=True)
    if branch_history[-1]["hashes"] != reconstructed.state_branch_hashes():
        raise ValueError("alternating final branch hashes do not bind model")
    reconstructed_count = sum(
        parameter.numel() for name in STATE_MODULES
        for parameter in getattr(reconstructed, name).parameters()
    )
    if reconstructed_count != parameter_count:
        raise ValueError("alternating reconstructed capacity differs")
    actual_hashes = {
        name: state_dict_sha256({
            key[len(name) + 1:]: value for key, value in model_state.items()
            if key.startswith(name + ".")
        })
        for name in ALL_TRAINABLE_MODULES
    }
    if actual_hashes != manifest.get("trainable_final_state_dict_sha256"):
        raise ValueError("alternating final module hashes differ")
    initial_hashes = manifest.get("trainable_initial_state_dict_sha256")
    if (
        not isinstance(initial_hashes, dict)
        or any(
            actual_hashes.get(name) != initial_hashes.get(name)
            for name in FROZEN_FUTURE_MODULES
        )
    ):
        raise ValueError("alternating frozen future modules changed")

    v12_path, v12 = _validated_v12_baseline(args.seed)
    diagnostics = manifest["final_diagnostics"]
    if (
        not isinstance(diagnostics, dict)
        or set(diagnostics) != DIAGNOSTIC_FIELDS
        or diagnostics.get("schema_version") != DIAGNOSTIC_SCHEMA
        or diagnostics.get("validation_only") is not True
        or diagnostics.get("test_accessed") is not False
        or diagnostics.get("seed") != args.seed
        or set(diagnostics.get("groups", {})) != set(GROUP_NAMES)
        or set(diagnostics.get("write_isolation", {}))
        != WRITE_ISOLATION_FIELDS
    ):
        raise ValueError("alternating diagnostics are incomplete")
    candidate_binding = {
        "overall_yaw_sign_accuracy": diagnostics["groups"]["overall"][
            "candidate_yaw_sign_accuracy"
        ],
        "diagnostics": diagnostics,
    }
    control_binding = {
        "overall_yaw_sign_accuracy": v12["diagnostics"]["groups"]["overall"][
            "candidate_yaw_sign_accuracy"
        ],
    }
    _assert_finite_tree(candidate_binding, path="v13")
    _validate_diagnostic_binding(candidate_binding, control_binding)
    source_manifest = json.loads(
        (Path(args.dataset).resolve() / "dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    current_dataset_manifest_sha256 = sha256_file(
        Path(args.dataset).resolve() / "dataset_manifest.json"
    )
    if (
        source_manifest.get("test_accessed") is not False
        or manifest.get("provenance", {}).get("test_accessed") is not False
        or current_dataset_manifest_sha256
        != contract.get("dataset_manifest_sha256")
        or current_dataset_manifest_sha256
        != manifest.get("provenance", {}).get("dataset", {}).get(
            "manifest_sha256"
        )
    ):
        raise ValueError("alternating source dataset provenance differs")
    expected_truth_sha = source_manifest.get("truth_history_manifest_sha256")
    if not isinstance(expected_truth_sha, str):
        raise ValueError("alternating source truth binding is missing")
    truth_index = MotionTruthIndex(
        args.truth_history, expected_manifest_sha256=expected_truth_sha,
    )
    train_dataset = _dataset(
        Path(args.dataset).resolve(), "train",
        sample_limit=args.train_limit_per_class,
    )
    validation_dataset = _dataset(
        Path(args.dataset).resolve(), "validation",
        sample_limit=args.validation_limit_per_class,
    )
    train_join = truth_index.attach(train_dataset, "train")
    validation_join = truth_index.attach(validation_dataset, "validation")
    recomputed_preflight = alternating_dataset_preflight(
        train_dataset, validation_dataset,
    )
    _validate_yaw_preflight_report(preflight, recomputed=recomputed_preflight)
    provenance = manifest["provenance"]
    recorded_dataset = provenance.get("dataset", {})
    if (
        provenance.get("truth_history") != truth_index.provenance
        or provenance.get("truth_join")
        != {"train": train_join, "validation": validation_join}
        or Path(str(recorded_dataset.get("path", ""))).resolve()
        != Path(args.dataset).resolve()
        or recorded_dataset.get("manifest_sha256")
        != current_dataset_manifest_sha256
        or recorded_dataset.get("train") != train_dataset.audit
        or recorded_dataset.get("validation") != validation_dataset.audit
        or contract.get("truth_manifest_sha256") != truth_index.manifest_sha256
        or contract.get("truth_key_set_sha256") != truth_index.key_set_sha256
        or contract.get("truth_label_sha256") != truth_index.label_sha256
        or contract.get("train_join_key_set_sha256")
        != train_join["joined_key_set_sha256"]
        or contract.get("validation_join_key_set_sha256")
        != validation_join["joined_key_set_sha256"]
    ):
        raise ValueError("alternating truth/preflight source binding differs")
    standard = manifest["final_validation"]["motion_state"]
    for name in ("overall", "combined", "combined_speed_gt_1_7"):
        if (
            _json_sha256(diagnostics["groups"][name]["candidate_velocity"])
            != _json_sha256(standard[name]["velocity_vector_error_mps"])
            or _json_sha256(diagnostics["groups"][name]["candidate_yaw"])
            != _json_sha256(standard[name]["yaw_absolute_error_rad_s"])
        ):
            raise ValueError(f"alternating validation binding differs: {name}")
    checks = _screen_checks(diagnostics, v12)
    result = {
        "schema_version": RESULT_SCHEMA,
        "seed": args.seed,
        "source_commit": recorded_git["git_commit"],
        "contract_sha256": manifest["contract_sha256"],
        "fixed_updates": 100,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "total_state_parameter_count": parameter_count,
        "capacity_ceiling": int(1.05 * V8_JOINT_REACHABLE_STATE_PARAMETERS),
        "future_position_decoder_status": "deferred_and_frozen",
        "test_accessed": False,
        "checks": checks,
        "passed": all(item["passed"] for item in checks.values()),
        "final_validation": manifest["final_validation"]["motion_state"],
        "diagnostics": diagnostics,
        "v12_baseline": {
            "path": str(v12_path.resolve()),
            "result_sha256": V12_RESULT_SHA256[args.seed],
            "checkpoint_sha256": v12["checkpoint_sha256"],
        },
    }
    return result


def _finalize_screen(args, checkpoint: Path, parameter_count: int) -> dict:
    result = _validate_completed_alternating_screen(
        args, checkpoint, parameter_count,
    )
    output = Path(args.output).resolve()
    _atomic_json(output / "screen_result.json", result)
    return result


def main() -> None:
    args = build_alternating_screen_parser().parse_args()
    _validate_args(args)
    git = _git_state()
    if git.get("worktree_dirty") is not False or git.get("git_commit") in {
        None, "unknown",
    }:
        raise RuntimeError("alternating screen requires clean committed source")
    v77 = Path(args.v77_control_checkpoint).resolve()
    _preflight_control(v77)
    parameter_count = _state_parameter_count(
        AnonymousEquivariantAlternatingTwistProbe, args,
    )
    if parameter_count > int(1.05 * V8_JOINT_REACHABLE_STATE_PARAMETERS):
        raise ValueError("alternating state capacity exceeds V8 plus five percent")
    checkpoint = train(
        args,
        model_class=AnonymousEquivariantAlternatingTwistProbe,
        state_step_function=equivariant_alternating_train_step,
        motion_state_cell_function=alternating_motion_state_cells,
        dataset_preflight_function=alternating_dataset_preflight,
        final_diagnostic_function=omega_first_ordered_validation_diagnostics,
        run_schema=RUN_SCHEMA,
        extra_source_paths={
            "equivariant_alternating_model_and_step": Path(__file__).with_name(
                "equivariant_alternating_twist_future.py"
            ),
            "equivariant_alternating_runner": Path(__file__),
            "v12_diagnostics": Path(__file__).with_name(
                "train_omega_first_ordered_closure_probe.py"
            ),
        },
        state_gate_only=True,
        frozen_initialization_checkpoint=v77,
        frozen_initialization_modules=FROZEN_FUTURE_MODULES,
    )
    result = _finalize_screen(args, checkpoint, parameter_count)
    print(json.dumps({
        "seed": result["seed"], "passed": result["passed"],
        "checkpoint": result["checkpoint"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
