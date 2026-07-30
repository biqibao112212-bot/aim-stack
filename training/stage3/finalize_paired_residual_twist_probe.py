"""Fail-closed aggregation for the two fixed V10 paired-residual probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from .joint_rigid_flow_probe import AnonymousJointTwistProbe
from .paired_residual_twist_future import AnonymousPairedResidualTwistProbe
from .train_anonymous_vehicle_motion import _json_sha256
from .train_causal_physical_ab import _git_state
from .train_joint_rigid_flow_probe import _state_parameter_count
from .train_paired_twist_set_probe import (
    PROBE_SEEDS,
    V8_JOINT_CONTROL_CHECKPOINTS,
    V8_JOINT_REACHABLE_STATE_PARAMETERS,
    V8_JOINT_TOTAL_STATE_PARAMETERS,
    _validate_args,
)
from .train_paired_residual_twist_probe import (
    RUN_SCHEMA,
    _validate_completed_residual_probe,
)
from .train_pnp_window_mapper_distillation import _atomic_json


def _load_candidate(root: Path) -> tuple[dict, dict, dict]:
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != RUN_SCHEMA:
        raise ValueError(f"v10 probe schema differs: {root}")
    contract_args = dict(manifest["contract"]["args"])
    contract_args["stop_after_update"] = 0
    args = SimpleNamespace(**contract_args)
    _validate_args(args)
    if Path(args.output).resolve() != root.resolve():
        raise ValueError(f"v10 probe output contract differs: {root}")
    count = _state_parameter_count(AnonymousPairedResidualTwistProbe, args)
    checkpoint = Path(manifest["fixed_final_checkpoint"]["path"]).resolve()
    result = _validate_completed_residual_probe(args, checkpoint, count)
    stored = json.loads((root / "probe_result.json").read_text(encoding="utf-8"))
    if _json_sha256(stored) != _json_sha256(result):
        raise ValueError(f"stored v10 result differs: {root}")
    return manifest, vars(args), result


def _control_result(seed: int) -> dict:
    root = V8_JOINT_CONTROL_CHECKPOINTS[seed].resolve().parent.parent
    result = json.loads((root / "probe_result.json").read_text(encoding="utf-8"))
    if (
        result.get("schema_version") != "stage3-v8-rigid-flow-probe-result-v1"
        or result.get("variant") != "joint"
        or result.get("seed") != seed
        or Path(result.get("checkpoint", "")).resolve()
        != V8_JOINT_CONTROL_CHECKPOINTS[seed].resolve()
        or result.get("test_accessed") is not False
    ):
        raise ValueError(f"v10 V8 control result identity differs: {seed}")
    return result


def _worsens(value: float, baseline: float, ratio: float, absolute: float) -> bool:
    return value > baseline and (
        value >= ratio * baseline or value >= baseline + absolute
    )


def _checks(candidate: dict, control: dict) -> dict:
    intact_high = candidate["high_speed_combined_velocity_mean_mps"]
    broken_high = candidate["broken_pairing_high_speed_velocity_mean_mps"]
    residual_used = (
        _worsens(
            candidate["zero_residual_combined_velocity_mean_mps"],
            candidate["combined_velocity_mean_mps"], 1.08, 0.10,
        )
        or _worsens(
            candidate["zero_residual_high_speed_velocity_mean_mps"],
            intact_high, 1.08, 0.10,
        )
    )
    checks = {
        "overall_velocity_improves_at_least_10_percent": (
            candidate["overall_velocity_mean_mps"]
            <= 0.90 * control["overall_velocity_mean_mps"]
        ),
        "combined_velocity_improves_at_least_15_percent": (
            candidate["combined_velocity_mean_mps"]
            <= 0.85 * control["combined_velocity_mean_mps"]
        ),
        "high_speed_combined_velocity_improves_at_least_15_percent": (
            intact_high <= 0.85 * control["high_speed_combined_velocity_mean_mps"]
        ),
        "overall_yaw_regresses_no_more_than_5_percent": (
            candidate["overall_yaw_mean_rad_s"]
            <= 1.05 * control["overall_yaw_mean_rad_s"]
        ),
        "combined_yaw_regresses_no_more_than_5_percent": (
            candidate["combined_yaw_mean_rad_s"]
            <= 1.05 * control["combined_yaw_mean_rad_s"]
        ),
        "high_speed_combined_yaw_regresses_no_more_than_5_percent": (
            candidate["high_speed_combined_yaw_mean_rad_s"]
            <= 1.05 * control["high_speed_combined_yaw_mean_rad_s"]
        ),
        "yaw_sign_regresses_no_more_than_0_01": (
            candidate["overall_yaw_sign_accuracy"]
            >= control["overall_yaw_sign_accuracy"] - 0.01
        ),
        "low_speed_history32_pair3_core_yaw_improves_at_least_10_percent": (
            candidate["core_yaw_mean_rad_s"]
            <= 0.90 * candidate["control_core_yaw_mean_rad_s"]
        ),
        "combined_pair1_2_velocity_improves_at_least_10_percent": (
            candidate["pair1_2_velocity_mean_mps"]
            <= 0.90 * candidate["control_pair1_2_velocity_mean_mps"]
        ),
        "combined_pair1_2_yaw_improves_at_least_10_percent": (
            candidate["pair1_2_yaw_mean_rad_s"]
            <= 0.90 * candidate["control_pair1_2_yaw_mean_rad_s"]
        ),
        "broken_pairing_worsens_high_speed_velocity": _worsens(
            broken_high, intact_high, 1.10, 0.15,
        ),
        "broken_pairing_worsens_pair1_2_velocity": _worsens(
            candidate["broken_pairing_pair1_2_velocity_mean_mps"],
            candidate["pair1_2_velocity_mean_mps"], 1.05, 0.05,
        ),
        "broken_pairing_worsens_pair1_2_yaw": _worsens(
            candidate["broken_pairing_pair1_2_yaw_mean_rad_s"],
            candidate["pair1_2_yaw_mean_rad_s"], 1.10, 0.30,
        ),
        "paired_rotation_residual_is_used": residual_used,
    }
    return {
        name: {"passed": bool(passed), "required": True}
        for name, passed in checks.items()
    }


def finalize(run_roots: list[Path], output: Path) -> dict:
    if len(run_roots) != 2 or len({path.resolve() for path in run_roots}) != 2:
        raise ValueError("v10 aggregation requires two distinct run roots")
    if output.exists():
        raise FileExistsError(f"refusing existing v10 aggregate root: {output}")
    git = _git_state()
    if git.get("worktree_dirty") is not False:
        raise ValueError("v10 aggregation requires a clean checkout")
    loaded = {}
    for root in run_roots:
        manifest, args, result = _load_candidate(root.resolve())
        seed = int(args["seed"])
        if seed in loaded:
            raise ValueError(f"duplicate v10 seed: {seed}")
        loaded[seed] = (manifest, args, result)
    if set(loaded) != set(PROBE_SEEDS):
        raise ValueError("v10 seed matrix differs")
    commits = {entry[0]["provenance"]["git"]["git_commit"] for entry in loaded.values()}
    if commits != {git.get("git_commit")}:
        raise ValueError("v10 source commit differs across runs or checkout")
    comparable_args = []
    for _, args, _ in loaded.values():
        comparable = dict(args)
        for name in ("output", "seed", "v8_joint_control_checkpoint"):
            comparable.pop(name)
        comparable_args.append(comparable)
    if len({_json_sha256(value) for value in comparable_args}) != 1:
        raise ValueError("v10 arms differ outside seed-dependent fields")
    controls = {seed: _control_result(seed) for seed in PROBE_SEEDS}
    comparisons = {}
    passed_all = True
    v8_total_count = None
    for seed in PROBE_SEEDS:
        candidate = loaded[seed][2]
        control = controls[seed]
        diagnostics = candidate["diagnostics"]["groups"]
        for group_name, velocity_key, yaw_key in (
            ("overall", "overall_velocity_mean_mps", "overall_yaw_mean_rad_s"),
            ("combined", "combined_velocity_mean_mps", "combined_yaw_mean_rad_s"),
            (
                "combined_speed_gt_1_7",
                "high_speed_combined_velocity_mean_mps",
                "high_speed_combined_yaw_mean_rad_s",
            ),
        ):
            if (
                diagnostics[group_name]["control_velocity"]["mean_m"]
                != control[velocity_key]
                or diagnostics[group_name]["control_yaw"]["mean_m"]
                != control[yaw_key]
            ):
                raise ValueError(f"v10 replayed control differs: {seed}/{group_name}")
        if diagnostics["overall"]["control_yaw_sign_accuracy"] != control[
            "overall_yaw_sign_accuracy"
        ]:
            raise ValueError(f"v10 replayed control sign differs: {seed}")
        args = SimpleNamespace(**{**loaded[seed][1], "stop_after_update": 0})
        count = _state_parameter_count(AnonymousJointTwistProbe, args)
        v8_total_count = count if v8_total_count is None else v8_total_count
        if count != v8_total_count or count != V8_JOINT_TOTAL_STATE_PARAMETERS:
            raise ValueError("v10 V8 control capacity differs by seed")
        relative_capacity = abs(
            candidate["total_state_parameter_count"]
            - V8_JOINT_REACHABLE_STATE_PARAMETERS
        ) / V8_JOINT_REACHABLE_STATE_PARAMETERS
        if relative_capacity > 0.05:
            raise ValueError("v10/V8 reachable state capacity differs by more than 5%")
        checks = _checks(candidate, control)
        passed = all(item["passed"] for item in checks.values())
        comparisons[str(seed)] = {
            "candidate": candidate,
            "v8_joint_control": control,
            "checks": checks,
            "passed": passed,
        }
        passed_all &= passed
    report = {
        "schema_version": "stage3-v10-paired-residual-probe-aggregate-v1",
        "status": "passed" if passed_all else "failed",
        "test_accessed": False,
        "source_commit": git["git_commit"],
        "required_seeds": list(PROBE_SEEDS),
        "capacity": {
            "v8_joint_optimizer_owned_state_parameters": v8_total_count,
            "v8_joint_gradient_reachable_state_parameters": (
                V8_JOINT_REACHABLE_STATE_PARAMETERS
            ),
            "v10_gradient_reachable_state_parameters": loaded[PROBE_SEEDS[0]][2][
                "total_state_parameter_count"
            ],
            "v10_all_state_parameters_gradient_reachable": True,
            "matched_within_5_percent": True,
        },
        "comparisons": comparisons,
        "authorized_full_v10": passed_all,
    }
    output.mkdir(parents=True)
    _atomic_json(output / "probe_comparison.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="aggregate two fixed V10 probes")
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = finalize(
        [Path(value).resolve() for value in args.run], Path(args.output).resolve(),
    )
    print(json.dumps({
        "status": report["status"],
        "authorized_full_v10": report["authorized_full_v10"],
    }, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
