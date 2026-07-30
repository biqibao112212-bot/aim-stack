"""Fail-closed aggregation for the six fixed V8 structural probe runs."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace

from .factorized_common_relative_motion_future import FactorizedCommonRelativeMotionStateV7
from .joint_rigid_flow_probe import (
    AnonymousJointTwistProbe,
    AnonymousSeparatedRigidFlowProbe,
)
from .train_anonymous_vehicle_motion import _json_sha256
from .train_causal_physical_ab import _git_state
from .train_joint_rigid_flow_probe import (
    PROBE_SEEDS,
    RUN_SCHEMA,
    _state_parameter_count,
    _validate_args,
    _validate_completed_probe,
)
from .train_pnp_window_mapper_distillation import _atomic_json


VARIANTS = ("v7_factorized", "separated", "joint")
MODEL_CLASSES = {
    "v7_factorized": FactorizedCommonRelativeMotionStateV7,
    "separated": AnonymousSeparatedRigidFlowProbe,
    "joint": AnonymousJointTwistProbe,
}


def _load_run(root: Path) -> tuple[dict, dict, dict]:
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != RUN_SCHEMA:
        raise ValueError(f"probe schema differs: {root}")
    contract_args = dict(manifest["contract"]["args"])
    # The shared trainer deliberately excludes interruption-only controls from
    # the immutable run contract.  Aggregation reconstructs a completed run,
    # so restore the only admissible V8 value before applying the runner gate.
    contract_args["stop_after_update"] = 0
    args = SimpleNamespace(**contract_args)
    _validate_args(args)
    if Path(args.output).resolve() != root.resolve():
        raise ValueError(f"probe output contract differs: {root}")
    model_class = MODEL_CLASSES[args.variant]
    count = _state_parameter_count(model_class, args)
    checkpoint = Path(manifest["fixed_final_checkpoint"]["path"]).resolve()
    result = _validate_completed_probe(args, model_class, checkpoint, count)
    stored = json.loads((root / "probe_result.json").read_text(encoding="utf-8"))
    if _json_sha256(stored) != _json_sha256(result):
        raise ValueError(f"stored probe result differs from checkpoint-bound result: {root}")
    return manifest, vars(args), result


def _candidate_checks(candidate: dict, control: dict) -> dict:
    checks = {
        "overall_yaw_improves_at_least_30_percent": (
            candidate["overall_yaw_mean_rad_s"]
            <= 0.70 * control["overall_yaw_mean_rad_s"]
        ),
        "combined_yaw_improves_at_least_30_percent": (
            candidate["combined_yaw_mean_rad_s"]
            <= 0.70 * control["combined_yaw_mean_rad_s"]
        ),
        "yaw_sign_does_not_regress_more_than_0_01": (
            candidate["overall_yaw_sign_accuracy"]
            >= control["overall_yaw_sign_accuracy"] - 0.01
        ),
        "overall_velocity_does_not_regress_more_than_0_03_mps": (
            candidate["overall_velocity_mean_mps"]
            <= control["overall_velocity_mean_mps"] + 0.03
        ),
        "combined_velocity_does_not_regress_more_than_0_03_mps": (
            candidate["combined_velocity_mean_mps"]
            <= control["combined_velocity_mean_mps"] + 0.03
        ),
        "high_speed_combined_yaw_improves_at_least_30_percent": (
            candidate["high_speed_combined_yaw_mean_rad_s"]
            <= 0.70 * control["high_speed_combined_yaw_mean_rad_s"]
        ),
        "high_speed_combined_velocity_does_not_regress_more_than_0_03_mps": (
            candidate["high_speed_combined_velocity_mean_mps"]
            <= control["high_speed_combined_velocity_mean_mps"] + 0.03
        ),
    }
    return {
        name: {"passed": bool(passed), "required": True}
        for name, passed in checks.items()
    }


def finalize(run_roots: list[Path], output: Path) -> dict:
    if len(run_roots) != 6 or len({path.resolve() for path in run_roots}) != 6:
        raise ValueError("v8 probe aggregation requires six distinct run roots")
    if output.exists():
        raise FileExistsError(f"refusing existing aggregate root: {output}")
    current_git = _git_state()
    if current_git.get("worktree_dirty") is not False:
        raise ValueError("v8 probe aggregation requires a clean checkout")
    loaded: dict[tuple[str, int], tuple[dict, dict, dict]] = {}
    for root in run_roots:
        manifest, args, result = _load_run(root.resolve())
        key = (str(args["variant"]), int(args["seed"]))
        if key in loaded:
            raise ValueError(f"duplicate v8 probe arm: {key}")
        loaded[key] = (manifest, args, result)
    expected = set(product(VARIANTS, PROBE_SEEDS))
    if set(loaded) != expected:
        raise ValueError(f"v8 probe arm matrix differs: {set(loaded) ^ expected}")
    commits = {entry[0]["provenance"]["git"]["git_commit"] for entry in loaded.values()}
    if commits != {current_git.get("git_commit")}:
        raise ValueError("v8 probe source commit differs across runs or current checkout")
    comparable_args = []
    for _, args, _ in loaded.values():
        comparable = dict(args)
        for name in ("output", "seed", "variant"):
            comparable.pop(name)
        comparable_args.append(comparable)
    if len({_json_sha256(value) for value in comparable_args}) != 1:
        raise ValueError("v8 probe arms differ outside variant, seed and output")
    provenance_fields = (
        "dataset", "truth_history", "frozen_initial_state_dict_sha256", "sampler",
        "frozen_module_initialization",
    )
    for field in provenance_fields:
        values = {
            _json_sha256(entry[0]["provenance"].get(field))
            for entry in loaded.values()
        }
        if len(values) != 1:
            raise ValueError(f"v8 probe provenance differs: {field}")
    separated_count = loaded[("separated", PROBE_SEEDS[0])][2][
        "total_state_parameter_count"
    ]
    joint_count = loaded[("joint", PROBE_SEEDS[0])][2][
        "total_state_parameter_count"
    ]
    capacity_delta = abs(separated_count - joint_count) / max(
        separated_count, joint_count,
    )
    if capacity_delta > 0.05:
        raise ValueError("separated and joint probe capacity differs by more than 5%")
    comparisons: dict[str, dict] = {}
    candidate_pass: dict[str, bool] = {}
    for variant in ("separated", "joint"):
        variant_pass = True
        comparisons[variant] = {}
        for seed in PROBE_SEEDS:
            candidate = loaded[(variant, seed)][2]
            control = loaded[("v7_factorized", seed)][2]
            checks = _candidate_checks(candidate, control)
            passed = all(item["passed"] for item in checks.values())
            comparisons[variant][str(seed)] = {
                "candidate": candidate,
                "v7_factorized_control": control,
                "checks": checks,
                "passed": passed,
            }
            variant_pass &= passed
        candidate_pass[variant] = variant_pass
    authorized = [name for name, passed in candidate_pass.items() if passed]
    report = {
        "schema_version": "stage3-v8-rigid-flow-probe-aggregate-v1",
        "status": "passed" if authorized else "failed",
        "test_accessed": False,
        "source_commit": current_git["git_commit"],
        "required_arm_matrix": [
            {"variant": variant, "seed": seed}
            for variant, seed in product(VARIANTS, PROBE_SEEDS)
        ],
        "capacity": {
            "separated_total_state_parameters": separated_count,
            "joint_total_state_parameters": joint_count,
            "relative_delta": capacity_delta,
            "matched_within_5_percent": True,
            "count_semantics": (
                "total parameters in optimizer-owned state modules; the common "
                "context contains shared legacy submodules that may be gradient-unreachable"
            ),
            "v7_control_counts_are_reported_but_not_capacity_matched": True,
        },
        "comparisons": comparisons,
        "authorized_full_v8_candidates": authorized,
        "interpretation_limit": (
            "separated-vs-joint is capacity matched; improvements over the smaller v7 "
            "control select a practical architecture but do not isolate capacity causally"
        ),
    }
    output.mkdir(parents=True)
    _atomic_json(output / "probe_comparison.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="aggregate six fixed V8 probe arms")
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = finalize(
        [Path(value).resolve() for value in args.run], Path(args.output).resolve(),
    )
    print(json.dumps({
        "status": report["status"],
        "authorized_full_v8_candidates": report["authorized_full_v8_candidates"],
    }, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
