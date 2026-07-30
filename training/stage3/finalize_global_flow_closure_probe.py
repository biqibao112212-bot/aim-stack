"""Aggregate two fixed V11 global-history-closure probes fail-closed."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .global_flow_closure_future import AnonymousGlobalFlowClosureProbe
from .train_causal_physical_ab import _git_state
from .train_global_flow_closure_probe import (
    RUN_SCHEMA,
    _validate_completed_closure_probe,
)
from .train_paired_twist_set_probe import (
    PROBE_SEEDS,
    V8_JOINT_CONTROL_CHECKPOINTS,
    V8_JOINT_CONTROL_SHA256,
    V8_JOINT_REACHABLE_STATE_PARAMETERS,
    _validate_args,
)
from .train_joint_rigid_flow_probe import _state_parameter_count
from .train_pnp_window_mapper_distillation import _atomic_json


AGGREGATE_SCHEMA = "stage3-v11-global-flow-closure-probe-aggregate-v1"


def _improves(candidate: float, control: float, fraction: float) -> bool:
    return candidate <= (1.0 - fraction) * control


def _not_worse(candidate: float, control: float, fraction: float) -> bool:
    return candidate <= (1.0 + fraction) * control


def _worsens(intervention: float, normal: float, fraction: float, absolute: float) -> bool:
    return intervention >= (1.0 + fraction) * normal or intervention >= normal + absolute


def _checks(candidate: dict, control: dict) -> dict[str, dict[str, bool]]:
    checks: dict[str, bool] = {
        "overall_velocity_improves_at_least_10_percent": _improves(
            candidate["overall_velocity_mean_mps"],
            control["overall_velocity_mean_mps"], 0.10,
        ),
        "combined_velocity_improves_at_least_15_percent": _improves(
            candidate["combined_velocity_mean_mps"],
            control["combined_velocity_mean_mps"], 0.15,
        ),
        "high_speed_velocity_improves_at_least_15_percent": _improves(
            candidate["high_speed_combined_velocity_mean_mps"],
            control["high_speed_combined_velocity_mean_mps"], 0.15,
        ),
        "overall_yaw_regresses_no_more_than_5_percent": _not_worse(
            candidate["overall_yaw_mean_rad_s"],
            control["overall_yaw_mean_rad_s"], 0.05,
        ),
        "combined_yaw_regresses_no_more_than_5_percent": _not_worse(
            candidate["combined_yaw_mean_rad_s"],
            control["combined_yaw_mean_rad_s"], 0.05,
        ),
        "high_speed_yaw_regresses_no_more_than_5_percent": _not_worse(
            candidate["high_speed_combined_yaw_mean_rad_s"],
            control["high_speed_combined_yaw_mean_rad_s"], 0.05,
        ),
        "core_yaw_improves_at_least_10_percent": _improves(
            candidate["core_yaw_mean_rad_s"],
            candidate["control_core_yaw_mean_rad_s"], 0.10,
        ),
        "yaw_sign_regresses_no_more_than_0_01": (
            candidate["overall_yaw_sign_accuracy"]
            >= control["overall_yaw_sign_accuracy"] - 0.01
        ),
        "broken_handle_worsens_overall_velocity": _worsens(
            candidate["broken_handle_overall_velocity_mean"],
            candidate["overall_velocity_mean_mps"], 0.10, 0.05,
        ),
        "closure_refinement_is_used": (
            _worsens(
                candidate["zero_refinement_combined_velocity_mean"],
                candidate["combined_velocity_mean_mps"], 0.08, 0.10,
            )
            or _worsens(
                candidate["zero_refinement_combined_yaw_mean"],
                candidate["combined_yaw_mean_rad_s"], 0.08, 0.30,
            )
        ),
    }
    for count in (1, 2):
        checks[f"pair{count}_velocity_improves_at_least_10_percent"] = _improves(
            candidate[f"pair{count}_velocity_mean"],
            candidate[f"control_pair{count}_velocity_mean"], 0.10,
        )
        checks[f"pair{count}_yaw_improves_at_least_10_percent"] = _improves(
            candidate[f"pair{count}_yaw_mean"],
            candidate[f"control_pair{count}_yaw_mean"], 0.10,
        )
        checks[f"broken_pair_does_not_improve_pair{count}_yaw"] = (
            candidate[f"broken_pair_pair{count}_yaw_mean"]
            >= 0.95 * candidate[f"pair{count}_yaw_mean"]
        )
    checks["pair3_velocity_regresses_no_more_than_5_percent"] = _not_worse(
        candidate["pair3_velocity_mean"],
        candidate["control_pair3_velocity_mean"], 0.05,
    )
    checks["pair3_yaw_regresses_no_more_than_5_percent"] = _not_worse(
        candidate["pair3_yaw_mean"], candidate["control_pair3_yaw_mean"], 0.05,
    )
    checks["broken_pair_worsens_pair3_yaw"] = _worsens(
        candidate["broken_pair_pair3_yaw_mean"],
        candidate["pair3_yaw_mean"], 0.10, 0.30,
    )
    cross = candidate["crossed_rotation_factors"]
    checks["crossed_velocity_follows_translation_source"] = (
        cross["velocity_error_to_translation_source"]["mean_m"]
        <= 0.80 * cross["velocity_error_to_rotation_donor"]["mean_m"]
    )
    checks["crossed_yaw_follows_rotation_donor"] = (
        cross["yaw_error_to_rotation_donor"]["mean_m"]
        <= 0.80 * cross["yaw_error_to_translation_source"]["mean_m"]
    )
    checks["crossed_yaw_sign_accuracy_at_least_0_85"] = (
        cross["yaw_sign_accuracy_to_rotation_donor"] >= 0.85
    )
    checks["broken_crossed_pairing_worsens_hybrid_state"] = (
        _worsens(
            cross["broken_pairing_velocity_error_to_hybrid"]["mean_m"],
            cross["velocity_error_to_translation_source"]["mean_m"],
            0.10, 0.05,
        )
        or _worsens(
            cross["broken_pairing_yaw_error_to_hybrid"]["mean_m"],
            cross["yaw_error_to_rotation_donor"]["mean_m"],
            0.10, 0.30,
        )
    )
    checks["broken_crossed_pairing_worsens_history_closure"] = _worsens(
        cross["broken_pairing_history_closure_error"]["mean_m"],
        cross["intact_history_closure_error"]["mean_m"],
        0.10, 0.02,
    )
    return {
        name: {"required": True, "passed": bool(passed)}
        for name, passed in checks.items()
    }


def _control(candidate: dict) -> dict:
    groups = candidate["diagnostics"]["groups"]
    result = {
        "overall_velocity_mean_mps": groups["overall"]["control_velocity"]["mean_m"],
        "overall_yaw_mean_rad_s": groups["overall"]["control_yaw"]["mean_m"],
        "combined_velocity_mean_mps": groups["combined"]["control_velocity"]["mean_m"],
        "combined_yaw_mean_rad_s": groups["combined"]["control_yaw"]["mean_m"],
        "high_speed_combined_velocity_mean_mps": groups[
            "combined_speed_gt_1_7"
        ]["control_velocity"]["mean_m"],
        "high_speed_combined_yaw_mean_rad_s": groups[
            "combined_speed_gt_1_7"
        ]["control_yaw"]["mean_m"],
        "overall_yaw_sign_accuracy": groups["overall"][
            "control_yaw_sign_accuracy"
        ],
    }
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in result.values()
    ):
        raise ValueError("v11 control metrics are invalid")
    return result


def _protected_control(seed: int) -> dict:
    checkpoint = V8_JOINT_CONTROL_CHECKPOINTS[seed].resolve()
    path = checkpoint.parents[1] / "probe_result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "stage3-v8-rigid-flow-probe-result-v1"
        or payload.get("seed") != seed
        or payload.get("variant") != "joint"
        or payload.get("fixed_updates") != 200
        or payload.get("test_accessed") is not False
        or Path(payload.get("checkpoint", "")).resolve() != checkpoint
        or payload.get("checkpoint_sha256") != V8_JOINT_CONTROL_SHA256[seed]
    ):
        raise ValueError("protected V8 joint control identity differs")
    return payload


def _validated_contract_namespace(contract_args: dict) -> argparse.Namespace:
    """Restore runtime-only defaults omitted from the immutable run contract."""
    values = dict(contract_args)
    values.setdefault("stop_after_update", 0)
    namespace = argparse.Namespace(**values)
    _validate_args(namespace)
    return namespace


def main() -> None:
    parser = argparse.ArgumentParser(description="aggregate two fixed V11 probes")
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.run) != 2:
        raise ValueError("v11 aggregate requires exactly two runs")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"v11 aggregate output already exists: {output}")
    git = _git_state()
    if git.get("worktree_dirty") is not False:
        raise RuntimeError("v11 aggregate requires a clean checkout")
    reports = []
    manifests = []
    for run in args.run:
        root = Path(run).resolve()
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != RUN_SCHEMA:
            raise ValueError("v11 run schema differs")
        namespace = _validated_contract_namespace(manifest["contract"]["args"])
        checkpoint = root / "checkpoints" / "checkpoint-update-000200.pt"
        recorded_report = json.loads((root / "probe_result.json").read_text(
            encoding="utf-8"
        ))
        parameter_count = _state_parameter_count(
            AnonymousGlobalFlowClosureProbe, namespace,
        )
        if recorded_report.get("total_state_parameter_count") != parameter_count:
            raise ValueError("v11 recorded state capacity differs")
        validated = _validate_completed_closure_probe(
            namespace, checkpoint, parameter_count,
        )
        if validated != recorded_report:
            raise ValueError("v11 recorded probe result differs")
        reports.append(validated)
        manifests.append(manifest)
    if {report["seed"] for report in reports} != set(PROBE_SEEDS):
        raise ValueError("v11 aggregate seed set differs")
    if any(report["source_commit"] != git.get("git_commit") for report in reports):
        raise ValueError("v11 aggregate source commit differs")
    counts = {report["total_state_parameter_count"] for report in reports}
    if len(counts) != 1:
        raise ValueError("v11 aggregate state capacity differs across seeds")
    parameter_count = next(iter(counts))
    capacity_matched = (
        abs(parameter_count - V8_JOINT_REACHABLE_STATE_PARAMETERS)
        / V8_JOINT_REACHABLE_STATE_PARAMETERS <= 0.05
    )
    if not capacity_matched:
        raise ValueError("v11 aggregate state capacity exceeds five percent")
    base_args = dict(manifests[0]["contract"]["args"])
    other_args = dict(manifests[1]["contract"]["args"])
    for name in ("seed", "output", "v8_joint_control_checkpoint"):
        base_args.pop(name, None)
        other_args.pop(name, None)
    if base_args != other_args:
        raise ValueError("v11 aggregate run arguments differ")
    comparisons = {}
    for candidate in sorted(reports, key=lambda item: item["seed"]):
        diagnostic_control = _control(candidate)
        control = _protected_control(candidate["seed"])
        for name, value in diagnostic_control.items():
            if control.get(name) != value:
                raise ValueError(f"v11 V8 control replay differs: {name}")
        checks = _checks(candidate, control)
        comparisons[str(candidate["seed"])] = {
            "candidate": candidate,
            "v8_joint_control": control,
            "checks": checks,
            "passed": all(item["passed"] for item in checks.values()),
        }
    passed = all(item["passed"] for item in comparisons.values())
    result = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "passed" if passed else "failed",
        "authorized_full_v11": passed,
        "required_seeds": list(PROBE_SEEDS),
        "source_commit": git["git_commit"],
        "test_accessed": False,
        "capacity": {
            "v11_gradient_reachable_state_parameters": reports[0][
                "total_state_parameter_count"
            ],
            "v8_joint_gradient_reachable_state_parameters": (
                V8_JOINT_REACHABLE_STATE_PARAMETERS
            ),
            "matched_within_5_percent": capacity_matched,
        },
        "comparisons": comparisons,
    }
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "probe_comparison.json", result)
    print(json.dumps({
        "status": result["status"],
        "authorized_full_v11": result["authorized_full_v11"],
    }, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
