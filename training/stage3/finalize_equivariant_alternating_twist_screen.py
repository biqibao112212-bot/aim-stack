"""Independently aggregate two fixed V13 typed-alternating screens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .equivariant_alternating_twist_future import (
    AnonymousEquivariantAlternatingTwistProbe,
)
from .train_causal_physical_ab import _git_state
from .train_equivariant_alternating_twist_screen import (
    RESULT_SCHEMA,
    RUN_SCHEMA,
    _validate_completed_alternating_screen,
    _validate_args,
)
from .train_joint_rigid_flow_probe import _state_parameter_count
from .train_paired_twist_set_probe import (
    PROBE_SEEDS,
    V8_JOINT_REACHABLE_STATE_PARAMETERS,
)
from .train_pnp_window_mapper_distillation import _atomic_json


AGGREGATE_SCHEMA = "stage3-equivariant-alternating-twist-screen-aggregate-v1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="aggregate two V13 typed-alternating screens",
    )
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.run) != 2:
        raise ValueError("alternating aggregate requires exactly two runs")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"alternating aggregate output exists: {output}")
    git = _git_state()
    if (
        git.get("worktree_dirty") is not False
        or git.get("git_commit") in {None, "unknown"}
    ):
        raise RuntimeError("alternating aggregate requires clean checkout")
    reports = []
    manifests = []
    for run in args.run:
        root = Path(run).resolve()
        manifest = json.loads(
            (root / "run_manifest.json").read_text(encoding="utf-8")
        )
        recorded = json.loads(
            (root / "screen_result.json").read_text(encoding="utf-8")
        )
        if (
            manifest.get("schema_version") != RUN_SCHEMA
            or recorded.get("schema_version") != RESULT_SCHEMA
        ):
            raise ValueError("alternating run/result schema differs")
        contract_args = dict(manifest["contract"]["args"])
        contract_args.setdefault("stop_after_update", 0)
        namespace = argparse.Namespace(**contract_args)
        _validate_args(namespace)
        parameter_count = _state_parameter_count(
            AnonymousEquivariantAlternatingTwistProbe, namespace,
        )
        checkpoint = root / "checkpoints" / "checkpoint-update-000100.pt"
        validated = _validate_completed_alternating_screen(
            namespace, checkpoint, parameter_count,
        )
        if validated != recorded:
            raise ValueError("alternating recorded result differs")
        reports.append(validated)
        manifests.append(manifest)
    if {item["seed"] for item in reports} != set(PROBE_SEEDS):
        raise ValueError("alternating aggregate seed set differs")
    if len({item["total_state_parameter_count"] for item in reports}) != 1:
        raise ValueError("alternating aggregate capacity differs")
    base_args = dict(manifests[0]["contract"]["args"])
    other_args = dict(manifests[1]["contract"]["args"])
    for name in ("seed", "output", "v8_joint_control_checkpoint"):
        base_args.pop(name, None)
        other_args.pop(name, None)
    if base_args != other_args:
        raise ValueError("alternating run arguments differ")
    parameter_count = reports[0]["total_state_parameter_count"]
    if parameter_count > int(1.05 * V8_JOINT_REACHABLE_STATE_PARAMETERS):
        raise ValueError("alternating aggregate exceeds capacity ceiling")
    passed = all(item["passed"] for item in reports)
    result = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "passed" if passed else "failed",
        "authorized_v13_state_continuation": passed,
        "future_position_decoder_status": "deferred_and_frozen",
        "source_commit": git.get("git_commit"),
        "test_accessed": False,
        "capacity": {
            "v13_gradient_reachable_state_parameters": parameter_count,
            "v8_joint_gradient_reachable_state_parameters": (
                V8_JOINT_REACHABLE_STATE_PARAMETERS
            ),
            "within_ceiling": True,
        },
        "comparisons": {
            str(item["seed"]): {
                "passed": item["passed"],
                "checks": item["checks"],
                "checkpoint_sha256": item["checkpoint_sha256"],
            }
            for item in reports
        },
    }
    output.mkdir(parents=True)
    _atomic_json(output / "screen_comparison.json", result)
    print(json.dumps({
        "status": result["status"],
        "authorized_v13_state_continuation": passed,
    }, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
