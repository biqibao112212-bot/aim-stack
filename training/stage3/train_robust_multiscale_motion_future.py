"""Train the v6 robust multi-scale physical-state bottleneck predictor."""

from __future__ import annotations

import json
import hashlib
import inspect
from pathlib import Path
import textwrap

import torch

from .robust_multiscale_motion_future import (
    RobustMultiScaleMotionBottleneckFutureModel,
    robust_multiscale_motion_future_loss,
    robust_multiscale_motion_state_loss,
)
from .observable_future_pnp_ab import sha256_file, state_dict_sha256
from .train_anonymous_vehicle_motion import _json_sha256
from .train_stable_motion_bottleneck_future import build_parser, train
from .train_stable_motion_bottleneck_future import motion_state_cells
from .train_increment_invariant_anonymous_future import (
    HierarchicalSessionHistorySampler,
    _history_label,
    apply_bin_preserving_prefix_dropout,
)
from .train_pnp_window_mapper_distillation import _atomic_json


RUN_SCHEMA = "stage3-robust-multiscale-motion-bottleneck-oracle-pilot-v6"
CONTROL_CHECKPOINT_SHA256 = (
    "f823f147183ea9ac1f6dc5e849008d3e3498f131b49b7169839be14a91b99662"
)
CONTROL_CONTRACT_SHA256 = (
    "c36717dfff88ba7a4d38b9a5fa578495552035dc5d14b28bb8a69609fec53871"
)
CONTROL_FIELDS = (
    "diagnostic_oracle_association", "allow_mapper_h_mismatch", "dataset",
    "truth_history", "mapper_checkpoint", "s_checkpoint", "h_checkpoint",
    "device", "seed", "batch_size", "validation_batch_size", "channels",
    "dropout", "message_layers", "basis_count", "motion_state_updates",
    "motion_state_learning_rate", "weight_decay", "gradient_clip_norm",
    "prefix_dropout_probability", "train_limit_per_class",
    "validation_limit_per_class",
)
FROZEN_FUTURE_MODULES = (
    "motion_state_encoder", "handle_encoder", "time_basis",
    "trajectory_coefficient_head", "role_coefficient_head",
)
CONTROL_SAMPLER_COMMIT = "39a23282160f158f6dfd3278aeb8c0d5e60b14fb"
CONTROL_SAMPLER_SOURCE_SHA256 = {
    "motion_state_cells": "5cdce7b2f5eaba4a14d9214c6f37ba742c928c17c64309ccc5b53c2358f4c2cc",
    "HierarchicalSessionHistorySampler": "165fd703087e823b73d51e08a38e3d185dd7270218d664ce0144bdc7e5d37f26",
    "apply_bin_preserving_prefix_dropout": "96457e8abf80e1f4b8713c26b336e40ff018c06985df7159cc85fe38dc0201ac",
    "_history_label": "a10497b1d13a13a500b5d349c43f5b7b890bcf668387c4bbabfda00db7aa88e9",
}


def _mean(group: dict, metric: str) -> float:
    return float(group[metric]["mean_m"])


def _semantic_source_sha256(value) -> str:
    source = textwrap.dedent(inspect.getsource(value)).strip() + "\n"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _sampler_source_hashes() -> dict[str, str]:
    return {
        "motion_state_cells": _semantic_source_sha256(motion_state_cells),
        "HierarchicalSessionHistorySampler": _semantic_source_sha256(
            HierarchicalSessionHistorySampler
        ),
        "apply_bin_preserving_prefix_dropout": _semantic_source_sha256(
            apply_bin_preserving_prefix_dropout
        ),
        "_history_label": _semantic_source_sha256(_history_label),
    }


def _preflight_control(control_checkpoint: Path) -> tuple[dict, dict]:
    actual_sampler_source = _sampler_source_hashes()
    if actual_sampler_source != CONTROL_SAMPLER_SOURCE_SHA256:
        raise ValueError(
            "state-gate sampler semantics differ from v77 control commit "
            + CONTROL_SAMPLER_COMMIT
        )
    if sha256_file(control_checkpoint) != CONTROL_CHECKPOINT_SHA256:
        raise ValueError("v77 update-800 control checkpoint hash differs")
    control = torch.load(control_checkpoint, map_location="cpu", weights_only=False)
    if control.get("contract_sha256") != CONTROL_CONTRACT_SHA256:
        raise ValueError("v77 update-800 control contract differs")
    if control.get("progress", {}).get("global_update") != 800:
        raise ValueError("v77 control is not the update-800 state endpoint")
    control_manifest_path = control_checkpoint.parent.parent / "run_manifest.json"
    control_manifest = json.loads(control_manifest_path.read_text(encoding="utf-8"))
    if control_manifest.get("contract_sha256") != CONTROL_CONTRACT_SHA256:
        raise ValueError("v77 control manifest contract reference differs")
    if _json_sha256(control_manifest.get("contract")) != CONTROL_CONTRACT_SHA256:
        raise ValueError("v77 control manifest contract payload differs")
    if control_manifest.get("run_id") != control.get("run_id"):
        raise ValueError("v77 control checkpoint/manifest run ID differs")
    for name in (
        "dataset", "truth_history", "frozen_initial_state_dict_sha256", "sampler",
    ):
        if control_manifest["provenance"][name] != control["provenance"][name]:
            raise ValueError(f"v77 control checkpoint/manifest {name} differs")
    return control, control_manifest


def _finalize_state_gate(
    output: Path, control_checkpoint: Path,
) -> dict:
    control, control_manifest = _preflight_control(control_checkpoint)
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or not manifest.get("state_gate_only"):
        raise ValueError("v6 state gate did not finish as a complete state-only run")
    if manifest.get("state_gate_future_modules_unchanged") is not True:
        raise ValueError("v6 state gate changed a frozen future module")
    control_args = control_manifest["contract"]["args"]
    actual_args = manifest["contract"]["args"]
    mismatch = {
        name: {"control": control_args.get(name), "actual": actual_args.get(name)}
        for name in CONTROL_FIELDS
        if control_args.get(name) != actual_args.get(name)
    }
    if mismatch:
        raise ValueError(f"v6 state gate differs from v77 control fields: {mismatch}")
    control_provenance = control["provenance"]
    actual_provenance = manifest["provenance"]
    lineage_checks = {
        "dataset_manifest_sha256": (
            control_provenance["dataset"]["manifest_sha256"],
            actual_provenance["dataset"]["manifest_sha256"],
        ),
        "truth_manifest_sha256": (
            control_provenance["truth_history"]["manifest_sha256"],
            actual_provenance["truth_history"]["manifest_sha256"],
        ),
        "frozen_upstream": (
            control_provenance["frozen_initial_state_dict_sha256"],
            actual_provenance["frozen_initial_state_dict_sha256"],
        ),
        "sampler": (
            control_provenance["sampler"], actual_provenance["sampler"],
        ),
    }
    if any(control_value != actual_value for control_value, actual_value in lineage_checks.values()):
        raise ValueError("v6 state gate lineage differs from the v77 control")
    control_future_hashes = {
        name: state_dict_sha256({
            key[len(name) + 1:]: value
            for key, value in control["model"].items()
            if key.startswith(name + ".")
        })
        for name in FROZEN_FUTURE_MODULES
    }
    actual_future_hashes = {
        name: manifest["trainable_final_state_dict_sha256"][name]
        for name in FROZEN_FUTURE_MODULES
    }
    if control_future_hashes != actual_future_hashes:
        raise ValueError("v6 state gate future modules differ from v77 update 800")

    state = manifest["final_validation"]["motion_state"]
    overall = state["overall"]
    combined = state["combined"]
    high_speed = state["combined_speed_gt_1_7"]
    high_speed_count = int(high_speed.get("sample_count", 0))
    if high_speed_count < 1:
        raise ValueError("v6 state gate has no combined speed>1.7 validation support")
    combined_11 = state["per_session"][
        "stage3-multistate-fixed6mm-20260730-v2-combined-11"
    ]
    checks = {
        "overall_velocity_mean_le_0_35_mps": (
            _mean(overall, "velocity_vector_error_mps"), 0.35, "le",
        ),
        "overall_yaw_mean_le_1_50_rad_s": (
            _mean(overall, "yaw_absolute_error_rad_s"), 1.50, "le",
        ),
        "overall_normalized_mae_le_0_08": (
            _mean(overall, "normalized_state_mae"), 0.08, "le",
        ),
        "combined_velocity_mean_le_0_57_mps": (
            _mean(combined, "velocity_vector_error_mps"), 0.57, "le",
        ),
        "combined_yaw_mean_le_2_10_rad_s": (
            _mean(combined, "yaw_absolute_error_rad_s"), 2.10, "le",
        ),
        "combined_speed_gt_1_7_velocity_mean_le_0_80_mps": (
            _mean(high_speed, "velocity_vector_error_mps"), 0.80, "le",
        ),
        "combined_11_normalized_mae_le_0_12": (
            _mean(combined_11, "normalized_state_mae"), 0.12, "le",
        ),
        "overall_yaw_sign_accuracy_ge_0_963": (
            float(overall["yaw_sign_accuracy_abs_truth_gt_0_5"]), 0.963, "ge",
        ),
    }
    check_report = {
        name: {
            "actual": actual, "threshold": threshold, "comparison": comparison,
            "passed": actual <= threshold if comparison == "le" else actual >= threshold,
            "required": True,
        }
        for name, (actual, threshold, comparison) in checks.items()
    }
    gate = {
        "schema_version": "stage3-v6-state-gate-v1",
        "status": "passed" if all(
            item["passed"] for item in check_report.values() if item["required"]
        ) else "failed",
        "test_accessed": False,
        "control": {
            "checkpoint": str(control_checkpoint),
            "checkpoint_sha256": CONTROL_CHECKPOINT_SHA256,
            "contract_sha256": CONTROL_CONTRACT_SHA256,
            "fixed_state_updates": 800,
            "sampler_commit": CONTROL_SAMPLER_COMMIT,
            "sampler_semantic_source_sha256": CONTROL_SAMPLER_SOURCE_SHA256,
        },
        "lineage": {
            name: {"control": pair[0], "actual": pair[1], "matched": pair[0] == pair[1]}
            for name, pair in lineage_checks.items()
        },
        "frozen_future_modules": {
            "control": control_future_hashes,
            "actual": actual_future_hashes,
            "matched": True,
        },
        "checks": check_report,
        "combined_speed_gt_1_7_sample_count": high_speed_count,
        "motion_state_metrics": state,
    }
    _atomic_json(output / "state_gate.json", gate)
    manifest["training_status"] = "complete"
    manifest["status"] = f"state_gate_{gate['status']}"
    manifest["state_gate"] = gate
    _atomic_json(manifest_path, manifest)
    return gate


def build_state_gate_parser():
    parser = build_parser()
    parser.description = "v6 robust multi-scale motion-state bottleneck pilot"
    parser.add_argument("--v77-control-checkpoint", required=True)
    parser.set_defaults(
        motion_state_updates=800,
        trajectory_updates=0,
        selector_updates=0,
        decoder_joint_updates=0,
        stop_after_update=0,
    )
    return parser


def _validate_state_gate_args(args) -> None:
    if (
        args.motion_state_updates != 800
        or any(value != 0 for value in (
            args.trajectory_updates, args.selector_updates, args.decoder_joint_updates,
        ))
        or args.stop_after_update != 0
    ):
        raise ValueError("v6 formal entry is fixed to a complete 800-update state-only gate")


def main() -> None:
    args = build_state_gate_parser().parse_args()
    _validate_state_gate_args(args)
    output = Path(args.output).resolve()
    control_checkpoint = Path(args.v77_control_checkpoint).resolve()
    _preflight_control(control_checkpoint)
    checkpoint = train(
        args,
        model_class=RobustMultiScaleMotionBottleneckFutureModel,
        loss_function=robust_multiscale_motion_future_loss,
        state_loss_function=robust_multiscale_motion_state_loss,
        run_schema=RUN_SCHEMA,
        extra_source_paths={"trainer_v6": Path(__file__)},
        state_gate_only=True,
        frozen_initialization_checkpoint=control_checkpoint,
        frozen_initialization_modules=FROZEN_FUTURE_MODULES,
    )
    try:
        gate = _finalize_state_gate(output, control_checkpoint)
    except Exception as error:
        manifest_path = output / "run_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["training_status"] = manifest.get("status")
            manifest["status"] = "state_gate_invalid"
            manifest["state_gate_finalization_error"] = str(error)
            _atomic_json(manifest_path, manifest)
        raise
    print(json.dumps({"state_gate": gate["status"]}, sort_keys=True))
    print(checkpoint)
    if gate["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
