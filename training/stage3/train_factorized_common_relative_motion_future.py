"""Run the fixed v7 factorized 800-update motion-state gate."""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path

import numpy as np
import torch

from .factorized_common_relative_motion_future import (
    FactorizedCommonRelativeMotionStateV7,
    factorized_motion_future_loss,
    factorized_state_train_step,
)
from .observable_future_pnp_ab import sha256_file, state_dict_sha256
from .robust_multiscale_motion_future import __file__ as V6_PRIMITIVES_SOURCE
from .train_pnp_window_mapper_distillation import _atomic_json
from .train_robust_multiscale_motion_future import (
    CONTROL_CHECKPOINT_SHA256,
    CONTROL_CONTRACT_SHA256,
    CONTROL_FIELDS,
    CONTROL_SAMPLER_COMMIT,
    CONTROL_SAMPLER_SOURCE_SHA256,
    FROZEN_FUTURE_MODULES,
    _mean,
    _preflight_control,
)
from .train_stable_motion_bottleneck_future import (
    ALL_TRAINABLE_MODULES,
    _cuda_amp_dtype,
    _prepare_batch,
    build_parser,
    train,
)
from .train_stable_motion_bottleneck_future import _callable_contract
from .train_anonymous_vehicle_motion import _distribution, _json_sha256
from .train_causal_physical_ab import _git_state, _to_device


RUN_SCHEMA = "stage3-factorized-common-relative-motion-state-gate-v7"


@torch.no_grad()
def factorized_validation_interventions(
    model: FactorizedCommonRelativeMotionStateV7,
    validation_loader,
    mapper: torch.nn.Module,
    s_model: torch.nn.Module,
    h_model: torch.nn.Module,
    device: torch.device,
) -> dict:
    """Validation-only PnP->clean and predicted->truth-yaw interventions."""
    model.eval()
    names = ("overall", "rotation", "combined", "combined_speed_gt_1_7")
    storage = {
        name: {
            "pnp_velocity": [], "pnp_yaw": [],
            "clean_velocity": [], "clean_yaw": [],
            "truth_yaw_conditioned_velocity": [],
        }
        for name in names
    }
    for raw_cpu in validation_loader:
        raw = _to_device(raw_cpu, device)
        amp = (
            torch.autocast("cuda", dtype=_cuda_amp_dtype())
            if device.type == "cuda" else nullcontext()
        )
        with amp:
            batch = _prepare_batch(mapper, s_model, h_model, raw)
            state_input = {
                name: batch[name] for name in (
                    "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
                    "history_event_mask", "history_time_s", "history_switch_step",
                )
            }
            pnp_output = model.estimate_motion_state(**state_input)
            pnp_state = pnp_output["state"]["motion_state_physical"]
            history = pnp_output["history"]
            truth_conditioned = model.motion_state_head(
                history["translation_scale_latent"],
                history["rotation_scale_latent"],
                history["translation_scale_available"],
                history["rotation_scale_available"],
                history["translation_scale_reliability"],
                history["rotation_scale_reliability"],
                rotation_condition_override=(
                    batch["target_motion_state_normalized"][:, 3]
                ),
            )["motion_state_normalized"] * model.motion_state_scale.to(pnp_state.dtype)

            clean_active = raw["clean_s_event_mask"].to(torch.bool)
            clean_visible = (
                raw["clean_s_obs_mask"].to(torch.bool) & clean_active.unsqueeze(-1)
            )
            clean_primary = (
                raw["clean_s_primary_mask"].to(torch.bool)
                & clean_active.unsqueeze(-1)
            )
            last = model.context._last_active(clean_active)
            rows = torch.arange(last.shape[0], device=device)
            primary = clean_primary[rows, last].to(torch.long).argmax(dim=1)
            clean_obs = raw["clean_s_obs_m"]
            clean_current = clean_obs[rows, last, primary]
            clean_relative = torch.where(
                clean_visible.unsqueeze(-1),
                clean_obs - clean_current[:, None, None],
                torch.zeros_like(clean_obs),
            )
            clean_output = model.estimate_motion_state(
                history_obs_rel_m=clean_relative,
                history_obs_mask=clean_visible,
                history_primary_mask=clean_primary,
                history_event_mask=clean_active,
                history_time_s=raw["clean_s_event_time_s"],
                history_switch_step=raw["clean_s_switch_step"],
            )
            clean_state = clean_output["state"]["motion_state_physical"]
            truth = batch["target_motion_state_physical"]
        motion_class = batch["motion_class"].to(torch.long)
        group_mask = {
            "overall": torch.ones_like(motion_class, dtype=torch.bool),
            "rotation": motion_class == 2,
            "combined": motion_class == 3,
            "combined_speed_gt_1_7": (
                (motion_class == 3) & (truth[:, :2].norm(dim=-1) > 1.7)
            ),
        }
        values = {
            "pnp_velocity": (pnp_state[:, :3] - truth[:, :3]).norm(dim=-1),
            "pnp_yaw": (pnp_state[:, 3] - truth[:, 3]).abs(),
            "clean_velocity": (clean_state[:, :3] - truth[:, :3]).norm(dim=-1),
            "clean_yaw": (clean_state[:, 3] - truth[:, 3]).abs(),
            "truth_yaw_conditioned_velocity": (
                truth_conditioned[:, :3] - truth[:, :3]
            ).norm(dim=-1),
        }
        for group, mask in group_mask.items():
            for metric, value in values.items():
                storage[group][metric].append(value[mask].detach().cpu().numpy())
    groups = {
        group: {
            metric: _distribution(np.concatenate(values).astype(np.float64))
            for metric, values in metrics.items()
        }
        for group, metrics in storage.items()
    }
    return {
        "schema_version": "stage3-v7-factorized-validation-interventions-v1",
        "validation_only": True,
        "test_accessed": False,
        "validation_sample_count": len(validation_loader.dataset),
        "validation_audit": validation_loader.dataset.audit,
        "pnp_to_clean_semantics": (
            "same clean anonymous S observations; no mapper, future or truth state input"
        ),
        "truth_yaw_intervention_semantics": (
            "truth normalized yaw replaces detached predicted-yaw planar condition only"
        ),
        "groups": groups,
    }


def _validated_error_distribution(
    value: object, *, name: str, require_nonempty: bool = True,
) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} distribution is missing")
    count = value.get("count")
    if not isinstance(count, int) or count < (1 if require_nonempty else 0):
        raise ValueError(f"{name} distribution count is invalid")
    if count == 0:
        return value
    for field in ("mean_m", "p50_m", "p95_m", "p99_m"):
        metric = value.get(field)
        if (
            isinstance(metric, bool) or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric)) or float(metric) < 0.0
        ):
            raise ValueError(f"{name}/{field} is not a finite nonnegative error")
    return value


def _validated_accuracy(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} is not a finite probability")
    return float(value)


def _load_bound_candidate(output: Path) -> tuple[dict, dict]:
    """Load candidate evidence from its final checkpoint, never JSON alone."""
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != RUN_SCHEMA:
        raise ValueError("v7 candidate manifest schema differs")
    contract = manifest.get("contract")
    contract_sha = manifest.get("contract_sha256")
    if not isinstance(contract, dict) or contract_sha != _json_sha256(contract):
        raise ValueError("v7 candidate manifest contract hash differs")
    fixed = manifest.get("fixed_final_checkpoint")
    if (
        not isinstance(fixed, dict) or fixed.get("update") != 800
        or fixed.get("selected_by_validation") is not False
    ):
        raise ValueError("v7 candidate fixed checkpoint record differs")
    checkpoint = Path(str(fixed.get("path", ""))).resolve()
    expected_dir = (output / "checkpoints").resolve()
    if (
        checkpoint.parent != expected_dir
        or checkpoint.name != "checkpoint-update-000800.pt"
        or not checkpoint.is_file()
    ):
        raise ValueError("v7 candidate checkpoint path escapes its fixed output")
    if fixed.get("sha256") != sha256_file(checkpoint):
        raise ValueError("v7 candidate checkpoint file hash differs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != RUN_SCHEMA
        or payload.get("contract_sha256") != contract_sha
        or payload.get("run_id") != manifest.get("run_id")
        or payload.get("model_class") != FactorizedCommonRelativeMotionStateV7.__name__
        or payload.get("model_config") != manifest.get("model_config")
        or payload.get("fixed_endpoint") is not True
        or payload.get("stage_endpoint") is not True
        or payload.get("checkpoint_role") != "fixed_final_endpoint"
        or payload.get("checkpoint_selected_by_validation") is not False
        or payload.get("progress") != {
            "global_update": 800, "stage": "motion_state", "stage_update": 800,
        }
    ):
        raise ValueError("v7 candidate checkpoint identity or endpoint differs")
    model_state = payload.get("model")
    if (
        not isinstance(model_state, dict)
        or payload.get("model_state_dict_sha256") != state_dict_sha256(model_state)
    ):
        raise ValueError("v7 candidate model state hash differs")
    config = payload["model_config"]
    scale = config.get("motion_state_scale")
    motion_context = config.get("motion_context")
    if (
        not isinstance(scale, list) or len(scale) != 4
        or not isinstance(motion_context, dict)
        or not isinstance(motion_context.get("lag_scales_s"), list)
    ):
        raise ValueError("v7 candidate model config is incomplete")
    try:
        bound_model = FactorizedCommonRelativeMotionStateV7(
            velocity_scale_mps=tuple(float(value) for value in scale[:3]),
            yaw_rate_scale_rad_s=float(scale[3]),
            channels=int(config["channels"]),
            dropout=float(config["dropout"]),
            message_layers=int(config["message_layers"]),
            trained_horizon_s=float(config["trained_horizon_s"]),
            maximum_absolute_step=int(config["maximum_absolute_step"]),
            position_scale_m=float(config["position_scale_m"]),
            history_scale_s=float(config["history_scale_s"]),
            residual_scale_m=float(config["residual_scale_m"]),
            basis_count=int(config["basis_count"]),
            lag_scales_s=tuple(float(value) for value in motion_context["lag_scales_s"]),
        )
        if _json_sha256(bound_model.config) != _json_sha256(config):
            raise ValueError("reconstructed config differs")
        bound_model.load_state_dict(model_state, strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("v7 candidate model/config cannot be reconstructed exactly") from error
    if _json_sha256(payload.get("provenance")) != _json_sha256(
        manifest.get("provenance")
    ):
        raise ValueError("v7 candidate checkpoint/manifest provenance differs")
    if _json_sha256(payload.get("validation_history")) != _json_sha256(
        manifest.get("validation_history")
    ):
        raise ValueError("v7 candidate checkpoint/manifest validation history differs")
    history = payload.get("validation_history")
    if (
        not isinstance(history, list) or not history
        or _json_sha256(history[-1].get("metrics"))
        != _json_sha256(manifest.get("final_validation"))
    ):
        raise ValueError("v7 candidate final validation is not checkpoint-bound")
    for name in (
        "state_substage_counts", "state_substage_transitions",
        "state_branch_hash_history", "gradient_isolation_verified",
        "final_diagnostics",
    ):
        if _json_sha256(payload.get(name)) != _json_sha256(manifest.get(name)):
            raise ValueError(f"v7 candidate checkpoint/manifest {name} differs")
    actual_module_hashes = {
        name: state_dict_sha256(getattr(bound_model, name).state_dict())
        for name in ALL_TRAINABLE_MODULES
    }
    if actual_module_hashes != manifest.get("trainable_final_state_dict_sha256"):
        raise ValueError("v7 candidate final module hashes differ")
    progress = manifest.get("progress", {})
    if (
        progress.get("global_update") != 800
        or Path(str(progress.get("latest_checkpoint", ""))).resolve() != checkpoint
    ):
        raise ValueError("v7 candidate manifest progress differs")
    return manifest, payload


def _finalize_state_gate(output: Path, control_checkpoint: Path) -> dict:
    control, control_manifest = _preflight_control(control_checkpoint)
    manifest_path = output / "run_manifest.json"
    manifest, candidate_checkpoint = _load_bound_candidate(output)
    if manifest.get("status") != "complete" or not manifest.get("state_gate_only"):
        raise ValueError("v7 state gate did not finish as a complete state-only run")
    if manifest.get("state_gate_future_modules_unchanged") is not True:
        raise ValueError("v7 state gate changed a frozen future module")
    if manifest.get("model_config", {}).get("state_substage_schedule") != {
        "angular_specialization": [1, 250],
        "translation_specialization": [251, 600],
        "joint_calibration": [601, 800],
    }:
        raise ValueError("v7 internal state schedule differs")
    expected_step = _callable_contract(factorized_state_train_step)
    if (
        manifest.get("contract", {}).get("state_step_function") != expected_step
        or manifest.get("provenance", {}).get("state_step_function") != expected_step
    ):
        raise ValueError("v7 state-step function is not semantic-source bound")
    expected_diagnostic = _callable_contract(factorized_validation_interventions)
    if (
        manifest.get("contract", {}).get("final_diagnostic_function")
        != expected_diagnostic
        or manifest.get("provenance", {}).get("final_diagnostic_function")
        != expected_diagnostic
    ):
        raise ValueError("v7 validation intervention is not semantic-source bound")
    if manifest.get("state_substage_counts") != {
        "angular_specialization": 250,
        "translation_specialization": 350,
        "joint_calibration": 200,
    }:
        raise ValueError("v7 actual state-substage counts differ")
    if manifest.get("state_substage_transitions") != [
        {"global_update": 1, "substage": "angular_specialization"},
        {"global_update": 251, "substage": "translation_specialization"},
        {"global_update": 601, "substage": "joint_calibration"},
    ]:
        raise ValueError("v7 actual state-substage transitions differ")
    branch_history = manifest.get("state_branch_hash_history")
    if not isinstance(branch_history, list) or [
        item.get("global_update") for item in branch_history
    ] != [0, 250, 600, 800]:
        raise ValueError("v7 branch hash boundary history differs")
    if (
        branch_history[0]["hashes"]["translation_vertical"]
        != branch_history[1]["hashes"]["translation_vertical"]
    ):
        raise ValueError("v7 translation branch changed during angular specialization")
    if (
        branch_history[1]["hashes"]["angular"]
        != branch_history[2]["hashes"]["angular"]
    ):
        raise ValueError("v7 angular branch changed while frozen")
    git = manifest.get("provenance", {}).get("git", {})
    if git.get("worktree_dirty") is not False or git.get("git_commit") in {None, "unknown"}:
        raise ValueError("v7 formal run did not start from a clean source commit")
    current_git = _git_state()
    if (
        current_git.get("worktree_dirty") is not False
        or current_git.get("git_commit") != git.get("git_commit")
    ):
        raise ValueError("v7 source checkout changed during the formal run")
    interventions = manifest.get("final_diagnostics")
    if (
        not isinstance(interventions, dict)
        or interventions.get("schema_version")
        != "stage3-v7-factorized-validation-interventions-v1"
        or interventions.get("validation_only") is not True
        or interventions.get("test_accessed") is not False
        or interventions.get("validation_audit")
        != manifest["provenance"]["dataset"]["validation"]
    ):
        raise ValueError("v7 validation interventions are missing or unbound")
    required_intervention_metrics = {
        "pnp_velocity", "pnp_yaw", "clean_velocity", "clean_yaw",
        "truth_yaw_conditioned_velocity",
    }
    intervention_groups = interventions.get("groups")
    if not isinstance(intervention_groups, dict) or set(intervention_groups) != {
        "overall", "rotation", "combined", "combined_speed_gt_1_7",
    }:
        raise ValueError("v7 validation intervention groups differ")

    control_args = control_manifest["contract"]["args"]
    actual_args = manifest["contract"]["args"]
    mismatch = {
        name: {"control": control_args.get(name), "actual": actual_args.get(name)}
        for name in CONTROL_FIELDS
        if control_args.get(name) != actual_args.get(name)
    }
    if mismatch:
        raise ValueError(f"v7 state gate differs from v77 control fields: {mismatch}")
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
    if any(a != b for a, b in lineage_checks.values()):
        raise ValueError("v7 state gate lineage differs from the v77 control")
    control_future_hashes = {
        name: state_dict_sha256({
            key[len(name) + 1:]: value
            for key, value in control["model"].items()
            if key.startswith(name + ".")
        })
        for name in FROZEN_FUTURE_MODULES
    }
    candidate_model = candidate_checkpoint["model"]
    actual_future_hashes = {
        name: state_dict_sha256({
            key[len(name) + 1:]: value
            for key, value in candidate_model.items() if key.startswith(name + ".")
        })
        for name in FROZEN_FUTURE_MODULES
    }
    if control_future_hashes != actual_future_hashes:
        raise ValueError("v7 state gate future modules differ from v77 update 800")

    state = manifest["final_validation"]["motion_state"]
    overall = state["overall"]
    combined = state["combined"]
    high_speed = state["combined_speed_gt_1_7"]
    high_speed_count = int(high_speed.get("sample_count", 0))
    if high_speed_count < 1:
        raise ValueError("v7 gate has no combined speed>1.7 validation support")
    combined_11 = state["per_session"][
        "stage3-multistate-fixed6mm-20260730-v2-combined-11"
    ]
    for group_name, group in (
        ("overall", overall), ("combined", combined),
        ("combined_speed_gt_1_7", high_speed), ("combined_11", combined_11),
    ):
        for metric_name in (
            "velocity_vector_error_mps", "yaw_absolute_error_rad_s",
            "normalized_state_mae",
        ):
            distribution = _validated_error_distribution(
                group.get(metric_name), name=f"{group_name}/{metric_name}",
            )
            if int(distribution["count"]) != int(group.get("sample_count", distribution["count"])):
                raise ValueError(f"{group_name} metric/sample count differs")
    overall_sign = _validated_accuracy(
        overall.get("yaw_sign_accuracy_abs_truth_gt_0_5"),
        name="overall yaw-sign accuracy",
    )
    high_speed_sign = _validated_accuracy(
        high_speed.get("yaw_sign_accuracy_abs_truth_gt_0_5"),
        name="high-speed combined yaw-sign accuracy",
    )
    intervention_count = interventions.get("validation_sample_count")
    if not isinstance(intervention_count, int) or intervention_count < 1:
        raise ValueError("v7 validation intervention total support is invalid")
    group_counts: dict[str, int] = {}
    for group_name, group in intervention_groups.items():
        if set(group) != required_intervention_metrics:
            raise ValueError(f"v7 intervention metrics differ for {group_name}")
        counts = set()
        for metric_name in required_intervention_metrics:
            distribution = _validated_error_distribution(
                group[metric_name],
                name=f"intervention/{group_name}/{metric_name}",
            )
            counts.add(int(distribution["count"]))
        if len(counts) != 1:
            raise ValueError(f"v7 intervention support differs for {group_name}")
        group_counts[group_name] = counts.pop()
        if _json_sha256(group["pnp_velocity"]) != _json_sha256(
            state[group_name]["velocity_vector_error_mps"]
        ) or _json_sha256(group["pnp_yaw"]) != _json_sha256(
            state[group_name]["yaw_absolute_error_rad_s"]
        ):
            raise ValueError(f"v7 intervention PnP baseline differs for {group_name}")
    if (
        group_counts["overall"] != intervention_count
        or group_counts["rotation"] + group_counts["combined"] != intervention_count
        or not 0 < group_counts["combined_speed_gt_1_7"] <= group_counts["combined"]
    ):
        raise ValueError("v7 validation intervention group counts differ")
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
            overall_sign, 0.963, "ge",
        ),
    }
    report = {
        name: {
            "actual": actual, "threshold": threshold,
            "comparison": comparison,
            "passed": actual <= threshold if comparison == "le" else actual >= threshold,
            "required": True,
        }
        for name, (actual, threshold, comparison) in checks.items()
    }
    diagnostics = {
        "combined_11_velocity_mean_mps": _mean(
            combined_11, "velocity_vector_error_mps",
        ),
        "combined_11_yaw_mean_rad_s": _mean(
            combined_11, "yaw_absolute_error_rad_s",
        ),
        "combined_speed_gt_1_7_yaw_mean_rad_s": _mean(
            high_speed, "yaw_absolute_error_rad_s",
        ),
        "combined_speed_gt_1_7_yaw_sign_accuracy": float(
            high_speed_sign
        ),
        "validation_interventions": interventions,
    }
    gate = {
        "schema_version": "stage3-v7-factorized-state-gate-v1",
        "status": "passed" if all(item["passed"] for item in report.values()) else "failed",
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
        "checks": report,
        "diagnostics": diagnostics,
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
    parser.description = "v7 factorized common/relative motion-state gate"
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
        raise ValueError("v7 formal entry is fixed to a complete 800-update state-only gate")


def main() -> None:
    args = build_state_gate_parser().parse_args()
    _validate_state_gate_args(args)
    output = Path(args.output).resolve()
    control_checkpoint = Path(args.v77_control_checkpoint).resolve()
    git = _git_state()
    if git.get("worktree_dirty") is not False or git.get("git_commit") in {None, "unknown"}:
        raise RuntimeError("v7 formal state gate requires a clean source commit")
    _preflight_control(control_checkpoint)
    checkpoint = train(
        args,
        model_class=FactorizedCommonRelativeMotionStateV7,
        loss_function=factorized_motion_future_loss,
        state_step_function=factorized_state_train_step,
        final_diagnostic_function=factorized_validation_interventions,
        run_schema=RUN_SCHEMA,
        extra_source_paths={
            "trainer_v7": Path(__file__),
            "model_v6_primitives": Path(V6_PRIMITIVES_SOURCE),
        },
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
