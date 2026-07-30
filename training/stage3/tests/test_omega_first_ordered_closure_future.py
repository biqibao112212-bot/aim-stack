from __future__ import annotations

from copy import deepcopy
import inspect

import pytest
import torch

from training.stage3.factorized_common_relative_motion_future import (
    apply_common_velocity_ramp,
)
from training.stage3.finalize_omega_first_ordered_closure_probe import (
    BASE_GROUP_DISTRIBUTIONS,
    PAIR_GROUP_DISTRIBUTIONS,
    _checks,
)
from training.stage3.omega_first_ordered_closure_future import (
    AnonymousOmegaFirstOrderedClosureProbe,
    _MaskedOrderedGRU,
    omega_first_ordered_state_loss,
    omega_first_ordered_train_step,
)
from training.stage3.tests.test_anonymous_vehicle_motion import _batch
from training.stage3.tests.test_stable_motion_bottleneck_future import (
    _supervised_batch,
)
from training.stage3.train_paired_twist_set_probe import (
    V8_JOINT_REACHABLE_STATE_PARAMETERS,
)
from training.stage3.train_omega_first_ordered_closure_probe import (
    DIAGNOSTIC_SCHEMA,
    _cross_sample_geometry_derangement,
    _reflect_relative_history_with_truth_common,
    _synthetic_twist_history_on_target_support,
)
from training.stage3.train_stable_motion_bottleneck_future import STATE_MODULES


def _model(*, channels: int = 32, dropout: float = 0.0):
    torch.manual_seed(20260731)
    return AnonymousOmegaFirstOrderedClosureProbe(
        velocity_scale_mps=(3.0, 3.2, 0.25),
        yaw_rate_scale_rad_s=17.25,
        channels=channels, dropout=dropout,
        message_layers=2 if channels == 32 else 3,
        basis_count=6 if channels == 32 else 8,
    )


def _fields(batch):
    return {
        name: batch[name] for name in (
            "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
            "history_event_mask", "history_time_s", "history_switch_step",
        )
    }


def test_omega_first_keeps_six_fields_and_strict_state_direction() -> None:
    expected = {
        "self", "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
        "history_event_mask", "history_time_s", "history_switch_step",
    }
    assert set(inspect.signature(
        AnonymousOmegaFirstOrderedClosureProbe.estimate_motion_state,
    ).parameters) == expected
    config = _model().config
    assert config["event_order_preserved"] is True
    assert config["shared_four_dimensional_update"] is False
    assert config["velocity_to_omega_path"] is False
    assert config["omega_to_velocity_path"] is True
    assert config["residual_encoder_reads_prediction_separately"] is False
    assert config["pair_writes_velocity_directly"] is False
    assert config["analytic_future_decoder"] is False


def test_ordered_gru_is_causal_and_not_set_invariant() -> None:
    torch.manual_seed(4)
    sequence = _MaskedOrderedGRU(8).eval()
    event = torch.randn(2, 5, 8)
    valid = torch.tensor([
        [False, True, True, True, True],
        [True, True, False, True, False],
    ])
    reference = sequence(event, valid)
    changed_inactive = event.clone()
    changed_inactive[~valid] = 1000.0
    torch.testing.assert_close(sequence(changed_inactive, valid), reference)
    reordered = event.clone()
    reordered[:, [1, 2]] = reordered[:, [2, 1]]
    assert torch.count_nonzero(sequence(reordered, valid) - reference) > 0


@torch.inference_mode()
def test_common_velocity_ramp_cannot_enter_omega_channel() -> None:
    model = _model().eval()
    batch = _supervised_batch()
    reference = model.estimate_motion_state(**_fields(batch))["state"]
    ramp = torch.tensor([[0.5, -0.2, 0.0], [-0.3, 0.4, 0.0]])
    augmented = apply_common_velocity_ramp(
        batch, ramp, model.motion_state_scale,
    )
    changed = model.estimate_motion_state(**_fields(augmented))["state"]
    torch.testing.assert_close(
        changed["angular_iteration_normalized"],
        reference["angular_iteration_normalized"], rtol=2e-5, atol=5e-6,
    )


@torch.inference_mode()
def test_angular_and_velocity_refinements_have_disjoint_write_sets() -> None:
    model = _model().eval()
    fields = _fields(_batch())
    final = model.estimate_motion_state(**fields)["state"]
    no_angular = model.estimate_motion_state_zero_angular_refinement(
        **fields
    )["state"]
    no_velocity = model.estimate_motion_state_zero_velocity_refinement(
        **fields
    )["state"]
    torch.testing.assert_close(
        no_velocity["motion_state_normalized"][:, 3],
        final["motion_state_normalized"][:, 3], rtol=0, atol=0,
    )
    assert torch.count_nonzero(
        no_velocity["motion_state_normalized"][:, :3]
        - final["motion_state_normalized"][:, :3]
    ) > 0
    assert torch.count_nonzero(
        no_angular["motion_state_normalized"][:, 3]
        - final["motion_state_normalized"][:, 3]
    ) > 0


@torch.inference_mode()
def test_pair0_uses_no_constant_pair_branch_and_remains_finite() -> None:
    model = _model().eval()
    batch = deepcopy(_batch())
    primary = batch["history_primary_mask"].to(torch.bool)
    batch["history_obs_mask"] = primary
    batch["history_obs_rel_m"] = torch.where(
        primary.unsqueeze(-1), batch["history_obs_rel_m"],
        torch.zeros_like(batch["history_obs_rel_m"]),
    )
    output = model.estimate_motion_state(**_fields(batch))["state"]
    assert not bool(output["pair_supported"].any())
    assert torch.isfinite(output["motion_state_normalized"]).all()


def test_omega_first_loss_reads_state_and_history_but_not_future() -> None:
    model = _model().train()
    batch = _supervised_batch()
    output = model.estimate_motion_state(**_fields(batch))
    prediction = {**output["history"], **output["state"]}
    loss, components = omega_first_ordered_state_loss(prediction, batch)
    poisoned = dict(batch)
    poisoned["future_target_position_m"] = torch.full((2, 9, 3), torch.nan)
    poisoned["q0_relation_m"] = torch.full((2, 4, 3), torch.nan)
    actual, _ = omega_first_ordered_state_loss(prediction, poisoned)
    torch.testing.assert_close(actual, loss)
    for name in (
        "angular_handle_history_closure", "pair_history_closure",
        "common_handle_history_closure",
    ):
        assert torch.isfinite(components[name])


def test_omega_first_train_step_reaches_all_state_parameters() -> None:
    model = _model(dropout=0.05).train()
    batch = _supervised_batch()
    _, loss, components = omega_first_ordered_train_step(model, batch, 1, 200)
    assert torch.isfinite(loss)
    assert components["state_substage"] == (
        "omega_first_ordered_closure_structural_probe"
    )
    loss.backward()
    for name in STATE_MODULES:
        assert all(
            parameter.grad is not None
            for parameter in getattr(model, name).parameters()
            if parameter.requires_grad
        )


def test_omega_first_capacity_matches_v8_within_five_percent() -> None:
    model = _model(channels=96)
    count = sum(
        parameter.numel() for name in STATE_MODULES
        for parameter in getattr(model, name).parameters()
    )
    assert abs(count - V8_JOINT_REACHABLE_STATE_PARAMETERS) / (
        V8_JOINT_REACHABLE_STATE_PARAMETERS
    ) <= 0.05


@torch.inference_mode()
def test_common_residual_path_is_zero_preserving() -> None:
    head = _model().eval().motion_state_head
    residual = torch.zeros(2, 7, 3)
    time = torch.randn(2, 7, 8)
    assert torch.count_nonzero(head.common_encoder(residual, time)) == 0


@torch.inference_mode()
def test_complete_state_path_cannot_use_support_or_time_without_motion() -> None:
    model = _model().eval()
    history = model.context(**_fields(_batch()))
    handle_kinematics = history["_handle_kinematics_raw"].clone()
    pair_kinematics = history["_pair_kinematics_raw"].clone()
    handle_kinematics[..., :6] = 0
    pair_kinematics[..., :6] = 0
    state = model.motion_state_head(
        history["_handle_geometry_raw"], handle_kinematics,
        history["_handle_raw_valid"], history["_pair_geometry_raw"],
        pair_kinematics, history["_pair_raw_valid"],
    )
    assert torch.count_nonzero(state["motion_state_normalized"]) == 0
    assert torch.count_nonzero(state["angular_iteration_normalized"]) == 0
    assert torch.count_nonzero(state["velocity_iteration_normalized"]) == 0


def test_cross_sample_pair_break_changes_single_factor_without_changing_support() -> None:
    valid = torch.tensor([
        [True, False, False],
        [True, False, False],
        [True, False, False],
        [True, True, False],
    ])
    geometry = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
    history = {
        "_pair_geometry_raw": geometry,
        "_pair_raw_valid": valid,
    }
    broken = _cross_sample_geometry_derangement(history, handle=False)
    torch.testing.assert_close(broken["_pair_raw_valid"], valid)
    changed = (
        broken["_pair_geometry_raw"] != geometry
    ).any(dim=-1) & valid
    assert changed[:3].sum().item() == 3
    assert not bool(changed[3].any())


@torch.inference_mode()
def test_relative_reflection_is_involutive_on_decoder_visible_geometry() -> None:
    model = _model().eval()
    batch = _supervised_batch()
    history = model.context(**_fields(batch))
    velocity = (
        batch["target_motion_state_normalized"][:, :3]
        * model.motion_state_scale[:3]
    )
    reflected = _reflect_relative_history_with_truth_common(
        model, history, velocity,
        event_count=batch["history_event_mask"].shape[1],
    )
    restored = _reflect_relative_history_with_truth_common(
        model, reflected, velocity,
        event_count=batch["history_event_mask"].shape[1],
    )
    original_factors = model.motion_state_head._reshape_factors(
        history["_handle_geometry_raw"], history["_handle_kinematics_raw"],
        history["_handle_raw_valid"], history["_pair_geometry_raw"],
        history["_pair_kinematics_raw"], history["_pair_raw_valid"],
    )
    reflected_factors = model.motion_state_head._reshape_factors(
        reflected["_handle_geometry_raw"],
        reflected["_handle_kinematics_raw"],
        reflected["_handle_raw_valid"], reflected["_pair_geometry_raw"],
        reflected["_pair_kinematics_raw"], reflected["_pair_raw_valid"],
    )
    restored_factors = model.motion_state_head._reshape_factors(
        restored["_handle_geometry_raw"], restored["_handle_kinematics_raw"],
        restored["_handle_raw_valid"], restored["_pair_geometry_raw"],
        restored["_pair_kinematics_raw"], restored["_pair_raw_valid"],
    )
    original_derived = model.motion_state_head._derived_relative_factors(
        original_factors[0], original_factors[1], original_factors[2],
    )
    reflected_derived = model.motion_state_head._derived_relative_factors(
        reflected_factors[0], reflected_factors[1], reflected_factors[2],
    )
    restored_derived = model.motion_state_head._derived_relative_factors(
        restored_factors[0], restored_factors[1], restored_factors[2],
    )
    reflection = torch.tensor([1.0, -1.0, 1.0])
    valid = original_factors[2]
    torch.testing.assert_close(
        reflected_derived[0][valid],
        (original_derived[0].reshape(*original_derived[0].shape[:-1], 3, 3)
         * reflection).reshape_as(original_derived[0])[valid],
        rtol=2e-5, atol=2e-6,
    )
    torch.testing.assert_close(
        reflected_derived[1][valid],
        (original_derived[1] * reflection)[valid], rtol=2e-5, atol=2e-6,
    )
    expected_common = velocity * (
        model.context.history_scale_s / model.context.position_scale_m
    )
    torch.testing.assert_close(
        reflected_derived[2], expected_common, rtol=2e-5, atol=2e-6,
    )
    torch.testing.assert_close(
        restored_derived[0][valid], original_derived[0][valid],
        rtol=2e-5, atol=2e-6,
    )
    current_minus_prior = (
        reflected_factors[0][..., 6:9] - reflected_factors[0][..., 9:12]
    )
    elapsed = reflected_derived[3]
    torch.testing.assert_close(
        reflected_factors[1][..., :3][valid],
        current_minus_prior[valid], rtol=2e-5, atol=2e-6,
    )
    torch.testing.assert_close(
        reflected_factors[1][..., :3][valid],
        (reflected_factors[1][..., 3:6] * elapsed.unsqueeze(-1))[valid],
        rtol=2e-5, atol=2e-6,
    )
    torch.testing.assert_close(
        restored["_pair_geometry_raw"], history["_pair_geometry_raw"],
    )
    torch.testing.assert_close(
        reflected["_pair_raw_valid"], history["_pair_raw_valid"],
    )


@torch.inference_mode()
def test_factorial_twist_cells_keep_support_and_separate_common_from_relative() -> None:
    model = _model().eval()
    batch = _supervised_batch()
    history = model.context(**_fields(batch))
    base = batch["target_motion_state_normalized"] * model.motion_state_scale
    changed_velocity = base[:, :3] + torch.tensor([
        [0.8, -0.7, 0.0], [-0.6, 0.9, 0.0],
    ])
    aa = _synthetic_twist_history_on_target_support(
        model, history, base[:, :3], base[:, 3],
        event_count=batch["history_event_mask"].shape[1],
    )
    ba = _synthetic_twist_history_on_target_support(
        model, history, changed_velocity, base[:, 3],
        event_count=batch["history_event_mask"].shape[1],
    )
    changed_yaw = -base[:, 3] + torch.tensor([0.7, -0.9])
    ab = _synthetic_twist_history_on_target_support(
        model, history, base[:, :3], changed_yaw,
        event_count=batch["history_event_mask"].shape[1],
    )
    for valid_name in ("_handle_raw_valid", "_pair_raw_valid"):
        torch.testing.assert_close(aa[valid_name], history[valid_name])
        torch.testing.assert_close(ba[valid_name], history[valid_name])
    aa_factors = model.motion_state_head._reshape_factors(
        aa["_handle_geometry_raw"], aa["_handle_kinematics_raw"],
        aa["_handle_raw_valid"], aa["_pair_geometry_raw"],
        aa["_pair_kinematics_raw"], aa["_pair_raw_valid"],
    )
    ba_factors = model.motion_state_head._reshape_factors(
        ba["_handle_geometry_raw"], ba["_handle_kinematics_raw"],
        ba["_handle_raw_valid"], ba["_pair_geometry_raw"],
        ba["_pair_kinematics_raw"], ba["_pair_raw_valid"],
    )
    ab_factors = model.motion_state_head._reshape_factors(
        ab["_handle_geometry_raw"], ab["_handle_kinematics_raw"],
        ab["_handle_raw_valid"], ab["_pair_geometry_raw"],
        ab["_pair_kinematics_raw"], ab["_pair_raw_valid"],
    )
    aa_derived = model.motion_state_head._derived_relative_factors(
        aa_factors[0], aa_factors[1], aa_factors[2],
    )
    ba_derived = model.motion_state_head._derived_relative_factors(
        ba_factors[0], ba_factors[1], ba_factors[2],
    )
    ab_derived = model.motion_state_head._derived_relative_factors(
        ab_factors[0], ab_factors[1], ab_factors[2],
    )
    valid = aa_factors[2]
    torch.testing.assert_close(
        aa_derived[0][valid], ba_derived[0][valid], rtol=2e-5, atol=2e-6,
    )
    torch.testing.assert_close(
        aa_derived[1][valid], ba_derived[1][valid], rtol=2e-5, atol=2e-6,
    )
    scale = model.context.history_scale_s / model.context.position_scale_m
    torch.testing.assert_close(
        aa_derived[2], base[:, :3] * scale, rtol=2e-5, atol=2e-6,
    )
    torch.testing.assert_close(
        ba_derived[2], changed_velocity * scale, rtol=2e-5, atol=2e-6,
    )
    torch.testing.assert_close(
        aa_derived[0][..., 3:6][valid],
        ab_derived[0][..., 3:6][valid], rtol=2e-5, atol=2e-6,
    )
    torch.testing.assert_close(
        aa_factors[3][..., 3:6][aa_factors[5]],
        ab_factors[3][..., 3:6][ab_factors[5]], rtol=0, atol=0,
    )


@torch.inference_mode()
def test_fixed_state_decoder_never_reestimates_supplied_state() -> None:
    model = _model().eval()
    fields = _fields(_batch())
    history = model.context(**fields)
    state = model.estimate_motion_state(**fields)["state"][
        "motion_state_normalized"
    ].clone()
    decoded = model.motion_state_head.decode_closure_at_state(
        history["_handle_geometry_raw"], history["_handle_kinematics_raw"],
        history["_handle_raw_valid"], history["_pair_geometry_raw"],
        history["_pair_kinematics_raw"], history["_pair_raw_valid"], state,
    )
    assert torch.isfinite(decoded["angular_handle_residual"]).all()
    assert torch.isfinite(decoded["common_handle_residual"]).all()
    assert decoded["handle_decoder_geometry"].shape[-1] == 3
    torch.testing.assert_close(
        state,
        model.estimate_motion_state(**fields)["state"][
            "motion_state_normalized"
        ],
    )


def test_omega_first_gate_requires_performance_typed_closure_and_transfer() -> None:
    def distribution(mean, count=200, *, p50=None, p95=None, p99=None):
        p50 = mean if p50 is None else p50
        p95 = max(p50, mean) if p95 is None else p95
        p99 = max(p95, mean) if p99 is None else p99
        return {
            "count": count, "mean_m": mean,
            "p50_m": p50, "p95_m": p95, "p99_m": p99,
        }
    coverage = {
        "handle_rows_touched": 4, "handle_factors_touched": 8,
        "handle_valid_factors": 16, "pair_rows_touched": 4,
        "pair_factors_touched": 8, "pair_valid_factors": 16,
    }
    group_counts = {
        "overall": 750, "combined": 300, "combined_speed_gt_1_7": 82,
        "core": 149, "pair0": 127, "combined_pair1": 25,
        "combined_pair2": 23, "combined_pair3": 229,
    }
    groups = {}
    for name, count in group_counts.items():
        group = {
            "sample_count": count,
            "candidate_yaw_sign_accuracy": 0.95,
            "control_yaw_sign_accuracy": 0.96,
            "intervention_coverage": deepcopy(coverage),
        }
        if name == "pair0":
            group["intervention_coverage"].update({
                "pair_rows_touched": 0, "pair_factors_touched": 0,
                "pair_valid_factors": 0,
            })
        for metric in BASE_GROUP_DISTRIBUTIONS:
            metric_count = 4 if (
                metric.startswith("fixed_") or "_handle_intervention_" in metric
            ) else count
            group[metric] = distribution(0.60, metric_count)
        if name != "pair0":
            for metric in PAIR_GROUP_DISTRIBUTIONS:
                metric_count = 4 if (
                    metric.startswith("fixed_") or "_pair_intervention_" in metric
                ) else count
                group[metric] = distribution(0.60, metric_count)
        group.update({
            "candidate_velocity": distribution(0.60, count),
            "candidate_yaw": distribution(1.80, count),
            "zero_angular_yaw": distribution(2.20, count),
            "zero_velocity_velocity": distribution(0.75, count),
            "fixed_intact_handle_closure": distribution(0.05, 4),
            "fixed_broken_handle_closure": distribution(0.06, 4),
            "candidate_handle_intervention_yaw": distribution(1.00, 4),
            "broken_handle_intervention_yaw": distribution(1.20, 4),
        })
        if name != "pair0":
            group.update({
                "fixed_intact_pair_closure": distribution(0.05, 4),
                "fixed_broken_pair_closure": distribution(0.07, 4),
                "candidate_pair_intervention_yaw": distribution(1.00, 4),
                "broken_pair_intervention_yaw": distribution(1.25, 4),
            })
        groups[name] = group
    cross_item = {
        "sample_count": 200,
        "unique_donor_count": 20,
        "donor_target_absolute_yaw_gap_rad_s": distribution(
            0.50, 200, p50=0.45, p95=0.80, p99=0.90,
        ),
        "velocity_error_to_injected_truth_mps": distribution(0.50, 200),
        "yaw_transfer": {
            "sample_count": 200,
            "yaw_mae_rad_s": 0.20,
            "zero_intercept_slope": 0.90,
            "pearson_correlation": 0.95,
            "median_absolute_ratio": 0.95,
            "prediction_margin_coverage": 0.90,
            "sign_accuracy_with_prediction_margin": 0.98,
        },
        "factorial_aa_ab_ba_bb": {
            "common_switch_truth_delta_magnitude_mps": distribution(0.80, 200),
            "common_switch_direction_quadrant_count": 4,
            "common_switch_velocity_axis_transfer": {
                "x": {
                    "support_count": 200,
                    "transfer": {
                        "sample_count": 200,
                        "yaw_mae_rad_s": 0.20,
                        "zero_intercept_slope": 0.90,
                        "pearson_correlation": 0.95,
                        "median_absolute_ratio": 0.95,
                        "prediction_margin_coverage": 0.90,
                        "sign_accuracy_with_prediction_margin": 0.95,
                    },
                },
                "y": {
                    "support_count": 200,
                    "transfer": {
                        "sample_count": 200,
                        "yaw_mae_rad_s": 0.20,
                        "zero_intercept_slope": 0.90,
                        "pearson_correlation": 0.95,
                        "median_absolute_ratio": 0.95,
                        "prediction_margin_coverage": 0.90,
                        "sign_accuracy_with_prediction_margin": 0.95,
                    },
                },
            },
            "common_switch_velocity_delta_error_mps": distribution(0.50, 400),
            "common_switch_yaw_leak_rad_s": distribution(
                0.005, 400, p50=0.003, p95=0.008, p99=0.01,
            ),
            "relative_switch_velocity_leak_mps": distribution(0.20, 400),
            "relative_switch_yaw_delta_transfer": {
                "sample_count": 400,
                "yaw_mae_rad_s": 0.20,
                "zero_intercept_slope": 0.90,
                "pearson_correlation": 0.95,
                "median_absolute_ratio": 0.95,
                "prediction_margin_coverage": 0.90,
                "sign_accuracy_with_prediction_margin": 0.98,
            },
        },
    }
    same_cross_item = deepcopy(cross_item)
    same_cross_item["donor_target_absolute_yaw_gap_rad_s"] = distribution(
        2.50, 200, p50=2.40, p95=3.00, p99=3.20,
    )
    candidate = {
        "overall_velocity_mean_mps": 0.44,
        "combined_velocity_mean_mps": 0.67,
        "high_speed_combined_velocity_mean_mps": 1.18,
        "overall_yaw_mean_rad_s": 1.95,
        "combined_yaw_mean_rad_s": 2.50,
        "high_speed_combined_yaw_mean_rad_s": 4.15,
        "overall_yaw_sign_accuracy": 0.951,
        "core_yaw_mean_rad_s": 1.35,
        "control_core_yaw_mean_rad_s": 1.51,
        "diagnostics": {
            "schema_version": DIAGNOSTIC_SCHEMA,
            "validation_only": True,
            "test_accessed": False,
            "seed": 20260730,
            "v8_joint_control_checkpoint": "protected-v8-control.pt",
            "v8_joint_control_checkpoint_sha256": "0" * 64,
            "groups": groups,
            "write_isolation": {
                "zero_velocity_max_absolute_yaw_difference_normalized": 0.0,
            },
            "factor_level_truth_common_donor_relative_cross": {
                "opposite_sign_similar_magnitude": deepcopy(cross_item),
                "same_sign_different_magnitude": same_cross_item,
            },
            "common_ramp_equivariance": {
                "velocity_delta_error_mps": distribution(0.10, 750),
                "yaw_invariance_error_rad_s": distribution(
                    0.005, 750, p50=0.003, p95=0.008, p99=0.01,
                ),
            },
            "relative_reversal_equivariance": {
                "sample_count": 200,
                "velocity_prediction_invariance_mps": distribution(0.10, 200),
                "yaw_prediction_antisymmetry_rad_s": distribution(0.30, 200),
                "velocity_error_to_unchanged_truth_mps": distribution(0.50, 200),
                "yaw_transfer_to_negated_truth": {
                    "sample_count": 200,
                    "yaw_mae_rad_s": 0.20,
                    "zero_intercept_slope": 0.90,
                    "pearson_correlation": 0.95,
                    "median_absolute_ratio": 0.95,
                    "prediction_margin_coverage": 0.90,
                    "sign_accuracy_with_prediction_margin": 0.98,
                },
            },
        },
    }
    for count in (1, 2):
        candidate.update({
            f"pair{count}_velocity_mean": 0.80,
            f"control_pair{count}_velocity_mean": 0.90,
            f"pair{count}_yaw_mean": 3.50,
            f"control_pair{count}_yaw_mean": 4.00,
        })
    candidate.update({
        "pair3_velocity_mean": 0.90,
        "control_pair3_velocity_mean": 0.90,
        "pair3_yaw_mean": 2.00,
        "control_pair3_yaw_mean": 2.00,
    })
    control = {
        "overall_velocity_mean_mps": 0.50,
        "combined_velocity_mean_mps": 0.80,
        "high_speed_combined_velocity_mean_mps": 1.40,
        "overall_yaw_mean_rad_s": 1.90,
        "combined_yaw_mean_rad_s": 2.40,
        "high_speed_combined_yaw_mean_rad_s": 4.00,
        "overall_yaw_sign_accuracy": 0.96,
    }
    assert all(item["passed"] for item in _checks(candidate, control).values())
    bypass = deepcopy(candidate)
    bypass["diagnostics"][
        "factor_level_truth_common_donor_relative_cross"
    ]["opposite_sign_similar_magnitude"]["yaw_transfer"][
        "zero_intercept_slope"
    ] = 0.60
    assert not _checks(bypass, control)[
        "cross_opposite_yaw_slope_at_least_0_80"
    ]["passed"]
    nonfinite = deepcopy(candidate)
    nonfinite["diagnostics"]["groups"]["combined"][
        "zero_angular_yaw"
    ]["mean_m"] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        _checks(nonfinite, control)
    invalid_probability = deepcopy(candidate)
    invalid_probability["overall_yaw_sign_accuracy"] = 2.0
    with pytest.raises(ValueError, match="probability"):
        _checks(invalid_probability, control)
    invalid_count = deepcopy(candidate)
    invalid_count["diagnostics"]["groups"]["overall"]["sample_count"] = 749
    with pytest.raises(ValueError, match="sample count"):
        _checks(invalid_count, control)
    invalid_percentiles = deepcopy(candidate)
    invalid_percentiles["diagnostics"][
        "factor_level_truth_common_donor_relative_cross"
    ]["opposite_sign_similar_magnitude"][
        "donor_target_absolute_yaw_gap_rad_s"
    ]["p50_m"] = 0.9
    with pytest.raises(ValueError, match="percentiles"):
        _checks(invalid_percentiles, control)
    boolean_zero = deepcopy(candidate)
    boolean_zero["diagnostics"]["write_isolation"][
        "zero_velocity_max_absolute_yaw_difference_normalized"
    ] = False
    with pytest.raises(ValueError, match="write isolation"):
        _checks(boolean_zero, control)
    unexpected_diagnostic = deepcopy(candidate)
    unexpected_diagnostic["diagnostics"]["unexpected"] = "ignored"
    with pytest.raises(ValueError, match="diagnostic fields"):
        _checks(unexpected_diagnostic, control)
    unexpected_coverage = deepcopy(candidate)
    unexpected_coverage["diagnostics"]["groups"]["overall"][
        "intervention_coverage"
    ]["unexpected"] = 0
    with pytest.raises(ValueError, match="coverage fields"):
        _checks(unexpected_coverage, control)
