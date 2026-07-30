"""Independently aggregate two fixed omega-first ordered closure probes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .finalize_global_flow_closure_probe import (
    _control,
    _improves,
    _not_worse,
    _protected_control,
    _validated_contract_namespace,
    _worsens,
)
from .omega_first_ordered_closure_future import (
    AnonymousOmegaFirstOrderedClosureProbe,
)
from .train_causal_physical_ab import _git_state
from .train_joint_rigid_flow_probe import _state_parameter_count
from .train_omega_first_ordered_closure_probe import (
    DIAGNOSTIC_FIELDS,
    DIAGNOSTIC_SCHEMA,
    INTERVENTION_COVERAGE_FIELDS,
    RUN_SCHEMA,
    WRITE_ISOLATION_FIELDS,
    _validate_completed_omega_first_probe,
)
from .train_paired_twist_set_probe import (
    PROBE_SEEDS,
    V8_JOINT_REACHABLE_STATE_PARAMETERS,
)
from .train_pnp_window_mapper_distillation import _atomic_json


AGGREGATE_SCHEMA = "stage3-v12-omega-first-ordered-closure-aggregate-v1"


def _assert_finite_tree(value, *, path: str) -> None:
    if isinstance(value, dict):
        for name, item in value.items():
            _assert_finite_tree(item, path=f"{path}.{name}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_tree(item, path=f"{path}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"omega-first non-finite gate value: {path}")


def _require_probability(value, *, name: str) -> None:
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"omega-first probability differs: {name}")


EXPECTED_GROUP_COUNTS = {
    "overall": 750,
    "combined": 300,
    "combined_speed_gt_1_7": 82,
    "core": 149,
    "pair0": 127,
    "combined_pair1": 25,
    "combined_pair2": 23,
    "combined_pair3": 229,
}
BASE_GROUP_DISTRIBUTIONS = {
    "candidate_velocity", "candidate_yaw", "candidate_yaw_signed",
    "control_velocity", "control_yaw", "control_yaw_signed",
    "broken_handle_velocity", "broken_handle_yaw",
    "broken_pair_velocity", "broken_pair_yaw",
    "zero_angular_velocity", "zero_angular_yaw",
    "zero_velocity_velocity", "zero_velocity_yaw",
    "zero_all_velocity", "zero_all_yaw",
    "angular_handle_closure", "common_handle_closure",
    "fixed_intact_handle_closure", "fixed_broken_handle_closure",
    "candidate_handle_intervention_velocity",
    "candidate_handle_intervention_yaw",
    "broken_handle_intervention_velocity",
    "broken_handle_intervention_yaw",
}
PAIR_GROUP_DISTRIBUTIONS = {
    "pair_closure", "fixed_intact_pair_closure", "fixed_broken_pair_closure",
    "candidate_pair_intervention_velocity", "candidate_pair_intervention_yaw",
    "broken_pair_intervention_velocity", "broken_pair_intervention_yaw",
}
TRANSFER_FIELDS = {
    "sample_count", "yaw_mae_rad_s", "zero_intercept_slope",
    "pearson_correlation", "median_absolute_ratio",
    "prediction_margin_coverage", "sign_accuracy_with_prediction_margin",
}


def _validated_distribution(
    value: dict, *, name: str, expected_count: int | None,
    nonnegative: bool = True,
) -> None:
    required = {"count", "mean_m", "p50_m", "p95_m", "p99_m"}
    if set(value) != required:
        raise ValueError(f"omega-first distribution fields differ: {name}")
    count = value["count"]
    if (
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        or (expected_count is not None and count != expected_count)
    ):
        raise ValueError(f"omega-first distribution count differs: {name}")
    raw_statistics = [value[key] for key in ("mean_m", "p50_m", "p95_m", "p99_m")]
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in raw_statistics
    ):
        raise ValueError(f"omega-first distribution statistic differs: {name}")
    mean, p50, p95, p99 = (float(item) for item in raw_statistics)
    if not p50 <= p95 <= p99:
        raise ValueError(f"omega-first distribution percentiles differ: {name}")
    if nonnegative and min(mean, p50, p95, p99) < 0.0:
        raise ValueError(f"omega-first error distribution is negative: {name}")


def _validated_transfer(value: dict, *, name: str, expected_count: int) -> None:
    if set(value) != TRANSFER_FIELDS:
        raise ValueError(f"omega-first transfer fields differ: {name}")
    count = value["sample_count"]
    if (
        isinstance(count, bool) or not isinstance(count, int)
        or count != expected_count or count < 2
    ):
        raise ValueError(f"omega-first transfer count differs: {name}")
    for key in (
        "yaw_mae_rad_s", "zero_intercept_slope", "pearson_correlation",
        "median_absolute_ratio",
    ):
        item = value[key]
        if (
            isinstance(item, bool) or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"omega-first transfer statistic differs: {name}.{key}")
    if value["yaw_mae_rad_s"] < 0 or value["median_absolute_ratio"] < 0:
        raise ValueError(f"omega-first transfer error differs: {name}")
    if not -1.0 <= value["pearson_correlation"] <= 1.0:
        raise ValueError(f"omega-first transfer correlation differs: {name}")
    _require_probability(
        value["prediction_margin_coverage"], name=f"{name}.coverage",
    )
    _require_probability(
        value["sign_accuracy_with_prediction_margin"], name=f"{name}.sign",
    )


def _validate_diagnostic_binding(candidate: dict, control: dict) -> None:
    _require_probability(
        candidate["overall_yaw_sign_accuracy"], name="candidate overall yaw sign",
    )
    _require_probability(
        control["overall_yaw_sign_accuracy"], name="control overall yaw sign",
    )
    diagnostics = candidate["diagnostics"]
    if not isinstance(diagnostics, dict) or set(diagnostics) != DIAGNOSTIC_FIELDS:
        raise ValueError("omega-first diagnostic fields differ")
    if (
        diagnostics["schema_version"] != DIAGNOSTIC_SCHEMA
        or diagnostics["validation_only"] is not True
        or diagnostics["test_accessed"] is not False
        or isinstance(diagnostics["seed"], bool)
        or diagnostics["seed"] not in PROBE_SEEDS
        or not isinstance(diagnostics["v8_joint_control_checkpoint"], str)
        or not diagnostics["v8_joint_control_checkpoint"]
        or not isinstance(
            diagnostics["v8_joint_control_checkpoint_sha256"], str,
        )
        or len(diagnostics["v8_joint_control_checkpoint_sha256"]) != 64
    ):
        raise ValueError("omega-first diagnostic identity differs")
    write_isolation = diagnostics["write_isolation"]
    if (
        not isinstance(write_isolation, dict)
        or set(write_isolation) != WRITE_ISOLATION_FIELDS
    ):
        raise ValueError("omega-first write isolation fields differ")
    write_isolation_value = write_isolation[
        "zero_velocity_max_absolute_yaw_difference_normalized"
    ]
    if (
        isinstance(write_isolation_value, bool)
        or not isinstance(write_isolation_value, (int, float))
        or not math.isfinite(float(write_isolation_value))
        or float(write_isolation_value) != 0.0
    ):
        raise ValueError("omega-first write isolation differs")
    groups = diagnostics["groups"]
    if set(groups) != set(EXPECTED_GROUP_COUNTS):
        raise ValueError("omega-first diagnostic group set differs")
    state_distributions = {
        "candidate_velocity", "candidate_yaw", "candidate_yaw_signed",
        "control_velocity", "control_yaw", "control_yaw_signed",
        "broken_handle_velocity", "broken_handle_yaw",
        "broken_pair_velocity", "broken_pair_yaw",
        "zero_angular_velocity", "zero_angular_yaw",
        "zero_velocity_velocity", "zero_velocity_yaw",
        "zero_all_velocity", "zero_all_yaw",
    }
    for group_name, expected_count in EXPECTED_GROUP_COUNTS.items():
        group = groups[group_name]
        required_distributions = set(BASE_GROUP_DISTRIBUTIONS)
        if group_name != "pair0":
            required_distributions.update(PAIR_GROUP_DISTRIBUTIONS)
        required_group_fields = required_distributions | {
            "sample_count", "candidate_yaw_sign_accuracy",
            "control_yaw_sign_accuracy", "intervention_coverage",
        }
        if set(group) != required_group_fields:
            raise ValueError(f"omega-first group fields differ: {group_name}")
        if group.get("sample_count") != expected_count:
            raise ValueError(f"omega-first group sample count differs: {group_name}")
        for sign_name in (
            "candidate_yaw_sign_accuracy", "control_yaw_sign_accuracy",
        ):
            _require_probability(group[sign_name], name=f"{group_name}.{sign_name}")
        coverage = group["intervention_coverage"]
        if (
            not isinstance(coverage, dict)
            or set(coverage) != set(INTERVENTION_COVERAGE_FIELDS)
        ):
            raise ValueError(
                f"omega-first intervention coverage fields differ: {group_name}"
            )
        for key, value in coverage.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"omega-first intervention coverage differs: {group_name}.{key}"
                )
        if (
            coverage["handle_rows_touched"] > expected_count
            or coverage["pair_rows_touched"] > expected_count
            or coverage["handle_factors_touched"]
            > coverage["handle_valid_factors"]
            or coverage["pair_factors_touched"] > coverage["pair_valid_factors"]
        ):
            raise ValueError(f"omega-first intervention coverage exceeds support: {group_name}")
        for metric_name, distribution in group.items():
            if not (
                isinstance(distribution, dict)
                and "count" in distribution and "mean_m" in distribution
            ):
                continue
            expected = None
            if metric_name in state_distributions:
                expected = expected_count
            elif metric_name.startswith("fixed_") and "handle" in metric_name:
                expected = coverage["handle_rows_touched"]
            elif metric_name.startswith("fixed_") and "pair" in metric_name:
                expected = coverage["pair_rows_touched"]
            elif "_handle_intervention_" in metric_name:
                expected = coverage["handle_rows_touched"]
            elif "_pair_intervention_" in metric_name:
                expected = coverage["pair_rows_touched"]
            _validated_distribution(
                distribution, name=f"{group_name}.{metric_name}",
                expected_count=expected,
                nonnegative=metric_name not in {
                    "candidate_yaw_signed", "control_yaw_signed",
                },
            )
            if expected is None and distribution["count"] > expected_count:
                raise ValueError(
                    f"omega-first closure count exceeds group: {group_name}.{metric_name}"
                )
    cross = diagnostics["factor_level_truth_common_donor_relative_cross"]
    for relation, item in cross.items():
        if set(item) != {
            "sample_count", "unique_donor_count",
            "donor_target_absolute_yaw_gap_rad_s",
            "velocity_error_to_injected_truth_mps", "yaw_transfer",
            "factorial_aa_ab_ba_bb",
        }:
            raise ValueError(f"omega-first cross fields differ: {relation}")
        sample_count = item["sample_count"]
        unique_count = item["unique_donor_count"]
        if (
            isinstance(sample_count, bool) or not isinstance(sample_count, int)
            or isinstance(unique_count, bool) or not isinstance(unique_count, int)
            or not 1 <= unique_count <= sample_count
        ):
            raise ValueError(f"omega-first cross support differs: {relation}")
        for key in (
            "donor_target_absolute_yaw_gap_rad_s",
            "velocity_error_to_injected_truth_mps",
        ):
            _validated_distribution(
                item[key], name=f"{relation}.{key}",
                expected_count=sample_count,
            )
        _validated_transfer(
            item["yaw_transfer"], name=f"{relation}.yaw_transfer",
            expected_count=sample_count,
        )
        factorial = item["factorial_aa_ab_ba_bb"]
        if set(factorial) != {
            "common_switch_truth_delta_magnitude_mps",
            "common_switch_direction_quadrant_count",
            "common_switch_velocity_axis_transfer",
            "common_switch_velocity_delta_error_mps",
            "common_switch_yaw_leak_rad_s",
            "relative_switch_velocity_leak_mps",
            "relative_switch_yaw_delta_transfer",
        }:
            raise ValueError(f"omega-first factorial fields differ: {relation}")
        _validated_distribution(
            factorial["common_switch_truth_delta_magnitude_mps"],
            name=f"{relation}.factorial.common_truth_delta",
            expected_count=sample_count,
        )
        quadrant_count = factorial["common_switch_direction_quadrant_count"]
        if (
            isinstance(quadrant_count, bool) or not isinstance(quadrant_count, int)
            or not 1 <= quadrant_count <= 4
        ):
            raise ValueError(f"omega-first common direction coverage differs: {relation}")
        for key in (
            "common_switch_velocity_delta_error_mps",
            "common_switch_yaw_leak_rad_s",
            "relative_switch_velocity_leak_mps",
        ):
            _validated_distribution(
                factorial[key], name=f"{relation}.factorial.{key}",
                expected_count=2 * sample_count,
            )
        _validated_transfer(
            factorial["relative_switch_yaw_delta_transfer"],
            name=f"{relation}.factorial.relative_yaw",
            expected_count=2 * sample_count,
        )
        common_axis_transfer = factorial["common_switch_velocity_axis_transfer"]
        if set(common_axis_transfer) != {"x", "y"}:
            raise ValueError(f"omega-first common velocity axes differ: {relation}")
        for axis in ("x", "y"):
            axis_item = common_axis_transfer[axis]
            if set(axis_item) != {"support_count", "transfer"}:
                raise ValueError(
                    f"omega-first common velocity axis fields differ: {relation}.{axis}"
                )
            support_count = axis_item["support_count"]
            if (
                isinstance(support_count, bool)
                or not isinstance(support_count, int)
                or support_count < 64
                or support_count % 2 != 0
                or support_count > 2 * sample_count
            ):
                raise ValueError(
                    f"omega-first common velocity axis support differs: "
                    f"{relation}.{axis}"
                )
            _validated_transfer(
                axis_item["transfer"],
                name=f"{relation}.factorial.common_velocity.{axis}",
                expected_count=support_count,
            )
    ramp = diagnostics["common_ramp_equivariance"]
    for key in ("velocity_delta_error_mps", "yaw_invariance_error_rad_s"):
        _validated_distribution(
            ramp[key], name=f"common_ramp.{key}", expected_count=750,
        )
    reversal = diagnostics["relative_reversal_equivariance"]
    reversal_count = reversal["sample_count"]
    for key in (
        "velocity_prediction_invariance_mps",
        "yaw_prediction_antisymmetry_rad_s",
        "velocity_error_to_unchanged_truth_mps",
    ):
        _validated_distribution(
            reversal[key], name=f"relative_reversal.{key}",
            expected_count=reversal_count,
        )
    _validated_transfer(
        reversal["yaw_transfer_to_negated_truth"],
        name="relative_reversal.yaw_transfer", expected_count=reversal_count,
    )


def _checks(candidate: dict, control: dict) -> dict[str, dict[str, bool]]:
    _assert_finite_tree(candidate, path="candidate")
    _assert_finite_tree(control, path="control")
    _validate_diagnostic_binding(candidate, control)
    diagnostics = candidate["diagnostics"]
    groups = diagnostics["groups"]
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
        "angular_refinement_improves_combined_or_core_yaw": (
            _worsens(
                groups["combined"]["zero_angular_yaw"]["mean_m"],
                groups["combined"]["candidate_yaw"]["mean_m"], 0.15, 0.30,
            )
            or _worsens(
                groups["core"]["zero_angular_yaw"]["mean_m"],
                groups["core"]["candidate_yaw"]["mean_m"], 0.15, 0.30,
            )
        ),
        "velocity_refinement_improves_combined_or_high_speed_velocity": (
            _worsens(
                groups["combined"]["zero_velocity_velocity"]["mean_m"],
                groups["combined"]["candidate_velocity"]["mean_m"], 0.08, 0.10,
            )
            or _worsens(
                groups["combined_speed_gt_1_7"]["zero_velocity_velocity"]["mean_m"],
                groups["combined_speed_gt_1_7"]["candidate_velocity"]["mean_m"],
                0.08, 0.10,
            )
        ),
        "fixed_handle_correspondence_worsens_pair3_angular_closure": _worsens(
            groups["combined_pair3"]["fixed_broken_handle_closure"]["mean_m"],
            groups["combined_pair3"]["fixed_intact_handle_closure"]["mean_m"],
            0.10, 0.005,
        ),
        "pair0_handle_geometry_changes_estimated_yaw": _worsens(
            groups["pair0"]["broken_handle_intervention_yaw"]["mean_m"],
            groups["pair0"]["candidate_handle_intervention_yaw"]["mean_m"],
            0.10, 0.15,
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
        checks[f"fixed_pair_correspondence_worsens_pair{count}_closure"] = (
            _worsens(
                groups[f"combined_pair{count}"][
                    "fixed_broken_pair_closure"
                ]["mean_m"],
                groups[f"combined_pair{count}"][
                    "fixed_intact_pair_closure"
                ]["mean_m"],
                0.10, 0.005,
            )
        )
        checks[f"pair{count}_geometry_changes_estimated_yaw"] = _worsens(
            groups[f"combined_pair{count}"][
                "broken_pair_intervention_yaw"
            ]["mean_m"],
            groups[f"combined_pair{count}"][
                "candidate_pair_intervention_yaw"
            ]["mean_m"],
            0.10, 0.20,
        )
    checks["pair3_velocity_regresses_no_more_than_5_percent"] = _not_worse(
        candidate["pair3_velocity_mean"], candidate["control_pair3_velocity_mean"],
        0.05,
    )
    checks["pair3_yaw_regresses_no_more_than_5_percent"] = _not_worse(
        candidate["pair3_yaw_mean"], candidate["control_pair3_yaw_mean"], 0.05,
    )
    checks["fixed_pair_correspondence_worsens_pair3_closure"] = _worsens(
        groups["combined_pair3"]["fixed_broken_pair_closure"]["mean_m"],
        groups["combined_pair3"]["fixed_intact_pair_closure"]["mean_m"],
        0.15, 0.01,
    )
    coverage = groups
    checks["handle_intervention_touches_every_support_group"] = all(
        coverage[name]["intervention_coverage"]["handle_rows_touched"] >= 4
        for name in ("pair0", "combined_pair1", "combined_pair2", "combined_pair3")
    )
    checks["pair_intervention_touches_pair1_pair2_pair3"] = all(
        coverage[f"combined_pair{count}"]["intervention_coverage"][
            "pair_rows_touched"
        ] >= 4
        for count in (1, 2, 3)
    )
    cross = candidate["diagnostics"][
        "factor_level_truth_common_donor_relative_cross"
    ]
    opposite = cross["opposite_sign_similar_magnitude"]
    same = cross["same_sign_different_magnitude"]
    checks["cross_opposite_has_at_least_128_samples"] = (
        opposite["sample_count"] >= 128
    )
    checks["cross_same_has_at_least_192_samples"] = same["sample_count"] >= 192
    checks["cross_opposite_has_at_least_16_unique_donors"] = (
        opposite["unique_donor_count"] >= 16
    )
    checks["cross_same_has_at_least_16_unique_donors"] = (
        same["unique_donor_count"] >= 16
    )
    checks["cross_opposite_yaw_gap_p95_at_most_1_rad_s"] = (
        opposite["donor_target_absolute_yaw_gap_rad_s"]["p95_m"] <= 1.0
    )
    checks["cross_same_yaw_gap_p50_at_least_2_rad_s"] = (
        same["donor_target_absolute_yaw_gap_rad_s"]["p50_m"] >= 2.0
    )
    checks["cross_opposite_velocity_error_at_most_0_60_mps"] = (
        opposite["velocity_error_to_injected_truth_mps"]["mean_m"] <= 0.60
    )
    checks["cross_same_velocity_error_at_most_0_60_mps"] = (
        same["velocity_error_to_injected_truth_mps"]["mean_m"] <= 0.60
    )
    for name, item, correlation_floor, sign_floor in (
        ("opposite", opposite, 0.85, 0.90),
        ("same", same, 0.90, 0.95),
    ):
        transfer = item["yaw_transfer"]
        for probability_name in (
            "prediction_margin_coverage", "sign_accuracy_with_prediction_margin",
        ):
            _require_probability(
                transfer[probability_name],
                name=f"cross_{name}_{probability_name}",
            )
        correlation = transfer["pearson_correlation"]
        if not -1.0 <= correlation <= 1.0:
            raise ValueError(f"omega-first correlation differs: cross_{name}")
        if transfer["sample_count"] != item["sample_count"]:
            raise ValueError(f"omega-first cross sample binding differs: {name}")
        checks[f"cross_{name}_yaw_slope_at_least_0_80"] = (
            transfer["zero_intercept_slope"] >= 0.80
        )
        checks[f"cross_{name}_yaw_correlation"] = (
            transfer["pearson_correlation"] >= correlation_floor
        )
        checks[f"cross_{name}_yaw_ratio_in_0_80_to_1_20"] = (
            0.80 <= transfer["median_absolute_ratio"] <= 1.20
        )
        checks[f"cross_{name}_yaw_sign"] = (
            transfer["sign_accuracy_with_prediction_margin"] >= sign_floor
        )
        checks[f"cross_{name}_prediction_margin_coverage_at_least_0_80"] = (
            transfer["prediction_margin_coverage"] >= 0.80
        )
        factorial = item["factorial_aa_ab_ba_bb"]
        checks[f"factorial_{name}_common_truth_delta_p50_at_least_0_50"] = (
            factorial["common_switch_truth_delta_magnitude_mps"]["p50_m"]
            >= 0.50
        )
        checks[f"factorial_{name}_common_has_all_4_direction_quadrants"] = (
            factorial["common_switch_direction_quadrant_count"] == 4
        )
        checks[f"factorial_{name}_common_velocity_delta_error_at_most_0_75"] = (
            factorial["common_switch_velocity_delta_error_mps"]["mean_m"]
            <= 0.75
        )
        checks[f"factorial_{name}_common_yaw_leak_p99_at_most_0_02"] = (
            factorial["common_switch_yaw_leak_rad_s"]["p99_m"] <= 0.02
        )
        checks[f"factorial_{name}_relative_velocity_leak_at_most_0_35"] = (
            factorial["relative_switch_velocity_leak_mps"]["mean_m"] <= 0.35
        )
        factorial_yaw = factorial["relative_switch_yaw_delta_transfer"]
        for probability_name in (
            "prediction_margin_coverage", "sign_accuracy_with_prediction_margin",
        ):
            _require_probability(
                factorial_yaw[probability_name],
                name=f"factorial_{name}_{probability_name}",
            )
        if not -1.0 <= factorial_yaw["pearson_correlation"] <= 1.0:
            raise ValueError(f"omega-first factorial correlation differs: {name}")
        if factorial_yaw["sample_count"] != 2 * item["sample_count"]:
            raise ValueError(f"omega-first factorial sample binding differs: {name}")
        checks[f"factorial_{name}_relative_yaw_slope_at_least_0_75"] = (
            factorial_yaw["zero_intercept_slope"] >= 0.75
        )
        checks[f"factorial_{name}_relative_yaw_correlation_at_least_0_85"] = (
            factorial_yaw["pearson_correlation"] >= 0.85
        )
        checks[f"factorial_{name}_relative_yaw_ratio_in_0_70_to_1_30"] = (
            0.70 <= factorial_yaw["median_absolute_ratio"] <= 1.30
        )
        checks[f"factorial_{name}_relative_yaw_sign_at_least_0_90"] = (
            factorial_yaw["sign_accuracy_with_prediction_margin"] >= 0.90
        )
        checks[f"factorial_{name}_relative_yaw_coverage_at_least_0_80"] = (
            factorial_yaw["prediction_margin_coverage"] >= 0.80
        )
        for axis in ("x", "y"):
            common_transfer = factorial["common_switch_velocity_axis_transfer"][
                axis
            ]["transfer"]
            checks[f"factorial_{name}_common_{axis}_slope_at_least_0_75"] = (
                common_transfer["zero_intercept_slope"] >= 0.75
            )
            checks[
                f"factorial_{name}_common_{axis}_correlation_at_least_0_80"
            ] = common_transfer["pearson_correlation"] >= 0.80
            checks[f"factorial_{name}_common_{axis}_ratio_in_0_70_to_1_30"] = (
                0.70 <= common_transfer["median_absolute_ratio"] <= 1.30
            )
            checks[f"factorial_{name}_common_{axis}_sign_at_least_0_85"] = (
                common_transfer["sign_accuracy_with_prediction_margin"] >= 0.85
            )
            checks[f"factorial_{name}_common_{axis}_coverage_at_least_0_70"] = (
                common_transfer["prediction_margin_coverage"] >= 0.70
            )
    ramp = diagnostics["common_ramp_equivariance"]
    checks["common_ramp_velocity_delta_error_at_most_0_15_mps"] = (
        ramp["velocity_delta_error_mps"]["mean_m"] <= 0.15
    )
    checks["common_ramp_yaw_invariance_p99_at_most_0_02_rad_s"] = (
        ramp["yaw_invariance_error_rad_s"]["p99_m"] <= 0.02
    )
    reversal = diagnostics["relative_reversal_equivariance"]
    checks["relative_reversal_has_at_least_128_samples"] = (
        reversal["sample_count"] >= 128
    )
    checks["relative_reversal_velocity_error_at_most_0_60_mps"] = (
        reversal["velocity_error_to_unchanged_truth_mps"]["mean_m"] <= 0.60
    )
    checks["relative_reversal_prediction_velocity_invariance_at_most_0_15_mps"] = (
        reversal["velocity_prediction_invariance_mps"]["mean_m"] <= 0.15
    )
    checks["relative_reversal_prediction_yaw_antisymmetry_at_most_0_50_rad_s"] = (
        reversal["yaw_prediction_antisymmetry_rad_s"]["mean_m"] <= 0.50
    )
    reversal_yaw = reversal["yaw_transfer_to_negated_truth"]
    for probability_name in (
        "prediction_margin_coverage", "sign_accuracy_with_prediction_margin",
    ):
        _require_probability(
            reversal_yaw[probability_name],
            name=f"relative_reversal_{probability_name}",
        )
    if not -1.0 <= reversal_yaw["pearson_correlation"] <= 1.0:
        raise ValueError("omega-first reversal correlation differs")
    if reversal_yaw["sample_count"] != reversal["sample_count"]:
        raise ValueError("omega-first reversal sample binding differs")
    checks["relative_reversal_yaw_slope_at_least_0_80"] = (
        reversal_yaw["zero_intercept_slope"] >= 0.80
    )
    checks["relative_reversal_yaw_correlation_at_least_0_90"] = (
        reversal_yaw["pearson_correlation"] >= 0.90
    )
    checks["relative_reversal_yaw_ratio_in_0_80_to_1_20"] = (
        0.80 <= reversal_yaw["median_absolute_ratio"] <= 1.20
    )
    checks["relative_reversal_yaw_sign_at_least_0_95"] = (
        reversal_yaw["sign_accuracy_with_prediction_margin"] >= 0.95
    )
    checks["relative_reversal_yaw_coverage_at_least_0_80"] = (
        reversal_yaw["prediction_margin_coverage"] >= 0.80
    )
    return {
        name: {"required": True, "passed": bool(passed)}
        for name, passed in checks.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="aggregate two omega-first probes")
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.run) != 2:
        raise ValueError("omega-first aggregate requires exactly two runs")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"omega-first aggregate output exists: {output}")
    git = _git_state()
    if git.get("worktree_dirty") is not False:
        raise RuntimeError("omega-first aggregate requires clean checkout")
    reports = []
    manifests = []
    for run in args.run:
        root = Path(run).resolve()
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != RUN_SCHEMA:
            raise ValueError("omega-first run schema differs")
        namespace = _validated_contract_namespace(manifest["contract"]["args"])
        checkpoint = root / "checkpoints" / "checkpoint-update-000200.pt"
        recorded = json.loads((root / "probe_result.json").read_text(encoding="utf-8"))
        parameter_count = _state_parameter_count(
            AnonymousOmegaFirstOrderedClosureProbe, namespace,
        )
        if recorded.get("total_state_parameter_count") != parameter_count:
            raise ValueError("omega-first recorded capacity differs")
        validated = _validate_completed_omega_first_probe(
            namespace, checkpoint, parameter_count,
        )
        if validated != recorded:
            raise ValueError("omega-first recorded result differs")
        reports.append(validated)
        manifests.append(manifest)
    if {item["seed"] for item in reports} != set(PROBE_SEEDS):
        raise ValueError("omega-first aggregate seed set differs")
    if any(item["source_commit"] != git.get("git_commit") for item in reports):
        raise ValueError("omega-first aggregate source commit differs")
    counts = {item["total_state_parameter_count"] for item in reports}
    if len(counts) != 1:
        raise ValueError("omega-first capacities differ across seeds")
    parameter_count = next(iter(counts))
    capacity_matched = (
        abs(parameter_count - V8_JOINT_REACHABLE_STATE_PARAMETERS)
        / V8_JOINT_REACHABLE_STATE_PARAMETERS <= 0.05
    )
    if not capacity_matched:
        raise ValueError("omega-first capacity exceeds five percent")
    base_args = dict(manifests[0]["contract"]["args"])
    other_args = dict(manifests[1]["contract"]["args"])
    for name in ("seed", "output", "v8_joint_control_checkpoint"):
        base_args.pop(name, None)
        other_args.pop(name, None)
    if base_args != other_args:
        raise ValueError("omega-first run arguments differ")
    comparisons = {}
    for candidate in sorted(reports, key=lambda item: item["seed"]):
        diagnostic_control = _control(candidate)
        control = _protected_control(candidate["seed"])
        for name, value in diagnostic_control.items():
            if control.get(name) != value:
                raise ValueError(f"omega-first V8 replay differs: {name}")
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
        "authorized_v12_state_training": passed,
        "future_position_decoder_status": "deferred_and_frozen",
        "required_seeds": list(PROBE_SEEDS),
        "source_commit": git["git_commit"],
        "test_accessed": False,
        "capacity": {
            "v12_gradient_reachable_state_parameters": parameter_count,
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
        "authorized_v12_state_training": result[
            "authorized_v12_state_training"
        ],
    }, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
