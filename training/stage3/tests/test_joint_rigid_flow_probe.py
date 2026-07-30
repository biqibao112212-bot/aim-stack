from __future__ import annotations

from copy import deepcopy
import inspect
from types import SimpleNamespace

import pytest
import torch

from training.stage3.joint_rigid_flow_probe import (
    AnonymousJointTwistProbe,
    AnonymousLocalRigidFlowContext,
    AnonymousSeparatedRigidFlowProbe,
    RigidFlowProbeHead,
    _deterministic_probe_ramp,
    rigid_flow_probe_train_step,
)
from training.stage3.factorized_common_relative_motion_future import (
    FactorizedCommonRelativeMotionStateV7,
)
from training.stage3.robust_multiscale_motion_future import (
    robust_multiscale_motion_state_loss,
)
from training.stage3.tests.test_anonymous_vehicle_motion import _batch
from training.stage3.tests.test_stable_motion_bottleneck_future import _supervised_batch
from training.stage3.train_joint_rigid_flow_probe import (
    EXPECTED_PATHS,
    LOCKED_VALUES,
    PROBE_SEEDS,
    _state_parameter_count,
    _validate_args,
    build_probe_parser,
)
from training.stage3.finalize_joint_rigid_flow_probe import _candidate_checks


def _model(kind="joint"):
    torch.manual_seed(20260730)
    cls = {
        "joint": AnonymousJointTwistProbe,
        "separated": AnonymousSeparatedRigidFlowProbe,
        "v7_factorized": FactorizedCommonRelativeMotionStateV7,
    }[kind]
    return cls(
        velocity_scale_mps=(3.0, 3.2, 0.25),
        yaw_rate_scale_rad_s=17.25,
        channels=32,
        dropout=0.0,
        message_layers=2,
        basis_count=6,
    )


def _fields(batch):
    return {
        name: batch[name] for name in (
            "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
            "history_event_mask", "history_time_s", "history_switch_step",
        )
    }


def test_v8_probe_parser_is_fixed_to_200_state_updates() -> None:
    parser = build_probe_parser()
    assert parser.get_default("motion_state_updates") == 200
    valid = SimpleNamespace(
        motion_state_updates=200, trajectory_updates=0, selector_updates=0,
        decoder_joint_updates=0, stop_after_update=0, seed=PROBE_SEEDS[0],
        diagnostic_oracle_association=True, allow_mapper_h_mismatch=True,
        **LOCKED_VALUES,
        **{name: str(path) for name, path in EXPECTED_PATHS.items()},
    )
    _validate_args(valid)
    invalid = SimpleNamespace(**vars(valid))
    invalid.trajectory_updates = 1
    with pytest.raises(ValueError, match="200 state-only"):
        _validate_args(invalid)


def test_separated_and_joint_probe_capacity_is_matched() -> None:
    args = SimpleNamespace(channels=96, dropout=0.05, message_layers=3, basis_count=8)
    separated = _state_parameter_count(AnonymousSeparatedRigidFlowProbe, args)
    joint = _state_parameter_count(AnonymousJointTwistProbe, args)
    assert abs(separated - joint) / max(separated, joint) < 0.05
    assert _state_parameter_count(FactorizedCommonRelativeMotionStateV7, args) > 0


def test_v8_state_api_is_exactly_six_fields_and_has_no_identity_input() -> None:
    expected = {
        "self", "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
        "history_event_mask", "history_time_s", "history_switch_step",
    }
    assert set(inspect.signature(
        AnonymousLocalRigidFlowContext.forward,
    ).parameters) == expected
    assert set(inspect.signature(
        AnonymousJointTwistProbe.estimate_motion_state,
    ).parameters) == expected
    config = _model().config
    for field in (
        "physical_id_input", "motion_class_input", "session_identity_input",
        "truth_state_input",
    ):
        assert config[field] is False
    assert config["motion_context"]["long_projective_yaw_lags"] is False
    assert config["motion_context"]["curvature_fallback"] is False


@torch.inference_mode()
def test_pair_flow_is_common_velocity_ramp_invariant() -> None:
    context = _model().context.eval()
    batch = _batch()
    reference = context(**_fields(batch))
    changed = deepcopy(batch)
    ramp = torch.tensor([[0.6, -0.4, 0.0], [-0.2, 0.5, 0.0]])
    offset = changed["history_time_s"][:, :, None, None] * ramp[:, None, None]
    valid = (
        changed["history_event_mask"][:, :, None]
        & changed["history_obs_mask"]
    )
    changed["history_obs_rel_m"] = torch.where(
        valid.unsqueeze(-1), changed["history_obs_rel_m"] + offset,
        changed["history_obs_rel_m"],
    )
    actual = context(**_fields(changed))
    assert torch.equal(actual["pair_flow_available"], reference["pair_flow_available"])
    torch.testing.assert_close(
        actual["pair_flow_latent"], reference["pair_flow_latent"],
        rtol=2e-5, atol=3e-6,
    )


def test_local_pair_lag_rejects_a_different_visible_set() -> None:
    context = AnonymousLocalRigidFlowContext(channels=32, dropout=0.0, message_layers=2)
    pair = torch.tensor([[[1.0, 0.0, 0.0], [0.98, 0.2, 0.0]]])
    time = torch.tensor([[-0.03, 0.0]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    changed_set = torch.tensor([[[True, True, False, False], [False, True, True, False]]])
    primary = torch.tensor([[[True, False, False, False], [False, True, False, False]]])
    _, _, available, _ = context._same_set_lag_bank(
        pair, time, valid, changed_set, primary,
    )
    assert not available.any()
    same_set = changed_set.clone()
    same_set[:, 1] = same_set[:, 0]
    _, _, available, _ = context._same_set_lag_bank(
        pair, time, valid, same_set, primary,
    )
    assert available[0, 1, 1]


def test_local_pair_lag_accepts_10ms_and_rejects_105ms_or_longer() -> None:
    context = AnonymousLocalRigidFlowContext(channels=32, dropout=0.0, message_layers=2)
    pair = torch.tensor([[[1.0, 0.0, 0.0], [0.99, 0.1, 0.0]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    visible = torch.tensor([[[True, True, False, False], [True, True, False, False]]])
    primary = torch.tensor([[[True, False, False, False], [True, False, False, False]]])
    _, _, available, _ = context._same_set_lag_bank(
        pair, torch.tensor([[-0.01, 0.0]]), valid, visible, primary,
    )
    assert available[0, 1, 0]
    _, _, available, _ = context._same_set_lag_bank(
        pair, torch.tensor([[-0.106, 0.0]]), valid, visible, primary,
    )
    assert not available.any()


def test_v8_candidate_gate_requires_both_yaw_improvements_without_velocity_regression() -> None:
    control = {
        "overall_yaw_mean_rad_s": 4.0,
        "combined_yaw_mean_rad_s": 3.0,
        "overall_yaw_sign_accuracy": 0.82,
        "overall_velocity_mean_mps": 0.45,
        "combined_velocity_mean_mps": 0.65,
        "high_speed_combined_yaw_mean_rad_s": 6.0,
        "high_speed_combined_velocity_mean_mps": 1.10,
    }
    passing = {
        "overall_yaw_mean_rad_s": 2.79,
        "combined_yaw_mean_rad_s": 2.09,
        "overall_yaw_sign_accuracy": 0.811,
        "overall_velocity_mean_mps": 0.479,
        "combined_velocity_mean_mps": 0.679,
        "high_speed_combined_yaw_mean_rad_s": 4.19,
        "high_speed_combined_velocity_mean_mps": 1.129,
    }
    assert all(item["passed"] for item in _candidate_checks(passing, control).values())
    failing = dict(passing, combined_yaw_mean_rad_s=2.11)
    checks = _candidate_checks(failing, control)
    assert not checks["combined_yaw_improves_at_least_30_percent"]["passed"]
    failing = dict(passing, high_speed_combined_yaw_mean_rad_s=4.21)
    checks = _candidate_checks(failing, control)
    assert not checks[
        "high_speed_combined_yaw_improves_at_least_30_percent"
    ]["passed"]


def test_probe_ramp_depends_on_seed_and_update_not_consumed_rng_state() -> None:
    torch.manual_seed(20260730)
    selected_a, ramp_a = _deterministic_probe_ramp(
        64, dtype=torch.float32, device=torch.device("cpu"), stage_update=17,
    )
    _ = torch.rand(10000)
    selected_b, ramp_b = _deterministic_probe_ramp(
        64, dtype=torch.float32, device=torch.device("cpu"), stage_update=17,
    )
    assert torch.equal(selected_a, selected_b)
    assert torch.equal(ramp_a, ramp_b)
    _, ramp_next = _deterministic_probe_ramp(
        64, dtype=torch.float32, device=torch.device("cpu"), stage_update=18,
    )
    assert not torch.equal(ramp_a, ramp_next)


def test_local_pair_orientation_uses_primary_swap_not_dot_product() -> None:
    context = AnonymousLocalRigidFlowContext(channels=32, dropout=0.0, message_layers=2)
    angle = torch.deg2rad(torch.tensor(100.0))
    current = torch.stack((angle.cos(), angle.sin(), torch.tensor(0.0)))
    pair = torch.stack((torch.tensor([1.0, 0.0, 0.0]), current)).unsqueeze(0)
    time = torch.tensor([[-0.07, 0.0]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    visible = torch.tensor([[[True, True, False, False], [True, True, False, False]]])
    same_primary = torch.tensor([[[True, False, False, False], [True, False, False, False]]])
    prior, _, available, _ = context._same_set_lag_bank(
        pair, time, valid, visible, same_primary,
    )
    assert available[0, 1, 2]
    cross = torch.cross(prior[0, 1, 2], current, dim=-1)
    assert cross[2] > 0.98

    swapped_pair = pair.clone()
    swapped_pair[:, 1] = -current
    swapped_primary = same_primary.clone()
    swapped_primary[:, 1] = torch.tensor([False, True, False, False])
    prior, _, available, _ = context._same_set_lag_bank(
        swapped_pair, time, valid, visible, swapped_primary,
    )
    cross = torch.cross(prior[0, 1, 2], swapped_pair[0, 1], dim=-1)
    assert cross[2] > 0.98

    outside = torch.tensor([[-0.106, 0.0]])
    _, _, available, _ = context._same_set_lag_bank(
        pair, outside, valid, visible, same_primary,
    )
    assert not available.any()


@pytest.mark.parametrize("kind", ["separated", "joint"])
@torch.inference_mode()
def test_v8_probe_is_c4_and_reflection_invariant(kind: str) -> None:
    model = _model(kind).eval()
    batch = _batch()
    reference = model.estimate_motion_state(**_fields(batch))["state"][
        "motion_state_normalized"
    ]
    cyclic = deepcopy(batch)
    for name in ("history_obs_rel_m", "history_obs_mask", "history_primary_mask"):
        cyclic[name] = torch.roll(cyclic[name], 1, dims=2)
    torch.testing.assert_close(
        model.estimate_motion_state(**_fields(cyclic))["state"][
            "motion_state_normalized"
        ], reference, rtol=2e-5, atol=4e-6,
    )
    reflected = deepcopy(batch)
    order = torch.tensor([0, 3, 2, 1])
    for name in ("history_obs_rel_m", "history_obs_mask", "history_primary_mask"):
        reflected[name] = reflected[name].index_select(2, order)
    reflected["history_switch_step"] = -reflected["history_switch_step"]
    torch.testing.assert_close(
        model.estimate_motion_state(**_fields(reflected))["state"][
            "motion_state_normalized"
        ], reference, rtol=2e-5, atol=4e-6,
    )


def test_separated_translation_cannot_read_pair_but_joint_translation_can() -> None:
    torch.manual_seed(7)
    common = torch.randn(2, 3, 128, requires_grad=True)
    pair = torch.randn(2, 3, 64, requires_grad=True)
    geometry = torch.randn(2, 64, requires_grad=True)
    available = torch.ones(2, 3, 4, dtype=torch.bool)
    pair_available = torch.ones(2, 3, dtype=torch.bool)
    reliability = torch.randn(2, 3, 7)
    separated = RigidFlowProbeHead(32, 0.0, variant="separated")
    separated_output = separated(
        common, pair, geometry, available, pair_available, reliability,
    )
    gradient = torch.autograd.grad(
        separated_output["motion_state_normalized"][:, :3].sum(), pair,
        retain_graph=True, allow_unused=True,
    )[0]
    assert gradient is None or torch.count_nonzero(gradient) == 0
    geometry_gradient = torch.autograd.grad(
        separated_output["motion_state_normalized"][:, :3].sum(), geometry,
        allow_unused=True,
    )[0]
    assert geometry_gradient is None or torch.count_nonzero(geometry_gradient) == 0
    joint = RigidFlowProbeHead(32, 0.0, variant="joint")
    joint_output = joint(common, pair, geometry, available, pair_available, reliability)
    joint_gradient = torch.autograd.grad(
        joint_output["motion_state_normalized"][:, :3].sum(), pair,
    )[0]
    assert torch.count_nonzero(joint_gradient) > 0


@pytest.mark.parametrize("kind", ["separated", "joint"])
def test_pair_route_masks_pair_gradient_when_evidence_is_unavailable(kind: str) -> None:
    torch.manual_seed(9)
    head = RigidFlowProbeHead(32, 0.0, variant=kind)
    common = torch.randn(2, 3, 128, requires_grad=True)
    pair = torch.randn(2, 3, 64, requires_grad=True)
    geometry = torch.randn(2, 64)
    available = torch.ones(2, 3, 4, dtype=torch.bool)
    reliability = torch.randn(2, 3, 7)
    unavailable = torch.zeros(2, 3, dtype=torch.bool)
    output = head(common, pair, geometry, available, unavailable, reliability)
    gradient = torch.autograd.grad(
        output["motion_state_normalized"][:, 3].sum(), pair,
        allow_unused=True,
    )[0]
    assert gradient is None or torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize("kind", ["v7_factorized", "separated", "joint"])
def test_common_200_update_train_step_is_finite(kind: str) -> None:
    torch.manual_seed(11)
    model = _model(kind).train()
    batch = _supervised_batch()
    _, loss, components = rigid_flow_probe_train_step(model, batch, 1, 200)
    assert torch.isfinite(loss)
    assert torch.isfinite(components["ramp_yaw_invariance"])
    assert torch.isfinite(components["ramp_translation_equivariance"])
    assert components["state_substage"] == "joint_structural_probe"
    loss.backward()


@pytest.mark.parametrize("kind", ["separated", "joint"])
def test_v8_state_loss_is_finite_and_future_isolated(kind: str) -> None:
    model = _model(kind).train()
    batch = _supervised_batch()
    output = model.estimate_motion_state(**_fields(batch))
    prediction = {**output["history"], **output["state"]}
    loss, components = robust_multiscale_motion_state_loss(prediction, batch)
    poisoned = dict(batch)
    poisoned["future_target_position_m"] = torch.full((2, 9, 3), torch.nan)
    poisoned["q0_relation_m"] = torch.full((2, 4, 3), torch.nan)
    poisoned_loss, _ = robust_multiscale_motion_state_loss(prediction, poisoned)
    torch.testing.assert_close(poisoned_loss, loss)
    assert torch.isfinite(loss)
    assert torch.isfinite(components["scale_aux"])
