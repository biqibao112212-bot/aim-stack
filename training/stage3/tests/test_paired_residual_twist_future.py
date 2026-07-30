from __future__ import annotations

from copy import deepcopy
import inspect

import torch

from training.stage3.finalize_paired_residual_twist_probe import _checks
from training.stage3.paired_residual_twist_future import (
    AnonymousPairedResidualTwistProbe,
    paired_residual_probe_train_step,
    paired_residual_state_loss,
)
from training.stage3.tests.test_anonymous_vehicle_motion import _batch
from training.stage3.tests.test_stable_motion_bottleneck_future import _supervised_batch
from training.stage3.train_paired_twist_set_probe import (
    V8_JOINT_REACHABLE_STATE_PARAMETERS,
)
from training.stage3.train_stable_motion_bottleneck_future import STATE_MODULES


def _model(*, dropout: float = 0.0, channels: int = 32):
    torch.manual_seed(20260730)
    return AnonymousPairedResidualTwistProbe(
        velocity_scale_mps=(3.0, 3.2, 0.25),
        yaw_rate_scale_rad_s=17.25,
        channels=channels,
        dropout=dropout,
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


def test_v10_keeps_six_field_boundary_and_removes_complete_4d_router() -> None:
    expected = {
        "self", "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
        "history_event_mask", "history_time_s", "history_switch_step",
    }
    assert set(inspect.signature(
        AnonymousPairedResidualTwistProbe.estimate_motion_state,
    ).parameters) == expected
    config = _model().config
    assert config["complete_4d_expert_mixture"] is False
    assert config["learned_expert_router"] is False
    assert config["pair1_pair2_pair3_all_consumed"] is True
    assert config["analytic_future_decoder"] is False
    for field in (
        "physical_id_input", "motion_class_input", "session_identity_input",
        "truth_state_input",
    ):
        assert config[field] is False


def test_v10_reachable_capacity_matches_v8_within_five_percent() -> None:
    model = _model(channels=96)
    count = sum(
        parameter.numel() for name in STATE_MODULES
        for parameter in getattr(model, name).parameters()
    )
    assert count == 1_498_568
    assert abs(count - V8_JOINT_REACHABLE_STATE_PARAMETERS) / (
        V8_JOINT_REACHABLE_STATE_PARAMETERS
    ) < 0.05


@torch.inference_mode()
def test_v10_consumes_every_available_pair_bundle_without_pair3_gate() -> None:
    model = _model().eval()
    output = model.estimate_motion_state(**_fields(_batch()))
    state = output["state"]
    valid = state["pair_bundle_valid"]
    reliability = state["pair_yaw_reliability"]
    assert torch.equal(state["pair_supported"], valid.any(dim=1))
    assert torch.all(reliability[~valid] == 0)
    torch.testing.assert_close(
        reliability.sum(dim=1),
        state["pair_supported"].to(reliability.dtype),
    )
    assert torch.count_nonzero(valid) > 0


@torch.inference_mode()
def test_v10_is_c4_and_handle_reflection_invariant() -> None:
    model = _model().eval()
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


@torch.inference_mode()
def test_v10_pair_free_path_has_structurally_zero_rotation_residual() -> None:
    model = _model().eval()
    batch = deepcopy(_batch())
    primary = batch["history_primary_mask"].to(torch.bool)
    batch["history_obs_mask"] = primary
    batch["history_obs_rel_m"] = torch.where(
        primary.unsqueeze(-1), batch["history_obs_rel_m"],
        torch.zeros_like(batch["history_obs_rel_m"]),
    )
    normal = model.estimate_motion_state(**_fields(batch))["state"]
    zero = model.estimate_motion_state_zero_rotation_residual(
        **_fields(batch)
    )["state"]
    assert not bool(normal["pair_supported"].any())
    assert torch.count_nonzero(
        normal["paired_rotation_residual_normalized"]
    ) == 0
    torch.testing.assert_close(
        normal["motion_state_normalized"], zero["motion_state_normalized"],
    )


@torch.inference_mode()
def test_v10_zero_residual_never_changes_vertical_velocity() -> None:
    model = _model().eval()
    fields = _fields(_batch())
    normal = model.estimate_motion_state(**fields)["state"]
    zero = model.estimate_motion_state_zero_rotation_residual(**fields)["state"]
    torch.testing.assert_close(
        normal["motion_state_normalized"][:, 2],
        zero["motion_state_normalized"][:, 2],
    )
    assert torch.count_nonzero(
        normal["motion_state_normalized"][:, :2]
        - zero["motion_state_normalized"][:, :2]
    ) > 0


@torch.inference_mode()
def test_v10_broken_pairing_preserves_support_but_changes_state() -> None:
    model = _model().eval()
    fields = _fields(_batch())
    normal = model.estimate_motion_state(**fields)
    broken = model.estimate_motion_state_broken_pairing(**fields)
    for name in (
        "pair_flow_available", "pair_flow_edge_valid", "history_active_count",
    ):
        assert torch.equal(normal["history"][name], broken["history"][name])
    assert torch.equal(
        normal["state"]["pair_bundle_valid"],
        broken["state"]["pair_bundle_valid"],
    )
    assert torch.count_nonzero(
        normal["state"]["motion_state_normalized"]
        - broken["state"]["motion_state_normalized"]
    ) > 0


@torch.inference_mode()
def test_v10_broken_pairing_changes_only_raw_geometry_before_v10_head() -> None:
    context = _model().context.eval()
    fields = _fields(_batch())
    normal = context(**fields)
    broken = context.forward_broken_pairing(**fields)
    for name in (
        "_handle_kinematics_raw", "_pair_kinematics_raw",
        "_handle_raw_valid", "_pair_raw_valid",
    ):
        assert torch.equal(normal[name], broken[name])
    for geometry, valid in (
        ("_handle_geometry_raw", "_handle_raw_valid"),
        ("_pair_geometry_raw", "_pair_raw_valid"),
    ):
        assert torch.count_nonzero(
            (normal[geometry] - broken[geometry])[normal[valid]]
        ) > 0


@torch.inference_mode()
def test_v10_paired_path_is_strictly_zero_preserving() -> None:
    model = _model().eval()
    for name in (
        "handle_geometry_projection", "handle_kinematics_projection",
        "pair_geometry_projection", "pair_kinematics_projection",
    ):
        getattr(model.motion_state_head, name).weight.zero_()
    batch = _batch()
    reference = model.estimate_motion_state(**_fields(batch))["state"]
    assert torch.count_nonzero(reference["pair_yaw_vote_normalized"]) == 0
    assert torch.count_nonzero(
        reference["paired_rotation_residual_normalized"]
    ) == 0
    changed = deepcopy(batch)
    ramp = torch.tensor([0.7, -0.4, 0.0])
    changed["history_obs_rel_m"] = changed["history_obs_rel_m"] + (
        changed["history_time_s"][:, :, None, None] * ramp[None, None, None]
    )
    actual = model.estimate_motion_state(**_fields(changed))["state"]
    assert torch.count_nonzero(actual["pair_yaw_vote_normalized"]) == 0
    assert torch.count_nonzero(
        actual["paired_rotation_residual_normalized"]
    ) == 0


def test_v10_loss_reads_only_state_and_yaw_identifiable_auxiliary() -> None:
    model = _model().train()
    batch = _supervised_batch()
    output = model.estimate_motion_state(**_fields(batch))
    prediction = {**output["history"], **output["state"]}
    loss, components = paired_residual_state_loss(prediction, batch)
    poisoned = dict(batch)
    poisoned["future_target_position_m"] = torch.full((2, 9, 3), torch.nan)
    poisoned["q0_relation_m"] = torch.full((2, 4, 3), torch.nan)
    actual, _ = paired_residual_state_loss(prediction, poisoned)
    torch.testing.assert_close(actual, loss)
    assert torch.isfinite(components["pair_yaw_aux"])
    assert torch.isfinite(components["yaw_calibration"])


def test_v10_train_step_reaches_all_state_parameters_and_freezes_future() -> None:
    model = _model(dropout=0.05).train()
    batch = _supervised_batch()
    _, loss, components = paired_residual_probe_train_step(model, batch, 1, 200)
    assert torch.isfinite(loss)
    assert components["state_substage"] == "paired_residual_structural_probe"
    loss.backward()
    for name in STATE_MODULES:
        assert all(
            parameter.grad is not None
            for parameter in getattr(model, name).parameters()
            if parameter.requires_grad
        )


def test_v10_gate_requires_pair12_and_rotation_residual_causality() -> None:
    control = {
        "overall_velocity_mean_mps": 0.50,
        "combined_velocity_mean_mps": 0.80,
        "high_speed_combined_velocity_mean_mps": 1.40,
        "overall_yaw_mean_rad_s": 1.9,
        "combined_yaw_mean_rad_s": 2.4,
        "high_speed_combined_yaw_mean_rad_s": 4.0,
        "overall_yaw_sign_accuracy": 0.96,
    }
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
        "pair1_2_velocity_mean_mps": 0.80,
        "control_pair1_2_velocity_mean_mps": 0.90,
        "pair1_2_yaw_mean_rad_s": 3.50,
        "control_pair1_2_yaw_mean_rad_s": 4.00,
        "broken_pairing_high_speed_velocity_mean_mps": 1.31,
        "broken_pairing_pair1_2_velocity_mean_mps": 0.86,
        "broken_pairing_pair1_2_yaw_mean_rad_s": 3.90,
        "zero_residual_combined_velocity_mean_mps": 0.78,
        "zero_residual_high_speed_velocity_mean_mps": 1.20,
    }
    assert all(item["passed"] for item in _checks(candidate, control).values())
    no_residual = dict(
        candidate,
        zero_residual_combined_velocity_mean_mps=0.68,
        zero_residual_high_speed_velocity_mean_mps=1.19,
    )
    assert not _checks(no_residual, control)[
        "paired_rotation_residual_is_used"
    ]["passed"]
    weak_pair12 = dict(candidate, pair1_2_yaw_mean_rad_s=3.70)
    assert not _checks(weak_pair12, control)[
        "combined_pair1_2_yaw_improves_at_least_10_percent"
    ]["passed"]
    bypass_pair12 = dict(
        candidate,
        broken_pairing_pair1_2_velocity_mean_mps=0.81,
        broken_pairing_pair1_2_yaw_mean_rad_s=3.55,
    )
    assert not _checks(bypass_pair12, control)[
        "broken_pairing_worsens_pair1_2_velocity"
    ]["passed"]
