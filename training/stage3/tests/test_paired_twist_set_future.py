from __future__ import annotations

from copy import deepcopy
import inspect
import sys
from types import SimpleNamespace

import torch

from training.stage3.joint_rigid_flow_probe import AnonymousJointTwistProbe
from training.stage3.paired_twist_set_future import (
    AnonymousPairedTwistSetProbe,
    AnonymousPairedTwistTokenContext,
    paired_twist_probe_train_step,
    paired_twist_state_loss,
)
from training.stage3.tests.test_anonymous_vehicle_motion import _batch
from training.stage3.tests.test_stable_motion_bottleneck_future import _supervised_batch
from training.stage3.train_stable_motion_bottleneck_future import (
    STATE_MODULES,
    _callable_contract,
)
from training.stage3.train_paired_twist_set_probe import (
    LOCKED_VALUES,
    PROBE_SEEDS,
    V8_EXPECTED_PATHS,
    V8_JOINT_CONTROL_CHECKPOINTS,
    V8_JOINT_REACHABLE_STATE_PARAMETERS,
    V8_JOINT_TOTAL_STATE_PARAMETERS,
    _validate_args,
    build_probe_parser,
)
from training.stage3.finalize_paired_twist_set_probe import _checks


def _model(*, dropout: float = 0.0, channels: int = 32):
    torch.manual_seed(20260730)
    return AnonymousPairedTwistSetProbe(
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


def test_v9_api_is_exactly_six_fields_and_forbids_identity_inputs() -> None:
    expected = {
        "self", "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
        "history_event_mask", "history_time_s", "history_switch_step",
    }
    assert set(inspect.signature(
        AnonymousPairedTwistTokenContext.forward,
    ).parameters) == expected
    assert set(inspect.signature(
        AnonymousPairedTwistSetProbe.estimate_motion_state,
    ).parameters) == expected
    config = _model().config
    for field in (
        "physical_id_input", "motion_class_input", "session_identity_input",
        "truth_state_input",
    ):
        assert config[field] is False
    assert config["motion_context"]["long_projective_yaw_lags"] is False
    assert config["motion_context"]["early_geometry_velocity_pooling"] is False
    assert config["per_coordinate_scale_selection"] is False


def test_callable_contract_canonicalizes_python_m_entrypoint(monkeypatch) -> None:
    def diagnostic_hook() -> None:
        return None

    diagnostic_hook.__module__ = "__main__"
    main_module = sys.modules["__main__"]
    monkeypatch.setattr(
        main_module, "__spec__",
        SimpleNamespace(name="training.stage3.synthetic_probe"),
        raising=False,
    )
    contract = _callable_contract(diagnostic_hook)
    assert contract is not None
    assert contract["module"] == "training.stage3.synthetic_probe"


def test_v9_reachable_state_capacity_matches_v8_joint_within_five_percent() -> None:
    kwargs = dict(
        velocity_scale_mps=(3.0, 3.2, 0.25), yaw_rate_scale_rad_s=17.25,
        channels=96, dropout=0.05, message_layers=3, basis_count=8,
    )
    v8 = AnonymousJointTwistProbe(**kwargs)
    v9 = AnonymousPairedTwistSetProbe(**kwargs)
    count = lambda model: sum(
        parameter.numel() for name in STATE_MODULES
        for parameter in getattr(model, name).parameters()
    )
    assert count(v8) == V8_JOINT_TOTAL_STATE_PARAMETERS
    assert (
        abs(count(v9) - V8_JOINT_REACHABLE_STATE_PARAMETERS)
        / V8_JOINT_REACHABLE_STATE_PARAMETERS < 0.05
    )


def test_v9_probe_parser_locks_200_updates_and_same_seed_control() -> None:
    parser = build_probe_parser()
    assert parser.get_default("motion_state_updates") == 200
    seed = PROBE_SEEDS[0]
    paths = dict(V8_EXPECTED_PATHS)
    paths["v8_joint_control_checkpoint"] = V8_JOINT_CONTROL_CHECKPOINTS[seed]
    valid = SimpleNamespace(
        motion_state_updates=200, trajectory_updates=0, selector_updates=0,
        decoder_joint_updates=0, stop_after_update=0, seed=seed,
        diagnostic_oracle_association=True, allow_mapper_h_mismatch=True,
        **LOCKED_VALUES,
        **{name: str(path) for name, path in paths.items()},
    )
    _validate_args(valid)
    invalid = SimpleNamespace(**vars(valid))
    invalid.v8_joint_control_checkpoint = str(
        V8_JOINT_CONTROL_CHECKPOINTS[PROBE_SEEDS[1]]
    )
    try:
        _validate_args(invalid)
    except ValueError as error:
        assert "artifact contract differs" in str(error)
    else:
        raise AssertionError("v9 accepted the wrong-seed V8 control")


def test_v9_gate_requires_velocity_core_and_pairing_causality() -> None:
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
        "broken_pairing_high_speed_velocity_mean_mps": 1.31,
    }
    assert all(item["passed"] for item in _checks(candidate, control).values())
    broken = dict(candidate, broken_pairing_high_speed_velocity_mean_mps=1.20)
    assert not _checks(broken, control)[
        "broken_pairing_worsens_high_speed_velocity"
    ]["passed"]
    zero = dict(
        candidate,
        high_speed_combined_velocity_mean_mps=0.0,
        broken_pairing_high_speed_velocity_mean_mps=0.0,
    )
    assert not _checks(zero, control)[
        "broken_pairing_worsens_high_speed_velocity"
    ]["passed"]


@torch.inference_mode()
def test_v9_is_c4_and_handle_reflection_invariant() -> None:
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
def test_v9_head_is_token_permutation_invariant_and_uses_one_expert_weight() -> None:
    model = _model().eval()
    history = model.context(**_fields(_batch()))
    head = model.motion_state_head
    reference = head(
        history["_token"], history["_token_valid"],
        history["_local_token_valid"], history["_steady_token_valid"],
        history["_handle_token_valid"], history["_expert_available"],
        history["_router_stats"],
    )
    permutation = torch.arange(history["_token"].shape[1] - 1, -1, -1)
    actual = head(
        history["_token"][:, permutation],
        history["_token_valid"][:, permutation],
        history["_local_token_valid"][:, permutation],
        history["_steady_token_valid"][:, permutation],
        history["_handle_token_valid"][:, permutation],
        history["_expert_available"],
        history["_router_stats"],
    )
    torch.testing.assert_close(
        actual["motion_state_normalized"], reference["motion_state_normalized"],
        rtol=2e-5, atol=4e-6,
    )
    reconstructed = (
        reference["expert_weight"].unsqueeze(-1)
        * reference["expert_motion_state_normalized"]
    ).sum(dim=1)
    torch.testing.assert_close(reconstructed, reference["motion_state_normalized"])
    assert reference["expert_weight"].shape == (2, 3)


def test_v9_expert_observability_masks_and_steady_translation_path_are_real() -> None:
    model = _model().eval()
    history = model.context(**_fields(_batch()))
    expected_steady = (
        (history["history_active_count"] == 32)
        & history["pair_flow_available"].all(dim=1)
    )
    expected_fallback = ~history["pair_flow_available"].any(dim=1)
    assert torch.equal(history["_expert_available"][:, 1], expected_steady)
    assert torch.equal(history["_expert_available"][:, 2], expected_fallback)
    assert torch.all(history["_local_token_valid"] <= history["_token_valid"])
    assert torch.all(history["_local_token_valid"].any(dim=1))

    token = history["_token"].detach().requires_grad_(True)
    # Force the already well-defined masks available only to inspect the
    # architectural path; formal routing still uses observation-only support.
    available = torch.ones(token.shape[0], 3, dtype=torch.bool)
    output = model.motion_state_head(
        token, history["_token_valid"], history["_local_token_valid"],
        history["_steady_token_valid"], history["_handle_token_valid"],
        available, history["_router_stats"],
    )
    steady_xy = output["expert_motion_state_normalized"][:, 1, :2].sum()
    gradient = torch.autograd.grad(steady_xy, token)[0]
    assert torch.count_nonzero(gradient[history["_handle_token_valid"]]) > 0


@torch.inference_mode()
def test_pairing_intervention_preserves_support_but_changes_paired_tokens() -> None:
    context = _model().context.eval()
    batch = _batch()
    intact = context(**_fields(batch))
    broken = context.forward_broken_pairing(**_fields(batch))
    for name in (
        "_token_valid", "_handle_token_valid", "_pair_token_valid",
        "_local_token_valid", "_steady_token_valid", "_expert_available",
        "pair_flow_available", "pair_flow_edge_valid", "history_active_count",
    ):
        assert torch.equal(broken[name], intact[name])
    valid = intact["_token_valid"]
    assert torch.count_nonzero(
        (broken["_token"] - intact["_token"])[valid]
    ) > 0


def test_grouped_pairing_intervention_preserves_each_stream_scale_marginal() -> None:
    geometry = torch.arange(1 * 4 * 6 * 3 * 2, dtype=torch.float32).reshape(
        1, 4, 6, 3, 2,
    )
    valid = torch.ones(1, 4, 6, 3, dtype=torch.bool)
    valid[:, :, 0, 1] = False
    actual = AnonymousPairedTwistTokenContext._roll_grouped_geometry(
        geometry, valid,
    )
    for handle in range(4):
        for scale in range(3):
            mask = valid[0, handle, :, scale]
            expected_rows = geometry[0, handle, mask, scale]
            actual_rows = actual[0, handle, mask, scale]
            torch.testing.assert_close(
                actual_rows.sort(dim=0).values,
                expected_rows.sort(dim=0).values,
            )


@torch.inference_mode()
def test_pairing_intervention_is_c4_and_reflection_invariant() -> None:
    model = _model().eval()
    batch = _batch()
    reference = model.estimate_motion_state_broken_pairing(**_fields(batch))["state"][
        "motion_state_normalized"
    ]
    cyclic = deepcopy(batch)
    for name in ("history_obs_rel_m", "history_obs_mask", "history_primary_mask"):
        cyclic[name] = torch.roll(cyclic[name], 1, dims=2)
    torch.testing.assert_close(
        model.estimate_motion_state_broken_pairing(**_fields(cyclic))["state"][
            "motion_state_normalized"
        ], reference, rtol=2e-5, atol=4e-6,
    )
    reflected = deepcopy(batch)
    order = torch.tensor([0, 3, 2, 1])
    for name in ("history_obs_rel_m", "history_obs_mask", "history_primary_mask"):
        reflected[name] = reflected[name].index_select(2, order)
    reflected["history_switch_step"] = -reflected["history_switch_step"]
    torch.testing.assert_close(
        model.estimate_motion_state_broken_pairing(**_fields(reflected))["state"][
            "motion_state_normalized"
        ], reference, rtol=2e-5, atol=4e-6,
    )


def test_v9_unified_loss_ignores_future_and_unsupervised_expert_proposals() -> None:
    model = _model().train()
    batch = _supervised_batch()
    output = model.estimate_motion_state(**_fields(batch))
    prediction = {**output["history"], **output["state"]}
    loss, components = paired_twist_state_loss(prediction, batch)
    poisoned_batch = dict(batch)
    poisoned_batch["future_target_position_m"] = torch.full((2, 9, 3), torch.nan)
    poisoned_batch["q0_relation_m"] = torch.full((2, 4, 3), torch.nan)
    poisoned_prediction = dict(prediction)
    poisoned_prediction["expert_motion_state_normalized"] = torch.full(
        (2, 3, 4), torch.nan,
    )
    poisoned_loss, _ = paired_twist_state_loss(
        poisoned_prediction, poisoned_batch,
    )
    torch.testing.assert_close(poisoned_loss, loss)
    assert torch.isfinite(components["planar_velocity"])
    assert torch.isfinite(components["vertical_velocity"])


def test_v9_common_200_update_train_step_is_finite_and_reaches_both_token_encoders() -> None:
    model = _model(dropout=0.05).train()
    batch = _supervised_batch()
    _, loss, components = paired_twist_probe_train_step(model, batch, 1, 200)
    assert torch.isfinite(loss)
    assert torch.isfinite(components["ramp_yaw_invariance"])
    assert torch.isfinite(components["ramp_translation_equivariance"])
    assert components["state_substage"] == "paired_twist_structural_probe"
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.context.handle_encoder.parameters())
    assert all(parameter.grad is not None for parameter in model.context.pair_encoder.parameters())
