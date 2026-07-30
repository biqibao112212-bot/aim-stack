from __future__ import annotations

from copy import deepcopy
import inspect

import torch

from training.stage3.global_flow_closure_future import (
    AnonymousGlobalFlowClosureProbe,
    global_flow_closure_state_loss,
    global_flow_closure_train_step,
)
from training.stage3.finalize_global_flow_closure_probe import (
    _checks,
    _validated_contract_namespace,
)
from training.stage3.tests.test_anonymous_vehicle_motion import _batch
from training.stage3.tests.test_stable_motion_bottleneck_future import (
    _supervised_batch,
)
from training.stage3.train_paired_twist_set_probe import (
    V8_JOINT_CONTROL_CHECKPOINTS,
    V8_JOINT_REACHABLE_STATE_PARAMETERS,
    build_probe_parser,
)
from training.stage3.train_global_flow_closure_probe import _cross_source_index
from training.stage3.train_stable_motion_bottleneck_future import STATE_MODULES


def _model(*, channels: int = 32, dropout: float = 0.0):
    torch.manual_seed(20260730)
    return AnonymousGlobalFlowClosureProbe(
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


def test_v11_keeps_six_field_boundary_and_has_no_local_yaw_vote() -> None:
    expected = {
        "self", "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
        "history_event_mask", "history_time_s", "history_switch_step",
    }
    assert set(inspect.signature(
        AnonymousGlobalFlowClosureProbe.estimate_motion_state,
    ).parameters) == expected
    config = _model().config
    assert config["local_yaw_votes"] is False
    assert config["pair_support_hard_switch"] is False
    assert config["observed_history_decoder"] is True
    assert config["history_decoder_reads_observed_displacement"] is False
    assert config["history_decoder_reads_current_endpoint"] is False
    assert config["analytic_future_decoder"] is False
    for name in (
        "physical_id_input", "motion_class_input", "session_identity_input",
        "truth_state_input",
    ):
        assert config[name] is False


def test_v11_capacity_matches_v8_within_five_percent() -> None:
    model = _model(channels=96)
    count = sum(
        parameter.numel() for name in STATE_MODULES
        for parameter in getattr(model, name).parameters()
    )
    assert count == 1_467_004
    assert abs(count - V8_JOINT_REACHABLE_STATE_PARAMETERS) / (
        V8_JOINT_REACHABLE_STATE_PARAMETERS
    ) < 0.05


def test_v11_finalizer_restores_runtime_default_missing_from_contract() -> None:
    parser = build_probe_parser()
    args = parser.parse_args([
        "--diagnostic-oracle-association", "--allow-mapper-h-mismatch",
        "--dataset", r"D:\仿真\dataset\autoaim-stage3-v1\derived\stage3-multistate-fixed6mm-v2-pnp-sf-20260730-r2",
        "--truth-history", r"D:\仿真\dataset\autoaim-stage3-v1\derived\stage3-multistate-fixed6mm-v2-truth-history-20260730-r2",
        "--mapper-checkpoint", r"D:\仿真\models\engines\stage3-training\20260727-v41-a4-aligned-anchored-window-capacity-r1\epoch-0002-update-000264.pt",
        "--s-checkpoint", r"D:\仿真\models\engines\stage3-training\20260724-v19-anchor-edge-state-restorer-120ep-seed0-r2\stage3-cyclic-anchor-edge-restorer-seed0-epoch110.pt",
        "--h-checkpoint", r"D:\仿真\models\engines\stage3-training\20260727-v35-a3-h0-warm-only-full-r1\diagnostic-sealed-epoch-0032-update-004224.pt",
        "--v77-control-checkpoint", r"D:\仿真\models\engines\stage3-training\20260730-v77-v5-multistate-latest32-full-r1\checkpoints\checkpoint-update-000800.pt",
        "--v8-joint-control-checkpoint", str(V8_JOINT_CONTROL_CHECKPOINTS[20260730]),
        "--output", "unused", "--seed", "20260730",
    ])
    contract_args = {
        name: value for name, value in vars(args).items()
        if name not in {"resume_checkpoint", "stop_after_update"}
    }
    assert "stop_after_update" not in contract_args
    restored = _validated_contract_namespace(contract_args)
    assert restored.stop_after_update == 0


@torch.inference_mode()
def test_v11_typed_history_decoder_has_block_sparse_state_incidence() -> None:
    model = _model().eval()
    history = model.context(**_fields(_batch()))
    head = model.motion_state_head
    hg, ht, pg, pt = head._prior_contexts(
        history["_handle_geometry_raw"],
        history["_handle_kinematics_raw"],
        history["_pair_geometry_raw"],
        history["_pair_kinematics_raw"],
    )
    changed_handle_geometry = history["_handle_geometry_raw"].clone()
    changed_handle_geometry[..., 9:12] += 123.0
    changed_hg, changed_ht, _, _ = head._prior_contexts(
        changed_handle_geometry, history["_handle_kinematics_raw"],
        history["_pair_geometry_raw"], history["_pair_kinematics_raw"],
    )
    torch.testing.assert_close(changed_hg, hg, rtol=0, atol=0)
    torch.testing.assert_close(changed_ht, ht, rtol=0, atol=0)
    zero = torch.zeros(hg.shape[0], 4)
    zero_handle, zero_pair = head._decode_history(zero, hg, ht, pg, pt)
    assert torch.count_nonzero(zero_handle) == 0
    assert torch.count_nonzero(zero_pair) == 0

    velocity = zero.clone()
    velocity[:, :3] = torch.tensor([0.3, -0.2, 0.1])
    velocity_handle, velocity_pair = head._decode_history(
        velocity, hg, ht, pg, pt,
    )
    assert torch.count_nonzero(velocity_handle) > 0
    torch.testing.assert_close(velocity_pair, zero_pair, rtol=0, atol=0)

    omega = zero.clone()
    omega[:, 3] = 0.4
    omega_handle, omega_pair = head._decode_history(omega, hg, ht, pg, pt)
    assert torch.count_nonzero(omega_handle) > 0
    assert torch.count_nonzero(omega_pair) > 0
    no_geometry_handle, no_geometry_pair = head._decode_history(
        omega, torch.zeros_like(hg), ht, torch.zeros_like(pg), pt,
    )
    assert torch.count_nonzero(no_geometry_handle) == 0
    assert torch.count_nonzero(no_geometry_pair) == 0


@torch.inference_mode()
def test_v11_pair_free_path_has_no_pair_branch_constant() -> None:
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
    width = model.motion_state_head.width
    zeros = torch.zeros(2, width)
    assert torch.count_nonzero(model.motion_state_head.pair_initial_yaw(zeros)) == 0
    assert torch.count_nonzero(model.motion_state_head.pair_yaw_update(zeros)) == 0


@torch.inference_mode()
def test_v11_pair_messages_require_geometry_motion_and_time_together() -> None:
    head = _model().eval().motion_state_head
    batch, factors = 2, 5
    initial_inputs = (
        torch.randn(batch, factors, 12),
        torch.randn(batch, factors, 6),
        torch.randn(batch, factors, 7),
    )
    residual_inputs = (
        torch.randn(batch, factors, 6),
        torch.randn(batch, factors, 3),
        torch.randn(batch, factors, 6),
    )
    for encoder, values in (
        (head.pair_initial_encoder, initial_inputs),
        (head.pair_residual_encoder, residual_inputs),
    ):
        assert torch.count_nonzero(encoder(*values)) > 0
        for index in range(3):
            changed = list(values)
            changed[index] = torch.zeros_like(changed[index])
            assert torch.count_nonzero(encoder(*changed)) == 0


@torch.inference_mode()
def test_v11_is_c4_and_handle_reflection_invariant() -> None:
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
def test_v11_separate_geometry_interventions_preserve_support() -> None:
    model = _model().eval()
    fields = _fields(_batch())
    normal = model.estimate_motion_state(**fields)
    handle = model.estimate_motion_state_broken_handle_geometry(**fields)
    pair = model.estimate_motion_state_broken_pair_geometry(**fields)
    for changed in (handle, pair):
        for name in (
            "pair_flow_available", "pair_flow_edge_valid", "history_active_count",
        ):
            assert torch.equal(normal["history"][name], changed["history"][name])
        assert torch.count_nonzero(
            normal["state"]["motion_state_normalized"]
            - changed["state"]["motion_state_normalized"]
        ) > 0


@torch.inference_mode()
def test_v11_crossed_rotation_factors_keep_target_common_and_donor_differential() -> None:
    model = _model().eval()
    fields = _fields(_batch())
    history = model.context(**fields)
    source = torch.tensor([1, 0])
    crossed = model._cross_rotation_factors(
        history, source, event_count=fields["history_event_mask"].shape[1],
        history_scale_s=float(model.context.history_scale_s),
    )
    events = fields["history_event_mask"].shape[1]
    valid = history["_handle_raw_valid"].reshape(2, 4, events, 3)
    weight = valid.unsqueeze(-1).to(torch.float32)
    target = history["_handle_kinematics_raw"].reshape(
        2, 4, events, 3, 14,
    )
    actual = crossed["_handle_kinematics_raw"].reshape(
        2, 4, events, 3, 14,
    )
    donor_valid = valid.index_select(0, source)
    target_common_rate = (
        target[..., 3:6] * weight
    ).sum(dim=(1, 2, 3)) / weight.sum(dim=(1, 2, 3)).clamp_min(1)
    elapsed = 0.01 * torch.expm1(actual[..., 6])
    donor_geometry = history["_handle_geometry_raw"].reshape(
        2, 4, events, 3, 12,
    ).index_select(0, source)
    common_delta = (
        target_common_rate[:, None, None, None]
        * (elapsed / float(model.context.history_scale_s)).unsqueeze(-1)
    )
    torch.testing.assert_close(
        (actual[..., :3] - common_delta)[donor_valid],
        (donor_geometry[..., :3] - donor_geometry[..., 3:6])[donor_valid],
        rtol=1e-5, atol=1e-6,
    )
    torch.testing.assert_close(
        actual[..., :3][donor_valid],
        (actual[..., 3:6] * (
            elapsed / float(model.context.history_scale_s)
        ).unsqueeze(-1))[donor_valid],
        rtol=1e-5, atol=1e-6,
    )
    crossed_geometry = crossed["_handle_geometry_raw"].reshape(
        2, 4, events, 3, 12,
    )
    torch.testing.assert_close(
        actual[..., :3][donor_valid],
        (crossed_geometry[..., 6:9] - crossed_geometry[..., 9:12])[donor_valid],
        rtol=1e-5, atol=1e-6,
    )
    torch.testing.assert_close(
        crossed["_pair_geometry_raw"],
        history["_pair_geometry_raw"].index_select(0, source),
    )
    torch.testing.assert_close(
        crossed["_pair_kinematics_raw"],
        history["_pair_kinematics_raw"].index_select(0, source),
    )
    broken = model._break_crossed_rotation_pairing(
        crossed, event_count=events,
    )
    assert torch.equal(
        broken["_handle_raw_valid"], crossed["_handle_raw_valid"],
    )
    assert torch.equal(
        broken["_pair_raw_valid"], crossed["_pair_raw_valid"],
    )
    torch.testing.assert_close(
        broken["_handle_kinematics_raw"], crossed["_handle_kinematics_raw"],
    )
    torch.testing.assert_close(
        broken["_pair_kinematics_raw"], crossed["_pair_kinematics_raw"],
    )
    assert torch.count_nonzero(
        broken["_handle_geometry_raw"] - crossed["_handle_geometry_raw"]
    ) > 0
    assert torch.count_nonzero(
        broken["_pair_geometry_raw"] - crossed["_pair_geometry_raw"]
    ) > 0


@torch.inference_mode()
def test_v11_two_shared_closure_refinements_change_the_global_state() -> None:
    model = _model().eval()
    fields = _fields(_batch())
    refined = model.estimate_motion_state(**fields)["state"]
    initial = model.estimate_motion_state_zero_refinement(**fields)["state"]
    assert torch.equal(refined["refinement_steps"], torch.full((2,), 2.0))
    assert torch.equal(initial["refinement_steps"], torch.zeros(2))
    assert torch.count_nonzero(
        refined["motion_state_normalized"]
        - initial["motion_state_normalized"]
    ) > 0


def test_v11_loss_uses_state_and_causal_history_but_not_future() -> None:
    model = _model().train()
    batch = _supervised_batch()
    output = model.estimate_motion_state(**_fields(batch))
    prediction = {**output["history"], **output["state"]}
    loss, components = global_flow_closure_state_loss(prediction, batch)
    poisoned = dict(batch)
    poisoned["future_target_position_m"] = torch.full((2, 9, 3), torch.nan)
    poisoned["q0_relation_m"] = torch.full((2, 4, 3), torch.nan)
    actual, _ = global_flow_closure_state_loss(prediction, poisoned)
    torch.testing.assert_close(actual, loss)
    assert torch.isfinite(components["handle_history_closure"])
    assert torch.isfinite(components["pair_history_closure"])


def test_v11_train_step_reaches_all_state_parameters() -> None:
    model = _model(dropout=0.05).train()
    batch = _supervised_batch()
    _, loss, components = global_flow_closure_train_step(model, batch, 1, 200)
    assert torch.isfinite(loss)
    assert components["state_substage"] == "global_flow_closure_structural_probe"
    loss.backward()
    for name in STATE_MODULES:
        assert all(
            parameter.grad is not None
            for parameter in getattr(model, name).parameters()
            if parameter.requires_grad
        )


def test_v11_gate_requires_pair_groups_crossed_state_and_closure() -> None:
    distribution = lambda mean: {"mean_m": mean}
    control = {
        "overall_velocity_mean_mps": 0.50,
        "combined_velocity_mean_mps": 0.80,
        "high_speed_combined_velocity_mean_mps": 1.40,
        "overall_yaw_mean_rad_s": 1.90,
        "combined_yaw_mean_rad_s": 2.40,
        "high_speed_combined_yaw_mean_rad_s": 4.00,
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
        "broken_handle_overall_velocity_mean": 0.50,
        "zero_refinement_combined_velocity_mean": 0.80,
        "zero_refinement_combined_yaw_mean": 2.70,
        "crossed_rotation_factors": {
            "velocity_error_to_translation_source": distribution(0.30),
            "velocity_error_to_rotation_donor": distribution(1.00),
            "yaw_error_to_translation_source": distribution(1.00),
            "yaw_error_to_rotation_donor": distribution(0.40),
            "yaw_sign_accuracy_to_rotation_donor": 0.90,
            "broken_pairing_velocity_error_to_hybrid": distribution(0.40),
            "broken_pairing_yaw_error_to_hybrid": distribution(0.50),
            "intact_history_closure_error": distribution(0.10),
            "broken_pairing_history_closure_error": distribution(0.13),
        },
    }
    for count in (1, 2):
        candidate.update({
            f"pair{count}_velocity_mean": 0.80,
            f"control_pair{count}_velocity_mean": 0.90,
            f"pair{count}_yaw_mean": 3.50,
            f"control_pair{count}_yaw_mean": 4.00,
            f"broken_pair_pair{count}_yaw_mean": 3.40,
        })
    candidate.update({
        "pair3_velocity_mean": 0.90,
        "control_pair3_velocity_mean": 0.90,
        "pair3_yaw_mean": 2.00,
        "control_pair3_yaw_mean": 2.00,
        "broken_pair_pair3_yaw_mean": 2.30,
    })
    assert all(item["passed"] for item in _checks(candidate, control).values())
    bypass = deepcopy(candidate)
    bypass["crossed_rotation_factors"] = dict(
        candidate["crossed_rotation_factors"],
        broken_pairing_history_closure_error=distribution(0.105),
    )
    assert not _checks(bypass, control)[
        "broken_crossed_pairing_worsens_history_closure"
    ]["passed"]


def test_v11_cross_source_matches_discrete_history_and_pair_support() -> None:
    handle_valid = torch.tensor([
        [True, False, True], [False, True, True],
    ])
    pair_valid = torch.tensor([
        [True, False, False, True, False, False],
        [True, False, False, False, False, False],
    ])
    active_count = torch.tensor([32, 32])
    target = torch.tensor([
        [0.0, 0.0, 0.0, -2.0], [1.0, 0.0, 0.0, 2.0],
    ])
    source, selected = _cross_source_index(
        handle_valid, pair_valid, active_count,
        target, torch.ones(2, dtype=torch.bool),
    )
    assert torch.equal(source, torch.tensor([1, 0]))
    assert bool(selected.all())
    active_count[1] = 24
    source, selected = _cross_source_index(
        handle_valid, pair_valid, active_count,
        target, torch.ones(2, dtype=torch.bool),
    )
    assert torch.equal(source, torch.arange(2))
    assert not bool(selected.any())
