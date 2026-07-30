from copy import deepcopy
import inspect
from types import SimpleNamespace

import pytest
import torch

from training.stage3.equivariant_alternating_twist_future import (
    AnonymousEquivariantAlternatingTwistProbe,
    EquivariantAlternatingTwistHead,
    _EquivariantOmegaStage,
    _InvariantVelocityStage,
    _PerStreamOrderedSummary,
    _reflect_physical_training_batch,
    alternating_substage_loss,
    equivariant_alternating_train_step,
)
from training.stage3.factorized_common_relative_motion_future import (
    apply_common_velocity_ramp,
)
from training.stage3.tests.test_stable_motion_bottleneck_future import (
    _supervised_batch,
)
from training.stage3.train_omega_first_ordered_closure_probe import (
    _synthetic_twist_history_on_target_support,
)
from training.stage3.train_paired_twist_set_probe import (
    V8_JOINT_REACHABLE_STATE_PARAMETERS,
)
from training.stage3.train_equivariant_alternating_twist_screen import (
    _validate_yaw_preflight_report,
    alternating_dataset_preflight,
    alternating_motion_state_cells,
)
import training.stage3.train_equivariant_alternating_twist_screen as screen_runner
from training.stage3.motion_truth_supervision import MOTION_TARGET_FIELD


def _model(*, channels: int = 32):
    torch.manual_seed(20260731)
    return AnonymousEquivariantAlternatingTwistProbe(
        velocity_scale_mps=(3.0, 3.2, 0.25),
        yaw_rate_scale_rad_s=17.25,
        channels=channels, dropout=0.0,
        message_layers=2 if channels == 32 else 3,
        basis_count=6 if channels == 32 else 8,
    )


def _fields(model, batch):
    return {name: batch[name] for name in model._field_names()}


def test_alternating_keeps_six_field_anonymous_contract() -> None:
    expected = {
        "self", "history_obs_rel_m", "history_obs_mask",
        "history_primary_mask", "history_event_mask", "history_time_s",
        "history_switch_step",
    }
    assert set(inspect.signature(
        AnonymousEquivariantAlternatingTwistProbe.estimate_motion_state,
    ).parameters) == expected
    config = _model().config
    assert config["typed_alternating"] is True
    assert config["strict_omega_first"] is False
    assert config["pre_recurrence_handle_scale_pooling"] is False
    assert config["visible_factor_gauge_is_center_velocity"] is False
    assert config["physical_id_input"] is False
    assert config["truth_state_input"] is False
    assert config["analytic_future_decoder"] is False


def test_each_handle_and_scale_recur_before_anonymous_pooling() -> None:
    summary = _PerStreamOrderedSummary(8).eval()
    token = torch.randn(2, 4, 5, 3, 8)
    valid = torch.ones(2, 4, 5, 3, dtype=torch.bool)
    reference = summary(token, valid)
    reordered = token.clone()
    reordered[:, :, [1, 2]] = reordered[:, :, [2, 1]]
    assert torch.count_nonzero(summary(reordered, valid) - reference) > 0
    permuted_streams = token[:, [2, 0, 3, 1]]
    torch.testing.assert_close(summary(permuted_streams, valid), reference)


def test_pair_primary_swap_canonicalization_preserves_signed_proposal() -> None:
    theta = torch.tensor(0.42)
    elapsed = torch.tensor([[[[0.05]]]])
    prior = torch.tensor([[[[[1.0, 0.0, 0.0]]]]])
    rotated = torch.tensor([[[[[torch.cos(theta), torch.sin(theta), 0.0]]]]])
    valid = torch.ones(1, 1, 1, 1, dtype=torch.bool)
    direct, direct_valid = EquivariantAlternatingTwistHead._signed_angle_proposal(
        prior, rotated, elapsed, valid, 17.25,
    )
    raw_swapped_current = -rotated
    canonical_swapped_current = -raw_swapped_current
    swapped, swapped_valid = EquivariantAlternatingTwistHead._signed_angle_proposal(
        prior, canonical_swapped_current, elapsed, valid, 17.25,
    )
    assert torch.equal(direct_valid, swapped_valid)
    torch.testing.assert_close(swapped, direct)
    torch.testing.assert_close(
        direct.squeeze(), theta / elapsed.squeeze() / 17.25,
    )


@torch.inference_mode()
def test_end_to_end_common_ramp_is_exact_for_both_typed_stages() -> None:
    model = _model().eval()
    batch = _supervised_batch()
    ramp = torch.tensor([[0.5, -0.2, 0.0], [-0.3, 0.4, 0.0]])
    augmented = apply_common_velocity_ramp(
        batch, ramp, model.motion_state_scale,
    )
    original = model.estimate_motion_state(**_fields(model, batch))["state"]
    changed = model.estimate_motion_state(**_fields(model, augmented))["state"]
    for index in (0, 1):
        torch.testing.assert_close(
            changed[f"omega{index}_normalized"],
            original[f"omega{index}_normalized"], rtol=0, atol=1e-7,
        )
        torch.testing.assert_close(
            changed[f"velocity{index}_normalized"]
            - original[f"velocity{index}_normalized"],
            ramp / model.motion_state_scale[:3], rtol=1e-6, atol=1e-7,
        )


@torch.inference_mode()
def test_common_ramp_remains_exact_after_nonzero_learned_corrections() -> None:
    model = _model().eval()
    for name, parameter in model.motion_state_head.named_parameters():
        if "coefficient.2.weight" in name or "fusion_logit.2.weight" in name:
            parameter.normal_(0.0, 0.15)
    batch = _supervised_batch()
    ramp = torch.tensor([[0.25, -0.35, 0.0], [-0.45, 0.15, 0.0]])
    augmented = apply_common_velocity_ramp(
        batch, ramp, model.motion_state_scale,
    )
    original = model.estimate_motion_state(**_fields(model, batch))["state"]
    changed = model.estimate_motion_state(**_fields(model, augmented))["state"]
    for index in (0, 1):
        torch.testing.assert_close(
            changed[f"omega{index}_normalized"],
            original[f"omega{index}_normalized"], rtol=0, atol=1e-7,
        )
        torch.testing.assert_close(
            changed[f"velocity{index}_normalized"]
            - original[f"velocity{index}_normalized"],
            ramp / model.motion_state_scale[:3], rtol=2e-6, atol=2e-7,
        )


@pytest.mark.parametrize(
    ("update", "prefix"),
    ((1, "omega0."), (36, "velocity0."), (56, "omega1."), (81, "velocity1.")),
)
def test_real_train_loss_has_exact_typed_nonzero_gradient_write_set(
    update: int, prefix: str,
) -> None:
    model = _model().train()
    batch = _supervised_batch()
    model.zero_grad(set_to_none=True)
    _, objective, _ = equivariant_alternating_train_step(model, batch, update, 100)
    objective.backward()
    reached = {
        name for name, parameter in model.motion_state_head.named_parameters()
        if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
    }
    assert reached
    assert all(name.startswith(prefix) for name in reached)
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_zero_angle_carrier_still_has_learnable_signed_omega_gradient() -> None:
    model = _model().train()
    batch = _supervised_batch()
    state = model.estimate_motion_state(**_fields(model, batch))["state"]
    assert torch.equal(
        state["omega0_carrier_normalized"],
        torch.zeros_like(state["omega0_carrier_normalized"]),
    )
    model.zero_grad(set_to_none=True)
    _, objective, _ = equivariant_alternating_train_step(model, batch, 1, 100)
    objective.backward()
    gradient = model.motion_state_head.omega0.pair_coefficient[-1].weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0


def test_omega1_is_explicit_detached_omega0_plus_signed_delta() -> None:
    model = _model().eval()
    state = model.estimate_motion_state(
        **_fields(model, _supervised_batch())
    )["state"]
    torch.testing.assert_close(
        state["omega1_normalized"],
        state["omega0_normalized"] + state["omega1_delta_normalized"],
    )


def test_pair_and_handle_are_softly_fused_when_both_are_supported() -> None:
    torch.manual_seed(7)
    stage = _EquivariantOmegaStage(16).eval()
    shape = (2, 1, 3, 1)
    handle_bundle = torch.randn(*shape, 3).clamp(-1, 1)
    pair_bundle = torch.randn(*shape, 3).clamp(-1, 1)
    valid = torch.ones(*shape, dtype=torch.bool)
    result = stage(
        handle_geometry_even=torch.rand(*shape, 9),
        handle_motion_even=torch.rand(*shape, 9),
        handle_time=torch.rand(*shape, 8),
        handle_token_valid=valid,
        handle_bundle=handle_bundle,
        handle_bundle_valid=valid.unsqueeze(-1).expand_as(handle_bundle),
        pair_geometry_even=torch.rand(*shape, 12),
        pair_motion_even=torch.rand(*shape, 6),
        pair_time=torch.rand(*shape, 7),
        pair_token_valid=valid,
        pair_bundle=pair_bundle,
        pair_bundle_valid=valid.unsqueeze(-1).expand_as(pair_bundle),
        base_omega_normalized=torch.zeros(2, 1),
    )
    assert bool((result["evidence_weight"] > 0).all())
    torch.testing.assert_close(
        result["evidence_weight"].sum(dim=1), torch.ones(2),
    )


@pytest.mark.parametrize("supported_source", ("handle", "pair"))
def test_single_supported_source_has_unit_weight_under_extreme_logits(
    supported_source: str,
) -> None:
    class ExtremeLogit(torch.nn.Module):
        def forward(self, value):
            return value.new_tensor([[-100.0, 100.0]]).expand(value.shape[0], -1)

    stage = _EquivariantOmegaStage(16).eval()
    stage.fusion_logit = ExtremeLogit()
    shape = (1, 1, 3, 1)
    handle_valid = torch.full(shape, supported_source == "handle")
    pair_valid = torch.full(shape, supported_source == "pair")
    handle_bundle = torch.ones(*shape, 3)
    pair_bundle = -torch.ones(*shape, 3)
    result = stage(
        handle_geometry_even=torch.ones(*shape, 9),
        handle_motion_even=torch.ones(*shape, 9),
        handle_time=torch.ones(*shape, 8),
        handle_token_valid=handle_valid,
        handle_bundle=handle_bundle,
        handle_bundle_valid=handle_valid.unsqueeze(-1).expand_as(handle_bundle),
        pair_geometry_even=torch.ones(*shape, 12),
        pair_motion_even=torch.ones(*shape, 6),
        pair_time=torch.ones(*shape, 7),
        pair_token_valid=pair_valid,
        pair_bundle=pair_bundle,
        pair_bundle_valid=pair_valid.unsqueeze(-1).expand_as(pair_bundle),
        base_omega_normalized=torch.zeros(1, 1),
    )
    expected = torch.tensor([[1.0, 0.0]]) if supported_source == "handle" else torch.tensor([[0.0, 1.0]])
    assert torch.equal(result["evidence_weight"], expected)


def test_irregular_chord_acceleration_uses_midpoint_time() -> None:
    rate = torch.tensor([[[[[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]]]])
    current = torch.tensor([[[[0.10], [0.30]]]])
    elapsed = torch.tensor([[[[0.10], [0.20]]]])
    valid = torch.ones(1, 1, 2, 1, dtype=torch.bool)
    acceleration, supported = EquivariantAlternatingTwistHead._handle_acceleration(
        rate, current, elapsed, valid, torch.ones_like(valid),
    )
    # Chord midpoints are 0.05 and 0.20, so delta-time is 0.15.
    torch.testing.assert_close(acceleration[0, 0, 1, 0, 0], torch.tensor(2.0 / 0.15))
    assert supported[0, 0, 1, 0]


def test_single_handle_edge_is_not_angularly_observable() -> None:
    rate = torch.zeros(1, 1, 2, 1, 3)
    current = torch.tensor([[[[0.0], [0.1]]]])
    elapsed = torch.tensor([[[[0.0], [0.1]]]])
    valid = torch.tensor([[[[False], [True]]]])
    _, supported = EquivariantAlternatingTwistHead._handle_acceleration(
        rate, current, elapsed, valid, torch.ones_like(valid),
    )
    assert not bool(supported.any())


def test_velocity_wls_exposes_unsupported_denominator() -> None:
    stage = _InvariantVelocityStage(16, history_scale_s=0.35).eval()
    shape = (2, 1, 2, 1)
    output = stage(
        omega_normalized=torch.zeros(2, 1), yaw_scale_rad_s=17.25,
        relative_geometry=torch.zeros(*shape, 9),
        handle_time=torch.zeros(*shape, 8),
        handle_delta=torch.zeros(*shape, 3),
        elapsed_normalized=torch.zeros(*shape),
        handle_valid=torch.zeros(*shape, dtype=torch.bool),
        beta_to_velocity_state=torch.ones(3),
    )
    assert not bool(output["velocity_supported"].any())
    assert torch.equal(
        output["velocity_normalized"], torch.zeros(2, 3),
    )


@torch.inference_mode()
def test_reflection_is_exact_after_nonzero_learned_corrections() -> None:
    model = _model().eval()
    for name, parameter in model.motion_state_head.named_parameters():
        if "coefficient.2.weight" in name or "fusion_logit.2.weight" in name:
            parameter.normal_(0.0, 0.2)
    batch = _supervised_batch()
    reflected_batch = _reflect_physical_training_batch(batch)
    original = model.estimate_motion_state(**_fields(model, batch))["state"]
    reflected = model.estimate_motion_state(
        **_fields(model, reflected_batch)
    )["state"]
    for index in (0, 1):
        torch.testing.assert_close(
            reflected[f"omega{index}_normalized"],
            -original[f"omega{index}_normalized"], rtol=0, atol=1e-7,
        )
        torch.testing.assert_close(
            reflected[f"velocity{index}_normalized"],
            original[f"velocity{index}_normalized"]
            * torch.tensor([1.0, -1.0, 1.0]), rtol=0, atol=1e-7,
        )


def _planar_transform_batch(
    batch: dict[str, torch.Tensor], matrix: torch.Tensor,
) -> dict[str, torch.Tensor]:
    result = deepcopy(batch)
    observations = result["history_obs_rel_m"].clone()
    observations[..., :2] = torch.einsum(
        "ij,...j->...i", matrix, observations[..., :2],
    )
    result["history_obs_rel_m"] = observations
    determinant = torch.linalg.det(matrix)
    target = result["target_motion_state_normalized"].clone()
    target[:, :2] = torch.einsum("ij,bj->bi", matrix, target[:, :2])
    target[:, 3] *= determinant
    result["target_motion_state_normalized"] = target
    if determinant < 0:
        result["history_switch_step"] = -result["history_switch_step"]
    return result


@pytest.mark.parametrize("kind", ("rotation", "slanted_reflection"))
@torch.inference_mode()
def test_planar_o2_equivariance_after_nonzero_learned_corrections(kind: str) -> None:
    model = _model().eval()
    for name, parameter in model.motion_state_head.named_parameters():
        if "coefficient.2.weight" in name or "fusion_logit.2.weight" in name:
            parameter.normal_(0.0, 0.2)
    if kind == "rotation":
        angle = torch.tensor(0.71)
        matrix = torch.tensor([
            [torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)],
        ])
    else:
        phi = torch.tensor(0.37)
        matrix = torch.tensor([
            [torch.cos(2 * phi), torch.sin(2 * phi)],
            [torch.sin(2 * phi), -torch.cos(2 * phi)],
        ])
    batch = _supervised_batch()
    transformed_batch = _planar_transform_batch(batch, matrix)
    original = model.estimate_motion_state(**_fields(model, batch))["state"]
    transformed = model.estimate_motion_state(
        **_fields(model, transformed_batch)
    )["state"]
    determinant = torch.linalg.det(matrix)
    for index in (0, 1):
        expected_velocity = original[f"velocity{index}_normalized"].clone()
        physical_planar = (
            expected_velocity[:, :2] * model.motion_state_scale[:2]
        )
        expected_velocity[:, :2] = torch.einsum(
            "ij,bj->bi", matrix, physical_planar,
        ) / model.motion_state_scale[:2]
        torch.testing.assert_close(
            transformed[f"velocity{index}_normalized"],
            expected_velocity, rtol=2e-5, atol=2e-6,
        )
        torch.testing.assert_close(
            transformed[f"omega{index}_normalized"],
            determinant * original[f"omega{index}_normalized"],
            rtol=2e-5, atol=2e-6,
        )


def test_yaw_alias_envelope_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="aliases"):
        EquivariantAlternatingTwistHead(
            32, 0.0, history_scale_s=0.35, position_scale_m=1.0,
            velocity_scale_mps=(3.0, 3.2, 0.25),
            yaw_rate_scale_rad_s=30.0,
            max_abs_yaw_rate_rad_s=30.0,
        )


def test_dataset_preflight_rejects_physical_yaw_outside_alias_envelope() -> None:
    def dataset(yaw: float):
        part = SimpleNamespace(tensors={
            MOTION_TARGET_FIELD: torch.tensor([[0.0, 0.0, 0.0, yaw]]),
        })
        return SimpleNamespace(parts=(part,))

    with pytest.raises(ValueError, match="exceeds alias envelope"):
        alternating_dataset_preflight(dataset(31.0), dataset(1.0))


def test_artifact_preflight_rejects_formula_and_truth_summary_tampering() -> None:
    def dataset(*yaw: float):
        part = SimpleNamespace(tensors={
            MOTION_TARGET_FIELD: torch.tensor([
                [0.0, 0.0, 0.0, value] for value in yaw
            ]),
        })
        return SimpleNamespace(parts=(part,))

    recomputed = alternating_dataset_preflight(
        dataset(1.0, -4.0), dataset(3.0),
    )
    _validate_yaw_preflight_report(recomputed, recomputed=deepcopy(recomputed))

    false_phase = deepcopy(recomputed)
    false_phase["phase_upper_bound_rad"] = 0.0
    with pytest.raises(ValueError, match="preflight differs"):
        _validate_yaw_preflight_report(false_phase, recomputed=recomputed)

    for field, value in (
        ("sample_count", 1), ("max_abs_yaw_rate_rad_s", 0.0),
    ):
        false_summary = deepcopy(recomputed)
        false_summary["splits"]["train"][field] = value
        with pytest.raises(ValueError, match="not bound to truth"):
            _validate_yaw_preflight_report(false_summary, recomputed=recomputed)


def test_alternating_sampler_cells_separate_exact_pair_scale_support() -> None:
    events = 8
    active = torch.ones(2, events, dtype=torch.bool)
    visible = torch.zeros(2, events, 4, dtype=torch.bool)
    visible[0, :, 0] = True
    visible[1, :, :2] = True
    part = SimpleNamespace(
        tensors={
            "pnp_s_event_mask": active,
            "pnp_s_obs_mask": visible,
            "pnp_s_event_time_s": torch.linspace(-0.07, 0.0, events).repeat(2, 1),
            MOTION_TARGET_FIELD: torch.tensor([
                [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0],
            ]),
        },
        session_ids=("same-session", "same-session"),
        motion_class=3,
    )
    dataset = SimpleNamespace(parts=(part,), offsets=(0,))
    cells = alternating_motion_state_cells(dataset)
    suffixes = {key[2].rsplit("/", 1)[-1] for key in cells}
    assert suffixes == {"pair0", "pair3"}


def test_screen_check_happy_path_and_zero_closure_do_not_false_pass(
    monkeypatch,
) -> None:
    monkeypatch.setattr(screen_runner, "_assert_finite_tree", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        screen_runner, "_validate_diagnostic_binding", lambda *args, **kwargs: None,
    )

    def distribution(value: float = 1.0) -> dict[str, float | int]:
        return {
            "count": 1, "mean_m": value, "p50_m": value,
            "p95_m": value, "p99_m": value,
        }

    def group() -> dict:
        return {
            "candidate_velocity": distribution(),
            "candidate_yaw": distribution(),
            "candidate_yaw_sign_accuracy": 0.8,
            "broken_handle_intervention_yaw": distribution(0.0),
            "candidate_handle_intervention_yaw": distribution(0.0),
            "fixed_broken_handle_closure": distribution(0.0),
            "fixed_intact_handle_closure": distribution(0.0),
            "broken_pair_intervention_yaw": distribution(0.0),
            "candidate_pair_intervention_yaw": distribution(0.0),
            "fixed_broken_pair_closure": distribution(0.0),
            "fixed_intact_pair_closure": distribution(0.0),
        }

    groups = {
        name: group() for name in (
            "overall", "pair0", "combined_pair1", "combined_pair2",
            "combined_pair3",
        )
    }
    diagnostics = {
        "groups": groups,
        "common_ramp_equivariance": {
            "velocity_delta_error_mps": distribution(0.0),
            "yaw_invariance_error_rad_s": distribution(0.0),
        },
        "relative_reversal_equivariance": {
            "velocity_prediction_invariance_mps": distribution(0.0),
            "yaw_prediction_antisymmetry_rad_s": distribution(0.0),
        },
        "write_isolation": {
            "zero_velocity_max_absolute_yaw_difference_normalized": 0.0,
        },
    }
    checks = screen_runner._screen_checks(
        diagnostics, {"diagnostics": deepcopy(diagnostics)},
    )
    assert "pair1_break_worsens_yaw" in checks
    assert checks["pair0_handle_break_worsens_closure"]["passed"] is False
    assert checks["pair1_break_worsens_closure"]["passed"] is False


def test_macro_substage_loss_ignores_future_and_rejects_unknown_stage() -> None:
    model = _model().train()
    batch = _supervised_batch()
    output = model.estimate_motion_state(**_fields(model, batch))
    prediction = {**output["history"], **output["state"]}
    expected, _ = alternating_substage_loss(
        prediction, batch, substage="velocity1",
    )
    poisoned = deepcopy(batch)
    poisoned["future_target_position_m"] = torch.full((2, 9, 3), torch.nan)
    actual, _ = alternating_substage_loss(
        prediction, poisoned, substage="velocity1",
    )
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="unknown alternating substage"):
        alternating_substage_loss(prediction, batch, substage="shared4d")


@pytest.mark.parametrize(
    ("update", "name", "phase_update", "phase_total", "endpoint"),
    (
        (1, "omega0", 1, 35, False), (35, "omega0", 35, 35, True),
        (36, "velocity0", 1, 20, False), (55, "velocity0", 20, 20, True),
        (56, "omega1", 1, 25, False), (80, "omega1", 25, 25, True),
        (81, "velocity1", 1, 20, False), (100, "velocity1", 20, 20, True),
    ),
)
def test_train_step_resets_lr_schedule_at_each_typed_substage(
    update: int, name: str, phase_update: int, phase_total: int,
    endpoint: bool,
) -> None:
    model = _model().train()
    _, loss, components = equivariant_alternating_train_step(
        model, _supervised_batch(), update, 100,
    )
    assert torch.isfinite(loss)
    assert components["state_substage"] == f"typed_alternating_{name}"
    assert components["state_lr_phase_update"] == phase_update
    assert components["state_lr_phase_total"] == phase_total
    assert components["state_substage_endpoint"] is endpoint


def test_formal_capacity_stays_below_v8_plus_five_percent() -> None:
    model = _model(channels=96)
    state_parameters = sum(
        parameter.numel() for name in ("context", "motion_state_head")
        for parameter in getattr(model, name).parameters()
    )
    assert state_parameters <= int(1.05 * V8_JOINT_REACHABLE_STATE_PARAMETERS)
    assert state_parameters > 0
