from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from training.stage3.anonymous_vehicle_motion_v2 import (
    VisibilityAwareAnonymousVehicleFutureModel,
    visibility_aware_future_loss,
)
from training.stage3.tests.test_anonymous_vehicle_motion import (
    HANDLE_REFLECTION,
    _batch,
    _reflect_handles,
    _roll_handles,
)


CHANNELS = 32


def _model() -> VisibilityAwareAnonymousVehicleFutureModel:
    torch.manual_seed(20260730)
    return VisibilityAwareAnonymousVehicleFutureModel(
        channels=CHANNELS, dropout=0.0, message_layers=2,
        trained_horizon_s=0.55, maximum_absolute_step=6,
        basis_count=6, latent_experts=3,
    ).eval()


def _assert_close(left: torch.Tensor, right: torch.Tensor) -> None:
    torch.testing.assert_close(left, right, rtol=1e-5, atol=2e-6)


@torch.inference_mode()
def test_v2_forward_shapes_tau_zero_and_tau_independent_state() -> None:
    batch = _batch()
    output = _model()(batch)
    assert output["handle_state"].shape == (2, 4, 2 * CHANNELS)
    assert output["trajectory_coefficient"].shape == (2, 4, 3, 6, 3)
    assert output["motion_regime_probability"].shape == (2, 3)
    assert output["role_position_m"].shape == (2, 4, 4, 3)
    assert output["role_probability"].shape == (2, 4, 4)
    assert output["conditional_position_m"].shape == (2, 4, 13, 3)
    assert output["switch_probability"].shape == (2, 4, 13)
    assert output["position_m"].shape == (2, 4, 3)
    assert torch.equal(
        output["role_delta_m"][:, 0],
        batch["q0_relation_m"],
    )
    assert torch.equal(output["selected_role"][:, 0], torch.zeros(2, dtype=torch.long))
    assert torch.equal(output["position_m"][:, 0], batch["current_position_m"])
    assert torch.equal(
        output["selected_switch_step_aux"][:, 0], torch.zeros(2, dtype=torch.long)
    )
    # Coefficients and latent motion-regime probabilities are predicted once
    # per history and do not carry a query dimension.
    assert output["trajectory_coefficient"].ndim == 5
    assert output["motion_regime_probability"].ndim == 2


@torch.inference_mode()
def test_active_but_invisible_handle_coordinates_are_ignored() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)
    polluted = deepcopy(batch)
    invisible = (
        polluted["history_event_mask"][:, :, None]
        & ~polluted["history_obs_mask"]
    )
    polluted["history_obs_rel_m"] = torch.where(
        invisible.unsqueeze(-1),
        torch.full_like(polluted["history_obs_rel_m"], float("nan")),
        polluted["history_obs_rel_m"],
    )
    actual = model(polluted)
    for name in reference:
        if reference[name].dtype.is_floating_point:
            _assert_close(actual[name], reference[name])
        else:
            assert torch.equal(actual[name], reference[name]), name


@torch.inference_mode()
def test_sparse_same_handle_velocity_uses_previous_visible_time() -> None:
    output = _model()(_batch())
    # Handle zero is visible at event 5 and then event 8 with two intervening
    # global events. Its configured constant velocity must use the full .03 s.
    assert bool(output["history_velocity_valid"][0, 8, 0])
    assert output["history_visible_elapsed_s"][0, 8, 0].item() == pytest.approx(0.03)
    _assert_close(
        output["history_local_velocity_mps"][0, 8, 0],
        torch.tensor([0.17, -0.09, 0.04]),
    )


@torch.inference_mode()
def test_same_modulo_role_is_one_structural_trajectory_and_ignores_row_metadata() -> None:
    model = _model()
    batch = _batch()
    polluted = deepcopy(batch)
    polluted["candidate_relation_m"] = torch.randn_like(
        polluted["candidate_relation_m"]
    ) * 1e5
    polluted["candidate_confidence"] = torch.rand_like(
        polluted["candidate_confidence"]
    )
    polluted["candidate_supported"] = ~polluted["candidate_supported"]
    reference = model(batch)
    actual = model(polluted)
    _assert_close(actual["conditional_position_m"], reference["conditional_position_m"])
    steps = batch["candidate_step"][0]
    for step in range(-6, 3):
        first = int(torch.nonzero(steps == step, as_tuple=False)[0])
        second = int(torch.nonzero(steps == step + 4, as_tuple=False)[0])
        assert torch.equal(
            actual["conditional_position_m"][:, :, first],
            actual["conditional_position_m"][:, :, second],
        )


@torch.inference_mode()
def test_role_probability_is_exact_probability_aggregated_modulo_four() -> None:
    output = _model()(_batch())
    aggregate = torch.zeros_like(output["role_probability"])
    aggregate.scatter_add_(
        2,
        output["candidate_role"][:, None].expand(-1, 4, -1),
        output["switch_probability"],
    )
    _assert_close(aggregate, output["role_probability"])
    _assert_close(output["role_probability"].sum(dim=-1), torch.ones(2, 4))
    _assert_close(output["switch_probability"].sum(dim=-1), torch.ones(2, 4))


@torch.inference_mode()
def test_duplicate_or_missing_signed_candidate_is_rejected() -> None:
    model = _model()
    batch = _batch()
    batch["candidate_step"][:, -1] = batch["candidate_step"][:, -2]
    with pytest.raises(ValueError, match="unique complete"):
        model(batch)


@pytest.mark.parametrize("shift", [0, 1, 2, 3])
@torch.inference_mode()
def test_v2_c4_equivariance(shift: int) -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)
    shifted = model(_roll_handles(batch, shift))
    _assert_close(
        shifted["handle_state"],
        torch.roll(reference["handle_state"], shifts=shift, dims=1),
    )
    for name in (
        "vehicle_state", "role_position_m", "role_probability",
        "conditional_position_m", "switch_probability", "position_m",
    ):
        _assert_close(shifted[name], reference[name])


@torch.inference_mode()
def test_v2_reflection_swaps_relative_roles_and_signed_crossing() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)
    reflected = model(_reflect_handles(batch))
    index = HANDLE_REFLECTION
    _assert_close(
        reflected["handle_state"], reference["handle_state"].index_select(1, index),
    )
    _assert_close(
        reflected["role_probability"],
        reference["role_probability"].index_select(2, index),
    )
    _assert_close(
        reflected["role_position_m"],
        reference["role_position_m"].index_select(2, index),
    )
    _assert_close(reflected["conditional_position_m"], reference["conditional_position_m"])
    _assert_close(reflected["switch_probability"], reference["switch_probability"])
    _assert_close(reflected["position_m"], reference["position_m"])
    assert torch.equal(
        reflected["selected_switch_step_aux"],
        -reference["selected_switch_step_aux"],
    )


@torch.inference_mode()
def test_query_and_candidate_permutations_preserve_v2_semantics() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)
    query_order = torch.tensor([2, 0, 3, 1])
    query_batch = deepcopy(batch)
    query_batch["tau_s"] = query_batch["tau_s"][:, query_order]
    query = model(query_batch)
    for name in (
        "time_basis", "role_logits", "role_probability", "role_position_m",
        "conditional_position_m", "switch_probability", "selected_role",
        "selected_switch_step_aux", "position_m",
    ):
        _assert_close(query[name], reference[name][:, query_order])
    candidate_order = torch.tensor([6, 0, 12, 2, 10, 4, 8, 1, 11, 3, 9, 5, 7])
    candidate_batch = deepcopy(batch)
    for name in (
        "candidate_relation_m", "candidate_step", "candidate_mask",
        "candidate_confidence", "candidate_supported",
    ):
        candidate_batch[name] = candidate_batch[name][:, candidate_order]
    candidate = model(candidate_batch)
    for name in ("conditional_position_m", "switch_probability"):
        _assert_close(candidate[name], reference[name][:, :, candidate_order])
    for name in (
        "role_probability", "role_position_m", "selected_role",
        "selected_switch_step_aux", "position_m",
    ):
        _assert_close(candidate[name], reference[name])


@torch.inference_mode()
def test_truth_motion_session_and_future_pnp_never_affect_v2_forward() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)
    polluted = deepcopy(batch)
    polluted.update({
        "target_visible_delta_m": torch.randn(2, 4, 3) * 1e6,
        "target_switch_count": torch.randint(-99, 100, (2, 4)),
        "truth_current_position_m": torch.randn(2, 3) * 1e6,
        "motion_class": torch.tensor([2, 3]),
        "session_index": torch.tensor([91234, -571]),
        "physical_armor_id": torch.randint(0, 999, (2, 10, 4)),
        "future_pnp_position_m": torch.randn(2, 4, 3) * 1e6,
    })
    actual = model(polluted)
    for name in reference:
        if reference[name].dtype.is_floating_point:
            assert torch.equal(actual[name], reference[name]), name
        else:
            assert torch.equal(actual[name], reference[name]), name


def test_loss_is_window_weighted_and_exact_crossing_is_auxiliary() -> None:
    model = _model().train()
    batch = _batch()
    batch.update({
        "target_query_mask": torch.ones(2, 4, dtype=torch.bool),
        "target_switch_count": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        "truth_current_position_m": batch["current_position_m"].clone(),
        "target_visible_delta_m": torch.zeros(2, 4, 3),
    })
    prediction = model(batch)
    objective, component = visibility_aware_future_loss(prediction, batch)
    assert torch.isfinite(objective)
    assert component["role"].item() > 0
    assert component["exact_crossing"].item() > 0
    objective.backward()
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in model.role_coefficient_head.parameters()
    )


def test_v1_and_v2_families_are_distinct() -> None:
    from training.stage3.anonymous_vehicle_motion import AnonymousVehicleFutureModel

    assert AnonymousVehicleFutureModel.model_family != _model().model_family
    with pytest.raises(RuntimeError):
        _model().load_state_dict(AnonymousVehicleFutureModel(
            channels=32, dropout=0.0, message_layers=2,
        ).state_dict(), strict=True)


def _supervised_batch() -> dict[str, torch.Tensor]:
    batch = _batch()
    batch.update({
        "target_query_mask": torch.ones(2, 4, dtype=torch.bool),
        "target_switch_count": torch.tensor([[0, 1, 2, 3], [0, -1, -2, -3]]),
        "truth_current_position_m": batch["current_position_m"].clone(),
        "target_visible_delta_m": torch.randn(2, 4, 3) * 0.2,
    })
    return batch


@pytest.mark.parametrize("stage", ["trajectory", "selector", "joint", "recalibration"])
def test_four_stage_allowlist_and_actual_optimizer_step(stage: str) -> None:
    from training.stage3.train_anonymous_vehicle_motion_v2 import (
        configure_stage,
        stage_loss_weights,
        stage_trainable_parameter_ids,
    )

    model = _model()
    configure_stage(model, stage)
    expected = stage_trainable_parameter_ids(model, stage)
    actual = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert actual == expected
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-2,
    )
    batch = _supervised_batch()
    prediction = model(batch, detach_selector_context=(stage == "joint"))
    loss, _ = visibility_aware_future_loss(
        prediction, batch, **stage_loss_weights(stage),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    optimizer.step()
    changed_active = 0
    for name, parameter in model.named_parameters():
        changed = not torch.equal(before[name], parameter.detach())
        if id(parameter) in expected:
            changed_active += int(changed)
        else:
            assert not changed, name
    assert changed_active > 0


def test_joint_selector_loss_is_strictly_detached_from_trajectory() -> None:
    from training.stage3.train_anonymous_vehicle_motion_v2 import (
        TRAJECTORY_MODULES,
        stage_loss_weights,
    )

    model = _model().train()
    batch = _supervised_batch()
    prediction = model(batch, detach_selector_context=True)
    selector_loss, _ = visibility_aware_future_loss(
        prediction, batch, **stage_loss_weights("recalibration"),
    )
    trajectory_parameters = [
        parameter
        for name in TRAJECTORY_MODULES
        for parameter in getattr(model, name).parameters()
    ]
    gradient = torch.autograd.grad(
        selector_loss, trajectory_parameters, allow_unused=True,
    )
    assert all(value is None or torch.count_nonzero(value) == 0 for value in gradient)
