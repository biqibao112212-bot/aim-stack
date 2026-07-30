from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy

import pytest
import torch

from training.stage3.anonymous_vehicle_motion_v2 import (
    VisibilityAwareAnonymousVehicleFutureModel,
)
from training.stage3.continuous_invariant_anonymous_future import (
    ContinuousInvariantAnonymousFutureModel,
    continuous_future_loss,
)
from training.stage3.tests.test_anonymous_vehicle_motion import (
    HANDLE_REFLECTION,
    _batch,
    _reflect_handles,
    _roll_handles,
)
from training.stage3.train_continuous_invariant_anonymous_future import (
    HierarchicalSessionHistorySampler,
    TRAJECTORY_MODULES,
    apply_bin_preserving_prefix_dropout,
    configure_stage,
    stage_loss_weights,
    stage_trainable_parameter_ids,
)


CHANNELS = 32


def _model() -> ContinuousInvariantAnonymousFutureModel:
    torch.manual_seed(20260730)
    return ContinuousInvariantAnonymousFutureModel(
        channels=CHANNELS, dropout=0.0, message_layers=2,
        trained_horizon_s=0.55, maximum_absolute_step=6,
        basis_count=6,
    ).eval()


def _supervised_batch() -> dict[str, torch.Tensor]:
    batch = _batch()
    batch.update({
        "target_query_mask": torch.ones(2, 4, dtype=torch.bool),
        "target_switch_count": torch.tensor([[0, 1, 2, 3], [0, -1, -2, -3]]),
        "truth_current_position_m": batch["current_position_m"].clone(),
        "target_visible_delta_m": torch.randn(2, 4, 3) * 0.2,
    })
    return batch


def _assert_close(left: torch.Tensor, right: torch.Tensor) -> None:
    torch.testing.assert_close(left, right, rtol=1e-5, atol=2e-6)


@torch.inference_mode()
def test_v3_shapes_tau_zero_and_single_continuous_field() -> None:
    batch = _batch()
    output = _model()(batch)
    assert output["trajectory_coefficient"].shape == (2, 4, 6, 3)
    assert output["role_delta_m"].shape == (2, 4, 4, 3)
    assert output["role_position_m"].shape == (2, 4, 4, 3)
    assert output["role_probability"].shape == (2, 4, 4)
    assert output["position_m"].shape == (2, 4, 3)
    assert torch.equal(output["role_delta_m"][:, 0], batch["q0_relation_m"])
    assert torch.equal(output["selected_role"][:, 0], torch.zeros(2, dtype=torch.long))
    assert torch.equal(output["position_m"][:, 0], batch["current_position_m"])
    forbidden = ("expert", "regime", "gate", "mixture")
    assert not any(
        any(token in name.lower() for token in forbidden)
        for name, _ in _model().named_modules()
    )
    assert _model().config["trajectory_experts"] is False


@torch.inference_mode()
def test_common_translation_changes_only_absolute_positions() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)
    translation = torch.tensor([[1.25, -0.7, 0.3], [-2.0, 0.4, 0.8]])
    shifted_batch = deepcopy(batch)
    shifted_batch["current_position_m"] = (
        shifted_batch["current_position_m"] + translation
    )
    shifted = model(shifted_batch)
    for name in (
        "trajectory_coefficient", "time_basis", "role_delta_m", "delta_m",
        "role_logits", "role_probability", "selected_role",
    ):
        assert torch.equal(shifted[name], reference[name]), name
    _assert_close(
        shifted["role_position_m"],
        reference["role_position_m"] + translation[:, None, None],
    )
    _assert_close(
        shifted["position_m"], reference["position_m"] + translation[:, None],
    )


def test_current_position_gradient_is_absent_from_delta_and_identity_for_position() -> None:
    model = _model()
    batch = _batch()
    current = batch["current_position_m"].clone().requires_grad_(True)
    batch["current_position_m"] = current
    output = model(batch)
    delta_gradient = torch.autograd.grad(
        output["delta_m"].sum(), current, retain_graph=True, allow_unused=True,
    )[0]
    assert delta_gradient is None or torch.count_nonzero(delta_gradient) == 0
    coordinate = output["position_m"][0, 1, 0]
    position_gradient = torch.autograd.grad(coordinate, current)[0]
    expected = torch.zeros_like(current)
    expected[0, 0] = 1.0
    assert torch.equal(position_gradient, expected)


@torch.inference_mode()
def test_truth_motion_session_id_candidate_and_future_pnp_are_not_forward_inputs() -> None:
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
        "candidate_relation_m": torch.randn_like(batch["candidate_relation_m"]) * 1e6,
        "candidate_step": torch.randint(-999, 1000, batch["candidate_step"].shape),
        "candidate_mask": ~batch["candidate_mask"],
        "candidate_confidence": torch.randn_like(batch["candidate_confidence"]),
        "candidate_supported": ~batch["candidate_supported"],
    })
    actual = model(polluted)
    for name in reference:
        assert torch.equal(actual[name], reference[name]), name


@torch.inference_mode()
def test_invisible_handle_coordinates_are_ignored_and_gap_dt_is_real() -> None:
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
    assert actual["history_visible_elapsed_s"][0, 8, 0].item() == pytest.approx(0.03)
    _assert_close(
        actual["history_local_velocity_mps"][0, 8, 0],
        torch.tensor([0.17, -0.09, 0.04]),
    )


@pytest.mark.parametrize("shift", [0, 1, 2, 3])
@torch.inference_mode()
def test_v3_c4_equivariance(shift: int) -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)
    shifted = model(_roll_handles(batch, shift))
    _assert_close(
        shifted["handle_state"],
        torch.roll(reference["handle_state"], shifts=shift, dims=1),
    )
    for name in (
        "vehicle_state", "trajectory_coefficient", "role_delta_m",
        "role_position_m", "role_probability", "position_m",
    ):
        _assert_close(shifted[name], reference[name])


@torch.inference_mode()
def test_v3_reflection_swaps_relative_roles() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)
    reflected = model(_reflect_handles(batch))
    index = HANDLE_REFLECTION
    _assert_close(
        reflected["handle_state"], reference["handle_state"].index_select(1, index),
    )
    for name in ("trajectory_coefficient", "role_delta_m", "role_position_m"):
        _assert_close(reflected[name], reference[name].index_select(2 if name != "trajectory_coefficient" else 1, index))
    _assert_close(
        reflected["role_probability"],
        reference["role_probability"].index_select(2, index),
    )
    _assert_close(reflected["position_m"], reference["position_m"])


@torch.inference_mode()
def test_query_permutation_preserves_semantics() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)
    order = torch.tensor([2, 0, 3, 1])
    permuted = deepcopy(batch)
    permuted["tau_s"] = permuted["tau_s"][:, order]
    actual = model(permuted)
    for name in (
        "time_basis", "role_logits", "role_probability", "role_delta_m",
        "role_position_m", "selected_role", "delta_m", "position_m",
    ):
        _assert_close(actual[name], reference[name][:, order])


def test_direct_loss_is_finite_and_uses_role_selector() -> None:
    model = _model().train()
    batch = _supervised_batch()
    objective, component = continuous_future_loss(model(batch), batch)
    assert torch.isfinite(objective)
    assert component["trajectory"].item() >= 0
    assert component["role"].item() > 0
    objective.backward()
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in model.role_coefficient_head.parameters()
    )


def test_v1_v2_v3_families_and_state_dicts_are_strictly_distinct() -> None:
    from training.stage3.anonymous_vehicle_motion import AnonymousVehicleFutureModel

    model = _model()
    assert len({
        model.model_family,
        AnonymousVehicleFutureModel.model_family,
        VisibilityAwareAnonymousVehicleFutureModel.model_family,
    }) == 3
    with pytest.raises(RuntimeError):
        model.load_state_dict(VisibilityAwareAnonymousVehicleFutureModel(
            channels=32, dropout=0.0, message_layers=2,
            basis_count=6, latent_experts=3,
        ).state_dict(), strict=True)


@pytest.mark.parametrize("stage", ["trajectory", "selector", "joint"])
def test_three_stage_allowlist_and_actual_optimizer_step(stage: str) -> None:
    model = _model()
    configure_stage(model, stage)
    expected = stage_trainable_parameter_ids(model, stage)
    actual = {id(value) for value in model.parameters() if value.requires_grad}
    assert actual == expected
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.SGD(
        [value for value in model.parameters() if value.requires_grad], lr=1e-2,
    )
    batch = _supervised_batch()
    prediction = model(batch, detach_selector_context=(stage == "joint"))
    loss, _ = continuous_future_loss(
        prediction, batch, **stage_loss_weights(stage),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    changed_active = 0
    for name, value in model.named_parameters():
        changed = not torch.equal(before[name], value.detach())
        if id(value) in expected:
            changed_active += int(changed)
        else:
            assert not changed, name
    assert changed_active > 0


def test_joint_selector_loss_is_detached_from_trajectory() -> None:
    model = _model().train()
    prediction = model(_supervised_batch(), detach_selector_context=True)
    batch = _supervised_batch()
    # Recompute from the same batch so labels and prediction have one contract.
    prediction = model(batch, detach_selector_context=True)
    selector_loss, _ = continuous_future_loss(
        prediction, batch, **stage_loss_weights("selector"),
    )
    trajectory = [
        parameter
        for name in TRAJECTORY_MODULES
        for parameter in getattr(model, name).parameters()
    ]
    gradient = torch.autograd.grad(
        selector_loss, trajectory, allow_unused=True,
    )
    assert all(value is None or torch.count_nonzero(value) == 0 for value in gradient)


def _sampler_cells() -> dict[tuple[int, str, str], list[int]]:
    result: dict[tuple[int, str, str], list[int]] = {}
    index = 0
    for motion, sessions in ((2, ("a", "b", "c")), (3, ("d", "e", "f", "g"))):
        for session in sessions:
            for history in ("8-15", "16-23", "24-plus"):
                size = 1 if session in {"a", "d"} else 17
                result[(motion, session, history)] = list(range(index, index + size))
                index += size
    return result


def test_hierarchical_sampler_balances_every_level_and_resumes_exactly() -> None:
    cells = _sampler_cells()
    reverse = {index: key for key, values in cells.items() for index in values}
    sampler = HierarchicalSessionHistorySampler(cells, seed=19)
    selected = sampler.draw(840)
    by_motion = Counter(reverse[index][0] for index in selected)
    assert max(by_motion.values()) - min(by_motion.values()) <= 1
    by_session: dict[int, Counter[str]] = defaultdict(Counter)
    by_history: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    for index in selected:
        motion, session, history = reverse[index]
        by_session[motion][session] += 1
        by_history[(motion, session)][history] += 1
    assert all(max(value.values()) - min(value.values()) <= 1 for value in by_session.values())
    assert all(max(value.values()) - min(value.values()) <= 1 for value in by_history.values())
    state = sampler.state_dict()
    expected = sampler.draw(73)
    restored = HierarchicalSessionHistorySampler(cells, seed=999)
    restored.load_state_dict(state)
    assert restored.draw(73) == expected


@pytest.mark.parametrize("count,lower", [(12, 8), (20, 16), (30, 24)])
def test_prefix_dropout_never_crosses_history_bin(count: int, lower: int) -> None:
    events = 32
    active = torch.arange(events)[None] >= events - count
    batch = {
        "pnp_s_event_mask": active,
        "pnp_s_obs_mask": active[:, :, None].expand(-1, -1, 4).clone(),
        "pnp_s_primary_mask": torch.zeros(1, events, 4, dtype=torch.bool),
        "pnp_s_obs_m": torch.randn(1, events, 4, 3),
        "pnp_s_event_time_s": torch.arange(events, dtype=torch.float32)[None],
        "pnp_s_switch_step": torch.zeros(1, events, dtype=torch.long),
    }
    batch["pnp_s_primary_mask"][..., 0] = active
    generator = torch.Generator().manual_seed(123)
    for _ in range(50):
        actual = apply_bin_preserving_prefix_dropout(
            batch, probability=1.0, generator=generator,
        )
        retained = int(actual["pnp_s_event_mask"].sum())
        assert lower <= retained <= count
