from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from training.stage3.continuous_invariant_anonymous_future import (
    ContinuousInvariantAnonymousFutureModel,
    continuous_future_loss,
)
from training.stage3.increment_invariant_anonymous_future import (
    IncrementInvariantAnonymousFutureModel,
)
from training.stage3.tests.test_anonymous_vehicle_motion import (
    HANDLE_REFLECTION,
    _batch,
    _reflect_handles,
    _roll_handles,
)
from training.stage3.train_increment_invariant_anonymous_future import (
    TRAJECTORY_MODULES,
    configure_stage,
    stage_loss_weights,
    stage_trainable_parameter_ids,
)


CHANNELS = 32


def _model() -> IncrementInvariantAnonymousFutureModel:
    torch.manual_seed(20260730)
    return IncrementInvariantAnonymousFutureModel(
        channels=CHANNELS, dropout=0.0, message_layers=2, basis_count=6,
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


def _capture_temporal_tokens(
    model: IncrementInvariantAnonymousFutureModel,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def hook(_module, values):
        captured.append(values[0].detach().clone())

    handle = model.context.history_projection[0].register_forward_pre_hook(hook)
    try:
        model(batch)
    finally:
        handle.remove()
    assert len(captured) == 1
    return captured[0]


@torch.inference_mode()
def test_v4_temporal_projection_uses_no_raw_handle_origin() -> None:
    model = _model()
    batch = _batch()
    reference = _capture_temporal_tokens(model, batch)
    shifted = deepcopy(batch)
    offset = torch.tensor([
        [[0.7, -0.2, 0.1], [-0.4, 0.8, 0.3], [0.2, 0.1, -0.5], [1.0, -0.6, 0.2]],
        [[-0.3, 0.4, 0.2], [0.9, 0.1, -0.2], [-0.5, -0.7, 0.1], [0.4, 0.3, 0.6]],
    ])
    shifted["history_obs_rel_m"] = shifted["history_obs_rel_m"] + offset[:, None]
    shifted["q0_relation_m"] = shifted["q0_relation_m"] + offset
    actual = _capture_temporal_tokens(model, shifted)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=3e-6)


@torch.inference_mode()
def test_first_visible_offset_is_relative_to_q0_and_is_not_discarded() -> None:
    model = _model()
    batch = _batch()
    reference = _capture_temporal_tokens(model, batch)
    changed = deepcopy(batch)
    visible = changed["history_obs_mask"][:, :, 2]
    changed["history_obs_rel_m"][:, :, 2] = torch.where(
        visible.unsqueeze(-1),
        changed["history_obs_rel_m"][:, :, 2] + torch.tensor([0.3, -0.1, 0.2]),
        changed["history_obs_rel_m"][:, :, 2],
    )
    actual = _capture_temporal_tokens(model, changed)
    assert not torch.equal(actual, reference)


@torch.inference_mode()
def test_history_dt_and_q0_quality_are_not_motion_law_features() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)
    polluted = deepcopy(batch)
    polluted["history_dt_s"] = torch.randn_like(polluted["history_dt_s"]) * 1e5
    polluted["q0_sigma_m"] = torch.randn_like(polluted["q0_sigma_m"]) * 1e5
    polluted["q0_confidence"] = torch.randn_like(polluted["q0_confidence"]) * 1e5
    polluted["q0_age_s"] = torch.randn_like(polluted["q0_age_s"]) * 1e5
    polluted["q0_support_class"] = torch.remainder(
        polluted["q0_support_class"] + 1, 4,
    )
    actual = model(polluted)
    for name in reference:
        assert torch.equal(actual[name], reference[name]), name


@torch.inference_mode()
def test_common_translation_still_changes_only_absolute_position() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)
    translation = torch.tensor([[1.25, -0.7, 0.3], [-2.0, 0.4, 0.8]])
    shifted = deepcopy(batch)
    shifted["current_position_m"] = shifted["current_position_m"] + translation
    actual = model(shifted)
    for name in ("trajectory_coefficient", "role_delta_m", "role_logits", "delta_m"):
        assert torch.equal(actual[name], reference[name]), name
    torch.testing.assert_close(
        actual["position_m"], reference["position_m"] + translation[:, None],
    )


@pytest.mark.parametrize("shift", [0, 1, 2, 3])
@torch.inference_mode()
def test_v4_c4_equivariance(shift: int) -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)
    actual = model(_roll_handles(batch, shift))
    torch.testing.assert_close(
        actual["handle_state"],
        torch.roll(reference["handle_state"], shifts=shift, dims=1),
    )
    for name in (
        "vehicle_state", "trajectory_coefficient", "role_delta_m",
        "role_position_m", "role_probability", "position_m",
    ):
        torch.testing.assert_close(actual[name], reference[name])


@torch.inference_mode()
def test_v4_reflection_swaps_relative_roles() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)
    actual = model(_reflect_handles(batch))
    index = HANDLE_REFLECTION
    torch.testing.assert_close(
        actual["handle_state"], reference["handle_state"].index_select(1, index),
    )
    torch.testing.assert_close(
        actual["trajectory_coefficient"],
        reference["trajectory_coefficient"].index_select(1, index),
    )
    torch.testing.assert_close(
        actual["role_delta_m"], reference["role_delta_m"].index_select(2, index),
    )
    torch.testing.assert_close(actual["position_m"], reference["position_m"])


@torch.inference_mode()
def test_v4_tau_zero_and_query_permutation() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)
    assert torch.equal(reference["role_delta_m"][:, 0], batch["q0_relation_m"])
    assert torch.equal(reference["position_m"][:, 0], batch["current_position_m"])
    order = torch.tensor([2, 0, 3, 1])
    permuted = deepcopy(batch)
    permuted["tau_s"] = permuted["tau_s"][:, order]
    actual = model(permuted)
    for name in (
        "time_basis", "role_logits", "role_probability", "role_delta_m",
        "role_position_m", "selected_role", "delta_m", "position_m",
    ):
        torch.testing.assert_close(actual[name], reference[name][:, order])


def test_v3_checkpoint_cannot_strict_load_into_v4() -> None:
    v4 = _model()
    v3 = ContinuousInvariantAnonymousFutureModel(
        channels=32, dropout=0.0, message_layers=2, basis_count=6,
    )
    assert v4.model_family != v3.model_family
    with pytest.raises(RuntimeError):
        v4.load_state_dict(v3.state_dict(), strict=True)


@pytest.mark.parametrize("stage", ["trajectory", "selector", "joint"])
def test_v4_three_stage_allowlist_and_optimizer_step(stage: str) -> None:
    model = _model()
    configure_stage(model, stage)
    expected = stage_trainable_parameter_ids(model, stage)
    assert {id(value) for value in model.parameters() if value.requires_grad} == expected
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.SGD(
        [value for value in model.parameters() if value.requires_grad], lr=1e-2,
    )
    batch = _supervised_batch()
    loss, _ = continuous_future_loss(
        model(batch, detach_selector_context=(stage == "joint")),
        batch, **stage_loss_weights(stage),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    changed = 0
    for name, value in model.named_parameters():
        did_change = not torch.equal(before[name], value.detach())
        if id(value) in expected:
            changed += int(did_change)
        else:
            assert not did_change, name
    assert changed > 0


def test_v4_joint_selector_loss_is_detached_from_trajectory() -> None:
    model = _model().train()
    batch = _supervised_batch()
    prediction = model(batch, detach_selector_context=True)
    selector_loss, _ = continuous_future_loss(
        prediction, batch, **stage_loss_weights("selector"),
    )
    trajectory = [
        parameter
        for name in TRAJECTORY_MODULES
        for parameter in getattr(model, name).parameters()
    ]
    gradient = torch.autograd.grad(selector_loss, trajectory, allow_unused=True)
    assert all(value is None or torch.count_nonzero(value) == 0 for value in gradient)
