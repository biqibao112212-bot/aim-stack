from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from training.stage3.anonymous_vehicle_motion import (
    AnonymousVehicleFutureModel,
)


CHANNELS = 32
HANDLE_REFLECTION = torch.tensor([0, 3, 2, 1])


def _model() -> AnonymousVehicleFutureModel:
    torch.manual_seed(20260730)
    model = AnonymousVehicleFutureModel(
        channels=CHANNELS,
        dropout=0.0,
        message_layers=2,
        trained_horizon_s=0.55,
        maximum_absolute_step=6,
    )
    return model.eval()


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    events = 10
    query_count = 4
    steps = torch.arange(-6, 7, dtype=torch.long)
    candidate_count = steps.numel()

    active = torch.zeros(batch_size, events, dtype=torch.bool)
    active[:, 2:] = True
    time = torch.zeros(batch_size, events)
    time[:, 2:] = torch.linspace(-0.07, 0.0, 8)
    dt = torch.zeros_like(time)
    dt[:, 3:] = 0.01

    # The primary advances once per active event and ends at anonymous handle
    # zero at q0.  No value here denotes a persistent physical armor ID.
    primary_index = torch.tensor([1, 2, 3, 0, 1, 2, 3, 0])
    primary = torch.zeros(batch_size, events, 4, dtype=torch.bool)
    visible = torch.zeros_like(primary)
    for event, handle in enumerate(primary_index.tolist(), start=2):
        primary[:, event, handle] = True
        visible[:, event, handle] = True
        visible[:, event, (handle + 1) % 4] = True

    switch = torch.zeros(batch_size, events, dtype=torch.long)
    switch[:, 3:] = 1
    q0_relation = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.32, 0.04, 0.01],
            [-0.03, 0.61, -0.02],
            [-0.29, 0.08, 0.03],
        ]
    ).expand(batch_size, -1, -1).clone()
    history = torch.zeros(batch_size, events, 4, 3)
    velocity = torch.tensor([0.17, -0.09, 0.04])
    for event in range(2, events):
        for handle in range(4):
            if bool(visible[0, event, handle]):
                history[:, event, handle] = (
                    q0_relation[:, handle]
                    + time[:, event, None] * velocity
                    + 0.002 * handle
                )

    q0_sigma = torch.tensor(
        [
            [0.010, 0.012, 0.014],
            [0.018, 0.016, 0.020],
            [0.025, 0.023, 0.021],
            [0.015, 0.019, 0.017],
        ]
    ).expand(batch_size, -1, -1).clone()
    q0_confidence = torch.tensor([0.97, 0.83, 0.68, 0.76]).expand(
        batch_size, -1
    ).clone()
    q0_age = torch.tensor([0.0, 0.03, 0.08, 0.05]).expand(
        batch_size, -1
    ).clone()
    q0_support_class = torch.tensor([3, 2, 1, 2], dtype=torch.long).expand(
        batch_size, -1
    ).clone()
    q0_supported = torch.tensor([True, True, False, True]).expand(
        batch_size, -1
    ).clone()

    handle = torch.remainder(steps, 4).expand(batch_size, -1)
    relation = q0_relation.gather(
        1, handle.unsqueeze(-1).expand(-1, -1, 3)
    )
    candidate_confidence = q0_confidence.gather(1, handle)
    candidate_supported = q0_supported.gather(1, handle)
    current = torch.tensor(
        [[1.2, -0.3, 5.1], [-0.4, 0.2, 6.0]], dtype=torch.float32
    )[:batch_size].clone()
    tau = torch.tensor([0.0, 0.1, 0.25, 0.5]).expand(
        batch_size, query_count
    ).clone()

    return {
        "history_obs_rel_m": history,
        "history_obs_mask": visible,
        "history_primary_mask": primary,
        "history_event_mask": active,
        "history_time_s": time,
        "history_dt_s": dt,
        "history_switch_step": switch,
        "q0_relation_m": q0_relation,
        "q0_sigma_m": q0_sigma,
        "q0_confidence": q0_confidence,
        "q0_age_s": q0_age,
        "q0_support_class": q0_support_class,
        "q0_supported": q0_supported,
        "current_position_m": current,
        "candidate_relation_m": relation,
        "candidate_step": steps.expand(batch_size, candidate_count).clone(),
        "candidate_mask": torch.ones(
            batch_size, candidate_count, dtype=torch.bool
        ),
        "candidate_confidence": candidate_confidence,
        "candidate_supported": candidate_supported,
        "tau_s": tau,
    }


def _assert_same(left: torch.Tensor, right: torch.Tensor) -> None:
    torch.testing.assert_close(left, right, rtol=1e-5, atol=2e-6)


def _roll_handles(batch: dict[str, torch.Tensor], shift: int) -> dict[str, torch.Tensor]:
    rolled = deepcopy(batch)
    for name, dimension in (
        ("history_obs_rel_m", 2),
        ("history_obs_mask", 2),
        ("history_primary_mask", 2),
        ("q0_relation_m", 1),
        ("q0_sigma_m", 1),
        ("q0_confidence", 1),
        ("q0_age_s", 1),
        ("q0_support_class", 1),
        ("q0_supported", 1),
    ):
        rolled[name] = torch.roll(rolled[name], shifts=shift, dims=dimension)
    return rolled


def _reflect_handles(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    reflected = deepcopy(batch)
    index = HANDLE_REFLECTION.to(batch["current_position_m"].device)
    for name, dimension in (
        ("history_obs_rel_m", 2),
        ("history_obs_mask", 2),
        ("history_primary_mask", 2),
        ("q0_relation_m", 1),
        ("q0_sigma_m", 1),
        ("q0_confidence", 1),
        ("q0_age_s", 1),
        ("q0_support_class", 1),
        ("q0_supported", 1),
    ):
        reflected[name] = reflected[name].index_select(dimension, index)
    reflected["history_switch_step"] = -reflected["history_switch_step"]
    reflected["candidate_step"] = -reflected["candidate_step"]
    return reflected


@torch.inference_mode()
def test_forward_shapes_and_exact_tau_zero_contract() -> None:
    batch = _batch()
    output = _model()(batch)
    batch_size, candidate_count = batch["candidate_step"].shape
    query_count = batch["tau_s"].shape[1]

    assert output["handle_state"].shape == (batch_size, 4, 2 * CHANNELS)
    assert output["vehicle_state"].shape == (batch_size, 4 * CHANNELS)
    assert output["candidate_handle"].shape == (batch_size, candidate_count)
    assert output["conditional_delta_m"].shape == (
        batch_size, query_count, candidate_count, 3
    )
    assert output["conditional_position_m"].shape == (
        batch_size, query_count, candidate_count, 3
    )
    assert output["switch_logits"].shape == (
        batch_size, query_count, candidate_count
    )
    assert output["switch_probability"].shape == (
        batch_size, query_count, candidate_count
    )
    assert output["selected_candidate_row"].shape == (batch_size, query_count)
    assert output["selected_switch_step"].shape == (batch_size, query_count)
    assert output["position_m"].shape == (batch_size, query_count, 3)

    assert torch.equal(
        output["conditional_delta_m"][:, 0], batch["candidate_relation_m"]
    )
    assert torch.equal(output["selected_switch_step"][:, 0], torch.zeros(batch_size, dtype=torch.long))
    assert torch.equal(output["delta_m"][:, 0], torch.zeros(batch_size, 3))
    assert torch.equal(output["position_m"][:, 0], batch["current_position_m"])
    zero_row = torch.nonzero(
        batch["candidate_step"][0] == 0, as_tuple=False
    ).squeeze(1).item()
    assert torch.equal(
        output["switch_probability"][:, 0, zero_row], torch.ones(batch_size)
    )


@torch.inference_mode()
def test_same_anonymous_role_k_and_k_plus_four_share_conditional_trajectory() -> None:
    batch = _batch()
    output = _model()(batch)
    steps = batch["candidate_step"][0]
    for step in range(-6, 3):
        first = torch.nonzero(steps == step, as_tuple=False).squeeze(1).item()
        second = torch.nonzero(steps == step + 4, as_tuple=False).squeeze(1).item()
        _assert_same(
            output["conditional_position_m"][:, :, first],
            output["conditional_position_m"][:, :, second],
        )


@pytest.mark.parametrize("shift", [0, 1, 2, 3])
@torch.inference_mode()
def test_all_four_c4_handle_shifts_are_equivariant_or_invariant(shift: int) -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)
    shifted = model(_roll_handles(batch, shift))

    _assert_same(
        shifted["handle_state"],
        torch.roll(reference["handle_state"], shifts=shift, dims=1),
    )
    assert torch.equal(
        shifted["primary_index"],
        torch.remainder(reference["primary_index"] + shift, 4),
    )
    assert torch.equal(
        shifted["candidate_handle"],
        torch.remainder(reference["candidate_handle"] + shift, 4),
    )
    for name in (
        "vehicle_state",
        "conditional_delta_m",
        "conditional_position_m",
        "switch_logits",
        "switch_probability",
        "delta_m",
        "position_m",
    ):
        _assert_same(shifted[name], reference[name])
    assert torch.equal(
        shifted["selected_switch_step"], reference["selected_switch_step"]
    )


@torch.inference_mode()
def test_query_and_candidate_row_permutations_preserve_semantics() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)

    query_permutation = torch.tensor([2, 0, 3, 1])
    query_batch = deepcopy(batch)
    query_batch["tau_s"] = query_batch["tau_s"][:, query_permutation]
    query_output = model(query_batch)
    for name in (
        "conditional_delta_m",
        "conditional_position_m",
        "switch_logits",
        "switch_probability",
        "selected_candidate_row",
        "selected_switch_step",
        "delta_m",
        "position_m",
    ):
        _assert_same(query_output[name], reference[name][:, query_permutation])

    candidate_permutation = torch.tensor([6, 0, 12, 2, 10, 4, 8, 1, 11, 3, 9, 5, 7])
    candidate_batch = deepcopy(batch)
    for name in (
        "candidate_relation_m",
        "candidate_step",
        "candidate_mask",
        "candidate_confidence",
        "candidate_supported",
    ):
        candidate_batch[name] = candidate_batch[name][:, candidate_permutation]
    candidate_output = model(candidate_batch)
    for name in ("candidate_handle",):
        _assert_same(candidate_output[name], reference[name][:, candidate_permutation])
    for name in (
        "conditional_delta_m",
        "conditional_position_m",
        "switch_logits",
        "switch_probability",
    ):
        _assert_same(candidate_output[name], reference[name][:, :, candidate_permutation])
    assert torch.equal(
        candidate_output["selected_switch_step"], reference["selected_switch_step"]
    )
    _assert_same(candidate_output["delta_m"], reference["delta_m"])
    _assert_same(candidate_output["position_m"], reference["position_m"])


@torch.inference_mode()
def test_forbidden_truth_motion_and_session_fields_cannot_affect_forward() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)
    polluted = deepcopy(batch)
    polluted.update(
        {
            "target_visible_delta_m": torch.randn(2, 4, 3) * 1e6,
            "target_switch_count": torch.randint(-99, 100, (2, 4)),
            "truth_current_position_m": torch.randn(2, 3) * 1e6,
            "truth_vehicle_state": torch.randn(2, 32) * 1e6,
            "motion_class": torch.tensor([2, 3]),
            "session_index": torch.tensor([91234, -571]),
            "future_pnp_position_m": torch.randn(2, 4, 3) * 1e6,
        }
    )
    polluted_output = model(polluted)
    assert polluted_output.keys() == reference.keys()
    for name in reference:
        if reference[name].dtype.is_floating_point:
            assert torch.equal(polluted_output[name], reference[name]), name
        else:
            assert torch.equal(polluted_output[name], reference[name]), name


@torch.inference_mode()
def test_inactive_prefix_nan_and_huge_values_are_strictly_ignored() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)
    polluted = deepcopy(batch)
    polluted["history_obs_rel_m"][:, :2] = torch.nan
    polluted["history_obs_mask"][:, :2] = True
    polluted["history_primary_mask"][:, :2] = True
    polluted["history_time_s"][:, :2] = torch.nan
    polluted["history_dt_s"][:, :2] = 1e30
    polluted["history_switch_step"][:, :2] = 999999
    polluted_output = model(polluted)
    for name in reference:
        _assert_same(polluted_output[name], reference[name])


@torch.inference_mode()
def test_handle_reflection_swaps_direction_semantics_exactly() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)
    reflected = model(_reflect_handles(batch))
    index = HANDLE_REFLECTION.to(reference["handle_state"].device)

    _assert_same(
        reflected["handle_state"], reference["handle_state"].index_select(1, index)
    )
    _assert_same(reflected["vehicle_state"], reference["vehicle_state"])
    assert torch.equal(
        reflected["candidate_handle"],
        torch.remainder(-reference["candidate_handle"], 4),
    )
    for name in (
        "conditional_delta_m",
        "conditional_position_m",
        "switch_logits",
        "switch_probability",
        "delta_m",
        "position_m",
    ):
        _assert_same(reflected[name], reference[name])
    for name in (
        "direction_logits",
        "direction_probability",
        "crossing_interval_s",
        "crossing_time_s",
        "crossing_temperature_s",
    ):
        _assert_same(reflected[name], reference[name].flip(1))
    assert torch.equal(
        reflected["selected_switch_step"], -reference["selected_switch_step"]
    )
