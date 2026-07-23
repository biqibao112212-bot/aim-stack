from __future__ import annotations

import inspect

import numpy as np
import torch

from training.stage3.cyclic_track_dataset import (
    construct_cyclic_visibility,
    cyclic_relabel,
)
from training.stage3.cyclic_track_loss import cyclic_track_expert_loss
from training.stage3.cyclic_track_model import CyclicTrackExpertSystem


def _batch(batch: int = 4, time: int = 8) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    obs = torch.zeros(batch, time, 4, 3)
    mask = torch.zeros(batch, time, 4, dtype=torch.bool)
    primary = torch.zeros_like(mask)
    for b in range(batch):
        for t in range(time):
            track = (b + t // 3) % 4
            obs[b, t, track] = torch.tensor((2.5 + 0.01 * t, 0.2 * track, 0.1))
            mask[b, t, track] = True
            primary[b, t, track] = True
            if t % 4 == 0:
                neighbor = (track + 1) % 4
                obs[b, t, neighbor] = torch.tensor((2.6, 0.2 * neighbor, 0.11))
                mask[b, t, neighbor] = True
    switch = torch.zeros(batch, time)
    switch[:, 3] = 1
    switch[:, 6] = 1
    tau = torch.tensor((0.0, 0.1, 0.2, 0.5)).repeat(batch, 1)
    return {
        "obs": obs, "obs_mask": mask, "primary_mask": primary,
        "event_mask": torch.ones(batch, time, dtype=torch.bool),
        "event_time_s": torch.linspace(-0.07, 0.0, time).repeat(batch, 1),
        "switch_step": switch, "tau": tau,
    }


def _model() -> CyclicTrackExpertSystem:
    return CyclicTrackExpertSystem(
        torch.tensor((3.0, 0.0, 0.1)),
        torch.tensor((1.0, 1.0, 0.1)),
        channels=8, dropout=0.0, history_events=8,
    )


def _forward(model: CyclicTrackExpertSystem, batch: dict[str, torch.Tensor]):
    return model(
        batch["obs"], batch["obs_mask"], batch["primary_mask"],
        batch["event_mask"], batch["event_time_s"], batch["switch_step"],
        batch["tau"],
    )


def test_forward_contract_has_no_truth_or_geometry_inputs() -> None:
    parameters = list(inspect.signature(CyclicTrackExpertSystem.forward).parameters)
    assert parameters == [
        "self", "obs", "obs_mask", "primary_mask", "event_mask",
        "event_time_s", "switch_step", "tau",
    ]
    config = _model().config()
    assert config["fixed_geometry"] is False
    assert config["slot_features"] is False
    assert not any(
        forbidden in name.lower()
        for name, _ in _model().named_parameters()
        for forbidden in ("geometry", "phase", "center", "slot_embedding", "slot_queries")
    )


def test_constructed_visibility_is_one_or_two_adjacent_with_adjacent_switches() -> None:
    angle = np.linspace(0.0, np.pi, 25, dtype=np.float32)
    center = np.asarray((3.0, 0.0, 0.1), dtype=np.float32)
    geometry = np.asarray([
        (0.2, 0.0, 0.0), (0.0, 0.2, 0.01),
        (-0.2, 0.0, 0.0), (0.0, -0.2, 0.01),
    ], dtype=np.float32)
    position = []
    for yaw in angle:
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.asarray(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=np.float32)
        position.append(center + geometry @ rotation.T)
    position = np.asarray(position)
    event = np.ones(len(position), dtype=np.bool_)
    visible, primary, switch = construct_cyclic_visibility(
        position, event, secondary_gap_ratio=0.25
    )
    assert np.all(np.isin(visible.sum(axis=1), (1, 2)))
    assert np.all(primary.sum(axis=1) == 1)
    assert np.all(primary <= visible)
    for row in visible[visible.sum(axis=1) == 2]:
        pair = np.flatnonzero(row)
        assert int((pair[1] - pair[0]) % 4) in (1, 3)
    assert set(np.unique(switch)).issubset({-1, 0, 1})


def test_cyclic_relabel_round_trip_preserves_temp_track_values() -> None:
    value = np.arange(2 * 4 * 3).reshape(2, 4, 3)
    for shift in range(4):
        shifted = cyclic_relabel(value, shift=shift, reverse=False, axis=1)
        assert np.array_equal(np.roll(shifted, -shift, axis=1), value)
    reversed_value = cyclic_relabel(value, shift=0, reverse=True, axis=1)
    assert np.array_equal(
        cyclic_relabel(reversed_value, shift=0, reverse=True, axis=1), value
    )


def test_direction_reversal_flips_switch_and_preserves_primary_consistency() -> None:
    primary = np.zeros((4, 4), dtype=np.bool_)
    primary[np.arange(4), np.arange(4)] = True
    reversed_primary = cyclic_relabel(primary, shift=0, reverse=True, axis=1)
    reversed_index = reversed_primary.argmax(axis=1)
    assert reversed_index.tolist() == [0, 3, 2, 1]
    original_switch = np.asarray((0, 1, 1, 1), dtype=np.int8)
    reversed_switch = -original_switch
    expected = [0]
    for current, previous in zip(reversed_index[1:], reversed_index[:-1]):
        delta = int((current - previous) % 4)
        expected.append(1 if delta == 1 else -1 if delta == 3 else 0)
    assert reversed_switch.tolist() == expected


def test_all_experts_are_c4_equivariant_and_router_is_invariant() -> None:
    model = _model().eval()
    batch = _batch()
    reference = _forward(model, batch)
    for shift in (1, 2, 3):
        shifted = dict(batch)
        for name in ("obs", "obs_mask", "primary_mask"):
            shifted[name] = torch.roll(batch[name], shifts=shift, dims=2)
        output = _forward(model, shifted)
        assert torch.allclose(output["router_logit"], reference["router_logit"], atol=1e-6)
        assert torch.equal(output["route_index"], reference["route_index"])
        assert torch.allclose(
            output["expert_position"],
            torch.roll(reference["expert_position"], shifts=shift, dims=3),
            atol=2e-6,
        )
        assert torch.allclose(
            output["position_mean"],
            torch.roll(reference["position_mean"], shifts=shift, dims=2),
            atol=2e-6,
        )


def test_combined_expert_is_parameter_and_output_independent() -> None:
    model = _model().eval()
    batch = _batch()
    combined_before = _forward(model, batch)["expert_position"][:, 3].detach().clone()
    translation_pointers = {p.data_ptr() for p in model.experts[1].parameters()}
    combined_pointers = {p.data_ptr() for p in model.experts[3].parameters()}
    assert translation_pointers.isdisjoint(combined_pointers)
    with torch.no_grad():
        for parameter in model.experts[0].parameters():
            parameter.add_(torch.randn_like(parameter))
        for parameter in model.experts[1].parameters():
            parameter.add_(torch.randn_like(parameter))
        for parameter in model.experts[2].parameters():
            parameter.add_(torch.randn_like(parameter))
    combined_after = _forward(model, batch)["expert_position"][:, 3]
    assert torch.equal(combined_before, combined_after)


def _module_has_gradient(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in module.parameters()
    )


def test_positive_only_loss_isolates_expert_gradients_and_has_no_shift_search() -> None:
    for route in range(4):
        model = _model().train()
        batch = _batch(batch=2)
        output = _forward(model, batch)
        target = output["expert_position"][:, route].detach() + 0.05
        labels = torch.full((2,), route, dtype=torch.long)
        current_primary = batch["primary_mask"][:, -1].long().argmax(dim=-1)
        total, parts = cyclic_track_expert_loss(
            output, target, labels, torch.ones(2, 4, dtype=torch.bool),
            current_primary, batch["obs_mask"][:, -1]
        )
        assert torch.isfinite(total)
        assert set(("position", "motion_delta", "self_rigid", "router")) <= parts.keys()
        total.backward()
        assert [_module_has_gradient(expert) for expert in model.experts] == [
            index == route for index in range(4)
        ]
        assert _module_has_gradient(model.router_context)


def test_model_rejects_more_than_two_visible_and_invalid_switch() -> None:
    model = _model()
    batch = _batch(batch=1)
    batch["obs_mask"][:, 0, :3] = True
    try:
        _forward(model, batch)
    except ValueError as error:
        assert "at most two" in str(error)
    else:
        raise AssertionError("three visible tracks must fail closed")
    batch = _batch(batch=1)
    batch["switch_step"][:, 0] = 2
    try:
        _forward(model, batch)
    except ValueError as error:
        assert "switch_step" in str(error)
    else:
        raise AssertionError("invalid switch must fail closed")


def test_ineligible_future_query_cannot_change_training_objective() -> None:
    model = _model().eval()
    batch = _batch(batch=2)
    output = _forward(model, batch)
    labels = torch.tensor((0, 1), dtype=torch.long)
    target = output["expert_position"][torch.arange(2), labels].detach().clone()
    primary = batch["primary_mask"][:, -1].long().argmax(dim=-1)
    rule = torch.tensor(((True, True, False, False), (True, True, False, False)))
    first, _ = cyclic_track_expert_loss(
        output, target, labels, rule, primary, batch["obs_mask"][:, -1]
    )
    changed = target.clone()
    changed[:, 2:] += 1000.0
    second, _ = cyclic_track_expert_loss(
        output, changed, labels, rule, primary, batch["obs_mask"][:, -1]
    )
    assert torch.equal(first, second)
