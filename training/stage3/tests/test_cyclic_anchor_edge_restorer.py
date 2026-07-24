from __future__ import annotations

import inspect

import torch

from training.stage3.cyclic_anchor_edge_loss import cyclic_anchor_edge_loss
from training.stage3.cyclic_anchor_edge_model import CyclicAnchorEdgeRestorer
from training.stage3.cyclic_state_model import CyclicStateRestorer


def _model() -> CyclicAnchorEdgeRestorer:
    return CyclicAnchorEdgeRestorer(
        torch.tensor((3.0, 0.0, 0.1)),
        torch.tensor((1.0, 1.0, 0.1)),
        channels=8,
        dropout=0.0,
        history_events=8,
    )


def _async_batch(batch: int = 2) -> dict[str, torch.Tensor]:
    torch.manual_seed(31)
    time = 8
    obs = torch.zeros(batch, time, 4, 3)
    visible = torch.zeros(batch, time, 4, dtype=torch.bool)
    primary = torch.zeros_like(visible)
    # Tracks 1 and 3 are independently seen before current primary 0.  No pair
    # is ever co-visible, so both relevant edges are asynchronous supports.
    sequence = (1, 1, 0, 0, 3, 3, 0, 0)
    for row in range(batch):
        for event, track in enumerate(sequence):
            obs[row, event, track] = torch.tensor((
                2.0 + 0.02 * event + 0.1 * row,
                0.2 * track - 0.01 * event,
                0.08 + 0.01 * track,
            ))
            visible[row, event, track] = True
            primary[row, event, track] = True
    switch = torch.zeros(batch, time)
    switch[:, 2] = -1
    switch[:, 4] = -1
    switch[:, 6] = 1
    return {
        "obs": obs,
        "obs_mask": visible,
        "primary_mask": primary,
        "event_mask": torch.ones(batch, time, dtype=torch.bool),
        "event_time_s": torch.linspace(-0.07, 0.0, time).repeat(batch, 1),
        "switch_step": switch,
    }


def _forward(model: CyclicAnchorEdgeRestorer, batch: dict[str, torch.Tensor]):
    return model(
        batch["obs"], batch["obs_mask"], batch["primary_mask"],
        batch["event_mask"], batch["event_time_s"], batch["switch_step"],
    )


def test_anchor_edge_forward_has_no_truth_class_or_geometry_input() -> None:
    assert list(inspect.signature(CyclicAnchorEdgeRestorer.forward).parameters) == [
        "self", "obs", "obs_mask", "primary_mask", "event_mask",
        "event_time_s", "switch_step",
    ]
    config = _model().config()
    assert config["fixed_geometry"] is False
    assert config["slot_features"] is False
    assert config["asynchronous_endpoint_edge_support"] is True
    assert config["hidden_position_parameterization"] == (
        "current anchor plus directed edge"
    )


def test_v1_foundation_loads_with_only_v2_heads_missing() -> None:
    v1 = CyclicStateRestorer(
        torch.tensor((3.0, 0.0, 0.1)),
        torch.tensor((1.0, 1.0, 0.1)),
        channels=8, dropout=0.0, history_events=8,
    )
    v2 = _model()
    incompatible = v2.load_state_dict(v1.state_dict(), strict=False)
    expected = {
        name for name in v2.state_dict()
        if name.startswith(v2.new_parameter_prefixes())
    }
    assert set(incompatible.missing_keys) == expected
    assert incompatible.unexpected_keys == []


def test_asynchronously_seen_neighbors_are_anchor_composed() -> None:
    model = _model().eval()
    output = _forward(model, _async_batch(batch=1))
    assert not bool(output["pair_seen"].any())
    assert bool(output["edge0_async_supported"][0, 0])
    assert bool(output["edge0_async_supported"][0, 3])
    assert bool(output["anchor_composed"][0, 1])
    assert bool(output["anchor_composed"][0, 3])
    anchor = output["q0_m"][0, 0]
    assert torch.allclose(
        output["q0_m"][0, 1], anchor + output["edge0_m"][0, 0], atol=1e-6
    )
    assert torch.allclose(
        output["q0_m"][0, 3], anchor - output["edge0_m"][0, 3], atol=1e-6
    )
    assert torch.equal(
        output["q0_valid"][0], torch.tensor((True, True, False, True))
    )


def test_async_residual_cannot_overwrite_pair_seen_edge() -> None:
    model = _model().eval()
    with torch.no_grad():
        model.async_edge_head[-1].bias.fill_(10.0)
    batch = _async_batch(batch=1)
    # Establish historical co-visibility for edge 0 -> 1 only.
    batch["obs_mask"][0, 2, 1] = True
    batch["obs"][0, 2, 1] = torch.tensor((2.1, 0.2, 0.09))
    output = _forward(model, batch)
    assert bool(output["pair_seen"][0, 0])
    assert not bool(output["edge0_async_supported"][0, 0])
    assert torch.equal(
        output["edge0_m"][0, 0], output["edge0_foundation_m"][0, 0]
    )
    assert bool(output["edge0_async_supported"][0, 3])
    assert not torch.equal(
        output["edge0_m"][0, 3], output["edge0_foundation_m"][0, 3]
    )


def test_all_current_visible_tracks_remain_exact_measurements() -> None:
    model = _model().eval()
    batch = _async_batch(batch=1)
    batch["obs_mask"][0, -1, 1] = True
    batch["obs"][0, -1, 1] = torch.tensor((2.7, 0.25, 0.11))
    output = _forward(model, batch)
    expected = model.position_mean + model.position_std * batch["obs"][:, -1]
    visible = batch["obs_mask"][:, -1]
    assert torch.equal(output["q0_m"][visible], expected[visible])
    assert torch.all(output["q0_valid"][visible])


def test_anchor_edge_model_is_exactly_c4_roll_equivariant() -> None:
    model = _model().eval()
    batch = _async_batch()
    reference = _forward(model, batch)
    for shift in (1, 2, 3):
        shifted = dict(batch)
        for name in ("obs", "obs_mask", "primary_mask"):
            shifted[name] = torch.roll(batch[name], shifts=shift, dims=2)
        output = _forward(model, shifted)
        for name in (
            "q0_m", "q0_valid", "q0_sigma_m", "confidence", "edge0_m",
            "edge0_valid", "edge0_supported", "anchor_composed",
            "current_visible", "warm_hidden", "cold", "adjacent",
        ):
            actual = output[name]
            expected = torch.roll(reference[name], shifts=shift, dims=1)
            if actual.dtype.is_floating_point:
                assert torch.allclose(actual, expected, atol=2e-6)
            else:
                assert torch.equal(actual, expected)


def test_async_hidden_targets_drive_new_edge_head_gradients() -> None:
    model = _model().train()
    output = _forward(model, _async_batch())
    target = output["q0_m"].detach().clone()
    target[output["anchor_composed"]] += torch.tensor((0.04, -0.03, 0.02))
    labels = torch.tensor((2, 3), dtype=torch.long)
    total, parts = cyclic_anchor_edge_loss(output, target, labels)
    assert torch.isfinite(total)
    assert parts["async_edge_fraction"] > 0
    total.backward()
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.async_edge_head.parameters()
    )


def test_cold_and_nonrotating_hidden_targets_remain_excluded() -> None:
    model = _model().eval()
    output = _forward(model, _async_batch())
    target = output["q0_m"].detach().clone()
    labels = torch.tensor((0, 1), dtype=torch.long)
    first, _ = cyclic_anchor_edge_loss(output, target, labels)
    changed = target.clone()
    changed[~output["current_visible"]] += 1000.0
    second, _ = cyclic_anchor_edge_loss(output, changed, labels)
    assert torch.equal(first, second)


def test_dynamic_cold_targets_remain_excluded() -> None:
    model = _model().eval()
    output = _forward(model, _async_batch())
    target = output["q0_m"].detach().clone()
    labels = torch.tensor((2, 3), dtype=torch.long)
    first, _ = cyclic_anchor_edge_loss(output, target, labels)
    changed = target.clone()
    changed[output["cold"]] += 1000.0
    second, _ = cyclic_anchor_edge_loss(output, changed, labels)
    assert torch.equal(first, second)
