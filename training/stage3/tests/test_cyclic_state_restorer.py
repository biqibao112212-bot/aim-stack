from __future__ import annotations

import inspect

import torch

from training.stage3.cyclic_state_loss import cyclic_state_loss
from training.stage3.cyclic_state_model import (
    CyclicStateRestorer,
    current_track_support,
)


def _batch(batch: int = 4, time: int = 8) -> dict[str, torch.Tensor]:
    torch.manual_seed(19)
    obs = torch.zeros(batch, time, 4, 3)
    mask = torch.zeros(batch, time, 4, dtype=torch.bool)
    primary = torch.zeros_like(mask)
    for row in range(batch):
        for event in range(time):
            track = (row + event // 3) % 4
            obs[row, event, track] = torch.tensor(
                (2.5 + 0.01 * event, 0.15 * track, 0.1)
            )
            mask[row, event, track] = True
            primary[row, event, track] = True
            if event in (2, 5):
                neighbor = (track + 1) % 4
                obs[row, event, neighbor] = torch.tensor(
                    (2.55 + 0.01 * event, 0.15 * neighbor, 0.11)
                )
                mask[row, event, neighbor] = True
    switch = torch.zeros(batch, time)
    switch[:, 3] = 1
    switch[:, 6] = 1
    return {
        "obs": obs,
        "obs_mask": mask,
        "primary_mask": primary,
        "event_mask": torch.ones(batch, time, dtype=torch.bool),
        "event_time_s": torch.linspace(-0.07, 0.0, time).repeat(batch, 1),
        "switch_step": switch,
    }


def _model() -> CyclicStateRestorer:
    return CyclicStateRestorer(
        torch.tensor((3.0, 0.0, 0.1)),
        torch.tensor((1.0, 1.0, 0.1)),
        channels=8, dropout=0.0, history_events=8,
    )


def _forward(model: CyclicStateRestorer, batch: dict[str, torch.Tensor]):
    return model(
        batch["obs"], batch["obs_mask"], batch["primary_mask"],
        batch["event_mask"], batch["event_time_s"], batch["switch_step"],
    )


def test_forward_contract_has_no_truth_class_or_geometry_input() -> None:
    assert list(inspect.signature(CyclicStateRestorer.forward).parameters) == [
        "self", "obs", "obs_mask", "primary_mask", "event_mask",
        "event_time_s", "switch_step",
    ]
    config = _model().config()
    assert config["q0_observed_identity_bypass"] is True
    assert config["cold_tracks_are_invalid"] is True
    assert config["fixed_geometry"] is False
    assert config["slot_features"] is False


def test_every_current_visible_track_is_an_exact_identity_bypass() -> None:
    model = _model().eval()
    batch = _batch()
    # Make the current event dual-visible; secondary must bypass as well.
    primary_index = batch["primary_mask"][:, -1].long().argmax(dim=-1)
    for row, track in enumerate(primary_index.tolist()):
        neighbor = (track + 1) % 4
        batch["obs_mask"][row, -1, neighbor] = True
        batch["obs"][row, -1, neighbor] = torch.tensor((2.8, 0.3, 0.12))
    output = _forward(model, batch)
    visible = batch["obs_mask"][:, -1]
    expected = model.position_mean + model.position_std * batch["obs"][:, -1]
    assert torch.equal(output["q0_m"][visible], expected[visible])
    assert torch.all(output["q0_valid"][visible])
    assert torch.all(output["confidence"][visible] > 0.999)


def test_visible_event_before_q0_is_propagated_not_forced_to_identity() -> None:
    model = _model().eval()
    batch = _batch(batch=1)
    batch["event_time_s"][:, -1] = -0.02
    output = _forward(model, batch)
    assert not bool(output["q0_observed"].any())
    visible = output["current_visible"]
    observed = model.position_mean + model.position_std * batch["obs"][:, -1]
    assert not torch.equal(output["q0_m"][visible], observed[visible])


def test_identity_bypass_uses_the_consumed_tail_not_the_source_prefix() -> None:
    model = _model().eval()
    batch = _batch(batch=1)
    prefix_obs = torch.full((1, 3, 4, 3), -99.0)
    prefix_mask = torch.zeros(1, 3, 4, dtype=torch.bool)
    prefix_primary = torch.zeros_like(prefix_mask)
    prefix_event = torch.zeros(1, 3, dtype=torch.bool)
    prefix_time = torch.full((1, 3), -10.0)
    prefix_switch = torch.zeros(1, 3)
    extended = {
        "obs": torch.cat((prefix_obs, batch["obs"]), dim=1),
        "obs_mask": torch.cat((prefix_mask, batch["obs_mask"]), dim=1),
        "primary_mask": torch.cat((prefix_primary, batch["primary_mask"]), dim=1),
        "event_mask": torch.cat((prefix_event, batch["event_mask"]), dim=1),
        "event_time_s": torch.cat((prefix_time, batch["event_time_s"]), dim=1),
        "switch_step": torch.cat((prefix_switch, batch["switch_step"]), dim=1),
    }
    output = _forward(model, extended)
    visible = batch["obs_mask"][:, -1]
    expected = model.position_mean + model.position_std * batch["obs"][:, -1]
    assert torch.equal(output["q0_m"][visible], expected[visible])


def test_support_marks_seen_hidden_warm_and_never_seen_cold() -> None:
    batch = _batch(batch=1)
    support = current_track_support(
        batch["obs_mask"], batch["primary_mask"], batch["event_mask"],
        batch["event_time_s"], history_events=8,
    )
    assert torch.equal(
        support["seen"], support["current_visible"] | support["warm_hidden"]
    )
    assert torch.equal(support["cold"], ~support["seen"])
    assert not bool((support["warm_hidden"] & support["cold"]).any())


def test_state_restorer_is_exactly_c4_roll_equivariant() -> None:
    model = _model().eval()
    batch = _batch()
    reference = _forward(model, batch)
    for shift in (1, 2, 3):
        shifted = dict(batch)
        for name in ("obs", "obs_mask", "primary_mask"):
            shifted[name] = torch.roll(batch[name], shifts=shift, dims=2)
        output = _forward(model, shifted)
        for name in (
            "q0_m", "q0_valid", "q0_sigma_m", "confidence",
            "edge0_m", "edge0_valid", "current_visible", "warm_hidden",
            "self_warm", "edge_warm", "cold", "adjacent", "pair_seen",
        ):
            dimension = 1
            assert torch.allclose(
                output[name] if output[name].dtype.is_floating_point else output[name].float(),
                torch.roll(reference[name], shifts=shift, dims=dimension)
                if reference[name].dtype.is_floating_point else
                torch.roll(reference[name], shifts=shift, dims=dimension).float(),
                atol=2e-6,
            )


def test_cold_and_nonrotating_hidden_targets_cannot_change_loss() -> None:
    model = _model().eval()
    batch = _batch(batch=2)
    output = _forward(model, batch)
    target = output["q0_m"].detach().clone()
    labels = torch.tensor((0, 1), dtype=torch.long)
    first, _ = cyclic_state_loss(output, target, labels)
    changed = target.clone()
    hidden_or_cold = ~output["current_visible"]
    changed[hidden_or_cold] += 1000.0
    second, _ = cyclic_state_loss(output, changed, labels)
    assert torch.equal(first, second)


def test_dynamic_cold_targets_cannot_change_loss() -> None:
    model = _model().eval()
    batch = _batch(batch=2)
    output = _forward(model, batch)
    target = output["q0_m"].detach().clone()
    labels = torch.tensor((2, 3), dtype=torch.long)
    first, _ = cyclic_state_loss(output, target, labels)
    changed = target.clone()
    changed[output["cold"]] += 1000.0
    second, _ = cyclic_state_loss(output, changed, labels)
    assert torch.equal(first, second)


def test_dynamic_warm_adjacent_tracks_drive_position_gradients() -> None:
    model = _model().train()
    batch = _batch(batch=2)
    output = _forward(model, batch)
    selected = output["warm_hidden"] & output["adjacent"]
    assert bool(selected.any())
    target = output["q0_m"].detach().clone()
    target[selected] += 0.05
    labels = torch.tensor((2, 3), dtype=torch.long)
    total, parts = cyclic_state_loss(output, target, labels)
    assert torch.isfinite(total)
    assert parts["selected_fraction"] > 0
    total.backward()
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.position_head.parameters()
    )


def test_support_rejects_more_than_two_visible_tracks() -> None:
    batch = _batch(batch=1)
    batch["obs_mask"][:, -1, :3] = True
    try:
        _forward(_model(), batch)
    except ValueError as error:
        assert "at most two" in str(error)
    else:
        raise AssertionError("three current visible tracks must fail closed")
