from __future__ import annotations

import torch

from training.stage3.causal_physical_state_model import (
    FactorizedExpertPhysicalPredictor, IndependentMotionExpertSystem,
)
from training.stage3.moving_refinement_loss import moving_refinement_loss
from training.stage3.moving_refinement_model import MovingRefinementSystem


MEAN = torch.tensor([3.0, 0.5, 0.7])
STD = torch.tensor([1.0, 1.0, 0.25])


def _geometry() -> torch.Tensor:
    return torch.tensor([
        [0.22, 0.00, -0.07], [0.00, -0.21, -0.06],
        [-0.20, 0.00, -0.07], [0.00, 0.21, -0.06],
    ])


def _base() -> IndependentMotionExpertSystem:
    arguments = (_geometry(), MEAN, STD)
    original = FactorizedExpertPhysicalPredictor(
        *arguments, channels=8, dropout=0.0, history_events=8,
    )
    augmented = FactorizedExpertPhysicalPredictor(
        *arguments, channels=8, dropout=0.0, history_events=8,
    )
    model = IndependentMotionExpertSystem(
        *arguments, channels=8, dropout=0.0, history_events=8,
    )
    model.initialize_from_factorized(augmented, original)
    model.freeze_foundations()
    return model


def _model() -> MovingRefinementSystem:
    return MovingRefinementSystem(
        _base(), position_mean=MEAN, position_std=STD,
        refinement_channels=8, refinement_dropout=0.0, history_events=8,
    )


def _constant_head(head: torch.nn.Sequential, values: list[float]) -> None:
    for parameter in head.parameters():
        parameter.data.zero_()
    head[-1].bias.data.copy_(torch.tensor(values))


def _input(batch: int = 2) -> tuple[torch.Tensor, ...]:
    physical = MEAN + 0.05 * torch.randn(batch, 8, 4, 3)
    normalized = (physical - MEAN) / STD
    angle = torch.arange(4) * torch.pi / 2
    slot = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
    slot = slot.view(1, 1, 4, 2).expand(batch, 8, 4, 2)
    obs = torch.cat((normalized, slot), dim=-1)
    obs_mask = torch.ones(batch, 8, 4, dtype=torch.bool)
    event_mask = torch.ones(batch, 8, dtype=torch.bool)
    event_time = torch.linspace(-0.07, 0.0, 8).repeat(batch, 1)
    tau = torch.tensor([0.0, 0.1, 0.2, 0.5]).repeat(batch, 1)
    return obs, obs_mask, event_mask, event_time, tau


def _future(
    velocity: torch.Tensor, omega: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = velocity.shape[0]
    tau = torch.tensor([0.0, 0.1, 0.2, 0.5]).repeat(batch, 1)
    center = MEAN[None, None] + tau[..., None] * velocity[:, None]
    angle = tau * omega[:, None]
    cosine, sine = torch.cos(angle), torch.sin(angle)
    geometry = _geometry()
    gx, gy = geometry[:, 0], geometry[:, 1]
    relative = torch.stack((
        cosine[..., None] * gx - sine[..., None] * gy,
        sine[..., None] * gx + cosine[..., None] * gy,
        geometry[:, 2].view(1, 1, 4).expand(batch, 4, 4),
    ), dim=-1)
    return center[:, :, None] + relative, tau, torch.ones_like(tau, dtype=torch.bool)


def test_relative_observation_is_translation_invariant() -> None:
    model = _model()
    obs, obs_mask, event_mask, _, _ = _input()
    shifted = obs.clone()
    translation = torch.tensor([1.2, -0.7, 0.1])
    shifted[..., :3] += translation / STD
    first = model.relative_observation(obs, obs_mask, event_mask)
    second = model.relative_observation(shifted, obs_mask, event_mask)
    # De-normalizing meter-scale positions in FP32 can lose a few ulps after
    # adding and then removing a global translation.  The residual is still
    # four orders of magnitude below the normalized rigid-shape signal.
    assert torch.allclose(first, second, atol=3e-6)


def test_only_refinement_parameters_are_trainable() -> None:
    model = _model()
    trainable = {
        name for name, value in model.named_parameters() if value.requires_grad
    }
    assert trainable
    assert all(name.startswith("refinement_") for name in trainable)


def test_moving_binary_loss_balances_translation_and_combined() -> None:
    velocity = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    omega = torch.tensor([0.0, 2.0])
    target, tau, rule = _future(velocity, omega)
    logit = torch.zeros(2, requires_grad=True)
    total, parts = moving_refinement_loss(
        logit, target, tau, rule, _geometry(),
    )
    total.backward()
    assert torch.isfinite(total)
    assert logit.grad is not None and bool(torch.count_nonzero(logit.grad))
    assert torch.isclose(parts["translation_fraction"], torch.tensor(0.5))
    assert torch.isclose(parts["combined_fraction"], torch.tensor(0.5))


def test_refinement_changes_only_the_moving_route() -> None:
    model = _model().eval()
    inputs = _input()
    _constant_head(model.base.router_head, [0.0, 10.0, 0.0, 0.0])
    _constant_head(model.refinement_head, [-10.0])
    translation = model(*inputs)
    assert torch.equal(translation["route_index"], torch.ones(2, dtype=torch.long))
    _constant_head(model.refinement_head, [10.0])
    combined = model(*inputs)
    assert torch.equal(combined["route_index"], torch.full((2,), 3, dtype=torch.long))

    _constant_head(model.base.router_head, [0.0, 0.0, 10.0, 0.0])
    _constant_head(model.refinement_head, [-10.0])
    rotation_a = model(*inputs)
    _constant_head(model.refinement_head, [10.0])
    rotation_b = model(*inputs)
    assert torch.equal(rotation_a["route_index"], torch.full((2,), 2, dtype=torch.long))
    assert torch.equal(rotation_a["route_index"], rotation_b["route_index"])
