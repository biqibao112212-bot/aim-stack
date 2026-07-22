from __future__ import annotations

import inspect

import torch

from training.stage3.causal_physical_state_model import (
    FactorizedExpertPhysicalPredictor,
    IndependentMotionExpertSystem,
)
from training.stage3.independent_motion_expert_loss import (
    independent_motion_expert_loss,
)


def _geometry() -> torch.Tensor:
    return torch.tensor([
        [0.22, 0.00, -0.07], [0.00, -0.21, -0.06],
        [-0.20, 0.00, -0.07], [0.00, 0.21, -0.06],
    ])


def _factorized() -> FactorizedExpertPhysicalPredictor:
    return FactorizedExpertPhysicalPredictor(
        _geometry(), torch.tensor([3.0, 0.5, 0.7]),
        torch.tensor([1.0, 1.0, 0.25]), channels=8,
        dropout=0.05, history_events=8,
    )


def _system() -> IndependentMotionExpertSystem:
    original = _factorized()
    augmented = _factorized()
    model = IndependentMotionExpertSystem(
        _geometry(), torch.tensor([3.0, 0.5, 0.7]),
        torch.tensor([1.0, 1.0, 0.25]), channels=8,
        dropout=0.05, history_events=8,
    )
    model.initialize_from_factorized(augmented, original)
    model.freeze_foundations()
    return model


def _input(batch: int = 2) -> tuple[torch.Tensor, ...]:
    obs = torch.randn(batch, 8, 4, 5)
    obs_mask = torch.ones(batch, 8, 4, dtype=torch.bool)
    event_mask = torch.ones(batch, 8, dtype=torch.bool)
    event_time = torch.linspace(-0.07, 0.0, 8).repeat(batch, 1)
    tau = torch.tensor([0.0, 0.1, 0.2, 0.5]).repeat(batch, 1)
    return obs, obs_mask, event_mask, event_time, tau


def _constant_head(head: torch.nn.Sequential, values: list[float]) -> None:
    for parameter in head.parameters():
        parameter.data.zero_()
    head[-1].bias.data.copy_(torch.tensor(values))


def _set_route(model: IndependentMotionExpertSystem, route: int) -> None:
    _constant_head(model.router_head, [0.0, 0.0, 0.0, 0.0])
    model.router_head[-1].bias.data[route] = 10.0


def _future(
    velocity: torch.Tensor, omega: float, batch: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tau = torch.tensor([0.0, 0.1, 0.2, 0.5]).repeat(batch, 1)
    center0 = torch.tensor([3.0, 0.5, 0.7]).repeat(batch, 1)
    center = center0[:, None] + tau[..., None] * velocity.repeat(batch, 1)[:, None]
    angle = tau * omega
    phase = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
    geometry = _geometry()
    cosine, sine = phase[..., 0], phase[..., 1]
    gx = geometry[:, 0].view(1, 1, 4)
    gy = geometry[:, 1].view(1, 1, 4)
    relative = torch.stack((
        cosine[..., None] * gx - sine[..., None] * gy,
        sine[..., None] * gx + cosine[..., None] * gy,
        geometry[:, 2].view(1, 1, 4).expand(batch, 4, 4),
    ), dim=-1)
    return center[:, :, None] + relative, tau, torch.ones_like(tau, dtype=torch.bool)


def test_forward_contract_has_no_truth_or_motion_state_inputs() -> None:
    parameters = list(inspect.signature(IndependentMotionExpertSystem.forward).parameters)
    assert parameters == [
        "self", "obs", "obs_mask", "event_mask", "event_time_s", "tau",
    ]


def test_source_mapping_and_foundations_remain_frozen_eval() -> None:
    original = _factorized()
    augmented = _factorized()
    with torch.no_grad():
        augmented.q0_head[-1].bias.fill_(0.25)
        original.translation_expert[-1].bias.fill_(-0.5)
    model = IndependentMotionExpertSystem(
        _geometry(), torch.tensor([3.0, 0.5, 0.7]),
        torch.tensor([1.0, 1.0, 0.25]), channels=8,
        dropout=0.05, history_events=8,
    )
    model.initialize_from_factorized(augmented, original)
    assert torch.equal(model.q0_head[-1].bias, augmented.q0_head[-1].bias)
    assert torch.equal(
        model.translation_head[-1].bias, original.translation_expert[-1].bias
    )
    assert torch.equal(
        model.rotation_head[-1].bias, original.rotation_expert[-1].bias
    )
    assert torch.equal(
        model.combined_rotation_head[-1].bias,
        original.rotation_expert[-1].bias,
    )
    assert model.rotation_encoder.projection[0].weight.data_ptr() != (
        model.combined_encoder.projection[0].weight.data_ptr()
    )
    model.freeze_foundations()
    model.train()
    for module in (
        model.pose_encoder, model.q0_head,
        model.translation_encoder, model.translation_head,
    ):
        assert not module.training
        assert all(not value.requires_grad for value in module.parameters())
    assert model.rotation_encoder.training
    assert model.combined_encoder.training
    assert model.router_encoder.training


def test_hard_routes_use_the_correct_independent_specialist_and_stay_rigid() -> None:
    model = _system().eval()
    _constant_head(model.translation_head, [0.3, 0.0, 0.0, 0.0])
    _constant_head(model.rotation_head, [0.2, 0.0])
    _constant_head(model.combined_translation_head, [-0.4, 0.1, 0.0, 0.0])
    _constant_head(model.combined_rotation_head, [-0.3, 0.0])
    inputs = _input()
    for route in range(4):
        _set_route(model, route)
        output = model(*inputs)
        if route == 0:
            assert torch.count_nonzero(output["velocity"]) == 0
            assert torch.count_nonzero(output["omega"]) == 0
        elif route == 1:
            assert torch.equal(output["velocity"], output["translation_velocity"])
            assert torch.count_nonzero(output["omega"]) == 0
        elif route == 2:
            assert torch.count_nonzero(output["velocity"]) == 0
            assert torch.equal(output["omega"], output["rotation_omega"])
        else:
            assert torch.equal(output["velocity"], output["combined_velocity"])
            assert torch.equal(output["omega"], output["combined_omega"])
            assert not torch.equal(output["velocity"], output["translation_velocity"])
            assert not torch.equal(output["omega"], output["rotation_omega"])
        position = output["position_mean"]
        predicted_distances = torch.linalg.vector_norm(
            position[:, :, :, None] - position[:, :, None, :], dim=-1,
        )
        geometry = _geometry()
        expected_distances = torch.linalg.vector_norm(
            geometry[:, None] - geometry[None, :], dim=-1,
        )
        assert torch.allclose(
            predicted_distances, expected_distances[None, None], atol=1e-5,
        )


def _gradient_nonzero(module: torch.nn.Module) -> bool:
    return any(
        value.grad is not None and bool(torch.count_nonzero(value.grad))
        for value in module.parameters()
    )


def test_loss_keeps_rotation_and_combined_regression_gradients_isolated() -> None:
    for velocity, omega, expected in (
        (torch.tensor([[0.0, 0.0, 0.0]]), 2.0, "rotation"),
        (torch.tensor([[1.0, 0.2, 0.0]]), -2.0, "combined"),
    ):
        model = _system().train()
        prediction = model.forward_trainable_experts(*_input()[:4])
        target, tau, rule = _future(velocity, omega)
        total, _ = independent_motion_expert_loss(
            prediction, target, tau, rule, _geometry(),
        )
        total.backward()
        rotation_grad = _gradient_nonzero(model.rotation_encoder) or (
            _gradient_nonzero(model.rotation_head)
        )
        combined_grad = _gradient_nonzero(model.combined_encoder) or (
            _gradient_nonzero(model.combined_translation_head)
            or _gradient_nonzero(model.combined_rotation_head)
        )
        assert _gradient_nonzero(model.router_encoder)
        assert rotation_grad == (expected == "rotation")
        assert combined_grad == (expected == "combined")


def test_optimizer_step_does_not_change_frozen_foundation() -> None:
    model = _system().train()
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.startswith((
            "pose_encoder.", "q0_head.",
            "translation_encoder.", "translation_head.",
        ))
    }
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad], lr=1e-3,
    )
    prediction = model.forward_trainable_experts(*_input()[:4])
    target, tau, rule = _future(torch.tensor([[1.0, 0.2, 0.0]]), 2.0)
    total, _ = independent_motion_expert_loss(
        prediction, target, tau, rule, _geometry(),
    )
    total.backward()
    optimizer.step()
    after = model.state_dict()
    assert all(torch.equal(value, after[name]) for name, value in before.items())
