from __future__ import annotations

import torch

from training.stage3.causal_physical_state_model import (
    FactorizedExpertPhysicalPredictor,
    IndependentMotionExpertSystem,
)
from training.stage3.router_only_loss import router_only_loss
from training.stage3.train_router_only import (
    _freeze_except_router,
    _lr_multiplier,
)


def _geometry() -> torch.Tensor:
    return torch.tensor([
        [0.22, 0.00, -0.07], [0.00, -0.21, -0.06],
        [-0.20, 0.00, -0.07], [0.00, 0.21, -0.06],
    ])


def _system() -> IndependentMotionExpertSystem:
    arguments = (
        _geometry(), torch.tensor([3.0, 0.5, 0.7]),
        torch.tensor([1.0, 1.0, 0.25]),
    )
    original = FactorizedExpertPhysicalPredictor(
        *arguments, channels=8, dropout=0.05, history_events=8,
    )
    augmented = FactorizedExpertPhysicalPredictor(
        *arguments, channels=8, dropout=0.05, history_events=8,
    )
    model = IndependentMotionExpertSystem(
        *arguments, channels=8, dropout=0.05, history_events=8,
    )
    model.initialize_from_factorized(augmented, original)
    model.freeze_foundations()
    return model


def _future(
    velocity: torch.Tensor, omega: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = velocity.shape[0]
    tau = torch.tensor([0.0, 0.1, 0.2, 0.5]).repeat(batch, 1)
    center0 = torch.tensor([3.0, 0.5, 0.7]).repeat(batch, 1)
    center = center0[:, None] + tau[..., None] * velocity[:, None]
    angle = tau * omega[:, None]
    cosine, sine = torch.cos(angle), torch.sin(angle)
    geometry = _geometry()
    gx = geometry[:, 0].view(1, 1, 4)
    gy = geometry[:, 1].view(1, 1, 4)
    relative = torch.stack((
        cosine[..., None] * gx - sine[..., None] * gy,
        sine[..., None] * gx + cosine[..., None] * gy,
        geometry[:, 2].view(1, 1, 4).expand(batch, 4, 4),
    ), dim=-1)
    return (
        center[:, :, None] + relative,
        tau,
        torch.ones_like(tau, dtype=torch.bool),
    )


def test_router_only_freezes_everything_except_router() -> None:
    model = _system()
    _freeze_except_router(model)
    trainable = {
        name for name, value in model.named_parameters() if value.requires_grad
    }
    assert trainable
    assert all(name.startswith(("router_encoder.", "router_head.")) for name in trainable)
    assert any(name.startswith("router_encoder.") for name in trainable)
    assert any(name.startswith("router_head.") for name in trainable)


def test_factor_aware_loss_trains_all_four_logits() -> None:
    velocity = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
    ])
    omega = torch.tensor([0.0, 0.0, 2.0, 2.0])
    target, tau, rule = _future(velocity, omega)
    logits = torch.zeros(4, 4, requires_grad=True)
    total, parts = router_only_loss(
        logits, target, tau, rule, _geometry(), label_smoothing=0.02,
    )
    total.backward()
    assert torch.isfinite(total)
    assert logits.grad is not None
    assert bool(torch.count_nonzero(logits.grad))
    assert set(parts) == {
        "four_class", "move_factor", "rotate_factor", "eligible_fraction",
        "valid_fraction", "stationary_fraction", "translation_fraction",
        "rotation_fraction", "combined_fraction",
    }
    for name in (
        "stationary_fraction", "translation_fraction",
        "rotation_fraction", "combined_fraction",
    ):
        assert torch.isclose(parts[name], torch.tensor(0.25))


def test_learning_rate_schedule_warms_then_decays() -> None:
    values = [
        _lr_multiplier(epoch, epochs=120, warmup_epochs=5)
        for epoch in (0, 1, 4, 5, 60, 120)
    ]
    assert values[0] < values[1] < values[2]
    assert values[2] >= values[3]
    assert values[3] > values[4] > values[5]
    assert abs(values[-1] - 0.02) < 1e-9
