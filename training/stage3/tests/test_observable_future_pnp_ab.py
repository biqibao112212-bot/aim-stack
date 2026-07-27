from __future__ import annotations

import torch

from training.stage3.observable_future_model import AnonymousCandidateFutureExpert
from training.stage3.observable_future_pnp_ab import (
    CausalSelectedPnPAdapter,
    sf_compose,
)


def _f_inputs() -> dict[str, torch.Tensor]:
    batch, events = 2, 32
    steps = torch.tensor((-1, 0, 1, 2), dtype=torch.long).repeat(batch, 1)
    relation = torch.randn(batch, 4, 3) * 0.1
    relation[:, 1] = 0.0
    time = torch.linspace(-0.31, 0.0, events).repeat(batch, 1)
    return {
        "history": torch.randn(batch, events, 3) * 0.1,
        "time": time,
        "dt": torch.cat((torch.zeros(batch, 1), torch.full((batch, events - 1), 0.01)), dim=1),
        "switch": torch.zeros(batch, events),
        "history_mask": torch.ones(batch, events, dtype=torch.bool),
        "current": torch.randn(batch, 3),
        "relation": relation,
        "step": steps,
        "candidate_mask": torch.ones(batch, 4, dtype=torch.bool),
        "confidence": torch.ones(batch, 4),
        "tau": torch.tensor((0.0, 0.1)),
    }


def test_f_differentiable_boundary_is_opt_in() -> None:
    model = AnonymousCandidateFutureExpert(
        "combined", channels=16, dropout=0.0,
        maximum_absolute_step=2, trajectory_rank=2,
    )
    frozen = _f_inputs()
    frozen["current"].requires_grad_()
    frozen["relation"].requires_grad_()
    frozen["confidence"].requires_grad_()
    prediction = model(
        frozen["history"], frozen["time"], frozen["dt"], frozen["switch"],
        frozen["history_mask"], frozen["current"], frozen["relation"],
        frozen["step"], frozen["candidate_mask"], frozen["confidence"],
        frozen["tau"],
    )
    prediction["conditional_position_m"][:, 1].sum().backward()
    assert frozen["current"].grad is None
    assert frozen["relation"].grad is None
    assert frozen["confidence"].grad is None

    differentiable = _f_inputs()
    differentiable["current"].requires_grad_()
    differentiable["relation"].requires_grad_()
    differentiable["confidence"].requires_grad_()
    prediction = model(
        differentiable["history"], differentiable["time"],
        differentiable["dt"], differentiable["switch"],
        differentiable["history_mask"], differentiable["current"],
        differentiable["relation"], differentiable["step"],
        differentiable["candidate_mask"], differentiable["confidence"],
        differentiable["tau"], detach_observation_inputs=False,
    )
    (
        prediction["conditional_position_m"][:, 1].sum()
        + prediction["switch_logits"][:, 1].sum()
    ).backward()
    assert differentiable["current"].grad is not None
    assert differentiable["relation"].grad is not None
    assert differentiable["confidence"].grad is not None


def test_adapter_clean_bypass_is_bit_exact_and_current_roles_stay_zero() -> None:
    inputs = _f_inputs()
    adapter = CausalSelectedPnPAdapter(channels=16, dropout=0.0)
    bypass = adapter(
        inputs["history"], inputs["time"], inputs["dt"], inputs["switch"],
        inputs["history_mask"], inputs["current"], inputs["relation"],
        inputs["step"], inputs["confidence"], bypass=True,
    )
    assert torch.equal(bypass["history_position_rel_m"], inputs["history"])
    assert torch.equal(bypass["current_position_m"], inputs["current"])
    assert torch.equal(bypass["candidate_relation_m"], inputs["relation"])

    corrected = adapter(
        inputs["history"], inputs["time"], inputs["dt"], inputs["switch"],
        inputs["history_mask"], inputs["current"], inputs["relation"],
        inputs["step"], inputs["confidence"],
    )
    current_role = torch.remainder(inputs["step"], 4) == 0
    assert torch.equal(
        corrected["candidate_relation_m"][current_role],
        torch.zeros_like(corrected["candidate_relation_m"][current_role]),
    )
    assert torch.equal(
        corrected["history_position_rel_m"][:, -1],
        torch.zeros_like(corrected["history_position_rel_m"][:, -1]),
    )


def test_sf_compose_uses_relative_steps_and_preserves_gradient() -> None:
    q0 = torch.randn(2, 4, 3, requires_grad=True)
    primary = torch.tensor((1, 3), dtype=torch.long)
    step = torch.tensor((-1, 0, 1, 2), dtype=torch.long).repeat(2, 1)
    batch = {
        "candidate_step": step,
        "pnp_current_position_m": torch.randn(2, 3),
        "pnp_history_position_rel_m": torch.randn(2, 32, 3),
    }
    output = sf_compose({
        "q0_m": q0,
        "primary_index": primary,
        "confidence": torch.ones(2, 4),
        "q0_valid": torch.tensor(((1, 0, 1, 1), (0, 1, 1, 1)), dtype=torch.bool),
    }, batch)
    assert torch.equal(
        output["candidate_relation_m"][:, 1],
        torch.zeros_like(output["candidate_relation_m"][:, 1]),
    )
    output["candidate_relation_m"].sum().backward()
    assert q0.grad is not None
    assert torch.count_nonzero(q0.grad) > 0

