from __future__ import annotations

import pytest
import torch

from training.stage3.pnp_history_domain_adapter import PnPHistoryDomainAdapter


def _inputs(batch: int = 2) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(31)
    position = torch.randn(batch, 32, 3) * 0.2
    position[:, -1] = 0
    time = torch.linspace(-0.31, 0.0, 32).repeat(batch, 1)
    dt = torch.zeros_like(time)
    dt[:, 1:] = time[:, 1:] - time[:, :-1]
    switch = torch.zeros(batch, 32)
    switch[:, 8] = 1
    switch[:, 24] = -1
    mask = torch.ones(batch, 32, dtype=torch.bool)
    return position, time, dt, switch, mask


def _model() -> PnPHistoryDomainAdapter:
    return PnPHistoryDomainAdapter(channels=16, dropout=0.0)


def test_history_adapter_zero_init_preserves_past_and_zero_q0() -> None:
    model = _model()
    inputs = _inputs()
    output = model(*inputs)
    assert torch.equal(
        output["corrected_history_position_rel_m"][:, :-1], inputs[0][:, :-1]
    )
    assert torch.equal(
        output["corrected_history_position_rel_m"][:, -1],
        torch.zeros_like(inputs[0][:, -1]),
    )
    assert model.config["physical_id_input"] is False
    assert model.config["primary_mask_input"] is False
    assert model.config["future_or_target_input"] is False


def test_history_adapter_is_closed_window_and_padding_poison_safe() -> None:
    model = _model().eval()
    with torch.no_grad():
        model.residual_head[-1].weight.normal_(0.0, 0.02)
    position, time, dt, switch, mask = _inputs(batch=1)
    reference = model(position, time, dt, switch, mask)[
        "corrected_history_position_rel_m"
    ]
    changed = position.clone()
    changed[:, -2] += 0.4
    changed_output = model(changed, time, dt, switch, mask)[
        "corrected_history_position_rel_m"
    ]
    assert not torch.allclose(reference[:, -6], changed_output[:, -6])

    padded_mask = mask.clone()
    padded_mask[:, 4] = False
    poisoned_position = position.clone()
    poisoned_time = time.clone()
    poisoned_dt = dt.clone()
    poisoned_switch = switch.clone()
    poisoned_position[:, 4] = torch.nan
    poisoned_time[:, 4] = torch.nan
    poisoned_dt[:, 4] = torch.nan
    poisoned_switch[:, 4] = torch.nan
    poisoned = model(
        poisoned_position, poisoned_time, poisoned_dt,
        poisoned_switch, padded_mask,
    )["corrected_history_position_rel_m"]
    clean_padding = model(
        position, time, dt, switch, padded_mask
    )["corrected_history_position_rel_m"]
    assert torch.equal(poisoned, clean_padding)


def test_history_adapter_fails_closed_on_time_contract() -> None:
    model = _model()
    position, time, dt, switch, mask = _inputs(batch=1)
    positive = time.clone()
    positive[:, -1] = 0.01
    with pytest.raises(ValueError, match="after q0"):
        model(position, positive, dt, switch, mask)
    nonmonotonic = time.clone()
    nonmonotonic[:, 10] = nonmonotonic[:, 11]
    with pytest.raises(ValueError, match="strictly increasing"):
        model(position, nonmonotonic, dt, switch, mask)
    no_q0 = time.clone()
    no_q0[:, -1] = -0.001
    with pytest.raises(ValueError, match="exactly one q0"):
        model(position, no_q0, dt, switch, mask)


def test_history_adapter_has_gradient_only_through_valid_past() -> None:
    model = _model()
    position, time, dt, switch, mask = _inputs(batch=1)
    output = model(position, time, dt, switch, mask)
    loss = output["corrected_history_position_rel_m"][:, :-1].square().mean()
    loss.backward()
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and bool(torch.any(parameter.grad != 0))
        for parameter in model.parameters()
    )
