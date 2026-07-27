from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.stage3.observable_future_model import AnonymousCandidateFutureExpert
from training.stage3.observable_future_selector_loss import observable_future_selector_loss
from training.stage3.train_observable_future_dedicated_selector import (
    DEDICATED_SELECTOR_PARAMETER_PREFIXES,
    FrozenTrajectoryDedicatedSelector,
    _canonical_selected_row,
    _module_state_sha256,
    freeze_dedicated_selector,
    set_dedicated_selector_train_mode,
)


def _models() -> tuple[
    AnonymousCandidateFutureExpert, AnonymousCandidateFutureExpert
]:
    base = AnonymousCandidateFutureExpert(
        "combined", channels=16, dropout=0.1,
        maximum_absolute_step=2, trajectory_rank=2,
    )
    selector = AnonymousCandidateFutureExpert(
        "combined", channels=16, dropout=0.1,
        maximum_absolute_step=2, trajectory_rank=2,
    )
    selector.load_state_dict(base.state_dict(), strict=True)
    return base, selector


def _batch() -> dict[str, torch.Tensor]:
    batch = 3
    history = 6
    position = torch.randn(batch, history, 3) * 0.05
    time = torch.linspace(-0.25, 0.0, history).expand(batch, -1)
    dt = torch.zeros_like(time)
    dt[:, 1:] = time[:, 1:] - time[:, :-1]
    relation = torch.randn(batch, 4, 3) * 0.2
    relation[:, 2] = 0
    return {
        "history_position_rel_m": position,
        "history_time_s": time,
        "history_dt_s": dt,
        "history_switch_step": torch.zeros(batch, history),
        "history_mask": torch.ones(batch, history, dtype=torch.bool),
        "current_position_m": torch.randn(batch, 3),
        "candidate_relation_m": relation,
        "candidate_step": torch.tensor([-2.0, -1.0, 0.0, 1.0]).expand(batch, -1),
        "candidate_mask": torch.ones(batch, 4, dtype=torch.bool),
        "candidate_confidence": torch.ones(batch, 4),
        "tau_s": torch.tensor([0.0, 0.2, 0.5]),
        "target_switch_count": torch.tensor([[0.0, -1.0, 1.0]]).expand(batch, -1),
        "target_visible_delta_m": torch.randn(batch, 3, 3) * 0.2,
        "target_query_mask": torch.ones(batch, 3, dtype=torch.bool),
    }


def _forward(
    system: FrozenTrajectoryDedicatedSelector,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return system(
        batch["history_position_rel_m"], batch["history_time_s"],
        batch["history_dt_s"], batch["history_switch_step"],
        batch["history_mask"], batch["current_position_m"],
        batch["candidate_relation_m"], batch["candidate_step"],
        batch["candidate_mask"], batch["candidate_confidence"],
        batch["tau_s"],
    )


def test_dedicated_selector_freeze_whitelist_is_exact() -> None:
    base, selector = _models()
    trainable, frozen = freeze_dedicated_selector(base, selector)
    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert trainable and frozen
    assert all(name.startswith(DEDICATED_SELECTOR_PARAMETER_PREFIXES) for name in trainable)
    assert all(not name.startswith(DEDICATED_SELECTOR_PARAMETER_PREFIXES) for name in frozen)


def test_selector_clone_has_equal_values_without_shared_storage() -> None:
    base, selector = _models()
    selector_named = dict(selector.named_parameters())
    for name, parameter in base.named_parameters():
        clone = selector_named[name]
        assert torch.equal(parameter, clone)
        assert parameter is not clone
        assert parameter.data_ptr() != clone.data_ptr()


def test_dedicated_selector_step_keeps_parent_trajectory_bit_exact() -> None:
    torch.manual_seed(19)
    base, selector = _models()
    freeze_dedicated_selector(base, selector)
    system = FrozenTrajectoryDedicatedSelector(base, selector)
    batch = _batch()
    system.eval()
    before = _forward(system, batch)
    conditional = before["conditional_delta_m"].detach().clone()
    coefficient = before["trajectory_coefficient_m"].detach().clone()
    time_basis = before["time_basis"].detach().clone()
    base_hash = _module_state_sha256(base)
    selector_frozen_hash = _module_state_sha256(
        selector, prefixes=DEDICATED_SELECTOR_PARAMETER_PREFIXES, invert=True,
    )
    selector_trainable_hash = _module_state_sha256(
        selector, prefixes=DEDICATED_SELECTOR_PARAMETER_PREFIXES,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in selector.parameters()
         if parameter.requires_grad], lr=1e-2,
    )
    set_dedicated_selector_train_mode(system)
    prediction = _forward(system, batch)
    objective, _ = observable_future_selector_loss(
        prediction, batch["candidate_step"], batch["candidate_mask"],
        batch["tau_s"], batch["target_switch_count"],
        batch["target_visible_delta_m"], batch["target_query_mask"],
    )
    objective.backward()
    optimizer.step()
    system.eval()
    after = _forward(system, batch)
    assert torch.equal(after["conditional_delta_m"], conditional)
    assert torch.equal(after["trajectory_coefficient_m"], coefficient)
    assert torch.equal(after["time_basis"], time_basis)
    assert _module_state_sha256(base) == base_hash
    assert _module_state_sha256(
        selector, prefixes=DEDICATED_SELECTOR_PARAMETER_PREFIXES, invert=True,
    ) == selector_frozen_hash
    assert _module_state_sha256(
        selector, prefixes=DEDICATED_SELECTOR_PARAMETER_PREFIXES,
    ) != selector_trainable_hash


def test_dedicated_selector_tau_zero_is_structural_identity() -> None:
    base, selector = _models()
    freeze_dedicated_selector(base, selector)
    system = FrozenTrajectoryDedicatedSelector(base, selector).eval()
    prediction = _forward(system, _batch())
    assert torch.equal(
        prediction["selected_switch_step"][:, 0],
        torch.zeros_like(prediction["selected_switch_step"][:, 0]),
    )
    assert torch.equal(
        prediction["delta_m"][:, 0],
        torch.zeros_like(prediction["delta_m"][:, 0]),
    )


def test_canonical_tie_break_is_candidate_order_independent() -> None:
    logits = torch.zeros(1, 2, 4)
    step = torch.tensor([[2, -1, 0, 1]])
    mask = torch.ones_like(step, dtype=torch.bool)
    first = _canonical_selected_row(logits, step, mask)
    assert torch.equal(step.gather(1, first), torch.full((1, 2), -1))
    permutation = torch.tensor([2, 0, 3, 1])
    changed = _canonical_selected_row(
        logits[:, :, permutation], step[:, permutation], mask[:, permutation]
    )
    assert torch.equal(
        step[:, permutation].gather(1, changed), torch.full((1, 2), -1)
    )


def test_hard_route_uses_selector_rows_and_parent_branches() -> None:
    base, selector = _models()
    freeze_dedicated_selector(base, selector)
    with torch.no_grad():
        selector.switch_logit.weight.zero_()
        selector.switch_logit.bias.zero_()
    system = FrozenTrajectoryDedicatedSelector(base, selector).eval()
    batch = _batch()
    prediction = _forward(system, batch)
    # Nonzero queries are exact ties and therefore select canonical step -2.
    assert torch.equal(
        prediction["selected_switch_step"][:, 1:],
        torch.full_like(prediction["selected_switch_step"][:, 1:], -2.0),
    )
    rows = prediction["selected_candidate_row"]
    gathered = prediction["conditional_delta_m"].gather(
        2, rows[:, :, None, None].expand(-1, -1, 1, 3)
    ).squeeze(2)
    gathered[:, 0] = 0
    assert torch.equal(prediction["delta_m"], gathered)


def test_dedicated_selector_gradients_never_enter_parent_or_unused_heads() -> None:
    base, selector = _models()
    trainable, frozen = freeze_dedicated_selector(base, selector)
    system = FrozenTrajectoryDedicatedSelector(base, selector)
    batch = _batch()
    set_dedicated_selector_train_mode(system)
    prediction = _forward(system, batch)
    objective, _ = observable_future_selector_loss(
        prediction, batch["candidate_step"], batch["candidate_mask"],
        batch["tau_s"], batch["target_switch_count"],
        batch["target_visible_delta_m"], batch["target_query_mask"],
    )
    objective.backward()
    assert all(parameter.grad is None for parameter in base.parameters())
    named = dict(selector.named_parameters())
    assert all(named[name].grad is None for name in frozen)
    assert any(named[name].grad is not None for name in trainable)
