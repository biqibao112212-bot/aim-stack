from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.stage3.observable_future_model import AnonymousCandidateFutureExpert
from training.stage3.observable_future_selector_loss import (
    observable_future_selector_loss,
)
from training.stage3.train_observable_future_selector import (
    SELECTOR_PARAMETER_PREFIXES,
    _model_forward,
    _tensor_state_sha256,
    _validate_foundation_provenance,
    freeze_for_selector_only,
    set_selector_train_mode,
)


def _loss_case() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    step = torch.tensor([[2.0, -1.0, 0.0, 1.0]])
    logits = torch.tensor([[[0.1, 0.2, -0.2, 0.3]]], requires_grad=True)
    delta = torch.tensor([[[
        [0.6, 0.0, 0.0], [0.3, -0.1, 0.0],
        [-0.4, 0.0, 0.0], [0.9, 0.0, 0.0],
    ]]], requires_grad=True)
    prediction = {"switch_logits": logits, "conditional_delta_m": delta}
    target = {
        "candidate_step": step,
        "candidate_mask": torch.ones_like(step, dtype=torch.bool),
        "tau_s": torch.tensor([0.2]),
        "target_switch_count": torch.tensor([[-1.0]]),
        "target_visible_delta_m": torch.tensor([[[0.3, -0.1, 0.0]]]),
        "target_query_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    return prediction, target


def _model_batch() -> dict[str, torch.Tensor]:
    batch = 3
    history = 6
    candidates = 4
    query = 3
    position = torch.randn(batch, history, 3) * 0.05
    time = torch.linspace(-0.25, 0.0, history).expand(batch, -1)
    dt = torch.zeros_like(time)
    dt[:, 1:] = time[:, 1:] - time[:, :-1]
    relation = torch.randn(batch, candidates, 3) * 0.2
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
        "candidate_mask": torch.ones(batch, candidates, dtype=torch.bool),
        "candidate_confidence": torch.ones(batch, candidates),
        "tau_s": torch.tensor([0.0, 0.2, 0.5]),
        "target_switch_count": torch.tensor([[0.0, -1.0, 1.0]]).expand(batch, -1),
        "target_visible_delta_m": torch.randn(batch, query, 3) * 0.2,
        "target_query_mask": torch.ones(batch, query, dtype=torch.bool),
    }


def test_distance_cost_backpropagates_only_to_switch_logits() -> None:
    prediction, target = _loss_case()
    objective, parts = observable_future_selector_loss(
        prediction, **target, switch_weight=0.0, distance_cost_weight=1.0
    )
    objective.backward()
    assert parts["distance_cost"] > 0
    assert prediction["switch_logits"].grad is not None
    assert torch.count_nonzero(prediction["switch_logits"].grad) > 0
    assert prediction["conditional_delta_m"].grad is None


def test_selector_loss_is_query_and_candidate_permutation_invariant() -> None:
    step = torch.tensor([[2.0, -1.0, 0.0, 1.0]])
    logits = torch.randn(1, 3, 4)
    delta = torch.randn(1, 3, 4, 3)
    tau = torch.tensor([0.3, 0.1, 0.2])
    target_step = torch.tensor([[1.0, 0.0, -1.0]])
    target_delta = torch.randn(1, 3, 3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    baseline, _ = observable_future_selector_loss(
        {"switch_logits": logits, "conditional_delta_m": delta},
        step, torch.ones_like(step, dtype=torch.bool), tau,
        target_step, target_delta, mask,
    )
    qp = torch.tensor([2, 0, 1])
    cp = torch.tensor([3, 1, 0, 2])
    changed, _ = observable_future_selector_loss(
        {
            "switch_logits": logits[:, qp][:, :, cp],
            "conditional_delta_m": delta[:, qp][:, :, cp],
        },
        step[:, cp], torch.ones_like(step, dtype=torch.bool)[:, cp], tau[qp],
        target_step[:, qp], target_delta[:, qp], mask[:, qp],
    )
    torch.testing.assert_close(changed, baseline)


def test_freeze_partition_contains_only_switch_heads() -> None:
    model = AnonymousCandidateFutureExpert(
        "combined", channels=16, dropout=0.1, maximum_absolute_step=2,
        trajectory_rank=2,
    )
    trainable, frozen = freeze_for_selector_only(model)
    assert trainable
    assert frozen
    assert all(name.startswith(SELECTOR_PARAMETER_PREFIXES) for name in trainable)
    assert all(not name.startswith(SELECTOR_PARAMETER_PREFIXES) for name in frozen)


def test_selector_optimizer_step_keeps_conditional_trajectory_bit_exact() -> None:
    torch.manual_seed(7)
    model = AnonymousCandidateFutureExpert(
        "combined", channels=16, dropout=0.1, maximum_absolute_step=2,
        trajectory_rank=2,
    )
    batch = _model_batch()
    model.eval()
    before = _model_forward(model, batch)["conditional_delta_m"].detach().clone()
    before_frozen_hash = _tensor_state_sha256(model, selector=False)
    before_selector = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith(SELECTOR_PARAMETER_PREFIXES)
    }
    freeze_for_selector_only(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    set_selector_train_mode(model)
    prediction = _model_forward(model, batch)
    objective, _ = observable_future_selector_loss(
        prediction, batch["candidate_step"], batch["candidate_mask"],
        batch["tau_s"], batch["target_switch_count"],
        batch["target_visible_delta_m"], batch["target_query_mask"],
    )
    objective.backward()
    optimizer.step()
    model.eval()
    after = _model_forward(model, batch)["conditional_delta_m"].detach()
    assert torch.equal(after, before)
    assert _tensor_state_sha256(model, selector=False) == before_frozen_hash
    assert any(
        not torch.equal(parameter.detach(), before_selector[name])
        for name, parameter in model.named_parameters()
        if name in before_selector
    )


def test_frozen_parameters_never_receive_gradients() -> None:
    model = AnonymousCandidateFutureExpert(
        "combined", channels=16, dropout=0.0, maximum_absolute_step=2,
        trajectory_rank=2,
    )
    _, frozen = freeze_for_selector_only(model)
    batch = _model_batch()
    set_selector_train_mode(model)
    prediction = _model_forward(model, batch)
    objective, _ = observable_future_selector_loss(
        prediction, batch["candidate_step"], batch["candidate_mask"],
        batch["tau_s"], batch["target_switch_count"],
        batch["target_visible_delta_m"], batch["target_query_mask"],
    )
    objective.backward()
    named = dict(model.named_parameters())
    assert all(named[name].grad is None for name in frozen)


def test_selector_foundation_provenance_fails_closed() -> None:
    sha = "a" * 64
    valid = {
        "provenance": {
            "test_accessed": False,
            "dataset_manifest_sha256": sha,
        }
    }
    _validate_foundation_provenance(valid, sha)
    with pytest.raises(ValueError, match="test sealed"):
        _validate_foundation_provenance(
            {"provenance": {"test_accessed": True,
                            "dataset_manifest_sha256": sha}}, sha,
        )
    with pytest.raises(ValueError, match="manifest mismatch"):
        _validate_foundation_provenance(valid, "b" * 64)
