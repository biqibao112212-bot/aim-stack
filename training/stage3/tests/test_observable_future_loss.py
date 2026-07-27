from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.stage3.observable_future_loss import (
    _balanced_group_mean,
    observable_future_loss,
)
from training.stage3.train_observable_future_expert import (
    _BalancedCachedSamples,
    _scheduled_learning_rate,
)


def _case() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    step = torch.tensor([[2.0, -1.0, 0.0, 1.0]])
    logits = torch.tensor([[[0.1, 2.0, -0.2, 0.3]]], requires_grad=True)
    delta = torch.tensor([[[
        [100.0, 100.0, 100.0], [0.4, -0.2, 0.1],
        [-100.0, -100.0, -100.0], [50.0, 50.0, 50.0],
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


def test_conditional_position_loss_reads_only_true_branch() -> None:
    prediction, target = _case()
    objective, parts = observable_future_loss(
        prediction, **target, switch_weight=0.0, position_weight=1.0
    )
    objective.backward()
    assert parts["position"] > 0
    gradient = prediction["conditional_delta_m"].grad
    assert gradient is not None
    assert torch.count_nonzero(gradient[0, 0, 1]) > 0
    assert torch.equal(gradient[0, 0, 0], torch.zeros(3))
    assert torch.equal(gradient[0, 0, 2], torch.zeros(3))
    assert torch.equal(gradient[0, 0, 3], torch.zeros(3))


def test_wrong_branch_coordinates_cannot_change_position_loss() -> None:
    prediction, target = _case()
    _, baseline = observable_future_loss(prediction, **target)
    changed_delta = prediction["conditional_delta_m"].detach().clone()
    changed_delta[:, :, [0, 2, 3]] = 1e6
    changed = {
        "switch_logits": prediction["switch_logits"],
        "conditional_delta_m": changed_delta,
    }
    _, result = observable_future_loss(changed, **target)
    assert torch.equal(result["position"], baseline["position"])


def test_loss_is_invariant_to_query_and_candidate_permutation() -> None:
    step = torch.tensor([[2.0, -1.0, 0.0, 1.0]])
    logits = torch.randn(1, 3, 4)
    delta = torch.randn(1, 3, 4, 3)
    tau = torch.tensor([0.3, 0.0, 0.1])
    target_step = torch.tensor([[1.0, 0.0, -1.0]])
    target_delta = torch.randn(1, 3, 3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    baseline, _ = observable_future_loss(
        {"switch_logits": logits, "conditional_delta_m": delta},
        step, torch.ones_like(step, dtype=torch.bool), tau,
        target_step, target_delta, mask, trend_weight=1.0,
    )
    query_permutation = torch.tensor([2, 0, 1])
    candidate_permutation = torch.tensor([3, 1, 0, 2])
    changed, _ = observable_future_loss(
        {
            "switch_logits": logits[:, query_permutation][:, :, candidate_permutation],
            "conditional_delta_m": delta[:, query_permutation][:, :, candidate_permutation],
        },
        step[:, candidate_permutation],
        torch.ones_like(step, dtype=torch.bool)[:, candidate_permutation],
        tau[query_permutation], target_step[:, query_permutation],
        target_delta[:, query_permutation], mask[:, query_permutation],
        trend_weight=1.0,
    )
    assert torch.allclose(changed, baseline)


def test_invalid_target_branch_fails_closed() -> None:
    prediction, target = _case()
    target["target_switch_count"] = torch.tensor([[5.0]])
    with pytest.raises(ValueError, match="missing"):
        observable_future_loss(prediction, **target)


def test_update_schedule_warms_up_then_decays_to_declared_floor() -> None:
    values = [
        _scheduled_learning_rate(
            update,
            base_learning_rate=3e-4,
            minimum_learning_rate=1e-5,
            warmup_updates=2,
            total_updates=6,
        )
        for update in range(1, 7)
    ]
    assert values[0] == 1.5e-4
    assert values[1] == 3e-4
    assert values[2] < values[1]
    assert values[-1] == 1e-5


def test_macro_micro_blend_keeps_common_and_rare_queries_in_the_loss() -> None:
    value = torch.tensor([[1.0, 1.0, 1.0, 9.0]])
    mask = torch.ones_like(value, dtype=torch.bool)
    step = torch.tensor([[0, 0, 0, 1]])
    micro, _, _ = _balanced_group_mean(value, mask, step, 0.0)
    macro, _, _ = _balanced_group_mean(value, mask, step, 1.0)
    blend, _, _ = _balanced_group_mean(value, mask, step, 0.5)
    assert micro == torch.tensor(3.0)
    assert macro == torch.tensor(5.0)
    assert blend == torch.tensor(4.0)


def test_structural_tau_zero_does_not_dilute_learned_loss() -> None:
    step = torch.tensor([[0.0, 1.0]])
    candidate_mask = torch.ones_like(step, dtype=torch.bool)
    learned_prediction = {
        "switch_logits": torch.tensor([[[2.0, -1.0]]]),
        "conditional_delta_m": torch.tensor([[[[0.3, 0.0, 0.0], [1.0, 0.0, 0.0]]]]),
    }
    learned_target = dict(
        candidate_step=step,
        candidate_mask=candidate_mask,
        tau_s=torch.tensor([[0.2]]),
        target_switch_count=torch.tensor([[0.0]]),
        target_visible_delta_m=torch.tensor([[[0.2, 0.0, 0.0]]]),
        target_query_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    baseline, _ = observable_future_loss(learned_prediction, **learned_target)
    with_zero_prediction = {
        "switch_logits": torch.tensor([[[0.0, -torch.inf], [2.0, -1.0]]]),
        "conditional_delta_m": torch.tensor([[[
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        ], [
            [0.3, 0.0, 0.0], [1.0, 0.0, 0.0],
        ]]]),
    }
    with_zero_target = dict(
        candidate_step=step,
        candidate_mask=candidate_mask,
        tau_s=torch.tensor([[0.0, 0.2]]),
        target_switch_count=torch.tensor([[0.0, 0.0]]),
        target_visible_delta_m=torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]]),
        target_query_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    changed, _ = observable_future_loss(with_zero_prediction, **with_zero_target)
    torch.testing.assert_close(changed, baseline)


def test_focal_switch_loss_downweights_an_already_easy_branch() -> None:
    prediction, target = _case()
    _, plain = observable_future_loss(
        prediction, **target, position_weight=0.0, switch_focal_gamma=0.0
    )
    _, focal = observable_future_loss(
        prediction, **target, position_weight=0.0, switch_focal_gamma=2.0
    )
    assert focal["switch"] < plain["switch"]


def test_rate_tau_floor_keeps_near_zero_query_gradient_finite() -> None:
    prediction, target = _case()
    target["tau_s"] = torch.tensor([1e-7])
    objective, parts = observable_future_loss(
        prediction, **target, switch_weight=0.0, position_weight=0.0,
        rate_weight=1.0, rate_tau_floor_s=0.05,
    )
    objective.backward()
    assert torch.isfinite(objective)
    assert torch.isfinite(parts["rate"])
    gradient = prediction["conditional_delta_m"].grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[0, 0, 1]) > 0


def test_position_tail_loss_reads_only_true_branch() -> None:
    prediction, target = _case()
    objective, parts = observable_future_loss(
        prediction, **target, switch_weight=0.0, position_weight=0.0,
        position_tail_weight=1.0,
    )
    objective.backward()
    assert parts["position_tail"] > 0
    gradient = prediction["conditional_delta_m"].grad
    assert gradient is not None
    assert torch.count_nonzero(gradient[0, 0, 1]) > 0
    assert torch.equal(gradient[0, 0, 0], torch.zeros(3))
    assert torch.equal(gradient[0, 0, 2], torch.zeros(3))
    assert torch.equal(gradient[0, 0, 3], torch.zeros(3))


def test_balanced_tiny_cache_is_deterministic_and_keeps_rare_roles() -> None:
    samples = []
    for index, step in enumerate([0, 0, 0, 0, 1, -2]):
        samples.append({
            "tau_s": torch.tensor([0.0, 0.15 + 0.01 * index]),
            "target_query_mask": torch.tensor([True, True]),
            "target_switch_count": torch.tensor([0, step]),
            "marker": torch.tensor(index),
        })
    first = _BalancedCachedSamples(samples, limit=4, seed=17)
    second = _BalancedCachedSamples(samples, limit=4, seed=17)
    assert first.selection_sha256 == second.selection_sha256
    selected_steps = {
        int(sample["target_switch_count"][1]) for sample in first.samples
    }
    assert {-2, 1}.issubset(selected_steps)
