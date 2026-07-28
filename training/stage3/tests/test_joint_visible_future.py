from copy import deepcopy

import torch

from training.stage3.joint_visible_future import (
    JointVisibleFutureModel,
    LearnedVisibleStateSelector,
    joint_visible_future_loss,
)
from training.stage3.observable_future_model import AnonymousCandidateFutureExpert


def _batch() -> dict[str, torch.Tensor]:
    batch, events, candidates = 2, 6, 5
    history = torch.randn(batch, events, 3) * 0.03
    history[:, -1] = 0.0
    relation = torch.randn(batch, candidates, 3) * 0.1
    relation[:, 2] = 0.0
    return {
        "history_position_rel_m": history,
        "history_time_s": torch.tensor(
            [[-0.5, -0.4, -0.3, -0.2, -0.1, 0.0]], dtype=torch.float32,
        ).expand(batch, -1),
        "history_dt_s": torch.tensor(
            [[0.0, 0.1, 0.1, 0.1, 0.1, 0.1]], dtype=torch.float32,
        ).expand(batch, -1),
        "history_switch_step": torch.zeros(batch, events),
        "history_mask": torch.ones(batch, events, dtype=torch.bool),
        "current_position_m": torch.randn(batch, 3),
        "candidate_relation_m": relation,
        "candidate_step": torch.tensor(
            [[-2, -1, 0, 1, 2]], dtype=torch.long,
        ).expand(batch, -1),
        "candidate_mask": torch.ones(batch, candidates, dtype=torch.bool),
        "candidate_confidence": torch.rand(batch, candidates),
        "tau_s": torch.tensor([0.0, 0.2, 0.4]),
        "target_switch_count": torch.tensor(
            [[0, 0, 1], [0, -1, -2]], dtype=torch.long,
        ),
        "target_visible_delta_m": torch.randn(batch, 3, 3) * 0.1,
        "target_query_mask": torch.ones(batch, 3, dtype=torch.bool),
    }


def _model() -> JointVisibleFutureModel:
    trajectory = AnonymousCandidateFutureExpert(
        "combined", channels=16, dropout=0.0,
        maximum_absolute_step=2, trajectory_rank=2,
    )
    selector = LearnedVisibleStateSelector(
        channels=16, dropout=0.0, maximum_absolute_step=2,
    )
    return JointVisibleFutureModel(trajectory, selector)


def test_joint_selector_is_anonymous_and_candidate_permutation_equivariant() -> None:
    torch.manual_seed(17)
    model = _model().eval()
    batch = _batch()
    first = model(batch)
    permutation = torch.tensor([3, 1, 4, 0, 2])
    permuted = dict(batch)
    for name in (
        "candidate_relation_m", "candidate_step", "candidate_mask",
        "candidate_confidence",
    ):
        permuted[name] = batch[name].index_select(1, permutation)
    second = model(permuted)
    inverse = torch.argsort(permutation)
    assert torch.allclose(
        first["conditional_position_m"],
        second["conditional_position_m"].index_select(2, inverse),
        atol=1e-6, rtol=1e-6,
    )
    assert torch.allclose(
        first["switch_logits"],
        second["switch_logits"].index_select(2, inverse),
        atol=1e-6, rtol=1e-6,
    )
    assert model.selector.config["physical_id_input"] is False
    assert model.selector.config["candidate_row_embedding"] is False


def test_selector_compatibility_does_not_consume_candidate_confidence() -> None:
    torch.manual_seed(23)
    model = _model().eval()
    batch = _batch()
    first = model(batch)["switch_logits"]
    changed = dict(batch)
    changed["candidate_confidence"] = 1.0 - batch["candidate_confidence"]
    # The frozen trajectory consumes confidence, so hold its candidate paths
    # fixed when checking the selector's explicit interface contract.
    with torch.no_grad():
        trajectory = model.trajectory(
            batch["history_position_rel_m"], batch["history_time_s"],
            batch["history_dt_s"], batch["history_switch_step"],
            batch["history_mask"], batch["current_position_m"],
            batch["candidate_relation_m"], batch["candidate_step"],
            batch["candidate_mask"], batch["candidate_confidence"],
            batch["tau_s"],
        )
        original = model.selector(
            batch["history_position_rel_m"], batch["history_time_s"],
            batch["history_dt_s"], batch["history_switch_step"],
            batch["history_mask"], batch["current_position_m"],
            batch["candidate_relation_m"], batch["candidate_step"],
            batch["candidate_mask"], batch["tau_s"],
            trajectory["conditional_position_m"],
        )["switch_logits"]
        repeated = model.selector(
            batch["history_position_rel_m"], batch["history_time_s"],
            batch["history_dt_s"], batch["history_switch_step"],
            batch["history_mask"], batch["current_position_m"],
            batch["candidate_relation_m"], batch["candidate_step"],
            batch["candidate_mask"], batch["tau_s"],
            trajectory["conditional_position_m"],
        )["switch_logits"]
    assert torch.equal(original, repeated)
    assert first.shape == original.shape


def test_joint_loss_can_train_selector_then_trajectory() -> None:
    torch.manual_seed(31)
    model = _model()
    batch = _batch()
    loss_kwargs = {
        "switch_weight": 1.0,
        "conditional_position_weight": 10.0,
        "mixture_weight": 0.1,
        "expected_cost_weight": 1.0,
        "mixture_sigma_m": 0.1,
        "huber_beta_m": 0.01,
        "macro_balance_weight": 0.5,
        "focal_gamma": 0.0,
    }

    for parameter in model.trajectory.parameters():
        parameter.requires_grad_(False)
    prediction = model(batch)
    loss, components = joint_visible_future_loss(
        prediction, batch, **loss_kwargs,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert set(components) == {
        "objective", "switch", "conditional_position", "mixture_nll",
        "expected_position_cost",
    }
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in model.selector.parameters()
    )
    assert all(parameter.grad is None for parameter in model.trajectory.parameters())

    joint = deepcopy(_model())
    prediction = joint(batch)
    joint_loss, _ = joint_visible_future_loss(
        prediction, batch, **loss_kwargs,
    )
    joint_loss.backward()
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for name, parameter in joint.trajectory.named_parameters()
        if not name.startswith(("switch_candidate_head.", "switch_logit."))
    )
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in joint.selector.parameters()
    )


def test_tau_zero_routes_relative_step_zero() -> None:
    model = _model().eval()
    batch = _batch()
    prediction = model(batch)
    assert torch.equal(
        prediction["selected_switch_step"][:, 0],
        torch.zeros(batch["current_position_m"].shape[0], dtype=torch.long),
    )
