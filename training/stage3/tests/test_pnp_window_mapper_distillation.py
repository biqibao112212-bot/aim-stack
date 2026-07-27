from __future__ import annotations

import pytest
import torch

from training.stage3.observable_future_pnp_ab import (
    canonicalize_direction_keep_c4,
)
from training.stage3.train_pnp_window_mapper_distillation import (
    MAPPER_INPUT_FIELDS,
    gather_true_branch,
    task_distillation_loss,
)


def _branch_batch() -> dict[str, torch.Tensor]:
    return {
        "candidate_step": torch.tensor(((0, 1, 1),)),
        "candidate_mask": torch.tensor(((True, True, False),)),
        "target_switch_count": torch.tensor(((0, 1),)),
        "target_query_mask": torch.tensor(((True, True),)),
        "tau_s": torch.tensor(((0.0, 0.1),)),
        "current_position_m": torch.zeros(1, 3),
        "target_visible_delta_m": torch.zeros(1, 2, 3),
    }


def test_true_branch_uses_candidate_mask_and_excludes_tau_zero() -> None:
    batch = _branch_batch()
    prediction = {
        "conditional_position_m": torch.arange(18, dtype=torch.float32).reshape(
            1, 2, 3, 3
        )
    }
    selected, _, eligible, row = gather_true_branch(prediction, batch)
    assert torch.equal(eligible, torch.tensor(((False, True),)))
    assert torch.equal(row, torch.tensor(((0, 1),)))
    assert torch.equal(selected[0, 1], prediction["conditional_position_m"][0, 1, 1])

    duplicate = dict(batch)
    duplicate["candidate_mask"] = torch.ones_like(batch["candidate_mask"])
    with pytest.raises(ValueError, match="exactly one"):
        gather_true_branch(prediction, duplicate)


def test_task_distillation_loss_is_loss_side_only_and_differentiable() -> None:
    scale = torch.tensor(0.2, requires_grad=True)
    batch = {
        **_branch_batch(),
        "pnp_history_mask": torch.tensor(((True, True),)),
        "history_mask": torch.tensor(((True, True),)),
        "pnp_candidate_mask": torch.tensor(((True, True, False),)),
        "pnp_s_obs_mask": torch.tensor(
            [[(True, False, False, False), (True, False, False, False)]]
        ),
        "clean_s_obs_m": torch.zeros(1, 2, 4, 3),
    }
    mapped = {"corrected_obs_m": scale * torch.ones(1, 2, 4, 3)}
    student_composed = {
        "history_position_rel_m": scale * torch.ones(1, 2, 3),
        "current_position_m": scale * torch.ones(1, 3),
        "candidate_relation_m": scale * torch.ones(1, 3, 3),
    }
    teacher_composed = {
        name: torch.zeros_like(value)
        for name, value in student_composed.items()
    }
    student_prediction = {
        "conditional_position_m": scale * torch.ones(1, 2, 3, 3)
    }
    teacher_prediction = {
        "conditional_position_m": torch.zeros(1, 2, 3, 3)
    }
    objective, components = task_distillation_loss(
        mapped, student_composed, student_prediction,
        teacher_composed, teacher_prediction, batch,
        huber_beta_m=0.05,
        future_weight=1.0,
        history_weight=0.5,
        current_weight=0.25,
        candidate_weight=0.1,
        observation_weight=0.1,
        future_target="clean-teacher",
    )
    assert set(components) == {
        "future_target_huber", "history_teacher_huber",
        "current_teacher_huber", "candidate_teacher_huber",
        "observation_clean_huber",
    }
    objective.backward()
    assert scale.grad is not None and torch.isfinite(scale.grad)
    assert float(scale.grad) > 0
    assert MAPPER_INPUT_FIELDS == (
        "pnp_s_obs_m", "pnp_s_obs_mask",
        "pnp_s_event_time_s", "pnp_s_event_mask",
    )


def test_physical_truth_future_target_changes_only_the_loss_target() -> None:
    scale = torch.tensor(0.2, requires_grad=True)
    batch = {
        **_branch_batch(),
        "pnp_history_mask": torch.tensor(((True, True),)),
        "history_mask": torch.tensor(((True, True),)),
        "pnp_candidate_mask": torch.tensor(((True, True, False),)),
        "pnp_s_obs_mask": torch.tensor(
            [[(True, False, False, False), (True, False, False, False)]]
        ),
        "clean_s_obs_m": torch.zeros(1, 2, 4, 3),
    }
    batch["target_visible_delta_m"][:, 1] = 0.5
    mapped = {"corrected_obs_m": scale * torch.ones(1, 2, 4, 3)}
    student_composed = {
        "history_position_rel_m": scale * torch.ones(1, 2, 3),
        "current_position_m": scale * torch.ones(1, 3),
        "candidate_relation_m": scale * torch.ones(1, 3, 3),
    }
    teacher_composed = {
        name: torch.zeros_like(value) for name, value in student_composed.items()
    }
    student_prediction = {
        "conditional_position_m": scale * torch.ones(1, 2, 3, 3)
    }
    teacher_prediction = {
        "conditional_position_m": torch.zeros(1, 2, 3, 3)
    }
    common = dict(
        huber_beta_m=0.05, future_weight=1.0, history_weight=0.0,
        current_weight=0.0, candidate_weight=0.0, observation_weight=0.0,
    )
    teacher_loss, _ = task_distillation_loss(
        mapped, student_composed, student_prediction,
        teacher_composed, teacher_prediction, batch,
        future_target="clean-teacher", **common,
    )
    truth_loss, _ = task_distillation_loss(
        mapped, student_composed, student_prediction,
        teacher_composed, teacher_prediction, batch,
        future_target="physical-truth", **common,
    )
    assert not torch.equal(teacher_loss, truth_loss)


def test_direction_canonicalization_keeps_pair_order_and_c4_origin() -> None:
    obs = torch.zeros(2, 1, 4, 3)
    obs[:, 0, :, 0] = torch.arange(4)
    primary = torch.zeros(2, 1, 4, dtype=torch.bool)
    primary[:, :, 1] = True
    signed = torch.tensor(((2,), (3,)))
    batch = {
        "pnp_s_obs_m": obs.clone(),
        "pnp_s_primary_mask": primary.clone(),
        "candidate_step": signed.clone(),
    }
    canonicalize_direction_keep_c4(
        batch, ("0000000000000000", "0000000000000004")
    )
    assert torch.equal(batch["pnp_s_obs_m"][0], obs[0])
    assert torch.equal(
        batch["pnp_s_obs_m"][1, 0, :, 0], torch.tensor((0.0, 3.0, 2.0, 1.0))
    )
    assert torch.equal(
        batch["pnp_s_primary_mask"][1, 0],
        torch.tensor((False, False, False, True)),
    )
    assert torch.equal(batch["candidate_step"], torch.tensor(((2,), (-3,))))
    with pytest.raises(ValueError, match="batch size"):
        canonicalize_direction_keep_c4(batch, ("0000000000000000",))
