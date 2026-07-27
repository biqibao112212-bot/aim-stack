from copy import deepcopy

import pytest
import torch
from torch import nn

from training.stage3.dual_domain_future import (
    ObservationDomain,
    assert_independent_models,
    route_future_expert,
)
from training.stage3.observable_future_model import AnonymousCandidateFutureExpert
from training.stage3.train_dual_domain_pnp_f import (
    _configure_trainable_parameters,
    _set_stage_training_mode,
)


def test_external_domain_router_is_explicit_and_fail_closed() -> None:
    clean = nn.Linear(3, 2)
    pnp = deepcopy(clean)
    assert route_future_expert(
        ObservationDomain.CLEAN, clean_f=clean, pnp_f=pnp,
    ) is clean
    assert route_future_expert(
        ObservationDomain.PNP_V41, clean_f=clean, pnp_f=pnp,
    ) is pnp
    with pytest.raises(TypeError, match="ObservationDomain"):
        route_future_expert("clean", clean_f=clean, pnp_f=pnp)  # type: ignore[arg-type]


def test_domain_models_must_not_share_storage() -> None:
    clean = nn.Linear(3, 2)
    pnp = deepcopy(clean)
    assert_independent_models(clean, pnp)
    with pytest.raises(ValueError, match="share storage"):
        assert_independent_models(clean, clean)


class _PartitionedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.history_encoder = nn.Linear(2, 2)
        self.trajectory_coefficient_head = nn.Linear(2, 2)
        self.switch_candidate_head = nn.Linear(2, 2)
        self.switch_logit = nn.Linear(2, 1)


@pytest.mark.parametrize("stage", ["trajectory", "selector"])
def test_stage_parameter_partition_is_exact(stage: str) -> None:
    model = _PartitionedModel()
    trainable, trainable_names, frozen_names = _configure_trainable_parameters(
        model, stage,
    )
    selector_names = {
        name for name, _ in model.named_parameters()
        if name.startswith(("switch_candidate_head.", "switch_logit."))
    }
    if stage == "selector":
        assert set(trainable_names) == selector_names
    else:
        assert set(frozen_names) == selector_names
    assert {id(parameter) for parameter in trainable} == {
        id(parameter) for name, parameter in model.named_parameters()
        if name in trainable_names
    }
    assert all(
        parameter.requires_grad == (name in trainable_names)
        for name, parameter in model.named_parameters()
    )


def test_selector_stage_keeps_frozen_trajectory_in_eval_mode() -> None:
    model = _PartitionedModel()
    _set_stage_training_mode(model, "selector")
    assert not model.history_encoder.training
    assert not model.trajectory_coefficient_head.training
    assert model.switch_candidate_head.training
    assert model.switch_logit.training


def _real_f_inputs() -> dict[str, torch.Tensor]:
    batch, events, candidates = 2, 6, 7
    history = torch.randn(batch, events, 3) * 0.05
    history[:, -1] = 0.0
    candidate_relation = torch.randn(batch, candidates, 3) * 0.1
    candidate_relation[:, 3] = 0.0
    return {
        "history_position_rel_m": history,
        "history_time_s": torch.tensor(
            [[-0.5, -0.4, -0.3, -0.2, -0.1, 0.0]]
        ).expand(batch, -1),
        "history_dt_s": torch.tensor(
            [[0.0, 0.1, 0.1, 0.1, 0.1, 0.1]]
        ).expand(batch, -1),
        "history_switch_step": torch.zeros(batch, events),
        "history_mask": torch.ones(batch, events, dtype=torch.bool),
        "current_position_m": torch.randn(batch, 3),
        "candidate_relation_m": candidate_relation,
        "candidate_step": torch.tensor(
            [[-3, -2, -1, 0, 1, 2, 3]]
        ).expand(batch, -1),
        "candidate_mask": torch.ones(batch, candidates, dtype=torch.bool),
        "candidate_confidence": torch.ones(batch, candidates),
        "tau_s": torch.tensor([0.1, 0.2]),
    }


def _small_real_f() -> AnonymousCandidateFutureExpert:
    return AnonymousCandidateFutureExpert(
        "combined", channels=16, dropout=0.0, trajectory_rank=2,
    )


def test_real_f_optimizer_steps_are_strictly_stage_isolated() -> None:
    torch.manual_seed(7)
    clean = _small_real_f()
    trajectory = deepcopy(clean)
    assert_independent_models(clean, trajectory)
    _, _, frozen_trajectory_names = _configure_trainable_parameters(
        trajectory, "trajectory",
    )
    before_clean = deepcopy(clean.state_dict())
    before_trajectory = deepcopy(trajectory.state_dict())
    optimizer = torch.optim.AdamW(
        [parameter for parameter in trajectory.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    optimizer.zero_grad(set_to_none=True)
    trajectory(**_real_f_inputs())["conditional_position_m"].square().mean().backward()
    optimizer.step()
    assert all(
        torch.equal(before_clean[name], clean.state_dict()[name])
        for name in before_clean
    )
    assert all(
        torch.equal(before_trajectory[name], trajectory.state_dict()[name])
        for name in frozen_trajectory_names
    )

    selector = deepcopy(trajectory)
    trajectory_parent = deepcopy(trajectory.state_dict())
    selector_parameters, _, frozen_selector_names = _configure_trainable_parameters(
        selector, "selector",
    )
    selector.eval()
    before_output = selector(**_real_f_inputs())["conditional_position_m"]
    optimizer = torch.optim.AdamW(selector_parameters, lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    selector(**_real_f_inputs())["switch_logits"].square().mean().backward()
    optimizer.step()
    selector.eval()
    # Conditional predictions are independent of both selector modules.
    after_inputs = _real_f_inputs()
    before_same_inputs = trajectory(**after_inputs)["conditional_position_m"]
    after_output = selector(**after_inputs)["conditional_position_m"]
    assert torch.equal(before_same_inputs, after_output)
    assert all(
        torch.equal(trajectory_parent[name], selector.state_dict()[name])
        for name in frozen_selector_names
    )
    assert before_output.shape == after_output.shape
