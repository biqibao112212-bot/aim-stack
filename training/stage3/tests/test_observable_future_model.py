from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.stage3.observable_future_model import AnonymousCandidateFutureExpert


def _inputs(batch: int = 2, time: int = 8) -> dict[str, torch.Tensor]:
    history_time = torch.linspace(-0.35, 0.0, time).repeat(batch, 1)
    history_rel = torch.zeros(batch, time, 3)
    history_rel[..., 0] = history_time
    history_dt = torch.zeros(batch, time)
    history_dt[:, 1:] = history_time[:, 1:] - history_time[:, :-1]
    steps = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32).repeat(batch, 1)
    relation = torch.stack((steps * 0.2, steps * -0.1, steps * 0.03), dim=-1)
    relation[:, 2] = 0.0
    return {
        "history_position_rel_m": history_rel,
        "history_time_s": history_time,
        "history_dt_s": history_dt,
        "history_switch_step": torch.zeros(batch, time),
        "history_mask": torch.ones(batch, time, dtype=torch.bool),
        "current_position_m": torch.tensor([[4.0, 1.0, 0.2]]).repeat(batch, 1),
        "candidate_relation_m": relation,
        "candidate_step": steps,
        "candidate_mask": torch.ones(batch, 5, dtype=torch.bool),
        "candidate_confidence": torch.ones(batch, 5),
        "tau_s": torch.tensor([0.2, 0.0, 0.1]),
    }


def _model() -> AnonymousCandidateFutureExpert:
    model = AnonymousCandidateFutureExpert(
        "rotation", channels=16, dropout=0.0, trained_horizon_s=0.5
    )
    return model.eval()


def test_tau_zero_is_exact_for_any_query_and_candidate_order() -> None:
    values = _inputs()
    result = _model()(**values)
    zero = 1
    current_row = (values["candidate_step"] == 0).to(torch.long).argmax(dim=1)
    expected = torch.zeros_like(result["switch_probability"][:, zero])
    expected.scatter_(1, current_row[:, None], 1.0)
    assert torch.equal(result["switch_probability"][:, zero], expected)
    assert torch.equal(result["delta_m"][:, zero], torch.zeros_like(result["delta_m"][:, zero]))
    assert torch.equal(result["position_m"][:, zero], values["current_position_m"])


def test_candidate_permutation_is_equivariant_per_sample() -> None:
    model = _model()
    values = _inputs()
    baseline = model(**values)
    permutation = torch.tensor([[4, 2, 0, 3, 1], [1, 4, 3, 0, 2]])
    changed = dict(values)
    for key in (
        "candidate_relation_m", "candidate_step", "candidate_mask",
        "candidate_confidence",
    ):
        value = values[key]
        index = permutation.unsqueeze(-1).expand(-1, -1, value.shape[-1]) if value.ndim == 3 else permutation
        changed[key] = value.gather(1, index)
    permuted = model(**changed)
    index3 = permutation[:, None, :].expand(-1, baseline["switch_logits"].shape[1], -1)
    assert torch.allclose(permuted["switch_logits"], baseline["switch_logits"].gather(2, index3))
    index4 = index3.unsqueeze(-1).expand(-1, -1, -1, 3)
    assert torch.allclose(
        permuted["conditional_delta_m"], baseline["conditional_delta_m"].gather(2, index4)
    )
    assert torch.equal(permuted["selected_switch_step"], baseline["selected_switch_step"])
    assert torch.allclose(permuted["position_m"], baseline["position_m"])


def test_masked_dummy_candidate_cannot_change_valid_outputs() -> None:
    model = _model()
    values = _inputs()
    baseline = model(**values)
    changed = dict(values)
    changed["candidate_relation_m"] = torch.cat((
        values["candidate_relation_m"], torch.full((2, 1, 3), 1e6)
    ), dim=1)
    changed["candidate_step"] = torch.cat((values["candidate_step"], torch.full((2, 1), 99.0)), dim=1)
    changed["candidate_confidence"] = torch.cat((values["candidate_confidence"], torch.full((2, 1), 1e6)), dim=1)
    changed["candidate_mask"] = torch.cat((values["candidate_mask"], torch.zeros(2, 1, dtype=torch.bool)), dim=1)
    result = model(**changed)
    assert torch.allclose(result["switch_logits"][..., :5], baseline["switch_logits"])
    assert torch.allclose(result["conditional_delta_m"][..., :5, :], baseline["conditional_delta_m"])
    assert torch.allclose(result["position_m"], baseline["position_m"])


def test_query_permutation_only_reorders_outputs() -> None:
    model = _model()
    values = _inputs()
    baseline = model(**values)
    permutation = torch.tensor([2, 0, 1])
    changed = dict(values)
    changed["tau_s"] = values["tau_s"][permutation]
    result = model(**changed)
    for key in (
        "switch_logits", "switch_probability", "conditional_delta_m",
        "conditional_position_m", "selected_candidate_row",
        "selected_switch_step", "delta_m", "position_m", "time_basis",
    ):
        assert torch.allclose(result[key], baseline[key][:, permutation])
    assert torch.equal(
        result["trajectory_coefficient_m"],
        baseline["trajectory_coefficient_m"],
    )


def test_trajectory_coefficients_are_shared_by_every_query() -> None:
    model = _model()
    values = _inputs()
    many = model(**values)
    single_values = dict(values)
    single_values["tau_s"] = values["tau_s"][:1]
    single = model(**single_values)
    assert torch.equal(
        many["trajectory_coefficient_m"],
        single["trajectory_coefficient_m"],
    )
    torch.testing.assert_close(
        many["conditional_delta_m"][:, :1],
        single["conditional_delta_m"],
    )


def test_time_basis_and_coefficients_both_receive_position_gradient() -> None:
    model = _model()
    values = _inputs()
    result = model(**values)
    result["conditional_delta_m"][:, 0].square().sum().backward()
    coefficient_output = model.trajectory_coefficient_head[-1]
    basis_output = model.time_basis_output
    assert coefficient_output.weight.grad is not None
    assert basis_output.weight.grad is not None
    assert torch.count_nonzero(coefficient_output.weight.grad) > 0
    assert torch.count_nonzero(basis_output.weight.grad) > 0


def test_time_basis_conditioning_is_shared_by_all_queries() -> None:
    model = _model()
    values = _inputs()
    many = model(**values)
    single_values = dict(values)
    single_values["tau_s"] = values["tau_s"][2:3]
    single = model(**single_values)
    torch.testing.assert_close(many["time_basis"][:, 2:3], single["time_basis"])


def test_s_boundary_is_detached_and_experts_share_no_parameters() -> None:
    model = _model()
    values = _inputs()
    for key in ("current_position_m", "candidate_relation_m", "candidate_confidence"):
        values[key] = values[key].clone().requires_grad_(True)
    result = model(**values)
    result["switch_logits"][:, 0].sum().backward()
    assert values["current_position_m"].grad is None
    assert values["candidate_relation_m"].grad is None
    assert values["candidate_confidence"].grad is None
    assert any(parameter.grad is not None for parameter in model.parameters())
    other = AnonymousCandidateFutureExpert("combined", channels=16, dropout=0.0)
    assert not ({id(value) for value in model.parameters()} & {id(value) for value in other.parameters()})
