from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip("torch")

from training.stage3.observable_future_model import AnonymousCandidateFutureExpert
from training.stage3.ordinal_visible_selector import (
    OrdinalVisibleFutureModel,
    OrdinalVisibleProgressSelector,
    ordinal_visible_selector_loss,
)


def _batch(batch_size: int = 3, device: str = "cpu") -> dict[str, torch.Tensor]:
    history_time = torch.linspace(-0.35, 0.0, 8, device=device).repeat(batch_size, 1)
    history_rel = torch.zeros(batch_size, 8, 3, device=device)
    history_rel[..., 0] = 0.2 * history_time
    history_rel[..., 1] = -0.1 * history_time
    history_dt = torch.zeros(batch_size, 8, device=device)
    history_dt[:, 1:] = history_time[:, 1:] - history_time[:, :-1]
    steps = torch.arange(-6, 7, dtype=torch.float32, device=device).repeat(batch_size, 1)
    relation = torch.stack((0.18 * steps, -0.08 * steps, 0.02 * steps), dim=-1)
    relation[:, 6] = 0.0
    tau = torch.tensor([0.0, 0.05, 0.12, 0.22, 0.36, 0.5], device=device)
    target = torch.tensor([0, 0, 0, 1, 1, 2], device=device).repeat(batch_size, 1)
    target[1] *= -1
    target[2] = 0
    target_visible = torch.zeros(batch_size, tau.numel(), 3, device=device)
    target_visible[..., 0] = target.to(torch.float32) * 0.18
    return {
        "history_position_rel_m": history_rel,
        "history_time_s": history_time,
        "history_dt_s": history_dt,
        "history_switch_step": torch.zeros(batch_size, 8, device=device),
        "history_mask": torch.ones(batch_size, 8, dtype=torch.bool, device=device),
        "current_position_m": torch.tensor(
            [[4.0, 1.0, 0.2]], device=device,
        ).repeat(batch_size, 1),
        "candidate_relation_m": relation,
        "candidate_step": steps,
        "candidate_mask": torch.ones(batch_size, 13, dtype=torch.bool, device=device),
        "candidate_confidence": torch.ones(batch_size, 13, device=device),
        "tau_s": tau,
        "target_switch_count": target,
        "target_visible_delta_m": target_visible,
        "target_query_mask": torch.ones(
            batch_size, tau.numel(), dtype=torch.bool, device=device,
        ),
    }


def _model(device: str = "cpu") -> OrdinalVisibleFutureModel:
    trajectory = AnonymousCandidateFutureExpert(
        "combined", channels=16, dropout=0.0,
        trained_horizon_s=0.5, maximum_absolute_step=6,
    ).to(device).eval()
    selector = OrdinalVisibleProgressSelector(
        frozen_context_features=4 * trajectory.channels + 15,
        channels=32, dropout=0.0,
        trained_horizon_s=trajectory.trained_horizon_s,
        maximum_absolute_step=trajectory.maximum_absolute_step,
    ).to(device)
    return OrdinalVisibleFutureModel(trajectory, selector)


def _state_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def test_frozen_trajectory_and_conditional_output_remain_bit_exact() -> None:
    model = _model().train()
    batch = _batch()
    before_state = _state_sha(model.trajectory)
    with torch.no_grad():
        expected = model.trajectory(
            batch["history_position_rel_m"], batch["history_time_s"],
            batch["history_dt_s"], batch["history_switch_step"],
            batch["history_mask"], batch["current_position_m"],
            batch["candidate_relation_m"], batch["candidate_step"],
            batch["candidate_mask"], batch["candidate_confidence"],
            batch["tau_s"],
        )["conditional_delta_m"]
    prediction = model(batch)
    assert torch.equal(prediction["conditional_delta_m"], expected)
    objective, _ = ordinal_visible_selector_loss(prediction, batch)
    objective.backward()
    assert any(parameter.grad is not None for parameter in model.selector.parameters())
    assert all(parameter.grad is None for parameter in model.trajectory.parameters())
    optimizer = torch.optim.AdamW(model.selector.parameters(), lr=1e-3)
    optimizer.step()
    assert _state_sha(model.trajectory) == before_state


def test_candidate_permutation_is_equivariant() -> None:
    model = _model().eval()
    batch = _batch()
    baseline = model(batch)
    permutation = torch.tensor([
        [12, 8, 2, 3, 10, 6, 1, 11, 0, 5, 9, 4, 7],
        [4, 7, 0, 12, 5, 1, 9, 3, 11, 6, 2, 8, 10],
        [1, 0, 4, 2, 8, 3, 12, 10, 6, 5, 7, 11, 9],
    ])
    changed = dict(batch)
    for key in (
        "candidate_relation_m", "candidate_step", "candidate_mask",
        "candidate_confidence",
    ):
        value = batch[key]
        index = (
            permutation.unsqueeze(-1).expand(-1, -1, value.shape[-1])
            if value.ndim == 3 else permutation
        )
        changed[key] = value.gather(1, index)
    result = model(changed)
    index3 = permutation[:, None].expand(-1, baseline["switch_logits"].shape[1], -1)
    torch.testing.assert_close(
        result["switch_logits"], baseline["switch_logits"].gather(2, index3),
    )
    assert torch.equal(result["selected_switch_step"], baseline["selected_switch_step"])
    torch.testing.assert_close(result["position_m"], baseline["position_m"])


def test_query_permutation_only_reorders_query_outputs() -> None:
    model = _model().eval()
    batch = _batch()
    baseline = model(batch)
    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    changed = dict(batch)
    changed["tau_s"] = batch["tau_s"][permutation]
    result = model(changed)
    for key in (
        "switch_logits", "switch_probability", "selected_candidate_row",
        "selected_switch_step", "expected_switch_step", "hard_switch_magnitude",
        "cumulative_boundary_probability", "delta_m", "position_m",
    ):
        torch.testing.assert_close(result[key], baseline[key][:, permutation])

    subset = dict(batch)
    subset["tau_s"] = batch["tau_s"][[1, 4]]
    subset_result = model(subset)
    for key in (
        "switch_logits", "switch_probability", "selected_switch_step",
        "expected_switch_step", "hard_switch_magnitude",
        "cumulative_boundary_probability",
    ):
        torch.testing.assert_close(subset_result[key], baseline[key][:, [1, 4]])


def test_tau_zero_is_exact_and_positive_progress_is_hard_monotone() -> None:
    model = _model().eval()
    with torch.no_grad():
        model.selector.direction_head.weight.zero_()
        model.selector.direction_head.bias.copy_(torch.tensor([-8.0, 8.0]))
        model.selector.crossing_interval_head.weight.zero_()
        model.selector.crossing_interval_head.bias.fill_(-3.0)
    batch = _batch()
    result = model(batch)
    assert torch.equal(
        result["selected_switch_step"][:, 0],
        torch.zeros(3),
    )
    assert torch.equal(
        result["switch_probability"][:, 0, 6], torch.ones(3),
    )
    difference = result["selected_switch_step"][:, 1:] - result["selected_switch_step"][:, :-1]
    assert bool(torch.all(difference >= 0))


def test_extreme_crossing_times_are_finite_and_permutation_equivariant() -> None:
    model = _model().eval()
    with torch.no_grad():
        model.selector.direction_head.weight.zero_()
        model.selector.direction_head.bias.copy_(torch.tensor([-100.0, 100.0]))
        model.selector.crossing_interval_head.weight.zero_()
        model.selector.crossing_interval_head.bias.fill_(-100.0)
    batch = _batch()
    baseline = model(batch)
    assert bool(torch.isfinite(baseline["switch_probability"]).all())
    torch.testing.assert_close(
        baseline["switch_probability"].sum(dim=-1),
        torch.ones_like(baseline["switch_probability"].sum(dim=-1)),
    )
    assert bool(torch.isfinite(baseline["crossing_time_s"]).all())
    assert bool(torch.all(
        baseline["crossing_time_s"][:, 1:]
        > baseline["crossing_time_s"][:, :-1]
    ))
    cumulative = baseline["cumulative_boundary_probability"]
    assert bool(torch.all(cumulative[..., 1:] <= cumulative[..., :-1]))
    assert bool(torch.all(cumulative[:, 1:] >= cumulative[:, :-1]))
    assert torch.equal(
        baseline["selected_switch_step"][:, -1],
        torch.full((3,), 6.0),
    )

    permutation = torch.tensor([12, 0, 11, 1, 10, 2, 9, 3, 8, 4, 7, 5, 6])
    changed = dict(batch)
    for key in (
        "candidate_relation_m", "candidate_step", "candidate_mask",
        "candidate_confidence",
    ):
        changed[key] = batch[key][:, permutation]
    permuted = model(changed)
    assert torch.equal(
        permuted["selected_switch_step"], baseline["selected_switch_step"],
    )


def test_exact_boundary_median_depends_on_step_not_row() -> None:
    model = _model().eval()
    selector = model.selector
    minimum_gap = 0.5 * 1e-4
    inverse_softplus = torch.log(torch.expm1(torch.tensor(
        (0.5 - minimum_gap) / 0.5,
    )))
    with torch.no_grad():
        selector.direction_head.weight.zero_()
        selector.direction_head.bias.copy_(torch.tensor([-100.0, 100.0]))
        selector.crossing_interval_head.weight.zero_()
        selector.crossing_interval_head.bias.fill_(100.0)
        selector.crossing_interval_head.bias[0] = inverse_softplus
    batch = _batch()
    batch["tau_s"] = torch.tensor([0.5])
    baseline = model(batch)
    assert torch.equal(
        baseline["selected_switch_step"], torch.ones(3, 1),
    )
    permutation = torch.tensor([7, 6, 0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12])
    changed = dict(batch)
    for key in (
        "candidate_relation_m", "candidate_step", "candidate_mask",
        "candidate_confidence",
    ):
        changed[key] = batch[key][:, permutation]
    assert torch.equal(
        model(changed)["selected_switch_step"],
        baseline["selected_switch_step"],
    )


def test_missing_or_duplicated_target_step_fails_closed() -> None:
    model = _model().eval()
    batch = _batch()
    prediction = model(batch)
    missing = dict(batch)
    missing["target_switch_count"] = batch["target_switch_count"].clone()
    missing["target_switch_count"][0, -1] = 7
    with pytest.raises(ValueError, match="missing or duplicated"):
        ordinal_visible_selector_loss(prediction, missing)

    duplicated = dict(batch)
    duplicated["candidate_step"] = batch["candidate_step"].clone()
    duplicated["candidate_step"][:, -1] = 2
    with pytest.raises(ValueError, match="missing or duplicated"):
        ordinal_visible_selector_loss(prediction, duplicated)

    invalid_candidates = dict(batch)
    invalid_candidates["candidate_step"] = batch["candidate_step"].clone()
    invalid_candidates["candidate_step"][:, -1] = 7
    with pytest.raises(ValueError, match="exceeds configured range"):
        model(invalid_candidates)


def test_config_excludes_identity_and_candidate_wise_shortcuts() -> None:
    config = _model().selector.config
    assert config["candidate_wise_mlp"] is False
    assert config["candidate_row_embedding"] is False
    assert config["physical_id_input"] is False
    assert config["motion_class_forward_input"] is False
    assert config["future_candidate_path_input"] is False
    assert config["fixed_switch_time"] is False
    assert config["physics_position_decoder"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_forward_backward_is_finite() -> None:
    model = _model("cuda").train()
    batch = _batch(device="cuda")
    prediction = model(batch)
    objective, _ = ordinal_visible_selector_loss(prediction, batch)
    assert bool(torch.isfinite(objective))
    objective.backward()
    gradients = [
        parameter.grad for parameter in model.selector.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(value).all()) for value in gradients)
