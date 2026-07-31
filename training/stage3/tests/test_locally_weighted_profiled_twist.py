from __future__ import annotations

from copy import deepcopy
import inspect

import torch

import training.stage3.locally_weighted_profiled_twist as local_solver_module
from training.stage3.locally_weighted_profiled_twist import (
    LocallyWeightedProfiledTwistAtOmega,
    centered_visible_observation_precision,
    visible_observation_precision,
)
from training.stage3.profiled_center_twist_future import (
    ProfiledRigidTwistAtOmega,
    translation_only_fwl,
)


def _fixture(batch_size: int = 2) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    events = 8
    time = torch.linspace(-0.07, 0.0, events).expand(batch_size, -1).clone()
    row_pattern = torch.arange(batch_size) % 2
    center = torch.tensor([
        [0.20, 0.00, 0.01], [0.10, -0.08, -0.02],
    ])[row_pattern]
    radial = torch.tensor([
        [-0.20, 0.00, 0.00], [0.00, 0.30, 0.01],
        [0.20, 0.00, -0.01], [0.00, -0.30, 0.00],
    ])
    q0 = center[:, None] + radial[None]
    velocity = torch.tensor([
        [0.55, -0.35, 0.08], [-0.25, 0.40, -0.03],
    ])[row_pattern]
    omega = torch.tensor([4.0, -6.0])[row_pattern]
    history = torch.zeros(batch_size, events, 4, 3)
    for row in range(batch_size):
        for event in range(events):
            theta = omega[row] * time[row, event]
            rotation = torch.tensor([
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ])
            history[row, event, :, :2] = (
                (torch.eye(2) - rotation) @ center[row, :2]
                + time[row, event] * velocity[row, :2]
                + (rotation @ q0[row, :, :2].T).T
            )
            history[row, event, :, 2] = (
                q0[row, :, 2] + time[row, event] * velocity[row, 2]
            )
    mask = torch.ones(batch_size, events, 4, dtype=torch.bool)
    batch = {
        "history_obs_rel_m": history,
        "history_obs_mask": mask,
        "history_event_mask": torch.ones(batch_size, events, dtype=torch.bool),
        "history_time_s": time,
    }
    prior = {
        "center_supported": torch.ones(batch_size, dtype=torch.bool),
        "center_offset_m": center,
        "center_log_variance_xy_z": torch.tensor([
            [-3.0, -3.5], [-2.5, -3.2],
        ])[row_pattern],
        "q0_relation_m": q0,
    }
    return batch, prior, omega


def _call(
    model: LocallyWeightedProfiledTwistAtOmega,
    batch: dict[str, torch.Tensor],
    prior: dict[str, torch.Tensor],
    omega: torch.Tensor,
    log_precision: torch.Tensor,
    alpha: torch.Tensor,
    center_alpha: torch.Tensor | None = None,
    *,
    use_q0_prior: bool = True,
) -> dict[str, torch.Tensor]:
    if center_alpha is None:
        center_alpha = alpha.mean(dim=-1)
    return model(
        batch["history_obs_rel_m"], batch["history_obs_mask"],
        batch["history_event_mask"], batch["history_time_s"], omega, prior,
        observation_log_precision=log_precision, anchor_alpha=alpha,
        center_alpha=center_alpha, use_q0_prior=use_q0_prior,
    )


@torch.inference_mode()
def test_neutral_history_and_q0_boundaries_delegate_exact_v14() -> None:
    batch, prior, omega = _fixture()
    kwargs = dict(center_precision=25.0, use_learned_center_variance=True)
    legacy = ProfiledRigidTwistAtOmega(**kwargs).eval()
    model = LocallyWeightedProfiledTwistAtOmega(**kwargs).eval()
    neutral = torch.zeros_like(batch["history_obs_mask"], dtype=torch.float32)
    for alpha_value, informed in ((0.0, False), (1.0, True)):
        alpha = torch.full((2, 4), alpha_value)
        expected = legacy(
            batch["history_obs_rel_m"], batch["history_obs_mask"],
            batch["history_event_mask"], batch["history_time_s"], omega,
            prior, use_q0_prior=informed,
        )
        actual = _call(model, batch, prior, omega, neutral, alpha)
        assert torch.all(actual["boundary_delegated"])
        for name in (
            "velocity_mps", "profiled_center_offset_m", "profile_energy",
            "profile_supported", "velocity_information_min_eigenvalue_s2",
            "velocity_information_condition", "vertical_velocity_information_s2",
        ):
            assert torch.equal(actual[name], expected[name]), name


def test_centered_visible_logits_have_unit_geometric_mean() -> None:
    logit = torch.tensor([[[0.0, 2.0, -4.0, 8.0], [2.0, 4.0, 1.0, 9.0]]])
    mask = torch.tensor([[[True, True, False, False], [True, True, False, False]]])
    precision = centered_visible_observation_precision(logit, mask)
    torch.testing.assert_close(
        torch.log(precision[mask]).mean(), torch.zeros(()), atol=1e-7, rtol=0,
    )
    assert torch.count_nonzero(precision[~mask]) == 0
    projected = torch.tensor([[[0.4, -0.7, 9.0, 8.0], [0.1, 0.2, 7.0, 6.0]]])
    direct = visible_observation_precision(projected, mask)
    torch.testing.assert_close(direct[mask], torch.exp(projected[mask]))
    assert torch.count_nonzero(direct[~mask]) == 0


def test_general_weight_path_is_fp32_and_all_precision_gradients_are_finite() -> None:
    batch, prior, omega = _fixture()
    batch = deepcopy(batch)
    noise = torch.randn(
        batch["history_obs_rel_m"].shape,
        generator=torch.Generator().manual_seed(4),
    )
    batch["history_obs_rel_m"] = batch["history_obs_rel_m"] + 0.003 * noise
    logit = torch.randn(2, 8, 4, generator=torch.Generator().manual_seed(5)).requires_grad_()
    raw_alpha = torch.randn(2, 4, generator=torch.Generator().manual_seed(6), requires_grad=True)
    raw_center_alpha = torch.randn(
        2, generator=torch.Generator().manual_seed(7), requires_grad=True,
    )
    alpha = torch.sigmoid(raw_alpha)
    center_alpha = torch.sigmoid(raw_center_alpha)
    result = _call(
        LocallyWeightedProfiledTwistAtOmega().eval(), batch, prior, omega,
        logit, alpha, center_alpha,
    )
    assert result["velocity_mps"].dtype == torch.float32
    assert torch.all(result["profile_supported"])
    loss = (
        result["velocity_mps"].square().mean()
        + result["profile_energy"].mean()
        + 1e-3 * result["observation_precision"].square().mean()
    )
    loss.backward()
    for gradient in (logit.grad, raw_alpha.grad, raw_center_alpha.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def test_general_fp32_path_matches_q0_boundary_without_delegation() -> None:
    batch, prior, omega = _fixture()
    kwargs = dict(center_precision=25.0, use_learned_center_variance=True)
    legacy = ProfiledRigidTwistAtOmega(**kwargs).eval()
    model = LocallyWeightedProfiledTwistAtOmega(**kwargs).eval()
    logit = torch.zeros(2, 8, 4, requires_grad=True)
    alpha = torch.ones(2, 4, requires_grad=True)
    actual = _call(model, batch, prior, omega, logit, alpha)
    expected = legacy(
        batch["history_obs_rel_m"], batch["history_obs_mask"],
        batch["history_event_mask"], batch["history_time_s"], omega,
        prior, use_q0_prior=True,
    )
    assert not torch.any(actual["boundary_delegated"])
    for name in (
        "velocity_mps", "profiled_center_offset_m", "profile_energy",
        "velocity_information_min_eigenvalue_s2",
        "velocity_information_condition", "vertical_velocity_information_s2",
    ):
        torch.testing.assert_close(actual[name], expected[name], atol=1e-5, rtol=1e-5)


def test_natural_parameter_interpolation_and_velocity_leverage_diagnostics() -> None:
    batch, prior, omega = _fixture()
    alpha = torch.full((2, 4), 0.5)
    logit = torch.linspace(-0.7, 0.9, 64).reshape(2, 8, 4)
    model = LocallyWeightedProfiledTwistAtOmega().eval()
    result = _call(model, batch, prior, omega, logit, alpha)
    expected_anchor = model.q_ridge + 0.5 * (
        model.q0_endpoint_precision - model.q_ridge
    )
    torch.testing.assert_close(
        result["effective_anchor_precision"],
        torch.full_like(alpha, expected_anchor),
    )
    torch.testing.assert_close(
        result["anchor_natural_rhs"],
        0.5 * model.q0_endpoint_precision * prior["q0_relation_m"],
    )
    torch.testing.assert_close(result["center_alpha"], torch.full((2,), 0.5))
    assert torch.isfinite(result["weighted_residual_m"]).all()
    assert torch.isfinite(result["velocity_leverage"]).all()
    torch.testing.assert_close(
        result["velocity_leverage_xy"].sum(dim=(1, 2)),
        torch.full((2,), 2.0), atol=2e-5, rtol=2e-5,
    )
    torch.testing.assert_close(
        result["velocity_leverage_z"].sum(dim=(1, 2)),
        torch.ones(2), atol=2e-5, rtol=2e-5,
    )


@torch.inference_mode()
def test_general_path_is_s4_invariant_and_role_diagnostics_are_equivariant() -> None:
    batch, prior, omega = _fixture()
    model = LocallyWeightedProfiledTwistAtOmega().eval()
    logit = torch.linspace(-0.8, 0.9, 64).reshape(2, 8, 4)
    alpha = torch.tensor([[0.2, 0.8, 0.4, 0.6], [0.7, 0.3, 0.9, 0.5]])
    reference = _call(model, batch, prior, omega, logit, alpha)
    order = torch.tensor([2, 0, 3, 1])
    changed = deepcopy(batch)
    changed["history_obs_rel_m"] = batch["history_obs_rel_m"][:, :, order]
    changed["history_obs_mask"] = batch["history_obs_mask"][:, :, order]
    changed_prior = dict(prior)
    changed_prior["q0_relation_m"] = prior["q0_relation_m"][:, order]
    actual = _call(
        model, changed, changed_prior, omega, logit[:, :, order], alpha[:, order],
    )
    for name in (
        "velocity_mps", "profiled_center_offset_m", "profile_energy",
        "velocity_information_min_eigenvalue_s2",
        "velocity_information_condition", "vertical_velocity_information_s2",
    ):
        torch.testing.assert_close(actual[name], reference[name], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        actual["weighted_residual_m"], reference["weighted_residual_m"][:, :, order],
        atol=2e-5, rtol=2e-5,
    )
    torch.testing.assert_close(
        actual["velocity_leverage"], reference["velocity_leverage"][:, :, order],
        atol=2e-5, rtol=2e-5,
    )


def _transform_xy(
    batch: dict[str, torch.Tensor], prior: dict[str, torch.Tensor],
    transform: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    changed = deepcopy(batch)
    changed_prior = deepcopy(prior)
    changed["history_obs_rel_m"][..., :2] = (
        batch["history_obs_rel_m"][..., :2] @ transform.T
    )
    changed_prior["q0_relation_m"][..., :2] = (
        prior["q0_relation_m"][..., :2] @ transform.T
    )
    changed_prior["center_offset_m"][..., :2] = (
        prior["center_offset_m"][..., :2] @ transform.T
    )
    return changed, changed_prior


@torch.inference_mode()
def test_general_path_is_o2_equivariant_including_reflection() -> None:
    batch, prior, omega = _fixture()
    model = LocallyWeightedProfiledTwistAtOmega().eval()
    logit = torch.linspace(-0.6, 0.8, 64).reshape(2, 8, 4)
    alpha = torch.tensor([[0.2, 0.8, 0.4, 0.6], [0.7, 0.3, 0.9, 0.5]])
    reference = _call(model, batch, prior, omega, logit, alpha)
    for transform in (
        torch.tensor([[0.6, -0.8], [0.8, 0.6]]),
        torch.tensor([[0.6, 0.8], [0.8, -0.6]]),
    ):
        changed, changed_prior = _transform_xy(batch, prior, transform)
        determinant = torch.linalg.det(transform)
        actual = _call(
            model, changed, changed_prior, determinant * omega, logit, alpha,
        )
        expected_velocity = reference["velocity_mps"].clone()
        expected_velocity[..., :2] = reference["velocity_mps"][..., :2] @ transform.T
        expected_center = reference["profiled_center_offset_m"].clone()
        expected_center[..., :2] = (
            reference["profiled_center_offset_m"][..., :2] @ transform.T
        )
        torch.testing.assert_close(
            actual["velocity_mps"], expected_velocity, atol=3e-5, rtol=3e-5,
        )
        torch.testing.assert_close(
            actual["profiled_center_offset_m"], expected_center,
            atol=3e-5, rtol=3e-5,
        )
        for name in (
            "profile_energy", "velocity_information_min_eigenvalue_s2",
            "velocity_information_condition", "vertical_velocity_information_s2",
            "velocity_leverage",
        ):
            torch.testing.assert_close(actual[name], reference[name], atol=3e-5, rtol=3e-5)


@torch.inference_mode()
def test_general_path_has_exact_common_ramp_equivariance() -> None:
    batch, prior, omega = _fixture()
    model = LocallyWeightedProfiledTwistAtOmega().eval()
    logit = torch.linspace(-0.5, 0.7, 64).reshape(2, 8, 4)
    alpha = torch.tensor([[0.2, 0.8, 0.4, 0.6], [0.7, 0.3, 0.9, 0.5]])
    reference = _call(model, batch, prior, omega, logit, alpha)
    ramp = torch.tensor([[0.3, -0.2, 0.05], [-0.1, 0.25, -0.04]])
    changed = deepcopy(batch)
    changed["history_obs_rel_m"] = (
        batch["history_obs_rel_m"]
        + batch["history_time_s"][:, :, None, None] * ramp[:, None, None]
    )
    actual = _call(model, changed, prior, omega, logit, alpha)
    torch.testing.assert_close(
        actual["velocity_mps"], reference["velocity_mps"] + ramp,
        atol=3e-5, rtol=3e-5,
    )
    for name in (
        "profiled_center_offset_m", "profile_energy",
        "velocity_information_min_eigenvalue_s2",
        "velocity_information_condition", "vertical_velocity_information_s2",
        "weighted_residual_m", "velocity_leverage",
    ):
        torch.testing.assert_close(actual[name], reference[name], atol=3e-5, rtol=3e-5)


def test_center_gate_varies_independently_from_anchor_gates() -> None:
    batch, prior, omega = _fixture()
    model = LocallyWeightedProfiledTwistAtOmega().eval()
    log_precision = torch.full((2, 8, 4), 0.15)
    anchor_alpha = torch.full((2, 4), 0.4)
    without_center = _call(
        model, batch, prior, omega, log_precision, anchor_alpha,
        torch.zeros(2),
    )
    with_center = _call(
        model, batch, prior, omega, log_precision, anchor_alpha,
        torch.ones(2),
    )
    torch.testing.assert_close(
        without_center["anchor_alpha"], with_center["anchor_alpha"],
    )
    torch.testing.assert_close(
        without_center["effective_anchor_precision"],
        with_center["effective_anchor_precision"],
    )
    torch.testing.assert_close(without_center["center_alpha"], torch.zeros(2))
    torch.testing.assert_close(with_center["center_alpha"], torch.ones(2))
    assert torch.all(
        with_center["effective_center_precision_xy"]
        > without_center["effective_center_precision_xy"]
    )
    assert torch.count_nonzero(without_center["center_natural_rhs_xy"]) == 0
    assert torch.count_nonzero(with_center["center_natural_rhs_xy"]) > 0


def test_unsupported_center_forces_both_prior_gates_to_zero() -> None:
    batch, prior, omega = _fixture()
    prior = deepcopy(prior)
    prior["center_supported"][0] = False
    model = LocallyWeightedProfiledTwistAtOmega().eval()
    result = _call(
        model, batch, prior, omega, torch.full((2, 8, 4), 0.1),
        torch.ones(2, 4), torch.tensor([1.0, 0.7]),
    )
    torch.testing.assert_close(result["anchor_alpha"][0], torch.zeros(4))
    torch.testing.assert_close(result["center_alpha"][0], torch.zeros(()))
    torch.testing.assert_close(
        result["effective_anchor_precision"][0],
        torch.full((4,), model.q_ridge),
    )
    torch.testing.assert_close(
        result["effective_center_precision_xy"][0],
        torch.tensor(model.history_center_precision),
    )
    assert torch.count_nonzero(result["anchor_natural_rhs"][0]) == 0
    assert torch.count_nonzero(result["center_natural_rhs_xy"][0]) == 0


@torch.inference_mode()
def test_translation_fallback_is_invariant_to_local_precision() -> None:
    batch, prior, omega = _fixture()
    batch = deepcopy(batch)
    batch["history_obs_mask"].zero_()
    batch["history_obs_mask"][:, 0, 0] = True
    batch["history_obs_mask"][:, -1, 0] = True
    model = LocallyWeightedProfiledTwistAtOmega().eval()
    anchor_alpha = torch.full((2, 4), 0.4)
    center_alpha = torch.full((2,), 0.6)
    low_first = torch.zeros(2, 8, 4)
    high_first = torch.zeros(2, 8, 4)
    low_first[:, 0, 0], low_first[:, -1, 0] = -2.0, 2.0
    high_first[:, 0, 0], high_first[:, -1, 0] = 2.0, -2.0
    first = _call(
        model, batch, prior, omega, low_first, anchor_alpha, center_alpha,
    )
    second = _call(
        model, batch, prior, omega, high_first, anchor_alpha, center_alpha,
    )
    expected = translation_only_fwl(
        batch["history_obs_rel_m"], batch["history_obs_mask"],
        batch["history_event_mask"], batch["history_time_s"],
        minimum_time_span_s=model.minimum_time_span_s,
    )
    assert not torch.any(first["profile_supported"])
    assert torch.all(first["fallback_supported"])
    torch.testing.assert_close(first["fallback_velocity_mps"], expected["velocity_mps"])
    torch.testing.assert_close(first["velocity_mps"], expected["velocity_mps"])
    torch.testing.assert_close(second["velocity_mps"], expected["velocity_mps"])


@torch.inference_mode()
def test_mixed_batch_delegates_each_endpoint_row_and_keeps_general_row() -> None:
    batch, prior, omega = _fixture(batch_size=3)
    model = LocallyWeightedProfiledTwistAtOmega().eval()
    legacy = ProfiledRigidTwistAtOmega().eval()
    log_precision = torch.zeros(3, 8, 4)
    log_precision[2] = 0.2
    anchor_alpha = torch.stack((
        torch.zeros(4), torch.ones(4), torch.full((4,), 0.4),
    ))
    center_alpha = torch.tensor([0.0, 1.0, 0.6])
    result = _call(
        model, batch, prior, omega, log_precision, anchor_alpha, center_alpha,
    )
    assert result["boundary_delegated"].tolist() == [True, True, False]
    legacy_fields = (
        "velocity_mps", "yaw_rate_rad_s", "profiled_center_offset_m",
        "profile_energy", "profile_energy_xy", "profile_energy_z",
        "profile_supported", "q0_prior_used", "fallback_velocity_mps",
        "fallback_supported", "state_supported",
        "velocity_information_min_eigenvalue_s2",
        "velocity_information_condition", "vertical_velocity_information_s2",
    )
    for row, informed in ((0, False), (1, True)):
        row_batch = {name: value[row:row + 1] for name, value in batch.items()}
        row_prior = {name: value[row:row + 1] for name, value in prior.items()}
        expected = legacy(
            row_batch["history_obs_rel_m"], row_batch["history_obs_mask"],
            row_batch["history_event_mask"], row_batch["history_time_s"],
            omega[row:row + 1], row_prior, use_q0_prior=informed,
        )
        for name in legacy_fields:
            assert torch.equal(result[name][row], expected[name][0]), (row, name)

    row = 2
    row_batch = {name: value[row:row + 1] for name, value in batch.items()}
    row_prior = {name: value[row:row + 1] for name, value in prior.items()}
    standalone = _call(
        model, row_batch, row_prior, omega[row:row + 1],
        log_precision[row:row + 1], anchor_alpha[row:row + 1],
        center_alpha[row:row + 1],
    )
    for name in (
        "velocity_mps", "profiled_center_offset_m", "profile_energy",
        "weighted_residual_m", "velocity_leverage",
    ):
        torch.testing.assert_close(
            result[name][row], standalone[name][0], atol=2e-5, rtol=2e-5,
        )


@torch.inference_mode()
def test_hidden_nan_is_cleaned_before_finite_translation_fallback() -> None:
    batch, prior, omega = _fixture(batch_size=1)
    batch = deepcopy(batch)
    batch["history_obs_mask"].zero_()
    batch["history_obs_mask"][:, 0, 0] = True
    batch["history_obs_mask"][:, -1, 0] = True
    visible = batch["history_obs_mask"].unsqueeze(-1).expand_as(
        batch["history_obs_rel_m"],
    )
    batch["history_obs_rel_m"] = torch.where(
        visible, batch["history_obs_rel_m"],
        torch.full_like(batch["history_obs_rel_m"], torch.nan),
    )
    model = LocallyWeightedProfiledTwistAtOmega().eval()
    result = _call(
        model, batch, prior, omega, torch.full((1, 8, 4), 0.1),
        torch.full((1, 4), 0.4), torch.full((1,), 0.6),
    )
    clean = torch.where(
        visible, batch["history_obs_rel_m"],
        torch.zeros_like(batch["history_obs_rel_m"]),
    )
    expected = translation_only_fwl(
        clean, batch["history_obs_mask"], batch["history_event_mask"],
        batch["history_time_s"],
        minimum_time_span_s=model.minimum_time_span_s,
    )
    assert not bool(result["profile_supported"].item())
    assert bool(result["fallback_supported"].item())
    assert bool(result["state_supported"].item())
    assert bool(torch.isfinite(result["velocity_mps"]).all())
    torch.testing.assert_close(result["velocity_mps"], expected["velocity_mps"])


@torch.inference_mode()
def test_float64_endpoint_inputs_return_uniform_fp32_outputs() -> None:
    batch, prior, omega = _fixture(batch_size=2)
    batch64 = {
        name: value.to(torch.float64) if value.is_floating_point() else value
        for name, value in batch.items()
    }
    prior64 = {
        name: value.to(torch.float64) if value.is_floating_point() else value
        for name, value in prior.items()
    }
    model = LocallyWeightedProfiledTwistAtOmega().eval()
    result = _call(
        model, batch64, prior64, omega.to(torch.float64),
        torch.zeros(2, 8, 4, dtype=torch.float64),
        torch.stack((
            torch.zeros(4, dtype=torch.float64),
            torch.ones(4, dtype=torch.float64),
        )),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
    )
    assert result["boundary_delegated"].tolist() == [True, True]
    for name, value in result.items():
        if value.is_floating_point():
            assert value.dtype == torch.float32, name

    legacy = ProfiledRigidTwistAtOmega().eval()
    for row, informed in ((0, False), (1, True)):
        expected = legacy(
            batch["history_obs_rel_m"][row:row + 1],
            batch["history_obs_mask"][row:row + 1],
            batch["history_event_mask"][row:row + 1],
            batch["history_time_s"][row:row + 1], omega[row:row + 1],
            {name: value[row:row + 1] for name, value in prior.items()},
            use_q0_prior=informed,
        )
        for name, expected_value in expected.items():
            assert torch.equal(result[name][row], expected_value[0]), (row, name)


def test_solver_source_has_no_unchecked_linalg_solve() -> None:
    source = inspect.getsource(local_solver_module)
    assert "torch.linalg.solve(" not in source
