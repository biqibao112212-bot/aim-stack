from __future__ import annotations

import inspect

import pytest
import torch

from training.stage3.prequential_profile_diagnostics import (
    linear_gaussian_crossfit_diagnostics,
    profile_refit_drift_summary,
)


def _problem(
    *, dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    fit_design = torch.tensor([
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 0.0],
            [1.0, 0.0, 2.0],
        ],
        [
            [1.0, -1.0, 0.0],
            [1.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
        ],
    ], dtype=dtype)
    fit_target = torch.tensor([
        [1.10, 1.45, 0.72, 1.18, 1.92, 0.35],
        [-0.20, 0.52, 1.31, 1.75, 2.01, -0.51],
    ], dtype=dtype)
    heldout_design = torch.tensor([
        [[1.0, 1.5, 0.5], [1.0, -0.5, 1.5]],
        [[1.0, 0.5, -0.5], [1.0, -0.5, 0.5]],
    ], dtype=dtype)
    heldout_target = torch.tensor([
        [1.42, 0.41], [1.01, 0.12],
    ], dtype=dtype)
    return {
        "fit_design": fit_design,
        "fit_target": fit_target,
        "fit_mask": torch.ones(2, 6, dtype=torch.bool),
        "heldout_design": heldout_design,
        "heldout_target": heldout_target,
        "heldout_mask": torch.ones(2, 2, dtype=torch.bool),
        "prior_precision": 0.2 * torch.eye(
            3, dtype=dtype,
        ).unsqueeze(0).expand(2, -1, -1).clone(),
        "prior_natural": torch.zeros(2, 3, dtype=dtype),
        "velocity_column_mask": torch.tensor([False, True, True]),
    }


def test_interface_contains_no_truth_identity_class_or_future_fields() -> None:
    parameters = set(inspect.signature(
        linear_gaussian_crossfit_diagnostics,
    ).parameters)
    forbidden = {"truth", "session", "motion_class", "physical_armor_id", "future"}
    assert parameters.isdisjoint(forbidden)


def test_heldout_target_cannot_change_fit_noise_covariance_or_leverage() -> None:
    inputs = _problem()
    first = linear_gaussian_crossfit_diagnostics(**inputs)
    changed = dict(inputs)
    changed["heldout_target"] = inputs["heldout_target"] + torch.tensor(
        [[100.0, -50.0], [-80.0, 120.0]], dtype=torch.float64,
    )
    second = linear_gaussian_crossfit_diagnostics(**changed)
    for name in (
        "parameter_mean", "parameter_covariance", "velocity_covariance",
        "fit_sse", "effective_parameter_count", "noise_variance",
        "heldout_prediction", "heldout_leverage",
        "heldout_predictive_covariance",
    ):
        assert torch.equal(first[name], second[name]), name
    assert not torch.equal(
        first["heldout_joint_score"], second["heldout_joint_score"],
    )


def test_joint_and_marginal_scores_retain_gaussian_log_determinant() -> None:
    result = linear_gaussian_crossfit_diagnostics(**_problem())
    assert result["fit_supported"].tolist() == [True, True]
    assert result["heldout_joint_valid"].tolist() == [True, True]
    expected = (
        result["heldout_joint_quadratic"]
        + result["heldout_joint_log_determinant"]
    ) / result["heldout_joint_dimension_count"]
    torch.testing.assert_close(result["heldout_joint_score"], expected)
    assert bool((result["heldout_joint_log_determinant"].abs() > 1e-8).all())
    marginal_expected = (
        result["heldout_marginal_quadratic"]
        + result["heldout_marginal_log_determinant"]
    )
    torch.testing.assert_close(result["heldout_marginal_score"], marginal_expected)


def test_masks_hide_nonfinite_padding_without_affecting_visible_scores() -> None:
    inputs = _problem()
    inputs["fit_mask"][0, -1] = False
    inputs["heldout_mask"][1, -1] = False
    reference = linear_gaussian_crossfit_diagnostics(**inputs)
    padded = {name: value.clone() for name, value in inputs.items()}
    padded["fit_design"][0, -1] = torch.nan
    padded["fit_target"][0, -1] = torch.nan
    padded["heldout_design"][1, -1] = torch.nan
    padded["heldout_target"][1, -1] = torch.nan
    actual = linear_gaussian_crossfit_diagnostics(**padded)
    for name in (
        "noise_variance", "parameter_mean", "parameter_covariance",
        "heldout_joint_score", "heldout_marginal_score",
    ):
        torch.testing.assert_close(actual[name], reference[name])
    assert actual["heldout_marginal_score"][1, -1].item() == 0.0


def test_singular_fit_normal_fails_closed() -> None:
    design = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
    result = linear_gaussian_crossfit_diagnostics(
        fit_design=design,
        fit_target=torch.tensor([[1.0, 2.0]]),
        fit_mask=torch.ones(1, 2, dtype=torch.bool),
        heldout_design=torch.tensor([[[3.0, 3.0]]]),
        heldout_target=torch.tensor([[3.0]]),
        heldout_mask=torch.ones(1, 1, dtype=torch.bool),
        prior_precision=torch.zeros(1, 2, 2),
        prior_natural=torch.zeros(1, 2),
        velocity_column_mask=torch.tensor([True, False]),
    )
    assert result["fit_supported"].tolist() == [False]
    assert result["heldout_joint_valid"].tolist() == [False]
    for name in (
        "parameter_mean", "parameter_covariance", "velocity_covariance",
        "noise_variance", "heldout_joint_score", "heldout_marginal_score",
    ):
        assert torch.count_nonzero(result[name]).item() == 0, name


def test_fit_support_requires_declared_minimum_residual_dof() -> None:
    inputs = _problem()
    # Six fit rows with an almost unregularized three-parameter model have
    # about three residual degrees of freedom.  Raising the declared minimum
    # above that must fail closed even though the normal matrix is solvable.
    result = linear_gaussian_crossfit_diagnostics(
        **inputs, minimum_residual_dof=4.0,
    )
    assert result["fit_supported"].tolist() == [False, False]
    assert result["heldout_joint_valid"].tolist() == [False, False]


def test_indefinite_gaussian_prior_is_rejected() -> None:
    inputs = _problem()
    inputs["prior_precision"][0, 0, 0] = -0.1
    with pytest.raises(ValueError, match="positive semidefinite"):
        linear_gaussian_crossfit_diagnostics(**inputs)


def _rotate_profile(
    mean: torch.Tensor, covariance: torch.Tensor, rotation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    transform = torch.eye(3, dtype=mean.dtype)
    transform[:2, :2] = rotation
    return (
        mean @ transform.transpose(0, 1),
        transform.unsqueeze(0) @ covariance @ transform.transpose(0, 1).unsqueeze(0),
    )


@pytest.mark.parametrize(
    "rotation",
    [
        torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64),
        torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    ],
)
def test_refit_drift_scalars_are_o2_invariant(rotation: torch.Tensor) -> None:
    full_mean = torch.tensor([[1.0, -0.5, 0.2]], dtype=torch.float64)
    lbo_mean = torch.tensor([[-0.2, 0.7, -0.1]], dtype=torch.float64)
    full_covariance = torch.tensor([[
        [0.4, 0.1, 0.0], [0.1, 0.6, 0.0], [0.0, 0.0, 0.3],
    ]], dtype=torch.float64)
    lbo_covariance = torch.tensor([[
        [0.5, -0.05, 0.0], [-0.05, 0.7, 0.0], [0.0, 0.0, 0.2],
    ]], dtype=torch.float64)
    velocity_mask = torch.tensor([True, True, False])
    original = profile_refit_drift_summary(
        full_parameter_mean=full_mean,
        lbo_parameter_mean=lbo_mean,
        full_parameter_covariance=full_covariance,
        lbo_parameter_covariance=lbo_covariance,
        velocity_column_mask=velocity_mask,
    )
    rotated_full_mean, rotated_full_covariance = _rotate_profile(
        full_mean, full_covariance, rotation,
    )
    rotated_lbo_mean, rotated_lbo_covariance = _rotate_profile(
        lbo_mean, lbo_covariance, rotation,
    )
    rotated = profile_refit_drift_summary(
        full_parameter_mean=rotated_full_mean,
        lbo_parameter_mean=rotated_lbo_mean,
        full_parameter_covariance=rotated_full_covariance,
        lbo_parameter_covariance=rotated_lbo_covariance,
        velocity_column_mask=velocity_mask,
    )
    assert original["valid"].tolist() == rotated["valid"].tolist() == [True]
    for name in (
        "parameter_drift_l2", "parameter_drift_scaled_quadratic",
        "velocity_drift_l2", "velocity_drift_squared_l2",
        "velocity_drift_scaled_quadratic", "velocity_covariance_trace",
        "velocity_covariance_log_determinant",
    ):
        torch.testing.assert_close(original[name], rotated[name], atol=1e-10, rtol=1e-10)


def test_singular_refit_covariance_fails_closed() -> None:
    result = profile_refit_drift_summary(
        full_parameter_mean=torch.tensor([[1.0, 2.0]]),
        lbo_parameter_mean=torch.zeros(1, 2),
        full_parameter_covariance=torch.zeros(1, 2, 2),
        lbo_parameter_covariance=torch.zeros(1, 2, 2),
        velocity_column_mask=torch.tensor([True, True]),
    )
    assert result["valid"].tolist() == [False]
    assert result["velocity_drift_scaled_quadratic"].item() == 0.0


def test_crossfit_and_drift_paths_have_finite_real_gradients() -> None:
    inputs = _problem(dtype=torch.float64)
    inputs["fit_design"].requires_grad_(True)
    inputs["fit_target"].requires_grad_(True)
    inputs["heldout_design"].requires_grad_(True)
    inputs["heldout_target"].requires_grad_(True)
    result = linear_gaussian_crossfit_diagnostics(**inputs)
    objective = (
        result["heldout_joint_score"].sum()
        + 0.1 * result["parameter_covariance"].sum()
    )
    objective.backward()
    for name in (
        "fit_design", "fit_target", "heldout_design", "heldout_target",
    ):
        gradient = inputs[name].grad
        assert gradient is not None and bool(torch.isfinite(gradient).all()), name

    full_mean = torch.tensor([[0.4, -0.2]], dtype=torch.float64, requires_grad=True)
    lbo_mean = torch.tensor([[0.1, 0.3]], dtype=torch.float64, requires_grad=True)
    full_scale = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
    lbo_scale = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    drift = profile_refit_drift_summary(
        full_parameter_mean=full_mean,
        lbo_parameter_mean=lbo_mean,
        full_parameter_covariance=torch.diag_embed(
            full_scale[:, None].expand(-1, 2),
        ),
        lbo_parameter_covariance=torch.diag_embed(
            lbo_scale[:, None].expand(-1, 2),
        ),
        velocity_column_mask=torch.tensor([True, True]),
    )
    (
        drift["velocity_drift_scaled_quadratic"]
        + drift["velocity_covariance_log_determinant"]
    ).sum().backward()
    for value in (full_mean, lbo_mean, full_scale, lbo_scale):
        assert value.grad is not None and bool(torch.isfinite(value.grad).all())


def test_implementation_does_not_use_explicit_matrix_inverse() -> None:
    source = inspect.getsource(linear_gaussian_crossfit_diagnostics)
    source += inspect.getsource(profile_refit_drift_summary)
    assert "torch.linalg.inv" not in source
    assert ".inverse(" not in source
