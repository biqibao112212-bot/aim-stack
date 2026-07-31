from __future__ import annotations

import torch

from training.stage3.prequential_predictive_risk import (
    fit_only_noise_variance,
    masked_gaussian_predictive_score,
    paired_predictive_risk_features,
)


def test_fit_only_variance_uses_effective_residual_dof() -> None:
    result = fit_only_noise_variance(
        torch.tensor([8.0, 1.0]), torch.tensor([10.0, 4.0]),
        torch.tensor([2.0, 3.5]), minimum_residual_dof=2.0,
    )
    torch.testing.assert_close(result["variance"], torch.tensor([1.0, 0.5]))
    assert result["valid"].tolist() == [True, False]
    torch.testing.assert_close(result["residual_dof"], torch.tensor([8.0, 0.5]))


def test_gaussian_predictive_score_keeps_logdet_and_mask() -> None:
    residual = torch.tensor([[1.0, 2.0, 99.0], [1.0, 0.0, 0.0]])
    covariance = torch.stack((torch.eye(3), 4.0 * torch.eye(3)))
    mask = torch.tensor([[True, True, False], [True, True, True]])
    result = masked_gaussian_predictive_score(residual, covariance, mask)
    torch.testing.assert_close(result["quadratic"], torch.tensor([5.0, 0.25]))
    torch.testing.assert_close(
        result["log_determinant"],
        torch.tensor([0.0, 3.0 * torch.log(torch.tensor(4.0))]),
    )
    assert result["valid"].tolist() == [True, True]
    assert result["dimension_count"].tolist() == [2.0, 3.0]


def test_gaussian_predictive_score_is_differentiable() -> None:
    residual = torch.tensor([[0.4, -0.2]], requires_grad=True)
    scale = torch.tensor([1.5], requires_grad=True)
    covariance = torch.diag_embed(scale[:, None].expand(-1, 2))
    result = masked_gaussian_predictive_score(
        residual, covariance, torch.ones(1, 2, dtype=torch.bool),
    )
    result["score"].sum().backward()
    assert residual.grad is not None and bool(torch.isfinite(residual.grad).all())
    assert scale.grad is not None and bool(torch.isfinite(scale.grad).all())


def test_paired_risk_swap_is_exactly_antisymmetric() -> None:
    q0 = torch.tensor([0.4, 1.5, 2.0])
    history = torch.tensor([1.2, 1.0, 3.0])
    valid = torch.tensor([True, True, False])
    forward = paired_predictive_risk_features(q0, history, valid, valid)
    swapped = paired_predictive_risk_features(history, q0, valid, valid)
    torch.testing.assert_close(
        forward["signed_evidence"], -swapped["signed_evidence"],
    )
    torch.testing.assert_close(
        forward["normalized_signed_evidence"],
        -swapped["normalized_signed_evidence"],
    )
    torch.testing.assert_close(
        forward["feature"][..., 0], swapped["feature"][..., 0],
    )
    assert torch.equal(forward["valid"], swapped["valid"])


def test_invalid_joint_covariance_fails_closed() -> None:
    residual = torch.tensor([[1.0, 2.0]])
    covariance = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]])
    result = masked_gaussian_predictive_score(
        residual, covariance, torch.ones(1, 2, dtype=torch.bool),
    )
    assert not bool(result["valid"].item())
    assert result["score"].item() == 0.0
