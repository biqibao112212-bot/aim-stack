"""Proper, expert-antisymmetric risk primitives for local precision F.

This module contains no model, identity, motion label, truth target, or future
field.  It scores held-out residuals against covariance estimated exclusively
from the complementary fit set.  The Gaussian log-determinant is retained, so
inflating predictive uncertainty cannot improve the score for free.
"""

from __future__ import annotations

import torch


def fit_only_noise_variance(
    residual_sum_squares: torch.Tensor,
    fitted_dimension_count: torch.Tensor,
    effective_parameter_count: torch.Tensor,
    *,
    minimum_residual_dof: float = 2.0,
    variance_floor: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Estimate noise from fit-only SSE and effective residual degrees of freedom."""
    if minimum_residual_dof <= 0 or variance_floor <= 0:
        raise ValueError("fit-only variance bounds must be positive")
    if not (
        residual_sum_squares.shape
        == fitted_dimension_count.shape
        == effective_parameter_count.shape
    ):
        raise ValueError("fit-only variance statistic shapes differ")
    for name, value in {
        "SSE": residual_sum_squares,
        "dimension count": fitted_dimension_count,
        "effective parameter count": effective_parameter_count,
    }.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"fit-only {name} is non-finite")
    residual_dof = fitted_dimension_count - effective_parameter_count
    denominator = residual_dof.clamp_min(float(minimum_residual_dof))
    variance = (residual_sum_squares.clamp_min(0.0) / denominator).clamp_min(
        float(variance_floor),
    )
    valid = (
        (residual_dof >= float(minimum_residual_dof))
        & (residual_sum_squares >= 0)
    )
    return {
        "variance": variance,
        "residual_dof": residual_dof,
        "valid": valid,
    }


def masked_gaussian_predictive_score(
    residual: torch.Tensor,
    covariance: torch.Tensor,
    dimension_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return per-dimension Gaussian proper score for a masked joint block.

    ``residual`` is ``[B,D]``, ``covariance`` is ``[B,D,D]`` and the boolean
    mask selects the jointly scored coordinates.  Invalid rows return zero and
    an explicit false support bit; covariance is never inverted explicitly.
    """
    if residual.ndim != 2:
        raise ValueError("predictive residual must have shape [B,D]")
    batch, width = residual.shape
    if covariance.shape != (batch, width, width):
        raise ValueError("predictive covariance shape differs")
    if dimension_mask.shape != (batch, width) or dimension_mask.dtype != torch.bool:
        raise ValueError("predictive dimension mask differs")
    if bool(torch.any(dimension_mask & ~torch.isfinite(residual))):
        raise ValueError("visible predictive residual is non-finite")
    if not bool(torch.isfinite(covariance).all()):
        raise ValueError("predictive covariance is non-finite")
    score: list[torch.Tensor] = []
    quadratic: list[torch.Tensor] = []
    log_determinant: list[torch.Tensor] = []
    valid_rows: list[torch.Tensor] = []
    dimension_count: list[torch.Tensor] = []
    for row in range(batch):
        selected = torch.nonzero(dimension_mask[row], as_tuple=False).flatten()
        count = int(selected.numel())
        dimension_count.append(residual.new_tensor(float(count)))
        if count == 0:
            zero = residual.new_zeros(())
            score.append(zero)
            quadratic.append(zero)
            log_determinant.append(zero)
            valid_rows.append(torch.tensor(False, device=residual.device))
            continue
        row_residual = residual[row, selected]
        row_covariance = covariance[row][selected][:, selected]
        row_covariance = 0.5 * (row_covariance + row_covariance.T)
        factor, info = torch.linalg.cholesky_ex(row_covariance)
        # ``cholesky_ex`` deliberately does not raise for a non-positive-
        # definite row.  Do not feed its partial factor to cholesky_solve:
        # that can manufacture infinities which later leak into gradients even
        # when a final ``where`` masks the score.  Invalid support is an
        # explicit, zero-valued result instead.
        factor_diagonal = torch.diagonal(factor)
        if bool((info != 0).item()) or not bool(
            (torch.isfinite(factor_diagonal) & (factor_diagonal > 0)).all()
        ):
            zero = residual.new_zeros(())
            score.append(zero)
            quadratic.append(zero)
            log_determinant.append(zero)
            valid_rows.append(torch.tensor(False, device=residual.device))
            continue
        solved = torch.cholesky_solve(row_residual[:, None], factor).squeeze(-1)
        quad = row_residual @ solved
        logdet = 2.0 * torch.log(factor_diagonal).sum()
        finite = torch.isfinite(quad) & torch.isfinite(logdet)
        normalized = (quad + logdet) / float(count)
        score.append(torch.where(finite, normalized, torch.zeros_like(normalized)))
        quadratic.append(torch.where(finite, quad, torch.zeros_like(quad)))
        log_determinant.append(torch.where(finite, logdet, torch.zeros_like(logdet)))
        valid_rows.append(finite)
    return {
        "score": torch.stack(score),
        "quadratic": torch.stack(quadratic),
        "log_determinant": torch.stack(log_determinant),
        "dimension_count": torch.stack(dimension_count),
        "valid": torch.stack(valid_rows).to(torch.bool),
    }


def paired_predictive_risk_features(
    q0_score: torch.Tensor,
    history_score: torch.Tensor,
    q0_valid: torch.Tensor,
    history_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build symmetric context plus a signed expert-antisymmetric preference."""
    if not (
        q0_score.shape == history_score.shape == q0_valid.shape == history_valid.shape
    ):
        raise ValueError("paired predictive risk shapes differ")
    if q0_valid.dtype != torch.bool or history_valid.dtype != torch.bool:
        raise ValueError("paired predictive risk support must be boolean")
    if not bool(torch.isfinite(q0_score).all()) or not bool(
        torch.isfinite(history_score).all()
    ):
        raise ValueError("paired predictive risk is non-finite")
    valid = q0_valid & history_valid
    # Positive signed evidence means the history-only predictive risk is worse,
    # and therefore favours the q0-informed prior.
    signed = history_score - q0_score
    scale = q0_score.abs() + history_score.abs() + 1e-6
    normalized = signed / scale
    symmetric = 0.5 * (q0_score + history_score)
    feature = torch.stack((symmetric, signed, normalized), dim=-1)
    feature = torch.where(valid.unsqueeze(-1), feature, torch.zeros_like(feature))
    return {
        "feature": feature,
        "signed_evidence": torch.where(valid, signed, torch.zeros_like(signed)),
        "normalized_signed_evidence": torch.where(
            valid, normalized, torch.zeros_like(normalized),
        ),
        "valid": valid,
    }


__all__ = [
    "fit_only_noise_variance",
    "masked_gaussian_predictive_score",
    "paired_predictive_risk_features",
]
