"""Truth-free linear-Gaussian diagnostics for prequential profile refits.

The routines in this module accept only padded designs, observations, masks,
and Gaussian prior natural parameters.  They do not know sample identity,
motion class, future state, or truth velocity.  Noise and parameter uncertainty
are estimated exclusively from the fit partition; held-out observations enter
only the predictive residual scored after the fit is complete.
"""

from __future__ import annotations

import torch

from .prequential_predictive_risk import (
    fit_only_noise_variance,
    masked_gaussian_predictive_score,
)


def _validate_crossfit_inputs(
    fit_design: torch.Tensor,
    fit_target: torch.Tensor,
    fit_mask: torch.Tensor,
    heldout_design: torch.Tensor,
    heldout_target: torch.Tensor,
    heldout_mask: torch.Tensor,
    prior_precision: torch.Tensor,
    prior_natural: torch.Tensor,
    velocity_column_mask: torch.Tensor,
) -> tuple[int, int, int, int, torch.Tensor]:
    if fit_design.ndim != 3:
        raise ValueError("fit design must have shape [B,N,P]")
    batch, fit_width, parameters = fit_design.shape
    if fit_target.shape != (batch, fit_width):
        raise ValueError("fit target shape differs")
    if fit_mask.shape != (batch, fit_width) or fit_mask.dtype != torch.bool:
        raise ValueError("fit mask shape/dtype differs")
    if heldout_design.ndim != 3 or heldout_design.shape[0] != batch:
        raise ValueError("heldout design must have shape [B,M,P]")
    heldout_width = heldout_design.shape[1]
    if heldout_design.shape[2] != parameters:
        raise ValueError("fit/heldout parameter widths differ")
    if heldout_target.shape != (batch, heldout_width):
        raise ValueError("heldout target shape differs")
    if (
        heldout_mask.shape != (batch, heldout_width)
        or heldout_mask.dtype != torch.bool
    ):
        raise ValueError("heldout mask shape/dtype differs")
    if prior_precision.shape != (batch, parameters, parameters):
        raise ValueError("prior precision shape differs")
    if prior_natural.shape != (batch, parameters):
        raise ValueError("prior natural-parameter shape differs")
    if (
        velocity_column_mask.shape != (parameters,)
        or velocity_column_mask.dtype != torch.bool
    ):
        raise ValueError("velocity column mask shape/dtype differs")
    velocity_indices = torch.nonzero(
        velocity_column_mask, as_tuple=False,
    ).flatten()
    if velocity_indices.numel() == 0:
        raise ValueError("velocity column mask is empty")
    if not fit_design.is_floating_point():
        raise ValueError("linear-Gaussian inputs must be floating point")
    if (
        fit_mask.device != fit_design.device
        or heldout_mask.device != fit_design.device
        or velocity_column_mask.device != fit_design.device
    ):
        raise ValueError("linear-Gaussian mask device differs")
    tensors = (
        fit_target, heldout_design, heldout_target,
        prior_precision, prior_natural,
    )
    if any(
        value.device != fit_design.device or value.dtype != fit_design.dtype
        for value in tensors
    ):
        raise ValueError("linear-Gaussian tensor device/dtype differs")
    if bool(torch.any(
        fit_mask & (
            ~torch.isfinite(fit_target)
            | ~torch.isfinite(fit_design).all(dim=-1)
        )
    )):
        raise ValueError("visible fit inputs are non-finite")
    if bool(torch.any(
        heldout_mask & (
            ~torch.isfinite(heldout_target)
            | ~torch.isfinite(heldout_design).all(dim=-1)
        )
    )):
        raise ValueError("visible heldout inputs are non-finite")
    if not bool(torch.isfinite(prior_precision).all()) or not bool(
        torch.isfinite(prior_natural).all()
    ):
        raise ValueError("Gaussian prior is non-finite")
    if not bool(torch.allclose(
        prior_precision, prior_precision.transpose(-1, -2),
        atol=1e-6, rtol=1e-6,
    )):
        raise ValueError("prior precision is not symmetric")
    if bool((torch.linalg.eigvalsh(prior_precision) < -1e-6).any()):
        raise ValueError("prior precision is not positive semidefinite")
    return batch, fit_width, heldout_width, parameters, velocity_indices


def linear_gaussian_crossfit_diagnostics(
    *,
    fit_design: torch.Tensor,
    fit_target: torch.Tensor,
    fit_mask: torch.Tensor,
    heldout_design: torch.Tensor,
    heldout_target: torch.Tensor,
    heldout_mask: torch.Tensor,
    prior_precision: torch.Tensor,
    prior_natural: torch.Tensor,
    velocity_column_mask: torch.Tensor,
    minimum_residual_dof: float = 2.0,
    variance_floor: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Fit a masked Gaussian linear model and score held-out observations.

    The fitted objective is ``||y-X theta||^2 + theta' P theta - 2 h' theta``.
    ``P`` and ``h`` are the supplied prior precision and natural parameter.
    The posterior covariance convention is ``sigma^2 (X'X+P)^-1`` and the
    predictive covariance is ``sigma^2 I + X_hold C X_hold'``.  Linear solves
    use Cholesky factors; no explicit matrix inverse is formed.

    Invalid/singular rows fail closed: all diagnostics are zero and the
    corresponding support bits are false.
    """
    if minimum_residual_dof <= 0 or variance_floor <= 0:
        raise ValueError("fit-only variance bounds must be positive")
    (
        batch, _, heldout_width, parameters, velocity_indices,
    ) = _validate_crossfit_inputs(
        fit_design, fit_target, fit_mask,
        heldout_design, heldout_target, heldout_mask,
        prior_precision, prior_natural, velocity_column_mask,
    )
    dtype, device = fit_design.dtype, fit_design.device
    parameter_mean_rows: list[torch.Tensor] = []
    unit_covariance_rows: list[torch.Tensor] = []
    fit_sse_rows: list[torch.Tensor] = []
    fit_count_rows: list[torch.Tensor] = []
    effective_parameter_rows: list[torch.Tensor] = []
    normal_valid_rows: list[torch.Tensor] = []
    heldout_prediction_rows: list[torch.Tensor] = []
    heldout_leverage_rows: list[torch.Tensor] = []
    joint_leverage_logdet_rows: list[torch.Tensor] = []
    for row in range(batch):
        selected_fit = torch.nonzero(fit_mask[row], as_tuple=False).flatten()
        count = int(selected_fit.numel())
        zero_parameter = fit_design.new_zeros(parameters)
        zero_covariance = fit_design.new_zeros(parameters, parameters)
        zero_heldout = fit_design.new_zeros(heldout_width)
        if count == 0:
            parameter_mean_rows.append(zero_parameter)
            unit_covariance_rows.append(zero_covariance)
            fit_sse_rows.append(fit_design.new_zeros(()))
            fit_count_rows.append(fit_design.new_zeros(()))
            effective_parameter_rows.append(fit_design.new_zeros(()))
            normal_valid_rows.append(torch.tensor(False, device=device))
            heldout_prediction_rows.append(zero_heldout)
            heldout_leverage_rows.append(zero_heldout)
            joint_leverage_logdet_rows.append(fit_design.new_zeros(()))
            continue
        design = fit_design[row, selected_fit]
        target = fit_target[row, selected_fit]
        precision = 0.5 * (
            prior_precision[row] + prior_precision[row].transpose(-1, -2)
        )
        normal = design.transpose(0, 1) @ design + precision
        normal = 0.5 * (normal + normal.transpose(0, 1))
        factor, factor_info = torch.linalg.cholesky_ex(normal)
        diagonal = torch.diagonal(factor)
        factor_valid = (
            bool((factor_info == 0).item())
            and bool((torch.isfinite(diagonal) & (diagonal > 0)).all())
        )
        if not factor_valid:
            parameter_mean_rows.append(zero_parameter)
            unit_covariance_rows.append(zero_covariance)
            fit_sse_rows.append(fit_design.new_zeros(()))
            fit_count_rows.append(fit_design.new_tensor(float(count)))
            effective_parameter_rows.append(fit_design.new_zeros(()))
            normal_valid_rows.append(torch.tensor(False, device=device))
            heldout_prediction_rows.append(zero_heldout)
            heldout_leverage_rows.append(zero_heldout)
            joint_leverage_logdet_rows.append(fit_design.new_zeros(()))
            continue
        rhs = design.transpose(0, 1) @ target + prior_natural[row]
        parameter_mean = torch.cholesky_solve(
            rhs.unsqueeze(-1), factor,
        ).squeeze(-1)
        unit_covariance = torch.cholesky_solve(
            torch.eye(parameters, dtype=dtype, device=device), factor,
        )
        projected_fit = torch.cholesky_solve(design.transpose(0, 1), factor)
        fit_leverage = (design * projected_fit.transpose(0, 1)).sum(dim=-1)
        fitted = design @ parameter_mean
        fit_sse = (target - fitted).square().sum()
        heldout = torch.where(
            heldout_mask[row].unsqueeze(-1), heldout_design[row],
            torch.zeros_like(heldout_design[row]),
        )
        prediction = heldout @ parameter_mean
        projected_heldout = heldout @ unit_covariance
        leverage_matrix = projected_heldout @ heldout.transpose(0, 1)
        leverage_matrix = 0.5 * (
            leverage_matrix + leverage_matrix.transpose(0, 1)
        )
        leverage = torch.diagonal(leverage_matrix)
        selected_heldout = torch.nonzero(
            heldout_mask[row], as_tuple=False,
        ).flatten()
        if selected_heldout.numel() == 0:
            leverage_logdet = fit_design.new_zeros(())
        else:
            selected_leverage = leverage_matrix[selected_heldout][
                :, selected_heldout
            ]
            identity = torch.eye(
                selected_heldout.numel(), dtype=dtype, device=device,
            )
            leverage_factor, leverage_info = torch.linalg.cholesky_ex(
                identity + selected_leverage,
            )
            leverage_diagonal = torch.diagonal(leverage_factor)
            leverage_valid = (
                bool((leverage_info == 0).item())
                and bool((
                    torch.isfinite(leverage_diagonal) & (leverage_diagonal > 0)
                ).all())
            )
            leverage_logdet = (
                2.0 * torch.log(leverage_diagonal).sum()
                if leverage_valid else fit_design.new_zeros(())
            )
        finite = (
            torch.isfinite(parameter_mean).all()
            & torch.isfinite(unit_covariance).all()
            & torch.isfinite(fit_sse)
            & torch.isfinite(fit_leverage).all()
            & torch.isfinite(prediction).all()
            & torch.isfinite(leverage).all()
        )
        parameter_mean_rows.append(torch.where(
            finite, parameter_mean, zero_parameter,
        ))
        unit_covariance_rows.append(torch.where(
            finite, unit_covariance, zero_covariance,
        ))
        fit_sse_rows.append(torch.where(
            finite, fit_sse, fit_design.new_zeros(()),
        ))
        fit_count_rows.append(fit_design.new_tensor(float(count)))
        effective_parameter_rows.append(torch.where(
            finite, fit_leverage.sum(), fit_design.new_zeros(()),
        ))
        normal_valid_rows.append(finite.to(torch.bool))
        heldout_prediction_rows.append(torch.where(
            finite, prediction, zero_heldout,
        ))
        heldout_leverage_rows.append(torch.where(
            finite, leverage, zero_heldout,
        ))
        joint_leverage_logdet_rows.append(torch.where(
            finite, leverage_logdet, fit_design.new_zeros(()),
        ))
    parameter_mean = torch.stack(parameter_mean_rows)
    unit_covariance = torch.stack(unit_covariance_rows)
    fit_sse = torch.stack(fit_sse_rows)
    fit_count = torch.stack(fit_count_rows)
    effective_parameter_count = torch.stack(effective_parameter_rows)
    normal_valid = torch.stack(normal_valid_rows).to(torch.bool)
    variance_result = fit_only_noise_variance(
        fit_sse, fit_count, effective_parameter_count,
        minimum_residual_dof=minimum_residual_dof,
        variance_floor=variance_floor,
    )
    fit_supported = normal_valid & variance_result["valid"]
    noise_variance = torch.where(
        fit_supported, variance_result["variance"],
        torch.zeros_like(variance_result["variance"]),
    )
    parameter_covariance = noise_variance[:, None, None] * unit_covariance
    velocity_covariance = parameter_covariance[:, velocity_indices][
        :, :, velocity_indices
    ]
    heldout_prediction = torch.stack(heldout_prediction_rows)
    heldout_leverage = torch.stack(heldout_leverage_rows)
    joint_leverage_logdet = torch.stack(joint_leverage_logdet_rows)
    safe_heldout_target = torch.where(
        heldout_mask, heldout_target, torch.zeros_like(heldout_target),
    )
    heldout_residual = safe_heldout_target - heldout_prediction
    heldout_residual = torch.where(
        heldout_mask & fit_supported.unsqueeze(-1), heldout_residual,
        torch.zeros_like(heldout_residual),
    )
    heldout_design_safe = torch.where(
        heldout_mask.unsqueeze(-1), heldout_design,
        torch.zeros_like(heldout_design),
    )
    predictive_covariance = (
        noise_variance[:, None, None]
        * torch.eye(heldout_width, dtype=dtype, device=device).unsqueeze(0)
        + heldout_design_safe @ parameter_covariance
        @ heldout_design_safe.transpose(-1, -2)
    )
    predictive_covariance = torch.where(
        fit_supported[:, None, None], predictive_covariance,
        torch.zeros_like(predictive_covariance),
    )
    joint_mask = heldout_mask & fit_supported.unsqueeze(-1)
    joint = masked_gaussian_predictive_score(
        heldout_residual, predictive_covariance, joint_mask,
    )
    predictive_variance = torch.diagonal(
        predictive_covariance, dim1=-2, dim2=-1,
    )
    marginal_valid = (
        joint_mask & torch.isfinite(predictive_variance)
        & (predictive_variance > 0)
    )
    safe_variance = predictive_variance.clamp_min(variance_floor)
    marginal_quadratic = heldout_residual.square() / safe_variance
    marginal_log_determinant = torch.log(safe_variance)
    marginal_score = marginal_quadratic + marginal_log_determinant
    marginal_score = torch.where(
        marginal_valid, marginal_score, torch.zeros_like(marginal_score),
    )
    marginal_quadratic = torch.where(
        marginal_valid, marginal_quadratic,
        torch.zeros_like(marginal_quadratic),
    )
    marginal_log_determinant = torch.where(
        marginal_valid, marginal_log_determinant,
        torch.zeros_like(marginal_log_determinant),
    )
    return {
        "parameter_mean": parameter_mean,
        "parameter_covariance": parameter_covariance,
        "velocity_covariance": velocity_covariance,
        "fit_sse": fit_sse,
        "fit_dimension_count": fit_count,
        "effective_parameter_count": effective_parameter_count,
        "residual_dof": variance_result["residual_dof"],
        "noise_variance": noise_variance,
        "fit_supported": fit_supported,
        "heldout_prediction": heldout_prediction,
        "heldout_residual": heldout_residual,
        "heldout_leverage": heldout_leverage,
        "heldout_joint_leverage_trace": heldout_leverage.sum(dim=-1),
        "heldout_joint_leverage_log_determinant": joint_leverage_logdet,
        "heldout_predictive_covariance": predictive_covariance,
        "heldout_predictive_variance": predictive_variance,
        "heldout_marginal_score": marginal_score,
        "heldout_marginal_quadratic": marginal_quadratic,
        "heldout_marginal_log_determinant": marginal_log_determinant,
        "heldout_marginal_valid": marginal_valid,
        "heldout_joint_score": joint["score"],
        "heldout_joint_quadratic": joint["quadratic"],
        "heldout_joint_log_determinant": joint["log_determinant"],
        "heldout_joint_dimension_count": joint["dimension_count"],
        "heldout_joint_valid": joint["valid"] & fit_supported,
    }


def profile_refit_drift_summary(
    *,
    full_parameter_mean: torch.Tensor,
    lbo_parameter_mean: torch.Tensor,
    full_parameter_covariance: torch.Tensor,
    lbo_parameter_covariance: torch.Tensor,
    velocity_column_mask: torch.Tensor,
    full_supported: torch.Tensor | None = None,
    lbo_supported: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return covariance-normalized full-vs-LBO drift scalar summaries.

    Euclidean norms and symmetric covariance-scaled quadratic forms are
    invariant under a consistent orthogonal change of basis, including planar
    O(2) rotations and reflections.  Full and leave-block-out fits are nested
    and correlated, so ``C_full + C_lbo`` is used only as a deterministic
    uncertainty scale.  The returned quadratic forms are deliberately not
    named or interpreted as Mahalanobis sampling distances.
    """
    if full_parameter_mean.ndim != 2:
        raise ValueError("profile parameter mean must have shape [B,P]")
    batch, parameters = full_parameter_mean.shape
    if lbo_parameter_mean.shape != (batch, parameters):
        raise ValueError("full/LBO parameter mean shapes differ")
    expected_covariance = (batch, parameters, parameters)
    if (
        full_parameter_covariance.shape != expected_covariance
        or lbo_parameter_covariance.shape != expected_covariance
    ):
        raise ValueError("full/LBO parameter covariance shapes differ")
    if (
        velocity_column_mask.shape != (parameters,)
        or velocity_column_mask.dtype != torch.bool
    ):
        raise ValueError("velocity column mask shape/dtype differs")
    velocity_indices = torch.nonzero(
        velocity_column_mask, as_tuple=False,
    ).flatten()
    if velocity_indices.numel() == 0:
        raise ValueError("velocity column mask is empty")
    tensors = (
        lbo_parameter_mean, full_parameter_covariance,
        lbo_parameter_covariance,
    )
    if any(
        value.device != full_parameter_mean.device
        or value.dtype != full_parameter_mean.dtype
        for value in tensors
    ):
        raise ValueError("full/LBO tensor device/dtype differs")
    if not all(bool(torch.isfinite(value).all()) for value in (
        full_parameter_mean, lbo_parameter_mean,
        full_parameter_covariance, lbo_parameter_covariance,
    )):
        raise ValueError("full/LBO profile diagnostics are non-finite")
    if full_supported is None:
        full_supported = torch.ones(batch, dtype=torch.bool, device=full_parameter_mean.device)
    if lbo_supported is None:
        lbo_supported = torch.ones(batch, dtype=torch.bool, device=full_parameter_mean.device)
    if (
        full_supported.shape != (batch,) or full_supported.dtype != torch.bool
        or lbo_supported.shape != (batch,) or lbo_supported.dtype != torch.bool
    ):
        raise ValueError("full/LBO support shape/dtype differs")
    if (
        full_supported.device != full_parameter_mean.device
        or lbo_supported.device != full_parameter_mean.device
        or velocity_column_mask.device != full_parameter_mean.device
    ):
        raise ValueError("full/LBO support or velocity-mask device differs")
    base_supported = full_supported & lbo_supported
    drift = full_parameter_mean - lbo_parameter_mean
    covariance = full_parameter_covariance + lbo_parameter_covariance
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    velocity_drift = drift[:, velocity_indices]
    velocity_covariance = covariance[:, velocity_indices][:, :, velocity_indices]
    parameter_scaled_quadratic: list[torch.Tensor] = []
    velocity_scaled_quadratic: list[torch.Tensor] = []
    velocity_logdet: list[torch.Tensor] = []
    valid_rows: list[torch.Tensor] = []
    for row in range(batch):
        zero = full_parameter_mean.new_zeros(())
        if not bool(base_supported[row]):
            parameter_scaled_quadratic.append(zero)
            velocity_scaled_quadratic.append(zero)
            velocity_logdet.append(zero)
            valid_rows.append(torch.tensor(False, device=drift.device))
            continue
        full_factor, full_info = torch.linalg.cholesky_ex(covariance[row])
        velocity_factor, velocity_info = torch.linalg.cholesky_ex(
            velocity_covariance[row],
        )
        full_diagonal = torch.diagonal(full_factor)
        velocity_diagonal = torch.diagonal(velocity_factor)
        valid = (
            bool((full_info == 0).item())
            and bool((velocity_info == 0).item())
            and bool((
                torch.isfinite(full_diagonal) & (full_diagonal > 0)
            ).all())
            and bool((
                torch.isfinite(velocity_diagonal) & (velocity_diagonal > 0)
            ).all())
        )
        if not valid:
            parameter_scaled_quadratic.append(zero)
            velocity_scaled_quadratic.append(zero)
            velocity_logdet.append(zero)
            valid_rows.append(torch.tensor(False, device=drift.device))
            continue
        solved_full = torch.cholesky_solve(
            drift[row].unsqueeze(-1), full_factor,
        ).squeeze(-1)
        solved_velocity = torch.cholesky_solve(
            velocity_drift[row].unsqueeze(-1), velocity_factor,
        ).squeeze(-1)
        full_value = drift[row] @ solved_full
        velocity_value = velocity_drift[row] @ solved_velocity
        logdet = 2.0 * torch.log(velocity_diagonal).sum()
        finite = (
            torch.isfinite(full_value)
            & torch.isfinite(velocity_value)
            & torch.isfinite(logdet)
        )
        parameter_scaled_quadratic.append(torch.where(finite, full_value, zero))
        velocity_scaled_quadratic.append(torch.where(finite, velocity_value, zero))
        velocity_logdet.append(torch.where(finite, logdet, zero))
        valid_rows.append(finite.to(torch.bool))
    valid = torch.stack(valid_rows).to(torch.bool)
    parameter_l2 = torch.linalg.vector_norm(drift, dim=-1)
    velocity_l2 = torch.linalg.vector_norm(velocity_drift, dim=-1)
    velocity_trace = torch.diagonal(
        velocity_covariance, dim1=-2, dim2=-1,
    ).sum(dim=-1)
    return {
        "parameter_drift_l2": torch.where(
            valid, parameter_l2, torch.zeros_like(parameter_l2),
        ),
        "parameter_drift_scaled_quadratic": torch.stack(
            parameter_scaled_quadratic,
        ),
        "velocity_drift_l2": torch.where(
            valid, velocity_l2, torch.zeros_like(velocity_l2),
        ),
        "velocity_drift_squared_l2": torch.where(
            valid, velocity_l2.square(), torch.zeros_like(velocity_l2),
        ),
        "velocity_drift_scaled_quadratic": torch.stack(
            velocity_scaled_quadratic,
        ),
        "velocity_covariance_trace": torch.where(
            valid, velocity_trace, torch.zeros_like(velocity_trace),
        ),
        "velocity_covariance_log_determinant": torch.stack(velocity_logdet),
        "valid": valid,
    }


__all__ = [
    "linear_gaussian_crossfit_diagnostics",
    "profile_refit_drift_summary",
]
