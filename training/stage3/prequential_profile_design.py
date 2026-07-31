"""Dense padded linear designs for the frozen fixed-omega V14 profiler.

This module is a pure geometry/prior adapter.  It consumes causal history and
the frozen center/q0 prior, and emits the scalar XY and Z systems expected by
``linear_gaussian_crossfit_diagnostics``.  It has no truth, identity, class,
session, or future-state input.
"""

from __future__ import annotations

from typing import Any

import torch


XY_PARAMETER_COUNT = 12
Z_PARAMETER_COUNT = 5
ROLE_COUNT = 4


def _validate_inputs(
    history_obs_rel_m: torch.Tensor,
    history_obs_mask: torch.Tensor,
    history_event_mask: torch.Tensor,
    history_time_s: torch.Tensor,
    omega_rad_s: torch.Tensor,
    center_prior: dict[str, torch.Tensor],
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    if history_obs_rel_m.ndim != 4 or history_obs_rel_m.shape[2:] != (4, 3):
        raise ValueError("history observations must have shape [B,T,4,3]")
    batch, events = history_obs_rel_m.shape[:2]
    if (
        history_obs_mask.shape != (batch, events, ROLE_COUNT)
        or history_obs_mask.dtype != torch.bool
    ):
        raise ValueError("history observation mask shape/dtype differs")
    if (
        history_event_mask.shape != (batch, events)
        or history_event_mask.dtype != torch.bool
    ):
        raise ValueError("history event mask shape/dtype differs")
    if history_time_s.shape != (batch, events):
        raise ValueError("history time shape differs")
    if omega_rad_s.shape not in {(batch,), (batch, 1)}:
        raise ValueError("omega shape differs")
    if not history_obs_rel_m.is_floating_point():
        raise ValueError("profile design inputs must be floating point")
    for value in (
        history_time_s, omega_rad_s,
    ):
        if (
            value.device != history_obs_rel_m.device
            or value.dtype != history_obs_rel_m.dtype
        ):
            raise ValueError("profile design tensor device/dtype differs")
    if (
        history_obs_mask.device != history_obs_rel_m.device
        or history_event_mask.device != history_obs_rel_m.device
    ):
        raise ValueError("profile design mask device differs")
    valid = history_obs_mask & history_event_mask.unsqueeze(-1)
    if bool(torch.any(valid & ~torch.isfinite(history_obs_rel_m).all(dim=-1))):
        raise ValueError("visible history observation is non-finite")
    active_event = valid.any(dim=-1)
    if bool(torch.any(active_event & ~torch.isfinite(history_time_s))):
        raise ValueError("visible history time is non-finite")
    omega_flat = omega_rad_s.reshape(batch)
    if not bool(torch.isfinite(omega_flat).all()):
        raise ValueError("omega is non-finite")
    center_mean = center_prior.get("center_offset_m")
    center_supported = center_prior.get("center_supported")
    if not isinstance(center_mean, torch.Tensor) or center_mean.shape != (batch, 3):
        raise ValueError("center prior mean shape differs")
    if (
        not isinstance(center_supported, torch.Tensor)
        or center_supported.shape != (batch,)
        or center_supported.dtype != torch.bool
    ):
        raise ValueError("center prior support shape/dtype differs")
    if (
        center_mean.device != history_obs_rel_m.device
        or center_mean.dtype != history_obs_rel_m.dtype
        or center_supported.device != history_obs_rel_m.device
    ):
        raise ValueError("center prior device/dtype differs")
    return batch, events, valid, omega_flat


def profiled_twist_dense_design(
    *,
    history_obs_rel_m: torch.Tensor,
    history_obs_mask: torch.Tensor,
    history_event_mask: torch.Tensor,
    history_time_s: torch.Tensor,
    omega_rad_s: torch.Tensor,
    center_prior: dict[str, torch.Tensor],
    use_q0_prior: bool,
    center_precision: float = 25.0,
    history_center_precision: float = 0.01,
    q_ridge: float = 1e-5,
    q0_endpoint_precision: float = 10.0,
    use_learned_center_variance: bool = True,
    reference_center_variance_xy: float = 0.15**2,
) -> dict[str, Any]:
    """Convert the V14 fixed-omega equations to padded scalar dense systems.

    XY parameter order is exactly ``[center_xy, velocity_xy, q_role_xy*4]``.
    Z parameter order is exactly ``[velocity_z, q_role_z*4]``.  Priors are
    returned as precision matrices and natural parameters, including V14's
    history center precision, q ridge, q0 endpoint precision, and optional
    learned center-variance scaling.
    """
    if min(
        center_precision, history_center_precision, q_ridge,
        q0_endpoint_precision, reference_center_variance_xy,
    ) <= 0:
        raise ValueError("V14 profile prior constants must be positive")
    if not isinstance(use_q0_prior, bool):
        raise ValueError("use_q0_prior must be boolean")
    batch, events, valid, omega = _validate_inputs(
        history_obs_rel_m, history_obs_mask, history_event_mask,
        history_time_s, omega_rad_s, center_prior,
    )
    dtype, device = history_obs_rel_m.dtype, history_obs_rel_m.device
    informed = center_prior["center_supported"] & use_q0_prior
    center_mean_all = center_prior["center_offset_m"]
    if bool(torch.any(informed & ~torch.isfinite(center_mean_all).all(dim=-1))):
        raise ValueError("supported center prior is non-finite")
    center_mean = torch.where(
        informed.unsqueeze(-1), center_mean_all,
        torch.zeros_like(center_mean_all),
    )
    center_precision_used = torch.where(
        informed,
        history_obs_rel_m.new_full((batch,), float(center_precision)),
        history_obs_rel_m.new_full((batch,), float(history_center_precision)),
    )
    center_log_variance = center_prior.get("center_log_variance_xy_z")
    if use_learned_center_variance and center_log_variance is not None:
        if (
            not isinstance(center_log_variance, torch.Tensor)
            or center_log_variance.shape != (batch, 2)
        ):
            raise ValueError("center log-variance shape differs")
        if (
            center_log_variance.device != device
            or center_log_variance.dtype != dtype
        ):
            raise ValueError("center log-variance device/dtype differs")
        if bool(torch.any(
            informed & ~torch.isfinite(center_log_variance).all(dim=-1)
        )):
            raise ValueError("supported center log-variance is non-finite")
        scale = (
            float(reference_center_variance_xy)
            * torch.exp(-center_log_variance[:, 0])
        ).clamp(
            float(history_center_precision) / float(center_precision), 40.0,
        )
        center_precision_used = torch.where(
            informed, center_precision_used * scale, center_precision_used,
        )
    q_reference_input = center_prior.get("q0_relation_m")
    endpoint_informed = informed.clone()
    if q_reference_input is None:
        q_reference = history_obs_rel_m.new_zeros((batch, ROLE_COUNT, 3))
        endpoint_informed = torch.zeros_like(informed)
    else:
        if (
            not isinstance(q_reference_input, torch.Tensor)
            or q_reference_input.shape != (batch, ROLE_COUNT, 3)
        ):
            raise ValueError("q0 endpoint-reference shape differs")
        if (
            q_reference_input.device != device
            or q_reference_input.dtype != dtype
        ):
            raise ValueError("q0 endpoint-reference device/dtype differs")
        if bool(torch.any(
            endpoint_informed.unsqueeze(-1).unsqueeze(-1)
            & ~torch.isfinite(q_reference_input)
        )):
            raise ValueError("supported q0 endpoint-reference is non-finite")
        q_reference = torch.where(
            endpoint_informed[:, None, None], q_reference_input,
            torch.zeros_like(q_reference_input),
        )
    endpoint_precision_used = torch.where(
        endpoint_informed,
        history_obs_rel_m.new_full((batch,), float(q0_endpoint_precision)),
        history_obs_rel_m.new_full((batch,), float(q_ridge)),
    )

    safe_time = torch.where(
        valid.any(dim=-1), history_time_s, torch.zeros_like(history_time_s),
    )
    theta = omega[:, None] * safe_time
    cosine, sine = torch.cos(theta), torch.sin(theta)
    rotation = torch.stack((
        cosine, -sine, sine, cosine,
    ), dim=-1).reshape(batch, events, 2, 2)
    eye = torch.eye(2, dtype=dtype, device=device).view(1, 1, 2, 2)
    xy_design = history_obs_rel_m.new_zeros(
        (batch, events, ROLE_COUNT, 2, XY_PARAMETER_COUNT),
    )
    xy_design[..., 0:2] = (eye - rotation).unsqueeze(2)
    xy_design[..., 2:4] = (
        safe_time[:, :, None, None] * eye
    ).unsqueeze(2)
    for role in range(ROLE_COUNT):
        xy_design[:, :, role, :, 4 + 2 * role:6 + 2 * role] = rotation
    xy_mask = valid.unsqueeze(-1).expand(-1, -1, -1, 2).reshape(batch, -1)
    xy_design = xy_design.reshape(batch, -1, XY_PARAMETER_COUNT)
    xy_design = torch.where(
        xy_mask.unsqueeze(-1), xy_design, torch.zeros_like(xy_design),
    )
    xy_target = history_obs_rel_m[..., :2].reshape(batch, -1)
    xy_target = torch.where(
        xy_mask, xy_target, torch.zeros_like(xy_target),
    )

    z_design = history_obs_rel_m.new_zeros(
        (batch, events, ROLE_COUNT, Z_PARAMETER_COUNT),
    )
    z_design[..., 0] = safe_time.unsqueeze(-1)
    for role in range(ROLE_COUNT):
        z_design[:, :, role, 1 + role] = 1.0
    z_mask = valid.reshape(batch, -1)
    z_design = z_design.reshape(batch, -1, Z_PARAMETER_COUNT)
    z_design = torch.where(
        z_mask.unsqueeze(-1), z_design, torch.zeros_like(z_design),
    )
    z_target = history_obs_rel_m[..., 2].reshape(batch, -1)
    z_target = torch.where(z_mask, z_target, torch.zeros_like(z_target))

    xy_prior_diagonal = history_obs_rel_m.new_zeros(
        (batch, XY_PARAMETER_COUNT),
    )
    xy_prior_diagonal[:, :2] = center_precision_used.unsqueeze(-1)
    xy_prior_diagonal[:, 4:] = endpoint_precision_used.unsqueeze(-1)
    xy_prior_natural = history_obs_rel_m.new_zeros(
        (batch, XY_PARAMETER_COUNT),
    )
    xy_prior_natural[:, :2] = center_precision_used.unsqueeze(-1) * center_mean[:, :2]
    xy_prior_natural[:, 4:] = (
        endpoint_precision_used[:, None, None] * q_reference[..., :2]
    ).reshape(batch, 8)

    z_prior_diagonal = history_obs_rel_m.new_zeros((batch, Z_PARAMETER_COUNT))
    z_prior_diagonal[:, 1:] = endpoint_precision_used.unsqueeze(-1)
    z_prior_natural = history_obs_rel_m.new_zeros((batch, Z_PARAMETER_COUNT))
    z_prior_natural[:, 1:] = endpoint_precision_used.unsqueeze(-1) * q_reference[..., 2]

    xy_velocity_mask = torch.zeros(
        XY_PARAMETER_COUNT, dtype=torch.bool, device=device,
    )
    xy_velocity_mask[2:4] = True
    z_velocity_mask = torch.zeros(
        Z_PARAMETER_COUNT, dtype=torch.bool, device=device,
    )
    z_velocity_mask[0] = True
    return {
        "xy": {
            "design": xy_design,
            "target": xy_target,
            "mask": xy_mask,
            "prior_precision": torch.diag_embed(xy_prior_diagonal),
            "prior_natural": xy_prior_natural,
            "velocity_column_mask": xy_velocity_mask,
        },
        "z": {
            "design": z_design,
            "target": z_target,
            "mask": z_mask,
            "prior_precision": torch.diag_embed(z_prior_diagonal),
            "prior_natural": z_prior_natural,
            "velocity_column_mask": z_velocity_mask,
        },
        "q0_prior_used": informed,
        "q0_endpoint_prior_used": endpoint_informed,
        "center_precision_used": center_precision_used,
        "endpoint_precision_used": endpoint_precision_used,
    }


__all__ = [
    "ROLE_COUNT", "XY_PARAMETER_COUNT", "Z_PARAMETER_COUNT",
    "profiled_twist_dense_design",
]
