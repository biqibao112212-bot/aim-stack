"""Differentiable locally weighted extension of the frozen V14 profiler.

The learned quantities are projected observation log-precisions and anonymous
anchor/center gates.  The network owns visible centering and clamping; this
solver only exponentiates its input.  Priors are interpolated in Gaussian
natural-parameter space.  Velocity is never regularized or clamped.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .profiled_center_twist_future import (
    ProfiledRigidTwistAtOmega,
    translation_only_fwl,
)


ROLE_COUNT = 4


def centered_visible_observation_precision(
    observation_log_precision: torch.Tensor,
    visible_mask: torch.Tensor,
) -> torch.Tensor:
    """Exponentiate visible-centered logits; invisible entries receive zero."""
    if observation_log_precision.shape != visible_mask.shape:
        raise ValueError("observation precision logit/mask shapes differ")
    if not torch.is_floating_point(observation_log_precision):
        raise ValueError("observation precision logits must be floating point")
    mask = visible_mask.to(torch.bool)
    if bool(torch.any(mask & ~torch.isfinite(observation_log_precision))):
        raise ValueError("visible observation precision logit is non-finite")
    weight = mask.to(observation_log_precision.dtype)
    count = weight.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    mean = (
        torch.where(mask, observation_log_precision, torch.zeros_like(
            observation_log_precision,
        )) * weight
    ).sum(dim=(1, 2), keepdim=True) / count
    centered = torch.where(
        mask, observation_log_precision - mean,
        torch.zeros_like(observation_log_precision),
    )
    precision = torch.exp(centered)
    return torch.where(mask, precision, torch.zeros_like(precision))


def visible_observation_precision(
    projected_log_precision: torch.Tensor,
    visible_mask: torch.Tensor,
) -> torch.Tensor:
    """Exponentiate an already centered/clamped visible log precision."""
    if projected_log_precision.shape != visible_mask.shape:
        raise ValueError("projected observation log precision/mask shapes differ")
    if not torch.is_floating_point(projected_log_precision):
        raise ValueError("projected observation log precision must be floating point")
    mask = visible_mask.to(torch.bool)
    if bool(torch.any(mask & ~torch.isfinite(projected_log_precision))):
        raise ValueError("visible projected observation log precision is non-finite")
    safe_log = torch.where(
        mask, projected_log_precision, torch.zeros_like(projected_log_precision),
    )
    precision = torch.exp(safe_log)
    if bool(torch.any(mask & ~torch.isfinite(precision))):
        raise ValueError("visible observation precision is non-finite")
    return torch.where(mask, precision, torch.zeros_like(precision))


class LocallyWeightedProfiledTwistAtOmega(nn.Module):
    """Batched FP32 Schur profiler with local observation/anchor confidence."""

    def __init__(
        self,
        *,
        center_precision: float = 25.0,
        history_center_precision: float = 0.01,
        q_ridge: float = 1e-5,
        q0_endpoint_precision: float = 10.0,
        use_learned_center_variance: bool = True,
        minimum_velocity_information_s2: float = 1e-4,
        maximum_velocity_condition: float = 1e6,
        minimum_time_span_s: float = 1e-3,
    ) -> None:
        super().__init__()
        if min(
            center_precision, history_center_precision, q_ridge,
            q0_endpoint_precision, minimum_velocity_information_s2,
            maximum_velocity_condition, minimum_time_span_s,
        ) <= 0:
            raise ValueError("profile precisions and support gates must be positive")
        if q0_endpoint_precision <= q_ridge:
            raise ValueError("q0 endpoint precision must exceed q ridge")
        self.center_precision = float(center_precision)
        self.history_center_precision = float(history_center_precision)
        self.q_ridge = float(q_ridge)
        self.q0_endpoint_precision = float(q0_endpoint_precision)
        self.use_learned_center_variance = bool(use_learned_center_variance)
        self.reference_center_variance_xy = 0.15**2
        self.minimum_velocity_information_s2 = float(
            minimum_velocity_information_s2,
        )
        self.maximum_velocity_condition = float(maximum_velocity_condition)
        self.minimum_time_span_s = float(minimum_time_span_s)
        self.reference = ProfiledRigidTwistAtOmega(
            center_precision=center_precision,
            history_center_precision=history_center_precision,
            q_ridge=q_ridge,
            q0_endpoint_precision=q0_endpoint_precision,
            use_learned_center_variance=use_learned_center_variance,
            minimum_velocity_information_s2=minimum_velocity_information_s2,
            maximum_velocity_condition=maximum_velocity_condition,
            minimum_time_span_s=minimum_time_span_s,
        )

    def _validate(
        self,
        observations: torch.Tensor,
        observation_mask: torch.Tensor,
        event_mask: torch.Tensor,
        time_s: torch.Tensor,
        omega: torch.Tensor,
        observation_log_precision: torch.Tensor,
        anchor_alpha: torch.Tensor,
        center_alpha: torch.Tensor,
    ) -> tuple[int, int, torch.Tensor]:
        if observations.ndim != 4 or observations.shape[2:] != (ROLE_COUNT, 3):
            raise ValueError("weighted profile observations must have shape [B,T,4,3]")
        batch, events = observations.shape[:2]
        if observation_mask.shape != (batch, events, ROLE_COUNT):
            raise ValueError("weighted profile observation mask differs")
        if event_mask.shape != (batch, events) or time_s.shape != (batch, events):
            raise ValueError("weighted profile event fields differ")
        if omega.shape not in {(batch,), (batch, 1)}:
            raise ValueError("weighted profile omega shape differs")
        if observation_log_precision.shape != (batch, events, ROLE_COUNT):
            raise ValueError("observation log precision must have shape [B,T,4]")
        if anchor_alpha.shape != (batch, ROLE_COUNT):
            raise ValueError("anchor alpha must have shape [B,4]")
        if center_alpha.shape != (batch,):
            raise ValueError("center alpha must have shape [B]")
        if not torch.is_floating_point(anchor_alpha) or bool(torch.any(
            ~torch.isfinite(anchor_alpha) | (anchor_alpha < 0) | (anchor_alpha > 1)
        )):
            raise ValueError("anchor alpha must be finite and inside [0,1]")
        if not torch.is_floating_point(center_alpha) or bool(torch.any(
            ~torch.isfinite(center_alpha) | (center_alpha < 0) | (center_alpha > 1)
        )):
            raise ValueError("center alpha must be finite and inside [0,1]")
        valid = observation_mask.to(torch.bool) & event_mask.to(torch.bool).unsqueeze(-1)
        if bool(torch.any(valid & ~torch.isfinite(observations).all(dim=-1))):
            raise ValueError("visible weighted-profile observation is non-finite")
        active_event = valid.any(dim=-1)
        if bool(torch.any(active_event & ~torch.isfinite(time_s))):
            raise ValueError("visible weighted-profile time is non-finite")
        if not bool(torch.isfinite(omega).all()):
            raise ValueError("weighted-profile omega is non-finite")
        return batch, events, valid

    @staticmethod
    def _time_span(time_s: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        positive = torch.full_like(time_s, torch.inf)
        negative = torch.full_like(time_s, -torch.inf)
        earliest = torch.where(active, time_s, positive).min(dim=1).values
        latest = torch.where(active, time_s, negative).max(dim=1).values
        return torch.where(
            active.sum(dim=1) >= 2, latest - earliest,
            torch.zeros_like(earliest),
        )

    def _natural_parameters(
        self,
        center_prior: dict[str, torch.Tensor],
        anchor_alpha: torch.Tensor,
        center_alpha: torch.Tensor,
        *,
        use_q0_prior: bool,
    ) -> dict[str, torch.Tensor]:
        batch = anchor_alpha.shape[0]
        supported = center_prior["center_supported"].to(torch.bool)
        center_mean = center_prior["center_offset_m"].to(torch.float32)
        if supported.shape != (batch,) or center_mean.shape != (batch, 3):
            raise ValueError("weighted profile center prior shapes differ")
        gate = anchor_alpha.to(torch.float32)
        center_gate = center_alpha.to(torch.float32)
        if not use_q0_prior:
            gate = torch.zeros_like(gate)
            center_gate = torch.zeros_like(center_gate)
        support_float = supported.to(gate.dtype)
        gate = gate * support_float[:, None]
        center_gate = center_gate * support_float
        q_reference = center_prior.get("q0_relation_m")
        if q_reference is None:
            if bool(torch.any(gate > 0)):
                raise ValueError("positive anchor alpha needs q0_relation_m")
            q_reference = center_mean.new_zeros((batch, ROLE_COUNT, 3))
        q_reference = q_reference.to(torch.float32)
        if q_reference.shape != (batch, ROLE_COUNT, 3):
            raise ValueError("weighted profile q0 reference shape differs")
        informed_precision = center_mean.new_full((batch,), self.center_precision)
        if self.use_learned_center_variance:
            log_variance = center_prior.get("center_log_variance_xy_z")
            if log_variance is not None:
                log_variance = log_variance.to(torch.float32)
                if log_variance.shape != (batch, 2):
                    raise ValueError("weighted center log variance shape differs")
                informed_precision = informed_precision * (
                    self.reference_center_variance_xy
                    * torch.exp(-log_variance[:, 0])
                ).clamp(
                    self.history_center_precision / self.center_precision,
                    40.0,
                )
        center_diagonal = (
            self.history_center_precision
            + center_gate * (informed_precision - self.history_center_precision)
        )
        center_rhs = (
            center_gate[:, None] * informed_precision[:, None] * center_mean[:, :2]
        )
        anchor_diagonal = (
            self.q_ridge
            + gate * (self.q0_endpoint_precision - self.q_ridge)
        )
        anchor_rhs = (
            gate[..., None] * self.q0_endpoint_precision * q_reference
        )
        return {
            "anchor_gate": gate, "center_gate": center_gate,
            "anchor_diagonal": anchor_diagonal,
            "anchor_rhs": anchor_rhs,
            "center_diagonal": center_diagonal,
            "center_rhs": center_rhs,
            "center_mean": center_mean, "q_reference": q_reference,
            "informed_center_precision": informed_precision,
        }

    def _xy_system(
        self,
        observations: torch.Tensor,
        precision: torch.Tensor,
        time_s: torch.Tensor,
        omega: torch.Tensor,
        natural: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        batch, events = time_s.shape
        eye = torch.eye(2, dtype=torch.float32, device=observations.device)
        theta = omega[:, None] * time_s
        cosine, sine = torch.cos(theta), torch.sin(theta)
        rotation = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
            batch, events, 2, 2,
        )
        center_design = (eye - rotation)[:, :, None].expand(-1, -1, ROLE_COUNT, -1, -1)
        velocity_design = (
            time_s[:, :, None, None, None] * eye.view(1, 1, 1, 2, 2)
        ).expand(-1, -1, ROLE_COUNT, -1, -1)
        role_eye = torch.eye(
            ROLE_COUNT, dtype=torch.float32, device=observations.device,
        )
        anchor_design = torch.einsum(
            "rh,btij->btrihj", role_eye, rotation,
        ).reshape(batch, events, ROLE_COUNT, 2, 8)
        design = torch.cat((center_design, velocity_design, anchor_design), dim=-1)
        matrix = design.reshape(batch, events * ROLE_COUNT * 2, 12)
        target = observations[..., :2].reshape(batch, events * ROLE_COUNT * 2)
        coordinate_precision = precision[..., None].expand(-1, -1, -1, 2).reshape(
            batch, events * ROLE_COUNT * 2,
        )
        weighted_matrix = matrix * coordinate_precision.unsqueeze(-1)
        velocity_matrix = matrix[:, :, 2:4]
        nuisance_matrix = torch.cat((matrix[:, :, :2], matrix[:, :, 4:]), dim=-1)
        weighted_velocity = velocity_matrix * coordinate_precision.unsqueeze(-1)
        weighted_nuisance = nuisance_matrix * coordinate_precision.unsqueeze(-1)
        h_xx = velocity_matrix.transpose(1, 2) @ weighted_velocity
        h_xz = velocity_matrix.transpose(1, 2) @ weighted_nuisance
        nuisance_normal = nuisance_matrix.transpose(1, 2) @ weighted_nuisance
        diagonal = torch.cat((
            natural["center_diagonal"][:, None].expand(-1, 2),
            natural["anchor_diagonal"].repeat_interleave(2, dim=-1),
        ), dim=-1)
        nuisance_normal = nuisance_normal + torch.diag_embed(diagonal)
        weighted_target = coordinate_precision * target
        r_x = velocity_matrix.transpose(1, 2) @ weighted_target.unsqueeze(-1)
        r_z = nuisance_matrix.transpose(1, 2) @ weighted_target.unsqueeze(-1)
        prior_rhs = torch.cat((
            natural["center_rhs"], natural["anchor_rhs"][..., :2].reshape(batch, 8),
        ), dim=-1)
        r_z = r_z.squeeze(-1) + prior_rhs
        projected_design, nuisance_info = torch.linalg.solve_ex(
            nuisance_normal, h_xz.transpose(1, 2),
        )
        projected_target, rhs_info = torch.linalg.solve_ex(
            nuisance_normal, r_z.unsqueeze(-1),
        )
        schur = h_xx - h_xz @ projected_design
        schur = 0.5 * (schur + schur.transpose(1, 2))
        eigenvalues = torch.linalg.eigvalsh(schur)
        minimum_information = eigenvalues[:, 0]
        condition = (
            eigenvalues[:, -1].clamp_min(0)
            / minimum_information.clamp_min(1e-20)
        )
        active = precision.gt(0).any(dim=-1)
        time_span = self._time_span(time_s, active)
        count = precision.gt(0).sum(dim=(1, 2))
        identifiable = (
            (count >= 3) & (nuisance_info == 0) & (rhs_info == 0)
            & torch.isfinite(eigenvalues).all(dim=-1)
            & (minimum_information >= self.minimum_velocity_information_s2)
            & (condition <= self.maximum_velocity_condition)
            & (time_span >= self.minimum_time_span_s)
        )
        safe_schur = torch.where(
            identifiable[:, None, None], schur,
            eye[None].expand(batch, -1, -1),
        )
        velocity, velocity_info = torch.linalg.solve_ex(
            safe_schur,
            r_x - h_xz @ projected_target,
        )
        velocity = velocity.squeeze(-1)
        residualized_velocity_design = (
            velocity_matrix - nuisance_matrix @ projected_design
        )
        leverage_solution, leverage_info = torch.linalg.solve_ex(
            safe_schur, residualized_velocity_design.transpose(1, 2),
        )
        leverage_solution = leverage_solution.transpose(1, 2)
        coordinate_leverage = coordinate_precision * (
            velocity_matrix * leverage_solution
        ).sum(dim=-1)
        leverage = coordinate_leverage.reshape(
            batch, events, ROLE_COUNT, 2,
        ).sum(dim=-1)
        leverage_valid = (
            identifiable & (leverage_info == 0)
            & torch.isfinite(leverage_solution).all(dim=(1, 2))
        )
        leverage = torch.where(
            leverage_valid[:, None, None], leverage, torch.zeros_like(leverage),
        )
        nuisance, final_info = torch.linalg.solve_ex(
            nuisance_normal,
            r_z.unsqueeze(-1) - h_xz.transpose(1, 2) @ velocity.unsqueeze(-1),
        )
        nuisance = nuisance.squeeze(-1)
        finite = (
            identifiable & (velocity_info == 0) & (final_info == 0)
            & (leverage_info == 0)
            & torch.isfinite(velocity).all(dim=-1)
            & torch.isfinite(nuisance).all(dim=-1)
        )
        solution = torch.cat((nuisance[:, :2], velocity, nuisance[:, 2:]), dim=-1)
        residual = target - (matrix @ solution.unsqueeze(-1)).squeeze(-1)
        residual_energy = (
            coordinate_precision * residual.square()
        ).sum(dim=-1)
        center = nuisance[:, :2]
        anchor = nuisance[:, 2:].reshape(batch, ROLE_COUNT, 2)
        gate = natural["anchor_gate"]
        center_gate = natural["center_gate"]
        center_prior_energy = (
            (1.0 - center_gate) * self.history_center_precision
            * center.square().sum(dim=-1)
            + center_gate * natural["informed_center_precision"]
            * (center - natural["center_mean"][:, :2]).square().sum(dim=-1)
        )
        anchor_prior_energy = (
            (1.0 - gate) * self.q_ridge * anchor.square().sum(dim=-1)
            + gate * self.q0_endpoint_precision
            * (anchor - natural["q_reference"][..., :2]).square().sum(dim=-1)
        ).sum(dim=-1)
        denominator = coordinate_precision.sum(dim=-1).clamp_min(1e-20)
        energy = (residual_energy + center_prior_energy + anchor_prior_energy) / denominator
        return {
            "velocity": velocity, "center": center, "anchor": anchor,
            "energy": energy, "supported": finite,
            "minimum_information": minimum_information,
            "condition": condition, "time_span": time_span,
            "residual": residual.reshape(batch, events, ROLE_COUNT, 2),
            "schur": schur, "leverage": leverage,
        }

    def _z_system(
        self,
        observations: torch.Tensor,
        precision: torch.Tensor,
        time_s: torch.Tensor,
        natural: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        batch, events = time_s.shape
        role_eye = torch.eye(
            ROLE_COUNT, dtype=torch.float32, device=observations.device,
        )
        anchor_design = role_eye.view(1, 1, ROLE_COUNT, ROLE_COUNT).expand(
            batch, events, -1, -1,
        )
        matrix = torch.cat((time_s[:, :, None, None].expand(-1, -1, ROLE_COUNT, 1), anchor_design), dim=-1)
        matrix = matrix.reshape(batch, events * ROLE_COUNT, 5)
        target = observations[..., 2].reshape(batch, events * ROLE_COUNT)
        weight = precision.reshape(batch, events * ROLE_COUNT)
        velocity_matrix = matrix[:, :, :1]
        nuisance_matrix = matrix[:, :, 1:]
        weighted_velocity = velocity_matrix * weight.unsqueeze(-1)
        weighted_nuisance = nuisance_matrix * weight.unsqueeze(-1)
        nuisance_normal = (
            nuisance_matrix.transpose(1, 2) @ weighted_nuisance
            + torch.diag_embed(natural["anchor_diagonal"])
        )
        h_xx = velocity_matrix.transpose(1, 2) @ weighted_velocity
        h_xz = velocity_matrix.transpose(1, 2) @ weighted_nuisance
        weighted_target = weight * target
        r_x = velocity_matrix.transpose(1, 2) @ weighted_target.unsqueeze(-1)
        r_z = (
            nuisance_matrix.transpose(1, 2) @ weighted_target.unsqueeze(-1)
        ).squeeze(-1) + natural["anchor_rhs"][..., 2]
        projected_design, nuisance_info = torch.linalg.solve_ex(
            nuisance_normal, h_xz.transpose(1, 2),
        )
        projected_target, rhs_info = torch.linalg.solve_ex(
            nuisance_normal, r_z.unsqueeze(-1),
        )
        information = (h_xx - h_xz @ projected_design).reshape(batch)
        active = precision.gt(0).any(dim=-1)
        time_span = self._time_span(time_s, active)
        count = precision.gt(0).sum(dim=(1, 2))
        identifiable = (
            (count >= 2) & (nuisance_info == 0) & (rhs_info == 0)
            & torch.isfinite(information)
            & (information >= self.minimum_velocity_information_s2)
            & (time_span >= self.minimum_time_span_s)
        )
        safe_information = torch.where(
            identifiable, information, torch.ones_like(information),
        )
        velocity = (
            r_x.squeeze(-1).squeeze(-1)
            - (h_xz @ projected_target).reshape(batch)
        ) / safe_information
        residualized_velocity_design = (
            velocity_matrix - nuisance_matrix @ projected_design
        ).squeeze(-1)
        leverage = (
            weight * velocity_matrix.squeeze(-1) * residualized_velocity_design
            / safe_information[:, None]
        ).reshape(batch, events, ROLE_COUNT)
        leverage = torch.where(
            identifiable[:, None, None], leverage, torch.zeros_like(leverage),
        )
        nuisance, final_info = torch.linalg.solve_ex(
            nuisance_normal,
            r_z.unsqueeze(-1) - h_xz.transpose(1, 2) * velocity[:, None, None],
        )
        nuisance = nuisance.squeeze(-1)
        finite = (
            identifiable & (final_info == 0) & torch.isfinite(velocity)
            & torch.isfinite(nuisance).all(dim=-1)
        )
        fitted = (
            matrix @ torch.cat((velocity[:, None], nuisance), dim=-1).unsqueeze(-1)
        ).squeeze(-1)
        residual = target - fitted
        gate = natural["anchor_gate"]
        prior = (
            (1.0 - gate) * self.q_ridge * nuisance.square()
            + gate * self.q0_endpoint_precision
            * (nuisance - natural["q_reference"][..., 2]).square()
        ).sum(dim=-1)
        energy = (
            (weight * residual.square()).sum(dim=-1) + prior
        ) / weight.sum(dim=-1).clamp_min(1e-20)
        return {
            "velocity": velocity, "anchor": nuisance, "energy": energy,
            "supported": finite, "information": information,
            "residual": residual.reshape(batch, events, ROLE_COUNT),
            "leverage": leverage,
        }

    @staticmethod
    def _fp32_prior_subset(
        center_prior: dict[str, Any],
        batch: int,
        indices: torch.Tensor,
    ) -> dict[str, Any]:
        subset: dict[str, Any] = {}
        for name, value in center_prior.items():
            if not isinstance(value, torch.Tensor):
                subset[name] = value
                continue
            selected = (
                value.index_select(0, indices)
                if value.ndim > 0 and value.shape[0] == batch
                else value
            )
            subset[name] = (
                selected.to(torch.float32)
                if torch.is_floating_point(selected) else selected
            )
        return subset

    def _boundary_delegates(
        self,
        observations: torch.Tensor,
        observation_mask: torch.Tensor,
        event_mask: torch.Tensor,
        time_s: torch.Tensor,
        omega: torch.Tensor,
        center_prior: dict[str, torch.Tensor],
        anchor_alpha: torch.Tensor,
        center_alpha: torch.Tensor,
        observation_log_precision: torch.Tensor,
        use_q0_prior: bool,
    ) -> list[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        prior_has_gradient = any(
            isinstance(value, torch.Tensor) and value.requires_grad
            for value in center_prior.values()
        )
        if (
            observations.requires_grad or time_s.requires_grad
            or omega.requires_grad or prior_has_gradient
            or observation_log_precision.requires_grad
            or anchor_alpha.requires_grad or center_alpha.requires_grad
        ):
            return []
        batch = observations.shape[0]
        visible = observation_mask.to(torch.bool) & event_mask.to(torch.bool).unsqueeze(-1)
        neutral = torch.where(
            visible, observation_log_precision == 0,
            torch.ones_like(visible),
        ).all(dim=(1, 2))
        all_zero = (anchor_alpha == 0).all(dim=-1)
        all_one = (anchor_alpha == 1).all(dim=-1)
        zero_endpoint = neutral & all_zero & (center_alpha == 0)
        one_endpoint = (
            neutral & all_one & (center_alpha == 1) & bool(use_q0_prior)
        )
        delegates: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
        for endpoint_mask, informed in (
            (zero_endpoint, False), (one_endpoint, True),
        ):
            indices = torch.nonzero(endpoint_mask, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            delegated = self.reference(
                observations.index_select(0, indices),
                observation_mask.index_select(0, indices),
                event_mask.index_select(0, indices),
                time_s.index_select(0, indices),
                omega.index_select(0, indices),
                self._fp32_prior_subset(center_prior, batch, indices),
                use_q0_prior=informed,
            )
            delegates.append((indices, delegated))
        return delegates

    def forward(
        self,
        history_obs_rel_m: torch.Tensor,
        history_obs_mask: torch.Tensor,
        history_event_mask: torch.Tensor,
        history_time_s: torch.Tensor,
        omega_rad_s: torch.Tensor,
        center_prior: dict[str, torch.Tensor],
        *,
        observation_log_precision: torch.Tensor,
        anchor_alpha: torch.Tensor,
        center_alpha: torch.Tensor,
        use_q0_prior: bool = True,
    ) -> dict[str, torch.Tensor]:
        batch, events, valid = self._validate(
            history_obs_rel_m, history_obs_mask, history_event_mask,
            history_time_s, omega_rad_s, observation_log_precision, anchor_alpha,
            center_alpha,
        )
        observations = torch.where(
            valid.unsqueeze(-1), history_obs_rel_m, torch.zeros_like(history_obs_rel_m),
        ).to(torch.float32)
        active_event = valid.any(dim=-1)
        time_s = torch.where(
            active_event, history_time_s, torch.zeros_like(history_time_s),
        ).to(torch.float32)
        omega = omega_rad_s.reshape(batch).to(torch.float32)
        log_precision = observation_log_precision.to(torch.float32)
        delegates = self._boundary_delegates(
            observations, valid, history_event_mask.to(torch.bool), time_s,
            omega, center_prior, anchor_alpha, center_alpha, log_precision,
            use_q0_prior,
        )
        precision = visible_observation_precision(log_precision, valid)
        natural = self._natural_parameters(
            center_prior, anchor_alpha, center_alpha,
            use_q0_prior=use_q0_prior,
        )
        fallback = translation_only_fwl(
            observations, valid, history_event_mask.to(torch.bool), time_s,
            minimum_time_span_s=self.minimum_time_span_s,
        )
        fallback_velocity_finite = torch.isfinite(
            fallback["velocity_mps"],
        ).all(dim=-1)
        fallback = dict(fallback)
        fallback["supported"] = fallback["supported"] & fallback_velocity_finite
        fallback["velocity_mps"] = torch.where(
            fallback_velocity_finite[:, None], fallback["velocity_mps"],
            torch.zeros_like(fallback["velocity_mps"]),
        )
        xy = self._xy_system(observations, precision, time_s, omega, natural)
        z = self._z_system(observations, precision, time_s, natural)
        supported = xy["supported"] & z["supported"]
        profile_velocity = torch.cat((xy["velocity"], z["velocity"][:, None]), dim=-1)
        velocity = torch.where(
            supported[:, None], profile_velocity, fallback["velocity_mps"],
        )
        center = torch.cat((
            xy["center"],
            natural["center_gate"][:, None] * natural["center_mean"][:, 2:3],
        ), dim=-1)
        center = torch.where(
            supported[:, None], center,
            natural["center_gate"][:, None] * natural["center_mean"],
        )
        residual = torch.cat((xy["residual"], z["residual"].unsqueeze(-1)), dim=-1)
        residual = torch.where(valid.unsqueeze(-1), residual, torch.zeros_like(residual))
        event_active = precision.gt(0).any(dim=-1)
        result = {
            "velocity_mps": velocity,
            "yaw_rate_rad_s": omega,
            "profiled_center_offset_m": center,
            "profile_energy": xy["energy"] + z["energy"],
            "profile_energy_xy": xy["energy"],
            "profile_energy_z": z["energy"],
            "profile_supported": supported,
            "q0_prior_used": natural["anchor_gate"].gt(0).any(dim=-1),
            "fallback_velocity_mps": fallback["velocity_mps"],
            "fallback_supported": fallback["supported"],
            "state_supported": supported | fallback["supported"],
            "velocity_information_min_eigenvalue_s2": xy["minimum_information"],
            "velocity_information_condition": xy["condition"],
            "vertical_velocity_information_s2": z["information"],
            "observation_precision": precision,
            "anchor_alpha": natural["anchor_gate"],
            "effective_anchor_precision": natural["anchor_diagonal"],
            "anchor_natural_rhs": natural["anchor_rhs"],
            "center_alpha": natural["center_gate"],
            "effective_center_precision_xy": natural["center_diagonal"],
            "center_natural_rhs_xy": natural["center_rhs"],
            "informed_center_precision_xy": natural[
                "informed_center_precision"
            ],
            "weighted_residual_m": residual,
            "weighted_residual_energy": (
                precision.unsqueeze(-1) * residual.square()
            ).sum(dim=(1, 2, 3)),
            "observation_precision_sum": precision.sum(dim=(1, 2)),
            "effective_observation_count": precision.gt(0).sum(dim=(1, 2)),
            "effective_event_count": event_active.sum(dim=1),
            "time_span_s": xy["time_span"],
            "velocity_schur_information_xy": xy["schur"],
            "velocity_leverage_xy": xy["leverage"],
            "velocity_leverage_z": z["leverage"],
            "velocity_leverage": xy["leverage"] + z["leverage"],
            "boundary_delegated": torch.zeros(
                batch, dtype=torch.bool, device=observations.device,
            ),
        }
        for indices, delegated in delegates:
            for name, value in delegated.items():
                if name not in result:
                    raise RuntimeError(f"V14 delegated an unknown field: {name}")
                result[name] = torch.index_copy(
                    result[name], 0, indices, value,
                )
            result["boundary_delegated"] = torch.index_fill(
                result["boundary_delegated"], 0, indices, True,
            )
        velocity_finite = torch.isfinite(result["velocity_mps"]).all(dim=-1)
        fallback_finite = torch.isfinite(
            result["fallback_velocity_mps"],
        ).all(dim=-1)
        result["velocity_mps"] = torch.where(
            velocity_finite[:, None], result["velocity_mps"],
            torch.zeros_like(result["velocity_mps"]),
        )
        result["fallback_velocity_mps"] = torch.where(
            fallback_finite[:, None], result["fallback_velocity_mps"],
            torch.zeros_like(result["fallback_velocity_mps"]),
        )
        result["fallback_supported"] = (
            result["fallback_supported"] & fallback_finite
        )
        result["profile_supported"] = (
            result["profile_supported"] & velocity_finite
        )
        result["velocity_mps"] = torch.where(
            velocity_finite[:, None], result["velocity_mps"],
            result["fallback_velocity_mps"],
        )
        final_velocity_finite = torch.isfinite(
            result["velocity_mps"],
        ).all(dim=-1)
        result["state_supported"] = (
            (result["profile_supported"] | result["fallback_supported"])
            & final_velocity_finite
        )
        return result


__all__ = [
    "LocallyWeightedProfiledTwistAtOmega",
    "centered_visible_observation_precision",
    "visible_observation_precision",
]
