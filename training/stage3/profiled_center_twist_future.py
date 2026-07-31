"""Anonymous center prior and profiled rigid-twist state mechanism.

This module estimates the *current* physical twist from causal relative
history.  It is deliberately separate from the learned future decoder.  No
physical armor identity, absolute range, future observation, or truth label is
accepted by its forward contract.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


CENTER_TWIST_FORWARD_FIELDS = (
    "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
    "history_event_mask", "history_time_s", "history_switch_step",
    "q0_relation_m", "q0_supported",
)


def _validate_center_inputs(
    q0_relation_m: torch.Tensor, q0_supported: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    if q0_relation_m.ndim != 3 or q0_relation_m.shape[1:] != (4, 3):
        raise ValueError("q0 relation must have shape [B,4,3]")
    if q0_supported.shape != q0_relation_m.shape[:2]:
        raise ValueError("q0 support shape differs")
    support = q0_supported.to(torch.bool)
    if bool(torch.any(support & ~torch.isfinite(q0_relation_m).all(dim=-1))):
        raise ValueError("supported q0 relation is non-finite")
    return q0_relation_m.shape[0], support


class AnonymousQ0CenterPrior(nn.Module):
    """O(2)-equivariant, permutation-invariant anonymous center prior.

    The zero-update carrier is the mean of all four finite H hypotheses;
    evidence support calibrates confidence and gates the all-unsupported case,
    but does not delete an inferred role.  Learning can only emit scalar
    coefficients on relative vectors, so there is no free XYZ vector,
    fixed-slot embedding, or physical-ID lookup.
    """

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.coefficient = nn.Sequential(
            nn.Linear(8, width), nn.SiLU(), nn.Linear(width, 1, bias=False),
        )
        nn.init.zeros_(self.coefficient[-1].weight)
        self.log_variance = nn.Sequential(
            nn.Linear(6, width), nn.SiLU(), nn.Linear(width, 2),
        )
        nn.init.zeros_(self.log_variance[-1].weight)
        with torch.no_grad():
            self.log_variance[-1].bias.copy_(torch.log(torch.tensor((0.15**2, 0.10**2))))

    def forward(
        self, q0_relation_m: torch.Tensor, q0_supported: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, support = _validate_center_inputs(q0_relation_m, q0_supported)
        if not bool(torch.isfinite(q0_relation_m).all()):
            raise ValueError("q0 relation hypotheses are non-finite")
        # H emits a finite hypothesis for every anonymous role even when a role
        # lacks direct evidence.  Validation shows those inferred hypotheses
        # improve the body distribution, so support calibrates uncertainty but
        # does not delete otherwise reasonable geometry.
        clean = q0_relation_m
        count = support.sum(dim=1)
        denominator = clean.new_full((batch, 1), 4.0)
        carrier = clean.mean(dim=1)
        supported = count >= 1

        norm2 = clean[..., :2].square().sum(dim=-1)
        carrier_norm2 = carrier[:, :2].square().sum(dim=-1, keepdim=True)
        dot_carrier = (
            clean[..., :2] * carrier[:, None, :2]
        ).sum(dim=-1, keepdim=True)
        support_fraction = (count.to(clean.dtype) / 4.0).view(batch, 1, 1)
        token = torch.cat((
            norm2.unsqueeze(-1), torch.sqrt(norm2.clamp_min(1e-12)).unsqueeze(-1),
            clean[..., 2:3], clean[..., 2:3].square(), dot_carrier,
            carrier_norm2[:, None].expand(-1, 4, -1),
            support_fraction.expand(-1, 4, -1),
            support.to(clean.dtype).unsqueeze(-1),
        ), dim=-1)
        coefficient = self.coefficient(token).squeeze(-1)
        correction = (
            coefficient.unsqueeze(-1) * clean
        ).sum(dim=1) / denominator
        mean = carrier + correction

        mean_point_norm2 = norm2.sum(dim=1, keepdim=True) / denominator
        centered = clean - carrier[:, None]
        planar_variance = centered[..., :2].square().sum(dim=-1).mean(
            dim=1, keepdim=True,
        )
        z_variance = centered[..., 2].square().mean(dim=1, keepdim=True)
        pooled = torch.cat((
            carrier_norm2, carrier[:, 2:3], mean_point_norm2,
            planar_variance, z_variance,
            count.to(clean.dtype).unsqueeze(-1) / 4.0,
        ), dim=-1)
        log_variance = self.log_variance(pooled).clamp(-9.0, 2.0)
        mean = torch.where(supported.unsqueeze(-1), mean, torch.zeros_like(mean))
        log_variance = torch.where(
            supported.unsqueeze(-1), log_variance,
            torch.full_like(log_variance, 2.0),
        )
        return {
            "center_offset_m": mean,
            "center_log_variance_xy_z": log_variance,
            "center_supported": supported,
            "center_support_count": count,
            "center_carrier_m": torch.where(
                supported.unsqueeze(-1), carrier, torch.zeros_like(carrier),
            ),
        }


def anonymous_center_prior_loss(
    prediction: dict[str, torch.Tensor], target_center_offset_m: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Gaussian center NLL on supported windows; labels are loss-only."""
    mean = prediction["center_offset_m"]
    if target_center_offset_m.shape != mean.shape:
        raise ValueError("center target shape differs")
    supported = prediction["center_supported"].to(torch.bool)
    if not bool(supported.any()):
        zero = mean.sum() * 0.0
        return zero, {"center_nll": zero, "center_l2_m": zero}
    error = mean - target_center_offset_m.detach()
    log_variance = prediction["center_log_variance_xy_z"]
    planar_square = error[:, :2].square().sum(dim=-1)
    vertical_square = error[:, 2].square()
    nll = 0.5 * (
        planar_square * torch.exp(-log_variance[:, 0])
        + 2.0 * log_variance[:, 0]
        + vertical_square * torch.exp(-log_variance[:, 1])
        + log_variance[:, 1]
    )
    center_nll = nll[supported].mean()
    center_l2 = torch.linalg.vector_norm(error, dim=-1)[supported].mean()
    return center_nll, {"center_nll": center_nll, "center_l2_m": center_l2}


def translation_only_fwl(
    history_obs_rel_m: torch.Tensor,
    history_obs_mask: torch.Tensor,
    history_event_mask: torch.Tensor,
    history_time_s: torch.Tensor,
    *,
    minimum_time_span_s: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """Shared-slope regression after eliminating one intercept per tracklet."""
    if minimum_time_span_s <= 0:
        raise ValueError("minimum fallback time span must be positive")
    if history_obs_rel_m.ndim != 4 or history_obs_rel_m.shape[2:] != (4, 3):
        raise ValueError("history observations must have shape [B,T,4,3]")
    batch, events = history_obs_rel_m.shape[:2]
    if history_obs_mask.shape != (batch, events, 4):
        raise ValueError("history observation mask shape differs")
    if history_event_mask.shape != (batch, events):
        raise ValueError("history event mask shape differs")
    if history_time_s.shape != (batch, events):
        raise ValueError("history time shape differs")
    valid = history_obs_mask.to(torch.bool) & history_event_mask.to(torch.bool).unsqueeze(-1)
    if bool(torch.any(valid & ~torch.isfinite(history_obs_rel_m).all(dim=-1))):
        raise ValueError("visible history contains non-finite observations")
    weight = valid.to(history_obs_rel_m.dtype)
    count = weight.sum(dim=1).clamp_min(1.0)
    time = history_time_s.unsqueeze(-1)
    mean_time = (time * weight).sum(dim=1) / count
    mean_position = (
        history_obs_rel_m * weight.unsqueeze(-1)
    ).sum(dim=1) / count.unsqueeze(-1)
    centered_time = time - mean_time[:, None]
    centered_position = history_obs_rel_m - mean_position[:, None]
    denominator = (centered_time.square() * weight).sum(dim=(1, 2))
    numerator = (
        centered_time.unsqueeze(-1) * centered_position * weight.unsqueeze(-1)
    ).sum(dim=(1, 2))
    event_valid = valid.any(dim=-1)
    positive_infinity = torch.full_like(history_time_s, torch.inf)
    negative_infinity = torch.full_like(history_time_s, -torch.inf)
    earliest_time = torch.where(
        event_valid, history_time_s, positive_infinity,
    ).min(dim=1).values
    latest_time = torch.where(
        event_valid, history_time_s, negative_infinity,
    ).max(dim=1).values
    time_span = torch.where(
        event_valid.sum(dim=1) >= 2,
        latest_time - earliest_time,
        torch.zeros_like(earliest_time),
    )
    supported = (denominator > 1e-8) & (time_span >= minimum_time_span_s)
    velocity = numerator / denominator.clamp_min(1e-8).unsqueeze(-1)
    velocity = torch.where(supported.unsqueeze(-1), velocity, torch.zeros_like(velocity))
    return {
        "velocity_mps": velocity,
        "supported": supported,
        "time_information": denominator,
        "time_span_s": time_span,
    }


class ProfiledRigidTwistAtOmega(nn.Module):
    """Profile center and anonymous tracklet endpoints at a supplied omega.

    This truth-omega-capable mechanism slice is used to isolate the center and
    velocity definition before a free omega posterior is trained.  Velocity is
    never regularized or clamped; all numerical priors act only on nuisance
    center/tracklet variables.
    """

    def __init__(
        self, *, center_precision: float = 25.0,
        history_center_precision: float = 0.01, q_ridge: float = 1e-5,
        q0_endpoint_precision: float = 10.0,
        use_learned_center_variance: bool = True,
        minimum_velocity_information_s2: float = 1e-4,
        maximum_velocity_condition: float = 1e6,
        minimum_time_span_s: float = 1e-3,
    ) -> None:
        super().__init__()
        if min(
            center_precision, history_center_precision, q_ridge,
            q0_endpoint_precision,
            minimum_velocity_information_s2, maximum_velocity_condition,
            minimum_time_span_s,
        ) <= 0:
            raise ValueError("profile nuisance precisions must be positive")
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

    @staticmethod
    def _xy_system(
        observations: torch.Tensor, valid: torch.Tensor, time_s: torch.Tensor,
        omega: torch.Tensor, center_mean: torch.Tensor,
        center_precision: torch.Tensor, q_reference: torch.Tensor,
        q_precision: torch.Tensor,
        minimum_velocity_information_s2: float,
        maximum_velocity_condition: float,
        minimum_time_span_s: float,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        """Schur-profile nuisance state and expose velocity identifiability."""
        device, dtype = observations.device, observations.dtype
        indices = torch.nonzero(valid, as_tuple=False)
        if indices.shape[0] < 3:
            zero2 = observations.new_zeros(2)
            return (
                zero2, zero2, observations.new_tensor(torch.inf),
                torch.tensor(False, device=device), observations.new_zeros(()),
                observations.new_tensor(torch.inf),
            )
        event_index, handle_index = indices[:, 0], indices[:, 1]
        t = time_s[event_index]
        theta = omega * t
        cosine, sine = torch.cos(theta), torch.sin(theta)
        rotation = torch.stack((
            cosine, -sine, sine, cosine,
        ), dim=-1).reshape(-1, 2, 2)
        eye = torch.eye(2, dtype=dtype, device=device).expand_as(rotation)
        count = indices.shape[0]
        design = observations.new_zeros((count, 2, 12))
        design[:, :, :2] = eye - rotation
        design[:, :, 2:4] = t[:, None, None] * eye
        for handle in range(4):
            selected = handle_index == handle
            if bool(selected.any()):
                design[selected, :, 4 + 2 * handle:6 + 2 * handle] = rotation[selected]
        matrix = design.reshape(-1, 12)
        target = observations[event_index, handle_index, :2].reshape(-1)
        velocity_design = matrix[:, 2:4]
        nuisance_design = torch.cat((matrix[:, :2], matrix[:, 4:]), dim=1)
        h_xx = velocity_design.T @ velocity_design
        h_xz = velocity_design.T @ nuisance_design
        nuisance_diagonal = observations.new_full((10,), q_precision)
        nuisance_diagonal[:2] = center_precision
        nuisance_normal = (
            nuisance_design.T @ nuisance_design + torch.diag(nuisance_diagonal)
        )
        r_x = velocity_design.T @ target
        r_z = nuisance_design.T @ target
        r_z = r_z + torch.cat((
            center_precision * center_mean[:2],
            q_precision * q_reference[:, :2].reshape(8),
        ))
        projected_design, nuisance_info = torch.linalg.solve_ex(
            nuisance_normal, h_xz.T,
        )
        projected_target, nuisance_rhs_info = torch.linalg.solve_ex(
            nuisance_normal, r_z,
        )
        schur = h_xx - h_xz @ projected_design
        schur = 0.5 * (schur + schur.T)
        eigenvalues = torch.linalg.eigvalsh(schur)
        minimum_information = eigenvalues[0]
        condition = eigenvalues[-1].clamp_min(0) / minimum_information.clamp_min(1e-20)
        time_span = t.max() - t.min()
        identifiable = (
            (nuisance_info == 0) & (nuisance_rhs_info == 0)
            & torch.isfinite(eigenvalues).all()
            & (minimum_information >= minimum_velocity_information_s2)
            & (condition <= maximum_velocity_condition)
            & (time_span >= minimum_time_span_s)
        )
        velocity, velocity_info = torch.linalg.solve_ex(
            schur, r_x - h_xz @ projected_target,
        )
        nuisance, final_nuisance_info = torch.linalg.solve_ex(
            nuisance_normal, r_z - h_xz.T @ velocity,
        )
        solution = torch.cat((
            nuisance[:2], velocity, nuisance[2:],
        ))
        finite = (
            identifiable & (velocity_info == 0) & (final_nuisance_info == 0)
            & torch.isfinite(solution).all()
        )
        residual = target - matrix @ solution
        prior = center_precision * (solution[:2] - center_mean[:2]).square().sum()
        prior = prior + q_precision * (
            solution[4:] - q_reference[:, :2].reshape(8)
        ).square().sum()
        energy = residual.square().mean() + prior / max(int(target.numel()), 1)
        return (
            solution[2:4], solution[:2], energy, finite,
            minimum_information, condition,
        )

    @staticmethod
    def _z_system(
        observations: torch.Tensor, valid: torch.Tensor, time_s: torch.Tensor,
        q_reference_z: torch.Tensor, q_precision: torch.Tensor,
        minimum_velocity_information_s2: float,
        minimum_time_span_s: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = observations.device
        indices = torch.nonzero(valid, as_tuple=False)
        if indices.shape[0] < 2:
            return (
                observations.new_zeros(()), torch.tensor(False, device=device),
                observations.new_zeros(()), observations.new_tensor(torch.inf),
            )
        event_index, handle_index = indices[:, 0], indices[:, 1]
        matrix = observations.new_zeros((indices.shape[0], 5))
        matrix[:, 0] = time_s[event_index]
        matrix[torch.arange(indices.shape[0], device=device), 1 + handle_index] = 1.0
        target = observations[event_index, handle_index, 2]
        velocity_design = matrix[:, :1]
        nuisance_design = matrix[:, 1:]
        nuisance_normal = (
            nuisance_design.T @ nuisance_design
            + torch.eye(4, dtype=matrix.dtype, device=device) * q_precision
        )
        h_xx = velocity_design.T @ velocity_design
        h_xz = velocity_design.T @ nuisance_design
        projected_design, nuisance_info = torch.linalg.solve_ex(
            nuisance_normal, h_xz.T,
        )
        projected_target, rhs_info = torch.linalg.solve_ex(
            nuisance_normal,
            nuisance_design.T @ target + q_precision * q_reference_z,
        )
        information = (h_xx - h_xz @ projected_design).reshape(())
        velocity = (
            velocity_design.T @ target - h_xz @ projected_target
        ).reshape(()) / information.clamp_min(1e-20)
        nuisance, final_info = torch.linalg.solve_ex(
            nuisance_normal,
            nuisance_design.T @ (target - velocity_design[:, 0] * velocity)
            + q_precision * q_reference_z,
        )
        time_span = time_s[event_index].max() - time_s[event_index].min()
        finite = (
            (nuisance_info == 0) & (rhs_info == 0) & (final_info == 0)
            & torch.isfinite(velocity) & torch.isfinite(nuisance).all()
            & (information >= minimum_velocity_information_s2)
            & (time_span >= minimum_time_span_s)
        )
        fitted = velocity_design[:, 0] * velocity + nuisance[handle_index]
        residual = target - fitted
        prior = q_precision * (nuisance - q_reference_z).square().sum()
        energy = residual.square().mean() + prior / max(int(target.numel()), 1)
        return velocity, finite, information, energy

    def forward(
        self,
        history_obs_rel_m: torch.Tensor,
        history_obs_mask: torch.Tensor,
        history_event_mask: torch.Tensor,
        history_time_s: torch.Tensor,
        omega_rad_s: torch.Tensor,
        center_prior: dict[str, torch.Tensor],
        *,
        use_q0_prior: bool = True,
    ) -> dict[str, torch.Tensor]:
        if history_obs_rel_m.ndim != 4 or history_obs_rel_m.shape[2:] != (4, 3):
            raise ValueError("profile history observations differ")
        batch, events = history_obs_rel_m.shape[:2]
        if omega_rad_s.shape not in {(batch,), (batch, 1)}:
            raise ValueError("profile omega shape differs")
        if history_obs_mask.shape != (batch, events, 4):
            raise ValueError("profile observation mask shape differs")
        if history_event_mask.shape != (batch, events) or history_time_s.shape != (batch, events):
            raise ValueError("profile event fields differ")
        valid = history_obs_mask.to(torch.bool) & history_event_mask.to(torch.bool).unsqueeze(-1)
        if bool(torch.any(valid & ~torch.isfinite(history_obs_rel_m).all(dim=-1))):
            raise ValueError("profile visible history is non-finite")
        omega_flat = omega_rad_s.reshape(batch)
        q0_supported = center_prior["center_supported"].to(torch.bool)
        center_mean_all = center_prior["center_offset_m"]
        if center_mean_all.shape != (batch, 3) or q0_supported.shape != (batch,):
            raise ValueError("profile center prior shapes differ")

        fallback = translation_only_fwl(
            history_obs_rel_m, history_obs_mask,
            history_event_mask, history_time_s,
            minimum_time_span_s=self.minimum_time_span_s,
        )
        velocities: list[torch.Tensor] = []
        centers: list[torch.Tensor] = []
        energies: list[torch.Tensor] = []
        energies_xy: list[torch.Tensor] = []
        energies_z: list[torch.Tensor] = []
        solved: list[torch.Tensor] = []
        q0_used: list[torch.Tensor] = []
        velocity_information: list[torch.Tensor] = []
        velocity_condition: list[torch.Tensor] = []
        vertical_information: list[torch.Tensor] = []
        for row in range(batch):
            informed = bool(use_q0_prior and q0_supported[row])
            center_mean = (
                center_mean_all[row] if informed
                else torch.zeros_like(center_mean_all[row])
            )
            precision = center_mean.new_tensor(
                self.center_precision if informed else self.history_center_precision,
            )
            if informed and self.use_learned_center_variance:
                log_variance = center_prior.get("center_log_variance_xy_z")
                if log_variance is not None:
                    if log_variance.shape != (batch, 2):
                        raise ValueError("profile center log-variance shape differs")
                    precision = precision * (
                        self.reference_center_variance_xy
                        * torch.exp(-log_variance[row, 0])
                    ).clamp(
                        self.history_center_precision / self.center_precision,
                        40.0,
                    )
            q_reference_all = center_prior.get("q0_relation_m")
            if informed and q_reference_all is not None:
                if q_reference_all.shape != (batch, 4, 3):
                    raise ValueError("profile q0 endpoint-reference shape differs")
                q_reference = q_reference_all[row]
                q_precision = center_mean.new_tensor(self.q0_endpoint_precision)
            else:
                q_reference = center_mean.new_zeros((4, 3))
                q_precision = center_mean.new_tensor(self.q_ridge)
            (
                velocity_xy, center_xy, energy, valid_xy,
                information_xy, condition_xy,
            ) = self._xy_system(
                history_obs_rel_m[row], valid[row], history_time_s[row],
                omega_flat[row], center_mean, precision,
                q_reference, q_precision,
                self.minimum_velocity_information_s2,
                self.maximum_velocity_condition, self.minimum_time_span_s,
            )
            velocity_z, valid_z, information_z, energy_z = self._z_system(
                history_obs_rel_m[row], valid[row], history_time_s[row],
                q_reference[:, 2], q_precision,
                self.minimum_velocity_information_s2, self.minimum_time_span_s,
            )
            valid_solution = valid_xy & valid_z
            profile_velocity = torch.cat((velocity_xy, velocity_z.reshape(1)))
            velocity = torch.where(
                valid_solution, profile_velocity, fallback["velocity_mps"][row],
            )
            center = torch.cat((center_xy, center_mean[2:3]))
            center = torch.where(valid_solution, center, center_mean)
            velocities.append(velocity)
            centers.append(center)
            energies.append(energy + energy_z)
            energies_xy.append(energy)
            energies_z.append(energy_z)
            solved.append(valid_solution)
            q0_used.append(torch.tensor(informed, device=history_obs_rel_m.device))
            velocity_information.append(information_xy)
            velocity_condition.append(condition_xy)
            vertical_information.append(information_z)
        profile_supported = torch.stack(solved)
        fallback_supported = fallback["supported"]
        return {
            "velocity_mps": torch.stack(velocities),
            "yaw_rate_rad_s": omega_flat,
            "profiled_center_offset_m": torch.stack(centers),
            "profile_energy": torch.stack(energies),
            "profile_energy_xy": torch.stack(energies_xy),
            "profile_energy_z": torch.stack(energies_z),
            "profile_supported": profile_supported,
            "q0_prior_used": torch.stack(q0_used),
            "fallback_velocity_mps": fallback["velocity_mps"],
            "fallback_supported": fallback_supported,
            "state_supported": profile_supported | fallback_supported,
            "velocity_information_min_eigenvalue_s2": torch.stack(
                velocity_information,
            ),
            "velocity_information_condition": torch.stack(velocity_condition),
            "vertical_velocity_information_s2": torch.stack(vertical_information),
        }


class CenterPriorProfiledTwistScreen(nn.Module):
    """Eight-field state-only screen; the learned future remains external."""

    forward_fields = CENTER_TWIST_FORWARD_FIELDS

    def __init__(
        self, *, width: int = 32, center_precision: float = 25.0,
    ) -> None:
        super().__init__()
        self.center_prior = AnonymousQ0CenterPrior(width=width)
        self.profile = ProfiledRigidTwistAtOmega(center_precision=center_precision)
        self.component_gate = nn.Sequential(
            nn.Linear(13, width), nn.SiLU(), nn.Linear(width, 1),
        )
        nn.init.zeros_(self.component_gate[-1].weight)
        nn.init.constant_(self.component_gate[-1].bias, 2.0)

    def estimate_center(
        self, q0_relation_m: torch.Tensor, q0_supported: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.center_prior(q0_relation_m, q0_supported)

    def forward_at_omega(
        self, batch: dict[str, torch.Tensor], omega_rad_s: torch.Tensor,
        *, use_q0_prior: bool = True,
    ) -> dict[str, torch.Tensor]:
        missing = set(self.forward_fields) - set(batch)
        if missing:
            raise ValueError(f"profile state forward fields missing: {sorted(missing)}")
        fields = {name: batch[name] for name in self.forward_fields}
        batch_size, events = fields["history_obs_rel_m"].shape[:2]
        if fields["history_primary_mask"].shape != (batch_size, events, 4):
            raise ValueError("history primary mask shape differs")
        if fields["history_switch_step"].shape != (batch_size, events):
            raise ValueError("history switch-step shape differs")
        active = fields["history_event_mask"].to(torch.bool)
        primary_count = fields["history_primary_mask"].to(torch.bool).sum(dim=-1)
        if bool(torch.any(active & (primary_count != 1))):
            raise ValueError("each active history event needs one anonymous primary")
        center = self.center_prior(fields["q0_relation_m"], fields["q0_supported"])
        profile_prior = {
            **center, "q0_relation_m": fields["q0_relation_m"],
        }
        informed = self.profile(
            fields["history_obs_rel_m"], fields["history_obs_mask"],
            fields["history_event_mask"], fields["history_time_s"],
            omega_rad_s, profile_prior, use_q0_prior=use_q0_prior,
        )
        if not use_q0_prior:
            return {**center, **informed}
        history = self.profile(
            fields["history_obs_rel_m"], fields["history_obs_mask"],
            fields["history_event_mask"], fields["history_time_s"],
            omega_rad_s, profile_prior, use_q0_prior=False,
        )
        q0_state_supported = informed["state_supported"]
        history_state_supported = history["state_supported"]
        q0_energy = torch.where(
            informed["profile_supported"], informed["profile_energy"],
            torch.full_like(informed["profile_energy"], 1e6),
        )
        history_energy = torch.where(
            history["profile_supported"], history["profile_energy"],
            torch.full_like(history["profile_energy"], 1e6),
        )
        log_variance = center["center_log_variance_xy_z"]
        support_fraction = center["center_support_count"].to(
            q0_energy.dtype,
        ).unsqueeze(-1) / 4.0
        energy_feature = torch.stack((
            informed["profile_energy_xy"], informed["profile_energy_z"],
            history["profile_energy_xy"], history["profile_energy_z"],
        ), dim=-1)
        information_feature = torch.stack((
            informed["velocity_information_min_eigenvalue_s2"],
            history["velocity_information_min_eigenvalue_s2"],
            informed["velocity_information_condition"].reciprocal(),
            history["velocity_information_condition"].reciprocal(),
            informed["vertical_velocity_information_s2"],
            history["vertical_velocity_information_s2"],
        ), dim=-1)
        gate_feature = torch.cat((
            support_fraction,
            log_variance,
            torch.log1p(1000.0 * energy_feature.clamp(0.0, 1e3)),
            torch.log1p(1000.0 * information_feature.clamp(0.0, 1e3)),
        ), dim=-1).detach()
        logit = self.component_gate(gate_feature).squeeze(-1)
        q0_weight = torch.sigmoid(logit)
        q0_weight = torch.where(
            q0_state_supported & ~history_state_supported,
            torch.ones_like(q0_weight), q0_weight,
        )
        q0_weight = torch.where(
            ~q0_state_supported & history_state_supported,
            torch.zeros_like(q0_weight), q0_weight,
        )
        q0_weight = torch.where(
            q0_state_supported | history_state_supported,
            q0_weight, torch.zeros_like(q0_weight),
        )
        weight = q0_weight.unsqueeze(-1)
        velocity = (
            weight * informed["velocity_mps"]
            + (1.0 - weight) * history["velocity_mps"]
        )
        profiled_center = (
            weight * informed["profiled_center_offset_m"]
            + (1.0 - weight) * history["profiled_center_offset_m"]
        )
        profile_supported = (
            informed["profile_supported"] | history["profile_supported"]
        )
        fallback_supported = (
            ~profile_supported
            & (informed["fallback_supported"] | history["fallback_supported"])
        )
        state = dict(informed)
        state.update({
            "velocity_mps": velocity,
            "profiled_center_offset_m": profiled_center,
            "profile_energy": (
                q0_weight * q0_energy + (1.0 - q0_weight) * history_energy
            ),
            "profile_supported": profile_supported,
            "fallback_supported": fallback_supported,
            "state_supported": q0_state_supported | history_state_supported,
            "q0_component_weight": q0_weight,
            "q0_component_logit": logit,
            "component_gate_feature": gate_feature,
            "q0_profile_energy": q0_energy,
            "history_profile_energy": history_energy,
            "q0_component_velocity_mps": informed["velocity_mps"],
            "history_component_velocity_mps": history["velocity_mps"],
            "q0_component_state_supported": q0_state_supported,
            "history_component_state_supported": history_state_supported,
        })
        return {**center, **state}


__all__ = [
    "CENTER_TWIST_FORWARD_FIELDS", "AnonymousQ0CenterPrior",
    "CenterPriorProfiledTwistScreen", "ProfiledRigidTwistAtOmega",
    "anonymous_center_prior_loss", "translation_only_fwl",
]
