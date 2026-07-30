"""Typed, equivariant alternating physical-state estimator for Stage3 F.

V12 tried to solve angular motion completely before translation.  Its visible
factor mean could still contain rotation, so the following omega stage was
asked to recover information already removed by the gauge.  This head instead
alternates strictly typed estimates:

    omega0 -> velocity0 -> omega1 -> velocity1

Every cross-stage state is detached.  Angular modules can only write yaw and
velocity modules can only write translation.  Omega is not constrained to be
a positive rescaling of one analytic carrier: even learned coefficients act
on several signed pseudoscalar proposals, so a zero or wrong-sign angle
carrier is recoverable.  Pair and handle evidence are precision-weighted
softly.  Velocity starts from exact weighted least squares and learns only
equivariant combinations of ramp-invariant residual vectors.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .factorized_common_relative_motion_future import apply_common_velocity_ramp
from .joint_rigid_flow_probe import _deterministic_probe_ramp
from .omega_first_ordered_closure_future import (
    AnonymousOmegaFirstOrderedClosureProbe,
    _MaskedOrderedGRU,
)
from .stable_motion_bottleneck_future import StableMotionBottleneckAnonymousFutureModel
from .observable_future_pnp_ab import state_dict_sha256


def _masked_mean(
    value: torch.Tensor, valid: torch.Tensor, dims: tuple[int, ...],
) -> torch.Tensor:
    weight = valid.unsqueeze(-1).to(value.dtype)
    return (value * weight).sum(dim=dims) / weight.sum(dim=dims).clamp_min(1)


def _three_vector_o2_invariants(value: torch.Tensor) -> torch.Tensor:
    """Nine scalar invariants for three XYZ vectors under planar O(2)."""
    if value.shape[-1] != 9:
        raise ValueError("three-vector invariant input differs")
    vector = value.reshape(*value.shape[:-1], 3, 3)
    planar = vector[..., :2]
    norm2 = planar.square().sum(dim=-1)
    z = vector[..., 2]
    dots = torch.stack((
        (planar[..., 0, :] * planar[..., 1, :]).sum(dim=-1),
        (planar[..., 0, :] * planar[..., 2, :]).sum(dim=-1),
        (planar[..., 1, :] * planar[..., 2, :]).sum(dim=-1),
    ), dim=-1)
    return torch.cat((norm2, z, dots), dim=-1)


def _four_vector_o2_invariants(value: torch.Tensor) -> torch.Tensor:
    """Twelve scalar invariants for four XYZ vectors under planar O(2)."""
    if value.shape[-1] != 12:
        raise ValueError("four-vector invariant input differs")
    vector = value.reshape(*value.shape[:-1], 4, 3)
    planar = vector[..., :2]
    norm2 = planar.square().sum(dim=-1)
    z = vector[..., 2]
    dots = torch.stack((
        (planar[..., 0, :] * planar[..., 1, :]).sum(dim=-1),
        (planar[..., 2, :] * planar[..., 3, :]).sum(dim=-1),
        (planar[..., 0, :] * planar[..., 2, :]).sum(dim=-1),
        (planar[..., 1, :] * planar[..., 3, :]).sum(dim=-1),
    ), dim=-1)
    return torch.cat((norm2, z, dots), dim=-1)


def _single_vector_o2_invariants(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] != 3:
        raise ValueError("single-vector invariant input differs")
    norm2 = value[..., :2].square().sum(dim=-1, keepdim=True)
    return torch.cat((norm2, torch.sqrt(norm2.clamp_min(1e-12)), value[..., 2:3]), dim=-1)


class _ZeroPreservingTripleEncoder(nn.Module):
    """Encode geometry-motion-time correspondence without a constant path."""

    def __init__(
        self, geometry_features: int, motion_features: int,
        time_features: int, width: int,
    ) -> None:
        super().__init__()
        self.geometry = nn.Linear(geometry_features, width, bias=False)
        self.motion = nn.Linear(motion_features, width, bias=False)
        self.time = nn.Linear(time_features, width, bias=False)
        self.output = nn.Sequential(
            nn.LayerNorm(width, elementwise_affine=False),
            nn.Linear(width, width, bias=False), nn.SiLU(),
        )

    def forward(
        self, geometry: torch.Tensor, motion: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        return self.output(
            F.silu(self.geometry(geometry))
            * F.silu(self.motion(motion))
            * F.silu(self.time(time))
        )


class _PerStreamOrderedSummary(nn.Module):
    """Run time recurrence before pooling anonymous handles or lag scales."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.sequence = _MaskedOrderedGRU(width)

    def forward(
        self, token: torch.Tensor, valid: torch.Tensor,
    ) -> torch.Tensor:
        if token.ndim != 5 or valid.shape != token.shape[:-1]:
            raise ValueError("alternating stream token shapes differ")
        batch, streams, events, scales, width = token.shape
        ordered = token.permute(0, 1, 3, 2, 4).reshape(
            batch * streams * scales, events, width,
        )
        ordered_valid = valid.permute(0, 1, 3, 2).reshape(
            batch * streams * scales, events,
        )
        hidden = self.sequence(ordered, ordered_valid).reshape(
            batch, streams, scales, width,
        )
        stream_valid = valid.any(dim=2)
        return _masked_mean(hidden, stream_valid, dims=(1, 2))


class _InvariantVelocityStage(nn.Module):
    """WLS carrier plus a ramp-invariant, O(2)-equivariant correction.

    The learned branch predicts scalar coefficients for residual vector bases;
    it never emits an unconstrained XYZ vector.  Therefore reflecting the
    observation coordinates reflects the velocity exactly.  A common velocity
    ramp changes only ``beta`` and leaves every learned input unchanged.
    """

    def __init__(
        self, width: int, *, history_scale_s: float,
        correction_scale: float = 0.20,
    ) -> None:
        super().__init__()
        self.encoder = _ZeroPreservingTripleEncoder(9, 3, 8, width)
        self.summary = _PerStreamOrderedSummary(width)
        self.coefficient = nn.Sequential(
            nn.Linear(width, width, bias=False), nn.SiLU(),
            nn.Linear(width, 3, bias=False),
        )
        nn.init.zeros_(self.coefficient[-1].weight)
        self.history_scale_s = float(history_scale_s)
        self.correction_scale = float(correction_scale)

    @staticmethod
    def _masked_vector_mean(
        value: torch.Tensor, valid: torch.Tensor,
    ) -> torch.Tensor:
        weight = valid.unsqueeze(-1).to(value.dtype)
        return (value * weight).sum(dim=(1, 2, 3)) / weight.sum(
            dim=(1, 2, 3),
        ).clamp_min(1)

    def forward(
        self,
        *,
        omega_normalized: torch.Tensor,
        yaw_scale_rad_s: float,
        relative_geometry: torch.Tensor,
        handle_time: torch.Tensor,
        handle_delta: torch.Tensor,
        elapsed_normalized: torch.Tensor,
        handle_valid: torch.Tensor,
        beta_to_velocity_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        elapsed_s = elapsed_normalized * self.history_scale_s
        theta = (
            omega_normalized[:, None, None, None, 0]
            * float(yaw_scale_rad_s) * elapsed_s
        )
        prior = relative_geometry[..., 3:6]
        cosine, sine = torch.cos(theta), torch.sin(theta)
        rotated = torch.stack((
            cosine * prior[..., 0] - sine * prior[..., 1],
            sine * prior[..., 0] + cosine * prior[..., 1],
            prior[..., 2],
        ), dim=-1)
        analytic_rotation = rotated - prior
        rotation = analytic_rotation
        de_rotated = handle_delta - analytic_rotation
        weight = handle_valid.unsqueeze(-1).to(de_rotated.dtype)
        h = elapsed_normalized.unsqueeze(-1)
        raw_denominator = (
            handle_valid.to(de_rotated.dtype) * elapsed_normalized.square()
        ).sum(dim=(1, 2, 3))
        supported = raw_denominator >= 1e-7
        denominator = raw_denominator.clamp_min(1e-7)
        beta = (
            (weight * h * de_rotated).sum(dim=(1, 2, 3))
            / denominator.unsqueeze(-1)
        )
        beta = torch.where(supported.unsqueeze(-1), beta, torch.zeros_like(beta))
        residual = de_rotated - beta[:, None, None, None] * h
        token = self.encoder(
            _three_vector_o2_invariants(relative_geometry),
            _single_vector_o2_invariants(residual), handle_time,
        )
        hidden = self.summary(token, handle_valid)
        safe_rate = torch.where(
            handle_valid.unsqueeze(-1),
            residual / h.clamp_min(1e-5), torch.zeros_like(residual),
        )
        current_time = handle_time[..., 1]
        elapsed_weight = elapsed_normalized
        bases = torch.stack((
            self._masked_vector_mean(safe_rate, handle_valid),
            self._masked_vector_mean(
                safe_rate * current_time.abs().unsqueeze(-1), handle_valid,
            ),
            self._masked_vector_mean(
                safe_rate * elapsed_weight.unsqueeze(-1), handle_valid,
            ),
        ), dim=1)
        coefficient = torch.tanh(self.coefficient(hidden))
        correction_beta = self.correction_scale * (
            bases * coefficient.unsqueeze(-1)
        ).sum(dim=1)
        correction_beta = torch.where(
            supported.unsqueeze(-1), correction_beta,
            torch.zeros_like(correction_beta),
        )
        velocity = (beta + correction_beta) * beta_to_velocity_state
        return {
            "velocity_normalized": velocity,
            "rotation_displacement_normalized": rotation,
            "wls_beta_normalized_rate": beta,
            "wls_denominator": raw_denominator,
            "velocity_supported": supported,
            "wls_residual_normalized": residual,
            "learned_correction_normalized_rate": correction_beta,
        }


class _EquivariantOmegaStage(nn.Module):
    """Learn a signed omega residual from handle and pair pseudoscalars.

    All neural inputs are reflection-even.  The only reflection-odd values are
    explicit signed proposal bundles, multiplied by unrestricted learned scalar
    coefficients.  This makes the output structurally antisymmetric while
    still allowing the network to reverse or replace an erroneous carrier.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.handle_encoder = _ZeroPreservingTripleEncoder(9, 9, 8, width)
        self.pair_encoder = _ZeroPreservingTripleEncoder(12, 6, 7, width)
        self.handle_summary = _PerStreamOrderedSummary(width)
        self.pair_summary = _PerStreamOrderedSummary(width)
        self.handle_coefficient = nn.Sequential(
            nn.Linear(width, width, bias=False), nn.SiLU(),
            nn.Linear(width, 3, bias=False),
        )
        self.pair_coefficient = nn.Sequential(
            nn.Linear(width, width, bias=False), nn.SiLU(),
            nn.Linear(width, 3, bias=False),
        )
        self.fusion_logit = nn.Sequential(
            nn.Linear(2 * width + 5, width, bias=False), nn.SiLU(),
            nn.Linear(width, 2, bias=False),
        )
        nn.init.zeros_(self.handle_coefficient[-1].weight)
        nn.init.zeros_(self.pair_coefficient[-1].weight)
        nn.init.zeros_(self.fusion_logit[-1].weight)

    @staticmethod
    def _robust_location(
        value: torch.Tensor, valid: torch.Tensor,
    ) -> torch.Tensor:
        if value.shape != valid.shape:
            raise ValueError("signed proposal and support shapes differ")
        dims = tuple(range(1, value.ndim))
        weight = valid.to(value.dtype)
        initial = (value * weight).sum(dim=dims) / weight.sum(
            dim=dims,
        ).clamp_min(1)
        broadcast = initial.reshape(initial.shape[0], *([1] * (value.ndim - 1)))
        residual = (value - broadcast).abs()
        robust = weight / (1.0 + (residual / 0.20).square())
        return (value * robust).sum(dim=dims) / robust.sum(
            dim=dims,
        ).clamp_min(1)

    @classmethod
    def _robust_bundle(
        cls, value: torch.Tensor, valid: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack([
            cls._robust_location(value[..., index], valid[..., index])
            for index in range(value.shape[-1])
        ], dim=-1)

    @staticmethod
    def _evidence_stats(
        value: torch.Tensor, valid: torch.Tensor, carrier: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dims = tuple(range(1, value.ndim))
        weight = valid.to(value.dtype)
        count = weight.sum(dim=dims)
        center = carrier.reshape(carrier.shape[0], *([1] * (value.ndim - 1)))
        variance = (
            weight * (value - center).square()
        ).sum(dim=dims) / count.clamp_min(1)
        return count, variance

    def forward(
        self,
        *,
        handle_geometry_even: torch.Tensor,
        handle_motion_even: torch.Tensor,
        handle_time: torch.Tensor,
        handle_token_valid: torch.Tensor,
        handle_bundle: torch.Tensor,
        handle_bundle_valid: torch.Tensor,
        pair_geometry_even: torch.Tensor,
        pair_motion_even: torch.Tensor,
        pair_time: torch.Tensor,
        pair_token_valid: torch.Tensor,
        pair_bundle: torch.Tensor,
        pair_bundle_valid: torch.Tensor,
        base_omega_normalized: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        handle_hidden = self.handle_summary(self.handle_encoder(
            handle_geometry_even, handle_motion_even, handle_time,
        ), handle_token_valid)
        pair_hidden = self.pair_summary(self.pair_encoder(
            pair_geometry_even, pair_motion_even, pair_time,
        ), pair_token_valid)

        handle_carrier = self._robust_location(
            handle_bundle[..., 0], handle_bundle_valid[..., 0],
        )
        pair_carrier = self._robust_location(
            pair_bundle[..., 0], pair_bundle_valid[..., 0],
        )
        base = base_omega_normalized.squeeze(-1)
        handle_residual_bundle = handle_bundle - base[:, None, None, None, None]
        pair_residual_bundle = pair_bundle - base[:, None, None, None, None]
        handle_residual_bundle = torch.where(
            handle_bundle_valid, handle_residual_bundle,
            torch.zeros_like(handle_residual_bundle),
        )
        pair_residual_bundle = torch.where(
            pair_bundle_valid, pair_residual_bundle,
            torch.zeros_like(pair_residual_bundle),
        )
        handle_delta = handle_carrier - base
        pair_delta = pair_carrier - base
        handle_delta = handle_delta + (
            self.handle_coefficient(handle_hidden)
            * self._robust_bundle(
                handle_residual_bundle, handle_bundle_valid,
            )
        ).sum(dim=-1)
        pair_delta = pair_delta + (
            self.pair_coefficient(pair_hidden)
            * self._robust_bundle(pair_residual_bundle, pair_bundle_valid)
        ).sum(dim=-1)

        handle_supported = handle_bundle_valid.any(dim=(1, 2, 3, 4))
        pair_supported = pair_bundle_valid.any(dim=(1, 2, 3, 4))
        handle_count, handle_variance = self._evidence_stats(
            handle_bundle[..., 0], handle_bundle_valid[..., 0], handle_carrier,
        )
        pair_count, pair_variance = self._evidence_stats(
            pair_bundle[..., 0], pair_bundle_valid[..., 0], pair_carrier,
        )
        even_fusion = torch.cat((
            handle_hidden, pair_hidden, base.abs().unsqueeze(-1),
            torch.log1p(handle_count).unsqueeze(-1),
            torch.log1p(pair_count).unsqueeze(-1),
            torch.log1p(handle_variance / 0.01).unsqueeze(-1),
            torch.log1p(pair_variance / 0.01).unsqueeze(-1),
        ), dim=-1)
        learned_logit = self.fusion_logit(even_fusion)
        evidence_logit = torch.stack((
            torch.log1p(handle_count) - torch.log1p(handle_variance / 0.01),
            torch.log1p(pair_count) - torch.log1p(pair_variance / 0.01),
        ), dim=-1)
        supported = torch.stack((handle_supported, pair_supported), dim=-1)
        any_supported = supported.any(dim=-1, keepdim=True)
        logits = torch.where(
            supported, evidence_logit + learned_logit,
            torch.full_like(learned_logit, -torch.inf),
        )
        safe_logits = torch.where(any_supported, logits, torch.zeros_like(logits))
        weights = torch.softmax(safe_logits, dim=-1)
        weights = torch.where(supported, weights, torch.zeros_like(weights))
        weights = torch.where(
            any_supported,
            weights / weights.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(weights.dtype).tiny,
            ),
            torch.zeros_like(weights),
        )
        delta = (
            weights * torch.stack((handle_delta, pair_delta), dim=-1)
        ).sum(dim=-1)
        omega_supported = supported.any(dim=-1)
        omega = torch.where(
            omega_supported, base + delta, torch.zeros_like(base),
        )
        return {
            "omega_normalized": omega.unsqueeze(-1),
            "omega_delta_normalized": delta.unsqueeze(-1),
            "carrier_normalized": (
                weights * torch.stack((handle_carrier, pair_carrier), dim=-1)
            ).sum(dim=-1, keepdim=True),
            "handle_carrier_normalized": handle_carrier.unsqueeze(-1),
            "pair_carrier_normalized": pair_carrier.unsqueeze(-1),
            "handle_supported": handle_supported,
            "pair_supported": pair_supported,
            "omega_supported": omega_supported,
            "evidence_weight": weights,
        }


class EquivariantAlternatingTwistHead(nn.Module):
    """Two typed omega/velocity stages with detached cross-stage state."""

    angular_refinement_steps = 2
    velocity_refinement_steps = 2

    def __init__(
        self,
        channels: int,
        dropout: float,
        *,
        history_scale_s: float,
        position_scale_m: float,
        velocity_scale_mps: tuple[float, float, float],
        yaw_rate_scale_rad_s: float,
        max_abs_yaw_rate_rad_s: float,
    ) -> None:
        super().__init__()
        del dropout
        if history_scale_s <= 0 or position_scale_m <= 0:
            raise ValueError("alternating physical scales must be positive")
        if yaw_rate_scale_rad_s <= 0 or min(velocity_scale_mps) <= 0:
            raise ValueError("alternating state scales must be positive")
        self.channels = int(channels)
        self.width = 5 * self.channels // 3
        if self.width < 48:
            raise ValueError("alternating hidden width is too small")
        self.history_scale_s = float(history_scale_s)
        self.position_scale_m = float(position_scale_m)
        self.yaw_rate_scale_rad_s = float(yaw_rate_scale_rad_s)
        self.max_abs_yaw_rate_rad_s = float(max_abs_yaw_rate_rad_s)
        self.max_factor_elapsed_s = 0.105
        if (
            self.max_abs_yaw_rate_rad_s <= 0
            or self.max_abs_yaw_rate_rad_s * self.max_factor_elapsed_s >= torch.pi
        ):
            raise ValueError(
                "alternating physical yaw envelope aliases inside the fixed lag envelope"
            )
        beta_to_state = [
            self.position_scale_m / self.history_scale_s / float(value)
            for value in velocity_scale_mps
        ]
        self.register_buffer(
            "beta_to_velocity_state",
            torch.tensor(beta_to_state, dtype=torch.float32),
        )

        self.omega0 = _EquivariantOmegaStage(self.width)
        self.velocity0 = _InvariantVelocityStage(
            self.width, history_scale_s=self.history_scale_s,
        )
        self.omega1 = _EquivariantOmegaStage(self.width)
        self.velocity1 = _InvariantVelocityStage(
            self.width, history_scale_s=self.history_scale_s,
        )

    @staticmethod
    def _reshape_factors(
        handle_geometry: torch.Tensor,
        handle_kinematics: torch.Tensor,
        handle_valid: torch.Tensor,
        pair_geometry: torch.Tensor,
        pair_kinematics: torch.Tensor,
        pair_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        scales = 3
        if pair_geometry.shape[1] % scales:
            raise ValueError("alternating pair factor count differs")
        events = pair_geometry.shape[1] // scales
        if handle_geometry.shape[1] != 4 * events * scales:
            raise ValueError("alternating handle factor count differs")
        return (
            handle_geometry.reshape(-1, 4, events, scales, 12),
            handle_kinematics.reshape(-1, 4, events, scales, 14),
            handle_valid.reshape(-1, 4, events, scales),
            pair_geometry.reshape(-1, 1, events, scales, 12),
            pair_kinematics.reshape(-1, 1, events, scales, 13),
            pair_valid.reshape(-1, 1, events, scales),
        )

    def _relative_factors_with_gauge(
        self,
        handle_geometry: torch.Tensor,
        handle_kinematics: torch.Tensor,
        handle_valid: torch.Tensor,
        gauge_rate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = handle_valid.unsqueeze(-1).to(handle_kinematics.dtype)
        elapsed_s = 0.01 * torch.expm1(handle_kinematics[..., 6]).clamp_min(0)
        elapsed_normalized = elapsed_s / self.history_scale_s
        current_time = handle_kinematics[..., 7]
        prior_time = current_time - elapsed_normalized
        current = handle_geometry[..., 6:9] - (
            gauge_rate[:, None, None, None] * current_time.unsqueeze(-1)
        )
        prior = handle_geometry[..., 9:12] - (
            gauge_rate[:, None, None, None] * prior_time.unsqueeze(-1)
        )
        center = ((current + prior) * 0.5 * weight).sum(
            dim=(1, 2, 3),
        ) / weight.sum(dim=(1, 2, 3)).clamp_min(1)
        current_relative = current - center[:, None, None, None]
        prior_relative = prior - center[:, None, None, None]
        relative_geometry = torch.cat((
            current_relative, prior_relative,
            current_relative - prior_relative,
        ), dim=-1)
        relative_delta = handle_kinematics[..., :3] - (
            gauge_rate[:, None, None, None]
            * elapsed_normalized.unsqueeze(-1)
        )
        return relative_geometry, relative_delta, elapsed_normalized

    def _derived_relative_factors(
        self,
        handle_geometry: torch.Tensor,
        handle_kinematics: torch.Tensor,
        handle_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = handle_valid.unsqueeze(-1).to(handle_kinematics.dtype)
        gauge_rate = (
            handle_kinematics[..., 3:6] * weight
        ).sum(dim=(1, 2, 3)) / weight.sum(dim=(1, 2, 3)).clamp_min(1)
        geometry, delta, elapsed = self._relative_factors_with_gauge(
            handle_geometry, handle_kinematics, handle_valid, gauge_rate,
        )
        return geometry, delta, gauge_rate, elapsed

    @staticmethod
    def _signed_angle_proposal(
        prior: torch.Tensor,
        current: torch.Tensor,
        elapsed_s: torch.Tensor,
        valid: torch.Tensor,
        yaw_scale_rad_s: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prior_xy, current_xy = prior[..., :2], current[..., :2]
        prior_norm = torch.linalg.vector_norm(prior_xy, dim=-1)
        current_norm = torch.linalg.vector_norm(current_xy, dim=-1)
        supported = (
            valid & (prior_norm > 1e-4) & (current_norm > 1e-4)
            & (elapsed_s > 1e-5)
        )
        denominator = (prior_norm * current_norm).clamp_min(1e-8)
        cosine = (prior_xy * current_xy).sum(dim=-1) / denominator
        sine = (
            prior_xy[..., 0] * current_xy[..., 1]
            - prior_xy[..., 1] * current_xy[..., 0]
        ) / denominator
        angle = torch.atan2(sine, cosine.clamp(-1.0, 1.0))
        proposal = angle / elapsed_s.clamp_min(1e-5) / float(yaw_scale_rad_s)
        proposal = proposal.clamp(-1.5, 1.5)
        return torch.where(supported, proposal, torch.zeros_like(proposal)), supported

    @staticmethod
    def _robust_carrier(
        proposal: torch.Tensor, valid: torch.Tensor,
    ) -> torch.Tensor:
        weight = valid.to(proposal.dtype)
        initial = (proposal * weight).sum(dim=(1, 2, 3)) / weight.sum(
            dim=(1, 2, 3),
        ).clamp_min(1)
        residual = (proposal - initial[:, None, None, None]).abs()
        robust = weight / (1.0 + (residual / 0.20).square())
        return (proposal * robust).sum(dim=(1, 2, 3)) / robust.sum(
            dim=(1, 2, 3),
        ).clamp_min(1)

    @staticmethod
    def _handle_acceleration(
        rate: torch.Tensor,
        current_time: torch.Tensor,
        elapsed_normalized: torch.Tensor,
        valid: torch.Tensor,
        no_switch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        midpoint_time = current_time - 0.5 * elapsed_normalized
        dt = midpoint_time[:, :, 1:] - midpoint_time[:, :, :-1]
        supported = (
            valid[:, :, 1:] & valid[:, :, :-1]
            & no_switch[:, :, 1:] & no_switch[:, :, :-1]
            & (dt > 1e-5)
        )
        acceleration = (
            rate[:, :, 1:] - rate[:, :, :-1]
        ) / dt.clamp_min(1e-5).unsqueeze(-1)
        acceleration = torch.where(
            supported.unsqueeze(-1), acceleration,
            torch.zeros_like(acceleration),
        )
        padded = F.pad(acceleration, (0, 0, 0, 0, 1, 0))
        padded_valid = F.pad(supported, (0, 0, 1, 0), value=False)
        return padded, padded_valid

    @staticmethod
    def _cross_xy(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]

    @classmethod
    def _sine_rate_proposal(
        cls,
        prior: torch.Tensor,
        current: torch.Tensor,
        elapsed_s: torch.Tensor,
        valid: torch.Tensor,
        yaw_scale_rad_s: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prior_norm = torch.linalg.vector_norm(prior[..., :2], dim=-1)
        current_norm = torch.linalg.vector_norm(current[..., :2], dim=-1)
        supported = (
            valid & (prior_norm > 1e-4) & (current_norm > 1e-4)
            & (elapsed_s > 1e-5)
        )
        sine = cls._cross_xy(prior, current) / (
            prior_norm * current_norm
        ).clamp_min(1e-8)
        proposal = sine / elapsed_s.clamp_min(1e-5) / float(yaw_scale_rad_s)
        proposal = proposal.clamp(-1.5, 1.5)
        return torch.where(supported, proposal, torch.zeros_like(proposal)), supported

    def _omega_stage(
        self,
        *,
        prefix: str,
        relative_geometry: torch.Tensor,
        relative_delta: torch.Tensor,
        elapsed_normalized: torch.Tensor,
        handle_kinematics: torch.Tensor,
        handle_valid: torch.Tensor,
        pair_geometry: torch.Tensor,
        pair_kinematics: torch.Tensor,
        pair_valid: torch.Tensor,
        base_omega_normalized: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        handle_time = handle_kinematics[..., 6:14]
        pair_time = pair_kinematics[..., 6:13]
        elapsed_s = elapsed_normalized * self.history_scale_s
        acceleration, acceleration_valid = self._handle_acceleration(
            handle_kinematics[..., 3:6], handle_kinematics[..., 7],
            elapsed_normalized, handle_valid,
            handle_kinematics[..., 8].abs() < 1e-6,
        )
        stream_supported = (
            acceleration_valid.any(dim=2)
            & (handle_valid.sum(dim=2) >= 2)
        )
        supported_handle_edge = handle_valid & stream_supported.unsqueeze(2)
        handle_proposal, handle_proposal_valid = self._signed_angle_proposal(
            relative_geometry[..., 3:6], relative_geometry[..., :3],
            elapsed_s, supported_handle_edge, self.yaw_rate_scale_rad_s,
        )
        relative_rate = relative_delta / elapsed_normalized.clamp_min(
            1e-5,
        ).unsqueeze(-1)
        midpoint = 0.5 * (
            relative_geometry[..., :3] + relative_geometry[..., 3:6]
        )
        midpoint_norm2 = midpoint[..., :2].square().sum(dim=-1)
        orbital_valid = (
            supported_handle_edge & (midpoint_norm2 > 1e-6)
            & (elapsed_normalized > 1e-5)
        )
        orbital = self._cross_xy(midpoint, relative_rate) / (
            midpoint_norm2.clamp_min(1e-6)
            * self.history_scale_s * self.yaw_rate_scale_rad_s
        )
        orbital = torch.where(
            orbital_valid, orbital.clamp(-1.5, 1.5),
            torch.zeros_like(orbital),
        )
        rate_norm2 = relative_rate[..., :2].square().sum(dim=-1)
        curvature_valid = acceleration_valid & (rate_norm2 > 1e-6)
        curvature = self._cross_xy(relative_rate, acceleration) / (
            rate_norm2.clamp_min(1e-6)
            * self.history_scale_s * self.yaw_rate_scale_rad_s
        )
        curvature = torch.where(
            curvature_valid, curvature.clamp(-1.5, 1.5),
            torch.zeros_like(curvature),
        )
        handle_bundle = torch.stack((
            handle_proposal, orbital, curvature,
        ), dim=-1)
        handle_bundle_valid = torch.stack((
            handle_proposal_valid, orbital_valid, curvature_valid,
        ), dim=-1)

        swap_sign = torch.where(
            pair_kinematics[..., 11] > 0.5, -1.0, 1.0,
        )
        pair_current_canonical = pair_geometry[..., :3] * swap_sign.unsqueeze(-1)
        pair_elapsed_s = 0.01 * torch.expm1(
            pair_kinematics[..., 8]
        ).clamp_min(0)
        pair_proposal, pair_proposal_valid = self._signed_angle_proposal(
            pair_geometry[..., 3:6], pair_current_canonical,
            pair_elapsed_s, pair_valid, self.yaw_rate_scale_rad_s,
        )
        pair_sine, pair_sine_valid = self._sine_rate_proposal(
            pair_geometry[..., 3:6], pair_current_canonical,
            pair_elapsed_s, pair_valid, self.yaw_rate_scale_rad_s,
        )
        pair_prior_norm2 = pair_geometry[..., 3:5].square().sum(dim=-1)
        pair_chord_valid = (
            pair_valid & (pair_prior_norm2 > 1e-6) & (pair_elapsed_s > 1e-5)
        )
        pair_chord = self._cross_xy(
            pair_geometry[..., 3:6],
            pair_current_canonical - pair_geometry[..., 3:6],
        ) / (
            pair_prior_norm2.clamp_min(1e-6)
            * pair_elapsed_s.clamp_min(1e-5) * self.yaw_rate_scale_rad_s
        )
        pair_chord = torch.where(
            pair_chord_valid, pair_chord.clamp(-1.5, 1.5),
            torch.zeros_like(pair_chord),
        )
        pair_bundle = torch.stack((
            pair_proposal, pair_sine, pair_chord,
        ), dim=-1)
        pair_bundle_valid = torch.stack((
            pair_proposal_valid, pair_sine_valid, pair_chord_valid,
        ), dim=-1)

        delta_planar = relative_delta[..., :2]
        acceleration_planar = acceleration[..., :2]
        handle_cross = self._cross_xy(relative_delta, acceleration)
        handle_motion = torch.cat((
            delta_planar.square().sum(dim=-1, keepdim=True),
            acceleration_planar.square().sum(dim=-1, keepdim=True),
            relative_delta[..., 2:3], acceleration[..., 2:3],
            (delta_planar * acceleration_planar).sum(dim=-1, keepdim=True),
            handle_cross.square().unsqueeze(-1), handle_bundle.abs(),
        ), dim=-1)
        handle_motion = torch.where(
            handle_bundle_valid.any(dim=-1).unsqueeze(-1), handle_motion,
            torch.zeros_like(handle_motion),
        )
        pair_delta = pair_kinematics[..., :3]
        pair_delta_norm2 = pair_delta[..., :2].square().sum(
            dim=-1, keepdim=True,
        )
        pair_motion = torch.cat((
            pair_delta_norm2, torch.sqrt(pair_delta_norm2.clamp_min(1e-12)),
            pair_delta[..., 2:3], pair_bundle.abs(),
        ), dim=-1)
        result = getattr(self, prefix)(
            handle_geometry_even=_three_vector_o2_invariants(relative_geometry),
            handle_motion_even=handle_motion,
            handle_time=handle_time,
            handle_token_valid=handle_bundle_valid.any(dim=-1),
            handle_bundle=handle_bundle,
            handle_bundle_valid=handle_bundle_valid,
            pair_geometry_even=_four_vector_o2_invariants(pair_geometry),
            pair_motion_even=pair_motion,
            pair_time=pair_time,
            pair_token_valid=pair_bundle_valid.any(dim=-1),
            pair_bundle=pair_bundle,
            pair_bundle_valid=pair_bundle_valid,
            base_omega_normalized=base_omega_normalized,
        )
        result["handle_acceleration_supported"] = acceleration_valid
        result["handle_observable_stream"] = stream_supported
        return result

    def _analytic_pair_rotation(
        self,
        omega: torch.Tensor,
        pair_geometry: torch.Tensor,
        pair_kinematics: torch.Tensor,
    ) -> torch.Tensor:
        elapsed_s = 0.01 * torch.expm1(pair_kinematics[..., 8]).clamp_min(0)
        theta = (
            omega[:, None, None, None, 0]
            * self.yaw_rate_scale_rad_s * elapsed_s
        )
        prior = pair_geometry[..., 3:6]
        cosine, sine = torch.cos(theta), torch.sin(theta)
        rotated = torch.stack((
            cosine * prior[..., 0] - sine * prior[..., 1],
            sine * prior[..., 0] + cosine * prior[..., 1],
            prior[..., 2],
        ), dim=-1)
        sign = torch.where(
            pair_kinematics[..., 11] > 0.5, -1.0, 1.0,
        ).unsqueeze(-1)
        return sign * rotated - prior

    def decode_closure_at_state(
        self,
        handle_geometry: torch.Tensor,
        handle_kinematics: torch.Tensor,
        handle_valid: torch.Tensor,
        pair_geometry: torch.Tensor,
        pair_kinematics: torch.Tensor,
        pair_valid: torch.Tensor,
        state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if state.ndim != 2 or state.shape[-1] != 4:
            raise ValueError("alternating fixed state shape differs")
        hg, hk, hv, pg, pk, pv = self._reshape_factors(
            handle_geometry, handle_kinematics, handle_valid,
            pair_geometry, pair_kinematics, pair_valid,
        )
        beta = state[:, :3] / self.beta_to_velocity_state.to(state.dtype)
        relative_geometry, relative_delta, elapsed = (
            self._relative_factors_with_gauge(hg, hk, hv, beta)
        )
        omega = state[:, 3:4]
        theta = (
            omega[:, None, None, None, 0]
            * self.yaw_rate_scale_rad_s * elapsed * self.history_scale_s
        )
        prior = relative_geometry[..., 3:6]
        cosine, sine = torch.cos(theta), torch.sin(theta)
        rotated = torch.stack((
            cosine * prior[..., 0] - sine * prior[..., 1],
            sine * prior[..., 0] + cosine * prior[..., 1], prior[..., 2],
        ), dim=-1)
        handle_rotation = rotated - prior
        translation = beta[:, None, None, None] * elapsed.unsqueeze(-1)
        pair_rotation = self._analytic_pair_rotation(omega, pg, pk)
        return {
            "relative_geometry": relative_geometry.reshape(
                handle_geometry.shape[0], -1, 9,
            ),
            "handle_decoder_geometry": relative_geometry[..., 3:6].reshape(
                handle_geometry.shape[0], -1, 3,
            ),
            "pair_decoder_geometry": torch.cat((
                pg[..., 3:6], pg[..., 9:12],
            ), dim=-1).reshape(pair_geometry.shape[0], -1, 6),
            "angular_handle_residual": (
                relative_delta - handle_rotation
            ).reshape_as(handle_kinematics[..., :3]),
            "pair_residual": (
                pk[..., :3] - pair_rotation
            ).reshape_as(pair_kinematics[..., :3]),
            "common_handle_residual": (
                hk[..., :3] - handle_rotation - translation
            ).reshape_as(handle_kinematics[..., :3]),
            "handle_factor_valid": handle_valid,
            "pair_factor_valid": pair_valid,
        }

    def forward(
        self,
        handle_geometry: torch.Tensor,
        handle_kinematics: torch.Tensor,
        handle_valid: torch.Tensor,
        pair_geometry: torch.Tensor,
        pair_kinematics: torch.Tensor,
        pair_valid: torch.Tensor,
        *,
        refinement_steps: int | None = None,
        angular_refinement_steps: int | None = None,
        velocity_refinement_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if refinement_steps is not None:
            angular_steps = velocity_steps = int(refinement_steps)
        else:
            angular_steps = 2 if angular_refinement_steps is None else int(
                angular_refinement_steps
            )
            velocity_steps = 2 if velocity_refinement_steps is None else int(
                velocity_refinement_steps
            )
        if angular_steps not in (0, 2) or velocity_steps not in (0, 2):
            raise ValueError("alternating refinement selection differs")
        hg, hk, hv, pg, pk, pv = self._reshape_factors(
            handle_geometry, handle_kinematics, handle_valid,
            pair_geometry, pair_kinematics, pair_valid,
        )
        weight = hv.unsqueeze(-1).to(hk.dtype)
        visible_factor_gauge_rate = (
            hk[..., 3:6] * weight
        ).sum(dim=(1, 2, 3)) / weight.sum(dim=(1, 2, 3)).clamp_min(1)
        relative0, delta0, elapsed = self._relative_factors_with_gauge(
            hg, hk, hv, visible_factor_gauge_rate,
        )
        omega0 = self._omega_stage(
            prefix="omega0", relative_geometry=relative0,
            relative_delta=delta0, elapsed_normalized=elapsed,
            handle_kinematics=hk, handle_valid=hv,
            pair_geometry=pg, pair_kinematics=pk, pair_valid=pv,
            base_omega_normalized=visible_factor_gauge_rate.new_zeros(
                (visible_factor_gauge_rate.shape[0], 1),
            ),
        )
        velocity0 = self.velocity0(
            omega_normalized=omega0["omega_normalized"].detach(),
            yaw_scale_rad_s=self.yaw_rate_scale_rad_s,
            relative_geometry=relative0, handle_time=hk[..., 6:14],
            handle_delta=hk[..., :3], elapsed_normalized=elapsed,
            handle_valid=hv,
            beta_to_velocity_state=self.beta_to_velocity_state.to(hk.dtype),
        )

        beta0 = velocity0["velocity_normalized"].detach() / (
            self.beta_to_velocity_state.to(hk.dtype)
        )
        relative1, delta1, elapsed1 = self._relative_factors_with_gauge(
            hg, hk, hv, beta0,
        )
        if angular_steps:
            omega1 = self._omega_stage(
                prefix="omega1", relative_geometry=relative1,
                relative_delta=delta1, elapsed_normalized=elapsed1,
                handle_kinematics=hk, handle_valid=hv,
                pair_geometry=pg, pair_kinematics=pk, pair_valid=pv,
                base_omega_normalized=omega0["omega_normalized"].detach(),
            )
        else:
            omega1 = dict(omega0)
        if velocity_steps:
            velocity1 = self.velocity1(
                omega_normalized=omega1["omega_normalized"].detach(),
                yaw_scale_rad_s=self.yaw_rate_scale_rad_s,
                relative_geometry=relative1, handle_time=hk[..., 6:14],
                handle_delta=hk[..., :3], elapsed_normalized=elapsed1,
                handle_valid=hv,
                beta_to_velocity_state=self.beta_to_velocity_state.to(hk.dtype),
            )
        else:
            velocity1 = dict(velocity0)

        selected_omega = omega1["omega_normalized"]
        selected_velocity = velocity1["velocity_normalized"]
        state = torch.cat((selected_velocity, selected_omega), dim=-1)
        initial_state = torch.cat((
            velocity0["velocity_normalized"], omega0["omega_normalized"],
        ), dim=-1)
        omega0_closure = self.decode_closure_at_state(
            handle_geometry, handle_kinematics, handle_valid,
            pair_geometry, pair_kinematics, pair_valid, torch.cat((
                velocity0["velocity_normalized"].detach(),
                omega0["omega_normalized"],
            ), dim=-1),
        )
        velocity0_closure = self.decode_closure_at_state(
            handle_geometry, handle_kinematics, handle_valid,
            pair_geometry, pair_kinematics, pair_valid, torch.cat((
                velocity0["velocity_normalized"],
                omega0["omega_normalized"].detach(),
            ), dim=-1),
        )
        omega1_closure = self.decode_closure_at_state(
            handle_geometry, handle_kinematics, handle_valid,
            pair_geometry, pair_kinematics, pair_valid, torch.cat((
                velocity0["velocity_normalized"].detach(),
                omega1["omega_normalized"],
            ), dim=-1),
        )
        closure = self.decode_closure_at_state(
            handle_geometry, handle_kinematics, handle_valid,
            pair_geometry, pair_kinematics, pair_valid, torch.cat((
                velocity1["velocity_normalized"],
                omega1["omega_normalized"].detach(),
            ), dim=-1),
        )
        support = torch.cat((
            velocity1["velocity_supported"].unsqueeze(-1).expand(-1, 3),
            omega1["omega_supported"].unsqueeze(-1),
        ), dim=-1)
        return {
            "motion_state_normalized": state,
            "motion_log_variance": torch.where(
                support, torch.zeros_like(state), torch.full_like(state, 5.0),
            ),
            "initial_motion_state_normalized": initial_state,
            "omega0_normalized": omega0["omega_normalized"],
            "velocity0_normalized": velocity0["velocity_normalized"],
            "omega1_normalized": omega1["omega_normalized"],
            "velocity1_normalized": velocity1["velocity_normalized"],
            "omega0_carrier_normalized": omega0["carrier_normalized"],
            "omega1_carrier_normalized": omega1["carrier_normalized"],
            "omega0_delta_normalized": omega0["omega_delta_normalized"],
            "omega1_delta_normalized": omega1["omega_delta_normalized"],
            "omega0_supported": omega0["omega_supported"],
            "omega1_supported": omega1["omega_supported"],
            "velocity0_supported": velocity0["velocity_supported"],
            "velocity1_supported": velocity1["velocity_supported"],
            "omega0_evidence_weight": omega0["evidence_weight"],
            "omega1_evidence_weight": omega1["evidence_weight"],
            "visible_factor_gauge_rate_normalized": visible_factor_gauge_rate,
            "velocity0_wls_beta_normalized_rate": velocity0[
                "wls_beta_normalized_rate"
            ],
            "velocity1_wls_beta_normalized_rate": velocity1[
                "wls_beta_normalized_rate"
            ],
            "angular_iteration_normalized": torch.stack((
                omega0["omega_normalized"], omega1["omega_normalized"],
            ), dim=1),
            "velocity_iteration_normalized": torch.stack((
                velocity0["velocity_normalized"],
                velocity1["velocity_normalized"],
            ), dim=1),
            "handle_closure_prediction_normalized": (
                handle_kinematics[..., :3]
                - closure["common_handle_residual"]
            ),
            "pair_closure_prediction_normalized": (
                pair_kinematics[..., :3] - closure["pair_residual"]
            ),
            "handle_closure_residual_normalized": closure[
                "common_handle_residual"
            ],
            "pair_closure_residual_normalized": closure["pair_residual"],
            "angular_handle_closure_residual_normalized": closure[
                "angular_handle_residual"
            ],
            "common_handle_closure_residual_normalized": closure[
                "common_handle_residual"
            ],
            "omega0_angular_handle_closure_residual_normalized": omega0_closure[
                "angular_handle_residual"
            ],
            "omega0_pair_closure_residual_normalized": omega0_closure[
                "pair_residual"
            ],
            "velocity0_common_handle_closure_residual_normalized": velocity0_closure[
                "common_handle_residual"
            ],
            "omega1_angular_handle_closure_residual_normalized": omega1_closure[
                "angular_handle_residual"
            ],
            "omega1_pair_closure_residual_normalized": omega1_closure[
                "pair_residual"
            ],
            "velocity1_common_handle_closure_residual_normalized": closure[
                "common_handle_residual"
            ],
            "handle_factor_valid": handle_valid,
            "pair_factor_valid": pair_valid,
            "pair_supported": pair_valid.any(dim=1),
            "angular_refinement_steps": state.new_full(
                (state.shape[0],), angular_steps,
            ),
            "velocity_refinement_steps": state.new_full(
                (state.shape[0],), velocity_steps,
            ),
        }


class AnonymousEquivariantAlternatingTwistProbe(
    AnonymousOmegaFirstOrderedClosureProbe
):
    """V13 typed alternating estimator with no permanent armor identity."""

    model_family = "anonymous-equivariant-alternating-twist-probe-v13"
    max_abs_yaw_rate_rad_s = 15.0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        velocity_scale = tuple(
            float(value) for value in self.motion_state_scale[:3].tolist()
        )
        self.motion_state_head = EquivariantAlternatingTwistHead(
            self.channels, self.dropout,
            history_scale_s=float(self.context.history_scale_s),
            position_scale_m=float(self.context.position_scale_m),
            velocity_scale_mps=velocity_scale,
            yaw_rate_scale_rad_s=float(self.motion_state_scale[3]),
            max_abs_yaw_rate_rad_s=self.max_abs_yaw_rate_rad_s,
        )

    @property
    def config(self) -> dict[str, Any]:
        parent = dict(super().config)
        parent.update({
            "family": self.model_family,
            "motion_state_fusion": (
                "typed detached omega0 velocity0 omega1 velocity1 with "
                "signed residual proposals, soft evidence fusion and "
                "equivariant residual-WLS velocity"
            ),
            "typed_alternating": True,
            "strict_omega_first": False,
            "pre_recurrence_handle_scale_pooling": False,
            "visible_factor_gauge_is_center_velocity": False,
            "pair_primary_swap_canonicalized": True,
            "pair_handle_hard_switch": False,
            "handle_only_minimum_positions": 3,
            "unsupported_state_is_explicit": True,
            "common_velocity_wls_equivariant_on_supported_rows": True,
            "reflection_equivariance_structural": True,
            "omega1_is_detached_omega0_plus_delta": True,
            "yaw_alias_envelope_max_elapsed_s": (
                self.motion_state_head.max_factor_elapsed_s
            ),
            "yaw_alias_envelope_max_abs_rate_rad_s": (
                self.motion_state_head.max_abs_yaw_rate_rad_s
            ),
            "analytic_future_decoder": False,
        })
        return parent

    def state_branch_hashes(self) -> dict[str, str]:
        state = self.state_dict()

        def selected(prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
            return {
                name: value for name, value in state.items()
                if any(name.startswith(prefix) for prefix in prefixes)
            }

        return {
            "context": state_dict_sha256(selected(("context.",))),
            "omega0": state_dict_sha256(selected(("motion_state_head.omega0.",))),
            "velocity0": state_dict_sha256(selected(("motion_state_head.velocity0.",))),
            "omega1": state_dict_sha256(selected(("motion_state_head.omega1.",))),
            "velocity1": state_dict_sha256(selected(("motion_state_head.velocity1.",))),
        }


def _macro_mean(value: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    pieces = [value[group == item].mean() for item in torch.unique(group)]
    if not pieces:
        raise ValueError("alternating macro loss lacks groups")
    return torch.stack(pieces).mean()


def _macro_mean_supported(
    value: torch.Tensor, group: torch.Tensor, supported: torch.Tensor,
) -> torch.Tensor:
    pieces = [
        value[(group == item) & supported].mean()
        for item in torch.unique(group[supported])
    ]
    if not pieces:
        return value.sum() * 0.0
    return torch.stack(pieces).mean()


def _factor_error(
    residual: torch.Tensor, valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if residual.shape[:2] != valid.shape:
        raise ValueError("alternating closure shapes differ")
    coordinate = F.smooth_l1_loss(
        residual, torch.zeros_like(residual), beta=0.02, reduction="none",
    ).mean(dim=-1)
    weight = valid.to(coordinate.dtype)
    per_sample = (coordinate * weight).sum(dim=1) / weight.sum(
        dim=1,
    ).clamp_min(1)
    return per_sample, valid.any(dim=1)


def _history_group(active_count: torch.Tensor) -> torch.Tensor:
    return torch.where(
        active_count >= 32, 3,
        torch.where(active_count >= 24, 2, torch.where(active_count >= 16, 1, 0)),
    )


def alternating_substage_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    substage: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = batch["target_motion_state_normalized"]
    pair_group = prediction["pair_flow_available"].sum(dim=1).to(torch.long)
    history_group = _history_group(prediction["history_active_count"])
    motion_group = batch.get(
        "motion_class", torch.zeros_like(pair_group),
    ).to(torch.long)
    joint_group = motion_group * 16 + pair_group * 4 + history_group
    zero = target.new_zeros(())
    if substage in {"omega0", "omega1"}:
        key = f"{substage}_normalized"
        supported = prediction[f"{substage}_supported"].to(torch.bool)
        per_sample = F.smooth_l1_loss(
            prediction[key].squeeze(-1), target[:, 3],
            beta=0.05, reduction="none",
        )
        macro = _macro_mean_supported(per_sample, joint_group, supported)
        handle_closure, handle_has_factor = _factor_error(
            prediction[
                f"{substage}_angular_handle_closure_residual_normalized"
            ], prediction["handle_factor_valid"],
        )
        pair_closure, pair_has_factor = _factor_error(
            prediction[f"{substage}_pair_closure_residual_normalized"],
            prediction["pair_factor_valid"],
        )
        closure = (
            0.10 * _macro_mean_supported(
                handle_closure, joint_group, supported & handle_has_factor,
            )
            + 0.15 * _macro_mean_supported(
                pair_closure, joint_group, supported & pair_has_factor,
            )
        )
        physical_target = batch.get("target_motion_state_physical")
        strong = supported & (
            physical_target[:, 3].abs() > 0.5
            if physical_target is not None else target[:, 3].abs() > 0.03
        )
        sign_margin = (
            F.relu(0.03 - prediction[key].squeeze(-1) * target[:, 3].sign())[
                strong
            ].mean() if bool(strong.any()) else zero
        )
        objective = macro + closure + 0.05 * sign_margin
        velocity, yaw = zero, macro
    elif substage in {"velocity0", "velocity1"}:
        key = f"{substage}_normalized"
        supported = prediction[f"{substage}_supported"].to(torch.bool)
        coordinate = F.smooth_l1_loss(
            prediction[key], target[:, :3], beta=0.05, reduction="none",
        )
        per_sample = coordinate[:, :2].mean(dim=1) + 0.25 * coordinate[:, 2]
        macro = _macro_mean_supported(per_sample, joint_group, supported)
        common_closure, has_factor = _factor_error(
            prediction[
                f"{substage}_common_handle_closure_residual_normalized"
            ], prediction["handle_factor_valid"],
        )
        closure = 0.10 * _macro_mean_supported(
            common_closure, joint_group, supported & has_factor,
        )
        objective = macro + closure
        velocity, yaw = macro, zero
    else:
        raise ValueError(f"unknown alternating substage: {substage}")
    return objective, {
        "objective": objective,
        "motion": objective,
        "velocity": velocity,
        "yaw_rate": yaw,
        "planar_velocity": velocity,
        "vertical_velocity": zero,
        "scale_heteroscedastic": zero,
        "trajectory": zero,
        "trend": zero,
        "role": zero,
        "distance_risk": zero,
        "typed_closure": closure,
        "typed_supported_fraction": supported.to(target.dtype).mean(),
    }


def _reflect_physical_training_batch(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Reflect physical Y without introducing a handle identity feature."""
    result = dict(batch)
    history = batch["history_obs_rel_m"].clone()
    history[..., 1] = -history[..., 1]
    result["history_obs_rel_m"] = history
    target = batch["target_motion_state_normalized"].clone()
    target[:, 1] = -target[:, 1]
    target[:, 3] = -target[:, 3]
    result["target_motion_state_normalized"] = target
    if "target_motion_state_physical" in batch:
        physical = batch["target_motion_state_physical"].clone()
        physical[:, 1] = -physical[:, 1]
        physical[:, 3] = -physical[:, 3]
        result["target_motion_state_physical"] = physical
    if "history_switch_step" in batch:
        result["history_switch_step"] = -batch["history_switch_step"]
    return result


def equivariant_alternating_train_step(
    model: StableMotionBottleneckAnonymousFutureModel,
    batch: dict[str, torch.Tensor],
    stage_update: int,
    stage_total: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    if int(stage_total) != 100 or not 1 <= int(stage_update) <= 100:
        raise ValueError("alternating structural screen is fixed to 100 updates")
    if stage_update <= 35:
        substage, endpoint = "omega0", stage_update == 35
        phase_update, phase_total = stage_update, 35
    elif stage_update <= 55:
        substage, endpoint = "velocity0", stage_update == 55
        phase_update, phase_total = stage_update - 35, 20
    elif stage_update <= 80:
        substage, endpoint = "omega1", stage_update == 80
        phase_update, phase_total = stage_update - 55, 25
    else:
        substage, endpoint = "velocity1", stage_update == 100
        phase_update, phase_total = stage_update - 80, 20
    field_names = AnonymousEquivariantAlternatingTwistProbe._field_names()
    original_output = model.estimate_motion_state(**{
        name: batch[name] for name in field_names
    })
    original = {**original_output["history"], **original_output["state"]}
    selected, ramp = _deterministic_probe_ramp(
        batch["history_obs_rel_m"].shape[0],
        dtype=batch["history_obs_rel_m"].dtype,
        device=batch["history_obs_rel_m"].device,
        stage_update=stage_update,
    )
    augmented_batch = apply_common_velocity_ramp(
        batch, ramp, model.motion_state_scale.to(ramp.dtype),
    )
    augmented_output = model.estimate_motion_state(**{
        name: augmented_batch[name] for name in field_names
    })
    augmented = {**augmented_output["history"], **augmented_output["state"]}
    reflected_batch = _reflect_physical_training_batch(batch)
    reflected_output = model.estimate_motion_state(**{
        name: reflected_batch[name] for name in field_names
    })
    reflected = {**reflected_output["history"], **reflected_output["state"]}
    original_loss, original_components = alternating_substage_loss(
        original, batch, substage=substage,
    )
    augmented_loss, augmented_components = alternating_substage_loss(
        augmented, augmented_batch, substage=substage,
    )
    reflected_loss, reflected_components = alternating_substage_loss(
        reflected, reflected_batch, substage=substage,
    )
    support_key = f"{substage}_supported"
    common_support = (
        original[support_key].to(torch.bool)
        & augmented[support_key].to(torch.bool)
    )
    selected_support = selected & common_support
    if substage.startswith("omega"):
        equivariance = (
            F.smooth_l1_loss(
                augmented[f"{substage}_normalized"][selected_support],
                original[f"{substage}_normalized"][selected_support], beta=0.002,
            ) if bool(selected_support.any()) else original_loss * 0.0
        )
        reflected_expected = -original[f"{substage}_normalized"]
    else:
        normalized_ramp = ramp[:, :3] / model.motion_state_scale[:3].to(ramp.dtype)
        equivariance = (
            F.smooth_l1_loss(
                augmented[f"{substage}_normalized"][selected_support]
                - original[f"{substage}_normalized"][selected_support],
                normalized_ramp[selected_support], beta=0.002,
            ) if bool(selected_support.any()) else original_loss * 0.0
        )
        reflection = original[f"{substage}_normalized"].new_tensor(
            [1.0, -1.0, 1.0],
        )
        reflected_expected = original[f"{substage}_normalized"] * reflection
    reflected_support = (
        original[support_key].to(torch.bool)
        & reflected[support_key].to(torch.bool)
    )
    reflection_equivariance = (
        F.smooth_l1_loss(
            reflected[f"{substage}_normalized"][reflected_support],
            reflected_expected[reflected_support], beta=0.002,
        ) if bool(reflected_support.any()) else original_loss * 0.0
    )
    objective = (
        (original_loss + augmented_loss + reflected_loss) / 3.0
        + 0.20 * equivariance + 0.10 * reflection_equivariance
    )
    components = dict(original_components)
    for name in (
        "motion", "velocity", "yaw_rate", "planar_velocity",
        "vertical_velocity", "scale_heteroscedastic",
    ):
        components[name] = (
            original_components[name] + augmented_components[name]
            + reflected_components[name]
        ) / 3.0
    components.update({
        "objective": objective,
        "typed_equivariance": equivariance,
        "typed_reflection_equivariance": reflection_equivariance,
        "state_substage": f"typed_alternating_{substage}",
        "state_substage_endpoint": endpoint,
        "state_lr_phase_update": int(phase_update),
        "state_lr_phase_total": int(phase_total),
    })
    return original, objective, components


__all__ = [
    "EquivariantAlternatingTwistHead",
    "AnonymousEquivariantAlternatingTwistProbe",
    "alternating_substage_loss",
    "equivariant_alternating_train_step",
]
