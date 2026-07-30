"""Typed omega-first ordered history-closure estimator.

The estimator keeps one q0 constant twist but removes V11's shared 4D update.
Translation-invariant, event-ordered relative factors estimate signed angular
rate first.  A learned observation decoder then predicts rotational handle
displacement.  Only the remaining de-rotated common residual may estimate or
refine translation.  The sole cross-state direction is omega -> velocity.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .factorized_common_relative_motion_future import apply_common_velocity_ramp
from .global_flow_closure_future import (
    AnonymousGlobalFlowClosureProbe,
    _OmegaGeometryDecoder,
    _StateTimeTranslationDecoder,
    _TripleGatedFactorEncoder,
    _masked_closure_loss,
)
from .joint_rigid_flow_probe import LOCAL_LAG_SCALES_S, _deterministic_probe_ramp
from .paired_twist_set_future import paired_twist_state_loss
from .stable_motion_bottleneck_future import StableMotionBottleneckAnonymousFutureModel


class _MaskedOrderedGRU(nn.Module):
    """Causal recurrent evidence accumulator whose inactive events are identity."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = int(width)
        self.cell = nn.GRUCell(self.width, self.width, bias=False)

    def forward(self, event: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if event.ndim != 3 or event.shape[:2] != valid.shape:
            raise ValueError("ordered event sequence shapes differ")
        hidden = event.new_zeros(event.shape[0], self.width)
        for index in range(event.shape[1]):
            updated = self.cell(event[:, index], hidden)
            hidden = torch.where(valid[:, index:index + 1], updated, hidden)
        return hidden


class _ResidualTimeEncoder(nn.Module):
    """Zero-preserving residual/time interaction with no time-only shortcut."""

    def __init__(self, residual_features: int, time_features: int, width: int) -> None:
        super().__init__()
        self.residual = nn.Linear(residual_features, width, bias=False)
        self.time = nn.Linear(time_features, width, bias=False)
        self.post = nn.Sequential(
            nn.LayerNorm(width, elementwise_affine=False),
            nn.Linear(width, width, bias=False), nn.SiLU(),
        )

    def forward(
        self, residual: torch.Tensor, time: torch.Tensor,
    ) -> torch.Tensor:
        return self.post(
            F.silu(self.residual(residual)) * F.silu(self.time(time))
        )


class OmegaFirstOrderedClosureHead(nn.Module):
    """Estimate omega from relative flow, then velocity from de-rotated flow."""

    angular_refinement_steps = 2
    velocity_refinement_steps = 2

    def __init__(
        self, channels: int, dropout: float, *, history_scale_s: float = 0.32,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.width = 11 * self.channels // 4
        if self.width < 64:
            raise ValueError("omega-first ordered width is too small")
        if history_scale_s <= 0:
            raise ValueError("omega-first history scale must be positive")
        self.history_scale_s = float(history_scale_s)
        self.handle_relative_encoder = _TripleGatedFactorEncoder(
            9, 3, 8, self.width,
        )
        self.pair_relative_encoder = _TripleGatedFactorEncoder(
            12, 3, 7, self.width,
        )
        self.angular_event_fusion = nn.Sequential(
            nn.Linear(2 * self.width, self.width, bias=False), nn.SiLU(),
        )
        self.angular_sequence = _MaskedOrderedGRU(self.width)
        self.omega_initial = nn.Sequential(
            nn.Linear(self.width, self.width, bias=False), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(self.width, 1, bias=False),
        )
        self.omega_update = nn.Sequential(
            nn.Linear(self.width + 1, self.width, bias=False), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(self.width, 1, bias=False),
        )
        self.handle_rotation_decoder = _OmegaGeometryDecoder(
            3, 8, self.width,
        )
        self.pair_rotation_decoder = _OmegaGeometryDecoder(
            6, 6, self.width,
        )
        self.common_encoder = _ResidualTimeEncoder(3, 8, self.width)
        self.common_sequence = _MaskedOrderedGRU(self.width)
        self.velocity_initial = nn.Sequential(
            nn.Linear(self.width, self.width, bias=False), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(self.width, 3, bias=False),
        )
        self.velocity_update = nn.Sequential(
            nn.Linear(self.width + 3, self.width, bias=False), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(self.width, 3, bias=False),
        )
        self.translation_decoder = _StateTimeTranslationDecoder(8, self.width)
        self.log_variance = nn.Sequential(
            nn.LayerNorm(2 * self.width + 4),
            nn.Linear(2 * self.width + 4, self.channels), nn.SiLU(),
            nn.Linear(self.channels, 4),
        )

    @staticmethod
    def _masked_mean(
        value: torch.Tensor, valid: torch.Tensor, dims: tuple[int, ...],
    ) -> torch.Tensor:
        weight = valid.unsqueeze(-1).to(value.dtype)
        return (value * weight).sum(dim=dims) / weight.sum(dim=dims).clamp_min(1)

    @staticmethod
    def _reshape_factors(
        handle_geometry: torch.Tensor,
        handle_kinematics: torch.Tensor,
        handle_valid: torch.Tensor,
        pair_geometry: torch.Tensor,
        pair_kinematics: torch.Tensor,
        pair_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        scales = len(LOCAL_LAG_SCALES_S)
        if pair_geometry.shape[1] % scales:
            raise ValueError("omega-first pair factor count differs")
        events = pair_geometry.shape[1] // scales
        if handle_geometry.shape[1] != 4 * events * scales:
            raise ValueError("omega-first handle factor count differs")
        return (
            handle_geometry.reshape(-1, 4, events, scales, 12),
            handle_kinematics.reshape(-1, 4, events, scales, 14),
            handle_valid.reshape(-1, 4, events, scales),
            pair_geometry.reshape(-1, 1, events, scales, 12),
            pair_kinematics.reshape(-1, 1, events, scales, 13),
            pair_valid.reshape(-1, 1, events, scales),
        )

    def _derived_relative_factors(
        self,
        handle_geometry: torch.Tensor,
        handle_kinematics: torch.Tensor,
        handle_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = handle_valid.unsqueeze(-1).to(handle_kinematics.dtype)
        common_rate = (
            handle_kinematics[..., 3:6] * weight
        ).sum(dim=(1, 2, 3)) / weight.sum(dim=(1, 2, 3)).clamp_min(1)
        elapsed_s = 0.01 * torch.expm1(handle_kinematics[..., 6]).clamp_min(0)
        elapsed_normalized = elapsed_s / self.history_scale_s
        current_time = handle_kinematics[..., 7]
        prior_time = current_time - elapsed_normalized
        current_detrended = handle_geometry[..., 6:9] - (
            common_rate[:, None, None, None] * current_time.unsqueeze(-1)
        )
        prior_detrended = handle_geometry[..., 9:12] - (
            common_rate[:, None, None, None] * prior_time.unsqueeze(-1)
        )
        center = (
            (current_detrended + prior_detrended) * 0.5 * weight
        ).sum(dim=(1, 2, 3)) / weight.sum(dim=(1, 2, 3)).clamp_min(1)
        current_relative = current_detrended - center[:, None, None, None]
        prior_relative = prior_detrended - center[:, None, None, None]
        relative_geometry = torch.cat((
            current_relative, prior_relative,
            current_relative - prior_relative,
        ), dim=-1)
        relative_delta = handle_kinematics[..., :3] - (
            common_rate[:, None, None, None]
            * elapsed_normalized.unsqueeze(-1)
        )
        return relative_geometry, relative_delta, common_rate, elapsed_normalized

    def _ordered_summary(
        self,
        handle_embedding: torch.Tensor,
        handle_valid: torch.Tensor,
        pair_embedding: torch.Tensor,
        pair_valid: torch.Tensor,
        *,
        angular: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        handle_event = self._masked_mean(
            handle_embedding, handle_valid, dims=(1, 3),
        )
        pair_event = self._masked_mean(
            pair_embedding, pair_valid, dims=(1, 3),
        )
        event_valid = handle_valid.any(dim=(1, 3)) | pair_valid.any(dim=(1, 3))
        if angular:
            event = self.angular_event_fusion(torch.cat((
                handle_event, pair_event,
            ), dim=-1))
            return self.angular_sequence(event, event_valid), event_valid
        # Common flow has no direct pair input; keep the same return contract.
        return self.common_sequence(handle_event, handle_valid.any(dim=(1, 3))), (
            handle_valid.any(dim=(1, 3))
        )

    @staticmethod
    def _decode_omega_factors(
        decoder: _OmegaGeometryDecoder,
        omega: torch.Tensor,
        geometry: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        shape = geometry.shape[:-1]
        decoded = decoder(
            omega,
            geometry.reshape(geometry.shape[0], -1, geometry.shape[-1]),
            time.reshape(time.shape[0], -1, time.shape[-1]),
        )
        return decoded.reshape(*shape, 3)

    @staticmethod
    def _decode_translation_factors(
        decoder: _StateTimeTranslationDecoder,
        velocity: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        shape = time.shape[:-1]
        decoded = decoder(
            velocity, time.reshape(time.shape[0], -1, time.shape[-1]),
        )
        return decoded.reshape(*shape, 3)

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
        """Decode supplied state without allowing the factors to re-estimate it."""
        if state.ndim != 2 or state.shape[-1] != 4:
            raise ValueError("fixed omega-first state shape differs")
        hg, hk, hv, pg, pk, pv = self._reshape_factors(
            handle_geometry, handle_kinematics, handle_valid,
            pair_geometry, pair_kinematics, pair_valid,
        )
        relative_geometry, relative_delta, _, _ = self._derived_relative_factors(
            hg, hk, hv,
        )
        handle_time = hk[..., 6:14]
        omega = state[:, 3:4]
        velocity = state[:, :3]
        handle_rotation = self._decode_omega_factors(
            self.handle_rotation_decoder,
            omega, relative_geometry[..., 3:6], handle_time,
        )
        pair_prior = torch.cat((pg[..., 3:6], pg[..., 9:12]), dim=-1)
        pair_rotation = self._decode_omega_factors(
            self.pair_rotation_decoder, omega, pair_prior, pk[..., 7:13],
        )
        translation = self._decode_translation_factors(
            self.translation_decoder, velocity, handle_time,
        )
        common_observed = hk[..., :3] - handle_rotation
        return {
            "relative_geometry": relative_geometry.reshape(
                handle_geometry.shape[0], -1, 9,
            ),
            "handle_decoder_geometry": relative_geometry[..., 3:6].reshape(
                handle_geometry.shape[0], -1, 3,
            ),
            "pair_decoder_geometry": pair_prior.reshape(
                pair_geometry.shape[0], -1, 6,
            ),
            "angular_handle_residual": (
                relative_delta - handle_rotation
            ).reshape_as(handle_kinematics[..., :3]),
            "pair_residual": (
                pk[..., :3] - pair_rotation
            ).reshape_as(pair_kinematics[..., :3]),
            "common_handle_residual": (
                common_observed - translation
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
        hg, hk, hv, pg, pk, pv = self._reshape_factors(
            handle_geometry, handle_kinematics, handle_valid,
            pair_geometry, pair_kinematics, pair_valid,
        )
        if refinement_steps is not None:
            angular_steps = velocity_steps = int(refinement_steps)
        else:
            angular_steps = (
                self.angular_refinement_steps if angular_refinement_steps is None
                else int(angular_refinement_steps)
            )
            velocity_steps = (
                self.velocity_refinement_steps if velocity_refinement_steps is None
                else int(velocity_refinement_steps)
            )
        if not 0 <= angular_steps <= self.angular_refinement_steps:
            raise ValueError("omega-first angular refinement count differs")
        if not 0 <= velocity_steps <= self.velocity_refinement_steps:
            raise ValueError("omega-first velocity refinement count differs")

        relative_geometry, relative_delta, _, _ = self._derived_relative_factors(
            hg, hk, hv,
        )
        handle_time = hk[..., 6:14]
        pair_time = pk[..., 6:13]
        handle_embedding = self.handle_relative_encoder(
            relative_geometry, relative_delta, handle_time,
        )
        pair_embedding = self.pair_relative_encoder(
            pg, pk[..., :3], pair_time,
        )
        angular_hidden, _ = self._ordered_summary(
            handle_embedding, hv, pair_embedding, pv, angular=True,
        )
        omega_logits = self.omega_initial(angular_hidden)
        initial_omega = torch.tanh(omega_logits)
        omega_iterations = [initial_omega]
        handle_rotation = torch.zeros_like(hk[..., :3])
        pair_rotation = torch.zeros_like(pk[..., :3])
        angular_handle_residual = relative_delta
        pair_residual = pk[..., :3]
        for _ in range(angular_steps):
            omega = torch.tanh(omega_logits)
            handle_rotation = self._decode_omega_factors(
                self.handle_rotation_decoder,
                omega, relative_geometry[..., 3:6], handle_time,
            )
            pair_prior = torch.cat((pg[..., 3:6], pg[..., 9:12]), dim=-1)
            pair_rotation = self._decode_omega_factors(
                self.pair_rotation_decoder, omega, pair_prior, pk[..., 7:13],
            )
            angular_handle_residual = relative_delta - handle_rotation
            pair_residual = pk[..., :3] - pair_rotation
            handle_message = self.handle_relative_encoder(
                relative_geometry, angular_handle_residual, handle_time,
            )
            pair_message = self.pair_relative_encoder(
                pg, pair_residual, pair_time,
            )
            angular_hidden, _ = self._ordered_summary(
                handle_message, hv, pair_message, pv, angular=True,
            )
            omega_logits = omega_logits + 0.5 * torch.tanh(
                self.omega_update(torch.cat((angular_hidden, omega), dim=-1))
            )
            omega_iterations.append(torch.tanh(omega_logits))
        omega = torch.tanh(omega_logits)
        handle_rotation = self._decode_omega_factors(
            self.handle_rotation_decoder,
            omega, relative_geometry[..., 3:6], handle_time,
        )
        pair_prior = torch.cat((pg[..., 3:6], pg[..., 9:12]), dim=-1)
        pair_rotation = self._decode_omega_factors(
            self.pair_rotation_decoder, omega, pair_prior, pk[..., 7:13],
        )
        angular_handle_residual = relative_delta - handle_rotation
        pair_residual = pk[..., :3] - pair_rotation

        common_observed = hk[..., :3] - handle_rotation
        common_embedding = self.common_encoder(common_observed, handle_time)
        common_hidden, _ = self._ordered_summary(
            common_embedding, hv,
            pair_embedding.new_zeros(pair_embedding.shape),
            torch.zeros_like(pv), angular=False,
        )
        velocity_logits = self.velocity_initial(common_hidden)
        initial_velocity = torch.tanh(velocity_logits)
        velocity_iterations = [initial_velocity]
        translation = torch.zeros_like(common_observed)
        common_residual = common_observed
        for _ in range(velocity_steps):
            velocity = torch.tanh(velocity_logits)
            translation = self._decode_translation_factors(
                self.translation_decoder, velocity, handle_time,
            )
            common_residual = common_observed - translation
            common_message = self.common_encoder(common_residual, handle_time)
            common_hidden, _ = self._ordered_summary(
                common_message, hv,
                pair_embedding.new_zeros(pair_embedding.shape),
                torch.zeros_like(pv), angular=False,
            )
            velocity_logits = velocity_logits + 0.5 * torch.tanh(
                self.velocity_update(torch.cat((
                    common_hidden, velocity,
                ), dim=-1))
            )
            velocity_iterations.append(torch.tanh(velocity_logits))
        velocity = torch.tanh(velocity_logits)
        translation = self._decode_translation_factors(
            self.translation_decoder, velocity, handle_time,
        )
        common_residual = common_observed - translation
        state = torch.cat((velocity, omega), dim=-1)
        initial_state = torch.cat((initial_velocity, initial_omega), dim=-1)
        uncertainty = self.log_variance(torch.cat((
            angular_hidden, common_hidden, state,
        ), dim=-1)).clamp(-5.0, 5.0)
        handle_prediction = handle_rotation + translation
        handle_residual = hk[..., :3] - handle_prediction
        return {
            "motion_state_normalized": state,
            "motion_log_variance": uncertainty,
            "initial_motion_state_normalized": initial_state,
            "angular_iteration_normalized": torch.stack(omega_iterations, dim=1),
            "velocity_iteration_normalized": torch.stack(
                velocity_iterations, dim=1,
            ),
            "handle_closure_prediction_normalized": handle_prediction.reshape_as(
                handle_kinematics[..., :3]
            ),
            "pair_closure_prediction_normalized": pair_rotation.reshape_as(
                pair_kinematics[..., :3]
            ),
            "handle_closure_residual_normalized": handle_residual.reshape_as(
                handle_kinematics[..., :3]
            ),
            "pair_closure_residual_normalized": pair_residual.reshape_as(
                pair_kinematics[..., :3]
            ),
            "angular_handle_closure_residual_normalized": (
                angular_handle_residual.reshape_as(handle_kinematics[..., :3])
            ),
            "common_handle_closure_residual_normalized": (
                common_residual.reshape_as(handle_kinematics[..., :3])
            ),
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


class AnonymousOmegaFirstOrderedClosureProbe(AnonymousGlobalFlowClosureProbe):
    """V12 strict omega-first ordered structural probe."""

    model_family = "anonymous-omega-first-ordered-closure-probe-v12"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.motion_state_head = OmegaFirstOrderedClosureHead(
            self.channels, self.dropout,
            history_scale_s=float(self.context.history_scale_s),
        )

    @property
    def config(self) -> dict[str, Any]:
        parent = dict(super().config)
        parent.update({
            "family": self.model_family,
            "motion_state_fusion": (
                "event-ordered relative omega followed by learned de-rotated "
                "common velocity closure"
            ),
            "event_order_preserved": True,
            "shared_four_dimensional_update": False,
            "velocity_to_omega_path": False,
            "omega_to_velocity_path": True,
            "residual_encoder_reads_prediction_separately": False,
            "pair_writes_velocity_directly": False,
            "analytic_future_decoder": False,
        })
        return parent

    def _estimate_typed(
        self,
        fields: dict[str, torch.Tensor],
        *,
        angular_steps: int,
        velocity_steps: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        history = self.context(**fields)
        state = self.motion_state_head(
            history["_handle_geometry_raw"], history["_handle_kinematics_raw"],
            history["_handle_raw_valid"], history["_pair_geometry_raw"],
            history["_pair_kinematics_raw"], history["_pair_raw_valid"],
            angular_refinement_steps=angular_steps,
            velocity_refinement_steps=velocity_steps,
        )
        state["motion_state_physical"] = (
            state["motion_state_normalized"]
            * self.motion_state_scale.to(state["motion_state_normalized"].dtype)
        )
        public_history = {
            name: value for name, value in history.items()
            if not name.startswith("_")
        }
        return {"history": public_history, "state": state}

    def estimate_motion_state_zero_angular_refinement(self, **fields):
        return self._estimate_typed(fields, angular_steps=0, velocity_steps=2)

    def estimate_motion_state_zero_velocity_refinement(self, **fields):
        return self._estimate_typed(fields, angular_steps=2, velocity_steps=0)


def omega_first_ordered_state_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    base, components = paired_twist_state_loss(prediction, batch)
    angular_handle = _masked_closure_loss(
        prediction["angular_handle_closure_residual_normalized"],
        prediction["handle_factor_valid"],
    )
    pair = _masked_closure_loss(
        prediction["pair_closure_residual_normalized"],
        prediction["pair_factor_valid"],
    )
    common = _masked_closure_loss(
        prediction["common_handle_closure_residual_normalized"],
        prediction["handle_factor_valid"],
    )
    target = batch["target_motion_state_normalized"]
    initial_velocity = F.smooth_l1_loss(
        prediction["initial_motion_state_normalized"][:, :3], target[:, :3],
        beta=0.1,
    )
    initial_omega = F.smooth_l1_loss(
        prediction["initial_motion_state_normalized"][:, 3], target[:, 3],
        beta=0.1,
    )
    objective = base + 0.15 * (
        angular_handle + pair + common
    ) + 0.05 * (initial_velocity + initial_omega)
    result = dict(components)
    result.update({
        "objective": objective,
        "angular_handle_history_closure": angular_handle,
        "pair_history_closure": pair,
        "common_handle_history_closure": common,
        "initial_velocity": initial_velocity,
        "initial_omega": initial_omega,
    })
    return objective, result


def omega_first_ordered_train_step(
    model: StableMotionBottleneckAnonymousFutureModel,
    batch: dict[str, torch.Tensor],
    stage_update: int,
    stage_total: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    if int(stage_total) != 200 or not 1 <= int(stage_update) <= 200:
        raise ValueError("omega-first ordered probe is fixed to 200 updates")
    field_names = AnonymousOmegaFirstOrderedClosureProbe._field_names()
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
    original_loss, original_components = omega_first_ordered_state_loss(
        original, batch,
    )
    augmented_loss, augmented_components = omega_first_ordered_state_loss(
        augmented, augmented_batch,
    )
    yaw_invariance = F.smooth_l1_loss(
        augmented["motion_state_normalized"][selected, 3],
        original["motion_state_normalized"][selected, 3], beta=0.02,
    )
    normalized_ramp = ramp[:, :2] / model.motion_state_scale[:2].to(ramp.dtype)
    translation_equivariance = F.smooth_l1_loss(
        augmented["motion_state_normalized"][selected, :2]
        - original["motion_state_normalized"][selected, :2],
        normalized_ramp[selected], beta=0.02,
    )
    objective = 0.5 * (original_loss + augmented_loss) + 0.20 * (
        yaw_invariance + translation_equivariance
    )
    components = dict(original_components)
    for name in (
        "motion", "velocity", "yaw_rate", "planar_velocity",
        "vertical_velocity", "scale_heteroscedastic",
        "angular_handle_history_closure", "pair_history_closure",
        "common_handle_history_closure", "initial_velocity", "initial_omega",
    ):
        components[name] = 0.5 * (
            original_components[name] + augmented_components[name]
        )
    components.update({
        "objective": objective,
        "ramp_yaw_invariance": yaw_invariance,
        "ramp_translation_equivariance": translation_equivariance,
        "state_substage": "omega_first_ordered_closure_structural_probe",
        "state_substage_endpoint": False,
    })
    return original, objective, components


__all__ = [
    "OmegaFirstOrderedClosureHead",
    "AnonymousOmegaFirstOrderedClosureProbe",
    "omega_first_ordered_state_loss",
    "omega_first_ordered_train_step",
]
