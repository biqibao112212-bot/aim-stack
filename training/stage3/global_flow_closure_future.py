"""Learned global rigid-flow closure estimator (V11 structural probe).

V10 asked every local pair bundle to vote for the complete global yaw rate.
V11 instead treats exact handle and pair edges as factors for one global twist.
A learned history decoder predicts already-observed displacements from that
twist and prior geometry; pooled closure residuals refine the same state twice.
The decoder never predicts a future position and never receives the observed
displacement as one of its prediction inputs.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .factorized_common_relative_motion_future import apply_common_velocity_ramp
from .joint_rigid_flow_probe import LOCAL_LAG_SCALES_S, _deterministic_probe_ramp
from .paired_residual_twist_future import AnonymousPairedResidualTwistProbe
from .paired_twist_set_future import paired_twist_state_loss
from .stable_motion_bottleneck_future import StableMotionBottleneckAnonymousFutureModel


class _FactorEncoder(nn.Module):
    def __init__(self, input_features: int, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_features, width), nn.LayerNorm(width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class _TripleGatedFactorEncoder(nn.Module):
    """Zero-preserving correspondence among geometry, motion and time."""

    def __init__(
        self, geometry_features: int, motion_features: int,
        time_features: int, width: int,
    ) -> None:
        super().__init__()
        self.geometry = nn.Linear(geometry_features, width, bias=False)
        self.motion = nn.Linear(motion_features, width, bias=False)
        self.time = nn.Linear(time_features, width, bias=False)
        self.post = nn.Sequential(
            nn.LayerNorm(width, elementwise_affine=False),
            nn.Linear(width, width, bias=False), nn.SiLU(),
        )

    def forward(
        self,
        geometry: torch.Tensor,
        motion: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        interaction = (
            F.silu(self.geometry(geometry))
            * F.silu(self.motion(motion))
            * F.silu(self.time(time))
        )
        return self.post(interaction)


class _StateTimeTranslationDecoder(nn.Module):
    """Learn translation from velocity and time only, with no geometry path."""

    def __init__(self, time_features: int, width: int) -> None:
        super().__init__()
        self.time = nn.Sequential(
            nn.Linear(time_features, width), nn.LayerNorm(width), nn.SiLU(),
        )
        self.velocity = nn.Linear(3, width, bias=False)
        self.output = nn.Linear(width, 3, bias=False)

    def forward(
        self, velocity: torch.Tensor, time_context: torch.Tensor,
    ) -> torch.Tensor:
        if velocity.ndim != 2 or velocity.shape[-1] != 3:
            raise ValueError("translation decoder velocity shape differs")
        return self.output(
            F.silu(self.velocity(velocity))[:, None]
            * F.silu(self.time(time_context))
        )


class _OmegaGeometryDecoder(nn.Module):
    """Learn rotation from omega, prior geometry and time with triple gating."""

    def __init__(self, geometry_features: int, time_features: int, width: int) -> None:
        super().__init__()
        self.omega = nn.Linear(1, width, bias=False)
        self.geometry = nn.Linear(geometry_features, width, bias=False)
        self.time = nn.Sequential(
            nn.Linear(time_features, width), nn.LayerNorm(width), nn.SiLU(),
        )
        self.output = nn.Linear(width, 3, bias=False)

    def forward(
        self,
        omega: torch.Tensor,
        prior_geometry: torch.Tensor,
        time_context: torch.Tensor,
    ) -> torch.Tensor:
        if omega.ndim != 2 or omega.shape[-1] != 1:
            raise ValueError("rotation decoder omega shape differs")
        return self.output(
            F.silu(self.omega(omega))[:, None]
            * F.silu(self.geometry(prior_geometry))
            * F.silu(self.time(time_context))
        )


class GlobalFlowClosureHead(nn.Module):
    """Infer one twist and close all observed handle/pair displacement factors."""

    refinement_steps = 2

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.channels = int(channels)
        self.width = 4 * self.channels
        self.dropout = nn.Dropout(float(dropout))
        self.handle_initial_encoder = _FactorEncoder(26, self.width)
        self.pair_initial_encoder = _TripleGatedFactorEncoder(
            12, 6, 7, self.width,
        )
        self.handle_initial_state = nn.Sequential(
            nn.LayerNorm(self.width),
            nn.Linear(self.width, self.width), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(self.width, self.width), nn.SiLU(),
            nn.Linear(self.width, 4),
        )
        self.pair_initial_yaw = nn.Sequential(
            nn.LayerNorm(self.width, elementwise_affine=False),
            nn.Linear(self.width, self.width, bias=False), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(self.width, 1, bias=False),
        )
        # Prior contexts contain no observed displacement and no current
        # endpoint geometry.  They are sufficient for a learned PnP-aware
        # history model but cannot copy the reconstruction target.
        self.handle_translation_decoder = _StateTimeTranslationDecoder(
            8, self.width,
        )
        self.handle_rotation_decoder = _OmegaGeometryDecoder(
            3, 8, self.width,
        )
        self.pair_rotation_decoder = _OmegaGeometryDecoder(
            6, 6, self.width,
        )
        self.handle_residual_encoder = _FactorEncoder(17, self.width)
        self.pair_residual_encoder = _TripleGatedFactorEncoder(
            6, 3, 6, self.width,
        )
        self.handle_state_update = nn.Sequential(
            nn.LayerNorm(self.width + 4),
            nn.Linear(self.width + 4, self.width), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(self.width, 4),
        )
        self.pair_yaw_update = nn.Sequential(
            nn.LayerNorm(self.width, elementwise_affine=False),
            nn.Linear(self.width, self.width, bias=False), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(self.width, 1, bias=False),
        )
        self.log_variance = nn.Sequential(
            nn.LayerNorm(2 * self.width + 4),
            nn.Linear(2 * self.width + 4, self.channels), nn.SiLU(),
            nn.Linear(self.channels, 4),
        )

    @staticmethod
    def _masked_mean(
        value: torch.Tensor, valid: torch.Tensor,
    ) -> torch.Tensor:
        if value.shape[:2] != valid.shape:
            raise ValueError("factor value/mask shapes differ")
        weight = valid.unsqueeze(-1).to(value.dtype)
        return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1)

    @staticmethod
    def _prior_contexts(
        handle_geometry: torch.Tensor,
        handle_kinematics: torch.Tensor,
        pair_geometry: torch.Tensor,
        pair_kinematics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # handle geometry: current-centered, prior-centered, current-observed,
        # prior-observed.  Only the two prior fields enter the decoder.
        # Only the translation-invariant prior centered coordinate is allowed.
        # prior_obs is q0-anchored and can equal the negative reconstruction
        # target on q0-primary edges, so it is deliberately excluded.
        handle_geometry_prior = handle_geometry[..., 3:6]
        handle_time = handle_kinematics[..., 6:14]
        # pair geometry: current, prior, current-unit, prior-unit.  Current norm
        # is also excluded; the remaining metadata are elapsed/support fields.
        pair_geometry_prior = torch.cat((
            pair_geometry[..., 3:6], pair_geometry[..., 9:12],
        ), dim=-1)
        pair_time = pair_kinematics[..., 7:13]
        return (
            handle_geometry_prior, handle_time,
            pair_geometry_prior, pair_time,
        )

    def _decode_history(
        self,
        state: torch.Tensor,
        handle_geometry_prior: torch.Tensor,
        handle_time: torch.Tensor,
        pair_geometry_prior: torch.Tensor,
        pair_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        handle_translation = self.handle_translation_decoder(
            state[:, :3], handle_time,
        )
        handle_rotation = self.handle_rotation_decoder(
            state[:, 3:4], handle_geometry_prior, handle_time,
        )
        pair_rotation = self.pair_rotation_decoder(
            state[:, 3:4], pair_geometry_prior, pair_time,
        )
        return handle_translation + handle_rotation, pair_rotation

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
    ) -> dict[str, torch.Tensor]:
        if handle_geometry.shape[-1] != 12 or handle_kinematics.shape[-1] != 14:
            raise ValueError("handle closure factor shape differs")
        if pair_geometry.shape[-1] != 12 or pair_kinematics.shape[-1] != 13:
            raise ValueError("pair closure factor shape differs")
        if handle_geometry.shape[:2] != handle_valid.shape:
            raise ValueError("handle closure mask shape differs")
        if pair_geometry.shape[:2] != pair_valid.shape:
            raise ValueError("pair closure mask shape differs")
        steps = self.refinement_steps if refinement_steps is None else int(
            refinement_steps
        )
        if not 0 <= steps <= self.refinement_steps:
            raise ValueError("closure refinement count differs")

        handle_factor = self.handle_initial_encoder(torch.cat((
            handle_geometry, handle_kinematics,
        ), dim=-1))
        pair_factor = self.pair_initial_encoder(
            pair_geometry, pair_kinematics[..., :6], pair_kinematics[..., 6:13],
        )
        handle_pool = self._masked_mean(handle_factor, handle_valid)
        pair_pool = self._masked_mean(pair_factor, pair_valid)
        logits = self.handle_initial_state(handle_pool)
        logits = torch.cat((
            logits[:, :3],
            logits[:, 3:4] + self.pair_initial_yaw(pair_pool),
        ), dim=-1)
        initial_state = torch.tanh(logits)
        (
            handle_geometry_prior, handle_time,
            pair_geometry_prior, pair_time,
        ) = self._prior_contexts(
            handle_geometry, handle_kinematics, pair_geometry, pair_kinematics,
        )
        handle_prior = torch.cat((handle_geometry_prior, handle_time), dim=-1)
        pair_prior = torch.cat((pair_geometry_prior, pair_time), dim=-1)
        observed_handle = handle_kinematics[..., :3]
        observed_pair = pair_kinematics[..., :3]
        states = [initial_state]
        handle_prediction = torch.zeros_like(observed_handle)
        pair_prediction = torch.zeros_like(observed_pair)
        handle_residual_pool = torch.zeros_like(handle_pool)
        pair_residual_pool = torch.zeros_like(pair_pool)
        for _ in range(steps):
            state = torch.tanh(logits)
            handle_prediction, pair_prediction = self._decode_history(
                state, handle_geometry_prior, handle_time,
                pair_geometry_prior, pair_time,
            )
            handle_residual = observed_handle - handle_prediction
            pair_residual = observed_pair - pair_prediction
            handle_message = self.handle_residual_encoder(torch.cat((
                handle_prior, handle_residual, handle_prediction,
            ), dim=-1))
            pair_message = self.pair_residual_encoder(
                pair_geometry_prior, pair_residual, pair_time,
            )
            handle_residual_pool = self._masked_mean(
                handle_message, handle_valid,
            )
            pair_residual_pool = self._masked_mean(pair_message, pair_valid)
            handle_update = self.handle_state_update(torch.cat((
                handle_residual_pool, state,
            ), dim=-1))
            pair_yaw_update = self.pair_yaw_update(pair_residual_pool)
            update = torch.cat((
                handle_update[:, :3],
                handle_update[:, 3:4] + pair_yaw_update,
            ), dim=-1)
            logits = logits + 0.5 * torch.tanh(update)
            states.append(torch.tanh(logits))

        state = torch.tanh(logits)
        # Always report closure at the returned state, including the zero-step
        # intervention used to prove that recurrent history closure matters.
        handle_prediction, pair_prediction = self._decode_history(
            state, handle_geometry_prior, handle_time,
            pair_geometry_prior, pair_time,
        )
        handle_residual = observed_handle - handle_prediction
        pair_residual = observed_pair - pair_prediction
        uncertainty = self.log_variance(torch.cat((
            handle_residual_pool, pair_residual_pool, state,
        ), dim=-1)).clamp(-5.0, 5.0)
        return {
            "motion_state_normalized": state,
            "motion_log_variance": uncertainty,
            "initial_motion_state_normalized": initial_state,
            "iteration_motion_state_normalized": torch.stack(states, dim=1),
            "handle_closure_prediction_normalized": handle_prediction,
            "pair_closure_prediction_normalized": pair_prediction,
            "handle_closure_residual_normalized": handle_residual,
            "pair_closure_residual_normalized": pair_residual,
            "handle_factor_valid": handle_valid,
            "pair_factor_valid": pair_valid,
            "pair_supported": pair_valid.any(dim=1),
            "refinement_steps": state.new_full((state.shape[0],), steps),
        }


class AnonymousGlobalFlowClosureProbe(AnonymousPairedResidualTwistProbe):
    """V11 global learned-history-closure probe with frozen future modules."""

    model_family = "anonymous-global-flow-closure-probe-v11"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.motion_state_head = GlobalFlowClosureHead(
            self.channels, self.dropout,
        )

    @property
    def config(self) -> dict[str, Any]:
        parent = dict(super().config)
        parent.update({
            "family": self.model_family,
            "motion_state_fusion": (
                "one global twist refined twice by learned observed-history "
                "handle and pair closure residuals"
            ),
            "local_yaw_votes": False,
            "pair_support_hard_switch": False,
            "observed_history_decoder": True,
            "history_decoder_reads_observed_displacement": False,
            "history_decoder_reads_current_endpoint": False,
            "analytic_future_decoder": False,
            "physical_id_input": False,
            "motion_class_input": False,
            "session_identity_input": False,
            "truth_state_input": False,
        })
        return parent

    @staticmethod
    def _roll_raw_geometry(
        history: dict[str, torch.Tensor], *, handle: bool, event_count: int,
    ) -> dict[str, torch.Tensor]:
        result = dict(history)
        if handle:
            geometry_name, valid_name, streams = (
                "_handle_geometry_raw", "_handle_raw_valid", 4,
            )
        else:
            geometry_name, valid_name, streams = (
                "_pair_geometry_raw", "_pair_raw_valid", 1,
            )
        geometry = history[geometry_name]
        valid = history[valid_name]
        grouped_geometry = geometry.reshape(
            geometry.shape[0], streams, event_count,
            len(LOCAL_LAG_SCALES_S), geometry.shape[-1],
        )
        grouped_valid = valid.reshape(
            valid.shape[0], streams, event_count, len(LOCAL_LAG_SCALES_S),
        )
        # Reuse the audited anonymous within-stream/event-scale intervention.
        result[geometry_name] = (
            AnonymousGlobalFlowClosureProbe._roll_grouped(
                grouped_geometry, grouped_valid,
            ).reshape_as(geometry)
        )
        return result

    @staticmethod
    def _roll_grouped(
        geometry: torch.Tensor, valid: torch.Tensor,
    ) -> torch.Tensor:
        # Kept as a model-level wrapper so interventions can break handle and
        # pair geometry independently without changing V9/V10 public behavior.
        from .paired_twist_set_future import AnonymousPairedTwistTokenContext
        return AnonymousPairedTwistTokenContext._roll_grouped_geometry(
            geometry, valid,
        )

    def _estimate_closure(
        self,
        fields: dict[str, torch.Tensor],
        *,
        break_handle_geometry: bool = False,
        break_pair_geometry: bool = False,
        refinement_steps: int | None = None,
        pair_source_index: torch.Tensor | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        history = self.context(**fields)
        event_count = int(fields["history_event_mask"].shape[1])
        if break_handle_geometry:
            history = self._roll_raw_geometry(
                history, handle=True, event_count=event_count,
            )
        if break_pair_geometry:
            history = self._roll_raw_geometry(
                history, handle=False, event_count=event_count,
            )
        if pair_source_index is not None:
            source = pair_source_index.to(
                device=history["_pair_raw_valid"].device, dtype=torch.long,
            )
            if source.shape != (history["_pair_raw_valid"].shape[0],):
                raise ValueError("crossed pair source shape differs")
            crossed_valid = history["_pair_raw_valid"].index_select(0, source)
            if not torch.equal(crossed_valid, history["_pair_raw_valid"]):
                raise ValueError("crossed pair source support differs")
            history = dict(history)
            for name in ("_pair_geometry_raw", "_pair_kinematics_raw"):
                history[name] = history[name].index_select(0, source)
        state = self.motion_state_head(
            history["_handle_geometry_raw"],
            history["_handle_kinematics_raw"],
            history["_handle_raw_valid"],
            history["_pair_geometry_raw"],
            history["_pair_kinematics_raw"],
            history["_pair_raw_valid"],
            refinement_steps=refinement_steps,
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

    @staticmethod
    def _cross_rotation_factors(
        history: dict[str, torch.Tensor],
        source: torch.Tensor,
        *,
        event_count: int,
        history_scale_s: float,
    ) -> dict[str, torch.Tensor]:
        """Resynthesize target common flow on the donor rotation time axis."""
        result = dict(history)
        scale_count = len(LOCAL_LAG_SCALES_S)
        target_valid = history["_handle_raw_valid"].reshape(
            -1, 4, event_count, scale_count,
        )
        handle_geometry = history["_handle_geometry_raw"].reshape(
            -1, 4, event_count, scale_count, 12,
        )
        handle_kinematics = history["_handle_kinematics_raw"].reshape(
            -1, 4, event_count, scale_count, 14,
        )
        donor_geometry = handle_geometry.index_select(0, source)
        donor_kinematics = handle_kinematics.index_select(0, source)
        donor_valid = target_valid.index_select(0, source)
        target_weight = target_valid.unsqueeze(-1).to(handle_kinematics.dtype)
        target_common_rate = (
            handle_kinematics[..., 3:6] * target_weight
        ).sum(dim=(1, 2, 3)) / target_weight.sum(dim=(1, 2, 3)).clamp_min(1)
        elapsed_s = 0.01 * torch.expm1(donor_kinematics[..., 6]).clamp_min(0)
        elapsed_normalized = elapsed_s / float(history_scale_s)
        donor_centered_delta = donor_geometry[..., :3] - donor_geometry[..., 3:6]
        hybrid_delta = (
            target_common_rate[:, None, None, None]
            * elapsed_normalized.unsqueeze(-1)
        ) + donor_centered_delta
        hybrid_rate = hybrid_delta / elapsed_normalized.clamp_min(1e-7).unsqueeze(-1)

        target_center = handle_geometry[..., 6:9] - handle_geometry[..., :3]
        q0_estimate = target_center - (
            target_common_rate[:, None, None, None]
            * handle_kinematics[..., 7].unsqueeze(-1)
        )
        q0_weight = target_weight
        q0_center = (
            q0_estimate * q0_weight
        ).sum(dim=(1, 2, 3)) / q0_weight.sum(dim=(1, 2, 3))
        donor_current_time_normalized = donor_kinematics[..., 7]
        hybrid_current_center = q0_center[:, None, None, None] + (
            target_common_rate[:, None, None, None]
            * donor_current_time_normalized.unsqueeze(-1)
        )
        hybrid_prior_center = hybrid_current_center - (
            target_common_rate[:, None, None, None]
            * elapsed_normalized.unsqueeze(-1)
        )
        hybrid_geometry = donor_geometry.clone()
        hybrid_geometry[..., 6:9] = (
            hybrid_current_center + donor_geometry[..., :3]
        )
        hybrid_geometry[..., 9:12] = (
            hybrid_prior_center + donor_geometry[..., 3:6]
        )
        hybrid_kinematics = donor_kinematics.clone()
        hybrid_kinematics[..., :3] = hybrid_delta
        hybrid_kinematics[..., 3:6] = hybrid_rate
        result["_handle_geometry_raw"] = hybrid_geometry.reshape_as(
            history["_handle_geometry_raw"]
        )
        result["_handle_kinematics_raw"] = hybrid_kinematics.reshape_as(
            history["_handle_kinematics_raw"]
        )
        result["_handle_raw_valid"] = donor_valid.reshape_as(
            history["_handle_raw_valid"]
        )
        for name in (
            "_pair_geometry_raw", "_pair_kinematics_raw", "_pair_raw_valid",
        ):
            result[name] = history[name].index_select(0, source)
        return result

    @staticmethod
    def _break_crossed_rotation_pairing(
        history: dict[str, torch.Tensor], *, event_count: int,
    ) -> dict[str, torch.Tensor]:
        """Break donor geometry/differential correspondence inside a hybrid."""
        result = dict(history)
        scale_count = len(LOCAL_LAG_SCALES_S)
        handle_geometry = history["_handle_geometry_raw"].reshape(
            -1, 4, event_count, scale_count, 12,
        )
        handle_valid = history["_handle_raw_valid"].reshape(
            -1, 4, event_count, scale_count,
        )
        target_current_center = handle_geometry[..., 6:9] - handle_geometry[..., :3]
        target_prior_center = handle_geometry[..., 9:12] - handle_geometry[..., 3:6]
        centered = AnonymousGlobalFlowClosureProbe._roll_grouped(
            handle_geometry[..., :6], handle_valid,
        )
        broken_handle = handle_geometry.clone()
        broken_handle[..., :6] = centered
        broken_handle[..., 6:9] = target_current_center + centered[..., :3]
        broken_handle[..., 9:12] = target_prior_center + centered[..., 3:6]
        result["_handle_geometry_raw"] = broken_handle.reshape_as(
            history["_handle_geometry_raw"]
        )
        pair_geometry = history["_pair_geometry_raw"].reshape(
            -1, 1, event_count, scale_count, 12,
        )
        pair_valid = history["_pair_raw_valid"].reshape(
            -1, 1, event_count, scale_count,
        )
        result["_pair_geometry_raw"] = (
            AnonymousGlobalFlowClosureProbe._roll_grouped(
                pair_geometry, pair_valid,
            ).reshape_as(history["_pair_geometry_raw"])
        )
        return result

    def _state_from_raw_history(
        self, history: dict[str, torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor]]:
        state = self.motion_state_head(
            history["_handle_geometry_raw"],
            history["_handle_kinematics_raw"],
            history["_handle_raw_valid"],
            history["_pair_geometry_raw"],
            history["_pair_kinematics_raw"],
            history["_pair_raw_valid"],
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

    def estimate_motion_state(
        self,
        *,
        history_obs_rel_m: torch.Tensor,
        history_obs_mask: torch.Tensor,
        history_primary_mask: torch.Tensor,
        history_event_mask: torch.Tensor,
        history_time_s: torch.Tensor,
        history_switch_step: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        return self._estimate_closure({
            "history_obs_rel_m": history_obs_rel_m,
            "history_obs_mask": history_obs_mask,
            "history_primary_mask": history_primary_mask,
            "history_event_mask": history_event_mask,
            "history_time_s": history_time_s,
            "history_switch_step": history_switch_step,
        })

    def estimate_motion_state_broken_handle_geometry(self, **fields):
        return self._estimate_closure(fields, break_handle_geometry=True)

    def estimate_motion_state_broken_pair_geometry(self, **fields):
        return self._estimate_closure(fields, break_pair_geometry=True)

    def estimate_motion_state_zero_refinement(self, **fields):
        return self._estimate_closure(fields, refinement_steps=0)

    def estimate_motion_state_crossed_pair_factors(
        self, pair_source_index: torch.Tensor, **fields,
    ):
        return self._estimate_closure(
            fields, pair_source_index=pair_source_index,
        )

    def estimate_motion_state_crossed_rotation_factors(
        self, rotation_source_index: torch.Tensor, **fields,
    ):
        history = self.context(**fields)
        source = rotation_source_index.to(
            device=history["_handle_raw_valid"].device, dtype=torch.long,
        )
        if source.shape != (history["_handle_raw_valid"].shape[0],):
            raise ValueError("crossed rotation source shape differs")
        history = self._cross_rotation_factors(
            history, source,
            event_count=int(fields["history_event_mask"].shape[1]),
            history_scale_s=float(self.context.history_scale_s),
        )
        return self._state_from_raw_history(history)

    def estimate_motion_state_crossed_rotation_broken_pairing(
        self, rotation_source_index: torch.Tensor, **fields,
    ):
        history = self.context(**fields)
        source = rotation_source_index.to(
            device=history["_handle_raw_valid"].device, dtype=torch.long,
        )
        if source.shape != (history["_handle_raw_valid"].shape[0],):
            raise ValueError("crossed rotation source shape differs")
        event_count = int(fields["history_event_mask"].shape[1])
        history = self._cross_rotation_factors(
            history, source, event_count=event_count,
            history_scale_s=float(self.context.history_scale_s),
        )
        history = self._break_crossed_rotation_pairing(
            history, event_count=event_count,
        )
        return self._state_from_raw_history(history)


def _masked_closure_loss(
    residual: torch.Tensor, valid: torch.Tensor,
) -> torch.Tensor:
    coordinate = F.smooth_l1_loss(
        residual, torch.zeros_like(residual), reduction="none", beta=0.02,
    ).mean(dim=-1)
    supported = valid.any(dim=1)
    if bool(supported.any()):
        per_sample = (
            coordinate * valid.to(coordinate.dtype)
        ).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return per_sample[supported].mean()
    return coordinate.new_zeros(())


def global_flow_closure_state_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise one global state and learned reconstruction of causal history."""
    base, components = paired_twist_state_loss(prediction, batch)
    handle_closure = _masked_closure_loss(
        prediction["handle_closure_residual_normalized"],
        prediction["handle_factor_valid"],
    )
    pair_closure = _masked_closure_loss(
        prediction["pair_closure_residual_normalized"],
        prediction["pair_factor_valid"],
    )
    target = batch["target_motion_state_normalized"]
    initial = F.smooth_l1_loss(
        prediction["initial_motion_state_normalized"], target, beta=0.1,
    )
    objective = base + 0.20 * handle_closure + 0.20 * pair_closure + 0.10 * initial
    result = dict(components)
    result.update({
        "objective": objective,
        "handle_history_closure": handle_closure,
        "pair_history_closure": pair_closure,
        "initial_state": initial,
    })
    return objective, result


def global_flow_closure_train_step(
    model: StableMotionBottleneckAnonymousFutureModel,
    batch: dict[str, torch.Tensor],
    stage_update: int,
    stage_total: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    if int(stage_total) != 200 or not 1 <= int(stage_update) <= 200:
        raise ValueError("v11 structural probe is fixed to 200 updates")
    field_names = AnonymousGlobalFlowClosureProbe._field_names()
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
    original_loss, original_components = global_flow_closure_state_loss(
        original, batch,
    )
    augmented_loss, augmented_components = global_flow_closure_state_loss(
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
        "vertical_velocity", "scale_heteroscedastic", "handle_history_closure",
        "pair_history_closure", "initial_state",
    ):
        components[name] = 0.5 * (
            original_components[name] + augmented_components[name]
        )
    components.update({
        "objective": objective,
        "ramp_yaw_invariance": yaw_invariance,
        "ramp_translation_equivariance": translation_equivariance,
        "state_substage": "global_flow_closure_structural_probe",
        "state_substage_endpoint": False,
    })
    return original, objective, components


__all__ = [
    "GlobalFlowClosureHead",
    "AnonymousGlobalFlowClosureProbe",
    "global_flow_closure_state_loss",
    "global_flow_closure_train_step",
]
