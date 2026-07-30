"""Anonymous paired-residual state estimator (V10 structural probe).

V9 mixed complete four-dimensional proposals with one expert weight.  V10
keeps the same six causal observation fields and anonymous local-edge encoder,
but gives translation and rotation distinct observable subspaces.  All handle
edges contribute to one coherent 3D velocity baseline.  Every available
event/scale pair edge is bundled with its matching handle summary through a
multiplicative interaction; those bundles estimate yaw and a planar correction
to the velocity baseline.  No physical identity, motion class, truth or future
field enters inference.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .factorized_common_relative_motion_future import apply_common_velocity_ramp
from .joint_rigid_flow_probe import LOCAL_LAG_SCALES_S, _deterministic_probe_ramp
from .paired_twist_set_future import (
    AnonymousPairedTwistSetProbe,
    _CrossAttentionBlock,
    paired_twist_state_loss,
)
from .stable_motion_bottleneck_future import StableMotionBottleneckAnonymousFutureModel


class _ZeroTokenEncoder(nn.Module):
    """Keep the inherited V9 raw builder without retaining dead encoders."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = int(width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.new_zeros((*value.shape[:-1], self.width))


class PairedResidualTwistHead(nn.Module):
    """Estimate translation, angular motion and a paired planar correction."""

    def __init__(self, channels: int, dropout: float, message_layers: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.width = 2 * self.channels
        self.query = nn.Parameter(torch.empty(1, self.width))
        nn.init.normal_(self.query, std=0.02)
        self.blocks = nn.ModuleList(
            _CrossAttentionBlock(self.width, dropout)
            for _ in range(int(message_layers))
        )
        self.common_kinematics_encoder = nn.Sequential(
            nn.Linear(14, self.width), nn.LayerNorm(self.width), nn.SiLU(),
            nn.Linear(self.width, self.width), nn.SiLU(),
        )
        # Bias-free, zero-preserving projections make a one-sided constant
        # bypass impossible on the paired yaw/correction path.
        self.handle_geometry_projection = nn.Linear(12, self.width, bias=False)
        self.handle_kinematics_projection = nn.Linear(14, self.width, bias=False)
        self.pair_geometry_projection = nn.Linear(12, self.width, bias=False)
        self.pair_kinematics_projection = nn.Linear(13, self.width, bias=False)
        self.bundle_norm = nn.LayerNorm(self.width, elementwise_affine=False)
        self.common_velocity = nn.Sequential(
            nn.LayerNorm(self.width),
            nn.Linear(self.width, self.channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(self.channels, 3),
        )
        self.fallback_yaw = nn.Sequential(
            nn.LayerNorm(self.width),
            nn.Linear(self.width, self.channels), nn.SiLU(),
            nn.Linear(self.channels, 1),
        )
        self.pair_yaw_vote = nn.Sequential(
            nn.LayerNorm(self.width, elementwise_affine=False),
            nn.Linear(self.width, self.channels, bias=False), nn.SiLU(),
            nn.Linear(self.channels, 2, bias=False),
        )
        # The correction receives only the multiplicative subspace interaction;
        # geometry or velocity marginals cannot bypass their learned pairing.
        self.paired_correction = nn.Sequential(
            nn.LayerNorm(self.width, elementwise_affine=False),
            nn.Linear(self.width, self.channels, bias=False), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(self.channels, 3, bias=False),
        )
        self.log_variance = nn.Sequential(
            nn.LayerNorm(2 * self.width),
            nn.Linear(2 * self.width, self.channels), nn.SiLU(),
            nn.Linear(self.channels, 4),
        )

    @staticmethod
    def _masked_weights(score: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if score.shape != valid.shape:
            raise ValueError("paired residual reliability fields differ")
        supported = valid.any(dim=1)
        safe_valid = valid.clone()
        safe_valid[~supported, 0] = True
        logits = score.float().masked_fill(~safe_valid, -torch.inf)
        weight = torch.softmax(logits, dim=1).to(score.dtype)
        return weight * supported[:, None].to(weight.dtype)

    def forward(
        self,
        handle_geometry: torch.Tensor,
        handle_kinematics: torch.Tensor,
        handle_valid: torch.Tensor,
        pair_geometry: torch.Tensor,
        pair_kinematics: torch.Tensor,
        pair_valid: torch.Tensor,
        *,
        event_count: int,
    ) -> dict[str, torch.Tensor]:
        if handle_geometry.ndim != 3 or pair_geometry.ndim != 3:
            raise ValueError("paired residual raw token ranks differ")
        batch = handle_geometry.shape[0]
        if event_count < 1:
            raise ValueError("paired residual event support differs")
        scale_count = len(LOCAL_LAG_SCALES_S)
        handle_count = 4 * int(event_count) * scale_count
        pair_count = int(event_count) * scale_count
        if (
            handle_geometry.shape != (batch, handle_count, 12)
            or handle_kinematics.shape != (batch, handle_count, 14)
            or handle_valid.shape != (batch, handle_count)
            or pair_geometry.shape != (batch, pair_count, 12)
            or pair_kinematics.shape != (batch, pair_count, 13)
            or pair_valid.shape != (batch, pair_count)
        ):
            raise ValueError("paired residual raw token layout differs")

        common_token = self.common_kinematics_encoder(handle_kinematics)
        handle_interaction = F.silu(
            self.handle_geometry_projection(handle_geometry)
        ) * F.silu(self.handle_kinematics_projection(handle_kinematics))
        pair_interaction = F.silu(
            self.pair_geometry_projection(pair_geometry)
        ) * F.silu(self.pair_kinematics_projection(pair_kinematics))
        grouped_handle = handle_interaction.reshape(
            batch, 4, event_count, scale_count, self.width,
        ).permute(0, 2, 3, 1, 4)
        grouped_handle_valid = handle_valid.reshape(
            batch, 4, event_count, scale_count,
        ).permute(0, 2, 3, 1)
        handle_support = grouped_handle_valid.sum(dim=3)
        handle_summary = (
            grouped_handle
            * grouped_handle_valid.unsqueeze(-1).to(grouped_handle.dtype)
        ).sum(dim=3) / handle_support.clamp_min(1).unsqueeze(-1)
        pair_interaction = pair_interaction.reshape(
            batch, event_count, scale_count, self.width,
        )
        pair_valid = pair_valid.reshape(batch, event_count, scale_count)
        bundle_valid = pair_valid & (handle_support > 0)
        bundle = self.bundle_norm(handle_summary * pair_interaction)
        bundle = torch.where(
            bundle_valid.unsqueeze(-1), bundle, torch.zeros_like(bundle),
        )
        bundle = bundle.reshape(batch, pair_count, self.width)
        bundle_valid = bundle_valid.reshape(batch, pair_count)

        pair_supported = bundle_valid.any(dim=1)
        latent = self.query[None].expand(batch, -1, -1)
        for block in self.blocks:
            latent = block(latent, common_token, handle_valid[:, None])
        common_latent = latent[:, 0]

        common_logits = self.common_velocity(common_latent)
        vote = self.pair_yaw_vote(bundle)
        pair_weight = self._masked_weights(vote[..., 1], bundle_valid)
        paired_yaw_logit = (pair_weight * vote[..., 0]).sum(dim=1)
        pair_latent = (pair_weight.unsqueeze(-1) * bundle).sum(dim=1)
        fallback_yaw_logit = self.fallback_yaw(common_latent).squeeze(-1)
        yaw_base_logit = torch.where(
            pair_supported, paired_yaw_logit, fallback_yaw_logit,
        )
        subspace_interaction = common_latent * pair_latent
        correction = self.paired_correction(subspace_interaction)
        support = pair_supported.to(correction.dtype)
        planar_logits = common_logits[:, :2] + 0.5 * support[:, None] * correction[:, :2]
        velocity = torch.cat((
            torch.tanh(planar_logits), torch.tanh(common_logits[:, 2:3]),
        ), dim=1)
        yaw = torch.tanh(yaw_base_logit + 0.25 * support * correction[:, 2])
        state = torch.cat((velocity, yaw[:, None]), dim=1)
        zero_residual_state = torch.cat((
            torch.tanh(common_logits), torch.tanh(yaw_base_logit)[:, None],
        ), dim=1)
        uncertainty_input = torch.cat((common_latent, pair_latent), dim=1)
        log_variance = self.log_variance(uncertainty_input).clamp(-5.0, 5.0)
        return {
            "motion_state_normalized": state,
            "motion_log_variance": log_variance,
            "common_velocity_normalized": torch.tanh(common_logits),
            "paired_rotation_residual_normalized": state - zero_residual_state,
            "zero_rotation_residual_state_normalized": zero_residual_state,
            "pair_yaw_vote_normalized": torch.tanh(vote[..., 0]),
            "pair_yaw_reliability": pair_weight,
            "pair_bundle_valid": bundle_valid,
            "pair_supported": pair_supported,
        }


class AnonymousPairedResidualTwistProbe(AnonymousPairedTwistSetProbe):
    """V10 paired-residual state probe with the frozen learned future stack."""

    model_family = "anonymous-paired-residual-twist-probe-v10"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        width = int(self.context.width)
        self.context.handle_encoder = _ZeroTokenEncoder(width)
        self.context.pair_encoder = _ZeroTokenEncoder(width)
        del self.context.scale_embedding
        self.context.register_buffer(
            "scale_embedding", torch.zeros(len(LOCAL_LAG_SCALES_S), width),
        )
        del self.context.type_embedding
        self.context.register_buffer("type_embedding", torch.zeros(2, width))
        self.motion_state_head = PairedResidualTwistHead(
            self.channels, self.dropout, self.message_layers,
        )

    @property
    def config(self) -> dict[str, Any]:
        parent = dict(super().config)
        parent.update({
            "family": self.model_family,
            "motion_state_fusion": (
                "one full-history handle velocity baseline plus all-available-scale "
                "paired angular votes and multiplicative planar correction"
            ),
            "complete_4d_expert_mixture": False,
            "learned_expert_router": False,
            "translation_subspace": "one coherent 3D vector",
            "rotation_subspace": "one angular scalar plus planar paired residual",
            "pair1_pair2_pair3_all_consumed": True,
            "inherited_mixed_token_encoder_used": False,
            "paired_subspace_projection": (
                "four bias-free zero-preserving raw geometry/kinematics "
                "projections joined only by elementwise products"
            ),
            "analytic_future_decoder": False,
        })
        return parent

    def _estimate_residual(
        self,
        fields: dict[str, torch.Tensor],
        *,
        break_pairing: bool,
        zero_rotation_residual: bool,
    ) -> dict[str, dict[str, torch.Tensor]]:
        history = (
            self.context.forward_broken_pairing(**fields)
            if break_pairing else self.context(**fields)
        )
        state = self.motion_state_head(
            history["_handle_geometry_raw"],
            history["_handle_kinematics_raw"],
            history["_handle_raw_valid"],
            history["_pair_geometry_raw"],
            history["_pair_kinematics_raw"],
            history["_pair_raw_valid"],
            event_count=int(fields["history_event_mask"].shape[1]),
        )
        if zero_rotation_residual:
            state["motion_state_normalized"] = state[
                "zero_rotation_residual_state_normalized"
            ]
        state["motion_state_physical"] = (
            state["motion_state_normalized"]
            * self.motion_state_scale.to(state["motion_state_normalized"].dtype)
        )
        public_history = {
            name: value for name, value in history.items() if not name.startswith("_")
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
        return self._estimate_residual({
            "history_obs_rel_m": history_obs_rel_m,
            "history_obs_mask": history_obs_mask,
            "history_primary_mask": history_primary_mask,
            "history_event_mask": history_event_mask,
            "history_time_s": history_time_s,
            "history_switch_step": history_switch_step,
        }, break_pairing=False, zero_rotation_residual=False)

    def estimate_motion_state_broken_pairing(self, **fields: torch.Tensor):
        return self._estimate_residual(
            fields, break_pairing=True, zero_rotation_residual=False,
        )

    def estimate_motion_state_zero_rotation_residual(self, **fields: torch.Tensor):
        return self._estimate_residual(
            fields, break_pairing=False, zero_rotation_residual=True,
        )


def paired_residual_state_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Unified state loss plus identifiable pair-yaw and zero-bias calibration."""
    base, components = paired_twist_state_loss(prediction, batch)
    target = batch["target_motion_state_normalized"]
    pair_vote = prediction["pair_yaw_vote_normalized"]
    pair_valid = prediction["pair_bundle_valid"]
    vote_loss = F.smooth_l1_loss(
        pair_vote, target[:, None, 3].expand_as(pair_vote),
        reduction="none", beta=0.05,
    )
    supported = pair_valid.any(dim=1)
    per_sample = (
        vote_loss * pair_valid.to(vote_loss.dtype)
    ).sum(dim=1) / pair_valid.sum(dim=1).clamp_min(1)
    pair_yaw_aux = (
        per_sample[supported].mean() if bool(supported.any()) else base.new_zeros(())
    )
    yaw_bias = prediction["motion_state_normalized"][:, 3] - target[:, 3]
    yaw_calibration = F.smooth_l1_loss(
        yaw_bias.mean(), yaw_bias.new_zeros(()), beta=0.02,
    )
    objective = base + 0.15 * pair_yaw_aux + 0.10 * yaw_calibration
    result = dict(components)
    result.update({
        "objective": objective,
        "pair_yaw_aux": pair_yaw_aux,
        "yaw_calibration": yaw_calibration,
    })
    return objective, result


def paired_residual_probe_train_step(
    model: StableMotionBottleneckAnonymousFutureModel,
    batch: dict[str, torch.Tensor],
    stage_update: int,
    stage_total: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    """Fixed 200-update V10 step with deterministic common-velocity ramps."""
    if int(stage_total) != 200 or not 1 <= int(stage_update) <= 200:
        raise ValueError("v10 structural probe is fixed to 200 updates")
    field_names = AnonymousPairedResidualTwistProbe._field_names()
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
    original_loss, original_components = paired_residual_state_loss(original, batch)
    augmented_loss, augmented_components = paired_residual_state_loss(
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
        "vertical_velocity", "scale_heteroscedastic", "pair_yaw_aux",
        "yaw_calibration",
    ):
        components[name] = 0.5 * (
            original_components[name] + augmented_components[name]
        )
    components.update({
        "objective": objective,
        "ramp_yaw_invariance": yaw_invariance,
        "ramp_translation_equivariance": translation_equivariance,
        "state_substage": "paired_residual_structural_probe",
        "state_substage_endpoint": False,
    })
    return original, objective, components


__all__ = [
    "PairedResidualTwistHead",
    "AnonymousPairedResidualTwistProbe",
    "paired_residual_state_loss",
    "paired_residual_probe_train_step",
]
