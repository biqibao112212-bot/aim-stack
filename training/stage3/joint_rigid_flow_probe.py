"""Anonymous local rigid-flow probes for the post-v7 state redesign.

The probes keep the deployed six-field observation boundary and the existing
four-dimensional motion-state contract.  They deliberately replace v7's
long-lag projective-axis yaw path with local, same-visible-set differential
flow, while retaining a learned first-order common-flow fallback.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .continuous_invariant_anonymous_future import V3_FORWARD_FIELDS
from .robust_multiscale_motion_future import (
    RobustMultiScaleIncrementMotionContext,
    _RobustEdgeConsensus,
)
from .stable_motion_bottleneck_future import StableMotionBottleneckAnonymousFutureModel
from .factorized_common_relative_motion_future import apply_common_velocity_ramp
from .robust_multiscale_motion_future import robust_multiscale_motion_state_loss


LOCAL_LAG_SCALES_S = (0.01, 0.03, 0.07)


class AnonymousLocalRigidFlowContext(nn.Module):
    """Encode common flow, local pair differential flow and current geometry."""

    model_family = "anonymous-local-rigid-flow-context-v8-probe"

    def __init__(
        self,
        *,
        channels: int = 96,
        dropout: float = 0.05,
        message_layers: int = 3,
        position_scale_m: float = 1.0,
        history_scale_s: float = 0.32,
        lag_scales_s: tuple[float, ...] = LOCAL_LAG_SCALES_S,
    ) -> None:
        super().__init__()
        numeric_scales = tuple(float(value) for value in lag_scales_s)
        if numeric_scales != LOCAL_LAG_SCALES_S:
            raise ValueError("v8 probe is fixed to local 10/30/70-ms scales")
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.message_layers = int(message_layers)
        self.position_scale_m = float(position_scale_m)
        self.history_scale_s = float(history_scale_s)
        self.common_context = RobustMultiScaleIncrementMotionContext(
            channels=channels,
            dropout=dropout,
            message_layers=message_layers,
            position_scale_m=position_scale_m,
            history_scale_s=history_scale_s,
            lag_scales_s=numeric_scales,
        )
        self.register_buffer(
            "lag_scales_s", torch.tensor(numeric_scales, dtype=torch.float32),
        )
        self.pair_scale_embedding = nn.Parameter(
            torch.empty(len(numeric_scales), channels),
        )
        nn.init.normal_(self.pair_scale_embedding, std=0.02)
        # Unit current/prior pair vectors, differential flow, cross/dot,
        # magnitudes, elapsed/time/switch/lag-ratio = 19 fields.
        self.pair_flow_consensus = _RobustEdgeConsensus(19, channels, dropout)
        self.pair_flow_projection = nn.Sequential(
            nn.Linear(2 * channels, 2 * channels),
            nn.LayerNorm(2 * channels), nn.SiLU(),
            nn.Linear(2 * channels, 2 * channels), nn.SiLU(),
        )
        # Current primary->secondary vector, symmetric moment, count and mask.
        self.geometry_projection = nn.Sequential(
            nn.Linear(11, 2 * channels), nn.LayerNorm(2 * channels), nn.SiLU(),
            nn.Linear(2 * channels, 2 * channels), nn.SiLU(),
        )

    @property
    def config(self) -> dict[str, Any]:
        return {
            "family": self.model_family,
            "channels": self.channels,
            "dropout": self.dropout,
            "message_layers": self.message_layers,
            "position_scale_m": self.position_scale_m,
            "history_scale_s": self.history_scale_s,
            "lag_scales_s": self.lag_scales_s.detach().cpu().tolist(),
            "common_flow": "v6 learned robust same-handle local increments",
            "pair_flow": (
                "primary-oriented two-handle differential flow; same visible set only"
            ),
            "pair_orientation": (
                "current-primary-oriented within the exact same visible set; "
                "reverse only on primary swap; no persistent physical identity"
            ),
            "long_projective_yaw_lags": False,
            "curvature_fallback": False,
            "physical_id_input": False,
            "session_or_motion_class_input": False,
            "absolute_position_or_range_input": False,
            "q0_geometry_or_quality_input": False,
            "truth_or_future_input": False,
            "c4_invariant_output": True,
        }

    def _same_set_lag_bank(
        self,
        pair_vector: torch.Tensor,
        time_s: torch.Tensor,
        pair_valid: torch.Tensor,
        visible: torch.Tensor,
        primary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, events, dimensions = pair_vector.shape
        if dimensions != 3:
            raise ValueError("pair vector must be three-dimensional")
        current_index = torch.arange(events, device=pair_vector.device)[None, :, None]
        prior_index = torch.arange(events, device=pair_vector.device)[None, None, :]
        elapsed_all = time_s[:, :, None] - time_s[:, None, :]
        same_set = (
            visible[:, :, None, :] == visible[:, None, :, :]
        ).all(dim=-1)
        causal = (
            pair_valid[:, :, None] & pair_valid[:, None, :]
            & same_set & (prior_index < current_index) & (elapsed_all > 1e-7)
        )
        scales = self.lag_scales_s.to(dtype=pair_vector.dtype, device=pair_vector.device)
        middle = torch.sqrt(scales[:-1] * scales[1:])
        lower = torch.cat((scales[:1] * 0.5, middle))
        upper = torch.cat((middle, scales[-1:] * 1.5))
        candidate = (
            causal.unsqueeze(-1)
            & (elapsed_all[:, :, :, None] >= lower[None, None, None])
            & (elapsed_all[:, :, :, None] < upper[None, None, None])
        )
        ratio = elapsed_all[:, :, :, None] / scales[None, None, None]
        cost = torch.log(ratio.clamp_min(1e-7)).abs()
        cost = torch.where(candidate, cost, torch.full_like(cost, torch.inf))
        selected = cost.argmin(dim=2)
        edge_valid = candidate.any(dim=2)
        selected_safe = selected.clamp(0, max(events - 1, 0))
        prior = pair_vector[:, None].expand(-1, events, -1, -1).gather(
            2, selected_safe.unsqueeze(-1).expand(-1, -1, -1, dimensions),
        )
        elapsed = elapsed_all[:, :, :, None].expand(
            -1, -1, -1, len(scales),
        ).gather(2, selected_safe.unsqueeze(2)).squeeze(2)
        elapsed = torch.where(edge_valid, elapsed, torch.zeros_like(elapsed))
        prior_primary = primary[:, None].expand(-1, events, -1, -1).gather(
            2, selected_safe.unsqueeze(-1).expand(-1, -1, -1, 4),
        )
        current_primary = primary[:, :, None].expand(-1, -1, len(scales), -1)
        same_primary = (current_primary & prior_primary).any(dim=-1)
        orientation = torch.where(
            same_primary.unsqueeze(-1), torch.ones_like(prior[..., :1]),
            -torch.ones_like(prior[..., :1]),
        )
        aligned_prior = prior * orientation
        return aligned_prior, elapsed, edge_valid, selected

    @staticmethod
    def _symmetric_moment(vector: torch.Tensor) -> torch.Tensor:
        outer = torch.einsum("...i,...j->...ij", vector, vector)
        return torch.stack((
            outer[..., 0, 0], outer[..., 1, 1], outer[..., 2, 2],
            outer[..., 0, 1], outer[..., 0, 2], outer[..., 1, 2],
        ), dim=-1)

    def forward(
        self,
        history_obs_rel_m: torch.Tensor,
        history_obs_mask: torch.Tensor,
        history_primary_mask: torch.Tensor,
        history_event_mask: torch.Tensor,
        history_time_s: torch.Tensor,
        history_switch_step: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        common = self.common_context(
            history_obs_rel_m, history_obs_mask, history_primary_mask,
            history_event_mask, history_time_s, history_switch_step,
        )
        active = history_event_mask.to(torch.bool)
        visible = history_obs_mask.to(torch.bool) & active.unsqueeze(-1)
        primary = history_primary_mask.to(torch.bool) & active.unsqueeze(-1)
        clean_obs = torch.where(
            visible.unsqueeze(-1), history_obs_rel_m, torch.zeros_like(history_obs_rel_m),
        )
        clean_time = torch.where(active, history_time_s, torch.zeros_like(history_time_s))
        primary_position = (
            clean_obs * primary.unsqueeze(-1).to(clean_obs.dtype)
        ).sum(dim=2)
        secondary = visible & ~primary
        secondary_position = (
            clean_obs * secondary.unsqueeze(-1).to(clean_obs.dtype)
        ).sum(dim=2)
        pair_event = (visible.sum(dim=2) == 2) & (secondary.sum(dim=2) == 1)
        pair_vector = torch.where(
            pair_event.unsqueeze(-1), secondary_position - primary_position,
            torch.zeros_like(primary_position),
        )
        aligned_prior, elapsed, edge_valid, prior_index = self._same_set_lag_bank(
            pair_vector, clean_time, pair_event, visible, primary,
        )
        current = pair_vector[:, :, None].expand_as(aligned_prior)
        current_norm = current.norm(dim=-1)
        prior_norm = aligned_prior.norm(dim=-1)
        unit_current = current / current_norm.clamp_min(1e-6).unsqueeze(-1)
        unit_prior = aligned_prior / prior_norm.clamp_min(1e-6).unsqueeze(-1)
        differential = torch.where(
            edge_valid.unsqueeze(-1),
            (current - aligned_prior) / elapsed.clamp_min(1e-7).unsqueeze(-1),
            torch.zeros_like(current),
        )
        cumulative = torch.cumsum(
            torch.where(active, history_switch_step, torch.zeros_like(history_switch_step)),
            dim=1,
        )
        prior_safe = prior_index.clamp_min(0)
        prior_cumulative = cumulative[:, None].expand(-1, clean_time.shape[1], -1).gather(
            2, prior_safe,
        )
        interval_switch = (cumulative[:, :, None] - prior_cumulative).abs()
        scales = self.lag_scales_s.to(dtype=clean_obs.dtype, device=clean_obs.device)
        pair_token = torch.cat((
            unit_current, unit_prior,
            differential * (self.history_scale_s / self.position_scale_m),
            torch.cross(unit_prior, unit_current, dim=-1),
            (unit_prior * unit_current).sum(dim=-1, keepdim=True),
            (current_norm / self.position_scale_m).unsqueeze(-1),
            (prior_norm / self.position_scale_m).unsqueeze(-1),
            torch.log1p(elapsed / 0.01).unsqueeze(-1),
            (clean_time[:, :, None] / self.history_scale_s).unsqueeze(-1).expand(
                -1, -1, len(scales), -1,
            ),
            (interval_switch / 6.0).unsqueeze(-1),
            (elapsed / scales[None, None]).unsqueeze(-1),
        ), dim=-1)
        pair_flow, pair_available, pair_mass, pair_ess = self.pair_flow_consensus(
            pair_token, edge_valid, self.pair_scale_embedding, elapsed, scales,
        )
        pair_flow = self.pair_flow_projection(pair_flow)
        pair_flow = torch.where(
            pair_available.unsqueeze(-1), pair_flow, torch.zeros_like(pair_flow),
        )

        last = self.common_context._last_active(active)
        rows = torch.arange(active.shape[0], device=active.device)
        current_pair = pair_vector[rows, last]
        current_pair_valid = pair_event[rows, last]
        scaled_pair = current_pair / self.position_scale_m
        geometry_feature = torch.cat((
            scaled_pair,
            self._symmetric_moment(scaled_pair),
            (visible[rows, last].sum(dim=1).to(clean_obs.dtype) / 2.0).unsqueeze(-1),
            current_pair_valid.to(clean_obs.dtype).unsqueeze(-1),
        ), dim=-1)
        geometry = self.geometry_projection(geometry_feature)
        handle_available = common["scale_handle_available"].any(dim=1)
        handle_reliability = torch.stack((
            torch.log1p(common["scale_handle_effective_sample_size"]),
            common["scale_handle_available"].sum(dim=1).to(clean_obs.dtype) / 4.0,
            torch.log1p(common["scale_handle_weight_mass"]),
            handle_available.to(clean_obs.dtype),
        ), dim=-1)
        reliability = torch.cat((
            handle_reliability,
            torch.log1p(pair_ess).unsqueeze(-1),
            pair_available.to(clean_obs.dtype).unsqueeze(-1),
            current_pair_valid[:, None].to(clean_obs.dtype).unsqueeze(-1).expand(
                -1, len(scales), -1,
            ),
        ), dim=-1)
        return {
            **common,
            "handle_only_scale_latent": common["scale_handle_vehicle_state"],
            "handle_only_coordinate_available": handle_available.unsqueeze(-1).expand(
                -1, -1, 4,
            ),
            "pair_flow_latent": pair_flow,
            "pair_flow_available": pair_available,
            "pair_flow_weight_mass": pair_mass,
            "pair_flow_effective_sample_size": pair_ess,
            "pair_flow_prior_index": prior_index,
            "pair_flow_edge_valid": edge_valid,
            "current_geometry_latent": geometry,
            "joint_reliability_feature": reliability,
            "current_pair_available": current_pair_valid,
        }


class RigidFlowProbeHead(nn.Module):
    """Separated-expert or joint twist head over the same local evidence."""

    def __init__(self, channels: int, dropout: float, *, variant: str) -> None:
        super().__init__()
        if variant not in {"separated", "joint"}:
            raise ValueError("unknown rigid-flow probe variant")
        self.variant = variant
        common_width = 4 * channels
        pair_width = 2 * channels
        geometry_width = 2 * channels
        full_width = common_width + pair_width + geometry_width + 7
        if variant == "separated":
            self.translation_state = nn.Sequential(
                nn.LayerNorm(common_width),
                nn.Linear(common_width, 2 * channels), nn.SiLU(),
                nn.Dropout(dropout), nn.Linear(2 * channels, 3), nn.Tanh(),
            )
            self.pair_yaw_state = nn.Sequential(
                nn.LayerNorm(pair_width + geometry_width + 3),
                nn.Linear(pair_width + geometry_width + 3, channels), nn.SiLU(),
                nn.Dropout(dropout), nn.Linear(channels, 1), nn.Tanh(),
            )
            self.fallback_yaw_state = nn.Sequential(
                nn.LayerNorm(common_width),
                nn.Linear(common_width, 2 * channels), nn.SiLU(),
                nn.Dropout(dropout), nn.Linear(2 * channels, 1), nn.Tanh(),
            )
            self.translation_log_variance = nn.Sequential(
                nn.LayerNorm(common_width), nn.Linear(common_width, channels), nn.SiLU(),
                nn.Linear(channels, 3),
            )
            self.yaw_log_variance = nn.Sequential(
                nn.LayerNorm(full_width), nn.Linear(full_width, channels), nn.SiLU(),
                nn.Linear(channels, 1),
            )
            self.translation_fusion = nn.Sequential(
                nn.LayerNorm(common_width + 10),
                nn.Linear(common_width + 10, channels), nn.SiLU(),
                nn.Dropout(dropout), nn.Linear(channels, 3),
            )
            self.yaw_fusion = nn.Sequential(
                nn.LayerNorm(full_width + 2),
                nn.Linear(full_width + 2, channels), nn.SiLU(),
                nn.Dropout(dropout), nn.Linear(channels, 1),
            )
        else:
            self.joint_state = nn.Sequential(
                nn.LayerNorm(full_width),
                nn.Linear(full_width, 3 * channels), nn.SiLU(),
                nn.Dropout(dropout), nn.Linear(3 * channels, channels), nn.SiLU(),
                nn.Linear(channels, 4), nn.Tanh(),
            )
            self.log_variance = nn.Sequential(
                nn.LayerNorm(full_width), nn.Linear(full_width, channels), nn.SiLU(),
                nn.Linear(channels, 4),
            )
            self.fusion = nn.Sequential(
                nn.LayerNorm(full_width + 8),
                nn.Linear(full_width + 8, channels), nn.SiLU(),
                nn.Dropout(dropout), nn.Linear(channels, 4),
            )

    def forward(
        self,
        common: torch.Tensor,
        pair: torch.Tensor,
        geometry: torch.Tensor,
        common_available: torch.Tensor,
        pair_available: torch.Tensor,
        reliability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if common.ndim != 3 or pair.ndim != 3 or geometry.ndim != 2:
            raise ValueError("rigid-flow head inputs have incompatible ranks")
        batch, scales = common.shape[:2]
        geometry_scale = geometry[:, None].expand(-1, scales, -1)
        pair_safe = torch.where(
            pair_available.unsqueeze(-1), pair, torch.zeros_like(pair),
        )
        full = torch.cat((common, pair_safe, geometry_scale, reliability), dim=-1)
        if self.variant == "separated":
            translation = self.translation_state(common)
            pair_yaw = self.pair_yaw_state(torch.cat((
                pair_safe, geometry_scale, reliability[..., -3:],
            ), dim=-1))
            fallback_yaw = self.fallback_yaw_state(common)
            yaw = torch.where(pair_available.unsqueeze(-1), pair_yaw, fallback_yaw)
            scale_state = torch.cat((translation, yaw), dim=-1)
            translation_log_variance = self.translation_log_variance(common).clamp(
                -5.0, 5.0,
            )
            yaw_log_variance = self.yaw_log_variance(full).clamp(-5.0, 5.0)
            scale_log_variance = torch.cat((
                translation_log_variance, yaw_log_variance,
            ), dim=-1)
            translation_logits = self.translation_fusion(torch.cat((
                common, translation, translation_log_variance,
                reliability[..., :4],
            ), dim=-1))
            yaw_logits = self.yaw_fusion(torch.cat((
                full, yaw, yaw_log_variance,
            ), dim=-1))
            logits = torch.cat((translation_logits, yaw_logits), dim=-1)
        else:
            scale_state = self.joint_state(full)
            scale_log_variance = self.log_variance(full).clamp(-5.0, 5.0)
            logits = self.fusion(torch.cat((
                full, scale_state, scale_log_variance,
            ), dim=-1))
        coordinate_available = common_available.clone()
        coordinate_available[..., 3] = (
            coordinate_available[..., 3] | pair_available
        )
        logits = logits.masked_fill(~coordinate_available, -torch.inf)
        weight = torch.softmax(logits.float(), dim=1).to(scale_state.dtype)
        fused = (weight * scale_state).sum(dim=1)
        fused_log_variance = (weight * scale_log_variance).sum(dim=1)
        return {
            "motion_state_normalized": fused,
            "scale_motion_state_normalized": scale_state,
            "scale_motion_log_variance": scale_log_variance,
            "scale_motion_weight": weight,
            "motion_log_variance": fused_log_variance,
            "scale_coordinate_available": coordinate_available,
        }


class AnonymousJointRigidFlowProbe(StableMotionBottleneckAnonymousFutureModel):
    """Base model for the two bounded post-v7 structural probes."""

    probe_variant = "joint"
    model_family = "anonymous-joint-rigid-flow-probe-v8"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.context = AnonymousLocalRigidFlowContext(
            channels=self.channels, dropout=self.dropout,
            message_layers=self.message_layers,
            position_scale_m=self.position_scale_m,
            history_scale_s=self.history_scale_s,
        )
        self.motion_state_head = RigidFlowProbeHead(
            self.channels, self.dropout, variant=self.probe_variant,
        )

    @property
    def config(self) -> dict[str, Any]:
        parent = dict(super().config)
        parent.update({
            "family": self.model_family,
            "probe_variant": self.probe_variant,
            "motion_context": self.context.config,
            "motion_state_fusion": (
                "separated local-pair/fallback experts" if self.probe_variant == "separated"
                else "joint common/differential/geometry twist"
            ),
            "decoder_temporal_input": "fused predicted 4D motion state only",
            "physical_id_input": False,
            "motion_class_input": False,
            "session_identity_input": False,
            "truth_state_input": False,
        })
        return parent

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
        history = self.context(
            history_obs_rel_m, history_obs_mask, history_primary_mask,
            history_event_mask, history_time_s, history_switch_step,
        )
        state = self.motion_state_head(
            history["handle_only_scale_latent"],
            history["pair_flow_latent"],
            history["current_geometry_latent"],
            history["handle_only_coordinate_available"],
            history["pair_flow_available"],
            history["joint_reliability_feature"],
        )
        state["motion_state_physical"] = (
            state["motion_state_normalized"] * self.motion_state_scale.to(
                state["motion_state_normalized"].dtype,
            )
        )
        return {"history": history, "state": state}

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        detach_motion_code: bool = True,
        detach_selector_context: bool = False,
    ) -> dict[str, torch.Tensor]:
        missing = set(V3_FORWARD_FIELDS) - set(batch)
        if missing:
            raise ValueError(f"v8 probe future fields missing: {sorted(missing)}")
        output = self.estimate_motion_state(**{
            name: batch[name] for name in (
                "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
                "history_event_mask", "history_time_s", "history_switch_step",
            )
        })
        history, state = output["history"], output["state"]
        motion = state["motion_state_normalized"]
        decoder_motion = motion.detach() if detach_motion_code else motion
        current = batch["current_position_m"]
        batch_size = current.shape[0]
        tau = self._tau(batch["tau_s"], batch_size)
        primary = history["primary_index"]
        relative_role = torch.arange(4, device=current.device)[None]
        ordered_handle = torch.remainder(primary[:, None] + relative_role, 4)
        q0_relation = batch["q0_relation_m"].gather(
            1, ordered_handle.unsqueeze(-1).expand(-1, -1, 3),
        ).clone()
        q0_relation[:, 0] = 0.0
        q0_supported = batch["q0_supported"].gather(1, ordered_handle)
        decoded = self.decode_ordered(
            current_position_m=current,
            tau_s=tau,
            ordered_q0_relation_m=q0_relation,
            ordered_q0_supported=q0_supported,
            motion_state_normalized=decoder_motion,
            detach_selector_context=detach_selector_context,
        )
        return {**history, **state, **decoded}


class AnonymousSeparatedRigidFlowProbe(AnonymousJointRigidFlowProbe):
    probe_variant = "separated"
    model_family = "anonymous-separated-rigid-flow-probe-v8"


class AnonymousJointTwistProbe(AnonymousJointRigidFlowProbe):
    probe_variant = "joint"
    model_family = "anonymous-joint-twist-probe-v8"


def _deterministic_probe_ramp(
    batch_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    stage_update: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a variant-independent ramp from only the run seed and update."""
    # torch.initial_seed() is the runner's immutable experiment seed; unlike
    # the current RNG state it is unaffected by architecture-specific dropout.
    seed = (
        int(torch.initial_seed())
        ^ 0x56485F52414D50
        ^ (int(stage_update) * 0x9E3779B97F4A7C15)
    ) % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    selected_cpu = torch.rand(batch_size, generator=generator) < 0.5
    if not bool(selected_cpu.any()):
        selected_cpu[0] = True
    ramp_cpu = torch.zeros(batch_size, 3, dtype=torch.float32)
    ramp_cpu[selected_cpu, :2] = (
        torch.rand(int(selected_cpu.sum()), 2, generator=generator) * 1.2 - 0.6
    )
    return selected_cpu.to(device=device), ramp_cpu.to(device=device, dtype=dtype)


def rigid_flow_probe_train_step(
    model: StableMotionBottleneckAnonymousFutureModel,
    batch: dict[str, torch.Tensor],
    stage_update: int,
    stage_total: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    """Common 200-update ramp contract for all three structural probe arms."""
    if int(stage_total) != 200 or not 1 <= int(stage_update) <= 200:
        raise ValueError("rigid-flow structural probe is fixed to 200 updates")
    field_names = (
        "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
        "history_event_mask", "history_time_s", "history_switch_step",
    )
    original_output = model.estimate_motion_state(**{
        name: batch[name] for name in field_names
    })
    original = {**original_output["history"], **original_output["state"]}
    batch_size = batch["history_obs_rel_m"].shape[0]
    selected, ramp = _deterministic_probe_ramp(
        batch_size,
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
    original_loss, original_components = robust_multiscale_motion_state_loss(
        original, batch,
    )
    augmented_loss, augmented_components = robust_multiscale_motion_state_loss(
        augmented, augmented_batch,
    )
    yaw_invariance = F.smooth_l1_loss(
        augmented["motion_state_normalized"][selected, 3],
        original["motion_state_normalized"][selected, 3],
        beta=0.02,
    )
    normalized_ramp = ramp[:, :2] / model.motion_state_scale[:2].to(ramp.dtype)
    translation_equivariance = F.smooth_l1_loss(
        (
            augmented["motion_state_normalized"][selected, :2]
            - original["motion_state_normalized"][selected, :2]
        ),
        normalized_ramp[selected],
        beta=0.02,
    )
    objective = 0.5 * (original_loss + augmented_loss) + 0.20 * (
        yaw_invariance + translation_equivariance
    )
    components = dict(original_components)
    for name in (
        "motion", "velocity", "yaw_rate", "scale_aux", "scale_heteroscedastic",
    ):
        components[name] = 0.5 * (
            original_components[name] + augmented_components[name]
        )
    components.update({
        "objective": objective,
        "ramp_yaw_invariance": yaw_invariance,
        "ramp_translation_equivariance": translation_equivariance,
        "state_substage": "joint_structural_probe",
        "state_substage_endpoint": False,
    })
    return original, objective, components


__all__ = [
    "LOCAL_LAG_SCALES_S",
    "AnonymousLocalRigidFlowContext",
    "RigidFlowProbeHead",
    "AnonymousSeparatedRigidFlowProbe",
    "AnonymousJointTwistProbe",
    "_deterministic_probe_ramp",
    "rigid_flow_probe_train_step",
]
