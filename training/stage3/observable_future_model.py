"""Learned anonymous observable-target F experts.

There is no physics rollout, fixed plate head, slot embedding, or motion-class
input here.  Translation, rotation, and combined motion instantiate separate
copies of this module.  A shared candidate head predicts a mixture over signed
sample-local switch counts.  A learned low-rank neural operator gives every
anonymous candidate one sample-level continuous trajectory; it is not a
hand-written motion decoder.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


DYNAMIC_EXPERTS = ("translation", "rotation", "combined")


class MaskedCausalResidualBlock(nn.Module):
    """Causal residual block that prevents padded bias from leaking forward."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.dilation = int(dilation)
        self.conv1 = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def _convolve(self, layer: nn.Conv1d, value: torch.Tensor) -> torch.Tensor:
        return layer(F.pad(value, (2 * self.dilation, 0)))

    @staticmethod
    def _mask(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask.unsqueeze(1), value, torch.zeros_like(value))

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = self._mask(value, mask)
        hidden = self._convolve(self.conv1, residual)
        hidden = self._mask(hidden, mask).transpose(1, 2)
        hidden = self.dropout(F.silu(self.norm1(hidden))).transpose(1, 2)
        hidden = self._mask(hidden, mask)
        hidden = self._convolve(self.conv2, hidden)
        hidden = self._mask(hidden, mask).transpose(1, 2)
        hidden = self.dropout(F.silu(self.norm2(hidden))).transpose(1, 2)
        return self._mask(residual + hidden, mask)


class VisibleStreamEncoder(nn.Module):
    """Encode one anonymous selected-target history, not four physical tracks."""

    def __init__(self, channels: int = 64, dropout: float = 0.05) -> None:
        super().__init__()
        if channels < 16:
            raise ValueError("visible-stream channels must be at least 16")
        self.channels = int(channels)
        # xyz, relative time, dt, event switch increment, cumulative anonymous
        # switch relative to q0, same-handle local xyz velocity and its valid
        # marker, plus the event valid flag.
        self.projection = nn.Sequential(
            nn.Linear(12, channels), nn.LayerNorm(channels), nn.SiLU(),
        )
        self.temporal = nn.ModuleList(
            MaskedCausalResidualBlock(channels, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16)
        )

    def forward(
        self,
        history_position_rel_m: torch.Tensor,
        history_time_s: torch.Tensor,
        history_dt_s: torch.Tensor,
        history_switch_step: torch.Tensor,
        history_mask: torch.Tensor,
        *,
        position_scale_m: float,
        history_scale_s: float,
        switch_scale: float,
    ) -> torch.Tensor:
        if history_position_rel_m.ndim != 3 or history_position_rel_m.shape[-1] != 3:
            raise ValueError("history_position_rel_m must have shape [B,T,3]")
        expected = history_position_rel_m.shape[:2]
        if any(value.shape != expected for value in (
            history_time_s, history_dt_s, history_switch_step, history_mask,
        )):
            raise ValueError("visible history scalar inputs must have shape [B,T]")
        mask = history_mask.to(torch.bool)
        if bool(torch.any(mask.sum(dim=1) < 2)):
            raise ValueError("every F history requires at least two valid events")
        finite = (
            torch.isfinite(history_position_rel_m).all(dim=-1)
            & torch.isfinite(history_time_s) & torch.isfinite(history_dt_s)
            & torch.isfinite(history_switch_step)
        )
        if bool(torch.any(mask & ~finite)):
            raise ValueError("valid F history values must be finite")
        if bool(torch.any(mask & (history_time_s > 1e-6))):
            raise ValueError("F history cannot contain future events")
        if bool(torch.any(mask & ~torch.isin(
            history_switch_step.to(torch.long),
            history_switch_step.new_tensor((-1, 0, 1), dtype=torch.long),
        ))):
            raise ValueError("history switch increments must be -1, 0, or +1")
        indices = torch.arange(expected[1], device=mask.device).unsqueeze(0)
        last_index = torch.where(mask, indices, torch.full_like(indices, -1)).amax(dim=1)
        if bool(torch.any(last_index < 0)):
            raise ValueError("F history has no valid event")
        last_time = history_time_s.gather(1, last_index[:, None]).squeeze(1)
        if bool(torch.any(torch.abs(last_time) > 1e-6)):
            raise ValueError("the last valid F history event must be q0")

        clean_position = torch.where(
            mask.unsqueeze(-1), history_position_rel_m, torch.zeros_like(history_position_rel_m)
        )
        clean_time = torch.where(mask, history_time_s, torch.zeros_like(history_time_s))
        clean_dt = torch.where(mask, history_dt_s, torch.zeros_like(history_dt_s))
        clean_switch = torch.where(
            mask, history_switch_step, torch.zeros_like(history_switch_step)
        )
        cumulative_switch = torch.cumsum(clean_switch, dim=1)
        current_cumulative = cumulative_switch.gather(
            1, last_index[:, None]
        )
        cumulative_relative_q0 = torch.where(
            mask,
            cumulative_switch - current_cumulative,
            torch.zeros_like(cumulative_switch),
        )
        local_velocity = torch.zeros_like(clean_position)
        velocity_valid = torch.zeros_like(mask)
        pair_valid = (
            mask[:, 1:] & mask[:, :-1]
            & (clean_switch[:, 1:] == 0)
            & (clean_dt[:, 1:] > 0)
        )
        pair_velocity = (
            (clean_position[:, 1:] - clean_position[:, :-1])
            / clean_dt[:, 1:].clamp_min(1e-6).unsqueeze(-1)
        )
        local_velocity[:, 1:] = torch.where(
            pair_valid.unsqueeze(-1), pair_velocity,
            torch.zeros_like(pair_velocity),
        )
        velocity_valid[:, 1:] = pair_valid
        feature = torch.cat((
            clean_position / float(position_scale_m),
            (clean_time / float(history_scale_s)).unsqueeze(-1),
            (clean_dt / float(history_scale_s)).unsqueeze(-1),
            clean_switch.to(clean_position.dtype).unsqueeze(-1),
            (cumulative_relative_q0.to(clean_position.dtype) / float(switch_scale)).unsqueeze(-1),
            local_velocity * (float(history_scale_s) / float(position_scale_m)),
            velocity_valid.to(clean_position.dtype).unsqueeze(-1),
            mask.to(clean_position.dtype).unsqueeze(-1),
        ), dim=-1)
        sequence = self.projection(feature)
        sequence = torch.where(mask.unsqueeze(-1), sequence, torch.zeros_like(sequence))
        sequence = sequence.transpose(1, 2)
        for block in self.temporal:
            sequence = block(sequence, mask)
        gather = last_index[:, None, None].expand(-1, self.channels, 1)
        last = sequence.gather(2, gather).squeeze(2)
        mean = sequence.sum(dim=2) / mask.sum(dim=1, keepdim=True).to(sequence.dtype)
        velocity_count = velocity_valid.sum(dim=1, keepdim=True).clamp_min(1)
        velocity_mean = local_velocity.sum(dim=1) / velocity_count.to(local_velocity.dtype)
        velocity_indices = torch.where(
            velocity_valid, indices, torch.full_like(indices, -1)
        )
        last_velocity_index = velocity_indices.amax(dim=1)
        has_velocity = last_velocity_index >= 0
        safe_velocity_index = last_velocity_index.clamp_min(0)
        velocity_last = local_velocity.gather(
            1, safe_velocity_index[:, None, None].expand(-1, 1, 3)
        ).squeeze(1)
        velocity_last = torch.where(
            has_velocity.unsqueeze(-1), velocity_last,
            torch.zeros_like(velocity_last),
        )
        raw_velocity_summary = torch.cat((
            velocity_last, velocity_mean,
        ), dim=-1) * (float(history_scale_s) / float(position_scale_m))
        return torch.cat((last, mean, raw_velocity_summary), dim=-1)


class AnonymousCandidateFutureExpert(nn.Module):
    """One independent learned expert for future observable-target motion."""

    def __init__(
        self,
        expert: str,
        *,
        channels: int = 64,
        dropout: float = 0.05,
        position_scale_m: float = 1.0,
        history_scale_s: float = 1.0,
        trained_horizon_s: float = 0.55,
        maximum_absolute_step: int = 5,
        trajectory_rank: int = 8,
    ) -> None:
        super().__init__()
        if expert not in DYNAMIC_EXPERTS:
            raise ValueError("expert must be translation, rotation, or combined")
        if min(position_scale_m, history_scale_s, trained_horizon_s) <= 0:
            raise ValueError("F model scales must be positive")
        if maximum_absolute_step < 1:
            raise ValueError("maximum_absolute_step must be positive")
        if trajectory_rank < 2:
            raise ValueError("trajectory_rank must be at least two")
        self.expert = expert
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.position_scale_m = float(position_scale_m)
        self.history_scale_s = float(history_scale_s)
        self.trained_horizon_s = float(trained_horizon_s)
        self.maximum_absolute_step = int(maximum_absolute_step)
        self.trajectory_rank = int(trajectory_rank)
        self.query_frequencies_hz = (0.5, 1.0, 2.0, 4.0, 8.0)
        self.history_encoder = VisibleStreamEncoder(channels, dropout)
        # xyz relation, signed cumulative step, confidence, current marker.
        self.candidate_encoder = nn.Sequential(
            nn.Linear(6, channels), nn.LayerNorm(channels), nn.SiLU(),
            nn.Linear(channels, channels), nn.SiLU(),
        )
        query_features = 3 + 2 * len(self.query_frequencies_hz)
        geometry_features = 6
        trajectory_context_features = (
            (2 * channels + 6) + 3 + channels + 2 * channels
            + geometry_features
        )
        switch_decoder_features = trajectory_context_features + query_features
        self.switch_candidate_head = nn.Sequential(
            nn.Linear(switch_decoder_features, 2 * channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(2 * channels, channels), nn.SiLU(),
        )
        # The coefficient head never sees tau.  All queries of one sample and
        # anonymous candidate therefore supervise the same trajectory object.
        # Parameters are row-shared, so no candidate identity is encoded.
        self.trajectory_coefficient_head = nn.Sequential(
            nn.Linear(trajectory_context_features, 2 * channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(2 * channels, channels), nn.SiLU(),
            nn.Linear(channels, trajectory_rank * 3),
        )
        # This is a learned generic time basis, not a circle/velocity formula or
        # a numerical integrator.  One tau-independent FiLM vector lets the
        # history condition time scale/shape without giving each query an
        # independent trajectory head.
        time_condition_features = (2 * channels + 6) + 3 + 2 * channels + 6
        self.time_basis_projection = nn.Linear(query_features, channels)
        self.time_basis_conditioner = nn.Sequential(
            nn.Linear(time_condition_features, channels), nn.SiLU(),
            nn.Linear(channels, 2 * channels),
        )
        self.time_basis_output = nn.Linear(channels, trajectory_rank)
        self.switch_logit = nn.Linear(channels, 1)
        nn.init.zeros_(self.switch_logit.bias)
        coefficient_output = self.trajectory_coefficient_head[-1]
        assert isinstance(coefficient_output, nn.Linear)
        nn.init.normal_(coefficient_output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(coefficient_output.bias)

    @property
    def config(self) -> dict[str, object]:
        return {
            "family": "anonymous-observable-future-expert-v9",
            "expert": self.expert,
            "channels": self.channels,
            "dropout": self.dropout,
            "position_scale_m": self.position_scale_m,
            "history_scale_s": self.history_scale_s,
            "trained_horizon_s": self.trained_horizon_s,
            "maximum_absolute_step": self.maximum_absolute_step,
            "trajectory_rank": self.trajectory_rank,
            "query_frequencies_hz": list(self.query_frequencies_hz),
            "history_cumulative_switch_feature": "signed sample-local relative to q0",
            "history_local_velocity_feature": "past-only same-handle finite difference",
            "history_raw_velocity_summary": "last-valid plus valid-event mean",
            "candidate_geometry_summary": "mod-four deduplicated mean and rms spread",
            "continuous_trajectory": "sample-candidate coefficients plus history-conditioned learned time basis",
            "trajectory_coefficients_read_tau": False,
            "time_basis_learned": True,
            "time_basis_conditioning": "one tau-independent FiLM vector per sample",
            "future_target": "visibility-selected anonymous target",
            "physical_identity_input": False,
            "physics_decoder": False,
            "motion_class_input": False,
        }

    @staticmethod
    def _expanded_tau(tau_s: torch.Tensor, batch: int) -> torch.Tensor:
        if tau_s.ndim == 1:
            return tau_s.unsqueeze(0).expand(batch, -1)
        if tau_s.ndim != 2 or tau_s.shape[0] != batch:
            raise ValueError("tau_s must have shape [Q] or [B,Q]")
        return tau_s

    def forward(
        self,
        history_position_rel_m: torch.Tensor,
        history_time_s: torch.Tensor,
        history_dt_s: torch.Tensor,
        history_switch_step: torch.Tensor,
        history_mask: torch.Tensor,
        current_position_m: torch.Tensor,
        candidate_relation_m: torch.Tensor,
        candidate_step: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_confidence: torch.Tensor,
        tau_s: torch.Tensor,
        *,
        detach_observation_inputs: bool = True,
    ) -> dict[str, torch.Tensor]:
        batch = history_position_rel_m.shape[0]
        if current_position_m.shape != (batch, 3):
            raise ValueError("current_position_m must have shape [B,3]")
        if candidate_relation_m.ndim != 3 or candidate_relation_m.shape[0] != batch or candidate_relation_m.shape[-1] != 3:
            raise ValueError("candidate_relation_m must have shape [B,K,3]")
        candidate_shape = candidate_relation_m.shape[:2]
        if any(value.shape != candidate_shape for value in (
            candidate_step, candidate_mask, candidate_confidence,
        )):
            raise ValueError("candidate scalar inputs must have shape [B,K]")
        mask = candidate_mask.to(torch.bool)
        finite_candidate = (
            torch.isfinite(candidate_relation_m).all(dim=-1)
            & torch.isfinite(candidate_step) & torch.isfinite(candidate_confidence)
        )
        if bool(torch.any(mask & ~finite_candidate)):
            raise ValueError("valid anonymous candidate values must be finite")
        if bool(torch.any(mask.sum(dim=1) < 1)):
            raise ValueError("each sample requires at least one candidate")

        # The default preserves the accepted clean-F frozen-S boundary.  The
        # explicit differentiable path is reserved for the paired PnP A/B
        # trainers, where observation adapters or S itself are trainable.
        current = (
            current_position_m.detach()
            if detach_observation_inputs else current_position_m
        )
        relation = (
            candidate_relation_m.detach()
            if detach_observation_inputs else candidate_relation_m
        )
        step = candidate_step.detach()
        confidence = (
            candidate_confidence.detach()
            if detach_observation_inputs else candidate_confidence
        )
        mask = mask.detach()
        current_candidate = mask & (step == 0)
        if bool(torch.any(current_candidate.sum(dim=1) != 1)):
            raise ValueError("every sample requires exactly one valid step-zero candidate")
        clean_relation = torch.where(mask.unsqueeze(-1), relation, torch.zeros_like(relation))
        anchor_relation = clean_relation[current_candidate]
        if not torch.equal(anchor_relation, torch.zeros_like(anchor_relation)):
            raise ValueError("the step-zero candidate relation must be bit-exact zero")
        clean_step = torch.where(mask, step, torch.zeros_like(step)).to(relation.dtype)
        clean_confidence = torch.where(mask, confidence, torch.zeros_like(confidence)).to(relation.dtype)
        candidate_feature = torch.cat((
            clean_relation / self.position_scale_m,
            (clean_step / float(self.maximum_absolute_step)).unsqueeze(-1),
            clean_confidence.unsqueeze(-1),
            current_candidate.to(relation.dtype).unsqueeze(-1),
        ), dim=-1)
        candidate_state = self.candidate_encoder(candidate_feature)
        candidate_state = torch.where(mask.unsqueeze(-1), candidate_state, torch.zeros_like(candidate_state))
        count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(candidate_state.dtype)
        pooled_mean = candidate_state.sum(dim=1) / count
        minimum = torch.finfo(candidate_state.dtype).min
        pooled_max = torch.where(
            mask.unsqueeze(-1), candidate_state, candidate_state.new_full((), minimum)
        ).amax(dim=1)
        pooled = torch.cat((pooled_mean, pooled_max), dim=-1)
        rounded_step = torch.round(clean_step).to(torch.long)
        if bool(torch.any(mask & (clean_step != rounded_step.to(clean_step.dtype)))):
            raise ValueError("anonymous candidate steps must be integral")
        residue = torch.remainder(rounded_step, 4)
        residue_relations: list[torch.Tensor] = []
        for role in range(4):
            role_mask = mask & (residue == role)
            if bool(torch.any(role_mask.sum(dim=1) < 1)):
                raise ValueError("candidate set must cover every mod-four relative role")
            role_count = role_mask.sum(dim=1, keepdim=True).to(relation.dtype)
            role_relation = (
                torch.where(
                    role_mask.unsqueeze(-1), clean_relation,
                    torch.zeros_like(clean_relation),
                ).sum(dim=1) / role_count
            )
            residue_relations.append(role_relation)
        unique_relation = torch.stack(residue_relations, dim=1)
        geometry_center = unique_relation.mean(dim=1)
        geometry_spread = torch.sqrt(
            (unique_relation - geometry_center[:, None, :]).square().mean(dim=1)
            + 1e-12
        )
        geometry_context = torch.cat((
            geometry_center / self.position_scale_m,
            geometry_spread / self.position_scale_m,
        ), dim=-1)

        history_state = self.history_encoder(
            history_position_rel_m,
            history_time_s,
            history_dt_s,
            history_switch_step,
            history_mask,
            position_scale_m=self.position_scale_m,
            history_scale_s=self.history_scale_s,
            switch_scale=float(self.maximum_absolute_step),
        )
        if not bool(torch.isfinite(current).all()):
            raise ValueError("current observable position must be finite")
        tau = self._expanded_tau(tau_s, batch)
        if not bool(torch.isfinite(tau).all()) or bool(torch.any(tau < 0)):
            raise ValueError("F query times must be finite and non-negative")
        if bool(torch.any(tau > self.trained_horizon_s + 1e-6)):
            raise ValueError("F query exceeds its trained horizon")
        polynomial_time = torch.stack((
            tau / self.trained_horizon_s,
            (tau / self.trained_horizon_s).square(),
            (tau / self.trained_horizon_s).pow(3),
        ), dim=-1)
        frequencies = tau.new_tensor(self.query_frequencies_hz)
        phase = (2.0 * math.pi) * tau.unsqueeze(-1) * frequencies
        query_feature = torch.cat((
            polynomial_time, torch.sin(phase), torch.cos(phase),
        ), dim=-1)
        query_count = tau.shape[1]
        candidate_count = relation.shape[1]
        trajectory_context = torch.cat((
            history_state[:, None, None, :].expand(-1, query_count, candidate_count, -1),
            (current / self.position_scale_m)[:, None, None, :].expand(-1, query_count, candidate_count, -1),
            candidate_state[:, None, :, :].expand(-1, query_count, -1, -1),
            pooled[:, None, None, :].expand(-1, query_count, candidate_count, -1),
            geometry_context[:, None, None, :].expand(-1, query_count, candidate_count, -1),
        ), dim=-1)
        # Every expanded row is identical over query, so take one copy before
        # producing the sample-candidate trajectory coefficients.
        coefficient_input = trajectory_context[:, 0]
        time_condition = torch.cat((
            history_state,
            current / self.position_scale_m,
            pooled,
            geometry_context,
        ), dim=-1)
        switch_input = torch.cat((
            trajectory_context,
            query_feature[:, :, None, :].expand(-1, -1, candidate_count, -1),
        ), dim=-1)
        switch_state = self.switch_candidate_head(switch_input)
        logits = self.switch_logit(switch_state).squeeze(-1)
        logits = logits.masked_fill(~mask[:, None, :], -torch.inf)
        trajectory_coefficient = self.trajectory_coefficient_head(
            coefficient_input
        ).reshape(batch, candidate_count, self.trajectory_rank, 3)
        trajectory_coefficient = trajectory_coefficient * self.position_scale_m
        normalized_tau = tau / self.trained_horizon_s
        film_scale, film_bias = self.time_basis_conditioner(time_condition).chunk(2, dim=-1)
        time_hidden = self.time_basis_projection(query_feature)
        time_hidden = (
            (1.0 + 0.25 * torch.tanh(film_scale))[:, None, :] * time_hidden
            + 0.25 * torch.tanh(film_bias)[:, None, :]
        )
        learned_basis = torch.tanh(self.time_basis_output(F.silu(time_hidden)))
        time_basis = normalized_tau.unsqueeze(-1) * learned_basis
        learned_motion = torch.einsum(
            "bqr,bkrc->bqkc", time_basis, trajectory_coefficient
        )
        conditional_delta = clean_relation[:, None, :, :] + learned_motion
        conditional_position = current[:, None, None, :] + conditional_delta

        zero_query = tau == 0
        zero_logits = torch.where(
            current_candidate[:, None, :], torch.zeros_like(logits),
            torch.full_like(logits, -torch.inf),
        )
        logits = torch.where(zero_query.unsqueeze(-1), zero_logits, logits)
        probability = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        selected_row = logits.argmax(dim=-1)
        selected_step = clean_step.gather(1, selected_row)
        gather = selected_row[:, :, None, None].expand(-1, -1, 1, 3)
        selected_delta = conditional_delta.gather(2, gather).squeeze(2)
        selected_delta = torch.where(
            zero_query.unsqueeze(-1), torch.zeros_like(selected_delta), selected_delta
        )
        selected_position = current[:, None, :] + selected_delta
        return {
            "switch_logits": logits,
            "switch_probability": probability,
            "conditional_delta_m": conditional_delta,
            "conditional_position_m": conditional_position,
            "trajectory_coefficient_m": trajectory_coefficient,
            "time_basis": time_basis,
            "selected_candidate_row": selected_row,
            "selected_switch_step": selected_step,
            "delta_m": selected_delta,
            "position_m": selected_position,
        }


def stationary_observable_future(
    current_position_m: torch.Tensor, tau_s: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Parameter-free stationary route; no learned expert is instantiated."""
    batch = current_position_m.shape[0]
    tau = AnonymousCandidateFutureExpert._expanded_tau(tau_s, batch)
    delta = current_position_m.new_zeros((batch, tau.shape[1], 3))
    return {"delta_m": delta, "position_m": current_position_m[:, None, :] + delta}
