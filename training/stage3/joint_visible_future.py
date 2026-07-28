"""Joint anonymous future positions and learned visible-candidate probabilities."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .observable_future_loss import _balanced_group_mean, _target_candidate_row
from .observable_future_model import (
    AnonymousCandidateFutureExpert,
    VisibleStreamEncoder,
)


def _query_features(
    tau_s: torch.Tensor, *, trained_horizon_s: float,
    frequencies_hz: tuple[float, ...],
) -> torch.Tensor:
    normalized = tau_s / float(trained_horizon_s)
    polynomial = torch.stack(
        (normalized, normalized.square(), normalized.pow(3)), dim=-1,
    )
    frequencies = tau_s.new_tensor(frequencies_hz)
    phase = (2.0 * math.pi) * tau_s.unsqueeze(-1) * frequencies
    return torch.cat((polynomial, torch.sin(phase), torch.cos(phase)), dim=-1)


class LearnedVisibleStateSelector(nn.Module):
    """Compare a learned future-visible state with anonymous candidate paths.

    The selector owns its history representation.  It sees sample-local signed
    candidate steps and candidate trajectories, but no permanent armor identity,
    motion-class label, physical circle decoder, or truth angular velocity.
    Candidate confidence is deliberately excluded from its compatibility score;
    it remains available to the caller as a reliability/abstention signal.
    """

    model_family = "learned-visible-state-selector-v1"

    def __init__(
        self,
        *,
        channels: int = 128,
        dropout: float = 0.05,
        position_scale_m: float = 1.0,
        history_scale_s: float = 1.0,
        trained_horizon_s: float = 0.55,
        maximum_absolute_step: int = 5,
    ) -> None:
        super().__init__()
        if channels < 16 or maximum_absolute_step < 1:
            raise ValueError("invalid learned visible-state selector size")
        if min(position_scale_m, history_scale_s, trained_horizon_s) <= 0:
            raise ValueError("selector scales must be positive")
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.position_scale_m = float(position_scale_m)
        self.history_scale_s = float(history_scale_s)
        self.trained_horizon_s = float(trained_horizon_s)
        self.maximum_absolute_step = int(maximum_absolute_step)
        self.query_frequencies_hz = (0.5, 1.0, 2.0, 4.0, 8.0)

        self.history_encoder = VisibleStreamEncoder(channels, dropout)
        history_features = 2 * channels + 6
        query_features = 3 + 2 * len(self.query_frequencies_hz)
        self.future_state = nn.Sequential(
            nn.Linear(history_features + 3 + query_features, 2 * channels),
            nn.LayerNorm(2 * channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * channels, channels),
            nn.SiLU(),
        )
        # q0 relation, future relation, learned candidate motion and one signed
        # sample-local step.  There is no row embedding or physical plate ID.
        self.candidate_encoder = nn.Sequential(
            nn.Linear(10, channels), nn.LayerNorm(channels), nn.SiLU(),
            nn.Linear(channels, channels), nn.SiLU(),
        )
        self.compatibility = nn.Sequential(
            nn.Linear(5 * channels, 2 * channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(2 * channels, channels), nn.SiLU(),
            nn.Linear(channels, 1),
        )

    @property
    def config(self) -> dict[str, Any]:
        return {
            "family": self.model_family,
            "channels": self.channels,
            "dropout": self.dropout,
            "position_scale_m": self.position_scale_m,
            "history_scale_s": self.history_scale_s,
            "trained_horizon_s": self.trained_horizon_s,
            "maximum_absolute_step": self.maximum_absolute_step,
            "candidate_confidence_forward_input": False,
            "physical_id_input": False,
            "motion_class_forward_input": False,
            "candidate_row_embedding": False,
        }

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
        tau_s: torch.Tensor,
        conditional_position_m: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = history_position_rel_m.shape[0]
        tau = AnonymousCandidateFutureExpert._expanded_tau(tau_s, batch)
        if conditional_position_m.shape[:2] != tau.shape:
            raise ValueError("candidate paths and selector queries disagree")
        if conditional_position_m.shape[2:] != (
            candidate_relation_m.shape[1], 3,
        ):
            raise ValueError("candidate paths must have shape [B,Q,K,3]")
        if candidate_step.shape != candidate_mask.shape or candidate_step.shape != candidate_relation_m.shape[:2]:
            raise ValueError("selector candidate tensors disagree")

        history_state = self.history_encoder(
            history_position_rel_m, history_time_s, history_dt_s,
            history_switch_step, history_mask,
            position_scale_m=self.position_scale_m,
            history_scale_s=self.history_scale_s,
            switch_scale=float(self.maximum_absolute_step),
        )
        query = _query_features(
            tau, trained_horizon_s=self.trained_horizon_s,
            frequencies_hz=self.query_frequencies_hz,
        )
        current = current_position_m / self.position_scale_m
        future_input = torch.cat((
            history_state[:, None].expand(-1, tau.shape[1], -1),
            current[:, None].expand(-1, tau.shape[1], -1), query,
        ), dim=-1)
        visible_state = self.future_state(future_input)

        q0_relation = candidate_relation_m / self.position_scale_m
        future_relation = (
            conditional_position_m - current_position_m[:, None, None]
        ) / self.position_scale_m
        learned_motion = future_relation - q0_relation[:, None]
        signed_step = (
            candidate_step.to(future_relation.dtype)
            / float(self.maximum_absolute_step)
        )
        candidate_features = torch.cat((
            q0_relation[:, None].expand(-1, tau.shape[1], -1, -1),
            future_relation, learned_motion,
            signed_step[:, None, :, None].expand(-1, tau.shape[1], -1, -1),
        ), dim=-1)
        candidate_state = self.candidate_encoder(candidate_features)
        mask = candidate_mask.to(torch.bool)
        masked_state = torch.where(
            mask[:, None, :, None], candidate_state,
            torch.zeros_like(candidate_state),
        )
        pooled = masked_state.sum(dim=2) / mask.sum(
            dim=1, keepdim=True,
        ).clamp_min(1).to(candidate_state.dtype)[:, None]
        visible = visible_state[:, :, None].expand_as(candidate_state)
        pooled = pooled[:, :, None].expand_as(candidate_state)
        score_input = torch.cat((
            visible, candidate_state, pooled,
            visible * candidate_state, torch.abs(visible - candidate_state),
        ), dim=-1)
        logits = self.compatibility(score_input).squeeze(-1)
        logits = logits.masked_fill(~mask[:, None], -torch.inf)

        zero_query = tau == 0
        current_candidate = mask & (candidate_step == 0)
        if bool(torch.any(current_candidate.sum(dim=1) != 1)):
            raise ValueError("selector requires exactly one relative step-zero row")
        zero_logits = torch.where(
            current_candidate[:, None], torch.zeros_like(logits),
            torch.full_like(logits, -torch.inf),
        )
        logits = torch.where(zero_query.unsqueeze(-1), zero_logits, logits)
        probability = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        selected_row = logits.argmax(dim=-1)
        selected_step = candidate_step.gather(1, selected_row)
        return {
            "switch_logits": logits,
            "switch_probability": probability,
            "selected_candidate_row": selected_row,
            "selected_switch_step": selected_step,
            "visible_state": visible_state,
        }


class JointVisibleFutureModel(nn.Module):
    """A PnP-domain trajectory expert coupled to an independent selector."""

    def __init__(
        self,
        trajectory: AnonymousCandidateFutureExpert,
        selector: LearnedVisibleStateSelector,
    ) -> None:
        super().__init__()
        if trajectory.trained_horizon_s != selector.trained_horizon_s:
            raise ValueError("trajectory and selector horizons differ")
        if trajectory.maximum_absolute_step != selector.maximum_absolute_step:
            raise ValueError("trajectory and selector step ranges differ")
        self.trajectory = trajectory
        self.selector = selector

    @property
    def config(self) -> dict[str, Any]:
        return {
            "family": "joint-visible-future-v1",
            "trajectory": self.trajectory.config,
            "selector": self.selector.config,
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        trajectory = self.trajectory(
            batch["history_position_rel_m"], batch["history_time_s"],
            batch["history_dt_s"], batch["history_switch_step"],
            batch["history_mask"], batch["current_position_m"],
            batch["candidate_relation_m"], batch["candidate_step"],
            batch["candidate_mask"], batch["candidate_confidence"],
            batch["tau_s"], detach_observation_inputs=True,
        )
        selected = self.selector(
            batch["history_position_rel_m"], batch["history_time_s"],
            batch["history_dt_s"], batch["history_switch_step"],
            batch["history_mask"], batch["current_position_m"],
            batch["candidate_relation_m"], batch["candidate_step"],
            batch["candidate_mask"], batch["tau_s"],
            trajectory["conditional_position_m"],
        )
        row = selected["selected_candidate_row"]
        gather = row[:, :, None, None].expand(-1, -1, 1, 3)
        position = trajectory["conditional_position_m"].gather(
            2, gather,
        ).squeeze(2)
        result = dict(trajectory)
        result.update(selected)
        result["position_m"] = position
        result["delta_m"] = position - batch["current_position_m"][:, None]
        return result


def joint_visible_future_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    switch_weight: float,
    conditional_position_weight: float,
    mixture_weight: float,
    expected_cost_weight: float,
    mixture_sigma_m: float,
    huber_beta_m: float,
    macro_balance_weight: float,
    focal_gamma: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if min(
        switch_weight, conditional_position_weight, mixture_weight,
        expected_cost_weight,
    ) < 0 or min(mixture_sigma_m, huber_beta_m) <= 0:
        raise ValueError("joint visible-future loss configuration is invalid")
    logits = prediction["switch_logits"]
    true_row, query_mask = _target_candidate_row(
        batch["candidate_step"], batch["candidate_mask"],
        batch["target_switch_count"], batch["target_query_mask"],
    )
    tau = AnonymousCandidateFutureExpert._expanded_tau(
        batch["tau_s"], logits.shape[0],
    )
    optimization_mask = query_mask & (tau > 0)
    if not bool(optimization_mask.any()):
        raise ValueError("joint visible-future batch has no learned query")

    flat_ce = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        true_row.reshape(-1), reduction="none",
    ).reshape_as(true_row)
    if focal_gamma > 0:
        flat_ce = (1.0 - torch.exp(-flat_ce)).pow(focal_gamma) * flat_ce
    switch, _, _ = _balanced_group_mean(
        flat_ce, optimization_mask, batch["target_switch_count"],
        macro_balance_weight,
    )

    target_position = (
        batch["current_position_m"][:, None]
        + batch["target_visible_delta_m"]
    )
    candidate_error = prediction["conditional_position_m"] - target_position[:, :, None]
    candidate_distance = torch.linalg.vector_norm(candidate_error, dim=-1)
    gather = true_row[:, :, None, None].expand(-1, -1, 1, 3)
    true_position = prediction["conditional_position_m"].gather(
        2, gather,
    ).squeeze(2)
    conditional_element = F.smooth_l1_loss(
        true_position, target_position, beta=huber_beta_m, reduction="none",
    ).mean(dim=-1)
    conditional, _, _ = _balanced_group_mean(
        conditional_element, optimization_mask,
        batch["target_switch_count"], macro_balance_weight,
    )

    sigma2 = float(mixture_sigma_m) ** 2
    component_log_likelihood = (
        -0.5 * candidate_error.square().sum(dim=-1) / sigma2
        -1.5 * math.log(2.0 * math.pi * sigma2)
    )
    mixture_element = -torch.logsumexp(
        torch.log_softmax(logits.float(), dim=-1)
        + component_log_likelihood.float(), dim=-1,
    )
    mixture, _, _ = _balanced_group_mean(
        mixture_element, optimization_mask,
        batch["target_switch_count"], macro_balance_weight,
    )
    expected_element = (
        torch.softmax(logits.float(), dim=-1)
        * candidate_distance.float().clamp_max(1.0)
    ).sum(dim=-1)
    expected, _, _ = _balanced_group_mean(
        expected_element, optimization_mask,
        batch["target_switch_count"], macro_balance_weight,
    )
    objective = (
        switch_weight * switch
        + conditional_position_weight * conditional
        + mixture_weight * mixture
        + expected_cost_weight * expected
    )
    return objective, {
        "objective": objective,
        "switch": switch,
        "conditional_position": conditional,
        "mixture_nll": mixture,
        "expected_position_cost": expected,
    }
