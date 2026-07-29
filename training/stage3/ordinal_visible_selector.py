"""Anonymous monotone ordinal selector over frozen visible-future trajectories."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .observable_future_model import AnonymousCandidateFutureExpert
from .observable_future_selector_loss import observable_future_selector_loss


def frozen_trajectory_selector_context(
    trajectory: AnonymousCandidateFutureExpert,
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
) -> torch.Tensor:
    """Reproduce V50's sample context without reading future candidate paths.

    The history and anonymous-candidate encoders are the accepted, frozen V50
    encoders.  Only their permutation-invariant sample summary is exposed to
    the new selector; there is no candidate-row feature at the decision head.
    """
    batch = history_position_rel_m.shape[0]
    if current_position_m.shape != (batch, 3):
        raise ValueError("current_position_m must have shape [B,3]")
    if (
        candidate_relation_m.ndim != 3
        or candidate_relation_m.shape[0] != batch
        or candidate_relation_m.shape[-1] != 3
    ):
        raise ValueError("candidate_relation_m must have shape [B,K,3]")
    expected = candidate_relation_m.shape[:2]
    if any(value.shape != expected for value in (
        candidate_step, candidate_mask, candidate_confidence,
    )):
        raise ValueError("ordinal candidate tensors disagree")
    mask = candidate_mask.to(torch.bool)
    finite = (
        torch.isfinite(candidate_relation_m).all(dim=-1)
        & torch.isfinite(candidate_step)
        & torch.isfinite(candidate_confidence)
    )
    if bool(torch.any(mask & ~finite)):
        raise ValueError("valid anonymous candidates must be finite")
    current_candidate = mask & (candidate_step == 0)
    if bool(torch.any(current_candidate.sum(dim=1) != 1)):
        raise ValueError("every sample requires exactly one step-zero candidate")

    relation = candidate_relation_m.detach()
    step = candidate_step.detach()
    confidence = candidate_confidence.detach()
    mask = mask.detach()
    clean_relation = torch.where(
        mask.unsqueeze(-1), relation, torch.zeros_like(relation),
    )
    anchor = clean_relation[current_candidate]
    if not torch.equal(anchor, torch.zeros_like(anchor)):
        raise ValueError("step-zero candidate relation must be bit-exact zero")
    clean_step = torch.where(mask, step, torch.zeros_like(step)).to(relation.dtype)
    rounded_step = torch.round(clean_step).to(torch.long)
    if bool(torch.any(mask & (clean_step != rounded_step.to(clean_step.dtype)))):
        raise ValueError("anonymous candidate steps must be integral")
    clean_confidence = torch.where(
        mask, confidence, torch.zeros_like(confidence),
    ).to(relation.dtype)
    candidate_feature = torch.cat((
        clean_relation / trajectory.position_scale_m,
        (
            clean_step / float(trajectory.maximum_absolute_step)
        ).unsqueeze(-1),
        clean_confidence.unsqueeze(-1),
        current_candidate.to(relation.dtype).unsqueeze(-1),
    ), dim=-1)
    candidate_state = trajectory.candidate_encoder(candidate_feature)
    candidate_state = torch.where(
        mask.unsqueeze(-1), candidate_state, torch.zeros_like(candidate_state),
    )
    count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(candidate_state.dtype)
    pooled_mean = candidate_state.sum(dim=1) / count
    minimum = torch.finfo(candidate_state.dtype).min
    pooled_max = torch.where(
        mask.unsqueeze(-1), candidate_state,
        candidate_state.new_full((), minimum),
    ).amax(dim=1)
    pooled = torch.cat((pooled_mean, pooled_max), dim=-1)

    residue = torch.remainder(rounded_step, 4)
    residue_relations: list[torch.Tensor] = []
    for role in range(4):
        role_mask = mask & (residue == role)
        if bool(torch.any(role_mask.sum(dim=1) < 1)):
            raise ValueError("candidate set must cover every mod-four relative role")
        role_count = role_mask.sum(dim=1, keepdim=True).to(relation.dtype)
        role_relation = torch.where(
            role_mask.unsqueeze(-1), clean_relation,
            torch.zeros_like(clean_relation),
        ).sum(dim=1) / role_count
        residue_relations.append(role_relation)
    unique_relation = torch.stack(residue_relations, dim=1)
    geometry_center = unique_relation.mean(dim=1)
    geometry_spread = torch.sqrt(
        (unique_relation - geometry_center[:, None]).square().mean(dim=1)
        + 1e-12
    )
    geometry_context = torch.cat((
        geometry_center / trajectory.position_scale_m,
        geometry_spread / trajectory.position_scale_m,
    ), dim=-1)
    history_state = trajectory.history_encoder(
        history_position_rel_m.detach(), history_time_s.detach(),
        history_dt_s.detach(), history_switch_step.detach(),
        history_mask.detach(),
        position_scale_m=trajectory.position_scale_m,
        history_scale_s=trajectory.history_scale_s,
        switch_scale=float(trajectory.maximum_absolute_step),
    )
    if not bool(torch.isfinite(current_position_m).all()):
        raise ValueError("current observable position must be finite")
    return torch.cat((
        history_state,
        current_position_m.detach() / trajectory.position_scale_m,
        pooled,
        geometry_context,
    ), dim=-1)


class OrdinalVisibleProgressSelector(nn.Module):
    """Predict ordered sequential boundary hazards shared by all rows."""

    model_family = "ordinal-visible-progress-selector-v4"

    def __init__(
        self,
        *,
        frozen_context_features: int,
        channels: int = 128,
        dropout: float = 0.05,
        trained_horizon_s: float = 0.55,
        maximum_absolute_step: int = 6,
    ) -> None:
        super().__init__()
        if frozen_context_features < 1 or channels < 16:
            raise ValueError("invalid ordinal selector size")
        if maximum_absolute_step < 1 or trained_horizon_s <= 0:
            raise ValueError("invalid ordinal selector scales")
        self.frozen_context_features = int(frozen_context_features)
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.trained_horizon_s = float(trained_horizon_s)
        self.maximum_absolute_step = int(maximum_absolute_step)
        self.context = nn.Sequential(
            nn.Linear(frozen_context_features, 2 * channels),
            nn.LayerNorm(2 * channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(2 * channels, channels), nn.SiLU(),
        )
        # Direction is separated from magnitude: a soft direction confidence
        # can never compress the accumulated number of crossed boundaries.
        self.direction_head = nn.Linear(channels, 2)
        self.crossing_interval_head = nn.Linear(channels, maximum_absolute_step)
        self.temperature_head = nn.Linear(channels, 1)
        nn.init.zeros_(self.direction_head.bias)
        nn.init.zeros_(self.crossing_interval_head.weight)
        nn.init.constant_(self.crossing_interval_head.bias, -1.5)
        nn.init.zeros_(self.temperature_head.weight)
        nn.init.zeros_(self.temperature_head.bias)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "family": self.model_family,
            "frozen_context_features": self.frozen_context_features,
            "frozen_feature_source": "V50 history/candidate encoders",
            "channels": self.channels,
            "dropout": self.dropout,
            "trained_horizon_s": self.trained_horizon_s,
            "maximum_absolute_step": self.maximum_absolute_step,
            "progress_constraint": "sample-conditioned strictly ordered crossing times",
            "candidate_logit_rule": "direction times ordered magnitude distribution",
            "hard_selection_rule": "count sample-conditioned crossings not later than tau",
            "crossing_time_is_global": False,
            "candidate_wise_mlp": False,
            "candidate_row_embedding": False,
            "physical_id_input": False,
            "motion_class_forward_input": False,
            "future_candidate_path_input": False,
            "fixed_switch_time": False,
            "physics_position_decoder": False,
        }

    def forward(
        self,
        frozen_context: torch.Tensor,
        candidate_step: torch.Tensor,
        candidate_mask: torch.Tensor,
        tau_s: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if frozen_context.ndim != 2 or frozen_context.shape[1] != self.frozen_context_features:
            raise ValueError("frozen_context has the wrong shape")
        batch = frozen_context.shape[0]
        if candidate_step.ndim != 2 or candidate_step.shape[0] != batch:
            raise ValueError("candidate_step must have shape [B,K]")
        if candidate_mask.shape != candidate_step.shape:
            raise ValueError("candidate_mask does not match candidate_step")
        mask = candidate_mask.to(torch.bool)
        if bool(torch.any(mask.sum(dim=1) < 2)):
            raise ValueError("ordinal selector requires at least two candidates")
        current_candidate = mask & (candidate_step == 0)
        if bool(torch.any(current_candidate.sum(dim=1) != 1)):
            raise ValueError("ordinal selector requires exactly one step-zero candidate")
        clean_step = torch.where(mask, candidate_step, torch.zeros_like(candidate_step))
        rounded = torch.round(clean_step).to(torch.long)
        if bool(torch.any(mask & (clean_step != rounded.to(clean_step.dtype)))):
            raise ValueError("anonymous candidate steps must be integral")
        if bool(torch.any(mask & (rounded.abs() > self.maximum_absolute_step))):
            raise ValueError("anonymous candidate step exceeds configured range")
        duplicate = (
            mask[:, :, None] & mask[:, None, :]
            & (rounded[:, :, None] == rounded[:, None, :])
        )
        identity = torch.eye(
            rounded.shape[1], dtype=torch.bool, device=rounded.device,
        )[None]
        if bool(torch.any(duplicate & ~identity)):
            raise ValueError("valid anonymous candidate steps must be unique")
        required = 2 * self.maximum_absolute_step + 1
        if bool(torch.any(mask.sum(dim=1) != required)):
            raise ValueError("ordinal selector requires the complete signed-step range")
        tau = AnonymousCandidateFutureExpert._expanded_tau(tau_s, batch)
        if bool(torch.any(~torch.isfinite(tau))) or bool(torch.any(tau < 0)):
            raise ValueError("ordinal query time must be finite and nonnegative")
        if bool(torch.any(tau > self.trained_horizon_s + 1e-6)):
            raise ValueError("ordinal query exceeds trained horizon")

        if not bool(torch.isfinite(frozen_context).all()):
            raise ValueError("frozen ordinal context must be finite")
        context = self.context(frozen_context)
        if not bool(torch.isfinite(context).all()):
            raise ValueError("ordinal context head produced nonfinite values")
        direction_logits = self.direction_head(context)
        direction_probability = torch.softmax(
            direction_logits.float(), dim=-1,
        ).to(context.dtype)
        # Positive sample-conditioned intervals make T1<T2<... strict while
        # allowing local acceleration or deceleration.  A first crossing past
        # the horizon is the natural right-censored "no switch" prediction.
        minimum_gap_s = self.trained_horizon_s * 1e-4
        crossing_interval = (
            minimum_gap_s
            + self.trained_horizon_s
            * F.softplus(self.crossing_interval_head(context))
        )
        crossing_time = crossing_interval.cumsum(dim=-1)
        temperature = (
            0.01 * self.trained_horizon_s
            + 0.09 * self.trained_horizon_s
            * torch.sigmoid(self.temperature_head(context))
        )
        cumulative_boundary = torch.sigmoid(
            (
                tau[:, :, None] - crossing_time[:, None, :]
            ).float() / temperature[:, None].float()
        )
        magnitude_probability = torch.cat((
            1.0 - cumulative_boundary[..., :1],
            cumulative_boundary[..., :-1] - cumulative_boundary[..., 1:],
            cumulative_boundary[..., -1:],
        ), dim=-1).clamp_min(0.0)
        magnitude_probability = magnitude_probability / magnitude_probability.sum(
            dim=-1, keepdim=True,
        ).clamp_min(1e-12)
        if not bool(
            torch.isfinite(crossing_time).all()
            and torch.isfinite(magnitude_probability).all()
        ):
            raise ValueError("ordinal crossing-time head produced nonfinite values")

        magnitude_index = rounded.abs()[:, None].expand(-1, tau.shape[1], -1)
        magnitude_mass = magnitude_probability.gather(2, magnitude_index)
        sign_mass = torch.where(
            rounded[:, None] < 0,
            direction_probability[:, None, 0, None],
            torch.where(
                rounded[:, None] > 0,
                direction_probability[:, None, 1, None],
                torch.ones_like(magnitude_mass),
            ),
        )
        probability = magnitude_mass * sign_mass
        probability = torch.where(
            mask[:, None], probability, torch.zeros_like(probability),
        )
        probability = probability / probability.sum(
            dim=-1, keepdim=True,
        ).clamp_min(1e-12)
        logits = torch.log(probability.clamp_min(1e-12)).to(context.dtype)
        logits = logits.masked_fill(~mask[:, None], -torch.inf)

        zero_query = tau == 0
        zero_logits = torch.where(
            current_candidate[:, None], torch.zeros_like(logits),
            torch.full_like(logits, -torch.inf),
        )
        logits = torch.where(zero_query.unsqueeze(-1), zero_logits, logits)
        probability = torch.softmax(logits.float(), dim=-1).to(context.dtype)
        hard_magnitude = (
            tau[:, :, None] >= crossing_time[:, None, :]
        ).sum(dim=-1)
        hard_sign = torch.where(
            direction_probability[:, 0] >= direction_probability[:, 1],
            torch.full((batch,), -1, dtype=torch.long, device=context.device),
            torch.ones((batch,), dtype=torch.long, device=context.device),
        )
        hard_step = hard_magnitude * hard_sign[:, None]
        hard_step = torch.where(zero_query, torch.zeros_like(hard_step), hard_step)
        hard_match = (
            mask[:, None] & (rounded[:, None] == hard_step.unsqueeze(-1))
        )
        if bool(torch.any(hard_match.sum(dim=-1) != 1)):
            raise ValueError("hard ordinal step is missing or duplicated")
        selected_row = hard_match.to(torch.long).argmax(dim=-1)
        selected_step = clean_step.gather(1, selected_row)
        expected_step = (
            probability * clean_step[:, None].to(probability.dtype)
        ).sum(dim=-1)
        return {
            "switch_logits": logits,
            "switch_probability": probability,
            "selected_candidate_row": selected_row,
            "selected_switch_step": selected_step,
            "expected_switch_step": expected_step,
            "direction_logits": direction_logits,
            "direction_probability": direction_probability,
            "hard_switch_magnitude": hard_magnitude,
            "crossing_interval_s": crossing_interval,
            "crossing_time_s": crossing_time,
            "crossing_temperature_s": temperature,
            "cumulative_boundary_probability": cumulative_boundary,
            "magnitude_probability": magnitude_probability,
        }


class OrdinalVisibleFutureModel(nn.Module):
    """Bit-exact frozen V50 trajectory plus the ordinal selection head."""

    def __init__(
        self,
        trajectory: AnonymousCandidateFutureExpert,
        selector: OrdinalVisibleProgressSelector,
    ) -> None:
        super().__init__()
        if trajectory.trained_horizon_s != selector.trained_horizon_s:
            raise ValueError("ordinal selector and trajectory horizons differ")
        if trajectory.maximum_absolute_step != selector.maximum_absolute_step:
            raise ValueError("ordinal selector and trajectory step ranges differ")
        expected_context = 4 * trajectory.channels + 15
        if selector.frozen_context_features != expected_context:
            raise ValueError("ordinal selector frozen context width differs")
        self.trajectory = trajectory
        self.selector = selector
        for parameter in self.trajectory.parameters():
            parameter.requires_grad_(False)
        self.trajectory.eval()

    @property
    def config(self) -> dict[str, Any]:
        return {
            "family": "ordinal-visible-future-v4",
            "trajectory": self.trajectory.config,
            "selector": self.selector.config,
        }

    def train(self, mode: bool = True) -> "OrdinalVisibleFutureModel":
        super().train(mode)
        self.trajectory.eval()
        self.selector.train(mode)
        return self

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.trajectory.training or any(
            parameter.requires_grad for parameter in self.trajectory.parameters()
        ):
            raise RuntimeError("ordinal trajectory must remain frozen and in eval mode")
        with torch.no_grad():
            trajectory = self.trajectory(
                batch["history_position_rel_m"], batch["history_time_s"],
                batch["history_dt_s"], batch["history_switch_step"],
                batch["history_mask"], batch["current_position_m"],
                batch["candidate_relation_m"], batch["candidate_step"],
                batch["candidate_mask"], batch["candidate_confidence"],
                batch["tau_s"], detach_observation_inputs=True,
            )
            context = frozen_trajectory_selector_context(
                self.trajectory,
                batch["history_position_rel_m"], batch["history_time_s"],
                batch["history_dt_s"], batch["history_switch_step"],
                batch["history_mask"], batch["current_position_m"],
                batch["candidate_relation_m"], batch["candidate_step"],
                batch["candidate_mask"], batch["candidate_confidence"],
            )
        selected = self.selector(
            context.detach(), batch["candidate_step"],
            batch["candidate_mask"], batch["tau_s"],
        )
        row = selected["selected_candidate_row"]
        delta = trajectory["conditional_delta_m"].gather(
            2, row[:, :, None, None].expand(-1, -1, 1, 3),
        ).squeeze(2)
        result = dict(trajectory)
        result.update(selected)
        result["delta_m"] = delta
        result["position_m"] = batch["current_position_m"][:, None] + delta
        return result


def _direction_target(
    target_switch_count: torch.Tensor,
    target_query_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = target_query_mask.to(torch.bool)
    positive = torch.any(mask & (target_switch_count > 0), dim=1)
    negative = torch.any(mask & (target_switch_count < 0), dim=1)
    if bool(torch.any(positive & negative)):
        raise ValueError("one ordinal target sequence cannot reverse direction")
    supervised = positive | negative
    target = torch.where(
        negative, torch.zeros_like(negative, dtype=torch.long),
        torch.ones_like(positive, dtype=torch.long),
    )
    return target, supervised


def ordinal_visible_selector_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    switch_weight: float = 1.0,
    macro_balance_weight: float = 0.25,
    switch_focal_gamma: float = 2.0,
    distance_cost_weight: float = 1.0,
    distance_cost_scale_m: float = 0.3,
    distance_cost_cap: float = 2.0,
    direction_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """V52 selector semantics plus a sample-level direction auxiliary."""
    if direction_weight < 0:
        raise ValueError("direction_weight must be nonnegative")
    objective, parts = observable_future_selector_loss(
        prediction,
        batch["candidate_step"], batch["candidate_mask"], batch["tau_s"],
        batch["target_switch_count"], batch["target_visible_delta_m"],
        batch["target_query_mask"],
        switch_weight=switch_weight,
        macro_balance_weight=macro_balance_weight,
        switch_focal_gamma=switch_focal_gamma,
        distance_cost_weight=distance_cost_weight,
        distance_cost_scale_m=distance_cost_scale_m,
        distance_cost_cap=distance_cost_cap,
    )
    direction_target, direction_mask = _direction_target(
        batch["target_switch_count"], batch["target_query_mask"],
    )
    direction = objective * 0.0
    if bool(direction_mask.any()):
        direction = F.cross_entropy(
            prediction["direction_logits"].float()[direction_mask],
            direction_target[direction_mask],
        )
    objective = objective + direction_weight * direction
    return objective, {**parts, "objective": objective, "direction": direction}
