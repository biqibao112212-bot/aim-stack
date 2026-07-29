"""Causal residual refinement of the final hard-routed future position."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .observable_future_loss import _balanced_group_mean
from .observable_future_model import AnonymousCandidateFutureExpert
from .ordinal_visible_selector import (
    OrdinalVisibleFutureModel,
    frozen_trajectory_selector_context,
)


class FinalVisiblePositionRefiner(nn.Module):
    """Predict a bounded residual after frozen trajectory and plate selection."""

    model_family = "final-visible-position-residual-v1"

    def __init__(
        self,
        *,
        frozen_context_features: int,
        channels: int = 128,
        dropout: float = 0.05,
        position_scale_m: float = 1.0,
        trained_horizon_s: float = 0.55,
        maximum_absolute_step: int = 6,
        maximum_residual_m: float = 0.75,
    ) -> None:
        super().__init__()
        if frozen_context_features < 1 or channels < 16:
            raise ValueError("invalid final-position refiner size")
        if min(
            position_scale_m, trained_horizon_s, maximum_residual_m,
        ) <= 0 or maximum_absolute_step < 1:
            raise ValueError("invalid final-position refiner scales")
        self.frozen_context_features = int(frozen_context_features)
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.position_scale_m = float(position_scale_m)
        self.trained_horizon_s = float(trained_horizon_s)
        self.maximum_absolute_step = int(maximum_absolute_step)
        self.maximum_residual_m = float(maximum_residual_m)
        self.query_frequencies_hz = (0.5, 1.0, 2.0, 4.0, 8.0)
        query_features = 3 + 2 * len(self.query_frequencies_hz)
        # selected delta xyz, selected/expected step, confidence and entropy.
        route_features = 3 + 4
        self.context = nn.Sequential(
            nn.Linear(frozen_context_features, 2 * channels),
            nn.LayerNorm(2 * channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(2 * channels, channels), nn.SiLU(),
        )
        self.residual = nn.Sequential(
            nn.Linear(channels + query_features + route_features, 2 * channels),
            nn.LayerNorm(2 * channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(2 * channels, channels), nn.SiLU(),
            nn.Linear(channels, 3),
        )
        output = self.residual[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "family": self.model_family,
            "frozen_context_features": self.frozen_context_features,
            "channels": self.channels,
            "dropout": self.dropout,
            "position_scale_m": self.position_scale_m,
            "trained_horizon_s": self.trained_horizon_s,
            "maximum_absolute_step": self.maximum_absolute_step,
            "maximum_residual_m": self.maximum_residual_m,
            "query_frequencies_hz": list(self.query_frequencies_hz),
            "input": "causal frozen context plus frozen hard-route outputs",
            "future_truth_forward_input": False,
            "physical_id_input": False,
            "motion_class_forward_input": False,
            "candidate_wise_head": False,
            "physics_decoder": False,
        }

    def forward(
        self,
        frozen_context: torch.Tensor,
        base_prediction: dict[str, torch.Tensor],
        tau_s: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if frozen_context.ndim != 2 or frozen_context.shape[1] != self.frozen_context_features:
            raise ValueError("final refiner frozen context has the wrong shape")
        batch = frozen_context.shape[0]
        tau = AnonymousCandidateFutureExpert._expanded_tau(tau_s, batch)
        if not bool(torch.isfinite(tau).all()) or bool(torch.any(tau < 0)):
            raise ValueError("final refiner query time must be finite and nonnegative")
        if bool(torch.any(tau > self.trained_horizon_s + 1e-6)):
            raise ValueError("final refiner query exceeds trained horizon")
        selected_delta = base_prediction["delta_m"].detach()
        selected_step = base_prediction["selected_switch_step"].detach()
        expected_step = base_prediction["expected_switch_step"].detach()
        probability = base_prediction["switch_probability"].detach().float()
        row = base_prediction["selected_candidate_row"].detach()
        selected_probability = probability.gather(
            2, row.unsqueeze(-1),
        ).squeeze(-1)
        entropy = -(probability * torch.log(probability.clamp_min(1e-12))).sum(dim=-1)
        if selected_delta.shape != tau.shape + (3,):
            raise ValueError("final refiner selected delta shape differs")
        if any(value.shape != tau.shape for value in (
            selected_step, expected_step, selected_probability, entropy,
        )):
            raise ValueError("final refiner route feature shape differs")

        normalized_tau = tau / self.trained_horizon_s
        polynomial = torch.stack((
            normalized_tau, normalized_tau.square(), normalized_tau.pow(3),
        ), dim=-1)
        frequencies = tau.new_tensor(self.query_frequencies_hz)
        phase = 2.0 * math.pi * tau.unsqueeze(-1) * frequencies
        query_feature = torch.cat((
            polynomial, torch.sin(phase), torch.cos(phase),
        ), dim=-1)
        route_feature = torch.cat((
            selected_delta / self.position_scale_m,
            (
                selected_step / float(self.maximum_absolute_step)
            ).unsqueeze(-1),
            (
                expected_step / float(self.maximum_absolute_step)
            ).unsqueeze(-1),
            selected_probability.unsqueeze(-1),
            entropy.unsqueeze(-1),
        ), dim=-1)
        context = self.context(frozen_context.detach())
        hidden = torch.cat((
            context[:, None].expand(-1, tau.shape[1], -1),
            query_feature, route_feature,
        ), dim=-1)
        raw = self.residual(hidden)
        residual_m = self.maximum_residual_m * torch.tanh(raw)
        if not bool(torch.isfinite(residual_m).all()):
            raise ValueError("final position residual is nonfinite")
        refined_position = base_prediction["position_m"].detach() + residual_m
        return {
            "position_residual_m": residual_m,
            "unrefined_position_m": base_prediction["position_m"].detach(),
            "position_m": refined_position,
            "selected_probability": selected_probability,
            "route_entropy": entropy,
        }


class FrozenVisibleSystemFinalRefiner(nn.Module):
    """Frozen PnP/V50/V66 system followed by the trainable final residual."""

    def __init__(
        self,
        base: OrdinalVisibleFutureModel,
        refiner: FinalVisiblePositionRefiner,
    ) -> None:
        super().__init__()
        expected_context = 4 * base.trajectory.channels + 15
        if refiner.frozen_context_features != expected_context:
            raise ValueError("final refiner context width differs from V50")
        if base.trajectory.trained_horizon_s != refiner.trained_horizon_s:
            raise ValueError("final refiner and V50 horizons differ")
        if base.trajectory.maximum_absolute_step != refiner.maximum_absolute_step:
            raise ValueError("final refiner and V66 step ranges differ")
        self.base = base
        self.refiner = refiner
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    @property
    def config(self) -> dict[str, Any]:
        return {
            "family": "frozen-visible-system-final-refiner-v1",
            "base": self.base.config,
            "refiner": self.refiner.config,
        }

    def train(self, mode: bool = True) -> "FrozenVisibleSystemFinalRefiner":
        super().train(mode)
        self.base.eval()
        self.refiner.train(mode)
        return self

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.base.training or any(
            parameter.requires_grad for parameter in self.base.parameters()
        ):
            raise RuntimeError("final refiner base system must remain frozen")
        with torch.no_grad():
            base_prediction = self.base(batch)
            context = frozen_trajectory_selector_context(
                self.base.trajectory,
                batch["history_position_rel_m"], batch["history_time_s"],
                batch["history_dt_s"], batch["history_switch_step"],
                batch["history_mask"], batch["current_position_m"],
                batch["candidate_relation_m"], batch["candidate_step"],
                batch["candidate_mask"], batch["candidate_confidence"],
            )
        refined = self.refiner(context, base_prediction, batch["tau_s"])
        result = dict(base_prediction)
        result.update(refined)
        return result


def final_visible_position_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    huber_beta_m: float = 0.02,
    euclidean_weight: float = 0.1,
    macro_balance_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if huber_beta_m <= 0 or euclidean_weight < 0:
        raise ValueError("invalid final position loss scale")
    if not 0 <= macro_balance_weight <= 1:
        raise ValueError("invalid final position macro balance")
    target_position = (
        batch["current_position_m"][:, None]
        + batch["target_visible_delta_m"]
    )
    if prediction["position_m"].shape != target_position.shape:
        raise ValueError("final predicted and target positions differ")
    query_mask = batch["target_query_mask"].to(torch.bool)
    smooth = F.smooth_l1_loss(
        prediction["position_m"], target_position,
        beta=huber_beta_m, reduction="none",
    ).mean(dim=-1)
    euclidean = torch.linalg.vector_norm(
        prediction["position_m"] - target_position, dim=-1,
    )
    smooth_loss, _, _ = _balanced_group_mean(
        smooth, query_mask, batch["target_switch_count"], macro_balance_weight,
    )
    euclidean_loss, _, _ = _balanced_group_mean(
        euclidean, query_mask, batch["target_switch_count"],
        macro_balance_weight,
    )
    objective = smooth_loss + euclidean_weight * euclidean_loss
    return objective, {
        "objective": objective,
        "smooth_l1": smooth_loss,
        "euclidean": euclidean_loss,
        "mean_residual_m": torch.linalg.vector_norm(
            prediction["position_residual_m"], dim=-1,
        )[query_mask].mean(),
    }
