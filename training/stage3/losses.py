"""Permutation-invariant losses with one joint permutation per sample."""

from __future__ import annotations

import itertools
import torch
from torch import nn
import torch.nn.functional as F

PERMUTATIONS = tuple(itertools.permutations(range(4)))


def _permuted(tensor: torch.Tensor, permutation: tuple[int, ...]) -> torch.Tensor:
    index = torch.tensor(permutation, dtype=torch.long, device=tensor.device)
    return tensor.index_select(2, index)


def stage3_loss(pred: dict[str, torch.Tensor], target_position: torch.Tensor,
                target_normal: torch.Tensor, target_motion: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    candidates = []
    for permutation in PERMUTATIONS:
        mean = _permuted(pred["position_mean"], permutation)
        logvar = _permuted(pred["position_logvar"], permutation)
        normal = _permuted(pred["normal"], permutation)
        error = target_position - mean
        nll = 0.5 * (error.square() * torch.exp(-logvar) + logvar).mean(dim=(1, 2, 3))
        variance_regularizer = 0.01 * logvar.exp().mean(dim=(1, 2, 3))
        normal_loss = (1.0 - F.cosine_similarity(normal, target_normal, dim=-1)).mean(dim=(1, 2))
        candidates.append(nll + variance_regularizer + 0.2 * normal_loss)
    set_losses = torch.stack(candidates, dim=1)
    best, best_index = set_losses.min(dim=1)
    motion_loss = nn.functional.cross_entropy(pred["motion_logits"], target_motion)
    total = best.mean() + 0.1 * motion_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "set_loss": float(best.mean().detach().cpu()),
        "motion_loss": float(motion_loss.detach().cpu()),
        "variance_mean": float(pred["position_logvar"].exp().mean().detach().cpu()),
        "mean_permutation": float(best_index.float().mean().detach().cpu()),
    }


def stage3_observation_loss(
    pred: dict[str, torch.Tensor],
    target_position: torch.Tensor,
    target_normal: torch.Tensor,
    target_motion: torch.Tensor,
    future_observation_position: torch.Tensor,
    future_observation_mask: torch.Tensor,
    future_observation_frame_available: torch.Tensor,
    future_observation_ambiguous: torch.Tensor,
    physical_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Joint physical-state and masked future-observation loss.

    The truth slot permutation is selected jointly per sample.  Observation
    position and visibility terms are only active for exact, unambiguous
    future frames.  A frame with zero candidates is an explicit all-negative
    visibility label; a missing or >4-candidate frame is unknown and masked.
    """
    permutation_index = torch.tensor(PERMUTATIONS, dtype=torch.long, device=target_position.device)

    def all_permuted(value: torch.Tensor) -> torch.Tensor:
        # Armor is dimension 2 for [B,Q,4,...].  A single gather computes all
        # 24 slot permutations without a Python loop over the batch.
        if value.ndim == 4:
            index = permutation_index.view(1, len(PERMUTATIONS), 1, 4, 1).expand(
                value.shape[0], -1, value.shape[1], -1, value.shape[3]
            )
            return value.unsqueeze(1).expand(-1, len(PERMUTATIONS), -1, -1, -1).gather(3, index)
        index = permutation_index.view(1, len(PERMUTATIONS), 1, 4).expand(
            value.shape[0], -1, value.shape[1], -1
        )
        return value.unsqueeze(1).expand(-1, len(PERMUTATIONS), -1, -1).gather(3, index)

    mean = all_permuted(pred["position_mean"])
    logvar = all_permuted(pred["position_logvar"])
    normal = all_permuted(pred["normal"])
    residual = all_permuted(pred["observation_residual_mean"])
    residual_logvar = all_permuted(pred["observation_residual_logvar"])
    visibility = all_permuted(pred["visibility_logits"])
    target_position_expanded = target_position.unsqueeze(1)
    target_normal_expanded = target_normal.unsqueeze(1)
    error = target_position_expanded - mean
    physical_nll = 0.5 * (error.square() * torch.exp(-logvar) + logvar).mean(dim=(2, 3, 4))
    variance_regularizer = 0.01 * logvar.exp().mean(dim=(2, 3, 4))
    normal_loss = (1.0 - F.cosine_similarity(normal, target_normal_expanded, dim=-1)).mean(dim=(2, 3))
    active_frame = future_observation_frame_available & ~future_observation_ambiguous
    position_mask = future_observation_mask & active_frame.unsqueeze(-1)
    position_mask_xyz = position_mask.unsqueeze(1).unsqueeze(-1)
    residual_target = (future_observation_position - target_position).unsqueeze(1)
    observation_prediction = mean + residual
    obs_error = future_observation_position.unsqueeze(1) - observation_prediction
    obs_nll_values = 0.5 * (obs_error.square() * torch.exp(-residual_logvar) + residual_logvar)
    mask_float = position_mask_xyz.to(obs_nll_values.dtype)
    obs_denominator = mask_float.sum(dim=(2, 3, 4)).clamp_min(1.0)
    observation_nll = (obs_nll_values * mask_float).sum(dim=(2, 3, 4)) / obs_denominator
    residual_error = residual - residual_target
    residual_denominator = mask_float.sum(dim=(2, 3, 4)).clamp_min(1.0)
    residual_loss = (F.smooth_l1_loss(residual, residual_target.expand_as(residual), reduction="none") * mask_float).sum(dim=(2, 3, 4)) / residual_denominator
    visibility_mask = active_frame.unsqueeze(1).unsqueeze(-1)
    visibility_target = future_observation_mask.unsqueeze(1).to(visibility.dtype)
    visibility_loss_values = F.binary_cross_entropy_with_logits(visibility, visibility_target.expand_as(visibility), reduction="none")
    visibility_float = visibility_mask.to(visibility_loss_values.dtype)
    visibility_denominator = visibility_float.sum(dim=(2, 3)).clamp_min(1.0)
    visibility_loss = (visibility_loss_values * visibility_float).sum(dim=(2, 3)) / visibility_denominator
    set_losses = (
        physical_weight * (physical_nll + variance_regularizer + 0.2 * normal_loss) +
        observation_nll + 0.5 * residual_loss + 0.2 * visibility_loss
    )
    best, best_index = set_losses.min(dim=1)
    motion_loss = nn.functional.cross_entropy(pred["motion_logits"], target_motion)
    total = best.mean() + 0.1 * motion_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "joint_set_loss": float(best.mean().detach().cpu()),
        "motion_loss": float(motion_loss.detach().cpu()),
        "variance_mean": float(pred["position_logvar"].exp().mean().detach().cpu()),
        "mean_permutation": float(best_index.float().mean().detach().cpu()),
    }


def stage3_direct_observation_loss(
    pred: dict[str, torch.Tensor],
    target_position: torch.Tensor,
    future_observation_position: torch.Tensor,
    future_observation_mask: torch.Tensor,
    future_observation_frame_available: torch.Tensor,
    future_observation_ambiguous: torch.Tensor,
    *,
    physical_aux_weight: float = 0.0,
    huber_beta_m: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Direct future-observation objective for the scratch A/B experiment.

    A single geometry-only permutation is selected from the direct observation
    output against future physical truth.  The selection is shared by the
    observation and optional physical heads and is also used by evaluation.
    Observation availability never participates in permutation selection.
    """
    if "observation_mean" not in pred:
        raise KeyError("direct observation loss requires observation_mean")
    if physical_aux_weight < 0.0:
        raise ValueError("physical_aux_weight must be non-negative")
    if huber_beta_m <= 0.0:
        raise ValueError("huber_beta_m must be positive")

    permutation_index = torch.tensor(
        PERMUTATIONS, dtype=torch.long, device=target_position.device
    )

    def all_permuted(value: torch.Tensor) -> torch.Tensor:
        index = permutation_index.view(1, len(PERMUTATIONS), 1, 4, 1).expand(
            value.shape[0], -1, value.shape[1], -1, value.shape[3]
        )
        return value.unsqueeze(1).expand(
            -1, len(PERMUTATIONS), -1, -1, -1
        ).gather(3, index)

    observation_candidates = all_permuted(pred["observation_mean"])
    target_expanded = target_position.unsqueeze(1)
    geometry_score = torch.linalg.vector_norm(
        observation_candidates - target_expanded, dim=-1
    ).mean(dim=(2, 3))
    best_index = geometry_score.argmin(dim=1)

    def select(candidates: torch.Tensor) -> torch.Tensor:
        gather = best_index.view(-1, 1, 1, 1, 1).expand(
            -1, 1, candidates.shape[2], candidates.shape[3], candidates.shape[4]
        )
        return torch.gather(candidates, 1, gather).squeeze(1)

    observation = select(observation_candidates)
    active = (
        future_observation_mask
        & future_observation_frame_available.unsqueeze(-1)
        & ~future_observation_ambiguous.unsqueeze(-1)
    )
    active_xyz = active.unsqueeze(-1).to(observation.dtype)
    observation_values = F.smooth_l1_loss(
        observation,
        future_observation_position,
        reduction="none",
        beta=huber_beta_m,
    )
    observation_denominator = active_xyz.sum().clamp_min(1.0) * observation.shape[-1]
    observation_loss = (observation_values * active_xyz).sum() / observation_denominator

    physical_loss = observation_loss.new_zeros(())
    if physical_aux_weight > 0.0:
        physical = select(all_permuted(pred["position_mean"]))
        physical_loss = F.smooth_l1_loss(
            physical, target_position, reduction="mean", beta=huber_beta_m
        )
    total = observation_loss + physical_aux_weight * physical_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "observation_huber": float(observation_loss.detach().cpu()),
        "physical_huber": float(physical_loss.detach().cpu()),
        "active_armor_count": float(active.sum().detach().cpu()),
        "mean_permutation": float(best_index.float().mean().detach().cpu()),
    }
