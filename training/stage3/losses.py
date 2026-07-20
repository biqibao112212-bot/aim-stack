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
