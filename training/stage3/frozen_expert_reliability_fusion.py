"""Frozen-expert continuous reliability fusion for the V15 oracle-omega audit.

The reliability head never sees truth, identities, absolute position, or future
fields.  It consumes the invariant diagnostics already emitted by the frozen
V14 q0-informed and history-only profilers.  Truth is used only to construct a
detached optimal convex coefficient and the final velocity loss.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


V14_DIAGNOSTIC_WIDTH = 13
INCOMPATIBILITY_INDICES = (3, 4)
CONTEXT_INDICES = (0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12)


def oracle_projection_coefficient(
    truth_velocity_mps: torch.Tensor,
    q0_velocity_mps: torch.Tensor,
    history_velocity_mps: torch.Tensor,
    q0_supported: torch.Tensor,
    history_supported: torch.Tensor,
    *,
    minimum_disagreement_mps: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Loss-only O(2)-invariant projection onto the two-expert segment."""
    if truth_velocity_mps.shape != q0_velocity_mps.shape:
        raise ValueError("truth and q0 velocity shapes differ")
    if history_velocity_mps.shape != q0_velocity_mps.shape:
        raise ValueError("expert velocity shapes differ")
    if q0_velocity_mps.ndim != 2 or q0_velocity_mps.shape[1] != 3:
        raise ValueError("expert velocities must have shape [B,3]")
    batch = q0_velocity_mps.shape[0]
    if q0_supported.shape != (batch,) or history_supported.shape != (batch,):
        raise ValueError("expert support shapes differ")
    if minimum_disagreement_mps <= 0:
        raise ValueError("minimum expert disagreement must be positive")
    if not bool(torch.isfinite(truth_velocity_mps).all()):
        raise ValueError("truth velocity is non-finite")
    difference = q0_velocity_mps.detach() - history_velocity_mps.detach()
    denominator = difference.square().sum(dim=-1)
    numerator = (
        (truth_velocity_mps.detach() - history_velocity_mps.detach())
        * difference
    ).sum(dim=-1)
    both = q0_supported.to(torch.bool) & history_supported.to(torch.bool)
    identifiable = both & (
        denominator >= float(minimum_disagreement_mps) ** 2
    )
    projected = (numerator / denominator.clamp_min(1e-12)).clamp(0.0, 1.0)
    weight = torch.full_like(projected, 0.5)
    weight = torch.where(both & (denominator > 0.0), projected, weight)
    weight = torch.where(
        q0_supported.to(torch.bool) & ~history_supported.to(torch.bool),
        torch.ones_like(weight), weight,
    )
    weight = torch.where(
        ~q0_supported.to(torch.bool) & history_supported.to(torch.bool),
        torch.zeros_like(weight), weight,
    )
    weight = torch.where(
        q0_supported.to(torch.bool) | history_supported.to(torch.bool),
        weight, torch.zeros_like(weight),
    )
    return {
        "weight": weight.detach(),
        "oracle_valid": identifiable.detach(),
        "expert_disagreement_mps": denominator.sqrt().detach(),
    }


def vector_huber_loss(
    prediction: torch.Tensor, target: torch.Tensor, *, beta: float = 0.25,
) -> torch.Tensor:
    """Robust Euclidean loss; unlike coordinate Huber it is O(2)-invariant."""
    if prediction.shape != target.shape or prediction.shape[-1] != 3:
        raise ValueError("vector Huber inputs differ")
    error_norm = torch.linalg.vector_norm(prediction - target.detach(), dim=-1)
    return F.smooth_l1_loss(
        error_norm, torch.zeros_like(error_norm), beta=beta,
    )


class MonotoneReliabilityFusion(nn.Module):
    """Continuous gate with a structural negative incompatibility derivative."""

    diagnostic_width = V14_DIAGNOSTIC_WIDTH
    context_indices = CONTEXT_INDICES
    incompatibility_indices = INCOMPATIBILITY_INDICES

    def __init__(self, *, width: int = 32) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("reliability width must be positive")
        self.context = nn.Sequential(
            nn.Linear(len(self.context_indices), width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        self.raw_incompatibility_slope = nn.Parameter(torch.full((2,), -2.0))
        self.register_buffer("feature_mean", torch.zeros(self.diagnostic_width))
        self.register_buffer("feature_scale", torch.ones(self.diagnostic_width))
        self.register_buffer("normalizer_fitted", torch.tensor(False))

    @torch.no_grad()
    def fit_normalizer(self, diagnostic_feature: torch.Tensor) -> None:
        if diagnostic_feature.ndim != 2 or diagnostic_feature.shape[1] != 13:
            raise ValueError("reliability diagnostics must have shape [N,13]")
        if diagnostic_feature.shape[0] < 2:
            raise ValueError("reliability normalizer needs at least two rows")
        if not bool(torch.isfinite(diagnostic_feature).all()):
            raise ValueError("reliability diagnostics are non-finite")
        self.feature_mean.copy_(diagnostic_feature.mean(dim=0))
        self.feature_scale.copy_(
            diagnostic_feature.std(dim=0, unbiased=False).clamp_min(1e-4)
        )
        self.normalizer_fitted.fill_(True)

    def reliability_logit(self, diagnostic_feature: torch.Tensor) -> torch.Tensor:
        if diagnostic_feature.ndim != 2 or diagnostic_feature.shape[1] != 13:
            raise ValueError("reliability diagnostics must have shape [B,13]")
        if not bool(self.normalizer_fitted):
            raise RuntimeError("reliability normalizer is not fitted")
        if not bool(torch.isfinite(diagnostic_feature).all()):
            raise ValueError("reliability diagnostics are non-finite")
        normalized = (
            diagnostic_feature - self.feature_mean
        ) / self.feature_scale
        context = normalized[:, self.context_indices]
        incompatibility = normalized[:, self.incompatibility_indices]
        slope = F.softplus(self.raw_incompatibility_slope)
        return self.context(context).squeeze(-1) - (
            incompatibility * slope
        ).sum(dim=-1)

    def forward(
        self,
        diagnostic_feature: torch.Tensor,
        q0_velocity_mps: torch.Tensor,
        history_velocity_mps: torch.Tensor,
        q0_supported: torch.Tensor,
        history_supported: torch.Tensor,
        fallback_velocity_mps: torch.Tensor | None = None,
        fallback_supported: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = diagnostic_feature.shape[0]
        if q0_velocity_mps.shape != (batch, 3):
            raise ValueError("q0 velocity shape differs")
        if history_velocity_mps.shape != (batch, 3):
            raise ValueError("history velocity shape differs")
        if q0_supported.shape != (batch,) or history_supported.shape != (batch,):
            raise ValueError("expert support shape differs")
        if (fallback_velocity_mps is None) != (fallback_supported is None):
            raise ValueError("fallback velocity/support must be supplied together")
        if fallback_velocity_mps is not None:
            if fallback_velocity_mps.shape != (batch, 3):
                raise ValueError("fallback velocity shape differs")
            if fallback_supported is None or fallback_supported.shape != (batch,):
                raise ValueError("fallback support shape differs")
        logit = self.reliability_logit(diagnostic_feature)
        learned_weight = torch.sigmoid(logit)
        q0_valid = q0_supported.to(torch.bool)
        history_valid = history_supported.to(torch.bool)
        weight = torch.where(
            q0_valid & ~history_valid, torch.ones_like(learned_weight),
            learned_weight,
        )
        weight = torch.where(
            ~q0_valid & history_valid, torch.zeros_like(weight), weight,
        )
        profile_supported = q0_valid | history_valid
        fallback_valid = (
            torch.zeros_like(profile_supported) if fallback_supported is None
            else fallback_supported.to(torch.bool) & ~profile_supported
        )
        state_supported = profile_supported | fallback_valid
        weight = torch.where(state_supported, weight, torch.zeros_like(weight))
        velocity = (
            weight.unsqueeze(-1) * q0_velocity_mps
            + (1.0 - weight).unsqueeze(-1) * history_velocity_mps
        )
        velocity = torch.where(
            state_supported.unsqueeze(-1), velocity, torch.zeros_like(velocity),
        )
        if fallback_velocity_mps is not None:
            velocity = torch.where(
                fallback_valid.unsqueeze(-1), fallback_velocity_mps, velocity,
            )
        return {
            "velocity_mps": velocity,
            "q0_weight": weight,
            "q0_logit": logit,
            "state_supported": state_supported,
            "learned_gate_valid": q0_valid & history_valid,
            "fallback_used": fallback_valid,
        }


def deterministic_cross_session_derangement(
    sample_keys: Sequence[str],
    session_ids: Sequence[str],
    *,
    domain: str,
    strata: Sequence[object] | None = None,
    require_cross_session: bool = True,
) -> dict[str, object]:
    """Create deterministic within-stratum bijections with no same-session pair."""
    count = len(sample_keys)
    if len(session_ids) != count or (strata is not None and len(strata) != count):
        raise ValueError("derangement metadata lengths differ")
    if count < 2:
        raise ValueError("derangement needs at least two rows")
    if len(set(sample_keys)) != count:
        raise ValueError("derangement sample keys must be unique")
    group_values = list(strata) if strata is not None else ["all"] * count
    groups: dict[object, list[int]] = {}
    for index, value in enumerate(group_values):
        groups.setdefault(value, []).append(index)
    donor = np.full(count, -1, dtype=np.int64)
    invalid_groups: list[str] = []
    for value, indices in sorted(groups.items(), key=lambda item: repr(item[0])):
        ordered = sorted(
            indices,
            key=lambda index: (
                (
                    session_ids[index],
                    hashlib.sha256(
                        f"{domain}\0{sample_keys[index]}".encode("utf-8")
                    ).digest(),
                )
                if require_cross_session else hashlib.sha256(
                    f"{domain}\0{sample_keys[index]}".encode("utf-8")
                ).digest()
            ),
        )
        chosen_shift = None
        for shift in range(1, len(ordered)):
            if all(
                (not require_cross_session)
                or session_ids[ordered[position]]
                != session_ids[ordered[(position + shift) % len(ordered)]]
                for position in range(len(ordered))
            ):
                chosen_shift = shift
                break
        if chosen_shift is None:
            invalid_groups.append(repr(value))
            continue
        for position, recipient in enumerate(ordered):
            donor[recipient] = ordered[(position + chosen_shift) % len(ordered)]
    valid = donor >= 0
    if bool(np.any(donor[valid] == np.arange(count, dtype=np.int64)[valid])):
        raise RuntimeError("derangement contains a fixed point")
    same_session = (
        np.asarray(session_ids, dtype=object)[donor[valid]]
        == np.asarray(session_ids, dtype=object)[valid]
    )
    if require_cross_session and bool(np.any(same_session)):
        raise RuntimeError("derangement contains a same-session donor")
    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8"))
    for index in range(count):
        donor_key = sample_keys[int(donor[index])] if valid[index] else "INVALID"
        digest.update(f"{sample_keys[index]}\0{donor_key}\n".encode("utf-8"))
    return {
        "donor_index": donor,
        "valid": valid,
        "pair_sha256": digest.hexdigest(),
        "valid_count": int(valid.sum()),
        "sample_count": count,
        "coverage": float(valid.mean()),
        "fixed_point_count": 0,
        "cross_session_count": int(valid.sum() - same_session.sum()),
        "require_cross_session": bool(require_cross_session),
        "invalid_groups": invalid_groups,
        "domain": domain,
    }


__all__ = [
    "CONTEXT_INDICES", "INCOMPATIBILITY_INDICES",
    "MonotoneReliabilityFusion", "deterministic_cross_session_derangement",
    "oracle_projection_coefficient", "vector_huber_loss",
]
