"""Truth-isolated upper bound for the local-precision F state space.

The callable in this module is an evaluator, never a deployed forward path.
Truth velocity is used only to optimize the same bounded local variables that
the deployable model may emit.  It is deliberately independent from feature
caches, normalizers and model kwargs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from .prequential_local_precision_irls import (
    project_visible_observation_log_precision,
)


LocalPrecisionSolve = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], dict[str, torch.Tensor]
]


def _per_sample_vector_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape[-1] != 3:
        raise ValueError("local-precision oracle velocity shapes differ")
    error = torch.linalg.vector_norm(prediction - target.detach(), dim=-1)
    return F.smooth_l1_loss(
        error, torch.zeros_like(error), beta=beta, reduction="none",
    )


def optimize_loss_only_local_precision_oracle(
    solve: LocalPrecisionSolve,
    *,
    observation_mask: torch.Tensor,
    q0_supported: torch.Tensor,
    target_velocity_mps: torch.Tensor,
    steps: int = 32,
    learning_rate: float = 0.08,
    logit_limit: float = 2.0,
    beta: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Optimize a deterministic three-start, per-sample state-space oracle.

    ``solve`` receives raw observation logits ``[3B,T,4]``, bounded endpoint
    alphas ``[3B,4]`` and bounded center alphas ``[3B]``.  It must return
    ``velocity_mps`` and ``state_supported``.  The leading three blocks are
    neutral, history-only and q0-informed starts.  All 32 projected Adam steps
    execute; there is no metric-dependent early stopping.
    """
    if observation_mask.ndim != 3 or observation_mask.shape[-1] != 4:
        raise ValueError("oracle observation mask must have shape [B,T,4]")
    batch, events, roles = observation_mask.shape
    if q0_supported.shape != (batch, roles):
        raise ValueError("oracle q0 support must have shape [B,4]")
    if target_velocity_mps.shape != (batch, 3):
        raise ValueError("oracle truth velocity must have shape [B,3]")
    if observation_mask.dtype != torch.bool or q0_supported.dtype != torch.bool:
        raise ValueError("oracle support fields must be boolean")
    if steps != 32:
        raise ValueError("local-precision oracle has exactly 32 steps")
    if min(learning_rate, logit_limit, beta) <= 0:
        raise ValueError("local-precision oracle bounds must be positive")
    if not bool(torch.isfinite(target_velocity_mps).all()):
        raise ValueError("local-precision oracle truth is non-finite")

    device, dtype = target_velocity_mps.device, target_velocity_mps.dtype
    repeated_support = q0_supported.repeat(3, 1)
    repeated_observation_mask = observation_mask.repeat(3, 1, 1)
    center_supported = repeated_support.any(dim=-1)
    repeated_target = target_velocity_mps.repeat(3, 1)

    raw_logits = torch.zeros(
        (3 * batch, events, roles), device=device, dtype=dtype,
        requires_grad=True,
    )
    start_anchor = torch.cat((
        torch.full((batch, roles), 0.5, device=device, dtype=dtype),
        torch.zeros((batch, roles), device=device, dtype=dtype),
        torch.ones((batch, roles), device=device, dtype=dtype),
    ), dim=0)
    start_center = torch.cat((
        torch.full((batch,), 0.5, device=device, dtype=dtype),
        torch.zeros((batch,), device=device, dtype=dtype),
        torch.ones((batch,), device=device, dtype=dtype),
    ), dim=0)
    anchor_alpha = start_anchor.masked_fill(~repeated_support, 0.0).requires_grad_()
    center_alpha = start_center.masked_fill(~center_supported, 0.0).requires_grad_()
    optimizer = torch.optim.Adam(
        (raw_logits, anchor_alpha, center_alpha), lr=learning_rate,
    )

    def projected_solve() -> tuple[
        dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        projected_logits = project_visible_observation_log_precision(
            raw_logits, repeated_observation_mask, logit_limit,
        )
        projected_anchor = torch.where(
            repeated_support, anchor_alpha, torch.zeros_like(anchor_alpha),
        )
        projected_center = torch.where(
            center_supported, center_alpha, torch.zeros_like(center_alpha),
        )
        return (
            solve(projected_logits, projected_anchor, projected_center),
            projected_logits, projected_anchor, projected_center,
        )

    for _ in range(steps):
        result, _, _, _ = projected_solve()
        velocity = result.get("velocity_mps")
        supported = result.get("state_supported")
        if velocity is None or velocity.shape != (3 * batch, 3):
            raise ValueError("oracle solve velocity shape differs")
        if supported is None or supported.shape != (3 * batch,):
            raise ValueError("oracle solve support shape differs")
        if supported.dtype != torch.bool:
            raise ValueError("oracle solve support must be boolean")
        if bool(supported.any()) and not bool(torch.isfinite(velocity[supported]).all()):
            raise RuntimeError("oracle solve returned non-finite supported velocity")
        loss = _per_sample_vector_huber(velocity, repeated_target, beta=beta)
        # Unsupported states are invalid candidates rather than a zero target.
        differentiable = supported & torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        if bool(differentiable.any()):
            # A sum makes every per-sample Adam trajectory independent of the
            # number of supported neighbours in the evaluator batch.  A mean
            # would change the update through Adam's epsilon.
            objective = loss[differentiable].sum()
            if objective.requires_grad:
                objective.backward()
                for variable in (raw_logits, anchor_alpha, center_alpha):
                    if variable.grad is not None and not bool(
                        torch.isfinite(variable.grad).all()
                    ):
                        raise RuntimeError(
                            "local-precision oracle gradient is non-finite"
                        )
                if anchor_alpha.grad is not None and bool(torch.any(
                    anchor_alpha.grad.masked_select(~repeated_support) != 0
                )):
                    raise RuntimeError("unsupported anchor received oracle gradient")
                if center_alpha.grad is not None and bool(torch.any(
                    center_alpha.grad.masked_select(~center_supported) != 0
                )):
                    raise RuntimeError("unsupported center received oracle gradient")
        optimizer.step()
        with torch.no_grad():
            # Match deployment exactly: only the *centered* visible logits are
            # clamped by ``project_visible_observation_log_precision``.  The
            # pre-projection carrier is intentionally unbounded; clamping it
            # would remove deployable asymmetric saturation patterns.
            anchor_alpha.clamp_(0.0, 1.0)
            anchor_alpha.masked_fill_(~repeated_support, 0.0)
            center_alpha.clamp_(0.0, 1.0)
            center_alpha.masked_fill_(~center_supported, 0.0)

    final, projected_logits, projected_anchor, projected_center = projected_solve()
    final_velocity = final["velocity_mps"]
    final_supported = final["state_supported"].to(torch.bool)
    final_loss = _per_sample_vector_huber(
        final_velocity, repeated_target, beta=beta,
    )
    final_loss = torch.where(
        final_supported & torch.isfinite(final_loss), final_loss,
        torch.full_like(final_loss, torch.inf),
    ).reshape(3, batch)
    best_start = torch.argmin(final_loss, dim=0)
    flat_index = best_start * batch + torch.arange(batch, device=device)
    best_supported = torch.isfinite(final_loss[best_start, torch.arange(batch, device=device)])
    return {
        "velocity_mps": final_velocity[flat_index].detach(),
        "observation_log_precision": projected_logits[flat_index].detach(),
        "anchor_alpha": projected_anchor[flat_index].detach(),
        "center_alpha": projected_center[flat_index].detach(),
        "loss": final_loss[best_start, torch.arange(batch, device=device)].detach(),
        "best_start": best_start.detach(),
        "state_supported": best_supported.detach(),
        "steps": torch.tensor(steps, device=device),
    }


def assert_forward_cache_truth_free(cache: dict[str, Any]) -> None:
    """Fail closed if a deployable cache carries loss-only information."""
    forbidden_fragments = (
        "truth", "target", "oracle", "motion_class", "session", "future",
    )
    offenders = sorted(
        str(key) for key in cache
        if any(fragment in str(key).lower() for fragment in forbidden_fragments)
    )
    if offenders:
        raise ValueError(f"forward cache contains loss-only fields: {offenders}")


__all__ = [
    "LocalPrecisionSolve", "assert_forward_cache_truth_free",
    "optimize_loss_only_local_precision_oracle",
]
