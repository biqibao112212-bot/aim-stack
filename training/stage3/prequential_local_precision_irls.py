"""Anonymous local precision weights for two-step IRLS.

The network consumes scalar invariant evidence only.  One encoder is shared by
all four anonymous roles and another by all six unordered same-time pairs.
Global conditioning is broadcast into visible local tokens; it has no direct
velocity, precision, or logit output path.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Callable, Mapping

import torch
from torch import nn


ROLE_COUNT = 4
PAIR_INDICES = tuple(combinations(range(ROLE_COUNT), 2))


def _require_bool(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.dtype != torch.bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _visible_finite(
    name: str, value: torch.Tensor, mask: torch.Tensor,
) -> None:
    if bool(torch.any(mask & ~torch.isfinite(value).all(dim=-1))):
        raise ValueError(f"visible {name} is non-finite")


def _masked_mean(
    value: torch.Tensor, mask: torch.Tensor, *, dim: int | tuple[int, ...],
) -> torch.Tensor:
    weight = mask.to(value.dtype).unsqueeze(-1)
    return (value * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def project_visible_observation_log_precision(
    raw: torch.Tensor,
    mask: torch.Tensor,
    max_abs: float,
) -> torch.Tensor:
    """Apply the canonical visible-center, clamp, and mask projection.

    The function is intentionally model-free so the learned head, loss-only
    oracle, and physical solver can share bit-identical state-space semantics.
    Invisible values never participate in the center and are returned as zero.
    """
    if raw.ndim != 3 or raw.shape[-1] != ROLE_COUNT:
        raise ValueError("raw observation log precision must have shape [B,T,4]")
    mask = _require_bool("observation precision mask", mask)
    if mask.shape != raw.shape:
        raise ValueError("observation precision mask shape differs")
    if max_abs <= 0:
        raise ValueError("max absolute observation log precision must be positive")
    if bool(torch.any(mask & ~torch.isfinite(raw))):
        raise ValueError("visible raw observation log precision is non-finite")
    visible = torch.where(
        mask, raw, torch.zeros_like(raw),
    )
    count = mask.sum(dim=(1, 2)).clamp_min(1).to(raw.dtype)
    center = visible.sum(dim=(1, 2)) / count
    projected = (raw - center[:, None, None]).clamp(
        -float(max_abs), float(max_abs),
    )
    return torch.where(mask, projected, torch.zeros_like(projected))


class PrequentialLocalPrecisionIRLS(nn.Module):
    """S4-equivariant local precision head with shared two-step parameters.

    ``event_feature``, ``anchor_feature``, pair features, and global
    conditioning must already be scalar O(2) invariants.  The module accepts no
    role identity, physical identity, vector velocity, or absolute pose.

    ``observation_log_precision`` is centered over each sample's visible
    observations before being clamped.  Consequently neutral raw logits give
    an exact multiplicative observation weight of ``exp(0) == 1``.  Invalid
    observations carry zero log precision but must still be masked by the
    solver.  ``anchor_alpha`` and ``center_alpha`` are unitless interpolation
    coefficients; the solver owns the two endpoint normal contributions.
    """

    def __init__(
        self,
        *,
        event_width: int,
        anchor_width: int,
        pair_width: int,
        global_width: int,
        width: int = 32,
        max_abs_log_precision: float = 2.0,
    ) -> None:
        super().__init__()
        if min(event_width, anchor_width, pair_width, global_width, width) <= 0:
            raise ValueError("local precision widths must be positive")
        if max_abs_log_precision <= 0:
            raise ValueError("max absolute log precision must be positive")
        self.event_width = int(event_width)
        self.anchor_width = int(anchor_width)
        self.pair_width = int(pair_width)
        self.global_width = int(global_width)
        self.width = int(width)
        self.max_abs_log_precision = float(max_abs_log_precision)

        # There are deliberately no role or pair embeddings.
        self.event_encoder = nn.Sequential(
            nn.Linear(event_width + 2, width, bias=False), nn.SiLU(),
            nn.Linear(width, width, bias=False), nn.SiLU(),
        )
        self.anchor_encoder = nn.Sequential(
            nn.Linear(anchor_width, width, bias=False), nn.SiLU(),
            nn.Linear(width, width, bias=False), nn.SiLU(),
        )
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_width, width, bias=False), nn.SiLU(),
            nn.Linear(width, width, bias=False), nn.SiLU(),
        )
        # Global conditioning can only rescale nonzero local evidence.  Since
        # every local encoder is zero-preserving, it cannot create evidence or
        # an output on its own.
        self.global_modulator = nn.Sequential(
            nn.Linear(global_width, width, bias=False), nn.Tanh(),
        )
        self.event_fusion = nn.Sequential(
            nn.Linear(3 * width, width, bias=False), nn.SiLU(),
            nn.Linear(width, width, bias=False), nn.SiLU(),
        )
        self.anchor_fusion = nn.Sequential(
            nn.Linear(3 * width, width, bias=False), nn.SiLU(),
            nn.Linear(width, width, bias=False), nn.SiLU(),
        )
        self.observation_head = nn.Linear(width, 1, bias=False)
        self.anchor_head = nn.Linear(width, 1, bias=False)
        self.center_head = nn.Sequential(
            nn.Linear(2 * width, width, bias=False), nn.SiLU(),
            nn.Linear(width, 1, bias=False),
        )

        self.register_buffer("event_mean", torch.zeros(event_width))
        self.register_buffer("event_scale", torch.ones(event_width))
        self.register_buffer("anchor_mean", torch.zeros(anchor_width))
        self.register_buffer("anchor_scale", torch.ones(anchor_width))
        self.register_buffer("pair_mean", torch.zeros(pair_width))
        self.register_buffer("pair_scale", torch.ones(pair_width))
        self.register_buffer("normalizer_fitted", torch.tensor(False))

    def _validate(
        self,
        *,
        event_feature: torch.Tensor,
        event_mask: torch.Tensor,
        anchor_feature: torch.Tensor,
        anchor_mask: torch.Tensor,
        pair_feature: torch.Tensor,
        pair_mask: torch.Tensor,
        global_feature: torch.Tensor,
    ) -> tuple[int, int]:
        if event_feature.ndim != 4 or event_feature.shape[2:] != (
            ROLE_COUNT, self.event_width,
        ):
            raise ValueError("event feature must have shape [B,T,4,F]")
        batch, events = event_feature.shape[:2]
        event_mask = _require_bool("event mask", event_mask)
        anchor_mask = _require_bool("anchor mask", anchor_mask)
        pair_mask = _require_bool("pair mask", pair_mask)
        if event_mask.shape != (batch, events, ROLE_COUNT):
            raise ValueError("event mask shape differs")
        if anchor_feature.shape != (batch, ROLE_COUNT, self.anchor_width):
            raise ValueError("anchor feature shape differs")
        if anchor_mask.shape != (batch, ROLE_COUNT):
            raise ValueError("anchor mask shape differs")
        if pair_feature.shape != (batch, events, 6, self.pair_width):
            raise ValueError("pair feature shape differs")
        if pair_mask.shape != (batch, events, 6):
            raise ValueError("pair mask shape differs")
        if global_feature.shape != (batch, self.global_width):
            raise ValueError("global conditioning shape differs")
        expected_pair_mask = torch.stack([
            event_mask[:, :, left] & event_mask[:, :, right]
            for left, right in PAIR_INDICES
        ], dim=2)
        if bool(torch.any(pair_mask & ~expected_pair_mask)):
            raise ValueError("pair mask exceeds incident observation support")
        _visible_finite("event feature", event_feature, event_mask)
        _visible_finite("anchor feature", anchor_feature, anchor_mask)
        _visible_finite("pair feature", pair_feature, pair_mask)
        if not bool(torch.isfinite(global_feature).all()):
            raise ValueError("global conditioning is non-finite")
        return batch, events

    @torch.no_grad()
    def fit_normalizer(
        self,
        *,
        event_feature: torch.Tensor,
        event_mask: torch.Tensor,
        anchor_feature: torch.Tensor,
        anchor_mask: torch.Tensor,
        pair_feature: torch.Tensor,
        pair_mask: torch.Tensor,
        global_feature: torch.Tensor,
    ) -> None:
        """Fit token statistics from visible entries and no padded token."""
        self._validate(
            event_feature=event_feature, event_mask=event_mask,
            anchor_feature=anchor_feature, anchor_mask=anchor_mask,
            pair_feature=pair_feature, pair_mask=pair_mask,
            global_feature=global_feature,
        )
        selections = {
            "event": event_feature[event_mask],
            "anchor": anchor_feature[anchor_mask],
            "pair": pair_feature[pair_mask],
        }
        for name, values in selections.items():
            if values.shape[0] == 0:
                getattr(self, f"{name}_mean").zero_()
                getattr(self, f"{name}_scale").fill_(1.0)
                continue
            if not bool(torch.isfinite(values).all()):
                raise ValueError(f"visible {name} normalizer data is non-finite")
            getattr(self, f"{name}_mean").copy_(values.mean(dim=0))
            getattr(self, f"{name}_scale").copy_(
                values.std(dim=0, unbiased=False).clamp_min(1e-4)
            )
        self.normalizer_fitted.fill_(True)

    def _normalize(self, name: str, value: torch.Tensor) -> torch.Tensor:
        return (
            value - getattr(self, f"{name}_mean")
        ) / getattr(self, f"{name}_scale")

    @staticmethod
    def _irls_scalar(
        name: str,
        value: torch.Tensor | None,
        shape: tuple[int, int, int],
        mask: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if value is None:
            return reference.new_zeros(shape)
        if value.shape != shape:
            raise ValueError(f"current {name} shape differs")
        if bool(torch.any(mask & ~torch.isfinite(value))):
            raise ValueError(f"visible current {name} is non-finite")
        value = torch.where(mask, value.to(reference.dtype), torch.zeros_like(reference[..., 0]))
        return torch.log1p(value.square())

    def forward(
        self,
        *,
        event_feature: torch.Tensor,
        event_mask: torch.Tensor,
        anchor_feature: torch.Tensor,
        anchor_mask: torch.Tensor,
        pair_feature: torch.Tensor,
        pair_mask: torch.Tensor,
        global_feature: torch.Tensor,
        current_standardized_residual: torch.Tensor | None = None,
        current_leverage: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not bool(self.normalizer_fitted):
            raise RuntimeError("local precision normalizer is not fitted")
        batch, events = self._validate(
            event_feature=event_feature, event_mask=event_mask,
            anchor_feature=anchor_feature, anchor_mask=anchor_mask,
            pair_feature=pair_feature, pair_mask=pair_mask,
            global_feature=global_feature,
        )
        residual = self._irls_scalar(
            "standardized residual", current_standardized_residual,
            (batch, events, ROLE_COUNT), event_mask, event_feature,
        )
        leverage = self._irls_scalar(
            "leverage", current_leverage,
            (batch, events, ROLE_COUNT), event_mask, event_feature,
        )
        # Padding is allowed to carry arbitrary sentinels, including NaN/Inf.
        # Sanitize before normalization and every learned affine operation;
        # masking only the hidden output is too late because 0 * NaN can still
        # poison Linear weight gradients during backward.
        clean_event_feature = torch.where(
            event_mask.unsqueeze(-1), event_feature,
            torch.zeros_like(event_feature),
        )
        clean_anchor_feature = torch.where(
            anchor_mask.unsqueeze(-1), anchor_feature,
            torch.zeros_like(anchor_feature),
        )
        clean_pair_feature = torch.where(
            pair_mask.unsqueeze(-1), pair_feature,
            torch.zeros_like(pair_feature),
        )
        event_input = torch.cat((
            self._normalize("event", clean_event_feature),
            residual.unsqueeze(-1), leverage.unsqueeze(-1),
        ), dim=-1)
        event_hidden = self.event_encoder(event_input)
        event_hidden = torch.where(
            event_mask.unsqueeze(-1), event_hidden, torch.zeros_like(event_hidden),
        )
        anchor_hidden = self.anchor_encoder(
            self._normalize("anchor", clean_anchor_feature)
        )
        anchor_hidden = torch.where(
            anchor_mask.unsqueeze(-1), anchor_hidden,
            torch.zeros_like(anchor_hidden),
        )
        pair_hidden = self.pair_encoder(
            self._normalize("pair", clean_pair_feature)
        )
        pair_hidden = torch.where(
            pair_mask.unsqueeze(-1), pair_hidden, torch.zeros_like(pair_hidden),
        )
        messages: list[torch.Tensor] = []
        for role in range(ROLE_COUNT):
            incident = [
                index for index, pair in enumerate(PAIR_INDICES) if role in pair
            ]
            messages.append(_masked_mean(
                pair_hidden[:, :, incident], pair_mask[:, :, incident], dim=2,
            ))
        pair_message = torch.stack(messages, dim=2)
        global_scale = 1.0 + self.global_modulator(global_feature)
        event_hidden = event_hidden * global_scale[:, None, None]
        anchor_hidden = anchor_hidden * global_scale[:, None]
        pair_message = pair_message * global_scale[:, None, None]
        event_fused = self.event_fusion(torch.cat((
            event_hidden, pair_message,
            anchor_hidden[:, None].expand(-1, events, -1, -1),
        ), dim=-1))
        event_fused = torch.where(
            event_mask.unsqueeze(-1), event_fused, torch.zeros_like(event_fused),
        )
        raw_log_precision = self.observation_head(event_fused).squeeze(-1)
        observation_log_precision = project_visible_observation_log_precision(
            raw_log_precision, event_mask, self.max_abs_log_precision,
        )

        event_role_summary = _masked_mean(
            event_fused, event_mask, dim=1,
        )
        pair_role_summary = _masked_mean(
            pair_message, event_mask, dim=1,
        )
        anchor_fused = self.anchor_fusion(torch.cat((
            anchor_hidden, event_role_summary, pair_role_summary,
        ), dim=-1))
        anchor_fused = torch.where(
            anchor_mask.unsqueeze(-1), anchor_fused,
            torch.zeros_like(anchor_fused),
        )
        anchor_alpha = torch.sigmoid(self.anchor_head(anchor_fused).squeeze(-1))
        anchor_alpha = torch.where(
            anchor_mask, anchor_alpha, torch.zeros_like(anchor_alpha),
        )
        local_role_mask = anchor_mask | event_mask.any(dim=1)
        center_summary = _masked_mean(
            torch.cat((anchor_fused, event_role_summary), dim=-1),
            local_role_mask, dim=1,
        )
        center_alpha = torch.sigmoid(self.center_head(center_summary).squeeze(-1))
        # A history observation is not evidence that a q0 center prior exists.
        # Keep all prior interpolation exactly at the history boundary when no
        # q0 anchor is supported; the physical solver repeats this fail-closed
        # gate as a second line of defence.
        center_alpha = torch.where(
            anchor_mask.any(dim=1), center_alpha, torch.zeros_like(center_alpha),
        )
        return {
            "observation_log_precision": observation_log_precision,
            "anchor_alpha": anchor_alpha,
            "center_alpha": center_alpha,
        }

    def run_two_step_irls(
        self,
        solver_callback: Callable[
            [int, Mapping[str, torch.Tensor]], Mapping[str, Any]
        ],
        **features: torch.Tensor,
    ) -> dict[str, Any]:
        """Run two shared-head weighting calls around a solver callback.

        After step 0 the callback must return ``standardized_residual`` and
        ``leverage`` tensors.  The callback's step-1 return value is opaque
        solver state, keeping the neural module independent of the solver.
        """
        first = self(**features)
        update = solver_callback(0, first)
        if not isinstance(update, Mapping):
            raise TypeError("IRLS solver callback must return a mapping")
        second = self(
            **features,
            current_standardized_residual=update.get("standardized_residual"),
            current_leverage=update.get("leverage"),
        )
        solution = solver_callback(1, second)
        return {"step1": first, "step2": second, "solution": solution}


__all__ = [
    "PAIR_INDICES", "PrequentialLocalPrecisionIRLS", "ROLE_COUNT",
    "project_visible_observation_log_precision",
]
