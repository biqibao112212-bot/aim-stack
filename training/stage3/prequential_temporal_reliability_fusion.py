"""Anonymous temporal-graph reliability fusion for the Stage-3 A2 probe.

All learned inputs are scalar invariants.  The module shares one masked
recurrent encoder across the four anonymous roles and another across the six
anonymous unordered role pairs.  Pair summaries are returned to their two
incident roles before a symmetric role-set readout.  There are no role, pair,
session, motion-class, or physical-identity embeddings.
"""

from __future__ import annotations

from itertools import combinations

import torch
from torch import nn


PAIR_INDICES = tuple(combinations(range(4), 2))


def _masked_mean(
    value: torch.Tensor, mask: torch.Tensor, *, dim: int,
) -> torch.Tensor:
    weight = mask.to(value.dtype).unsqueeze(-1)
    return (value * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def _masked_max(
    value: torch.Tensor, mask: torch.Tensor, *, dim: int,
) -> torch.Tensor:
    filled = value.masked_fill(~mask.unsqueeze(-1), -torch.inf)
    maximum = filled.max(dim=dim).values
    supported = mask.any(dim=dim).unsqueeze(-1)
    return torch.where(supported, maximum, torch.zeros_like(maximum))


def _check_visible_finite(
    name: str, value: torch.Tensor, mask: torch.Tensor,
) -> None:
    if bool(torch.any(mask & ~torch.isfinite(value).all(dim=-1))):
        raise ValueError(f"visible {name} is non-finite")


class _SharedMaskedTemporalEncoder(nn.Module):
    """One recurrent encoder shared across every anonymous set element."""

    def __init__(self, input_width: int, width: int) -> None:
        super().__init__()
        self.input_width = int(input_width)
        self.width = int(width)
        self.token = nn.Sequential(
            nn.Linear(input_width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.cell = nn.GRUCell(width, width)
        self.recurrent_summary = nn.Sequential(
            nn.Linear(3 * width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )

    def forward(
        self,
        value: torch.Tensor,
        mask: torch.Tensor,
        *,
        temporal_collapsed: bool,
    ) -> torch.Tensor:
        if value.ndim != 4 or value.shape[-1] != self.input_width:
            raise ValueError("temporal value shape differs")
        batch, events, entities = value.shape[:3]
        if mask.shape != (batch, events, entities):
            raise ValueError("temporal mask shape differs")
        value = torch.where(
            mask.unsqueeze(-1), value, torch.zeros_like(value),
        )
        token = self.token(value)
        if temporal_collapsed:
            mean = _masked_mean(token, mask, dim=1)
            maximum = _masked_max(token, mask, dim=1)
            # Share the trained summary projection with the recurrent path so
            # this is a true temporal ablation rather than an untrained head.
            summary = self.recurrent_summary(torch.cat((mean, mean, maximum), dim=-1))
            return torch.where(
                mask.any(dim=1).unsqueeze(-1), summary,
                torch.zeros_like(summary),
            )

        hidden = torch.zeros(
            batch * entities, self.width, dtype=value.dtype, device=value.device,
        )
        sequence: list[torch.Tensor] = []
        flattened = token.transpose(1, 2).reshape(
            batch * entities, events, self.width,
        )
        flattened_mask = mask.transpose(1, 2).reshape(batch * entities, events)
        for event in range(events):
            candidate = self.cell(flattened[:, event], hidden)
            active = flattened_mask[:, event].unsqueeze(-1)
            hidden = torch.where(active, candidate, hidden)
            sequence.append(hidden.reshape(batch, entities, self.width))
        hidden_sequence = torch.stack(sequence, dim=1)
        mean = _masked_mean(hidden_sequence, mask, dim=1)
        maximum = _masked_max(hidden_sequence, mask, dim=1)
        latest = hidden.reshape(batch, entities, self.width)
        summary = self.recurrent_summary(torch.cat((latest, mean, maximum), dim=-1))
        return torch.where(
            mask.any(dim=1).unsqueeze(-1), summary,
            torch.zeros_like(summary),
        )


class PrequentialTemporalReliabilityFusion(nn.Module):
    """S4-invariant temporal graph head with explicit local/global logits."""

    def __init__(
        self,
        *,
        event_width: int = 11,
        pair_width: int = 8,
        global_width: int = 11,
        width: int = 64,
    ) -> None:
        super().__init__()
        if min(event_width, pair_width, global_width, width) <= 0:
            raise ValueError("temporal reliability widths must be positive")
        self.event_width = int(event_width)
        self.pair_width = int(pair_width)
        self.global_width = int(global_width)
        self.width = int(width)
        self.role_temporal = _SharedMaskedTemporalEncoder(event_width, width)
        self.pair_temporal = _SharedMaskedTemporalEncoder(pair_width, width)
        self.pair_message = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.role_update = nn.Sequential(
            nn.Linear(2 * width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.local_head = nn.Sequential(
            nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, 1),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.global_head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1),
        )
        self.logit_bias = nn.Parameter(torch.zeros(()))
        self.register_buffer("event_mean", torch.zeros(event_width))
        self.register_buffer("event_scale", torch.ones(event_width))
        self.register_buffer("pair_mean", torch.zeros(pair_width))
        self.register_buffer("pair_scale", torch.ones(pair_width))
        self.register_buffer("global_mean", torch.zeros(global_width))
        self.register_buffer("global_scale", torch.ones(global_width))
        self.register_buffer("normalizer_fitted", torch.tensor(False))
        self.register_buffer(
            "pair_left", torch.tensor([left for left, _ in PAIR_INDICES]),
        )
        self.register_buffer(
            "pair_right", torch.tensor([right for _, right in PAIR_INDICES]),
        )

    @torch.no_grad()
    def fit_normalizer(
        self,
        *,
        event_feature: torch.Tensor,
        event_mask: torch.Tensor,
        event_pair_feature: torch.Tensor,
        pair_mask: torch.Tensor,
        global_feature: torch.Tensor,
        row_mask: torch.Tensor,
    ) -> None:
        batch = global_feature.shape[0]
        self._validate_feature_shapes(
            event_feature, event_mask, event_pair_feature, pair_mask,
            global_feature,
        )
        if row_mask.shape != (batch,):
            raise ValueError("temporal normalizer row mask differs")
        row_mask = row_mask.to(torch.bool)
        selections = {
            "event": event_feature[row_mask][event_mask[row_mask]],
            "pair": event_pair_feature[row_mask][pair_mask[row_mask]],
            "global": global_feature[row_mask],
        }
        expected = {
            "event": self.event_width,
            "pair": self.pair_width,
            "global": self.global_width,
        }
        for name, values in selections.items():
            if (
                values.ndim != 2
                or values.shape[1] != expected[name]
                or values.shape[0] < 2
                or not bool(torch.isfinite(values).all())
            ):
                raise ValueError(f"temporal {name} normalizer population invalid")
            getattr(self, f"{name}_mean").copy_(values.mean(dim=0))
            getattr(self, f"{name}_scale").copy_(
                values.std(dim=0, unbiased=False).clamp_min(1e-4)
            )
        self.normalizer_fitted.fill_(True)

    def _validate_feature_shapes(
        self,
        event_feature: torch.Tensor,
        event_mask: torch.Tensor,
        event_pair_feature: torch.Tensor,
        pair_mask: torch.Tensor,
        global_feature: torch.Tensor,
    ) -> None:
        if (
            event_feature.ndim != 4
            or event_feature.shape[2:] != (4, self.event_width)
        ):
            raise ValueError("temporal event feature shape differs")
        batch, events = event_feature.shape[:2]
        if event_mask.shape != (batch, events, 4):
            raise ValueError("temporal event mask differs")
        if event_pair_feature.shape != (batch, events, 6, self.pair_width):
            raise ValueError("temporal pair feature shape differs")
        if pair_mask.shape != (batch, events, 6):
            raise ValueError("temporal pair mask differs")
        if global_feature.shape != (batch, self.global_width):
            raise ValueError("temporal global feature shape differs")
        event_mask_bool = event_mask.to(torch.bool)
        pair_mask_bool = pair_mask.to(torch.bool)
        pair_support = torch.stack([
            event_mask_bool[:, :, left] & event_mask_bool[:, :, right]
            for left, right in PAIR_INDICES
        ], dim=-1)
        if bool(torch.any(pair_mask_bool & ~pair_support)):
            raise ValueError("temporal pair mask exceeds its incident roles")
        _check_visible_finite("event feature", event_feature, event_mask_bool)
        _check_visible_finite(
            "pair feature", event_pair_feature, pair_mask_bool,
        )
        if not bool(torch.isfinite(global_feature).all()):
            raise ValueError("temporal global feature is non-finite")

    def _normalize(self, name: str, value: torch.Tensor) -> torch.Tensor:
        return (
            value - getattr(self, f"{name}_mean")
        ) / getattr(self, f"{name}_scale")

    def reliability_logits(
        self,
        *,
        event_feature: torch.Tensor,
        event_mask: torch.Tensor,
        event_pair_feature: torch.Tensor,
        pair_mask: torch.Tensor,
        global_feature: torch.Tensor,
        local_disabled: bool = False,
        temporal_collapsed: bool = False,
        pair_message_disabled: bool = False,
    ) -> dict[str, torch.Tensor]:
        if not bool(self.normalizer_fitted):
            raise RuntimeError("temporal reliability normalizer is not fitted")
        self._validate_feature_shapes(
            event_feature, event_mask, event_pair_feature, pair_mask,
            global_feature,
        )
        event_mask = event_mask.to(torch.bool)
        pair_mask = pair_mask.to(torch.bool)
        normalized_event = self._normalize("event", event_feature)
        normalized_pair = self._normalize("pair", event_pair_feature)
        role_summary = self.role_temporal(
            normalized_event, event_mask,
            temporal_collapsed=temporal_collapsed,
        )
        pair_summary = self.pair_temporal(
            normalized_pair, pair_mask,
            temporal_collapsed=temporal_collapsed,
        )
        pair_active = pair_mask.any(dim=1)
        encoded_message = self.pair_message(pair_summary)
        role_messages: list[torch.Tensor] = []
        for role in range(4):
            incident = [
                index for index, (left, right) in enumerate(PAIR_INDICES)
                if role in (left, right)
            ]
            active = pair_active[:, incident]
            message = _masked_mean(
                encoded_message[:, incident], active, dim=1,
            )
            role_messages.append(message)
        role_message = torch.stack(role_messages, dim=1)
        if pair_message_disabled:
            role_message = torch.zeros_like(role_message)
        role_active = event_mask.any(dim=1)
        role_encoded = self.role_update(torch.cat((role_summary, role_message), dim=-1))
        role_encoded = torch.where(
            role_active.unsqueeze(-1), role_encoded,
            torch.zeros_like(role_encoded),
        )
        role_mean = _masked_mean(role_encoded, role_active, dim=1)
        role_max = _masked_max(role_encoded, role_active, dim=1)
        local_branch = self.local_head(
            torch.cat((role_mean, role_max), dim=-1)
        ).squeeze(-1)
        global_branch = self.global_head(
            self.global_encoder(self._normalize("global", global_feature))
        ).squeeze(-1)
        local_logit = self.logit_bias + local_branch
        global_only_logit = self.logit_bias + global_branch
        full_logit = self.logit_bias + global_branch + local_branch
        if local_disabled:
            full_logit = global_only_logit
        return {
            "local_logit": local_logit,
            "global_only_logit": global_only_logit,
            "full_logit": full_logit,
            "role_active": role_active,
            "pair_active": pair_active,
        }

    @staticmethod
    def _support_override(
        learned_weight: torch.Tensor,
        q0_valid: torch.Tensor,
        history_valid: torch.Tensor,
        state_supported: torch.Tensor,
    ) -> torch.Tensor:
        weight = torch.where(
            q0_valid & ~history_valid, torch.ones_like(learned_weight),
            learned_weight,
        )
        weight = torch.where(
            ~q0_valid & history_valid, torch.zeros_like(weight), weight,
        )
        return torch.where(state_supported, weight, torch.zeros_like(weight))

    def forward(
        self,
        *,
        event_feature: torch.Tensor,
        event_mask: torch.Tensor,
        event_pair_feature: torch.Tensor,
        pair_mask: torch.Tensor,
        global_feature: torch.Tensor,
        q0_velocity_mps: torch.Tensor,
        history_velocity_mps: torch.Tensor,
        q0_supported: torch.Tensor,
        history_supported: torch.Tensor,
        fallback_velocity_mps: torch.Tensor | None = None,
        fallback_supported: torch.Tensor | None = None,
        local_disabled: bool = False,
        temporal_collapsed: bool = False,
        pair_message_disabled: bool = False,
    ) -> dict[str, torch.Tensor]:
        logits = self.reliability_logits(
            event_feature=event_feature, event_mask=event_mask,
            event_pair_feature=event_pair_feature, pair_mask=pair_mask,
            global_feature=global_feature, local_disabled=local_disabled,
            temporal_collapsed=temporal_collapsed,
            pair_message_disabled=pair_message_disabled,
        )
        batch = global_feature.shape[0]
        if q0_velocity_mps.shape != (batch, 3):
            raise ValueError("temporal q0 velocity shape differs")
        if history_velocity_mps.shape != (batch, 3):
            raise ValueError("temporal history velocity shape differs")
        if q0_supported.shape != (batch,) or history_supported.shape != (batch,):
            raise ValueError("temporal expert support shape differs")
        if (fallback_velocity_mps is None) != (fallback_supported is None):
            raise ValueError("temporal fallback velocity/support differ")
        if not bool(torch.isfinite(q0_velocity_mps).all()) or not bool(
            torch.isfinite(history_velocity_mps).all()
        ):
            raise ValueError("temporal expert velocity is non-finite")
        q0_valid = q0_supported.to(torch.bool)
        history_valid = history_supported.to(torch.bool)
        profile_supported = q0_valid | history_valid
        fallback_valid = (
            torch.zeros_like(profile_supported)
            if fallback_supported is None else
            fallback_supported.to(torch.bool) & ~profile_supported
        )
        state_supported = profile_supported | fallback_valid
        raw_local = torch.sigmoid(logits["local_logit"])
        raw_global = torch.sigmoid(logits["global_only_logit"])
        raw_full = torch.sigmoid(logits["full_logit"])
        local_weight = self._support_override(
            raw_local, q0_valid, history_valid, state_supported,
        )
        global_weight = self._support_override(
            raw_global, q0_valid, history_valid, state_supported,
        )
        full_weight = self._support_override(
            raw_full, q0_valid, history_valid, state_supported,
        )
        velocity = (
            full_weight.unsqueeze(-1) * q0_velocity_mps
            + (1.0 - full_weight).unsqueeze(-1) * history_velocity_mps
        )
        velocity = torch.where(
            state_supported.unsqueeze(-1), velocity, torch.zeros_like(velocity),
        )
        if fallback_velocity_mps is not None:
            if fallback_velocity_mps.shape != (batch, 3):
                raise ValueError("temporal fallback velocity shape differs")
            if not bool(torch.isfinite(fallback_velocity_mps).all()):
                raise ValueError("temporal fallback velocity is non-finite")
            velocity = torch.where(
                fallback_valid.unsqueeze(-1), fallback_velocity_mps, velocity,
            )
        learned_gate_valid = (
            q0_valid & history_valid & logits["role_active"].any(dim=-1)
        )
        return {
            "velocity_mps": velocity,
            "local_logit": logits["local_logit"],
            "global_only_logit": logits["global_only_logit"],
            "full_logit": logits["full_logit"],
            "local_q0_weight": local_weight,
            "global_only_q0_weight": global_weight,
            "q0_weight": full_weight,
            "raw_local_q0_weight": raw_local,
            "raw_global_only_q0_weight": raw_global,
            "raw_q0_weight": raw_full,
            "state_supported": state_supported,
            "learned_gate_valid": learned_gate_valid,
            "fallback_used": fallback_valid,
        }


__all__ = [
    "PAIR_INDICES", "PrequentialTemporalReliabilityFusion",
]
