"""Anonymous endpoint-token reliability for the V15-A1 train-only probe.

The scalar V15-A0 head pooled four roles before inference.  This module keeps
event/role correspondence long enough to compare the q0-informed and
history-only frozen expert explanations, then uses shared encoders and
symmetric pooling.  It has no role embedding, physical identity, session,
motion label, truth velocity, or future input.
"""

from __future__ import annotations

from itertools import combinations

import torch
from torch import nn


EVENT_TOKEN_WIDTH = 11
ROLE_TOKEN_WIDTH = 9
PAIR_TOKEN_WIDTH = 8
GLOBAL_CONTRAST_WIDTH = 11
PAIR_INDICES = tuple(combinations(range(4), 2))


def _check_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} is non-finite")


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


def _deproject_endpoint(
    observation_m: torch.Tensor,
    time_s: torch.Tensor,
    omega_rad_s: torch.Tensor,
    velocity_mps: torch.Tensor,
    center_offset_m: torch.Tensor,
) -> torch.Tensor:
    """Map each observed role back to its q0 endpoint under one expert."""
    theta = omega_rad_s[:, None] * time_s
    cosine, sine = torch.cos(theta), torch.sin(theta)
    center_x = center_offset_m[:, 0, None]
    center_y = center_offset_m[:, 1, None]
    center_term_x = (1.0 - cosine) * center_x + sine * center_y
    center_term_y = -sine * center_x + (1.0 - cosine) * center_y
    adjusted_x = (
        observation_m[..., 0] - center_term_x[..., None]
        - time_s[..., None] * velocity_mps[:, None, None, 0]
    )
    adjusted_y = (
        observation_m[..., 1] - center_term_y[..., None]
        - time_s[..., None] * velocity_mps[:, None, None, 1]
    )
    q_x = cosine[..., None] * adjusted_x + sine[..., None] * adjusted_y
    q_y = -sine[..., None] * adjusted_x + cosine[..., None] * adjusted_y
    q_z = (
        observation_m[..., 2]
        - time_s[..., None] * velocity_mps[:, None, None, 2]
    )
    return torch.stack((q_x, q_y, q_z), dim=-1)


def endpoint_token_features(
    *,
    history_obs_rel_m: torch.Tensor,
    history_obs_mask: torch.Tensor,
    history_event_mask: torch.Tensor,
    history_time_s: torch.Tensor,
    history_primary_mask: torch.Tensor,
    history_switch_step: torch.Tensor,
    q0_relation_m: torch.Tensor,
    q0_supported: torch.Tensor,
    omega_rad_s: torch.Tensor,
    q0_velocity_mps: torch.Tensor,
    history_velocity_mps: torch.Tensor,
    q0_center_offset_m: torch.Tensor,
    history_center_offset_m: torch.Tensor,
    pooled_diagnostic_feature: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Construct O(2)-invariant, role-equivariant paired local evidence."""
    if history_obs_rel_m.ndim != 4 or history_obs_rel_m.shape[2:] != (4, 3):
        raise ValueError("endpoint history must have shape [B,T,4,3]")
    batch, events = history_obs_rel_m.shape[:2]
    if history_obs_mask.shape != (batch, events, 4):
        raise ValueError("endpoint history mask differs")
    if history_event_mask.shape != (batch, events):
        raise ValueError("endpoint event mask differs")
    if history_time_s.shape != (batch, events):
        raise ValueError("endpoint time shape differs")
    if history_primary_mask.shape != (batch, events, 4):
        raise ValueError("endpoint primary mask differs")
    if history_switch_step.shape != (batch, events):
        raise ValueError("endpoint switch-step shape differs")
    if q0_relation_m.shape != (batch, 4, 3):
        raise ValueError("endpoint q0 relation shape differs")
    if q0_supported.shape != (batch, 4):
        raise ValueError("endpoint q0 support shape differs")
    if omega_rad_s.reshape(-1).shape != (batch,):
        raise ValueError("endpoint omega shape differs")
    for name, value in {
        "q0_velocity": q0_velocity_mps,
        "history_velocity": history_velocity_mps,
        "q0_center": q0_center_offset_m,
        "history_center": history_center_offset_m,
    }.items():
        if value.shape != (batch, 3):
            raise ValueError(f"endpoint {name} shape differs")
    if pooled_diagnostic_feature.shape != (batch, 13):
        raise ValueError("pooled V14 diagnostic shape differs")
    valid = (
        history_obs_mask.to(torch.bool)
        & history_event_mask.to(torch.bool).unsqueeze(-1)
    )
    if bool(torch.any(valid & ~torch.isfinite(history_obs_rel_m).all(dim=-1))):
        raise ValueError("visible endpoint history is non-finite")
    for name, value in {
        "time": history_time_s,
        "q0_relation": q0_relation_m,
        "omega": omega_rad_s,
        "pooled_diagnostic": pooled_diagnostic_feature,
    }.items():
        _check_finite(name, value)

    omega = omega_rad_s.reshape(batch)
    q0_endpoint = _deproject_endpoint(
        history_obs_rel_m, history_time_s, omega,
        q0_velocity_mps, q0_center_offset_m,
    )
    history_endpoint = _deproject_endpoint(
        history_obs_rel_m, history_time_s, omega,
        history_velocity_mps, history_center_offset_m,
    )
    q0_reference = q0_relation_m[:, None]
    q0_residual = q0_endpoint - q0_reference
    history_residual = history_endpoint - q0_reference
    expert_residual = q0_endpoint - history_endpoint

    def xy_square(value: torch.Tensor) -> torch.Tensor:
        return value[..., :2].square().sum(dim=-1)

    def log_energy(value: torch.Tensor) -> torch.Tensor:
        return torch.log1p(1000.0 * value.clamp_min(0.0))

    q0_xy = xy_square(q0_residual)
    history_xy = xy_square(history_residual)
    expert_xy = xy_square(expert_residual)
    q0_z2 = q0_residual[..., 2].square()
    history_z2 = history_residual[..., 2].square()
    expert_z2 = expert_residual[..., 2].square()
    xy_dot = (q0_residual[..., :2] * history_residual[..., :2]).sum(dim=-1)
    xy_cosine = xy_dot / (q0_xy * history_xy).sqrt().clamp_min(1e-8)
    z_sign_agreement = (
        q0_residual[..., 2] * history_residual[..., 2]
        / (q0_z2 * history_z2).sqrt().clamp_min(1e-8)
    )
    event_feature = torch.stack((
        history_time_s[:, :, None].expand(-1, -1, 4),
        history_primary_mask.to(history_obs_rel_m.dtype),
        history_switch_step.to(history_obs_rel_m.dtype)[:, :, None].expand(
            -1, -1, 4,
        ),
        log_energy(q0_xy), log_energy(q0_z2),
        log_energy(history_xy), log_energy(history_z2),
        log_energy(expert_xy), log_energy(expert_z2),
        xy_cosine.clamp(-1.0, 1.0), z_sign_agreement.clamp(-1.0, 1.0),
    ), dim=-1)
    # V14's q0 expert uses all four finite anonymous H hypotheses whenever
    # any direct q0 support exists.  Per-role support is confidence metadata;
    # it must not delete an inferred role that the frozen expert actually used.
    q0_prior_available = q0_supported.to(torch.bool).any(dim=-1)
    event_mask = valid & q0_prior_available[:, None, None]
    event_feature = torch.where(
        event_mask.unsqueeze(-1), event_feature,
        torch.zeros_like(event_feature),
    )

    valid_weight = valid.to(history_obs_rel_m.dtype)
    visible_count = valid_weight.sum(dim=1)
    history_endpoint_mean = (
        history_endpoint * valid_weight.unsqueeze(-1)
    ).sum(dim=1) / visible_count.clamp_min(1.0).unsqueeze(-1)
    role_mask = (visible_count > 0) & q0_prior_available[:, None]
    active_count = history_event_mask.to(history_obs_rel_m.dtype).sum(
        dim=1, keepdim=True,
    ).clamp_min(1.0)
    positive_inf = torch.full_like(history_time_s[:, :, None], torch.inf)
    negative_inf = torch.full_like(history_time_s[:, :, None], -torch.inf)
    earliest = torch.where(
        valid, history_time_s[:, :, None], positive_inf,
    ).min(dim=1).values
    latest = torch.where(
        valid, history_time_s[:, :, None], negative_inf,
    ).max(dim=1).values
    time_span = torch.where(
        visible_count >= 2, latest - earliest, torch.zeros_like(latest),
    )
    latest_age = torch.where(
        visible_count > 0, -latest, torch.zeros_like(latest),
    )
    primary_fraction = (
        (history_primary_mask.to(torch.bool) & valid)
        .to(history_obs_rel_m.dtype).sum(dim=1)
        / visible_count.clamp_min(1.0)
    )
    history_gap = history_endpoint_mean - q0_relation_m
    role_feature = torch.stack((
        q0_supported.to(history_obs_rel_m.dtype),
        visible_count / active_count,
        time_span, latest_age, primary_fraction,
        log_energy(q0_relation_m[..., :2].square().sum(dim=-1)),
        log_energy(q0_relation_m[..., 2].square()),
        log_energy(history_gap[..., :2].square().sum(dim=-1)),
        log_energy(history_gap[..., 2].square()),
    ), dim=-1)
    role_feature = torch.where(
        role_mask.unsqueeze(-1), role_feature, torch.zeros_like(role_feature),
    )

    pair_values: list[torch.Tensor] = []
    pair_masks: list[torch.Tensor] = []
    for left, right in PAIR_INDICES:
        q0_pair = q0_relation_m[:, right] - q0_relation_m[:, left]
        history_pair = (
            history_endpoint_mean[:, right] - history_endpoint_mean[:, left]
        )
        q0_pair_xy = q0_pair[:, :2].square().sum(dim=-1)
        history_pair_xy = history_pair[:, :2].square().sum(dim=-1)
        pair_xy_dot = (q0_pair[:, :2] * history_pair[:, :2]).sum(dim=-1)
        pair_xy_cosine = pair_xy_dot / (
            q0_pair_xy * history_pair_xy
        ).sqrt().clamp_min(1e-8)
        q0_pair_z2 = q0_pair[:, 2].square()
        history_pair_z2 = history_pair[:, 2].square()
        pair_z_agreement = q0_pair[:, 2] * history_pair[:, 2] / (
            q0_pair_z2 * history_pair_z2
        ).sqrt().clamp_min(1e-8)
        pair_values.append(torch.stack((
            log_energy(q0_pair_xy), log_energy(history_pair_xy),
            pair_xy_cosine.clamp(-1.0, 1.0),
            log_energy((q0_pair[:, :2] - history_pair[:, :2]).square().sum(dim=-1)),
            log_energy(q0_pair_z2), log_energy(history_pair_z2),
            pair_z_agreement.clamp(-1.0, 1.0),
            log_energy((q0_pair[:, 2] - history_pair[:, 2]).square()),
        ), dim=-1))
        pair_masks.append(role_mask[:, left] & role_mask[:, right])
    pair_feature = torch.stack(pair_values, dim=1)
    pair_mask = torch.stack(pair_masks, dim=1)
    pair_feature = torch.where(
        pair_mask.unsqueeze(-1), pair_feature, torch.zeros_like(pair_feature),
    )

    pooled = pooled_diagnostic_feature
    velocity_difference = q0_velocity_mps - history_velocity_mps
    global_feature = torch.stack((
        pooled[:, 0], pooled[:, 1], pooled[:, 2],
        pooled[:, 3] - pooled[:, 5], pooled[:, 4] - pooled[:, 6],
        pooled[:, 7] - pooled[:, 8], pooled[:, 9] - pooled[:, 10],
        pooled[:, 11] - pooled[:, 12],
        log_energy(velocity_difference[:, :2].square().sum(dim=-1)),
        log_energy(velocity_difference[:, 2].square()),
        role_mask.to(history_obs_rel_m.dtype).mean(dim=-1),
    ), dim=-1)
    for name, value in {
        "event_feature": event_feature,
        "role_feature": role_feature,
        "pair_feature": pair_feature,
        "global_feature": global_feature,
    }.items():
        _check_finite(name, value)
    return {
        "event_feature": event_feature.detach(),
        "event_mask": event_mask.detach(),
        "role_feature": role_feature.detach(),
        "role_mask": role_mask.detach(),
        "pair_feature": pair_feature.detach(),
        "pair_mask": pair_mask.detach(),
        "global_feature": global_feature.detach(),
    }


class EndpointTokenReliabilityFusion(nn.Module):
    """Shared event/role/pair set encoder with a continuous expert mixture."""

    def __init__(self, *, width: int = 32) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("endpoint reliability width must be positive")
        self.width = int(width)
        self.event_encoder = nn.Sequential(
            nn.Linear(EVENT_TOKEN_WIDTH, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.role_encoder = nn.Sequential(
            nn.Linear(ROLE_TOKEN_WIDTH + 2 * width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.pair_encoder = nn.Sequential(
            nn.Linear(PAIR_TOKEN_WIDTH, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_CONTRAST_WIDTH, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(5 * width, 2 * width), nn.SiLU(),
            nn.Linear(2 * width, 1),
        )
        for name, width_value in (
            ("event", EVENT_TOKEN_WIDTH), ("role", ROLE_TOKEN_WIDTH),
            ("pair", PAIR_TOKEN_WIDTH),
            ("global", GLOBAL_CONTRAST_WIDTH),
        ):
            self.register_buffer(f"{name}_mean", torch.zeros(width_value))
            self.register_buffer(f"{name}_scale", torch.ones(width_value))
        self.register_buffer("normalizer_fitted", torch.tensor(False))

    @torch.no_grad()
    def fit_normalizer(
        self,
        *,
        event_feature: torch.Tensor,
        event_mask: torch.Tensor,
        role_feature: torch.Tensor,
        role_mask: torch.Tensor,
        pair_feature: torch.Tensor,
        pair_mask: torch.Tensor,
        global_feature: torch.Tensor,
        row_mask: torch.Tensor,
    ) -> None:
        if row_mask.ndim != 1 or row_mask.shape[0] != global_feature.shape[0]:
            raise ValueError("endpoint normalizer row mask differs")
        selections = {
            "event": event_feature[row_mask][event_mask[row_mask]],
            "role": role_feature[row_mask][role_mask[row_mask]],
            "pair": pair_feature[row_mask][pair_mask[row_mask]],
            "global": global_feature[row_mask],
        }
        expected = {
            "event": EVENT_TOKEN_WIDTH, "role": ROLE_TOKEN_WIDTH,
            "pair": PAIR_TOKEN_WIDTH, "global": GLOBAL_CONTRAST_WIDTH,
        }
        for name, values in selections.items():
            if values.ndim != 2 or values.shape[1] != expected[name]:
                raise ValueError(f"endpoint {name} normalizer shape differs")
            if values.shape[0] < 2 or not bool(torch.isfinite(values).all()):
                raise ValueError(f"endpoint {name} normalizer population invalid")
            getattr(self, f"{name}_mean").copy_(values.mean(dim=0))
            getattr(self, f"{name}_scale").copy_(
                values.std(dim=0, unbiased=False).clamp_min(1e-4)
            )
        self.normalizer_fitted.fill_(True)

    def _normalize(self, name: str, value: torch.Tensor) -> torch.Tensor:
        return (
            value - getattr(self, f"{name}_mean")
        ) / getattr(self, f"{name}_scale")

    def reliability_logit(
        self,
        *,
        event_feature: torch.Tensor,
        event_mask: torch.Tensor,
        role_feature: torch.Tensor,
        role_mask: torch.Tensor,
        pair_feature: torch.Tensor,
        pair_mask: torch.Tensor,
        global_feature: torch.Tensor,
        ablate_local: bool = False,
    ) -> torch.Tensor:
        if not bool(self.normalizer_fitted):
            raise RuntimeError("endpoint reliability normalizer is not fitted")
        batch, events, roles, event_width = event_feature.shape
        if roles != 4 or event_width != EVENT_TOKEN_WIDTH:
            raise ValueError("endpoint event feature shape differs")
        if event_mask.shape != (batch, events, 4):
            raise ValueError("endpoint event mask differs")
        if role_feature.shape != (batch, 4, ROLE_TOKEN_WIDTH):
            raise ValueError("endpoint role feature shape differs")
        if role_mask.shape != (batch, 4):
            raise ValueError("endpoint role mask differs")
        if pair_feature.shape != (batch, 6, PAIR_TOKEN_WIDTH):
            raise ValueError("endpoint pair feature shape differs")
        if pair_mask.shape != (batch, 6):
            raise ValueError("endpoint pair mask differs")
        if global_feature.shape != (batch, GLOBAL_CONTRAST_WIDTH):
            raise ValueError("endpoint global feature shape differs")
        normalized_event = self._normalize("event", event_feature)
        normalized_role = self._normalize("role", role_feature)
        normalized_pair = self._normalize("pair", pair_feature)
        if ablate_local:
            normalized_event = torch.zeros_like(normalized_event)
            normalized_role = torch.zeros_like(normalized_role)
            normalized_pair = torch.zeros_like(normalized_pair)
        event_encoded = self.event_encoder(normalized_event)
        event_mean = _masked_mean(event_encoded, event_mask, dim=1)
        event_max = _masked_max(event_encoded, event_mask, dim=1)
        role_encoded = self.role_encoder(torch.cat((
            normalized_role, event_mean, event_max,
        ), dim=-1))
        role_mean = _masked_mean(role_encoded, role_mask, dim=1)
        role_max = _masked_max(role_encoded, role_mask, dim=1)
        pair_encoded = self.pair_encoder(normalized_pair)
        pair_mean = _masked_mean(pair_encoded, pair_mask, dim=1)
        pair_max = _masked_max(pair_encoded, pair_mask, dim=1)
        global_encoded = self.global_encoder(
            self._normalize("global", global_feature)
        )
        return self.output(torch.cat((
            role_mean, role_max, pair_mean, pair_max, global_encoded,
        ), dim=-1)).squeeze(-1)

    def forward(
        self,
        *,
        event_feature: torch.Tensor,
        event_mask: torch.Tensor,
        role_feature: torch.Tensor,
        role_mask: torch.Tensor,
        pair_feature: torch.Tensor,
        pair_mask: torch.Tensor,
        global_feature: torch.Tensor,
        q0_velocity_mps: torch.Tensor,
        history_velocity_mps: torch.Tensor,
        q0_supported: torch.Tensor,
        history_supported: torch.Tensor,
        fallback_velocity_mps: torch.Tensor | None = None,
        fallback_supported: torch.Tensor | None = None,
        ablate_local: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch = global_feature.shape[0]
        if q0_velocity_mps.shape != (batch, 3):
            raise ValueError("endpoint q0 velocity shape differs")
        if history_velocity_mps.shape != (batch, 3):
            raise ValueError("endpoint history velocity shape differs")
        if q0_supported.shape != (batch,) or history_supported.shape != (batch,):
            raise ValueError("endpoint expert support shape differs")
        if (fallback_velocity_mps is None) != (fallback_supported is None):
            raise ValueError("endpoint fallback velocity/support differ")
        logit = self.reliability_logit(
            event_feature=event_feature, event_mask=event_mask,
            role_feature=role_feature, role_mask=role_mask,
            pair_feature=pair_feature, pair_mask=pair_mask,
            global_feature=global_feature, ablate_local=ablate_local,
        )
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
            torch.zeros_like(profile_supported)
            if fallback_supported is None else
            fallback_supported.to(torch.bool) & ~profile_supported
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
            if fallback_velocity_mps.shape != (batch, 3):
                raise ValueError("endpoint fallback velocity shape differs")
            velocity = torch.where(
                fallback_valid.unsqueeze(-1), fallback_velocity_mps, velocity,
            )
        return {
            "velocity_mps": velocity,
            "q0_weight": weight,
            "q0_logit": logit,
            "state_supported": state_supported,
            "learned_gate_valid": q0_valid & history_valid & role_mask.any(dim=-1),
            "fallback_used": fallback_valid,
        }


__all__ = [
    "EVENT_TOKEN_WIDTH", "EndpointTokenReliabilityFusion",
    "GLOBAL_CONTRAST_WIDTH", "PAIR_INDICES", "PAIR_TOKEN_WIDTH",
    "ROLE_TOKEN_WIDTH", "endpoint_token_features",
]
