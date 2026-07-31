"""Invariant helpers for prequential anonymous endpoint evidence.

The helpers in this module only arrange already-constructed scalar endpoint
features.  They never accept a vector position/velocity, identity, session,
motion label, truth target, or future sample.  Consequently an upstream
feature that is planar-O(2), reflection, and common-ramp invariant keeps those
properties here.  Role operations are shared and anonymous; unordered-pair
features are symmetric in their two endpoints.
"""

from __future__ import annotations

from itertools import combinations

import torch


ROLE_COUNT = 4
UNORDERED_PAIR_INDICES = tuple(combinations(range(ROLE_COUNT), 2))
DOSE_VALUES = (1, 2, 4)


def _boolean_mask(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.dtype != torch.bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _check_endpoint_feature(value: torch.Tensor) -> tuple[int, int, int]:
    if value.ndim != 4 or value.shape[2] != ROLE_COUNT:
        raise ValueError("endpoint scalar feature must have shape [B,T,4,F]")
    if value.shape[-1] <= 0:
        raise ValueError("endpoint scalar feature width must be positive")
    return int(value.shape[0]), int(value.shape[1]), int(value.shape[3])


def fixed_history_leave_block_out_masks(
    history_event_mask: torch.Tensor,
    *,
    block_size: int = 1,
    minimum_fit_events: int = 3,
) -> dict[str, torch.Tensor]:
    """Create deterministic contiguous-active-rank leave-block-out masks.

    The block axis has fixed width ``ceil(T / block_size)``.  Active history
    events are ranked in temporal tensor order, so padding holes do not change
    which *active* events share a block.  A block is exposed only when it holds
    at least one event out and leaves ``minimum_fit_events`` active events for
    the frozen fit.  Events belonging only to an ineligible block remain
    explicitly uncovered rather than leaking into both fit and score masks.
    """
    event_mask = _boolean_mask("history event mask", history_event_mask)
    if event_mask.ndim != 2:
        raise ValueError("history event mask must have shape [B,T]")
    if block_size <= 0:
        raise ValueError("leave-block-out block size must be positive")
    if minimum_fit_events <= 0:
        raise ValueError("minimum fit events must be positive")
    batch, events = event_mask.shape
    if events == 0:
        raise ValueError("history needs at least one event position")
    block_count = (events + block_size - 1) // block_size
    active_rank = event_mask.to(torch.int64).cumsum(dim=1) - 1
    raw_block = torch.div(
        active_rank.clamp_min(0), block_size, rounding_mode="floor",
    )
    block_axis = torch.arange(
        block_count, device=event_mask.device, dtype=torch.int64,
    ).view(1, block_count, 1)
    heldout = (
        event_mask[:, None, :]
        & (raw_block[:, None, :] == block_axis)
    )
    fit = event_mask[:, None, :] & ~heldout
    block_valid = (heldout.sum(dim=-1) == int(block_size)) & (
        fit.sum(dim=-1) >= int(minimum_fit_events)
    )
    heldout = heldout & block_valid.unsqueeze(-1)
    fit = fit & block_valid.unsqueeze(-1)
    owner_count = heldout.to(torch.int64).sum(dim=1)
    if bool(torch.any(owner_count > 1)):
        raise RuntimeError("leave-block-out event belongs to multiple blocks")
    owner = torch.where(
        owner_count == 1,
        heldout.to(torch.int64).argmax(dim=1),
        torch.full_like(owner_count, -1),
    )
    return {
        "fit_event_mask": fit,
        "heldout_event_mask": heldout,
        "block_valid": block_valid,
        "heldout_block_index": owner,
        "heldout_coverage_mask": owner_count == 1,
    }


def combine_blockwise_endpoint_event_features(
    block_event_feature: torch.Tensor,
    heldout_event_mask: torch.Tensor,
    *,
    block_event_role_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Select each held-out block prediction back into ``[B,T,4,F]``.

    Features outside their held-out block are ignored and may contain a
    sentinel such as NaN.  A role/event can have at most one source block.
    """
    if block_event_feature.ndim != 5 or block_event_feature.shape[3] != ROLE_COUNT:
        raise ValueError(
            "block endpoint feature must have shape [B,K,T,4,F]"
        )
    batch, blocks, events, roles, width = block_event_feature.shape
    if width <= 0:
        raise ValueError("block endpoint feature width must be positive")
    heldout = _boolean_mask("heldout event mask", heldout_event_mask)
    if heldout.shape != (batch, blocks, events):
        raise ValueError("heldout event mask shape differs")
    if block_event_role_mask is None:
        source_mask = heldout.unsqueeze(-1).expand(-1, -1, -1, roles)
    else:
        role_mask = _boolean_mask(
            "block event-role mask", block_event_role_mask,
        )
        if role_mask.shape != (batch, blocks, events, roles):
            raise ValueError("block event-role mask shape differs")
        source_mask = heldout.unsqueeze(-1) & role_mask
    source_count = source_mask.to(torch.int64).sum(dim=1)
    if bool(torch.any(source_count > 1)):
        raise ValueError("endpoint event feature has multiple source blocks")
    finite = torch.isfinite(block_event_feature).all(dim=-1)
    if bool(torch.any(source_mask & ~finite)):
        raise ValueError("selected block endpoint feature is non-finite")
    selected = torch.where(
        source_mask.unsqueeze(-1), block_event_feature,
        torch.zeros_like(block_event_feature),
    ).sum(dim=1)
    output_mask = source_count == 1
    selected = torch.where(
        output_mask.unsqueeze(-1), selected, torch.zeros_like(selected),
    )
    return {
        "event_feature": selected,
        "event_mask": output_mask,
        "source_count": source_count,
    }


def same_time_unordered_pair_temporal_features(
    event_feature: torch.Tensor,
    event_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build six symmetric same-time pair tokens from scalar role tokens.

    For each unordered pair the token is ``[mean, absolute difference,
    product]`` channel-wise.  All operations are scalar and symmetric in the
    two roles, so a synchronized S4 relabel only permutes the six pair tokens.
    """
    batch, events, width = _check_endpoint_feature(event_feature)
    mask = _boolean_mask("endpoint event mask", event_mask)
    if mask.shape != (batch, events, ROLE_COUNT):
        raise ValueError("endpoint event mask shape differs")
    if bool(torch.any(mask & ~torch.isfinite(event_feature).all(dim=-1))):
        raise ValueError("visible endpoint scalar feature is non-finite")
    values: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for left, right in UNORDERED_PAIR_INDICES:
        left_value = event_feature[:, :, left]
        right_value = event_feature[:, :, right]
        values.append(torch.cat((
            0.5 * (left_value + right_value),
            (left_value - right_value).abs(),
            left_value * right_value,
        ), dim=-1))
        masks.append(mask[:, :, left] & mask[:, :, right])
    pair_feature = torch.stack(values, dim=2)
    pair_mask = torch.stack(masks, dim=2)
    pair_feature = torch.where(
        pair_mask.unsqueeze(-1), pair_feature,
        torch.zeros_like(pair_feature),
    )
    if pair_feature.shape != (batch, events, 6, 3 * width):
        raise RuntimeError("unordered pair feature shape differs")
    return {"pair_feature": pair_feature, "pair_mask": pair_mask}


def _per_role_temporal_donor_plan(
    event_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one temporal derangement per sample and anonymous role."""
    mask = _boolean_mask("temporal break event mask", event_mask)
    if mask.ndim != 3 or mask.shape[-1] != ROLE_COUNT:
        raise ValueError("temporal break event mask must have shape [B,T,4]")
    batch, events, roles = mask.shape
    donor = torch.arange(events, device=mask.device).view(
        1, events, 1,
    ).expand(batch, -1, roles).clone()
    valid_role = mask.sum(dim=1) >= 2
    for row in range(batch):
        for role in range(roles):
            visible = torch.nonzero(mask[row, :, role], as_tuple=False).flatten()
            if visible.numel() >= 2:
                donor[row, visible, role] = visible.roll(-1)
    event_index = torch.arange(events, device=mask.device).view(1, -1, 1)
    if bool(torch.any(
        mask & valid_role[:, None, :] & (donor == event_index)
    )):
        raise RuntimeError("per-role temporal break has a visible fixed point")
    return donor, valid_role


def _gather_time_per_role(
    value: torch.Tensor, donor: torch.Tensor,
) -> torch.Tensor:
    """Gather ``[B,T,4,...]`` along time with a ``[B,T,4]`` plan."""
    if value.ndim < 3 or value.shape[:3] != donor.shape:
        raise ValueError("per-role temporal gather shapes differ")
    trailing = value.shape[3:]
    transposed = value.transpose(1, 2)
    donor_transposed = donor.transpose(1, 2)
    gather_index = donor_transposed.view(
        *donor_transposed.shape, *(1 for _ in trailing),
    ).expand(*donor_transposed.shape, *trailing)
    return torch.gather(transposed, 2, gather_index).transpose(1, 2)


def _validate_per_role_temporal_donor_plan(
    event_mask: torch.Tensor,
    donor: torch.Tensor,
) -> torch.Tensor:
    """Validate a visible-time derangement and return its valid-role mask."""
    batch, events, roles = event_mask.shape
    if donor.shape != (batch, events, roles):
        raise ValueError("temporal donor event index shape differs")
    if bool(torch.any((donor < 0) | (donor >= events))):
        raise ValueError("temporal donor event index is out of range")
    valid_role = event_mask.sum(dim=1) >= 2
    for row in range(batch):
        for role in range(roles):
            visible = torch.nonzero(
                event_mask[row, :, role], as_tuple=False,
            ).flatten()
            if visible.numel() < 2:
                continue
            selected = donor[row, visible, role]
            if torch.equal(selected.sort().values, visible):
                continue
            raise ValueError(
                "temporal donor must permute each role's visible times",
            )
    event_index = torch.arange(events, device=donor.device).view(1, -1, 1)
    if bool(torch.any(
        event_mask & valid_role[:, None, :] & (donor == event_index)
    )):
        raise ValueError("temporal donor has a visible fixed point")
    return valid_role


def deterministic_temporal_association_break(
    event_feature: torch.Tensor,
    event_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Derange only residual suffixes within each role's visible times.

    Channels ``0:3`` are the recipient event's time/primary/switch context and
    remain bit-exact.  Visibility also remains bit-exact.  Only channels
    ``3:`` move, independently for every sample and anonymous role.
    """
    batch, events, width = _check_endpoint_feature(event_feature)
    if width <= 3:
        raise ValueError("temporal association break needs residual suffix 3:")
    role_mask = _boolean_mask("endpoint event mask", event_mask)
    if role_mask.shape != (batch, events, ROLE_COUNT):
        raise ValueError("endpoint event mask shape differs")
    donor, valid_role = _per_role_temporal_donor_plan(role_mask)
    moved_suffix = _gather_time_per_role(event_feature[..., 3:], donor)
    suffix = torch.where(
        valid_role[:, None, :, None], moved_suffix, event_feature[..., 3:],
    )
    return {
        "event_feature": torch.cat((event_feature[..., :3], suffix), dim=-1),
        "event_mask": role_mask.clone(),
        "donor_event_index": donor,
        "break_valid": valid_role,
    }


def conjugate_role_donor_index(
    role_donor_index: torch.Tensor,
    role_permutation: torch.Tensor,
) -> torch.Tensor:
    """Conjugate a role-donor plan under ``new[j] = old[perm[j]]``."""
    if role_donor_index.ndim != 3 or role_donor_index.shape[-1] != ROLE_COUNT:
        raise ValueError("role donor index must have shape [B,T,4]")
    permutation = role_permutation.to(
        device=role_donor_index.device, dtype=torch.int64,
    )
    if permutation.shape != (ROLE_COUNT,) or sorted(permutation.tolist()) != list(
        range(ROLE_COUNT)
    ):
        raise ValueError("role permutation must contain 0..3 exactly once")
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(
        ROLE_COUNT, device=permutation.device,
    )
    old_donor_for_new_recipient = role_donor_index[:, :, permutation]
    return inverse[old_donor_for_new_recipient]


def deterministic_per_event_s4_continuity_break(
    event_feature: torch.Tensor,
    event_mask: torch.Tensor,
    *,
    role_donor_index: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Apply a synchronized no-fixed role permutation at every active event.

    The default schedule cycles through shifts 1, 2, and 3 by active-event
    rank.  A caller testing another S4 labelling may pass the conjugated donor
    plan returned by :func:`conjugate_role_donor_index`.
    """
    batch, events, width = _check_endpoint_feature(event_feature)
    role_mask = _boolean_mask("endpoint event mask", event_mask)
    if role_mask.shape != (batch, events, ROLE_COUNT):
        raise ValueError("endpoint event mask shape differs")
    active = role_mask.any(dim=-1)
    if role_donor_index is None:
        rank = active.to(torch.int64).cumsum(dim=1) - 1
        shift = 1 + torch.remainder(rank.clamp_min(0), ROLE_COUNT - 1)
        recipient = torch.arange(
            ROLE_COUNT, device=event_feature.device,
        ).view(1, 1, ROLE_COUNT)
        donor = torch.remainder(recipient + shift.unsqueeze(-1), ROLE_COUNT)
        donor = torch.where(active.unsqueeze(-1), donor, recipient)
    else:
        donor = role_donor_index.to(
            device=event_feature.device, dtype=torch.int64,
        )
        if donor.shape != (batch, events, ROLE_COUNT):
            raise ValueError("role donor index shape differs")
        if bool(torch.any((donor < 0) | (donor >= ROLE_COUNT))):
            raise ValueError("role donor index is out of range")
        sorted_donor = donor.sort(dim=-1).values
        expected = torch.arange(
            ROLE_COUNT, device=donor.device,
        ).view(1, 1, ROLE_COUNT)
        if bool(torch.any(active.unsqueeze(-1) & (sorted_donor != expected))):
            raise ValueError("active role donor index is not an S4 permutation")
    recipient = torch.arange(
        ROLE_COUNT, device=event_feature.device,
    ).view(1, 1, ROLE_COUNT)
    if bool(torch.any(active.unsqueeze(-1) & (donor == recipient))):
        raise ValueError("active per-event S4 break contains a fixed role")
    gathered_feature = torch.gather(
        event_feature, 2, donor.unsqueeze(-1).expand(-1, -1, -1, width),
    )
    gathered_mask = torch.gather(role_mask, 2, donor)
    return {
        "event_feature": gathered_feature,
        "event_mask": gathered_mask,
        "donor_role_index": donor,
        "break_valid": active,
    }


def deterministic_endpoint_dose_temporal_break(
    event_feature: torch.Tensor,
    event_mask: torch.Tensor,
    *,
    dose: int,
    donor_event_index: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Enumerate every anonymous endpoint subset at dose 1, 2, or 4.

    Selected endpoint streams receive the reusable per-role no-fixed temporal
    donor plan; unselected streams stay exact.  Enumerating *all* subsets
    avoids choosing a privileged role.  Under S4 relabelling, variants merely
    permute according to their returned ``damaged_role_mask``.
    """
    if dose not in DOSE_VALUES:
        raise ValueError("endpoint dose must be one of 1, 2, or 4")
    batch, events, width = _check_endpoint_feature(event_feature)
    if width <= 3:
        raise ValueError("endpoint dose break needs residual suffix 3:")
    role_mask = _boolean_mask("endpoint event mask", event_mask)
    if role_mask.shape != (batch, events, ROLE_COUNT):
        raise ValueError("endpoint event mask shape differs")
    if donor_event_index is None:
        donor, valid_role = _per_role_temporal_donor_plan(role_mask)
    else:
        donor = donor_event_index.to(
            device=event_feature.device, dtype=torch.int64,
        )
        valid_role = _validate_per_role_temporal_donor_plan(role_mask, donor)
    moved_suffix = _gather_time_per_role(event_feature[..., 3:], donor)
    subsets = tuple(combinations(range(ROLE_COUNT), dose))
    damaged = torch.zeros(
        len(subsets), ROLE_COUNT, dtype=torch.bool, device=event_feature.device,
    )
    for variant, subset in enumerate(subsets):
        damaged[variant, list(subset)] = True
    damage = damaged.view(1, len(subsets), 1, ROLE_COUNT)
    valid = valid_role.view(batch, 1, 1, ROLE_COUNT)
    suffix = torch.where(
        (damage & valid).unsqueeze(-1), moved_suffix[:, None],
        event_feature[:, None, ..., 3:],
    )
    prefix = event_feature[:, None, ..., :3].expand(
        -1, len(subsets), -1, -1, -1,
    )
    feature = torch.cat((prefix, suffix), dim=-1)
    mask = role_mask[:, None].expand(-1, len(subsets), -1, -1).clone()
    return {
        "event_feature": feature,
        "event_mask": mask,
        "damaged_role_mask": damaged,
        "donor_event_index": donor,
        "break_valid": valid_role,
    }


__all__ = [
    "DOSE_VALUES", "ROLE_COUNT", "UNORDERED_PAIR_INDICES",
    "combine_blockwise_endpoint_event_features",
    "conjugate_role_donor_index",
    "deterministic_endpoint_dose_temporal_break",
    "deterministic_per_event_s4_continuity_break",
    "deterministic_temporal_association_break",
    "fixed_history_leave_block_out_masks",
    "same_time_unordered_pair_temporal_features",
]
