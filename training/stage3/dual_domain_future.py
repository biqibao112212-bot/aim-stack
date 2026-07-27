"""Fail-closed external routing for independent clean and PnP F experts."""

from __future__ import annotations

from enum import Enum

import torch


class ObservationDomain(str, Enum):
    CLEAN = "clean"
    PNP_V41 = "pnp_v41"


def assert_independent_models(
    clean_f: torch.nn.Module, pnp_f: torch.nn.Module,
) -> None:
    """Reject accidental parameter/buffer sharing across domain experts."""
    clean_state = clean_f.state_dict()
    pnp_state = pnp_f.state_dict()
    if clean_state.keys() != pnp_state.keys():
        raise ValueError("clean and PnP F state contracts differ")
    shared = [
        name for name in clean_state
        if clean_state[name].data_ptr() == pnp_state[name].data_ptr()
    ]
    if shared:
        raise ValueError(f"clean and PnP F share storage: {shared[:3]}")


def route_future_expert(
    domain: ObservationDomain,
    *,
    clean_f: torch.nn.Module,
    pnp_f: torch.nn.Module,
) -> torch.nn.Module:
    """Route by trusted observation provenance, never by a learned feature."""
    if not isinstance(domain, ObservationDomain):
        raise TypeError("observation domain must be an ObservationDomain enum")
    if domain is ObservationDomain.CLEAN:
        return clean_f
    if domain is ObservationDomain.PNP_V41:
        return pnp_f
    raise ValueError(f"unsupported observation domain: {domain!r}")
