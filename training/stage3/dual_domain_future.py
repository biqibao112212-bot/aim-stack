"""Fail-closed external routing for independent clean and PnP F experts."""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
from typing import Any

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


def load_formal_dual_domain_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_stage: str = "selector",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load only a passed, canonical-protocol fixed-final PnP F artifact."""
    if expected_stage not in {"trajectory", "selector"}:
        raise ValueError("formal dual-domain stage must be trajectory or selector")
    path = Path(checkpoint_path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "stage3-dual-domain-pnp-f-v1":
        raise ValueError("formal dual-domain checkpoint schema mismatch")
    provenance = payload.get("provenance", {})
    if (
        provenance.get("training_stage") != expected_stage
        or provenance.get("formal_oracle_evaluation") is not True
        or provenance.get("fixed_final_checkpoint") is not True
        or provenance.get("diagnostic_only") is not False
        or provenance.get("deployable_pipeline") is not False
        or not isinstance(provenance.get("formal_source_contract"), dict)
    ):
        raise ValueError("formal dual-domain checkpoint provenance is incomplete")

    from .formal_run_contract import (
        load_protocol,
        repository_root,
        require_formal_checkpoint_manifest,
        sha256_file,
    )
    from .observable_future_pnp_ab import load_observable_f_checkpoint

    protocol_path, protocol = load_protocol()
    contract = provenance["formal_source_contract"]
    if (
        contract.get("protocol_path")
        != protocol_path.relative_to(repository_root()).as_posix()
        or contract.get("protocol_sha256") != sha256_file(protocol_path)
        or contract.get("protocol_schema_version")
        != protocol["schema_version"]
    ):
        raise ValueError("formal dual-domain checkpoint protocol is not canonical")
    manifest_path = path.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require_formal_checkpoint_manifest(
        f"formal dual-domain {expected_stage}",
        path,
        manifest,
        expected_update=int(protocol[expected_stage]["fixed_final_update"]),
        checkpoint_update=int(payload.get("update", -1)),
    )
    if expected_stage == "selector" and not (
        manifest.get("gate", {}).get("conditional_output_bit_exact") is True
        and manifest.get("gate", {}).get("upstream_input_bit_exact") is True
    ):
        raise ValueError("formal selector did not preserve its frozen partitions")
    model, loaded = load_observable_f_checkpoint(path)
    loaded["formal_provenance"] = provenance
    loaded["formal_manifest_path"] = str(manifest_path)
    loaded["formal_manifest_sha256"] = sha256_file(manifest_path)
    return model, loaded
