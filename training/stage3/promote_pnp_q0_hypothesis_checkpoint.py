"""Seal a pre-contract H run as diagnostic evidence with audited splits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from .cyclic_future_foundation import load_frozen_v19
from .observable_future_pnp_ab import (
    load_observable_f_checkpoint,
    sha256_file,
    state_dict_sha256,
)
from .pnp_q0_hypothesis_adapter import (
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)


RUN_SCHEMA = "stage3-pnp-q0-hypothesis-adapter-run-v1"
AUDIT_SCHEMA = "stage3-pnp-q0-hypothesis-formal-split-audit-v1"


def promote(
    source_checkpoint: str | Path,
    run_manifest: str | Path,
    split_audit: str | Path,
    output_checkpoint: str | Path,
) -> Path:
    source = Path(source_checkpoint).resolve()
    manifest_path = Path(run_manifest).resolve()
    audit_path = Path(split_audit).resolve()
    output = Path(output_checkpoint).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite promoted H: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    best = manifest.get("best", {})
    frozen_ok = all(bool(manifest.get(name, False)) for name in (
        "frozen_mapper_verified_unchanged",
        "frozen_s_verified_unchanged",
        "frozen_f_verified_unchanged",
    ))
    if (
        manifest.get("schema_version") != RUN_SCHEMA
        or manifest.get("status") != "complete"
        or bool(manifest.get("validation_from_train", True))
        or bool(manifest.get("test_accessed", True))
        or not frozen_ok
        or best.get("path") != source.name
        or best.get("sha256") != sha256_file(source)
    ):
        raise ValueError("H run is not eligible for diagnostic sealing")
    if (
        audit.get("schema_version") != AUDIT_SCHEMA
        or audit.get("qualified_split_isolation") is not True
        or audit.get("diagnostic_only") is not True
        or audit.get("dataset_manifest_sha256")
        != manifest.get("dataset_manifest_sha256")
        or audit.get("run_manifest_sha256") != sha256_file(manifest_path)
        or audit.get("source_checkpoint_sha256") != sha256_file(source)
        or audit.get("session_overlap_count") != 0
        or audit.get("sample_key_overlap_count") != 0
    ):
        raise ValueError("H independent split audit is invalid")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != RUN_SCHEMA
        or payload.get("model_class") != "C4Q0HypothesisAdapter"
        or payload.get("validation") != best.get("validation")
        or list(payload.get("selection", ())) != list(best.get("selection", ()))
    ):
        raise ValueError("H best checkpoint and run manifest disagree")
    provenance = dict(payload.get("provenance", {}))
    existing_diagnostic = provenance.get("diagnostic_only")
    existing_from_train = provenance.get("validation_from_train")
    if existing_diagnostic not in (None, True) or existing_from_train not in (
        None, False
    ):
        raise ValueError("legacy H checkpoint diagnostic provenance is invalid")
    mapper, mapper_info = load_frozen_pnp_mapper(
        provenance["frozen_mapper"]["path"]
    )
    s_model, s_info = load_frozen_v19(
        provenance["frozen_s"]["checkpoint_path"]
    )
    f_model, f_info = load_observable_f_checkpoint(
        provenance["frozen_f"]["path"]
    )
    if mapper_info["state_dict_sha256"] != provenance["frozen_mapper"][
        "state_dict_sha256"
    ]:
        raise ValueError("promotion mapper state provenance mismatch")
    if s_info["state_dict_sha256"] != provenance["frozen_s"][
        "state_dict_sha256"
    ]:
        raise ValueError("promotion S state provenance mismatch")
    if (
        mapper_info["provenance"]["frozen_s"]["state_dict_sha256"]
        != s_info["state_dict_sha256"]
    ):
        raise ValueError("promotion mapper/S provenance chain is broken")
    if (
        f_info["sha256"] != provenance["frozen_f"]["sha256"]
        or f_info["state_dict_sha256"]
        != provenance["frozen_f"]["state_dict_sha256"]
    ):
        raise ValueError("diagnostic seal F provenance mismatch")
    provenance.update({
        "validation_from_train": False,
        "diagnostic_only": True,
        "diagnostic_reasons": ["legacy_training_source_bundle_unrecoverable"],
        "support_metric_diagnostic_only": True,
        "support_output_consumed_by_f": False,
        "diagnostic_seal": {
            "source_checkpoint_path": str(source),
            "source_checkpoint_sha256": sha256_file(source),
            "run_manifest_path": str(manifest_path),
            "run_manifest_sha256": sha256_file(manifest_path),
            "split_audit_path": str(audit_path),
            "split_audit_sha256": sha256_file(audit_path),
        },
    })
    promoted = dict(payload)
    promoted["provenance"] = provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(
        f".{output.name}.pending-{os.getpid()}-{time.time_ns()}"
    )
    try:
        torch.save(promoted, pending)
        loaded, info = load_frozen_hypothesis_adapter(
            pending, allow_diagnostic=True
        )
        source_state_sha = state_dict_sha256(payload["model"])
        if info["state_dict_sha256"] != source_state_sha:
            raise RuntimeError("diagnostic sealing changed H model state")
        del loaded
        pending.replace(output)
    finally:
        if pending.exists():
            pending.unlink()
    del mapper, s_model, f_model
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--split-audit", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    args = parser.parse_args()
    print(promote(
        args.source_checkpoint, args.run_manifest, args.split_audit,
        args.output_checkpoint,
    ))


if __name__ == "__main__":
    main()
