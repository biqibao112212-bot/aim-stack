"""Independently seal split isolation for a pre-contract formal A3 H run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cyclic_future_foundation import load_frozen_v19
from .observable_future_pnp_ab import (
    load_observable_f_checkpoint,
    sha256_file,
)
from .pnp_q0_hypothesis_adapter import load_frozen_pnp_mapper


AUDIT_SCHEMA = "stage3-pnp-q0-hypothesis-formal-split-audit-v1"
RUN_SCHEMA = "stage3-pnp-q0-hypothesis-adapter-run-v1"


def _set_sha256(values: set[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _scan_split(
    dataset: Path,
    manifest: dict[str, Any],
    split: str,
) -> tuple[dict[str, Any], set[str], set[str]]:
    sessions: set[str] = set()
    sample_keys: set[str] = set()
    selected_count = 0
    shard_records: set[str] = set()
    for item in manifest["shards"]:
        if str(item["split"]) != split:
            continue
        path = dataset / Path(str(item["path"]).replace("\\", "/"))
        current_sha = sha256_file(path)
        if current_sha != str(item["sha256"]):
            raise ValueError(f"split audit shard hash mismatch: {path}")
        shard_records.add(f"{item['path']}\x1f{current_sha}")
        with np.load(path, allow_pickle=False) as loaded:
            keep = loaded["pnp_sf_common_usable"].astype(np.bool_)
            selected_count += int(keep.sum())
            for session, t0_ns, pair_id in zip(
                loaded["session_id"][keep],
                loaded["t0_ns"][keep],
                loaded["pair_id"][keep],
            ):
                session_text = str(session)
                sessions.add(session_text)
                sample_keys.add(
                    f"{session_text}\x1f{int(t0_ns)}\x1f{str(pair_id)}"
                )
    return ({
        "split": split,
        "sample_count": selected_count,
        "unique_sample_key_count": len(sample_keys),
        "duplicate_sample_key_count": selected_count - len(sample_keys),
        "session_count": len(sessions),
        "session_set_sha256": _set_sha256(sessions),
        "sample_key_set_sha256": _set_sha256(sample_keys),
        "shard_set_sha256": _set_sha256(shard_records),
        "sample_strategy": "full_split",
        "sample_limit": 0,
        "motion_class": None,
    }, sessions, sample_keys)


def audit(args: argparse.Namespace) -> Path:
    dataset = Path(args.dataset).resolve()
    manifest_path = dataset / "dataset_manifest.json"
    manifest_sha = sha256_file(manifest_path)
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_manifest_path = Path(args.run_manifest).resolve()
    run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    best = run.get("best", {})
    matching_history = [
        item for item in run.get("history", [])
        if item.get("checkpoint") == checkpoint_path.name
    ]
    if len(matching_history) != 1:
        raise ValueError("formal H best must have exactly one history record")
    history = matching_history[0]
    arguments = run.get("training_arguments", {})
    if (
        run.get("schema_version") != RUN_SCHEMA
        or run.get("status") != "complete"
        or bool(run.get("validation_from_train", True))
        or bool(run.get("test_accessed", True))
        or arguments.get("train_limit") != 0
        or arguments.get("validation_limit") != 0
        or arguments.get("motion_class") != -1
        or bool(arguments.get("validation_from_train", True))
        or run.get("dataset_manifest_sha256") != manifest_sha
        or best.get("path") != checkpoint_path.name
        or best.get("sha256") != checkpoint_sha
        or best.get("epoch") != checkpoint.get("epoch")
        or best.get("update") != checkpoint.get("update")
        or best.get("selection") != list(checkpoint.get("selection", ()))
        or best.get("validation") != checkpoint.get("validation")
        or history.get("checkpoint_sha256") != checkpoint_sha
        or history.get("epoch") != checkpoint.get("epoch")
        or history.get("update") != checkpoint.get("update")
        or history.get("selection") != list(checkpoint.get("selection", ()))
        or history.get("validation") != checkpoint.get("validation")
    ):
        raise ValueError("formal H run/checkpoint linkage is invalid")
    if (
        dataset_manifest.get("qualification_passed") is not True
        or dataset_manifest.get("test_accessed") is not False
        or dataset_manifest.get("diagnostic_subset") is not False
    ):
        raise ValueError("formal H dataset qualification is invalid")
    if not all(bool(run.get(name, False)) for name in (
        "frozen_mapper_verified_unchanged",
        "frozen_s_verified_unchanged",
        "frozen_f_verified_unchanged",
    )):
        raise ValueError("formal H run did not preserve every frozen model")

    train_audit, train_sessions, train_keys = _scan_split(
        dataset, dataset_manifest, "train"
    )
    validation_audit, validation_sessions, validation_keys = _scan_split(
        dataset, dataset_manifest, "validation"
    )
    if (
        train_audit["sample_count"] != run.get("train_sample_count")
        or validation_audit["sample_count"]
        != run.get("validation_sample_count")
        or train_audit["duplicate_sample_key_count"] != 0
        or validation_audit["duplicate_sample_key_count"] != 0
    ):
        raise ValueError("formal H split counts or uniqueness are invalid")
    session_overlap = train_sessions & validation_sessions
    sample_overlap = train_keys & validation_keys
    if session_overlap or sample_overlap:
        raise ValueError("formal H train/validation membership overlaps")

    provenance = checkpoint.get("provenance", {})
    if provenance.get("dataset_manifest_sha256") != manifest_sha:
        raise ValueError("formal H checkpoint dataset provenance mismatch")
    mapper, mapper_info = load_frozen_pnp_mapper(
        provenance["frozen_mapper"]["path"]
    )
    s_model, s_info = load_frozen_v19(
        provenance["frozen_s"]["checkpoint_path"]
    )
    f_model, f_info = load_observable_f_checkpoint(
        provenance["frozen_f"]["path"]
    )
    for name, recorded, loaded in (
        ("mapper", provenance["frozen_mapper"], mapper_info),
        ("F", provenance["frozen_f"], f_info),
    ):
        if any(recorded.get(key) != loaded.get(key) for key in (
            "sha256", "state_dict_sha256"
        )):
            raise ValueError(f"formal H {name} provenance mismatch")
    if (
        provenance["frozen_s"].get("checkpoint_sha256")
        != s_info.get("checkpoint_sha256")
        or provenance["frozen_s"].get("state_dict_sha256")
        != s_info.get("state_dict_sha256")
        or mapper_info["provenance"]["frozen_s"]["state_dict_sha256"]
        != s_info["state_dict_sha256"]
    ):
        raise ValueError("formal H mapper/S provenance chain mismatch")
    del mapper, s_model, f_model

    source_files = (
        Path(__file__),
        Path(__file__).with_name("pnp_q0_hypothesis_adapter.py"),
        Path(__file__).with_name("train_pnp_q0_hypothesis_adapter.py"),
        Path(__file__).with_name("promote_pnp_q0_hypothesis_checkpoint.py"),
    )
    result = {
        "schema_version": AUDIT_SCHEMA,
        "qualified_split_isolation": True,
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": manifest_sha,
        "run_manifest_path": str(run_manifest_path),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "source_checkpoint_path": str(checkpoint_path),
        "source_checkpoint_sha256": checkpoint_sha,
        "source_checkpoint_epoch": int(checkpoint["epoch"]),
        "source_checkpoint_update": int(checkpoint["update"]),
        "train": train_audit,
        "validation": validation_audit,
        "session_overlap_count": 0,
        "sample_key_overlap_count": 0,
        "validation_from_train": False,
        "diagnostic_only": True,
        "diagnostic_reasons": ["legacy_training_source_bundle_unrecoverable"],
        "common_usable_only": True,
        "test_accessed": False,
        "frozen_mapper": mapper_info,
        "frozen_s": s_info,
        "frozen_f": f_info,
        "legacy_training_source_bundle_recoverable": False,
        "audit_and_promotion_source_sha256": {
            path.name: sha256_file(path) for path in source_files
        },
        "training_arguments": arguments,
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite formal H audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    print(audit(parser.parse_args()))


if __name__ == "__main__":
    main()
