"""Freeze the one formal fixed-6-mm multistate capture and split contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .build_dataset import _load_formal_manifest, stratified_session_split


SCHEMA_VERSION = "stage3-multistate-capture-contract-v1"
PROFILE = "fixed6mm-24session-12segment-3second-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def freeze_contract(manifest: Path, output: Path, split_seed: int) -> dict[str, Any]:
    manifest = manifest.resolve()
    records = _load_formal_manifest(manifest)
    if len(records) != 24:
        raise ValueError("formal multistate capture requires exactly 24 sessions")
    family_counts = {"spin": 0, "linear_and_spin": 0}
    dataset_ids: set[str] = set()
    for record in records:
        session_id = str(record["session_id"])
        if record.get("schema_version") != "stage3-multistate-manifest-v2":
            raise ValueError(f"formal session is not multistate v2: {session_id}")
        mode = str(record.get("mode"))
        if mode not in family_counts:
            raise ValueError(f"formal session has invalid family: {session_id}/{mode}")
        family_counts[mode] += 1
        dataset_ids.add(str(record.get("dataset_id", "")))
        if record.get("camera_profile") != "wide_6mm" or bool(record.get("dual_focal", True)):
            raise ValueError(f"formal session is not fixed wide_6mm: {session_id}")
        segments = list(record.get("segments", ()))
        if len(segments) != 12:
            raise ValueError(f"formal session must contain exactly 12 segments: {session_id}")
        stationary = 0
        active = 0
        for index, segment in enumerate(segments):
            if int(segment.get("segment_index", -1)) != index:
                raise ValueError(f"non-contiguous formal segment index: {session_id}/{index}")
            if not math.isclose(float(segment.get("duration_s", 0.0)), 3.0, abs_tol=1e-12):
                raise ValueError(f"formal segment duration is not 3 seconds: {session_id}/{index}")
            segment_mode = str(segment.get("mode"))
            if segment_mode == "stationary":
                stationary += 1
            elif segment_mode == mode:
                active += 1
            else:
                raise ValueError(f"formal segment leaves its family: {session_id}/{index}")
        if stationary != 1 or active != 11:
            raise ValueError(f"formal session must contain 1 stationary + 11 active: {session_id}")
        if not math.isclose(float(record.get("duration_s", 0.0)), 36.0, abs_tol=1e-12):
            raise ValueError(f"formal session duration is not 36 seconds: {session_id}")
    if family_counts != {"spin": 12, "linear_and_spin": 12}:
        raise ValueError(f"formal family counts differ: {family_counts}")
    if len(dataset_ids) != 1 or "" in dataset_ids:
        raise ValueError("formal manifest must contain one nonempty dataset_id")

    splits = stratified_session_split(records, split_seed)
    if {name: len(values) for name, values in splits.items()} != {
        "train": 14, "validation": 5, "test": 5,
    }:
        raise AssertionError("formal split sizes are not 14/5/5")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "profile": PROFILE,
        "dataset_id": next(iter(dataset_ids)),
        "formal_manifest": str(manifest),
        "formal_manifest_sha256": _sha256(manifest),
        "session_count": 24,
        "family_session_counts": family_counts,
        "segments_per_session": 12,
        "stationary_segments_per_session": 1,
        "active_segments_per_session": 11,
        "segment_duration_s": 3.0,
        "camera_profile": "wide_6mm",
        "dual_focal": False,
        "split_seed": int(split_seed),
        "splits": splits,
        "splits_sha256": _canonical_sha256(splits),
        "post_capture_requirements": {
            "raw_camera_profile_scan": True,
            "per_segment_survival_table": True,
            "minimum_heldout_active_segments_with_samples": 8,
            "derived_test_accessed": False,
        },
    }
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite capture contract: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    args = parser.parse_args()
    payload = freeze_contract(Path(args.manifest), Path(args.output), args.split_seed)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
