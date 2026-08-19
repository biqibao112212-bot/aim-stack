#!/usr/bin/env python3
"""Qualify a create-once SDK RGBA + exposure capture for Release validators."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_EXPOSURE_FLAGS = 0b111
NATIVE_RGBA_BYTES = 1440 * 1080 * 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError(f"empty evidence: {path}")
    return rows


def identity(row: dict[str, object]) -> tuple[int, int, int]:
    return tuple(int(row[name]) for name in ("producer_epoch", "frame_seq", "timestamp_ns"))


def write_new_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def qualify(session: Path) -> dict[str, object]:
    session = session.resolve(strict=True)
    events_path = session / "capture-events.jsonl"
    exposures_path = session / "exposure-states.jsonl"
    ledger_path = session / "tcp-identities.jsonl"
    capture_manifest_path = session / "capture-manifest.json"
    exposure_manifest_path = session / "exposure-manifest.json"
    for output in (ledger_path, capture_manifest_path, exposure_manifest_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite protected evidence: {output}")
    events = load_jsonl(events_path)
    exposures = load_jsonl(exposures_path)
    event_keys = [identity(row) for row in events]
    exposure_keys = [identity(row) for row in exposures]
    if (
        len(set(event_keys)) != len(event_keys)
        or event_keys != sorted(event_keys)
        or len({key[0] for key in event_keys}) != 1
    ):
        raise ValueError("capture identities are duplicated or not monotonic")
    if (
        len(set(exposure_keys)) != len(exposure_keys)
        or exposure_keys != sorted(exposure_keys)
        or not set(exposure_keys) <= set(event_keys)
    ):
        raise ValueError("exposure identities are duplicated or not a capture subset")
    qualified: list[dict[str, object]] = []
    for row in events:
        if (
            row["pixel_format"] != "rgba32"
            or int(row["width"]) != 1440
            or int(row["height"]) != 1080
            or int(row["payload_bytes"]) != NATIVE_RGBA_BYTES
        ):
            raise ValueError("capture contains a non-native RGBA frame")
        relative = Path(str(row["raw_rgba_file"]))
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("frames",):
            raise ValueError(f"unsafe raw frame path: {relative}")
        raw = session / relative
        if not raw.is_file() or raw.stat().st_size != int(row["payload_bytes"]):
            raise ValueError(f"raw payload mismatch: {raw}")
        digest = sha256(raw)
        qualified.append({**row, "payload_sha256": digest, "raw_rgba_sha256": digest})
    for row in exposures:
        if row.get("schema_version") != "aim-stack.exposure-frame/1":
            raise ValueError("unsupported exposure schema")
        if int(row["state_flags"]) & REQUIRED_EXPOSURE_FLAGS != REQUIRED_EXPOSURE_FLAGS:
            raise ValueError("exposure row lacks chassis/gimbal/camera world pose")
        if row.get("online_target_truth_read") is not False or row.get("future_truth_included") is not False:
            raise ValueError("exposure sidecar violated truth boundary")
    with ledger_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in qualified:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    capture_manifest = {
        "schema_version": "daedalus.offline-frame-capture/1",
        "capture_mode": "until_eof",
        "frame_count": len(qualified),
        "producer_epoch": event_keys[0][0],
        "last_frame_seq": event_keys[-1][1],
        "image_format": "rgba32-raw",
        "frame_directory": "frames",
        "identity_ledger": "tcp-identities.jsonl",
        "online_truth_read": False,
        "future_truth_included": False,
    }
    write_new_json(capture_manifest_path, capture_manifest)
    exposure_manifest = {
        "schema_version": "aim-stack.exposure-capture/1",
        "status": "complete",
        "source": "DaedalusSimSdk-1.3.1/readExposureStateForFrame",
        "identity_contract": "exact producer_epoch/frame_seq/timestamp_ns join; no nearest-frame substitution",
        "required_state_flags": [
            "chassis_world_pose", "gimbal_world_pose", "camera_world_pose",
        ],
        "capture_identity_count": len(event_keys),
        "exposure_identity_count": len(exposure_keys),
        "coverage_fraction": len(exposure_keys) / len(event_keys),
        "online_target_truth_read": False,
        "future_truth_included": False,
        "artifacts": {
            "capture_events": {"path": str(events_path), "sha256": sha256(events_path)},
            "exposure_states": {"path": str(exposures_path), "sha256": sha256(exposures_path)},
            "tcp_identities": {"path": str(ledger_path), "sha256": sha256(ledger_path)},
        },
    }
    write_new_json(exposure_manifest_path, exposure_manifest)
    return {"capture": capture_manifest, "exposure": exposure_manifest}


def main() -> None:
    result = qualify(parse_args().session_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
