#!/usr/bin/env python3
"""Verify the pinned source snapshot and optional workspace dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise SystemExit(f"missing locked file: {path}")
    actual = sha256(path)
    if actual.lower() != expected.lower():
        raise SystemExit(
            f"locked file hash mismatch: {path}\nexpected={expected}\nactual={actual}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()

    module = Path(__file__).resolve().parents[1]
    lock = json.loads((module / "implementation.lock.json").read_text("utf-8"))
    if lock["status"] != "sole_internal_implementation":
        raise SystemExit("implementation lock is not authoritative")

    vendor = module / "third_party" / "tongji_sp_vision_25"
    for relative, expected in lock["autoaim"]["vendored_sha256"].items():
        require_hash(vendor / relative, expected)

    if args.workspace is not None:
        workspace = args.workspace.resolve()
        require_hash(
            workspace / lock["model"]["relative_to_workspace"],
            lock["model"]["sha256"],
        )
        release = workspace / lock["simulator"]["release_relative_to_workspace"]
        require_hash(
            release / "release.json", lock["simulator"]["release_json_sha256"]
        )
        require_hash(
            release / "release-manifest.json",
            lock["simulator"]["release_manifest_sha256"],
        )
        runtime = workspace / lock["inference_runtime"]["relative_to_workspace"]
        require_hash(
            runtime / "lib" / "libonnxruntime.so",
            lock["inference_runtime"]["libonnxruntime_sha256"],
        )

    print(
        "implementation_lock_ok "
        f"upstream={lock['autoaim']['upstream_commit']} "
        f"simulator={lock['simulator']['exact_release']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
