#!/usr/bin/env python3
"""Build and attest the locked Linux TensorRT detector engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    module_root = Path(__file__).resolve().parents[1]
    repository = module_root.parents[1]
    workspace = repository.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        type=Path,
        default=workspace / "models/engines/linux/armor-0526-fp16.onnx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=workspace
        / "models/engines/linux/armor-0526-trt11.2.1-cuda13.3-sm89-fp16.engine",
    )
    parser.add_argument(
        "--builder",
        type=Path,
        default=repository / "build/autoaim-research/autoaim_research_build_engine",
    )
    parser.add_argument("--workspace-mib", type=int, default=2048)
    args = parser.parse_args()

    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_partial = metadata_path.with_suffix(metadata_path.suffix + ".partial")
    engine_partial = Path(str(args.output) + ".partial")
    for path in (args.output, metadata_path, metadata_partial, engine_partial):
        if path.exists():
            parser.error(f"refusing to overwrite protected artifact: {path}")
    for path in (args.onnx, args.builder):
        if not path.is_file():
            parser.error(f"required input is missing: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    tensorrt_library_dir = (
        workspace
        / "deps/tensorrt-11.2.1-cuda13.3/root/usr/lib/x86_64-linux-gnu"
    )
    environment["LD_LIBRARY_PATH"] = str(tensorrt_library_dir) + (
        ":" + environment["LD_LIBRARY_PATH"]
        if environment.get("LD_LIBRARY_PATH")
        else ""
    )
    completed = subprocess.run(
        [
            str(args.builder),
            "--onnx",
            str(args.onnx),
            "--output",
            str(args.output),
            "--workspace-mib",
            str(args.workspace_mib),
            "--fp16",
        ],
        text=True,
        stdout=subprocess.PIPE,
        env=environment,
        check=True,
    )
    print(completed.stdout, end="")
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader",
        ],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    metadata = {
        "schema": "aim-stack.tensorrt-engine/v1",
        "kind": "linux_tensorrt_engine",
        "engine": str(args.output),
        "engine_bytes": args.output.stat().st_size,
        "engine_sha256": sha256(args.output),
        "onnx": str(args.onnx),
        "onnx_bytes": args.onnx.stat().st_size,
        "onnx_sha256": sha256(args.onnx),
        "tensorrt": "11.2.1",
        "cuda": "13.3",
        "precision": "fp16",
        "workspace_mib": args.workspace_mib,
        "gpu": gpu,
        "builder_output": completed.stdout.splitlines(),
        "validation": "build and same-process deserialization succeeded",
    }
    metadata_partial.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata_partial.replace(metadata_path)
    print(f"metadata={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
