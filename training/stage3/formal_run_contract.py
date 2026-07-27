"""Fail-closed source and environment contract for formal-oracle Stage3 runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

import torch


PROTOCOL_SCHEMA = "stage3-formal-oracle-protocol-v1"
SOURCE_CONTRACT_SCHEMA = "stage3-formal-source-contract-v1"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_protocol(path: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    canonical = Path(__file__).with_name("formalization_protocol.json").resolve()
    resolved = Path(path).resolve() if path is not None else canonical
    if resolved != canonical:
        raise ValueError("formal-oracle runs require the canonical protocol path")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported Stage3 formalization protocol")
    if payload.get("runtime") != {
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "tf32": False,
    }:
        raise ValueError("formal runtime contract differs from the implementation")
    return resolved, payload


def _git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True,
        encoding="utf-8", errors="strict",
    ).strip()


def _nvidia_identity() -> dict[str, str]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,pci.bus_id,uuid",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        text=True, encoding="utf-8", errors="strict",
    ).strip()
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("formal runtime requires one resolved CUDA device 0")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 3 or not all(fields):
        raise RuntimeError("could not resolve formal NVIDIA device identity")
    return {
        "driver_version": fields[0],
        "pci_bus_id": fields[1],
        "gpu_uuid": fields[2],
    }


def capture_formal_contract(
    source_paths: Iterable[str | Path],
    *,
    protocol_path: str | Path | None = None,
    requested_device: str,
    workers: int,
) -> dict[str, Any]:
    root = repository_root()
    commit = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError("formal-oracle training requires a clean worktree")
    protocol, payload = load_protocol(protocol_path)
    files: dict[str, str] = {}
    for raw_path in source_paths:
        path = Path(raw_path)
        resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("formal source must stay inside the repository") from error
        if not resolved.is_file():
            raise FileNotFoundError(f"formal source is missing: {relative}")
        tracked = _git_output(root, "ls-files", "--error-unmatch", relative)
        if tracked != relative:
            raise RuntimeError(f"formal source is not tracked exactly: {relative}")
        files[relative] = sha256_file(resolved)
    protocol_relative = protocol.relative_to(root).as_posix()
    if protocol_relative not in files:
        files[protocol_relative] = sha256_file(protocol)
    return {
        "schema_version": SOURCE_CONTRACT_SCHEMA,
        "git_commit": commit,
        "git_branch": _git_output(root, "branch", "--show-current"),
        "worktree_clean": True,
        "source_sha256": dict(sorted(files.items())),
        "protocol_path": protocol_relative,
        "protocol_sha256": sha256_file(protocol),
        "protocol_schema_version": payload["schema_version"],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available() else None
            ),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "cuda_device_index": (
                torch.cuda.current_device() if torch.cuda.is_available() else None
            ),
            "cuda_capability": (
                list(torch.cuda.get_device_capability(0))
                if torch.cuda.is_available() else None
            ),
            "cuda_total_memory": (
                torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available() else None
            ),
            "requested_device": requested_device,
            "resolved_device": f"cuda:{torch.cuda.current_device()}",
            "workers": int(workers),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "nvidia": _nvidia_identity(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
    }


def verify_formal_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema_version") != SOURCE_CONTRACT_SCHEMA:
        raise ValueError("invalid formal source contract")
    current = capture_formal_contract(
        contract["source_sha256"].keys(),
        protocol_path=repository_root() / contract["protocol_path"],
        requested_device=contract["environment"]["requested_device"],
        workers=int(contract["environment"]["workers"]),
    )
    for key in (
        "git_commit", "git_branch", "worktree_clean", "source_sha256",
        "protocol_path", "protocol_sha256", "protocol_schema_version",
        "environment",
    ):
        if current[key] != contract[key]:
            raise RuntimeError(f"formal source contract changed during run: {key}")


def require_exact_protocol_arguments(
    arguments: Any,
    section: dict[str, Any],
    names: Iterable[str],
) -> None:
    mismatches = {
        name: {"expected": section[name], "actual": getattr(arguments, name)}
        for name in names
        if getattr(arguments, name) != section[name]
    }
    if mismatches:
        raise ValueError(
            "formal arguments differ from protocol: "
            + json.dumps(mismatches, sort_keys=True)
        )


def resolve_formal_schedule(
    requested_schedule_updates: int,
    planned_epoch_updates: int,
    section: dict[str, Any],
) -> tuple[int, int]:
    schedule_updates = int(section["schedule_total_updates"])
    fixed_final_update = int(section["fixed_final_update"])
    if requested_schedule_updates != schedule_updates:
        raise ValueError("formal LR schedule horizon differs from protocol")
    if planned_epoch_updates < schedule_updates:
        raise ValueError("formal epoch count cannot cover the LR schedule horizon")
    if not (0 < fixed_final_update <= schedule_updates):
        raise ValueError("formal fixed-final update is outside the LR schedule")
    return schedule_updates, fixed_final_update


def configure_formal_runtime(requested_device: str, workers: int) -> None:
    if requested_device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal-oracle Stage3 requires the locked CUDA runtime")
    if workers != 0:
        raise ValueError("formal-oracle Stage3 requires workers=0")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "formal-oracle Stage3 requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "before Python starts"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def require_asset_binding(
    label: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    normalized_actual = {
        "file_sha256": actual.get("sha256") or actual.get("checkpoint_sha256"),
        "state_dict_sha256": actual.get("state_dict_sha256"),
    }
    if normalized_actual != expected:
        raise ValueError(
            f"formal asset binding mismatch for {label}: "
            + json.dumps(
                {"expected": expected, "actual": normalized_actual},
                sort_keys=True,
            )
        )


def require_compatible_contracts(
    label: str,
    parent: dict[str, Any],
    current: dict[str, Any],
) -> None:
    for name in (
        "git_commit", "git_branch", "protocol_sha256",
        "protocol_schema_version", "environment",
    ):
        if parent.get(name) != current.get(name):
            raise ValueError(
                f"formal source contract mismatch for {label}: {name}"
            )


def require_fixed_final_state(
    label: str,
    section: dict[str, Any],
    *,
    update: int,
    stop_reason: str,
    history: list[dict[str, Any]],
    best: dict[str, Any] | None,
) -> None:
    expected_update = int(section["fixed_final_update"])
    if section.get("fixed_final_checkpoint") is not True:
        raise ValueError(f"{label} protocol is not fixed-final")
    if (
        update != expected_update
        or stop_reason != "fixed_final_update"
        or len(history) != 1
        or best is None
        or int(best.get("update", -1)) != expected_update
    ):
        raise RuntimeError(
            f"{label} did not produce exactly one fixed-final validation"
        )


def require_formal_checkpoint_manifest(
    label: str,
    checkpoint_path: str | Path,
    manifest: dict[str, Any],
    *,
    expected_update: int,
    checkpoint_update: int | None = None,
) -> None:
    path = Path(checkpoint_path).resolve()
    provenance = manifest.get("provenance", manifest)
    best = manifest.get("best", {})
    history = manifest.get("history", [])
    if (
        manifest.get("status") != "complete"
        or manifest.get("stop_reason") != "fixed_final_update"
        or manifest.get("formal_gate_passed") is not True
        or provenance.get("formal_oracle_evaluation") is not True
        or provenance.get("fixed_final_checkpoint") is not True
        or best.get("path") != path.name
        or best.get("sha256") != sha256_file(path)
        or int(manifest.get("update", -1)) != int(expected_update)
        or int(best.get("update", -1)) != int(expected_update)
        or len(history) != 1
        or int(history[0].get("update", -1)) != int(expected_update)
        or (
            checkpoint_update is not None
            and int(checkpoint_update) != int(expected_update)
        )
    ):
        raise ValueError(f"{label} is not a passed manifest-declared fixed final")
