"""Strict loading and freezing of the accepted V19 q0 state restorer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .cyclic_anchor_edge_model import CyclicAnchorEdgeRestorer


EXPECTED_FAMILY = "cyclic-equivariant-anchor-edge-q0-restorer-v2"
EXPECTED_SCHEMA = "stage3-cyclic-anchor-edge-restorer-run-v2"


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_frozen_v19(
    checkpoint_path: str | Path,
    *,
    expected_dataset_manifest_sha256: str | None = None,
) -> tuple[CyclicAnchorEdgeRestorer, dict[str, Any]]:
    """Load only a clean, completed, test-sealed V19 best checkpoint."""
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"V19 checkpoint missing: {checkpoint_path}")
    manifest_path = checkpoint_path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("V19 checkpoint requires final run_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(checkpoint_path)
    best = manifest.get("best", {})
    architecture = manifest.get("architecture_contract", {})
    if (
        manifest.get("schema_version") != EXPECTED_SCHEMA
        or
        manifest.get("status") != "complete"
        or manifest.get("stop_reason") != "epoch_limit"
        or bool(manifest.get("test_accessed", True))
        or bool(manifest.get("worktree_dirty", True))
        or not bool(manifest.get("frozen_foundation_verified_unchanged", False))
        or architecture.get("purpose") != "q0 current state recovery only"
        or bool(architecture.get("future_prediction", True))
    ):
        raise ValueError("V19 foundation is not a clean completed q0-only run")
    if (
        best.get("path") != checkpoint_path.name
        or best.get("sha256") != checkpoint_sha
        or not bool(best.get("trained_checkpoint", False))
    ):
        raise ValueError("V19 foundation must be the manifest-declared best checkpoint")
    if (
        expected_dataset_manifest_sha256 is not None
        and manifest.get("dataset_manifest_sha256")
        != expected_dataset_manifest_sha256
    ):
        raise ValueError("V19 foundation dataset differs from the F-layer dataset")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload.get("model_config", {})
    provenance = payload.get("provenance", {})
    if (
        payload.get("model_class") != "CyclicAnchorEdgeRestorer"
        or int(payload.get("epoch", -1)) != int(best.get("epoch", -2))
        or payload.get("checkpoint_role") != "validation"
        or config.get("family") != EXPECTED_FAMILY
        or (
            "model_config" in manifest
            and config != manifest.get("model_config")
        )
        or provenance.get("schema_version") != EXPECTED_SCHEMA
        or provenance.get("dataset_manifest_sha256")
        != manifest.get("dataset_manifest_sha256")
        or provenance.get("git_commit") != manifest.get("git_commit")
        or bool(provenance.get("worktree_dirty", True))
        or provenance.get("source_sha256") != manifest.get("source_sha256")
        or provenance.get("architecture_contract")
        != manifest.get("architecture_contract")
        or provenance.get("objective_contract") != manifest.get("objective_contract")
        or provenance.get("selection_contract") != manifest.get("selection_contract")
        or bool(provenance.get("test_accessed", True))
    ):
        raise ValueError("V19 checkpoint embedded provenance is incompatible")
    model = CyclicAnchorEdgeRestorer(
        torch.tensor(config["position_mean"], dtype=torch.float32),
        torch.tensor(config["position_std"], dtype=torch.float32),
        channels=int(config["channels"]),
        dropout=float(config["dropout"]),
        history_events=int(config["history_events"]),
    )
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False)
    model.eval()
    state_sha = state_dict_sha256(model.state_dict())
    return model, {
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": int(payload["epoch"]),
        "source_git_commit": manifest["git_commit"],
        "source_model_family": config["family"],
        "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "state_dict_sha256": state_sha,
        "test_accessed": False,
    }


class FrozenV19Adapter(torch.nn.Module):
    """Keep the S layer detached and deterministically in eval mode."""

    def __init__(self, foundation: CyclicAnchorEdgeRestorer) -> None:
        super().__init__()
        self.foundation = foundation
        self.foundation.requires_grad_(False)
        self.foundation.eval()

    def train(self, mode: bool = True) -> "FrozenV19Adapter":
        super().train(mode)
        self.foundation.eval()
        return self

    def forward(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        primary_mask: torch.Tensor,
        event_mask: torch.Tensor,
        event_time_s: torch.Tensor,
        switch_step: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.foundation.eval()
        with torch.no_grad():
            result = self.foundation(
                obs, obs_mask, primary_mask, event_mask,
                event_time_s, switch_step,
            )
        return {name: value.detach() for name, value in result.items()}
