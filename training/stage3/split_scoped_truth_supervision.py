"""Split-scoped physical truth joins for sealed Stage-3 experiments.

Older truth indexes eagerly opened both train and validation shards.  This
index reads the manifest globally but hashes and loads only the explicitly
requested split, so train-only structural screens cannot observe validation
truth even incidentally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .center_offset_supervision import CENTER_POSITION_TARGET_FIELD
from .motion_truth_supervision import MOTION_TARGET_FIELD, TRUTH_SCHEMA
from .observable_future_pnp_ab import sha256_file


class SplitScopedTruthIndex:
    """Immutable exact-key truth index for exactly one declared split."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        expected_manifest_sha256: str | None = None,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split-scoped truth forbids test/unknown split")
        self.root = Path(root).resolve()
        self.split = split
        self.manifest_path = self.root / "dataset_manifest.json"
        self.manifest_sha256 = sha256_file(self.manifest_path)
        if (
            expected_manifest_sha256 is not None
            and self.manifest_sha256 != expected_manifest_sha256
        ):
            raise ValueError("split-scoped truth manifest differs")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != TRUTH_SCHEMA:
            raise ValueError("split-scoped truth schema mismatch")
        if self.manifest.get("test_accessed") is not False:
            raise ValueError("split-scoped truth accessed test")
        if self.manifest.get("qualification_passed") is not True:
            raise ValueError("split-scoped truth is not qualified")
        if set(self.manifest.get("splits", ())) != {"train", "validation"}:
            raise ValueError("split-scoped truth manifest split set differs")
        required = {
            "session_id", "t0_ns", "motion_class",
            "anchor_velocity_mps", "anchor_yaw_rate_rad_s",
            "anchor_center_position_m",
        }
        records: dict[tuple[str, int], tuple[int, np.ndarray, np.ndarray]] = {}
        self.accessed_shards: list[dict[str, Any]] = []
        for shard in self.manifest["shards"]:
            if str(shard["split"]) != split:
                continue
            path = self.root / Path(str(shard["path"]).replace("\\", "/"))
            actual_sha = sha256_file(path)
            if actual_sha != str(shard["sha256"]):
                raise ValueError(f"split-scoped truth shard hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as loaded:
                missing = required - set(loaded.files)
                if missing:
                    raise ValueError(
                        f"split-scoped truth fields missing: {sorted(missing)}"
                    )
                count = len(loaded["t0_ns"])
                if count != int(shard["sample_count"]):
                    raise ValueError("split-scoped truth shard count differs")
                velocity = loaded["anchor_velocity_mps"].astype(
                    np.float32, copy=False,
                )
                yaw = loaded["anchor_yaw_rate_rad_s"].astype(
                    np.float32, copy=False,
                )
                center = loaded["anchor_center_position_m"].astype(
                    np.float32, copy=False,
                )
                state = np.concatenate((velocity, yaw[:, None]), axis=1)
                if (
                    state.shape != (count, 4)
                    or center.shape != (count, 3)
                    or not np.isfinite(state).all()
                    or not np.isfinite(center).all()
                ):
                    raise ValueError("split-scoped truth values are invalid")
                for row in range(count):
                    key = (
                        str(loaded["session_id"][row]),
                        int(loaded["t0_ns"][row]),
                    )
                    if key in records:
                        raise ValueError(f"duplicate split-scoped truth key: {key}")
                    records[key] = (
                        int(loaded["motion_class"][row]),
                        state[row].copy(), center[row].copy(),
                    )
            self.accessed_shards.append({
                "path": str(path), "sha256": actual_sha,
                "sample_count": int(shard["sample_count"]),
            })
        expected_count = sum(
            int(shard["sample_count"])
            for shard in self.manifest["shards"]
            if str(shard["split"]) == split
        )
        if not self.accessed_shards or len(records) != expected_count:
            raise ValueError("split-scoped truth record count differs")
        self.records = records

    def attach(self, dataset: Any) -> dict[str, Any]:
        matched = 0
        joined: set[tuple[str, int]] = set()
        for part in dataset.parts:
            states: list[np.ndarray] = []
            centers: list[np.ndarray] = []
            for session_id, t0_ns in zip(
                part.session_ids, part.t0_ns, strict=True,
            ):
                key = (str(session_id), int(t0_ns))
                if key in joined:
                    raise ValueError(f"duplicate paired split key: {key}")
                joined.add(key)
                record = self.records.get(key)
                if record is None:
                    raise ValueError(f"paired row lacks {self.split} truth: {key}")
                motion_class, state, center = record
                if motion_class != int(part.motion_class):
                    raise ValueError(f"motion class differs at split key: {key}")
                states.append(state)
                centers.append(center)
            if (
                MOTION_TARGET_FIELD in part.tensors
                or CENTER_POSITION_TARGET_FIELD in part.tensors
            ):
                raise ValueError("split-scoped truth was already attached")
            part.tensors[MOTION_TARGET_FIELD] = torch.from_numpy(
                np.stack(states).astype(np.float32, copy=False)
            )
            part.tensors[CENTER_POSITION_TARGET_FIELD] = torch.from_numpy(
                np.stack(centers).astype(np.float32, copy=False)
            )
            matched += len(states)
        if matched != len(dataset) or len(joined) != len(dataset):
            raise ValueError("split-scoped truth join is not exact 1:1")
        return {
            "split": self.split,
            "paired_count": len(dataset),
            "matched_count": matched,
            "accessed_shard_count": len(self.accessed_shards),
            "validation_truth_accessed": self.split == "validation",
        }

    def assert_unchanged(self) -> None:
        if sha256_file(self.manifest_path) != self.manifest_sha256:
            raise RuntimeError("split-scoped truth manifest changed")
        for shard in self.accessed_shards:
            if sha256_file(shard["path"]) != shard["sha256"]:
                raise RuntimeError("split-scoped truth accessed shard changed")


def assert_manifest_split_shards_unchanged(
    root: str | Path,
    manifest: dict[str, Any],
    *,
    split: str,
    label: str,
) -> None:
    """Rehash only shards whose split was actually authorized and opened."""
    if split not in {"train", "validation"}:
        raise ValueError("manifest split check forbids test/unknown split")
    base = Path(root).resolve()
    selected = [
        shard for shard in manifest["shards"] if str(shard["split"]) == split
    ]
    if not selected:
        raise ValueError(f"{label} has no {split} shards")
    for shard in selected:
        path = base / Path(str(shard["path"]).replace("\\", "/"))
        if sha256_file(path) != str(shard["sha256"]):
            raise RuntimeError(f"{label} {split} shard changed: {path}")


__all__ = ["SplitScopedTruthIndex", "assert_manifest_split_shards_unchanged"]
