"""Strict physical-motion truth joins for the Stage-3 future predictor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .observable_future_pnp_ab import sha256_file


TRUTH_SCHEMA = "stage3-truth-history-v1"
JOIN_KEY_SCHEMA = "split/session_id/exposure_t0_ns-v1"
MOTION_TARGET_FIELD = "target_motion_state_physical"
NORMALIZED_MOTION_TARGET_FIELD = "target_motion_state_normalized"


class MotionTruthIndex:
    """Immutable unique-key index over train/validation truth-history rows."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_manifest_sha256: str | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        manifest_path = self.root / "dataset_manifest.json"
        self.manifest_sha256 = sha256_file(manifest_path)
        if (
            expected_manifest_sha256 is not None
            and self.manifest_sha256 != expected_manifest_sha256
        ):
            raise ValueError("truth-history manifest differs from paired dataset binding")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != TRUTH_SCHEMA:
            raise ValueError("motion truth schema mismatch")
        if bool(self.manifest.get("test_accessed", True)):
            raise ValueError("motion truth accessed test")
        if not bool(self.manifest.get("qualification_passed", False)):
            raise ValueError("motion truth is not qualified")
        if set(self.manifest.get("splits", ())) != {"train", "validation"}:
            raise ValueError("motion truth must contain only train and validation")

        records: dict[tuple[str, str, int], tuple[int, np.ndarray]] = {}
        shard_tokens: list[str] = []
        required = {
            "session_id", "t0_ns", "motion_class",
            "anchor_velocity_mps", "anchor_yaw_rate_rad_s",
        }
        for shard in self.manifest["shards"]:
            split = str(shard["split"])
            if split not in {"train", "validation"}:
                raise ValueError("motion truth shard opened a forbidden split")
            path = self.root / Path(str(shard["path"]).replace("\\", "/"))
            actual_sha = sha256_file(path)
            if actual_sha != str(shard["sha256"]):
                raise ValueError(f"motion truth shard hash mismatch: {path}")
            shard_tokens.append(f"{split}\x1f{shard['path']}\x1f{actual_sha}")
            with np.load(path, allow_pickle=False) as loaded:
                missing = required - set(loaded.files)
                if missing:
                    raise ValueError(f"motion truth shard fields missing: {sorted(missing)}")
                count = len(loaded["t0_ns"])
                if count != int(shard["sample_count"]):
                    raise ValueError("motion truth shard sample count mismatch")
                state = np.concatenate((
                    loaded["anchor_velocity_mps"].astype(np.float32, copy=False),
                    loaded["anchor_yaw_rate_rad_s"].astype(
                        np.float32, copy=False,
                    )[:, None],
                ), axis=1)
                if state.shape != (count, 4) or not np.isfinite(state).all():
                    raise ValueError("motion truth contains invalid 4D state")
                for row in range(count):
                    key = (
                        split, str(loaded["session_id"][row]),
                        int(loaded["t0_ns"][row]),
                    )
                    if key in records:
                        raise ValueError(f"duplicate motion truth join key: {key}")
                    records[key] = (int(loaded["motion_class"][row]), state[row].copy())
        if len(records) != int(self.manifest["sample_count"]):
            raise ValueError("motion truth manifest sample count mismatch")
        self.records = records
        key_digest = hashlib.sha256()
        label_digest = hashlib.sha256()
        for key in sorted(records):
            encoded = "\x1f".join((key[0], key[1], str(key[2]))).encode("utf-8")
            key_digest.update(encoded)
            motion_class, state = records[key]
            label_digest.update(encoded)
            label_digest.update(str(motion_class).encode("ascii"))
            label_digest.update(state.astype("<f4", copy=False).tobytes())
        self.key_set_sha256 = key_digest.hexdigest()
        self.label_sha256 = label_digest.hexdigest()
        self.shard_set_sha256 = hashlib.sha256(
            "\n".join(sorted(shard_tokens)).encode("utf-8")
        ).hexdigest()

    def attach(self, dataset: Any, split: str) -> dict[str, Any]:
        """Attach one physical label per paired window using an exact 1:1 join."""
        if split not in {"train", "validation"}:
            raise ValueError("motion truth attachment forbids test")
        matched = 0
        joined_keys: set[tuple[str, str, int]] = set()
        for part in dataset.parts:
            states: list[np.ndarray] = []
            for session_id, t0_ns in zip(
                part.session_ids, part.t0_ns, strict=True,
            ):
                key = (split, str(session_id), int(t0_ns))
                if key in joined_keys:
                    raise ValueError(f"paired dataset has duplicate join key: {key}")
                joined_keys.add(key)
                record = self.records.get(key)
                if record is None:
                    raise ValueError(f"paired window lacks exact motion truth: {key}")
                motion_class, state = record
                if motion_class != int(part.motion_class):
                    raise ValueError(f"motion class differs at truth join key: {key}")
                states.append(state)
            tensor = torch.from_numpy(np.stack(states).astype(np.float32, copy=False))
            if MOTION_TARGET_FIELD in part.tensors:
                raise ValueError("motion truth target was already attached")
            part.tensors[MOTION_TARGET_FIELD] = tensor
            matched += len(states)
        split_truth_count = sum(key[0] == split for key in self.records)
        join_digest = hashlib.sha256()
        for key in sorted(joined_keys):
            join_digest.update(
                "\x1f".join((key[0], key[1], str(key[2]))).encode("utf-8")
            )
        return {
            "join_key_schema": JOIN_KEY_SCHEMA,
            "split": split,
            "paired_count": len(dataset),
            "matched_count": matched,
            "missing_count": len(dataset) - matched,
            "truth_split_count": split_truth_count,
            "extra_truth_count": split_truth_count - matched,
            "joined_key_set_sha256": join_digest.hexdigest(),
            "physical_label_semantics": (
                "tracker/chassis vx,vy,vz and physical yaw rate; invariant to "
                "anonymous C4 origin and slot reflection"
            ),
        }

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "path": str(self.root),
            "schema_version": TRUTH_SCHEMA,
            "manifest_sha256": self.manifest_sha256,
            "join_key_schema": JOIN_KEY_SCHEMA,
            "record_count": len(self.records),
            "key_set_sha256": self.key_set_sha256,
            "label_sha256": self.label_sha256,
            "shard_set_sha256": self.shard_set_sha256,
            "test_accessed": False,
        }


def fit_motion_scales(dataset: Any) -> torch.Tensor:
    """Fit fixed normalization from exactly the joined eligible train windows."""
    values = torch.cat([
        part.tensors[MOTION_TARGET_FIELD].to(torch.float64)
        for part in dataset.parts
    ], dim=0)
    if values.ndim != 2 or values.shape[1] != 4 or not bool(torch.isfinite(values).all()):
        raise ValueError("joined motion labels are invalid")
    p99 = torch.quantile(values.abs(), 0.99, dim=0)
    minimum = torch.tensor((0.25, 0.25, 0.25, 0.5), dtype=torch.float64)
    return torch.maximum(minimum, 1.25 * p99).to(torch.float32)


def normalize_attached_motion(dataset: Any, scale: torch.Tensor) -> None:
    if scale.shape != (4,) or not bool(torch.all(scale > 0)):
        raise ValueError("motion normalization scale must have shape [4]")
    for part in dataset.parts:
        physical = part.tensors[MOTION_TARGET_FIELD]
        part.tensors[NORMALIZED_MOTION_TARGET_FIELD] = physical / scale


__all__ = [
    "JOIN_KEY_SCHEMA", "MOTION_TARGET_FIELD", "NORMALIZED_MOTION_TARGET_FIELD",
    "MotionTruthIndex", "fit_motion_scales", "normalize_attached_motion",
]
