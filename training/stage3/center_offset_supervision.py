"""Strict loss-only chassis-center truth joins for Stage-3 state screens.

The stored label is an absolute physical center only so the immutable truth
join is independent of whichever frozen S/H checkpoint defines the online
current-primary origin.  Training code must form ``center - H_current`` after
``frozen_upstream_batch`` and must never include either target in forward
fields.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .motion_truth_supervision import JOIN_KEY_SCHEMA, TRUTH_SCHEMA
from .observable_future_pnp_ab import sha256_file


CENTER_POSITION_TARGET_FIELD = "target_anchor_center_position_m_physical"
CENTER_OFFSET_TARGET_FIELD = "target_center_offset_from_h_current_m"


class CenterTruthIndex:
    """Immutable exact-key index containing no inference-time feature."""

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
            raise ValueError("center truth manifest differs from paired binding")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != TRUTH_SCHEMA:
            raise ValueError("center truth schema mismatch")
        if bool(self.manifest.get("test_accessed", True)):
            raise ValueError("center truth accessed test")
        if not bool(self.manifest.get("qualification_passed", False)):
            raise ValueError("center truth is not qualified")
        if set(self.manifest.get("splits", ())) != {"train", "validation"}:
            raise ValueError("center truth must contain only train and validation")

        records: dict[tuple[str, str, int], tuple[int, np.ndarray]] = {}
        shard_tokens: list[str] = []
        required = {
            "session_id", "t0_ns", "motion_class", "anchor_center_position_m",
        }
        for shard in self.manifest["shards"]:
            split = str(shard["split"])
            if split not in {"train", "validation"}:
                raise ValueError("center truth shard opened a forbidden split")
            path = self.root / Path(str(shard["path"]).replace("\\", "/"))
            actual_sha = sha256_file(path)
            if actual_sha != str(shard["sha256"]):
                raise ValueError(f"center truth shard hash mismatch: {path}")
            shard_tokens.append(f"{split}\x1f{shard['path']}\x1f{actual_sha}")
            with np.load(path, allow_pickle=False) as loaded:
                missing = required - set(loaded.files)
                if missing:
                    raise ValueError(f"center truth shard fields missing: {sorted(missing)}")
                count = len(loaded["t0_ns"])
                if count != int(shard["sample_count"]):
                    raise ValueError("center truth shard sample count mismatch")
                centers = loaded["anchor_center_position_m"].astype(
                    np.float32, copy=False,
                )
                if centers.shape != (count, 3) or not np.isfinite(centers).all():
                    raise ValueError("center truth contains invalid 3D centers")
                for row in range(count):
                    key = (
                        split, str(loaded["session_id"][row]),
                        int(loaded["t0_ns"][row]),
                    )
                    if key in records:
                        raise ValueError(f"duplicate center truth join key: {key}")
                    records[key] = (
                        int(loaded["motion_class"][row]), centers[row].copy(),
                    )
        if len(records) != int(self.manifest["sample_count"]):
            raise ValueError("center truth manifest sample count mismatch")
        self.records = records

        key_digest = hashlib.sha256()
        label_digest = hashlib.sha256()
        for key in sorted(records):
            encoded = "\x1f".join((key[0], key[1], str(key[2]))).encode("utf-8")
            key_digest.update(encoded)
            motion_class, center = records[key]
            label_digest.update(encoded)
            label_digest.update(str(motion_class).encode("ascii"))
            label_digest.update(center.astype("<f4", copy=False).tobytes())
        self.key_set_sha256 = key_digest.hexdigest()
        self.label_sha256 = label_digest.hexdigest()
        self.shard_set_sha256 = hashlib.sha256(
            "\n".join(sorted(shard_tokens)).encode("utf-8")
        ).hexdigest()

    def attach(self, dataset: Any, split: str) -> dict[str, Any]:
        """Attach one loss-only absolute center using an exact 1:1 join."""
        if split not in {"train", "validation"}:
            raise ValueError("center truth attachment forbids test")
        matched = 0
        joined_keys: set[tuple[str, str, int]] = set()
        for part in dataset.parts:
            centers: list[np.ndarray] = []
            for session_id, t0_ns in zip(
                part.session_ids, part.t0_ns, strict=True,
            ):
                key = (split, str(session_id), int(t0_ns))
                if key in joined_keys:
                    raise ValueError(f"paired dataset has duplicate join key: {key}")
                joined_keys.add(key)
                record = self.records.get(key)
                if record is None:
                    raise ValueError(f"paired window lacks exact center truth: {key}")
                motion_class, center = record
                if motion_class != int(part.motion_class):
                    raise ValueError(f"motion class differs at center join key: {key}")
                centers.append(center)
            if CENTER_POSITION_TARGET_FIELD in part.tensors:
                raise ValueError("center truth target was already attached")
            part.tensors[CENTER_POSITION_TARGET_FIELD] = torch.from_numpy(
                np.stack(centers).astype(np.float32, copy=False),
            )
            matched += len(centers)

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
                "loss-only chassis center in tracker coordinates; converted to "
                "an anonymous H-current offset after frozen upstream inference"
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


def attach_center_offset_after_frozen_upstream(
    prepared: dict[str, torch.Tensor], raw: dict[str, torch.Tensor],
) -> None:
    """Create the only train-time center offset in the H-current frame."""
    if CENTER_POSITION_TARGET_FIELD not in raw:
        raise ValueError("joined absolute center target missing")
    current = prepared.get("current_position_m")
    target = raw[CENTER_POSITION_TARGET_FIELD]
    if current is None or current.shape != target.shape or target.shape[-1] != 3:
        raise ValueError("center target/current shapes differ")
    offset = target.detach() - current.detach()
    if not bool(torch.isfinite(offset).all()):
        raise ValueError("center offset target is non-finite")
    prepared[CENTER_OFFSET_TARGET_FIELD] = offset


__all__ = [
    "CENTER_OFFSET_TARGET_FIELD", "CENTER_POSITION_TARGET_FIELD",
    "CenterTruthIndex", "attach_center_offset_after_frozen_upstream",
]
