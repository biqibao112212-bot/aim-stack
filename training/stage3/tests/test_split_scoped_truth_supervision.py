from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from training.stage3.center_offset_supervision import CENTER_POSITION_TARGET_FIELD
from training.stage3.motion_truth_supervision import (
    MOTION_TARGET_FIELD,
    TRUTH_SCHEMA,
)
from training.stage3.observable_future_pnp_ab import sha256_file
from training.stage3.split_scoped_truth_supervision import (
    SplitScopedTruthIndex,
    assert_manifest_split_shards_unchanged,
)


class _Part:
    def __init__(self) -> None:
        self.session_ids = np.asarray(["train-session"])
        self.t0_ns = np.asarray([17], dtype=np.int64)
        self.motion_class = 2
        self.tensors: dict[str, torch.Tensor] = {}


class _Dataset:
    def __init__(self) -> None:
        self.parts = [_Part()]

    def __len__(self) -> int:
        return 1


def _write_truth_root(root: Path) -> Path:
    root.mkdir()
    train_path = root / "train.npz"
    np.savez(
        train_path,
        session_id=np.asarray(["train-session"]),
        t0_ns=np.asarray([17], dtype=np.int64),
        motion_class=np.asarray([2], dtype=np.int64),
        anchor_velocity_mps=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        anchor_yaw_rate_rad_s=np.asarray([4.0], dtype=np.float32),
        anchor_center_position_m=np.asarray([[5.0, 6.0, 7.0]], dtype=np.float32),
    )
    # Intentionally do not create validation.npz.  A train-scoped index must
    # succeed without hashing or opening it.
    manifest = {
        "schema_version": TRUTH_SCHEMA,
        "test_accessed": False,
        "qualification_passed": True,
        "splits": ["train", "validation"],
        "shards": [
            {
                "split": "train", "path": "train.npz",
                "sha256": sha256_file(train_path), "sample_count": 1,
            },
            {
                "split": "validation", "path": "validation.npz",
                "sha256": "a" * 64, "sample_count": 999,
            },
        ],
    }
    manifest_path = root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_train_scoped_truth_never_opens_validation_shard(tmp_path: Path) -> None:
    manifest_path = _write_truth_root(tmp_path / "truth")
    index = SplitScopedTruthIndex(
        manifest_path.parent, split="train",
        expected_manifest_sha256=sha256_file(manifest_path),
    )
    assert [Path(item["path"]).name for item in index.accessed_shards] == [
        "train.npz"
    ]
    dataset = _Dataset()
    joined = index.attach(dataset)
    assert joined["validation_truth_accessed"] is False
    part = dataset.parts[0]
    assert torch.equal(
        part.tensors[MOTION_TARGET_FIELD],
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
    )
    assert torch.equal(
        part.tensors[CENTER_POSITION_TARGET_FIELD],
        torch.tensor([[5.0, 6.0, 7.0]]),
    )
    index.assert_unchanged()
    assert_manifest_split_shards_unchanged(
        manifest_path.parent, index.manifest,
        split="train", label="truth fixture",
    )


def test_validation_scope_requires_validation_shard_to_exist(tmp_path: Path) -> None:
    manifest_path = _write_truth_root(tmp_path / "truth")
    with pytest.raises(FileNotFoundError):
        SplitScopedTruthIndex(
            manifest_path.parent, split="validation",
            expected_manifest_sha256=sha256_file(manifest_path),
        )


def test_split_scoped_truth_forbids_test(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        SplitScopedTruthIndex(tmp_path, split="test")
