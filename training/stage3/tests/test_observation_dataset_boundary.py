from __future__ import annotations

import pytest

from training.stage3.build_observation_dataset import _select_training_shards
from training.stage3.build_observable_future_pnp_upper_bound_dataset import (
    _assert_sealed_observation_manifest,
)


def test_observation_derivative_selects_only_train_and_validation() -> None:
    manifest = {
        "shards": [
            {"path": "train.npz", "split": "train", "sample_count": 3},
            {"path": "validation.npz", "split": "validation", "sample_count": 2},
            {"path": "test-must-not-open.npz", "split": "test", "sample_count": 7},
        ]
    }

    selected = _select_training_shards(manifest)

    assert [item["path"] for item in selected] == ["train.npz", "validation.npz"]
    assert all(item["split"] != "test" for item in selected)


@pytest.mark.parametrize("present_split", ["train", "validation"])
def test_observation_derivative_requires_both_training_splits(present_split: str) -> None:
    manifest = {
        "shards": [
            {"path": f"{present_split}.npz", "split": present_split, "sample_count": 1},
            {"path": "test.npz", "split": "test", "sample_count": 1},
        ]
    }

    with pytest.raises(ValueError, match="complete train/validation"):
        _select_training_shards(manifest)


def test_pnp_accepts_only_sealed_train_validation_observation_manifest() -> None:
    manifest = {
        "test_accessed": False,
        "included_splits": ["train", "validation"],
        "shards": [
            {"split": "train"},
            {"split": "validation"},
        ],
    }

    _assert_sealed_observation_manifest(manifest)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"test_accessed": True}, "accessed test"),
        ({"test_accessed": None}, "accessed test"),
        ({"included_splits": ["train", "validation", "test"]}, "train/validation only"),
        ({"shards": [{"split": "test"}]}, "materialized a test shard"),
    ],
)
def test_pnp_rejects_unsealed_observation_manifest(
    update: dict[str, object], message: str
) -> None:
    manifest: dict[str, object] = {
        "test_accessed": False,
        "included_splits": ["train", "validation"],
        "shards": [{"split": "train"}, {"split": "validation"}],
    }
    if "test_accessed" in update and update["test_accessed"] is None:
        manifest.pop("test_accessed")
    else:
        manifest.update(update)

    with pytest.raises(ValueError, match=message):
        _assert_sealed_observation_manifest(manifest)
