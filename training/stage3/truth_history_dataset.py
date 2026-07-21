"""Bounded-memory loader for causal exact-truth history shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TruthHistoryShardDataset(IterableDataset):
    def __init__(
        self, dataset_dir: str | Path, split: str, *, seed: int,
        shuffle: bool = False, sample_limit: int = 0,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir).resolve()
        if split not in {"train", "validation"}:
            raise ValueError("truth-history loader only permits train or validation")
        self.split = split
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.sample_limit = int(sample_limit)
        self.epoch = 0
        manifest = json.loads(
            (self.dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("schema_version") != "stage3-truth-history-v1" or not bool(
            manifest.get("qualification_passed", False)
        ):
            raise ValueError("truth-history dataset is not qualified")
        if bool(manifest.get("test_accessed", True)):
            raise ValueError("truth-history training dataset must record test_accessed=false")
        self.shards = [item for item in manifest["shards"] if item["split"] == split]
        if not self.shards:
            raise ValueError(f"truth-history dataset has no {split} shards")
        normalization_path = self.dataset_dir / str(manifest["normalization"])
        if _sha256(normalization_path) != str(manifest["normalization_sha256"]):
            raise ValueError("truth-history normalization hash mismatch")
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        self.mean = np.asarray(normalization["position_m"]["mean"], dtype=np.float32)
        self.std = np.asarray(normalization["position_m"]["std"], dtype=np.float32)
        if self.mean.shape != (3,) or self.std.shape != (3,) or np.any(self.std <= 0):
            raise ValueError("invalid truth-history normalization")
        declared = sum(int(item["sample_count"]) for item in self.shards)
        self._length = min(declared, sample_limit) if sample_limit > 0 else declared
        for item in self.shards:
            path = self.dataset_dir / str(item["path"])
            if _sha256(path) != str(item["sha256"]):
                raise ValueError(f"truth-history shard hash mismatch: {path}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rng = np.random.default_rng(self.seed + self.epoch * 100003)
        shard_order = np.arange(len(self.shards))
        if self.shuffle:
            rng.shuffle(shard_order)
        emitted = 0
        for shard_index in shard_order:
            item = self.shards[int(shard_index)]
            path = self.dataset_dir / str(item["path"])
            if _sha256(path) != str(item["sha256"]):
                raise ValueError(f"truth-history shard changed before open: {path}")
            with np.load(path, allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            indices = np.arange(int(item["sample_count"]))
            if self.shuffle:
                rng.shuffle(indices)
            for index in indices:
                if self.sample_limit > 0 and emitted >= self.sample_limit:
                    return
                obs = arrays["truth_obs"][index].astype(np.float32, copy=True)
                obs = (obs - self.mean) / self.std
                emitted += 1
                yield {
                    "obs": torch.from_numpy(obs),
                    "obs_mask": torch.from_numpy(
                        arrays["truth_obs_mask"][index].astype(np.bool_, copy=False)
                    ),
                    "event_mask": torch.from_numpy(
                        arrays["event_mask"][index].astype(np.bool_, copy=False)
                    ),
                    "event_time_s": torch.from_numpy(
                        arrays["event_time_s"][index].astype(np.float32, copy=False)
                    ),
                    "tau": torch.from_numpy(
                        arrays["tau"][index].astype(np.float32, copy=False)
                    ),
                    "future_position": torch.from_numpy(
                        arrays["future_position"][index].astype(np.float32, copy=False)
                    ),
                    "motion_class": torch.tensor(
                        int(arrays["motion_class"][index]), dtype=torch.long
                    ),
                    "anchor_velocity_mps": torch.from_numpy(
                        arrays["anchor_velocity_mps"][index].astype(np.float32, copy=False)
                    ),
                    "anchor_yaw_rate_rad_s": torch.tensor(
                        float(arrays["anchor_yaw_rate_rad_s"][index]), dtype=torch.float32
                    ),
                    "anchor_center_position_m": torch.from_numpy(
                        arrays["anchor_center_position_m"][index].astype(np.float32, copy=False)
                    ),
                    "rule_query": torch.from_numpy(
                        arrays["rule_query"][index].astype(np.bool_, copy=False)
                    ),
                    "distance_m": torch.tensor(
                        float(arrays["distance_m"][index]), dtype=torch.float32
                    ),
                }
