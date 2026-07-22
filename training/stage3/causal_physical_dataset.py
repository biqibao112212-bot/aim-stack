"""Bounded-memory loader for causal clean-physics train/validation shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CausalPhysicalShardDataset(IterableDataset):
    """Expose normalized xyz+cyclic-slot features and raw metre histories."""

    def __init__(
        self, dataset_dir: str | Path, split: str, *, seed: int,
        shuffle: bool = False, sample_limit: int = 0,
        session_ids: Iterable[str] | None = None,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir).resolve()
        if split not in {"train", "validation"}:
            raise ValueError("causal physical loader only permits train or validation")
        self.split = split
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.sample_limit = int(sample_limit)
        self.session_ids = (
            None if session_ids is None
            else frozenset(str(value) for value in session_ids)
        )
        self.epoch = 0
        manifest_path = self.dataset_dir / "dataset_manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != "stage3-causal-physical-v1":
            raise ValueError("causal physical dataset schema mismatch")
        if not bool(self.manifest.get("qualification_passed", False)):
            raise ValueError("causal physical dataset is not qualified")
        if bool(self.manifest.get("test_accessed", True)):
            raise ValueError("causal physical dataset must record test_accessed=false")
        self.shards = [
            item for item in self.manifest["shards"] if item["split"] == split
        ]
        if self.session_ids is not None:
            self.shards = [
                item for item in self.shards
                if self.session_ids.intersection(
                    str(value) for value in item.get("session_ids", ())
                )
            ]
        if not self.shards:
            raise ValueError(f"causal physical dataset has no {split} shards")
        normalization_path = self.dataset_dir / str(self.manifest["normalization"])
        if _sha256(normalization_path) != str(self.manifest["normalization_sha256"]):
            raise ValueError("causal physical normalization hash mismatch")
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        self.mean = np.asarray(normalization["position_m"]["mean"], dtype=np.float32)
        self.std = np.asarray(normalization["position_m"]["std"], dtype=np.float32)
        if self.mean.shape != (3,) or self.std.shape != (3,) or np.any(self.std <= 0):
            raise ValueError("invalid causal physical normalization")
        declared = sum(int(item["sample_count"]) for item in self.shards)
        self._length = min(declared, sample_limit) if sample_limit > 0 else declared
        found_sessions = {
            str(value) for item in self.shards for value in item.get("session_ids", ())
            if self.session_ids is not None and str(value) in self.session_ids
        }
        if self.session_ids is not None and found_sessions != set(self.session_ids):
            raise ValueError(
                f"causal physical sessions missing from {split}: "
                f"{sorted(set(self.session_ids) - found_sessions)}"
            )
        for item in self.shards:
            path = self.dataset_dir / str(item["path"])
            if _sha256(path) != str(item["sha256"]):
                raise ValueError(f"causal physical shard hash mismatch: {path}")
        angle = np.arange(4, dtype=np.float32) * (np.pi / 2.0)
        self.slot_features = np.stack((np.sin(angle), np.cos(angle)), axis=-1)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._length

    @staticmethod
    def _validate_arrays(arrays: dict[str, np.ndarray]) -> None:
        required = {
            "history_position_m", "history_obs_mask", "event_mask", "event_time_s",
            "tau", "future_position", "motion_class", "rule_query", "distance_m",
            "session_id", "t0_ns",
        }
        missing = required - arrays.keys()
        if missing:
            raise ValueError(f"causal physical shard is missing arrays: {sorted(missing)}")
        if arrays["history_position_m"].shape[1:] != (200, 4, 3):
            raise ValueError("causal physical history shape must be [N,200,4,3]")
        if arrays["history_obs_mask"].shape[1:] != (200, 4):
            raise ValueError("causal physical history mask shape must be [N,200,4]")
        if not np.array_equal(
            arrays["event_mask"], arrays["history_obs_mask"].any(axis=2)
        ):
            raise ValueError("event_mask must equal history_obs_mask.any(-1)")
        if arrays["future_position"].shape[1:] != (8, 4, 3):
            raise ValueError("causal physical future shape must be [N,8,4,3]")
        if arrays["tau"].shape[1:] != (8,) or arrays["rule_query"].shape[1:] != (8,):
            raise ValueError("causal physical query shape must be [N,8]")

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
                raise ValueError(f"causal physical shard changed before open: {path}")
            with np.load(path, allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            self._validate_arrays(arrays)
            indices = np.arange(int(item["sample_count"]))
            if self.shuffle:
                rng.shuffle(indices)
            for index in indices:
                if self.sample_limit > 0 and emitted >= self.sample_limit:
                    return
                position_m = arrays["history_position_m"][index].astype(
                    np.float32, copy=True
                )
                mask = arrays["history_obs_mask"][index].astype(np.bool_, copy=False)
                normalized = (position_m - self.mean) / self.std
                slot = np.broadcast_to(self.slot_features[None, :, :], (200, 4, 2))
                obs = np.concatenate((normalized, slot), axis=-1).astype(np.float32)
                obs = np.where(mask[..., None], obs, 0.0).astype(np.float32, copy=False)
                position_m = np.where(
                    mask[..., None], position_m, 0.0
                ).astype(np.float32, copy=False)
                emitted += 1
                yield {
                    "obs": torch.from_numpy(obs),
                    "history_position_m": torch.from_numpy(position_m),
                    "obs_mask": torch.from_numpy(mask),
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
                    "rule_query": torch.from_numpy(
                        arrays["rule_query"][index].astype(np.bool_, copy=False)
                    ),
                    "distance_m": torch.tensor(
                        float(arrays["distance_m"][index]), dtype=torch.float32
                    ),
                }
