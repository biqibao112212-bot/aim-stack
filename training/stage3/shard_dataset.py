"""Bounded-memory loader for qualified Stage-3 v3 event-sequence shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_shard_arrays(arrays: dict[str, np.ndarray], declared_count: int) -> None:
    required = (
        "obs", "obs_mask", "event_mask", "event_time_s", "tau", "future_position", "future_normal",
        "motion_class", "session_id", "t0_ns",
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"shard is missing arrays: {missing}")
    lengths = {name: int(arrays[name].shape[0]) for name in required}
    if set(lengths.values()) != {declared_count}:
        raise ValueError(f"shard first-dimension mismatch: {lengths}, declared={declared_count}")
    if arrays["obs"].shape[1:] != (200, 4, 5):
        raise ValueError(f"invalid observation tensor shape: {arrays['obs'].shape}")
    if arrays["event_mask"].shape[1:] != (200,) or arrays["event_time_s"].shape[1:] != (200,):
        raise ValueError("invalid event mask/time tensor shape")
    if not np.array_equal(arrays["event_mask"], arrays["obs_mask"].any(axis=2)):
        raise ValueError("event_mask must equal obs_mask.any(-1)")
    for mask, times in zip(arrays["event_mask"], arrays["event_time_s"]):
        valid_times = times[mask]
        if np.any(valid_times > 1e-6) or np.any(np.diff(valid_times) < 0.0):
            raise ValueError("valid event times must be ordered and no later than the anchor")
    if arrays["tau"].shape[1:] != (8,) or arrays["future_position"].shape[1:] != (8, 4, 3):
        raise ValueError("invalid query or target tensor shape")


class Stage3ShardDataset(IterableDataset):
    """Load one compressed shard at a time and yield normalized samples.

    Shards are split-specific, so selecting train or validation never opens a
    test file.  Session filters support overfit and pilot selection manifests.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        *,
        augment: bool,
        seed: int,
        shuffle: bool = False,
        session_ids: Iterable[str] | None = None,
        sample_limit: int = 0,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir).resolve()
        self.split = split
        self.augment = augment
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.sample_limit = sample_limit
        self.session_ids = None if session_ids is None else frozenset(str(value) for value in session_ids)
        manifest = json.loads((self.dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "stage3-dataset-v3":
            raise ValueError("formal shard training requires stage3-dataset-v3")
        if not bool(manifest.get("qualification_passed", False)):
            raise ValueError("dataset qualification did not pass")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported split: {split}")
        self.shards = [item for item in manifest.get("shards", ()) if item.get("split") == split]
        if self.session_ids is not None:
            self.shards = [
                item for item in self.shards
                if self.session_ids.intersection(str(value) for value in item.get("session_ids", ()))
            ]
        if not self.shards:
            raise ValueError(f"no {split} shards match the requested sessions")
        normalization_path = self.dataset_dir / str(manifest["normalization"])
        expected_normalization = manifest.get("artifact_sha256", {}).get("normalization")
        if not expected_normalization or _file_sha256(normalization_path) != expected_normalization:
            raise ValueError("normalization artifact hash mismatch")
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        self.obs_mean = np.asarray(normalization["obs_xyz"]["mean"], dtype=np.float32)
        self.obs_std = np.asarray(normalization["obs_xyz"]["std"], dtype=np.float32)
        if self.obs_mean.shape != (3,) or self.obs_std.shape != (3,):
            raise ValueError("normalization mean/std must each have shape (3,)")
        if not np.all(np.isfinite(self.obs_mean)) or not np.all(np.isfinite(self.obs_std)):
            raise ValueError("normalization mean/std must be finite")
        if np.any(self.obs_std <= 0):
            raise ValueError("normalization standard deviation must be positive")
        declared = sum(int(item["sample_count"]) for item in self.shards)
        self._length = min(declared, sample_limit) if sample_limit > 0 and self.session_ids is None else declared
        selected = 0
        found_sessions: set[str] = set()
        for item in self.shards:
            shard_path = self.dataset_dir / item["path"]
            if _file_sha256(shard_path) != str(item.get("sha256", "")):
                raise ValueError(f"shard hash mismatch: {shard_path}")
            if self.session_ids is not None:
                with np.load(shard_path, allow_pickle=False) as arrays:
                    _validate_shard_arrays(
                        {key: arrays[key] for key in arrays.files}, int(item["sample_count"])
                    )
                    present = {str(value) for value in arrays["session_id"]}
                    found_sessions.update(present & self.session_ids)
                    selected += sum(str(value) in self.session_ids for value in arrays["session_id"])
        if self.session_ids is not None:
            if found_sessions != set(self.session_ids):
                raise ValueError(
                    f"requested {split} sessions do not exactly match shard contents; "
                    f"missing={sorted(set(self.session_ids) - found_sessions)}"
                )
            self._length = min(selected, sample_limit) if sample_limit > 0 else selected
        if self._length <= 0:
            raise ValueError("selected shard dataset has no samples")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._length

    def _augment_events(
        self, obs: np.ndarray, obs_mask: np.ndarray, event_mask: np.ndarray,
        event_time_s: np.ndarray, source_seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(source_seed)
        original_obs = obs.copy()
        original_obs_mask = obs_mask.copy()
        original_event_mask = event_mask.copy()
        original_event_time_s = event_time_s.copy()
        for t in range(obs_mask.shape[0]):
            for armor in range(obs_mask.shape[1]):
                if obs_mask[t, armor] and rng.random() < 0.05:
                    obs_mask[t, armor] = False
        if rng.random() < 0.30:
            valid = np.flatnonzero(event_mask)
            if len(valid) >= 2:
                length = min(int(rng.integers(2, 11)), len(valid))
                start = int(rng.integers(0, len(valid) - length + 1))
                obs_mask[valid[start:start + length]] = False
        keep = np.flatnonzero(obs_mask.any(axis=1))
        if len(keep) == 0:
            return original_obs, original_obs_mask, original_event_mask, original_event_time_s
        compact_obs = np.zeros_like(obs)
        compact_obs_mask = np.zeros_like(obs_mask)
        compact_event_mask = np.zeros_like(event_mask)
        compact_event_time_s = np.zeros_like(event_time_s)
        start = len(event_mask) - len(keep)
        compact_obs[start:] = obs[keep]
        compact_obs_mask[start:] = obs_mask[keep]
        compact_event_mask[start:] = True
        compact_event_time_s[start:] = event_time_s[keep]
        recent = compact_event_mask & (compact_event_time_s >= -0.2)
        if np.count_nonzero(recent) < 8 or compact_event_time_s[-1] < -0.05:
            return original_obs, original_obs_mask, original_event_mask, original_event_time_s
        return compact_obs, compact_obs_mask, compact_event_mask, compact_event_time_s

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rng = np.random.default_rng(self.seed + self.epoch * 100003)
        shard_order = np.arange(len(self.shards))
        if self.shuffle:
            rng.shuffle(shard_order)
        emitted = 0
        for shard_position in shard_order:
            shard = self.shards[int(shard_position)]
            shard_path = self.dataset_dir / shard["path"]
            if _file_sha256(shard_path) != str(shard.get("sha256", "")):
                raise ValueError(f"shard hash changed before open: {shard_path}")
            with np.load(shard_path, allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            _validate_shard_arrays(arrays, int(shard["sample_count"]))
            indices = np.arange(len(arrays["session_id"]))
            if self.session_ids is not None:
                indices = np.asarray([
                    index for index in indices
                    if str(arrays["session_id"][index]) in self.session_ids
                ], dtype=np.int64)
            if self.shuffle:
                rng.shuffle(indices)
            for index in indices:
                if self.sample_limit > 0 and emitted >= self.sample_limit:
                    return
                obs = arrays["obs"][index].astype(np.float32, copy=True)
                obs_mask = arrays["obs_mask"][index].astype(np.bool_, copy=True)
                event_mask = arrays["event_mask"][index].astype(np.bool_, copy=True)
                event_time_s = arrays["event_time_s"][index].astype(np.float32, copy=True)
                if self.augment:
                    source_seed = (
                        self.seed + self.epoch * 100003 +
                        int(arrays["t0_ns"][index] % 2_147_483_647)
                    )
                    obs, obs_mask, event_mask, event_time_s = self._augment_events(
                        obs, obs_mask, event_mask, event_time_s, source_seed
                    )
                obs[..., :3] = (obs[..., :3] - self.obs_mean) / self.obs_std
                future_position = arrays["future_position"][index].astype(np.float32, copy=False)
                emitted += 1
                yield {
                    "obs": torch.from_numpy(obs),
                    "obs_mask": torch.from_numpy(obs_mask),
                    "event_mask": torch.from_numpy(event_mask),
                    "event_time_s": torch.from_numpy(event_time_s),
                    "tau": torch.from_numpy(arrays["tau"][index].astype(np.float32, copy=False)),
                    "future_position": torch.from_numpy(future_position),
                    "future_normal": torch.from_numpy(arrays["future_normal"][index].astype(np.float32, copy=False)),
                    "motion_class": torch.tensor(int(arrays["motion_class"][index]), dtype=torch.long),
                }
