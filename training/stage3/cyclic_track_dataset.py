"""Virtual clean-physics observations for four temporary cyclic armor tracks.

The source dataset contains all four truth trajectories.  This loader exposes
only the nearest one or two plates at every causal history event.  The four
indices are tracker-owned temporary state handles: no slot feature, radius,
height, center, phase, or fixed geometry is exposed to the predictor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset


SCHEMA_VERSION = "stage3-cyclic-track-physical-virtual-v1"
VISIBILITY_POLICY = "nearest-horizontal-range-primary-gap-ratio-secondary-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def construct_cyclic_visibility(
    position_m: np.ndarray,
    event_mask: np.ndarray,
    *,
    secondary_gap_ratio: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct causal 1--2 plate visibility and adjacent switch steps.

    Visibility depends only on the four physical positions at the current
    event.  The closest horizontal-range plate is primary.  The second closest
    (which must be a cyclic neighbor) is exposed near a range-order boundary.
    This creates observation availability, not predictor features or labels.
    """
    if position_m.ndim != 3 or position_m.shape[1:] != (4, 3):
        raise ValueError("position_m must have shape [T,4,3]")
    if event_mask.shape != position_m.shape[:1]:
        raise ValueError("event_mask must have shape [T]")
    if not 0.0 <= secondary_gap_ratio <= 1.0:
        raise ValueError("secondary_gap_ratio must be within [0,1]")
    finite = np.isfinite(position_m).all(axis=(1, 2))
    valid_event = event_mask.astype(np.bool_, copy=False) & finite
    horizontal_range = np.linalg.norm(position_m[..., :2], axis=-1)
    order = np.argsort(horizontal_range, axis=-1, kind="stable")
    primary = order[:, 0].astype(np.int64, copy=False)
    secondary = order[:, 1].astype(np.int64, copy=False)
    separation = (secondary - primary) % 4
    invalid_neighbor = valid_event & ~np.isin(separation, (1, 3))
    if np.any(invalid_neighbor):
        bad = int(np.flatnonzero(invalid_neighbor)[0])
        raise ValueError(f"second visible candidate is not adjacent at event {bad}")

    ordered_range = np.take_along_axis(horizontal_range, order, axis=-1)
    range_span = ordered_range[:, 3] - ordered_range[:, 0]
    gap_ratio = (ordered_range[:, 1] - ordered_range[:, 0]) / np.maximum(
        range_span, 1e-6
    )
    expose_secondary = valid_event & (gap_ratio <= secondary_gap_ratio)
    visible = np.zeros(position_m.shape[:2], dtype=np.bool_)
    valid_indices = np.flatnonzero(valid_event)
    visible[valid_indices, primary[valid_indices]] = True
    dual_indices = np.flatnonzero(expose_secondary)
    visible[dual_indices, secondary[dual_indices]] = True

    primary_mask = np.zeros_like(visible)
    primary_mask[valid_indices, primary[valid_indices]] = True
    switch_step = np.zeros(position_m.shape[0], dtype=np.int8)
    previous: int | None = None
    for event in valid_indices:
        current = int(primary[event])
        if previous is not None:
            delta = (current - previous) % 4
            if delta == 1:
                switch_step[event] = 1
            elif delta == 3:
                switch_step[event] = -1
            elif delta != 0:
                raise ValueError(
                    f"primary track jumped by a non-adjacent step at event {event}"
                )
        previous = current
    counts = visible.sum(axis=1)
    if np.any(valid_event & ~np.isin(counts, (1, 2))):
        raise RuntimeError("valid events must expose one or two tracks")
    if np.any(~valid_event & (counts != 0)):
        raise RuntimeError("padded events cannot expose tracks")
    return visible, primary_mask, switch_step


def cyclic_relabel(
    array: np.ndarray, *, shift: int, reverse: bool, axis: int,
) -> np.ndarray:
    """Apply one temporary cyclic-origin change and optional direction flip."""
    result = np.roll(array, int(shift) % 4, axis=axis)
    if reverse:
        result = np.take(result, np.asarray((0, 3, 2, 1)), axis=axis)
    return np.ascontiguousarray(result)


class CyclicTrackPhysicalDataset(IterableDataset):
    """Read qualified train/validation truth and expose virtual observations."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        *,
        seed: int,
        shuffle: bool = False,
        sample_limit: int = 0,
        secondary_gap_ratio: float = 0.25,
        augment_cyclic_origin: bool = False,
        augment_direction: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir).resolve()
        if split not in {"train", "validation"}:
            raise ValueError("cyclic-track loader only permits train or validation")
        if split != "train" and (augment_cyclic_origin or augment_direction):
            raise ValueError("validation cyclic labels must remain deterministic")
        self.split = split
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.sample_limit = int(sample_limit)
        self.secondary_gap_ratio = float(secondary_gap_ratio)
        self.augment_cyclic_origin = bool(augment_cyclic_origin)
        self.augment_direction = bool(augment_direction)
        self.epoch = 0

        manifest_path = self.dataset_dir / "dataset_manifest.json"
        self.manifest_path = manifest_path
        self.manifest_sha256 = _sha256(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != "stage3-causal-physical-v1":
            raise ValueError("cyclic-track source schema mismatch")
        if not bool(self.manifest.get("qualification_passed", False)):
            raise ValueError("cyclic-track source dataset is not qualified")
        if bool(self.manifest.get("test_accessed", True)):
            raise ValueError("cyclic-track source must record test_accessed=false")
        self.shards = [
            item for item in self.manifest["shards"] if item["split"] == split
        ]
        if not self.shards:
            raise ValueError(f"cyclic-track source has no {split} shards")
        normalization_path = self.dataset_dir / str(self.manifest["normalization"])
        if _sha256(normalization_path) != self.manifest["normalization_sha256"]:
            raise ValueError("cyclic-track normalization hash mismatch")
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        self.mean = np.asarray(normalization["position_m"]["mean"], dtype=np.float32)
        self.std = np.asarray(normalization["position_m"]["std"], dtype=np.float32)
        if self.mean.shape != (3,) or self.std.shape != (3,) or np.any(self.std <= 0):
            raise ValueError("invalid cyclic-track normalization")
        for item in self.shards:
            path = self.dataset_dir / str(item["path"])
            if _sha256(path) != str(item["sha256"]):
                raise ValueError(f"cyclic-track source shard hash mismatch: {path}")
        declared = sum(int(item["sample_count"]) for item in self.shards)
        self._length = min(declared, sample_limit) if sample_limit > 0 else declared

    @property
    def virtual_contract(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_manifest_sha256": self.manifest_sha256,
            "visibility_policy": VISIBILITY_POLICY,
            "secondary_gap_ratio": self.secondary_gap_ratio,
            "maximum_visible_tracks_per_event": 2,
            "minimum_visible_tracks_per_valid_event": 1,
            "switch_policy": "primary changes only by 0,+1,-1 modulo four",
            "track_identity": "temporary cyclic state handles with no feature embedding",
            "predictor_geometry_inputs": [],
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._length

    @staticmethod
    def _validate_arrays(arrays: dict[str, np.ndarray]) -> None:
        required = {
            "history_position_m", "history_obs_mask", "event_mask",
            "event_time_s", "tau", "future_position", "motion_class",
            "rule_query", "distance_m", "session_id", "t0_ns",
        }
        missing = required - arrays.keys()
        if missing:
            raise ValueError(f"cyclic-track shard is missing: {sorted(missing)}")
        if arrays["history_position_m"].shape[1:] != (200, 4, 3):
            raise ValueError("cyclic-track source history must be [N,200,4,3]")
        if arrays["history_obs_mask"].shape[1:] != (200, 4):
            raise ValueError("cyclic-track source mask must be [N,200,4]")
        if arrays["future_position"].shape[1:] != (8, 4, 3):
            raise ValueError("cyclic-track source future must be [N,8,4,3]")
        full = arrays["history_obs_mask"]
        event = arrays["event_mask"]
        if not np.array_equal(full, np.broadcast_to(event[..., None], full.shape)):
            raise ValueError("cyclic-track virtual source requires complete four-track truth")

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        # Keep bounded-smoke membership stable across epochs.  Epoch-dependent
        # randomness is reserved for relabel augmentation, not sample choice.
        order_rng = np.random.default_rng(self.seed)
        augment_rng = np.random.default_rng(self.seed + self.epoch * 100003)
        shard_order = np.arange(len(self.shards))
        if self.shuffle:
            order_rng.shuffle(shard_order)
        emitted = 0
        for shard_index in shard_order:
            item = self.shards[int(shard_index)]
            path = self.dataset_dir / str(item["path"])
            if _sha256(path) != str(item["sha256"]):
                raise ValueError(f"cyclic-track source shard changed before open: {path}")
            with np.load(path, allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            self._validate_arrays(arrays)
            indices = np.arange(int(item["sample_count"]))
            if self.shuffle:
                order_rng.shuffle(indices)
            for index in indices:
                if self.sample_limit > 0 and emitted >= self.sample_limit:
                    return
                full_position = arrays["history_position_m"][index].astype(
                    np.float32, copy=True
                )
                event_mask = arrays["event_mask"][index].astype(np.bool_, copy=False)
                visible, primary, switch_step = construct_cyclic_visibility(
                    full_position, event_mask,
                    secondary_gap_ratio=self.secondary_gap_ratio,
                )
                shift = (
                    int(augment_rng.integers(0, 4))
                    if self.augment_cyclic_origin else 0
                )
                reverse = (
                    bool(augment_rng.integers(0, 2))
                    if self.augment_direction else False
                )
                full_position = cyclic_relabel(
                    full_position, shift=shift, reverse=reverse, axis=1
                )
                visible = cyclic_relabel(
                    visible, shift=shift, reverse=reverse, axis=1
                )
                primary = cyclic_relabel(
                    primary, shift=shift, reverse=reverse, axis=1
                )
                future = cyclic_relabel(
                    arrays["future_position"][index].astype(np.float32, copy=True),
                    shift=shift, reverse=reverse, axis=1,
                )
                if reverse:
                    switch_step = -switch_step
                normalized = (full_position - self.mean) / self.std
                obs = np.where(
                    visible[..., None], normalized, 0.0
                ).astype(np.float32, copy=False)
                physical = np.where(
                    visible[..., None], full_position, 0.0
                ).astype(np.float32, copy=False)
                valid_indices = np.flatnonzero(event_mask)
                current_event = int(valid_indices[-1])
                current_primary = int(np.flatnonzero(primary[current_event])[0])
                current_visible = visible[current_event]
                adjacent = np.zeros(4, dtype=np.bool_)
                adjacent[(current_primary - 1) % 4] = True
                adjacent[(current_primary + 1) % 4] = True
                adjacent_hidden = adjacent & ~current_visible
                emitted += 1
                yield {
                    "obs": torch.from_numpy(obs),
                    "history_position_m": torch.from_numpy(physical),
                    "obs_mask": torch.from_numpy(visible),
                    "primary_mask": torch.from_numpy(primary),
                    "event_mask": torch.from_numpy(event_mask),
                    "event_time_s": torch.from_numpy(
                        arrays["event_time_s"][index].astype(np.float32, copy=False)
                    ),
                    "switch_step": torch.from_numpy(switch_step.astype(np.float32)),
                    "tau": torch.from_numpy(
                        arrays["tau"][index].astype(np.float32, copy=False)
                    ),
                    "future_position": torch.from_numpy(future),
                    "motion_class": torch.tensor(
                        int(arrays["motion_class"][index]), dtype=torch.long
                    ),
                    "rule_query": torch.from_numpy(
                        arrays["rule_query"][index].astype(np.bool_, copy=False)
                    ),
                    "distance_m": torch.tensor(
                        float(arrays["distance_m"][index]), dtype=torch.float32
                    ),
                    "current_primary_index": torch.tensor(
                        current_primary, dtype=torch.long
                    ),
                    "current_visible_mask": torch.from_numpy(current_visible.copy()),
                    "adjacent_hidden_mask": torch.from_numpy(adjacent_hidden),
                    "current_visible_count": torch.tensor(
                        int(current_visible.sum()), dtype=torch.long
                    ),
                }
