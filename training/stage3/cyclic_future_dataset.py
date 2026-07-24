"""Expert-filtered view of the qualified cyclic clean-physics dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .cyclic_track_dataset import CyclicTrackPhysicalDataset
from .cyclic_future_model import DYNAMIC_EXPERTS


EXPERT_CLASS = {"translation": 1, "rotation": 2, "combined": 3}


class CyclicFutureExpertDataset(IterableDataset):
    """Expose exactly one dynamic class without weakening source qualification."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        *,
        expert: str,
        seed: int,
        shuffle: bool = False,
        sample_limit: int = 0,
        secondary_gap_ratio: float = 0.25,
        augment_cyclic_origin: bool = False,
        augment_direction: bool = False,
    ) -> None:
        super().__init__()
        if expert not in DYNAMIC_EXPERTS:
            raise ValueError(f"unsupported dynamic expert: {expert}")
        self.expert = expert
        self.class_id = EXPERT_CLASS[expert]
        self.sample_limit = int(sample_limit)
        self.source = CyclicTrackPhysicalDataset(
            dataset_dir,
            split,
            seed=seed,
            shuffle=shuffle,
            sample_limit=0,
            secondary_gap_ratio=secondary_gap_ratio,
            augment_cyclic_origin=augment_cyclic_origin,
            augment_direction=augment_direction,
        )
        count = 0
        for item in self.source.shards:
            path = self.source.dataset_dir / str(item["path"])
            with np.load(path, allow_pickle=False) as loaded:
                count += int(np.count_nonzero(loaded["motion_class"] == self.class_id))
        if count == 0:
            raise ValueError(f"dataset has no {split} samples for {expert}")
        self._length = min(count, self.sample_limit) if self.sample_limit > 0 else count

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        self.source.set_epoch(epoch)

    @property
    def manifest(self) -> dict[str, object]:
        return self.source.manifest

    @property
    def manifest_path(self) -> Path:
        return self.source.manifest_path

    @property
    def manifest_sha256(self) -> str:
        return self.source.manifest_sha256

    @property
    def mean(self) -> np.ndarray:
        return self.source.mean

    @property
    def std(self) -> np.ndarray:
        return self.source.std

    @property
    def virtual_contract(self) -> dict[str, object]:
        return {
            **self.source.virtual_contract,
            "expert": self.expert,
            "motion_class_filter": self.class_id,
            "filter_role": "outer training/evaluation only; not a predictor input",
        }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        emitted = 0
        for sample in self.source:
            if int(sample["motion_class"].item()) != self.class_id:
                continue
            if self.sample_limit > 0 and emitted >= self.sample_limit:
                return
            emitted += 1
            yield sample
