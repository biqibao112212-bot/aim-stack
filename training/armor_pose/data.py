"""Read-only adapter from verified corner-PnP packs to separated online/label batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from training.corner_pnp.data import load_pack


ONLINE_FIELDS = (
    "patch", "features", "raw", "raw_patch", "inverse_transform", "scale", "intrinsics",
)
SUPERVISION_FIELDS = ("target", "target_patch", "rotation", "translation")
FORBIDDEN_ONLINE_TOKENS = (
    "target", "exact", "truth", "translation", "rotation", "range", "motion",
    "session", "identity", "history", "future", "tracker",
)


@dataclass(frozen=True)
class LoadedArmorPosePack:
    values: dict[str, np.ndarray]
    manifest: dict[str, object]
    feature_mean: np.ndarray
    feature_std: np.ndarray


def load_development_pack(path: Path, *, expected_split: str,
                          feature_mean: np.ndarray | None = None,
                          feature_std: np.ndarray | None = None) -> LoadedArmorPosePack:
    if expected_split not in {"train", "validation"}:
        raise PermissionError("V19 development adapter refuses test packs")
    values, manifest = load_pack(path, expected_split=expected_split)
    required = set(ONLINE_FIELDS) | set(SUPERVISION_FIELDS) | {"session_id", "mode"}
    missing = required - set(values)
    if missing:
        raise ValueError(f"legacy pack lacks required V19 fields: {sorted(missing)}")
    if feature_mean is None:
        feature_mean = values["features"].mean(axis=0).astype(np.float32)
    if feature_std is None:
        feature_std = np.maximum(values["features"].std(axis=0), 1.0e-6).astype(np.float32)
    return LoadedArmorPosePack(values, manifest, np.asarray(feature_mean, dtype=np.float32),
                               np.asarray(feature_std, dtype=np.float32))


class ArmorPoseDataset(Dataset[dict[str, object]]):
    """Keep online tensors structurally separate from offline supervision."""

    def __init__(self, pack: LoadedArmorPosePack, *, maximum_samples: int | None = None) -> None:
        self.pack = pack
        count = len(pack.values["raw"])
        self.count = count if maximum_samples is None else min(count, int(maximum_samples))
        if self.count <= 0:
            raise ValueError("dataset must contain at least one sample")

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, object]:
        value = self.pack.values
        normalized_geometry = (value["features"][index] - self.pack.feature_mean) / self.pack.feature_std
        online = {
            "patch_rgb": torch.from_numpy(value["patch"][index].astype(np.float32)),
            "detector_geometry": torch.from_numpy(normalized_geometry.astype(np.float32)),
            "raw_corners_px": torch.from_numpy(value["raw"][index].astype(np.float32)),
            "raw_corners_patch_px": torch.from_numpy(value["raw_patch"][index].astype(np.float32)),
            "patch_to_image_h": torch.from_numpy(value["inverse_transform"][index].astype(np.float32)),
            "raw_scale_px": torch.as_tensor(value["scale"][index], dtype=torch.float32),
            "intrinsics": torch.from_numpy(value["intrinsics"][index].astype(np.float32)),
        }
        if any(any(token in name for token in FORBIDDEN_ONLINE_TOKENS) for name in online):
            raise AssertionError("truth-like field entered the online tensor namespace")
        supervision = {
            "target_corners_px": torch.from_numpy(value["target"][index].astype(np.float32)),
            "target_corners_patch_px": torch.from_numpy(value["target_patch"][index].astype(np.float32)),
            "reference_rotation": torch.from_numpy(value["rotation"][index].astype(np.float32)),
            "reference_translation_m": torch.from_numpy(value["translation"][index].astype(np.float32)),
        }
        return {
            "online": online,
            "supervision": supervision,
            "metadata": {"session_id": str(value["session_id"][index]), "mode": str(value["mode"][index])},
        }


def move_tensor_tree(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_tensor_tree(item, device) for key, item in value.items()}
    return value
