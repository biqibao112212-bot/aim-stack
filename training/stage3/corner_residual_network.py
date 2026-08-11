"""Joint four-corner residual model for the simulation-only pilot.

Only raw detector quadrilateral geometry enters the network.  Truth, range,
incidence, motion, identity, PnP and future fields are labels or evaluation
metadata and are deliberately absent from :func:`observable_features`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


CORNER_ORDER = ("bl", "tl", "tr", "br")
IMAGE_WIDTH = 1440.0
IMAGE_HEIGHT = 1080.0
FEATURE_DIM = 15
OUTPUT_DIM = 8


def polygon_signed_area(corners_px: np.ndarray) -> np.ndarray:
    """Return signed quadrilateral area for arrays shaped ``[...,4,2]``."""
    corners = np.asarray(corners_px, dtype=np.float64)
    if corners.shape[-2:] != (4, 2):
        raise ValueError("corners must have shape [...,4,2]")
    x = corners[..., :, 0]
    y = corners[..., :, 1]
    return 0.5 * np.sum(x * np.roll(y, -1, axis=-1) - y * np.roll(x, -1, axis=-1), axis=-1)


def observable_features(
    corners_px: np.ndarray,
    *,
    image_width: float = IMAGE_WIDTH,
    image_height: float = IMAGE_HEIGHT,
) -> np.ndarray:
    """Build the frozen 15-D feature vector using raw corners only."""
    corners = np.asarray(corners_px, dtype=np.float64)
    if corners.ndim == 2:
        corners = corners[None, ...]
        squeeze = True
    else:
        squeeze = False
    if corners.ndim != 3 or corners.shape[1:] != (4, 2):
        raise ValueError("corners must have shape [N,4,2] or [4,2]")
    if not np.isfinite(corners).all():
        raise ValueError("corners must be finite")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    center = corners.mean(axis=1)
    area = np.maximum(np.abs(polygon_signed_area(corners)), 1.0)
    scale = np.sqrt(area)
    relative = (corners - center[:, None, :]) / scale[:, None, None]
    left_center = 0.5 * (corners[:, 0] + corners[:, 1])
    right_center = 0.5 * (corners[:, 2] + corners[:, 3])
    axis = right_center - left_center
    orientation = np.arctan2(axis[:, 1], axis[:, 0])
    width = np.linalg.norm(axis, axis=1)
    left_height = np.linalg.norm(corners[:, 0] - corners[:, 1], axis=1)
    right_height = np.linalg.norm(corners[:, 3] - corners[:, 2], axis=1)
    height = 0.5 * (left_height + right_height)
    features = np.column_stack(
        (
            relative.reshape(len(corners), 8),
            (center[:, 0] - 0.5 * image_width) / (0.5 * image_width),
            (center[:, 1] - 0.5 * image_height) / (0.5 * image_height),
            np.log(np.maximum(scale, 1.0) / 32.0),
            np.log(np.maximum(width, 1.0) / np.maximum(height, 1.0)),
            width / image_width,
            np.sin(orientation),
            np.cos(orientation),
        )
    ).astype(np.float32)
    if features.shape[1] != FEATURE_DIM:
        raise AssertionError("feature contract changed")
    return features[0] if squeeze else features


@dataclass(frozen=True)
class Standardization:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, targets_px: np.ndarray) -> "Standardization":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets_px, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
            raise ValueError("features have the wrong shape")
        if y.ndim != 2 or y.shape[1] != OUTPUT_DIM or len(y) != len(x):
            raise ValueError("targets have the wrong shape")
        x_std = x.std(axis=0)
        y_std = y.std(axis=0)
        return cls(
            feature_mean=x.mean(axis=0).astype(np.float32),
            feature_std=np.maximum(x_std, 1.0e-6).astype(np.float32),
            target_mean=y.mean(axis=0).astype(np.float32),
            target_std=np.maximum(y_std, 1.0e-6).astype(np.float32),
        )

    def normalize_features(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.feature_mean) / self.feature_std).astype(np.float32)

    def normalize_targets(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.target_mean) / self.target_std).astype(np.float32)

    def denormalize_targets(self, values: np.ndarray) -> np.ndarray:
        return (values * self.target_std + self.target_mean).astype(np.float32)


class JointCornerResidualMLP(nn.Module):
    """Small joint 8-D residual model with an identity initialization."""

    family = "joint-four-corner-residual-mlp-v1"

    def __init__(self, hidden: int = 64, dropout: float = 0.05) -> None:
        super().__init__()
        if hidden < 16 or not 0.0 <= dropout < 1.0:
            raise ValueError("invalid corner network configuration")
        self.hidden = int(hidden)
        self.dropout = float(dropout)
        self.network = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, OUTPUT_DIM),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError("corner network input must have shape [B,15]")
        return self.network(features)

    @property
    def config(self) -> dict[str, object]:
        return {
            "family": self.family,
            "feature_dimension": FEATURE_DIM,
            "output_dimension": OUTPUT_DIM,
            "hidden": self.hidden,
            "dropout": self.dropout,
            "truth_input": False,
            "range_or_incidence_input": False,
            "motion_or_identity_input": False,
        }


def deterministic_seed(seed: int) -> None:
    """Apply the deterministic settings used by every fold and seed."""
    import os
    import random

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def correction_metrics(residual_px: np.ndarray) -> dict[str, float]:
    """Return scalar navigation metrics without discarding row-level values."""
    residual = np.asarray(residual_px, dtype=np.float64).reshape(-1, 4, 2)
    coordinate_rms = np.sqrt(np.mean(np.square(residual), axis=(1, 2)))
    corner_norm = np.linalg.norm(residual, axis=2)
    return {
        "samples": int(len(residual)),
        "coordinate_rms_mean_px": float(coordinate_rms.mean()),
        "coordinate_rms_p50_px": float(np.quantile(coordinate_rms, 0.50)),
        "coordinate_rms_p90_px": float(np.quantile(coordinate_rms, 0.90)),
        "coordinate_rms_p95_px": float(np.quantile(coordinate_rms, 0.95)),
        "coordinate_rms_p99_px": float(np.quantile(coordinate_rms, 0.99)),
        "corner_norm_mean_px": float(corner_norm.mean()),
        "corner_norm_p95_px": float(np.quantile(corner_norm, 0.95)),
    }


def cyclic_learning_rate(
    initial: float, minimum: float, update: int, total_updates: int
) -> float:
    """Deterministic cosine schedule without a framework dependency."""
    if min(initial, minimum) <= 0 or total_updates <= 0:
        raise ValueError("invalid learning-rate schedule")
    fraction = min(max(update / total_updates, 0.0), 1.0)
    return minimum + 0.5 * (initial - minimum) * (1.0 + math.cos(math.pi * fraction))
