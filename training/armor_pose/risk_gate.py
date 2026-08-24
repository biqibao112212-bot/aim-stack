"""CUDA-only observable risk gate for sparse probabilistic pose candidates.

Reference pose is used only by :func:`offline_benefit_labels`.  Online feature
extraction, gate inference, and raw fallback consume same-frame sparse output
and truth-free GPU-PnP diagnostics only.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from training.corner_pnp.data import sha256, write_json_new
from training.corner_pnp.pnp import NOMINAL_OBJECT_POINTS_M

from .backbone import ArmorPoseBackbone
from .data import LoadedArmorPosePack, load_development_pack
from .gpu_pnp import GpuPnPResult, solve_weighted_planar_pnp
from .labels import calibrated_grid_moments
from .sparse_prob_head import SparsePrediction


SCHEMA = "aim-stack.armor-pose-risk-gate-checkpoint/1"
FEATURE_NAMES = (
    "correction_rms_norm", "correction_max_norm",
    "heatmap_entropy_mean_norm", "heatmap_entropy_max_norm",
    "in_context_probability_mean", "in_context_probability_min",
    "corner_cov_trace_mean_norm_log1p", "corner_cov_trace_max_norm_log1p",
    "corner_cov_eigen_ratio_mean_log1p", "tail_cov_trace_mean_norm_log1p",
    "raw_mode0_valid", "raw_mode1_valid", "raw_reprojection_log1p",
    "raw_objective_per_point_log1p", "raw_condition_log1p",
    "raw_pose_covariance_trace_log1p", "raw_mode_objective_gap_log1p",
    "candidate_mode0_valid", "candidate_mode1_valid", "candidate_reprojection_log1p",
    "candidate_objective_per_point_log1p", "candidate_condition_log1p",
    "candidate_pose_covariance_trace_log1p", "candidate_mode_objective_gap_log1p",
    "pose_delta_x_over_raw_z", "pose_delta_y_over_raw_z", "pose_delta_z_over_raw_z",
    "pose_delta_norm_over_raw_z", "pose_rotation_delta_rad",
    "candidate_to_raw_reprojection_log_ratio", "candidate_to_raw_objective_log_ratio",
    "candidate_to_raw_pose_covariance_log_ratio",
)
FEATURE_DIMENSION = len(FEATURE_NAMES)
FORBIDDEN_PATH_TOKENS = ("test-v15", "test-v18", "sealed")


@dataclass(frozen=True)
class RiskLabelPolicy:
    minimum_position_gain_mm: float = 0.0
    minimum_depth_gain_mm: float = 0.0
    maximum_ray_regression_mrad: float = 0.0

    def validate(self) -> None:
        values = (
            self.minimum_position_gain_mm,
            self.minimum_depth_gain_mm,
            self.maximum_ray_regression_mrad,
        )
        if not all(np.isfinite(values)) or any(value < 0.0 for value in values):
            raise ValueError("risk-label margins must be finite and nonnegative")

    @property
    def config(self) -> dict[str, float]:
        return {
            "minimum_position_gain_mm": self.minimum_position_gain_mm,
            "minimum_depth_gain_mm": self.minimum_depth_gain_mm,
            "maximum_ray_regression_mrad": self.maximum_ray_regression_mrad,
        }


@dataclass(frozen=True)
class OfflineRiskLabels:
    beneficial: torch.Tensor
    eligible: torch.Tensor
    raw_position_mm: torch.Tensor
    candidate_position_mm: torch.Tensor
    raw_depth_mm: torch.Tensor
    candidate_depth_mm: torch.Tensor
    raw_ray_mrad: torch.Tensor
    candidate_ray_mrad: torch.Tensor


@dataclass(frozen=True)
class RiskGateDecision:
    benefit_probability: torch.Tensor
    use_candidate: torch.Tensor
    fallback_to_raw: torch.Tensor
    translation_m: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class LoadedSparse:
    model: nn.Module
    checkpoint: Path
    checkpoint_hash: str
    source_plan_hash: str
    model_config: dict[str, Any]
    feature_mean: np.ndarray
    feature_std: np.ndarray


@dataclass(frozen=True)
class LoadedRiskGate:
    model: "ObservablePoseRiskGate"
    checkpoint: Path
    checkpoint_hash: str
    sparse_checkpoint_hash: str
    source_sparse_plan_hash: str
    trust_scale: float
    threshold: float
    label_policy: RiskLabelPolicy


class ObservablePoseRiskGate(nn.Module):
    """Small MLP over an explicitly frozen same-frame diagnostic vector."""

    family = "same-frame-observable-sparse-pose-risk-gate-v1"

    def __init__(self, *, hidden: int = 64, dropout: float = 0.05) -> None:
        super().__init__()
        if hidden < 16 or not 0.0 <= dropout < 1.0:
            raise ValueError("invalid risk-gate configuration")
        self.hidden, self.dropout = int(hidden), float(dropout)
        self.network = nn.Sequential(
            nn.Linear(FEATURE_DIMENSION, self.hidden), nn.SiLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden // 2), nn.SiLU(),
            nn.Linear(self.hidden // 2, 1),
        )
        self.register_buffer("feature_mean", torch.zeros(FEATURE_DIMENSION))
        self.register_buffer("feature_std", torch.ones(FEATURE_DIMENSION))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (FEATURE_DIMENSION,) or std.shape != mean.shape:
            raise ValueError("risk-gate normalization shape changed")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0.0):
            raise ValueError("risk-gate normalization must be finite and positive")
        with torch.no_grad():
            self.feature_mean.copy_(mean.to(device=self.feature_mean.device, dtype=self.feature_mean.dtype))
            self.feature_std.copy_(std.to(device=self.feature_std.device, dtype=self.feature_std.dtype))

    def forward(self, observable_features: torch.Tensor) -> torch.Tensor:
        if observable_features.ndim != 2 or observable_features.shape[1] != FEATURE_DIMENSION:
            raise ValueError(f"observable_features must be [B,{FEATURE_DIMENSION}]")
        normalized = (observable_features - self.feature_mean) / self.feature_std
        return self.network(normalized).squeeze(1)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "hidden": self.hidden,
            "dropout": self.dropout,
            "feature_dimension": FEATURE_DIMENSION,
            "feature_names": list(FEATURE_NAMES),
            "online_inputs": [
                "raw and trusted sparse corners", "sparse heatmap/mixture uncertainty",
                "raw GPU-PnP diagnostics", "candidate GPU-PnP diagnostics",
                "raw-to-candidate pose delta",
            ],
            "online_truth_input": False,
            "temporal_input": False,
        }


class _FrozenV19ProbabilisticCornerNet(nn.Module):
    """Byte-compatible V19 source architecture, independent of future defaults."""

    family = "same-frame-four-heatmap-probabilistic-corners-v1"

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ArmorPoseBackbone()
        self.heatmaps = nn.Conv2d(self.backbone.output_channels, 4, 1)
        self.tail = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(self.backbone.output_channels, 64), nn.SiLU(),
            nn.Linear(64, 4 * 6),
        )

    def forward(self, patch: torch.Tensor, geometry: torch.Tensor,
                raw_full: torch.Tensor, raw_patch: torch.Tensor,
                inverse_transform: torch.Tensor,
                scale: torch.Tensor) -> SparsePrediction:
        if raw_full.shape != (patch.shape[0], 4, 2) or raw_patch.shape != raw_full.shape:
            raise ValueError("frozen V19 sparse corner contract changed")
        feature = self.backbone(patch, geometry)
        logits = self.heatmaps(feature)
        local_image_mean, local_covariance, entropy = calibrated_grid_moments(
            logits, raw_patch, raw_full, inverse_transform,
        )
        tail = self.tail(feature).reshape(-1, 4, 6)
        tail_residual = 4.0 * torch.tanh(tail[..., :2])
        tail_image_mean = raw_full + tail_residual * scale[:, None, None]
        tail_std_norm = torch.exp(tail[..., 2:4].clamp(-5.0, 0.6931471805599453))
        rho = 0.95 * torch.tanh(tail[..., 4])
        sigma_x, sigma_y = tail_std_norm.unbind(dim=-1)
        tail_covariance_norm = torch.stack(
            (sigma_x.square(), rho * sigma_x * sigma_y,
             rho * sigma_x * sigma_y, sigma_y.square()), dim=-1,
        ).reshape(-1, 4, 2, 2)
        identity = torch.eye(2, dtype=patch.dtype, device=patch.device)
        tail_covariance = (
            tail_covariance_norm * scale[:, None, None, None].square()
            + 1.0e-4 * identity
        )
        in_context_probability = torch.sigmoid(tail[..., 5])
        mixture = in_context_probability[..., None]
        image_mean = mixture * local_image_mean + (1.0 - mixture) * tail_image_mean
        local_delta = local_image_mean - image_mean
        tail_delta = tail_image_mean - image_mean
        image_covariance = (
            in_context_probability[..., None, None]
            * (local_covariance + local_delta[..., :, None] * local_delta[..., None, :])
            + (1.0 - in_context_probability)[..., None, None]
            * (tail_covariance + tail_delta[..., :, None] * tail_delta[..., None, :])
        )
        return SparsePrediction(
            heatmap_logits=logits, visibility_logits=tail[..., 5], entropy=entropy,
            image_mean=image_mean, image_covariance=image_covariance,
            local_image_mean=local_image_mean, tail_image_mean=tail_image_mean,
            in_context_probability=in_context_probability,
            tail_covariance=tail_covariance,
        )


def _finite(value: torch.Tensor, *, minimum: float = -20.0,
            maximum: float = 20.0) -> torch.Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=maximum, neginf=minimum).clamp(minimum, maximum)


def _pnp_diagnostics(result: GpuPnPResult, *, correspondence_count: int) -> list[torch.Tensor]:
    valid0, valid1 = result.valid[:, 0], result.valid[:, 1]
    reprojection = torch.log1p(result.reprojection_rms_px[:, 0].clamp_min(0.0))
    objective = torch.log1p((result.objective[:, 0] / float(correspondence_count)).clamp_min(0.0))
    condition = torch.log1p(result.condition[:, 0].clamp_min(0.0))
    pose_covariance = 0.5 * (
        result.covariance_local[:, 0] + result.covariance_local[:, 0].transpose(-1, -2)
    )
    covariance_trace = torch.log1p(
        torch.diagonal(pose_covariance, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(0.0)
    )
    both = valid0 & valid1
    mode_gap = torch.where(
        both,
        ((result.objective[:, 1] - result.objective[:, 0]).clamp_min(0.0)
         / float(correspondence_count)),
        torch.zeros_like(result.objective[:, 0]),
    )
    return [
        valid0.to(result.objective.dtype), valid1.to(result.objective.dtype),
        _finite(reprojection), _finite(objective), _finite(condition),
        _finite(covariance_trace), _finite(torch.log1p(mode_gap)),
    ]


def observable_pose_risk_features(raw_corners_px: torch.Tensor,
                                  candidate_corners_px: torch.Tensor,
                                  raw_scale_px: torch.Tensor,
                                  sparse_prediction: SparsePrediction,
                                  raw_pnp: GpuPnPResult,
                                  candidate_pnp: GpuPnPResult) -> torch.Tensor:
    """Build the online gate vector without labels, metadata, or history."""
    batch = raw_corners_px.shape[0]
    if (raw_corners_px.shape != (batch, 4, 2)
            or candidate_corners_px.shape != raw_corners_px.shape
            or raw_scale_px.shape != (batch,)):
        raise ValueError("risk-gate corner input contract changed")
    scale = raw_scale_px.clamp_min(1.0)
    correction = (candidate_corners_px - raw_corners_px) / scale[:, None, None]
    correction_norm = torch.linalg.vector_norm(correction, dim=-1)
    entropy_scale = float(np.log(64 * 128))
    entropy = sparse_prediction.entropy / entropy_scale
    covariance = 0.5 * (
        sparse_prediction.image_covariance + sparse_prediction.image_covariance.transpose(-1, -2)
    )
    covariance_norm = covariance / scale[:, None, None, None].square()
    covariance_eigen = torch.linalg.eigvalsh(covariance_norm).clamp_min(1.0e-8)
    covariance_trace = covariance_eigen.sum(dim=-1)
    covariance_ratio = covariance_eigen[..., 1] / covariance_eigen[..., 0]
    tail_trace = torch.diagonal(
        sparse_prediction.tail_covariance / scale[:, None, None, None].square(),
        dim1=-2, dim2=-1,
    ).sum(dim=-1)
    features: list[torch.Tensor] = [
        torch.sqrt(correction.square().mean(dim=(1, 2))), correction_norm.amax(dim=1),
        entropy.mean(dim=1), entropy.amax(dim=1),
        sparse_prediction.in_context_probability.mean(dim=1),
        sparse_prediction.in_context_probability.amin(dim=1),
        torch.log1p(covariance_trace.mean(dim=1)), torch.log1p(covariance_trace.amax(dim=1)),
        torch.log1p(covariance_ratio.mean(dim=1)), torch.log1p(tail_trace.mean(dim=1)),
    ]
    raw_diagnostics = _pnp_diagnostics(raw_pnp, correspondence_count=4)
    candidate_diagnostics = _pnp_diagnostics(candidate_pnp, correspondence_count=4)
    features.extend(raw_diagnostics)
    features.extend(candidate_diagnostics)

    raw_translation = raw_pnp.translation_m[:, 0]
    candidate_translation = candidate_pnp.translation_m[:, 0]
    translation_delta = candidate_translation - raw_translation
    raw_z = raw_translation[:, 2].abs().clamp_min(0.10)
    normalized_delta = translation_delta / raw_z[:, None]
    relative_rotation = (
        candidate_pnp.rotation_camera_from_pnp[:, 0]
        @ raw_pnp.rotation_camera_from_pnp[:, 0].transpose(-1, -2)
    )
    trace = torch.diagonal(relative_rotation, dim1=-2, dim2=-1).sum(dim=-1)
    rotation_angle = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6))
    raw_reprojection = raw_pnp.reprojection_rms_px[:, 0].clamp_min(1.0e-6)
    candidate_reprojection = candidate_pnp.reprojection_rms_px[:, 0].clamp_min(1.0e-6)
    raw_objective = (raw_pnp.objective[:, 0] / 4.0).clamp_min(1.0e-6)
    candidate_objective = (candidate_pnp.objective[:, 0] / 4.0).clamp_min(1.0e-6)
    raw_covariance_trace = torch.diagonal(
        raw_pnp.covariance_local[:, 0], dim1=-2, dim2=-1,
    ).sum(dim=-1).abs().clamp_min(1.0e-8)
    candidate_covariance_trace = torch.diagonal(
        candidate_pnp.covariance_local[:, 0], dim1=-2, dim2=-1,
    ).sum(dim=-1).abs().clamp_min(1.0e-8)
    features.extend([
        normalized_delta[:, 0], normalized_delta[:, 1], normalized_delta[:, 2],
        torch.linalg.vector_norm(normalized_delta, dim=1), rotation_angle,
        torch.log(candidate_reprojection / raw_reprojection),
        torch.log(candidate_objective / raw_objective),
        torch.log(candidate_covariance_trace / raw_covariance_trace),
    ])
    result = torch.stack([_finite(value) for value in features], dim=1)
    if result.shape != (batch, FEATURE_DIMENSION) or not torch.isfinite(result).all():
        raise FloatingPointError("observable risk-gate feature construction failed")
    return result


def trusted_sparse_corners(raw_corners_px: torch.Tensor, sparse_prediction: SparsePrediction,
                           *, trust_scale: float) -> torch.Tensor:
    if not np.isfinite(trust_scale) or not 0.0 <= trust_scale <= 1.0:
        raise ValueError("trust_scale must be in [0,1]")
    return raw_corners_px + float(trust_scale) * (sparse_prediction.image_mean - raw_corners_px)


def offline_benefit_labels(raw_pnp: GpuPnPResult, candidate_pnp: GpuPnPResult,
                           reference_translation_m: torch.Tensor,
                           policy: RiskLabelPolicy) -> OfflineRiskLabels:
    """Construct offline supervision after online candidate inference is frozen."""
    policy.validate()
    if reference_translation_m.shape != (raw_pnp.translation_m.shape[0], 3):
        raise ValueError("reference translation shape changed")
    raw_translation = raw_pnp.translation_m[:, 0]
    candidate_translation = candidate_pnp.translation_m[:, 0]
    raw_delta = raw_translation - reference_translation_m
    candidate_delta = candidate_translation - reference_translation_m
    reference_z = reference_translation_m[:, 2].clamp_min(1.0e-4)
    raw_position = 1.0e3 * torch.linalg.vector_norm(raw_delta, dim=1)
    candidate_position = 1.0e3 * torch.linalg.vector_norm(candidate_delta, dim=1)
    raw_depth = 1.0e3 * raw_delta[:, 2].abs()
    candidate_depth = 1.0e3 * candidate_delta[:, 2].abs()
    raw_ray = 1.0e3 * torch.linalg.vector_norm(raw_delta[:, :2] / reference_z[:, None], dim=1)
    candidate_ray = 1.0e3 * torch.linalg.vector_norm(candidate_delta[:, :2] / reference_z[:, None], dim=1)
    eligible = raw_pnp.valid[:, 0] & torch.isfinite(reference_translation_m).all(dim=1)
    beneficial = (
        eligible & candidate_pnp.valid[:, 0]
        & ((raw_position - candidate_position) > policy.minimum_position_gain_mm)
        & ((raw_depth - candidate_depth) > policy.minimum_depth_gain_mm)
        & ((candidate_ray - raw_ray) <= policy.maximum_ray_regression_mrad)
    )
    return OfflineRiskLabels(
        beneficial, eligible, raw_position, candidate_position, raw_depth,
        candidate_depth, raw_ray, candidate_ray,
    )


def apply_observable_pose_risk_gate(gate: ObservablePoseRiskGate,
                                    raw_corners_px: torch.Tensor,
                                    candidate_corners_px: torch.Tensor,
                                    raw_scale_px: torch.Tensor,
                                    sparse_prediction: SparsePrediction,
                                    raw_pnp: GpuPnPResult,
                                    candidate_pnp: GpuPnPResult,
                                    *, threshold: float) -> RiskGateDecision:
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("risk-gate threshold must be in [0,1]")
    feature = observable_pose_risk_features(
        raw_corners_px, candidate_corners_px, raw_scale_px,
        sparse_prediction, raw_pnp, candidate_pnp,
    )
    probability = torch.sigmoid(gate(feature))
    use_candidate = candidate_pnp.valid[:, 0] & (probability >= threshold)
    fallback = ~use_candidate
    translation = torch.where(
        use_candidate[:, None], candidate_pnp.translation_m[:, 0], raw_pnp.translation_m[:, 0],
    )
    valid = torch.where(use_candidate, candidate_pnp.valid[:, 0], raw_pnp.valid[:, 0])
    return RiskGateDecision(probability, use_candidate, fallback, translation, valid)


def _reject_protected_pack(path: Path, *, expected_split: str) -> Path:
    lowered = str(path).lower()
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS) or path.name.lower().startswith("test"):
        raise PermissionError("risk-gate training refuses sealed/test packs")
    resolved = path.resolve(strict=True)
    lowered = str(resolved).lower()
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS) or resolved.name.lower().startswith("test"):
        raise PermissionError("resolved risk-gate pack path points at sealed/test evidence")
    if expected_split not in {"train", "validation"}:
        raise PermissionError("risk-gate development supports train/validation only")
    return resolved


def _sparse_model_from_config(model_config: dict[str, Any]) -> nn.Module:
    family = model_config.get("family")
    factories: dict[str, type[nn.Module]] = {
        "same-frame-four-heatmap-probabilistic-corners-v1": _FrozenV19ProbabilisticCornerNet,
    }
    model_type = factories.get(str(family))
    if model_type is None:
        raise ValueError(f"unsupported source sparse architecture: {family}")
    return model_type()


def _load_sparse(path: Path, *, device: torch.device) -> LoadedSparse:
    resolved = path.resolve(strict=True)
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    if (checkpoint.get("schema_version") != "aim-stack.armor-pose-checkpoint/1"
            or checkpoint.get("branch") != "sparse"):
        raise ValueError("risk gate requires a sparse armor-pose checkpoint")
    if checkpoint.get("online_truth_input") is not False or checkpoint.get("test_accessed") is not False:
        raise PermissionError("sparse checkpoint provenance is not truth/test clean")
    source_plan_hash = str(checkpoint.get("plan_sha256", ""))
    if len(source_plan_hash) != 64 or any(character not in "0123456789abcdef" for character in source_plan_hash):
        raise ValueError("source sparse checkpoint has no valid source-plan hash")
    model_config = checkpoint.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("source sparse checkpoint has no architecture config")
    feature_mean = np.asarray(checkpoint.get("feature_mean"), dtype=np.float32)
    feature_std = np.asarray(checkpoint.get("feature_std"), dtype=np.float32)
    if (feature_mean.shape != (15,) or feature_std.shape != (15,)
            or not np.isfinite(feature_mean).all() or not np.isfinite(feature_std).all()
            or np.any(feature_std <= 0.0)):
        raise ValueError("sparse checkpoint feature normalization is invalid")
    model = _sparse_model_from_config(model_config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return LoadedSparse(
        model, resolved, sha256(resolved), source_plan_hash, model_config,
        feature_mean, feature_std,
    )


@torch.inference_mode()
def _gate_examples(pack: LoadedArmorPosePack, sparse: LoadedSparse,
                   *, trust_scale: float, label_policy: RiskLabelPolicy,
                   batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    values = pack.values
    feature_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    eligible_chunks: list[torch.Tensor] = []
    raw_valid_count = candidate_valid_count = 0
    for start in range(0, len(values["raw"]), batch_size):
        end = min(len(values["raw"]), start + batch_size)
        patch = torch.from_numpy(values["patch"][start:end].astype(np.float32)).to(device)
        geometry = torch.from_numpy(
            ((values["features"][start:end] - sparse.feature_mean) / sparse.feature_std).astype(np.float32)
        ).to(device)
        raw = torch.from_numpy(values["raw"][start:end].astype(np.float32)).to(device)
        raw_patch = torch.from_numpy(values["raw_patch"][start:end].astype(np.float32)).to(device)
        inverse = torch.from_numpy(values["inverse_transform"][start:end].astype(np.float32)).to(device)
        scale = torch.from_numpy(values["scale"][start:end].astype(np.float32)).to(device)
        intrinsics = torch.from_numpy(values["intrinsics"][start:end].astype(np.float32)).to(device)
        object_points = torch.as_tensor(
            NOMINAL_OBJECT_POINTS_M, dtype=raw.dtype, device=device,
        )[None].expand(end - start, -1, -1)
        prediction = sparse.model(patch, geometry, raw, raw_patch, inverse, scale)
        candidate = trusted_sparse_corners(raw, prediction, trust_scale=trust_scale)
        raw_pnp = solve_weighted_planar_pnp(raw, object_points, intrinsics)
        candidate_pnp = solve_weighted_planar_pnp(
            candidate, object_points, intrinsics, covariance=prediction.image_covariance,
        )
        # The online feature vector is finalized before any reference tensor is loaded.
        feature = observable_pose_risk_features(raw, candidate, scale, prediction, raw_pnp, candidate_pnp)
        reference = torch.from_numpy(values["translation"][start:end].astype(np.float32)).to(device)
        labels = offline_benefit_labels(raw_pnp, candidate_pnp, reference, label_policy)
        feature_chunks.append(feature.cpu())
        label_chunks.append(labels.beneficial.cpu())
        eligible_chunks.append(labels.eligible.cpu())
        raw_valid_count += int(raw_pnp.valid[:, 0].sum())
        candidate_valid_count += int(candidate_pnp.valid[:, 0].sum())
    features = torch.cat(feature_chunks)
    labels = torch.cat(label_chunks)
    eligible = torch.cat(eligible_chunks)
    if not eligible.any():
        raise RuntimeError("GPU raw PnP produced no eligible risk-gate examples")
    diagnostics = {
        "samples": len(features),
        "eligible_samples": int(eligible.sum()),
        "positive_samples": int((labels & eligible).sum()),
        "positive_fraction": float((labels & eligible).sum() / eligible.sum()),
        "raw_valid_fraction": raw_valid_count / len(features),
        "candidate_valid_fraction": candidate_valid_count / len(features),
    }
    return features, labels, eligible, diagnostics


def _classification(model: ObservablePoseRiskGate, features: torch.Tensor,
                    labels: torch.Tensor, *, threshold: float,
                    positive_weight: torch.Tensor) -> dict[str, float]:
    logits = model(features)
    loss = F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype), pos_weight=positive_weight)
    prediction = torch.sigmoid(logits) >= threshold
    true_positive = (prediction & labels).sum()
    false_positive = (prediction & ~labels).sum()
    false_negative = (~prediction & labels).sum()
    true_negative = (~prediction & ~labels).sum()
    precision = true_positive / (true_positive + false_positive).clamp_min(1)
    recall = true_positive / (true_positive + false_negative).clamp_min(1)
    specificity = true_negative / (true_negative + false_positive).clamp_min(1)
    return {
        "bce": float(loss.detach()),
        "accuracy": float((prediction == labels).to(torch.float32).mean()),
        "precision": float(precision), "recall": float(recall),
        "specificity": float(specificity),
        "acceptance_fraction": float(prediction.to(torch.float32).mean()),
        "samples": int(len(features)),
    }


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_risk_gate(*, plan_path: Path, train_pack_path: Path,
                    validation_pack_path: Path, sparse_checkpoint_path: Path,
                    output_dir: Path, trust_scale: float = 1.0,
                    gate_threshold: float = 0.5,
                    label_policy: RiskLabelPolicy = RiskLabelPolicy(),
                    epochs: int = 50, batch_size: int = 128,
                    learning_rate: float = 1.0e-3, seed: int = 20260824,
                    hidden: int = 64, dropout: float = 0.05,
                    patience: int = 10) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("risk-gate training requires CUDA and refuses CPU fallback")
    if not 0.0 <= trust_scale <= 1.0 or not 0.0 <= gate_threshold <= 1.0:
        raise ValueError("trust_scale and gate_threshold must be in [0,1]")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0 or patience <= 0:
        raise ValueError("invalid risk-gate optimization configuration")
    label_policy.validate()
    plan_path = plan_path.resolve(strict=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "aim-stack.armor-pose-experiment-plan/1":
        raise ValueError("unsupported armor-pose plan")
    plan_hash = sha256(plan_path)
    train_pack_path = _reject_protected_pack(train_pack_path, expected_split="train")
    validation_pack_path = _reject_protected_pack(validation_pack_path, expected_split="validation")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse protected risk-gate run: {output_dir}")
    device = torch.device("cuda")
    sparse = _load_sparse(sparse_checkpoint_path, device=device)
    train_pack = load_development_pack(
        train_pack_path, expected_split="train",
        feature_mean=sparse.feature_mean, feature_std=sparse.feature_std,
    )
    validation_pack = load_development_pack(
        validation_pack_path, expected_split="validation",
        feature_mean=sparse.feature_mean, feature_std=sparse.feature_std,
    )
    train_sessions = set(map(str, train_pack.values["session_id"]))
    validation_sessions = set(map(str, validation_pack.values["session_id"]))
    if train_sessions & validation_sessions:
        raise PermissionError("risk-gate train/validation must be whole-session disjoint")
    _seed(seed)
    train_feature, train_label, train_eligible, train_generation = _gate_examples(
        train_pack, sparse, trust_scale=trust_scale, label_policy=label_policy,
        batch_size=batch_size, device=device,
    )
    validation_feature, validation_label, validation_eligible, validation_generation = _gate_examples(
        validation_pack, sparse, trust_scale=trust_scale, label_policy=label_policy,
        batch_size=batch_size, device=device,
    )
    train_feature = train_feature[train_eligible]
    train_label = train_label[train_eligible]
    validation_feature = validation_feature[validation_eligible]
    validation_label = validation_label[validation_eligible]
    feature_mean = train_feature.mean(dim=0)
    feature_std = train_feature.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    model = ObservablePoseRiskGate(hidden=hidden, dropout=dropout).to(device)
    model.set_normalization(feature_mean.to(device), feature_std.to(device))
    train_feature = train_feature.to(device)
    train_label = train_label.to(device)
    validation_feature = validation_feature.to(device)
    validation_label = validation_label.to(device)
    positives = train_label.sum()
    negatives = (~train_label).sum()
    positive_weight = torch.where(
        positives > 0, negatives.to(torch.float32) / positives.clamp_min(1).to(torch.float32),
        torch.ones((), device=device),
    ).clamp(0.25, 20.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    (output_dir / "checkpoints").mkdir()
    write_json_new(output_dir / "run-manifest.json", {
        "schema_version": "aim-stack.armor-pose-risk-gate-training-run/1",
        "plan": str(plan_path), "plan_sha256": plan_hash,
        "train_pack": str(train_pack_path), "train_pack_sha256": sha256(train_pack_path),
        "validation_pack": str(validation_pack_path), "validation_pack_sha256": sha256(validation_pack_path),
        "sparse_checkpoint": str(sparse.checkpoint), "sparse_checkpoint_sha256": sparse.checkpoint_hash,
        "source_sparse_plan_sha256": sparse.source_plan_hash,
        "source_sparse_model": sparse.model_config,
        "gate_plan_sha256": plan_hash,
        "train_sessions": sorted(train_sessions), "validation_sessions": sorted(validation_sessions),
        "whole_session_split": True, "trust_scale": trust_scale,
        "gate_threshold": gate_threshold, "label_policy": label_policy.config,
        "feature_names": list(FEATURE_NAMES), "feature_dimension": FEATURE_DIMENSION,
        "train_generation": train_generation, "validation_generation": validation_generation,
        "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "seed": seed, "patience": patience, "model": model.config,
        "device": "cuda", "gpu": torch.cuda.get_device_name(0), "cpu_fallback": False,
        "online_truth_input": False, "truth_usage": "offline risk label only",
        "exploratory_only": True, "test_accessed": False,
    })
    history: list[dict[str, Any]] = []
    best = float("inf")
    selected_checkpoint: Path | None = None
    stale = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_feature), generator=generator, device=device)
        loss_sum = 0.0
        for start in range(0, len(permutation), batch_size):
            index = permutation[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_feature[index])
            loss = F.binary_cross_entropy_with_logits(
                logits, train_label[index].to(logits.dtype), pos_weight=positive_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite risk-gate BCE")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
        model.eval()
        with torch.no_grad():
            validation = _classification(
                model, validation_feature, validation_label,
                threshold=gate_threshold, positive_weight=positive_weight,
            )
        history.append({
            "epoch": epoch,
            "train_bce": loss_sum / len(train_feature),
            "validation": validation,
        })
        if validation["bce"] < best - 1.0e-10:
            best, stale = validation["bce"], 0
            selected_checkpoint = output_dir / "checkpoints" / f"best-epoch-{epoch:03d}.pt"
            torch.save({
                "schema_version": SCHEMA, "state_dict": model.state_dict(),
                "model": model.config, "feature_mean": feature_mean.numpy(),
                "feature_std": feature_std.numpy(), "plan_sha256": plan_hash,
                "sparse_checkpoint_sha256": sparse.checkpoint_hash,
                "source_sparse_plan_sha256": sparse.source_plan_hash,
                "gate_plan_sha256": plan_hash,
                "trust_scale": trust_scale, "gate_threshold": gate_threshold,
                "label_policy": label_policy.config, "epoch": epoch,
                "validation": validation, "online_truth_input": False,
                "test_accessed": False, "device": "cuda",
            }, selected_checkpoint)
        else:
            stale += 1
        if stale >= patience:
            break
    if selected_checkpoint is None:
        raise RuntimeError("risk-gate training selected no checkpoint")
    result: dict[str, Any] = {
        "schema_version": "aim-stack.armor-pose-risk-gate-training-result/1",
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_sha256": sha256(selected_checkpoint),
        "sparse_checkpoint_sha256": sparse.checkpoint_hash,
        "source_sparse_plan_sha256": sparse.source_plan_hash,
        "gate_plan_sha256": plan_hash,
        "trust_scale": trust_scale, "gate_threshold": gate_threshold,
        "label_policy": label_policy.config, "best_validation_bce": best,
        "epochs_completed": len(history), "history": history,
        "test_accessed": False, "test_used_for_selection": False,
        "exploratory_only": True,
    }
    write_json_new(output_dir / "training-result.json", result)
    return result


def load_risk_gate_checkpoint(path: Path, *, plan_hash: str,
                              sparse_checkpoint_hash: str,
                              device: torch.device) -> LoadedRiskGate:
    resolved = path.resolve(strict=True)
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != SCHEMA:
        raise ValueError("unsupported risk-gate checkpoint")
    if checkpoint.get("plan_sha256") != plan_hash:
        raise ValueError("risk gate belongs to another experiment plan")
    if checkpoint.get("sparse_checkpoint_sha256") != sparse_checkpoint_hash:
        raise ValueError("risk gate is not bound to this sparse checkpoint")
    source_sparse_plan_hash = str(checkpoint.get("source_sparse_plan_sha256", ""))
    if len(source_sparse_plan_hash) != 64:
        raise ValueError("risk gate lacks source sparse-plan provenance")
    if checkpoint.get("online_truth_input") is not False or checkpoint.get("test_accessed") is not False:
        raise PermissionError("risk-gate checkpoint provenance is not truth/test clean")
    model_data = checkpoint["model"]
    if (model_data.get("family") != ObservablePoseRiskGate.family
            or model_data.get("feature_dimension") != FEATURE_DIMENSION
            or model_data.get("feature_names") != list(FEATURE_NAMES)):
        raise ValueError("risk-gate online feature/model contract changed")
    model = ObservablePoseRiskGate(
        hidden=int(model_data["hidden"]), dropout=float(model_data["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    policy = RiskLabelPolicy(**checkpoint["label_policy"])
    policy.validate()
    trust_scale = float(checkpoint["trust_scale"])
    threshold = float(checkpoint["gate_threshold"])
    if (not np.isfinite(trust_scale) or not 0.0 <= trust_scale <= 1.0
            or not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0):
        raise ValueError("risk-gate trust scale/threshold is invalid")
    return LoadedRiskGate(
        model, resolved, sha256(resolved), sparse_checkpoint_hash, source_sparse_plan_hash,
        trust_scale, threshold, policy,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--train-pack", type=Path, required=True)
    parser.add_argument("--validation-pack", type=Path, required=True)
    parser.add_argument("--sparse-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trust-scale", type=float, default=1.0)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-position-gain-mm", type=float, default=0.0)
    parser.add_argument("--minimum-depth-gain-mm", type=float, default=0.0)
    parser.add_argument("--maximum-ray-regression-mrad", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(train_risk_gate(
        plan_path=args.plan, train_pack_path=args.train_pack,
        validation_pack_path=args.validation_pack,
        sparse_checkpoint_path=args.sparse_checkpoint, output_dir=args.output_dir,
        trust_scale=args.trust_scale, gate_threshold=args.gate_threshold,
        label_policy=RiskLabelPolicy(
            args.minimum_position_gain_mm, args.minimum_depth_gain_mm,
            args.maximum_ray_regression_mrad,
        ),
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed,
        hidden=args.hidden, dropout=args.dropout, patience=args.patience,
    ), indent=2))


if __name__ == "__main__":
    main()
