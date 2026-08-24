"""CUDA-only exploratory validation for sparse/dense probabilistic armor pose.

The evaluator deliberately has no test-split entry point.  Neural heads and
all PnP candidates receive same-frame observable tensors only; labelled pose
is read after inference and is used exclusively to compute offline metrics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.corner_pnp.data import sha256, write_json_new
from training.corner_pnp.pnp import NOMINAL_OBJECT_POINTS_M

from .data import load_development_pack
from .dense_correspondence_head import (
    DenseCorrespondences, build_dense_correspondence_net, stratified_correspondences,
)
from .gpu_pnp import (
    GpuPnPResult,
    ObservablePnPInitialization,
    observable_initialization_from_result,
    solve_weighted_planar_pnp,
)
from .sparse_prob_head import ProbabilisticCornerNet, SparsePrediction


SCHEMA = "aim-stack.armor-pose-exploratory-evaluation/1"
FORBIDDEN_PACK_TOKENS = ("test-v15", "test-v18", "sealed")
MEASURES = ("position_mm", "depth_abs_mm", "ray_mrad")


@dataclass(frozen=True)
class PoseEstimate:
    translation_m: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class LoadedBranch:
    model: torch.nn.Module
    checkpoint: Path
    checkpoint_hash: str
    source_plan_hash: str
    feature_mean: np.ndarray
    feature_std: np.ndarray


def _reject_non_validation_pack(path: Path) -> None:
    lowered = str(path).lower()
    if any(token in lowered for token in FORBIDDEN_PACK_TOKENS):
        raise PermissionError("sealed V15/V18 test evidence is forbidden to exploratory evaluation")
    if path.name.lower().startswith("test"):
        raise PermissionError("exploratory evaluator accepts validation packs only")


def _load_branch(path: Path, *, branch: str, plan_hash: str,
                 device: torch.device,
                 allow_source_plan_mismatch: bool = False) -> LoadedBranch:
    resolved = path.resolve(strict=True)
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "aim-stack.armor-pose-checkpoint/1":
        raise ValueError(f"unsupported {branch} checkpoint schema")
    if checkpoint.get("branch") != branch:
        raise ValueError(f"expected {branch} checkpoint, got {checkpoint.get('branch')}")
    source_plan_hash = str(checkpoint.get("plan_sha256", ""))
    if (source_plan_hash != plan_hash) and not allow_source_plan_mismatch:
        raise ValueError(f"{branch} checkpoint belongs to another experiment plan")
    if len(source_plan_hash) != 64 or any(character not in "0123456789abcdef" for character in source_plan_hash):
        raise ValueError(f"{branch} checkpoint has invalid source-plan provenance")
    if checkpoint.get("online_truth_input") is not False or checkpoint.get("test_accessed") is not False:
        raise PermissionError(f"{branch} checkpoint provenance is not truth/test clean")
    feature_mean = np.asarray(checkpoint.get("feature_mean"), dtype=np.float32)
    feature_std = np.asarray(checkpoint.get("feature_std"), dtype=np.float32)
    if (feature_mean.shape != (15,) or feature_std.shape != (15,)
            or not np.isfinite(feature_mean).all() or not np.isfinite(feature_std).all()
            or np.any(feature_std <= 0.0)):
        raise ValueError(f"{branch} checkpoint has invalid feature normalization")
    if branch == "sparse":
        model: torch.nn.Module = ProbabilisticCornerNet()
    else:
        model_config = checkpoint.get("model")
        if not isinstance(model_config, dict):
            raise ValueError("dense checkpoint omitted its reconstructible model config")
        model = build_dense_correspondence_net(model_config=model_config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return LoadedBranch(
        model, resolved, sha256(resolved), source_plan_hash, feature_mean, feature_std,
    )


def _mode_zero(result: GpuPnPResult) -> tuple[torch.Tensor, torch.Tensor]:
    return result.translation_m[:, 0], result.valid[:, 0]


def _solve_fusion(sparse: SparsePrediction, dense: DenseCorrespondences,
                  intrinsics: torch.Tensor,
                  initialization: ObservablePnPInitialization | None = None) -> GpuPnPResult:
    batch = sparse.image_mean.shape[0]
    sparse_object = torch.as_tensor(
        NOMINAL_OBJECT_POINTS_M, dtype=sparse.image_mean.dtype, device=sparse.image_mean.device,
    )[None].expand(batch, -1, -1)
    sparse_quality = torch.reciprocal(
        torch.diagonal(sparse.image_covariance, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(1.0e-4)
    )
    sparse_weight = sparse_quality / sparse_quality.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    dense_weight = dense.weights / dense.weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    weights = torch.cat((sparse_weight, dense_weight), dim=1)
    image_points = torch.cat((sparse.image_mean, dense.image_points), dim=1)
    object_points = torch.cat((sparse_object, dense.object_points), dim=1)
    identity = torch.eye(2, dtype=image_points.dtype, device=image_points.device)
    dense_covariance = identity.expand(batch, dense.image_points.shape[1], 2, 2)
    covariance = torch.cat((sparse.image_covariance, dense_covariance), dim=1)
    return solve_weighted_planar_pnp(
        image_points, object_points, intrinsics, weights=weights, covariance=covariance,
        initialization=initialization,
    )


def _online_batch(values: dict[str, np.ndarray], start: int, end: int,
                  sparse: LoadedBranch, dense: LoadedBranch, device: torch.device,
                  *, dense_count: int, fusion: bool) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Run observable-only inference.  This signature intentionally has no labels."""
    patch = torch.from_numpy(values["patch"][start:end].astype(np.float32)).to(device)
    raw_feature = values["features"][start:end]
    sparse_geometry = torch.from_numpy(
        ((raw_feature - sparse.feature_mean) / sparse.feature_std).astype(np.float32)
    ).to(device)
    dense_geometry = torch.from_numpy(
        ((raw_feature - dense.feature_mean) / dense.feature_std).astype(np.float32)
    ).to(device)
    raw = torch.from_numpy(values["raw"][start:end].astype(np.float32)).to(device)
    raw_patch = torch.from_numpy(values["raw_patch"][start:end].astype(np.float32)).to(device)
    inverse = torch.from_numpy(values["inverse_transform"][start:end].astype(np.float32)).to(device)
    scale = torch.from_numpy(values["scale"][start:end].astype(np.float32)).to(device)
    intrinsics = torch.from_numpy(values["intrinsics"][start:end].astype(np.float32)).to(device)
    object_points = torch.as_tensor(
        NOMINAL_OBJECT_POINTS_M, dtype=raw.dtype, device=device,
    )[None].expand(end - start, -1, -1)

    raw_pnp = solve_weighted_planar_pnp(raw, object_points, intrinsics)
    sparse_prediction = sparse.model(patch, sparse_geometry, raw, raw_patch, inverse, scale)
    assert isinstance(sparse_prediction, SparsePrediction)
    sparse_pnp = solve_weighted_planar_pnp(
        sparse_prediction.image_mean, object_points, intrinsics,
        covariance=sparse_prediction.image_covariance,
    )
    dense_prediction = dense.model(patch, dense_geometry, raw_patch)
    dense_set = stratified_correspondences(dense_prediction, inverse, count=dense_count)
    dense_initialization = (
        observable_initialization_from_result(raw_pnp)
        if dense.model.config.get("architecture") == "spatial_projective_v2"
        else None
    )
    dense_pnp = solve_weighted_planar_pnp(
        dense_set.image_points, dense_set.object_points, intrinsics,
        weights=dense_set.weights, initialization=dense_initialization,
    )
    result = {
        "raw": _mode_zero(raw_pnp),
        "sparse": _mode_zero(sparse_pnp),
        "dense": _mode_zero(dense_pnp),
    }
    if fusion:
        result["fusion"] = _mode_zero(_solve_fusion(
            sparse_prediction, dense_set, intrinsics,
            initialization=dense_initialization,
        ))
    return result


def _metric_arrays(translation: np.ndarray, reference: np.ndarray) -> dict[str, np.ndarray]:
    delta = translation - reference
    reference_z = np.maximum(reference[:, 2], 1.0e-6)
    return {
        "position_mm": 1.0e3 * np.linalg.norm(delta, axis=1),
        "depth_abs_mm": 1.0e3 * np.abs(delta[:, 2]),
        "ray_mrad": 1.0e3 * np.linalg.norm(delta[:, :2] / reference_z[:, None], axis=1),
    }


def _metrics(translation: np.ndarray, valid: np.ndarray,
             reference: np.ndarray) -> dict[str, Any]:
    count = int(valid.sum())
    arrays = _metric_arrays(translation, reference)
    result: dict[str, Any] = {"samples": count}
    for name in MEASURES:
        selected = arrays[name][valid]
        result[name] = {
            "p50": float(np.quantile(selected, 0.50)) if count else None,
            "p95": float(np.quantile(selected, 0.95)) if count else None,
        }
    return result


def _paired_improvement(candidate_translation: np.ndarray, candidate_valid: np.ndarray,
                        raw: PoseEstimate, reference: np.ndarray) -> dict[str, Any]:
    paired = candidate_valid & raw.valid
    candidate_metrics = _metrics(candidate_translation, paired, reference)
    raw_metrics = _metrics(raw.translation_m, paired, reference)
    improvement: dict[str, dict[str, float | None]] = {}
    for name in MEASURES:
        improvement[name] = {}
        for quantile in ("p50", "p95"):
            baseline = raw_metrics[name][quantile]
            value = candidate_metrics[name][quantile]
            improvement[name][quantile] = (
                (baseline - value) / baseline
                if baseline is not None and value is not None and baseline > 1.0e-12 else None
            )
    return {
        "paired_samples": int(paired.sum()),
        "raw_metrics": raw_metrics,
        "candidate_metrics": candidate_metrics,
        "relative_improvement_fraction": improvement,
    }


def _method_summary(method: PoseEstimate, raw: PoseEstimate, reference: np.ndarray,
                    selected: np.ndarray) -> dict[str, Any]:
    count = int(selected.sum())
    candidate_valid = selected & method.valid
    fail_closed_translation = method.translation_m
    fallback_used = selected & ~method.valid & raw.valid
    fallback_valid = selected & (method.valid | raw.valid)
    fallback_translation = method.translation_m.copy()
    fallback_translation[fallback_used] = raw.translation_m[fallback_used]
    return {
        "candidate_valid_count": int(candidate_valid.sum()),
        "candidate_valid_rate": float(candidate_valid.sum() / count) if count else 0.0,
        "fail_closed": {
            "policy": "emit no pose when this candidate is invalid",
            "valid_count": int(candidate_valid.sum()),
            "valid_rate": float(candidate_valid.sum() / count) if count else 0.0,
            "metrics": _metrics(fail_closed_translation, candidate_valid, reference),
            "paired_vs_raw": _paired_improvement(fail_closed_translation, candidate_valid, raw, reference),
        },
        "raw_fallback": {
            "policy": "use raw GPU-PnP mode-0 pose when this candidate is invalid",
            "fallback_used_count": int(fallback_used.sum()),
            "valid_count": int(fallback_valid.sum()),
            "valid_rate": float(fallback_valid.sum() / count) if count else 0.0,
            "metrics": _metrics(fallback_translation, fallback_valid, reference),
            "paired_vs_raw": _paired_improvement(fallback_translation, fallback_valid, raw, reference),
        },
    }


def _group_summary(estimates: dict[str, PoseEstimate], reference: np.ndarray,
                   selected: np.ndarray) -> dict[str, Any]:
    raw = estimates["raw"]
    return {
        "samples": int(selected.sum()),
        "methods": {
            name: _method_summary(value, raw, reference, selected)
            for name, value in estimates.items()
        },
    }


def evaluate(*, plan_path: Path, validation_pack_path: Path,
             sparse_checkpoint_path: Path, dense_checkpoint_path: Path,
             output_dir: Path, batch_size: int = 8, maximum_samples: int | None = None,
             dense_count: int = 64, fusion: bool = True) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("armor-pose evaluation requires CUDA and refuses CPU fallback")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if maximum_samples is not None and maximum_samples <= 0:
        raise ValueError("maximum_samples must be positive")
    if dense_count not in {32, 64, 128}:
        raise ValueError("dense_count must be one of 32/64/128")
    _reject_non_validation_pack(validation_pack_path)
    validation_pack_path = validation_pack_path.resolve(strict=True)
    # Re-check the resolved path so an innocently named symlink cannot expose
    # one of the sealed V15/V18 packs before the split manifest is inspected.
    _reject_non_validation_pack(validation_pack_path)
    plan_path = plan_path.resolve(strict=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "aim-stack.armor-pose-experiment-plan/1":
        raise ValueError("unsupported armor-pose experiment plan")
    plan_hash = sha256(plan_path)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite protected evaluation: {output_dir}")
    device = torch.device("cuda")
    allow_frozen_v19_sparse = str(plan.get("name", "")).endswith("-v20")
    sparse = _load_branch(
        sparse_checkpoint_path, branch="sparse", plan_hash=plan_hash, device=device,
        allow_source_plan_mismatch=allow_frozen_v19_sparse,
    )
    dense = _load_branch(dense_checkpoint_path, branch="dense", plan_hash=plan_hash, device=device)
    if fusion and not (np.array_equal(sparse.feature_mean, dense.feature_mean)
                       and np.array_equal(sparse.feature_std, dense.feature_std)):
        raise ValueError("fusion requires checkpoints with identical detector-feature normalization")
    # expected_split is fixed here: there is no evaluator path that requests a test pack.
    pack = load_development_pack(validation_pack_path, expected_split="validation",
                                 feature_mean=sparse.feature_mean, feature_std=sparse.feature_std)
    count = len(pack.values["raw"])
    if maximum_samples is not None:
        count = min(count, maximum_samples)
    if count <= 0:
        raise ValueError("validation pack supplies no rows")

    chunks: dict[str, list[np.ndarray]] = {name: [] for name in ("raw", "sparse", "dense")}
    validity: dict[str, list[np.ndarray]] = {name: [] for name in chunks}
    if fusion:
        chunks["fusion"], validity["fusion"] = [], []
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            end = min(count, start + batch_size)
            output = _online_batch(pack.values, start, end, sparse, dense, device,
                                   dense_count=dense_count, fusion=fusion)
            for name, (translation, valid) in output.items():
                chunks[name].append(translation.detach().cpu().numpy().astype(np.float64))
                validity[name].append(valid.detach().cpu().numpy().astype(bool))
    estimates = {
        name: PoseEstimate(np.concatenate(chunks[name]), np.concatenate(validity[name]))
        for name in chunks
    }
    # Reference translation is intentionally touched only after online inference is complete.
    reference = pack.values["translation"][:count].astype(np.float64)
    modes = pack.values["mode"][:count].astype(str)
    sessions = pack.values["session_id"][:count].astype(str)
    all_rows = np.ones(count, dtype=bool)
    aggregate = _group_summary(estimates, reference, all_rows)
    by_mode = {
        name: _group_summary(estimates, reference, modes == name)
        for name in sorted(set(modes.tolist()))
    }
    by_session = {
        name: _group_summary(estimates, reference, sessions == name)
        for name in sorted(set(sessions.tolist()))
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "split": "validation",
        "exploratory_only": True,
        "test_accessed": False,
        "test_used_for_selection": False,
        "online_truth_input": False,
        "reference_usage": "reference translation is read only after inference and used only for metrics",
        "plan": str(plan_path),
        "plan_sha256": plan_hash,
        "validation_pack": str(validation_pack_path),
        "validation_pack_sha256": sha256(validation_pack_path),
        "pack_manifest_split": pack.manifest.get("split"),
        "checkpoints": {
            "sparse": {"path": str(sparse.checkpoint), "sha256": sparse.checkpoint_hash,
                       "source_plan_sha256": sparse.source_plan_hash},
            "dense": {"path": str(dense.checkpoint), "sha256": dense.checkpoint_hash,
                      "source_plan_sha256": dense.source_plan_hash},
        },
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "neural_cpu_fallback": False,
        "geometry_cpu_fallback": False,
        "raw_baseline": "truth-free CUDA weighted planar PnP from raw YOLO corners and nominal 135x55-mm points",
        "invalid_candidate_reporting": ["fail_closed", "raw_fallback"],
        "dense_correspondence_count": dense_count,
        "fusion_enabled": fusion,
        "batch_size": batch_size,
        "maximum_samples": maximum_samples,
        "samples": count,
        "aggregate": aggregate,
        "by_mode": by_mode,
        "by_session": by_session,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    write_json_new(output_dir / "evaluation-manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--validation-pack", type=Path, required=True)
    parser.add_argument("--sparse-checkpoint", type=Path, required=True)
    parser.add_argument("--dense-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--maximum-samples", type=int)
    parser.add_argument("--dense-count", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--fusion", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(
        plan_path=args.plan, validation_pack_path=args.validation_pack,
        sparse_checkpoint_path=args.sparse_checkpoint, dense_checkpoint_path=args.dense_checkpoint,
        output_dir=args.output_dir, batch_size=args.batch_size, maximum_samples=args.maximum_samples,
        dense_count=args.dense_count, fusion=args.fusion,
    ), indent=2))


if __name__ == "__main__":
    main()
