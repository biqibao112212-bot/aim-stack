from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from training.armor_pose.evaluate import (
    PoseEstimate, _method_summary, _reject_non_validation_pack, evaluate,
)
from training.armor_pose.sparse_prob_head import ProbabilisticCornerNet
from training.armor_pose.dense_correspondence_head import DenseCorrespondenceNet
from training.corner_pnp.data import load_pack, sha256


ROOT = Path("/home/potato/Projects/仿真")
REPOSITORY = ROOT / "repos/aim-stack"
PLAN = REPOSITORY / "training/armor_pose/experiment_plan_v19.json"
TRAIN_PACK = ROOT / "runtime/autoaim-b-corner-pnp-20260824-v1/packs/train-v12-pilot-calibrated-stable.npz"
VALIDATION_PACK = ROOT / "runtime/autoaim-b-corner-pnp-20260824-v1/packs/validation-v17-fresh-candidate-reduced-scale.npz"


def test_evaluator_rejects_test_and_sealed_paths_before_loading() -> None:
    for name in ("test-anything.npz", "test-v15-sealed-candidate.npz", "validation-sealed.npz"):
        with pytest.raises(PermissionError):
            _reject_non_validation_pack(Path("/does/not/exist") / name)


def test_invalid_candidate_reports_fail_closed_and_explicit_raw_fallback() -> None:
    reference = np.asarray([[0.0, 0.0, 4.0], [0.0, 0.0, 4.0], [0.0, 0.0, 4.0]])
    raw = PoseEstimate(
        np.asarray([[0.01, 0.0, 4.0], [0.02, 0.0, 4.0], [9.0, 9.0, 9.0]]),
        np.asarray([True, True, False]),
    )
    candidate = PoseEstimate(
        np.asarray([[0.005, 0.0, 4.0], [8.0, 8.0, 8.0], [0.03, 0.0, 4.0]]),
        np.asarray([True, False, True]),
    )
    summary = _method_summary(candidate, raw, reference, np.ones(3, dtype=bool))
    assert summary["candidate_valid_rate"] == pytest.approx(2.0 / 3.0)
    assert summary["fail_closed"]["valid_count"] == 2
    assert summary["raw_fallback"]["fallback_used_count"] == 1
    assert summary["raw_fallback"]["valid_count"] == 3
    assert summary["fail_closed"]["paired_vs_raw"]["paired_samples"] == 1
    assert summary["raw_fallback"]["paired_vs_raw"]["paired_samples"] == 2


def _write_checkpoint(path: Path, branch: str, model: torch.nn.Module,
                      feature_mean: np.ndarray, feature_std: np.ndarray) -> None:
    torch.save({
        "schema_version": "aim-stack.armor-pose-checkpoint/1",
        "branch": branch,
        "state_dict": model.state_dict(),
        "model": model.config,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "plan_sha256": sha256(PLAN),
        "online_truth_input": False,
        "test_accessed": False,
        "device": "cuda",
    }, path)


def test_cuda_evaluator_writes_non_overwritable_validation_manifest(tmp_path: Path) -> None:
    assert torch.cuda.is_available(), "armor-pose evaluation tests fail closed without CUDA"
    values, _ = load_pack(TRAIN_PACK, expected_split="train")
    feature_mean = values["features"].mean(axis=0).astype(np.float32)
    feature_std = np.maximum(values["features"].std(axis=0), 1.0e-6).astype(np.float32)
    sparse_checkpoint = tmp_path / "sparse.pt"
    dense_checkpoint = tmp_path / "dense.pt"
    _write_checkpoint(sparse_checkpoint, "sparse", ProbabilisticCornerNet(), feature_mean, feature_std)
    _write_checkpoint(dense_checkpoint, "dense", DenseCorrespondenceNet(), feature_mean, feature_std)
    output_dir = tmp_path / "evaluation"
    result = evaluate(
        plan_path=PLAN, validation_pack_path=VALIDATION_PACK,
        sparse_checkpoint_path=sparse_checkpoint, dense_checkpoint_path=dense_checkpoint,
        output_dir=output_dir, batch_size=1, maximum_samples=1, dense_count=32, fusion=True,
    )
    manifest_path = output_dir / "evaluation-manifest.json"
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result == on_disk
    assert result["split"] == "validation"
    assert result["test_accessed"] is False
    assert set(result["aggregate"]["methods"]) == {"raw", "sparse", "dense", "fusion"}
    assert result["validation_pack_sha256"] == sha256(VALIDATION_PACK)
    assert result["checkpoints"]["sparse"]["sha256"] == sha256(sparse_checkpoint)
    with pytest.raises(FileExistsError):
        evaluate(
            plan_path=PLAN, validation_pack_path=VALIDATION_PACK,
            sparse_checkpoint_path=sparse_checkpoint, dense_checkpoint_path=dense_checkpoint,
            output_dir=output_dir, batch_size=1, maximum_samples=1, dense_count=32, fusion=False,
        )
