from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from training.armor_pose.gpu_pnp import GpuPnPResult
from training.armor_pose.risk_gate import (
    FEATURE_DIMENSION, ObservablePoseRiskGate, RiskLabelPolicy,
    _FrozenV19ProbabilisticCornerNet, _load_sparse, _reject_protected_pack,
    apply_observable_pose_risk_gate, observable_pose_risk_features,
    offline_benefit_labels, trusted_sparse_corners,
)
from training.armor_pose.risk_gate_evaluate import evaluate_risk_gate
from training.armor_pose.sparse_prob_head import ProbabilisticCornerNet
from training.corner_pnp.data import sha256


ROOT = Path("/home/potato/Projects/仿真")
PLAN_V20 = ROOT / "repos/aim-stack/training/armor_pose/experiment_plan_v20.json"
VALIDATION_PACK = ROOT / "runtime/autoaim-b-corner-pnp-20260824-v1/packs/validation-v17-fresh-candidate-reduced-scale.npz"
SOURCE_SPARSE = ROOT / "runtime/autoaim-b-prob-dense-pnp-20260824-v1/sparse-full-v19-e12-all-sessions-gpu/checkpoints/best-epoch-010.pt"


def _device() -> torch.device:
    assert torch.cuda.is_available(), "risk-gate tests fail closed without CUDA"
    return torch.device("cuda")


def _pnp(translation: torch.Tensor, *, valid0: bool = True,
         objective0: float = 1.0) -> GpuPnPResult:
    batch = translation.shape[0]
    device, dtype = translation.device, translation.dtype
    rotation = torch.eye(3, device=device, dtype=dtype)[None, None].expand(batch, 2, -1, -1).clone()
    translations = translation[:, None].expand(-1, 2, -1).clone()
    covariance = torch.eye(6, device=device, dtype=dtype)[None, None].expand(batch, 2, -1, -1).clone()
    objective = torch.tensor([[objective0, objective0 + 2.0]], device=device, dtype=dtype).expand(batch, -1).clone()
    valid = torch.tensor([[valid0, valid0]], device=device).expand(batch, -1).clone()
    condition = torch.tensor([[100.0, 120.0]], device=device, dtype=dtype).expand(batch, -1).clone()
    reprojection = torch.tensor([[0.2, 0.3]], device=device, dtype=dtype).expand(batch, -1).clone()
    log_weight = torch.log_softmax(-0.5 * objective, dim=1)
    return GpuPnPResult(rotation, translations, covariance, objective, log_weight,
                        valid, condition, reprojection)


def _observable(batch: int = 2) -> tuple[torch.Tensor, ...]:
    device = _device()
    raw_patch = torch.tensor(
        [[[31.0, 48.0], [31.0, 16.0], [96.0, 16.0], [96.0, 48.0]]], device=device,
    ).expand(batch, -1, -1).clone()
    raw = raw_patch + raw_patch.new_tensor([300.0, 200.0])
    inverse = torch.eye(3, device=device)[None].expand(batch, -1, -1).clone()
    inverse[:, :2, 2] = inverse.new_tensor([300.0, 200.0])
    scale = torch.full((batch,), 45.0, device=device)
    sparse = ProbabilisticCornerNet().to(device).eval()
    prediction = sparse(torch.rand(batch, 3, 64, 128, device=device),
                        torch.randn(batch, 15, device=device), raw, raw_patch, inverse, scale)
    candidate = trusted_sparse_corners(raw, prediction, trust_scale=0.5)
    raw_pnp = _pnp(torch.tensor([[0.10, 0.0, 4.10]], device=device).expand(batch, -1).clone())
    candidate_pnp = _pnp(torch.tensor([[0.05, 0.0, 4.05]], device=device).expand(batch, -1).clone(), objective0=0.5)
    return raw, candidate, scale, prediction, raw_pnp, candidate_pnp


def test_gate_online_signatures_are_truth_free() -> None:
    forbidden = ("reference", "target", "truth", "session", "mode", "range", "history", "future")
    for function in (
        ObservablePoseRiskGate.forward, observable_pose_risk_features,
        apply_observable_pose_risk_gate,
    ):
        names = inspect.signature(function).parameters
        assert not any(token in name for token in forbidden for name in names)


def test_observable_features_are_cuda_finite_and_reference_independent() -> None:
    online = _observable()
    feature_before = observable_pose_risk_features(*online)
    assert feature_before.shape == (2, FEATURE_DIMENSION)
    assert feature_before.is_cuda and torch.isfinite(feature_before).all()
    reference_a = torch.tensor([[0.0, 0.0, 4.0], [0.0, 0.0, 4.0]], device=_device())
    reference_b = torch.tensor([[1.0, 0.0, 4.0], [1.0, 0.0, 4.0]], device=_device())
    labels_a = offline_benefit_labels(online[-2], online[-1], reference_a, RiskLabelPolicy())
    labels_b = offline_benefit_labels(online[-2], online[-1], reference_b, RiskLabelPolicy())
    feature_after = observable_pose_risk_features(*online)
    assert torch.equal(feature_before, feature_after)
    assert not torch.equal(labels_a.beneficial, labels_b.beneficial)


def test_offline_label_requires_position_depth_gain_and_ray_nonregression() -> None:
    device = _device()
    reference = torch.tensor([[0.0, 0.0, 4.0], [0.0, 0.0, 4.0]], device=device)
    raw = _pnp(torch.tensor([[0.10, 0.0, 4.10], [0.10, 0.0, 4.10]], device=device))
    candidate = _pnp(torch.tensor([[0.05, 0.0, 4.05], [0.20, 0.0, 4.05]], device=device))
    labels = offline_benefit_labels(raw, candidate, reference, RiskLabelPolicy())
    assert labels.beneficial.tolist() == [True, False]
    invalid = _pnp(torch.tensor([[0.05, 0.0, 4.05], [0.05, 0.0, 4.05]], device=device), valid0=False)
    assert not offline_benefit_labels(raw, invalid, reference, RiskLabelPolicy()).beneficial.any()


def test_gate_low_or_candidate_invalid_uses_raw_fallback_cuda() -> None:
    raw, candidate, scale, prediction, raw_pnp, candidate_pnp = _observable()
    gate = ObservablePoseRiskGate().to(_device()).eval()
    with torch.no_grad():
        gate.network[-1].weight.zero_()
        gate.network[-1].bias.fill_(-20.0)
    decision = apply_observable_pose_risk_gate(
        gate, raw, candidate, scale, prediction, raw_pnp, candidate_pnp, threshold=0.5,
    )
    assert decision.fallback_to_raw.all()
    assert torch.equal(decision.translation_m, raw_pnp.translation_m[:, 0])
    assert decision.valid.all() and decision.translation_m.is_cuda
    with torch.no_grad():
        gate.network[-1].bias.fill_(20.0)
    candidate_pnp.valid[0, 0] = False
    decision = apply_observable_pose_risk_gate(
        gate, raw, candidate, scale, prediction, raw_pnp, candidate_pnp, threshold=0.5,
    )
    assert decision.fallback_to_raw[0] and not decision.use_candidate[0]
    assert decision.use_candidate[1]


def test_v20_gate_loader_preserves_v19_source_plan_and_architecture(tmp_path: Path) -> None:
    checkpoint = tmp_path / "v19-sparse.pt"
    source_plan_hash = "a" * 64
    model = _FrozenV19ProbabilisticCornerNet()
    torch.save({
        "schema_version": "aim-stack.armor-pose-checkpoint/1",
        "branch": "sparse", "state_dict": model.state_dict(),
        "model": {
            "family": "same-frame-four-heatmap-probabilistic-corners-v1",
            "online_truth_input": False,
        },
        "feature_mean": np.zeros(15, dtype=np.float32),
        "feature_std": np.ones(15, dtype=np.float32),
        "plan_sha256": source_plan_hash,
        "online_truth_input": False, "test_accessed": False,
    }, checkpoint)
    loaded = _load_sparse(checkpoint, device=_device())
    assert isinstance(loaded.model, _FrozenV19ProbabilisticCornerNet)
    assert loaded.source_plan_hash == source_plan_hash
    assert loaded.model_config["family"].endswith("probabilistic-corners-v1")


def test_gate_training_rejects_test_paths_before_access() -> None:
    with pytest.raises(PermissionError):
        _reject_protected_pack(Path("/does/not/exist/test-v18-sealed.npz"), expected_split="validation")


def test_validation_only_gate_evaluator_reports_candidate_and_fallback(tmp_path: Path) -> None:
    assert torch.cuda.is_available(), "risk-gate evaluator fails closed without CUDA"
    source = torch.load(SOURCE_SPARSE, map_location="cpu", weights_only=False)
    gate_model = ObservablePoseRiskGate()
    gate_checkpoint = tmp_path / "risk-gate.pt"
    torch.save({
        "schema_version": "aim-stack.armor-pose-risk-gate-checkpoint/1",
        "state_dict": gate_model.state_dict(), "model": gate_model.config,
        "feature_mean": np.zeros(FEATURE_DIMENSION, dtype=np.float32),
        "feature_std": np.ones(FEATURE_DIMENSION, dtype=np.float32),
        "plan_sha256": sha256(PLAN_V20),
        "gate_plan_sha256": sha256(PLAN_V20),
        "sparse_checkpoint_sha256": sha256(SOURCE_SPARSE),
        "source_sparse_plan_sha256": source["plan_sha256"],
        "trust_scale": 0.2, "gate_threshold": 0.5,
        "label_policy": RiskLabelPolicy().config,
        "online_truth_input": False, "test_accessed": False,
    }, gate_checkpoint)
    output_dir = tmp_path / "evaluation"
    result = evaluate_risk_gate(
        plan_path=PLAN_V20, validation_pack_path=VALIDATION_PACK,
        sparse_checkpoint_path=SOURCE_SPARSE,
        risk_gate_checkpoint_path=gate_checkpoint,
        output_dir=output_dir, batch_size=128,
    )
    disk = json.loads((output_dir / "evaluation-manifest.json").read_text(encoding="utf-8"))
    assert result == disk
    assert result["test_accessed"] is False and result["split"] == "validation"
    assert set(result["aggregate"]["methods"]) == {"raw", "candidate", "gated"}
    assert result["source_sparse_plan_sha256"] != result["gate_plan_sha256"]
    assert result["gate_decision"]["raw_fallback_count"] >= 0
