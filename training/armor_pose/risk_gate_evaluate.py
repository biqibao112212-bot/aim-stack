"""Validation-only CUDA evaluation for a V20 observable sparse-pose risk gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.corner_pnp.data import sha256, write_json_new
from training.corner_pnp.pnp import NOMINAL_OBJECT_POINTS_M

from .data import load_development_pack
from .evaluate import PoseEstimate, _group_summary
from .gpu_pnp import solve_weighted_planar_pnp
from .risk_gate import (
    _load_sparse, _reject_protected_pack, apply_observable_pose_risk_gate,
    load_risk_gate_checkpoint, trusted_sparse_corners,
)


SCHEMA = "aim-stack.armor-pose-risk-gate-evaluation/1"


def _decision_summary(probability: np.ndarray, use_candidate: np.ndarray,
                      valid: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    count = int(selected.sum())
    values = probability[selected]
    return {
        "samples": count,
        "benefit_probability_p50": float(np.quantile(values, 0.50)) if count else None,
        "benefit_probability_p95": float(np.quantile(values, 0.95)) if count else None,
        "candidate_use_count": int((selected & use_candidate).sum()),
        "candidate_use_rate": float((selected & use_candidate).sum() / count) if count else 0.0,
        "raw_fallback_count": int((selected & ~use_candidate).sum()),
        "raw_fallback_rate": float((selected & ~use_candidate).sum() / count) if count else 0.0,
        "output_valid_count": int((selected & valid).sum()),
        "output_valid_rate": float((selected & valid).sum() / count) if count else 0.0,
    }


def evaluate_risk_gate(*, plan_path: Path, validation_pack_path: Path,
                       sparse_checkpoint_path: Path, risk_gate_checkpoint_path: Path,
                       output_dir: Path, batch_size: int = 128) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("risk-gate evaluation requires CUDA and refuses CPU fallback")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    validation_pack_path = _reject_protected_pack(
        validation_pack_path, expected_split="validation",
    )
    plan_path = plan_path.resolve(strict=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "aim-stack.armor-pose-experiment-plan/1":
        raise ValueError("unsupported gate evaluation plan")
    plan_hash = sha256(plan_path)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite risk-gate evaluation: {output_dir}")
    device = torch.device("cuda")
    sparse = _load_sparse(sparse_checkpoint_path, device=device)
    gate = load_risk_gate_checkpoint(
        risk_gate_checkpoint_path, plan_hash=plan_hash,
        sparse_checkpoint_hash=sparse.checkpoint_hash, device=device,
    )
    if gate.source_sparse_plan_hash != sparse.source_plan_hash:
        raise ValueError("risk gate records a different source sparse plan")
    pack = load_development_pack(
        validation_pack_path, expected_split="validation",
        feature_mean=sparse.feature_mean, feature_std=sparse.feature_std,
    )
    values = pack.values
    translations: dict[str, list[np.ndarray]] = {
        "raw": [], "candidate": [], "gated": [],
    }
    validity: dict[str, list[np.ndarray]] = {name: [] for name in translations}
    probabilities: list[np.ndarray] = []
    uses: list[np.ndarray] = []
    with torch.inference_mode():
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
            objects = torch.as_tensor(
                NOMINAL_OBJECT_POINTS_M, dtype=raw.dtype, device=device,
            )[None].expand(end - start, -1, -1)
            prediction = sparse.model(patch, geometry, raw, raw_patch, inverse, scale)
            candidate = trusted_sparse_corners(raw, prediction, trust_scale=gate.trust_scale)
            raw_pnp = solve_weighted_planar_pnp(raw, objects, intrinsics)
            candidate_pnp = solve_weighted_planar_pnp(
                candidate, objects, intrinsics, covariance=prediction.image_covariance,
            )
            decision = apply_observable_pose_risk_gate(
                gate.model, raw, candidate, scale, prediction, raw_pnp, candidate_pnp,
                threshold=gate.threshold,
            )
            for name, translation, valid in (
                ("raw", raw_pnp.translation_m[:, 0], raw_pnp.valid[:, 0]),
                ("candidate", candidate_pnp.translation_m[:, 0], candidate_pnp.valid[:, 0]),
                ("gated", decision.translation_m, decision.valid),
            ):
                translations[name].append(translation.cpu().numpy().astype(np.float64))
                validity[name].append(valid.cpu().numpy().astype(bool))
            probabilities.append(decision.benefit_probability.cpu().numpy().astype(np.float64))
            uses.append(decision.use_candidate.cpu().numpy().astype(bool))
    estimates = {
        name: PoseEstimate(np.concatenate(translations[name]), np.concatenate(validity[name]))
        for name in translations
    }
    probability = np.concatenate(probabilities)
    use_candidate = np.concatenate(uses)
    gated_valid = estimates["gated"].valid
    # Labels are read only after every online decision is finalized.
    reference = values["translation"].astype(np.float64)
    modes = values["mode"].astype(str)
    sessions = values["session_id"].astype(str)
    selected_all = np.ones(len(reference), dtype=bool)
    result: dict[str, Any] = {
        "schema_version": SCHEMA, "split": "validation",
        "exploratory_only": True, "test_accessed": False,
        "test_used_for_selection": False, "online_truth_input": False,
        "reference_usage": "reference translation is read only after online gate decisions",
        "plan": str(plan_path), "gate_plan_sha256": plan_hash,
        "validation_pack": str(validation_pack_path),
        "validation_pack_sha256": sha256(validation_pack_path),
        "source_sparse_checkpoint": str(sparse.checkpoint),
        "source_sparse_checkpoint_sha256": sparse.checkpoint_hash,
        "source_sparse_plan_sha256": sparse.source_plan_hash,
        "risk_gate_checkpoint": str(gate.checkpoint),
        "risk_gate_checkpoint_sha256": gate.checkpoint_hash,
        "trust_scale": gate.trust_scale, "gate_threshold": gate.threshold,
        "label_policy": gate.label_policy.config,
        "device": "cuda", "gpu": torch.cuda.get_device_name(0),
        "cpu_fallback": False, "samples": len(reference),
        "aggregate": _group_summary(estimates, reference, selected_all),
        "gate_decision": _decision_summary(probability, use_candidate, gated_valid, selected_all),
        "by_mode": {
            name: {
                "pose": _group_summary(estimates, reference, modes == name),
                "gate_decision": _decision_summary(probability, use_candidate, gated_valid, modes == name),
            }
            for name in sorted(set(modes.tolist()))
        },
        "by_session": {
            name: {
                "pose": _group_summary(estimates, reference, sessions == name),
                "gate_decision": _decision_summary(probability, use_candidate, gated_valid, sessions == name),
            }
            for name in sorted(set(sessions.tolist()))
        },
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
    parser.add_argument("--risk-gate-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(evaluate_risk_gate(
        plan_path=args.plan, validation_pack_path=args.validation_pack,
        sparse_checkpoint_path=args.sparse_checkpoint,
        risk_gate_checkpoint_path=args.risk_gate_checkpoint,
        output_dir=args.output_dir, batch_size=args.batch_size,
    ), indent=2))


if __name__ == "__main__":
    main()
