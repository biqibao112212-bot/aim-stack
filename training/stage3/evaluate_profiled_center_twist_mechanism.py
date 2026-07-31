"""Validation-only zero-update mechanism audit for the profiled center twist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .center_offset_supervision import (
    CENTER_OFFSET_TARGET_FIELD,
    CenterTruthIndex,
    attach_center_offset_after_frozen_upstream,
)
from .cyclic_future_foundation import load_frozen_v19
from .motion_truth_supervision import MOTION_TARGET_FIELD, MotionTruthIndex
from .observable_future_pnp_ab import sha256_file, state_dict_sha256
from .pnp_q0_hypothesis_adapter import (
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .profiled_center_twist_future import (
    AnonymousQ0CenterPrior,
    ProfiledRigidTwistAtOmega,
    translation_only_fwl,
)
from .train_anonymous_vehicle_motion import (
    _dataset, _json_sha256, _validate_bindings, frozen_upstream_batch,
)
from .train_causal_physical_ab import _to_device
from .train_pnp_window_mapper_distillation import _atomic_json


SCHEMA_VERSION = "stage3-profiled-center-twist-zero-update-audit-v1"
TRAINED_SCREEN_SCHEMA = "stage3-profiled-center-prior-truth-omega-screen-v14-b0"


def _validate_checkpoint_manifest_identity(
    payload: dict[str, Any], run_manifest: dict[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Validate checkpoint identity before checking external artifact bindings."""
    if (
        payload.get("schema_version") != TRAINED_SCREEN_SCHEMA
        or run_manifest.get("schema_version") != TRAINED_SCREEN_SCHEMA
    ):
        raise ValueError("center audit checkpoint schema differs")
    if (
        int(payload.get("global_update", -1)) != 100
        or int(run_manifest.get("global_update", -1)) != 100
    ):
        raise ValueError("center audit requires the fixed update-100 checkpoint")
    if run_manifest.get("status") != "completed":
        raise ValueError("center audit run did not complete")
    contract = run_manifest.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("center audit run contract missing")
    contract_sha256 = _json_sha256(contract)
    if (
        run_manifest.get("contract_sha256") != contract_sha256
        or payload.get("contract_sha256") != contract_sha256
    ):
        raise ValueError("center audit contract content/hash differs")
    if run_manifest.get("final_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("center audit checkpoint hash differs from run manifest")
    ledger = [
        item for item in run_manifest.get("checkpoint_ledger", [])
        if int(item.get("global_update", -1)) == 100
    ]
    if (
        len(ledger) != 1 or ledger[0].get("sha256") != checkpoint_sha256
        or ledger[0].get("kind") != "training_boundary"
    ):
        raise ValueError("center audit final checkpoint ledger differs")
    return contract


def _distribution(values: list[np.ndarray]) -> dict[str, float | int]:
    data = np.concatenate(values).astype(np.float64, copy=False)
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError("mechanism audit distribution is empty or non-finite")
    return {
        "count": int(data.size),
        "mean": float(data.mean()),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "coverage_le_0_75": float(np.mean(data <= 0.75)),
    }


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    dataset_path = Path(args.dataset).resolve()
    manifest_path = dataset_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("test_accessed", True)):
        raise ValueError("mechanism audit refuses a dataset that accessed test")
    expected_truth_sha = manifest.get("truth_history_manifest_sha256")
    if not isinstance(expected_truth_sha, str):
        raise ValueError("paired dataset lacks truth-history binding")
    dataset = _dataset(dataset_path, "validation", sample_limit=0)
    motion_truth = MotionTruthIndex(
        args.truth_history, expected_manifest_sha256=expected_truth_sha,
    )
    center_truth = CenterTruthIndex(
        args.truth_history, expected_manifest_sha256=expected_truth_sha,
    )
    motion_join = motion_truth.attach(dataset, "validation")
    center_join = center_truth.attach(dataset, "validation")
    if motion_join["joined_key_set_sha256"] != center_join["joined_key_set_sha256"]:
        raise ValueError("motion and center truth joins differ")

    mapper, mapper_info = load_frozen_pnp_mapper(args.mapper_checkpoint)
    s_model, s_info = load_frozen_v19(args.s_checkpoint)
    h_model, h_info = load_frozen_hypothesis_adapter(
        args.h_checkpoint, allow_diagnostic=True,
    )
    binding = _validate_bindings(
        sha256_file(manifest_path), mapper_info, s_info, h_info,
    )
    for model in (mapper, s_model, h_model):
        model.to(device).eval().requires_grad_(False)
    frozen_hashes = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    }

    center_model = AnonymousQ0CenterPrior(width=args.width).to(device).eval()
    checkpoint_provenance: dict[str, Any] | None = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
        run_manifest_path = checkpoint_path.parent.parent / "run_manifest.json"
        if checkpoint_path.parent.name != "checkpoints" or not run_manifest_path.is_file():
            raise ValueError("center audit checkpoint lacks its owning run manifest")
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_sha = sha256_file(checkpoint_path)
        contract = _validate_checkpoint_manifest_identity(
            payload, run_manifest, checkpoint_sha,
        )
        if (
            contract.get("dataset_manifest_sha256") != sha256_file(manifest_path)
            or contract.get("truth_manifest_sha256") != expected_truth_sha
        ):
            raise ValueError("center audit checkpoint dataset/truth contract differs")
        contract_args = contract.get("args", {})
        mechanism = contract.get("profile_mechanism", {})
        if (
            int(contract_args.get("width", -1)) != args.width
            or float(mechanism.get("center_precision", -1)) != args.center_precision
        ):
            raise ValueError("center audit reconstruction config differs")
        expected_frozen_files = contract.get("frozen_checkpoint_sha256")
        actual_frozen_files = {
            "mapper": sha256_file(args.mapper_checkpoint),
            "s": sha256_file(args.s_checkpoint),
            "h": sha256_file(args.h_checkpoint),
        }
        if expected_frozen_files != actual_frozen_files:
            raise ValueError("center audit frozen checkpoint files differ")
        if contract.get("frozen_state_dict_sha256") != frozen_hashes:
            raise ValueError("center audit frozen upstream states differ")
        source_paths = run_manifest.get("provenance", {}).get("source_paths", {})
        source_hashes = contract.get("source_sha256", {})
        if set(source_paths) != set(source_hashes) or any(
            sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise ValueError("center audit bound source changed")
        state = payload.get("model")
        if not isinstance(state, dict):
            raise ValueError("center audit checkpoint lacks model state")
        center_state = {
            key[len("center_prior."):]: value
            for key, value in state.items() if key.startswith("center_prior.")
        }
        center_model.load_state_dict(center_state, strict=True)
        checkpoint_provenance = {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "run_manifest": str(run_manifest_path),
            "schema_version": payload.get("schema_version"),
            "global_update": payload.get("global_update"),
            "contract_sha256": payload.get("contract_sha256"),
        }
    soft_profile = ProfiledRigidTwistAtOmega(
        center_precision=args.center_precision,
    ).to(device).eval()
    fixed_profile = ProfiledRigidTwistAtOmega(
        center_precision=args.fixed_center_precision,
        use_learned_center_variance=False,
    ).to(device).eval()
    variants = (
        "history_wide_center", "q0_soft_center", "q0_fixed_center",
        "oracle_fixed_center", "translation_only",
    )
    group_names = ("overall", "rotation", "combined")
    velocity_errors = {
        variant: {group: [] for group in group_names} for variant in variants
    }
    velocity_by_support = {
        variant: {str(count): [] for count in range(5)} for variant in variants
    }
    solver_coverage = {
        variant: {
            group: {"profile": 0, "fallback": 0, "state": 0, "count": 0}
            for group in group_names
        }
        for variant in variants if variant != "translation_only"
    }
    profile_energies = {
        variant: {group: [] for group in group_names}
        for variant in variants if variant != "translation_only"
    }
    center_errors = {group: [] for group in group_names}
    center_by_support = {str(count): [] for count in range(5)}
    support_histogram = {str(count): 0 for count in range(5)}

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    for raw_cpu in loader:
        raw = _to_device(raw_cpu, device)
        batch = frozen_upstream_batch(mapper, s_model, h_model, raw)
        attach_center_offset_after_frozen_upstream(batch, raw)
        target_state = raw[MOTION_TARGET_FIELD]
        target_center = batch[CENTER_OFFSET_TARGET_FIELD]
        omega = target_state[:, 3]
        prior = center_model(batch["q0_relation_m"], batch["q0_supported"])
        prior["q0_relation_m"] = batch["q0_relation_m"]
        oracle_prior = {
            "center_offset_m": target_center,
            "center_supported": torch.ones(
                target_center.shape[0], dtype=torch.bool, device=device,
            ),
        }
        profile_predictions = {
            "history_wide_center": soft_profile(
                batch["history_obs_rel_m"], batch["history_obs_mask"],
                batch["history_event_mask"], batch["history_time_s"],
                omega, prior, use_q0_prior=False,
            ),
            "q0_soft_center": soft_profile(
                batch["history_obs_rel_m"], batch["history_obs_mask"],
                batch["history_event_mask"], batch["history_time_s"],
                omega, prior,
            ),
            "q0_fixed_center": fixed_profile(
                batch["history_obs_rel_m"], batch["history_obs_mask"],
                batch["history_event_mask"], batch["history_time_s"],
                omega, prior,
            ),
            "oracle_fixed_center": fixed_profile(
                batch["history_obs_rel_m"], batch["history_obs_mask"],
                batch["history_event_mask"], batch["history_time_s"],
                omega, oracle_prior,
            ),
        }
        translation = translation_only_fwl(
                batch["history_obs_rel_m"], batch["history_obs_mask"],
                batch["history_event_mask"], batch["history_time_s"],
        )
        predictions = {
            name: value["velocity_mps"] for name, value in profile_predictions.items()
        }
        predictions["translation_only"] = translation["velocity_mps"]
        motion_class = batch["motion_class"]
        masks = {
            "overall": torch.ones_like(motion_class, dtype=torch.bool),
            "rotation": motion_class == 2,
            "combined": motion_class == 3,
        }
        center_error = torch.linalg.vector_norm(
            prior["center_offset_m"] - target_center, dim=-1,
        )
        for group, mask in masks.items():
            if bool(mask.any()):
                center_errors[group].append(center_error[mask].cpu().numpy())
                for variant, prediction in predictions.items():
                    error = torch.linalg.vector_norm(
                        prediction - target_state[:, :3], dim=-1,
                    )
                    velocity_errors[variant][group].append(error[mask].cpu().numpy())
                    if variant in profile_predictions:
                        profiled = profile_predictions[variant]
                        coverage = solver_coverage[variant][group]
                        coverage["count"] += int(mask.sum())
                        coverage["profile"] += int(profiled["profile_supported"][mask].sum())
                        coverage["fallback"] += int((
                            ~profiled["profile_supported"]
                            & profiled["fallback_supported"]
                        )[mask].sum())
                        coverage["state"] += int(profiled["state_supported"][mask].sum())
                        profile_energies[variant][group].append(
                            profiled["profile_energy"][mask].cpu().numpy()
                        )
        support_count = prior["center_support_count"]
        for count in range(5):
            selected = support_count == count
            support_histogram[str(count)] += int(selected.sum())
            if bool(selected.any()):
                center_by_support[str(count)].append(center_error[selected].cpu().numpy())
                for variant, prediction in predictions.items():
                    error = torch.linalg.vector_norm(
                        prediction - target_state[:, :3], dim=-1,
                    )
                    velocity_by_support[variant][str(count)].append(
                        error[selected].cpu().numpy()
                    )

    after_hashes = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    }
    if after_hashes != frozen_hashes:
        raise RuntimeError("zero-update audit modified frozen upstream")
    return {
        "schema_version": SCHEMA_VERSION,
        "test_accessed": False,
        "zero_training_updates": checkpoint_provenance is None,
        "checkpoint": checkpoint_provenance,
        "truth_omega_forward_input": True,
        "truth_omega_semantics": "mechanism-bound diagnostic only",
        "future_decoder_accessed": False,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "truth_manifest_sha256": expected_truth_sha,
        "motion_truth": motion_truth.provenance,
        "center_truth": center_truth.provenance,
        "motion_join": motion_join,
        "center_join": center_join,
        "binding": binding,
        "frozen_state_dict_sha256": frozen_hashes,
        "support_histogram": support_histogram,
        "center_error_m": {
            group: _distribution(values) for group, values in center_errors.items()
        },
        "center_error_m_by_support_count": {
            count: (_distribution(values) if values else {"count": 0})
            for count, values in center_by_support.items()
        },
        "truth_omega_velocity_error_mps": {
            variant: {
                group: _distribution(values) for group, values in groups.items()
            }
            for variant, groups in velocity_errors.items()
        },
        "truth_omega_velocity_error_mps_by_support_count": {
            variant: {
                count: (_distribution(values) if values else {"count": 0})
                for count, values in counts.items()
            }
            for variant, counts in velocity_by_support.items()
        },
        "solver_coverage": {
            variant: {
                group: {
                    "count": values["count"],
                    "profile_fraction": values["profile"] / values["count"],
                    "fallback_fraction": values["fallback"] / values["count"],
                    "state_supported_fraction": values["state"] / values["count"],
                }
                for group, values in groups.items()
            }
            for variant, groups in solver_coverage.items()
        },
        "profile_energy": {
            variant: {
                group: _distribution(values) for group, values in groups.items()
            }
            for variant, groups in profile_energies.items()
        },
        "config": {
            "center_precision": args.center_precision,
            "fixed_center_precision": args.fixed_center_precision,
            "q0_carrier": "anonymous arithmetic mean of all four finite H hypotheses",
            "q0_support_usage": "uncertainty/support metadata only; no geometry deletion",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--truth-history", required=True)
    parser.add_argument("--mapper-checkpoint", required=True)
    parser.add_argument("--s-checkpoint", required=True)
    parser.add_argument("--h-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--center-precision", type=float, default=25.0)
    parser.add_argument("--fixed-center-precision", type=float, default=1000000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing audit output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, result)
    print(json.dumps({
        "output": str(output),
        "center_error_m": result["center_error_m"],
        "truth_omega_velocity_error_mps": result["truth_omega_velocity_error_mps"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
