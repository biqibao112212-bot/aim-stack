"""Verify v7 interruption/resume equality across both internal boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .factorized_common_relative_motion_future import (
    FactorizedCommonRelativeMotionStateV7,
    factorized_motion_future_loss,
    factorized_state_train_step,
)
from .observable_future_pnp_ab import sha256_file
from .train_pnp_window_mapper_distillation import _atomic_json
from .train_stable_motion_bottleneck_future import train


RUN_SCHEMA = "stage3-v7-factorized-state-recovery-smoke-v1"


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
            and left.dtype == right.dtype and left.shape == right.shape
            and np.array_equal(left, right, equal_nan=True)
        )
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype and left.shape == right.shape
            and torch.equal(left.cpu(), right.cpu())
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict) and isinstance(right, dict)
            and set(left) == set(right)
            and all(_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, type(right)) and len(left) == len(right)
            and all(_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _args(base: argparse.Namespace, output: Path, *, stop: int, resume: str = ""):
    return SimpleNamespace(
        diagnostic_oracle_association=True,
        allow_mapper_h_mismatch=True,
        dataset=base.dataset,
        truth_history=base.truth_history,
        mapper_checkpoint=base.mapper_checkpoint,
        s_checkpoint=base.s_checkpoint,
        h_checkpoint=base.h_checkpoint,
        output=str(output), resume_checkpoint=resume,
        stop_after_update=stop, device=base.device, seed=base.seed,
        batch_size=base.batch_size,
        validation_batch_size=base.validation_batch_size,
        channels=base.channels, dropout=base.dropout,
        message_layers=base.message_layers, basis_count=base.basis_count,
        motion_state_updates=800, trajectory_updates=0,
        selector_updates=0, decoder_joint_updates=0,
        motion_state_learning_rate=3e-4,
        trajectory_learning_rate=3e-4,
        selector_learning_rate=3e-4,
        joint_trajectory_learning_rate=1e-4,
        joint_selector_learning_rate=1e-4,
        weight_decay=1e-4, gradient_clip_norm=1.0,
        prefix_dropout_probability=0.75,
        train_limit_per_class=base.train_limit_per_class,
        validation_limit_per_class=base.validation_limit_per_class,
        log_interval=base.log_interval,
    )


def _run(base: argparse.Namespace, output: Path, *, stop: int, resume: str = "") -> Path:
    return train(
        _args(base, output, stop=stop, resume=resume),
        model_class=FactorizedCommonRelativeMotionStateV7,
        loss_function=factorized_motion_future_loss,
        state_step_function=factorized_state_train_step,
        run_schema=RUN_SCHEMA,
        extra_source_paths={"recovery_verifier": Path(__file__)},
        state_gate_only=True,
        # Recovery semantics are architecture/sampler/RNG properties.  Keep the
        # identically seeded future modules frozen without forcing the formal
        # 96-channel v77 initialization into this reduced-capacity smoke.
        frozen_initialization_checkpoint=None,
        frozen_initialization_modules=(),
    )


def _case(
    base: argparse.Namespace,
    root: Path,
    *,
    name: str,
    split_update: int,
    endpoint: int,
) -> dict[str, Any]:
    continuous_root = root / f"{name}-continuous"
    split_root = root / f"{name}-split"
    continuous = _run(base, continuous_root, stop=endpoint)
    split_checkpoint = _run(base, split_root, stop=split_update)
    resumed = _run(
        base, split_root, stop=endpoint, resume=str(split_checkpoint),
    )
    continuous_payload = torch.load(
        continuous, map_location="cpu", weights_only=False,
    )
    resumed_payload = torch.load(resumed, map_location="cpu", weights_only=False)
    fields = (
        "model", "optimizers", "scaler", "rng", "validation_history",
        "state_substage_counts", "state_substage_transitions",
        "state_branch_hash_history", "gradient_isolation_verified",
    )
    comparisons = {
        field: _equal(continuous_payload[field], resumed_payload[field])
        for field in fields
    }
    if not all(comparisons.values()):
        failed = [name for name, passed in comparisons.items() if not passed]
        raise RuntimeError(f"v7 recovery differs for {name}: {failed}")
    return {
        "split_update": split_update,
        "endpoint": endpoint,
        "continuous_checkpoint": str(continuous),
        "continuous_sha256": sha256_file(continuous),
        "resumed_checkpoint": str(resumed),
        "resumed_sha256": sha256_file(resumed),
        "model_state_dict_sha256": continuous_payload["model_state_dict_sha256"],
        "comparisons": comparisons,
        "passed": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="v7 bit-exact recovery verification across updates 251 and 601",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--truth-history", required=True)
    parser.add_argument("--mapper-checkpoint", required=True)
    parser.add_argument("--s-checkpoint", required=True)
    parser.add_argument("--h-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-batch-size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--message-layers", type=int, default=2)
    parser.add_argument("--basis-count", type=int, default=6)
    parser.add_argument("--train-limit-per-class", type=int, default=64)
    parser.add_argument("--validation-limit-per-class", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=150)
    parser.add_argument(
        "--case", choices=("all", "cross-250", "cross-600"), default="all",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.output_root).resolve()
    if root.exists():
        raise FileExistsError(f"refusing existing recovery root: {root}")
    root.mkdir(parents=True)
    cases: dict[str, Any] = {}
    if args.case in {"all", "cross-250"}:
        cases["cross_angular_to_translation"] = _case(
            args, root, name="cross-250", split_update=150, endpoint=300,
        )
    if args.case in {"all", "cross-600"}:
        cases["cross_translation_to_joint"] = _case(
            args, root, name="cross-600", split_update=600, endpoint=750,
        )
    report = {
        "schema_version": "stage3-v7-factorized-recovery-verification-v1",
        "test_accessed": False,
        "cases": cases,
    }
    report["passed"] = all(item["passed"] for item in report["cases"].values())
    _atomic_json(root / "recovery-verification.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
