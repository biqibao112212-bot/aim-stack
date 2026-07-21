"""Evaluate and package the parameter-free analytic physical core."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from .physical_baseline import ExactStateRigidRollout, RigidTwoFrameRollout
from .train_physical_ab import _selection_tuple, _validate_model
from .truth_history_dataset import TruthHistoryShardDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _provenance() -> dict[str, object]:
    root = Path(__file__).resolve().parent
    names = (
        "evaluate_physical_baseline.py", "physical_baseline.py",
        "physical_metrics.py", "physical_loss.py", "train_physical_ab.py",
        "truth_history_dataset.py",
    )
    repo = root.parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(repo), "status", "--short"], text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    return {
        "source_sha256": {name: _sha256(root / name) for name in names},
        "git_commit": commit,
        "worktree_dirty": dirty,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
    }


def evaluate(args: argparse.Namespace) -> Path:
    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite analytic core output: {output}")
    output.mkdir(parents=True)
    validation = TruthHistoryShardDataset(
        dataset, "validation", seed=0, shuffle=False, sample_limit=0
    )
    loader = DataLoader(
        validation, batch_size=args.batch_size, num_workers=0, pin_memory=False
    )
    device = torch.device(args.device)
    if args.method == "exact-state":
        model = ExactStateRigidRollout().to(device)
    else:
        model = RigidTwoFrameRollout(
            torch.from_numpy(validation.mean), torch.from_numpy(validation.std)
        ).to(device)
    metric_args = SimpleNamespace(
        amp="off", huber_beta_m=0.05, state_weight=2.0,
        motion_weight=1.0, absolute_weight=1.0, rigidity_weight=0.2,
    )
    metrics = _validate_model(model, loader, device, metric_args)
    headline = metrics["queries"][1:4]
    gates = {
        "state_q0_p95_le_1mm": float(metrics["state_q0"]["p95_m"]) <= 0.001,
        "rule_motion_q1_q3_p95_le_1mm": max(
            float(item["rule"]["motion_delta"]["p95_m"]) for item in headline
        ) <= 0.001,
        "rule_absolute_q1_q3_p95_le_1mm": max(
            float(item["rule"]["absolute_pg"]["p95_m"]) for item in headline
        ) <= 0.001,
    }
    report = {
        "schema_version": "stage3-analytic-physical-core-evaluation-v1",
        "model_family": model.model_family,
        "model_config": model.config(),
        "dataset": str(dataset),
        "dataset_manifest_sha256": _sha256(dataset / "dataset_manifest.json"),
        "split": "validation",
        "test_accessed": False,
        "metrics": metrics,
        "selection_tuple": _selection_tuple(metrics),
        "acceptance_gates": gates,
        "qualified": all(gates.values()),
        "provenance": _provenance(),
    }
    report_path = output / "validation_report.json"
    _write_json(report_path, report)
    checkpoint_path = output / "analytic-physical-core.pt"
    torch.save({
        "model": model.state_dict(),
        "model_class": model.__class__.__name__,
        "model_config": model.config(),
        "validation_report": report,
        "checkpoint_role": "qualified_analytic_physical_core" if all(gates.values()) else "failed_candidate",
        "test_accessed": False,
    }, checkpoint_path)
    _write_json(output / "artifact_manifest.json", {
        "schema_version": "stage3-analytic-physical-core-artifact-v1",
        "qualified": all(gates.values()),
        "test_accessed": False,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "validation_report": report_path.name,
        "validation_report_sha256": _sha256(report_path),
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--method", choices=("two-frame", "exact-state"), default="two-frame")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    print(evaluate(args))


if __name__ == "__main__":
    main()
