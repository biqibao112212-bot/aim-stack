"""Evaluate the fixed-slot causal input-sufficiency oracle without test data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from .causal_physical_dataset import CausalPhysicalShardDataset
from .physical_baseline import RigidPoseLeastSquaresRollout
from .train_causal_physical_ab import _load_geometry, _selection_tuple, _validate, _write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(args: argparse.Namespace) -> Path:
    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite causal oracle output: {output}")
    output.mkdir(parents=True)
    manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "stage3-causal-physical-v1":
        raise ValueError("causal oracle requires stage3-causal-physical-v1")
    if bool(manifest.get("test_accessed", True)):
        raise ValueError("causal oracle refuses a test-accessed dataset")
    validation = CausalPhysicalShardDataset(
        dataset, "validation", seed=0, shuffle=False, sample_limit=0
    )
    geometry, geometry_payload, geometry_sha256 = _load_geometry(dataset)
    model = RigidPoseLeastSquaresRollout(
        geometry, torch.from_numpy(validation.mean), torch.from_numpy(validation.std),
        fit_history_s=args.fit_history_s,
        fit_events=args.fit_events,
    ).to(args.device)
    loader = DataLoader(validation, batch_size=args.batch_size, num_workers=0)
    metrics = _validate(model, loader, torch.device(args.device), SimpleNamespace(amp="off"))
    headline = metrics["queries"][1:4]
    gates = {
        "state_q0_p95_le_1mm": float(metrics["state_q0"]["p95_m"]) <= 0.001,
        "rule_motion_q1_q3_p95_le_1mm": max(
            float(item["rule"]["motion_delta"]["p95_m"]) for item in headline
        ) <= 0.001,
        "rule_absolute_q1_q3_p95_le_1mm": max(
            float(item["rule"]["absolute"]["p95_m"]) for item in headline
        ) <= 0.001,
        "every_motion_class_q3_rule_motion_p95_le_1mm": all(
            int(item["q3_rule_motion"].get("count", 0)) > 0
            and float(item["q3_rule_motion"]["p95_m"]) <= 0.001
            for item in metrics["strata"]["motion_class"].values()
        ),
    }
    report = {
        "schema_version": "stage3-causal-input-sufficiency-oracle-v1",
        "role": "parameter-free evidence only; not the learned deployment candidate",
        "dataset": str(dataset),
        "dataset_manifest_sha256": _sha256(dataset / "dataset_manifest.json"),
        "geometry_hash": geometry_payload.get("geometry_hash"),
        "geometry_template_sha256": geometry_sha256,
        "split": "validation",
        "test_accessed": False,
        "model_config": model.config(),
        "metrics": metrics,
        "selection_tuple": _selection_tuple(metrics),
        "acceptance_gates": gates,
        "qualified_input_sufficiency": all(gates.values()),
    }
    report_path = output / "validation_report.json"
    _write_json(report_path, report)
    checkpoint = output / "causal-input-sufficiency-oracle.pt"
    torch.save({
        "model": model.state_dict(), "model_class": model.__class__.__name__,
        "model_config": model.config(), "validation_report": report,
        "checkpoint_role": "input_sufficiency_oracle", "test_accessed": False,
    }, checkpoint)
    _write_json(output / "artifact_manifest.json", {
        "schema_version": "stage3-causal-input-sufficiency-oracle-artifact-v1",
        "qualified_input_sufficiency": all(gates.values()), "test_accessed": False,
        "checkpoint": checkpoint.name, "checkpoint_sha256": _sha256(checkpoint),
        "validation_report": report_path.name,
        "validation_report_sha256": _sha256(report_path),
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fit-history-s", type=float, default=1.0)
    parser.add_argument("--fit-events", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1 or args.fit_history_s <= 0 or args.fit_events < 2:
        parser.error("batch-size/history must be positive and fit-events >= 2")
    print(evaluate(args))


if __name__ == "__main__":
    main()
