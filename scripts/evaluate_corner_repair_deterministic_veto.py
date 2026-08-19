#!/usr/bin/env python3
"""Select a transparent reject-only veto from runtime counterfactual features."""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
from pathlib import Path

import numpy as np

from train_corner_repair_benefit_gate import FEATURE_NAMES, Sample, gate_metrics, sha256


def load_samples(path: Path) -> tuple[list[Sample], list[dict[str, str]]]:
    samples = []
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            samples.append(
                Sample(
                    session=row["session"],
                    split=row["split"],
                    mode=row["mode"],
                    identity=tuple(
                        int(row[name])
                        for name in (
                            "producer_epoch",
                            "frame_seq",
                            "timestamp_ns",
                            "observation_id",
                        )
                    ),
                    features=np.asarray(
                        [float(row[name]) for name in FEATURE_NAMES], dtype=np.float32
                    ),
                    old_applied=row["old_applied"] == "1",
                    outcome=row["outcome"],
                    raw={
                        "angular_error_deg": float(row["raw_angular_error_deg"]),
                        "radial_error_abs_mm": float(row["raw_radial_error_abs_mm"]),
                        "transverse_error_mm": float(row["raw_transverse_error_mm"]),
                    },
                    proposed={
                        "angular_error_deg": float(row["proposed_angular_error_deg"]),
                        "radial_error_abs_mm": float(row["proposed_radial_error_abs_mm"]),
                        "transverse_error_mm": float(row["proposed_transverse_error_mm"]),
                    },
                )
            )
    return samples, rows


def decisions(rows: list[dict[str, str]], policy: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            row["old_applied"] == "1"
            and float(row["v3_reliability_probability"]) >= policy["minimum_reliability"]
            and abs(float(row["proposal_radial_shift_m"])) <= policy["maximum_abs_radial_shift_m"]
            and float(row["proposal_transverse_shift_m"]) <= policy["maximum_transverse_shift_m"]
            and float(row["proposal_ray_shift_rad"]) <= policy["maximum_ray_shift_rad"]
            and abs(float(row["proposal_width_ratio"]) - 1.0) <= policy["maximum_abs_width_ratio_delta"]
            and abs(float(row["proposal_area_ratio"]) - 1.0) <= policy["maximum_abs_area_ratio_delta"]
            and float(row["proposal_pnp_valid"]) == 1.0
            for row in rows
        ],
        dtype=np.float32,
    )


def policy_grid() -> list[dict[str, float]]:
    keys = (
        "minimum_reliability",
        "maximum_abs_radial_shift_m",
        "maximum_transverse_shift_m",
        "maximum_ray_shift_rad",
        "maximum_abs_width_ratio_delta",
        "maximum_abs_area_ratio_delta",
    )
    values = (
        (0.5, 0.7, 0.9, 0.99),
        (0.02, 0.05, 0.1, 0.2, 0.4),
        (0.0025, 0.005, 0.01, 0.02),
        (0.0005, 0.001, 0.002, 0.005),
        (0.02, 0.05, 0.1, 0.2, 0.3),
        (0.02, 0.05, 0.1, 0.2, 0.3),
    )
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source = args.samples.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite deterministic-veto evidence: {output}")
    output.mkdir(parents=True)
    samples, rows = load_samples(source)
    train_indices = [index for index, sample in enumerate(samples) if sample.split == "train"]
    validation_indices = [index for index, sample in enumerate(samples) if sample.split == "validation"]
    train_samples = [samples[index] for index in train_indices]
    train_rows = [rows[index] for index in train_indices]
    validation_samples = [samples[index] for index in validation_indices]
    validation_rows = [rows[index] for index in validation_indices]
    evaluated = []
    for policy in policy_grid():
        metric = gate_metrics(train_samples, decisions(train_rows, policy), 0.5)
        evaluated.append((policy, metric))
    candidates = [(policy, metric) for policy, metric in evaluated if metric["deployment_candidate"]]
    if candidates:
        selected_policy, development = max(
            candidates,
            key=lambda item: (
                item[1]["session_macro_transverse_improvement_fraction"],
                item[1]["benefit_precision"],
                -item[1]["harm_apply_fraction"],
                item[1]["applied"],
            ),
        )
        decision = "deterministic_veto_candidate"
        validation = gate_metrics(
            validation_samples, decisions(validation_rows, selected_policy), 0.5
        )
    else:
        selected_policy = None
        decision = "reject_all"
        development = gate_metrics(train_samples, np.zeros(len(train_samples)), 0.5)
        validation = gate_metrics(validation_samples, np.zeros(len(validation_samples)), 0.5)
    top = sorted(
        evaluated,
        key=lambda item: (
            item[1]["deployment_candidate"],
            item[1]["benefit_precision"],
            -item[1]["harm_apply_fraction"],
            item[1]["session_macro_transverse_improvement_fraction"],
        ),
        reverse=True,
    )[:20]
    registry = {
        "schema_version": "aim-stack.corner-repair-deterministic-veto/1",
        "claim": "development-only transparent veto sweep; current validation is diagnostic and sealed test remains closed",
        "source": {"path": str(source), "sha256": sha256(source)},
        "runtime_feature_only": True,
        "future_input": False,
        "truth_used_for_policy_selection_only": True,
        "grid_policies": len(evaluated),
        "selection_gate": {
            "inherited_from_benefit_gate": True,
            "minimum_applied": 10,
            "minimum_benefit_precision": 0.8,
            "maximum_harm_apply_fraction": 0.05,
            "minimum_session_macro_transverse_p95_improvement_fraction": 0.01,
            "per_session_tail_noninferiority": True,
        },
        "decision": decision,
        "selected_policy": selected_policy,
        "development": development,
        "validation_diagnostic": validation,
        "fresh_validation_required": True,
        "sealed_test_access_authorized": False,
        "top_grid_rows": [
            {"policy": policy, "metrics": metric}
            for policy, metric in top
        ],
    }
    path = output / "deterministic_veto_registry.json"
    path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    (output / "retention_manifest.json").write_text(
        json.dumps(
            {
                "classification": "long_term_private_method_development_evidence",
                "deletion_allowed": False,
                "artifacts": {
                    path.name: {
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: registry[key] for key in ("decision", "selected_policy", "development", "validation_diagnostic")}, indent=2))


if __name__ == "__main__":
    main()
