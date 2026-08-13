#!/usr/bin/env python3
"""Aggregate processor and association evidence into the next-stage decision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), quantile))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline_dir = root / "method-analysis" / "cv-ridge-replay-v2-hold-safe"
    learned_dir = root / "method-analysis" / "learned-association-threshold-sweep-v1"
    nested_dir = root / "method-analysis" / "cyclic-association-nested-v1"
    cyclic_dir = root / "method-analysis" / "cv-ridge-cyclic-e2e-nested-v2-confidence"
    baseline = read_json(baseline_dir / "replay_summary.json")
    learned = read_json(learned_dir / "threshold_sweep_summary.json")
    learned_rows = read_csv(learned_dir / "association_threshold_summary.csv")
    learned_candidate = max(
        (row for row in learned_rows if row["variant"] != "baseline_cv_gate25"),
        key=lambda row: float(row["global_mapping_accuracy_mean"]),
    )
    nested = read_json(nested_dir / "nested_cyclic_summary.json")
    cyclic = read_json(cyclic_dir / "cyclic_processor_summary.json")
    predictions = [
        row
        for row in read_csv(cyclic_dir / "replay_predictions.csv")
        if abs(float(row["horizon_s"]) - 0.1) < 1e-9
    ]
    selective = []
    for threshold in (0.0, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        rows = [
            row
            for row in predictions
            if float(row["association_confidence"]) >= threshold
        ]
        errors = [float(row["chain_error_deg"]) for row in rows]
        selective.append(
            {
                "association_confidence_threshold": threshold,
                "coverage": len(rows) / len(predictions),
                "identity_correct_rate": float(
                    np.mean([row["identity_correct"] == "True" for row in rows])
                ),
                "chain_error_p50_deg": percentile(errors, 50),
                "chain_error_p90_deg": percentile(errors, 90),
                "chain_error_p95_deg": percentile(errors, 95),
            }
        )
    write_csv(output / "cyclic_selective_metrics_100ms.csv", selective)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    labels = ["baseline CV", "nested cyclic"]
    accuracy = [
        float(nested["baseline_accuracy_mean"]),
        float(nested["cyclic_accuracy_mean"]),
    ]
    purity = [
        float(nested["baseline_track_purity_mean"]),
        float(nested["cyclic_track_purity_mean"]),
    ]
    x = np.arange(2)
    axes[0].bar(x - 0.18, accuracy, 0.36, label="mapping accuracy")
    axes[0].bar(x + 0.18, purity, 0.36, label="track purity")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Observation-only identity")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    coverage = [float(row["coverage"]) for row in selective]
    p95 = [float(row["chain_error_p95_deg"]) for row in selective]
    identity = [float(row["identity_correct_rate"]) for row in selective]
    axes[1].plot(coverage, p95, marker="o", label="100 ms chain P95 (deg)")
    identity_axis = axes[1].twinx()
    identity_axis.plot(coverage, identity, marker="s", color="#D55E00", label="identity accuracy")
    axes[1].set_xlabel("retained prediction fraction")
    axes[1].set_ylabel("angular error P95 (deg)")
    identity_axis.set_ylabel("identity accuracy")
    axes[1].set_title("Single-hypothesis confidence rejection")
    axes[1].grid(alpha=0.25)
    lines = axes[1].lines + identity_axis.lines
    axes[1].legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    fig.savefig(output / "processor_association_decision.png", dpi=180)
    fig.savefig(output / "processor_association_decision.svg")
    plt.close(fig)
    evidence = {
        "schema_version": 1,
        "kind": "processor_and_association_stage_decision",
        "processor_decision": "accept offline causal CV plus u/v Ridge with hold-safe fallback",
        "association_decision": "reject independent learned pair cost; retain cyclic topology only as the foundation for a multi-hypothesis C4 belief tracker",
        "online_claim": False,
        "baseline_processor": {
            "identity_correct_error_p95_deg": baseline["identity_correct_chain_error_deg"]["p95"],
            "identity_correct_error_p99_deg": baseline["identity_correct_chain_error_deg"]["p99"],
            "processor_latency_p99_us": baseline["processor_predict_latency_us"]["p99"],
        },
        "learned_pair_best_accuracy": float(
            learned_candidate["global_mapping_accuracy_mean"]
        ),
        "nested_cyclic": {
            "accuracy": nested["cyclic_accuracy_mean"],
            "baseline_accuracy": nested["baseline_accuracy_mean"],
            "condition_equal_improvement_fraction": nested["paired_bootstrap"]["condition_equal_improvement_fraction"],
            "bootstrap_ci95": [
                nested["paired_bootstrap"]["bootstrap_ci95_low"],
                nested["paired_bootstrap"]["bootstrap_ci95_high"],
            ],
            "associated_fraction": nested["cyclic_associated_fraction"],
            "track_purity": nested["cyclic_track_purity_mean"],
        },
        "cyclic_end_to_end": {
            "identity_correct_prediction_rate": cyclic["identity_correct_prediction_rate"],
            "chain_error_p50_deg": cyclic["all_chain_error_deg"]["p50"],
            "chain_error_p95_deg": cyclic["all_chain_error_deg"]["p95"],
            "identity_correct_error_p95_deg": cyclic["identity_correct_chain_error_deg"]["p95"],
            "processor_latency_p99_us": cyclic["processor_predict_latency_us"]["p99"],
            "association_latency_p99_us": cyclic["association_event_latency_us"]["p99"],
        },
        "selective_100ms": selective,
        "next_task": "implement a multi-hypothesis C4 identity belief tracker that emits top-k/permutation probabilities and lets fire control reject unresolved identity",
    }
    (output / "processor_association_decision.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Processor and association stage decision

## Accepted processor

- Keep causal constant-velocity plus u/v-only Ridge residual correction.
- Use hold, not short-history Kalman, after startup or reacquisition. With correct identity the hold-safe replay is {baseline['identity_correct_chain_error_deg']['p95']:.3f} deg P95 and {baseline['identity_correct_chain_error_deg']['p99']:.3f} deg P99 across all horizons; offline Python processor latency is {baseline['processor_predict_latency_us']['p99']:.1f} us P99.

## Association evidence

- Independent learned pair cost is rejected; its best threshold sweep does not beat the {nested['baseline_accuracy_mean']:.3f} baseline.
- Nested-repeat cyclic segment association improves mean mapping accuracy to {nested['cyclic_accuracy_mean']:.3f}, a condition-equal gain of {nested['paired_bootstrap']['condition_equal_improvement_fraction']:.3f} with 95% CI [{nested['paired_bootstrap']['bootstrap_ci95_low']:.3f}, {nested['paired_bootstrap']['bootstrap_ci95_high']:.3f}].
- The remaining hard-ID failures are catastrophic. The cyclic chain has {cyclic['all_chain_error_deg']['p50']:.3f} deg P50 but {cyclic['all_chain_error_deg']['p95']:.3f} deg P95, while identity-correct P95 is only {cyclic['identity_correct_chain_error_deg']['p95']:.3f} deg.
- Confidence thresholding is insufficient: at threshold 0.95 it retains {selective[-1]['coverage']:.3f} of 100 ms predictions but identity accuracy is only {selective[-1]['identity_correct_rate']:.3f} and chain P95 remains {selective[-1]['chain_error_p95_deg']:.3f} deg.

## Decision

Do not tune Ridge further and do not deploy a single hard identity. The next component is a multi-hypothesis C4 belief tracker over cyclic permutations/phase. It must emit identity probabilities or top-k hypotheses so unresolved identity can remain a mixture or be rejected. Score top-k coverage, NLL/Brier calibration, end-to-end mixture position error, reacquisition and P99 latency on the retained cache before online integration.
"""
    (output / "PROCESSOR_ASSOCIATION_DECISION.md").write_text(report, encoding="utf-8")
    source_paths = {
        "baseline_replay": baseline_dir / "replay_summary.json",
        "learned_pair_sweep": learned_dir / "threshold_sweep_summary.json",
        "nested_cyclic": nested_dir / "nested_cyclic_summary.json",
        "cyclic_replay": cyclic_dir / "cyclic_processor_summary.json",
        "cyclic_predictions": cyclic_dir / "replay_predictions.csv",
    }
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "retention_manifest.json"
    }
    (output / "retention_manifest.json").write_text(
        json.dumps(
            {
                "classification": "long_term_private_evidence",
                "deletion_allowed": False,
                "sources": {
                    name: {"path": str(path), "sha256": sha256_file(path)}
                    for name, path in source_paths.items()
                },
                "artifacts": artifacts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
