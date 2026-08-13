#!/usr/bin/env python3
"""Rank processing methods and compute paired complete-run bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BOOTSTRAP_SAMPLES = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-rows", required=True, type=Path)
    parser.add_argument("--periodic-run-rows", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_rows(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        converted = dict(row)
        for field in ("horizon_s", "distance_m", "scale", "error_p95_deg"):
            converted[field] = float(row[field])
        converted["samples"] = int(float(row["samples"]))
        converted["seed"] = int(float(row["seed"])) if row.get("seed") not in (None, "") else None
        converted["input_tier"] = row.get("input_tier") or "unspecified"
        converted["test_example_hash"] = row.get("test_example_hash") or ""
        converted["dataset_fingerprint"] = row.get("dataset_fingerprint") or ""
        result.append(converted)
    return result


def validate_unique_rows(rows: list[dict]) -> None:
    seen = set()
    for row in rows:
        key = (
            row["dataset_fingerprint"], row["split"], row["fold"], row["horizon_s"],
            row["input_tier"], row["method"], row["motion"], row["distance_m"],
            row["scale"], row["run"], row["seed"], row.get("config", ""),
        )
        if key in seen:
            raise ValueError(f"duplicate method run row: {key}")
        seen.add(key)


def collapse_seeds(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset_fingerprint"], row["split"], row["fold"], row["horizon_s"],
            row["input_tier"], row["method"], row["motion"],
            row["distance_m"], row["scale"], row["run"],
        )
        grouped[key].append(row)
    result = []
    for group in grouped.values():
        if len({item["test_example_hash"] for item in group}) != 1:
            raise ValueError("seed rows disagree on test example set")
        if len({item["samples"] for item in group}) != 1:
            raise ValueError("seed rows disagree on per-run sample count")
        row = dict(group[0])
        row["error_p95_deg"] = float(np.median([item["error_p95_deg"] for item in group]))
        row["seed_count"] = len({item["seed"] for item in group if item["seed"] is not None}) or 1
        row["seed"] = None
        result.append(row)
    return result


def validate_comparability(rows: list[dict]) -> None:
    fingerprints = {row["dataset_fingerprint"] for row in rows}
    if "" in fingerprints or len(fingerprints) != 1:
        raise ValueError(f"method inputs do not share one dataset fingerprint: {fingerprints}")
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["test_example_hash"], row["horizon_s"], row["input_tier"])].append(row)
    for key, group in grouped.items():
        hashes = {row["test_example_hash"] for row in group}
        if "" in hashes or len(hashes) != 1:
            raise ValueError(f"methods disagree on test examples for {key}: {hashes}")
        by_method: dict[str, dict[tuple, int]] = defaultdict(dict)
        for row in group:
            run_key = (row["test_example_hash"], row["run"])
            if run_key in by_method[row["method"]]:
                raise ValueError(f"duplicate collapsed run row for {key}, {row['method']}, {run_key}")
            by_method[row["method"]][run_key] = row["samples"]
        reference_name = sorted(by_method)[0]
        reference = by_method[reference_name]
        for method, coverage in by_method.items():
            if coverage != reference:
                raise ValueError(
                    f"method coverage mismatch for {key}: {reference_name} vs {method}"
                )


def ranking(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["horizon_s"], row["input_tier"], row["method"])].append(row)
    result = []
    for (split, horizon, input_tier, method), group in grouped.items():
        condition_groups: dict[tuple, list[float]] = defaultdict(list)
        for row in group:
            condition_groups[(row["motion"], row["distance_m"], row["scale"])].append(row["error_p95_deg"])
        condition_values = [float(np.mean(values)) for values in condition_groups.values()]
        result.append(
            {
                "split": split,
                "horizon_s": horizon,
                "input_tier": input_tier,
                "method": method,
                "runs": len(group),
                "conditions": len(condition_values),
                "run_equal_p95_deg": float(np.mean([row["error_p95_deg"] for row in group])),
                "condition_equal_p95_deg": float(np.mean(condition_values)),
                "worst_condition_p95_deg": float(np.max(condition_values)),
            }
        )
    return sorted(
        result,
        key=lambda row: (
            row["split"], row["horizon_s"], row["input_tier"], row["condition_equal_p95_deg"]
        ),
    )


def paired_bootstrap(rows: list[dict], first: str, second: str, split: str, horizon: float, input_tier: str) -> dict | None:
    selected = [row for row in rows if row["split"] == split and row["horizon_s"] == horizon and row["input_tier"] == input_tier]
    by_method = defaultdict(dict)
    for row in selected:
        key = (row["test_example_hash"], row["run"])
        by_method[row["method"]][key] = row
    if set(by_method[first]) != set(by_method[second]):
        raise ValueError(
            f"paired methods have different complete-run keys: {input_tier}/{split}/{horizon}, {first} vs {second}"
        )
    keys = sorted(by_method[first])
    if len(keys) < 5:
        return None
    condition_keys: dict[tuple, list[tuple]] = defaultdict(list)
    for key in keys:
        row = by_method[first][key]
        condition_keys[(row["motion"], row["distance_m"], row["scale"])].append(key)
    conditions = sorted(condition_keys)

    def condition_equal_difference(sampled_conditions: list[tuple], rng: np.random.RandomState | None) -> float:
        differences = []
        for condition in sampled_conditions:
            members = condition_keys[condition]
            if rng is not None:
                members = [members[index] for index in rng.randint(0, len(members), len(members))]
            candidate = float(np.mean([by_method[first][key]["error_p95_deg"] for key in members]))
            reference = float(np.mean([by_method[second][key]["error_p95_deg"] for key in members]))
            differences.append((reference - candidate) / max(reference, 1e-12))
        return float(np.mean(differences))

    rng = np.random.RandomState(20260809)
    improvements = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        sampled = [conditions[value] for value in rng.randint(0, len(conditions), len(conditions))]
        improvements[index] = condition_equal_difference(sampled, rng)
    observed = condition_equal_difference(conditions, None)
    return {
        "split": split,
        "horizon_s": horizon,
        "input_tier": input_tier,
        "candidate": first,
        "reference": second,
        "paired_runs": len(keys),
        "paired_conditions": len(conditions),
        "improvement_fraction": observed,
        "bootstrap_ci95_low": float(np.percentile(improvements, 2.5)),
        "bootstrap_ci95_high": float(np.percentile(improvements, 97.5)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    sources = [args.baseline_run_rows.resolve()]
    rows = normalized_rows(read_csv(sources[0]))
    if args.periodic_run_rows:
        sources.append(args.periodic_run_rows.resolve())
        rows.extend(normalized_rows(read_csv(sources[-1])))
    validate_unique_rows(rows)
    collapsed = collapse_seeds(rows)
    validate_comparability(collapsed)
    ranks = ranking(collapsed)
    comparisons = []
    for split, horizon, input_tier in sorted({(row["split"], row["horizon_s"], row["input_tier"]) for row in ranks}):
        group = [row for row in ranks if row["split"] == split and row["horizon_s"] == horizon and row["input_tier"] == input_tier]
        if len(group) >= 2:
            comparison = paired_bootstrap(collapsed, group[0]["method"], group[1]["method"], split, horizon, input_tier)
            if comparison:
                comparisons.append(comparison)
        for candidate, reference in (
            ("periodic_ukf", "periodic_ekf"),
            ("periodic_ukf_shared", "periodic_ekf_shared"),
            ("ukf_coordinated_turn", "ekf_coordinated_turn"),
            ("ukf_coordinated_turn_shared", "ekf_coordinated_turn_shared"),
            ("ridge_uv_residual", "kalman_cv"),
            ("ridge_uv_residual", "mlp_uv_residual"),
            ("ridge_uv_yaw_residual", "periodic_ekf_shared"),
            ("ridge_uv_yaw_residual", "periodic_ukf_shared"),
            ("ridge_uv_yaw_residual", "mlp_uv_yaw_residual"),
            ("ridge_residual", "mlp_residual"),
        ):
            methods = {row["method"] for row in group}
            if candidate in methods and reference in methods:
                comparison = paired_bootstrap(collapsed, candidate, reference, split, horizon, input_tier)
                if comparison:
                    comparisons.append(comparison)
    comparisons = list(
        {
            (
                row["split"], row["horizon_s"], row["input_tier"],
                row["candidate"], row["reference"],
            ): row
            for row in comparisons
        }.values()
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "method_ranking.csv", ranks)
    write_csv(output / "paired_run_bootstrap.csv", comparisons)
    summary = {
        "schema_version": 1,
        "kind": "trajectory_processing_method_selection_summary",
        "bootstrap_unit": "hierarchical condition then complete run; overlapping windows are never resampled independently",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "oracle_identity_upper_bound": True,
        "sources": {str(path): sha256(path) for path in sources},
        "rankings": ranks,
        "paired_comparisons": comparisons,
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    }
    (output / "method_selection_evidence.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "retention_manifest.json").write_text(json.dumps({"classification": "long_term_private_evidence", "deletion_allowed": False, "sources": summary["sources"], "artifacts": ["method_ranking.csv", "paired_run_bootstrap.csv", "method_selection_evidence.json"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
