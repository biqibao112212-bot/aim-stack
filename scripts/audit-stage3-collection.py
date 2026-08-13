#!/usr/bin/env python3
"""Create a durable root-level acceptance record for a truth-gated collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--expected-runs", type=int, default=60)
    parser.add_argument("--minimum-fps", type=float, default=100.0)
    parser.add_argument("--target-fps", type=float, default=100.0)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    collection_path = root / "collection_manifest.json"
    if not collection_path.exists():
        raise SystemExit(f"collection manifest is absent: {collection_path}")
    collection = read_json(collection_path)
    run_dirs = sorted(path for path in root.iterdir() if path.is_dir() and (path / "collection_run_manifest.json").exists())
    records = []
    failures = []
    by_scale = defaultdict(lambda: {"even": [], "odd": []})
    by_distance = defaultdict(lambda: {"camera": [], "gimbal": []})
    for run_dir in run_dirs:
        paths = {
            "manifest": run_dir / "collection_run_manifest.json",
            "audit": run_dir / "truth_motion_audit.json",
            "performance": run_dir / "summary.json",
        }
        if not all(path.exists() for path in paths.values()):
            failures.append(f"incomplete:{run_dir.name}")
            continue
        manifest, audit, performance = (read_json(paths[name]) for name in ("manifest", "audit", "performance"))
        fps = float(performance["processed_fps"])
        if not audit.get("passed"):
            failures.append(f"truth_audit:{run_dir.name}")
        if not math.isfinite(fps) or fps < args.minimum_fps:
            failures.append(f"fps:{run_dir.name}:{fps}")
        requested_scale = float(audit["requested"]["radial_scale"])
        requested_distance = float(manifest["requested_distance_m"])
        by_scale[requested_scale]["even"].append(float(audit["measured"]["radius_even_m_p50"]))
        by_scale[requested_scale]["odd"].append(float(audit["measured"]["radius_odd_m_p50"]))
        by_distance[requested_distance]["camera"].append(float(audit["measured"]["target_distance_camera_m_p50"]))
        by_distance[requested_distance]["gimbal"].append(float(audit["measured"]["target_distance_gimbal_m_p50"]))
        records.append(
            {
                "run": run_dir.name,
                "fps": fps,
                "fps_target_met": fps >= args.target_fps,
                "truth_passed": bool(audit.get("passed")),
                "spin_error_deg_s": abs(float(audit["measured"]["spin_deg_s_p50"]) - float(audit["requested"]["spin_deg_s"])),
                "linear_speed_error_mps": abs(float(audit["measured"]["linear_speed_mps_p50"]) - float(audit["requested"]["linear_speed_mps"])),
                "files": {name: sha256(path) for name, path in paths.items()},
            }
        )
    if len(run_dirs) != args.expected_runs:
        failures.append(f"run_count:{len(run_dirs)}!={args.expected_runs}")
    fps_values = [record["fps"] for record in records]
    summary = {
        "schema_version": 1,
        "kind": "stage3_truth_gated_collection_acceptance",
        "root": str(root),
        "collection_manifest_sha256": sha256(collection_path),
        "requested": {
            "motion_mode": collection.get("motion_mode"),
            "spin_deg_s": collection.get("spin_deg_s"),
            "linear_speed_mps": collection.get("linear_speed_mps"),
            "linear_span_m": collection.get("linear_span_m"),
            "radial_scales": collection.get("grid", {}).get("radial_scales"),
            "distances_m": collection.get("grid", {}).get("distances_m"),
            "repeats": collection.get("grid", {}).get("repeats"),
        },
        "acceptance": {
            "expected_runs": args.expected_runs,
            "completed_runs": len(records),
            "minimum_fps": args.minimum_fps,
            "target_fps": args.target_fps,
            "runs_below_target_fps": sum(not record["fps_target_met"] for record in records),
            "failures": failures,
            "passed": not failures,
        },
        "processed_fps": quantiles(fps_values) if fps_values else None,
        "max_spin_error_deg_s": max((record["spin_error_deg_s"] for record in records), default=None),
        "max_linear_speed_error_mps": max((record["linear_speed_error_mps"] for record in records), default=None),
        "measured_radius_m": {str(scale): {plate: quantiles(values) for plate, values in plates.items()} for scale, plates in sorted(by_scale.items())},
        "measured_distance_m": {str(distance): {reference: quantiles(values) for reference, values in references.items()} for distance, references in sorted(by_distance.items())},
        "runs": records,
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    }
    output = root / "collection_acceptance.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    retention = {
        "schema_version": 1,
        "kind": "stage3_collection_retention_manifest",
        "classification": "protected_raw_research_data",
        "deletion_allowed": False,
        "collection_acceptance_sha256": sha256(output),
        "raw_runs": len(records),
    }
    (root / "retention_manifest.json").write_text(json.dumps(retention, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["acceptance"], ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
