#!/usr/bin/env python3
"""Fail-closed truth audit for one controlled Stage3 collection run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


STOCK_RADIUS_EVEN_M = 0.21131764
STOCK_RADIUS_ODD_M = 0.21176702


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--radial-scale", required=True, type=float)
    parser.add_argument("--spin-deg-s", required=True, type=float)
    parser.add_argument("--linear-speed-mps", required=True, type=float)
    parser.add_argument("--linear-span-m", required=True, type=float)
    parser.add_argument("--radius-tolerance-m", type=float, default=0.002)
    parser.add_argument("--spin-tolerance-deg-s", type=float, default=0.15)
    parser.add_argument("--speed-tolerance-mps", type=float, default=0.05)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def select_target(row: dict) -> dict | None:
    exposure = row.get("exposure_state") or {}
    camera = exposure.get("camera_position_world_m")
    targets = [
        target
        for target in (row.get("ground_truth") or {}).get("targets", [])
        if target.get("armor_label") == 3 and target.get("armor_count") == 4
    ]
    if not camera or not targets:
        return None
    return max(targets, key=lambda target: euclidean(target["world_position_m"], camera))


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> None:
    args = parse_args()
    truth_path = args.truth.resolve()
    rows = []
    selected = []
    camera_distances = []
    gimbal_distances = []
    with truth_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            target = select_target(row)
            if target is None:
                continue
            selected.append(target)
            exposure = row["exposure_state"]
            camera_distances.append(
                euclidean(target["world_position_m"], exposure["camera_position_world_m"])
            )
            gimbal_distances.append(
                euclidean(target["world_position_m"], exposure["gimbal_position_world_m"])
            )
    if not rows or not selected:
        raise RuntimeError("truth file has no selectable complete target-3 frames")
    radii_even = [float(target["radius_even_m"]) for target in selected]
    radii_odd = [float(target["radius_odd_m"]) for target in selected]
    spin = [math.degrees(float(target.get("vyaw_rad_s", 0.0))) for target in selected]
    speed = [
        math.sqrt(sum(float(value) ** 2 for value in target.get("world_velocity_mps", [0.0, 0.0, 0.0])))
        for target in selected
    ]
    positions = [target["world_position_m"] for target in selected]
    axis_span = [
        max(float(position[axis]) for position in positions)
        - min(float(position[axis]) for position in positions)
        for axis in range(3)
    ]
    measured_even = median(radii_even)
    measured_odd = median(radii_odd)
    measured_spin = median(spin)
    measured_speed = median(speed)
    expected_even = STOCK_RADIUS_EVEN_M * args.radial_scale
    expected_odd = STOCK_RADIUS_ODD_M * args.radial_scale
    checks = {
        "target_selection_coverage": len(selected) / len(rows) >= 0.99,
        "radius_even_matches_requested_scale": abs(measured_even - expected_even)
        <= args.radius_tolerance_m,
        "radius_odd_matches_requested_scale": abs(measured_odd - expected_odd)
        <= args.radius_tolerance_m,
        "spin_matches_request": abs(measured_spin - args.spin_deg_s)
        <= args.spin_tolerance_deg_s,
        "linear_speed_matches_request": abs(measured_speed - args.linear_speed_mps)
        <= args.speed_tolerance_mps,
    }
    if args.linear_speed_mps > 0.0 and args.linear_span_m > 0.0:
        checks["linear_span_reached"] = max(axis_span) >= 0.90 * args.linear_span_m
    else:
        checks["linear_span_reached"] = max(axis_span) <= 0.02
    summary = {
        "schema_version": 1,
        "kind": "stage3_truth_motion_audit",
        "truth_path": str(truth_path),
        "truth_sha256": sha256_file(truth_path),
        "truth_rows": len(rows),
        "selected_target_rows": len(selected),
        "requested": {
            "radial_scale": args.radial_scale,
            "spin_deg_s": args.spin_deg_s,
            "linear_speed_mps": args.linear_speed_mps,
            "linear_span_m": args.linear_span_m,
        },
        "measured": {
            "radius_even_m_p50": measured_even,
            "radius_odd_m_p50": measured_odd,
            "normalized_radius_even_m_p50": measured_even / args.radial_scale,
            "normalized_radius_odd_m_p50": measured_odd / args.radial_scale,
            "spin_deg_s_p50": measured_spin,
            "linear_speed_mps_p50": measured_speed,
            "world_axis_span_m": axis_span,
            "target_distance_camera_m_p50": median(camera_distances),
            "target_distance_gimbal_m_p50": median(gimbal_distances),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.resolve().write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
