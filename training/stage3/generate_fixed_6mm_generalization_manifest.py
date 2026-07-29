"""Generate a small disjoint Stage-3 manifest for native 6 mm collection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


DISTANCE_BINS = ((1.5, 2.8), (2.8, 4.6), (4.6, 6.2))
SPEED_BANDS = ((0.2, 1.0), (1.0, 2.0), (2.0, 3.0))
OMEGA_BANDS = ((0.5, 5.0), (5.0, 10.0), (10.0, 15.0))
MAX_NOMINAL_RANGE_M = 6.5
MIN_FORWARD_M = 0.75
MAX_ABS_YAW_DEG = 75.0
TRUTH_GIMBAL_LEAD_S = 0.10


def capture_envelope(
    distance_m: float,
    direction_deg: float,
    linear_speed_mps: float,
    linear_span_m: float,
) -> dict[str, float | bool]:
    extent = 0.5 * linear_span_m + TRUTH_GIMBAL_LEAD_S * linear_speed_mps
    heading = math.radians(direction_deg)
    lateral_axis = math.sin(heading)
    forward_axis = math.cos(heading)
    max_range = 0.0
    min_forward = math.inf
    max_yaw = 0.0
    for sign in (-1.0, 1.0):
        lateral = sign * extent * lateral_axis
        forward = distance_m + sign * extent * forward_axis
        max_range = max(max_range, math.hypot(lateral, forward))
        min_forward = min(min_forward, forward)
        yaw = abs(math.degrees(math.atan2(lateral, forward))) if forward > 0.0 else 180.0
        max_yaw = max(max_yaw, yaw)
    accepted = (
        max_range <= MAX_NOMINAL_RANGE_M
        and min_forward >= MIN_FORWARD_M
        and max_yaw <= MAX_ABS_YAW_DEG
    )
    return {
        "accepted": accepted,
        "max_nominal_range_m": max_range,
        "min_forward_m": min_forward,
        "max_abs_yaw_deg": max_yaw,
    }


def _uniform(rng: np.random.Generator, limits: tuple[float, float]) -> float:
    return float(rng.uniform(*limits))


def _base_record(
    dataset_id: str,
    session_id: str,
    seed: int,
    duration_s: float,
) -> dict[str, object]:
    return {
        "schema_version": "stage3-manifest-v1",
        "dataset_id": dataset_id,
        "session_id": session_id,
        "target_number": 3,
        "duration_s": duration_s,
        "seed": seed,
        "camera_profile": "wide_6mm",
        "dual_focal": False,
    }


def generate_records(
    seed: int,
    dataset_id: str,
    spin_count: int = 6,
    combined_count: int = 6,
    duration_s: float = 20.0,
) -> tuple[list[dict[str, object]], int]:
    if spin_count <= 0 or combined_count <= 0 or duration_s <= 0.0:
        raise ValueError("counts and duration must be positive")
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    rejected = 0

    for index in range(spin_count):
        distance_bin = index % len(DISTANCE_BINS)
        omega_band = index % len(OMEGA_BANDS)
        omega = _uniform(rng, OMEGA_BANDS[omega_band])
        if index % 2:
            omega = -omega
        record = _base_record(
            dataset_id,
            f"{dataset_id}-spin-{index:02d}",
            seed + index,
            duration_s,
        )
        record.update(
            {
                "mode": "spin",
                "distance_m": _uniform(rng, DISTANCE_BINS[distance_bin]),
                "initial_yaw_rad": float(rng.uniform(-math.pi, math.pi)),
                "direction_deg": float(rng.uniform(-180.0, 180.0)),
                "linear_speed_mps": 0.0,
                "linear_span_m": 0.0,
                "spin_rad_s": omega,
                "distance_bin": distance_bin,
                "omega_band": omega_band,
            }
        )
        records.append(record)

    for index in range(combined_count):
        distance_bin = index % len(DISTANCE_BINS)
        speed_band = index % len(SPEED_BANDS)
        omega_band = (index + 1) % len(OMEGA_BANDS)
        for _ in range(10_000):
            distance = _uniform(rng, DISTANCE_BINS[distance_bin])
            direction = float(rng.uniform(-180.0, 180.0))
            speed = _uniform(rng, SPEED_BANDS[speed_band])
            span = float(rng.uniform(1.0, 8.0))
            envelope = capture_envelope(distance, direction, speed, span)
            if envelope["accepted"]:
                break
            rejected += 1
        else:
            raise RuntimeError(f"could not sample safe combined session {index}")

        omega = _uniform(rng, OMEGA_BANDS[omega_band])
        if index % 2:
            omega = -omega
        record = _base_record(
            dataset_id,
            f"{dataset_id}-combined-{index:02d}",
            seed + spin_count + index,
            duration_s,
        )
        record.update(
            {
                "mode": "linear_and_spin",
                "distance_m": distance,
                "initial_yaw_rad": float(rng.uniform(-math.pi, math.pi)),
                "direction_deg": direction,
                "linear_speed_mps": speed,
                "linear_span_m": span,
                "spin_rad_s": omega,
                "distance_bin": distance_bin,
                "speed_band": speed_band,
                "omega_band": omega_band,
                "capture_envelope": envelope,
            }
        )
        records.append(record)
    return records, rejected


def write_manifest(records: list[dict[str, object]], output: Path, seed: int, rejected: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "stage3-fixed-6mm-generalization-summary-v1",
        "dataset_id": records[0]["dataset_id"],
        "seed": seed,
        "session_count": len(records),
        "mode_counts": {
            mode: sum(record["mode"] == mode for record in records)
            for mode in ("spin", "linear_and_spin")
        },
        "camera_profile": "wide_6mm",
        "dual_focal": False,
        "combined_rejected_draws": rejected,
        "capture_envelope": {
            "max_nominal_range_m": MAX_NOMINAL_RANGE_M,
            "min_forward_m": MIN_FORWARD_M,
            "max_abs_yaw_deg": MAX_ABS_YAW_DEG,
            "truth_gimbal_lead_s": TRUTH_GIMBAL_LEAD_S,
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--spin-count", type=int, default=6)
    parser.add_argument("--combined-count", type=int, default=6)
    parser.add_argument("--duration-s", type=float, default=20.0)
    args = parser.parse_args()
    records, rejected = generate_records(
        args.seed,
        args.dataset_id,
        spin_count=args.spin_count,
        combined_count=args.combined_count,
        duration_s=args.duration_s,
    )
    write_manifest(records, Path(args.output), args.seed, rejected)


if __name__ == "__main__":
    main()
