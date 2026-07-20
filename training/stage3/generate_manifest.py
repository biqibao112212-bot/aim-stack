"""Generate the deterministic 360-session Stage-3 collection manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


DISTANCE_BINS = ((1.0, 3.0), (3.0, 5.5), (5.5, 8.0))
SPEED_BANDS = ((0.1, 1.0), (1.0, 2.0), (2.0, 3.0))
OMEGA_BANDS = ((0.2, 5.0), (5.0, 10.0), (10.0, 15.0))
MODE_COUNTS = (("stationary", 36), ("linear", 90), ("spin", 90), ("linear_and_spin", 144))


def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def generate(seed: int, output: Path, dataset_id: str) -> None:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    ordinal = 0
    for mode, count in MODE_COUNTS:
        for local_index in range(count):
            distance_bin = local_index % len(DISTANCE_BINS)
            distance = _uniform(rng, *DISTANCE_BINS[distance_bin])
            direction_sector = local_index % 8
            direction = direction_sector * 45.0 + _uniform(rng, -22.5, 22.5)
            initial_yaw = _uniform(rng, -math.pi, math.pi)
            speed = 0.0
            omega = 0.0
            if mode in ("linear", "linear_and_spin"):
                speed_band = (local_index // len(DISTANCE_BINS)) % len(SPEED_BANDS)
                speed = _uniform(rng, *SPEED_BANDS[speed_band])
            if mode in ("spin", "linear_and_spin"):
                omega_band = (local_index // len(DISTANCE_BINS)) % len(OMEGA_BANDS)
                omega = _uniform(rng, *OMEGA_BANDS[omega_band])
                if local_index % 2:
                    omega = -omega
            record = {
                "schema_version": "stage3-manifest-v1",
                "dataset_id": dataset_id,
                "session_id": f"{dataset_id}-{ordinal:04d}",
                "target_number": 3,
                "mode": mode,
                "distance_m": distance,
                "initial_yaw_rad": initial_yaw,
                "direction_deg": direction,
                "linear_speed_mps": speed,
                "linear_span_m": 8.0 if mode in ("linear", "linear_and_spin") else 0.0,
                "spin_rad_s": omega,
                "duration_s": 30.0,
                "seed": int(seed + ordinal),
                "distance_bin": distance_bin,
                "direction_sector": direction_sector,
            }
            records.append(record)
            ordinal += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "dataset_id": dataset_id,
        "seed": seed,
        "session_count": len(records),
        "mode_counts": {mode: sum(record["mode"] == mode for record in records) for mode, _ in MODE_COUNTS},
        "distance_bin_counts": {str(index): sum(record["distance_bin"] == index for record in records) for index in range(3)},
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-id", default="stage3-20260719-v1")
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    generate(args.seed, Path(args.output), args.dataset_id)


if __name__ == "__main__":
    main()
