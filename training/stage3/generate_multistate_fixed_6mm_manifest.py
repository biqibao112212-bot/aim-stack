"""Generate fixed-6 mm sessions with many independent motion states per session."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .generate_fixed_6mm_generalization_manifest import capture_envelope


SCHEMA_VERSION = "stage3-multistate-manifest-v2"
DISTANCE_RANGE_M = (1.5, 6.2)
SPEED_RANGE_MPS = (0.2, 3.0)
OMEGA_RANGE_RAD_S = (0.5, 15.0)


def _latin_hypercube(
    rng: np.random.Generator, count: int, limits: tuple[float, float]
) -> np.ndarray:
    if count <= 0:
        raise ValueError("Latin-hypercube count must be positive")
    unit = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
    rng.shuffle(unit)
    return limits[0] + unit * (limits[1] - limits[0])


def _maximum_safe_span(
    distance_m: float, direction_deg: float, speed_mps: float
) -> float:
    if not bool(capture_envelope(distance_m, direction_deg, speed_mps, 0.0)["accepted"]):
        return 0.0
    low, high = 0.0, 8.0
    if bool(capture_envelope(distance_m, direction_deg, speed_mps, high)["accepted"]):
        return high
    for _ in range(48):
        middle = 0.5 * (low + high)
        if bool(capture_envelope(distance_m, direction_deg, speed_mps, middle)["accepted"]):
            low = middle
        else:
            high = middle
    return low


def _signed_omega(
    rng: np.random.Generator, count: int
) -> np.ndarray:
    magnitude = _latin_hypercube(rng, count, OMEGA_RANGE_RAD_S)
    signs = np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
    rng.shuffle(signs)
    return magnitude * signs


def _stationary_segment(index: int, duration_s: float) -> dict[str, object]:
    return {
        "segment_index": index,
        "mode": "stationary",
        "direction_deg": 0.0,
        "linear_speed_mps": 0.0,
        "linear_span_m": 0.0,
        "spin_rad_s": 0.0,
        "duration_s": duration_s,
    }


def _base_record(
    *, dataset_id: str, session_id: str, seed: int, distance_m: float,
    initial_yaw_rad: float, family_mode: str, segments: list[dict[str, object]],
) -> dict[str, object]:
    first_active = next(segment for segment in segments if segment["mode"] != "stationary")
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "session_id": session_id,
        "target_number": 3,
        "duration_s": float(sum(float(segment["duration_s"]) for segment in segments)),
        "seed": seed,
        "camera_profile": "wide_6mm",
        "dual_focal": False,
        "motion_family": "rotation" if family_mode == "spin" else "combined",
        # Root motion fields name the family and retain a deterministic summary;
        # the v2 builder obtains the actual state from the ACK-bound segment.
        "mode": family_mode,
        "distance_m": distance_m,
        "initial_yaw_rad": initial_yaw_rad,
        "direction_deg": first_active["direction_deg"],
        "linear_speed_mps": first_active["linear_speed_mps"],
        "linear_span_m": first_active["linear_span_m"],
        "spin_rad_s": first_active["spin_rad_s"],
        "segments": segments,
        "segment_policy": {
            "boundary": "scene-control ACK applied_timestamp_ns; half-open [start,end)",
            "admission": "all retained history and all future queries inside one segment",
            "full_truth_constant_motion_required": True,
        },
    }


def generate_records(
    seed: int,
    dataset_id: str,
    *,
    spin_count: int = 12,
    combined_count: int = 12,
    segments_per_session: int = 12,
    segment_duration_s: float = 3.0,
) -> tuple[list[dict[str, object]], int]:
    if spin_count <= 0 or combined_count <= 0:
        raise ValueError("spin and combined session counts must be positive")
    if segments_per_session < 3:
        raise ValueError("at least three segments per session are required")
    if not math.isfinite(segment_duration_s) or segment_duration_s <= 0.0:
        raise ValueError("segment duration must be finite and positive")

    rng = np.random.default_rng(seed)
    session_count = spin_count + combined_count
    distances = _latin_hypercube(rng, session_count, DISTANCE_RANGE_M)
    initial_yaws = _latin_hypercube(rng, session_count, (-math.pi, math.pi))
    active_per_session = segments_per_session - 1
    spin_omegas = _signed_omega(rng, spin_count * active_per_session)
    combined_omegas = _signed_omega(rng, combined_count * active_per_session)
    combined_speeds = _latin_hypercube(
        rng, combined_count * active_per_session, SPEED_RANGE_MPS
    )
    combined_directions = _latin_hypercube(
        rng, combined_count * active_per_session, (-180.0, 180.0)
    )
    combined_span_fractions = _latin_hypercube(
        rng, combined_count * active_per_session, (0.0, 1.0)
    )

    records: list[dict[str, object]] = []
    rejected_directions = 0
    spin_cursor = 0
    for session_index in range(spin_count):
        stationary_index = int(rng.integers(1, segments_per_session - 1))
        segments: list[dict[str, object]] = []
        for segment_index in range(segments_per_session):
            if segment_index == stationary_index:
                segments.append(_stationary_segment(segment_index, segment_duration_s))
                continue
            segments.append({
                "segment_index": segment_index,
                "mode": "spin",
                "direction_deg": 0.0,
                "linear_speed_mps": 0.0,
                "linear_span_m": 0.0,
                "spin_rad_s": float(spin_omegas[spin_cursor]),
                "duration_s": segment_duration_s,
            })
            spin_cursor += 1
        records.append(_base_record(
            dataset_id=dataset_id,
            session_id=f"{dataset_id}-spin-{session_index:02d}",
            seed=seed + session_index,
            distance_m=float(distances[session_index]),
            initial_yaw_rad=float(initial_yaws[session_index]),
            family_mode="spin",
            segments=segments,
        ))

    combined_cursor = 0
    for session_index in range(combined_count):
        global_session = spin_count + session_index
        distance = float(distances[global_session])
        stationary_index = int(rng.integers(1, segments_per_session - 1))
        segments = []
        for segment_index in range(segments_per_session):
            if segment_index == stationary_index:
                segments.append(_stationary_segment(segment_index, segment_duration_s))
                continue
            speed = float(combined_speeds[combined_cursor])
            direction = float(combined_directions[combined_cursor])
            best_lateral_span = max(
                _maximum_safe_span(distance, -90.0, speed),
                _maximum_safe_span(distance, 90.0, speed),
            )
            minimum_span = min(max(1.5, speed), 0.85 * best_lateral_span)
            minimum_span = max(0.5, minimum_span)
            maximum_span = _maximum_safe_span(distance, direction, speed)
            while maximum_span < minimum_span:
                rejected_directions += 1
                direction = float(rng.uniform(-180.0, 180.0))
                maximum_span = _maximum_safe_span(distance, direction, speed)
            fraction = float(combined_span_fractions[combined_cursor])
            span = minimum_span + fraction * (maximum_span - minimum_span)
            envelope = capture_envelope(distance, direction, speed, span)
            if not bool(envelope["accepted"]):
                raise AssertionError("accepted combined draw failed the capture envelope")
            segments.append({
                "segment_index": segment_index,
                "mode": "linear_and_spin",
                "direction_deg": direction,
                "linear_speed_mps": speed,
                "linear_span_m": span,
                "spin_rad_s": float(combined_omegas[combined_cursor]),
                "duration_s": segment_duration_s,
                "capture_envelope": envelope,
            })
            combined_cursor += 1
        records.append(_base_record(
            dataset_id=dataset_id,
            session_id=f"{dataset_id}-combined-{session_index:02d}",
            seed=seed + global_session,
            distance_m=distance,
            initial_yaw_rad=float(initial_yaws[global_session]),
            family_mode="linear_and_spin",
            segments=segments,
        ))
    return records, rejected_directions


def write_manifest(
    records: list[dict[str, object]], output: Path, seed: int,
    rejected_directions: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    segment_count = sum(len(record["segments"]) for record in records)
    summary = {
        "schema_version": "stage3-multistate-fixed-6mm-summary-v1",
        "dataset_id": records[0]["dataset_id"],
        "seed": seed,
        "session_count": len(records),
        "segment_count": segment_count,
        "stationary_segment_fraction": len(records) / segment_count,
        "mode_counts": {
            mode: sum(record["mode"] == mode for record in records)
            for mode in ("spin", "linear_and_spin")
        },
        "camera_profile": "wide_6mm",
        "dual_focal": False,
        "rejected_direction_draws": rejected_directions,
        "sampling": "continuous Latin-hypercube values with safe-envelope direction repair",
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--spin-count", type=int, default=12)
    parser.add_argument("--combined-count", type=int, default=12)
    parser.add_argument("--segments-per-session", type=int, default=12)
    parser.add_argument("--segment-duration-s", type=float, default=3.0)
    args = parser.parse_args()
    records, rejected = generate_records(
        args.seed,
        args.dataset_id,
        spin_count=args.spin_count,
        combined_count=args.combined_count,
        segments_per_session=args.segments_per_session,
        segment_duration_s=args.segment_duration_s,
    )
    write_manifest(records, Path(args.output), args.seed, rejected)


if __name__ == "__main__":
    main()
