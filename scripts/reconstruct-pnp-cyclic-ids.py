#!/usr/bin/env python3
"""Replay an observation-set JSONL and add a derived cyclic plate id.

The raw observation set remains authoritative.  This pass deliberately does
not read tracker_id/tracked_armor/jump_flag.  It uses only the measured armor
yaw, exposure time, and the known constant spin direction/rate.  The first
visible target observation is canonical id 0; subsequent unmatched plates are
allocated in cyclic order 0,1,2,3.  Every assignment is annotated so a later
policy can be replayed without recollecting images.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def wrap180(angle_deg: float) -> float:
    return (angle_deg + 90.0) % 180.0 - 90.0


def angular_error(observed: float, expected: float) -> float:
    return abs(wrap180(observed - expected))


def solve_assignment(
    observations: Sequence[dict],
    slots: Dict[int, dict],
    timestamp_ns: int,
    spin_deg_per_sec: float,
    threshold_deg: float,
) -> Tuple[Dict[int, int], Dict[int, float]]:
    """Find a minimum-cost one-to-one match against the four prior slots."""
    candidates: List[Tuple[int, float]] = []
    for slot_id, state in slots.items():
        dt = max(0.0, (timestamp_ns - int(state["timestamp_ns"])) / 1e9)
        expected = float(state["yaw_deg"]) + spin_deg_per_sec * dt
        for obs_index, observation in enumerate(observations):
            yaw = observation.get("armor_yaw_absolute_deg")
            if finite(yaw):
                candidates.append((slot_id, angular_error(float(yaw), expected)))

    best: Optional[Tuple[float, Dict[int, int], Dict[int, float]]] = None
    slot_ids = sorted(slots)
    for count in range(1, min(len(slot_ids), len(observations)) + 1):
        for selected_slots in itertools.combinations(slot_ids, count):
            for selected_obs in itertools.permutations(range(len(observations)), count):
                assignment: Dict[int, int] = {}
                costs: Dict[int, float] = {}
                total = 0.0
                valid = True
                for slot_id, obs_index in zip(selected_slots, selected_obs):
                    yaw = observations[obs_index].get("armor_yaw_absolute_deg")
                    state = slots[slot_id]
                    dt = max(0.0, (timestamp_ns - int(state["timestamp_ns"])) / 1e9)
                    expected = float(state["yaw_deg"]) + spin_deg_per_sec * dt
                    cost = angular_error(float(yaw), expected)
                    if cost > threshold_deg:
                        valid = False
                        break
                    assignment[obs_index] = slot_id
                    costs[obs_index] = cost
                    total += cost
                if not valid:
                    continue
                # Prefer more matches; use cost only as the tie breaker.
                score = (float(-count), total)
                if best is None or score < (float(-len(best[1])), best[0]):
                    best = (total, assignment, costs)
    if best is None:
        return {}, {}
    return best[1], best[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--vehicle-number", type=int, default=3)
    parser.add_argument("--spin-deg-per-sec", type=float, default=30.0)
    parser.add_argument("--match-threshold-deg", type=float, default=40.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    slots: Dict[int, dict] = {}
    next_id = 0
    frame_count = 0
    target_frame_count = 0
    target_observation_count = 0
    assigned_count = 0
    event_counts: Dict[str, int] = {}
    assignment_costs: List[float] = []
    assignment_rows: List[dict] = []

    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            timestamp_ns = record.get("source_capture_timestamp_ns")
            if not finite(timestamp_ns):
                timestamp_ns = record.get("vision_completion_timestamp_ns")
            if not finite(timestamp_ns):
                timestamp_ns = 0
            timestamp_ns = int(timestamp_ns)
            target = [
                observation
                for observation in record.get("observations", [])
                if observation.get("target_candidate")
                and observation.get("number") == args.vehicle_number
                and finite(observation.get("armor_yaw_absolute_deg"))
            ]
            target.sort(key=lambda item: float(item["armor_yaw_absolute_deg"]))
            frame_count += 1
            target_frame_count += bool(target)
            target_observation_count += len(target)

            assignments, costs = solve_assignment(
                target, slots, timestamp_ns, args.spin_deg_per_sec, args.match_threshold_deg
            )
            used_ids = set(assignments.values())
            for index, observation in enumerate(target):
                if index in assignments:
                    canonical_id = assignments[index]
                    event = "matched"
                    cost = costs[index]
                else:
                    # Allocate the next cyclic id not already used in this frame.
                    for _ in range(4):
                        candidate = next_id % 4
                        next_id = (next_id + 1) % 4
                        if candidate not in used_ids:
                            break
                    canonical_id = candidate
                    used_ids.add(canonical_id)
                    event = "initialized" if not slots else "new_cyclic"
                    cost = None
                observation["canonical_plate_id"] = canonical_id
                observation["canonical_assignment_event"] = event
                observation["canonical_assignment_cost_deg"] = cost
                state = {
                    "timestamp_ns": timestamp_ns,
                    "yaw_deg": float(observation["armor_yaw_absolute_deg"]),
                }
                slots[canonical_id] = state
                next_id = (canonical_id + 1) % 4
                event_counts[event] = event_counts.get(event, 0) + 1
                if cost is not None:
                    assignment_costs.append(cost)
                    assigned_count += 1
                assignment_rows.append(
                    {
                        "source_image_seq": record.get("source_image_seq"),
                        "timestamp_ns": timestamp_ns,
                        "observation_id": observation.get("observation_id"),
                        "canonical_plate_id": canonical_id,
                        "event": event,
                        "cost_deg": cost,
                        "yaw_deg": observation.get("armor_yaw_absolute_deg"),
                    }
                )
            # Keep non-target observations intact and emit the enriched record.
            destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            destination.write("\n")

    assignment_costs.sort()
    summary = {
        "source": str(input_path),
        "frames": frame_count,
        "frames_with_target_observations": target_frame_count,
        "target_observations": target_observation_count,
        "matched_observations": assigned_count,
        "new_or_initialized_observations": target_observation_count - assigned_count,
        "event_counts": event_counts,
        "canonical_id_counts": {
            str(slot_id): sum(row["canonical_plate_id"] == slot_id for row in assignment_rows)
            for slot_id in range(4)
        },
        "assignment_cost_deg": {
            "p50": assignment_costs[len(assignment_costs) // 2] if assignment_costs else None,
            "p95": assignment_costs[int(0.95 * (len(assignment_costs) - 1))]
            if assignment_costs
            else None,
            "max": max(assignment_costs) if assignment_costs else None,
        },
        "rule": {
            "vehicle_number": args.vehicle_number,
            "spin_deg_per_sec": args.spin_deg_per_sec,
            "match_threshold_deg": args.match_threshold_deg,
            "first_visible_plate_is_zero": True,
            "raw_observations_preserved": True,
            "tracker_identity_used": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
