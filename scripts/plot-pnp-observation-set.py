#!/usr/bin/env python3
"""Plot the complete target-3 PnP observation set and derived cyclic ids."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load(path: Path, vehicle_number: int) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            timestamp = record.get("source_capture_timestamp_ns")
            if not finite(timestamp):
                continue
            for observation in record.get("observations", []):
                if (
                    observation.get("target_candidate")
                    and observation.get("number") == vehicle_number
                    and finite(observation.get("armor_yaw_absolute_deg"))
                ):
                    rows.append(
                        {
                            "t": float(timestamp) * 1e-9,
                            "yaw": float(observation["armor_yaw_absolute_deg"]),
                            "distance_m": float(observation.get("distance_mm", math.nan)) * 1e-3,
                            "id": observation.get("canonical_plate_id"),
                            "event": observation.get("canonical_assignment_event"),
                        }
                    )
    rows.sort(key=lambda row: row["t"])
    return rows


def summarize(rows: List[dict]) -> Dict[str, object]:
    if not rows:
        return {"target_observations": 0}
    costs = [row["distance_m"] for row in rows if finite(row.get("distance_m"))]
    return {
        "target_observations": len(rows),
        "duration_s": rows[-1]["t"] - rows[0]["t"],
        "yaw_min_deg": min(row["yaw"] for row in rows),
        "yaw_max_deg": max(row["yaw"] for row in rows),
        "distance_m_p50": sorted(costs)[len(costs) // 2] if costs else None,
        "canonical_ids": sorted({row["id"] for row in rows if row["id"] is not None}),
        "new_cyclic_events": sum(row["event"] == "new_cyclic" for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--vehicle-number", type=int, default=3)
    args = parser.parse_args()

    runs: List[Tuple[str, List[dict]]] = []
    for item in args.run:
        label, separator, raw_path = item.partition("=")
        if not separator or not label or not raw_path:
            parser.error("invalid --run value: %r" % item)
        runs.append((label, load(Path(raw_path), args.vehicle_number)))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    figure, axes = plt.subplots(len(runs), 1, figsize=(15, 4.5 * len(runs)), squeeze=False)
    summaries: Dict[str, object] = {}
    for row_index, (label, rows) in enumerate(runs):
        axis = axes[row_index][0]
        summaries[label] = summarize(rows)
        if not rows:
            axis.text(0.5, 0.5, "no target observations", ha="center", va="center")
            axis.set_title(label)
            continue
        t0 = rows[0]["t"]
        for plate_id in range(4):
            points = [item for item in rows if item["id"] == plate_id]
            if points:
                axis.scatter(
                    [item["t"] - t0 for item in points],
                    [item["yaw"] for item in points],
                    s=5,
                    alpha=0.7,
                    color=colors[plate_id],
                    label="canonical plate %d" % plate_id,
                )
        axis.set_title(
            "%s: N=%d, yaw=[%.1f, %.1f] deg, IDs=%s"
            % (
                label,
                len(rows),
                min(item["yaw"] for item in rows),
                max(item["yaw"] for item in rows),
                ",".join(str(item) for item in summaries[label]["canonical_ids"]),
            )
        )
        axis.set_ylabel("armor_yaw_absolute (deg)")
        axis.set_ylim(-95, 95)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", markerscale=2)
    axes[-1][0].set_xlabel("time since first target observation (s)")
    figure.suptitle("All target-3 PnP observations with offline cyclic IDs")
    figure.tight_layout()
    figure.savefig(args.output, dpi=180)
    Path(args.summary_output).write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
