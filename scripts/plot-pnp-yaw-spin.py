#!/usr/bin/env python3
"""Plot tracked PnP yaw from high-rate pipeline JSONL telemetry.

The plot intentionally keeps the wrapped yaw and marks tracker-ID changes so
the expected plate-switch jumps remain visible. It does not associate GT or
smooth the observations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load_samples(path: Path) -> list[dict[str, float | int | str]]:
    samples: list[dict[str, float | int | str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            tracker = record.get("tracker") or {}
            armor = tracker.get("tracked_armor")
            if not isinstance(armor, dict):
                continue
            yaw = armor.get("armor_yaw_deg")
            yaw_absolute = armor.get("armor_yaw_absolute_deg")
            timestamp = record.get("source_capture_timestamp_ns")
            tracked_id = tracker.get("tracked_id")
            if not finite(yaw) or not finite(timestamp) or not finite(tracked_id):
                continue
            sample: dict[str, float | int | str] = {
                "t": float(timestamp) * 1e-9,
                "yaw": float(yaw),
                "id": int(tracked_id),
                "source": str(record.get("source_image_seq", "")),
            }
            if finite(yaw_absolute):
                sample["yaw_absolute"] = float(yaw_absolute)
            samples.append(sample)
    samples.sort(key=lambda item: float(item["t"]))
    return samples


def summarize(samples: list[dict[str, float | int | str]]) -> dict[str, object]:
    if not samples:
        return {"tracked_samples": 0, "ids": [], "id_transitions": 0}
    ids = [int(item["id"]) for item in samples]
    transitions = sum(left != right for left, right in zip(ids, ids[1:]))
    return {
        "tracked_samples": len(samples),
        "duration_s": float(samples[-1]["t"]) - float(samples[0]["t"]),
        "ids": sorted(set(ids)),
        "id_transitions": transitions,
        "yaw_min_deg": min(float(item["yaw"]) for item in samples),
        "yaw_max_deg": max(float(item["yaw"]) for item in samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    args = parser.parse_args()

    runs: list[tuple[str, list[dict[str, float | int | str]]]] = []
    for item in args.run:
        label, separator, raw_path = item.partition("=")
        if not separator or not label or not raw_path:
            parser.error(f"invalid --run value: {item!r}")
        runs.append((label, load_samples(Path(raw_path))))

    figure, axes = plt.subplots(len(runs), 1, figsize=(14, 4.2 * len(runs)), squeeze=False)
    summaries: dict[str, object] = {}
    for row, (label, samples) in enumerate(runs):
        axis = axes[row][0]
        summaries[label] = summarize(samples)
        if not samples:
            axis.text(0.5, 0.5, "no tracked_armor samples", ha="center", va="center")
            axis.set_title(label)
            continue

        t0 = float(samples[0]["t"])
        times = [float(item["t"]) - t0 for item in samples]
        yaws = [float(item["yaw"]) for item in samples]
        axis.plot(times, yaws, color="#1f77b4", linewidth=0.8, label="tracked PnP yaw")
        if all("yaw_absolute" in item for item in samples):
            axis.plot(
                times,
                [float(item["yaw_absolute"]) for item in samples],
                color="#ff7f0e",
                linewidth=0.55,
                alpha=0.7,
                label="camera-relative yaw_absolute",
            )
        ids = [int(item["id"]) for item in samples]
        for index in range(1, len(ids)):
            if ids[index] != ids[index - 1]:
                axis.axvline(times[index], color="#777777", linewidth=0.35, alpha=0.5)
        axis.set_title(f"{label}: {summaries[label]}")
        axis.set_ylabel("yaw (deg)")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1][0].set_xlabel("time since first tracked sample (s)")
    figure.suptitle("PnP yaw during target-3 slow spin (wrapped yaw; ID changes marked)")
    figure.tight_layout()
    figure.savefig(args.output, dpi=180)
    if args.summary_output:
        Path(args.summary_output).write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
