#!/usr/bin/env python3
"""Plot only the corrected production chassis yaw as connected curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            timestamp = record.get("source_capture_timestamp_ns")
            for armor in record.get("observations") or []:
                if not isinstance(armor, dict) or not armor.get("target_candidate"):
                    continue
                if armor.get("number") != 3 or not isinstance(timestamp, (int, float)):
                    continue
                pnp = armor.get("pnp_ab") or {}
                yaw = pnp.get("corrected_chassis_yaw_deg")
                plate_id = armor.get("canonical_plate_id")
                if not isinstance(yaw, (int, float)) or not isinstance(plate_id, int):
                    continue
                rows.append(
                    {
                        "t": float(timestamp) * 1e-9,
                        "id": plate_id,
                        "yaw": float(yaw),
                    }
                )
    rows.sort(key=lambda row: (row["t"], row["id"]))
    return rows


def plot_run(axis: plt.Axes, label: str, rows: list[dict]) -> None:
    if not rows:
        axis.set_title(label)
        return
    t0 = rows[0]["t"]
    for plate_id in range(4):
        plate = [row for row in rows if row["id"] == plate_id]
        segments: list[list[dict]] = []
        segment: list[dict] = []
        for row in plate:
            if segment and row["t"] - segment[-1]["t"] > 0.2:
                segments.append(segment)
                segment = []
            segment.append(row)
        if segment:
            segments.append(segment)
        for segment_index, segment in enumerate(segments):
            axis.plot(
                [row["t"] - t0 for row in segment],
                [row["yaw"] for row in segment],
                color=COLORS[plate_id],
                linewidth=0.9,
                marker=".",
                markersize=2.0,
                alpha=0.82,
                label=f"plate {plate_id}" if segment_index == 0 else None,
            )
    axis.set_title(label)
    axis.set_ylim(-90, 90)
    axis.set_ylabel("corrected chassis yaw (deg)")
    axis.set_xlabel("seconds from first observation")
    axis.grid(True, alpha=0.25)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=JSONL")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    figure, axes = plt.subplots(len(args.run), 1, figsize=(15, 4.0 * len(args.run)), squeeze=False)
    for index, raw in enumerate(args.run):
        label, separator, path_text = raw.partition("=")
        if not separator:
            parser.error(f"invalid --run: {raw!r}")
        plot_run(axes[index][0], label, load_rows(Path(path_text)))
    axes[0][0].legend(loc="upper right", fontsize=9, ncol=4)
    figure.suptitle(
        "Target 3 slow spin: corrected production PnP yaw\n"
        "connected observations; gaps >0.2 s are intentionally left open",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(args.output, dpi=200)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
