#!/usr/bin/env python3
"""Render raw/exact/proposed corner overlays for offline diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from score_linux_repair_observer_validation import key, read_jsonl
from train_corner_repair_benefit_gate import corners, load_session_samples
from training.stage3.train_image_corner_repair_pilot import load_release_ledger, load_release_rgba


COLORS = {
    "raw": (40, 40, 235),
    "exact": (40, 210, 40),
    "proposal": (235, 120, 30),
}


def candidate_by_identity(
    frames: list[dict[str, Any]], identity: tuple[int, int, int, int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    for frame in frames:
        if key(frame) != identity[:3]:
            continue
        for candidate in frame["candidates"]:
            if int(candidate["observation_id"]) == identity[3]:
                return frame, candidate
    raise KeyError(identity)


def matched_exact(session: Path, frame: dict[str, Any], candidate: dict[str, Any]) -> np.ndarray:
    labels = [row for row in read_jsonl(session / "exact-corners.jsonl") if key(row) == key(frame)]
    raw = corners(candidate, "raw_corners_px")
    costs = np.asarray(
        [
            float(
                np.sqrt(
                    np.mean(
                        np.square(raw - np.asarray(label["exact_corners_px"], dtype=np.float64))
                    )
                )
            )
            for label in labels
        ]
    )
    if not len(costs) or float(costs.min()) > 25.0:
        raise ValueError("selected candidate lacks a <=25 px exact match")
    return np.asarray(labels[int(np.argmin(costs))]["exact_corners_px"], dtype=np.float64)


def draw_quad(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    integer = np.round(points).astype(np.int32)
    cv2.polylines(image, [integer], True, color, 2, cv2.LINE_AA)
    for point in integer:
        cv2.circle(image, tuple(point), 4, color, -1, cv2.LINE_AA)


def render(session: Path, sidecar_name: str, output: Path, requested_outcome: str | None) -> dict[str, Any]:
    samples = load_session_samples(session, 25.0, sidecar_name)
    eligible = [sample for sample in samples if sample.old_applied]
    if requested_outcome is None:
        counts = {
            outcome: sum(sample.outcome == outcome for sample in eligible)
            for outcome in ("BENEFIT", "HARM")
        }
        requested_outcome = max(counts, key=counts.get)
    selected = [sample for sample in eligible if sample.outcome == requested_outcome]
    if not selected:
        raise ValueError(f"no applied {requested_outcome} sample in {session.name}")
    selected.sort(key=lambda sample: sample.raw["transverse_error_mm"])
    sample = selected[len(selected) // 2]
    frames = read_jsonl(session / sidecar_name)
    frame, candidate = candidate_by_identity(frames, sample.identity)
    exact = matched_exact(session, frame, candidate)
    raw = corners(candidate, "raw_corners_px")
    proposal = corners(candidate, "model_proposed_corners_px")
    ledger = load_release_ledger(session)
    rgba = load_release_rgba(session, ledger[key(frame)])
    image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    draw_quad(image, raw, COLORS["raw"], "raw")
    draw_quad(image, exact, COLORS["exact"], "exact")
    draw_quad(image, proposal, COLORS["proposal"], "proposal")
    combined = np.concatenate((raw, exact, proposal), axis=0)
    minimum = np.floor(combined.min(axis=0) - 45).astype(int)
    maximum = np.ceil(combined.max(axis=0) + 45).astype(int)
    minimum = np.maximum(minimum, 0)
    maximum = np.minimum(maximum, np.asarray([image.shape[1], image.shape[0]]))
    crop = image[minimum[1] : maximum[1], minimum[0] : maximum[0]]
    crop = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST)
    header = np.zeros((90, crop.shape[1], 3), dtype=np.uint8)
    lines = [
        f"{session.name}  {requested_outcome}",
        (
            f"raw transverse={sample.raw['transverse_error_mm']:.2f} mm  "
            f"proposal={sample.proposed['transverse_error_mm']:.2f} mm"
        ),
        "red=raw detector  green=exact label  blue=model proposal",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            header,
            line,
            (10, 24 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    rendered = np.concatenate((header, crop), axis=0)
    if not cv2.imwrite(str(output), rendered):
        raise OSError(f"failed to write {output}")
    return {
        "session": session.name,
        "outcome": requested_outcome,
        "identity": list(sample.identity),
        "output": str(output),
        "raw": sample.raw,
        "proposal": sample.proposed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", action="append", required=True, type=Path)
    parser.add_argument("--pnp-sidecar-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--outcome",
        action="append",
        choices=("BENEFIT", "HARM"),
        help="optional outcome per --session; otherwise choose the majority applied outcome",
    )
    args = parser.parse_args()
    if args.outcome and len(args.outcome) != len(args.session):
        raise ValueError("--outcome count must match --session count")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic overlays: {output}")
    output.mkdir(parents=True)
    rows = []
    for index, supplied in enumerate(args.session):
        session = supplied.resolve(strict=True)
        outcome = None if not args.outcome else args.outcome[index]
        rows.append(
            render(
                session,
                args.pnp_sidecar_name,
                output / f"{index + 1:02d}-{session.name}.png",
                outcome,
            )
        )
    (output / "overlay_manifest.json").write_text(
        json.dumps({"schema_version": "aim-stack.corner-repair-overlay/1", "rows": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
