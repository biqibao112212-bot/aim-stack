#!/usr/bin/env python3
"""Run the protected ONNX detector on collected Linux exact-corner frames.

Exact corners are used only after detector inference, to match a detected
quadrilateral and make a supervised target.  The emitted `raw_*` fields are
the detector's own output, never copied from simulator labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


ORDER = ("bl", "tl", "tr", "br")
IMAGE_WIDTH, IMAGE_HEIGHT = 1440, 1080


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--session", required=True, type=Path)
    result.add_argument("--labels", required=True, type=Path)
    result.add_argument("--model", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--score-threshold", type=float, default=0.25)
    result.add_argument("--nms-threshold", type=float, default=0.45)
    result.add_argument("--match-rms-px", type=float, default=25.0)
    result.add_argument(
        "--minimum-frame-seq",
        type=int,
        default=0,
        help="exclude frames before a recorded Scene Control geometry/motion commit boundary",
    )
    result.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    result.add_argument(
        "--allow-empty", action="store_true",
        help="write a header-only evidence table when detector applicability is zero",
    )
    result.add_argument(
        "--max-labeled-exposures",
        type=int,
        help="deterministically process at most this many label-bearing frames; omitted means all",
    )
    result.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="deterministically retain every Nth label-bearing frame before --max-labeled-exposures",
    )
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_release_rgba(session: Path, identity: dict[str, object]) -> np.ndarray:
    """Read a simulator-owned full-frame export without parsing the TCP wire."""
    required = ("raw_rgba_file", "raw_rgba_sha256", "payload_sha256", "payload_bytes")
    if any(field not in identity for field in required):
        raise ValueError("Release ledger lacks required full-frame export metadata")
    relative = Path(str(identity["raw_rgba_file"]))
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("frames",):
        raise ValueError(f"unsafe Release raw-frame path: {relative}")
    image_file = session / relative
    payload = image_file.read_bytes()
    if len(payload) != int(identity["payload_bytes"]) or len(payload) != IMAGE_WIDTH * IMAGE_HEIGHT * 4:
        raise ValueError(f"invalid Release raw RGBA payload size: {image_file}")
    payload_hash = hashlib.sha256(payload).hexdigest()
    if payload_hash != identity["payload_sha256"] or payload_hash != identity["raw_rgba_sha256"]:
        raise ValueError(f"Release raw-frame hash does not match ledger: {image_file}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 4).copy()


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -80.0, 80.0)))


def preprocess(rgba: np.ndarray) -> np.ndarray:
    # Release frames are literal RGBA32 bytes, not OpenCV-decoded PNG BGRA.
    # Keep this conversion aligned with the public SDK contract and the repair
    # patch path below; otherwise red and blue are swapped before ONNX input.
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    resized = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((640, 640, 3), dtype=np.uint8)
    canvas[:480] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]


def ordered_corners(row: np.ndarray) -> np.ndarray:
    # This is the existing detector's `[0,6,4,2] / [1,7,5,3]` vertex decode,
    # followed by its left/right and top/bottom canonicalization.
    points = np.asarray([[row[0], row[1]], [row[6], row[7]], [row[4], row[5]], [row[2], row[3]]], dtype=np.float64)
    points *= IMAGE_WIDTH / 640.0
    by_x = points[np.argsort(points[:, 0], kind="stable")]
    left, right = by_x[:2], by_x[2:]
    left = left[np.argsort(left[:, 1], kind="stable")]
    right = right[np.argsort(right[:, 1], kind="stable")]
    return np.asarray([left[1], left[0], right[0], right[1]], dtype=np.float64)


def decoded_number(number_index: int) -> int:
    if number_index == 0:
        return 7
    if 1 <= number_index <= 5:
        return number_index
    if number_index == 6:
        return 6
    if number_index in (7, 8):
        return 8
    raise ValueError(f"invalid detector number index: {number_index}")


def decode_with_diagnostics(
    output: np.ndarray, score_threshold: float, nms_threshold: float
) -> tuple[list[dict[str, object]], dict[str, int]]:
    values = np.asarray(output, dtype=np.float32).reshape(-1, 22)
    scores = sigmoid(values[:, 8])
    candidates: list[dict[str, object]] = []
    diagnostics = {
        "raw_output_rows": len(values),
        "nonfinite_score_rows": 0,
        "below_score_rows": 0,
        "nonfinite_corner_rows": 0,
        "degenerate_box_rows": 0,
        "pre_nms_candidates": 0,
        "nms_rejected_candidates": 0,
        "post_nms_candidates": 0,
    }
    for output_row_index, (row, score) in enumerate(zip(values, scores)):
        if not math.isfinite(float(score)):
            diagnostics["nonfinite_score_rows"] += 1
            continue
        if float(score) < score_threshold:
            diagnostics["below_score_rows"] += 1
            continue
        corners = ordered_corners(row)
        if not np.isfinite(corners).all():
            diagnostics["nonfinite_corner_rows"] += 1
            continue
        minimum, maximum = corners.min(axis=0), corners.max(axis=0)
        width, height = maximum - minimum
        if width < 1.0 or height < 1.0:
            diagnostics["degenerate_box_rows"] += 1
            continue
        color_logits = np.asarray(row[9:13], dtype=np.float64)
        number_logits = np.asarray(row[13:22], dtype=np.float64)
        color_index = int(np.argmax(color_logits))
        number_index = int(np.argmax(number_logits))
        number = decoded_number(number_index)
        candidates.append(
            {
                "output_row_index": output_row_index,
                "corners": corners,
                "score": float(score),
                "objectness_logit": float(row[8]),
                "box": [float(minimum[0]), float(minimum[1]), float(width), float(height)],
                "color_index": color_index,
                "color_logits": color_logits.tolist(),
                "number_index": number_index,
                "number": number,
                "number_logits": number_logits.tolist(),
                "armor_type": "large" if number == 1 else "small",
            }
        )
    diagnostics["pre_nms_candidates"] = len(candidates)
    if not candidates:
        return [], diagnostics
    indices = cv2.dnn.NMSBoxes([item["box"] for item in candidates], [item["score"] for item in candidates], score_threshold, nms_threshold)
    selected = [candidates[int(index)] for index in np.asarray(indices).reshape(-1)]
    diagnostics["nms_rejected_candidates"] = len(candidates) - len(selected)
    diagnostics["post_nms_candidates"] = len(selected)
    return selected, diagnostics


def decode(output: np.ndarray, score_threshold: float, nms_threshold: float) -> list[dict[str, object]]:
    return decode_with_diagnostics(output, score_threshold, nms_threshold)[0]


def main() -> None:
    args = parser().parse_args()
    session, labels, model, output = (args.session.resolve(strict=True), args.labels.resolve(strict=True), args.model.resolve(strict=True), args.output.resolve())
    if output.exists():
        raise FileExistsError(f"refusing to overwrite detector evidence: {output}")
    ledger = {}
    with (session / "tcp-identities.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            key = tuple(int(item[name]) for name in ("producer_epoch", "frame_seq", "timestamp_ns"))
            if key in ledger:
                raise ValueError(f"duplicate TCP identity: {key}")
            ledger[key] = item
    label_rows: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    with labels.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = tuple(int(row[name]) for name in ("producer_epoch", "frame_seq", "timestamp_ns"))
            if key not in ledger:
                raise ValueError(f"label lacks a stored TCP image: {key}")
            label_rows.setdefault(key, []).append(row)
    if args.minimum_frame_seq < 0:
        raise ValueError("--minimum-frame-seq must be non-negative")
    keys = [key for key in sorted(label_rows) if key[1] >= args.minimum_frame_seq]
    if args.sample_stride <= 0:
        raise ValueError("--sample-stride must be positive")
    keys = keys[::args.sample_stride]
    if args.max_labeled_exposures is not None:
        if args.max_labeled_exposures <= 0:
            raise ValueError("--max-labeled-exposures must be positive")
        keys = keys[:args.max_labeled_exposures]
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.provider == "cuda" else ["CPUExecutionProvider"]
    inference = ort.InferenceSession(str(model), providers=providers)
    input_name = inference.get_inputs()[0].name
    model_sha256 = sha256(model)
    rows: list[dict[str, object]] = []
    for key in keys:
        labels_for_frame = label_rows[key]
        rgba = load_release_rgba(session, ledger[key])
        detections = decode(inference.run(None, {input_name: preprocess(rgba)})[0], args.score_threshold, args.nms_threshold)
        unmatched = set(range(len(detections)))
        for label in sorted(labels_for_frame, key=lambda item: int(item["relative_slot"])):
            truth = np.asarray(label["exact_corners_px"], dtype=np.float64)
            candidates = [(float(np.sqrt(np.mean(np.square(np.asarray(detections[index]["corners"]) - truth)))), index) for index in unmatched]
            if not candidates:
                continue
            rms, index = min(candidates)
            if rms > args.match_rms_px:
                continue
            unmatched.remove(index)
            detection = detections[index]
            row: dict[str, object] = {
                "session_id": session.name, "producer_epoch": key[0], "frame_seq": key[1], "timestamp_ns": key[2],
                "relative_slot": int(label["relative_slot"]), "motion_uniform": bool(label["motion_uniform"]),
                "distance_m": float(label["distance_m"]), "image_file": str(ledger[key]["raw_rgba_file"]),
                "detector_score": float(detection["score"]), "match_corner_rms_px": rms,
                "model_sha256": model_sha256, "future_truth_included": False,
            }
            for name, raw, exact in zip(ORDER, np.asarray(detection["corners"]), truth):
                row[f"raw_{name}_x_px"], row[f"raw_{name}_y_px"] = float(raw[0]), float(raw[1])
                row[f"exact_{name}_x_px"], row[f"exact_{name}_y_px"] = float(exact[0]), float(exact[1])
            rows.append(row)
    if not rows and not args.allow_empty:
        raise ValueError("detector produced no label-matched raw corner rows")
    fields = [
        "session_id", "producer_epoch", "frame_seq", "timestamp_ns", "relative_slot",
        "motion_uniform", "distance_m", "image_file", "detector_score", "match_corner_rms_px",
        "model_sha256", "future_truth_included",
        *(f"{prefix}_{corner}_{axis}_px" for corner in ORDER for prefix in ("raw", "exact") for axis in ("x", "y")),
    ]
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "labeled_exposures": len(keys), "matched_exposures": len({(r['producer_epoch'], r['frame_seq'], r['timestamp_ns']) for r in rows}), "providers": inference.get_providers(), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
