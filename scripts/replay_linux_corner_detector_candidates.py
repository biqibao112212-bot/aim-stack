#!/usr/bin/env python3
"""Replay every stored Linux RGBA frame into a complete detector sidecar."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


CORNER_ORDER = ("bl", "tl", "tr", "br")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_detector_module(script: Path):
    spec = importlib.util.spec_from_file_location("linux_corner_detector_rows", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import detector implementation: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--session", required=True, type=Path)
    result.add_argument("--model", required=True, type=Path)
    result.add_argument("--detector-script", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--manifest", required=True, type=Path)
    result.add_argument("--score-threshold", type=float, default=0.25)
    result.add_argument("--nms-threshold", type=float, default=0.45)
    result.add_argument("--minimum-frame-seq", type=int, default=0)
    result.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--max-frames", type=int)
    return result


def corner_json(points: np.ndarray) -> dict[str, list[float]]:
    return {
        name: [float(point[0]), float(point[1])]
        for name, point in zip(CORNER_ORDER, np.asarray(points, dtype=float))
    }


def write_new_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def main() -> None:
    args = parser().parse_args()
    session = args.session.resolve(strict=True)
    model = args.model.resolve(strict=True)
    detector_script = args.detector_script.resolve(strict=True)
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    if output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite complete detector sidecar evidence")
    if args.minimum_frame_seq < 0:
        raise ValueError("--minimum-frame-seq must be non-negative")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    detector = load_detector_module(detector_script)
    ledger = []
    seen = set()
    with (session / "tcp-identities.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            key = tuple(int(item[name]) for name in ("producer_epoch", "frame_seq", "timestamp_ns"))
            if key in seen:
                raise ValueError(f"duplicate TCP identity: {key}")
            seen.add(key)
            if key[1] >= args.minimum_frame_seq:
                ledger.append((key, item))
    ledger.sort(key=lambda item: item[0])
    if args.max_frames is not None:
        ledger = ledger[: args.max_frames]
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.provider == "cuda"
        else ["CPUExecutionProvider"]
    )
    inference = ort.InferenceSession(str(model), providers=providers)
    input_name = inference.get_inputs()[0].name
    candidate_total = zero_candidate_frames = 0
    latency_us = []
    started = time.time()
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for key, identity in ledger:
            rgba = detector.load_release_rgba(session, identity)
            start_ns = time.perf_counter_ns()
            tensor = detector.preprocess(rgba)
            inference_start_ns = time.perf_counter_ns()
            raw_output = inference.run(None, {input_name: tensor})[0]
            inference_end_ns = time.perf_counter_ns()
            detections, diagnostics = detector.decode_with_diagnostics(
                raw_output, args.score_threshold, args.nms_threshold
            )
            end_ns = time.perf_counter_ns()
            candidates = []
            for rank, detection in enumerate(detections):
                candidates.append(
                    {
                        "candidate_rank": rank,
                        "observation_id": rank,
                        "output_row_index": int(detection["output_row_index"]),
                        "objectness": float(detection["score"]),
                        "objectness_logit": float(detection["objectness_logit"]),
                        "box_xywh_px": detection["box"],
                        "raw_corners_order": list(CORNER_ORDER),
                        "raw_corners_px": corner_json(detection["corners"]),
                        "decoded_color": int(detection["color_index"]),
                        "color_logits": detection["color_logits"],
                        "decoded_number": int(detection["number"]),
                        "number_index": int(detection["number_index"]),
                        "number_logits": detection["number_logits"],
                        "decoded_armor_type": detection["armor_type"],
                        "filter_status": "post_score_post_nms_unfiltered_team",
                    }
                )
            candidate_total += len(candidates)
            zero_candidate_frames += int(not candidates)
            total_us = (end_ns - start_ns) / 1000.0
            latency_us.append(total_us)
            record = {
                "schema_version": "aim-stack.linux-offline-detector-frame/1",
                "session_id": session.name,
                "runtime_instance_id": f"offline-onnx-{session.name}",
                "producer_epoch": key[0],
                "frame_seq": key[1],
                "timestamp_ns": key[2],
                "image": {
                    "raw_rgba_file": identity["raw_rgba_file"],
                    "raw_rgba_sha256": identity["raw_rgba_sha256"],
                    "payload_bytes": int(identity["payload_bytes"]),
                    "width": int(identity.get("width", detector.IMAGE_WIDTH)),
                    "height": int(identity.get("height", detector.IMAGE_HEIGHT)),
                    "pixel_format": identity.get("pixel_format", "rgba32"),
                },
                "submission_status": "offline_replay_complete",
                "completion_status": "ok",
                "drop_status": "not_applicable_offline",
                "sink_status": "ok",
                "candidate_count": len(candidates),
                "candidates": candidates,
                "decode_diagnostics": diagnostics,
                "timing_us": {
                    "preprocess": (inference_start_ns - start_ns) / 1000.0,
                    "inference": (inference_end_ns - inference_start_ns) / 1000.0,
                    "decode_nms": (end_ns - inference_end_ns) / 1000.0,
                    "total": total_us,
                },
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    output_hash = sha256(output)
    manifest = {
        "schema_version": "aim-stack.linux-offline-detector-sidecar-manifest/1",
        "claim": "deterministic offline ONNXRuntime candidate replay; not TensorRT parity or native latency",
        "session": str(session),
        "frames": len(ledger),
        "candidates": candidate_total,
        "zero_candidate_frames": zero_candidate_frames,
        "identity_unique": len(ledger) == len({key for key, _ in ledger}),
        "model": str(model),
        "model_sha256": sha256(model),
        "detector_script": str(detector_script),
        "detector_script_sha256": sha256(detector_script),
        "sidecar_script": str(Path(__file__).resolve()),
        "sidecar_script_sha256": sha256(Path(__file__).resolve()),
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "minimum_frame_seq": args.minimum_frame_seq,
        "requested_provider": args.provider,
        "active_providers": inference.get_providers(),
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "onnxruntime": ort.__version__,
        },
        "offline_total_latency_us": {
            "p50": float(np.percentile(latency_us, 50)) if latency_us else None,
            "p95": float(np.percentile(latency_us, 95)) if latency_us else None,
            "p99": float(np.percentile(latency_us, 99)) if latency_us else None,
            "max": max(latency_us, default=None),
            "production_claim": False,
        },
        "elapsed_wall_s": time.time() - started,
        "output": str(output),
        "output_sha256": output_hash,
        "output_bytes": output.stat().st_size,
        "truth_fields_included": False,
        "retention": {
            "classification": "protected_derived_detector_evidence",
            "deletion_allowed": False,
        },
    }
    write_new_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
