#!/usr/bin/env python3
"""Truth-free shadow replay of raw and repaired PnP sets through the observer."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from causal_ray_observer import CausalRayObserver, ObserverConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def candidate_for_domain(candidate: dict[str, Any], domain: str) -> dict[str, Any]:
    group = candidate["raw_pnp"] if domain == "raw" else candidate["selected_pnp"]
    selected = next((item for item in group["candidates"] if item["selected"]), None)
    return {
        "valid": selected is not None,
        "camera_tvec_m": None if selected is None else selected["tvec_m"],
        "yaw_absolute_rad": None if selected is None else selected["observed_yaw_rad"],
        "detector_confidence": candidate["detector_confidence"],
        "reprojection_rms_px": None if selected is None else selected["reprojection_rms_px"],
        "reprojection_max_px": None if selected is None else selected["reprojection_max_px"],
        "corner_source": "raw" if domain == "raw" else candidate["corner_source"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pnp-sidecar", required=True, type=Path)
    parser.add_argument("--pnp-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sidecar = args.pnp_sidecar.resolve(strict=True)
    manifest_path = args.pnp_manifest.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite observer shadow evidence: {output}")
    output.mkdir(parents=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["output_sha256"] != sha256(sidecar):
        raise ValueError("repair/PnP sidecar hash mismatch")
    frames = read_jsonl(sidecar)
    if len(frames) != int(manifest["frames"]):
        raise ValueError("repair/PnP frame count mismatch")
    observers = {domain: CausalRayObserver(ObserverConfig()) for domain in ("raw", "repaired")}
    statuses = {domain: Counter() for domain in observers}
    latency = {domain: [] for domain in observers}
    records = []
    status_disagreements = 0
    for frame in frames:
        outputs = {}
        for domain, observer in observers.items():
            runtime_frame = {
                "schema_version": "autoaim-observer-frame-v1",
                "session_id": frame["session_id"],
                "producer_epoch": frame["producer_epoch"],
                "frame_seq": frame["frame_seq"],
                "capture_timestamp_ns": frame["timestamp_ns"],
                "observation_sink_status": frame["observation_sink_status"],
                "candidates": [
                    candidate_for_domain(candidate, domain) for candidate in frame["candidates"]
                ],
            }
            start = time.perf_counter_ns()
            observer_output = observer.update(runtime_frame)
            latency[domain].append((time.perf_counter_ns() - start) / 1000.0)
            statuses[domain][observer_output["observer_status"]] += 1
            outputs[domain] = observer_output
        status_disagreements += int(
            outputs["raw"]["observer_status"] != outputs["repaired"]["observer_status"]
        )
        records.append(
            {
                "schema_version": "aim-stack.linux-repair-observer-shadow-frame/1",
                "producer_epoch": frame["producer_epoch"],
                "frame_seq": frame["frame_seq"],
                "timestamp_ns": frame["timestamp_ns"],
                "repair_applied_candidates": sum(
                    bool(candidate["repair"]["applied"]) for candidate in frame["candidates"]
                ),
                "raw": outputs["raw"],
                "repaired": outputs["repaired"],
            }
        )
    frame_output = output / "observer_shadow_frames.jsonl"
    with frame_output.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "aim-stack.linux-repair-observer-shadow-summary/1",
        "claim": "truth-free single-session development shadow; not formal observer or predictor acceptance",
        "frames": len(frames),
        "detector_candidates": sum(len(frame["candidates"]) for frame in frames),
        "repair_applied_candidates": sum(
            bool(candidate["repair"]["applied"])
            for frame in frames
            for candidate in frame["candidates"]
        ),
        "status_disagreement_frames": status_disagreements,
        "domains": {
            domain: {
                "status_counts": dict(statuses[domain]),
                "observed_anonymous_fraction": statuses[domain]["OBSERVED_ANONYMOUS"] / len(frames),
                "ambiguous_fraction": statuses[domain]["AMBIGUOUS_SET"] / len(frames),
                "stale_fraction": statuses[domain]["STALE"] / len(frames),
                "offline_python_latency_us": {
                    "p50": float(np.percentile(latency[domain], 50)),
                    "p95": float(np.percentile(latency[domain], 95)),
                    "p99": float(np.percentile(latency[domain], 99)),
                    "max": max(latency[domain]),
                    "production_claim": False,
                },
            }
            for domain in observers
        },
        "guards": {
            "truth_fields_in_runtime_input": False,
            "physical_identity_resolved": False,
            "prediction_valid": False,
            "fire_control_valid": False,
        },
        "pnp_sidecar": str(sidecar),
        "pnp_sidecar_sha256": sha256(sidecar),
        "pnp_manifest": str(manifest_path),
        "pnp_manifest_sha256": sha256(manifest_path),
        "observer_config": ObserverConfig().__dict__,
        "frame_output": str(frame_output),
        "frame_output_sha256": sha256(frame_output),
    }
    summary_path = output / "observer_shadow_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "retention_manifest.json").write_text(
        json.dumps(
            {
                "classification": "long_term_private_development_evidence",
                "deletion_allowed": False,
                "artifacts": {
                    path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in output.iterdir()
                    if path.is_file() and path.name != "retention_manifest.json"
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
