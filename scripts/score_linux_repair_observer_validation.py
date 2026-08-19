#!/usr/bin/env python3
"""Join truth only after runtime replay and score raw/repaired PnP observations."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


CORNER_ORDER = ("bl", "tl", "tr", "br")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row: dict[str, Any]) -> tuple[int, int, int]:
    return tuple(int(row[name]) for name in ("producer_epoch", "frame_seq", "timestamp_ns"))


def label_matrix(label: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = label["camera"]["intrinsics"]
    matrix = np.asarray(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return matrix, np.asarray(intrinsics["distortion"], dtype=np.float64)


def exact_camera_tvec(label: dict[str, Any]) -> np.ndarray:
    corners_armor_m = np.asarray(
        label["plate_geometry"]["object_corners_armor_m"], dtype=np.float32
    )
    object_points = np.column_stack(
        (
            corners_armor_m[:, 0] * 1000.0,
            -corners_armor_m[:, 2] * 1000.0,
            np.zeros(4, dtype=np.float32),
        )
    ).astype(np.float32)
    corners = np.asarray(label["exact_corners_px"], dtype=np.float32)
    matrix, distortion = label_matrix(label)
    result = cv2.solvePnPGeneric(
        object_points, corners, matrix, distortion, flags=cv2.SOLVEPNP_IPPE
    )
    if not bool(result[0]) or not result[1]:
        raise ValueError("exact-corner IPPE failed")
    candidates = []
    for index, (rvec, tvec) in enumerate(zip(result[1], result[2])):
        point = np.asarray(tvec, dtype=np.float64).reshape(3)
        if not np.isfinite(point).all() or point[2] <= 0.0:
            continue
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, matrix, distortion
        )
        rms = float(
            np.sqrt(np.mean(np.sum(np.square(projected.reshape(4, 2) - corners), axis=1)))
        )
        candidates.append((rms, index, point / 1000.0))
    if not candidates:
        raise ValueError("exact-corner IPPE has no positive-depth candidate")
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def raw_corners(candidate: dict[str, Any]) -> np.ndarray:
    values = candidate["repair"]["raw_corners_px"]
    return np.asarray([values[name] for name in CORNER_ORDER], dtype=np.float64)


def selected_tvec(candidate: dict[str, Any], domain: str) -> np.ndarray | None:
    group = candidate["raw_pnp"] if domain == "raw" else candidate["selected_pnp"]
    selected = next((item for item in group["candidates"] if item["selected"]), None)
    if selected is None:
        return None
    point = np.asarray(selected["tvec_m"], dtype=np.float64)
    return point if point.shape == (3,) and np.isfinite(point).all() else None


def error_components(observed: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    truth_range = float(np.linalg.norm(truth))
    observed_range = float(np.linalg.norm(observed))
    truth_ray = truth / truth_range
    observed_ray = observed / observed_range
    cosine = float(np.clip(np.dot(truth_ray, observed_ray), -1.0, 1.0))
    residual = observed - truth
    radial = float(np.dot(residual, truth_ray))
    transverse = residual - radial * truth_ray
    return {
        "angular_error_deg": math.degrees(math.acos(cosine)),
        "radial_error_abs_mm": abs(radial) * 1000.0,
        "transverse_error_mm": float(np.linalg.norm(transverse)) * 1000.0,
        "position_error_mm": float(np.linalg.norm(residual)) * 1000.0,
    }


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def session_score(session: Path, analysis: Path, match_gate_px: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = json.loads((session / "session-result.json").read_text(encoding="utf-8"))
    pnp_path = session / "repair-pnp-complete-candidates-v1.jsonl"
    pnp_manifest = json.loads(
        (session / "repair-pnp-complete-candidates-v1-manifest.json").read_text(encoding="utf-8")
    )
    if pnp_manifest["output_sha256"] != sha256(pnp_path):
        raise ValueError(f"PnP sidecar hash mismatch: {session}")
    observer_summary = json.loads(
        (analysis / session.name / "observer_shadow_summary.json").read_text(encoding="utf-8")
    )
    labels: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for label in read_jsonl(session / "exact-corners.jsonl"):
        if int(label["frame_seq"]) >= int(result["first_eligible_frame_seq"]):
            labels[key(label)].append(label)
    truth_cache: dict[tuple[int, int, int, int], np.ndarray] = {}
    samples = []
    matched = total_labels = false_positive = 0
    frames = read_jsonl(pnp_path)
    for frame in frames:
        frame_labels = labels.get(key(frame), [])
        total_labels += len(frame_labels)
        candidates = frame["candidates"]
        if not frame_labels or not candidates:
            false_positive += len(candidates)
            continue
        costs = np.zeros((len(candidates), len(frame_labels)), dtype=np.float64)
        for candidate_index, candidate in enumerate(candidates):
            raw = raw_corners(candidate)
            for label_index, label in enumerate(frame_labels):
                exact = np.asarray(label["exact_corners_px"], dtype=np.float64)
                costs[candidate_index, label_index] = float(
                    np.sqrt(np.mean(np.square(raw - exact)))
                )
        rows, columns = linear_sum_assignment(costs)
        assigned_candidates = set()
        for candidate_index, label_index in zip(rows, columns):
            corner_rms = float(costs[candidate_index, label_index])
            if corner_rms > match_gate_px:
                continue
            assigned_candidates.add(int(candidate_index))
            candidate = candidates[int(candidate_index)]
            label = frame_labels[int(label_index)]
            truth_key = (*key(frame), int(label["relative_slot"]))
            if truth_key not in truth_cache:
                truth_cache[truth_key] = exact_camera_tvec(label)
            truth = truth_cache[truth_key]
            matched += 1
            for domain in ("raw", "repaired"):
                observed = selected_tvec(candidate, domain)
                if observed is None:
                    continue
                samples.append(
                    {
                        "session": session.name,
                        "split": result["planned"]["split"],
                        "mode": result["planned"]["mode"],
                        "producer_epoch": frame["producer_epoch"],
                        "frame_seq": frame["frame_seq"],
                        "timestamp_ns": frame["timestamp_ns"],
                        "relative_slot": int(label["relative_slot"]),
                        "domain": domain,
                        "repair_applied": bool(candidate["repair"]["applied"]),
                        "detector_score": float(candidate["detector_confidence"]),
                        "detector_truth_corner_rms_px": corner_rms,
                        **error_components(observed, truth),
                    }
                )
        false_positive += len(candidates) - len(assigned_candidates)
    metrics = {}
    for domain in ("raw", "repaired"):
        selected = [row for row in samples if row["domain"] == domain]
        metrics[domain] = {
            field: describe([float(row[field]) for row in selected])
            for field in (
                "angular_error_deg",
                "radial_error_abs_mm",
                "transverse_error_mm",
                "position_error_mm",
            )
        }
        metrics[domain]["observed_anonymous_fraction"] = observer_summary["domains"][domain][
            "observed_anonymous_fraction"
        ]
        metrics[domain]["ambiguous_fraction"] = observer_summary["domains"][domain][
            "ambiguous_fraction"
        ]
    exposure = json.loads((session / "exposure-manifest.json").read_text(encoding="utf-8"))
    planned = result["planned"]
    summary = {
        "session": session.name,
        "split": planned["split"],
        "mode": planned["mode"],
        "distance_proxy_radial_scale": planned["radial_scale"],
        "frames": len(frames),
        "truth_labels": total_labels,
        "matched_candidates": matched,
        "truth_label_match_fraction": matched / total_labels if total_labels else 0.0,
        "unmatched_detector_candidates": false_positive,
        "exposure_identity_join_fraction": exposure["coverage_fraction"],
        "metrics": metrics,
    }
    return summary, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", action="append", required=True, type=Path)
    parser.add_argument("--analysis-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--match-gate-px", type=float, default=25.0)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite scored validation evidence: {output}")
    output.mkdir(parents=True)
    analysis = args.analysis_root.resolve(strict=True)
    sessions = []
    for root in args.collection_root:
        sessions.extend(
            sorted(
                path.parent
                for path in root.resolve(strict=True).glob("*/session-result.json")
                if (path.parent / "repair-pnp-complete-candidates-v1.jsonl").exists()
            )
        )
    summaries = []
    sample_rows = []
    for session in sessions:
        summary, samples = session_score(session, analysis, args.match_gate_px)
        summaries.append(summary)
        sample_rows.extend(samples)
    sample_path = output / "pnp_scored_samples.csv.gz"
    with gzip.open(sample_path, "xt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    rows_path = output / "session_metrics.jsonl"
    with rows_path.open("x", encoding="utf-8", newline="\n") as handle:
        for summary in summaries:
            handle.write(json.dumps(summary, separators=(",", ":")) + "\n")
    gate_rows = []
    for summary in summaries:
        if summary["split"] != "validation":
            continue
        raw = summary["metrics"]["raw"]
        repaired = summary["metrics"]["repaired"]
        raw_angular_p95 = raw["angular_error_deg"]["p95"]
        repaired_angular_p95 = repaired["angular_error_deg"]["p95"]
        raw_depth_p95 = raw["radial_error_abs_mm"]["p95"]
        repaired_depth_p95 = repaired["radial_error_abs_mm"]["p95"]
        gates = {
            "exact_identity_join": summary["exposure_identity_join_fraction"] >= 0.98,
            "observer_availability_noninferior": (
                repaired["observed_anonymous_fraction"]
                >= raw["observed_anonymous_fraction"] - 0.01
            ),
            "observer_ambiguity_noninferior": (
                repaired["ambiguous_fraction"] <= raw["ambiguous_fraction"] + 0.01
            ),
            "angular_p95_noninferior": (
                raw_angular_p95 is not None
                and repaired_angular_p95 is not None
                and repaired_angular_p95 <= raw_angular_p95 + 0.02
            ),
            "depth_p95_noninferior": (
                raw_depth_p95 is not None
                and repaired_depth_p95 is not None
                and repaired_depth_p95 <= raw_depth_p95 * 1.02
            ),
        }
        gate_rows.append(
            {
                "session": summary["session"],
                "mode": summary["mode"],
                "gates": gates,
                "passed": all(gates.values()),
                "practical_applicability_warning": (
                    raw["observed_anonymous_fraction"] < 0.5
                ),
            }
        )
    formal_pass = bool(gate_rows) and all(row["passed"] for row in gate_rows)
    practical_pass = formal_pass and not any(
        row["practical_applicability_warning"] for row in gate_rows
    )
    registry = {
        "schema_version": "aim-stack.linux-repair-observer-validation/1",
        "claim": "truth joined after runtime outputs for scoring only; test remains sealed",
        "sessions": len(summaries),
        "validation_sessions": len(gate_rows),
        "match_gate_px": args.match_gate_px,
        "truth_runtime_input": False,
        "gate_rows": gate_rows,
        "pre_registered_gate_passed": formal_pass,
        "practical_applicability_passed": practical_pass,
        "test_access_authorized": practical_pass,
        "decision": (
            "authorize sealed test without changing thresholds"
            if practical_pass
            else "reject sealed test access; retain validation failure and investigate applicability"
        ),
        "artifacts": {
            sample_path.name: {"bytes": sample_path.stat().st_size, "sha256": sha256(sample_path)},
            rows_path.name: {"bytes": rows_path.stat().st_size, "sha256": sha256(rows_path)},
        },
    }
    registry_path = output / "validation_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    (output / "retention_manifest.json").write_text(
        json.dumps(
            {
                "classification": "long_term_private_validation_evidence",
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
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
