#!/usr/bin/env python3
"""Build a manifest-bound detector-row dataset from qualified Release sessions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


SCHEMA = "aim-stack.corner-repair-detector-dataset/1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--detector-script", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), action="append")
    parser.add_argument("--test-authorization", type=Path)
    parser.add_argument("--match-rms-px", type=float, default=25.0)
    parser.add_argument("--artifact-tag", default="match25-v2")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def validate_test_authorization(path: Path | None, detector_model_sha256: str) -> dict[str, object] | None:
    if path is None:
        return None
    resolved = path.resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if value.get("schema_version") != "aim-stack.corner-repair-validation-result/1":
        raise ValueError("test authorization is not a formal validation result")
    if value.get("validation_gate_passed") is not True:
        raise ValueError("validation gate did not authorize test access")
    if value.get("detector_model_sha256") != detector_model_sha256:
        raise ValueError("test authorization detector hash does not match --model")
    checkpoint_sha256 = value.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise ValueError("test authorization does not bind a repair checkpoint")
    return {
        "path": str(resolved), "sha256": sha256(resolved),
        "repair_checkpoint_sha256": checkpoint_sha256,
    }


def qualified_exposures(result: dict[str, object]) -> int:
    match = re.search(r"(?:^|\s)exposures=(\d+)(?:\s|$)", str(result.get("validator_stdout", "")))
    if match is None:
        raise ValueError("qualified session validator output lacks exposure count")
    return int(match.group(1))


def inspect_rows(path: Path, expected_model_sha256: str, minimum_frame_seq: int,
                 maximum_match_rms_px: float) -> dict[str, object]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no detector rows: {path}")
    if {row["model_sha256"] for row in rows} != {expected_model_sha256}:
        raise ValueError(f"detector model hash mismatch: {path}")
    if any(int(row["frame_seq"]) < minimum_frame_seq for row in rows):
        raise ValueError(f"row precedes Scene Control eligibility boundary: {path}")
    if any(row["future_truth_included"] != "False" for row in rows):
        raise ValueError(f"future truth contract violated: {path}")
    if any(float(row["match_corner_rms_px"]) > maximum_match_rms_px for row in rows):
        raise ValueError(f"row exceeds declared detector-to-truth association gate: {path}")
    identities = {(row["producer_epoch"], row["frame_seq"], row["timestamp_ns"]) for row in rows}
    return {
        "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
        "rows": len(rows), "uniform_rows": sum(row["motion_uniform"] == "True" for row in rows),
        "matched_exposures": len(identities),
    }


def main() -> None:
    args = parse_args()
    plan_path = args.plan.resolve(strict=True)
    root = args.collection_root.resolve(strict=True)
    model = args.model.resolve(strict=True)
    detector_script = args.detector_script.resolve(strict=True)
    if args.match_rms_px <= 0:
        raise ValueError("--match-rms-px must be positive")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.artifact_tag) is None:
        raise ValueError("--artifact-tag must contain only lowercase letters, digits and hyphens")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    collection_plan = json.loads((root / "collection-plan.json").read_text(encoding="utf-8"))
    if collection_plan["plan_sha256"] != sha256(plan_path):
        raise ValueError("collection root is bound to a different plan")
    if sha256(model) != plan["detector_model_sha256"]:
        raise ValueError("detector model does not match the predeclared plan")
    splits = set(args.split or ("train", "validation"))
    if "test" in splits and args.test_authorization is None:
        raise PermissionError("test rows require --test-authorization from a passing validation gate")
    if args.test_authorization is not None and "test" not in splits:
        raise ValueError("--test-authorization is only valid when --split test is requested")
    authorization = None
    if "test" in splits:
        authorization = validate_test_authorization(args.test_authorization, sha256(model))

    entries: list[dict[str, object]] = []
    for planned in plan["sessions"]:
        if planned["split"] not in splits:
            continue
        session_dir = root / planned["id"]
        result_path = session_dir / "session-result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["planned"] != planned:
            raise ValueError(f"session result does not match plan: {planned['id']}")
        for name, recorded in result["artifacts"].items():
            actual = session_dir / name
            if sha256(actual) != recorded["sha256"] or actual.stat().st_size != recorded["bytes"]:
                raise ValueError(f"qualified session artifact changed: {actual}")
        rows_path = session_dir / f"detector-matched-rows-rgba-{args.artifact_tag}.csv"
        if not rows_path.exists():
            command = [
                sys.executable, str(detector_script), "--session", str(session_dir),
                "--labels", str(session_dir / "exact-corners.jsonl"), "--model", str(model),
                "--output", str(rows_path), "--minimum-frame-seq", str(result["first_eligible_frame_seq"]),
                "--score-threshold", "0.25", "--match-rms-px", str(args.match_rms_px),
            ]
            subprocess.run(command, check=True)
        row_summary = inspect_rows(
            rows_path, sha256(model), int(result["first_eligible_frame_seq"]), args.match_rms_px,
        )
        exposures = qualified_exposures(result)
        entry = {
            "session_id": planned["id"], "split": planned["split"], "mode": planned["mode"],
            "radial_scale": planned["radial_scale"], "first_eligible_frame_seq": result["first_eligible_frame_seq"],
            "session_result": {"path": str(result_path), "sha256": sha256(result_path)},
            "qualified_exposures": exposures,
            "detector_exposure_coverage_fraction": row_summary["matched_exposures"] / exposures,
            "rows": row_summary,
        }
        entries.append(entry)
    expected = sum(1 for item in plan["sessions"] if item["split"] in splits)
    if len(entries) != expected:
        raise ValueError(f"expected {expected} declared sessions, built {len(entries)}")
    prefix = "test-detector-dataset-manifest" if splits == {"test"} else "detector-dataset-manifest"
    output = root / f"{prefix}-{args.artifact_tag}.json"
    write_new_json(output, {
        "schema_version": SCHEMA, "plan": str(plan_path), "plan_sha256": sha256(plan_path),
        "collection_root": str(root), "detector_script": str(detector_script),
        "detector_script_sha256": sha256(detector_script), "detector_model": str(model),
        "detector_model_sha256": sha256(model), "match_rms_px": args.match_rms_px,
        "artifact_tag": args.artifact_tag, "splits": sorted(splits), "sessions": entries,
        "test_accessed": "test" in splits, "test_authorization": authorization,
    })
    print(json.dumps({"manifest": str(output), "sessions": len(entries), "splits": sorted(splits)}, indent=2))


if __name__ == "__main__":
    main()
