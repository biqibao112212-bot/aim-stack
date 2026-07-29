"""Audit raw fixed-6-mm Stage-3 capture coverage against exact exposure truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error


def _key(record: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(record["producer_epoch"]),
        int(record["frame_seq"]),
        int(record["timestamp_ns"]),
    )


def _selected_distance_m(truth: dict[str, Any]) -> float | None:
    if not truth.get("has_exact_exposure_truth", False):
        return None
    exposure = truth.get("exposure_state") or {}
    camera = exposure.get("camera_position_world_m")
    selected_target_id = truth.get("selected_target_id")
    targets = (truth.get("ground_truth") or {}).get("targets") or []
    selected = next(
        (target for target in targets if target.get("target_id") == selected_target_id),
        None,
    )
    if camera is None or selected is None:
        return None
    target = selected.get("world_position_m")
    if target is None or len(camera) != 3 or len(target) != 3:
        return None
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(target, camera)))


def _session_metrics(
    manifest: dict[str, Any], result_path: Path
) -> dict[str, Any]:
    result = _read_json(result_path)
    if result.get("session_id") != manifest.get("session_id"):
        raise ValueError(f"session mismatch in {result_path}")

    observations_path = Path(result["observations"])
    truth_path = Path(result["truth"])
    observations: dict[tuple[int, int, int], dict[str, Any]] = {}
    profiles: Counter[str] = Counter()
    frames = valid_any = target3 = empty = wrong_only = 0

    for observation in _read_jsonl(observations_path):
        frames += 1
        key = _key(observation)
        if key in observations:
            raise ValueError(f"duplicate observation key in {observations_path}: {key}")
        observations[key] = observation
        profiles[str(observation.get("camera_profile_id", "missing"))] += 1
        armors = observation.get("armors") or []
        valid_armors = [armor for armor in armors if armor.get("valid", False)]
        has_valid = bool(valid_armors)
        has_target3 = any(int(armor.get("detector_number", -1)) == 3 for armor in valid_armors)
        valid_any += int(has_valid)
        target3 += int(has_target3)
        empty += int(not armors)
        wrong_only += int(has_valid and not has_target3)

    distances: list[float] = []
    joined_exact = 0
    truth_frames = 0
    for truth in _read_jsonl(truth_path):
        truth_frames += 1
        key = _key(truth)
        if key not in observations:
            continue
        distance = _selected_distance_m(truth)
        if distance is not None:
            joined_exact += 1
            distances.append(distance)

    if frames == 0:
        raise ValueError(f"no observations in {observations_path}")
    if joined_exact != frames:
        raise ValueError(
            f"exact truth join incomplete for {manifest['session_id']}: "
            f"{joined_exact}/{frames} observations (truth frames={truth_frames})"
        )

    return {
        "session_id": manifest["session_id"],
        "mode": manifest["mode"],
        "frames": frames,
        "camera_profiles": dict(sorted(profiles.items())),
        "result_camera_profile": result.get("camera_profile"),
        "result_dual_focal": result.get("dual_focal"),
        "valid_any_frames": valid_any,
        "target3_frames": target3,
        "empty_frames": empty,
        "wrong_only_frames": wrong_only,
        "valid_any_pct": 100.0 * valid_any / frames,
        "target3_pct": 100.0 * target3 / frames,
        "empty_pct": 100.0 * empty / frames,
        "wrong_only_pct": 100.0 * wrong_only / frames,
        "exact_truth_joined": joined_exact,
        "distance_min_m": min(distances),
        "distance_max_m": max(distances),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frames = sum(int(row["frames"]) for row in rows)
    if frames == 0:
        raise ValueError("cannot aggregate zero frames")
    return {
        "sessions": len(rows),
        "frames": frames,
        "valid_any_pct": 100.0 * sum(int(row["valid_any_frames"]) for row in rows) / frames,
        "target3_pct": 100.0 * sum(int(row["target3_frames"]) for row in rows) / frames,
        "empty_pct": 100.0 * sum(int(row["empty_frames"]) for row in rows) / frames,
        "wrong_only_pct": 100.0 * sum(int(row["wrong_only_frames"]) for row in rows) / frames,
        "distance_min_m": min(float(row["distance_min_m"]) for row in rows),
        "distance_max_m": max(float(row["distance_max_m"]) for row in rows),
    }


def analyze(manifest_path: Path, evidence_root: Path) -> dict[str, Any]:
    manifest_rows = list(_read_jsonl(manifest_path))
    if not manifest_rows:
        raise ValueError(f"empty manifest: {manifest_path}")
    rows = []
    for manifest in manifest_rows:
        result_path = evidence_root / manifest["session_id"] / "session_result.json"
        if not result_path.is_file():
            raise ValueError(f"missing session result: {result_path}")
        rows.append(_session_metrics(manifest, result_path))

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["mode"])].append(row)
    aggregates = {mode: _aggregate(group) for mode, group in sorted(groups.items())}
    aggregates["overall"] = _aggregate(rows)

    all_fixed_6mm = all(
        row["camera_profiles"] == {"wide_6mm": row["frames"]}
        and row["result_camera_profile"] == "wide_6mm"
        and row["result_dual_focal"] is False
        for row in rows
    )
    return {
        "schema_version": "stage3-fixed-6mm-capture-audit-v1",
        "manifest": str(manifest_path.resolve()),
        "evidence_root": str(evidence_root.resolve()),
        "all_fixed_6mm": all_fixed_6mm,
        "sessions": rows,
        "aggregates": aggregates,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "session_id",
        "mode",
        "frames",
        "valid_any_pct",
        "target3_pct",
        "empty_pct",
        "wrong_only_pct",
        "distance_min_m",
        "distance_max_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    report = analyze(args.manifest, args.evidence_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(args.output_csv, report["sessions"])
    print(json.dumps(report["aggregates"], indent=2, ensure_ascii=False))
    print(f"all_fixed_6mm={report['all_fixed_6mm']}")


if __name__ == "__main__":
    main()
