"""Audit formal ACK-bound multistate history and per-segment survival.

The audit is deliberately train/validation-only.  Test session names and shard
descriptors are counted from manifests, but neither test shards nor test raw
records are opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


AUDIT_SCHEMA = "stage3-multistate-segment-survival-audit-v1"
HISTORY_CONTRACT_ID = "formal-ack-bound-latest-32-valid-observation-events-v1"
ALLOWED_SPLITS = ("train", "validation")
EXPECTED_STAGE_SCHEMAS = {
    "base": "stage3-dataset-v3",
    "truth": "stage3-truth-history-v1",
    "causal": "stage3-causal-physical-v1",
    "clean": "stage3-observable-future-v1",
    "pnp": "stage3-observable-future-real-pnp-observed-stream-v2",
    "pnp_sf": "stage3-observable-future-real-pnp-sf-observed-stream-v2",
}
MOTION_CLASS_BY_MODE = {
    "stationary": 0,
    "linear": 1,
    "spin": 2,
    "linear_and_spin": 3,
}
CORE_FIELDS = (
    "motion_command_epoch",
    "motion_segment_start_ns",
    "motion_segment_end_ns",
    "history_start_ns",
    "future_end_ns",
    "window_constant_motion",
    "motion_class",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            yield value


def _manifest_path(root: Path, value: object) -> Path:
    return root / Path(str(value).replace("\\", "/"))


def _load_manifest(root: Path) -> tuple[Path, dict[str, Any], str]:
    path = root / "dataset_manifest.json"
    return path, _json(path), _sha256(path)


def _selected_shards(
    root: Path,
    manifest: Mapping[str, Any],
    expected_split_by_session: Mapping[str, str],
) -> tuple[list[tuple[str, str, Path, Mapping[str, Any]]], int]:
    selected: list[tuple[str, str, Path, Mapping[str, Any]]] = []
    test_descriptors = 0
    for item in manifest.get("shards", ()):
        if not isinstance(item, Mapping):
            raise ValueError(f"invalid shard descriptor in {root}")
        split = str(item.get("split", ""))
        sessions = item.get("session_ids")
        if not isinstance(sessions, list) or len(sessions) != 1:
            raise ValueError(f"formal audit requires one-session shards in {root}")
        session_id = str(sessions[0])
        expected_split = expected_split_by_session.get(session_id)
        if expected_split is None or split != expected_split:
            raise ValueError(
                f"shard descriptor differs from frozen split before open: "
                f"{session_id}/{split}/{expected_split}"
            )
        if split == "test":
            test_descriptors += 1
            continue
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"unexpected split {split!r} in {root}")
        selected.append((split, session_id, _manifest_path(root, item["path"]), item))
    return selected, test_descriptors


def _row_key(split: str, session_id: object, t0_ns: object) -> tuple[str, str, int]:
    return split, str(session_id), int(t0_ns)


def _active_times(
    arrays: Mapping[str, np.ndarray],
    row: int,
    time_key: str,
    mask_key: str,
) -> np.ndarray:
    mask = np.asarray(arrays[mask_key][row], dtype=np.bool_)
    times = np.asarray(arrays[time_key][row], dtype=np.float64)
    if mask.shape != times.shape:
        raise ValueError(f"{time_key}/{mask_key} shape mismatch")
    active = times[mask]
    if active.size and np.any(np.diff(active) <= 0.0):
        raise ValueError(f"{time_key} is not strictly increasing")
    return active


def _require_right_aligned(mask: np.ndarray, stage: str) -> None:
    active = np.flatnonzero(mask)
    if active.size and not np.array_equal(
        active, np.arange(mask.size - active.size, mask.size)
    ):
        raise ValueError(f"{stage} history mask is not a contiguous right-aligned suffix")


def _is_subset_times(child: np.ndarray, parent: np.ndarray, atol_s: float = 2e-6) -> bool:
    if child.size == 0:
        return True
    if parent.size == 0:
        return False
    return all(float(np.min(np.abs(parent - value))) <= atol_s for value in child)


def _history_bin(count: int) -> str:
    if 8 <= count <= 15:
        return "8_15"
    if 16 <= count <= 23:
        return "16_23"
    if 24 <= count <= 31:
        return "24_31"
    if count == 32:
        return "32"
    return "other"


def _history_view(stage: str, arrays: Mapping[str, np.ndarray], row: int) -> np.ndarray | None:
    if stage in {"base", "truth", "causal"}:
        value = _active_times(arrays, row, "event_time_s", "event_mask")
        mask = np.asarray(arrays["event_mask"][row], dtype=np.bool_)
        if mask.shape != (200,):
            raise ValueError(f"{stage} history tensor capacity is not 200")
        _require_right_aligned(mask, stage)
        return value
    if stage == "clean":
        value = _active_times(arrays, row, "history_time_s", "history_mask")
        mask = np.asarray(arrays["history_mask"][row], dtype=np.bool_)
        if mask.shape != (32,):
            raise ValueError("clean history tensor capacity is not 32")
        _require_right_aligned(mask, stage)
        return value
    if stage == "pnp":
        value = _active_times(arrays, row, "pnp_history_time_s", "pnp_history_mask")
        mask = np.asarray(arrays["pnp_history_mask"][row], dtype=np.bool_)
        if mask.shape != (32,):
            raise ValueError("pnp history tensor capacity is not 32")
        _require_right_aligned(mask, stage)
        return value
    if stage == "pnp_sf":
        value = _active_times(arrays, row, "pnp_s_event_time_s", "pnp_s_event_mask")
        mask = np.asarray(arrays["pnp_s_event_mask"][row], dtype=np.bool_)
        if mask.shape != (32,):
            raise ValueError("pnp_sf history tensor capacity is not 32")
        _require_right_aligned(mask, stage)
        return value
    return None


def _load_stage(
    stage: str,
    root: Path,
    expected_capture_sha: str,
    expected_split_by_session: Mapping[str, str],
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, Any]]:
    manifest_path, manifest, manifest_sha = _load_manifest(root)
    if manifest.get("qualification_passed") is not True:
        raise ValueError(f"{stage} is not qualified")
    if manifest.get("schema_version") != EXPECTED_STAGE_SCHEMAS[stage]:
        raise ValueError(f"{stage} schema version mismatch")
    if str(manifest.get("capture_contract_sha256", "")) != expected_capture_sha:
        raise ValueError(f"{stage} capture contract hash mismatch")
    if stage != "base" and manifest.get("test_accessed") is not False:
        raise ValueError(f"{stage} is not test-sealed")
    if stage == "base":
        history = str(manifest.get("tensor_contract", {}).get("history_selection", ""))
        if "formal_ack_bound_latest_32" not in history:
            raise ValueError("base does not declare the formal latest-32 history contract")
        if int(manifest.get("tensor_contract", {}).get(
            "formal_motion_history_event_limit", -1
        )) != 32:
            raise ValueError("base formal history limit is not 32")
    selected, test_descriptors = _selected_shards(
        root, manifest, expected_split_by_session
    )
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    shard_count = 0
    for split, descriptor_session, path, item in selected:
        if _sha256(path) != str(item.get("sha256", "")):
            raise ValueError(f"{stage} shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        required = {"session_id", "t0_ns", *CORE_FIELDS}
        if stage != "base":
            required.add("rule_query")
        missing = required - arrays.keys()
        if missing:
            raise ValueError(f"{stage} shard missing {sorted(missing)}")
        sample_count = int(arrays["t0_ns"].shape[0])
        if sample_count != int(item.get("sample_count", -1)):
            raise ValueError(f"{stage} shard count mismatch: {path}")
        for row in range(sample_count):
            key = _row_key(split, arrays["session_id"][row], arrays["t0_ns"][row])
            if key[1] != descriptor_session:
                raise ValueError(f"{stage} row session differs from its shard descriptor")
            if key in rows:
                raise ValueError(f"duplicate {stage} row: {key}")
            if stage == "base":
                event_mask = np.asarray(arrays["event_mask"][row], dtype=np.bool_)
                active_indices = np.flatnonzero(event_mask)
                if active_indices.size and not np.array_equal(
                    active_indices,
                    np.arange(event_mask.size - active_indices.size, event_mask.size),
                ):
                    raise ValueError(f"base latest-32 history is not right aligned: {key}")
            core = {field: np.asarray(arrays[field][row]).copy() for field in CORE_FIELDS}
            if not bool(core["window_constant_motion"]):
                raise ValueError(f"{stage} row is not constant-motion: {key}")
            history = _history_view(stage, arrays, row)
            if history is not None:
                if stage in {"base", "truth", "causal"} and not 8 <= history.size <= 32:
                    raise ValueError(f"{stage} history is outside 8..32 events: {key}")
                if stage == "clean" and history.size != 32:
                    raise ValueError(f"clean row does not contain 32 qualified events: {key}")
                if stage in {"pnp", "pnp_sf"} and history.size > 32:
                    raise ValueError(f"{stage} history exceeds 32 associated events: {key}")
            row_value: dict[str, Any] = {
                "core": core,
                "history_time_s": history,
            }
            if stage != "base":
                rule_query = np.asarray(arrays["rule_query"][row], dtype=np.bool_)
                if rule_query.ndim != 1 or rule_query.size == 0 or not rule_query.all():
                    raise ValueError(f"{stage} row does not admit every rule query: {key}")
                row_value["rule_query"] = rule_query
            if stage == "pnp":
                usable = bool(arrays["pnp_forward_usable"][row])
                if usable and history.size < int(manifest["minimum_history_events"]):
                    raise ValueError(f"usable PnP row has too few associated events: {key}")
                row_value["usable"] = usable
            elif stage == "pnp_sf":
                pnp_usable = bool(arrays["pnp_forward_usable"][row])
                pnp_s_usable = bool(arrays["pnp_s_forward_usable"][row])
                common_usable = bool(arrays["pnp_sf_common_usable"][row])
                if pnp_s_usable and history.size < int(manifest["minimum_history_events"]):
                    raise ValueError(f"usable PnP/S row has too few associated events: {key}")
                if common_usable != (pnp_usable and pnp_s_usable):
                    raise ValueError(f"PnP/SF common usability is inconsistent: {key}")
                row_value["pnp_s_usable"] = pnp_s_usable
                row_value["common_usable"] = common_usable
            rows[key] = row_value
        shard_count += 1
    return rows, {
        "path": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "schema_version": manifest.get("schema_version"),
        "opened_shards": shard_count,
        "test_shard_descriptors_seen": test_descriptors,
        "test_shards_opened": 0,
        "row_count": len(rows),
    }


def _same_value(left: object, right: object) -> bool:
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _validate_lineage(
    stages: Mapping[str, Mapping[tuple[str, str, int], Mapping[str, Any]]],
) -> None:
    order = ("base", "truth", "causal", "clean", "pnp", "pnp_sf")
    for parent_name, child_name in zip(order, order[1:]):
        parent = stages[parent_name]
        child = stages[child_name]
        missing = set(child) - set(parent)
        if missing:
            raise ValueError(f"{child_name} contains rows outside {parent_name}: {len(missing)}")
        for key, child_row in child.items():
            parent_row = parent[key]
            for field in CORE_FIELDS:
                if not _same_value(child_row["core"][field], parent_row["core"][field]):
                    raise ValueError(f"{child_name} {field} differs from {parent_name}: {key}")
            if parent_name != "base" and not _same_value(
                child_row["rule_query"], parent_row["rule_query"]
            ):
                raise ValueError(f"{child_name} rule_query differs from {parent_name}: {key}")
    base = stages["base"]
    for stage_name, rows in stages.items():
        if stage_name == "base":
            continue
        for key, row in rows.items():
            times = row.get("history_time_s")
            if times is not None and not _is_subset_times(times, base[key]["history_time_s"]):
                raise ValueError(f"{stage_name} history leaves base latest-32 history: {key}")


def _validate_manifest_lineage(
    manifests: Mapping[str, Mapping[str, Any]],
    reports: Mapping[str, Mapping[str, Any]],
) -> None:
    expected = {
        "truth": {
            "source_dataset_manifest_sha256": "base",
        },
        "causal": {
            "source_observation_manifest_sha256": "base",
            "source_truth_history_manifest_sha256": "truth",
        },
        "clean": {
            "source_dataset_manifest_sha256": "causal",
            "truth_history_manifest_sha256": "truth",
        },
        "pnp": {
            "source_v3_manifest_sha256": "base",
            "truth_history_manifest_sha256": "truth",
            "causal_physical_manifest_sha256": "causal",
            "clean_dataset_manifest_sha256": "clean",
        },
        "pnp_sf": {
            "source_v3_manifest_sha256": "base",
            "truth_history_manifest_sha256": "truth",
            "causal_physical_manifest_sha256": "causal",
            "parent_paired_manifest_sha256": "pnp",
        },
    }
    for stage, fields in expected.items():
        for field, parent in fields.items():
            if manifests[stage].get(field) != reports[parent]["manifest_sha256"]:
                raise ValueError(f"{stage} manifest does not bind {parent}: {field}")


def _validate_observation_manifest(
    manifests: Mapping[str, Mapping[str, Any]],
    reports: Mapping[str, Mapping[str, Any]],
    capture_sha: str,
) -> dict[str, Any]:
    pnp = manifests["pnp"]
    pnp_sf = manifests["pnp_sf"]
    path = Path(str(pnp.get("observation_dataset", ""))).resolve()
    if path != Path(str(pnp_sf.get("observation_dataset", ""))).resolve():
        raise ValueError("PnP and PnP/SF observation datasets differ")
    manifest_path = path / "dataset_manifest.json"
    manifest_sha = _sha256(manifest_path)
    observation = _json(manifest_path)
    if (
        pnp.get("observation_manifest_sha256") != manifest_sha
        or pnp_sf.get("observation_manifest_sha256") != manifest_sha
        or observation.get("schema_version") != "stage3-dataset-v4-observation"
        or observation.get("qualification_passed") is not True
        or observation.get("capture_contract_sha256") != capture_sha
        or observation.get("source_v3_manifest_sha256") != reports["base"]["manifest_sha256"]
        or observation.get("test_accessed") is not False
        or int(observation.get("test_shards_opened", -1)) != 0
    ):
        raise ValueError("observation derivative provenance or test sealing is invalid")
    return {
        "path": str(path),
        "manifest_sha256": manifest_sha,
        "test_shards_opened": 0,
        "test_accessed": False,
    }


def _raw_valid_history(
    observations_path: Path,
    expected_sha: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if _sha256(observations_path) != expected_sha:
        raise ValueError(f"raw observation hash mismatch: {observations_path}")
    valid_timestamps: list[int] = []
    frame_count = 0
    profile_count: Counter[str] = Counter()
    for record in _jsonl(observations_path):
        frame_count += 1
        profile_count[str(record.get("camera_profile_id", ""))] += 1
        armors = record.get("armors", ())
        valid = False
        if isinstance(armors, list):
            for armor in armors:
                if not isinstance(armor, Mapping) or not bool(armor.get("valid", False)):
                    continue
                values = [*armor.get("position_m", ()), armor.get("yaw_rad")]
                if len(values) == 4 and all(
                    isinstance(value, (int, float)) and math.isfinite(float(value))
                    for value in values
                ):
                    valid = True
                    break
        if valid:
            valid_timestamps.append(int(record["timestamp_ns"]))
    if set(profile_count) != {"wide_6mm"}:
        raise ValueError(f"raw observation contains non-6mm profile: {profile_count}")
    return np.asarray(valid_timestamps, dtype=np.int64), {
        "observation_path": str(observations_path),
        "frame_count": frame_count,
        "valid_event_count": len(valid_timestamps),
        "camera_profile_count": dict(profile_count),
    }


def _validate_base_against_raw(
    base_rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
    canonical_sources_path: Path,
    selected_sessions: set[str],
) -> tuple[dict[str, Any], int]:
    sources = {str(value["session_id"]): value for value in _jsonl(canonical_sources_path)}
    missing = selected_sessions - sources.keys()
    if missing:
        raise ValueError(f"canonical raw sources missing sessions: {sorted(missing)}")
    reports: dict[str, Any] = {}
    checked_rows = 0
    for session_id in sorted(selected_sessions):
        source = sources[session_id]
        timestamps, report = _raw_valid_history(
            Path(str(source["observations"])), str(source["observation_sha256"])
        )
        session_rows = [
            (key, row) for key, row in base_rows.items() if key[1] == session_id
        ]
        for key, row in session_rows:
            t0_ns = key[2]
            core = row["core"]
            start_ns = int(core["motion_segment_start_ns"])
            end_ns = int(core["motion_segment_end_ns"])
            available = timestamps[
                (timestamps >= start_ns) & (timestamps <= t0_ns) & (timestamps < end_ns)
            ]
            expected = available[-32:]
            actual_time_s = np.asarray(row["history_time_s"], dtype=np.float64)
            actual = t0_ns + np.rint(actual_time_s * 1e9).astype(np.int64)
            if expected.size != actual.size:
                raise ValueError(f"base history count is not latest min(32, available): {key}")
            if expected.size and np.max(np.abs(expected - actual), initial=0) > 2_000:
                raise ValueError(f"base history timestamps differ from raw latest-32: {key}")
            if expected.size and int(core["history_start_ns"]) != int(expected[0]):
                raise ValueError(f"base history_start_ns differs from raw latest-32: {key}")
            if int(core["history_start_ns"]) - start_ns < 2_000:
                raise ValueError(f"base history violates the 2us start guard: {key}")
            if end_ns - 1 - int(core["future_end_ns"]) < 2_000:
                raise ValueError(f"base future violates the 2us end guard: {key}")
            checked_rows += 1
        report["audited_base_row_count"] = len(session_rows)
        reports[session_id] = report
    return reports, checked_rows


def _plan_rows(
    contract: Mapping[str, Any],
    canonical_sources_path: Path,
    family_by_session: Mapping[str, str],
) -> list[dict[str, Any]]:
    split_by_session = {
        session_id: split
        for split in ALLOWED_SPLITS
        for session_id in contract["splits"][split]
    }
    sources = {str(value["session_id"]): value for value in _jsonl(canonical_sources_path)}
    rows: list[dict[str, Any]] = []
    for session_id, split in split_by_session.items():
        source = sources.get(session_id)
        if source is None:
            raise ValueError(f"capture plan missing canonical source: {session_id}")
        for segment in source.get("motion_segments", ()):
            mode = str(segment["mode"])
            if mode not in MOTION_CLASS_BY_MODE:
                raise ValueError(f"unknown canonical motion mode: {mode}")
            rows.append({
                "split": split,
                "session_id": session_id,
                "motion_family": family_by_session[session_id],
                "motion_command_epoch": int(segment["motion_command_epoch"]),
                "mode": mode,
                "motion_class": MOTION_CLASS_BY_MODE[mode],
                "motion_segment_start_ns": int(segment["start_timestamp_ns"]),
                "motion_segment_end_ns": int(segment["end_timestamp_ns"]),
                "active": mode != "stationary",
            })
    return sorted(rows, key=lambda row: (
        row["split"], row["session_id"], row["motion_command_epoch"]
    ))


def _validate_base_against_plan(
    base_rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
    plan: list[dict[str, Any]],
) -> None:
    plan_by_epoch = {
        (row["split"], row["session_id"], row["motion_command_epoch"]): row
        for row in plan
    }
    if len(plan_by_epoch) != len(plan):
        raise ValueError("canonical plan contains duplicate session epochs")
    seen_sessions: set[tuple[str, str]] = set()
    for key, row in base_rows.items():
        core = row["core"]
        epoch_key = (key[0], key[1], int(core["motion_command_epoch"]))
        planned = plan_by_epoch.get(epoch_key)
        if planned is None:
            raise ValueError(f"base row references an unplanned motion epoch: {key}")
        if (
            int(core["motion_segment_start_ns"]) != planned["motion_segment_start_ns"]
            or int(core["motion_segment_end_ns"]) != planned["motion_segment_end_ns"]
            or int(core["motion_class"]) != planned["motion_class"]
        ):
            raise ValueError(f"base row differs from canonical ACK segment: {key}")
        if not (
            planned["motion_segment_start_ns"] <= key[2]
            < planned["motion_segment_end_ns"]
        ):
            raise ValueError(f"base anchor is outside its canonical ACK segment: {key}")
        seen_sessions.add((key[0], key[1]))
    planned_sessions = {
        (row["split"], row["session_id"]) for row in plan
    }
    missing_sessions = planned_sessions - seen_sessions
    if missing_sessions:
        raise ValueError(f"canonical train/validation sessions have no base rows: {sorted(missing_sessions)}")


def audit(args: argparse.Namespace) -> Path:
    contract_path = Path(args.capture_contract).resolve()
    contract = _json(contract_path)
    contract_sha = _sha256(contract_path)
    if (
        contract.get("schema_version") != "stage3-multistate-capture-contract-v1"
        or contract.get("camera_profile") != "wide_6mm"
        or contract.get("dual_focal") is not False
    ):
        raise ValueError("capture contract is not the fixed wide-6mm formal contract")
    formal_manifest_path = Path(str(contract["formal_manifest"])).resolve()
    if _sha256(formal_manifest_path) != str(contract.get("formal_manifest_sha256", "")):
        raise ValueError("formal capture manifest hash mismatch")
    family_by_session = {
        str(record["session_id"]): str(record["motion_family"])
        for record in _jsonl(formal_manifest_path)
    }
    expected_split_by_session = {
        str(session_id): split
        for split in ("train", "validation", "test")
        for session_id in contract["splits"][split]
    }
    if set(family_by_session) != set(expected_split_by_session):
        raise ValueError("formal manifest sessions differ from the frozen split contract")

    roots = {
        "base": Path(args.base).resolve(),
        "truth": Path(args.truth_history).resolve(),
        "causal": Path(args.causal_physical).resolve(),
        "clean": Path(args.observable_clean).resolve(),
        "pnp": Path(args.pnp).resolve(),
        "pnp_sf": Path(args.pnp_sf).resolve(),
    }
    stage_rows: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    stage_reports: dict[str, Any] = {}
    manifests: dict[str, dict[str, Any]] = {}
    test_descriptors = 0
    for stage, root in roots.items():
        stage_rows[stage], stage_reports[stage] = _load_stage(
            stage, root, contract_sha, expected_split_by_session
        )
        manifests[stage] = _json(root / "dataset_manifest.json")
        test_descriptors += int(stage_reports[stage]["test_shard_descriptors_seen"])
    _validate_lineage(stage_rows)
    _validate_manifest_lineage(manifests, stage_reports)
    observation_report = _validate_observation_manifest(
        manifests, stage_reports, contract_sha
    )

    base_manifest = _json(roots["base"] / "dataset_manifest.json")
    canonical_sources_path = _manifest_path(
        roots["base"], base_manifest["canonical_sources"]
    )
    if base_manifest.get("artifact_sha256", {}).get(
        "canonical_sources"
    ) != _sha256(canonical_sources_path):
        raise ValueError("base canonical_sources artifact hash mismatch")
    plan = _plan_rows(contract, canonical_sources_path, family_by_session)
    _validate_base_against_plan(stage_rows["base"], plan)
    selected_sessions = {
        str(session_id)
        for split in ALLOWED_SPLITS
        for session_id in contract["splits"][split]
    }
    raw_reports, raw_checked_rows = _validate_base_against_raw(
        stage_rows["base"], canonical_sources_path, selected_sessions
    )

    counts: dict[str, Counter[tuple[str, str, int]]] = {
        stage: Counter((key[0], key[1], int(row["core"]["motion_command_epoch"]))
                       for key, row in rows.items())
        for stage, rows in stage_rows.items()
    }
    pnp_usable: Counter[tuple[str, str, int]] = Counter()
    for key, row in stage_rows["pnp"].items():
        if row["usable"]:
            pnp_usable[(key[0], key[1], int(row["core"]["motion_command_epoch"]))] += 1
    pnp_s_usable: Counter[tuple[str, str, int]] = Counter()
    common_usable: Counter[tuple[str, str, int]] = Counter()
    for key, row in stage_rows["pnp_sf"].items():
        epoch_key = (key[0], key[1], int(row["core"]["motion_command_epoch"]))
        if row["pnp_s_usable"]:
            pnp_s_usable[epoch_key] += 1
        if row["common_usable"]:
            common_usable[epoch_key] += 1
    base_history_bins: dict[tuple[str, str, int], Counter[str]] = defaultdict(Counter)
    total_history_bins: Counter[str] = Counter()
    for key, row in stage_rows["base"].items():
        epoch_key = (key[0], key[1], int(row["core"]["motion_command_epoch"]))
        bin_name = _history_bin(int(row["history_time_s"].size))
        base_history_bins[epoch_key][bin_name] += 1
        total_history_bins[bin_name] += 1

    csv_rows: list[dict[str, Any]] = []
    for plan_row in plan:
        epoch_key = (
            plan_row["split"], plan_row["session_id"],
            plan_row["motion_command_epoch"],
        )
        csv_rows.append({
            **plan_row,
            "base": counts["base"][epoch_key],
            "base_history_8_15": base_history_bins[epoch_key]["8_15"],
            "base_history_16_23": base_history_bins[epoch_key]["16_23"],
            "base_history_24_31": base_history_bins[epoch_key]["24_31"],
            "base_history_32": base_history_bins[epoch_key]["32"],
            "base_history_other": base_history_bins[epoch_key]["other"],
            "truth": counts["truth"][epoch_key],
            "causal": counts["causal"][epoch_key],
            "clean": counts["clean"][epoch_key],
            "pnp": counts["pnp"][epoch_key],
            "pnp_forward_usable": pnp_usable[epoch_key],
            "pnp_sf": counts["pnp_sf"][epoch_key],
            "pnp_s_forward_usable": pnp_s_usable[epoch_key],
            "pnp_sf_common_usable": common_usable[epoch_key],
        })

    minimum = int(contract["post_capture_requirements"][
        "minimum_heldout_active_segments_with_samples"
    ])
    validation_sessions: dict[str, dict[str, Any]] = {}
    for session_id in contract["splits"]["validation"]:
        session_rows = [
            row for row in csv_rows
            if row["session_id"] == session_id and row["active"]
        ]
        survived = sum(row["pnp_sf_common_usable"] > 0 for row in session_rows)
        validation_sessions[str(session_id)] = {
            "active_segment_count": len(session_rows),
            "survived_active_segment_count": survived,
            "required_survived_active_segment_count": minimum,
            "passed": survived >= minimum,
        }
    gate_passed = all(value["passed"] for value in validation_sessions.values())

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite survival audit: {output}")
    output.mkdir(parents=True)
    csv_path = output / "segment-survival.csv"
    fieldnames = list(csv_rows[0]) if csv_rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    result = {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed" if gate_passed else "failed",
        "history_contract_id": HISTORY_CONTRACT_ID,
        "history_capacity": 32,
        "capture_contract": str(contract_path),
        "capture_contract_sha256": contract_sha,
        "stages": stage_reports,
        "observation_stage": observation_report,
        "raw_camera_scan": raw_reports,
        "raw_base_rows_checked": raw_checked_rows,
        "base_history_bin_counts": dict(sorted(total_history_bins.items())),
        "derived_history_outside_base_count": 0,
        "segment_row_count": len(csv_rows),
        "active_segment_count": sum(bool(row["active"]) for row in csv_rows),
        "validation_sessions": validation_sessions,
        "validation_gate_passed": gate_passed,
        "minimum_validation_active_segments_with_samples": minimum,
        "test_session_metadata_count": len(contract["splits"]["test"]),
        "test_shard_descriptors_seen": test_descriptors,
        "test_shards_opened": 0,
        "test_raw_opened": 0,
        "test_accessed": False,
        "segment_csv": csv_path.name,
    }
    result_path = output / "audit.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-contract", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--truth-history", required=True)
    parser.add_argument("--causal-physical", required=True)
    parser.add_argument("--observable-clean", required=True)
    parser.add_argument("--pnp", required=True)
    parser.add_argument("--pnp-sf", required=True)
    parser.add_argument("--output", required=True)
    result_path = audit(parser.parse_args())
    print(result_path)
    result = _json(result_path)
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
