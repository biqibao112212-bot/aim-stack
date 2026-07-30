from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from training.stage3.audit_multistate_segment_survival import audit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _core_arrays(session_id: str, t0_ns: int, history_start_ns: int) -> dict[str, np.ndarray]:
    return {
        "session_id": np.asarray([session_id]),
        "t0_ns": np.asarray([t0_ns], dtype=np.int64),
        "motion_command_epoch": np.asarray([0], dtype=np.int64),
        "motion_segment_start_ns": np.asarray([t0_ns - 780_002_000], dtype=np.int64),
        "motion_segment_end_ns": np.asarray([t0_ns + 100_000_000], dtype=np.int64),
        "history_start_ns": np.asarray([history_start_ns], dtype=np.int64),
        "future_end_ns": np.asarray([t0_ns], dtype=np.int64),
        "window_constant_motion": np.asarray([True]),
        "motion_class": np.asarray([2], dtype=np.int64),
    }


def _stage_arrays(
    stage: str,
    session_id: str,
    timestamps: np.ndarray,
) -> dict[str, np.ndarray]:
    t0_ns = int(timestamps[-1])
    selected = timestamps[-32:]
    times = (selected.astype(np.float64) - t0_ns) / 1e9
    arrays = _core_arrays(session_id, t0_ns, int(selected[0]))
    rule_query = np.ones((1, 8), dtype=np.bool_)
    if stage in {"base", "truth", "causal"}:
        event_mask = np.zeros((1, 200), dtype=np.bool_)
        event_mask[:, -32:] = True
        event_time = np.zeros((1, 200), dtype=np.float32)
        event_time[:, -32:] = times
        arrays.update({"event_mask": event_mask, "event_time_s": event_time})
    if stage == "clean":
        arrays.update({
            "history_mask": np.ones((1, 32), dtype=np.bool_),
            "history_time_s": times.astype(np.float32)[None, :],
        })
    if stage == "pnp":
        associated = 0 if session_id.startswith("train") else 12
        mask = np.zeros((1, 32), dtype=np.bool_)
        history_time = np.zeros((1, 32), dtype=np.float32)
        if associated:
            mask[:, -associated:] = True
            history_time[:, -associated:] = times[-associated:]
        arrays.update({
            "pnp_history_mask": mask,
            "pnp_history_time_s": history_time,
            "pnp_forward_usable": np.asarray([associated >= 8]),
        })
    if stage == "pnp_sf":
        associated = 0 if session_id.startswith("train") else 12
        mask = np.zeros((1, 32), dtype=np.bool_)
        history_time = np.zeros((1, 32), dtype=np.float32)
        if associated:
            mask[:, -associated:] = True
            history_time[:, -associated:] = times[-associated:]
        usable = associated >= 8
        arrays.update({
            "pnp_s_event_mask": mask,
            "pnp_s_event_time_s": history_time,
            "pnp_forward_usable": np.asarray([usable]),
            "pnp_s_forward_usable": np.asarray([usable]),
            "pnp_sf_common_usable": np.asarray([usable]),
        })
    if stage != "base":
        arrays["rule_query"] = rule_query
    return arrays


def _raw_records(session_id: str, timestamps: np.ndarray) -> list[dict[str, object]]:
    return [{
        "schema_version": "stage3-observation-v2",
        "session_id": session_id,
        "producer_epoch": 1,
        "frame_seq": index,
        "timestamp_ns": int(timestamp),
        "camera_profile_id": "wide_6mm",
        "armors": [{
            "valid": True,
            "position_m": [3.0, 0.0, 0.2],
            "yaw_rad": 0.0,
        }],
    } for index, timestamp in enumerate(timestamps)]


def test_survival_audit_uses_latest_32_and_never_opens_test(tmp_path: Path) -> None:
    sessions = {"train": "train-spin", "validation": "validation-spin", "test": "test-spin"}
    contract_path = tmp_path / "capture-contract.json"
    formal_manifest_path = tmp_path / "formal-manifest.jsonl"
    _write_jsonl(formal_manifest_path, [{
        "session_id": session_id,
        "motion_family": "rotation",
    } for session_id in sessions.values()])
    contract = {
        "schema_version": "stage3-multistate-capture-contract-v1",
        "camera_profile": "wide_6mm",
        "dual_focal": False,
        "formal_manifest": str(formal_manifest_path),
        "formal_manifest_sha256": _sha256(formal_manifest_path),
        "splits": {split: [session] for split, session in sessions.items()},
        "post_capture_requirements": {
            "minimum_heldout_active_segments_with_samples": 1,
        },
    }
    _write_json(contract_path, contract)
    contract_sha = _sha256(contract_path)

    roots = {stage: tmp_path / stage for stage in (
        "base", "truth", "causal", "clean", "pnp", "pnp_sf"
    )}
    canonical: list[dict[str, object]] = []
    timestamps_by_session: dict[str, np.ndarray] = {}
    for split, session_id in sessions.items():
        t0_ns = 2_000_000_000 + len(timestamps_by_session) * 2_000_000_000
        timestamps = np.asarray([
            t0_ns - offset * 20_000_000 for offset in reversed(range(40))
        ], dtype=np.int64)
        timestamps_by_session[session_id] = timestamps
        raw_path = tmp_path / "raw" / session_id / "observations.jsonl"
        if split != "test":
            _write_jsonl(raw_path, _raw_records(session_id, timestamps))
            raw_sha = _sha256(raw_path)
        else:
            raw_sha = "test-raw-must-not-be-opened"
        canonical.append({
            "session_id": session_id,
            "observations": str(raw_path),
            "observation_sha256": raw_sha,
            "motion_segments": [{
                "motion_command_epoch": 0,
                "mode": "spin",
                "start_timestamp_ns": int(timestamps[0]) - 2_000,
                "end_timestamp_ns": int(timestamps[-1]) + 100_000_000,
            }],
        })
    _write_jsonl(roots["base"] / "canonical_sources.jsonl", canonical)

    manifest_hashes: dict[str, str] = {}
    observation_root = tmp_path / "observation"
    observation_manifest_sha = ""
    for stage, root in roots.items():
        descriptors = []
        for split, session_id in sessions.items():
            relative = Path("shards") / f"{split}-{session_id}.npz"
            path = root / relative
            if split == "test":
                descriptors.append({
                    "split": split, "path": str(relative), "sample_count": 1,
                    "session_ids": [session_id],
                    "sha256": "poison-test-shard-must-not-be-opened",
                })
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **_stage_arrays(
                stage, session_id, timestamps_by_session[session_id]
            ))
            descriptors.append({
                "split": split, "path": str(relative), "sample_count": 1,
                "session_ids": [session_id],
                "sha256": _sha256(path),
            })
        manifest = {
            "schema_version": {
                "base": "stage3-dataset-v3",
                "truth": "stage3-truth-history-v1",
                "causal": "stage3-causal-physical-v1",
                "clean": "stage3-observable-future-v1",
                "pnp": "stage3-observable-future-real-pnp-observed-stream-v2",
                "pnp_sf": "stage3-observable-future-real-pnp-sf-observed-stream-v2",
            }[stage],
            "qualification_passed": True,
            "capture_contract_sha256": contract_sha,
            "shards": descriptors,
        }
        if stage == "base":
            manifest.update({
                "canonical_sources": "canonical_sources.jsonl",
                "artifact_sha256": {
                    "canonical_sources": _sha256(root / "canonical_sources.jsonl"),
                },
                "tensor_contract": {
                    "history_selection": (
                        "formal_ack_bound_latest_32_valid_observation_events_right_aligned_to_200; "
                        "legacy_latest_200_valid_observation_events"
                    ),
                    "formal_motion_history_event_limit": 32,
                },
            })
        else:
            manifest["test_accessed"] = False
        if stage == "truth":
            manifest["source_dataset_manifest_sha256"] = manifest_hashes["base"]
        elif stage == "causal":
            manifest.update({
                "source_observation_manifest_sha256": manifest_hashes["base"],
                "source_truth_history_manifest_sha256": manifest_hashes["truth"],
            })
        elif stage == "clean":
            manifest.update({
                "source_dataset_manifest_sha256": manifest_hashes["causal"],
                "truth_history_manifest_sha256": manifest_hashes["truth"],
            })
        elif stage == "pnp":
            observation_manifest = {
                "schema_version": "stage3-dataset-v4-observation",
                "qualification_passed": True,
                "capture_contract_sha256": contract_sha,
                "source_v3_manifest_sha256": manifest_hashes["base"],
                "test_accessed": False,
                "test_shards_opened": 0,
            }
            _write_json(observation_root / "dataset_manifest.json", observation_manifest)
            observation_manifest_sha = _sha256(observation_root / "dataset_manifest.json")
            manifest.update({
                "source_v3_manifest_sha256": manifest_hashes["base"],
                "truth_history_manifest_sha256": manifest_hashes["truth"],
                "causal_physical_manifest_sha256": manifest_hashes["causal"],
                "clean_dataset_manifest_sha256": manifest_hashes["clean"],
                "observation_dataset": str(observation_root),
                "observation_manifest_sha256": observation_manifest_sha,
                "minimum_history_events": 8,
            })
        elif stage == "pnp_sf":
            manifest.update({
                "source_v3_manifest_sha256": manifest_hashes["base"],
                "truth_history_manifest_sha256": manifest_hashes["truth"],
                "causal_physical_manifest_sha256": manifest_hashes["causal"],
                "parent_paired_manifest_sha256": manifest_hashes["pnp"],
                "observation_dataset": str(observation_root),
                "observation_manifest_sha256": observation_manifest_sha,
                "minimum_history_events": 8,
            })
        _write_json(root / "dataset_manifest.json", manifest)
        manifest_hashes[stage] = _sha256(root / "dataset_manifest.json")

    output = tmp_path / "audit"
    result_path = audit(argparse.Namespace(
        capture_contract=str(contract_path),
        base=str(roots["base"]),
        truth_history=str(roots["truth"]),
        causal_physical=str(roots["causal"]),
        observable_clean=str(roots["clean"]),
        pnp=str(roots["pnp"]),
        pnp_sf=str(roots["pnp_sf"]),
        output=str(output),
    ))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "passed"
    assert result["validation_gate_passed"] is True
    assert result["raw_base_rows_checked"] == 2
    assert result["test_session_metadata_count"] == 1
    assert result["test_shard_descriptors_seen"] == 6
    assert result["test_shards_opened"] == 0
    assert result["test_raw_opened"] == 0
    assert result["test_accessed"] is False

    base_manifest_path = roots["base"] / "dataset_manifest.json"
    bad_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    test_descriptor = next(
        item for item in bad_manifest["shards"] if item["split"] == "test"
    )
    test_descriptor["split"] = "validation"
    _write_json(base_manifest_path, bad_manifest)
    with pytest.raises(ValueError, match="frozen split before open"):
        audit(argparse.Namespace(
            capture_contract=str(contract_path),
            base=str(roots["base"]),
            truth_history=str(roots["truth"]),
            causal_physical=str(roots["causal"]),
            observable_clean=str(roots["clean"]),
            pnp=str(roots["pnp"]),
            pnp_sf=str(roots["pnp_sf"]),
            output=str(tmp_path / "bad-audit"),
        ))
