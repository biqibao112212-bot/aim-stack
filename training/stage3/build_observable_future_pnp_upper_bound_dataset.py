"""Build a hash-bound actual-observed-primary PnP dataset for frozen F.

The builder replays the accepted observable-r6 construction bit-exact before
attaching real PnP histories.  Unordered PnP rows are associated with
same-exposure past truth only, after every point is rebased into the q0 anchor
tracker frame.  The actual observed q0 role seeds history and future labels;
temporary assignments are never exported.  Oracle association and the truth-S
candidate set keep this an explicitly non-deployable upper bound.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .build_observable_future_dataset import (
    _dense_times,
    _rollout_dense_truth,
)
from .build_truth_history_dataset import (
    _load_jsonl,
    _nearest_indices,
    _parse_truth,
    _resolve_truth,
    _rotation_matrix,
)
from .observable_future_dataset import (
    DEFAULT_CANDIDATE_STEPS,
    SCHEMA_VERSION as CLEAN_SCHEMA_VERSION,
    construct_observable_future_sample,
)
from .observable_future_pnp_upper_bound import (
    OBSERVED_STREAM_EXPERIMENT_KIND as EXPERIMENT_KIND,
    OBSERVED_STREAM_SCHEMA_VERSION as SCHEMA_VERSION,
    construct_observed_primary_pnp_sample,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _key_index(arrays: dict[str, np.ndarray], label: str) -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}
    for index, (session, t0) in enumerate(zip(arrays["session_id"], arrays["t0_ns"])):
        key = (str(session), int(t0))
        if key in result:
            raise ValueError(f"duplicate {label} sample key: {key}")
        result[key] = index
    return result


def _assert_clean_replay(
    replayed: dict[str, np.ndarray], clean: dict[str, np.ndarray], clean_index: int
) -> None:
    for key, value in replayed.items():
        if key not in clean:
            raise ValueError(f"clean shard is missing replayed field: {key}")
        expected = clean[key][clean_index]
        if not np.array_equal(value, expected):
            raise ValueError(f"observable-r6 replay is not bit-exact: {key}/{clean_index}")


def _pair_id(session_id: str, t0_ns: int) -> str:
    return hashlib.sha256(f"{session_id}:{int(t0_ns)}".encode("utf-8")).hexdigest()


def _build_shard(task: dict[str, Any]) -> dict[str, Any]:
    source = _load_npz(Path(task["source_shard"]))
    truth = _load_npz(Path(task["truth_shard"]))
    observation = _load_npz(Path(task["observation_shard"]))
    clean = _load_npz(Path(task["clean_shard"]))
    truth_index = _key_index(truth, "truth-history")
    observation_index = _key_index(observation, "observation")

    frames = _parse_truth(Path(task["raw_truth_path"]), str(task["raw_truth_sha256"]))
    timestamps = np.asarray([frame.timestamp_ns for frame in frames], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("raw truth timestamps are not strictly increasing")

    paired: list[dict[str, np.ndarray]] = []
    clean_index = 0
    maximum_rollout_error_m = 0.0
    for source_index, (session_raw, t0_raw) in enumerate(zip(
        source["session_id"], source["t0_ns"]
    )):
        session_id = str(session_raw)
        t0_ns = int(t0_raw)
        key = (session_id, t0_ns)
        if key not in truth_index or key not in observation_index:
            raise ValueError(f"paired source is missing exact window key: {key}")
        truth_row = truth_index[key]
        observation_row = observation_index[key]
        if not np.array_equal(
            source["event_time_s"][source_index], truth["event_time_s"][truth_row]
        ) or not np.array_equal(
            source["event_time_s"][source_index], observation["event_time_s"][observation_row]
        ):
            raise ValueError(f"paired event times differ: {key}")

        tau = source["tau"][source_index].astype(np.float32, copy=False)
        zero = np.flatnonzero(tau == 0.0)
        if zero.size < 1:
            raise ValueError("causal source row has no exact q0")
        dense_time = _dense_times(tau, float(task["dense_step_s"]))
        dense_position = _rollout_dense_truth(
            source["future_position"][source_index, int(zero[0])],
            truth["anchor_center_position_m"][truth_row],
            truth["anchor_velocity_mps"][truth_row],
            float(truth["anchor_yaw_rate_rad_s"][truth_row]),
            dense_time,
        )
        endpoint_error: list[float] = []
        for query, query_time in enumerate(tau):
            distance = np.abs(dense_time - float(query_time))
            minimum = float(distance.min())
            matches = np.flatnonzero(distance == minimum)
            if matches.size != 1 or minimum > float(task["query_tolerance_s"]):
                raise ValueError("dense replay lost a source query")
            dense_index = int(matches[0])
            error = np.linalg.norm(
                dense_position[dense_index] - source["future_position"][source_index, query],
                axis=-1,
            ).max()
            if bool(source["rule_query"][source_index, query]):
                endpoint_error.append(float(error))
            dense_position[dense_index] = source["future_position"][source_index, query]
        row_error = max(endpoint_error, default=0.0)
        maximum_rollout_error_m = max(maximum_rollout_error_m, row_error)
        if row_error > float(task["maximum_rollout_error_m"]):
            continue
        try:
            clean_replay = construct_observable_future_sample(
                source["history_position_m"][source_index],
                source["event_mask"][source_index],
                source["event_time_s"][source_index],
                dense_position,
                dense_time,
                tau,
                source["rule_query"][source_index],
                history_events=32,
                candidate_steps=DEFAULT_CANDIDATE_STEPS,
                tie_epsilon_m=float(task["tie_epsilon_m"]),
                query_match_tolerance_s=float(task["query_tolerance_s"]),
            )
        except ValueError:
            continue
        if clean_index >= int(clean["motion_class"].shape[0]):
            raise ValueError("observable-r6 replay produced extra samples")
        _assert_clean_replay(clean_replay, clean, clean_index)
        if int(source["motion_class"][source_index]) != int(
            clean["motion_class"][clean_index]
        ):
            raise ValueError("clean replay motion class differs")

        valid_events = np.flatnonzero(
            source["event_mask"][source_index]
            & np.isfinite(source["event_time_s"][source_index])
            & (source["event_time_s"][source_index] <= 1e-6)
        )[-32:]
        event_timestamps = t0_ns + np.rint(
            source["event_time_s"][source_index, valid_events].astype(np.float64)
            * 1e9
        ).astype(np.int64)
        event_frame_index, event_error_ns = _nearest_indices(timestamps, event_timestamps)
        anchor_frame_index, anchor_error_ns = _nearest_indices(
            timestamps, np.asarray([t0_ns], dtype=np.int64)
        )
        if int(anchor_error_ns[0]) != 0 or np.any(event_error_ns > 2_000):
            raise ValueError(f"raw truth pose is not exposure-aligned: {key}")
        anchor_frame = frames[int(anchor_frame_index[0])]
        event_origins = np.asarray([
            frames[int(index)].gimbal_origin_world_m for index in event_frame_index
        ], dtype=np.float64)
        event_rotations = np.stack([
            _rotation_matrix(frames[int(index)].chassis_quaternion_world_wxyz)
            for index in event_frame_index
        ])
        pnp = construct_observed_primary_pnp_sample(
            clean_replay,
            source["history_position_m"][source_index],
            source["event_mask"][source_index],
            source["event_time_s"][source_index],
            dense_position,
            dense_time,
            tau,
            source["rule_query"][source_index],
            truth["truth_obs"][truth_row],
            truth["truth_obs_mask"][truth_row],
            observation["obs"][observation_row, :, :, :3],
            observation["obs_mask"][observation_row],
            event_origins,
            event_rotations,
            np.asarray(anchor_frame.gimbal_origin_world_m, dtype=np.float64),
            _rotation_matrix(anchor_frame.chassis_quaternion_world_wxyz),
            minimum_history_events=int(task["minimum_history_events"]),
            tie_epsilon_m=float(task["tie_epsilon_m"]),
            primary_switch_hysteresis_m=float(
                task["primary_switch_hysteresis_m"]
            ),
            query_match_tolerance_s=float(task["query_tolerance_s"]),
            association_ambiguity_epsilon_m=float(
                task["association_ambiguity_epsilon_m"]
            ),
        )
        row = {key: value.copy() for key, value in pnp.items()}
        row["motion_class"] = np.asarray(
            clean["motion_class"][clean_index], dtype=np.int64
        )
        row["session_id"] = np.asarray(session_id)
        row["t0_ns"] = np.asarray(t0_ns, dtype=np.int64)
        row["pair_id"] = np.asarray(_pair_id(session_id, t0_ns))
        paired.append(row)
        clean_index += 1

    if clean_index != int(clean["motion_class"].shape[0]):
        raise ValueError(
            f"observable-r6 replay count mismatch: {clean_index} != "
            f"{clean['motion_class'].shape[0]}"
        )
    if not paired:
        raise ValueError("paired PnP shard produced no clean samples")
    arrays = {
        key: np.stack([sample[key] for sample in paired], axis=0)
        for key in paired[0]
    }
    output_path = Path(task["output_shard"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    usable = arrays["pnp_forward_usable"].astype(np.bool_, copy=False)
    q0 = arrays["pnp_q0_associated"].astype(np.bool_, copy=False)
    full = arrays["pnp_full_history_associated"].astype(np.bool_, copy=False)
    associated_events = arrays["pnp_history_associated_mask"].astype(
        np.bool_, copy=False
    )
    ambiguous_events = arrays["pnp_history_ambiguous_mask"].astype(
        np.bool_, copy=False
    )
    candidate_events = arrays["pnp_history_candidate_count"].astype(
        np.int64, copy=False
    ) > 0
    active_count = arrays["pnp_history_active_count"].astype(
        np.int64, copy=False
    )
    track_break = arrays["pnp_history_track_break_count"].astype(
        np.int64, copy=False
    )
    future_coherent = arrays["pnp_future_label_coherent"].astype(
        np.bool_, copy=False
    )
    return {
        "path": str(output_path),
        "split": task["split"],
        "session_id": task["session_id"],
        "sample_count": len(paired),
        "pnp_forward_usable_count": int(usable.sum()),
        "pnp_q0_associated_count": int(q0.sum()),
        "pnp_full_history_associated_count": int(full.sum()),
        "history_event_count": int(associated_events.size),
        "history_event_associated_count": int(associated_events.sum()),
        "history_event_ambiguous_count": int(ambiguous_events.sum()),
        "history_observation_candidate_event_count": int(candidate_events.sum()),
        "history_active_count": int(active_count.sum()),
        "history_track_break_count": int(track_break.sum()),
        "future_label_coherent_count": int(future_coherent.sum()),
        "minimum_active_history_count": int(active_count.min(initial=32)),
        "maximum_active_history_count": int(active_count.max(initial=0)),
        "failure_code_count": {
            str(code): int(np.count_nonzero(arrays["pnp_failure_code"] == code))
            for code in range(4)
        },
        "maximum_rollout_error_m": maximum_rollout_error_m,
        "sha256": _sha256(output_path),
        "bytes": output_path.stat().st_size,
    }


def _one_session_map(manifest: dict[str, Any], allowed_splits: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for item in manifest["shards"]:
        split = str(item["split"])
        if split not in allowed_splits:
            continue
        sessions = [str(value) for value in item["session_ids"]]
        if len(sessions) != 1:
            raise ValueError("paired PnP builder requires one-session shards")
        key = (split, sessions[0])
        if key in result:
            raise ValueError(f"duplicate session shard: {key}")
        result[key] = item
    return result


def build(args: argparse.Namespace) -> Path:
    clean_dir = Path(args.clean_dataset).resolve()
    observation_dir = Path(args.observation_dataset).resolve()
    output_dir = Path(args.output).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite paired PnP dataset: {output_dir}")

    clean_manifest_path = clean_dir / "dataset_manifest.json"
    clean_manifest = json.loads(clean_manifest_path.read_text(encoding="utf-8"))
    if clean_manifest.get("schema_version") != CLEAN_SCHEMA_VERSION:
        raise ValueError("clean source must be observable-future v1")
    if not bool(clean_manifest.get("qualification_passed", False)):
        raise ValueError("clean observable source is not qualified")
    if bool(clean_manifest.get("test_accessed", True)):
        raise ValueError("clean observable source accessed test")

    source_dir = Path(str(clean_manifest["source_dataset"])).resolve()
    source_manifest_path = source_dir / "dataset_manifest.json"
    if _sha256(source_manifest_path) != str(
        clean_manifest["source_dataset_manifest_sha256"]
    ):
        raise ValueError("clean observable source-manifest binding changed")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema_version") != "stage3-causal-physical-v1":
        raise ValueError("clean source no longer points to causal physical v1")
    if not bool(source_manifest.get("qualification_passed", False)) or bool(
        source_manifest.get("test_accessed", True)
    ):
        raise ValueError("causal physical source is not sealed and qualified")

    truth_dir = Path(str(clean_manifest["truth_history_dataset"])).resolve()
    truth_manifest_path = truth_dir / "dataset_manifest.json"
    if _sha256(truth_manifest_path) != str(
        clean_manifest["truth_history_manifest_sha256"]
    ):
        raise ValueError("clean truth-history binding changed")
    truth_manifest = json.loads(truth_manifest_path.read_text(encoding="utf-8"))
    if not bool(truth_manifest.get("qualification_passed", False)) or bool(
        truth_manifest.get("test_accessed", True)
    ):
        raise ValueError("truth-history source is not sealed and qualified")

    observation_manifest_path = observation_dir / "dataset_manifest.json"
    observation_manifest = json.loads(
        observation_manifest_path.read_text(encoding="utf-8")
    )
    if observation_manifest.get("schema_version") != "stage3-dataset-v4-observation":
        raise ValueError("PnP source must be observation v4")
    if not bool(observation_manifest.get("qualification_passed", False)):
        raise ValueError("PnP observation source is not qualified")
    if str(observation_manifest.get("source_v3_manifest_sha256")) != str(
        truth_manifest.get("source_dataset_manifest_sha256")
    ):
        raise ValueError("PnP and truth history do not share the qualified v3 source")

    v3_dir = Path(str(truth_manifest["source_dataset"])).resolve()
    v3_manifest_path = v3_dir / "dataset_manifest.json"
    v3_manifest = json.loads(v3_manifest_path.read_text(encoding="utf-8"))
    if _sha256(v3_manifest_path) != str(
        truth_manifest["source_dataset_manifest_sha256"]
    ):
        raise ValueError("qualified v3 source manifest changed")
    raw_root = v3_dir.parent.parent
    canonical_sources = {
        str(item["session_id"]): item
        for item in _load_jsonl(v3_dir / str(v3_manifest["canonical_sources"]))
    }

    allowed_splits = {"train", "validation"}
    clean_shards = _one_session_map(clean_manifest, allowed_splits)
    source_shards = _one_session_map(source_manifest, allowed_splits)
    truth_shards = _one_session_map(truth_manifest, allowed_splits)
    observation_shards = _one_session_map(observation_manifest, allowed_splits)
    selected: list[dict[str, Any]] = []
    split_count = {"train": 0, "validation": 0}
    for (split, session_id), clean_item in sorted(clean_shards.items()):
        if args.session_limit > 0 and split_count[split] >= args.session_limit:
            continue
        key = (split, session_id)
        if key not in source_shards or key not in truth_shards or key not in observation_shards:
            raise ValueError(f"paired session shard is missing: {key}")
        if session_id not in canonical_sources:
            raise ValueError(f"canonical raw source is missing: {session_id}")
        canonical = canonical_sources[session_id]
        raw_truth_path = _resolve_truth(raw_root, canonical)
        selected.append({
            "clean_shard": str(clean_dir / str(clean_item["path"])),
            "source_shard": str(source_dir / str(source_shards[key]["path"])),
            "truth_shard": str(truth_dir / str(truth_shards[key]["path"])),
            "observation_shard": str(
                observation_dir / str(observation_shards[key]["path"])
            ),
            "raw_truth_path": str(raw_truth_path),
            "raw_truth_sha256": str(canonical["truth_sha256"]),
            "output_shard": str(
                output_dir / "shards" / Path(str(clean_item["path"])).name
            ),
            "split": split,
            "session_id": session_id,
            "dense_step_s": float(clean_manifest["dense_label_policy"]["step_ms"]) / 1000.0,
            "maximum_rollout_error_m": float(
                clean_manifest["dense_label_policy"][
                    "maximum_allowed_endpoint_rollout_error_m"
                ]
            ),
            "query_tolerance_s": float(args.query_tolerance_us) / 1e6,
            "tie_epsilon_m": float(args.tie_epsilon_m),
            "association_ambiguity_epsilon_m": float(
                args.association_ambiguity_epsilon_m
            ),
            "minimum_history_events": int(args.minimum_history_events),
            "primary_switch_hysteresis_m": float(
                args.primary_switch_hysteresis_m
            ),
        })
        split_count[split] += 1
    if not selected or not all(split_count.values()):
        raise ValueError(f"paired train/validation selection is incomplete: {split_count}")

    output_dir.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_build_shard, task): task for task in selected}
            for completed, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                print(json.dumps({
                    "completed": completed,
                    "total": len(selected),
                    "session_id": result["session_id"],
                    "samples": result["sample_count"],
                    "usable": result["pnp_forward_usable_count"],
                }, sort_keys=True), flush=True)
    except Exception:
        _write_json(output_dir / "build_failed.json", {
            "status": "failed",
            "completed_shards": len(results),
            "clean_dataset": str(clean_dir),
            "observation_dataset": str(observation_dir),
        })
        raise
    results.sort(key=lambda item: (item["split"], item["session_id"]))

    sample_count = sum(int(item["sample_count"]) for item in results)
    usable_count = sum(int(item["pnp_forward_usable_count"]) for item in results)
    q0_count = sum(int(item["pnp_q0_associated_count"]) for item in results)
    full_count = sum(
        int(item["pnp_full_history_associated_count"]) for item in results
    )
    event_count = sum(int(item["history_event_count"]) for item in results)
    event_associated = sum(
        int(item["history_event_associated_count"]) for item in results
    )
    event_ambiguous = sum(
        int(item["history_event_ambiguous_count"]) for item in results
    )
    candidate_event_count = sum(
        int(item["history_observation_candidate_event_count"])
        for item in results
    )
    active_count = sum(int(item["history_active_count"]) for item in results)
    track_break_count = sum(
        int(item["history_track_break_count"]) for item in results
    )
    future_coherent_count = sum(
        int(item["future_label_coherent_count"]) for item in results
    )
    failure_code_count = {
        str(code): sum(
            int(item["failure_code_count"][str(code)]) for item in results
        )
        for code in range(4)
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_kind": EXPERIMENT_KIND,
        "qualification_passed": args.session_limit == 0,
        "diagnostic_subset": args.session_limit > 0,
        "test_accessed": False,
        "splits": ["train", "validation"],
        "deployable_pipeline": False,
        "pnp_source": "real stage3 observation-v4 xyz",
        "oracle_association": True,
        "oracle_history_switch": True,
        "truth_s_candidates": True,
        "future_truth_in_model_input": False,
        "physical_identity_exported": False,
        "strict_complete_32_event_history": False,
        "observed_primary_stream": True,
        "history_admission_policy": (
            "q0 PnP-range-nearest among actual observations; maximum coherent "
            "contiguous suffix with same/adjacent anonymous-handle transitions, "
            "one switch direction and range hysteresis"
        ),
        "failure_code_semantics": {
            "0": "usable",
            "1": "q0 has no unambiguous actual observation",
            "2": "coherent history suffix has fewer than the minimum events",
            "3": "observed q0 cannot seed an adjacent-only future label",
        },
        "minimum_history_events": int(args.minimum_history_events),
        "coordinate_policy": (
            "each PnP point event-tracker -> world -> q0-anchor-tracker"
        ),
        "current_policy": (
            "q0 nearest by exposure-local PnP horizontal range among actual "
            "observations; truth is association/label-only"
        ),
        "candidate_policy": (
            "truth-S q0 candidates with current role replaced by PnP current; "
            "all step modulo 4 equals 0 relations are bit-exact zero"
        ),
        "target_policy": (
            "future truth replay seeded by the actual-observed q0 handle; "
            "PnP delta recomputed from PnP current"
        ),
        "clean_dataset": str(clean_dir),
        "clean_dataset_manifest_sha256": _sha256(clean_manifest_path),
        "causal_physical_dataset": str(source_dir),
        "causal_physical_manifest_sha256": _sha256(source_manifest_path),
        "truth_history_dataset": str(truth_dir),
        "truth_history_manifest_sha256": _sha256(truth_manifest_path),
        "observation_dataset": str(observation_dir),
        "observation_manifest_sha256": _sha256(observation_manifest_path),
        "source_v3_dataset": str(v3_dir),
        "source_v3_manifest_sha256": _sha256(v3_manifest_path),
        "sample_count": sample_count,
        "pnp_forward_usable_count": usable_count,
        "pnp_forward_usable_fraction": usable_count / sample_count,
        "pnp_q0_associated_count": q0_count,
        "pnp_q0_associated_fraction": q0_count / sample_count,
        "pnp_full_history_associated_count": full_count,
        "pnp_full_history_associated_fraction": full_count / sample_count,
        "history_event_count": event_count,
        "history_event_associated_count": event_associated,
        "history_event_associated_fraction": event_associated / event_count,
        "history_event_ambiguous_count": event_ambiguous,
        "history_observation_candidate_event_count": candidate_event_count,
        "history_observation_candidate_event_fraction": (
            candidate_event_count / event_count
        ),
        "history_active_count": active_count,
        "history_active_fraction": active_count / event_count,
        "history_track_break_count": track_break_count,
        "future_label_coherent_count": future_coherent_count,
        "future_label_coherent_fraction": future_coherent_count / sample_count,
        "failure_code_count": failure_code_count,
        "history_events": 32,
        "candidate_steps": list(DEFAULT_CANDIDATE_STEPS),
        "association_ambiguity_epsilon_m": float(
            args.association_ambiguity_epsilon_m
        ),
        "primary_tie_epsilon_m": float(args.tie_epsilon_m),
        "primary_switch_hysteresis_m": float(
            args.primary_switch_hysteresis_m
        ),
        "source_clean_replay_bit_exact": True,
        "output_clean_reanchored_to_observed_q0": True,
        "maximum_endpoint_rollout_error_m": max(
            float(item["maximum_rollout_error_m"]) for item in results
        ),
        "builder_source_sha256": {
            "build_observable_future_pnp_upper_bound_dataset.py": _sha256(
                Path(__file__)
            ),
            "observable_future_pnp_upper_bound.py": _sha256(
                Path(__file__).with_name("observable_future_pnp_upper_bound.py")
            ),
        },
        "shards": [{
            "path": str(Path(item["path"]).relative_to(output_dir)),
            "split": item["split"],
            "session_ids": [item["session_id"]],
            "sample_count": item["sample_count"],
            "pnp_forward_usable_count": item["pnp_forward_usable_count"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        } for item in results],
    }
    _write_json(output_dir / "dataset_manifest.json", manifest)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dataset", required=True)
    parser.add_argument("--observation-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--session-limit", type=int, default=0)
    parser.add_argument("--query-tolerance-us", type=float, default=2.0)
    parser.add_argument("--tie-epsilon-m", type=float, default=1e-6)
    parser.add_argument("--association-ambiguity-epsilon-m", type=float, default=1e-6)
    parser.add_argument("--minimum-history-events", type=int, default=8)
    parser.add_argument("--primary-switch-hysteresis-m", type=float, default=0.02)
    args = parser.parse_args()
    if (
        args.workers < 1 or args.session_limit < 0
        or args.query_tolerance_us < 0 or args.tie_epsilon_m < 0
        or args.association_ambiguity_epsilon_m < 0
        or not 2 <= args.minimum_history_events <= 32
        or args.primary_switch_hysteresis_m < 0
    ):
        parser.error("paired PnP builder arguments are invalid")
    print(build(args))


if __name__ == "__main__":
    main()
