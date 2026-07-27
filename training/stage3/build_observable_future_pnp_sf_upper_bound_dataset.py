"""Build the anonymous four-handle PnP sidecar required by the S->F B arm.

The parent paired dataset already contains the strict selected-target PnP arm.
This builder reopens only the same train/validation windows and adds the
oracle-associated, virtual-visible four-handle history consumed by V19 S.
Every window receives an independent C4 origin and optional direction reversal;
the exported handles therefore have no stable physical identity across windows.
This remains a non-deployable upper bound because past truth performs the
same-exposure association and supplies the primary/switch supervision.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .build_observable_future_pnp_upper_bound_dataset import (
    _key_index,
    _load_npz,
    _one_session_map,
    _pair_id,
    _sha256,
    _write_json,
)
from .build_truth_history_dataset import (
    _load_jsonl,
    _nearest_indices,
    _parse_truth,
    _resolve_truth,
    _rotation_matrix,
)
from .cyclic_track_dataset import construct_cyclic_visibility, cyclic_relabel
from .observable_future_pnp_upper_bound import (
    oracle_injective_assignment,
    rebase_tracker_points_to_anchor,
)


SCHEMA_VERSION = "stage3-observable-future-real-pnp-sf-upper-bound-v1"
EXPERIMENT_KIND = "real_pnp_oracle_association_anonymous_sf_upper_bound"
SIGNED_FIELDS = (
    "history_switch_step",
    "pnp_history_switch_step",
    "candidate_step",
    "pnp_candidate_step",
    "target_switch_count",
)


def _manifest_path(root: Path, value: object) -> Path:
    return root / Path(str(value).replace("\\", "/"))


def _window_relabel(pair_id: str) -> tuple[int, bool]:
    value = int(pair_id[:16], 16)
    return value % 4, bool((value >> 2) & 1)


def _build_shard(task: dict[str, Any]) -> dict[str, Any]:
    parent = _load_npz(Path(task["parent_shard"]))
    source = _load_npz(Path(task["source_shard"]))
    truth = _load_npz(Path(task["truth_shard"]))
    observation = _load_npz(Path(task["observation_shard"]))
    source_index = _key_index(source, "causal physical")
    truth_index = _key_index(truth, "truth history")
    observation_index = _key_index(observation, "observation")

    frames = _parse_truth(Path(task["raw_truth_path"]), str(task["raw_truth_sha256"]))
    timestamps = np.asarray([frame.timestamp_ns for frame in frames], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("raw truth timestamps are not strictly increasing")

    rows: list[dict[str, np.ndarray]] = []
    primary_event_count = 0
    associated_primary_count = 0
    ambiguous_event_count = 0
    usable_count = 0
    common_usable_count = 0
    for parent_row, (session_raw, t0_raw) in enumerate(zip(
        parent["session_id"], parent["t0_ns"]
    )):
        session_id = str(session_raw)
        t0_ns = int(t0_raw)
        key = (session_id, t0_ns)
        if key not in source_index or key not in truth_index or key not in observation_index:
            raise ValueError(f"S/F sidecar source is missing exact key: {key}")
        source_row = source_index[key]
        truth_row = truth_index[key]
        observation_row = observation_index[key]
        event_time = source["event_time_s"][source_row]
        if not np.array_equal(event_time, truth["event_time_s"][truth_row]) or not np.array_equal(
            event_time, observation["event_time_s"][observation_row]
        ):
            raise ValueError(f"S/F sidecar event times differ: {key}")
        valid_events = np.flatnonzero(
            source["event_mask"][source_row]
            & np.isfinite(event_time)
            & (event_time <= 1e-6)
        )[-32:]
        if valid_events.size != 32:
            raise ValueError("S/F sidecar requires exactly the qualified last 32 events")
        selected_time = event_time[valid_events].astype(np.float32, copy=True)
        event_timestamps = t0_ns + np.rint(
            selected_time.astype(np.float64) * 1e9
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
        anchor_origin = np.asarray(anchor_frame.gimbal_origin_world_m, dtype=np.float64)
        anchor_rotation = _rotation_matrix(anchor_frame.chassis_quaternion_world_wxyz)

        clean_full = source["history_position_m"][source_row, valid_events].astype(
            np.float32, copy=True
        )
        event_mask = np.ones(32, dtype=np.bool_)
        visible, primary, switch_step = construct_cyclic_visibility(
            clean_full, event_mask,
            secondary_gap_ratio=float(task["secondary_gap_ratio"]),
        )
        pnp_handle = np.zeros((32, 4, 3), dtype=np.float32)
        associated = np.zeros((32, 4), dtype=np.bool_)
        ambiguous = np.zeros(32, dtype=np.bool_)
        obs = observation["obs"][observation_row, :, :, :3]
        obs_mask = observation["obs_mask"][observation_row]
        truth_obs = truth["truth_obs"][truth_row]
        truth_mask = truth["truth_obs_mask"][truth_row]
        for local_row, event in enumerate(valid_events):
            if not bool(truth_mask[event].all()):
                continue
            row_mask = obs_mask[event]
            rebased = rebase_tracker_points_to_anchor(
                obs[event, row_mask], event_origins[local_row],
                event_rotations[local_row], anchor_origin, anchor_rotation,
            )
            match = oracle_injective_assignment(
                rebased, truth_obs[event],
                ambiguity_epsilon_m=float(task["association_ambiguity_epsilon_m"]),
            )
            if match is None:
                continue
            _, assignment, is_ambiguous = match
            ambiguous[local_row] = bool(is_ambiguous)
            if is_ambiguous:
                continue
            for handle in np.flatnonzero(visible[local_row]):
                handle_int = int(handle)
                if handle_int in assignment:
                    pnp_handle[local_row, handle_int] = rebased[assignment[handle_int]]
                    associated[local_row, handle_int] = True
        pnp_mask = visible & associated
        primary_associated = (pnp_mask & primary).sum(axis=1) == 1
        q0_associated = bool(primary_associated[-1])
        sf_usable = bool(primary_associated.all() and not ambiguous.any())
        parent_usable = bool(parent["pnp_forward_usable"][parent_row])
        common_usable = sf_usable and parent_usable
        primary_event_count += 32
        associated_primary_count += int(primary_associated.sum())
        ambiguous_event_count += int(ambiguous.sum())
        usable_count += int(sf_usable)
        common_usable_count += int(common_usable)

        clean_obs = np.where(visible[..., None], clean_full, 0.0).astype(
            np.float32, copy=False
        )
        pair = str(parent["pair_id"][parent_row])
        if pair != _pair_id(session_id, t0_ns):
            raise ValueError("parent pair_id no longer matches session/t0")
        shift, reverse = _window_relabel(pair)
        for value, axis in (
            (pnp_handle, 1), (pnp_mask, 1), (clean_obs, 1),
            (visible, 1), (primary, 1), (clean_full[-1], 0),
        ):
            relabelled = cyclic_relabel(value, shift=shift, reverse=reverse, axis=axis)
            if value is pnp_handle:
                pnp_handle = relabelled
            elif value is pnp_mask:
                pnp_mask = relabelled
            elif value is clean_obs:
                clean_obs = relabelled
            elif value is visible:
                visible = relabelled
            elif value is primary:
                primary = relabelled
            else:
                truth_q0 = relabelled
        direction_sign = -1 if reverse else 1
        switch_step = (switch_step.astype(np.int64) * direction_sign).astype(
            np.int64, copy=False
        )

        output = {name: value[parent_row].copy() for name, value in parent.items()}
        for name in SIGNED_FIELDS:
            output[name] = (output[name].astype(np.int64) * direction_sign).astype(
                output[name].dtype, copy=False
            )
        output.update({
            "pnp_s_obs_m": pnp_handle,
            "pnp_s_obs_mask": pnp_mask,
            "pnp_s_primary_mask": primary,
            "pnp_s_event_mask": event_mask,
            "pnp_s_event_time_s": selected_time,
            "pnp_s_switch_step": switch_step,
            "pnp_s_truth_q0_m": truth_q0.astype(np.float32, copy=False),
            "clean_s_obs_m": clean_obs,
            "clean_s_obs_mask": visible,
            "clean_s_primary_mask": primary.copy(),
            "clean_s_event_mask": event_mask.copy(),
            "clean_s_event_time_s": selected_time.copy(),
            "clean_s_switch_step": switch_step.copy(),
            "pnp_s_primary_associated_mask": primary_associated,
            "pnp_s_ambiguous_event_mask": ambiguous,
            "pnp_s_q0_associated": np.asarray(q0_associated, dtype=np.bool_),
            "pnp_s_forward_usable": np.asarray(sf_usable, dtype=np.bool_),
            "pnp_sf_common_usable": np.asarray(common_usable, dtype=np.bool_),
            "pnp_s_window_shift": np.asarray(shift, dtype=np.int64),
            "pnp_s_direction_sign": np.asarray(direction_sign, dtype=np.int64),
        })
        if common_usable:
            q0_primary = int(np.flatnonzero(primary[-1])[0])
            if not np.allclose(
                pnp_handle[-1, q0_primary], output["pnp_current_position_m"],
                atol=2e-6, rtol=0.0,
            ):
                raise ValueError("S q0 primary and selected PnP current disagree")
            if not np.allclose(
                truth_q0[q0_primary], output["current_position_m"],
                atol=5e-5, rtol=0.0,
            ):
                raise ValueError("S truth q0 primary and clean F current disagree")
        rows.append(output)

    arrays = {
        name: np.stack([row[name] for row in rows], axis=0)
        for name in rows[0]
    }
    output_path = Path(task["output_shard"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return {
        "path": str(output_path),
        "split": task["split"],
        "session_id": task["session_id"],
        "sample_count": len(rows),
        "pnp_s_forward_usable_count": usable_count,
        "pnp_sf_common_usable_count": common_usable_count,
        "primary_event_count": primary_event_count,
        "associated_primary_count": associated_primary_count,
        "ambiguous_event_count": ambiguous_event_count,
        "sha256": _sha256(output_path),
        "bytes": output_path.stat().st_size,
    }


def build(args: argparse.Namespace) -> Path:
    parent_dir = Path(args.parent_dataset).resolve()
    output_dir = Path(args.output).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite S/F PnP dataset: {output_dir}")
    parent_manifest_path = parent_dir / "dataset_manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if (
        parent_manifest.get("schema_version")
        != "stage3-observable-future-real-pnp-upper-bound-v1"
        or not bool(parent_manifest.get("qualification_passed", False))
        or bool(parent_manifest.get("test_accessed", True))
    ):
        raise ValueError("S/F sidecar requires the qualified sealed paired PnP parent")

    source_dir = Path(str(parent_manifest["causal_physical_dataset"])).resolve()
    truth_dir = Path(str(parent_manifest["truth_history_dataset"])).resolve()
    observation_dir = Path(str(parent_manifest["observation_dataset"])).resolve()
    v3_dir = Path(str(parent_manifest["source_v3_dataset"])).resolve()
    manifest_paths = {
        "source": source_dir / "dataset_manifest.json",
        "truth": truth_dir / "dataset_manifest.json",
        "observation": observation_dir / "dataset_manifest.json",
        "v3": v3_dir / "dataset_manifest.json",
    }
    for name, path in manifest_paths.items():
        expected = parent_manifest[
            "causal_physical_manifest_sha256" if name == "source" else
            "truth_history_manifest_sha256" if name == "truth" else
            "observation_manifest_sha256" if name == "observation" else
            "source_v3_manifest_sha256"
        ]
        if _sha256(path) != str(expected):
            raise ValueError(f"S/F sidecar source manifest changed: {name}")
    source_manifest = json.loads(manifest_paths["source"].read_text(encoding="utf-8"))
    truth_manifest = json.loads(manifest_paths["truth"].read_text(encoding="utf-8"))
    observation_manifest = json.loads(
        manifest_paths["observation"].read_text(encoding="utf-8")
    )
    v3_manifest = json.loads(manifest_paths["v3"].read_text(encoding="utf-8"))
    if any(bool(manifest.get("test_accessed", False)) for manifest in (
        source_manifest, truth_manifest,
    )):
        raise ValueError("S/F sidecar source accessed test")

    allowed = {"train", "validation"}
    parent_shards = _one_session_map(parent_manifest, allowed)
    source_shards = _one_session_map(source_manifest, allowed)
    truth_shards = _one_session_map(truth_manifest, allowed)
    observation_shards = _one_session_map(observation_manifest, allowed)
    raw_root = v3_dir.parent.parent
    canonical_sources = {
        str(item["session_id"]): item
        for item in _load_jsonl(v3_dir / str(v3_manifest["canonical_sources"]))
    }
    selected: list[dict[str, Any]] = []
    split_count = {"train": 0, "validation": 0}
    for (split, session_id), parent_item in sorted(parent_shards.items()):
        if args.session_limit > 0 and split_count[split] >= args.session_limit:
            continue
        key = (split, session_id)
        if key not in source_shards or key not in truth_shards or key not in observation_shards:
            raise ValueError(f"S/F sidecar session source is missing: {key}")
        canonical = canonical_sources[session_id]
        selected.append({
            "parent_shard": str(_manifest_path(parent_dir, parent_item["path"])),
            "source_shard": str(_manifest_path(source_dir, source_shards[key]["path"])),
            "truth_shard": str(_manifest_path(truth_dir, truth_shards[key]["path"])),
            "observation_shard": str(_manifest_path(
                observation_dir, observation_shards[key]["path"]
            )),
            "raw_truth_path": str(_resolve_truth(raw_root, canonical)),
            "raw_truth_sha256": str(canonical["truth_sha256"]),
            "output_shard": str(
                output_dir / "shards" / Path(str(parent_item["path"])).name
            ),
            "split": split,
            "session_id": session_id,
            "secondary_gap_ratio": float(args.secondary_gap_ratio),
            "association_ambiguity_epsilon_m": float(
                parent_manifest["association_ambiguity_epsilon_m"]
            ),
        })
        split_count[split] += 1
    if not selected or not all(split_count.values()):
        raise ValueError(f"S/F sidecar selection is incomplete: {split_count}")

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
                    "common_usable": result["pnp_sf_common_usable_count"],
                }, sort_keys=True), flush=True)
    except Exception:
        _write_json(output_dir / "build_failed.json", {
            "status": "failed",
            "completed_shards": len(results),
            "parent_dataset": str(parent_dir),
        })
        raise
    results.sort(key=lambda item: (item["split"], item["session_id"]))
    sample_count = sum(int(item["sample_count"]) for item in results)
    sf_usable = sum(int(item["pnp_s_forward_usable_count"]) for item in results)
    common_usable = sum(int(item["pnp_sf_common_usable_count"]) for item in results)
    primary_events = sum(int(item["primary_event_count"]) for item in results)
    associated_primary = sum(int(item["associated_primary_count"]) for item in results)
    ambiguous_events = sum(int(item["ambiguous_event_count"]) for item in results)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_kind": EXPERIMENT_KIND,
        "qualification_passed": args.session_limit == 0,
        "diagnostic_subset": args.session_limit > 0,
        "test_accessed": False,
        "splits": ["train", "validation"],
        "deployable_pipeline": False,
        "oracle_association": True,
        "oracle_primary_and_switch": True,
        "future_truth_in_model_input": False,
        "physical_identity_exported": False,
        "handle_identity": "window-local C4-shifted and optionally direction-reversed",
        "mandatory_anonymization": "pair-id hash C4 origin plus direction reversal",
        "pnp_feature_contract": "xyz and validity only",
        "candidate_mask_semantics": (
            "all signed-step hypotheses exist; S confidence zero marks unsupported roles"
        ),
        "parent_paired_dataset": str(parent_dir),
        "parent_paired_manifest_sha256": _sha256(parent_manifest_path),
        "causal_physical_dataset": str(source_dir),
        "causal_physical_manifest_sha256": _sha256(manifest_paths["source"]),
        "truth_history_dataset": str(truth_dir),
        "truth_history_manifest_sha256": _sha256(manifest_paths["truth"]),
        "observation_dataset": str(observation_dir),
        "observation_manifest_sha256": _sha256(manifest_paths["observation"]),
        "source_v3_dataset": str(v3_dir),
        "source_v3_manifest_sha256": _sha256(manifest_paths["v3"]),
        "sample_count": sample_count,
        "pnp_s_forward_usable_count": sf_usable,
        "pnp_s_forward_usable_fraction": sf_usable / sample_count,
        "pnp_sf_common_usable_count": common_usable,
        "pnp_sf_common_usable_fraction": common_usable / sample_count,
        "primary_event_count": primary_events,
        "associated_primary_count": associated_primary,
        "associated_primary_fraction": associated_primary / primary_events,
        "ambiguous_event_count": ambiguous_events,
        "history_events": 32,
        "candidate_steps": list(parent_manifest["candidate_steps"]),
        "secondary_gap_ratio": float(args.secondary_gap_ratio),
        "builder_source_sha256": _sha256(Path(__file__)),
        "shards": [{
            "path": str(Path(item["path"]).relative_to(output_dir)),
            "split": item["split"],
            "session_ids": [item["session_id"]],
            "sample_count": item["sample_count"],
            "pnp_sf_common_usable_count": item["pnp_sf_common_usable_count"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        } for item in results],
    }
    _write_json(output_dir / "dataset_manifest.json", manifest)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--session-limit", type=int, default=0)
    parser.add_argument("--secondary-gap-ratio", type=float, default=0.25)
    args = parser.parse_args()
    if (
        args.workers < 1 or args.session_limit < 0
        or not 0.0 <= args.secondary_gap_ratio <= 1.0
    ):
        parser.error("S/F sidecar arguments are invalid")
    print(build(args))


if __name__ == "__main__":
    main()
