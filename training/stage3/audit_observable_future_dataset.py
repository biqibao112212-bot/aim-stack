"""Independently audit a built anonymous observable-target derivative."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .observable_future_dataset import SCHEMA_VERSION, _manifest_path


FORBIDDEN_KEYS = {
    "physical_id", "armor_id", "slot_id", "handle_id", "primary_index",
    "primary_mask", "future_position",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(dataset_dir: str | Path) -> dict[str, object]:
    dataset = Path(dataset_dir).resolve()
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or not bool(manifest.get("qualification_passed", False))
        or bool(manifest.get("test_accessed", True))
        or set(manifest.get("splits", ())) != {"train", "validation"}
    ):
        raise ValueError("observable F manifest is not qualified and test-sealed")
    declared_steps = np.asarray(manifest["candidate_steps"], dtype=np.int64)
    summary: dict[str, dict[str, object]] = {
        split: {
            "sample_count": 0, "eligible_query_count": 0,
            "changed_query_count": 0, "motion_class_count": {},
            "switch_count": {},
        } for split in ("train", "validation")
    }
    total_samples = 0
    for item in manifest["shards"]:
        split = str(item["split"])
        if split not in summary:
            raise ValueError("observable F manifest contains a forbidden split")
        path = _manifest_path(dataset, item["path"])
        if _sha256(path) != str(item["sha256"]):
            raise ValueError(f"observable F shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        forbidden = FORBIDDEN_KEYS & arrays.keys()
        if forbidden:
            raise ValueError(f"observable F shard exports identity/fixed-track keys: {forbidden}")
        required = {
            "history_position_rel_m", "history_time_s", "history_dt_s",
            "history_switch_step", "history_mask", "current_position_m",
            "candidate_relation_m", "candidate_step", "candidate_mask",
            "candidate_confidence", "tau_s", "target_switch_count",
            "target_candidate_onehot", "target_visible_delta_m",
            "target_query_mask", "motion_class",
        }
        if required - arrays.keys():
            raise ValueError(f"observable F shard is missing {sorted(required - arrays.keys())}")
        sample_count = int(arrays["motion_class"].shape[0])
        if sample_count != int(item["sample_count"]):
            raise ValueError("observable F shard sample count differs from manifest")
        expected_steps = np.broadcast_to(declared_steps, arrays["candidate_step"].shape)
        if not np.array_equal(arrays["candidate_step"], expected_steps):
            raise ValueError("observable F candidate steps differ from manifest")
        if not np.array_equal(arrays["candidate_mask"], np.ones_like(arrays["candidate_mask"])):
            raise ValueError("truth-S derivative must cover every declared candidate")
        zero_candidate = arrays["candidate_step"] == 0
        if np.any(zero_candidate.sum(axis=1) != 1):
            raise ValueError("observable F sample does not have one step-zero candidate")
        if np.any(arrays["candidate_relation_m"][zero_candidate] != 0):
            raise ValueError("step-zero candidate relation is not exact zero")
        query_mask = arrays["target_query_mask"].astype(np.bool_, copy=False)
        onehot = arrays["target_candidate_onehot"].astype(np.bool_, copy=False)
        if np.any(onehot.sum(axis=-1)[query_mask] != 1):
            raise ValueError("eligible observable query lacks one-hot candidate coverage")
        target = arrays["target_switch_count"]
        if np.any((onehot & (arrays["candidate_step"][:, None, :] != target[:, :, None]))[query_mask]):
            raise ValueError("observable target one-hot points at the wrong signed step")
        zero_query = arrays["tau_s"] == 0
        if (
            np.any(target[zero_query] != 0)
            or np.any(arrays["target_visible_delta_m"][zero_query] != 0)
            or np.any(~query_mask[zero_query])
        ):
            raise ValueError("observable F q0 target is not structurally exact")
        if np.any(~np.isin(arrays["history_switch_step"], (-1, 0, 1))):
            raise ValueError("observable F history contains a non-adjacent switch")
        valid_history = arrays["history_mask"].astype(np.bool_, copy=False)
        if np.any(valid_history.sum(axis=1) != int(manifest["history_events"])):
            raise ValueError("observable F history length differs from qualification")
        values = target[query_mask]
        split_summary = summary[split]
        split_summary["sample_count"] = int(split_summary["sample_count"]) + sample_count
        split_summary["eligible_query_count"] = int(split_summary["eligible_query_count"]) + int(values.size)
        split_summary["changed_query_count"] = int(split_summary["changed_query_count"]) + int((values != 0).sum())
        class_count = split_summary["motion_class_count"]
        for value, count in zip(*np.unique(arrays["motion_class"], return_counts=True)):
            class_count[str(int(value))] = class_count.get(str(int(value)), 0) + int(count)
        step_count = split_summary["switch_count"]
        for value, count in zip(*np.unique(values, return_counts=True)):
            step_count[str(int(value))] = step_count.get(str(int(value)), 0) + int(count)
        total_samples += sample_count
    if total_samples != int(manifest["sample_count"]):
        raise ValueError("observable F total sample count differs from manifest")
    observed = [
        int(step) for value in summary.values() for step in value["switch_count"]
    ]
    if min(observed) != int(manifest["minimum_switch_count"]) or max(observed) != int(
        manifest["maximum_switch_count"]
    ):
        raise ValueError("observable F observed switch range differs from manifest")
    if int(manifest.get("uncovered_query_count", -1)) != 0:
        raise ValueError("observable F manifest records uncovered eligible queries")
    return {
        "status": "passed",
        "dataset": str(dataset),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "test_accessed": False,
        "candidate_steps": declared_steps.tolist(),
        "observed_switch_range": [min(observed), max(observed)],
        "splits": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
