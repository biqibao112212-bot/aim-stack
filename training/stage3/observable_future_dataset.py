"""ID-free observable-target samples for the Stage-3 F experts.

This module deliberately requires a dense future physical-truth stream.  The
eight sparse endpoint queries in the older r4 dataset cannot distinguish no
switch from a complete revolution back to the same plate.  Source slots exist
only inside :func:`construct_observable_future_sample`; exported tensors use a
signed, sample-local switch count and anonymous S candidate relations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import torch
    from torch.utils.data import IterableDataset
except ModuleNotFoundError as error:  # Keep the NumPy label builder torch-free.
    if error.name != "torch":
        raise
    torch = None  # type: ignore[assignment]

    class IterableDataset:  # type: ignore[no-redef]
        pass


SCHEMA_VERSION = "stage3-observable-future-v1"
VISIBILITY_POLICY = "nearest-horizontal-range-continuity-tie-v1"
EXPERT_TO_MOTION_CLASS = {"translation": 1, "rotation": 2, "combined": 3}
DEFAULT_CANDIDATE_STEPS = tuple(range(-6, 7))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(root: Path, value: object) -> Path:
    """Resolve manifests produced on either Windows or POSIX."""
    return root / Path(str(value).replace("\\", "/"))


def _select_with_continuity(
    position_m: np.ndarray,
    previous: int | None,
    *,
    tie_epsilon_m: float,
) -> int:
    """Select the nearest candidate without using a source-slot tie break."""
    if position_m.shape != (4, 3) or not np.isfinite(position_m).all():
        raise ValueError("visibility selection requires finite [4,3] positions")
    horizontal_range = np.linalg.norm(position_m[:, :2], axis=-1)
    order = np.argsort(horizontal_range, kind="stable")
    tied = (
        float(horizontal_range[order[1]] - horizontal_range[order[0]])
        <= tie_epsilon_m
    )
    if tied:
        pair = {int(order[0]), int(order[1])}
        if previous is None or int(previous) not in pair:
            raise ValueError("visibility tie has no continuity-preserving selection")
        return int(previous)
    return int(order[0])


def _signed_adjacent_step(previous: int, current: int) -> int:
    delta = (int(current) - int(previous)) % 4
    if delta == 0:
        return 0
    if delta == 1:
        return 1
    if delta == 3:
        return -1
    raise ValueError("observable target jumped to an opposite source slot")


def _exact_query_indices(
    dense_time_s: np.ndarray,
    query_time_s: np.ndarray,
    tolerance_s: float,
) -> np.ndarray:
    result = np.empty(query_time_s.size, dtype=np.int64)
    for index, query in enumerate(query_time_s):
        distance = np.abs(dense_time_s - query)
        minimum = float(distance.min())
        matches = np.flatnonzero(distance == minimum)
        if matches.size != 1 or minimum > tolerance_s:
            raise ValueError("each query requires exactly one dense future frame")
        result[index] = int(matches[0])
    return result


def construct_future_targets_from_current_source(
    dense_future_position_m: np.ndarray,
    dense_future_time_s: np.ndarray,
    query_time_s: np.ndarray,
    rule_query: np.ndarray,
    current_source: int,
    current_position_m: np.ndarray,
    *,
    candidate_steps: tuple[int, ...] = DEFAULT_CANDIDATE_STEPS,
    tie_epsilon_m: float = 1e-6,
    query_match_tolerance_s: float = 2e-6,
) -> dict[str, np.ndarray]:
    """Build anonymous future labels from an explicitly chosen q0 handle.

    The handle index is construction-only metadata.  Exported labels remain
    relative signed steps and positions, so choosing the armor that was
    actually observed at q0 does not introduce a persistent physical ID.
    """
    dense_position = np.asarray(dense_future_position_m, dtype=np.float32)
    dense_time = np.asarray(dense_future_time_s, dtype=np.float64)
    tau = np.asarray(query_time_s, dtype=np.float32)
    rule = np.asarray(rule_query, dtype=np.bool_)
    current_position = np.asarray(current_position_m, dtype=np.float32)
    steps = np.asarray(candidate_steps, dtype=np.int64)
    if dense_position.ndim != 3 or dense_position.shape[1:] != (4, 3):
        raise ValueError("dense_future_position_m must have shape [U,4,3]")
    if dense_time.shape != dense_position.shape[:1] or dense_time.size < 2:
        raise ValueError("a nontrivial dense future time stream is required")
    if tau.ndim != 1 or rule.shape != tau.shape:
        raise ValueError("query_time_s and rule_query must have shape [Q]")
    if not 0 <= int(current_source) < 4 or current_position.shape != (3,):
        raise ValueError("current source/position is invalid")
    if steps.ndim != 1 or steps.size < 3 or len(set(steps.tolist())) != steps.size:
        raise ValueError("candidate_steps must be a unique one-dimensional set")
    if int(np.count_nonzero(steps == 0)) != 1:
        raise ValueError("candidate_steps must contain exactly one zero anchor")
    if tie_epsilon_m < 0 or query_match_tolerance_s < 0:
        raise ValueError("future-label tolerances must be non-negative")
    if not np.isfinite(tau).all() or np.any(tau < 0):
        raise ValueError("query times must be finite and non-negative")
    if not np.isfinite(dense_time).all() or np.any(np.diff(dense_time) <= 0):
        raise ValueError("dense future times must be finite and strictly increasing")
    if not np.isclose(float(dense_time[0]), 0.0, atol=query_match_tolerance_s):
        raise ValueError("dense future must start at q0")
    if not np.allclose(
        dense_position[0, int(current_source)], current_position,
        atol=5e-5, rtol=0.0,
    ):
        raise ValueError("chosen current position and dense q0 source disagree")
    zero_queries = np.flatnonzero(tau == 0.0)
    if zero_queries.size < 1 or not bool(rule[zero_queries].all()):
        raise ValueError("queries must contain at least one eligible exact q0")
    query_indices = _exact_query_indices(
        dense_time, tau, query_match_tolerance_s
    )

    dense_selected = np.empty(dense_time.size, dtype=np.int64)
    dense_switch_count = np.zeros(dense_time.size, dtype=np.int64)
    dense_selected[0] = int(current_source)
    previous = int(current_source)
    total = 0
    for row in range(1, dense_time.size):
        selected = _select_with_continuity(
            dense_position[row], previous, tie_epsilon_m=tie_epsilon_m
        )
        total += _signed_adjacent_step(previous, selected)
        dense_selected[row] = selected
        dense_switch_count[row] = total
        previous = selected

    target_switch = dense_switch_count[query_indices]
    selected_query_position = dense_position[
        query_indices, dense_selected[query_indices]
    ]
    target_delta = selected_query_position - current_position[None, :]
    target_switch[zero_queries] = 0
    target_delta[zero_queries] = 0.0
    target_mask = rule & np.isin(target_switch, steps)
    target_mask[zero_queries] = True

    q0_all = dense_position[0]
    candidate_relation = np.stack([
        q0_all[(int(current_source) + int(step)) % 4] - current_position
        for step in steps
    ]).astype(np.float32, copy=False)
    current_row = int(np.flatnonzero(steps == 0)[0])
    candidate_relation[current_row] = 0.0
    target_onehot = steps[None, :] == target_switch[:, None]
    target_onehot &= target_mask[:, None]
    if np.any(target_mask & (target_onehot.sum(axis=1) != 1)):
        raise RuntimeError("eligible query does not have exactly one candidate branch")
    return {
        "candidate_relation_m": candidate_relation,
        "candidate_step": steps.copy(),
        "candidate_mask": np.ones(steps.size, dtype=np.bool_),
        "candidate_confidence": np.ones(steps.size, dtype=np.float32),
        "tau_s": tau,
        "target_switch_count": target_switch,
        "target_candidate_onehot": target_onehot,
        "target_visible_delta_m": target_delta.astype(np.float32, copy=False),
        "target_query_mask": target_mask,
    }


def construct_observable_future_sample_from_selected_history(
    history_position_m: np.ndarray,
    selected_event_indices: np.ndarray,
    selected_source_slots: np.ndarray,
    selected_event_mask: np.ndarray,
    history_time_s: np.ndarray,
    dense_future_position_m: np.ndarray,
    dense_future_time_s: np.ndarray,
    query_time_s: np.ndarray,
    rule_query: np.ndarray,
    *,
    future_targets: dict[str, np.ndarray] | None = None,
    candidate_steps: tuple[int, ...] = DEFAULT_CANDIDATE_STEPS,
    tie_epsilon_m: float = 1e-6,
    query_match_tolerance_s: float = 2e-6,
) -> dict[str, np.ndarray]:
    """Build F tensors for a construction-time selected observed track.

    Only a contiguous active suffix is accepted.  Earlier, incoherent history
    remains explicitly masked instead of rejecting the whole window or
    fabricating an opposite-armor switch.
    """
    history = np.asarray(history_position_m, dtype=np.float32)
    events = np.asarray(selected_event_indices, dtype=np.int64)
    slots = np.asarray(selected_source_slots, dtype=np.int64)
    mask = np.asarray(selected_event_mask, dtype=np.bool_)
    event_time = np.asarray(history_time_s, dtype=np.float32)
    if history.ndim != 3 or history.shape[1:] != (4, 3):
        raise ValueError("history_position_m must have shape [T,4,3]")
    if events.shape != (32,) or slots.shape != (32,) or mask.shape != (32,):
        raise ValueError("selected observed history must contain 32 events")
    if event_time.shape != history.shape[:1]:
        raise ValueError("history_time_s must have shape [T]")
    if np.any(events < 0) or np.any(events >= history.shape[0]):
        raise ValueError("selected history event index is out of range")
    active = np.flatnonzero(mask)
    if active.size < 1 or int(active[-1]) != 31:
        raise ValueError("selected history must end with an active q0 event")
    if not np.array_equal(active, np.arange(int(active[0]), 32)):
        raise ValueError("selected history mask must be a contiguous suffix")
    if np.any((slots[mask] < 0) | (slots[mask] >= 4)):
        raise ValueError("active selected source slots must be in [0,3]")
    selected_time = event_time[events]
    if not np.isclose(float(selected_time[-1]), 0.0, atol=1e-6):
        raise ValueError("the final selected event must be exact q0")
    if np.any(np.diff(selected_time) <= 0):
        raise ValueError("selected history event times must be strictly increasing")

    current_source = int(slots[-1])
    current_position = history[events[-1], current_source].copy()
    history_relative = np.zeros((32, 3), dtype=np.float32)
    history_step = np.zeros(32, dtype=np.int64)
    history_dt = np.zeros(32, dtype=np.float32)
    for row in active:
        history_relative[row] = (
            history[events[row], int(slots[row])] - current_position
        )
        if row > int(active[0]):
            history_step[row] = _signed_adjacent_step(
                int(slots[row - 1]), int(slots[row])
            )
            history_dt[row] = selected_time[row] - selected_time[row - 1]
    history_relative[-1] = 0.0

    future = (
        construct_future_targets_from_current_source(
            dense_future_position_m, dense_future_time_s, query_time_s,
            rule_query, current_source, current_position,
            candidate_steps=candidate_steps,
            tie_epsilon_m=tie_epsilon_m,
            query_match_tolerance_s=query_match_tolerance_s,
        )
        if future_targets is None
        else {key: value.copy() for key, value in future_targets.items()}
    )
    return {
        "history_position_rel_m": history_relative,
        "history_time_s": selected_time.astype(np.float32, copy=False),
        "history_dt_s": history_dt,
        "history_switch_step": history_step,
        "history_mask": mask.copy(),
        "current_position_m": current_position.astype(np.float32, copy=False),
        **future,
    }


def construct_observable_future_sample(
    history_position_m: np.ndarray,
    history_event_mask: np.ndarray,
    history_time_s: np.ndarray,
    dense_future_position_m: np.ndarray,
    dense_future_time_s: np.ndarray,
    query_time_s: np.ndarray,
    rule_query: np.ndarray,
    *,
    history_events: int = 32,
    candidate_steps: tuple[int, ...] = DEFAULT_CANDIDATE_STEPS,
    tie_epsilon_m: float = 1e-6,
    query_match_tolerance_s: float = 2e-6,
) -> dict[str, np.ndarray]:
    """Collapse complete physical truth into the anonymous F contract.

    ``dense_future_*`` must contain q0 and every visibility transition up to
    the maximum query.  It may contain extra times and queries may be in any
    order.  A missing dense stream fails closed instead of falling back to
    endpoint modulo-four labels.
    """
    history = np.asarray(history_position_m, dtype=np.float32)
    event_mask = np.asarray(history_event_mask, dtype=np.bool_)
    event_time = np.asarray(history_time_s, dtype=np.float32)
    dense_position = np.asarray(dense_future_position_m, dtype=np.float32)
    dense_time = np.asarray(dense_future_time_s, dtype=np.float64)
    tau = np.asarray(query_time_s, dtype=np.float32)
    rule = np.asarray(rule_query, dtype=np.bool_)
    steps = np.asarray(candidate_steps, dtype=np.int64)
    if history.ndim != 3 or history.shape[1:] != (4, 3):
        raise ValueError("history_position_m must have shape [T,4,3]")
    if event_mask.shape != history.shape[:1] or event_time.shape != event_mask.shape:
        raise ValueError("history mask/time must have shape [T]")
    if dense_position.ndim != 3 or dense_position.shape[1:] != (4, 3):
        raise ValueError("dense_future_position_m must have shape [U,4,3]")
    if dense_time.shape != dense_position.shape[:1] or dense_time.size < 2:
        raise ValueError("a nontrivial dense future time stream is required")
    if tau.ndim != 1 or rule.shape != tau.shape:
        raise ValueError("query_time_s and rule_query must have shape [Q]")
    if history_events != 32:
        raise ValueError("the current source qualification permits exactly 32 events")
    if steps.ndim != 1 or steps.size < 3 or len(set(steps.tolist())) != steps.size:
        raise ValueError("candidate_steps must be a unique one-dimensional set")
    if int(np.count_nonzero(steps == 0)) != 1:
        raise ValueError("candidate_steps must contain exactly one zero anchor")
    if tie_epsilon_m < 0 or query_match_tolerance_s < 0:
        raise ValueError("visibility tolerances must be non-negative")
    if not np.isfinite(tau).all() or np.any(tau < 0):
        raise ValueError("query times must be finite and non-negative")
    if not np.isfinite(dense_time).all() or np.any(np.diff(dense_time) <= 0):
        raise ValueError("dense future times must be finite and strictly increasing")
    if not np.isclose(float(dense_time[0]), 0.0, atol=query_match_tolerance_s):
        raise ValueError("dense future must start at q0")
    zero_queries = np.flatnonzero(tau == 0.0)
    if zero_queries.size < 1 or not bool(rule[zero_queries].all()):
        raise ValueError("queries must contain at least one eligible exact q0")
    query_indices = _exact_query_indices(dense_time, tau, query_match_tolerance_s)

    valid_history = np.flatnonzero(
        event_mask & np.isfinite(event_time) & (event_time <= 1e-6)
    )
    if valid_history.size < history_events:
        raise ValueError("sample has fewer than 32 qualified history events")
    valid_history = valid_history[-history_events:]
    if not np.isclose(float(event_time[valid_history[-1]]), 0.0, atol=1e-6):
        raise ValueError("the final history event must be exact q0")
    if not np.allclose(
        history[valid_history[-1]], dense_position[0], atol=5e-5, rtol=0.0
    ):
        raise ValueError("history and dense physical truth disagree at q0")

    history_selected = np.empty(history_events, dtype=np.int64)
    previous: int | None = None
    for row, source_index in enumerate(valid_history):
        previous = _select_with_continuity(
            history[source_index], previous, tie_epsilon_m=tie_epsilon_m
        )
        history_selected[row] = previous
    current_source = int(history_selected[-1])
    current_position = history[valid_history[-1], current_source].copy()

    history_step = np.zeros(history_events, dtype=np.int64)
    for row in range(1, history_events):
        history_step[row] = _signed_adjacent_step(
            int(history_selected[row - 1]), int(history_selected[row])
        )
    selected_history_position = history[
        valid_history, history_selected
    ]
    history_relative = selected_history_position - current_position[None, :]
    history_dt = np.zeros(history_events, dtype=np.float32)
    history_dt[1:] = np.diff(event_time[valid_history])
    if np.any(history_dt[1:] <= 0):
        raise ValueError("valid history event times must be strictly increasing")

    dense_selected = np.empty(dense_time.size, dtype=np.int64)
    dense_switch_count = np.zeros(dense_time.size, dtype=np.int64)
    dense_selected[0] = current_source  # q0 inherits history; never re-argmin.
    previous = current_source
    total = 0
    for row in range(1, dense_time.size):
        selected = _select_with_continuity(
            dense_position[row], previous, tie_epsilon_m=tie_epsilon_m
        )
        total += _signed_adjacent_step(previous, selected)
        dense_selected[row] = selected
        dense_switch_count[row] = total
        previous = selected

    target_switch = dense_switch_count[query_indices]
    selected_query_position = dense_position[
        query_indices, dense_selected[query_indices]
    ]
    target_delta = selected_query_position - current_position[None, :]
    target_switch[zero_queries] = 0
    target_delta[zero_queries] = 0.0
    covered = np.isin(target_switch, steps)
    target_mask = rule & covered
    target_mask[zero_queries] = True

    q0_all = dense_position[0]
    candidate_relation = np.stack(
        [q0_all[(current_source + int(step)) % 4] - current_position for step in steps],
        axis=0,
    ).astype(np.float32, copy=False)
    current_row = int(np.flatnonzero(steps == 0)[0])
    candidate_relation[current_row] = 0.0
    candidate_mask = np.ones(steps.size, dtype=np.bool_)
    candidate_confidence = np.ones(steps.size, dtype=np.float32)
    target_onehot = steps[None, :] == target_switch[:, None]
    target_onehot &= target_mask[:, None]
    if np.any(target_mask & (target_onehot.sum(axis=1) != 1)):
        raise RuntimeError("eligible query does not have exactly one candidate branch")

    return {
        "history_position_rel_m": history_relative.astype(np.float32, copy=False),
        "history_time_s": event_time[valid_history].astype(np.float32, copy=False),
        "history_dt_s": history_dt,
        "history_switch_step": history_step,
        "history_mask": np.ones(history_events, dtype=np.bool_),
        "current_position_m": current_position.astype(np.float32, copy=False),
        "candidate_relation_m": candidate_relation,
        "candidate_step": steps.copy(),
        "candidate_mask": candidate_mask,
        "candidate_confidence": candidate_confidence,
        "tau_s": tau,
        "target_switch_count": target_switch,
        "target_candidate_onehot": target_onehot,
        "target_visible_delta_m": target_delta.astype(np.float32, copy=False),
        "target_query_mask": target_mask,
    }


class ObservableFutureDataset(IterableDataset):
    """Load a built anonymous derivative; test data is intentionally forbidden."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        expert: str,
        *,
        seed: int,
        shuffle: bool = False,
        sample_limit: int = 0,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir).resolve()
        if split not in {"train", "validation"}:
            raise ValueError("observable F loader only permits train or validation")
        if expert not in EXPERT_TO_MOTION_CLASS:
            raise ValueError("expert must be translation, rotation, or combined")
        self.split = split
        self.expert = expert
        self.motion_class = EXPERT_TO_MOTION_CLASS[expert]
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.sample_limit = int(sample_limit)
        self.epoch = 0

        manifest_path = self.dataset_dir / "dataset_manifest.json"
        self.manifest_sha256 = _sha256(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("observable F dataset schema mismatch")
        if not bool(self.manifest.get("qualification_passed", False)):
            raise ValueError("observable F dataset is not qualified")
        if bool(self.manifest.get("test_accessed", True)):
            raise ValueError("observable F dataset must keep test_accessed=false")
        if self.manifest.get("candidate_steps") != list(DEFAULT_CANDIDATE_STEPS):
            raise ValueError("observable F candidate-step contract mismatch")
        self.shards = [
            item for item in self.manifest["shards"] if item["split"] == split
        ]
        if not self.shards:
            raise ValueError(f"observable F dataset has no {split} shards")
        for item in self.shards:
            path = _manifest_path(self.dataset_dir, item["path"])
            if _sha256(path) != str(item["sha256"]):
                raise ValueError(f"observable F shard hash mismatch: {path}")
        self._source_length = sum(int(item["sample_count"]) for item in self.shards)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return min(self._source_length, self.sample_limit) if self.sample_limit > 0 else self._source_length

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if torch is None:
            raise RuntimeError("ObservableFutureDataset requires PyTorch")
        rng = np.random.default_rng(self.seed + self.epoch * 100003)
        shard_order = np.arange(len(self.shards))
        if self.shuffle:
            rng.shuffle(shard_order)
        emitted = 0
        for shard_index in shard_order:
            item = self.shards[int(shard_index)]
            path = _manifest_path(self.dataset_dir, item["path"])
            if _sha256(path) != str(item["sha256"]):
                raise ValueError(f"observable F shard changed before open: {path}")
            with np.load(path, allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            indices = np.arange(int(item["sample_count"]))
            if self.shuffle:
                rng.shuffle(indices)
            for index in indices:
                if int(arrays["motion_class"][index]) != self.motion_class:
                    continue
                if self.sample_limit > 0 and emitted >= self.sample_limit:
                    return
                emitted += 1
                yield {
                    key: torch.from_numpy(value[index].copy())
                    for key, value in arrays.items()
                    if key not in {"motion_class"}
                }
