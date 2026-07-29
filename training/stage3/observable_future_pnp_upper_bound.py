"""Real-PnP, oracle-associated inputs for observable F.

This module intentionally does *not* define a deployable PnP tracker.  It uses
same-exposure physical truth at ``t <= q0`` to associate unordered PnP rows.
The legacy v1 path requires the clean-rule primary at all 32 events.  The v2
path instead builds a coherent anonymous track through actually observed
plates, masks an incoherent older prefix, and rebuilds all labels from the same
observed q0 role.

All PnP points must first be rebased from their exposure tracker frame into the
q0 anchor tracker frame.  Future truth is never used while constructing model
inputs.  Temporary physical slots and assignments are not exported.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from .observable_future_dataset import (
    _select_with_continuity,
    construct_observable_future_sample_from_selected_history,
)


SCHEMA_VERSION = "stage3-observable-future-real-pnp-upper-bound-v1"
OBSERVED_STREAM_SCHEMA_VERSION = (
    "stage3-observable-future-real-pnp-observed-stream-v2"
)
EXPERIMENT_KIND = "real_pnp_oracle_association_truth_s_upper_bound"
OBSERVED_STREAM_EXPERIMENT_KIND = (
    "real_pnp_oracle_association_observed_primary_stream"
)
FORWARD_KEYS = (
    "history_position_rel_m",
    "history_time_s",
    "history_dt_s",
    "history_switch_step",
    "history_mask",
    "current_position_m",
    "candidate_relation_m",
    "candidate_step",
    "candidate_mask",
    "candidate_confidence",
    "tau_s",
)


def rebase_tracker_points_to_anchor(
    position_m: np.ndarray,
    event_origin_world_m: np.ndarray,
    event_tracker_to_world_rotation: np.ndarray,
    anchor_origin_world_m: np.ndarray,
    anchor_tracker_to_world_rotation: np.ndarray,
) -> np.ndarray:
    """Express event-local tracker points in the q0 anchor tracker frame."""
    position = np.asarray(position_m, dtype=np.float64)
    event_origin = np.asarray(event_origin_world_m, dtype=np.float64)
    event_rotation = np.asarray(event_tracker_to_world_rotation, dtype=np.float64)
    anchor_origin = np.asarray(anchor_origin_world_m, dtype=np.float64)
    anchor_rotation = np.asarray(anchor_tracker_to_world_rotation, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position_m must have shape [N,3]")
    if event_origin.shape != (3,) or anchor_origin.shape != (3,):
        raise ValueError("tracker origins must have shape [3]")
    if event_rotation.shape != (3, 3) or anchor_rotation.shape != (3, 3):
        raise ValueError("tracker rotations must have shape [3,3]")
    if not all(np.isfinite(value).all() for value in (
        position, event_origin, event_rotation, anchor_origin, anchor_rotation,
    )):
        raise ValueError("tracker rebase inputs must be finite")
    world = position @ event_rotation.T + event_origin[None, :]
    return ((world - anchor_origin[None, :]) @ anchor_rotation).astype(
        np.float32, copy=False
    )


def _anchor_points_to_event_tracker(
    position_m: np.ndarray,
    event_origin_world_m: np.ndarray,
    event_tracker_to_world_rotation: np.ndarray,
    anchor_origin_world_m: np.ndarray,
    anchor_tracker_to_world_rotation: np.ndarray,
) -> np.ndarray:
    """Express q0-anchor tracker points in one exposure's tracker frame."""
    position = np.asarray(position_m, dtype=np.float64)
    event_origin = np.asarray(event_origin_world_m, dtype=np.float64)
    event_rotation = np.asarray(event_tracker_to_world_rotation, dtype=np.float64)
    anchor_origin = np.asarray(anchor_origin_world_m, dtype=np.float64)
    anchor_rotation = np.asarray(anchor_tracker_to_world_rotation, dtype=np.float64)
    world = position @ anchor_rotation.T + anchor_origin[None, :]
    return ((world - event_origin[None, :]) @ event_rotation).astype(
        np.float32, copy=False
    )


def oracle_injective_assignment(
    observation_position_m: np.ndarray,
    truth_position_m: np.ndarray,
    *,
    ambiguity_epsilon_m: float,
) -> tuple[float, dict[int, int], bool] | None:
    """Minimum-cost truth-slot -> observation-row assignment.

    The ambiguity flag is invariant to observation row order.  It is true when
    a second distinct assignment is within ``ambiguity_epsilon_m`` mean cost of
    the optimum.  Ambiguous assignments must not enter the strict upper bound.
    """
    observation = np.asarray(observation_position_m, dtype=np.float64)
    truth = np.asarray(truth_position_m, dtype=np.float64)
    if observation.ndim != 2 or observation.shape[1] != 3:
        raise ValueError("observation positions must have shape [N,3]")
    if truth.shape != (4, 3):
        raise ValueError("truth positions must have shape [4,3]")
    if ambiguity_epsilon_m < 0:
        raise ValueError("association ambiguity epsilon must be non-negative")
    count = int(observation.shape[0])
    if count < 1 or count > 4:
        return None
    if not np.isfinite(observation).all() or not np.isfinite(truth).all():
        return None
    distances = np.linalg.norm(
        observation[:, None, :] - truth[None, :, :], axis=-1
    )
    scored: list[tuple[float, tuple[int, ...]]] = []
    for truth_slots in itertools.permutations(range(4), count):
        cost = float(sum(distances[row, slot] for row, slot in enumerate(truth_slots)))
        scored.append((cost / count, tuple(int(slot) for slot in truth_slots)))
    scored.sort(key=lambda item: (item[0], item[1]))
    best_cost, best_slots = scored[0]
    ambiguous = (
        len(scored) > 1
        and scored[1][0] - best_cost <= float(ambiguity_epsilon_m)
    )
    mapping = {slot: row for row, slot in enumerate(best_slots)}
    return best_cost, mapping, ambiguous


def _selected_history_slots(
    history_position_m: np.ndarray,
    event_mask: np.ndarray,
    event_time_s: np.ndarray,
    *,
    tie_epsilon_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    history = np.asarray(history_position_m, dtype=np.float32)
    mask = np.asarray(event_mask, dtype=np.bool_)
    time = np.asarray(event_time_s, dtype=np.float32)
    valid = np.flatnonzero(mask & np.isfinite(time) & (time <= 1e-6))
    if valid.size < 32:
        raise ValueError("PnP upper bound requires the qualified last 32 events")
    valid = valid[-32:]
    selected = np.empty(32, dtype=np.int64)
    previous: int | None = None
    for row, event in enumerate(valid):
        previous = _select_with_continuity(
            history[event], previous, tie_epsilon_m=tie_epsilon_m
        )
        selected[row] = previous
    return valid, selected


def associate_observed_primary_history(
    physical_event_mask: np.ndarray,
    physical_event_time_s: np.ndarray,
    truth_history_position_m: np.ndarray,
    truth_history_mask: np.ndarray,
    observation_position_m: np.ndarray,
    observation_mask: np.ndarray,
    event_origins_world_m: np.ndarray,
    event_tracker_to_world_rotation: np.ndarray,
    anchor_origin_world_m: np.ndarray,
    anchor_tracker_to_world_rotation: np.ndarray,
    *,
    primary_tie_epsilon_m: float = 1e-6,
    primary_switch_hysteresis_m: float = 0.02,
    association_ambiguity_epsilon_m: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Associate a coherent anonymous suffix through actually observed plates.

    The q0 primary is the nearest real PnP observation in its exposure-local
    tracker frame.  Earlier primaries are selected by a minimum-range dynamic
    program constrained to same/adjacent cyclic transitions.  If no coherent
    predecessor exists, older history is masked instead of inventing a
    two-plate jump.  Truth slots are temporary construction handles only.
    """
    event_mask = np.asarray(physical_event_mask, dtype=np.bool_)
    event_time = np.asarray(physical_event_time_s, dtype=np.float32)
    valid_events = np.flatnonzero(
        event_mask & np.isfinite(event_time) & (event_time <= 1e-6)
    )
    if valid_events.size < 32:
        raise ValueError("observed primary association requires the last 32 events")
    valid_events = valid_events[-32:]
    truth = np.asarray(truth_history_position_m, dtype=np.float32)
    truth_mask = np.asarray(truth_history_mask, dtype=np.bool_)
    observation = np.asarray(observation_position_m, dtype=np.float32)
    obs_mask = np.asarray(observation_mask, dtype=np.bool_)
    origins = np.asarray(event_origins_world_m, dtype=np.float64)
    rotations = np.asarray(event_tracker_to_world_rotation, dtype=np.float64)
    if truth.ndim != 3 or truth.shape[1:] != (4, 3):
        raise ValueError("truth history must have shape [T,4,3]")
    if truth_mask.shape != truth.shape[:2]:
        raise ValueError("truth history mask must have shape [T,4]")
    if observation.ndim != 3 or observation.shape[1:] != (4, 3):
        raise ValueError("observation positions must have shape [T,4,3]")
    if obs_mask.shape != observation.shape[:2]:
        raise ValueError("observation mask must have shape [T,4]")
    if origins.shape != (32, 3) or rotations.shape != (32, 3, 3):
        raise ValueError("event rebase poses must cover the selected 32 events")
    if primary_tie_epsilon_m < 0 or primary_switch_hysteresis_m < 0:
        raise ValueError("primary selection tolerances must be non-negative")

    handle_position = np.zeros((32, 4, 3), dtype=np.float32)
    handle_mask = np.zeros((32, 4), dtype=np.bool_)
    local_range = np.full((32, 4), np.inf, dtype=np.float64)
    selection_range = np.full((32, 4), np.inf, dtype=np.float64)
    association_error = np.zeros((32, 4), dtype=np.float32)
    ambiguous = np.zeros(32, dtype=np.bool_)
    candidate_count = np.zeros(32, dtype=np.int64)
    for row, event in enumerate(valid_events):
        if not bool(truth_mask[event].all()):
            continue
        row_mask = obs_mask[event]
        candidate_count[row] = int(row_mask.sum())
        local = observation[event, row_mask]
        rebased = rebase_tracker_points_to_anchor(
            local, origins[row], rotations[row],
            anchor_origin_world_m, anchor_tracker_to_world_rotation,
        )
        match = oracle_injective_assignment(
            rebased, truth[event],
            ambiguity_epsilon_m=association_ambiguity_epsilon_m,
        )
        if match is None:
            continue
        _, assignment, is_ambiguous = match
        ambiguous[row] = bool(is_ambiguous)
        if is_ambiguous:
            continue
        active_rows = np.flatnonzero(row_mask)
        truth_event_local = _anchor_points_to_event_tracker(
            truth[event], origins[row], rotations[row],
            anchor_origin_world_m, anchor_tracker_to_world_rotation,
        )
        for slot, compact_row in assignment.items():
            slot_int = int(slot)
            compact_int = int(compact_row)
            handle_position[row, slot_int] = rebased[compact_int]
            handle_mask[row, slot_int] = True
            local_range[row, slot_int] = float(np.linalg.norm(
                observation[event, int(active_rows[compact_int]), :2]
            ))
            selection_range[row, slot_int] = float(np.linalg.norm(
                truth_event_local[slot_int, :2]
            ))
            association_error[row, slot_int] = np.float32(np.linalg.norm(
                rebased[compact_int] - truth[event, slot_int]
            ))

    selected_slot = np.full(32, -1, dtype=np.int64)
    selected_mask = np.zeros(32, dtype=np.bool_)
    q0_slots = np.flatnonzero(handle_mask[-1])
    q0_tied = False
    if q0_slots.size:
        order = q0_slots[np.argsort(
            selection_range[-1, q0_slots], kind="stable"
        )]
        q0_tied = bool(
            order.size > 1
            and selection_range[-1, order[1]] - selection_range[-1, order[0]]
            <= primary_tie_epsilon_m
        )
        if not q0_tied:
            q0 = int(order[0])
            # State is (temporary handle, established switch direction).  A
            # nonzero direction cannot reverse inside one coherent suffix.
            next_cost: dict[tuple[int, int], float] = {(q0, 0): 0.0}
            next_choice: list[
                dict[tuple[int, int], tuple[int, int]]
            ] = [dict() for _ in range(32)]
            start = 31
            for row in range(30, -1, -1):
                slots = np.flatnonzero(handle_mask[row])
                if slots.size == 0:
                    break
                minimum_range = float(selection_range[row, slots].min())
                current_cost: dict[tuple[int, int], float] = {}
                for slot_raw in slots:
                    slot = int(slot_raw)
                    for (next_slot, next_direction), cost in next_cost.items():
                        delta = (next_slot - slot) % 4
                        if delta == 0:
                            step = 0
                        elif delta == 1:
                            step = 1
                        elif delta == 3:
                            step = -1
                        else:
                            continue
                        if step and next_direction not in (0, step):
                            continue
                        direction = step if step else next_direction
                        state = (slot, direction)
                        candidate_cost = (
                            cost
                            + float(selection_range[row, slot] - minimum_range)
                            + primary_switch_hysteresis_m * int(step != 0)
                        )
                        previous_cost = current_cost.get(state)
                        if previous_cost is None or candidate_cost < previous_cost - 1e-12:
                            current_cost[state] = candidate_cost
                            next_choice[row][state] = (next_slot, next_direction)
                        elif abs(candidate_cost - previous_cost) <= 1e-12:
                            previous_next = next_choice[row][state]
                            candidate_key = tuple(
                                float(x) for x in handle_position[row + 1, next_slot]
                            )
                            previous_key = tuple(
                                float(x) for x in handle_position[
                                    row + 1, previous_next[0]
                                ]
                            )
                            if candidate_key < previous_key:
                                next_choice[row][state] = (
                                    next_slot, next_direction
                                )
                if not current_cost:
                    break
                next_cost = current_cost
                start = row

            first_state = min(
                next_cost,
                key=lambda value: (
                    next_cost[value],
                    tuple(
                        float(x) for x in handle_position[start, value[0]]
                    ),
                ),
            )
            selected_slot[start] = int(first_state[0])
            selected_mask[start] = True
            state = first_state
            for row in range(start, 31):
                state = next_choice[row][state]
                selected_slot[row + 1] = int(state[0])
                selected_mask[row + 1] = True

    selected_error = np.zeros(32, dtype=np.float32)
    active = np.flatnonzero(selected_mask)
    if active.size:
        selected_error[active] = association_error[
            active, selected_slot[active]
        ]
    return {
        "valid_event_index": valid_events.astype(np.int64, copy=False),
        "handle_position_m": handle_position,
        "handle_mask": handle_mask,
        "local_horizontal_range_m": local_range,
        "selection_horizontal_range_m": selection_range,
        "selected_source_slot": selected_slot,
        "selected_event_mask": selected_mask,
        "selected_association_error_m": selected_error,
        "ambiguous_event_mask": ambiguous,
        "candidate_count": candidate_count,
        "q0_tied": np.asarray(q0_tied, dtype=np.bool_),
    }


def construct_observed_primary_pnp_sample(
    clean_fallback_sample: dict[str, np.ndarray],
    physical_history_position_m: np.ndarray,
    physical_event_mask: np.ndarray,
    physical_event_time_s: np.ndarray,
    dense_future_position_m: np.ndarray,
    dense_future_time_s: np.ndarray,
    query_time_s: np.ndarray,
    rule_query: np.ndarray,
    truth_history_position_m: np.ndarray,
    truth_history_mask: np.ndarray,
    observation_position_m: np.ndarray,
    observation_mask: np.ndarray,
    event_origins_world_m: np.ndarray,
    event_tracker_to_world_rotation: np.ndarray,
    anchor_origin_world_m: np.ndarray,
    anchor_tracker_to_world_rotation: np.ndarray,
    *,
    minimum_history_events: int = 8,
    tie_epsilon_m: float = 1e-6,
    primary_switch_hysteresis_m: float = 0.02,
    query_match_tolerance_s: float = 2e-6,
    association_ambiguity_epsilon_m: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Build a paired clean/PnP sample around the actual observed q0 plate."""
    if minimum_history_events < 2 or minimum_history_events > 32:
        raise ValueError("minimum observed history must be within [2,32]")
    association = associate_observed_primary_history(
        physical_event_mask, physical_event_time_s,
        truth_history_position_m, truth_history_mask,
        observation_position_m, observation_mask,
        event_origins_world_m, event_tracker_to_world_rotation,
        anchor_origin_world_m, anchor_tracker_to_world_rotation,
        primary_tie_epsilon_m=tie_epsilon_m,
        primary_switch_hysteresis_m=primary_switch_hysteresis_m,
        association_ambiguity_epsilon_m=association_ambiguity_epsilon_m,
    )
    selected_mask = association["selected_event_mask"]
    selected_slot = association["selected_source_slot"]
    active_count = int(selected_mask.sum())
    q0_associated = bool(selected_mask[-1])
    usable = q0_associated and active_count >= minimum_history_events
    future_coherent = q0_associated

    if q0_associated:
        try:
            clean = construct_observable_future_sample_from_selected_history(
                physical_history_position_m,
                association["valid_event_index"], selected_slot, selected_mask,
                physical_event_time_s,
                dense_future_position_m, dense_future_time_s,
                query_time_s, rule_query,
                tie_epsilon_m=tie_epsilon_m,
                query_match_tolerance_s=query_match_tolerance_s,
            )
        except ValueError as error:
            if "opposite source slot" not in str(error):
                raise
            future_coherent = False
            usable = False
            clean = {
                key: value.copy() for key, value in clean_fallback_sample.items()
            }
    else:
        clean = {key: value.copy() for key, value in clean_fallback_sample.items()}

    pnp_history_position = np.zeros((32, 3), dtype=np.float32)
    pnp_current = np.zeros(3, dtype=np.float32)
    pnp_candidate_relation = np.zeros_like(
        clean["candidate_relation_m"], dtype=np.float32
    )
    pnp_target_delta = np.zeros_like(
        clean["target_visible_delta_m"], dtype=np.float32
    )
    if q0_associated:
        pnp_handle = association["handle_position_m"]
        pnp_current = pnp_handle[-1, int(selected_slot[-1])].copy()
        active = np.flatnonzero(selected_mask)
        pnp_history_position[active] = (
            pnp_handle[active, selected_slot[active]] - pnp_current[None, :]
        )
        pnp_history_position[-1] = 0.0
        anchor_shift = clean["current_position_m"] - pnp_current
        pnp_candidate_relation = (
            clean["candidate_relation_m"] + anchor_shift[None, :]
        ).astype(np.float32, copy=False)
        current_role = np.remainder(clean["candidate_step"], 4) == 0
        pnp_candidate_relation[current_role] = 0.0
        future_absolute = (
            clean["current_position_m"][None, :]
            + clean["target_visible_delta_m"]
        )
        pnp_target_delta = (future_absolute - pnp_current[None, :]).astype(
            np.float32, copy=False
        )

    failure_code = (
        0 if usable else 1 if not q0_associated
        else 3 if not future_coherent else 2
    )
    full_history = bool(active_count == 32)
    return {
        **clean,
        "pnp_history_position_rel_m": pnp_history_position,
        "pnp_history_time_s": clean["history_time_s"].astype(
            np.float32, copy=True
        ),
        "pnp_history_dt_s": clean["history_dt_s"].astype(np.float32, copy=True),
        "pnp_history_switch_step": clean["history_switch_step"].astype(
            np.int64, copy=True
        ),
        "pnp_history_mask": selected_mask.copy(),
        "pnp_current_position_m": pnp_current,
        "pnp_candidate_relation_m": pnp_candidate_relation,
        "pnp_candidate_step": clean["candidate_step"].astype(np.int64, copy=True),
        "pnp_candidate_mask": clean["candidate_mask"].astype(np.bool_, copy=True),
        "pnp_candidate_confidence": clean["candidate_confidence"].astype(
            np.float32, copy=True
        ),
        "pnp_tau_s": clean["tau_s"].astype(np.float32, copy=True),
        "pnp_target_visible_delta_m": pnp_target_delta,
        "pnp_forward_usable": np.asarray(usable, dtype=np.bool_),
        "pnp_q0_associated": np.asarray(q0_associated, dtype=np.bool_),
        "pnp_full_history_associated": np.asarray(full_history, dtype=np.bool_),
        "pnp_failure_code": np.asarray(failure_code, dtype=np.int64),
        "pnp_history_associated_mask": selected_mask.copy(),
        "pnp_history_ambiguous_mask": association["ambiguous_event_mask"].copy(),
        "pnp_history_candidate_count": association["candidate_count"].copy(),
        "pnp_history_association_error_m": association[
            "selected_association_error_m"
        ].copy(),
        "pnp_q0_anchor_error_m": np.asarray(
            association["selected_association_error_m"][-1]
            if q0_associated else 0.0,
            dtype=np.float32,
        ),
        "pnp_history_active_count": np.asarray(active_count, dtype=np.int64),
        "pnp_history_track_break_count": np.asarray(
            int(q0_associated and active_count < 32), dtype=np.int64
        ),
        "pnp_future_label_coherent": np.asarray(
            future_coherent, dtype=np.bool_
        ),
    }


def construct_real_pnp_upper_bound_sample(
    clean_sample: dict[str, np.ndarray],
    physical_history_position_m: np.ndarray,
    physical_event_mask: np.ndarray,
    physical_event_time_s: np.ndarray,
    truth_history_position_m: np.ndarray,
    truth_history_mask: np.ndarray,
    observation_position_m: np.ndarray,
    observation_mask: np.ndarray,
    event_origins_world_m: np.ndarray,
    event_tracker_to_world_rotation: np.ndarray,
    anchor_origin_world_m: np.ndarray,
    anchor_tracker_to_world_rotation: np.ndarray,
    *,
    tie_epsilon_m: float = 1e-6,
    association_ambiguity_epsilon_m: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Build one strict, paired PnP arm while retaining coverage failures."""
    valid_events, selected_slots = _selected_history_slots(
        physical_history_position_m,
        physical_event_mask,
        physical_event_time_s,
        tie_epsilon_m=tie_epsilon_m,
    )
    if clean_sample["history_position_rel_m"].shape != (32, 3):
        raise ValueError("clean observable sample must use 32 history events")
    if not np.array_equal(
        clean_sample["history_time_s"],
        np.asarray(physical_event_time_s, dtype=np.float32)[valid_events],
    ):
        raise ValueError("clean history times do not match replayed source events")

    truth_history = np.asarray(truth_history_position_m, dtype=np.float32)
    truth_mask = np.asarray(truth_history_mask, dtype=np.bool_)
    observation = np.asarray(observation_position_m, dtype=np.float32)
    obs_mask = np.asarray(observation_mask, dtype=np.bool_)
    origins = np.asarray(event_origins_world_m, dtype=np.float64)
    rotations = np.asarray(event_tracker_to_world_rotation, dtype=np.float64)
    if truth_history.ndim != 3 or truth_history.shape[1:] != (4, 3):
        raise ValueError("truth history must have shape [T,4,3]")
    if truth_mask.shape != truth_history.shape[:2]:
        raise ValueError("truth history mask must have shape [T,4]")
    if observation.ndim != 3 or observation.shape[1:] != (4, 3):
        raise ValueError("observation positions must have shape [T,4,3]")
    if obs_mask.shape != observation.shape[:2]:
        raise ValueError("observation mask must have shape [T,4]")
    if origins.shape != (32, 3) or rotations.shape != (32, 3, 3):
        raise ValueError("event rebase poses must cover the selected 32 events")

    associated = np.zeros(32, dtype=np.bool_)
    ambiguous = np.zeros(32, dtype=np.bool_)
    candidate_count = np.zeros(32, dtype=np.int64)
    association_error = np.zeros(32, dtype=np.float32)
    selected_pnp = np.zeros((32, 3), dtype=np.float32)
    for row, (event, selected_slot) in enumerate(zip(valid_events, selected_slots)):
        if not bool(truth_mask[event].all()):
            continue
        row_mask = obs_mask[event]
        candidate_count[row] = int(row_mask.sum())
        rebased = rebase_tracker_points_to_anchor(
            observation[event, row_mask],
            origins[row],
            rotations[row],
            anchor_origin_world_m,
            anchor_tracker_to_world_rotation,
        )
        match = oracle_injective_assignment(
            rebased,
            truth_history[event],
            ambiguity_epsilon_m=association_ambiguity_epsilon_m,
        )
        if match is None:
            continue
        _, assignment, is_ambiguous = match
        ambiguous[row] = bool(is_ambiguous)
        if is_ambiguous or int(selected_slot) not in assignment:
            continue
        observation_row = assignment[int(selected_slot)]
        selected_pnp[row] = rebased[observation_row]
        association_error[row] = np.float32(np.linalg.norm(
            selected_pnp[row] - truth_history[event, int(selected_slot)]
        ))
        associated[row] = True

    full_history = bool(associated.all())
    q0_associated = bool(associated[-1])
    usable = full_history and q0_associated
    pnp_history_position = np.zeros((32, 3), dtype=np.float32)
    pnp_history_time = np.zeros(32, dtype=np.float32)
    pnp_history_dt = np.zeros(32, dtype=np.float32)
    pnp_history_switch = np.zeros(32, dtype=np.int64)
    pnp_history_mask = np.zeros(32, dtype=np.bool_)
    pnp_current = np.zeros(3, dtype=np.float32)
    pnp_candidate_relation = np.zeros_like(
        clean_sample["candidate_relation_m"], dtype=np.float32
    )
    pnp_target_delta = np.zeros_like(
        clean_sample["target_visible_delta_m"], dtype=np.float32
    )
    if usable:
        pnp_current = selected_pnp[-1].copy()
        pnp_history_position = selected_pnp - pnp_current[None, :]
        pnp_history_position[-1] = 0.0
        pnp_history_time = clean_sample["history_time_s"].astype(
            np.float32, copy=True
        )
        pnp_history_dt = clean_sample["history_dt_s"].astype(
            np.float32, copy=True
        )
        pnp_history_switch = clean_sample["history_switch_step"].astype(
            np.int64, copy=True
        )
        pnp_history_mask[:] = True

        clean_current = clean_sample["current_position_m"].astype(
            np.float32, copy=False
        )
        anchor_shift = clean_current - pnp_current
        step = clean_sample["candidate_step"].astype(np.int64, copy=False)
        pnp_candidate_relation = (
            clean_sample["candidate_relation_m"].astype(np.float32, copy=False)
            + anchor_shift[None, :]
        ).astype(np.float32, copy=False)
        same_current_role = np.remainder(step, 4) == 0
        pnp_candidate_relation[same_current_role] = 0.0
        if not np.array_equal(
            pnp_candidate_relation[same_current_role],
            np.zeros_like(pnp_candidate_relation[same_current_role]),
        ):
            raise RuntimeError("all current-role PnP candidate relations must be zero")
        future_absolute = (
            clean_current[None, :]
            + clean_sample["target_visible_delta_m"].astype(np.float32, copy=False)
        )
        pnp_target_delta = (future_absolute - pnp_current[None, :]).astype(
            np.float32, copy=False
        )

    failure_code = 0
    if not q0_associated:
        failure_code = 1
    elif not full_history:
        failure_code = 2
    return {
        "pnp_history_position_rel_m": pnp_history_position,
        "pnp_history_time_s": pnp_history_time,
        "pnp_history_dt_s": pnp_history_dt,
        "pnp_history_switch_step": pnp_history_switch,
        "pnp_history_mask": pnp_history_mask,
        "pnp_current_position_m": pnp_current,
        "pnp_candidate_relation_m": pnp_candidate_relation,
        "pnp_candidate_step": clean_sample["candidate_step"].astype(
            np.int64, copy=True
        ),
        "pnp_candidate_mask": clean_sample["candidate_mask"].astype(
            np.bool_, copy=True
        ),
        "pnp_candidate_confidence": clean_sample["candidate_confidence"].astype(
            np.float32, copy=True
        ),
        "pnp_tau_s": clean_sample["tau_s"].astype(np.float32, copy=True),
        "pnp_target_visible_delta_m": pnp_target_delta,
        "pnp_forward_usable": np.asarray(usable, dtype=np.bool_),
        "pnp_q0_associated": np.asarray(q0_associated, dtype=np.bool_),
        "pnp_full_history_associated": np.asarray(full_history, dtype=np.bool_),
        "pnp_failure_code": np.asarray(failure_code, dtype=np.int64),
        "pnp_history_associated_mask": associated,
        "pnp_history_ambiguous_mask": ambiguous,
        "pnp_history_candidate_count": candidate_count,
        "pnp_history_association_error_m": association_error,
        "pnp_q0_anchor_error_m": np.asarray(
            association_error[-1] if q0_associated else 0.0, dtype=np.float32
        ),
    }


def model_inputs_from_arrays(
    arrays: dict[str, Any], *, prefix: str = ""
) -> dict[str, Any]:
    """Select only the eleven public F forward tensors from a paired batch."""
    return {key: arrays[f"{prefix}{key}"] for key in FORWARD_KEYS}
