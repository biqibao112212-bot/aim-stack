from __future__ import annotations

import copy

import numpy as np

from training.stage3.build_observable_future_pnp_upper_bound_dataset import (
    _assert_clean_replay,
)
from training.stage3.observable_future_dataset import DEFAULT_CANDIDATE_STEPS
from training.stage3.observable_future_pnp_upper_bound import (
    FORWARD_KEYS,
    construct_observed_primary_pnp_sample,
    construct_real_pnp_upper_bound_sample,
    model_inputs_from_arrays,
    oracle_injective_assignment,
    rebase_tracker_points_to_anchor,
)


def _observed_stream_sample(
    clean: dict[str, np.ndarray], inputs: dict[str, np.ndarray],
    *, minimum_history_events: int = 8,
) -> dict[str, np.ndarray]:
    physical = inputs["physical_history_position_m"]
    dense = np.stack((physical[-1], physical[-1]), axis=0)
    return construct_observed_primary_pnp_sample(
        clean,
        physical,
        inputs["physical_event_mask"],
        inputs["physical_event_time_s"],
        dense,
        np.asarray([0.0, 0.1], dtype=np.float64),
        clean["tau_s"],
        np.ones(clean["tau_s"].shape, dtype=np.bool_),
        inputs["truth_history_position_m"],
        inputs["truth_history_mask"],
        inputs["observation_position_m"],
        inputs["observation_mask"],
        inputs["event_origins_world_m"],
        inputs["event_tracker_to_world_rotation"],
        inputs["anchor_origin_world_m"],
        inputs["anchor_tracker_to_world_rotation"],
        minimum_history_events=minimum_history_events,
    )


def _fixture() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    time = np.linspace(-0.31, 0.0, 32, dtype=np.float32)
    dt = np.zeros(32, dtype=np.float32)
    dt[1:] = np.diff(time)
    q0 = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.1],
        [-3.0, 0.0, 0.2],
        [0.0, -4.0, 0.3],
    ], dtype=np.float32)
    physical = np.broadcast_to(q0, (32, 4, 3)).copy()
    steps = np.asarray(DEFAULT_CANDIDATE_STEPS, dtype=np.int64)
    candidate_relation = np.stack(
        [q0[int(step) % 4] - q0[0] for step in steps]
    ).astype(np.float32)
    target_delta = np.asarray([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
    ], dtype=np.float32)
    clean = {
        "history_position_rel_m": np.zeros((32, 3), dtype=np.float32),
        "history_time_s": time,
        "history_dt_s": dt,
        "history_switch_step": np.zeros(32, dtype=np.int64),
        "history_mask": np.ones(32, dtype=np.bool_),
        "current_position_m": q0[0].copy(),
        "candidate_relation_m": candidate_relation,
        "candidate_step": steps,
        "candidate_mask": np.ones(len(steps), dtype=np.bool_),
        "candidate_confidence": np.ones(len(steps), dtype=np.float32),
        "tau_s": np.asarray([0.0, 0.1], dtype=np.float32),
        "target_switch_count": np.asarray([0, 0], dtype=np.int64),
        "target_candidate_onehot": steps[None, :] == np.asarray([[0], [0]]),
        "target_visible_delta_m": target_delta,
        "target_query_mask": np.ones(2, dtype=np.bool_),
    }
    observation = physical.copy()
    observation[:, 0, 0] += np.linspace(0.002, 0.010, 32, dtype=np.float32)
    inputs = {
        "physical_history_position_m": physical,
        "physical_event_mask": np.ones(32, dtype=np.bool_),
        "physical_event_time_s": time,
        "truth_history_position_m": physical,
        "truth_history_mask": np.ones((32, 4), dtype=np.bool_),
        "observation_position_m": observation,
        "observation_mask": np.ones((32, 4), dtype=np.bool_),
        "event_origins_world_m": np.zeros((32, 3), dtype=np.float64),
        "event_tracker_to_world_rotation": np.broadcast_to(
            np.eye(3, dtype=np.float64), (32, 3, 3)
        ).copy(),
        "anchor_origin_world_m": np.zeros(3, dtype=np.float64),
        "anchor_tracker_to_world_rotation": np.eye(3, dtype=np.float64),
    }
    return clean, inputs


def test_rebase_tracker_points_uses_event_and_anchor_poses() -> None:
    angle = np.pi / 2.0
    event_rotation = np.asarray([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    result = rebase_tracker_points_to_anchor(
        np.asarray([[1.0, 0.0, 0.0]]),
        np.asarray([10.0, 0.0, 0.0]),
        event_rotation,
        np.asarray([9.0, 1.0, 0.0]),
        np.eye(3),
    )
    assert np.allclose(result, [[1.0, 0.0, 0.0]], atol=1e-7)


def test_oracle_assignment_is_detection_permutation_invariant() -> None:
    truth = np.asarray([
        [1.0, 0.0, 0.0], [0.0, 2.0, 0.0],
        [-3.0, 0.0, 0.0], [0.0, -4.0, 0.0],
    ])
    observation = truth[[2, 0, 3]] + np.asarray([
        [0.01, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.03],
    ])
    first = oracle_injective_assignment(
        observation, truth, ambiguity_epsilon_m=1e-9
    )
    order = np.asarray([2, 0, 1])
    second = oracle_injective_assignment(
        observation[order], truth, ambiguity_epsilon_m=1e-9
    )
    assert first is not None and second is not None
    assert not first[2] and not second[2]
    first_positions = {slot: observation[row] for slot, row in first[1].items()}
    second_positions = {
        slot: observation[order][row] for slot, row in second[1].items()
    }
    assert first_positions.keys() == second_positions.keys()
    for slot in first_positions:
        assert np.array_equal(first_positions[slot], second_positions[slot])


def test_oracle_assignment_marks_equal_cost_ambiguity() -> None:
    truth = np.asarray([
        [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 3.0, 0.0], [0.0, -3.0, 0.0],
    ])
    result = oracle_injective_assignment(
        np.asarray([[0.0, 0.0, 0.0]]), truth,
        ambiguity_epsilon_m=1e-9,
    )
    assert result is not None
    assert result[2]


def test_construct_upper_bound_reanchors_real_pnp_and_preserves_truth() -> None:
    clean, inputs = _fixture()
    result = construct_real_pnp_upper_bound_sample(clean, **inputs)
    assert bool(result["pnp_forward_usable"])
    assert bool(result["pnp_full_history_associated"])
    assert np.array_equal(
        result["pnp_history_position_rel_m"][-1], np.zeros(3, dtype=np.float32)
    )
    assert np.allclose(result["pnp_current_position_m"], [1.01, 0.0, 0.0])
    current_roles = np.remainder(result["pnp_candidate_step"], 4) == 0
    assert np.array_equal(
        result["pnp_candidate_relation_m"][current_roles],
        np.zeros((int(current_roles.sum()), 3), dtype=np.float32),
    )
    clean_absolute = (
        clean["current_position_m"][None, :] + clean["target_visible_delta_m"]
    )
    pnp_absolute = (
        result["pnp_current_position_m"][None, :]
        + result["pnp_target_visible_delta_m"]
    )
    assert np.array_equal(clean_absolute, pnp_absolute)


def test_missing_selected_pnp_event_is_ineligible_without_truth_fill() -> None:
    clean, inputs = _fixture()
    inputs["observation_mask"][7, 0] = False
    result = construct_real_pnp_upper_bound_sample(clean, **inputs)
    assert not bool(result["pnp_forward_usable"])
    assert bool(result["pnp_q0_associated"])
    assert not bool(result["pnp_full_history_associated"])
    assert int(result["pnp_failure_code"]) == 2
    assert not bool(result["pnp_history_associated_mask"][7])
    assert not bool(result["pnp_history_mask"].any())


def test_missing_pnp_q0_is_explicitly_ineligible() -> None:
    clean, inputs = _fixture()
    inputs["observation_mask"][-1, 0] = False
    result = construct_real_pnp_upper_bound_sample(clean, **inputs)
    assert not bool(result["pnp_forward_usable"])
    assert not bool(result["pnp_q0_associated"])
    assert int(result["pnp_failure_code"]) == 1
    assert np.array_equal(
        result["pnp_current_position_m"], np.zeros(3, dtype=np.float32)
    )


def test_observed_stream_keeps_an_adjacent_plate_when_old_primary_is_missing() -> None:
    clean, inputs = _fixture()
    inputs["observation_mask"][:] = False
    inputs["observation_mask"][:, 1] = True
    result = _observed_stream_sample(clean, inputs)
    assert bool(result["pnp_forward_usable"])
    assert int(result["pnp_history_active_count"]) == 32
    assert np.allclose(result["current_position_m"], [0.0, 2.0, 0.1])
    assert np.allclose(result["pnp_current_position_m"], [0.0, 2.0, 0.1])
    assert np.array_equal(result["history_switch_step"], np.zeros(32))
    assert np.array_equal(result["target_switch_count"], [0, -1])


def test_observed_stream_masks_incoherent_prefix_instead_of_encoding_opposite() -> None:
    clean, inputs = _fixture()
    inputs["observation_mask"][:] = False
    inputs["observation_mask"][:20, 2] = True
    inputs["observation_mask"][20:, 0] = True
    result = _observed_stream_sample(clean, inputs)
    assert bool(result["pnp_forward_usable"])
    assert int(result["pnp_history_active_count"]) == 12
    assert not bool(result["history_mask"][:20].any())
    assert bool(result["history_mask"][20:].all())
    assert set(result["history_switch_step"].tolist()) == {0}
    assert not any(
        token in key for key in result
        for token in ("physical_id", "armor_id", "source_slot", "handle_id")
    )


def test_observed_stream_never_reverses_switch_direction_inside_suffix() -> None:
    clean, inputs = _fixture()
    inputs["observation_mask"][:] = False
    inputs["observation_mask"][:, 0] = True
    inputs["observation_mask"][1] = False
    inputs["observation_mask"][1, 1] = True
    result = _observed_stream_sample(clean, inputs)
    assert bool(result["pnp_forward_usable"])
    active_switch = result["history_switch_step"][result["history_mask"]]
    nonzero = active_switch[active_switch != 0]
    assert nonzero.size <= 1 or bool(np.all(nonzero == nonzero[0]))
    assert not np.any(active_switch[:-1] * active_switch[1:] == -1)


def test_observed_stream_is_invariant_to_observation_row_permutation() -> None:
    clean, inputs = _fixture()
    first = _observed_stream_sample(clean, inputs)
    order = np.asarray([2, 0, 3, 1])
    permuted = copy.deepcopy(inputs)
    permuted["observation_position_m"] = inputs["observation_position_m"][:, order]
    permuted["observation_mask"] = inputs["observation_mask"][:, order]
    second = _observed_stream_sample(clean, permuted)
    for key in first:
        assert np.array_equal(first[key], second[key]), key


def test_model_input_filter_excludes_targets_and_pair_metadata() -> None:
    clean, _ = _fixture()
    batch = {key: np.expand_dims(value, 0) for key, value in clean.items()}
    batch["pair_id"] = np.asarray(["not-a-model-input"])
    batch["session_id"] = np.asarray(["session"])
    selected = model_inputs_from_arrays(batch)
    assert tuple(selected) == FORWARD_KEYS
    assert not any(
        token in key for key in selected
        for token in ("target", "future", "session", "pair", "slot", "armor")
    )


def test_clean_replay_check_is_bit_exact() -> None:
    clean, _ = _fixture()
    stacked = {key: np.expand_dims(value, 0) for key, value in clean.items()}
    _assert_clean_replay(clean, stacked, 0)
    changed = copy.deepcopy(stacked)
    changed["current_position_m"][0, 0] += np.float32(1e-7)
    try:
        _assert_clean_replay(clean, changed, 0)
    except ValueError as error:
        assert "bit-exact" in str(error)
    else:
        raise AssertionError("clean replay check accepted a changed tensor")
