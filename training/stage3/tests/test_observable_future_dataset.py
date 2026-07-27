from __future__ import annotations

import numpy as np
import pytest

from training.stage3.observable_future_dataset import (
    construct_observable_future_sample,
)


def _history(q0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = np.broadcast_to(q0, (32, 4, 3)).copy().astype(np.float32)
    mask = np.ones(32, dtype=np.bool_)
    time = np.linspace(-0.31, 0.0, 32, dtype=np.float32)
    return position, mask, time


def _user_switch_fixture() -> tuple[np.ndarray, ...]:
    q0 = np.asarray([
        [0.50, 0.0, 0.0], [0.70, 0.0, 0.0],
        [1.20, 0.0, 0.0], [0.90, 0.0, 0.0],
    ], dtype=np.float32)
    history, mask, history_time = _history(q0)
    dense = np.stack((q0, q0.copy(), q0.copy()))
    dense[1:, 0, 0] = 0.80
    dense[1:, 1, 0] = 0.25
    dense_time = np.asarray([0.0, 0.05, 0.10], dtype=np.float64)
    tau = np.asarray([0.0, 0.10], dtype=np.float32)
    rule = np.ones(2, dtype=np.bool_)
    return history, mask, history_time, dense, dense_time, tau, rule


def _build(values: tuple[np.ndarray, ...]) -> dict[str, np.ndarray]:
    return construct_observable_future_sample(*values)


def test_future_target_follows_selected_plate_not_anchor_handle() -> None:
    sample = _build(_user_switch_fixture())
    assert sample["target_switch_count"].tolist() == [0, 1]
    assert sample["target_visible_delta_m"][1, 0] == pytest.approx(-0.25)
    assert sample["target_visible_delta_m"][1, 0] != pytest.approx(0.30)
    assert bool(sample["target_candidate_onehot"][1].any())


def test_hidden_anchor_future_cannot_change_observable_target() -> None:
    values = list(_user_switch_fixture())
    baseline = _build(tuple(values))
    modified_dense = values[3].copy()
    modified_dense[1:, 0, 0] = 8.0
    values[3] = modified_dense
    changed = _build(tuple(values))
    for key in baseline:
        assert np.array_equal(baseline[key], changed[key]), key


def test_cyclic_source_reindex_is_erased_before_model_tensors() -> None:
    values = list(_user_switch_fixture())
    baseline = _build(tuple(values))
    values[0] = np.roll(values[0], 2, axis=1)
    values[3] = np.roll(values[3], 2, axis=1)
    shifted = _build(tuple(values))
    for key in baseline:
        assert np.array_equal(baseline[key], shifted[key]), key
    forbidden = ("physical_id", "armor_id", "slot_id", "handle_id", "primary_index")
    assert not any(token in key for key in baseline for token in forbidden)


def test_multi_switch_count_is_unwrapped_not_modulo_four() -> None:
    q0 = np.asarray([
        [0.50, 0.0, 0.0], [0.70, 0.0, 0.0],
        [0.90, 0.0, 0.0], [1.10, 0.0, 0.0],
    ], dtype=np.float32)
    history, mask, history_time = _history(q0)
    dense = np.broadcast_to(q0, (5, 4, 3)).copy()
    for row, selected in enumerate((0, 1, 2, 3, 0)):
        if row == 0:
            continue
        dense[row, :, 0] = np.asarray([0.8, 0.9, 1.0, 1.1], dtype=np.float32)
        dense[row, selected, 0] = 0.25
    sample = construct_observable_future_sample(
        history, mask, history_time,
        dense, np.asarray([0.0, 0.1, 0.2, 0.3, 0.4]),
        np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        np.ones(5, dtype=np.bool_),
    )
    assert sample["target_switch_count"].tolist() == [0, 1, 2, 3, 4]
    assert sample["target_switch_count"][-1] != 0


def test_q0_inherits_history_selection_even_if_dense_argmin_would_flip() -> None:
    q0 = np.asarray([
        [0.500000, 0.0, 0.0], [0.500010, 0.0, 0.0],
        [0.90, 0.0, 0.0], [1.10, 0.0, 0.0],
    ], dtype=np.float32)
    history, mask, history_time = _history(q0)
    dense = np.stack((q0.copy(), q0.copy()))
    dense[0, 0, 0] = 0.500002
    dense[0, 1, 0] = 0.499999
    sample = construct_observable_future_sample(
        history, mask, history_time, dense, np.asarray([0.0, 0.1]),
        np.asarray([0.1, 0.0], dtype=np.float32), np.ones(2, dtype=np.bool_),
    )
    zero = 1
    assert sample["target_switch_count"][zero] == 0
    assert np.array_equal(sample["target_visible_delta_m"][zero], np.zeros(3, np.float32))


def test_query_permutation_only_reorders_targets() -> None:
    q0 = np.asarray([
        [0.50, 0.0, 0.0], [0.70, 0.0, 0.0],
        [0.90, 0.0, 0.0], [1.10, 0.0, 0.0],
    ], dtype=np.float32)
    history, mask, history_time = _history(q0)
    dense = np.broadcast_to(q0, (3, 4, 3)).copy()
    dense[1:, 0, 0] = 0.8
    dense[1:, 1, 0] = 0.3
    dense_time = np.asarray([0.0, 0.2, 0.4])
    original_tau = np.asarray([0.0, 0.2, 0.4], dtype=np.float32)
    original = construct_observable_future_sample(
        history, mask, history_time, dense, dense_time,
        original_tau, np.ones(3, dtype=np.bool_),
    )
    permutation = np.asarray([2, 0, 1])
    permuted = construct_observable_future_sample(
        history, mask, history_time, dense, dense_time,
        original_tau[permutation], np.ones(3, dtype=np.bool_),
    )
    for key in (
        "tau_s", "target_switch_count", "target_candidate_onehot",
        "target_visible_delta_m", "target_query_mask",
    ):
        assert np.array_equal(permuted[key], original[key][permutation]), key


def test_missing_dense_timeline_fails_closed() -> None:
    values = list(_user_switch_fixture())
    values[3] = values[3][[0, 2]]
    values[4] = values[4][[0, 2]]
    # Query endpoints alone are not a dense transition stream.
    with pytest.raises(ValueError, match="opposite|dense"):
        q0 = values[3][0]
        values[3] = np.stack((q0, np.roll(q0, 2, axis=0)))
        _build(tuple(values))
