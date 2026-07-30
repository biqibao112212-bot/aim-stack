from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from training.stage3.diagnose_global_flow_closure_mechanism import (
    _fixed_state_closure,
    _physical_donor_index,
    _refinement_forward,
    _compatible_roll,
    _truth_common_rotation_hybrid,
)
from training.stage3.global_flow_closure_future import (
    AnonymousGlobalFlowClosureProbe,
)
from training.stage3.tests.test_anonymous_vehicle_motion import _batch


def _model() -> AnonymousGlobalFlowClosureProbe:
    torch.manual_seed(20260731)
    return AnonymousGlobalFlowClosureProbe(
        velocity_scale_mps=(3.0, 3.2, 0.25),
        yaw_rate_scale_rad_s=17.25,
        channels=32, dropout=0.0, message_layers=2, basis_count=6,
    ).eval()


def _fields(batch):
    return {
        name: batch[name] for name in (
            "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
            "history_event_mask", "history_time_s", "history_switch_step",
        )
    }


def test_compatible_roll_stays_within_stream_and_scale() -> None:
    value = torch.arange(2 * 2 * 3 * 3 * 2, dtype=torch.float32).reshape(
        2, 2 * 3 * 3, 2,
    )
    valid = torch.ones(2, 2 * 3 * 3, dtype=torch.bool)
    valid[1] = False
    valid[1, 0] = True
    changed = _compatible_roll(value, valid, streams=2, event_count=3)
    torch.testing.assert_close(changed[~valid], value[~valid])
    grouped = changed[0].reshape(2, 3, 3, 2)
    original = value[0].reshape(2, 3, 3, 2)
    for stream in range(2):
        for scale in range(3):
            assert {
                tuple(row.tolist()) for row in grouped[stream, :, scale]
            } == {
                tuple(row.tolist()) for row in original[stream, :, scale]
            }
    torch.testing.assert_close(changed[1], value[1])


@torch.inference_mode()
def test_mechanism_replay_normal_matches_native_v11_head() -> None:
    model = _model()
    fields = _fields(_batch())
    history = model.context(**fields)
    native = model.estimate_motion_state(**fields)["state"]
    replay = _refinement_forward(model, history, mode="normal")
    torch.testing.assert_close(
        replay["motion_state_normalized"],
        native["motion_state_normalized"], rtol=2e-5, atol=4e-6,
    )
    blind = _refinement_forward(model, history, mode="prediction_blind")
    shuffled = _refinement_forward(model, history, mode="prediction_shuffled")
    assert torch.count_nonzero(
        blind["motion_state_normalized"] - replay["motion_state_normalized"]
    ) > 0
    assert torch.count_nonzero(
        shuffled["motion_state_normalized"] - replay["motion_state_normalized"]
    ) > 0
    assert bool(shuffled["shuffled_handle_touched"].any())


@torch.inference_mode()
def test_fixed_state_closure_changes_without_reestimating_state() -> None:
    model = _model()
    fields = _fields(_batch())
    normal = model.context(**fields)
    state = _refinement_forward(model, normal, mode="normal")[
        "motion_state_normalized"
    ]
    normal_handle, normal_pair = _fixed_state_closure(model, normal, state)
    broken = dict(normal)
    broken_geometry = normal["_handle_geometry_raw"].clone()
    index = torch.nonzero(normal["_handle_raw_valid"][0], as_tuple=False).flatten()
    assert index.numel() > 1
    broken_geometry[0, index, 3:6] = broken_geometry[
        0, torch.roll(index, 1), 3:6
    ]
    broken["_handle_geometry_raw"] = broken_geometry
    broken_handle, broken_pair = _fixed_state_closure(model, broken, state)
    assert torch.count_nonzero(broken_handle - normal_handle) > 0
    torch.testing.assert_close(broken_pair, normal_pair)


@torch.inference_mode()
def test_truth_common_hybrid_preserves_both_displacement_identities() -> None:
    model = _model()
    fields = _fields(_batch())
    history = model.context(**fields)
    source = torch.tensor([1, 0])
    velocity = torch.tensor([[0.7, -0.2, 0.0], [-0.4, 0.5, 0.0]])
    event_count = fields["history_event_mask"].shape[1]
    hybrid = _truth_common_rotation_hybrid(
        model, history, source, velocity, event_count=event_count,
    )
    scales = 3
    valid = history["_handle_raw_valid"].reshape(
        2, 4, event_count, scales,
    ).index_select(0, source)
    geometry = hybrid["_handle_geometry_raw"].reshape(
        2, 4, event_count, scales, 12,
    )
    kinematics = hybrid["_handle_kinematics_raw"].reshape(
        2, 4, event_count, scales, 14,
    )
    elapsed = 0.01 * torch.expm1(kinematics[..., 6])
    elapsed_normalized = elapsed / model.context.history_scale_s
    donor_geometry = history["_handle_geometry_raw"].reshape(
        2, 4, event_count, scales, 12,
    ).index_select(0, source)
    common_rate = velocity * model.context.history_scale_s
    donor_centered_delta = donor_geometry[..., :3] - donor_geometry[..., 3:6]
    expected_common_delta = (
        common_rate[:, None, None, None]
        * elapsed_normalized.unsqueeze(-1)
    ).expand_as(donor_centered_delta)
    torch.testing.assert_close(
        (kinematics[..., :3] - donor_centered_delta)[valid],
        expected_common_delta[valid], rtol=1e-5, atol=1e-6,
    )
    torch.testing.assert_close(
        (geometry[..., 6:9] - geometry[..., 9:12])[valid],
        kinematics[..., :3][valid], rtol=1e-5, atol=1e-6,
    )
    torch.testing.assert_close(
        kinematics[..., :3][valid],
        (geometry[..., 6:9] - geometry[..., 9:12])[valid],
        rtol=1e-5, atol=1e-6,
    )
    torch.testing.assert_close(
        kinematics[..., :3][valid],
        (kinematics[..., 3:6] * (
            elapsed / model.context.history_scale_s
        ).unsqueeze(-1))[valid],
        rtol=1e-5, atol=1e-6,
    )
    torch.testing.assert_close(
        hybrid["_pair_geometry_raw"],
        history["_pair_geometry_raw"].index_select(0, source),
    )
    torch.testing.assert_close(
        hybrid["_pair_kinematics_raw"],
        history["_pair_kinematics_raw"].index_select(0, source),
    )
    assert torch.equal(
        hybrid["_pair_raw_valid"],
        history["_pair_raw_valid"].index_select(0, source),
    )


def test_physical_donor_selector_separates_sign_and_magnitude() -> None:
    model = _model()
    history = model.context(**_fields(_batch()))
    history = deepcopy(history)
    for name in ("_handle_raw_valid", "_pair_raw_valid"):
        history[name] = history[name][:1].repeat(4, 1)
    history["history_active_count"] = history["history_active_count"][:1].repeat(4)
    history["pair_flow_available"] = history["pair_flow_available"][:1].repeat(4, 1)
    target = torch.tensor([
        [0.0, 0.0, 0.0, 4.0], [0.0, 0.0, 0.0, -4.2],
        [0.0, 0.0, 0.0, 9.0], [0.0, 0.0, 0.0, 2.0],
    ])
    combined = torch.ones(4, dtype=torch.bool)
    opposite, selected = _physical_donor_index(
        history, target, combined,
        relation="opposite_sign_similar_magnitude",
    )
    assert bool(selected[:2].all())
    assert torch.equal(opposite[:2], torch.tensor([1, 0]))
    same, selected = _physical_donor_index(
        history, target, combined,
        relation="same_sign_different_magnitude",
    )
    assert bool(selected[[0, 2, 3]].all())
    assert float(target[same[0], 3]) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@torch.inference_mode()
def test_fixed_state_closure_accepts_formal_cuda_autocast_state() -> None:
    device = torch.device("cuda:0")
    model = _model().to(device)
    fields = {name: value.to(device) for name, value in _fields(_batch()).items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        history = model.context(**fields)
        state = _refinement_forward(model, history, mode="normal")[
            "motion_state_normalized"
        ]
    assert state.dtype == torch.bfloat16
    with torch.autocast("cuda", dtype=torch.bfloat16):
        handle, pair = _fixed_state_closure(model, history, state)
    assert torch.isfinite(handle).all()
    assert torch.isfinite(pair).all()
