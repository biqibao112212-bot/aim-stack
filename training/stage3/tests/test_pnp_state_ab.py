from __future__ import annotations

import hashlib
import json

import pytest
import torch

from training.stage3.pnp_state_loss import (
    pnp_state_constrained_loss,
    pnp_state_position_loss,
)
from training.stage3.pnp_state_model import (
    ExplicitStatePnPAdapter,
    ImplicitQueryPosePredictor,
    trainable_parameter_count,
)
from training.stage3.train_pnp_state_ab import (
    _load_session_selection,
    _to_device,
    _trajectory_consistency,
)
from training.stage3.pnp_state_targets import (
    decoded_trajectory_state,
    geometry_c4_asymmetry_m,
    truth_trajectory_targets,
)


GEOMETRY = torch.tensor([
    [0.01, -0.21, -0.061],
    [0.22, 0.00, -0.075],
    [0.01, 0.21, -0.061],
    [-0.20, 0.00, -0.075],
], dtype=torch.float32)


def _inputs(batch: int = 2, time: int = 8) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(7)
    obs = torch.randn(batch, time, 4, 7)
    obs_mask = torch.ones(batch, time, 4, dtype=torch.bool)
    event_mask = torch.ones(batch, time, dtype=torch.bool)
    event_time = torch.linspace(-0.14, 0.0, time).expand(batch, -1).clone()
    tau = torch.tensor([0.0, 0.1, 0.2, 0.5]).expand(batch, -1).clone()
    return obs, obs_mask, event_mask, event_time, tau


def _models() -> tuple[ExplicitStatePnPAdapter, ImplicitQueryPosePredictor]:
    torch.manual_seed(11)
    a = ExplicitStatePnPAdapter(GEOMETRY, channels=32, dropout=0.0)
    b = ImplicitQueryPosePredictor(GEOMETRY, channels=32, dropout=0.0)
    b.encoder.load_state_dict(a.encoder.state_dict())
    return a.eval(), b.eval()


def test_paired_parameter_counts_are_within_one_percent() -> None:
    a, b = _models()
    counts = trainable_parameter_count(a), trainable_parameter_count(b)
    assert abs(counts[0] - counts[1]) / max(counts) < 0.01


def test_explicit_state_has_a_zero_motion_neutral_initialization() -> None:
    model, _ = _models()
    output = model(*_inputs())
    assert torch.equal(output["velocity"], torch.zeros_like(output["velocity"]))
    assert torch.equal(output["omega"], torch.zeros_like(output["omega"]))


def test_both_models_always_decode_the_fixed_rigid_geometry() -> None:
    inputs = _inputs()
    pair_i, pair_j = torch.triu_indices(4, 4, offset=1)
    expected = torch.linalg.vector_norm(GEOMETRY[pair_i] - GEOMETRY[pair_j], dim=-1)
    for model in _models():
        output = model(*inputs)["position_mean"]
        actual = torch.linalg.vector_norm(
            output[:, :, pair_i] - output[:, :, pair_j], dim=-1
        )
        assert torch.allclose(actual, expected.view(1, 1, -1), atol=2e-6)


def test_bfloat16_encoding_cannot_change_the_decoded_rigid_geometry() -> None:
    inputs = _inputs()
    pair_i, pair_j = torch.triu_indices(4, 4, offset=1)
    expected = torch.linalg.vector_norm(GEOMETRY[pair_i] - GEOMETRY[pair_j], dim=-1)
    for model in _models():
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            position = model(*inputs)["position_mean"]
        actual = torch.linalg.vector_norm(
            position[:, :, pair_i] - position[:, :, pair_j], dim=-1
        )
        assert position.dtype == torch.float32
        assert torch.allclose(actual, expected.view(1, 1, -1), atol=2e-6)


def test_explicit_state_is_propagated_by_the_frozen_equation() -> None:
    model = ExplicitStatePnPAdapter(
        GEOMETRY, channels=32, dropout=0.0,
        center_reference=torch.zeros(3), center_scale=torch.ones(3),
    ).eval()
    for parameter in model.parameters():
        parameter.data.zero_()
    center0 = torch.tensor([1.0, 2.0, 3.0])
    velocity = torch.tensor([0.7, -0.35, 0.14])
    yaw0 = torch.tensor(0.3)
    omega = torch.tensor(0.6)
    bias = model.state_head[-1].bias
    bias.data[:3] = center0
    bias.data[3:6] = torch.atanh(velocity / model.maximum_speed_mps)
    bias.data[6] = torch.atanh(yaw0 / torch.pi)
    bias.data[7] = torch.atanh(omega / model.maximum_yaw_rate_rad_s)
    inputs = _inputs(batch=1)
    output = model(*inputs)
    tau = inputs[-1]
    assert torch.allclose(output["query_center"], center0 + tau[..., None] * velocity, atol=1e-6)
    expected_angle = yaw0 + tau * omega
    expected_phase = torch.stack((torch.cos(expected_angle), torch.sin(expected_angle)), dim=-1)
    assert torch.allclose(output["query_phase"], expected_phase, atol=1e-6)
    center_rms, yaw_rms = _trajectory_consistency(
        output["query_center"], output["query_phase"], tau
    )
    assert center_rms.max() < 1e-6
    assert yaw_rms.max() < 1e-6


def test_implicit_query_pose_has_no_velocity_or_omega_and_is_tau_equivariant() -> None:
    _, model = _models()
    obs, obs_mask, event_mask, event_time, tau = _inputs(batch=1)
    reference = model(obs, obs_mask, event_mask, event_time, tau)
    assert "velocity" not in reference and "omega" not in reference
    order = torch.tensor([3, 1, 3, 0, 2])
    reordered = model(obs, obs_mask, event_mask, event_time, tau[:, order])
    assert torch.allclose(
        reordered["position_mean"], reference["position_mean"][:, order], atol=1e-6
    )


def test_masked_nonfinite_values_cannot_affect_either_model() -> None:
    obs, obs_mask, event_mask, event_time, tau = _inputs()
    obs_mask[:, 2, 1] = False
    clean = obs.clone()
    clean[:, 2, 1] = 0.0
    poisoned = clean.clone()
    poisoned[0, 2, 1] = torch.tensor([
        float("nan"), float("inf"), -float("inf"), 9e20, -9e20, 1e30, -1e30
    ])
    for model in _models():
        expected = model(clean, obs_mask, event_mask, event_time, tau)["position_mean"]
        actual = model(poisoned, obs_mask, event_mask, event_time, tau)["position_mean"]
        assert torch.equal(expected, actual)
        assert torch.isfinite(actual).all()


def test_unordered_set_loss_is_invariant_to_label_row_order_without_search() -> None:
    target = GEOMETRY.view(1, 1, 4, 3).expand(1, 4, -1, -1).clone()
    tau = torch.tensor([[0.0, 0.1, 0.2, 0.5]])
    exact, _ = pnp_state_position_loss(target, target, tau)
    swapped, _ = pnp_state_position_loss(target, target.roll(1, dims=2), tau)
    assert exact == 0
    assert swapped == 0


def test_masked_nonfinite_event_time_cannot_pollute_encoder() -> None:
    obs, obs_mask, event_mask, event_time, tau = _inputs()
    event_mask[:, 1] = False
    obs_mask[:, 1] = False
    clean_time = event_time.clone()
    clean_time[:, 1] = 0.0
    poisoned_time = clean_time.clone()
    poisoned_time[0, 1] = float("nan")
    poisoned_time[1, 1] = float("inf")
    for model in _models():
        expected = model(obs, obs_mask, event_mask, clean_time, tau)["position_mean"]
        actual = model(obs, obs_mask, event_mask, poisoned_time, tau)["position_mean"]
        assert torch.equal(expected, actual)
        assert torch.isfinite(actual).all()


def test_high_valid_yaw_rate_does_not_alias_consistency_diagnostic() -> None:
    tau = torch.tensor([[0.0, 0.1, 0.2, 0.5]])
    omega = 14.0
    center = torch.zeros(1, 4, 3)
    angle = tau * omega
    phase = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
    center_rms, yaw_rms = _trajectory_consistency(center, phase, tau)
    assert center_rms.max() < 1e-7
    assert yaw_rms.max() < 1e-6


def test_position_loss_backpropagates_into_explicit_state_head() -> None:
    model, _ = _models()
    inputs = _inputs()
    prediction = model(*inputs)["position_mean"]
    target = prediction.detach() + 0.1
    loss, _ = pnp_state_position_loss(prediction, target, inputs[-1])
    loss.backward()
    gradient = model.state_head[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_truth_state_targets_recover_exact_high_rate_constant_twist() -> None:
    tau = torch.tensor([[0.0, 0.1, 0.2, 0.5], [0.0, 0.12, 0.2, 0.5]])
    center0 = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]])
    velocity = torch.tensor([[0.7, -0.2, 0.1], [-1.1, 0.3, 0.0]])
    yaw0 = torch.tensor([0.3, -2.4])
    omega = torch.tensor([14.0, -14.0])
    center = center0[:, None] + tau[..., None] * velocity[:, None]
    yaw = yaw0[:, None] + tau * omega[:, None]
    phase = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
    position = _models()[0].decoder(center, phase)
    truth = truth_trajectory_targets(position, tau, GEOMETRY)
    assert truth["constant_motion"].all()
    assert torch.allclose(truth["center0"], center0, atol=2e-6)
    assert torch.allclose(truth["velocity"], velocity, atol=2e-6)
    assert torch.allclose(truth["omega"], omega, atol=2e-5)
    decoded = decoded_trajectory_state(
        {"position_mean": position}, tau, GEOMETRY
    )
    assert torch.allclose(decoded["velocity"], velocity, atol=1e-6)
    assert torch.allclose(decoded["omega"], omega, atol=2e-5)


def test_constrained_loss_is_common_and_zero_on_exact_decoded_trajectory() -> None:
    tau = torch.tensor([[0.0, 0.1, 0.2, 0.5]])
    center = torch.tensor([[[1.0, 2.0, 3.0], [1.1, 2.0, 3.0],
                            [1.2, 2.0, 3.0], [1.5, 2.0, 3.0]]])
    yaw = 0.4 + 1.2 * tau
    phase = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
    position = _models()[0].decoder(center, phase)
    output = {
        "position_mean": position,
        "query_center": center,
        "query_phase": phase,
    }
    loss, parts = pnp_state_constrained_loss(output, position, tau, GEOMETRY)
    assert loss < 1e-10
    assert parts["constant_motion_fraction"] == 1


def test_truth_state_targets_reject_a_future_motion_discontinuity() -> None:
    tau = torch.tensor([[0.0, 0.1, 0.2, 0.5]])
    center = torch.zeros(1, 4, 3)
    center[:, 3, 0] = 0.2
    phase = torch.tensor([1.0, 0.0]).view(1, 1, 2).expand(1, 4, -1)
    position = _models()[0].decoder(center, phase)
    truth = truth_trajectory_targets(position, tau, GEOMETRY)
    assert not truth["constant_motion"].item()


def test_full_relative_yaw_requires_nonsymmetric_geometry() -> None:
    symmetric = torch.tensor([
        [0.2, 0.0, -0.07], [0.0, 0.2, -0.07],
        [-0.2, 0.0, -0.07], [0.0, -0.2, -0.07],
    ])
    assert geometry_c4_asymmetry_m(GEOMETRY) > 0.005
    assert geometry_c4_asymmetry_m(symmetric) < 1e-6
    position = symmetric.view(1, 1, 4, 3).expand(1, 4, -1, -1)
    tau = torch.tensor([[0.0, 0.1, 0.2, 0.5]])
    with pytest.raises(ValueError, match="C4 asymmetry"):
        truth_trajectory_targets(position, tau, symmetric)


def test_decoded_state_is_reparsed_from_position_not_internal_pose() -> None:
    tau = torch.tensor([[0.0, 0.1, 0.2, 0.5]])
    center = torch.zeros(1, 4, 3)
    phase = torch.tensor([1.0, 0.0]).view(1, 1, 2).expand(1, 4, -1)
    position = _models()[0].decoder(center, phase)
    output = {
        "position_mean": position,
        "query_center": torch.full_like(center, 123.0),
        "query_phase": torch.flip(phase, dims=(-1,)),
    }
    decoded = decoded_trajectory_state(output, tau, GEOMETRY)
    assert torch.allclose(decoded["query_center"], center, atol=1e-6)
    assert torch.allclose(decoded["query_phase"], phase, atol=1e-6)


def test_constrained_loss_handles_a_zero_eligible_batch_without_crashing() -> None:
    tau = torch.tensor([[0.0, 0.1, 0.2, 0.5]])
    center = torch.zeros(1, 4, 3)
    center[:, 3, 0] = 0.2
    phase = torch.tensor([1.0, 0.0]).view(1, 1, 2).expand(1, 4, -1)
    position = _models()[0].decoder(center, phase)
    predicted = position.clone().requires_grad_(True)
    loss, parts = pnp_state_constrained_loss(
        {"position_mean": predicted}, position, tau, GEOMETRY
    )
    loss.backward()
    assert loss == 0
    assert parts["constant_motion_fraction"] == 0
    assert torch.equal(predicted.grad, torch.zeros_like(predicted))


def test_q1_alias_guard_uses_q1_delta_and_requires_exact_q0() -> None:
    center = torch.zeros(1, 4, 3)
    phase = torch.tensor([1.0, 0.0]).view(1, 1, 2).expand(1, 4, -1)
    position = _models()[0].decoder(center, phase)
    with pytest.raises(ValueError, match="tau=0"):
        truth_trajectory_targets(
            position, torch.tensor([[0.01, 0.11, 0.21, 0.51]]), GEOMETRY
        )
    with pytest.raises(ValueError, match="alias"):
        truth_trajectory_targets(
            position, torch.tensor([[0.0, 0.21, 0.3, 0.5]]), GEOMETRY
        )


def test_training_device_allowlist_excludes_future_observation_and_state_labels() -> None:
    obs, obs_mask, event_mask, event_time, tau = _inputs(batch=1)
    raw = {
        "obs": obs,
        "obs_mask": obs_mask,
        "event_mask": event_mask,
        "event_time_s": event_time,
        "tau": tau,
        "future_position": torch.zeros(1, 4, 4, 3),
        "motion_class": torch.zeros(1, dtype=torch.long),
        "future_observation_position": torch.full((1, 4, 4, 3), 123.0),
        "exact_velocity": torch.full((1, 3), 456.0),
    }
    allowed = _to_device(raw, torch.device("cpu"))
    assert set(allowed) == {
        "obs", "obs_mask", "event_mask", "event_time_s", "tau",
        "future_position", "motion_class",
    }


def test_pilot_session_selection_is_manifest_bound_and_keeps_test_empty(tmp_path) -> None:
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text('{"schema_version":"stage3-dataset-v4-observation"}', encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "schema_version": "stage3-pnp-state-pilot-selection-v1",
        "dataset_manifest_sha256": digest,
        "purpose": "dynamic_pilot",
        "validation_source_split": "validation",
        "train": ["train-a", "train-b"],
        "validation": ["validation-a"],
        "test": [],
    }), encoding="utf-8")
    train, validation, record = _load_session_selection(str(selection), manifest)
    assert train == ["train-a", "train-b"]
    assert validation == ["validation-a"]
    assert record is not None and record["test"] == []


def test_pilot_session_selection_rejects_overlap_or_any_test_entry(tmp_path) -> None:
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    base = {
        "schema_version": "stage3-pnp-state-pilot-selection-v1",
        "dataset_manifest_sha256": digest,
        "purpose": "dynamic_pilot",
        "validation_source_split": "validation",
        "train": ["shared"],
        "validation": ["shared"],
        "test": [],
    }
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="disjoint"):
        _load_session_selection(str(selection), manifest)
    base["validation"] = ["validation-a"]
    base["test"] = ["forbidden"]
    selection.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="test empty"):
        _load_session_selection(str(selection), manifest)


def test_train_sourced_selection_requires_the_exact_same_sessions(tmp_path) -> None:
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    selection = tmp_path / "selection.json"
    payload = {
        "schema_version": "stage3-pnp-state-pilot-selection-v1",
        "dataset_manifest_sha256": digest,
        "purpose": "dynamic_overfit_combined",
        "validation_source_split": "train",
        "train": ["combined-session"],
        "validation": ["combined-session"],
        "test": [],
    }
    selection.write_text(json.dumps(payload), encoding="utf-8")
    train, validation, record = _load_session_selection(str(selection), manifest)
    assert train == validation == ["combined-session"]
    assert record is not None and record["validation_source_split"] == "train"
    payload["validation"] = ["different-session"]
    selection.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="reuse train sessions"):
        _load_session_selection(str(selection), manifest)
