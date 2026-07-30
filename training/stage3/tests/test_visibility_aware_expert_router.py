from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from training.stage3.anonymous_vehicle_motion_v2 import (
    VisibilityAwareAnonymousVehicleFutureModel,
)
from training.stage3.observable_future_pnp_ab import sha256_file, state_dict_sha256
from training.stage3.tests.test_anonymous_vehicle_motion import _batch
from training.stage3.train_anonymous_vehicle_motion_v2 import RUN_SCHEMA as PARENT_SCHEMA
from training.stage3.train_visibility_aware_expert_router import (
    _frozen_motion_state,
    _hard_expert_role_position,
    _load_parent,
    _validate_resume_metadata,
    best_expert_targets,
    expert_counterfactual_role_position,
)


def _model() -> VisibilityAwareAnonymousVehicleFutureModel:
    torch.manual_seed(20260730)
    return VisibilityAwareAnonymousVehicleFutureModel(
        channels=32, dropout=0.0, message_layers=2,
        trained_horizon_s=0.55, maximum_absolute_step=6,
        basis_count=6, latent_experts=3,
    ).eval()


def _supervised_batch() -> dict[str, torch.Tensor]:
    batch = _batch()
    batch.update({
        "target_query_mask": torch.ones(2, 4, dtype=torch.bool),
        "target_switch_count": torch.tensor([[0, 1, 2, 3], [0, -1, -2, -3]]),
        "truth_current_position_m": batch["current_position_m"].clone(),
        "target_visible_delta_m": torch.randn(2, 4, 3) * 0.2,
        "motion_class": torch.tensor([2, 3]),
    })
    return batch


@torch.inference_mode()
def test_counterfactual_experts_match_manual_and_forced_one_hot_parent() -> None:
    model = _model()
    batch = _supervised_batch()
    prediction = model(batch)
    expert = expert_counterfactual_role_position(model, prediction, batch)

    coefficient = prediction["trajectory_coefficient"].float()
    basis = prediction["time_basis"].float()
    dynamic = torch.einsum("bqr,bherc->bqhec", basis, coefficient)
    residual = torch.tanh(dynamic / model.basis_count**0.5) * model.residual_scale_m
    primary = prediction["primary_index"].to(torch.long)
    role = torch.arange(4)[None]
    handle = torch.remainder(primary[:, None] + role, 4)
    q0 = batch["q0_relation_m"].gather(
        1, handle.unsqueeze(-1).expand(-1, -1, 3),
    ).float()
    q0[:, 0] = 0.0
    expected = (
        batch["current_position_m"].float()[:, None, None, None]
        + q0[:, None, :, None]
        + (batch["tau_s"] / model.trained_horizon_s)[:, :, None, None, None]
        * residual
    )
    torch.testing.assert_close(expert, expected)

    for expert_index in range(model.latent_experts):
        forced = deepcopy(model)
        for parameter in forced.motion_regime_gate.parameters():
            parameter.zero_()
        forced.motion_regime_gate[-1].bias.fill_(-100.0)
        forced.motion_regime_gate[-1].bias[expert_index] = 100.0
        forced_prediction = forced(batch)
        torch.testing.assert_close(
            forced_prediction["role_position_m"], expert[..., expert_index, :],
            rtol=1e-5, atol=2e-6,
        )


@torch.inference_mode()
def test_counterfactual_tau_zero_and_one_window_one_hard_expert() -> None:
    model = _model()
    batch = _supervised_batch()
    prediction = model(batch)
    expert = expert_counterfactual_role_position(model, prediction, batch)
    assert torch.equal(
        expert[:, 0],
        prediction["role_position_m"][:, 0, :, None].expand_as(expert[:, 0]),
    )
    choice = torch.tensor([0, 2])
    selected = _hard_expert_role_position(expert, choice)
    assert torch.equal(selected[0], expert[0, :, :, 0])
    assert torch.equal(selected[1], expert[1, :, :, 2])


def test_oracle_label_uses_true_role_positive_queries_and_window_mean() -> None:
    expert = torch.full((2, 3, 4, 3, 3), 50.0)
    batch = {
        "tau_s": torch.tensor([[0.0, 0.1, 0.2], [0.0, 0.1, 0.2]]),
        "target_query_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "target_switch_count": torch.tensor([[3, 1, 2], [3, 1, 2]]),
        "truth_current_position_m": torch.zeros(2, 3),
        "target_visible_delta_m": torch.zeros(2, 3, 3),
    }
    # Window zero uses true roles 1 and 2. Expert one wins by its mean (3 m),
    # despite expert zero winning one individual query. Tau zero is ignored.
    expert[0, 1, 1, 0] = torch.tensor([0.0, 0.0, 0.0])
    expert[0, 2, 2, 0] = torch.tensor([10.0, 0.0, 0.0])
    expert[0, 1, 1, 1] = torch.tensor([3.0, 0.0, 0.0])
    expert[0, 2, 2, 1] = torch.tensor([3.0, 0.0, 0.0])
    expert[0, 1, 1, 2] = torch.tensor([4.0, 0.0, 0.0])
    expert[0, 2, 2, 2] = torch.tensor([4.0, 0.0, 0.0])
    # Window one has only query one and expert two is best on the true role 1.
    expert[1, 1, 1, 0] = torch.tensor([8.0, 0.0, 0.0])
    expert[1, 1, 1, 1] = torch.tensor([4.0, 0.0, 0.0])
    expert[1, 1, 1, 2] = torch.tensor([1.0, 0.0, 0.0])
    target, error, valid = best_expert_targets(expert, batch)
    assert torch.equal(target, torch.tensor([1, 2]))
    assert torch.equal(valid, torch.tensor([True, True]))
    torch.testing.assert_close(error[0], torch.tensor([5.0, 3.0, 4.0]))


@torch.inference_mode()
def test_future_truth_pollution_changes_oracle_not_history_gate_forward() -> None:
    model = _model()
    batch = _supervised_batch()
    reference = model(batch)
    reference_expert = expert_counterfactual_role_position(model, reference, batch)
    _, reference_error, _ = best_expert_targets(reference_expert, batch)
    polluted = deepcopy(batch)
    polluted["truth_current_position_m"] += 1000.0
    polluted["target_visible_delta_m"] *= -999.0
    polluted["target_switch_count"] += 17
    polluted["motion_class"] = torch.tensor([999, -999])
    actual = model(polluted)
    for name in reference:
        assert torch.equal(actual[name], reference[name]), name
    actual_expert = expert_counterfactual_role_position(model, actual, polluted)
    _, actual_error, _ = best_expert_targets(actual_expert, polluted)
    assert not torch.equal(actual_error, reference_error)


def test_only_history_gate_updates_and_oracle_target_is_detached() -> None:
    model = _model().requires_grad_(False)
    model.motion_regime_gate.train().requires_grad_(True)
    batch = _supervised_batch()
    frozen_before = state_dict_sha256(_frozen_motion_state(model))
    gate_before = state_dict_sha256(model.motion_regime_gate.state_dict())
    prediction = model(batch)
    expert = expert_counterfactual_role_position(model, prediction, batch)
    target, error, _ = best_expert_targets(expert, batch)
    assert not target.requires_grad
    assert not error.requires_grad
    loss = F.cross_entropy(prediction["motion_regime_logits"].float(), target)
    optimizer = torch.optim.SGD(model.motion_regime_gate.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert all(parameter.grad is None for parameter in model.context.parameters())
    optimizer.step()
    assert state_dict_sha256(_frozen_motion_state(model)) == frozen_before
    assert state_dict_sha256(model.motion_regime_gate.state_dict()) != gate_before


def test_parent_loader_accepts_only_sealed_fixed_v2_endpoint(tmp_path) -> None:
    model = _model()
    payload = {
        "schema_version": PARENT_SCHEMA,
        "checkpoint_role": "fixed_final_endpoint",
        "fixed_endpoint": True,
        "progress": {"global_update": 2400},
        "provenance": {
            "oracle_association": True,
            "deployable_pipeline": False,
            "test_accessed": False,
        },
        "model_config": model.config,
        "model": model.state_dict(),
        "model_state_dict_sha256": state_dict_sha256(model.state_dict()),
    }
    accepted = tmp_path / "accepted.pt"
    torch.save(payload, accepted)
    loaded, info = _load_parent(
        accepted, expected_sha256=sha256_file(accepted),
    )
    assert info["global_update"] == 2400
    assert state_dict_sha256(loaded.state_dict()) == payload["model_state_dict_sha256"]

    payload["progress"]["global_update"] = 2399
    rejected = tmp_path / "rejected.pt"
    torch.save(payload, rejected)
    with pytest.raises(ValueError, match="fixed update 2400"):
        _load_parent(rejected, expected_sha256=sha256_file(rejected))

    with pytest.raises(ValueError, match="hash-locked v70"):
        _load_parent(accepted, expected_sha256="0" * 64)


def test_resume_metadata_fails_closed_on_filename_and_history_mismatch(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-update-000150.pt"
    checkpoint.touch()
    payload = {
        "progress": {"global_update": 150},
        "validation_history": [{"global_update": 0, "metrics": {}}],
    }
    update, history = _validate_resume_metadata(payload, checkpoint)
    assert update == 150 and history[0]["global_update"] == 0
    payload["progress"]["global_update"] = 149
    with pytest.raises(ValueError, match="filename and payload"):
        _validate_resume_metadata(payload, checkpoint)
    payload["progress"]["global_update"] = 150
    payload["validation_history"] = [{"global_update": 151, "metrics": {}}]
    with pytest.raises(ValueError, match="history is inconsistent"):
        _validate_resume_metadata(payload, checkpoint)
