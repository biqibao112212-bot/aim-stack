from __future__ import annotations

from copy import deepcopy
import inspect
import json
from types import SimpleNamespace

import torch
import pytest

from training.stage3.robust_multiscale_motion_future import (
    RobustMultiScaleIncrementMotionContext,
    RobustMultiScaleMotionBottleneckFutureModel,
    robust_multiscale_motion_future_loss,
)
from training.stage3.tests.test_anonymous_vehicle_motion import _batch
from training.stage3.tests.test_stable_motion_bottleneck_future import (
    _supervised_batch,
)
from training.stage3.train_stable_motion_bottleneck_future import (
    STATE_MODULES,
    stage_loss_weights,
)
from training.stage3.train_robust_multiscale_motion_future import (
    CONTROL_FIELDS,
    FROZEN_FUTURE_MODULES,
    _finalize_state_gate,
    _validate_state_gate_args,
    build_state_gate_parser,
)
import training.stage3.train_robust_multiscale_motion_future as gate_module
from training.stage3.observable_future_pnp_ab import state_dict_sha256


def _model() -> RobustMultiScaleMotionBottleneckFutureModel:
    torch.manual_seed(20260730)
    return RobustMultiScaleMotionBottleneckFutureModel(
        velocity_scale_mps=(3.0, 3.2, 0.25),
        yaw_rate_scale_rad_s=17.25,
        channels=32, dropout=0.0, message_layers=2, basis_count=6,
    ).eval()


def test_formal_parser_defaults_to_one_complete_state_only_stage() -> None:
    parser = build_state_gate_parser()
    assert parser.get_default("motion_state_updates") == 800
    assert parser.get_default("trajectory_updates") == 0
    assert parser.get_default("selector_updates") == 0
    assert parser.get_default("decoder_joint_updates") == 0
    assert parser.get_default("stop_after_update") == 0


def test_formal_state_gate_rejects_stage_override() -> None:
    valid = SimpleNamespace(
        motion_state_updates=800, trajectory_updates=0, selector_updates=0,
        decoder_joint_updates=0, stop_after_update=0,
    )
    _validate_state_gate_args(valid)
    invalid = SimpleNamespace(**vars(valid))
    invalid.trajectory_updates = 1
    with pytest.raises(ValueError, match="800-update state-only"):
        _validate_state_gate_args(invalid)


def _metric(mean: float) -> dict[str, float | int]:
    return {"count": 8, "mean_m": mean}


def test_failed_state_gate_is_explicit_and_binds_frozen_control(
    tmp_path, monkeypatch,
) -> None:
    control_root = tmp_path / "control"
    checkpoint_dir = control_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    control_checkpoint = checkpoint_dir / "checkpoint-update-000800.pt"
    model_state = {
        f"{name}.weight": torch.tensor([float(index + 1)])
        for index, name in enumerate(FROZEN_FUTURE_MODULES)
    }
    provenance = {
        "dataset": {"manifest_sha256": "dataset"},
        "truth_history": {"manifest_sha256": "truth"},
        "frozen_initial_state_dict_sha256": {
            "mapper": "mapper", "s": "s", "h": "h",
        },
    }
    common_args = {name: f"value-{name}" for name in CONTROL_FIELDS}
    control_contract = {"args": common_args}
    contract_sha = gate_module._json_sha256(control_contract)
    torch.save({
        "contract_sha256": contract_sha, "model": model_state,
        "provenance": provenance, "progress": {"global_update": 800},
        "run_id": "control-run",
    }, control_checkpoint)
    monkeypatch.setattr(
        gate_module, "CONTROL_CHECKPOINT_SHA256",
        gate_module.sha256_file(control_checkpoint),
    )
    monkeypatch.setattr(gate_module, "CONTROL_CONTRACT_SHA256", contract_sha)
    (control_root / "run_manifest.json").write_text(json.dumps({
        "contract": control_contract, "contract_sha256": contract_sha,
        "provenance": provenance, "run_id": "control-run",
    }), encoding="utf-8")

    output = tmp_path / "candidate"
    output.mkdir()
    future_hashes = {
        name: state_dict_sha256({"weight": model_state[f"{name}.weight"]})
        for name in FROZEN_FUTURE_MODULES
    }
    group = {
        "sample_count": 8,
        "velocity_vector_error_mps": _metric(0.2),
        "yaw_absolute_error_rad_s": _metric(1.0),
        "normalized_state_mae": _metric(0.05),
        "yaw_sign_accuracy_abs_truth_gt_0_5": 0.99,
    }
    failed_overall = dict(group)
    failed_overall["velocity_vector_error_mps"] = _metric(0.4)
    metrics = {
        "overall": failed_overall,
        "combined": group,
        "combined_speed_gt_1_7": group,
        "per_session": {
            "stage3-multistate-fixed6mm-20260730-v2-combined-11": group,
        },
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps({
        "status": "complete", "state_gate_only": True,
        "state_gate_future_modules_unchanged": True,
        "contract": {"args": common_args}, "provenance": provenance,
        "trainable_final_state_dict_sha256": future_hashes,
        "final_validation": {"motion_state": metrics},
    }), encoding="utf-8")
    gate = _finalize_state_gate(output, control_checkpoint)
    assert gate["status"] == "failed"
    assert gate["combined_speed_gt_1_7_sample_count"] == 8
    assert gate["frozen_future_modules"]["matched"] is True
    finalized = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert finalized["training_status"] == "complete"
    assert finalized["status"] == "state_gate_failed"


@torch.inference_mode()
def test_multiscale_state_has_available_normalized_weights() -> None:
    output = _model()(_batch())
    assert output["scale_available"].shape == (2, 5)
    assert output["scale_available"][:, :3].all()
    assert torch.isfinite(output["motion_state_normalized"]).all()
    torch.testing.assert_close(
        output["scale_motion_weight"].sum(dim=1),
        torch.ones(2, 4), rtol=1e-6, atol=1e-6,
    )
    assert torch.count_nonzero(
        output["scale_motion_weight"]
        * (~output["scale_coordinate_available"])
    ) == 0


@torch.inference_mode()
def test_multiscale_state_is_global_translation_invariant() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)["motion_state_normalized"]
    changed = deepcopy(batch)
    changed["history_obs_rel_m"] = (
        changed["history_obs_rel_m"]
        + torch.tensor([[[[7.0, -11.0, 5.0]]], [[[-3.0, 2.0, 9.0]]]])
    )
    actual = model(changed)["motion_state_normalized"]
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=3e-6)


@torch.inference_mode()
def test_multiscale_state_is_c4_relabel_invariant() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)["motion_state_normalized"]
    changed = deepcopy(batch)
    for name in ("history_obs_rel_m", "history_obs_mask", "history_primary_mask"):
        changed[name] = torch.roll(changed[name], shifts=1, dims=2)
    for name in (
        "q0_relation_m", "q0_sigma_m", "q0_confidence", "q0_age_s",
        "q0_support_class", "q0_supported",
    ):
        changed[name] = torch.roll(changed[name], shifts=1, dims=1)
    actual = model(changed)["motion_state_normalized"]
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=3e-6)


@torch.inference_mode()
def test_multiscale_state_is_handle_reflection_invariant() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)["motion_state_normalized"]
    changed = deepcopy(batch)
    reflection = torch.tensor([0, 3, 2, 1])
    for name in ("history_obs_rel_m", "history_obs_mask", "history_primary_mask"):
        changed[name] = changed[name].index_select(2, reflection)
    for name in (
        "q0_relation_m", "q0_sigma_m", "q0_confidence", "q0_age_s",
        "q0_support_class", "q0_supported",
    ):
        changed[name] = changed[name].index_select(1, reflection)
    changed["history_switch_step"] = -changed["history_switch_step"]
    actual = model(changed)["motion_state_normalized"]
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=3e-6)


@torch.inference_mode()
def test_multiscale_state_ignores_padding_q0_quality_and_forbidden_fields() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)["motion_state_normalized"]
    changed = deepcopy(batch)
    inactive = ~changed["history_event_mask"]
    changed["history_obs_rel_m"][inactive] = 1e6
    changed["history_time_s"][inactive] = -1e6
    changed["history_switch_step"][inactive] = 777
    changed["q0_relation_m"] = torch.randn_like(changed["q0_relation_m"]) * 100.0
    changed["q0_sigma_m"] = torch.randn_like(changed["q0_sigma_m"]) * 100.0
    changed["q0_confidence"] = torch.randn_like(changed["q0_confidence"]) * 100.0
    changed["q0_age_s"] = torch.randn_like(changed["q0_age_s"]) * 100.0
    changed["q0_support_class"] = torch.remainder(changed["q0_support_class"] + 1, 4)
    changed["q0_supported"] = ~changed["q0_supported"]
    changed["current_position_m"] = torch.randn_like(
        changed["current_position_m"]
    ) * 1000.0
    changed["tau_s"] = torch.flip(changed["tau_s"], dims=(1,))
    changed["candidate_relation_m"] = torch.randn_like(
        changed["candidate_relation_m"]
    ) * 1000.0
    changed.update({
        "target_motion_state_normalized": torch.randn(2, 4) * 1e6,
        "target_motion_state_physical": torch.randn(2, 4) * 1e6,
        "session_id": torch.tensor([123, 456]),
        "t0_ns": torch.tensor([7, 8]),
        "motion_class": torch.tensor([2, 3]),
        "future_truth": torch.randn(2, 9, 3) * 1e6,
    })
    actual = model(changed)["motion_state_normalized"]
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=3e-6)


def test_state_api_accepts_only_causal_observation_fields() -> None:
    expected = {
        "self", "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
        "history_event_mask", "history_time_s", "history_switch_step",
    }
    assert set(inspect.signature(
        RobustMultiScaleIncrementMotionContext.forward,
    ).parameters) == expected
    assert set(inspect.signature(
        RobustMultiScaleMotionBottleneckFutureModel.estimate_motion_state,
    ).parameters) == expected
    batch = _batch()
    output = _model().estimate_motion_state(**{
        name: batch[name] for name in expected if name != "self"
    })
    assert set(output) == {"history", "state"}
    assert output["state"]["motion_state_normalized"].shape == (2, 4)


@torch.inference_mode()
def test_pair_shape_stays_translation_invariant_under_bfloat16_autocast() -> None:
    observations = torch.tensor([[[
        [0.25, 0.03, -0.08], [0.0, 0.0, 0.0],
        [-0.24, -0.02, 0.07], [0.0, 0.0, 0.0],
    ]]], dtype=torch.float32)
    visible = torch.tensor([[[True, False, True, False]]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        reference, mask = RobustMultiScaleIncrementMotionContext._symmetric_pair_shape(
            observations, visible,
        )
        translated, translated_mask = (
            RobustMultiScaleIncrementMotionContext._symmetric_pair_shape(
                observations + torch.tensor([7.0, -11.0, 5.0]), visible,
            )
        )
    assert torch.equal(mask, translated_mask)
    torch.testing.assert_close(translated, reference, rtol=2e-5, atol=2e-6)


@torch.inference_mode()
def test_lag_scales_use_non_overlapping_time_bands_without_prior_reuse() -> None:
    context = RobustMultiScaleIncrementMotionContext(
        channels=32, dropout=0.0, message_layers=2,
    )
    time = torch.tensor([[-0.28, -0.20, -0.15, -0.10, -0.07, -0.03, -0.01, 0.0]])
    value = torch.stack((time, 2 * time, -time), dim=-1)
    valid = torch.ones_like(time, dtype=torch.bool)
    _, elapsed, edge_valid, prior = context._lag_bank(value, time, valid)
    endpoint = -1
    selected = prior[0, endpoint, edge_valid[0, endpoint]].tolist()
    assert len(selected) == len(set(selected))
    scale = context.lag_scales_s
    middle = torch.sqrt(scale[:-1] * scale[1:])
    lower = torch.cat((scale[:1] * 0.5, middle))
    upper = torch.cat((middle, scale[-1:] * 1.5))
    assert torch.all(elapsed[0, endpoint][edge_valid[0, endpoint]] >= lower[edge_valid[0, endpoint]])
    assert torch.all(elapsed[0, endpoint][edge_valid[0, endpoint]] < upper[edge_valid[0, endpoint]])


@torch.inference_mode()
def test_multiscale_lag_bank_is_strictly_causal() -> None:
    output = _model()(_batch())
    prior = output["lag_prior_index"]
    valid = output["lag_edge_valid"]
    endpoint = torch.arange(prior.shape[2])[None, None, :, None]
    assert torch.all(prior[valid] >= 0)
    assert torch.all(prior[valid] < endpoint.expand_as(prior)[valid])


def test_multiscale_auxiliary_state_is_supervised_and_future_is_isolated() -> None:
    model = _model().train()
    batch = _supervised_batch()
    prediction = model(batch, detach_motion_code=True)
    loss, components = robust_multiscale_motion_future_loss(
        prediction, batch, **stage_loss_weights("motion_state"),
    )
    assert components["scale_aux"] > 0
    loss.backward()
    state_gradient = [
        parameter.grad for name in STATE_MODULES
        for parameter in getattr(model, name).parameters()
    ]
    assert any(value is not None and torch.count_nonzero(value) > 0 for value in state_gradient)
    future_gradient = [
        parameter.grad for name in (
            "motion_state_encoder", "handle_encoder", "time_basis",
            "trajectory_coefficient_head", "role_coefficient_head",
        )
        for parameter in getattr(model, name).parameters()
    ]
    assert all(value is None or torch.count_nonzero(value) == 0 for value in future_gradient)

    model.zero_grad(set_to_none=True)
    prediction = model(batch, detach_motion_code=True)
    trajectory, _ = robust_multiscale_motion_future_loss(
        prediction, batch, **stage_loss_weights("trajectory"),
    )
    gradients = torch.autograd.grad(
        trajectory,
        [
            parameter for name in STATE_MODULES
            for parameter in getattr(model, name).parameters()
        ],
        allow_unused=True,
    )
    assert all(value is None or torch.count_nonzero(value) == 0 for value in gradients)
