from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from training.stage3.factorized_common_relative_motion_future import (
    FactorizedCommonRelativeMotionContext,
    FactorizedCommonRelativeMotionStateV7,
    FactorizedMotionStateHead,
    apply_common_velocity_ramp,
    factorized_motion_state_loss,
    factorized_state_train_step,
)
from training.stage3.tests.test_anonymous_vehicle_motion import _batch
from training.stage3.tests.test_stable_motion_bottleneck_future import _supervised_batch
from training.stage3.train_factorized_common_relative_motion_future import (
    RUN_SCHEMA,
    _load_bound_candidate,
    _validated_accuracy,
    _validated_error_distribution,
    _validate_state_gate_args,
    build_state_gate_parser,
)
from training.stage3.train_stable_motion_bottleneck_future import _callable_contract
from training.stage3.train_stable_motion_bottleneck_future import ALL_TRAINABLE_MODULES
from training.stage3.observable_future_pnp_ab import sha256_file, state_dict_sha256
from training.stage3.train_anonymous_vehicle_motion import _json_sha256


def _model() -> FactorizedCommonRelativeMotionStateV7:
    torch.manual_seed(20260730)
    return FactorizedCommonRelativeMotionStateV7(
        velocity_scale_mps=(3.0, 3.2, 0.25),
        yaw_rate_scale_rad_s=17.25,
        channels=32, dropout=0.0, message_layers=2, basis_count=6,
    )


def _state(model, batch):
    return model.estimate_motion_state(**{
        name: batch[name] for name in (
            "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
            "history_event_mask", "history_time_s", "history_switch_step",
        )
    })["state"]


def test_v7_formal_parser_is_fixed_to_complete_state_gate() -> None:
    parser = build_state_gate_parser()
    assert parser.get_default("motion_state_updates") == 800
    assert parser.get_default("trajectory_updates") == 0
    assert parser.get_default("selector_updates") == 0
    assert parser.get_default("decoder_joint_updates") == 0
    valid = SimpleNamespace(
        motion_state_updates=800, trajectory_updates=0, selector_updates=0,
        decoder_joint_updates=0, stop_after_update=0,
    )
    _validate_state_gate_args(valid)
    invalid = SimpleNamespace(**vars(valid))
    invalid.motion_state_updates = 801
    with pytest.raises(ValueError, match="800-update state-only"):
        _validate_state_gate_args(invalid)


def test_state_step_callable_is_semantic_source_bound() -> None:
    contract = _callable_contract(factorized_state_train_step)
    assert contract == {
        "module": factorized_state_train_step.__module__,
        "qualname": factorized_state_train_step.__qualname__,
        "semantic_source_sha256": contract["semantic_source_sha256"],
    }
    assert len(contract["semantic_source_sha256"]) == 64

    def different_step():
        return None

    assert _callable_contract(different_step) != contract


def _write_bound_candidate(root: Path) -> Path:
    output = root / "candidate"
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "checkpoint-update-000800.pt"
    model = _model()
    model_state = model.state_dict()
    contract = {"args": {"seed": 20260730}, "purpose": "unit-test"}
    contract_sha = _json_sha256(contract)
    provenance = {"dataset": {"validation": {"split": "validation"}}}
    final_validation = {"motion_state": {"unit": {"count": 2}}}
    validation_history = [{
        "global_update": 800,
        "stage": "motion_state",
        "metrics": final_validation,
    }]
    substage_counts = {
        "angular_specialization": 250,
        "translation_specialization": 350,
        "joint_calibration": 200,
    }
    substage_transitions = [
        {"global_update": 1, "substage": "angular_specialization"},
        {"global_update": 251, "substage": "translation_specialization"},
        {"global_update": 601, "substage": "joint_calibration"},
    ]
    branch_history = [{
        "global_update": update,
        "hashes": {"angular": f"a{update}", "translation_vertical": f"t{update}"},
    } for update in (0, 250, 600, 800)]
    isolation = {"future_to_state": True, "selector_to_decoder": False}
    diagnostics = {"schema_version": "unit-diagnostics-v1", "count": 2}
    payload = {
        "schema_version": RUN_SCHEMA,
        "model_class": FactorizedCommonRelativeMotionStateV7.__name__,
        "model_config": model.config,
        "model": model_state,
        "model_state_dict_sha256": state_dict_sha256(model_state),
        "progress": {"global_update": 800, "stage": "motion_state", "stage_update": 800},
        "validation_history": validation_history,
        "provenance": provenance,
        "contract_sha256": contract_sha,
        "checkpoint_role": "fixed_final_endpoint",
        "checkpoint_selected_by_validation": False,
        "fixed_endpoint": True,
        "stage_endpoint": True,
        "run_id": "unit-bound-candidate",
        "gradient_isolation_verified": isolation,
        "state_substage_counts": substage_counts,
        "state_substage_transitions": substage_transitions,
        "state_branch_hash_history": branch_history,
        "final_diagnostics": diagnostics,
    }
    torch.save(payload, checkpoint)
    module_hashes = {
        name: state_dict_sha256({
            key[len(name) + 1:]: value
            for key, value in model_state.items() if key.startswith(name + ".")
        })
        for name in ALL_TRAINABLE_MODULES
    }
    manifest = {
        "schema_version": RUN_SCHEMA,
        "run_id": payload["run_id"],
        "model_config": model.config,
        "contract": contract,
        "contract_sha256": contract_sha,
        "provenance": provenance,
        "validation_history": validation_history,
        "final_validation": final_validation,
        "gradient_isolation_verified": isolation,
        "state_substage_counts": substage_counts,
        "state_substage_transitions": substage_transitions,
        "state_branch_hash_history": branch_history,
        "final_diagnostics": diagnostics,
        "trainable_final_state_dict_sha256": module_hashes,
        "fixed_final_checkpoint": {
            "path": str(checkpoint), "sha256": sha256_file(checkpoint), "update": 800,
            "selected_by_validation": False,
        },
        "progress": {"global_update": 800, "latest_checkpoint": str(checkpoint)},
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8",
    )
    return output


def test_v7_candidate_loader_binds_manifest_to_exact_checkpoint(tmp_path: Path) -> None:
    output = _write_bound_candidate(tmp_path)
    manifest, payload = _load_bound_candidate(output)
    assert manifest["run_id"] == payload["run_id"] == "unit-bound-candidate"

    manifest_path = output / "run_manifest.json"
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["final_validation"] = {"motion_state": {"forged": {"count": 2}}}
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint-bound"):
        _load_bound_candidate(output)


def test_v7_candidate_loader_rejects_checkpoint_byte_tampering(tmp_path: Path) -> None:
    output = _write_bound_candidate(tmp_path)
    checkpoint = output / "checkpoints" / "checkpoint-update-000800.pt"
    with checkpoint.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="file hash"):
        _load_bound_candidate(output)


def test_v7_candidate_loader_rejects_rehashed_incomplete_model(tmp_path: Path) -> None:
    output = _write_bound_candidate(tmp_path)
    checkpoint = output / "checkpoints" / "checkpoint-update-000800.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model"].pop(next(iter(payload["model"])))
    payload["model_state_dict_sha256"] = state_dict_sha256(payload["model"])
    torch.save(payload, checkpoint)
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fixed_final_checkpoint"]["sha256"] = sha256_file(checkpoint)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be reconstructed exactly"):
        _load_bound_candidate(output)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_v7_error_distribution_rejects_nonfinite_or_negative(value: float) -> None:
    distribution = {
        "count": 3, "mean_m": 0.1, "p50_m": 0.1, "p95_m": 0.2, "p99_m": 0.3,
    }
    distribution["p95_m"] = value
    with pytest.raises(ValueError, match="finite nonnegative"):
        _validated_error_distribution(distribution, name="unit")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01, True])
def test_v7_accuracy_rejects_nonfinite_or_out_of_range(value: object) -> None:
    with pytest.raises(ValueError, match="finite probability"):
        _validated_accuracy(value, name="unit")


def test_v7_state_api_accepts_exactly_six_causal_fields() -> None:
    expected = {
        "self", "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
        "history_event_mask", "history_time_s", "history_switch_step",
    }
    assert set(inspect.signature(
        FactorizedCommonRelativeMotionContext.forward,
    ).parameters) == expected
    assert set(inspect.signature(
        FactorizedCommonRelativeMotionStateV7.estimate_motion_state,
    ).parameters) == expected
    output = _state(_model().eval(), _batch())
    assert output["motion_state_normalized"].shape == (2, 4)


@torch.inference_mode()
def test_v7_scale_weights_are_available_and_coordinate_normalized() -> None:
    output = _model().eval()(_batch())
    assert output["scale_coordinate_available"].shape == (2, 5, 4)
    torch.testing.assert_close(
        output["scale_motion_weight"].sum(dim=1),
        torch.ones(2, 4), rtol=1e-6, atol=1e-6,
    )
    assert torch.count_nonzero(
        output["scale_motion_weight"] * (~output["scale_coordinate_available"])
    ) == 0


@torch.inference_mode()
def test_v7_state_is_invariant_to_global_translation() -> None:
    model = _model().eval()
    batch = _batch()
    reference = _state(model, batch)["motion_state_normalized"]
    changed = deepcopy(batch)
    changed["history_obs_rel_m"] = changed["history_obs_rel_m"] + torch.tensor(
        [[[[7.0, -11.0, 5.0]]], [[[-3.0, 2.0, 9.0]]]],
    )
    actual = _state(model, changed)["motion_state_normalized"]
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=4e-6)


@torch.inference_mode()
def test_v7_state_is_c4_and_handle_reflection_invariant() -> None:
    model = _model().eval()
    batch = _batch()
    reference = _state(model, batch)["motion_state_normalized"]
    cyclic = deepcopy(batch)
    for name in ("history_obs_rel_m", "history_obs_mask", "history_primary_mask"):
        cyclic[name] = torch.roll(cyclic[name], shifts=1, dims=2)
    torch.testing.assert_close(
        _state(model, cyclic)["motion_state_normalized"],
        reference, rtol=2e-5, atol=4e-6,
    )
    reflected = deepcopy(batch)
    reflection = torch.tensor([0, 3, 2, 1])
    for name in ("history_obs_rel_m", "history_obs_mask", "history_primary_mask"):
        reflected[name] = reflected[name].index_select(2, reflection)
    reflected["history_switch_step"] = -reflected["history_switch_step"]
    torch.testing.assert_close(
        _state(model, reflected)["motion_state_normalized"],
        reference, rtol=2e-5, atol=4e-6,
    )


@torch.inference_mode()
def test_v7_ignores_padding_and_all_forbidden_future_fields() -> None:
    model = _model().eval()
    batch = _batch()
    reference = _state(model, batch)["motion_state_normalized"]
    changed = deepcopy(batch)
    inactive = ~changed["history_event_mask"]
    changed["history_obs_rel_m"][inactive] = 1e6
    changed["history_time_s"][inactive] = -1e6
    changed["history_switch_step"][inactive] = 777
    changed.update({
        "current_position_m": torch.randn(2, 3) * 1e5,
        "q0_relation_m": torch.randn(2, 4, 3) * 1e5,
        "q0_sigma_m": torch.randn(2, 4) * 1e5,
        "q0_confidence": torch.randn(2, 4) * 1e5,
        "q0_age_s": torch.randn(2, 4) * 1e5,
        "q0_support_class": torch.randint(0, 4, (2, 4)),
        "q0_supported": torch.randint(0, 2, (2, 4), dtype=torch.bool),
        "target_motion_state_normalized": torch.randn(2, 4) * 1e5,
        "future_truth": torch.randn(2, 9, 3) * 1e5,
        "session_id": torch.tensor([123, 456]),
        "motion_class": torch.tensor([2, 3]),
    })
    actual = _state(model, changed)["motion_state_normalized"]
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=4e-6)


@torch.inference_mode()
def test_normalized_pair_shape_is_translation_and_scale_invariant_in_bf16() -> None:
    observations = torch.tensor([[[
        [0.25, 0.03, -0.08], [0.0, 0.0, 0.0],
        [-0.24, -0.02, 0.07], [0.0, 0.0, 0.0],
    ]]], dtype=torch.float32)
    visible = torch.tensor([[[True, False, True, False]]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        reference, mask = FactorizedCommonRelativeMotionContext._normalized_pair_shape(
            observations, visible,
        )
        translated, translated_mask = (
            FactorizedCommonRelativeMotionContext._normalized_pair_shape(
                observations + torch.tensor([7.0, -11.0, 5.0]), visible,
            )
        )
        scaled, scaled_mask = FactorizedCommonRelativeMotionContext._normalized_pair_shape(
            observations * 4.5, visible,
        )
    assert torch.equal(mask, translated_mask)
    assert torch.equal(mask, scaled_mask)
    torch.testing.assert_close(translated, reference, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(scaled, reference, rtol=3e-5, atol=3e-6)


def test_v7_branch_information_flow_is_one_way_and_detached() -> None:
    torch.manual_seed(5)
    head = FactorizedMotionStateHead(channels=32, dropout=0.0).eval()
    translation = torch.randn(2, 5, 64, requires_grad=True)
    rotation = torch.randn(2, 5, 64, requires_grad=True)
    available = torch.ones(2, 5, dtype=torch.bool)
    translation_reliability = torch.randn(2, 5, 5)
    rotation_reliability = torch.randn(2, 5, 3)
    output = head(
        translation, rotation, available, available,
        translation_reliability, rotation_reliability,
    )
    planar_gradient = torch.autograd.grad(
        output["motion_state_normalized"][:, :2].sum(), rotation,
        retain_graph=True, allow_unused=True,
    )[0]
    assert planar_gradient is None or torch.count_nonzero(planar_gradient) == 0
    yaw_gradient = torch.autograd.grad(
        output["motion_state_normalized"][:, 3].sum(), translation,
        allow_unused=True,
    )[0]
    assert yaw_gradient is None or torch.count_nonzero(yaw_gradient) == 0


@torch.inference_mode()
def test_v7_yaw_never_reads_translation_and_interaction_can_be_disabled() -> None:
    torch.manual_seed(7)
    head = FactorizedMotionStateHead(channels=32, dropout=0.0).eval()
    translation = torch.randn(2, 5, 64)
    rotation = torch.randn(2, 5, 64)
    available = torch.ones(2, 5, dtype=torch.bool)
    tr = torch.randn(2, 5, 5)
    rr = torch.randn(2, 5, 3)
    first = head(translation, rotation, available, available, tr, rr)
    changed_translation = head(
        translation + 100.0, rotation, available, available, tr, rr,
    )
    torch.testing.assert_close(
        first["motion_state_normalized"][:, 3],
        changed_translation["motion_state_normalized"][:, 3],
    )
    changed_rotation_enabled = head(
        translation, rotation * -13.0, available, available, tr, rr * 9.0,
    )
    torch.testing.assert_close(
        first["motion_state_normalized"][:, 2],
        changed_rotation_enabled["motion_state_normalized"][:, 2],
    )
    head.interaction_enabled = False
    base = head(translation, rotation, available, available, tr, rr)
    changed_rotation = head(
        translation, rotation * -13.0, available, available, tr, rr * 9.0,
    )
    torch.testing.assert_close(
        base["motion_state_normalized"][:, :3],
        changed_rotation["motion_state_normalized"][:, :3],
    )


@torch.inference_mode()
def test_single_handle_curvature_yaw_is_common_velocity_ramp_invariant() -> None:
    model = _model().eval()
    events = 32
    time = torch.linspace(-0.31, 0.0, events).unsqueeze(0)
    position = torch.zeros(1, events, 4, 3)
    position[0, :, 0, 0] = 0.4 * torch.cos(5.0 * time[0]) + 0.7 * time[0]
    position[0, :, 0, 1] = 0.4 * torch.sin(5.0 * time[0]) - 0.2 * time[0]
    position[0, :, 0, 2] = 0.1
    mask = torch.zeros(1, events, 4, dtype=torch.bool)
    mask[:, :, 0] = True
    fields = {
        "history_obs_rel_m": position,
        "history_obs_mask": mask,
        "history_primary_mask": mask.clone(),
        "history_event_mask": torch.ones(1, events, dtype=torch.bool),
        "history_time_s": time,
        "history_switch_step": torch.zeros(1, events, dtype=torch.long),
    }
    reference = model.estimate_motion_state(**fields)["state"][
        "motion_state_normalized"
    ][:, 3]
    ramp = torch.tensor([1.1, -0.8, 0.0])
    changed = dict(fields)
    changed["history_obs_rel_m"] = position + time[:, :, None, None] * ramp
    actual = model.estimate_motion_state(**changed)["state"][
        "motion_state_normalized"
    ][:, 3]
    torch.testing.assert_close(actual, reference, rtol=3e-4, atol=3e-5)


@torch.inference_mode()
def test_short_history_without_angular_evidence_uses_finite_missing_prior() -> None:
    model = _model().eval()
    batch = _batch()
    batch["history_obs_mask"] = batch["history_primary_mask"].clone()
    output = model.estimate_motion_state(**{
        name: batch[name] for name in (
            "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
            "history_event_mask", "history_time_s", "history_switch_step",
        )
    })
    assert torch.isfinite(output["state"]["motion_state_normalized"]).all()
    missing = ~output["history"]["rotation_scale_available"].any(dim=1)
    assert missing.any()
    torch.testing.assert_close(
        output["state"]["scale_motion_weight"][missing, :, 3].sum(dim=1),
        torch.ones(int(missing.sum())),
    )


def test_common_velocity_ramp_changes_only_visible_history_and_velocity_label() -> None:
    batch = _supervised_batch()
    scale = torch.tensor([3.0, 3.2, 0.25, 17.25])
    ramp = torch.tensor([[0.4, -0.2, 0.0], [-0.3, 0.5, 0.0]])
    changed = apply_common_velocity_ramp(batch, ramp, scale)
    valid = (
        batch["history_event_mask"][:, :, None]
        & batch["history_obs_mask"]
    )
    expected_offset = batch["history_time_s"][:, :, None, None] * ramp[:, None, None]
    torch.testing.assert_close(
        changed["history_obs_rel_m"][valid],
        (batch["history_obs_rel_m"] + expected_offset)[valid],
    )
    torch.testing.assert_close(
        changed["history_obs_rel_m"][~valid], batch["history_obs_rel_m"][~valid],
    )
    torch.testing.assert_close(
        changed["target_motion_state_normalized"][:, :3],
        batch["target_motion_state_normalized"][:, :3] + ramp / scale[:3],
    )
    torch.testing.assert_close(
        changed["target_motion_state_normalized"][:, 3],
        batch["target_motion_state_normalized"][:, 3],
    )


def test_v7_fixed_substages_freeze_the_opposite_branch() -> None:
    model = _model().train()
    assert model.set_state_training_update(1) == "angular_specialization"
    assert any(p.requires_grad for p in model.context.pair_rotation_consensus.parameters())
    assert all(
        not p.requires_grad for p in model.context.handle_translation_consensus.parameters()
    )
    assert model.set_state_training_update(251) == "translation_specialization"
    assert all(
        not p.requires_grad for p in model.context.pair_rotation_consensus.parameters()
    )
    assert any(p.requires_grad for p in model.context.handle_translation_consensus.parameters())
    assert model.set_state_training_update(601) == "joint_calibration"
    assert all(p.requires_grad for p in model.context.parameters())
    assert model.motion_state_head.interaction_enabled is True
    hashes = model.state_branch_hashes()
    assert set(hashes) == {"angular", "translation_vertical"}
    assert all(len(value) == 64 for value in hashes.values())


def test_v7_state_loss_is_future_isolated_and_training_step_is_finite() -> None:
    model = _model().train()
    batch = _supervised_batch()
    state_output = model.estimate_motion_state(**{
        name: batch[name] for name in (
            "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
            "history_event_mask", "history_time_s", "history_switch_step",
        )
    })
    prediction = {**state_output["history"], **state_output["state"]}
    loss, components = factorized_motion_state_loss(prediction, batch)
    poisoned = dict(batch)
    poisoned.update({
        "future_target_position_m": torch.full((2, 9, 3), torch.nan),
        "q0_relation_m": torch.full((2, 4, 3), torch.nan),
        "current_position_m": torch.full((2, 3), torch.nan),
    })
    poisoned_loss, _ = factorized_motion_state_loss(prediction, poisoned)
    torch.testing.assert_close(poisoned_loss, loss)
    assert torch.isfinite(components["scale_aux"])
    torch.manual_seed(11)
    _, train_loss, train_components = factorized_state_train_step(
        model, batch, 601, 800,
    )
    assert torch.isfinite(train_loss)
    assert train_components["state_substage"] == "joint_calibration"
    train_loss.backward()
    assert all(
        parameter.grad is None
        for module in (
            model.motion_state_encoder, model.handle_encoder, model.time_basis,
            model.trajectory_coefficient_head, model.role_coefficient_head,
        )
        for parameter in module.parameters()
    )
