from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from training.stage3.center_offset_supervision import (
    CENTER_OFFSET_TARGET_FIELD,
    CENTER_POSITION_TARGET_FIELD,
    CenterTruthIndex,
    attach_center_offset_after_frozen_upstream,
)
from training.stage3.evaluate_profiled_center_twist_mechanism import (
    TRAINED_SCREEN_SCHEMA,
    _validate_checkpoint_manifest_identity,
)
from training.stage3.profiled_center_twist_future import (
    CENTER_TWIST_FORWARD_FIELDS,
    AnonymousQ0CenterPrior,
    CenterPriorProfiledTwistScreen,
    anonymous_center_prior_loss,
    translation_only_fwl,
)
from training.stage3.train_profiled_center_prior_screen import (
    _global_half_cycle_derangement,
)


def _rigid_batch(batch_size: int = 2) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    events = 8
    time = torch.linspace(-0.07, 0.0, events).expand(batch_size, -1).clone()
    center = torch.tensor([[0.20, 0.00, 0.01], [0.10, -0.08, -0.02]])[:batch_size]
    radial = torch.tensor([
        [-0.20, 0.00, 0.00], [0.00, 0.30, 0.01],
        [0.20, 0.00, -0.01], [0.00, -0.30, 0.00],
    ])
    q0 = center[:, None] + radial[None]
    velocity = torch.tensor([[0.55, -0.35, 0.08], [-0.25, 0.40, -0.03]])[:batch_size]
    omega = torch.tensor([4.0, -6.0])[:batch_size]
    history = torch.zeros(batch_size, events, 4, 3)
    for row in range(batch_size):
        for event in range(events):
            theta = omega[row] * time[row, event]
            rotation = torch.stack((
                torch.cos(theta), -torch.sin(theta),
                torch.sin(theta), torch.cos(theta),
            )).reshape(2, 2)
            history[row, event, :, :2] = (
                (torch.eye(2) - rotation) @ center[row, :2]
                + time[row, event] * velocity[row, :2]
                + (rotation @ q0[row, :, :2].T).T
            )
            history[row, event, :, 2] = q0[row, :, 2] + time[row, event] * velocity[row, 2]
    mask = torch.ones(batch_size, events, 4, dtype=torch.bool)
    primary = torch.zeros_like(mask)
    primary[:, :, 0] = True
    batch = {
        "history_obs_rel_m": history,
        "history_obs_mask": mask,
        "history_primary_mask": primary,
        "history_event_mask": torch.ones(batch_size, events, dtype=torch.bool),
        "history_time_s": time,
        "history_switch_step": torch.zeros(batch_size, events, dtype=torch.long),
        "q0_relation_m": q0,
        "q0_supported": torch.ones(batch_size, 4, dtype=torch.bool),
    }
    return batch, velocity, omega


def test_forward_contract_has_exactly_six_history_plus_two_q0_fields() -> None:
    assert CENTER_TWIST_FORWARD_FIELDS == (
        "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
        "history_event_mask", "history_time_s", "history_switch_step",
        "q0_relation_m", "q0_supported",
    )


@torch.inference_mode()
def test_zero_update_center_carrier_uses_all_four_anonymous_hypotheses() -> None:
    prior = AnonymousQ0CenterPrior(width=12).eval()
    relation = torch.tensor([[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0],
                              [0.2, 0.3, 0.0], [0.2, -0.3, 0.0]]])
    support = torch.tensor([[True, True, False, True]])
    result = prior(relation, support)
    torch.testing.assert_close(result["center_offset_m"], relation.mean(dim=1))
    changed = relation.clone()
    changed[:, 2, 0] += 0.4
    actual = prior(changed, support)["center_offset_m"]
    assert not torch.equal(actual, result["center_offset_m"])


@torch.inference_mode()
def test_center_prior_is_permutation_and_o2_equivariant() -> None:
    torch.manual_seed(9)
    prior = AnonymousQ0CenterPrior(width=16).eval()
    relation = torch.randn(3, 4, 3)
    support = torch.tensor([
        [True, True, False, True], [True, False, False, False],
        [True, True, True, True],
    ])
    reference = prior(relation, support)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = prior(relation[:, permutation], support[:, permutation])
    torch.testing.assert_close(permuted["center_offset_m"], reference["center_offset_m"])
    torch.testing.assert_close(
        permuted["center_log_variance_xy_z"],
        reference["center_log_variance_xy_z"],
    )

    transform = torch.tensor([[0.6, -0.8], [-0.8, -0.6]])
    reflected = relation.clone()
    reflected[..., :2] = relation[..., :2] @ transform.T
    transformed = prior(reflected, support)
    expected = reference["center_offset_m"].clone()
    expected[..., :2] = reference["center_offset_m"][..., :2] @ transform.T
    torch.testing.assert_close(transformed["center_offset_m"], expected, atol=2e-6, rtol=1e-5)


@torch.inference_mode()
def test_profile_recovers_noiseless_twist_and_exact_common_ramp() -> None:
    batch, velocity, omega = _rigid_batch()
    model = CenterPriorProfiledTwistScreen(width=16, center_precision=1000.0).eval()
    reference = model.forward_at_omega(batch, omega)
    torch.testing.assert_close(
        reference["q0_component_velocity_mps"], velocity,
        atol=8e-4, rtol=2e-3,
    )

    ramp = torch.tensor([[0.30, -0.20, 0.05], [-0.15, 0.25, -0.04]])
    changed = deepcopy(batch)
    changed["history_obs_rel_m"] = (
        batch["history_obs_rel_m"]
        + batch["history_time_s"][:, :, None, None] * ramp[:, None, None]
    )
    actual = model.forward_at_omega(changed, omega)
    torch.testing.assert_close(
        actual["velocity_mps"], reference["velocity_mps"] + ramp,
        atol=4e-5, rtol=2e-5,
    )
    torch.testing.assert_close(actual["yaw_rate_rad_s"], reference["yaw_rate_rad_s"])
    torch.testing.assert_close(
        actual["q0_component_weight"], reference["q0_component_weight"],
        atol=2e-6, rtol=1e-5,
    )


@torch.inference_mode()
def test_profile_is_o2_equivariant_with_yaw_pseudoscalar() -> None:
    batch, _, omega = _rigid_batch()
    model = CenterPriorProfiledTwistScreen(width=16, center_precision=1000.0).eval()
    reference = model.forward_at_omega(batch, omega)
    transform = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    changed = deepcopy(batch)
    for name in ("history_obs_rel_m", "q0_relation_m"):
        value = batch[name].clone()
        value[..., :2] = value[..., :2] @ transform.T
        changed[name] = value
    actual = model.forward_at_omega(changed, -omega)
    expected_velocity = reference["velocity_mps"].clone()
    expected_velocity[..., :2] = reference["velocity_mps"][..., :2] @ transform.T
    torch.testing.assert_close(actual["velocity_mps"], expected_velocity, atol=5e-5, rtol=2e-4)
    torch.testing.assert_close(actual["yaw_rate_rad_s"], -reference["yaw_rate_rad_s"])
    torch.testing.assert_close(
        actual["q0_component_weight"], reference["q0_component_weight"],
        atol=2e-6, rtol=1e-5,
    )


@torch.inference_mode()
def test_history_only_branch_ignores_q0_geometry() -> None:
    batch, _, omega = _rigid_batch()
    model = CenterPriorProfiledTwistScreen(width=16).eval()
    reference = model.forward_at_omega(batch, omega, use_q0_prior=False)
    changed = deepcopy(batch)
    changed["q0_relation_m"] = torch.randn_like(changed["q0_relation_m"]) * 100.0
    actual = model.forward_at_omega(changed, omega, use_q0_prior=False)
    torch.testing.assert_close(actual["velocity_mps"], reference["velocity_mps"])


def test_shuffled_q0_diagnostics_can_train_gate_toward_history_branch() -> None:
    batch, velocity, omega = _rigid_batch()
    model = CenterPriorProfiledTwistScreen(width=16).eval()
    blind = model.forward_at_omega(batch, omega, use_q0_prior=False)
    corrupted = deepcopy(batch)
    corrupted["q0_relation_m"] = torch.roll(batch["q0_relation_m"], 1, dims=0)
    corrupted["q0_supported"] = torch.roll(batch["q0_supported"], 1, dims=0)
    before = model.forward_at_omega(corrupted, omega)
    assert bool(torch.all(
        before["q0_profile_energy"] > before["history_profile_energy"]
    ))
    optimizer = torch.optim.SGD(model.component_gate.parameters(), lr=0.5)
    loss = F.binary_cross_entropy_with_logits(
        before["q0_component_logit"], torch.zeros_like(before["q0_component_logit"]),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    after = model.forward_at_omega(corrupted, omega)
    assert bool(torch.all(
        after["q0_component_weight"] < before["q0_component_weight"]
    ))
    assert torch.isfinite(blind["velocity_mps"]).all()


def test_center_truth_only_changes_loss_not_forward() -> None:
    batch, _, omega = _rigid_batch()
    model = CenterPriorProfiledTwistScreen(width=16).eval()
    reference = model.forward_at_omega(batch, omega)
    polluted = dict(batch)
    polluted[CENTER_OFFSET_TARGET_FIELD] = torch.randn(2, 3) * 1e6
    actual = model.forward_at_omega(polluted, omega)
    for name in reference:
        assert torch.equal(actual[name], reference[name]), name
    first, _ = anonymous_center_prior_loss(reference, torch.zeros(2, 3))
    second, _ = anonymous_center_prior_loss(reference, torch.ones(2, 3))
    assert not torch.equal(first, second)


def test_truth_omega_profile_loss_backpropagates_only_into_center_prior() -> None:
    batch, velocity, omega = _rigid_batch()
    model = CenterPriorProfiledTwistScreen(width=16).train()
    prediction = model.forward_at_omega(batch, omega)
    loss = F.smooth_l1_loss(
        prediction["velocity_mps"], velocity + 0.1, beta=0.25,
    )
    loss.backward()
    gradients = {
        name: parameter.grad for name, parameter in model.named_parameters()
    }
    assert gradients
    assert all(value is not None for value in gradients.values())
    assert all(bool(torch.isfinite(value).all()) for value in gradients.values())
    assert any(bool(torch.any(value != 0)) for value in gradients.values())


@torch.inference_mode()
def test_translation_fallback_has_exact_common_ramp() -> None:
    batch, _, _ = _rigid_batch()
    reference = translation_only_fwl(
        batch["history_obs_rel_m"], batch["history_obs_mask"],
        batch["history_event_mask"], batch["history_time_s"],
    )
    ramp = torch.tensor([[0.2, -0.1, 0.03], [-0.3, 0.4, -0.02]])
    changed = batch["history_obs_rel_m"] + (
        batch["history_time_s"][:, :, None, None] * ramp[:, None, None]
    )
    actual = translation_only_fwl(
        changed, batch["history_obs_mask"],
        batch["history_event_mask"], batch["history_time_s"],
    )
    torch.testing.assert_close(
        actual["velocity_mps"], reference["velocity_mps"] + ramp,
        atol=2e-6, rtol=1e-5,
    )


@torch.inference_mode()
def test_pathological_short_arc_is_not_silently_profiled_or_supported() -> None:
    batch, _, omega = _rigid_batch(batch_size=1)
    tiny_time = torch.linspace(-1e-6, 0.0, 8).unsqueeze(0)
    batch["history_time_s"] = tiny_time
    batch["history_obs_rel_m"] = (
        batch["q0_relation_m"][:, None].expand(-1, 8, -1, -1).clone()
    )
    batch["history_obs_mask"][:, :, 1:] = False
    model = CenterPriorProfiledTwistScreen(width=16).eval()
    result = model.forward_at_omega(batch, omega[:1])
    assert not bool(result["profile_supported"][0])
    assert not bool(result["fallback_supported"][0])
    assert not bool(result["state_supported"][0])
    assert result["velocity_information_min_eigenvalue_s2"][0] < 1e-4
    assert torch.equal(result["velocity_mps"][0], torch.zeros(3))


@torch.inference_mode()
def test_unsupported_fallback_makes_no_global_ramp_claim() -> None:
    batch, _, _ = _rigid_batch(batch_size=1)
    batch["history_time_s"].zero_()
    reference = translation_only_fwl(
        batch["history_obs_rel_m"], batch["history_obs_mask"],
        batch["history_event_mask"], batch["history_time_s"],
    )
    assert not bool(reference["supported"][0])
    assert torch.equal(reference["velocity_mps"][0], torch.zeros(3))


@torch.inference_mode()
def test_fallback_rejects_many_observations_inside_submillisecond_span() -> None:
    batch, _, _ = _rigid_batch(batch_size=1)
    batch["history_time_s"] = torch.linspace(-5e-4, 0.0, 8).unsqueeze(0)
    ramp = torch.tensor([[[[200.0, -100.0, 25.0]]]])
    batch["history_obs_rel_m"] = (
        batch["q0_relation_m"][:, None]
        + batch["history_time_s"][:, :, None, None] * ramp
    )
    result = translation_only_fwl(
        batch["history_obs_rel_m"], batch["history_obs_mask"],
        batch["history_event_mask"], batch["history_time_s"],
    )
    assert result["time_information"][0] > 1e-8
    assert result["time_span_s"][0] < 1e-3
    assert not bool(result["supported"][0])
    assert torch.equal(result["velocity_mps"][0], torch.zeros(3))


def _write_center_truth_fixture(
    root: Path, *, duplicate: bool = False, nonfinite: bool = False,
) -> str:
    shard = root / "shards" / "train.npz"
    shard.parent.mkdir(parents=True)
    sessions = np.asarray(["session-a", "session-a"] if duplicate else ["session-a"])
    t0 = np.asarray([100, 100] if duplicate else [100], dtype=np.int64)
    count = len(t0)
    center = np.zeros((count, 3), dtype=np.float32)
    if nonfinite:
        center[0, 0] = np.nan
    np.savez_compressed(
        shard, session_id=sessions, t0_ns=t0,
        motion_class=np.full(count, 3, dtype=np.int64),
        anchor_center_position_m=center,
    )
    shard_sha = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "stage3-truth-history-v1",
        "test_accessed": False, "qualification_passed": True,
        "splits": ["train", "validation"], "sample_count": count,
        "shards": [{
            "path": "shards\\train.npz", "sha256": shard_sha,
            "split": "train", "sample_count": count,
        }],
    }
    path = root / "dataset_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_center_truth_index_rejects_duplicate_and_nonfinite(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate"
    manifest_sha = _write_center_truth_fixture(duplicate, duplicate=True)
    with pytest.raises(ValueError, match="duplicate center truth join key"):
        CenterTruthIndex(duplicate, expected_manifest_sha256=manifest_sha)
    nonfinite = tmp_path / "nonfinite"
    manifest_sha = _write_center_truth_fixture(nonfinite, nonfinite=True)
    with pytest.raises(ValueError, match="invalid 3D centers"):
        CenterTruthIndex(nonfinite, expected_manifest_sha256=manifest_sha)


def test_center_offset_is_formed_after_h_current() -> None:
    raw = {CENTER_POSITION_TARGET_FIELD: torch.tensor([[1.0, 2.0, 3.0]])}
    prepared = {"current_position_m": torch.tensor([[0.4, -0.2, 2.5]])}
    attach_center_offset_after_frozen_upstream(prepared, raw)
    torch.testing.assert_close(
        prepared[CENTER_OFFSET_TARGET_FIELD], torch.tensor([[0.6, 2.2, 0.5]]),
    )
    assert not prepared[CENTER_OFFSET_TARGET_FIELD].requires_grad


@pytest.mark.parametrize("sample_count", [2, 3, 64, 750])
def test_global_q0_counterfactual_is_deterministic_derangement(
    sample_count: int,
) -> None:
    first, first_hash = _global_half_cycle_derangement(
        sample_count, torch.device("cpu"),
    )
    second, second_hash = _global_half_cycle_derangement(
        sample_count, torch.device("cpu"),
    )
    assert torch.equal(first, second)
    assert first_hash == second_hash
    assert not bool(torch.any(first == torch.arange(sample_count)))
    assert torch.equal(torch.sort(first).values, torch.arange(sample_count))


def _checkpoint_identity_fixture() -> tuple[dict[str, object], dict[str, object]]:
    contract = {"dataset_manifest_sha256": "dataset", "args": {"width": 32}}
    digest = hashlib.sha256(json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()
    payload: dict[str, object] = {
        "schema_version": TRAINED_SCREEN_SCHEMA,
        "global_update": 100, "contract_sha256": digest,
    }
    manifest: dict[str, object] = {
        "schema_version": TRAINED_SCREEN_SCHEMA,
        "status": "completed", "global_update": 100,
        "contract": contract, "contract_sha256": digest,
        "final_checkpoint_sha256": "checkpoint",
        "checkpoint_ledger": [{
            "global_update": 100, "sha256": "checkpoint",
            "kind": "training_boundary",
        }],
    }
    return payload, manifest


def test_checkpoint_identity_recomputes_contract_and_ledger() -> None:
    payload, manifest = _checkpoint_identity_fixture()
    assert _validate_checkpoint_manifest_identity(
        payload, manifest, "checkpoint",
    ) == manifest["contract"]
    changed = deepcopy(manifest)
    changed["contract"]["args"]["width"] = 64
    with pytest.raises(ValueError, match="contract content/hash differs"):
        _validate_checkpoint_manifest_identity(payload, changed, "checkpoint")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("payload_update", "fixed update-100"),
        ("manifest_update", "fixed update-100"),
        ("checkpoint_hash", "checkpoint hash differs"),
        ("ledger_hash", "checkpoint ledger differs"),
        ("ledger_kind", "checkpoint ledger differs"),
    ],
)
def test_checkpoint_identity_rejects_wrong_update_hash_and_kind(
    mutation: str, message: str,
) -> None:
    payload, manifest = _checkpoint_identity_fixture()
    checkpoint_sha = "checkpoint"
    if mutation == "payload_update":
        payload["global_update"] = 75
    elif mutation == "manifest_update":
        manifest["global_update"] = 75
    elif mutation == "checkpoint_hash":
        checkpoint_sha = "other"
    elif mutation == "ledger_hash":
        manifest["checkpoint_ledger"][0]["sha256"] = "other"
    else:
        manifest["checkpoint_ledger"][0]["kind"] = "other"
    with pytest.raises(ValueError, match=message):
        _validate_checkpoint_manifest_identity(payload, manifest, checkpoint_sha)
