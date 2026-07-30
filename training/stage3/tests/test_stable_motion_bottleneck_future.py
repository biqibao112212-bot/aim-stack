from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from training.stage3.motion_truth_supervision import MotionTruthIndex
from training.stage3.stable_motion_bottleneck_future import (
    StableMotionBottleneckAnonymousFutureModel,
    stable_motion_future_loss,
)
from training.stage3.tests.test_anonymous_vehicle_motion import _batch
from training.stage3.train_stable_motion_bottleneck_future import (
    STATE_MODULES,
    configure_stage,
    stage_loss_weights,
    stage_modules,
)


def _model() -> StableMotionBottleneckAnonymousFutureModel:
    torch.manual_seed(20260730)
    return StableMotionBottleneckAnonymousFutureModel(
        velocity_scale_mps=(3.0, 3.2, 0.25),
        yaw_rate_scale_rad_s=17.25,
        channels=32, dropout=0.0, message_layers=2, basis_count=6,
    ).eval()


def _supervised_batch() -> dict[str, torch.Tensor]:
    batch = _batch()
    batch.update({
        "target_query_mask": torch.ones(2, 4, dtype=torch.bool),
        "target_switch_count": torch.tensor([[0, 1, 2, 3], [0, -1, -2, -3]]),
        "truth_current_position_m": batch["current_position_m"].clone(),
        "target_visible_delta_m": torch.randn(2, 4, 3) * 0.2,
        "target_motion_state_normalized": torch.tensor([
            [0.2, -0.3, 0.0, 0.6], [-0.1, 0.4, 0.0, -0.5],
        ]),
    })
    return batch


@torch.inference_mode()
def test_motion_estimator_has_no_q0_or_raw_origin_shortcut() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)["motion_state_normalized"]
    changed = deepcopy(batch)
    offset = torch.randn(2, 4, 3)
    changed["history_obs_rel_m"] = changed["history_obs_rel_m"] + offset[:, None]
    changed["q0_relation_m"] = torch.randn_like(changed["q0_relation_m"]) * 100.0
    changed["q0_sigma_m"] = torch.randn_like(changed["q0_sigma_m"]) * 100.0
    changed["q0_confidence"] = torch.randn_like(changed["q0_confidence"]) * 100.0
    changed["q0_age_s"] = torch.randn_like(changed["q0_age_s"]) * 100.0
    changed["q0_support_class"] = torch.remainder(changed["q0_support_class"] + 1, 4)
    changed["q0_supported"] = ~changed["q0_supported"]
    actual = model(changed)["motion_state_normalized"]
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=3e-6)


@torch.inference_mode()
def test_explicit_decoder_zero_motion_has_exactly_zero_dynamic_displacement() -> None:
    model = _model()
    batch = _batch()
    decoded = model.decode_ordered(
        current_position_m=batch["current_position_m"],
        tau_s=batch["tau_s"],
        ordered_q0_relation_m=batch["q0_relation_m"],
        ordered_q0_supported=batch["q0_supported"],
        motion_state_normalized=torch.zeros(2, 4),
    )
    expected = batch["q0_relation_m"][:, None].expand_as(decoded["role_delta_m"])
    assert torch.equal(decoded["role_delta_m"], expected)


@torch.inference_mode()
def test_full_role_trajectory_is_reconstructed_by_explicit_decoder() -> None:
    model = _model()
    batch = _batch()
    full = model(batch)
    primary = full["primary_index"]
    roles = torch.arange(4)[None]
    ordered = torch.remainder(primary[:, None] + roles, 4)
    relation = batch["q0_relation_m"].gather(
        1, ordered.unsqueeze(-1).expand(-1, -1, 3),
    ).clone()
    relation[:, 0] = 0.0
    supported = batch["q0_supported"].gather(1, ordered)
    decoded = model.decode_ordered(
        current_position_m=batch["current_position_m"],
        tau_s=batch["tau_s"],
        ordered_q0_relation_m=relation,
        ordered_q0_supported=supported,
        motion_state_normalized=full["motion_state_normalized"],
    )
    for name in (
        "trajectory_coefficient", "role_logits", "role_delta_m",
        "role_position_m", "position_m",
    ):
        assert torch.equal(decoded[name], full[name]), name


@torch.inference_mode()
def test_truth_fields_are_not_forward_inputs() -> None:
    model = _model()
    batch = _batch()
    reference = model(batch)
    polluted = deepcopy(batch)
    polluted.update({
        "target_motion_state_normalized": torch.randn(2, 4) * 1e6,
        "target_motion_state_physical": torch.randn(2, 4) * 1e6,
        "session_id": torch.tensor([123, 456]),
        "t0_ns": torch.tensor([7, 8]),
        "motion_class": torch.tensor([2, 3]),
        "future_truth": torch.randn(2, 9, 3) * 1e6,
    })
    actual = model(polluted)
    for name in reference:
        assert torch.equal(actual[name], reference[name]), name


def test_future_loss_cannot_update_motion_estimator() -> None:
    model = _model().train()
    batch = _supervised_batch()
    prediction = model(batch, detach_motion_code=True)
    loss, _ = stable_motion_future_loss(
        prediction, batch, **stage_loss_weights("trajectory"),
    )
    state_parameters = [
        parameter for name in STATE_MODULES
        for parameter in getattr(model, name).parameters()
    ]
    gradient = torch.autograd.grad(loss, state_parameters, allow_unused=True)
    assert all(value is None or torch.count_nonzero(value) == 0 for value in gradient)


@pytest.mark.parametrize(
    "stage", ["motion_state", "trajectory", "selector", "decoder_joint"],
)
def test_four_stage_parameter_allowlist(stage: str) -> None:
    model = _model()
    configure_stage(model, stage)
    expected = {
        id(parameter) for name in stage_modules(stage)
        for parameter in getattr(model, name).parameters()
    }
    assert {id(value) for value in model.parameters() if value.requires_grad} == expected


def _write_truth_fixture(
    root: Path, *, duplicate: bool = False,
) -> str:
    shard = root / "shards" / "train.npz"
    shard.parent.mkdir(parents=True)
    sessions = np.asarray(["session-a", "session-a"] if duplicate else ["session-a"])
    t0 = np.asarray([100, 100] if duplicate else [100], dtype=np.int64)
    count = len(t0)
    np.savez_compressed(
        shard,
        session_id=sessions, t0_ns=t0,
        motion_class=np.full(count, 3, dtype=np.int64),
        anchor_velocity_mps=np.zeros((count, 3), dtype=np.float32),
        anchor_yaw_rate_rad_s=np.zeros(count, dtype=np.float32),
    )
    payload = shard.read_bytes()
    shard_sha = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema_version": "stage3-truth-history-v1",
        "test_accessed": False,
        "qualification_passed": True,
        "splits": ["train", "validation"],
        "sample_count": count,
        "shards": [{
            "path": "shards\\train.npz", "sha256": shard_sha,
            "split": "train", "sample_count": count,
        }],
    }
    path = root / "dataset_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_truth_index_rejects_duplicate_exact_join_key(tmp_path: Path) -> None:
    manifest_sha = _write_truth_fixture(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="duplicate motion truth join key"):
        MotionTruthIndex(tmp_path, expected_manifest_sha256=manifest_sha)
