from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from training.stage3.build_observable_future_pnp_sf_upper_bound_dataset import (
    LEGACY_SCHEMA_VERSION,
    SCHEMA_VERSION,
)
from training.stage3.pnp_observation_mapper import (
    AlignedAnchoredWindowPnPObservationMapper,
    AnchoredWindowPnPObservationMapper,
    INPUT_FIELDS,
    LABEL_FIELDS,
    CausalPnPObservationMapper,
    PnPObservationMappingDataset,
    WindowPnPObservationMapper,
    _masked_event_time_and_delta,
    compose_corrected_observations,
)
from training.stage3.pnp_q0_hypothesis_adapter import load_frozen_pnp_mapper
from training.stage3.train_pnp_observation_mapper import mapping_loss


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(batch: int = 2) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(7)
    obs = torch.randn(batch, 32, 4, 3)
    mask = torch.rand(batch, 32, 4) > 0.35
    mask[:, :, 0] = True
    time = torch.linspace(-0.31, 0.0, 32).repeat(batch, 1)
    event = torch.ones(batch, 32, dtype=torch.bool)
    return obs, mask, time, event


def _model() -> CausalPnPObservationMapper:
    return CausalPnPObservationMapper(
        torch.tensor((3.0, 0.0, -0.07)),
        torch.tensor((1.5, 0.8, 0.01)),
        channels=16,
        dropout=0.0,
    )


def _window_model() -> WindowPnPObservationMapper:
    return WindowPnPObservationMapper(
        torch.tensor((3.0, 0.0, -0.07)),
        torch.tensor((1.5, 0.8, 0.01)),
        channels=16, dropout=0.0,
    )


def test_masked_event_delta_does_not_cross_inactive_prefix() -> None:
    event_time = torch.tensor(((-0.31, -0.30, -0.29, -0.28),))
    valid_event = torch.tensor(((False, False, True, True),))
    clean_time, dt = _masked_event_time_and_delta(event_time, valid_event)
    assert torch.equal(clean_time, torch.tensor(((0.0, 0.0, -0.29, -0.28),)))
    assert torch.allclose(dt, torch.tensor(((0.0, 0.0, 0.0, 0.01),)), atol=1e-7)


def test_mapper_zero_initialization_preserves_valid_pnp() -> None:
    model = _model()
    obs, mask, time, event = _inputs()
    output = model(obs, mask, time, event)
    assert torch.equal(output["corrected_obs_m"][mask], obs[mask])
    assert torch.equal(
        output["corrected_obs_m"][~mask],
        torch.zeros_like(output["corrected_obs_m"][~mask]),
    )
    assert model.config["physical_id_input"] is False
    assert model.config["primary_or_switch_input"] is False
    assert model.config["future_or_motion_class_input"] is False


def test_mapper_is_c4_equivariant() -> None:
    model = _model().eval()
    with torch.no_grad():
        model.residual_head[-1].weight.normal_(0.0, 0.02)
    obs, mask, time, event = _inputs()
    reference = model(obs, mask, time, event)["corrected_obs_m"]
    shifted = model(
        torch.roll(obs, 1, dims=2), torch.roll(mask, 1, dims=2), time, event
    )["corrected_obs_m"]
    assert torch.allclose(shifted, torch.roll(reference, 1, dims=2), atol=1e-6)


def test_mapper_is_strictly_causal() -> None:
    model = _model().eval()
    with torch.no_grad():
        model.residual_head[-1].weight.normal_(0.0, 0.02)
    obs, mask, time, event = _inputs()
    changed = obs.clone()
    changed[:, 20:] += 5.0
    reference = model(obs, mask, time, event)["corrected_obs_m"]
    perturbed = model(changed, mask, time, event)["corrected_obs_m"]
    assert torch.allclose(reference[:, :20], perturbed[:, :20], atol=1e-6)


def test_mapper_accepts_one_observed_event_without_filling_missing_points() -> None:
    model = _model()
    obs, mask, time, event = _inputs(batch=1)
    mask.zero_()
    mask[:, -1, 2] = True
    output = model(obs, mask, time, event)
    assert torch.equal(output["corrected_obs_m"][mask], obs[mask])
    assert torch.count_nonzero(output["corrected_obs_m"][~mask]) == 0


def test_window_mapper_zero_init_and_no_post_q0_input() -> None:
    model = _window_model()
    obs, mask, time, event = _inputs()
    output = model(obs, mask, time, event)
    assert torch.equal(output["corrected_obs_m"][mask], obs[mask])
    after_q0 = time.clone()
    after_q0[:, -1] = 0.01
    with pytest.raises(ValueError, match="after q0"):
        model(obs, mask, after_q0, event)


def test_window_mapper_is_equivariant_to_every_handle_permutation() -> None:
    model = _window_model().eval()
    with torch.no_grad():
        model.residual_head[-1].weight.normal_(0.0, 0.02)
    obs, mask, time, event = _inputs(batch=1)
    obs = obs * torch.tensor((0.3, 0.2, 0.005)) + torch.tensor((3.0, 0.0, -0.07))
    reference = model(obs, mask, time, event)["corrected_obs_m"]
    for permutation in itertools.permutations(range(4)):
        index = torch.tensor(permutation)
        output = model(
            obs[:, :, index], mask[:, :, index], time, event
        )["corrected_obs_m"]
        assert torch.allclose(
            output, reference[:, :, index], atol=1e-6, rtol=0.0
        )


def test_window_mapper_uses_later_past_but_ignores_masked_poison() -> None:
    model = _window_model().eval()
    with torch.no_grad():
        model.residual_head[-1].weight.normal_(0.0, 0.02)
    obs, mask, time, event = _inputs(batch=1)
    obs = obs * torch.tensor((0.3, 0.2, 0.005)) + torch.tensor((3.0, 0.0, -0.07))
    reference = model(obs, mask, time, event)["corrected_obs_m"]
    changed = obs.clone()
    changed[:, -1, 0] += 0.3
    later_changed = model(changed, mask, time, event)["corrected_obs_m"]
    assert not torch.allclose(
        reference[:, -4, 0], later_changed[:, -4, 0], atol=1e-7, rtol=0.0
    )
    poisoned = obs.clone()
    poisoned[~mask] = torch.nan
    poison_output = model(poisoned, mask, time, event)["corrected_obs_m"]
    assert torch.equal(poison_output[mask], reference[mask])
    assert torch.count_nonzero(poison_output[~mask]) == 0


def test_window_mapper_rejects_invalid_observed_event_and_nonmonotonic_time() -> None:
    model = _window_model()
    obs, mask, time, event = _inputs(batch=1)
    invalid_event = event.clone()
    invalid_event[:, 10] = False
    with pytest.raises(ValueError, match="invalid event"):
        model(obs, mask, time, invalid_event)

    mask[:, 10] = False
    time[:, 10] = 99.0
    reference = model(obs, mask, time, invalid_event)["corrected_obs_m"]
    time[:, 10] = torch.nan
    poisoned = model(obs, mask, time, invalid_event)["corrected_obs_m"]
    assert torch.equal(reference, poisoned)

    mask[:, 10, 0] = True
    invalid_event[:, 10] = True
    time[:, 10] = time[:, 11]
    with pytest.raises(ValueError, match="strictly increasing"):
        model(obs, mask, time, invalid_event)


def test_anchored_window_mapper_preserves_q0_and_delegates_past() -> None:
    anchor = _model().eval()
    window = _window_model().eval()
    with torch.no_grad():
        anchor.residual_head[-1].weight.normal_(0.0, 0.02)
        window.residual_head[-1].weight.normal_(0.0, 0.02)
    model = AnchoredWindowPnPObservationMapper(anchor, window)
    obs, mask, time, event = _inputs(batch=1)
    obs = obs * torch.tensor((0.3, 0.2, 0.005)) + torch.tensor((3.0, 0.0, -0.07))
    anchor_output = anchor(obs, mask, time, event)["corrected_obs_m"]
    window_output = window(obs, mask, time, event)["corrected_obs_m"]
    output = model(obs, mask, time, event)["corrected_obs_m"]
    q0 = time.abs().le(1e-6)[:, :, None, None].expand_as(output)
    assert torch.equal(output[q0], anchor_output[q0])
    assert torch.equal(output[~q0], window_output[~q0])
    model.train()
    assert not model.anchor_mapper.training
    assert model.window_smoother.training
    assert not any(
        parameter.requires_grad for parameter in model.anchor_mapper.parameters()
    )


def test_aligned_anchored_window_preserves_q0_and_aligns_past_equivariantly() -> None:
    anchor = _model().eval()
    window = _window_model().eval()
    with torch.no_grad():
        anchor.residual_head[-1].weight.normal_(0.0, 0.02)
        window.residual_head[-1].weight.normal_(0.0, 0.02)
    model = AlignedAnchoredWindowPnPObservationMapper(anchor, window).eval()
    obs, mask, time, event = _inputs(batch=1)
    obs = obs * torch.tensor((0.3, 0.2, 0.005)) + torch.tensor((3.0, 0.0, -0.07))
    anchor_output = anchor(obs, mask, time, event)["corrected_obs_m"]
    window_output = window(obs, mask, time, event)["corrected_obs_m"]
    result = model(obs, mask, time, event)
    q0_event = time.abs().le(1e-6)
    q0 = q0_event[:, :, None, None].expand_as(result["corrected_obs_m"])
    assert torch.equal(result["corrected_obs_m"][q0], anchor_output[q0])
    expected_past = window_output + result["past_alignment_m"][:, None, None]
    valid_past = mask.unsqueeze(-1) & ~q0
    assert torch.allclose(
        result["corrected_obs_m"][valid_past], expected_past[valid_past],
        atol=1e-7, rtol=0.0,
    )
    reference = result["corrected_obs_m"]
    reference_alignment = result["past_alignment_m"]
    for permutation in itertools.permutations(range(4)):
        index = torch.tensor(permutation)
        shifted = model(obs[:, :, index], mask[:, :, index], time, event)
        assert torch.allclose(
            shifted["corrected_obs_m"], reference[:, :, index],
            atol=1e-6, rtol=0.0,
        )
        assert torch.allclose(
            shifted["past_alignment_m"], reference_alignment,
            atol=1e-6, rtol=0.0,
        )
    assert model.config["physical_id_input"] is False
    assert model.config["primary_or_switch_input"] is False
    assert model.config["future_or_motion_class_input"] is False


def test_aligned_anchored_window_checkpoint_round_trip(tmp_path: Path) -> None:
    model = AlignedAnchoredWindowPnPObservationMapper(
        _model(), _window_model()
    ).eval()
    checkpoint = tmp_path / "aligned-window.pt"
    torch.save({
        "schema_version": "stage3-pnp-observation-mapper-run-v1",
        "model_class": type(model).__name__,
        "epoch": 3,
        "update": 17,
        "model_config": model.config,
        "model": model.state_dict(),
        "provenance": {"test_accessed": False},
    }, checkpoint)
    loaded, provenance = load_frozen_pnp_mapper(checkpoint)
    assert isinstance(loaded, AlignedAnchoredWindowPnPObservationMapper)
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    assert loaded.config == model.config
    assert provenance["epoch"] == 3 and provenance["update"] == 17
    for name, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value)


def test_mapping_dataset_exposes_only_allowlisted_fields(tmp_path: Path) -> None:
    count = 3
    arrays = {
        "pnp_s_obs_m": np.ones((count, 32, 4, 3), dtype=np.float32),
        "pnp_s_obs_mask": np.ones((count, 32, 4), dtype=np.bool_),
        "pnp_s_event_time_s": np.linspace(-0.31, 0.0, 32, dtype=np.float32)[None].repeat(count, 0),
        "pnp_s_event_mask": np.ones((count, 32), dtype=np.bool_),
        "clean_s_obs_m": np.full((count, 32, 4, 3), 2.0, dtype=np.float32),
        "clean_s_obs_mask": np.ones((count, 32, 4), dtype=np.bool_),
        "pnp_s_primary_mask": np.eye(4, dtype=np.bool_)[
            np.zeros((count, 32), dtype=np.int64)
        ],
        # These forbidden values deliberately exist in the physical shard.
        "pnp_s_truth_q0_m": np.full((count, 4, 3), 999.0, dtype=np.float32),
        "target_switch_count": np.full((count, 8), 999, dtype=np.int64),
        "motion_class": np.full(count, 3, dtype=np.int64),
        "session_id": np.asarray([f"session-{index}" for index in range(count)]),
        "t0_ns": np.arange(count, dtype=np.int64),
        "pair_id": np.asarray([f"pair-{index}" for index in range(count)]),
    }
    shard = tmp_path / "train.npz"
    np.savez_compressed(shard, **arrays)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "qualification_passed": True,
        "test_accessed": False,
        "oracle_association": True,
        "deployable_pipeline": False,
        "shards": [{
            "split": "train", "path": shard.name, "sha256": _sha256(shard),
        }],
    }
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    dataset = PnPObservationMappingDataset(tmp_path, "train")
    assert set(dataset[0]) == set(INPUT_FIELDS + LABEL_FIELDS)
    assert "pnp_s_truth_q0_m" not in dataset.tensors
    assert "target_switch_count" not in dataset.tensors
    assert "motion_class" not in dataset.tensors
    manifest["schema_version"] = LEGACY_SCHEMA_VERSION
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    legacy = PnPObservationMappingDataset(tmp_path, "train")
    assert len(legacy) == len(dataset)


def test_mapping_dataset_selection_fields_never_reach_model_item(tmp_path: Path) -> None:
    count = 2
    shard = tmp_path / "train.npz"
    obs_mask = np.ones((count, 32, 4), dtype=np.bool_)
    np.savez_compressed(
        shard,
        pnp_s_obs_m=np.ones((count, 32, 4, 3), dtype=np.float32),
        pnp_s_obs_mask=obs_mask,
        pnp_s_event_time_s=np.linspace(-0.31, 0.0, 32, dtype=np.float32)[None].repeat(count, 0),
        pnp_s_event_mask=np.ones((count, 32), dtype=np.bool_),
        clean_s_obs_m=np.ones((count, 32, 4, 3), dtype=np.float32),
        clean_s_obs_mask=obs_mask,
        pnp_s_primary_mask=np.eye(4, dtype=np.bool_)[np.zeros((count, 32), dtype=np.int64)],
        motion_class=np.asarray((3, 2), dtype=np.int64),
        pnp_sf_common_usable=np.asarray((True, True), dtype=np.bool_),
        session_id=np.asarray(("session-a", "session-b")),
        t0_ns=np.asarray((1, 2), dtype=np.int64),
        pair_id=np.asarray(("pair-a", "pair-b")),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION, "qualification_passed": True,
        "test_accessed": False, "oracle_association": True,
        "deployable_pipeline": False,
        "shards": [{"split": "train", "path": shard.name, "sha256": _sha256(shard)}],
    }
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    dataset = PnPObservationMappingDataset(
        tmp_path, "train", motion_class=3, require_common=True
    )
    assert len(dataset) == 1
    assert "motion_class" not in dataset[0]
    assert "pnp_sf_common_usable" not in dataset[0]


def test_corrected_compose_never_reads_raw_pnp_history() -> None:
    batch, events = 2, 32
    primary_index = torch.tensor((0, 2), dtype=torch.long)
    primary = torch.nn.functional.one_hot(primary_index, 4).bool()
    primary = primary[:, None, :].expand(-1, events, -1).clone()
    mask = primary.clone()
    corrected = torch.randn(batch, events, 4, 3)
    q0 = torch.randn(batch, 4, 3)
    step = torch.tensor((-1, 0, 1, 2), dtype=torch.long).repeat(batch, 1)
    output = compose_corrected_observations({
        "q0_m": q0,
        "primary_index": primary_index,
        "confidence": torch.ones(batch, 4),
        "q0_valid": torch.ones(batch, 4, dtype=torch.bool),
    }, corrected, mask, primary, step)
    expected = (corrected * primary.unsqueeze(-1)).sum(dim=2)
    assert torch.equal(output["selected_history_absolute_m"], expected)
    torch.testing.assert_close(
        output["history_position_rel_m"]
        + output["current_position_m"][:, None, :],
        expected,
        atol=1e-6,
        rtol=0.0,
    )


def test_mapper_trajectory_losses_ignore_common_bias_but_penalize_jitter() -> None:
    batch, events = 1, 4
    clean = torch.zeros(batch, events, 4, 3)
    clean[0, :, 0, 0] = torch.arange(events, dtype=torch.float32) * 0.01
    mask = torch.zeros(batch, events, 4, dtype=torch.bool)
    mask[:, :, 0] = True
    primary = mask.clone()
    common_bias = torch.tensor((0.2, -0.1, 0.03))
    corrected = torch.where(
        mask.unsqueeze(-1), clean + common_bias, torch.zeros_like(clean)
    )
    sample = {
        "pnp_s_obs_m": clean.clone(),
        "pnp_s_obs_mask": mask,
        "pnp_s_primary_mask": primary,
        "pnp_s_event_mask": torch.ones(batch, events, dtype=torch.bool),
        "pnp_s_event_time_s": torch.tensor(((-0.03, -0.02, -0.01, 0.0),)),
        "clean_s_obs_m": clean,
    }
    _, common = mapping_loss(
        {"corrected_obs_m": corrected}, sample,
        huber_beta_m=0.01, mse_weight=0.0, tail_weight=0.0,
        tail_fraction=0.25, q0_primary_weight=0.0,
        primary_history_weight=0.0, primary_relative_weight=1.0,
        primary_increment_weight=1.0,
    )
    assert float(common["primary_relative_huber"]) < 1e-12
    assert float(common["primary_increment_huber"]) < 1e-12
    poisoned_sample = dict(sample)
    poisoned_sample["clean_s_obs_m"] = clean.clone()
    poisoned_prediction = corrected.clone()
    poisoned_sample["clean_s_obs_m"][~mask] = torch.nan
    poisoned_prediction[~mask] = torch.nan
    poisoned_objective, poisoned = mapping_loss(
        {"corrected_obs_m": poisoned_prediction}, poisoned_sample,
        huber_beta_m=0.01, mse_weight=0.0, tail_weight=0.0,
        tail_fraction=0.25, q0_primary_weight=0.0,
        primary_history_weight=0.0, primary_relative_weight=1.0,
        primary_increment_weight=1.0,
    )
    assert torch.isfinite(poisoned_objective)
    assert torch.equal(
        poisoned["primary_relative_huber"], common["primary_relative_huber"]
    )
    assert torch.equal(
        poisoned["primary_increment_huber"], common["primary_increment_huber"]
    )

    jitter_scale = torch.tensor(0.05, requires_grad=True)
    time_ramp = torch.arange(events, dtype=torch.float32)[None, :, None, None]
    jittered = corrected + mask.unsqueeze(-1) * time_ramp * jitter_scale
    objective, jitter = mapping_loss(
        {"corrected_obs_m": jittered}, sample,
        huber_beta_m=0.01, mse_weight=0.0, tail_weight=0.0,
        tail_fraction=0.25, q0_primary_weight=0.0,
        primary_history_weight=0.0, primary_relative_weight=1.0,
        primary_increment_weight=1.0,
    )
    assert float(jitter["primary_relative_huber"]) > 0
    assert float(jitter["primary_increment_huber"]) > 0
    objective.backward()
    assert jitter_scale.grad is not None and float(jitter_scale.grad) != 0.0


def test_mapper_trajectory_loss_is_c4_invariant() -> None:
    torch.manual_seed(23)
    batch, events = 2, 5
    clean = torch.randn(batch, events, 4, 3)
    corrected = (clean + torch.randn_like(clean) * 0.03).requires_grad_()
    primary_index = torch.tensor(((0, 1, 1, 2, 2), (3, 3, 0, 0, 1)))
    primary = torch.nn.functional.one_hot(primary_index, 4).bool()
    mask = primary.clone()
    sample = {
        "pnp_s_obs_mask": mask,
        "pnp_s_primary_mask": primary,
        "pnp_s_event_mask": torch.ones(batch, events, dtype=torch.bool),
        "pnp_s_event_time_s": torch.tensor(
            ((-0.04, -0.03, -0.02, -0.01, 0.0),) * batch
        ),
        "clean_s_obs_m": clean,
    }
    kwargs = dict(
        huber_beta_m=0.01, mse_weight=0.0, tail_weight=0.0,
        tail_fraction=0.2, q0_primary_weight=0.0,
        primary_history_weight=0.0, primary_relative_weight=1.0,
        primary_increment_weight=1.0,
    )
    reference, _ = mapping_loss(
        {"corrected_obs_m": corrected}, sample, **kwargs
    )
    reference_gradient, = torch.autograd.grad(reference, corrected)
    for shift in range(4):
        shifted_corrected = torch.roll(
            corrected.detach(), shift, dims=2
        ).requires_grad_()
        shifted_sample = dict(sample)
        for name in ("pnp_s_obs_mask", "pnp_s_primary_mask", "clean_s_obs_m"):
            shifted_sample[name] = torch.roll(sample[name], shift, dims=2)
        shifted_loss, _ = mapping_loss(
            {"corrected_obs_m": shifted_corrected}, shifted_sample, **kwargs
        )
        shifted_gradient, = torch.autograd.grad(shifted_loss, shifted_corrected)
        assert torch.equal(shifted_loss, reference)
        assert torch.allclose(
            torch.roll(shifted_gradient, -shift, dims=2),
            reference_gradient, atol=1e-7, rtol=0.0,
        )
