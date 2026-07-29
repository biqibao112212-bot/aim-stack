from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from training.stage3.cyclic_track_dataset import cyclic_relabel
from training.stage3.pnp_q0_hypothesis_adapter import (
    C4Q0HypothesisAdapter,
    H_INPUT_FIELDS,
    PnPQ0HypothesisDataset,
    _undo_reflection_keep_c4_origin,
    compose_hypothesis_for_f,
    hypothesis_forward,
    load_frozen_hypothesis_adapter,
    roll_s_output_c4,
)
from training.stage3.train_pnp_q0_hypothesis_adapter import (
    _acquire_run_lock,
    _capture_rng_state,
    _load_recovery_state,
    _read_run_lock,
    _release_run_lock,
    _restore_rng_state,
    _train_cache_contract,
    _write_recovery_state,
    TRAIN_CACHE_SCHEMA,
    build_h_optimizer,
    hypothesis_loss,
    parser,
)
from training.stage3.observable_future_model import AnonymousCandidateFutureExpert
from training.stage3.train_observable_future_dedicated_selector import (
    FrozenTrajectoryDedicatedSelector,
    _module_state_sha256,
)
from training.stage3.train_observable_future_selector import _selector_loss
from training.stage3.train_pnp_q0_hypothesis_selector import (
    PNP_SELECTOR_TRAINABLE_PREFIXES,
    _metric_batch,
    _selector_loss_prediction,
    _validate_selector_run_manifest,
    freeze_pnp_selector,
    set_pnp_selector_train_mode,
)


def _s_output(batch: int = 2) -> dict[str, torch.Tensor]:
    q0 = torch.tensor((
        ((1.0, 0.0, 0.0), (1.2, 0.1, 0.0), (1.0, 0.3, 0.1), (0.8, 0.1, 0.0)),
        ((2.0, 0.0, 0.0), (2.2, 0.1, 0.0), (2.0, 0.3, 0.1), (1.8, 0.1, 0.0)),
    ), dtype=torch.float32)[:batch]
    primary = torch.zeros(batch, 4, dtype=torch.bool)
    primary[:, 0] = True
    current_visible = torch.zeros_like(primary)
    current_visible[:, 0] = True
    if batch > 1:
        current_visible[1, 3] = True
    seen = current_visible.clone()
    seen[:, 1] = True
    seen[0, 2] = True
    warm = seen & ~current_visible
    cold = ~seen
    adjacent = torch.zeros_like(primary)
    adjacent[:, 1] = True
    adjacent[:, 3] = True
    edge = torch.roll(q0, shifts=-1, dims=1) - q0
    edge_supported = seen & torch.roll(seen, shifts=-1, dims=1)
    anchor_composed = warm & adjacent
    confidence = torch.full((batch, 4), 0.5)
    confidence = torch.where(cold, torch.zeros_like(confidence), confidence)
    return {
        "q0_m": q0,
        "direct_q0_m": q0 + 0.01,
        "q0_sigma_m": torch.full((batch, 4, 1), 0.05),
        "confidence": confidence,
        "edge0_m": edge,
        "edge0_sigma_m": torch.full((batch, 4, 1), 0.08),
        "edge0_supported": edge_supported,
        "age_s": torch.where(seen, torch.full((batch, 4), 0.05), torch.inf),
        "current_primary": primary,
        "current_visible": current_visible,
        "warm_hidden": warm,
        "cold": cold,
        "adjacent": adjacent,
        "anchor_composed": anchor_composed,
        "primary_index": torch.zeros(batch, dtype=torch.long),
    }


def _model() -> C4Q0HypothesisAdapter:
    return C4Q0HypothesisAdapter(
        torch.tensor((1.5, 0.0, 0.0)), torch.tensor((0.7, 0.4, 0.2)),
        channels=32, dropout=0.0, message_layers=2,
    )


def test_h_forward_signature_and_dataset_allowlist() -> None:
    names = tuple(inspect.signature(C4Q0HypothesisAdapter.forward).parameters)
    forbidden = ("future", "tau", "switch", "motion", "session", "pair", "candidate")
    assert not any(token in name for name in names for token in forbidden)
    assert "pnp_s_direction_sign" not in H_INPUT_FIELDS
    assert "pnp_s_window_shift" not in H_INPUT_FIELDS
    assert "motion_class" not in H_INPUT_FIELDS
    assert set(PnPQ0HypothesisDataset.__getitem__.__annotations__) == {"index", "return"}


def test_h_c4_equivariance_all_four_shifts_and_observed_identity() -> None:
    torch.manual_seed(9)
    model = _model().eval()
    for parameter in model.parameters():
        if parameter.ndim > 1:
            torch.nn.init.normal_(parameter, std=0.03)
    source = _s_output()
    reference = hypothesis_forward(model, source)
    assert torch.equal(reference["q0_m"][reference["observed"]], source["q0_m"][reference["observed"]])
    assert torch.equal(reference["q0_m"][reference["true_cold"]], source["q0_m"][reference["true_cold"]])
    assert torch.equal(reference["confidence_for_f"], source["confidence"])
    for shift in range(4):
        output = hypothesis_forward(model, roll_s_output_c4(source, shift))
        for name in (
            "q0_m", "edge0_m", "hypothesis_sigma_m", "support_probability",
            "confidence_for_f", "support_class", "evidence_supported",
        ):
            expected = torch.roll(reference[name], shifts=shift, dims=1)
            if expected.is_floating_point():
                assert torch.allclose(output[name], expected, atol=1e-6, rtol=0.0)
            else:
                assert torch.equal(output[name], expected)


def test_h_rejects_nonzero_true_cold_confidence() -> None:
    source = _s_output()
    source["confidence"] = source["confidence"].clone()
    source["confidence"][source["cold"]] = 0.25
    with pytest.raises(ValueError, match="true-cold confidence"):
        hypothesis_forward(_model().eval(), source)


@pytest.mark.parametrize(
    "provenance, message",
    (({}, "lacks formal/diagnostic"),
     ({"validation_from_train": True, "diagnostic_only": True},
      "diagnostic H checkpoint")),
)
def test_formal_h_loader_rejects_unsealed_or_diagnostic_checkpoint(
    tmp_path, provenance: dict[str, bool], message: str,
) -> None:
    checkpoint = tmp_path / "h.pt"
    torch.save({
        "schema_version": "stage3-pnp-q0-hypothesis-adapter-run-v1",
        "model_class": "C4Q0HypothesisAdapter",
        "provenance": provenance,
    }, checkpoint)
    with pytest.raises(ValueError, match=message):
        load_frozen_hypothesis_adapter(checkpoint)


def test_formal_h_loader_rejects_tampered_contract(tmp_path) -> None:
    checkpoint = tmp_path / "h.pt"
    torch.save({
        "schema_version": "stage3-pnp-q0-hypothesis-adapter-run-v1",
        "model_class": "C4Q0HypothesisAdapter",
        "provenance": {
            "validation_from_train": False,
            "diagnostic_only": False,
            "test_accessed": True,
            "oracle_association": True,
            "deployable_pipeline": False,
            "optimizer_only_h": True,
            "cold_coordinate_supervision": False,
        },
    }, checkpoint)
    with pytest.raises(ValueError, match="test_accessed"):
        load_frozen_hypothesis_adapter(checkpoint)


def test_selector_manifest_gate_failed_checkpoint_is_not_formal(tmp_path) -> None:
    checkpoint = tmp_path / "epoch-0001-update-000001.pt"
    payload = {
        "schema_version": "stage3-pnp-q0-hypothesis-selector-run-v1",
        "model_class": "PnPQ0HypothesisFrozenTrajectorySelector",
        "epoch": 1, "update": 1, "selection": [1.0], "validation": {},
    }
    torch.save(payload, checkpoint)
    from training.stage3.observable_future_pnp_ab import sha256_file
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({
            "schema_version": payload["schema_version"],
            "status": "gate_failed",
            "gates": {"accepted": False},
            "best": {
                "path": checkpoint.name,
                "sha256": sha256_file(checkpoint),
                "epoch": 1, "update": 1,
                "selection": [1.0], "validation": {},
            },
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest-declared best"):
        _validate_selector_run_manifest(checkpoint, payload)


def test_true_cold_coordinate_target_is_never_consumed_or_differentiated() -> None:
    source = _s_output()
    output = hypothesis_forward(_model(), source)
    target = source["q0_m"].clone()
    loss1, _ = hypothesis_loss(output, target, huber_beta_m=0.01)
    poison = target.clone()
    poison[output["true_cold"]] = float("nan")
    loss2, _ = hypothesis_loss(output, poison, huber_beta_m=0.01)
    assert torch.equal(loss1, loss2)

    q0 = output["q0_m"].detach().clone().requires_grad_(True)
    edge = output["edge0_m"].detach().clone().requires_grad_(True)
    logits = output["support_logits"].detach().clone().requires_grad_(True)
    sigma = output["hypothesis_sigma_m"].detach().clone().requires_grad_(True)
    copied = dict(output)
    copied.update({
        "q0_m": q0, "edge0_m": edge,
        "support_logits": logits, "hypothesis_sigma_m": sigma,
    })
    loss, _ = hypothesis_loss(copied, target, huber_beta_m=0.01)
    loss.backward()
    assert torch.count_nonzero(q0.grad[output["true_cold"]]) == 0
    assert torch.isfinite(loss)


def test_reflection_is_removed_but_window_c4_roll_is_retained() -> None:
    original = np.arange(1 * 2 * 4 * 3, dtype=np.float32).reshape(1, 2, 4, 3)
    mask = np.ones((1, 2, 4), dtype=np.bool_)
    primary = np.zeros_like(mask)
    primary[:, :, 0] = True
    truth = original[:, -1].copy()
    shift = 2
    arrays = {
        "pnp_s_obs_m": cyclic_relabel(original, shift=shift, reverse=True, axis=2),
        "pnp_s_obs_mask": cyclic_relabel(mask, shift=shift, reverse=True, axis=2),
        "pnp_s_primary_mask": cyclic_relabel(primary, shift=shift, reverse=True, axis=2),
        "pnp_s_truth_q0_m": cyclic_relabel(truth, shift=shift, reverse=True, axis=1),
        "pnp_s_switch_step": np.asarray(((0, -1),), dtype=np.int64),
        "clean_s_obs_m": cyclic_relabel(original, shift=shift, reverse=True, axis=2),
        "clean_s_obs_mask": cyclic_relabel(mask, shift=shift, reverse=True, axis=2),
        "clean_s_primary_mask": cyclic_relabel(primary, shift=shift, reverse=True, axis=2),
        "clean_s_switch_step": np.asarray(((0, -1),), dtype=np.int64),
    }
    _undo_reflection_keep_c4_origin(arrays, np.asarray((-1,), dtype=np.int64))
    assert np.array_equal(arrays["pnp_s_obs_m"], np.roll(original, shift, axis=2))
    assert np.array_equal(arrays["pnp_s_truth_q0_m"], np.roll(truth, shift, axis=1))
    assert np.array_equal(arrays["pnp_s_switch_step"], np.asarray(((0, 1),)))
    assert np.array_equal(arrays["clean_s_switch_step"], np.asarray(((0, 1),)))
    assert not np.array_equal(arrays["pnp_s_obs_m"], original)


def test_optimizer_contains_exactly_h_parameters() -> None:
    model = _model()
    optimizer = build_h_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert actual == expected
    assert sum(parameter.numel() for parameter in model.parameters()) <= 150_000


def test_index_loader_preserves_shuffle_order_and_generator_state() -> None:
    class IndexedDictDataset(Dataset):
        def __len__(self) -> int:
            return 29

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {"index": torch.tensor(index, dtype=torch.long)}

    original_generator = torch.Generator().manual_seed(20260727)
    cached_generator = torch.Generator().manual_seed(20260727)
    original = DataLoader(
        IndexedDictDataset(), batch_size=8, shuffle=True,
        generator=original_generator, num_workers=0,
    )
    cached = DataLoader(
        range(29), batch_size=8, shuffle=True,
        generator=cached_generator, num_workers=0,
    )
    for _ in range(2):
        original_order = torch.cat([batch["index"] for batch in original])
        cached_order = torch.cat([batch for batch in cached])
        assert torch.equal(original_order, cached_order)
        assert torch.equal(original_generator.get_state(), cached_generator.get_state())


def test_train_cache_contract_excludes_only_build_time() -> None:
    metadata = {
        "schema_version": TRAIN_CACHE_SCHEMA,
        "content_sha256": "abc",
        "build_elapsed_s": 1.25,
        "validation_cached": False,
    }
    assert _train_cache_contract(metadata) == {
        "schema_version": TRAIN_CACHE_SCHEMA,
        "content_sha256": "abc",
        "validation_cached": False,
    }


def test_formal_h_recovery_round_trip_is_state_only_and_non_overwriting(
    tmp_path,
) -> None:
    model = _model()
    optimizer = build_h_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
    generator = torch.Generator().manual_seed(31)
    arguments = argparse.Namespace(
        formal_oracle=True,
        resume=False,
        output=str(tmp_path),
        seed=31,
    )
    values = {
        "formal_contract": {"schema_version": "contract", "git_commit": "abc"},
        "dataset_manifest_sha256": "dataset",
        "mapper_state_sha256": "mapper",
        "s_state_sha256": "s",
        "f_state_sha256": "f",
    }
    first = _write_recovery_state(
        tmp_path,
        epoch=1,
        update=7,
        elapsed_s=1.25,
        model=model,
        optimizer=optimizer,
        train_generator=generator,
        arguments=arguments,
        **values,
    )
    second = _write_recovery_state(
        tmp_path,
        epoch=1,
        update=7,
        elapsed_s=1.5,
        model=model,
        optimizer=optimizer,
        train_generator=generator,
        arguments=arguments,
        **values,
    )
    assert first != second
    assert first.is_file() and second.is_file()
    loaded_path, payload = _load_recovery_state(tmp_path.resolve())
    assert loaded_path == second.resolve()
    assert payload["completed_epoch"] == 1
    assert payload["update"] == 7
    assert payload["training_arguments"] == {
        "formal_oracle": True,
        "output": str(tmp_path),
        "seed": 31,
    }
    assert payload["validation_accessed"] is False
    assert payload["model_selection_performed"] is False
    assert payload["test_accessed"] is False

    pointer_path = tmp_path / "recovery_latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["sha256"] = "0" * 64
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        _load_recovery_state(tmp_path.resolve())


def test_formal_h_recovery_restores_all_host_rng_and_loader_state() -> None:
    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    generator = torch.Generator().manual_seed(41)
    state = _capture_rng_state(generator)
    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(5),
        torch.randperm(17, generator=generator),
    )
    _restore_rng_state(state, generator)
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(5),
        torch.randperm(17, generator=generator),
    )
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[3], expected[3])


def test_h_resume_cli_is_explicit() -> None:
    action = next(item for item in parser()._actions if item.dest == "resume")
    assert action.default is False


def test_h_run_lock_rejects_concurrent_owner_and_releases_on_exit(tmp_path) -> None:
    path, token, stream = _acquire_run_lock(tmp_path)
    owner = _read_run_lock(path)
    assert owner["pid"] == os.getpid()
    assert owner["released"] is False
    with pytest.raises(RuntimeError, match="live training process"):
        _acquire_run_lock(tmp_path)
    _release_run_lock(path, token, stream)
    assert _read_run_lock(path)["released"] is True

    next_path, next_token, next_stream = _acquire_run_lock(tmp_path)
    try:
        assert next_path == path
        assert _read_run_lock(path)["token"] == next_token
    finally:
        _release_run_lock(next_path, next_token, next_stream)


def test_h_compose_uses_corrected_observations_and_is_c4_invariant() -> None:
    source = _s_output(batch=1)
    h = hypothesis_forward(_model().eval(), source)
    corrected = torch.randn(1, 3, 4, 3)
    obs_mask = torch.zeros(1, 3, 4, dtype=torch.bool)
    primary_mask = torch.zeros_like(obs_mask)
    obs_mask[:, :, 0] = True
    primary_mask[:, :, 0] = True
    step = torch.arange(-6, 7).view(1, 13)
    reference = compose_hypothesis_for_f(
        h, source["primary_index"], corrected, obs_mask, primary_mask, step
    )
    assert torch.count_nonzero(reference["candidate_relation_m"][step.remainder(4) == 0]) == 0
    assert torch.equal(
        reference["selected_history_absolute_m"], corrected[:, :, 0]
    )
    for shift in (1, 2, 3):
        rolled_h = {
            name: torch.roll(value, shift, dims=1)
            if value.ndim >= 2 and value.shape[1] == 4 else value
            for name, value in h.items()
        }
        output = compose_hypothesis_for_f(
            rolled_h, torch.remainder(source["primary_index"] + shift, 4),
            torch.roll(corrected, shift, dims=2),
            torch.roll(obs_mask, shift, dims=2),
            torch.roll(primary_mask, shift, dims=2), step,
        )
        for name in (
            "current_position_m", "candidate_relation_m", "candidate_confidence",
            "candidate_supported", "candidate_support_class",
            "history_position_rel_m", "selected_history_absolute_m",
        ):
            if reference[name].is_floating_point():
                assert torch.allclose(output[name], reference[name], atol=1e-6, rtol=0.0)
            else:
                assert torch.equal(output[name], reference[name])


def test_compose_signature_has_no_raw_or_batch_fallback() -> None:
    names = tuple(inspect.signature(compose_hypothesis_for_f).parameters)
    assert "batch" not in names
    assert not any("raw" in name or name.startswith("pnp_") for name in names)


def test_h_compose_masks_inactive_prefix_without_negative_current_artifact() -> None:
    source = _s_output(batch=1)
    h = hypothesis_forward(_model().eval(), source)
    corrected = torch.randn(1, 3, 4, 3)
    obs_mask = torch.zeros(1, 3, 4, dtype=torch.bool)
    primary_mask = torch.zeros_like(obs_mask)
    event_mask = torch.tensor([[False, True, True]])
    obs_mask[:, 1:, 0] = True
    primary_mask[:, 1:, 0] = True
    step = torch.arange(-6, 7).view(1, 13)
    output = compose_hypothesis_for_f(
        h, source["primary_index"], corrected, obs_mask, primary_mask, step,
        event_mask,
    )
    assert torch.count_nonzero(output["history_position_rel_m"][:, 0]) == 0
    assert torch.count_nonzero(output["selected_history_absolute_m"][:, 0]) == 0


def test_pnp_selector_changes_only_switch_heads_and_keeps_trajectory_bit_exact() -> None:
    torch.manual_seed(17)
    trajectory = AnonymousCandidateFutureExpert(
        "combined", channels=32, dropout=0.0,
        position_scale_m=1.0, history_scale_s=0.32,
        trained_horizon_s=0.5, maximum_absolute_step=2,
        trajectory_rank=4,
    )
    selector = AnonymousCandidateFutureExpert(
        "combined", channels=32, dropout=0.0,
        position_scale_m=1.0, history_scale_s=0.32,
        trained_horizon_s=0.5, maximum_absolute_step=2,
        trajectory_rank=4,
    )
    selector.load_state_dict(trajectory.state_dict())
    system = FrozenTrajectoryDedicatedSelector(trajectory, selector)
    trainable, _ = freeze_pnp_selector(system)
    assert trainable
    assert all(name.startswith(PNP_SELECTOR_TRAINABLE_PREFIXES) for name in trainable)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in selector.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    assert {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    } == {id(parameter) for parameter in selector.parameters() if parameter.requires_grad}
    batch, events, queries = 2, 32, 3
    steps = torch.tensor(((-2, -1, 0, 1, 2),) * batch)
    relation = torch.randn(batch, 5, 3) * 0.1
    relation[:, 2] = 0.0
    inputs = (
        torch.randn(batch, events, 3) * 0.05,
        torch.linspace(-0.31, 0.0, events).repeat(batch, 1),
        torch.full((batch, events), 0.01),
        torch.zeros(batch, events, dtype=torch.long),
        torch.ones(batch, events, dtype=torch.bool),
        torch.randn(batch, 3), relation, steps,
        torch.ones(batch, 5, dtype=torch.bool),
        torch.ones(batch, 5),
        torch.tensor(((0.1, 0.3, 0.5),) * batch),
    )
    system.eval()
    with torch.no_grad():
        reference = system(*inputs)["conditional_delta_m"].clone()
    trajectory_hash = _module_state_sha256(trajectory)
    selector_frozen_hash = _module_state_sha256(
        selector, prefixes=PNP_SELECTOR_TRAINABLE_PREFIXES, invert=True
    )
    set_pnp_selector_train_mode(system)
    optimizer.zero_grad(set_to_none=True)
    prediction = system(*inputs)
    (-prediction["switch_logits"][..., 0].mean()).backward()
    optimizer.step()
    system.eval()
    with torch.no_grad():
        current = system(*inputs)["conditional_delta_m"]
    assert torch.equal(current, reference)
    assert _module_state_sha256(trajectory) == trajectory_hash
    assert _module_state_sha256(
        selector, prefixes=PNP_SELECTOR_TRAINABLE_PREFIXES, invert=True
    ) == selector_frozen_hash


def test_pnp_selector_loss_rebases_frozen_trajectory_to_clean_current() -> None:
    logits = torch.zeros(1, 1, 2, requires_grad=True)
    prediction = {
        "switch_logits": logits,
        # Frozen F deltas are relative to the H current position x=11.
        "conditional_delta_m": torch.tensor(((((1.0, 0.0, 0.0),
                                                 (0.0, 0.0, 0.0)),),)),
        "conditional_position_m": torch.tensor(((((12.0, 0.0, 0.0),
                                                    (11.0, 0.0, 0.0)),),)),
    }
    batch = {
        "current_position_m": torch.tensor(((10.0, 0.0, 0.0),)),
        "target_visible_delta_m": torch.tensor((((2.0, 0.0, 0.0),),)),
        "target_switch_count": torch.zeros(1, 1, dtype=torch.long),
        "target_query_mask": torch.ones(1, 1, dtype=torch.bool),
        "pnp_candidate_step": torch.tensor(((0, 1),)),
        "pnp_candidate_mask": torch.ones(1, 2, dtype=torch.bool),
        "pnp_candidate_confidence": torch.ones(1, 2),
        "pnp_tau_s": torch.tensor(((0.1,),)),
    }
    args = SimpleNamespace(
        switch_weight=1.0, macro_balance_weight=0.5,
        switch_focal_gamma=0.0, distance_cost_weight=1.0,
        distance_cost_scale_m=0.3, distance_cost_cap=2.0,
    )
    rebased = _selector_loss_prediction(prediction, batch)
    assert torch.equal(
        rebased["conditional_delta_m"],
        torch.tensor(((((2.0, 0.0, 0.0), (1.0, 0.0, 0.0)),),)),
    )
    objective, details = _selector_loss(
        rebased, _metric_batch(batch), args
    )
    assert float(details["mean_true_branch_error_m"]) == 0.0
    objective.backward()
    assert logits.grad is not None
