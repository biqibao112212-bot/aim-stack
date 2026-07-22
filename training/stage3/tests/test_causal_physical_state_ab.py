from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys

import pytest
import torch

from training.stage3.causal_physical_state_model import (
    ExplicitStatePhysicalPredictor,
    FixedSlotHistoryEncoder,
    ImplicitQueryPhysicalPredictor,
    trainable_parameter_count,
)
from training.stage3.physical_loss import causal_physical_base_loss
from training.stage3.train_causal_physical_ab import _load_selection, _train_one


GEOMETRY = torch.tensor([
    [0.01, -0.21, -0.061], [0.22, 0.00, -0.075],
    [0.01, 0.21, -0.061], [-0.20, 0.00, -0.075],
], dtype=torch.float32)
MEAN = torch.tensor([4.0, 0.0, -0.07])
STD = torch.tensor([2.0, 1.0, 0.1])


def _inputs(batch: int = 2, time: int = 8) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(17)
    obs = torch.randn(batch, time, 4, 5)
    mask = torch.ones(batch, time, 4, dtype=torch.bool)
    event = torch.ones(batch, time, dtype=torch.bool)
    timestamp = torch.linspace(-0.14, 0.0, time).expand(batch, -1).clone()
    tau = torch.tensor([0.0, 0.1, 0.2, 0.5]).expand(batch, -1).clone()
    return obs, mask, event, timestamp, tau


def _models(channels: int = 16):
    torch.manual_seed(23)
    common = dict(
        geometry=GEOMETRY, position_mean=MEAN, position_std=STD,
        channels=channels, dropout=0.0, history_events=8,
    )
    a = ExplicitStatePhysicalPredictor(**common)
    b = ImplicitQueryPhysicalPredictor(**common)
    b.encoder.load_state_dict(a.encoder.state_dict())
    return a.eval(), b.eval()


def test_no_analytic_state_recovery_exists_in_frozen_model_source() -> None:
    source = inspect.getsource(
        __import__(
            "training.stage3.causal_physical_state_model", fromlist=["unused"]
        )
    )
    for forbidden in ("LeastSquares", "lstsq", "pinv", "polyfit"):
        assert forbidden not in source


def test_active_trainer_has_one_common_loss_and_no_arm_specific_branch() -> None:
    source = inspect.getsource(_train_one)
    assert "causal_physical_base_loss" in source
    assert "causal_physical_history_regularizers" not in source
    assert "if label" not in source


def test_center_is_unbounded_and_yaw_parameterization_is_always_unit() -> None:
    model, _ = _models()
    raw = torch.tensor([[3.0, -4.0, 5.0]])
    center = model._center(raw)
    assert torch.allclose(
        center, model.center_reference + model.center_scale * raw
    )
    phase = model._phase(torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0]))
    assert torch.allclose(
        torch.linalg.vector_norm(phase, dim=-1), torch.ones(5), atol=1e-6
    )


def test_four_slots_are_encoded_independently_before_fixed_order_fusion() -> None:
    encoder = FixedSlotHistoryEncoder(
        input_features=5, channels=16, dropout=0.0, history_events=8
    ).eval()
    obs, mask, event, timestamp, _ = _inputs(batch=1)
    reference = encoder(obs, mask, event, timestamp)
    swapped = encoder(obs[:, :, [1, 0, 2, 3]], mask, event, timestamp)
    assert not torch.allclose(reference, swapped)


def test_masked_nonfinite_slot_history_cannot_pollute_models() -> None:
    obs, mask, event, timestamp, tau = _inputs()
    mask[:, 2, 1] = False
    clean = obs.clone()
    clean[:, 2, 1] = 0.0
    poison = clean.clone()
    poison[0, 2, 1] = torch.tensor([
        float("nan"), float("inf"), -float("inf"), 1e30, -1e30
    ])
    for model in _models():
        expected = model(clean, mask, event, timestamp, tau)["position_mean"]
        actual = model(poison, mask, event, timestamp, tau)["position_mean"]
        assert torch.equal(expected, actual)
        assert torch.isfinite(actual).all()


def test_explicit_arm_uses_one_state_and_exact_constant_twist() -> None:
    model, _ = _models()
    output = model(*_inputs(batch=1))
    tau = _inputs(batch=1)[-1]
    expected_center = (
        output["center0"][:, None]
        + tau.unsqueeze(-1) * output["velocity"][:, None]
    )
    assert output["velocity"].shape == (1, 3)
    assert output["omega"].shape == (1,)
    assert torch.allclose(output["query_center"], expected_center, atol=1e-7)
    assert torch.equal(output["velocity"], torch.zeros_like(output["velocity"]))
    assert torch.equal(output["omega"], torch.zeros_like(output["omega"]))


def test_implicit_arm_has_no_velocity_or_omega_and_tau_reorders() -> None:
    _, model = _models()
    with torch.no_grad():
        model.query_head[-1].weight.normal_(0.0, 0.01)
        model.query_head[-1].bias.normal_(0.0, 0.01)
    obs, mask, event, timestamp, tau = _inputs(batch=1)
    reference = model(obs, mask, event, timestamp, tau)
    assert "velocity" not in reference and "omega" not in reference
    order = torch.tensor([3, 1, 3, 0, 2])
    reordered = model(obs, mask, event, timestamp, tau[:, order])
    assert torch.allclose(
        reordered["position_mean"], reference["position_mean"][:, order],
        atol=1e-6,
    )


def test_both_arms_preserve_exact_fixed_geometry() -> None:
    pair_i, pair_j = torch.triu_indices(4, 4, offset=1)
    expected = torch.linalg.vector_norm(
        GEOMETRY[pair_i] - GEOMETRY[pair_j], dim=-1
    )
    for model in _models():
        position = model(*_inputs())["position_mean"]
        actual = torch.linalg.vector_norm(
            position[:, :, pair_i] - position[:, :, pair_j], dim=-1
        )
        assert torch.allclose(actual, expected.view(1, 1, -1), atol=2e-6)


def test_fixed_slot_loss_exposes_slot_swaps_without_permutation_search() -> None:
    target = GEOMETRY.view(1, 1, 4, 3).expand(1, 4, -1, -1).clone()
    prediction = {"position_mean": target.roll(1, dims=2)}
    tau = torch.tensor([[0.0, 0.1, 0.2, 0.5]])
    active = torch.ones_like(tau, dtype=torch.bool)
    loss, _ = causal_physical_base_loss(prediction, target, tau, active)
    assert loss > 0.01


def test_paired_parameter_counts_are_within_one_percent() -> None:
    a, b = _models(channels=64)
    counts = trainable_parameter_count(a), trainable_parameter_count(b)
    assert abs(counts[0] - counts[1]) / max(counts) < 0.01


def test_selection_is_hash_bound_disjoint_and_test_sealed(tmp_path) -> None:
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    selection = tmp_path / "selection.json"
    payload = {
        "schema_version": "stage3-causal-physical-state-ab-selection-v1",
        "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "purpose": "pilot", "train": ["train-a"],
        "validation": ["validation-a"], "test": [],
    }
    selection.write_text(json.dumps(payload), encoding="utf-8")
    train, validation, record = _load_selection(str(selection), manifest)
    assert train == ["train-a"] and validation == ["validation-a"]
    assert record is not None and record["test"] == []
    payload["test"] = ["forbidden"]
    selection.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="test empty"):
        _load_selection(str(selection), manifest)


def test_capacity_selection_reuses_train_without_opening_test(tmp_path) -> None:
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    selection = tmp_path / "selection.json"
    payload = {
        "schema_version": "stage3-causal-physical-state-ab-selection-v1",
        "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "purpose": "capacity", "validation_source_split": "train",
        "train": ["combined"], "validation": ["combined"], "test": [],
    }
    selection.write_text(json.dumps(payload), encoding="utf-8")
    train, validation, record = _load_selection(str(selection), manifest)
    assert train == validation == ["combined"]
    assert record is not None and record["validation_source_split"] == "train"


def test_cli_accepts_zero_weight_decay() -> None:
    result = subprocess.run(
        [
            sys.executable, "-m", "training.stage3.train_causal_physical_ab",
            "--dataset", "missing", "--output", "missing",
            "--weight-decay", "0",
        ],
        capture_output=True, text=True,
    )
    assert "arguments must be positive" not in result.stderr
    assert "weight decay cannot be negative" not in result.stderr
