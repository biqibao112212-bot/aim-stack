from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys

import pytest
import torch

from training.stage3 import build_causal_physical_dataset as causal_builder
from training.stage3.causal_physical_state_model import (
    ExplicitStatePhysicalPredictor,
    FixedSlotHistoryEncoder,
    ImplicitQueryPhysicalPredictor,
    trainable_parameter_count,
)
from training.stage3.physical_loss import (
    causal_physical_base_loss,
    causal_physical_state_loss,
)
from training.stage3.train_causal_physical_ab import (
    _audit_dataset_contract,
    _load_selection,
    _train_one,
    _validate_history_contract,
)


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
    assert "causal_physical_state_loss" in source
    assert "causal_physical_history_regularizers" not in source
    assert "if label" not in source


def test_official_trainer_fails_closed_on_dirty_provenance() -> None:
    source = inspect.getsource(
        __import__(
            "training.stage3.train_causal_physical_ab", fromlist=["train"]
        ).train
    )
    assert "allow_dirty_worktree" in source
    assert "requires a clean worktree" in source


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
    assert torch.isfinite(output["velocity"]).all()
    assert torch.isfinite(output["omega"]).all()
    assert torch.max(torch.abs(output["velocity"])) < model.maximum_speed_mps
    assert torch.max(torch.abs(output["omega"])) < model.maximum_yaw_rate_rad_s


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


def test_small_random_head_initialization_reaches_encoder_on_first_update() -> None:
    model, _ = _models()
    model.train()
    obs, mask, event, timestamp, tau = _inputs(batch=2)
    output = model(obs, mask, event, timestamp, tau)
    target = output["position_mean"].detach().clone()
    # A nonzero physical state target must propagate through the initially
    # small, but nonzero, final head into the history encoder immediately.
    target[:, 1:, :, 0] += tau[:, 1:, None]
    active = torch.ones_like(tau, dtype=torch.bool)
    loss, _ = causal_physical_state_loss(
        output, target, tau, active, model.decoder.geometry,
    )
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.encoder.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert max(float(gradient.abs().max()) for gradient in gradients) > 0.0


def test_state_loss_is_zero_for_exact_constant_twist() -> None:
    model, _ = _models()
    output = model(*_inputs())
    target = output["position_mean"].detach().clone()
    tau = _inputs()[-1]
    active = torch.ones_like(tau, dtype=torch.bool)
    loss, parts = causal_physical_state_loss(
        output, target, tau, active, model.decoder.geometry,
    )
    assert float(loss) < 1e-9
    assert float(parts["active_fraction"]) == 1.0


def test_state_loss_rejects_querywise_nonconstant_motion() -> None:
    model, _ = _models()
    output = model(*_inputs())
    target = output["position_mean"].detach().clone()
    nonlinear = {"position_mean": target.clone()}
    nonlinear["position_mean"][:, 3, :, 0] += 0.02
    tau = _inputs()[-1]
    active = torch.ones_like(tau, dtype=torch.bool)
    loss, parts = causal_physical_state_loss(
        nonlinear, target, tau, active, model.decoder.geometry,
    )
    assert float(loss) > 0.0
    assert float(parts["center_consistency"]) > 0.0


def test_history_contract_covers_every_model_input_event() -> None:
    manifest = {
        "identity_contract": {
            "constant_motion_fit_events": 32,
            "minimum_events_before_prediction": 32,
        }
    }
    _validate_history_contract(manifest, 32)
    manifest["identity_contract"]["constant_motion_fit_events"] = 4
    with pytest.raises(ValueError, match="every consumed history event"):
        _validate_history_contract(manifest, 32)


def test_dataset_builder_retains_zero_sample_sessions_as_manifest_evidence() -> None:
    shard_source = inspect.getsource(causal_builder._build_shard)
    build_source = inspect.getsource(causal_builder.build)
    assert '"sample_count": 0' in shard_source
    assert '"zero_sample_sessions"' in build_source
    assert "for item in admitted" in build_source


def _qualification_dataset(
    *, corrupt_history: bool = False, eligible: bool = True,
    aliased_interval: bool = False,
):
    model, _ = _models()
    history_time = torch.linspace(-1.75 if aliased_interval else -0.35, 0.0, 8)
    tau = torch.tensor([0.0, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9])

    def decode(time: torch.Tensor) -> torch.Tensor:
        center = torch.stack((1.0 + 0.4 * time, 2.0 - 0.2 * time, 0.1 * time), -1)
        angle = 0.7 * time
        phase = torch.stack((torch.cos(angle), torch.sin(angle)), -1)
        return model.decoder(center[None], phase[None])[0]

    history = decode(history_time)
    if corrupt_history:
        history[4, :, 0] += 0.01
    sample = {
        "event_mask": torch.ones(8, dtype=torch.bool),
        "obs_mask": torch.ones(8, 4, dtype=torch.bool),
        "event_time_s": history_time,
        "history_position_m": history,
        "future_position": decode(tau),
        "tau": tau,
        "rule_query": torch.full((8,), eligible, dtype=torch.bool),
        "motion_class": torch.tensor(3, dtype=torch.long),
    }

    class QualificationDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int):
            return sample

    return QualificationDataset(), model.decoder.geometry


def test_row_level_dataset_qualification_checks_motion_and_coverage() -> None:
    dataset, geometry = _qualification_dataset()
    report = _audit_dataset_contract(dataset, geometry, 8, 0.85)
    assert report["sample_count"] == report["eligible_count"] == 2
    assert report["coverage"] == 1.0

    corrupt, geometry = _qualification_dataset(corrupt_history=True)
    with pytest.raises(ValueError, match="constant-twist residual"):
        _audit_dataset_contract(corrupt, geometry, 8, 0.85)

    uncovered, geometry = _qualification_dataset(eligible=False)
    with pytest.raises(ValueError, match="coverage"):
        _audit_dataset_contract(uncovered, geometry, 8, 0.85)

    aliased, geometry = _qualification_dataset(aliased_interval=True)
    with pytest.raises(ValueError, match="alias"):
        _audit_dataset_contract(aliased, geometry, 8, 0.85)

    with pytest.raises(ValueError, match="every registered motion class"):
        _audit_dataset_contract(dataset, geometry, 8, 0.85, {0, 1, 2, 3})


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
