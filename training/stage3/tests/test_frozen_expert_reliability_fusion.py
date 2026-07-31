from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import pytest
import torch
from torch import nn

from training.stage3.frozen_expert_reliability_fusion import (
    MonotoneReliabilityFusion,
    deterministic_cross_session_derangement,
    oracle_projection_coefficient,
    vector_huber_loss,
)
from training.stage3.screen_frozen_expert_reliability_oracle_omega import (
    RUN_SCHEMA,
    _aligned_donor_q0,
    _build_map,
    _evaluate_head,
    validate_reliability_a0_artifacts,
)
from training.stage3.observable_future_pnp_ab import sha256_file, state_dict_sha256
from training.stage3.train_anonymous_vehicle_motion import _json_sha256


def _fitted_model() -> MonotoneReliabilityFusion:
    torch.manual_seed(4)
    model = MonotoneReliabilityFusion(width=8)
    model.fit_normalizer(torch.randn(64, 13))
    return model


@torch.inference_mode()
def test_oracle_projection_is_exact_clamped_segment_projection() -> None:
    history = torch.zeros(4, 3)
    q0 = torch.tensor([[2.0, 0.0, 0.0]]).expand(4, -1).clone()
    truth = torch.tensor([
        [0.5, 1.0, 0.0], [3.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0], [1.5, -2.0, 0.0],
    ])
    support = torch.ones(4, dtype=torch.bool)
    result = oracle_projection_coefficient(
        truth, q0, history, support, support,
    )
    torch.testing.assert_close(
        result["weight"], torch.tensor([0.25, 1.0, 0.0, 0.75]),
    )
    assert bool(result["oracle_valid"].all())


@torch.inference_mode()
def test_oracle_projection_handles_degenerate_and_single_expert_support() -> None:
    history = torch.zeros(4, 3)
    q0 = history.clone()
    truth = torch.ones(4, 3)
    q_support = torch.tensor([True, True, False, False])
    h_support = torch.tensor([True, False, True, False])
    result = oracle_projection_coefficient(
        truth, q0, history, q_support, h_support,
    )
    torch.testing.assert_close(
        result["weight"], torch.tensor([0.5, 1.0, 0.0, 0.0]),
    )
    assert not bool(result["oracle_valid"].any())


def test_truth_and_oracle_are_loss_only() -> None:
    model = _fitted_model()
    feature = torch.randn(5, 13)
    q0 = torch.randn(5, 3)
    history = torch.randn(5, 3)
    support = torch.ones(5, dtype=torch.bool)
    reference = model(feature, q0, history, support, support)
    first = oracle_projection_coefficient(
        torch.zeros(5, 3), q0, history, support, support,
    )
    second = oracle_projection_coefficient(
        torch.ones(5, 3), q0, history, support, support,
    )
    actual = model(feature, q0, history, support, support)
    for name in reference:
        assert torch.equal(reference[name], actual[name]), name
    assert not torch.equal(first["weight"], second["weight"])
    assert not first["weight"].requires_grad


def test_incompatibility_gate_is_monotone_nonincreasing() -> None:
    model = _fitted_model().double()
    base = torch.zeros(9, 13, dtype=torch.double)
    base[:, 3] = torch.linspace(-2.0, 2.0, 9)
    base.requires_grad_(True)
    logit = model.reliability_logit(base)
    weight = torch.sigmoid(logit)
    assert bool(torch.all(weight[1:] <= weight[:-1]))
    gradient = torch.autograd.grad(weight.sum(), base)[0][:, 3]
    assert bool(torch.all(gradient < 0))


@torch.inference_mode()
def test_fusion_is_exact_continuous_convex_combination_and_support_override() -> None:
    model = _fitted_model()
    feature = torch.randn(4, 13)
    q0 = torch.randn(4, 3)
    history = torch.randn(4, 3)
    q_support = torch.tensor([True, True, False, False])
    h_support = torch.tensor([True, False, True, False])
    result = model(feature, q0, history, q_support, h_support)
    weight = result["q0_weight"]
    assert bool(torch.all((weight >= 0) & (weight <= 1)))
    torch.testing.assert_close(weight[1:], torch.tensor([1.0, 0.0, 0.0]))
    expected = weight[:, None] * q0 + (1.0 - weight[:, None]) * history
    expected[-1].zero_()
    torch.testing.assert_close(result["velocity_mps"], expected)
    assert result["state_supported"].tolist() == [True, True, True, False]


@torch.inference_mode()
def test_both_invalid_profiles_use_explicit_common_fallback() -> None:
    model = _fitted_model()
    feature = torch.randn(3, 13)
    q0 = torch.randn(3, 3)
    history = torch.randn(3, 3)
    profile_support = torch.zeros(3, dtype=torch.bool)
    fallback = torch.randn(3, 3)
    fallback_support = torch.tensor([True, False, True])
    result = model(
        feature, q0, history, profile_support, profile_support,
        fallback, fallback_support,
    )
    torch.testing.assert_close(result["velocity_mps"][[0, 2]], fallback[[0, 2]])
    assert result["fallback_used"].tolist() == [True, False, True]
    assert result["learned_gate_valid"].tolist() == [False, False, False]
    assert result["state_supported"].tolist() == [True, False, True]


@torch.inference_mode()
def test_fusion_has_exact_common_ramp_and_scalar_weight() -> None:
    model = _fitted_model()
    feature = torch.randn(6, 13)
    q0 = torch.randn(6, 3)
    history = torch.randn(6, 3)
    support = torch.ones(6, dtype=torch.bool)
    reference = model(feature, q0, history, support, support)
    ramp = torch.randn(6, 3)
    changed = model(feature, q0 + ramp, history + ramp, support, support)
    torch.testing.assert_close(changed["q0_weight"], reference["q0_weight"])
    torch.testing.assert_close(
        changed["velocity_mps"], reference["velocity_mps"] + ramp,
    )


def test_optimizer_updates_only_reliability_parameters() -> None:
    expert = nn.Linear(3, 3).eval().requires_grad_(False)
    before = {name: value.detach().clone() for name, value in expert.state_dict().items()}
    model = _fitted_model().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    feature = torch.randn(12, 13)
    q0 = expert(torch.randn(12, 3)).detach()
    history = expert(torch.randn(12, 3)).detach()
    support = torch.ones(12, dtype=torch.bool)
    output = model(feature, q0, history, support, support)
    loss = vector_huber_loss(output["velocity_mps"], torch.randn(12, 3))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    changed_before = [value.detach().clone() for value in model.parameters()]
    optimizer.step()
    assert all(parameter.grad is None for parameter in expert.parameters())
    assert all(torch.equal(before[name], value) for name, value in expert.state_dict().items())
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(changed_before, model.parameters())
    )


def test_cross_session_derangement_is_deterministic_bijective_and_stratified() -> None:
    keys = [f"key-{index}" for index in range(8)]
    sessions = ["a", "b", "c", "d"] * 2
    strata = [0] * 4 + [1] * 4
    first = deterministic_cross_session_derangement(
        keys, sessions, domain="train/hard", strata=strata,
    )
    second = deterministic_cross_session_derangement(
        keys, sessions, domain="train/hard", strata=strata,
    )
    np.testing.assert_array_equal(first["donor_index"], second["donor_index"])
    assert first["pair_sha256"] == second["pair_sha256"]
    donor = first["donor_index"]
    assert sorted(donor.tolist()) == list(range(8))
    assert all(donor[index] != index for index in range(8))
    assert all(sessions[donor[index]] != sessions[index] for index in range(8))
    assert all(strata[donor[index]] == strata[index] for index in range(8))


def test_cross_session_derangement_fails_closed_for_singletons() -> None:
    result = deterministic_cross_session_derangement(
        ["a", "b", "c"], ["s1", "s2", "s3"],
        domain="validation/hard", strata=[0, 1, 1],
    )
    assert result["donor_index"][0] == -1
    assert not bool(result["valid"][0])
    assert result["coverage"] == pytest.approx(2 / 3)


def test_donor_q0_is_c4_aligned_to_recipient_primary() -> None:
    primary = torch.tensor([0, 1, 2, 3])
    primary_mask = torch.nn.functional.one_hot(primary, num_classes=4).bool()
    prepared = {
        "history_event_mask": torch.ones(4, 1, dtype=torch.bool),
        "history_time_s": torch.zeros(4, 1),
        "history_primary_mask": primary_mask[:, None],
        "q0_relation_m": torch.stack([
            torch.roll(torch.arange(4)[:, None].expand(-1, 3).float(), i, 0)
            for i in range(4)
        ]),
        "q0_supported": torch.stack([
            torch.roll(torch.tensor([True, False, True, False]), i, 0)
            for i in range(4)
        ]),
    }
    recipients = torch.tensor([0, 1, 2, 3])
    donors = torch.tensor([1, 2, 3, 0])
    relation, supported, shift = _aligned_donor_q0(
        prepared, recipients, donors,
    )
    row = torch.arange(4)
    torch.testing.assert_close(relation[row, primary, 0], torch.zeros(4))
    assert torch.equal(supported[row, primary], torch.ones(4, dtype=torch.bool))
    assert torch.equal(shift, torch.tensor([3, 3, 3, 3]))

    relabeled = {name: value.clone() for name, value in prepared.items()}
    relabeled["q0_relation_m"][1] = torch.roll(
        relabeled["q0_relation_m"][1], 2, dims=0,
    )
    relabeled["q0_supported"][1] = torch.roll(
        relabeled["q0_supported"][1], 2, dims=0,
    )
    relabeled["history_primary_mask"][1] = torch.roll(
        relabeled["history_primary_mask"][1], 2, dims=-1,
    )
    changed, changed_supported, _ = _aligned_donor_q0(
        relabeled, recipients[:1], donors[:1],
    )
    torch.testing.assert_close(changed, relation[:1])
    assert torch.equal(changed_supported, supported[:1])


def test_build_map_handles_noncontiguous_indices_and_relaxes_only_declared_strata() -> None:
    keys = [f"key-{index}" for index in range(12)]
    sessions = [f"s{index % 4}" for index in range(12)]
    indices = np.asarray([1, 2, 4, 7, 8, 11], dtype=np.int64)
    exact = [(0, value) for value in [0, 0, 1, 1, 2, 3]]
    fallback = [0] * 12
    recipients, donors, manifest = _build_map(
        indices, keys, sessions, domain="test/noncontiguous",
        strata=[(-1, -1) if index not in indices else exact[list(indices).index(index)] for index in range(12)],
        fallback_strata=fallback,
    )
    assert set(recipients.tolist()) == set(donors.tolist())
    assert set(recipients.tolist()).issubset(set(indices.tolist()))
    assert all(r != d for r, d in zip(recipients, donors))
    assert all(sessions[r] != sessions[d] for r, d in zip(recipients, donors))
    assert manifest["coverage"] == 1.0
    assert manifest["relaxed_valid_count"] > 0


@torch.inference_mode()
def test_counterfactual_metrics_use_profile_common_support_and_motion_groups() -> None:
    model = _fitted_model().eval()
    count = 12
    history = torch.zeros(count, 3)
    q0 = torch.zeros(count, 3)
    q0[:, 0] = 2.0
    truth = torch.zeros(count, 3)
    truth[1::2, 0] = 2.0
    q_support = torch.ones(count, dtype=torch.bool)
    h_support = torch.ones(count, dtype=torch.bool)
    h_support[-2] = False
    q_support[-1] = False
    h_support[-1] = False
    fallback_support = torch.zeros(count, dtype=torch.bool)
    fallback_support[-1] = True
    oracle = oracle_projection_coefficient(
        truth, q0, history, q_support, h_support,
    )
    arm = {
        "feature": torch.randn(count, 13),
        "q0_velocity": q0, "history_velocity": history,
        "old_velocity": 0.5 * (q0 + history),
        "q0_supported": q_support, "history_supported": h_support,
        "fallback_velocity": torch.full((count, 3), 7.0),
        "fallback_supported": fallback_support,
        "target_velocity": truth,
        "motion_class": torch.tensor([2] * 6 + [3] * 6),
        "alignment_shift": torch.zeros(count, dtype=torch.int64),
        "recipient_global": torch.arange(count),
        **oracle,
    }
    metrics = _evaluate_head(
        model, {"intact": arm, "global": arm, "hard": arm},
        torch.device("cpu"),
    )
    for name in ("intact", "global", "hard"):
        assert metrics["arms"][name]["full_count"] == 12
        assert metrics["arms"][name]["common_count"] == 10
        assert metrics["arms"][name]["groups"]["overall"][
            "blind_error_mps"
        ]["count"] == 10
        assert metrics["arms"][name]["groups"]["rotation"]["common_count"] == 6
        assert metrics["arms"][name]["groups"]["combined"]["common_count"] == 4
    assert metrics["arms"]["global"]["groups"]["overall"]["paired_count"] == 10


def _write_artifact_fixture(root: Path) -> None:
    root.mkdir()
    ledger_path = root.parent / "validation-ledger.json"
    checkpoint_path = root / "reliability-head.pt"
    contract = {"schema_version": RUN_SCHEMA, "seed": 1}
    contract_sha = _json_sha256(contract)
    model = _fitted_model().state_dict()
    parent_sha = "a" * 64
    torch.save({
        "schema_version": RUN_SCHEMA,
        "diagnostic_only": True,
        "truth_omega_forward_input": True,
        "experiment_contract": contract,
        "experiment_contract_sha256": contract_sha,
        "parent_checkpoint_sha256": parent_sha,
        "model": model,
    }, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    scope_sha = "b" * 64
    result = {
        "schema_version": RUN_SCHEMA, "status": "passed",
        "diagnostic_only": True, "truth_omega_forward_input": True,
        "formal_v15": False, "experiment_contract": contract,
        "experiment_contract_sha256": contract_sha,
        "reliability_checkpoint": str(checkpoint_path.resolve()),
        "reliability_checkpoint_sha256": checkpoint_sha,
        "reliability_state_dict_sha256": state_dict_sha256(model),
        "validation_scope_sha256": scope_sha,
        "validation_ledger": str(ledger_path.resolve()),
        "parent": {"sha256": parent_sha},
    }
    result_path = root / "screen_result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    result_sha = sha256_file(result_path)
    ledger_path.write_text(json.dumps({
        "schema_version": RUN_SCHEMA, "status": "consumed",
        "experiment_contract_sha256": contract_sha,
        "validation_scope_sha256": scope_sha,
        "screen_result_sha256": result_sha,
        "reliability_checkpoint_sha256": checkpoint_sha,
        "parent_checkpoint_sha256": parent_sha,
    }), encoding="utf-8")
    (root / "run_state.json").write_text(json.dumps({
        "schema_version": RUN_SCHEMA, "status": "passed",
        "experiment_contract_sha256": contract_sha,
        "screen_result_sha256": result_sha,
        "reliability_checkpoint_sha256": checkpoint_sha,
    }), encoding="utf-8")


def test_a0_artifact_loader_accepts_bound_fixture(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_artifact_fixture(root)
    payload, result = validate_reliability_a0_artifacts(root)
    assert payload["schema_version"] == RUN_SCHEMA
    assert result["status"] == "passed"


@pytest.mark.parametrize("tamper", ["checkpoint", "result", "ledger", "run_state"])
def test_a0_artifact_loader_rejects_tampering(tmp_path: Path, tamper: str) -> None:
    root = tmp_path / "run"
    _write_artifact_fixture(root)
    result = json.loads((root / "screen_result.json").read_text(encoding="utf-8"))
    if tamper == "checkpoint":
        payload = torch.load(
            root / "reliability-head.pt", map_location="cpu", weights_only=False,
        )
        payload["model"]["feature_mean"][0] += 1.0
        torch.save(payload, root / "reliability-head.pt")
    elif tamper == "result":
        result["experiment_contract"]["seed"] = 2
        (root / "screen_result.json").write_text(
            json.dumps(result), encoding="utf-8",
        )
    elif tamper == "ledger":
        ledger = Path(result["validation_ledger"])
        value = json.loads(ledger.read_text(encoding="utf-8"))
        value["status"] = "claimed"
        ledger.write_text(json.dumps(value), encoding="utf-8")
    else:
        state_path = root / "run_state.json"
        value = json.loads(state_path.read_text(encoding="utf-8"))
        value["status"] = "running"
        state_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_reliability_a0_artifacts(root)
