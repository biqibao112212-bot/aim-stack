from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from training.stage3.endpoint_token_reliability_fusion import (
    EndpointTokenReliabilityFusion,
)
from training.stage3.observable_future_pnp_ab import (
    sha256_file,
    state_dict_sha256,
)
from training.stage3.screen_endpoint_token_reliability_oracle_omega import (
    A0_RUN_SCHEMA,
    FIXED_FOLDS,
    HARD_MAP_POLICY,
    RUN_SCHEMA,
    _a1_source_paths,
    _all_boolean_gates_pass,
    _attach_motion_exact_coverage,
    _build_corruption_maps,
    _build_maximal_balanced_exact_map,
    _cross_fold_gates,
    _cv_authorized,
    _fold_gates,
    _finalize_preflight_failure,
    _load_train_dataset,
    _pair_common_mask,
    _validate_corruption_manifest,
    validate_a0_rejection,
    validate_a1_p0_artifacts,
)
from training.stage3.train_anonymous_vehicle_motion import _json_sha256


def _write_a0_fixture(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    parent = root.parent / "stage3-training" / "v14" / "checkpoints" / "parent.pt"
    parent.parent.mkdir(parents=True, exist_ok=True)
    parent_model = {"weight": torch.tensor([3.0])}
    torch.save({"model": parent_model}, parent)
    a0_sources = {
        "runner": _a1_source_paths()["a0_runner"],
        "fusion": _a1_source_paths()["a0_fusion"],
        "split_truth": _a1_source_paths()["split_truth"],
    }
    result = {
        "schema_version": A0_RUN_SCHEMA,
        "status": "cv_failed",
        "diagnostic_only": True,
        "truth_omega_forward_input": True,
        "formal_v15": False,
        "validation_accessed": False,
        "test_accessed": False,
        "future_modules_loaded": False,
        "authorized_a1_endpoint_probe": True,
        "authorized_formal_two_stage": False,
        "experiment_contract": {
            "source_sha256": {
                name: sha256_file(path) for name, path in a0_sources.items()
            },
        },
        "validation_scope_sha256": "a" * 64,
        "parent": {
            "checkpoint": str(parent.resolve()),
            "sha256": sha256_file(parent),
            "model_state_dict_sha256": state_dict_sha256(parent_model),
        },
        "dataset_manifest_sha256": "c" * 64,
        "truth_manifest_sha256": "d" * 64,
        "cv": [{"passed": False}, {"passed": False}],
    }
    result_path = root / "screen_result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    (root / "run_state.json").write_text(json.dumps({
        "schema_version": A0_RUN_SCHEMA,
        "status": "cv_failed",
        "validation_claimed": False,
        "validation_consumed": False,
        "test_accessed": False,
        "screen_result_sha256": sha256_file(result_path),
    }), encoding="utf-8")
    return result_path, parent


def _rewrite_a0_result_with_bound_hash(path: Path, result: dict[str, Any]) -> None:
    path.write_text(json.dumps(result), encoding="utf-8")
    state_path = path.parent / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["screen_result_sha256"] = sha256_file(path)
    state_path.write_text(json.dumps(state), encoding="utf-8")


def test_a0_rejection_loader_accepts_sealed_train_only_fixture(tmp_path: Path) -> None:
    result_path, _ = _write_a0_fixture(tmp_path / "a0")
    assert validate_a0_rejection(result_path)["status"] == "cv_failed"


@pytest.mark.parametrize(
    "tamper",
    [
        "status", "authorization", "validation", "test", "future",
        "folds", "source", "run_hash", "run_claim", "ledger",
    ],
)
def test_a0_rejection_loader_rejects_targeted_tamper(
    tmp_path: Path, tamper: str,
) -> None:
    result_path, parent = _write_a0_fixture(tmp_path / "a0")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if tamper == "status":
        result["status"] = "passed"
        _rewrite_a0_result_with_bound_hash(result_path, result)
    elif tamper == "authorization":
        result["authorized_a1_endpoint_probe"] = False
        _rewrite_a0_result_with_bound_hash(result_path, result)
    elif tamper in {"validation", "test", "future"}:
        result[{
            "validation": "validation_accessed",
            "test": "test_accessed",
            "future": "future_modules_loaded",
        }[tamper]] = True
        _rewrite_a0_result_with_bound_hash(result_path, result)
    elif tamper == "folds":
        result["cv"] = result["cv"][:1]
        _rewrite_a0_result_with_bound_hash(result_path, result)
    elif tamper == "source":
        result["experiment_contract"]["source_sha256"]["runner"] = "e" * 64
        _rewrite_a0_result_with_bound_hash(result_path, result)
    elif tamper == "run_hash":
        state_path = result_path.parent / "run_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["screen_result_sha256"] = "f" * 64
        state_path.write_text(json.dumps(state), encoding="utf-8")
    elif tamper == "run_claim":
        state_path = result_path.parent / "run_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["validation_claimed"] = True
        state_path.write_text(json.dumps(state), encoding="utf-8")
    else:
        ledger = (
            parent.parent.parent.parent / "_v15-validation-access-ledger"
            / f"{result['validation_scope_sha256']}.json"
        )
        ledger.parent.mkdir(parents=True)
        ledger.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_a0_rejection(result_path)


def test_train_dataset_boundary_constructs_only_train_split(tmp_path: Path) -> None:
    calls: list[tuple[Path, str, int]] = []

    def loader(path: Path, split: str, *, sample_limit: int) -> object:
        calls.append((path, split, sample_limit))
        if split != "train":
            raise AssertionError("A1 attempted a sealed split")
        return object()

    value = _load_train_dataset(tmp_path, loader=loader)
    assert value is not None
    assert calls == [(tmp_path, "train", 0)]


def _map_manifest(*, sample: int, valid: int) -> dict[str, Any]:
    excluded = sample - valid
    return {
        "sample_count": sample,
        "valid_count": valid,
        "exact_valid_count": valid,
        "coverage": valid / sample,
        "require_cross_session": True,
        "fixed_point_count": 0,
        "cross_session_count": valid,
        "relaxed_valid_count": 0,
        "balancing_policy": HARD_MAP_POLICY,
        "maximal_exact_selected_count": valid,
        "unavoidable_exclusion_count": excluded,
        "unavoidable_exclusion_groups": (
            [] if excluded == 0 else [{"excluded_count": excluded}]
        ),
        "motion_exact_coverage": {
            "overall": {"sample_count": sample, "valid_count": valid},
            "rotation": {"sample_count": 5, "valid_count": min(valid, 4)},
            "combined": {"sample_count": 5, "valid_count": min(valid, 4)},
        },
    }


def test_corruption_manifest_requires_exact_cross_session_maps() -> None:
    _validate_corruption_manifest(_map_manifest(sample=10, valid=10), kind="global")
    _validate_corruption_manifest(_map_manifest(sample=10, valid=8), kind="hard")
    for field, value in (
        ("fixed_point_count", 1), ("cross_session_count", 7),
        ("relaxed_valid_count", 1), ("exact_valid_count", 7),
        ("require_cross_session", False),
    ):
        manifest = _map_manifest(sample=10, valid=8)
        manifest[field] = value
        with pytest.raises(RuntimeError):
            _validate_corruption_manifest(manifest, kind="hard")
    with pytest.raises(RuntimeError):
        _validate_corruption_manifest(_map_manifest(sample=10, valid=7), kind="hard")
    biased = _map_manifest(sample=10, valid=8)
    biased["motion_exact_coverage"]["rotation"]["valid_count"] = 3
    with pytest.raises(RuntimeError):
        _validate_corruption_manifest(biased, kind="hard")
    for field, value in (
        ("balancing_policy", "relaxed"),
        ("maximal_exact_selected_count", 7),
        ("unavoidable_exclusion_count", 1),
        ("unavoidable_exclusion_groups", [{"excluded_count": 1}]),
    ):
        tampered = _map_manifest(sample=10, valid=8)
        tampered[field] = value
        with pytest.raises(RuntimeError, match="balancing policy"):
            _validate_corruption_manifest(tampered, kind="hard")


def test_hard_map_uses_deterministic_maximal_balanced_exact_subset() -> None:
    indices = np.arange(4, dtype=np.int64)
    keys = [f"sample-{index}" for index in indices]
    sessions = ["a", "a", "a", "b"]
    strata = [(2, 4)] * 4
    first = _build_maximal_balanced_exact_map(
        indices, keys, sessions, strata, domain="test/maximal",
    )
    second = _build_maximal_balanced_exact_map(
        indices, keys, sessions, strata, domain="test/maximal",
    )
    recipients, donors, manifest = first
    assert recipients.size == donors.size == 2
    assert sorted(recipients.tolist()) == sorted(donors.tolist())
    assert all(
        sessions[int(recipient)] != sessions[int(donor)]
        for recipient, donor in zip(recipients, donors, strict=True)
    )
    assert not bool(np.any(recipients == donors))
    assert manifest["balancing_policy"] == HARD_MAP_POLICY
    assert manifest["unavoidable_exclusion_count"] == 2
    assert manifest["relaxed_valid_count"] == 0
    assert manifest["recipient_index_sha256"] == second[2][
        "recipient_index_sha256"
    ]
    assert manifest["donor_index_sha256"] == second[2]["donor_index_sha256"]


def test_real_balanced_map_keeps_full_motion_family_denominator() -> None:
    indices = np.arange(20, dtype=np.int64)
    keys = [f"sample-{index}" for index in indices]
    motion = np.asarray([2] * 10 + [3] * 10, dtype=np.int64)
    strata = (
        [(2, 4)] * 8 + [(2, 3)] * 2
        + [(3, 4)] * 8 + [(3, 3)] * 2
    )
    sessions = (
        ["ra"] * 4 + ["rb"] * 4 + ["rx"] * 2
        + ["ca"] * 4 + ["cb"] * 4 + ["cx"] * 2
    )
    recipients, _, manifest = _build_maximal_balanced_exact_map(
        indices, keys, sessions, strata, domain="test/family/pass",
    )
    _attach_motion_exact_coverage(manifest, recipients, indices, motion)
    assert manifest["motion_exact_coverage"]["rotation"] == {
        "sample_count": 10, "valid_count": 8, "coverage": 0.8,
    }
    assert manifest["motion_exact_coverage"]["combined"] == {
        "sample_count": 10, "valid_count": 8, "coverage": 0.8,
    }
    _validate_corruption_manifest(manifest, kind="hard")
    map_indices, map_manifests = _build_corruption_maps(
        indices, keys, sessions, strata, motion, domain="test/fold/integration",
    )
    assert map_manifests["hard"]["motion_exact_coverage"]["rotation"][
        "valid_count"
    ] == 8
    population = set(indices.tolist())
    for recipients, donors in map_indices.values():
        assert set(recipients.tolist()) <= population
        assert set(donors.tolist()) <= population

    failing_sessions = (
        ["ra"] * 5 + ["rb"] * 3 + ["rx"] * 2
        + ["ca"] * 5 + ["cb"] * 5
    )
    failing_strata = [(2, 4)] * 8 + [(2, 3)] * 2 + [(3, 4)] * 10
    recipients, _, manifest = _build_maximal_balanced_exact_map(
        indices, keys, failing_sessions, failing_strata,
        domain="test/family/fail",
    )
    _attach_motion_exact_coverage(manifest, recipients, indices, motion)
    assert manifest["motion_exact_coverage"]["overall"]["valid_count"] == 16
    assert manifest["motion_exact_coverage"]["rotation"]["valid_count"] == 6
    with pytest.raises(RuntimeError, match="motion-family"):
        _validate_corruption_manifest(manifest, kind="hard")


def test_preflight_failure_is_sealed_and_preserves_failed_domain(
    tmp_path: Path,
) -> None:
    hard = _map_manifest(sample=10, valid=7)
    hard["motion_exact_coverage"]["rotation"] = {
        "sample_count": 10, "valid_count": 7, "coverage": 0.7,
    }
    details = {
        "schema_version": RUN_SCHEMA,
        "diagnostic_only": True,
        "truth_omega_feature_construction": True,
        "formal_v15": False,
        "train_only": True,
        "experiment_contract": {},
        "experiment_contract_sha256": "a" * 64,
        "metadata_preflight": [{
            "fold": 0,
            "status": "failed",
            "failed_domain": "heldout",
            "heldout_maps": {"hard": hard},
        }],
        "preflight_error": "coverage",
    }
    result = _finalize_preflight_failure(
        tmp_path, contract_sha="a" * 64, details=details,
    )
    state = json.loads((tmp_path / "run_state.json").read_text(encoding="utf-8"))
    stored = json.loads(
        (tmp_path / "screen_result.json").read_text(encoding="utf-8")
    )
    assert result == stored
    assert state["status"] == result["status"] == "preflight_failed"
    assert result["authorized_a1_counterfactual_probe"] is False
    assert result["authorized_formal_two_stage"] is False
    assert result["validation_accessed"] is False
    assert result["test_accessed"] is False
    assert result["future_modules_loaded"] is False
    failed = result["metadata_preflight"][0]
    assert failed["failed_domain"] == "heldout"
    assert failed["heldout_maps"]["hard"]["motion_exact_coverage"][
        "rotation"
    ]["valid_count"] == 7
    assert failed["heldout_maps"]["hard"][
        "unavoidable_exclusion_groups"
    ]


def test_pair_loss_mask_requires_both_arms_common_endpoint_domain() -> None:
    intact_batch = {"oracle_valid": torch.tensor([True, True, True, True])}
    intact_fused = {
        "learned_gate_valid": torch.tensor([True, False, True, True]),
    }
    corrupt_batch = {
        "oracle_valid": torch.tensor([True, True, False, True]),
        "q0_supported": torch.tensor([True, True, True, False]),
        "history_supported": torch.tensor([True, True, True, True]),
        "role_mask": torch.tensor([
            [True, False, False, False],
            [True, False, False, False],
            [True, False, False, False],
            [True, False, False, False],
        ]),
    }
    assert torch.equal(
        _pair_common_mask(intact_batch, intact_fused, corrupt_batch),
        torch.tensor([True, False, False, False]),
    )
    corrupt_batch["role_mask"][0] = False
    assert not torch.any(
        _pair_common_mask(intact_batch, intact_fused, corrupt_batch)
    )


def _group(*, corrupt: bool = False, auc: float = 0.80) -> dict[str, Any]:
    value: dict[str, Any] = {
        "common_fraction": 0.99,
        "component_preference_auc": auc,
        "coefficient_mae": 0.20,
        "local_auc_gain": 0.05,
        "local_mae_relative_improvement": 0.15,
        "parent_error_mps": {"mean": 0.80, "p50": 0.40, "p95": 2.0},
        "fused_error_mps": {"mean": 0.60, "p50": 0.25, "p95": 1.8},
        "oracle_error_mps": {"mean": 0.40, "p50": 0.10, "p95": 1.0},
        "blind_error_mps": {"mean": 0.70, "p50": 0.30, "p95": 2.0},
    }
    if corrupt:
        value.update({
            "paired_fraction": 0.99,
            "paired_weight_separation_mean": 0.16,
        })
    return value


def _metrics() -> dict[str, Any]:
    return {"arms": {
        "intact": {
            "full_count": 100, "state_supported_count": 100,
            "groups": {name: _group() for name in ("overall", "rotation", "combined")},
        },
        "global": {
            "full_count": 100, "state_supported_count": 100,
            "groups": {name: _group(corrupt=True) for name in ("overall", "rotation", "combined")},
        },
        "hard": {
            "full_count": 80, "state_supported_count": 80,
            "groups": {name: _group(corrupt=True) for name in ("overall", "rotation", "combined")},
        },
    }}


def test_fold_and_cross_fold_gates_fail_closed_at_thresholds() -> None:
    metrics = _metrics()
    assert _all_boolean_gates_pass(_fold_gates(metrics))
    metrics["arms"]["intact"]["groups"]["rotation"][
        "component_preference_auc"
    ] = 0.749
    assert not _fold_gates(metrics)["rotation_intact_auc_ge_0_75"]
    left, right = _metrics(), _metrics()
    cv = [{"metrics": left}, {"metrics": right}]
    assert _all_boolean_gates_pass(_cross_fold_gates(cv))
    right["arms"]["intact"]["groups"]["combined"][
        "component_preference_auc"
    ] = 0.69
    cross = _cross_fold_gates(cv)
    assert not cross["intact_combined_auc_fold_delta_le_0_10"]
    passed_folds = [
        {"passed": True, "gates": {"fold_ok": True}},
        {"passed": True, "gates": {"fold_ok": True}},
    ]
    assert _cv_authorized(
        passed_folds, {"stable": True},
    )
    assert not _cv_authorized(
        [passed_folds[0], {"passed": False, "gates": {"fold_ok": True}}],
        {"stable": True},
    )
    assert not _cv_authorized(
        passed_folds, {"stable": False},
    )


def test_normalizer_is_fit_only_and_not_reused_between_folds() -> None:
    torch.manual_seed(4)
    count, events = 6, 3

    def fields(offset: float) -> dict[str, torch.Tensor]:
        return {
            "event_feature": torch.randn(count, events, 4, 11) + offset,
            "event_mask": torch.ones(count, events, 4, dtype=torch.bool),
            "role_feature": torch.randn(count, 4, 9) + offset,
            "role_mask": torch.ones(count, 4, dtype=torch.bool),
            "pair_feature": torch.randn(count, 6, 8) + offset,
            "pair_mask": torch.ones(count, 6, dtype=torch.bool),
            "global_feature": torch.randn(count, 11) + offset,
            "row_mask": torch.ones(count, dtype=torch.bool),
        }

    fit = fields(0.0)
    heldout = fields(1000.0)
    first = EndpointTokenReliabilityFusion(width=8)
    first.fit_normalizer(**fit)
    before = first.global_mean.clone()
    _ = heldout["global_feature"]
    assert torch.equal(before, first.global_mean)
    second = EndpointTokenReliabilityFusion(width=8)
    second.fit_normalizer(**fields(5.0))
    assert not torch.equal(first.global_mean, second.global_mean)


def _write_a1_fixture(
    root: Path,
) -> dict[str, torch.nn.Module]:
    a0_path, parent = _write_a0_fixture(root.parent / "a0")
    root.mkdir()
    a0_sha = sha256_file(a0_path)
    dataset_manifest = root.parent / "dataset-manifest.json"
    truth_manifest = root.parent / "truth-manifest.json"
    dataset_manifest.write_text('{"sealed":"dataset"}', encoding="utf-8")
    truth_manifest.write_text('{"sealed":"truth"}', encoding="utf-8")
    frozen_root = root.parent / "frozen"
    frozen_root.mkdir()
    frozen_paths = {
        name: frozen_root / f"{name}.pt" for name in ("mapper", "s", "h")
    }
    for name, path in frozen_paths.items():
        path.write_bytes(f"sealed-{name}".encode("ascii"))
    torch.manual_seed(31)
    frozen_models = {
        name: torch.nn.Linear(1, 1) for name in ("mapper", "s", "h")
    }
    parent_payload = torch.load(parent, map_location="cpu", weights_only=False)
    parent_state_sha = state_dict_sha256(parent_payload["model"])
    contract = {
        "schema_version": RUN_SCHEMA,
        "diagnostic_only": True,
        "truth_omega_feature_construction": True,
        "train_only": True,
        "source_sha256": {
            name: sha256_file(path) for name, path in _a1_source_paths().items()
        },
        "a0_result_sha256": a0_sha,
        "parent_checkpoint_sha256": sha256_file(parent),
        "parent_model_state_dict_sha256": parent_state_sha,
        "dataset_manifest_path": str(dataset_manifest.resolve()),
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "truth_manifest_path": str(truth_manifest.resolve()),
        "truth_manifest_sha256": sha256_file(truth_manifest),
        "frozen_checkpoint_paths": {
            name: str(path.resolve()) for name, path in frozen_paths.items()
        },
        "frozen_checkpoint_sha256": {
            name: sha256_file(path) for name, path in frozen_paths.items()
        },
        "frozen_state_dict_sha256": {
            name: state_dict_sha256(model.state_dict())
            for name, model in frozen_models.items()
        },
        "folds": FIXED_FOLDS, "updates": 200, "width": 32,
        "seed": 20260730, "batch_size": 192, "expert_batch_size": 96,
        "role_embedding": False, "symmetric_set_pooling": True,
        "absolute_energy_monotonicity": False,
        "hard_map_policy": HARD_MAP_POLICY,
    }
    contract_sha = _json_sha256(contract)
    torch.manual_seed(41)
    fold_states: list[dict[str, torch.Tensor]] = []
    for _ in range(FIXED_FOLDS):
        model = EndpointTokenReliabilityFusion(width=32)
        count, events = 4, 3
        model.fit_normalizer(
            event_feature=torch.randn(count, events, 4, 11),
            event_mask=torch.ones(count, events, 4, dtype=torch.bool),
            role_feature=torch.randn(count, 4, 9),
            role_mask=torch.ones(count, 4, dtype=torch.bool),
            pair_feature=torch.randn(count, 6, 8),
            pair_mask=torch.ones(count, 6, dtype=torch.bool),
            global_feature=torch.randn(count, 11),
            row_mask=torch.ones(count, dtype=torch.bool),
        )
        fold_states.append(model.state_dict())
    checkpoint_path = root / "fold-heads.pt"
    torch.save({
        "schema_version": RUN_SCHEMA,
        "diagnostic_only": True,
        "truth_omega_feature_construction": True,
        "experiment_contract": contract,
        "experiment_contract_sha256": contract_sha,
        "a0_result_sha256": a0_sha,
        "parent_checkpoint_sha256": sha256_file(parent),
        "fold_models": fold_states,
        "fold_state_dict_sha256": [
            state_dict_sha256(state) for state in fold_states
        ],
    }, checkpoint_path)
    state_hashes = [state_dict_sha256(state) for state in fold_states]
    cv: list[dict[str, Any]] = []
    for fold in range(FIXED_FOLDS):
        metrics = _metrics()
        maps = {
            "global": _map_manifest(sample=10, valid=10),
            "hard": _map_manifest(sample=10, valid=8),
        }
        cv.append({
            "fold": fold, "passed": True, "metrics": metrics,
            "gates": _fold_gates(metrics),
            "train_maps": maps, "heldout_maps": maps,
        })
    cross_fold_gates = _cross_fold_gates(cv)
    result = {
        "schema_version": RUN_SCHEMA, "status": "passed",
        "diagnostic_only": True, "truth_omega_feature_construction": True,
        "formal_v15": False, "train_only": True,
        "validation_accessed": False, "test_accessed": False,
        "future_modules_loaded": False,
        "authorized_a1_counterfactual_probe": True,
        "authorized_formal_two_stage": False,
        "experiment_contract": contract,
        "experiment_contract_sha256": contract_sha,
        "a0": {"result": str(a0_path.resolve()), "sha256": a0_sha},
        "parent": {
            "checkpoint": str(parent.resolve()), "sha256": sha256_file(parent),
            "model_state_dict_sha256": parent_state_sha,
        },
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "truth_manifest_sha256": sha256_file(truth_manifest),
        "cv": cv,
        "cross_fold_gates": cross_fold_gates,
        "fold_checkpoint": str(checkpoint_path.resolve()),
        "fold_checkpoint_sha256": sha256_file(checkpoint_path),
        "fold_state_dict_sha256": state_hashes,
    }
    result_path = root / "screen_result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    (root / "run_state.json").write_text(json.dumps({
        "schema_version": RUN_SCHEMA, "status": "passed",
        "validation_accessed": False, "test_accessed": False,
        "future_modules_loaded": False,
        "experiment_contract_sha256": contract_sha,
        "screen_result_sha256": sha256_file(result_path),
        "fold_checkpoint_sha256": sha256_file(checkpoint_path),
    }), encoding="utf-8")
    return frozen_models


def _patch_frozen_loaders(
    monkeypatch: pytest.MonkeyPatch,
    models: dict[str, torch.nn.Module],
) -> None:
    import training.stage3.screen_endpoint_token_reliability_oracle_omega as screen

    monkeypatch.setattr(
        screen, "load_frozen_pnp_mapper", lambda path: (models["mapper"], {}),
    )
    monkeypatch.setattr(
        screen, "load_frozen_v19", lambda path: (models["s"], {}),
    )
    monkeypatch.setattr(
        screen, "load_frozen_hypothesis_adapter",
        lambda path, allow_diagnostic: (models["h"], {}),
    )


def test_a1_artifact_loader_accepts_bound_passed_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    models = _write_a1_fixture(root)
    _patch_frozen_loaders(monkeypatch, models)
    payload, result = validate_a1_p0_artifacts(root)
    assert len(payload["fold_models"]) == FIXED_FOLDS
    assert result["authorized_a1_counterfactual_probe"] is True


@pytest.mark.parametrize("tamper", ["checkpoint", "result", "run_state", "a0"])
def test_a1_artifact_loader_rejects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str,
) -> None:
    root = tmp_path / "run"
    models = _write_a1_fixture(root)
    _patch_frozen_loaders(monkeypatch, models)
    result_path = root / "screen_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if tamper == "checkpoint":
        payload = torch.load(
            root / "fold-heads.pt", map_location="cpu", weights_only=False,
        )
        first_key = next(iter(payload["fold_models"][0]))
        payload["fold_models"][0][first_key].reshape(-1)[0] += 1.0
        torch.save(payload, root / "fold-heads.pt")
    elif tamper == "result":
        result["authorized_a1_counterfactual_probe"] = False
        result_path.write_text(json.dumps(result), encoding="utf-8")
    elif tamper == "run_state":
        state_path = root / "run_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["validation_accessed"] = True
        state_path.write_text(json.dumps(state), encoding="utf-8")
    else:
        Path(result["a0"]["result"]).write_text("{}", encoding="utf-8")
    with pytest.raises((ValueError, KeyError)):
        validate_a1_p0_artifacts(root)
