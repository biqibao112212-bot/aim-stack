from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import training.stage3.screen_local_precision_state_space_oracle_omega as screen
from training.stage3.profiled_center_twist_future import (
    CENTER_TWIST_FORWARD_FIELDS,
    CenterPriorProfiledTwistScreen,
)


def _forward_fixture(
    batch: int = 2, events: int = 8,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    time = torch.linspace(-0.07, 0.0, events).expand(batch, -1).clone()
    center = torch.tensor([[0.2, 0.0, 0.01], [0.1, -0.08, -0.02]])[:batch]
    q0 = center[:, None] + torch.tensor([
        [-0.2, 0.0, 0.0], [0.0, 0.3, 0.01],
        [0.2, 0.0, -0.01], [0.0, -0.3, 0.0],
    ])[None]
    velocity = torch.tensor([[0.55, -0.35, 0.08], [-0.25, 0.4, -0.03]])[:batch]
    omega = torch.tensor([4.0, -6.0])[:batch]
    history = torch.zeros(batch, events, 4, 3)
    for row in range(batch):
        for event in range(events):
            theta = omega[row] * time[row, event]
            rotation = torch.stack((
                torch.stack((torch.cos(theta), -torch.sin(theta))),
                torch.stack((torch.sin(theta), torch.cos(theta))),
            ))
            history[row, event, :, :2] = (
                (torch.eye(2) - rotation) @ center[row, :2]
                + time[row, event] * velocity[row, :2]
                + (rotation @ q0[row, :, :2].T).T
            )
            history[row, event, :, 2] = (
                q0[row, :, 2] + time[row, event] * velocity[row, 2]
            )
    mask = torch.ones(batch, events, 4, dtype=torch.bool)
    state = {
        "history_obs_rel_m": history,
        "history_obs_mask": mask,
        "history_primary_mask": torch.nn.functional.one_hot(
            torch.zeros(batch, events, dtype=torch.long), num_classes=4,
        ).to(torch.bool),
        "history_event_mask": torch.ones(batch, events, dtype=torch.bool),
        "history_time_s": time,
        "history_switch_step": torch.zeros(batch, events, dtype=torch.long),
        "q0_relation_m": q0,
        "q0_supported": torch.ones(batch, 4, dtype=torch.bool),
    }
    prior = {
        "center_supported": torch.tensor([True, False])[:batch],
        "center_offset_m": center,
        "center_log_variance_xy_z": torch.tensor([
            [-3.0, -3.5], [-2.5, -3.2],
        ])[:batch],
        "q0_relation_m": q0,
    }
    return state, prior, omega


def _sidecar(
    expected: dict[str, object], contract_sha: str, *, supported: bool = True,
    sample_indices: torch.Tensor | None = None,
    motion_class: torch.Tensor | None = None,
) -> dict[str, object]:
    count = int(expected["stop"]) - int(expected["start"])
    indices = (
        torch.arange(count, dtype=torch.int64) + 10
        if sample_indices is None else sample_indices.to(torch.int64)
    )
    if screen._indices_sha256(indices) != expected["sample_indices_sha256"]:
        raise ValueError("test sidecar indices do not match expected shard")
    reference = torch.ones(count, dtype=torch.bool)
    local_supported = torch.full((count,), supported, dtype=torch.bool)
    return {
        "schema_version": screen.RUN_SCHEMA,
        "experiment_contract_sha256": contract_sha,
        **expected,
        "sample_indices": indices,
        "raw_label_fields": [],
        "motion_class": (
            torch.full((count,), 2, dtype=torch.int64)
            if motion_class is None else motion_class.to(torch.int64)
        ),
        "parent_error_mps": torch.full((count,), 2.0),
        "projection_error_mps": torch.full((count,), 1.0),
        "local_error_mps": torch.full((count,), 1.5),
        "reference_common": reference,
        "common": reference & local_supported,
        "local_profile_supported": local_supported,
        "best_start": torch.zeros(count, dtype=torch.int64),
        "anchor_alpha": torch.zeros(count, 4),
        "center_alpha": torch.zeros(count),
        "observation_log_precision": torch.zeros(count, 3, 4),
    }


def _completed_validator_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "completed-oracle"
    root.mkdir()
    (root / screen.SIDECAR_DIRECTORY).mkdir()
    dataset_manifest = tmp_path / "dataset" / "dataset_manifest.json"
    truth_manifest = tmp_path / "truth" / "truth_manifest.json"
    dataset_manifest.parent.mkdir()
    truth_manifest.parent.mkdir()
    dataset_manifest.write_text('{"dataset":"sealed"}', encoding="utf-8")
    truth_manifest.write_text('{"truth":"train-only"}', encoding="utf-8")
    parent_root = tmp_path / "parent"
    parent_path = parent_root / "checkpoints" / "fold.pt"
    parent_path.parent.mkdir(parents=True)
    parent_state = {"weight": torch.tensor([1.25])}
    torch.save({"model": parent_state}, parent_path)
    (parent_root / "run_manifest.json").write_text(
        '{"manifest":"sealed"}', encoding="utf-8",
    )
    (parent_root / "screen_result.json").write_text(
        '{"result":"sealed"}', encoding="utf-8",
    )
    frozen_paths: dict[str, Path] = {}
    frozen_states: dict[str, dict[str, torch.Tensor]] = {}
    for offset, name in enumerate(("mapper", "s", "h"), start=1):
        path = tmp_path / "frozen" / f"{name}.pt"
        path.parent.mkdir(exist_ok=True)
        torch.save({"name": name}, path)
        frozen_paths[name] = path
        frozen_states[name] = {"weight": torch.tensor([float(offset)])}

    class Frozen:
        def __init__(self, state: dict[str, torch.Tensor]) -> None:
            self._state = state

        def state_dict(self) -> dict[str, torch.Tensor]:
            return self._state

    monkeypatch.setattr(
        screen, "load_frozen_pnp_mapper",
        lambda _path: (Frozen(frozen_states["mapper"]), {}),
    )
    monkeypatch.setattr(
        screen, "load_frozen_v19",
        lambda _path: (Frozen(frozen_states["s"]), {}),
    )
    monkeypatch.setattr(
        screen, "load_frozen_hypothesis_adapter",
        lambda _path, allow_diagnostic: (Frozen(frozen_states["h"]), {}),
    )
    sessions = ["session-a", "session-a", "session-b", "session-b"]
    monkeypatch.setattr(screen, "_load_train_dataset", lambda _path: object())
    monkeypatch.setattr(
        screen, "_combined_metadata", lambda _dataset: (None, sessions),
    )
    monkeypatch.setattr(
        screen, "_fold_assignments",
        lambda _sessions, folds: [["session-a"], ["session-b"]],
    )
    monkeypatch.setattr(
        screen, "assert_manifest_split_shards_unchanged",
        lambda *_args, **_kwargs: None,
    )

    class TruthIndex:
        def __init__(self, *_args, **_kwargs) -> None:
            self.attached = False

        def attach(self, _dataset: object) -> None:
            self.attached = True

        def assert_unchanged(self) -> None:
            assert self.attached

    monkeypatch.setattr(screen, "SplitScopedTruthIndex", TruthIndex)
    v93_path = tmp_path / "v93" / "screen_result.json"
    v93_path.parent.mkdir()
    v93_path.write_text("{}", encoding="utf-8")
    expected_reference_counts = [
        {
            "overall": {"full_count": 2, "min_reference_common_count": 2},
            "rotation": {"full_count": 1, "min_reference_common_count": 1},
            "combined": {"full_count": 1, "min_reference_common_count": 1},
        }
        for _ in range(2)
    ]
    a2_result = {
        "experiment_contract_sha256": "b" * 64,
        "cv": [
            {
                "fold": fold,
                "metrics": {"arms": {"intact": {"groups": {
                    group: {
                        "full_count": counts["full_count"],
                        "common_count": counts["min_reference_common_count"],
                    }
                    for group, counts in expected_reference_counts[fold].items()
                }}}},
            }
            for fold in range(2)
        ],
    }
    monkeypatch.setattr(
        screen, "validate_v93_failure",
        lambda _path: ({}, a2_result, v93_path.resolve()),
    )
    fold_indices = [np.asarray([0, 1]), np.asarray([2, 3])]
    expected = screen._expected_shards(fold_indices, chunk_size=32)
    contract = {
        "schema_version": screen.RUN_SCHEMA,
        "diagnostic_only": True,
        "train_only": True,
        "truth_omega_feature_construction": True,
        "truth_velocity_loss_only": True,
        "network_trained": False,
        "v93_result_path": str(v93_path.resolve()),
        "v93_result_sha256": screen.V93_RESULT_SHA256,
        "v93_contract_sha256": a2_result["experiment_contract_sha256"],
        "parent_checkpoint_path": str(parent_path.resolve()),
        "parent_checkpoint_sha256": screen.sha256_file(parent_path),
        "parent_model_state_dict_sha256": screen.state_dict_sha256(parent_state),
        "parent_manifest_sha256": screen.sha256_file(
            parent_root / "run_manifest.json"
        ),
        "parent_result_sha256": screen.sha256_file(
            parent_root / "screen_result.json"
        ),
        "dataset_manifest_path": str(dataset_manifest.resolve()),
        "dataset_manifest_sha256": screen.sha256_file(dataset_manifest),
        "truth_manifest_path": str(truth_manifest.resolve()),
        "truth_manifest_sha256": screen.sha256_file(truth_manifest),
        "frozen_checkpoint_paths": {
            name: str(path.resolve()) for name, path in frozen_paths.items()
        },
        "frozen_checkpoint_sha256": {
            name: screen.sha256_file(path) for name, path in frozen_paths.items()
        },
        "frozen_state_dict_sha256": {
            name: screen.state_dict_sha256(state)
            for name, state in frozen_states.items()
        },
        "source_sha256": {
            name: screen.sha256_file(path)
            for name, path in screen._source_paths().items()
        },
        "git_commit": "1" * 40,
        "seed": screen.FIXED_SEED,
        "folds": screen.FIXED_FOLDS,
        "oracle_steps": screen.FIXED_ORACLE_STEPS,
        "oracle_starts": 3,
        "oracle_chunk_size": screen.FIXED_ORACLE_CHUNK_SIZE,
        "recovery_gate": screen.RECOVERY_GATE,
        "minimum_parent_projection_headroom_mps": (
            screen.MIN_PARENT_PROJECTION_HEADROOM_MPS
        ),
        "minimum_parent_local_improvement_mps": (
            screen.MIN_PARENT_LOCAL_IMPROVEMENT_MPS
        ),
        "anchor_availability": "center-supported-expanded-four-roles",
        "forward_cache_contains_truth": False,
        "forward_excludes": [
            "session", "motion_class", "physical_armor_id",
            "truth_velocity", "future", "absolute_pose",
        ],
        "fold_sample_indices_sha256": [
            screen._indices_sha256(indices) for indices in fold_indices
        ],
        "expected_shards": expected,
        "expected_reference_counts": expected_reference_counts,
    }
    contract_sha = screen._json_sha256(contract)
    screen._atomic_json(root / screen.CONTRACT_NAME, contract)
    completed: dict[str, dict[str, str]] = {}
    payloads: list[dict[str, object]] = []
    for fold, indices in enumerate(fold_indices):
        shard_id = sorted(
            key for key, value in expected.items() if value["fold"] == fold
        )[0]
        payload = _sidecar(
            expected[shard_id], contract_sha,
            sample_indices=torch.from_numpy(indices),
            motion_class=torch.tensor([2, 3]),
        )
        payloads.append(payload)
        path = root / screen.SIDECAR_DIRECTORY / f"{shard_id}.pt"
        torch.save(payload, path)
        completed[shard_id] = {
            "path": str(path.resolve()), "sha256": screen.sha256_file(path),
        }
    cv = []
    for fold, payload in enumerate(payloads):
        metrics = screen._summarize_fold(
            [payload], expected_reference_counts[fold],
        )
        gates = screen._fold_gates(metrics)
        cv.append({
            "fold": fold, "passed": screen._all_boolean_gates_pass(gates),
            "heldout_sessions": [["session-a"], ["session-b"]][fold],
            "heldout_count": 2, "metrics": metrics, "gates": gates,
        })
    screen._write_checkpoint_and_state(
        root, contract_sha, completed, torch.device("cpu"), status="running",
        generation=1,
    )
    final_generation = 2
    result = {
        "schema_version": screen.RUN_SCHEMA,
        "status": "passed", "diagnostic_only": True, "train_only": True,
        "network_trained": False, "truth_omega_feature_construction": True,
        "truth_velocity_loss_only": True, "validation_accessed": False,
        "test_accessed": False, "future_modules_loaded": False,
        "plots_created": False, "authorized_train_local_precision": True,
        "authorized_validation": False, "authorized_formal_two_stage": False,
        "experiment_contract": contract,
        "experiment_contract_sha256": contract_sha,
        "v93": {
            "result": str(v93_path.resolve()),
            "sha256": screen.V93_RESULT_SHA256,
        },
        "cv": cv, "completed_shards": completed,
        "fp32_boundary_samples_checked_this_process": 4,
        "oracle_checkpoint_generation": final_generation,
        "oracle_checkpoint": str(
            screen._checkpoint_path(root, final_generation).resolve()
        ),
    }
    screen._atomic_json(root / "screen_result.json", result)
    screen._write_checkpoint_and_state(
        root, contract_sha, completed, torch.device("cpu"), status="passed",
        result_sha=screen.sha256_file(root / "screen_result.json"),
        authorized=True, generation=final_generation,
    )
    return root, result


def test_v93_is_strict_loaded_before_hash_and_scope_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "v93"
    root.mkdir()
    (root / "screen_result.json").write_text("{}", encoding="utf-8")
    calls: list[Path] = []
    valid = {
        "status": "failed", "validation_accessed": False,
        "test_accessed": False, "future_modules_loaded": False,
        "train_only": True, "authorized_a2_train_counterfactual": False,
    }

    def validator(path: str | Path):
        calls.append(Path(path).resolve())
        return {}, dict(valid)

    monkeypatch.setattr(screen, "sha256_file", lambda _: screen.V93_RESULT_SHA256)
    _, result, path = screen.validate_v93_failure(root, validator=validator)
    assert calls == [root.resolve()]
    assert result == valid
    assert path == (root / "screen_result.json").resolve()

    for key, bad in (
        ("status", "passed"), ("validation_accessed", True),
        ("test_accessed", True), ("future_modules_loaded", True),
        ("authorized_a2_train_counterfactual", True),
    ):
        def rejected(_path: str | Path, key=key, bad=bad):
            calls.append(Path(_path).resolve())
            changed = dict(valid)
            changed[key] = bad
            return {}, changed

        with pytest.raises(ValueError, match="status/scope"):
            screen.validate_v93_failure(root, validator=rejected)

    monkeypatch.setattr(screen, "sha256_file", lambda _: "0" * 64)
    with pytest.raises(ValueError, match="result hash"):
        screen.validate_v93_failure(root, validator=validator)


def test_source_binding_includes_projection_dependency() -> None:
    paths = screen._source_paths()
    assert paths["prequential_local_precision_irls"].name == (
        "prequential_local_precision_irls.py"
    )


def test_truth_free_solve_cache_and_mixed_anchor_availability() -> None:
    state, prior, omega = _forward_fixture()
    cache = screen._forward_cache(state, omega, prior)
    assert set(cache) == set(CENTER_TWIST_FORWARD_FIELDS) | {
        "omega_rad_s", "center_prior",
    }
    assert not any(
        token in key for key in cache
        for token in ("truth", "target", "future", "motion_class", "session")
    )
    calls: list[tuple[tuple[torch.Tensor, ...], dict[str, object]]] = []

    class FakeWeighted:
        def __call__(self, *args: torch.Tensor, **kwargs: object):
            calls.append((args, kwargs))
            rows = args[0].shape[0]
            return {
                "velocity_mps": torch.zeros(rows, 3),
                "state_supported": torch.ones(rows, dtype=torch.bool),
                "profile_supported": (
                    torch.arange(rows) % 2 == 0
                ),
            }

    seen: dict[str, object] = {}

    def fake_optimize(solve, **kwargs):
        seen.update(kwargs)
        rows = 3 * state["history_obs_rel_m"].shape[0]
        solved = solve(
            torch.zeros(rows, state["history_obs_rel_m"].shape[1], 4),
            torch.zeros(rows, 4), torch.zeros(rows),
        )
        assert torch.equal(
            solved["state_supported"], solved["profile_supported"]
        )
        return {
            "state_supported": torch.ones(2, dtype=torch.bool),
            "velocity_mps": torch.zeros(2, 3),
        }

    optimized = screen._optimize_chunk(
        FakeWeighted(), cache, torch.randn(2, 3), optimize=fake_optimize,
    )
    assert torch.equal(
        seen["q0_supported"],
        torch.tensor([[True, True, True, True], [False, False, False, False]]),
    )
    assert seen["steps"] == 32
    assert len(calls) == 1
    assert torch.equal(
        optimized["profile_supported"],
        torch.ones(2, dtype=torch.bool),
    )
    _, forward_kwargs = calls[0]
    assert not any(
        token in key.lower() for key in forward_kwargs
        for token in ("truth", "target", "future", "motion", "session")
    )


@torch.inference_mode()
def test_real_fp32_boundaries_cover_mixed_supported_batch_exactly() -> None:
    state, prior, omega = _forward_fixture()
    base = CenterPriorProfiledTwistScreen().eval()
    cache = screen._forward_cache(state, omega, prior)
    assert screen._verify_fp32_boundaries(
        screen._weighted_like(base).eval(), base, cache,
    ) == 2


def test_recovery_gates_include_reference_domain_coverage() -> None:
    payload = {
        "motion_class": torch.tensor([2, 2, 3, 3]),
        "parent_error_mps": torch.full((4,), 2.0),
        "projection_error_mps": torch.full((4,), 1.0),
        "local_error_mps": torch.full((4,), 1.5),
        "reference_common": torch.ones(4, dtype=torch.bool),
        "common": torch.ones(4, dtype=torch.bool),
        "local_profile_supported": torch.ones(4, dtype=torch.bool),
    }
    expected_counts = {
        "overall": {"full_count": 4, "min_reference_common_count": 4},
        "rotation": {"full_count": 2, "min_reference_common_count": 2},
        "combined": {"full_count": 2, "min_reference_common_count": 2},
    }
    metrics = screen._summarize_fold([payload], expected_counts)
    gates = screen._fold_gates(metrics)
    assert screen._all_boolean_gates_pass(gates)
    for group in ("overall", "rotation", "combined"):
        assert gates[f"{group}_mean_recovery"] == pytest.approx(0.5)
        assert gates[f"{group}_p50_recovery"] == pytest.approx(0.5)
        assert gates[f"{group}_local_coverage_100pct"] is True

    payload["local_profile_supported"] = torch.tensor([True, True, True, False])
    payload["common"] = (
        payload["reference_common"] & payload["local_profile_supported"]
    )
    gates = screen._fold_gates(
        screen._summarize_fold([payload], expected_counts)
    )
    assert gates["combined_local_coverage_100pct"] is False
    assert not screen._all_boolean_gates_pass(gates)

    payload["local_profile_supported"] = torch.ones(4, dtype=torch.bool)
    payload["common"] = torch.ones(4, dtype=torch.bool)
    too_small = {
        **expected_counts,
        "combined": {"full_count": 2, "min_reference_common_count": 3},
    }
    gates = screen._fold_gates(screen._summarize_fold([payload], too_small))
    assert gates["combined_reference_common_ge_a2_min"] is False

    payload["projection_error_mps"] = torch.full((4,), 1.99)
    payload["local_error_mps"] = torch.full((4,), 1.98)
    gates = screen._fold_gates(screen._summarize_fold([payload], expected_counts))
    assert gates["overall_mean_headroom_ge_0_02_mps"] is False
    payload["projection_error_mps"] = torch.full((4,), 1.0)
    payload["local_error_mps"] = torch.full((4,), 1.995)
    gates = screen._fold_gates(screen._summarize_fold([payload], expected_counts))
    assert gates["overall_mean_improvement_ge_0_01_mps"] is False


def test_resume_is_exact_and_never_overwrites_accepted_shard(
    tmp_path: Path,
) -> None:
    output = tmp_path / "oracle"
    contract = {"schema_version": screen.RUN_SCHEMA, "tag": "sealed"}
    contract_sha = screen._json_sha256(contract)
    expected = screen._expected_shards(
        [np.asarray([10, 11]), np.asarray([20])], chunk_size=32,
    )
    completed, _ = screen._initialize_or_resume(
        output, contract, expected, torch.device("cpu"),
    )
    first_id = sorted(expected)[0]
    payload = _sidecar(expected[first_id], contract_sha)
    screen._commit_shard(
        output, contract_sha, expected[first_id], payload,
        completed, torch.device("cpu"),
    )
    path = Path(completed[first_id]["path"])
    accepted_sha = screen.sha256_file(path)
    resumed, _ = screen._initialize_or_resume(
        output, contract, expected, torch.device("cpu"),
    )
    assert resumed[first_id]["sha256"] == accepted_sha
    with pytest.raises(FileExistsError, match="overwrite"):
        screen._commit_shard(
            output, contract_sha, expected[first_id], payload,
            resumed, torch.device("cpu"),
        )
    assert screen.sha256_file(path) == accepted_sha
    with pytest.raises(ValueError, match="contract differs"):
        screen._initialize_or_resume(
            output, {**contract, "tag": "changed"}, expected,
            torch.device("cpu"),
        )
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="accepted oracle shard changed"):
        screen._initialize_or_resume(
            output, contract, expected, torch.device("cpu"),
        )


def test_sidecar_rejects_truth_fields_and_support_domain_mismatch() -> None:
    expected = {
        "shard_id": "fold00-000000-000001", "fold": 0,
        "start": 0, "stop": 1,
        "sample_indices_sha256": screen._indices_sha256(
            torch.tensor([10], dtype=torch.int64)
        ),
    }
    contract_sha = "a" * 64
    payload = _sidecar(expected, contract_sha)
    screen._validate_sidecar_payload(payload, expected, contract_sha)
    payload["target_velocity_mps"] = torch.zeros(1, 3)
    with pytest.raises(ValueError, match="identity/scope"):
        screen._validate_sidecar_payload(payload, expected, contract_sha)
    del payload["target_velocity_mps"]
    payload["common"] = torch.zeros(1, dtype=torch.bool)
    with pytest.raises(ValueError, match="support/domain"):
        screen._validate_sidecar_payload(payload, expected, contract_sha)


def test_completed_artifact_validator_recomputes_gates_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, original_result = _completed_validator_fixture(tmp_path, monkeypatch)
    checkpoint, validated = screen.validate_local_precision_oracle_artifacts(root)
    assert checkpoint["status"] == "passed"
    assert validated == original_result

    changed = json.loads(json.dumps(original_result))
    changed["cv"][0]["gates"]["overall_mean_recovery_ge_0_40"] = False
    screen._atomic_json(root / "screen_result.json", changed)
    run_state = json.loads((root / "run_state.json").read_text(encoding="utf-8"))
    run_state["screen_result_sha256"] = screen.sha256_file(
        root / "screen_result.json"
    )
    screen._atomic_json(root / "run_state.json", run_state)
    with pytest.raises(ValueError, match="recomputed fold"):
        screen.validate_local_precision_oracle_artifacts(root)

    screen._atomic_json(root / "screen_result.json", original_result)
    run_state["screen_result_sha256"] = screen.sha256_file(
        root / "screen_result.json"
    )
    screen._atomic_json(root / "run_state.json", run_state)
    shard_path = next((root / screen.SIDECAR_DIRECTORY).glob("*.pt"))
    original_shard = shard_path.read_bytes()
    with shard_path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="accepted oracle shard changed"):
        screen.validate_local_precision_oracle_artifacts(root)

    shard_path.write_bytes(original_shard)
    changed = json.loads(json.dumps(original_result))
    changed["plots_created"] = True
    screen._atomic_json(root / "screen_result.json", changed)
    with pytest.raises(ValueError, match="status/scope|artifact hashes"):
        screen.validate_local_precision_oracle_artifacts(root)


def test_resume_uses_commit_pointer_when_new_checkpoint_precedes_old_state(
    tmp_path: Path,
) -> None:
    output = tmp_path / "generation-window"
    contract = {"schema_version": screen.RUN_SCHEMA, "tag": "generation"}
    contract_sha = screen._json_sha256(contract)
    expected = screen._expected_shards(
        [np.asarray([10]), np.asarray([20])], chunk_size=32,
    )
    completed, _ = screen._initialize_or_resume(
        output, contract, expected, torch.device("cpu"),
    )
    assert json.loads((output / "run_state.json").read_text())["generation"] == 1
    shard_id = sorted(expected)[0]
    payload = _sidecar(expected[shard_id], contract_sha)
    shard_path = output / screen.SIDECAR_DIRECTORY / f"{shard_id}.pt"
    screen._atomic_torch_create(shard_path, payload)
    orphan_completed = {
        shard_id: {
            "path": str(shard_path.resolve()),
            "sha256": screen.sha256_file(shard_path),
        }
    }
    orphan_checkpoint = screen._checkpoint_payload(
        contract_sha, orphan_completed, torch.device("cpu"),
        status="running", generation=2,
    )
    screen._atomic_torch_create(
        screen._checkpoint_path(output, 2), orphan_checkpoint,
    )
    accepted_shard_sha = screen.sha256_file(shard_path)
    resumed, _ = screen._initialize_or_resume(
        output, contract, expected, torch.device("cpu"),
    )
    state = json.loads((output / "run_state.json").read_text())
    assert state["generation"] == 3
    assert state["completed_shards"] == resumed
    assert shard_id in resumed
    assert screen._checkpoint_path(output, 2).exists()
    assert screen.sha256_file(shard_path) == accepted_shard_sha


@pytest.mark.parametrize("final_checkpoint_exists", [False, True])
def test_validator_converges_final_result_across_both_final_commit_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    final_checkpoint_exists: bool,
) -> None:
    root, result = _completed_validator_fixture(tmp_path, monkeypatch)
    contract_sha = result["experiment_contract_sha256"]
    completed = result["completed_shards"]
    final_path = Path(result["oracle_checkpoint"])
    result_sha = screen.sha256_file(root / "screen_result.json")
    shard_hashes = {
        path: screen.sha256_file(path)
        for path in (root / screen.SIDECAR_DIRECTORY).glob("*.pt")
    }
    # Rewind only the atomic commit pointer to the preceding running generation.
    screen._write_checkpoint_and_state(
        root, contract_sha, completed, torch.device("cpu"),
        status="running", generation=1,
    )
    if not final_checkpoint_exists:
        final_path.unlink()
    checkpoint, validated = screen.validate_local_precision_oracle_artifacts(root)
    state = json.loads((root / "run_state.json").read_text())
    assert state["status"] == "passed"
    assert state["generation"] == 2
    assert checkpoint["generation"] == 2
    assert validated == result
    assert final_path.exists()
    assert screen.sha256_file(root / "screen_result.json") == result_sha
    assert {
        path: screen.sha256_file(path) for path in shard_hashes
    } == shard_hashes


def test_validator_reconstructs_truth_index_and_rejects_changed_truth_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _completed_validator_fixture(tmp_path, monkeypatch)
    calls: list[tuple[Path, str, str]] = []

    class ChangedTruth:
        def __init__(
            self, path: str | Path, *, split: str,
            expected_manifest_sha256: str,
        ) -> None:
            calls.append((Path(path).resolve(), split, expected_manifest_sha256))

        def attach(self, _dataset: object) -> None:
            pass

        def assert_unchanged(self) -> None:
            raise ValueError("truth shard changed")

    monkeypatch.setattr(screen, "SplitScopedTruthIndex", ChangedTruth)
    with pytest.raises(ValueError, match="truth shard changed"):
        screen.validate_local_precision_oracle_artifacts(root)
    assert len(calls) == 1
    assert calls[0][1] == "train"
