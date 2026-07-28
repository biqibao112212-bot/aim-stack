from pathlib import Path
from types import SimpleNamespace

import pytest

from training.stage3.formal_run_contract import (
    PROTOCOL_SCHEMA,
    load_protocol,
    require_asset_binding,
    require_compatible_contracts,
    require_exact_protocol_arguments,
    require_fixed_final_state,
    require_formal_checkpoint_manifest,
    resolve_formal_schedule,
)
from training.stage3.split_audit import (
    build_split_audit,
    require_formal_split_isolation,
)


def test_formalization_protocol_is_versioned_and_fixed_final() -> None:
    path, protocol = load_protocol()
    assert path.name == "formalization_protocol.json"
    assert protocol["schema_version"] == PROTOCOL_SCHEMA
    assert protocol["claim"] == {
        "formal_oracle_evaluation": True,
        "deployable_pipeline": False,
        "oracle_association": True,
        "full_chain_provenance_clean": False,
        "legacy_initial_assets": True,
        "test_accessed_during_training": False,
    }
    for stage in ("mapper", "hypothesis", "trajectory", "selector"):
        assert protocol[stage]["fixed_final_checkpoint"] is True
        assert 0 < protocol[stage]["fixed_final_update"] < protocol[stage]["schedule_total_updates"]
    hypothesis = protocol["hypothesis"]
    assert hypothesis["execution_mode"] == "frozen_train_s_cache_v1"
    assert hypothesis["train_cache"] == {
        "schema_version": "stage3-formal-h-train-cache-v1",
        "device_type": "cuda",
        "float_dtype": "torch.float32",
        "precompute_batch_size": 128,
        "validation_cached": False,
        "partial_training_batch_cached": False,
    }


def test_protocol_argument_check_fails_closed() -> None:
    class Arguments:
        seed = 7
        max_updates = 10

    require_exact_protocol_arguments(
        Arguments(), {"seed": 7, "max_updates": 10}, ("seed", "max_updates")
    )
    with pytest.raises(ValueError, match="max_updates"):
        require_exact_protocol_arguments(
            Arguments(), {"seed": 7, "max_updates": 11},
            ("seed", "max_updates"),
        )


def test_protocol_file_is_inside_stage3_source_tree() -> None:
    path, _ = load_protocol()
    assert path.parent == Path(__file__).resolve().parents[1]


def test_noncanonical_protocol_path_is_rejected(tmp_path: Path) -> None:
    custom = tmp_path / "protocol.json"
    custom.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        load_protocol(custom)


def test_asset_and_parent_contract_checks_fail_closed() -> None:
    expected = {"file_sha256": "file", "state_dict_sha256": "state"}
    require_asset_binding(
        "asset", {"sha256": "file", "state_dict_sha256": "state"}, expected
    )
    with pytest.raises(ValueError, match="asset"):
        require_asset_binding(
            "asset", {"sha256": "wrong", "state_dict_sha256": "state"},
            expected,
        )
    current = {
        "git_commit": "abc", "git_branch": "main",
        "protocol_sha256": "protocol", "protocol_schema_version": "v1",
        "environment": {"torch": "x"},
    }
    require_compatible_contracts("parent", dict(current), current)
    parent = dict(current)
    parent["git_commit"] = "wrong"
    with pytest.raises(ValueError, match="git_commit"):
        require_compatible_contracts("parent", parent, current)


def test_fixed_final_state_requires_one_validation_at_exact_budget() -> None:
    section = {
        "fixed_final_update": 11,
        "fixed_final_checkpoint": True,
    }
    history = [{"update": 11}]
    best = {"update": 11}
    require_fixed_final_state(
        "stage", section, update=11, stop_reason="fixed_final_update",
        history=history, best=best,
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        require_fixed_final_state(
            "stage", section, update=10, stop_reason="epoch_limit",
            history=history, best=best,
        )


def test_formal_checkpoint_manifest_accepts_only_declared_fixed_final(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"checkpoint")
    import hashlib

    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "status": "complete",
        "stop_reason": "fixed_final_update",
        "formal_gate_passed": True,
        "formal_oracle_evaluation": True,
        "fixed_final_checkpoint": True,
        "update": 7,
        "best": {"path": checkpoint.name, "sha256": checkpoint_sha, "update": 7},
        "history": [{"update": 7}],
    }
    require_formal_checkpoint_manifest(
        "parent", checkpoint, manifest, expected_update=7,
        checkpoint_update=7,
    )
    manifest["formal_gate_passed"] = False
    with pytest.raises(ValueError, match="fixed final"):
        require_formal_checkpoint_manifest(
            "parent", checkpoint, manifest, expected_update=7,
            checkpoint_update=7,
        )


def test_formal_split_audit_rejects_overlap() -> None:
    def dataset(split: str, session: str, timestamp: int) -> SimpleNamespace:
        audit, sessions, keys = build_split_audit(
            split=split,
            session_ids=[session],
            t0_ns=[timestamp],
            pair_ids=["pair"],
            shard_tokens=[f"{split}-shard"],
            sample_limit=0,
            motion_class=3,
            sample_strategy="full_split",
        )
        return SimpleNamespace(
            split_audit=audit, session_set=sessions, sample_key_set=keys
        )

    train = dataset("train", "train-session", 1)
    validation = dataset("validation", "validation-session", 2)
    assert require_formal_split_isolation(train, validation) == {
        "train_validation_session_overlap_count": 0,
        "train_validation_sample_key_overlap_count": 0,
    }
    overlapping = dataset("validation", "train-session", 1)
    with pytest.raises(ValueError, match="isolation"):
        require_formal_split_isolation(train, overlapping)


def test_formal_schedule_separates_lr_horizon_from_fixed_final() -> None:
    section = {
        "schedule_total_updates": 400,
        "fixed_final_update": 264,
    }
    assert resolve_formal_schedule(400, 528, section) == (400, 264)
    with pytest.raises(ValueError, match="horizon"):
        resolve_formal_schedule(264, 528, section)
    with pytest.raises(ValueError, match="epoch"):
        resolve_formal_schedule(400, 399, section)
