from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.stage3.build_dataset import _load_capture_contract_binding
from training.stage3.freeze_multistate_capture_contract import freeze_contract
from training.stage3.generate_multistate_fixed_6mm_manifest import (
    generate_records,
    write_manifest,
)


SOURCE_COMMIT = "1" * 40


def _manifest(tmp_path: Path) -> Path:
    records, rejected = generate_records(1234, "formal-test")
    path = tmp_path / "manifest.jsonl"
    write_manifest(records, path, 1234, rejected)
    return path


def test_freeze_contract_binds_exact_manifest_and_14_5_5_split(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "capture-contract.json"
    payload = freeze_contract(manifest, output, split_seed=9876, source_git_commit=SOURCE_COMMIT)
    assert payload["session_count"] == 24
    assert payload["family_session_counts"] == {"spin": 12, "linear_and_spin": 12}
    assert {name: len(ids) for name, ids in payload["splits"].items()} == {
        "train": 14, "validation": 5, "test": 5,
    }
    assert json.loads(output.read_text(encoding="utf-8"))["formal_manifest_sha256"]


def test_freeze_contract_rejects_nonformal_segment_count(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    lines = manifest.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["segments"] = first["segments"][:-1]
    first["duration_s"] = 33.0
    lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 12"):
        freeze_contract(
            manifest, tmp_path / "capture-contract.json", split_seed=9876,
            source_git_commit=SOURCE_COMMIT,
        )


def test_formal_builder_binding_is_mandatory_and_fail_closed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    with pytest.raises(ValueError, match="requires --capture-contract"):
        _load_capture_contract_binding(manifest, records, 9876, None)
    contract_path = tmp_path / "capture-contract.json"
    freeze_contract(
        manifest, contract_path, split_seed=9876, source_git_commit=SOURCE_COMMIT,
    )
    path, contract = _load_capture_contract_binding(
        manifest, records, 9876, contract_path
    )
    assert path == contract_path.resolve()
    assert contract is not None
    with pytest.raises(ValueError, match="split seed"):
        _load_capture_contract_binding(manifest, records, 9877, contract_path)
    tampered = json.loads(contract_path.read_text(encoding="utf-8"))
    tampered["splits"]["train"], tampered["splits"]["validation"] = (
        tampered["splits"]["validation"], tampered["splits"]["train"]
    )
    tampered_path = tmp_path / "tampered-contract.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="computed session split"):
        _load_capture_contract_binding(manifest, records, 9876, tampered_path)
