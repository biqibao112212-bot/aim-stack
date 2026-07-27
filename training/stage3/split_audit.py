"""Reusable fail-closed split isolation evidence for Stage3 formal runs."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Sequence


def _stable_string_set_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(set(values)):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_split_audit(
    *,
    split: str,
    session_ids: Sequence[object],
    t0_ns: Sequence[object],
    pair_ids: Sequence[object],
    shard_tokens: Iterable[str],
    sample_limit: int,
    motion_class: int | None,
    sample_strategy: str,
) -> tuple[dict[str, Any], frozenset[str], frozenset[str]]:
    if not (len(session_ids) == len(t0_ns) == len(pair_ids)):
        raise ValueError("split audit metadata lengths differ")
    sessions_sequence = tuple(str(value) for value in session_ids)
    keys_sequence = tuple(
        f"{session}\x1f{int(timestamp)}\x1f{str(pair_id)}"
        for session, timestamp, pair_id in zip(
            sessions_sequence, t0_ns, pair_ids
        )
    )
    sessions = frozenset(sessions_sequence)
    sample_keys = frozenset(keys_sequence)
    audit = {
        "split": split,
        "sample_count": len(keys_sequence),
        "unique_sample_key_count": len(sample_keys),
        "duplicate_sample_key_count": len(keys_sequence) - len(sample_keys),
        "session_count": len(sessions),
        "session_set_sha256": _stable_string_set_sha256(sessions),
        "sample_key_set_sha256": _stable_string_set_sha256(sample_keys),
        "shard_set_sha256": _stable_string_set_sha256(shard_tokens),
        "sample_strategy": sample_strategy,
        "sample_limit": int(sample_limit),
        "motion_class": motion_class,
    }
    return audit, sessions, sample_keys


def require_formal_split_isolation(
    train_dataset: Any,
    validation_dataset: Any,
) -> dict[str, int]:
    train_audit = train_dataset.split_audit
    validation_audit = validation_dataset.split_audit
    session_overlap = train_dataset.session_set & validation_dataset.session_set
    sample_overlap = train_dataset.sample_key_set & validation_dataset.sample_key_set
    if (
        train_audit.get("split") != "train"
        or validation_audit.get("split") != "validation"
        or train_audit.get("sample_limit") != 0
        or validation_audit.get("sample_limit") != 0
        or train_audit.get("duplicate_sample_key_count") != 0
        or validation_audit.get("duplicate_sample_key_count") != 0
        or session_overlap
        or sample_overlap
    ):
        raise ValueError("formal train/validation split isolation failed")
    return {
        "train_validation_session_overlap_count": len(session_overlap),
        "train_validation_sample_key_overlap_count": len(sample_overlap),
    }
