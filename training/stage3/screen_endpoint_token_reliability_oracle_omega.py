"""V15-A1-P0 anonymous endpoint-token reliability sufficiency screen.

This is a train-only, truth-omega diagnostic.  It is authorized by a valid
V15-A0 train-CV rejection and never constructs validation/test datasets or a
future-position module.  Mapper/S/H and both V14 profiled experts stay frozen;
only the endpoint-token reliability head is optimized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .cyclic_future_foundation import load_frozen_v19
from .endpoint_token_reliability_fusion import (
    EndpointTokenReliabilityFusion,
    endpoint_token_features,
)
from .evaluate_profiled_center_twist_mechanism import (
    _validate_checkpoint_manifest_identity,
)
from .frozen_expert_reliability_fusion import (
    oracle_projection_coefficient,
    vector_huber_loss,
)
from .motion_truth_supervision import MOTION_TARGET_FIELD
from .observable_future_pnp_ab import sha256_file, state_dict_sha256
from .pnp_q0_hypothesis_adapter import (
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .profiled_center_twist_future import (
    CENTER_TWIST_FORWARD_FIELDS,
    CenterPriorProfiledTwistScreen,
)
from .split_scoped_truth_supervision import (
    SplitScopedTruthIndex,
    assert_manifest_split_shards_unchanged,
)
from .screen_frozen_expert_reliability_oracle_omega import (
    FIXED_FOLDS,
    RUN_SCHEMA as A0_RUN_SCHEMA,
    _aligned_donor_q0,
    _binary_auc,
    _build_map,
    _cache_prepared_state,
    _combined_metadata,
    _distribution,
    _fold_assignments,
    _slice_arm,
)
from .train_anonymous_vehicle_motion import _dataset, _json_sha256
from .train_causal_physical_ab import _git_state, _seed
from .train_pnp_window_mapper_distillation import (
    _atomic_checkpoint,
    _atomic_json,
)


RUN_SCHEMA = "stage3-endpoint-token-reliability-oracle-omega-v15-a1-p0"
FIXED_UPDATES = 200
FIXED_WIDTH = 32
HARD_MAP_POLICY = "maximal-balanced-exact-motion-support-cross-session-v1"
FEATURE_NAMES = (
    "event_feature", "event_mask", "role_feature", "role_mask",
    "pair_feature", "pair_mask", "global_feature",
)


def _a1_source_paths() -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "fusion": Path(__file__).with_name(
            "endpoint_token_reliability_fusion.py"
        ).resolve(),
        "a0_runner": Path(__file__).with_name(
            "screen_frozen_expert_reliability_oracle_omega.py"
        ).resolve(),
        "a0_fusion": Path(__file__).with_name(
            "frozen_expert_reliability_fusion.py"
        ).resolve(),
        "split_truth": Path(__file__).with_name(
            "split_scoped_truth_supervision.py"
        ).resolve(),
    }


def _load_train_dataset(
    dataset_path: Path,
    *,
    loader: Any = _dataset,
) -> Any:
    """The only dataset construction boundary in A1-P0."""
    return loader(dataset_path, "train", sample_limit=0)


def _validate_corruption_manifest(
    manifest: dict[str, Any], *, kind: str,
) -> None:
    sample_count = int(manifest["sample_count"])
    valid_count = int(manifest["valid_count"])
    exact_count = int(manifest["exact_valid_count"])
    if (
        manifest.get("require_cross_session") is not True
        or int(manifest["fixed_point_count"]) != 0
        or int(manifest["cross_session_count"]) != valid_count
        or int(manifest["relaxed_valid_count"]) != 0
        or exact_count != valid_count
    ):
        raise RuntimeError(f"A1 {kind} corruption identity differs")
    if kind == "global":
        if valid_count != sample_count or float(manifest["coverage"]) != 1.0:
            raise RuntimeError("A1 global corruption coverage is incomplete")
    elif kind == "hard":
        exclusion_groups = manifest.get("unavoidable_exclusion_groups")
        if (
            manifest.get("balancing_policy") != HARD_MAP_POLICY
            or int(manifest.get("maximal_exact_selected_count", -1))
            != valid_count
            or int(manifest.get("unavoidable_exclusion_count", -1))
            != sample_count - valid_count
            or not isinstance(exclusion_groups, list)
            or any(not isinstance(group, dict) for group in exclusion_groups)
            or sum(
                int(group.get("excluded_count", -1))
                for group in exclusion_groups
            ) != sample_count - valid_count
        ):
            raise RuntimeError("A1 hard balancing policy differs")
        if exact_count < math.ceil(0.80 * sample_count):
            raise RuntimeError("A1 exact hard corruption coverage is insufficient")
        motion_coverage = manifest.get("motion_exact_coverage")
        if not isinstance(motion_coverage, dict) or any(
            int(motion_coverage[group]["valid_count"])
            < math.ceil(0.80 * int(motion_coverage[group]["sample_count"]))
            for group in ("overall", "rotation", "combined")
        ):
            raise RuntimeError("A1 hard motion-family coverage is insufficient")
    else:
        raise ValueError(f"unknown A1 corruption kind: {kind}")


def _build_maximal_balanced_exact_map(
    indices: np.ndarray,
    keys: Sequence[str],
    sessions: Sequence[str],
    strata: Sequence[object],
    *,
    domain: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build the largest exact cross-session bijection in every stratum."""
    if indices.ndim != 1 or len(set(map(int, indices))) != int(indices.size):
        raise ValueError("A1 hard-map population indices differ")
    if len(keys) != len(sessions) or len(keys) != len(strata):
        raise ValueError("A1 hard-map metadata lengths differ")
    groups: dict[object, list[int]] = {}
    for raw_index in indices:
        index = int(raw_index)
        groups.setdefault(strata[index], []).append(index)
    selected: set[int] = set()
    exclusion_groups: list[dict[str, Any]] = []
    for value, group in sorted(groups.items(), key=lambda item: repr(item[0])):
        by_session: dict[str, list[int]] = {}
        for index in group:
            by_session.setdefault(sessions[index], []).append(index)
        majority_session, majority = max(
            by_session.items(), key=lambda item: (len(item[1]), item[0]),
        )
        other_count = len(group) - len(majority)
        if len(majority) <= other_count:
            keep = set(group)
        else:
            ranked_majority = sorted(
                majority,
                key=lambda index: (
                    hashlib.sha256(
                        f"{domain}\0{value!r}\0{keys[index]}".encode("utf-8")
                    ).digest(),
                    keys[index],
                ),
            )
            keep = {
                index for index in group if sessions[index] != majority_session
            }
            keep.update(ranked_majority[:other_count])
        selected.update(keep)
        excluded_count = len(group) - len(keep)
        if excluded_count:
            exclusion_groups.append({
                "stratum": repr(value),
                "sample_count": len(group),
                "selected_count": len(keep),
                "excluded_count": excluded_count,
                "session_count": len(by_session),
                "max_session_count": len(majority),
                "other_session_count": other_count,
            })
    selected_indices = np.asarray(
        [int(index) for index in indices if int(index) in selected],
        dtype=np.int64,
    )
    if selected_indices.size < 2:
        raise RuntimeError("A1 hard map has no exact cross-session subset")
    recipients, donors, manifest = _build_map(
        selected_indices, keys, sessions, domain=domain, strata=strata,
        require_cross_session=True,
    )
    if (
        int(manifest["valid_count"]) != int(selected_indices.size)
        or manifest["exact_invalid_groups"]
        or int(manifest["relaxed_valid_count"]) != 0
    ):
        raise RuntimeError("A1 balanced exact hard map is not fully bijective")
    manifest.update({
        "sample_count": int(indices.size),
        "valid_count": int(recipients.size),
        "coverage": float(recipients.size / indices.size),
        "exact_valid_count": int(recipients.size),
        "balancing_policy": HARD_MAP_POLICY,
        "maximal_exact_selected_count": int(selected_indices.size),
        "unavoidable_exclusion_count": int(indices.size - selected_indices.size),
        "unavoidable_exclusion_groups": exclusion_groups,
        "population_index_sha256": hashlib.sha256(
            indices.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "selected_index_sha256": hashlib.sha256(
            selected_indices.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
    })
    return recipients, donors, manifest


def _attach_motion_exact_coverage(
    manifest: dict[str, Any],
    recipients: np.ndarray,
    indices: np.ndarray,
    motion: np.ndarray,
) -> None:
    manifest["motion_exact_coverage"] = {}
    for group, value in (("overall", None), ("rotation", 2), ("combined", 3)):
        population = indices if value is None else indices[motion[indices] == value]
        selected = (
            recipients if value is None
            else recipients[motion[recipients] == value]
        )
        if population.size == 0:
            raise RuntimeError(f"A1 empty {group} corruption population")
        manifest["motion_exact_coverage"][group] = {
            "sample_count": int(population.size),
            "valid_count": int(selected.size),
            "coverage": float(selected.size / population.size),
        }


def _build_corruption_maps(
    indices: np.ndarray,
    keys: Sequence[str],
    sessions: Sequence[str],
    hard_strata: Sequence[object],
    motion: np.ndarray,
    *,
    domain: str,
    validate: bool = True,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]], dict[str, dict[str, Any]],
]:
    global_recipient, global_donor, global_manifest = _build_map(
        indices, keys, sessions, domain=f"{domain}/global", strata=None,
        require_cross_session=True,
    )
    hard_recipient, hard_donor, hard_manifest = _build_maximal_balanced_exact_map(
        indices, keys, sessions, hard_strata,
        domain=f"{domain}/matched-hard",
    )
    for manifest, recipients in (
        (global_manifest, global_recipient), (hard_manifest, hard_recipient),
    ):
        _attach_motion_exact_coverage(manifest, recipients, indices, motion)
    if validate:
        _validate_corruption_manifest(global_manifest, kind="global")
        _validate_corruption_manifest(hard_manifest, kind="hard")
    return {
        "global": (global_recipient, global_donor),
        "hard": (hard_recipient, hard_donor),
    }, {"global": global_manifest, "hard": hard_manifest}


def _finalize_preflight_failure(
    output: Path,
    *,
    contract_sha: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    result = {
        **details,
        "status": "preflight_failed",
        "validation_accessed": False,
        "test_accessed": False,
        "future_modules_loaded": False,
        "authorized_a1_counterfactual_probe": False,
        "authorized_formal_two_stage": False,
    }
    _atomic_json(output / "screen_result.json", result)
    _atomic_json(output / "run_state.json", {
        "schema_version": RUN_SCHEMA, "status": "preflight_failed",
        "validation_accessed": False, "test_accessed": False,
        "future_modules_loaded": False,
        "experiment_contract_sha256": contract_sha,
        "screen_result_sha256": sha256_file(output / "screen_result.json"),
    })
    return result


def validate_a0_rejection(path: str | Path) -> dict[str, Any]:
    """Accept only a hash-bound A0 rejection that never opened validation."""
    result_path = Path(path).resolve()
    if result_path.is_dir():
        result_path = result_path / "screen_result.json"
    root = result_path.parent
    run_state_path = root / "run_state.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
    if (
        result.get("schema_version") != A0_RUN_SCHEMA
        or run_state.get("schema_version") != A0_RUN_SCHEMA
        or result.get("status") != "cv_failed"
        or run_state.get("status") != "cv_failed"
        or result.get("diagnostic_only") is not True
        or result.get("truth_omega_forward_input") is not True
        or result.get("formal_v15") is not False
        or result.get("validation_accessed") is not False
        or result.get("test_accessed") is not False
        or result.get("future_modules_loaded") is not False
        or result.get("authorized_a1_endpoint_probe") is not True
        or result.get("authorized_formal_two_stage") is not False
        or run_state.get("validation_claimed") is not False
        or run_state.get("validation_consumed") is not False
        or run_state.get("test_accessed") is not False
    ):
        raise ValueError("A1 requires a sealed train-only A0 rejection")
    if run_state.get("screen_result_sha256") != sha256_file(result_path):
        raise ValueError("A0 rejection result/run-state binding differs")
    if len(result.get("cv", [])) != FIXED_FOLDS or any(
        item.get("passed") is not False for item in result["cv"]
    ):
        raise ValueError("A0 rejection fold state differs")
    source_paths = {
        "runner": Path(__file__).with_name(
            "screen_frozen_expert_reliability_oracle_omega.py"
        ).resolve(),
        "fusion": Path(__file__).with_name(
            "frozen_expert_reliability_fusion.py"
        ).resolve(),
        "split_truth": Path(__file__).with_name(
            "split_scoped_truth_supervision.py"
        ).resolve(),
    }
    if {
        name: sha256_file(source) for name, source in source_paths.items()
    } != result["experiment_contract"].get("source_sha256"):
        raise ValueError("A0 rejection implementation source differs")
    scope_sha = result.get("validation_scope_sha256")
    parent_checkpoint = Path(result["parent"]["checkpoint"]).resolve()
    ledger = (
        parent_checkpoint.parent.parent.parent
        / "_v15-validation-access-ledger" / f"{scope_sha}.json"
    )
    if "validation_ledger" in result or ledger.exists():
        raise ValueError("A0 rejection unexpectedly has a validation claim")
    return result


def validate_a1_p0_artifacts(
    output: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed before a passed P0 may authorize its counterfactual probe."""
    root = Path(output).resolve()
    result_path = root / "screen_result.json"
    run_state_path = root / "run_state.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
    if (
        result.get("schema_version") != RUN_SCHEMA
        or run_state.get("schema_version") != RUN_SCHEMA
        or result.get("status") != "passed"
        or run_state.get("status") != "passed"
        or result.get("diagnostic_only") is not True
        or result.get("truth_omega_feature_construction") is not True
        or result.get("formal_v15") is not False
        or result.get("train_only") is not True
        or result.get("validation_accessed") is not False
        or result.get("test_accessed") is not False
        or result.get("future_modules_loaded") is not False
        or result.get("authorized_a1_counterfactual_probe") is not True
        or result.get("authorized_formal_two_stage") is not False
        or run_state.get("validation_accessed") is not False
        or run_state.get("test_accessed") is not False
        or run_state.get("future_modules_loaded") is not False
    ):
        raise ValueError("A1-P0 artifact status/scope differs")
    contract = result.get("experiment_contract")
    if not isinstance(contract, dict):
        raise ValueError("A1-P0 experiment contract is missing")
    if (
        contract.get("schema_version") != RUN_SCHEMA
        or contract.get("diagnostic_only") is not True
        or contract.get("truth_omega_feature_construction") is not True
        or contract.get("train_only") is not True
        or contract.get("folds") != FIXED_FOLDS
        or contract.get("updates") != FIXED_UPDATES
        or contract.get("width") != FIXED_WIDTH
        or contract.get("seed") != 20260730
        or contract.get("batch_size") != 192
        or contract.get("expert_batch_size") != 96
        or contract.get("role_embedding") is not False
        or contract.get("symmetric_set_pooling") is not True
        or contract.get("absolute_energy_monotonicity") is not False
        or contract.get("hard_map_policy") != HARD_MAP_POLICY
    ):
        raise ValueError("A1-P0 fixed experiment contract differs")
    contract_sha = _json_sha256(contract)
    if (
        result.get("experiment_contract_sha256") != contract_sha
        or run_state.get("experiment_contract_sha256") != contract_sha
    ):
        raise ValueError("A1-P0 experiment contract hash differs")
    if {
        name: sha256_file(path) for name, path in _a1_source_paths().items()
    } != contract.get("source_sha256"):
        raise ValueError("A1-P0 implementation source differs")
    result_sha = sha256_file(result_path)
    if run_state.get("screen_result_sha256") != result_sha:
        raise ValueError("A1-P0 result/run-state binding differs")
    a0_path = Path(result["a0"]["result"]).resolve()
    if (
        sha256_file(a0_path) != result["a0"].get("sha256")
        or result["a0"].get("sha256") != contract.get("a0_result_sha256")
    ):
        raise ValueError("A1-P0 A0 result binding differs")
    a0_result = validate_a0_rejection(a0_path)
    parent_path = Path(result["parent"]["checkpoint"]).resolve()
    parent_sha = sha256_file(parent_path)
    if (
        parent_sha != result["parent"].get("sha256")
        or parent_sha != contract.get("parent_checkpoint_sha256")
        or parent_sha != a0_result["parent"].get("sha256")
    ):
        raise ValueError("A1-P0 parent checkpoint binding differs")
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent_state_sha = state_dict_sha256(parent_payload["model"])
    if (
        parent_state_sha != result["parent"].get("model_state_dict_sha256")
        or parent_state_sha != contract.get("parent_model_state_dict_sha256")
    ):
        raise ValueError("A1-P0 parent model state binding differs")
    frozen_paths = contract.get("frozen_checkpoint_paths")
    frozen_hashes = contract.get("frozen_checkpoint_sha256")
    frozen_states = contract.get("frozen_state_dict_sha256")
    if not all(isinstance(value, dict) for value in (
        frozen_paths, frozen_hashes, frozen_states,
    )):
        raise ValueError("A1-P0 frozen dependency contract is missing")
    resolved_frozen = {
        name: Path(frozen_paths[name]).resolve() for name in ("mapper", "s", "h")
    }
    if {
        name: sha256_file(path) for name, path in resolved_frozen.items()
    } != frozen_hashes:
        raise ValueError("A1-P0 frozen checkpoint files differ")
    mapper, _ = load_frozen_pnp_mapper(resolved_frozen["mapper"])
    s_model, _ = load_frozen_v19(resolved_frozen["s"])
    h_model, _ = load_frozen_hypothesis_adapter(
        resolved_frozen["h"], allow_diagnostic=True,
    )
    if {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    } != frozen_states:
        raise ValueError("A1-P0 frozen dependency states differ")
    dataset_manifest_path = Path(contract["dataset_manifest_path"]).resolve()
    truth_manifest_path = Path(contract["truth_manifest_path"]).resolve()
    if (
        result.get("dataset_manifest_sha256")
        != contract.get("dataset_manifest_sha256")
        or sha256_file(dataset_manifest_path)
        != contract.get("dataset_manifest_sha256")
        or result.get("truth_manifest_sha256")
        != contract.get("truth_manifest_sha256")
        or sha256_file(truth_manifest_path)
        != contract.get("truth_manifest_sha256")
    ):
        raise ValueError("A1-P0 dataset/truth binding differs")
    cv = result.get("cv")
    if (
        not isinstance(cv, list) or len(cv) != FIXED_FOLDS
        or any(item.get("passed") is not True for item in cv)
    ):
        raise ValueError("A1-P0 fold gates differ")
    for item in cv:
        recomputed = _fold_gates(item["metrics"])
        if item.get("gates") != recomputed or not _all_boolean_gates_pass(
            recomputed
        ):
            raise ValueError("A1-P0 recomputed fold gates differ")
        for maps in (item["train_maps"], item["heldout_maps"]):
            _validate_corruption_manifest(maps["global"], kind="global")
            _validate_corruption_manifest(maps["hard"], kind="hard")
    recomputed_cross = _cross_fold_gates(cv)
    if (
        result.get("cross_fold_gates") != recomputed_cross
        or not _all_boolean_gates_pass(recomputed_cross)
    ):
        raise ValueError("A1-P0 recomputed cross-fold gates differ")
    checkpoint_path = Path(result["fold_checkpoint"]).resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    if (
        checkpoint_path.parent != root
        or checkpoint_sha != result.get("fold_checkpoint_sha256")
        or checkpoint_sha != run_state.get("fold_checkpoint_sha256")
    ):
        raise ValueError("A1-P0 fold checkpoint binding differs")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    states = payload.get("fold_models")
    state_hashes = payload.get("fold_state_dict_sha256")
    if (
        payload.get("schema_version") != RUN_SCHEMA
        or payload.get("diagnostic_only") is not True
        or payload.get("truth_omega_feature_construction") is not True
        or payload.get("experiment_contract") != contract
        or payload.get("experiment_contract_sha256") != contract_sha
        or payload.get("a0_result_sha256") != result["a0"]["sha256"]
        or payload.get("parent_checkpoint_sha256") != parent_sha
        or not isinstance(states, list) or len(states) != FIXED_FOLDS
        or state_hashes != [state_dict_sha256(state) for state in states]
        or result.get("fold_state_dict_sha256") != state_hashes
    ):
        raise ValueError("A1-P0 checkpoint payload differs")
    for state in states:
        model = EndpointTokenReliabilityFusion(width=int(contract["width"]))
        model.load_state_dict(state, strict=True)
        if not bool(model.normalizer_fitted):
            raise ValueError("A1-P0 fold checkpoint normalizer is not fitted")
        for name in ("event", "role", "pair", "global"):
            mean = getattr(model, f"{name}_mean")
            scale = getattr(model, f"{name}_scale")
            if (
                not bool(torch.isfinite(mean).all())
                or not bool(torch.isfinite(scale).all())
                or not bool((scale > 0).all())
            ):
                raise ValueError("A1-P0 fold checkpoint normalizer is invalid")
    return payload, result


def _endpoint_expert_arm(
    base: CenterPriorProfiledTwistScreen,
    prepared: dict[str, torch.Tensor],
    recipients: np.ndarray,
    donors: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    if recipients.shape != donors.shape or recipients.ndim != 1:
        raise ValueError("A1 recipient/donor indices differ")
    names = (
        *FEATURE_NAMES, "q0_velocity", "history_velocity", "old_velocity",
        "q0_supported", "history_supported", "fallback_velocity",
        "fallback_supported", "target_velocity", "motion_class",
        "alignment_shift",
    )
    output: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    for start in range(0, len(recipients), batch_size):
        stop = min(start + batch_size, len(recipients))
        recipient = torch.from_numpy(recipients[start:stop]).long()
        donor = torch.from_numpy(donors[start:stop]).long()
        state = {
            name: prepared[name][recipient].to(device, non_blocking=True)
            for name in CENTER_TWIST_FORWARD_FIELDS
        }
        relation, supported, alignment_shift = _aligned_donor_q0(
            prepared, recipient, donor,
        )
        state["q0_relation_m"] = relation.to(device, non_blocking=True)
        state["q0_supported"] = supported.to(device, non_blocking=True)
        target = prepared[MOTION_TARGET_FIELD][recipient].to(
            device, non_blocking=True,
        )
        profiled = base.forward_at_omega(state, target[:, 3])
        center = base.estimate_center(
            state["q0_relation_m"], state["q0_supported"],
        )
        profile_prior = {**center, "q0_relation_m": state["q0_relation_m"]}
        profile_args = (
            state["history_obs_rel_m"], state["history_obs_mask"],
            state["history_event_mask"], state["history_time_s"],
            target[:, 3], profile_prior,
        )
        q0_profile = base.profile(*profile_args, use_q0_prior=True)
        history_profile = base.profile(*profile_args, use_q0_prior=False)
        features = endpoint_token_features(
            history_obs_rel_m=state["history_obs_rel_m"],
            history_obs_mask=state["history_obs_mask"],
            history_event_mask=state["history_event_mask"],
            history_time_s=state["history_time_s"],
            history_primary_mask=state["history_primary_mask"],
            history_switch_step=state["history_switch_step"],
            q0_relation_m=state["q0_relation_m"],
            q0_supported=state["q0_supported"],
            omega_rad_s=target[:, 3],
            q0_velocity_mps=q0_profile["velocity_mps"],
            history_velocity_mps=history_profile["velocity_mps"],
            q0_center_offset_m=q0_profile["profiled_center_offset_m"],
            history_center_offset_m=history_profile["profiled_center_offset_m"],
            pooled_diagnostic_feature=profiled["component_gate_feature"],
        )
        values = {
            **features,
            "q0_velocity": q0_profile["velocity_mps"],
            "history_velocity": history_profile["velocity_mps"],
            "old_velocity": profiled["velocity_mps"],
            "q0_supported": q0_profile["profile_supported"],
            "history_supported": history_profile["profile_supported"],
            "fallback_velocity": q0_profile["fallback_velocity_mps"],
            "fallback_supported": (
                q0_profile["fallback_supported"]
                & history_profile["fallback_supported"]
            ),
            "target_velocity": target[:, :3],
            "motion_class": prepared["motion_class"][recipient].to(device),
            "alignment_shift": alignment_shift.to(device),
        }
        for name, value in values.items():
            output[name].append(value.detach().cpu())
    arm = {name: torch.cat(values, dim=0) for name, values in output.items()}
    arm.update(oracle_projection_coefficient(
        arm["target_velocity"], arm["q0_velocity"], arm["history_velocity"],
        arm["q0_supported"], arm["history_supported"],
    ))
    arm["recipient_global"] = torch.from_numpy(recipients.copy()).long()
    arm["donor_global"] = torch.from_numpy(donors.copy()).long()
    return arm


def _make_arms(
    base: CenterPriorProfiledTwistScreen,
    prepared: dict[str, torch.Tensor],
    intact_full: dict[str, torch.Tensor],
    indices: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
    map_indices: dict[str, tuple[np.ndarray, np.ndarray]],
    map_manifests: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    global_recipient, global_donor = map_indices["global"]
    hard_recipient, hard_donor = map_indices["hard"]
    global_manifest = map_manifests["global"]
    hard_manifest = map_manifests["hard"]
    _validate_corruption_manifest(global_manifest, kind="global")
    _validate_corruption_manifest(hard_manifest, kind="hard")
    arms = {
        "intact": _slice_arm(intact_full, indices),
        "global": _endpoint_expert_arm(
            base, prepared, global_recipient, global_donor, device,
            batch_size=batch_size,
        ),
        "hard": _endpoint_expert_arm(
            base, prepared, hard_recipient, hard_donor, device,
            batch_size=batch_size,
        ),
    }
    return arms, {"global": global_manifest, "hard": hard_manifest}


def _model_kwargs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "event_feature": batch["event_feature"],
        "event_mask": batch["event_mask"],
        "role_feature": batch["role_feature"],
        "role_mask": batch["role_mask"],
        "pair_feature": batch["pair_feature"],
        "pair_mask": batch["pair_mask"],
        "global_feature": batch["global_feature"],
        "q0_velocity_mps": batch["q0_velocity"],
        "history_velocity_mps": batch["history_velocity"],
        "q0_supported": batch["q0_supported"],
        "history_supported": batch["history_supported"],
        "fallback_velocity_mps": batch["fallback_velocity"],
        "fallback_supported": batch["fallback_supported"],
    }


def _arm_batch(
    arm: dict[str, torch.Tensor], rows: torch.Tensor, device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: value[rows].to(device, non_blocking=True)
        for name, value in arm.items()
    }


def _pair_common_mask(
    intact_batch: dict[str, torch.Tensor],
    intact_fused: dict[str, torch.Tensor],
    corrupt_batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    return (
        intact_batch["oracle_valid"]
        & intact_fused["learned_gate_valid"]
        & corrupt_batch["oracle_valid"]
        & corrupt_batch["q0_supported"]
        & corrupt_batch["history_supported"]
        & corrupt_batch["role_mask"].any(dim=-1)
    )


def _train_head(
    arms: dict[str, dict[str, torch.Tensor]],
    *,
    device: torch.device,
    seed: int,
    updates: int,
    width: int,
    batch_size: int,
) -> tuple[EndpointTokenReliabilityFusion, list[dict[str, float]]]:
    model = EndpointTokenReliabilityFusion(width=width).to(device)
    concatenated = {
        name: torch.cat([arm[name] for arm in arms.values()], dim=0).to(device)
        for name in FEATURE_NAMES
    }
    common = torch.cat([
        arm["q0_supported"] & arm["history_supported"]
        & arm["role_mask"].any(dim=-1)
        for arm in arms.values()
    ], dim=0).to(device)
    model.fit_normalizer(**concatenated, row_mask=common)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    arm_weight = {"intact": 0.50, "global": 0.25, "hard": 0.25}
    logs: list[dict[str, float]] = []
    model.train()
    for update in range(1, updates + 1):
        objective = torch.zeros((), device=device)
        predictions: dict[
            str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        for name, arm in arms.items():
            count = arm["global_feature"].shape[0]
            rows = torch.randint(
                count, (min(batch_size, count),), generator=generator,
            )
            batch = _arm_batch(arm, rows, device)
            fused = model(**_model_kwargs(batch))
            valid = batch["oracle_valid"] & fused["learned_gate_valid"]
            if not bool(valid.any()):
                raise RuntimeError(f"A1 has no oracle-valid {name} rows")
            coefficient_loss = F.smooth_l1_loss(
                fused["q0_weight"][valid], batch["weight"][valid], beta=0.10,
            )
            velocity_loss = vector_huber_loss(
                fused["velocity_mps"][valid],
                batch["target_velocity"][valid], beta=0.25,
            )
            objective = objective + arm_weight[name] * (
                coefficient_loss + velocity_loss
            )
            predictions[name] = (rows, fused["q0_weight"], batch["weight"])
        intact = arms["intact"]
        intact_lookup = {
            int(value): index for index, value in enumerate(
                intact["recipient_global"].tolist()
            )
        }
        pair_loss = torch.zeros((), device=device)
        for name in ("global", "hard"):
            rows, corrupt_weight, corrupt_oracle = predictions[name]
            recipients = arms[name]["recipient_global"][rows].tolist()
            intact_rows = torch.tensor([
                intact_lookup[int(value)] for value in recipients
            ], dtype=torch.long)
            intact_batch = _arm_batch(intact, intact_rows, device)
            intact_fused = model(**_model_kwargs(intact_batch))
            corrupt_batch = _arm_batch(arms[name], rows, device)
            pair_valid = _pair_common_mask(
                intact_batch, intact_fused, corrupt_batch,
            )
            if not bool(pair_valid.any()):
                raise RuntimeError(f"A1 has no valid {name} pairs")
            pair_loss = pair_loss + 0.5 * F.smooth_l1_loss(
                (intact_fused["q0_weight"] - corrupt_weight)[pair_valid],
                (intact_batch["weight"] - corrupt_oracle)[pair_valid],
                beta=0.10,
            )
        objective = objective + 0.5 * pair_loss
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if any(
            parameter.grad is None
            or not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise RuntimeError("A1 head has missing/non-finite gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        phase = update / updates
        learning_rate = 1e-4 + 0.5 * (1e-3 - 1e-4) * (
            1.0 + math.cos(math.pi * phase)
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        if update in {1, updates // 2, updates}:
            logs.append({
                "update": float(update),
                "objective": float(objective.detach()),
                "learning_rate": learning_rate,
            })
    return model.eval(), logs


@torch.inference_mode()
def _evaluate_head(
    model: EndpointTokenReliabilityFusion,
    arms: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    result: dict[str, Any] = {"arms": {}}
    predictions: dict[str, dict[str, torch.Tensor]] = {}
    for name, arm in arms.items():
        batch = {key: value.to(device) for key, value in arm.items()}
        fused = model(**_model_kwargs(batch))
        ablated = model(**_model_kwargs(batch), ablate_local=True)
        valid = batch["oracle_valid"] & fused["learned_gate_valid"]
        oracle_velocity = (
            batch["weight"].unsqueeze(-1) * batch["q0_velocity"]
            + (1.0 - batch["weight"]).unsqueeze(-1)
            * batch["history_velocity"]
        )
        q0_error = torch.linalg.vector_norm(
            batch["q0_velocity"] - batch["target_velocity"], dim=-1,
        )
        history_error = torch.linalg.vector_norm(
            batch["history_velocity"] - batch["target_velocity"], dim=-1,
        )
        predictions[name] = {
            "weight": fused["q0_weight"].cpu(),
            "recipient_global": arm["recipient_global"],
            "valid": valid.cpu(), "motion_class": arm["motion_class"],
        }
        arm_result: dict[str, Any] = {
            "full_count": int(valid.numel()),
            "common_count": int(valid.sum()),
            "common_fraction": float(valid.double().mean()),
            "state_supported_count": int(fused["state_supported"].sum()),
            "state_supported_fraction": float(
                fused["state_supported"].double().mean()
            ),
            "fallback_count": int(fused["fallback_used"].sum()),
            "groups": {},
        }
        for group, group_mask in {
            "overall": torch.ones_like(valid),
            "rotation": batch["motion_class"] == 2,
            "combined": batch["motion_class"] == 3,
        }.items():
            common = valid & group_mask
            if not bool(common.any()):
                raise RuntimeError(f"A1 empty {name}/{group} common population")

            def error(velocity: torch.Tensor) -> torch.Tensor:
                return torch.linalg.vector_norm(
                    velocity[common] - batch["target_velocity"][common], dim=-1,
                )

            preference = q0_error[common] < history_error[common]
            full_auc = _binary_auc(
                fused["q0_weight"][common].cpu(), preference.cpu(),
            )
            ablated_auc = _binary_auc(
                ablated["q0_weight"][common].cpu(), preference.cpu(),
            )
            coefficient_mae = float((
                fused["q0_weight"][common] - batch["weight"][common]
            ).abs().mean())
            ablated_mae = float((
                ablated["q0_weight"][common] - batch["weight"][common]
            ).abs().mean())
            group_result: dict[str, Any] = {
                "full_count": int(group_mask.sum()),
                "common_count": int(common.sum()),
                "common_fraction": float(
                    common.sum().double() / group_mask.sum().clamp_min(1)
                ),
                "fused_error_mps": _distribution(error(fused["velocity_mps"])),
                "parent_error_mps": _distribution(error(batch["old_velocity"])),
                "oracle_error_mps": _distribution(error(oracle_velocity)),
                "blind_error_mps": _distribution(error(batch["history_velocity"])),
                "q0_weight": _distribution(fused["q0_weight"][common]),
                "oracle_weight": _distribution(batch["weight"][common]),
                "component_preference_auc": full_auc,
                "local_ablated_component_preference_auc": ablated_auc,
                "local_auc_gain": full_auc - ablated_auc,
                "coefficient_mae": coefficient_mae,
                "local_ablated_coefficient_mae": ablated_mae,
                "local_mae_relative_improvement": (
                    (ablated_mae - coefficient_mae) / max(ablated_mae, 1e-8)
                ),
                "q0_better_count": int(preference.sum()),
                "history_better_or_equal_count": int((~preference).sum()),
            }
            arm_result["groups"][group] = group_result
        result["arms"][name] = arm_result
    intact_lookup = {
        int(value): index for index, value in enumerate(
            predictions["intact"]["recipient_global"].tolist()
        )
    }
    for name in ("global", "hard"):
        reference = torch.tensor([
            intact_lookup[int(value)]
            for value in predictions[name]["recipient_global"].tolist()
        ], dtype=torch.long)
        paired_valid = (
            predictions["intact"]["valid"][reference]
            & predictions[name]["valid"]
        )
        motion = predictions[name]["motion_class"]
        for group, group_mask in {
            "overall": torch.ones_like(paired_valid),
            "rotation": motion == 2,
            "combined": motion == 3,
        }.items():
            common = paired_valid & group_mask
            if not bool(common.any()):
                raise RuntimeError(f"A1 empty {name}/{group} paired population")
            result["arms"][name]["groups"][group].update({
                "paired_count": int(common.sum()),
                "paired_fraction": float(
                    common.sum().double() / group_mask.sum().clamp_min(1)
                ),
                "paired_weight_separation_mean": float((
                    predictions["intact"]["weight"][reference][common]
                    - predictions[name]["weight"][common]
                ).mean()),
            })
    return result


def _fold_gates(metrics: dict[str, Any]) -> dict[str, bool | float]:
    arms = metrics["arms"]
    gates: dict[str, bool | float] = {}
    for name in ("intact", "global", "hard"):
        gates[f"{name}_state_supported_complete"] = (
            arms[name]["state_supported_count"] == arms[name]["full_count"]
        )
    for group in ("overall", "rotation", "combined"):
        intact = arms["intact"]["groups"][group]
        parent = intact["parent_error_mps"]
        fused = intact["fused_error_mps"]
        oracle = intact["oracle_error_mps"]
        headroom_mean = parent["mean"] - oracle["mean"]
        headroom_p50 = parent["p50"] - oracle["p50"]
        identifiable = headroom_mean >= 0.02 and headroom_p50 >= 0.01
        recovery_mean = (
            (parent["mean"] - fused["mean"]) / headroom_mean
            if identifiable else 0.0
        )
        recovery_p50 = (
            (parent["p50"] - fused["p50"]) / headroom_p50
            if identifiable else 0.0
        )
        coefficient_limit = 0.28 if group == "overall" else 0.32
        gates[f"{group}_intact_oracle_headroom_identifiable"] = identifiable
        gates[f"{group}_intact_oracle_gap_recovery_mean"] = recovery_mean
        gates[f"{group}_intact_oracle_gap_recovery_p50"] = recovery_p50
        gates[f"{group}_intact_recovers_30pct_mean"] = recovery_mean >= 0.30
        gates[f"{group}_intact_recovers_30pct_p50"] = recovery_p50 >= 0.30
        gates[f"{group}_intact_auc_ge_0_75"] = (
            intact["component_preference_auc"] >= 0.75
        )
        gates[f"{group}_intact_coefficient_mae_within_limit"] = (
            intact["coefficient_mae"] <= coefficient_limit
        )
        gates[f"{group}_intact_local_auc_gain_ge_0_03"] = (
            intact["local_auc_gain"] >= 0.03
        )
        gates[f"{group}_intact_local_mae_improves_10pct"] = (
            intact["local_mae_relative_improvement"] >= 0.10
        )
        gates[f"{group}_intact_common_fraction_ge_0_95"] = (
            intact["common_fraction"] >= 0.95
        )
        for name in ("global", "hard"):
            arm = arms[name]["groups"][group]
            gates[f"{group}_{name}_auc_ge_0_75"] = (
                arm["component_preference_auc"] >= 0.75
            )
            gates[f"{group}_{name}_common_fraction_ge_0_95"] = (
                arm["common_fraction"] >= 0.95
            )
            gates[f"{group}_{name}_paired_fraction_ge_0_95"] = (
                arm["paired_fraction"] >= 0.95
            )
            gates[f"{group}_{name}_weight_separation_ge_0_15"] = (
                arm["paired_weight_separation_mean"] >= 0.15
            )
            gates[f"{group}_{name}_within_blind_mean"] = (
                arm["fused_error_mps"]["mean"]
                <= 1.02 * arm["blind_error_mps"]["mean"]
            )
            gates[f"{group}_{name}_within_blind_p50"] = (
                arm["fused_error_mps"]["p50"]
                <= 1.02 * arm["blind_error_mps"]["p50"]
            )
            gates[f"{group}_{name}_p95_guard"] = (
                arm["fused_error_mps"]["p95"]
                <= 1.10 * arm["blind_error_mps"]["p95"]
            )
    return gates


def _cross_fold_gates(cv: list[dict[str, Any]]) -> dict[str, bool | float]:
    gates: dict[str, bool | float] = {}
    for arm in ("intact", "global", "hard"):
        for group in ("overall", "rotation", "combined"):
            left = cv[0]["metrics"]["arms"][arm]["groups"][group]
            right = cv[1]["metrics"]["arms"][arm]["groups"][group]
            auc_delta = abs(
                left["component_preference_auc"]
                - right["component_preference_auc"]
            )
            gates[f"{arm}_{group}_auc_fold_delta"] = auc_delta
            gates[f"{arm}_{group}_auc_fold_delta_le_0_10"] = auc_delta <= 0.10
            if arm != "intact":
                separation_delta = abs(
                    left["paired_weight_separation_mean"]
                    - right["paired_weight_separation_mean"]
                )
                gates[f"{arm}_{group}_separation_fold_delta"] = separation_delta
                gates[f"{arm}_{group}_separation_fold_delta_le_0_05"] = (
                    separation_delta <= 0.05
                )
    return gates


def _all_boolean_gates_pass(gates: dict[str, bool | float]) -> bool:
    return all(value for value in gates.values() if isinstance(value, bool))


def _nonempty_boolean_gates_pass(gates: object) -> bool:
    return (
        isinstance(gates, dict)
        and any(isinstance(value, bool) for value in gates.values())
        and _all_boolean_gates_pass(gates)
    )


def _cv_authorized(
    cv: list[dict[str, Any]], cross_fold_gates: dict[str, bool | float],
) -> bool:
    return (
        len(cv) == FIXED_FOLDS
        and all(
            item.get("passed") is True
            and _nonempty_boolean_gates_pass(item.get("gates"))
            for item in cv
        )
        and _nonempty_boolean_gates_pass(cross_fold_gates)
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V15-A1-P0 requires local CUDA")
    git_start = _git_state()
    if git_start["git_commit"] == "unknown" or git_start["worktree_dirty"]:
        raise RuntimeError("V15-A1-P0 requires a clean known git commit")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing A1 output: {output}")
    output.mkdir(parents=True)
    _atomic_json(output / "run_state.json", {
        "schema_version": RUN_SCHEMA, "status": "initializing",
        "validation_accessed": False, "test_accessed": False,
        "future_modules_loaded": False,
    })
    a0_result_path = Path(args.a0_result).resolve()
    a0_result = validate_a0_rejection(a0_result_path)
    a0_result_sha = sha256_file(
        a0_result_path / "screen_result.json"
        if a0_result_path.is_dir() else a0_result_path
    )
    parent_checkpoint = Path(args.parent_checkpoint).resolve()
    if parent_checkpoint != Path(a0_result["parent"]["checkpoint"]).resolve():
        raise ValueError("A1 parent checkpoint differs from A0")
    parent_root = parent_checkpoint.parent.parent
    parent_manifest_path = parent_root / "run_manifest.json"
    parent_result_path = parent_root / "screen_result.json"
    parent_manifest_sha = sha256_file(parent_manifest_path)
    parent_result_sha = sha256_file(parent_result_path)
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_payload = torch.load(
        parent_checkpoint, map_location="cpu", weights_only=False,
    )
    parent_sha = sha256_file(parent_checkpoint)
    parent_contract = _validate_checkpoint_manifest_identity(
        parent_payload, parent_manifest, parent_sha,
    )
    if parent_sha != a0_result["parent"]["sha256"]:
        raise ValueError("A1 parent checkpoint hash differs from A0")
    parent_args = parent_contract["args"]
    dataset_path = Path(parent_args["dataset"]).resolve()
    truth_path = Path(parent_args["truth_history"]).resolve()
    mapper_path = Path(parent_args["mapper_checkpoint"]).resolve()
    s_path = Path(parent_args["s_checkpoint"]).resolve()
    h_path = Path(parent_args["h_checkpoint"]).resolve()
    _seed(args.seed)
    train_dataset = _load_train_dataset(dataset_path)
    dataset_manifest_path = dataset_path / "dataset_manifest.json"
    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if (
        sha256_file(dataset_manifest_path)
        != a0_result["dataset_manifest_sha256"]
        or manifest.get("test_accessed") is not False
    ):
        raise ValueError("A1 dataset manifest differs or accessed test")
    truth_sha = manifest["truth_history_manifest_sha256"]
    if truth_sha != a0_result["truth_manifest_sha256"]:
        raise ValueError("A1 truth manifest differs from A0")
    train_truth = SplitScopedTruthIndex(
        truth_path, split="train", expected_manifest_sha256=truth_sha,
    )
    train_truth.attach(train_dataset)
    mapper, _ = load_frozen_pnp_mapper(mapper_path)
    s_model, _ = load_frozen_v19(s_path)
    h_model, _ = load_frozen_hypothesis_adapter(h_path, allow_diagnostic=True)
    for frozen in (mapper, s_model, h_model):
        frozen.to(device).eval().requires_grad_(False)
    frozen_state = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    }
    frozen_files = {
        "mapper": sha256_file(mapper_path), "s": sha256_file(s_path),
        "h": sha256_file(h_path),
    }
    if (
        frozen_state != parent_contract["frozen_state_dict_sha256"]
        or frozen_files != parent_contract["frozen_checkpoint_sha256"]
    ):
        raise ValueError("A1 frozen Mapper/S/H differs")
    base = CenterPriorProfiledTwistScreen(
        width=int(parent_args["width"]),
        center_precision=float(parent_args["center_precision"]),
    ).to(device)
    base.load_state_dict(parent_payload["model"], strict=True)
    base.eval().requires_grad_(False)
    base_sha = state_dict_sha256(base.state_dict())
    if base_sha != a0_result["parent"]["model_state_dict_sha256"]:
        raise ValueError("A1 frozen V14 state differs from A0")
    source_paths = _a1_source_paths()
    source_sha = {name: sha256_file(path) for name, path in source_paths.items()}
    experiment_contract = {
        "schema_version": RUN_SCHEMA,
        "diagnostic_only": True,
        "truth_omega_feature_construction": True,
        "train_only": True,
        "a0_result_sha256": a0_result_sha,
        "parent_checkpoint_sha256": parent_sha,
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "dataset_manifest_path": str(dataset_manifest_path),
        "truth_manifest_sha256": truth_sha,
        "truth_manifest_path": str(truth_path / "dataset_manifest.json"),
        "frozen_checkpoint_sha256": frozen_files,
        "frozen_checkpoint_paths": {
            "mapper": str(mapper_path), "s": str(s_path), "h": str(h_path),
        },
        "frozen_state_dict_sha256": frozen_state,
        "parent_model_state_dict_sha256": base_sha,
        "git_commit": git_start["git_commit"],
        "source_sha256": source_sha,
        "folds": FIXED_FOLDS, "updates": args.updates,
        "width": args.width, "seed": args.seed,
        "batch_size": args.batch_size,
        "expert_batch_size": args.expert_batch_size,
        "forward_excludes": [
            "session", "motion_class", "physical_armor_id",
            "truth_velocity", "future", "absolute_pose",
        ],
        "role_embedding": False,
        "symmetric_set_pooling": True,
        "absolute_energy_monotonicity": False,
        "hard_map_policy": HARD_MAP_POLICY,
    }
    contract_sha = _json_sha256(experiment_contract)
    _atomic_json(output / "run_state.json", {
        "schema_version": RUN_SCHEMA, "status": "metadata_preflight",
        "validation_accessed": False, "test_accessed": False,
        "future_modules_loaded": False,
        "experiment_contract_sha256": contract_sha,
    })
    keys, sessions = _combined_metadata(train_dataset)
    prepared = _cache_prepared_state(
        train_dataset, mapper, s_model, h_model, device,
        batch_size=args.expert_batch_size,
    )
    all_indices = np.arange(len(train_dataset), dtype=np.int64)
    support_count = prepared["q0_supported"].sum(dim=-1).numpy().astype(np.int64)
    motion_class = prepared["motion_class"].numpy().astype(np.int64)
    hard_strata = [
        (int(motion_class[index]), int(support_count[index]))
        for index in range(len(train_dataset))
    ]
    fold_sessions = _fold_assignments(sessions, folds=FIXED_FOLDS)
    fold_plan: list[dict[str, Any]] = []
    metadata_preflight: list[dict[str, Any]] = []
    try:
        for fold, heldout_sessions in enumerate(fold_sessions):
            heldout = np.asarray([
                index for index, session in enumerate(sessions)
                if session in heldout_sessions
            ], dtype=np.int64)
            training = np.asarray([
                index for index, session in enumerate(sessions)
                if session not in heldout_sessions
            ], dtype=np.int64)
            if set(sessions[int(index)] for index in training) & heldout_sessions:
                raise RuntimeError("A1 train/heldout sessions overlap")
            audit: dict[str, Any] = {
                "fold": fold,
                "status": "running",
                "train_count": int(training.size),
                "heldout_count": int(heldout.size),
            }
            metadata_preflight.append(audit)
            train_map_indices, train_maps = _build_corruption_maps(
                training, keys, sessions, hard_strata, motion_class,
                domain=f"train/a1/cv{fold}/fit",
                validate=False,
            )
            audit["train_maps"] = train_maps
            try:
                _validate_corruption_manifest(train_maps["global"], kind="global")
                _validate_corruption_manifest(train_maps["hard"], kind="hard")
            except RuntimeError:
                audit["status"] = "failed"
                audit["failed_domain"] = "fit"
                raise
            heldout_map_indices, heldout_maps = _build_corruption_maps(
                heldout, keys, sessions, hard_strata, motion_class,
                domain=f"train/a1/cv{fold}/heldout",
                validate=False,
            )
            audit["heldout_maps"] = heldout_maps
            try:
                _validate_corruption_manifest(
                    heldout_maps["global"], kind="global",
                )
                _validate_corruption_manifest(heldout_maps["hard"], kind="hard")
            except RuntimeError:
                audit["status"] = "failed"
                audit["failed_domain"] = "heldout"
                raise
            audit["status"] = "passed"
            fold_plan.append({
                "fold": fold,
                "heldout_sessions": heldout_sessions,
                "training": training,
                "heldout": heldout,
                "train_map_indices": train_map_indices,
                "heldout_map_indices": heldout_map_indices,
                "train_maps": train_maps,
                "heldout_maps": heldout_maps,
            })
    except RuntimeError as error:
        return _finalize_preflight_failure(output, contract_sha=contract_sha, details={
            "schema_version": RUN_SCHEMA,
            "diagnostic_only": True,
            "truth_omega_feature_construction": True,
            "formal_v15": False,
            "train_only": True,
            "git": git_start,
            "experiment_contract": experiment_contract,
            "experiment_contract_sha256": contract_sha,
            "a0": {
                "result": str(
                    a0_result_path / "screen_result.json"
                    if a0_result_path.is_dir() else a0_result_path
                ),
                "sha256": a0_result_sha,
            },
            "parent": {"checkpoint": str(parent_checkpoint), "sha256": parent_sha},
            "dataset_manifest_sha256": experiment_contract[
                "dataset_manifest_sha256"
            ],
            "truth_manifest_sha256": truth_sha,
            "metadata_preflight": metadata_preflight,
            "preflight_error": str(error),
        })
    _atomic_json(output / "run_state.json", {
        "schema_version": RUN_SCHEMA, "status": "train_cv",
        "validation_accessed": False, "test_accessed": False,
        "future_modules_loaded": False,
        "experiment_contract_sha256": contract_sha,
    })
    intact_full = _endpoint_expert_arm(
        base, prepared, all_indices, all_indices, device,
        batch_size=args.expert_batch_size,
    )
    cv: list[dict[str, Any]] = []
    fold_models: list[dict[str, torch.Tensor]] = []
    for plan in fold_plan:
        fold = int(plan["fold"])
        heldout_sessions = plan["heldout_sessions"]
        training = plan["training"]
        heldout = plan["heldout"]
        train_arms, train_maps = _make_arms(
            base, prepared, intact_full, training, device,
            batch_size=args.expert_batch_size,
            map_indices=plan["train_map_indices"],
            map_manifests=plan["train_maps"],
        )
        heldout_arms, heldout_maps = _make_arms(
            base, prepared, intact_full, heldout, device,
            batch_size=args.expert_batch_size,
            map_indices=plan["heldout_map_indices"],
            map_manifests=plan["heldout_maps"],
        )
        head, logs = _train_head(
            train_arms, device=device, seed=args.seed + 100 + fold,
            updates=args.updates, width=args.width, batch_size=args.batch_size,
        )
        metrics = _evaluate_head(head, heldout_arms, device)
        gates = _fold_gates(metrics)
        passed = _all_boolean_gates_pass(gates)
        cv.append({
            "fold": fold, "passed": passed,
            "heldout_sessions": sorted(heldout_sessions),
            "train_count": int(training.size),
            "heldout_count": int(heldout.size),
            "train_maps": train_maps, "heldout_maps": heldout_maps,
            "logs": logs, "metrics": metrics, "gates": gates,
        })
        fold_models.append({
            name: value.detach().cpu() for name, value in head.state_dict().items()
        })
    cross_fold_gates = _cross_fold_gates(cv)
    passed = _cv_authorized(cv, cross_fold_gates)
    if state_dict_sha256(base.state_dict()) != base_sha:
        raise RuntimeError("A1 changed frozen V14 parent")
    if {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    } != frozen_state:
        raise RuntimeError("A1 changed frozen Mapper/S/H")
    if {
        "mapper": sha256_file(mapper_path), "s": sha256_file(s_path),
        "h": sha256_file(h_path),
    } != frozen_files:
        raise RuntimeError("A1 frozen checkpoint files changed")
    if sha256_file(parent_checkpoint) != parent_sha:
        raise RuntimeError("A1 parent checkpoint changed")
    if (
        sha256_file(parent_manifest_path) != parent_manifest_sha
        or sha256_file(parent_result_path) != parent_result_sha
    ):
        raise RuntimeError("A1 parent metadata changed")
    if sha256_file(dataset_manifest_path) != experiment_contract[
        "dataset_manifest_sha256"
    ]:
        raise RuntimeError("A1 dataset manifest changed")
    if sha256_file(truth_path / "dataset_manifest.json") != truth_sha:
        raise RuntimeError("A1 truth manifest changed")
    assert_manifest_split_shards_unchanged(
        dataset_path, manifest, split="train", label="paired dataset",
    )
    train_truth.assert_unchanged()
    if {name: sha256_file(path) for name, path in source_paths.items()} != source_sha:
        raise RuntimeError("A1 implementation source changed")
    if _git_state() != git_start:
        raise RuntimeError("A1 git state changed during run")
    result: dict[str, Any] = {
        "schema_version": RUN_SCHEMA,
        "status": "passed" if passed else "failed",
        "diagnostic_only": True,
        "truth_omega_feature_construction": True,
        "formal_v15": False,
        "train_only": True,
        "validation_accessed": False,
        "test_accessed": False,
        "future_modules_loaded": False,
        "authorized_a1_counterfactual_probe": passed,
        "authorized_formal_two_stage": False,
        "git": git_start,
        "experiment_contract": experiment_contract,
        "experiment_contract_sha256": contract_sha,
        "a0": {
            "result": str(
                a0_result_path / "screen_result.json"
                if a0_result_path.is_dir() else a0_result_path
            ),
            "sha256": a0_result_sha,
            "status": a0_result["status"],
        },
        "parent": {
            "checkpoint": str(parent_checkpoint), "sha256": parent_sha,
            "model_state_dict_sha256": base_sha,
        },
        "dataset_manifest_sha256": experiment_contract[
            "dataset_manifest_sha256"
        ],
        "truth_manifest_sha256": truth_sha,
        "metadata_preflight": metadata_preflight,
        "cv": cv,
        "cross_fold_gates": cross_fold_gates,
    }
    if passed:
        checkpoint_path = output / "fold-heads.pt"
        _atomic_checkpoint(checkpoint_path, {
            "schema_version": RUN_SCHEMA,
            "diagnostic_only": True,
            "truth_omega_feature_construction": True,
            "experiment_contract": experiment_contract,
            "experiment_contract_sha256": contract_sha,
            "a0_result_sha256": a0_result_sha,
            "parent_checkpoint_sha256": parent_sha,
            "fold_models": fold_models,
            "fold_state_dict_sha256": [
                state_dict_sha256(state) for state in fold_models
            ],
        })
        result["fold_checkpoint"] = str(checkpoint_path)
        result["fold_checkpoint_sha256"] = sha256_file(checkpoint_path)
        result["fold_state_dict_sha256"] = [
            state_dict_sha256(state) for state in fold_models
        ]
    _atomic_json(output / "screen_result.json", result)
    _atomic_json(output / "run_state.json", {
        "schema_version": RUN_SCHEMA, "status": result["status"],
        "validation_accessed": False, "test_accessed": False,
        "future_modules_loaded": False,
        "experiment_contract_sha256": contract_sha,
        "screen_result_sha256": sha256_file(output / "screen_result.json"),
        "fold_checkpoint_sha256": result.get("fold_checkpoint_sha256"),
    })
    if passed:
        validate_a1_p0_artifacts(output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0-result", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--updates", type=int, default=FIXED_UPDATES)
    parser.add_argument("--width", type=int, default=FIXED_WIDTH)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--expert-batch-size", type=int, default=96)
    args = parser.parse_args()
    if args.updates != FIXED_UPDATES or args.width != FIXED_WIDTH:
        raise ValueError("V15-A1-P0 updates/width are fixed")
    if args.seed != 20260730:
        raise ValueError("V15-A1-P0 seed is fixed")
    if args.batch_size != 192 or args.expert_batch_size != 96:
        raise ValueError("V15-A1-P0 batch sizes are fixed")
    return args


def main() -> None:
    result = run(parse_args())
    print(json.dumps({
        "status": result["status"],
        "authorized_a1_counterfactual_probe": result[
            "authorized_a1_counterfactual_probe"
        ],
        "cross_fold_gates": result["cross_fold_gates"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
