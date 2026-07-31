"""V15-A0 frozen-expert reliability audit with an explicit truth-omega scope.

This is a mechanism-sufficiency diagnostic, not a formal deployable state
estimator: the frozen profilers receive loss truth omega.  Model selection is
performed only by session-grouped train cross-validation.  Validation is
opened exactly once and only if every train fold passes the predeclared gates.
No future module or test split is loaded.
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
from torch.utils.data import DataLoader

from .center_offset_supervision import CenterTruthIndex
from .cyclic_future_foundation import load_frozen_v19
from .evaluate_profiled_center_twist_mechanism import (
    _validate_checkpoint_manifest_identity,
)
from .frozen_expert_reliability_fusion import (
    MonotoneReliabilityFusion,
    deterministic_cross_session_derangement,
    oracle_projection_coefficient,
    vector_huber_loss,
)
from .motion_truth_supervision import MOTION_TARGET_FIELD, MotionTruthIndex
from .observable_future_pnp_ab import sha256_file, state_dict_sha256
from .pnp_q0_hypothesis_adapter import (
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .profiled_center_twist_future import (
    CENTER_TWIST_FORWARD_FIELDS,
    CenterPriorProfiledTwistScreen,
)
from .train_anonymous_vehicle_motion import _dataset, _json_sha256
from .train_causal_physical_ab import _git_state, _seed, _to_device
from .train_profiled_center_prior_screen import (
    _assert_manifest_shards_unchanged,
    _prepare,
)
from .train_pnp_window_mapper_distillation import _atomic_checkpoint, _atomic_json


RUN_SCHEMA = "stage3-frozen-reliability-oracle-omega-v15-a0"
FIXED_FOLDS = 2
DEFAULT_UPDATES = 200


def validate_reliability_a0_artifacts(
    output: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail-closed loader for a passed V15-A0 diagnostic artifact set."""
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
        or result.get("truth_omega_forward_input") is not True
        or result.get("formal_v15") is not False
    ):
        raise ValueError("V15-A0 artifact status/schema/scope differs")
    contract = result.get("experiment_contract")
    if not isinstance(contract, dict):
        raise ValueError("V15-A0 experiment contract is missing")
    contract_sha = _json_sha256(contract)
    if (
        result.get("experiment_contract_sha256") != contract_sha
        or run_state.get("experiment_contract_sha256") != contract_sha
    ):
        raise ValueError("V15-A0 experiment contract hash differs")
    checkpoint_path = Path(result["reliability_checkpoint"]).resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    if (
        checkpoint_path.parent != root
        or result.get("reliability_checkpoint_sha256") != checkpoint_sha
        or run_state.get("reliability_checkpoint_sha256") != checkpoint_sha
    ):
        raise ValueError("V15-A0 reliability checkpoint binding differs")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != RUN_SCHEMA
        or payload.get("diagnostic_only") is not True
        or payload.get("truth_omega_forward_input") is not True
        or payload.get("experiment_contract") != contract
        or payload.get("experiment_contract_sha256") != contract_sha
        or payload.get("parent_checkpoint_sha256")
        != result["parent"]["sha256"]
    ):
        raise ValueError("V15-A0 checkpoint contract/parent differs")
    if state_dict_sha256(payload["model"]) != result.get(
        "reliability_state_dict_sha256"
    ):
        raise ValueError("V15-A0 reliability state hash differs")
    result_sha = sha256_file(result_path)
    if run_state.get("screen_result_sha256") != result_sha:
        raise ValueError("V15-A0 run-state result hash differs")
    ledger_path = Path(result["validation_ledger"]).resolve()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if (
        ledger.get("schema_version") != RUN_SCHEMA
        or ledger.get("status") != "consumed"
        or ledger.get("experiment_contract_sha256") != contract_sha
        or ledger.get("validation_scope_sha256")
        != result.get("validation_scope_sha256")
        or ledger.get("screen_result_sha256") != result_sha
        or ledger.get("reliability_checkpoint_sha256") != checkpoint_sha
        or ledger.get("parent_checkpoint_sha256") != result["parent"]["sha256"]
    ):
        raise ValueError("V15-A0 validation ledger binding differs")
    return payload, result


def _distribution(values: torch.Tensor) -> dict[str, float | int]:
    data = values.detach().cpu().numpy().astype(np.float64, copy=False)
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError("reliability distribution is empty or non-finite")
    return {
        "count": int(data.size), "mean": float(data.mean()),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
    }


def _binary_auc(score: torch.Tensor, label: torch.Tensor) -> float:
    values = score.detach().cpu().numpy().astype(np.float64, copy=False)
    target = label.detach().cpu().numpy().astype(np.bool_, copy=False)
    positive, negative = int(target.sum()), int((~target).sum())
    if positive == 0 or negative == 0:
        return 0.5
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return float(
        (ranks[target].sum() - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def _combined_metadata(dataset: Any) -> tuple[list[str], list[str]]:
    keys: list[str] = []
    sessions: list[str] = []
    for part in dataset.parts:
        for session, t0_ns in zip(part.session_ids, part.t0_ns):
            sessions.append(str(session))
            keys.append(f"{part.split}\x1f{session}\x1f{int(t0_ns)}")
    if len(keys) != len(dataset) or len(set(keys)) != len(keys):
        raise ValueError("combined dataset key metadata differs")
    return keys, sessions


def _current_primary_index(
    prepared: dict[str, torch.Tensor], indices: torch.Tensor,
) -> torch.Tensor:
    event_mask = prepared["history_event_mask"][indices].to(torch.bool)
    time_s = prepared["history_time_s"][indices]
    primary_mask = prepared["history_primary_mask"][indices].to(torch.bool)
    if bool(torch.any(event_mask & (primary_mask.sum(dim=-1) != 1))):
        raise ValueError("active event has a non-unique primary role")
    current_event = torch.where(
        event_mask, time_s, torch.full_like(time_s, -torch.inf),
    ).argmax(dim=1)
    row = torch.arange(indices.shape[0])
    return primary_mask[row, current_event].to(torch.int64).argmax(dim=-1)


def _aligned_donor_q0(
    prepared: dict[str, torch.Tensor],
    recipients: torch.Tensor, donors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recipient_primary = _current_primary_index(prepared, recipients)
    donor_primary = _current_primary_index(prepared, donors)
    shift = torch.remainder(recipient_primary - donor_primary, 4)
    target_role = torch.arange(4).unsqueeze(0).expand(recipients.shape[0], -1)
    donor_role = torch.remainder(target_role - shift.unsqueeze(-1), 4)
    relation = torch.gather(
        prepared["q0_relation_m"][donors], 1,
        donor_role.unsqueeze(-1).expand(-1, -1, 3),
    )
    supported = torch.gather(
        prepared["q0_supported"][donors], 1, donor_role,
    )
    return relation, supported, shift


@torch.inference_mode()
def _cache_prepared_state(
    dataset: Any, mapper: torch.nn.Module, s_model: torch.nn.Module,
    h_model: torch.nn.Module, device: torch.device, *, batch_size: int,
) -> dict[str, torch.Tensor]:
    names = (*CENTER_TWIST_FORWARD_FIELDS, MOTION_TARGET_FIELD, "motion_class")
    cached: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=True,
    )
    for raw_cpu in loader:
        raw = _to_device(raw_cpu, device)
        batch = _prepare(mapper, s_model, h_model, raw)
        for name in names:
            cached[name].append(batch[name].detach().cpu())
    result = {name: torch.cat(values, dim=0) for name, values in cached.items()}
    if result[MOTION_TARGET_FIELD].shape[0] != len(dataset):
        raise RuntimeError("prepared reliability cache count differs")
    return result


@torch.inference_mode()
def _expert_arm(
    base: CenterPriorProfiledTwistScreen,
    prepared: dict[str, torch.Tensor],
    recipients: np.ndarray,
    donors: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    if recipients.shape != donors.shape or recipients.ndim != 1:
        raise ValueError("expert recipient/donor indices differ")
    output: dict[str, list[torch.Tensor]] = {
        name: [] for name in (
            "feature", "q0_velocity", "history_velocity", "old_velocity",
            "q0_supported", "history_supported", "fallback_velocity",
            "fallback_supported", "target_velocity", "motion_class",
            "alignment_shift",
        )
    }
    for start in range(0, len(recipients), batch_size):
        stop = min(start + batch_size, len(recipients))
        recipient = torch.from_numpy(recipients[start:stop]).long()
        donor = torch.from_numpy(donors[start:stop]).long()
        state = {
            name: prepared[name][recipient].to(device, non_blocking=True)
            for name in CENTER_TWIST_FORWARD_FIELDS
        }
        aligned_relation, aligned_supported, alignment_shift = _aligned_donor_q0(
            prepared, recipient, donor,
        )
        state["q0_relation_m"] = aligned_relation.to(device, non_blocking=True)
        state["q0_supported"] = aligned_supported.to(device, non_blocking=True)
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
        values = {
            "feature": profiled["component_gate_feature"],
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
    oracle = oracle_projection_coefficient(
        arm["target_velocity"], arm["q0_velocity"], arm["history_velocity"],
        arm["q0_supported"], arm["history_supported"],
    )
    arm.update({name: value.cpu() for name, value in oracle.items()})
    arm["recipient_global"] = torch.from_numpy(recipients.copy()).long()
    return arm


def _slice_arm(arm: dict[str, torch.Tensor], indices: np.ndarray) -> dict[str, torch.Tensor]:
    selected = torch.from_numpy(indices).long()
    return {name: value[selected] for name, value in arm.items()}


def _build_map(
    indices: np.ndarray, keys: Sequence[str], sessions: Sequence[str],
    *, domain: str, strata: Sequence[object] | None,
    fallback_strata: Sequence[object] | None = None,
    require_cross_session: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    subset_keys = [keys[int(index)] for index in indices]
    subset_sessions = [sessions[int(index)] for index in indices]
    subset_strata = None if strata is None else [strata[int(index)] for index in indices]
    manifest = deterministic_cross_session_derangement(
        subset_keys, subset_sessions, domain=domain, strata=subset_strata,
        require_cross_session=require_cross_session,
    )
    valid = np.asarray(manifest["valid"], dtype=np.bool_).copy()
    local_donor = np.asarray(manifest["donor_index"], dtype=np.int64).copy()
    exact_valid_count = int(valid.sum())
    fallback_valid_count = 0
    fallback_invalid_groups: list[str] = []
    invalid_position = np.flatnonzero(~valid)
    if fallback_strata is not None and invalid_position.size >= 2:
        subset_fallback = [fallback_strata[int(index)] for index in indices]
        fallback = deterministic_cross_session_derangement(
            [subset_keys[int(position)] for position in invalid_position],
            [subset_sessions[int(position)] for position in invalid_position],
            domain=f"{domain}/relaxed", strata=[
                subset_fallback[int(position)] for position in invalid_position
            ],
            require_cross_session=require_cross_session,
        )
        fallback_valid = np.asarray(fallback["valid"], dtype=np.bool_)
        fallback_donor = np.asarray(fallback["donor_index"], dtype=np.int64)
        accepted = invalid_position[fallback_valid]
        local_donor[accepted] = invalid_position[fallback_donor[fallback_valid]]
        valid[accepted] = True
        fallback_valid_count = int(fallback_valid.sum())
        fallback_invalid_groups = list(fallback["invalid_groups"])
    recipients = indices[valid]
    donors = indices[local_donor[valid]]
    if sorted(local_donor[valid].tolist()) != sorted(np.flatnonzero(valid).tolist()):
        raise RuntimeError("combined corruption map is not bijective")
    pair_digest = hashlib.sha256()
    pair_digest.update(domain.encode("utf-8"))
    for position, key in enumerate(subset_keys):
        donor_key = subset_keys[int(local_donor[position])] if valid[position] else "INVALID"
        pair_digest.update(f"{key}\0{donor_key}\n".encode("utf-8"))
    public_manifest = {
        "domain": domain, "sample_count": int(indices.size),
        "valid_count": int(valid.sum()), "coverage": float(valid.mean()),
        "fixed_point_count": 0,
        "cross_session_count": int(np.sum(
            np.asarray(subset_sessions, dtype=object)[local_donor[valid]]
            != np.asarray(subset_sessions, dtype=object)[valid]
        )),
        "require_cross_session": bool(require_cross_session),
        "pair_sha256": pair_digest.hexdigest(),
        "exact_valid_count": exact_valid_count,
        "relaxed_valid_count": fallback_valid_count,
        "exact_invalid_groups": list(manifest["invalid_groups"]),
        "relaxed_invalid_groups": fallback_invalid_groups,
        "strata_schema": (
            "all" if strata is None else "motion_class/q0_support_count"
        ),
        "relaxed_strata_schema": (
            None if fallback_strata is None else "motion_class"
        ),
        "relaxed_require_cross_session": (
            None if fallback_strata is None else bool(require_cross_session)
        ),
    }
    public_manifest["recipient_index_sha256"] = hashlib.sha256(
        recipients.astype("<i8", copy=False).tobytes()
    ).hexdigest()
    public_manifest["donor_index_sha256"] = hashlib.sha256(
        donors.astype("<i8", copy=False).tobytes()
    ).hexdigest()
    return recipients, donors, public_manifest


def _arm_batch(
    arm: dict[str, torch.Tensor], rows: torch.Tensor, device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: value[rows].to(device, non_blocking=True)
        for name, value in arm.items()
    }


def _train_head(
    arms: dict[str, dict[str, torch.Tensor]],
    *,
    device: torch.device,
    seed: int,
    width: int,
    updates: int,
    batch_size: int,
) -> tuple[MonotoneReliabilityFusion, list[dict[str, float]]]:
    model = MonotoneReliabilityFusion(width=width).to(device)
    all_features = torch.cat([
        arm["feature"][arm["q0_supported"] & arm["history_supported"]]
        for arm in arms.values()
    ]).to(device)
    model.fit_normalizer(all_features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    logs: list[dict[str, float]] = []
    model.train()
    for update in range(1, updates + 1):
        arm_losses: list[torch.Tensor] = []
        predicted: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for name, arm in arms.items():
            count = arm["feature"].shape[0]
            rows = torch.randint(
                count, (min(batch_size, count),), generator=generator,
            )
            batch = _arm_batch(arm, rows, device)
            result = model(
                batch["feature"], batch["q0_velocity"],
                batch["history_velocity"], batch["q0_supported"],
                batch["history_supported"], batch["fallback_velocity"],
                batch["fallback_supported"],
            )
            valid = batch["oracle_valid"] & result["learned_gate_valid"]
            if not bool(valid.any()):
                raise RuntimeError(f"no oracle-valid rows in {name} reliability arm")
            coefficient_loss = F.smooth_l1_loss(
                result["q0_weight"][valid], batch["weight"][valid], beta=0.10,
            )
            mixture_loss = vector_huber_loss(
                result["velocity_mps"][valid],
                batch["target_velocity"][valid], beta=0.25,
            )
            arm_losses.append(coefficient_loss + mixture_loss)
            predicted[name] = (rows, result["q0_weight"], batch["weight"])
        pair_losses: list[torch.Tensor] = []
        intact_arm = arms["intact"]
        intact_lookup = {
            int(value): index
            for index, value in enumerate(intact_arm["recipient_global"].tolist())
        }
        for name in ("global", "hard"):
            rows, corrupt_weight, corrupt_oracle = predicted[name]
            recipient = arms[name]["recipient_global"][rows].tolist()
            intact_rows = torch.tensor(
                [intact_lookup[int(value)] for value in recipient], dtype=torch.long,
            )
            intact_batch = _arm_batch(intact_arm, intact_rows, device)
            intact_result = model(
                intact_batch["feature"], intact_batch["q0_velocity"],
                intact_batch["history_velocity"], intact_batch["q0_supported"],
                intact_batch["history_supported"],
                intact_batch["fallback_velocity"],
                intact_batch["fallback_supported"],
            )
            corrupt_batch = _arm_batch(arms[name], rows, device)
            pair_valid = (
                intact_batch["oracle_valid"]
                & intact_result["learned_gate_valid"]
                & corrupt_batch["oracle_valid"]
            )
            if not bool(pair_valid.any()):
                raise RuntimeError(f"no oracle-valid reliability pairs in {name}")
            pair_losses.append(F.smooth_l1_loss(
                (intact_result["q0_weight"] - corrupt_weight)[pair_valid],
                (intact_batch["weight"] - corrupt_oracle)[pair_valid], beta=0.10,
            ))
        objective = torch.stack(arm_losses).mean() + torch.stack(pair_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if any(
            parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise RuntimeError("reliability head has missing/non-finite gradients")
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
                "update": float(update), "objective": float(objective.detach()),
                "learning_rate": learning_rate,
            })
    return model.eval(), logs


@torch.inference_mode()
def _evaluate_head(
    model: MonotoneReliabilityFusion,
    arms: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    result: dict[str, Any] = {"arms": {}}
    predictions: dict[str, dict[str, torch.Tensor]] = {}
    for name, arm in arms.items():
        batch = {key: value.to(device) for key, value in arm.items()}
        fused = model(
            batch["feature"], batch["q0_velocity"], batch["history_velocity"],
            batch["q0_supported"], batch["history_supported"],
            batch["fallback_velocity"], batch["fallback_supported"],
        )
        supported = fused["state_supported"]
        oracle_velocity = (
            batch["weight"].unsqueeze(-1) * batch["q0_velocity"]
            + (1.0 - batch["weight"]).unsqueeze(-1) * batch["history_velocity"]
        )
        valid = batch["oracle_valid"] & fused["learned_gate_valid"]
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
            "common_fraction": float(valid.float().mean()),
            "state_supported_fraction": float(supported.float().mean()),
            "fallback_fraction": float(fused["fallback_used"].float().mean()),
            "groups": {},
        }
        group_masks = {
            "overall": torch.ones_like(valid),
            "rotation": batch["motion_class"] == 2,
            "combined": batch["motion_class"] == 3,
        }
        for group, group_mask in group_masks.items():
            common = valid & group_mask
            if not bool(common.any()):
                raise RuntimeError(f"empty {name}/{group} common-support population")
            def error(velocity: torch.Tensor) -> torch.Tensor:
                return torch.linalg.vector_norm(
                    velocity[common] - batch["target_velocity"][common], dim=-1,
                )
            preference = q0_error[common] < history_error[common]
            arm_result["groups"][group] = {
                "full_count": int(group_mask.sum()),
                "common_count": int(common.sum()),
                "common_fraction": float(
                    common.sum() / group_mask.sum().clamp_min(1)
                ),
                "fused_error_mps": _distribution(error(fused["velocity_mps"])),
                "blind_error_mps": _distribution(error(batch["history_velocity"])),
                "parent_error_mps": _distribution(error(batch["old_velocity"])),
                "oracle_error_mps": _distribution(error(oracle_velocity)),
                "q0_weight": _distribution(fused["q0_weight"][common]),
                "oracle_weight": _distribution(batch["weight"][common]),
                "coefficient_mae": float((
                    fused["q0_weight"][common] - batch["weight"][common]
                ).abs().mean()),
                "coefficient_brier": float((
                    fused["q0_weight"][common] - batch["weight"][common]
                ).square().mean()),
                "q0_better_count": int(preference.sum()),
                "history_better_or_equal_count": int((~preference).sum()),
                "component_preference_auc": _binary_auc(
                    fused["q0_weight"][common].cpu(), preference.cpu(),
                ),
            }
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
                raise RuntimeError(f"empty {name}/{group} paired reliability population")
            result["arms"][name]["groups"][group].update({
                "paired_count": int(common.sum()),
                "paired_fraction": float(
                    common.sum() / group_mask.sum().clamp_min(1)
                ),
                "paired_weight_separation_mean": float((
                    predictions["intact"]["weight"][reference][common]
                    - predictions[name]["weight"][common]
                ).mean()),
            })
    return result


def _gate_metrics(metrics: dict[str, Any]) -> dict[str, bool | float]:
    arms = metrics["arms"]
    gates: dict[str, bool | float] = {}
    for name in ("intact", "global", "hard"):
        gates[f"{name}_state_supported_complete"] = (
            arms[name]["state_supported_fraction"] == 1.0
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
        gates[f"{group}_intact_oracle_headroom_identifiable"] = identifiable
        gates[f"{group}_intact_oracle_gap_recovery_mean"] = recovery_mean
        gates[f"{group}_intact_oracle_gap_recovery_p50"] = recovery_p50
        gates[f"{group}_intact_recovers_30pct_mean"] = recovery_mean >= 0.30
        gates[f"{group}_intact_recovers_30pct_p50"] = recovery_p50 >= 0.30
        gates[f"{group}_intact_auc_ge_0_75"] = (
            intact["component_preference_auc"] >= 0.75
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
            gates[f"{group}_{name}_mean_within_blind_1_02"] = (
                arm["fused_error_mps"]["mean"]
                <= 1.02 * arm["blind_error_mps"]["mean"]
            )
            gates[f"{group}_{name}_p50_within_blind_1_02"] = (
                arm["fused_error_mps"]["p50"]
                <= 1.02 * arm["blind_error_mps"]["p50"]
            )
    return gates


def _fold_assignments(sessions: Sequence[str], *, folds: int) -> list[set[str]]:
    counts = {session: sessions.count(session) for session in set(sessions)}
    unique = list(counts)
    if len(unique) < folds * 2:
        raise ValueError("reliability CV has too few sessions")
    assignment = [set() for _ in range(folds)]
    load = [{"rotation": 0, "combined": 0} for _ in range(folds)]
    groups = {
        "combined": [value for value in unique if "combined" in value],
        "rotation": [value for value in unique if "combined" not in value],
    }
    for group, values in groups.items():
        ordered = sorted(
            values,
            key=lambda value: (
                -counts[value],
                hashlib.sha256(
                    f"v15/a0/folds\0{value}".encode("utf-8")
                ).digest(),
            ),
        )
        for session in ordered:
            fold = min(
                range(folds), key=lambda index: (load[index][group], index),
            )
            assignment[fold].add(session)
            load[fold][group] += counts[session]
    return assignment


def _make_arms(
    base: CenterPriorProfiledTwistScreen,
    prepared: dict[str, torch.Tensor],
    intact_full: dict[str, torch.Tensor],
    indices: np.ndarray,
    keys: Sequence[str], sessions: Sequence[str],
    hard_strata: Sequence[object],
    device: torch.device, *, domain: str, batch_size: int,
    require_cross_session: bool = True,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    global_recipient, global_donor, global_manifest = _build_map(
        indices, keys, sessions, domain=f"{domain}/global", strata=None,
        require_cross_session=require_cross_session,
    )
    hard_recipient, hard_donor, hard_manifest = _build_map(
        indices, keys, sessions, domain=f"{domain}/matched-hard",
        strata=hard_strata,
        require_cross_session=require_cross_session,
    )
    if global_manifest["coverage"] != 1.0:
        raise RuntimeError("global reliability corruption is incomplete")
    if hard_manifest["coverage"] < 0.80:
        raise RuntimeError("matched-hard reliability corruption covers below 80%")
    if hard_manifest["exact_valid_count"] < 0.80 * hard_manifest["sample_count"]:
        raise RuntimeError("matched-hard exact motion/support coverage is below 80%")
    if hard_manifest["relaxed_valid_count"] != 0:
        raise RuntimeError("matched-hard reliability corruption was relaxed")
    for label, manifest in (("global", global_manifest), ("hard", hard_manifest)):
        if manifest["cross_session_count"] != manifest["valid_count"]:
            raise RuntimeError(f"{label} reliability corruption is not cross-session")
    arms = {
        "intact": _slice_arm(intact_full, indices),
        "global": _expert_arm(
            base, prepared, global_recipient, global_donor, device,
            batch_size=batch_size,
        ),
        "hard": _expert_arm(
            base, prepared, hard_recipient, hard_donor, device,
            batch_size=batch_size,
        ),
    }
    for name, manifest in (("global", global_manifest), ("hard", hard_manifest)):
        shift = arms[name]["alignment_shift"].numpy().astype(np.uint8, copy=False)
        manifest["c4_alignment_shift_sha256"] = hashlib.sha256(
            shift.tobytes()
        ).hexdigest()
        manifest["c4_alignment_shift_histogram"] = {
            str(value): int(np.sum(shift == value)) for value in range(4)
        }
    return arms, {"global": global_manifest, "hard": hard_manifest}


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V15-A0 requires local CUDA")
    git_start = _git_state()
    if git_start["git_commit"] == "unknown" or git_start["worktree_dirty"]:
        raise RuntimeError("V15-A0 requires a clean known git commit")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing reliability output: {output}")
    output.mkdir(parents=True)
    _atomic_json(output / "run_state.json", {
        "schema_version": RUN_SCHEMA, "status": "initializing",
        "validation_claimed": False, "test_accessed": False,
    })

    parent_checkpoint = Path(args.parent_checkpoint).resolve()
    parent_root = parent_checkpoint.parent.parent
    parent_manifest = json.loads(
        (parent_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    parent_manifest_path = parent_root / "run_manifest.json"
    parent_manifest_file_sha256 = sha256_file(parent_manifest_path)
    parent_payload = torch.load(
        parent_checkpoint, map_location="cpu", weights_only=False,
    )
    parent_sha256 = sha256_file(parent_checkpoint)
    parent_contract = _validate_checkpoint_manifest_identity(
        parent_payload, parent_manifest, parent_sha256,
    )
    parent_result_path = parent_root / "screen_result.json"
    parent_result_file_sha256 = sha256_file(parent_result_path)
    if sha256_file(parent_result_path) != parent_manifest.get("screen_result_sha256"):
        raise ValueError("V15-A0 parent result hash differs")
    parent_result = json.loads(parent_result_path.read_text(encoding="utf-8"))
    if (
        parent_result.get("schema_version") != parent_payload.get("schema_version")
        or parent_result.get("checkpoint_sha256") != parent_sha256
        or parent_result.get("contract_sha256") != parent_manifest.get("contract_sha256")
    ):
        raise ValueError("V15-A0 parent result identity differs")
    if parent_result.get("test_accessed") is not False:
        raise ValueError("V15-A0 parent accessed test")
    parent_source_paths = {
        name: Path(path).resolve()
        for name, path in parent_manifest["provenance"]["source_paths"].items()
    }
    if {
        name: sha256_file(path) for name, path in parent_source_paths.items()
    } != parent_contract["source_sha256"]:
        raise RuntimeError("V15-A0 parent implementation source differs")
    parent_args = parent_contract["args"]
    dataset_path = Path(parent_args["dataset"]).resolve()
    truth_path = Path(parent_args["truth_history"]).resolve()
    mapper_path = Path(parent_args["mapper_checkpoint"]).resolve()
    s_path = Path(parent_args["s_checkpoint"]).resolve()
    h_path = Path(parent_args["h_checkpoint"]).resolve()

    _seed(args.seed)
    train_dataset = _dataset(dataset_path, "train", sample_limit=0)
    dataset_manifest_path = dataset_path / "dataset_manifest.json"
    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if sha256_file(dataset_manifest_path) != parent_contract["dataset_manifest_sha256"]:
        raise RuntimeError("V15-A0 dataset manifest differs from parent")
    if manifest.get("test_accessed") is not False:
        raise ValueError("V15-A0 dataset accessed test")
    truth_sha = manifest["truth_history_manifest_sha256"]
    if truth_sha != parent_contract["truth_manifest_sha256"]:
        raise RuntimeError("V15-A0 truth manifest differs from parent")
    motion_truth = MotionTruthIndex(
        truth_path, expected_manifest_sha256=truth_sha,
    )
    center_truth = CenterTruthIndex(
        truth_path, expected_manifest_sha256=truth_sha,
    )
    motion_truth.attach(train_dataset, "train")
    center_truth.attach(train_dataset, "train")
    mapper, _ = load_frozen_pnp_mapper(mapper_path)
    s_model, _ = load_frozen_v19(s_path)
    h_model, _ = load_frozen_hypothesis_adapter(h_path, allow_diagnostic=True)
    for frozen in (mapper, s_model, h_model):
        frozen.to(device).eval().requires_grad_(False)
    expected_frozen = parent_contract["frozen_state_dict_sha256"]
    actual_frozen = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    }
    if actual_frozen != expected_frozen:
        raise RuntimeError("V15-A0 frozen upstream differs from parent")
    frozen_checkpoint_sha = {
        "mapper": sha256_file(mapper_path), "s": sha256_file(s_path),
        "h": sha256_file(h_path),
    }
    if frozen_checkpoint_sha != parent_contract["frozen_checkpoint_sha256"]:
        raise RuntimeError("V15-A0 frozen checkpoint files differ from parent")
    base = CenterPriorProfiledTwistScreen(
        width=int(parent_args["width"]),
        center_precision=float(parent_args["center_precision"]),
    ).to(device)
    base.load_state_dict(parent_payload["model"], strict=True)
    base.eval().requires_grad_(False)
    base_initial_sha = state_dict_sha256(base.state_dict())
    source_paths = {
        "runner": Path(__file__).resolve(),
        "fusion": Path(__file__).with_name(
            "frozen_expert_reliability_fusion.py"
        ).resolve(),
    }
    source_sha = {name: sha256_file(path) for name, path in source_paths.items()}
    experiment_contract = {
        "schema_version": RUN_SCHEMA,
        "diagnostic_only": True,
        "truth_omega_forward_input": True,
        "parent_checkpoint_sha256": parent_sha256,
        "parent_screen_result_sha256": parent_manifest["screen_result_sha256"],
        "parent_run_manifest_sha256": parent_manifest_file_sha256,
        "parent_contract_sha256": parent_manifest["contract_sha256"],
        "parent_model_state_dict_sha256": base_initial_sha,
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "truth_manifest_sha256": truth_sha,
        "frozen_checkpoint_sha256": frozen_checkpoint_sha,
        "frozen_state_dict_sha256": actual_frozen,
        "git_commit": git_start["git_commit"],
        "source_sha256": source_sha,
        "folds": FIXED_FOLDS, "updates": args.updates,
        "seed": args.seed, "width": args.width,
        "batch_size": args.batch_size,
        "expert_batch_size": args.expert_batch_size,
        "forward_fields": list(CENTER_TWIST_FORWARD_FIELDS),
        "inherited_parent_validation": True,
    }
    experiment_contract_sha = _json_sha256(experiment_contract)
    validation_scope = {
        "stage_slot": "v15-a0-single-validation",
        "parent_checkpoint_sha256": parent_sha256,
        "dataset_manifest_sha256": experiment_contract["dataset_manifest_sha256"],
        "truth_manifest_sha256": truth_sha,
    }
    validation_scope_sha = _json_sha256(validation_scope)
    _atomic_json(output / "run_state.json", {
        "schema_version": RUN_SCHEMA, "status": "train_cv",
        "validation_claimed": False, "test_accessed": False,
        "experiment_contract": experiment_contract,
        "experiment_contract_sha256": experiment_contract_sha,
        "validation_scope": validation_scope,
        "validation_scope_sha256": validation_scope_sha,
    })
    keys, sessions = _combined_metadata(train_dataset)
    prepared = _cache_prepared_state(
        train_dataset, mapper, s_model, h_model, device,
        batch_size=args.expert_batch_size,
    )
    all_indices = np.arange(len(train_dataset), dtype=np.int64)
    intact_full = _expert_arm(
        base, prepared, all_indices, all_indices, device,
        batch_size=args.expert_batch_size,
    )
    support_count = prepared["q0_supported"].sum(dim=-1).numpy().astype(np.int64)
    motion_class = prepared["motion_class"].numpy().astype(np.int64)
    hard_strata = [
        (int(motion_class[index]), int(support_count[index]))
        for index in range(len(train_dataset))
    ]
    fold_sessions = _fold_assignments(sessions, folds=FIXED_FOLDS)
    cv_results: list[dict[str, Any]] = []
    for fold, heldout_sessions in enumerate(fold_sessions):
        heldout = np.asarray([
            index for index, session in enumerate(sessions)
            if session in heldout_sessions
        ], dtype=np.int64)
        training = np.asarray([
            index for index, session in enumerate(sessions)
            if session not in heldout_sessions
        ], dtype=np.int64)
        train_arms, train_maps = _make_arms(
            base, prepared, intact_full, training, keys, sessions, hard_strata,
            device, domain=f"train/cv{fold}/fit", batch_size=args.expert_batch_size,
        )
        heldout_arms, heldout_maps = _make_arms(
            base, prepared, intact_full, heldout, keys, sessions, hard_strata,
            device, domain=f"train/cv{fold}/heldout",
            batch_size=args.expert_batch_size,
        )
        head, logs = _train_head(
            train_arms, device=device, seed=args.seed + 100 + fold,
            width=args.width, updates=args.updates, batch_size=args.batch_size,
        )
        metrics = _evaluate_head(head, heldout_arms, device)
        gates = _gate_metrics(metrics)
        cv_results.append({
            "fold": fold, "heldout_sessions": sorted(heldout_sessions),
            "train_count": int(training.size), "heldout_count": int(heldout.size),
            "train_maps": train_maps, "heldout_maps": heldout_maps,
            "logs": logs, "metrics": metrics, "gates": gates,
            "passed": all(value for value in gates.values() if isinstance(value, bool)),
        })
    cv_passed = all(item["passed"] for item in cv_results)
    result: dict[str, Any] = {
        "schema_version": RUN_SCHEMA,
        "status": "cv_passed" if cv_passed else "cv_failed",
        "diagnostic_only": True,
        "truth_omega_forward_input": True,
        "formal_v15": False,
        "test_accessed": False,
        "future_modules_loaded": False,
        "validation_accessed": False,
        "git": git_start,
        "experiment_contract": experiment_contract,
        "experiment_contract_sha256": experiment_contract_sha,
        "validation_scope": validation_scope,
        "validation_scope_sha256": validation_scope_sha,
        "parent": {
            "checkpoint": str(parent_checkpoint), "sha256": parent_sha256,
            "screen_result_sha256": parent_manifest["screen_result_sha256"],
            "contract_sha256": parent_manifest["contract_sha256"],
            "model_state_dict_sha256": base_initial_sha,
            "source_commit": parent_contract["git"]["git_commit"],
        },
        "dataset_manifest_sha256": sha256_file(
            dataset_path / "dataset_manifest.json"
        ),
        "truth_manifest_sha256": truth_sha,
        "fixed_folds": FIXED_FOLDS,
        "fixed_updates": args.updates,
        "cv": cv_results,
    }
    if cv_passed:
        full_arms, train_maps = _make_arms(
            base, prepared, intact_full, all_indices, keys, sessions, hard_strata,
            device, domain="train/full", batch_size=args.expert_batch_size,
        )
        final_head, final_logs = _train_head(
            full_arms, device=device, seed=args.seed + 1000,
            width=args.width, updates=args.updates, batch_size=args.batch_size,
        )
        final_head.eval().requires_grad_(False)
        final_head_sha_before_validation = state_dict_sha256(final_head.state_dict())
        validation_ledger_dir = parent_root.parent / "_v15-validation-access-ledger"
        validation_ledger_dir.mkdir(exist_ok=True)
        validation_ledger_path = validation_ledger_dir / (
            validation_scope_sha + ".json"
        )
        validation_claim = {
            "schema_version": RUN_SCHEMA,
            "status": "claimed",
            "experiment_contract_sha256": experiment_contract_sha,
            "validation_scope": validation_scope,
            "validation_scope_sha256": validation_scope_sha,
            "dataset_manifest_sha256": experiment_contract[
                "dataset_manifest_sha256"
            ],
            "parent_checkpoint_sha256": parent_sha256,
            "reliability_state_dict_sha256": final_head_sha_before_validation,
            "output": str(output), "test_accessed": False,
        }
        try:
            with validation_ledger_path.open("x", encoding="utf-8") as handle:
                json.dump(validation_claim, handle, indent=2, sort_keys=True)
                handle.write("\n")
        except FileExistsError as exc:
            raise RuntimeError(
                "V15-A0 validation was already claimed for this experiment contract"
            ) from exc
        _atomic_json(output / "run_state.json", {
            "schema_version": RUN_SCHEMA, "status": "validation_claimed",
            "validation_claimed": True, "test_accessed": False,
            "experiment_contract_sha256": experiment_contract_sha,
            "validation_ledger": str(validation_ledger_path),
            "reliability_state_dict_sha256": final_head_sha_before_validation,
        })
        validation_dataset = _dataset(dataset_path, "validation", sample_limit=0)
        motion_truth.attach(validation_dataset, "validation")
        center_truth.attach(validation_dataset, "validation")
        validation_keys, validation_sessions = _combined_metadata(validation_dataset)
        validation_prepared = _cache_prepared_state(
            validation_dataset, mapper, s_model, h_model, device,
            batch_size=args.expert_batch_size,
        )
        validation_indices = np.arange(len(validation_dataset), dtype=np.int64)
        validation_intact = _expert_arm(
            base, validation_prepared, validation_indices, validation_indices,
            device, batch_size=args.expert_batch_size,
        )
        validation_support = validation_prepared["q0_supported"].sum(
            dim=-1
        ).numpy().astype(np.int64)
        validation_motion = validation_prepared["motion_class"].numpy().astype(
            np.int64
        )
        validation_strata = [
            (int(validation_motion[index]), int(validation_support[index]))
            for index in range(len(validation_dataset))
        ]
        validation_arms, validation_maps = _make_arms(
            base, validation_prepared, validation_intact, validation_indices,
            validation_keys, validation_sessions, validation_strata, device,
            domain="validation/locked", batch_size=args.expert_batch_size,
        )
        validation_metrics = _evaluate_head(final_head, validation_arms, device)
        if state_dict_sha256(final_head.state_dict()) != final_head_sha_before_validation:
            raise RuntimeError("V15-A0 changed the locked head during validation")
        validation_gates = _gate_metrics(validation_metrics)
        result.update({
            "status": "passed" if all(
                value for value in validation_gates.values()
                if isinstance(value, bool)
            ) else "failed",
            "validation_accessed": True,
            "train_maps": train_maps,
            "validation_maps": validation_maps,
            "final_logs": final_logs,
            "validation_metrics": validation_metrics,
            "validation_gates": validation_gates,
            "reliability_state_dict_sha256": state_dict_sha256(
                final_head.state_dict()
            ),
            "validation_ledger": str(validation_ledger_path),
        })
        result["authorized_a1_endpoint_probe"] = result["status"] != "passed"
        result["authorized_formal_two_stage"] = result["status"] == "passed"
    else:
        result["authorized_a1_endpoint_probe"] = True
        result["authorized_formal_two_stage"] = False
    if state_dict_sha256(base.state_dict()) != base_initial_sha:
        raise RuntimeError("V15-A0 changed its frozen V14 parent")
    if sha256_file(parent_checkpoint) != parent_sha256:
        raise RuntimeError("V15-A0 parent checkpoint changed")
    if sha256_file(parent_manifest_path) != parent_manifest_file_sha256:
        raise RuntimeError("V15-A0 parent run manifest changed")
    if sha256_file(parent_result_path) != parent_result_file_sha256:
        raise RuntimeError("V15-A0 parent screen result changed")
    if {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    } != actual_frozen:
        raise RuntimeError("V15-A0 changed frozen Mapper/S/H")
    if {
        "mapper": sha256_file(mapper_path), "s": sha256_file(s_path),
        "h": sha256_file(h_path),
    } != frozen_checkpoint_sha:
        raise RuntimeError("V15-A0 frozen checkpoint files changed")
    if sha256_file(dataset_manifest_path) != experiment_contract[
        "dataset_manifest_sha256"
    ]:
        raise RuntimeError("V15-A0 dataset manifest changed")
    if sha256_file(truth_path / "dataset_manifest.json") != truth_sha:
        raise RuntimeError("V15-A0 truth manifest changed")
    _assert_manifest_shards_unchanged(
        dataset_path, manifest, label="paired dataset",
    )
    _assert_manifest_shards_unchanged(
        truth_path, motion_truth.manifest, label="truth",
    )
    if {name: sha256_file(path) for name, path in source_paths.items()} != source_sha:
        raise RuntimeError("V15-A0 implementation source changed")
    if {
        name: sha256_file(path) for name, path in parent_source_paths.items()
    } != parent_contract["source_sha256"]:
        raise RuntimeError("V15-A0 parent source changed")
    if _git_state() != git_start:
        raise RuntimeError("V15-A0 source changed during the run")
    if cv_passed:
        checkpoint_path = output / "reliability-head.pt"
        _atomic_checkpoint(checkpoint_path, {
            "schema_version": RUN_SCHEMA,
            "diagnostic_only": True,
            "truth_omega_forward_input": True,
            "experiment_contract": experiment_contract,
            "experiment_contract_sha256": experiment_contract_sha,
            "parent_checkpoint_sha256": parent_sha256,
            "model": final_head.state_dict(),
        })
        result["reliability_checkpoint"] = str(checkpoint_path)
        result["reliability_checkpoint_sha256"] = sha256_file(checkpoint_path)
    _atomic_json(output / "screen_result.json", result)
    if cv_passed:
        result_sha = sha256_file(output / "screen_result.json")
        _atomic_json(validation_ledger_path, {
            **validation_claim,
            "status": "consumed",
            "screen_result_sha256": result_sha,
            "reliability_checkpoint_sha256": result[
                "reliability_checkpoint_sha256"
            ],
        })
    _atomic_json(output / "run_state.json", {
        "schema_version": RUN_SCHEMA, "status": result["status"],
        "validation_claimed": bool(cv_passed),
        "validation_consumed": bool(cv_passed),
        "test_accessed": False,
        "experiment_contract_sha256": experiment_contract_sha,
        "screen_result_sha256": sha256_file(output / "screen_result.json"),
        "reliability_checkpoint_sha256": result.get(
            "reliability_checkpoint_sha256"
        ),
    })
    if result["status"] == "passed":
        validate_reliability_a0_artifacts(output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--expert-batch-size", type=int, default=96)
    args = parser.parse_args()
    if args.updates != DEFAULT_UPDATES:
        raise ValueError(f"V15-A0 fixed updates must be {DEFAULT_UPDATES}")
    fixed = {
        "seed": 20260730, "width": 32, "batch_size": 256,
        "expert_batch_size": 96,
    }
    for name, expected in fixed.items():
        if getattr(args, name) != expected:
            raise ValueError(f"V15-A0 fixed {name} must be {expected}")
    return args


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
