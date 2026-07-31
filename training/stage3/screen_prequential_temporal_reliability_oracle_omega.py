"""A2 train-only prequential temporal reliability sufficiency screen.

The screen is authorized only by the sealed V15-A1-R2 rejection.  It keeps
Mapper/S/H and the two V14 profiled experts frozen, constructs cross-fitted
event evidence by leaving two history events out at a time, and trains only an
anonymous temporal reliability head.  Truth omega is a declared diagnostic
feature-construction input; truth velocity remains loss-only.  Validation,
test, free omega, and future-position modules are never constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import permutations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .cyclic_future_foundation import load_frozen_v19
from .endpoint_token_reliability_fusion import endpoint_token_features
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
from .prequential_endpoint_innovation import (
    DOSE_VALUES,
    combine_blockwise_endpoint_event_features,
    conjugate_role_donor_index,
    deterministic_endpoint_dose_temporal_break,
    deterministic_per_event_s4_continuity_break,
    deterministic_temporal_association_break,
    fixed_history_leave_block_out_masks,
    same_time_unordered_pair_temporal_features,
)
from .prequential_temporal_reliability_fusion import (
    PrequentialTemporalReliabilityFusion,
)
from .profiled_center_twist_future import (
    CENTER_TWIST_FORWARD_FIELDS,
    CenterPriorProfiledTwistScreen,
)
from .screen_endpoint_token_reliability_oracle_omega import (
    HARD_MAP_POLICY,
    _a1_source_paths,
    _all_boolean_gates_pass,
    _binary_auc,
    _build_corruption_maps,
    _cache_prepared_state,
    _combined_metadata,
    _distribution,
    _fold_assignments,
    _json_sha256,
    _load_train_dataset,
    _validate_corruption_manifest,
    validate_a0_rejection,
)
from .screen_endpoint_token_reliability_oracle_omega import (
    RUN_SCHEMA as A1_RUN_SCHEMA,
)
from .screen_endpoint_token_reliability_oracle_omega import (
    _cross_fold_gates as _a1_cross_fold_gates,
)
from .screen_endpoint_token_reliability_oracle_omega import (
    _fold_gates as _a1_fold_gates,
)
from .screen_frozen_expert_reliability_oracle_omega import (
    FIXED_FOLDS,
    _aligned_donor_q0,
    _slice_arm,
)
from .split_scoped_truth_supervision import (
    SplitScopedTruthIndex,
    assert_manifest_split_shards_unchanged,
)
from .train_causal_physical_ab import _git_state, _seed
from .train_pnp_window_mapper_distillation import (
    _atomic_checkpoint,
    _atomic_json,
)


RUN_SCHEMA = "stage3-prequential-temporal-reliability-oracle-omega-v15-a2-p0"
A1_R2_RESULT_SHA256 = (
    "c60beb32377d9cf767207268851dae6a47579a5ec03fd4d308a240bd227670d5"
)
A1_R2_CONTRACT_SHA256 = (
    "a7ffcc886c2998ae83a15d6c7cb1bb0710c2b424b09828014b88a6ec85280494"
)
A1_R2_GIT_COMMIT = "011074010cf5a9ca38dbe3efc17f33ba73c7edbb"
FIXED_UPDATES = 200
FIXED_WIDTH = 32
FIXED_BATCH_SIZE = 192
FIXED_EXPERT_BATCH_SIZE = 32
BLOCK_SIZE = 2
MINIMUM_FIT_EVENTS = 6
EVENT_WIDTH = 11
PAIR_WIDTH = 33
GLOBAL_WIDTH = 11
ARM_WEIGHT = {"intact": 0.50, "global": 0.25, "hard": 0.25}
VARIANT_NAMES = (
    "full", "local_disabled", "temporal_collapsed",
    "pair_message_disabled", "association_break", "continuity_break",
    "dose1", "dose2", "dose4",
)


def _source_paths() -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "fusion": Path(__file__).with_name(
            "prequential_temporal_reliability_fusion.py"
        ).resolve(),
        "innovation": Path(__file__).with_name(
            "prequential_endpoint_innovation.py"
        ).resolve(),
        **{f"a1_{name}": path for name, path in _a1_source_paths().items()},
    }


def validate_a1_r2_rejection(path: str | Path) -> dict[str, Any]:
    """Accept exactly the sealed train-only A1-R2 structural rejection."""
    result_path = Path(path).resolve()
    if result_path.is_dir():
        result_path = result_path / "screen_result.json"
    root = result_path.parent
    run_state_path = root / "run_state.json"
    if sha256_file(result_path) != A1_R2_RESULT_SHA256:
        raise ValueError("A2 requires the sealed A1-R2 result hash")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
    if (
        result.get("schema_version") != A1_RUN_SCHEMA
        or run_state.get("schema_version") != A1_RUN_SCHEMA
        or result.get("status") != "failed"
        or run_state.get("status") != "failed"
        or result.get("diagnostic_only") is not True
        or result.get("truth_omega_feature_construction") is not True
        or result.get("formal_v15") is not False
        or result.get("train_only") is not True
        or result.get("validation_accessed") is not False
        or result.get("test_accessed") is not False
        or result.get("future_modules_loaded") is not False
        or result.get("authorized_a1_counterfactual_probe") is not False
        or result.get("authorized_formal_two_stage") is not False
        or run_state.get("validation_accessed") is not False
        or run_state.get("test_accessed") is not False
        or run_state.get("future_modules_loaded") is not False
        or run_state.get("fold_checkpoint_sha256") is not None
        or "fold_checkpoint" in result
    ):
        raise ValueError("A1-R2 rejection status/scope differs")
    if run_state.get("screen_result_sha256") != A1_R2_RESULT_SHA256:
        raise ValueError("A1-R2 result/run-state binding differs")
    contract = result.get("experiment_contract")
    if (
        not isinstance(contract, dict)
        or result.get("experiment_contract_sha256") != A1_R2_CONTRACT_SHA256
        or run_state.get("experiment_contract_sha256") != A1_R2_CONTRACT_SHA256
        or _json_sha256(contract) != A1_R2_CONTRACT_SHA256
        or result.get("git", {}).get("git_commit") != A1_R2_GIT_COMMIT
        or result.get("git", {}).get("worktree_dirty") is not False
        or contract.get("git_commit") != A1_R2_GIT_COMMIT
    ):
        raise ValueError("A1-R2 fixed experiment contract differs")
    if {
        name: sha256_file(source) for name, source in _a1_source_paths().items()
    } != contract.get("source_sha256"):
        raise ValueError("A1-R2 implementation source differs")
    cv = result.get("cv")
    if (
        not isinstance(cv, list) or len(cv) != FIXED_FOLDS
        or any(item.get("passed") is not False for item in cv)
    ):
        raise ValueError("A1-R2 fold rejection differs")
    for item in cv:
        gates = _a1_fold_gates(item["metrics"])
        if item.get("gates") != gates or _all_boolean_gates_pass(gates):
            raise ValueError("A1-R2 recomputed fold rejection differs")
        for maps in (item["train_maps"], item["heldout_maps"]):
            _validate_corruption_manifest(maps["global"], kind="global")
            _validate_corruption_manifest(maps["hard"], kind="hard")
    cross = _a1_cross_fold_gates(cv)
    if result.get("cross_fold_gates") != cross:
        raise ValueError("A1-R2 cross-fold metrics differ")
    a0_path = Path(result["a0"]["result"]).resolve()
    if (
        sha256_file(a0_path) != result["a0"].get("sha256")
        or result["a0"].get("sha256") != contract.get("a0_result_sha256")
    ):
        raise ValueError("A1-R2 A0 binding differs")
    validate_a0_rejection(a0_path)
    parent_path = Path(result["parent"]["checkpoint"]).resolve()
    parent_sha = sha256_file(parent_path)
    if (
        parent_sha != result["parent"].get("sha256")
        or parent_sha != contract.get("parent_checkpoint_sha256")
    ):
        raise ValueError("A1-R2 parent binding differs")
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    if state_dict_sha256(parent_payload["model"]) != contract.get(
        "parent_model_state_dict_sha256"
    ):
        raise ValueError("A1-R2 parent state differs")
    for path_key, sha_key in (
        ("dataset_manifest_path", "dataset_manifest_sha256"),
        ("truth_manifest_path", "truth_manifest_sha256"),
    ):
        if sha256_file(Path(contract[path_key]).resolve()) != contract.get(sha_key):
            raise ValueError(f"A1-R2 {path_key} binding differs")
    frozen_paths = contract.get("frozen_checkpoint_paths")
    frozen_hashes = contract.get("frozen_checkpoint_sha256")
    if not isinstance(frozen_paths, dict) or not isinstance(frozen_hashes, dict):
        raise ValueError("A1-R2 frozen checkpoint contract is missing")
    if {
        name: sha256_file(Path(frozen_paths[name]).resolve())
        for name in ("mapper", "s", "h")
    } != frozen_hashes:
        raise ValueError("A1-R2 frozen checkpoint files differ")
    return result


def _model_kwargs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "event_feature": batch["event_feature"],
        "event_mask": batch["event_mask"],
        "event_pair_feature": batch["event_pair_feature"],
        "pair_mask": batch["pair_mask"],
        "global_feature": batch["global_feature"],
        "q0_velocity_mps": batch["q0_velocity"],
        "history_velocity_mps": batch["history_velocity"],
        "q0_supported": batch["q0_supported"],
        "history_supported": batch["history_supported"],
        "fallback_velocity_mps": batch["fallback_velocity"],
        "fallback_supported": batch["fallback_supported"],
    }


def _mixture_velocity(
    weight: torch.Tensor, q0: torch.Tensor, history: torch.Tensor,
) -> torch.Tensor:
    if weight.ndim == 1:
        return (
            weight.unsqueeze(-1) * q0
            + (1.0 - weight).unsqueeze(-1) * history
        )
    if weight.ndim == 2:
        return (
            weight.unsqueeze(-1) * q0.unsqueeze(1)
            + (1.0 - weight).unsqueeze(-1) * history.unsqueeze(1)
        )
    raise ValueError("A2 mixture weight must have shape [B] or [B,V]")


def _block_availability(
    event_mask: torch.Tensor, motion_class: torch.Tensor,
) -> dict[str, Any]:
    plan = fixed_history_leave_block_out_masks(
        event_mask.to(torch.bool), block_size=BLOCK_SIZE,
        minimum_fit_events=MINIMUM_FIT_EVENTS,
    )
    complete_block = (
        plan["heldout_event_mask"].sum(dim=-1) == BLOCK_SIZE
    )
    if bool(torch.any(plan["block_valid"] & ~complete_block)):
        raise RuntimeError("A2 accepted an incomplete heldout event block")
    row_available = plan["heldout_coverage_mask"].sum(dim=-1) >= 2
    result: dict[str, Any] = {"groups": {}}
    for group, mask in {
        "overall": torch.ones_like(row_available),
        "rotation": motion_class == 2,
        "combined": motion_class == 3,
    }.items():
        count = int(mask.sum())
        if count == 0:
            raise RuntimeError(f"A2 empty {group} block population")
        valid = int((row_available & mask).sum())
        result["groups"][group] = {
            "sample_count": count, "valid_count": valid,
            "coverage": valid / count,
        }
        if valid < math.ceil(0.80 * count):
            raise RuntimeError(f"A2 {group} block availability is insufficient")
    return result


def _crossfit_temporal_supported(event_role_mask: torch.Tensor) -> torch.Tensor:
    if event_role_mask.ndim != 3 or event_role_mask.shape[-1] != 4:
        raise ValueError("A2 crossfit event-role mask must have shape [B,T,4]")
    return event_role_mask.to(torch.bool).any(dim=-1).sum(dim=-1) >= 2


@torch.inference_mode()
def _extract_arm(
    base: CenterPriorProfiledTwistScreen,
    prepared: dict[str, torch.Tensor],
    recipients: np.ndarray,
    donors: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    if recipients.ndim != 1 or recipients.shape != donors.shape:
        raise ValueError("A2 recipient/donor indices differ")
    names = (
        "event_feature", "event_mask", "event_pair_feature", "pair_mask",
        "global_feature", "q0_velocity", "history_velocity", "old_velocity",
        "q0_supported", "history_supported", "fallback_velocity",
        "fallback_supported", "target_velocity", "motion_class",
        "crossfit_supported", "alignment_shift",
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
        relation, supported, shift = _aligned_donor_q0(
            prepared, recipient, donor,
        )
        state["q0_relation_m"] = relation.to(device, non_blocking=True)
        state["q0_supported"] = supported.to(device, non_blocking=True)
        target = prepared[MOTION_TARGET_FIELD][recipient].to(device)
        omega = target[:, 3]
        old = base.forward_at_omega(state, omega)
        center = base.estimate_center(
            state["q0_relation_m"], state["q0_supported"],
        )
        prior = {**center, "q0_relation_m": state["q0_relation_m"]}
        full_args = (
            state["history_obs_rel_m"], state["history_obs_mask"],
            state["history_event_mask"], state["history_time_s"], omega, prior,
        )
        q0_final = base.profile(*full_args, use_q0_prior=True)
        history_final = base.profile(*full_args, use_q0_prior=False)
        final_features = endpoint_token_features(
            history_obs_rel_m=state["history_obs_rel_m"],
            history_obs_mask=state["history_obs_mask"],
            history_event_mask=state["history_event_mask"],
            history_time_s=state["history_time_s"],
            history_primary_mask=state["history_primary_mask"],
            history_switch_step=state["history_switch_step"],
            q0_relation_m=state["q0_relation_m"],
            q0_supported=state["q0_supported"], omega_rad_s=omega,
            q0_velocity_mps=q0_final["velocity_mps"],
            history_velocity_mps=history_final["velocity_mps"],
            q0_center_offset_m=q0_final["profiled_center_offset_m"],
            history_center_offset_m=history_final["profiled_center_offset_m"],
            pooled_diagnostic_feature=old["component_gate_feature"],
        )
        block_plan = fixed_history_leave_block_out_masks(
            state["history_event_mask"].to(torch.bool), block_size=BLOCK_SIZE,
            minimum_fit_events=MINIMUM_FIT_EVENTS,
        )
        blocks = block_plan["fit_event_mask"].shape[1]
        block_features: list[torch.Tensor] = []
        block_role_masks: list[torch.Tensor] = []
        for block in range(blocks):
            fit_event = block_plan["fit_event_mask"][:, block]
            heldout_event = block_plan["heldout_event_mask"][:, block]
            fit_obs = state["history_obs_mask"].to(torch.bool) & fit_event.unsqueeze(-1)
            block_args = (
                state["history_obs_rel_m"], fit_obs, fit_event,
                state["history_time_s"], omega, prior,
            )
            q0_block = base.profile(*block_args, use_q0_prior=True)
            history_block = base.profile(*block_args, use_q0_prior=False)
            feature = endpoint_token_features(
                history_obs_rel_m=state["history_obs_rel_m"],
                history_obs_mask=state["history_obs_mask"],
                history_event_mask=state["history_event_mask"],
                history_time_s=state["history_time_s"],
                history_primary_mask=state["history_primary_mask"],
                history_switch_step=state["history_switch_step"],
                q0_relation_m=state["q0_relation_m"],
                q0_supported=state["q0_supported"], omega_rad_s=omega,
                q0_velocity_mps=q0_block["velocity_mps"],
                history_velocity_mps=history_block["velocity_mps"],
                q0_center_offset_m=q0_block["profiled_center_offset_m"],
                history_center_offset_m=history_block["profiled_center_offset_m"],
                pooled_diagnostic_feature=old["component_gate_feature"],
            )
            supported_block = (
                q0_block["profile_supported"] & history_block["profile_supported"]
                & block_plan["block_valid"][:, block]
                & (heldout_event.sum(dim=-1) == BLOCK_SIZE)
            )
            role_mask = (
                feature["event_mask"] & heldout_event.unsqueeze(-1)
                & supported_block[:, None, None]
            )
            block_features.append(feature["event_feature"])
            block_role_masks.append(role_mask)
        combined = combine_blockwise_endpoint_event_features(
            torch.stack(block_features, dim=1),
            block_plan["heldout_event_mask"],
            block_event_role_mask=torch.stack(block_role_masks, dim=1),
        )
        pair = same_time_unordered_pair_temporal_features(
            combined["event_feature"], combined["event_mask"],
        )
        crossfit_supported = _crossfit_temporal_supported(
            combined["event_mask"],
        )
        values = {
            "event_feature": combined["event_feature"],
            "event_mask": combined["event_mask"],
            "event_pair_feature": pair["pair_feature"],
            "pair_mask": pair["pair_mask"],
            "global_feature": final_features["global_feature"],
            "q0_velocity": q0_final["velocity_mps"],
            "history_velocity": history_final["velocity_mps"],
            "old_velocity": old["velocity_mps"],
            "q0_supported": q0_final["profile_supported"],
            "history_supported": history_final["profile_supported"],
            "fallback_velocity": q0_final["fallback_velocity_mps"],
            "fallback_supported": (
                q0_final["fallback_supported"] & history_final["fallback_supported"]
            ),
            "target_velocity": target[:, :3],
            "motion_class": prepared["motion_class"][recipient].to(device),
            "crossfit_supported": crossfit_supported,
            "alignment_shift": shift.to(device),
        }
        for name, value in values.items():
            output[name].append(value.detach().cpu())
    arm = {name: torch.cat(value, dim=0) for name, value in output.items()}
    arm.update(oracle_projection_coefficient(
        arm["target_velocity"], arm["q0_velocity"], arm["history_velocity"],
        arm["q0_supported"], arm["history_supported"],
    ))
    arm["recipient_global"] = torch.from_numpy(recipients.copy()).long()
    arm["donor_global"] = torch.from_numpy(donors.copy()).long()
    return arm


def _arm_batch(
    arm: dict[str, torch.Tensor], rows: torch.Tensor, device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: value[rows].to(device, non_blocking=True)
        for name, value in arm.items()
    }


def _common(batch: dict[str, torch.Tensor], output: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        batch["oracle_valid"] & batch["crossfit_supported"]
        & output["learned_gate_valid"]
    )


def _train_head(
    arms: dict[str, dict[str, torch.Tensor]],
    *,
    device: torch.device,
    seed: int,
    updates: int,
) -> tuple[PrequentialTemporalReliabilityFusion, list[dict[str, float]]]:
    model = PrequentialTemporalReliabilityFusion(
        event_width=EVENT_WIDTH, pair_width=PAIR_WIDTH,
        global_width=GLOBAL_WIDTH, width=FIXED_WIDTH,
    ).to(device)
    feature_names = (
        "event_feature", "event_mask", "event_pair_feature", "pair_mask",
        "global_feature",
    )
    concatenated = {
        name: torch.cat([arm[name] for arm in arms.values()], dim=0).to(device)
        for name in feature_names
    }
    row_mask = torch.cat([
        arm["oracle_valid"] & arm["crossfit_supported"]
        & arm["q0_supported"] & arm["history_supported"]
        for arm in arms.values()
    ]).to(device)
    model.fit_normalizer(**concatenated, row_mask=row_mask)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    logs: list[dict[str, float]] = []
    model.train()
    for update in range(1, updates + 1):
        objective = torch.zeros((), device=device)
        predictions: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for name, arm in arms.items():
            count = arm["global_feature"].shape[0]
            rows = torch.randint(
                count, (min(FIXED_BATCH_SIZE, count),), generator=generator,
            )
            batch = _arm_batch(arm, rows, device)
            output = model(**_model_kwargs(batch))
            valid = _common(batch, output)
            if not bool(valid.any()):
                raise RuntimeError(f"A2 has no common {name} rows")
            full_coefficient = F.smooth_l1_loss(
                output["q0_weight"][valid], batch["weight"][valid], beta=0.10,
            )
            full_velocity = vector_huber_loss(
                output["velocity_mps"][valid], batch["target_velocity"][valid],
                beta=0.25,
            )
            objective = objective + ARM_WEIGHT[name] * (
                full_coefficient + full_velocity
            )
            if name == "intact":
                local_coefficient = F.smooth_l1_loss(
                    output["local_q0_weight"][valid], batch["weight"][valid],
                    beta=0.10,
                )
                preference = (
                    torch.linalg.vector_norm(
                        batch["q0_velocity"] - batch["target_velocity"], dim=-1,
                    ) < torch.linalg.vector_norm(
                        batch["history_velocity"] - batch["target_velocity"],
                        dim=-1,
                    )
                ).to(output["local_logit"].dtype)
                local_preference = F.binary_cross_entropy_with_logits(
                    output["local_logit"][valid], preference[valid],
                )
                objective = objective + 0.50 * (
                    local_coefficient + local_preference
                )
            predictions[name] = (rows, output["q0_weight"], batch["weight"])
        intact = arms["intact"]
        lookup = {
            int(value): index for index, value in enumerate(
                intact["recipient_global"].tolist()
            )
        }
        pair_loss = torch.zeros((), device=device)
        for name in ("global", "hard"):
            rows, corrupt_weight, corrupt_oracle = predictions[name]
            recipients = arms[name]["recipient_global"][rows].tolist()
            intact_rows = torch.tensor(
                [lookup[int(value)] for value in recipients], dtype=torch.long,
            )
            intact_batch = _arm_batch(intact, intact_rows, device)
            intact_output = model(**_model_kwargs(intact_batch))
            corrupt_batch = _arm_batch(arms[name], rows, device)
            valid = (
                _common(intact_batch, intact_output)
                & corrupt_batch["oracle_valid"] & corrupt_batch["crossfit_supported"]
                & corrupt_batch["q0_supported"] & corrupt_batch["history_supported"]
                & corrupt_batch["event_mask"].any(dim=(1, 2))
            )
            if not bool(valid.any()):
                raise RuntimeError(f"A2 has no common {name} paired rows")
            pair_loss = pair_loss + 0.5 * F.smooth_l1_loss(
                (intact_output["q0_weight"] - corrupt_weight)[valid],
                (intact_batch["weight"] - corrupt_oracle)[valid], beta=0.10,
            )
        objective = objective + 0.25 * pair_loss
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if any(
            parameter.grad is None
            or not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise RuntimeError("A2 head has missing/non-finite gradients")
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


def _variant_from_weight(
    weight: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    variant_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    result = {
        "weight": weight,
        "velocity": _mixture_velocity(
            weight, batch["q0_velocity"], batch["history_velocity"],
        ),
    }
    if variant_valid is not None:
        if variant_valid.shape != weight.shape:
            raise ValueError("A2 variant validity shape differs")
        result["variant_valid"] = variant_valid.to(torch.bool)
    return result


def _counterfactual_input(
    batch: dict[str, torch.Tensor], *, kind: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if kind == "association":
        changed = deterministic_temporal_association_break(
            batch["event_feature"], batch["event_mask"],
        )
    elif kind == "continuity":
        changed = deterministic_per_event_s4_continuity_break(
            batch["event_feature"], batch["event_mask"],
        )
    else:
        raise ValueError(f"unknown A2 counterfactual: {kind}")
    pair = same_time_unordered_pair_temporal_features(
        changed["event_feature"], changed["event_mask"],
    )
    result = dict(batch)
    result["event_feature"] = changed["event_feature"]
    result["event_mask"] = changed["event_mask"]
    result["event_pair_feature"] = pair["pair_feature"]
    result["pair_mask"] = pair["pair_mask"]
    valid = changed["break_valid"]
    if kind == "association":
        valid = valid.any(dim=-1)
    else:
        valid = valid.any(dim=-1)
    return result, valid.to(torch.bool)


def _replicate_temporal_variants(
    batch: dict[str, torch.Tensor],
    event_feature: torch.Tensor,
    event_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Flatten [B,V,...] variants while preserving every static model input."""
    batch_size, variants = event_feature.shape[:2]
    event = event_feature.reshape(
        batch_size * variants, *event_feature.shape[2:],
    )
    mask = event_mask.reshape(batch_size * variants, *event_mask.shape[2:])
    pair = same_time_unordered_pair_temporal_features(event, mask)
    replicated: dict[str, torch.Tensor] = {}
    for name, value in batch.items():
        if value.ndim == 0 or value.shape[0] != batch_size:
            continue
        replicated[name] = value[:, None].expand(
            batch_size, variants, *value.shape[1:]
        ).reshape(batch_size * variants, *value.shape[1:])
    replicated["event_feature"] = event
    replicated["event_mask"] = mask
    replicated["event_pair_feature"] = pair["pair_feature"]
    replicated["pair_mask"] = pair["pair_mask"]
    return replicated


@torch.inference_mode()
def _continuity_variant(
    model: PrequentialTemporalReliabilityFusion,
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Average the complete S4 conjugacy group action, avoiding slot bias."""
    base = deterministic_per_event_s4_continuity_break(
        batch["event_feature"], batch["event_mask"],
    )
    features: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for values in permutations(range(4)):
        permutation = torch.tensor(
            values, dtype=torch.long, device=batch["event_feature"].device,
        )
        donor = conjugate_role_donor_index(
            base["donor_role_index"], permutation,
        )
        changed = deterministic_per_event_s4_continuity_break(
            batch["event_feature"], batch["event_mask"],
            role_donor_index=donor,
        )
        features.append(changed["event_feature"])
        masks.append(changed["event_mask"])
    event = torch.stack(features, dim=1)
    mask = torch.stack(masks, dim=1)
    replicated = _replicate_temporal_variants(batch, event, mask)
    output = model(**_model_kwargs(replicated))
    batch_size, variants = event.shape[:2]
    weight = output["q0_weight"].reshape(batch_size, variants)
    sample_valid = base["break_valid"].any(dim=-1).to(torch.bool)
    variant_valid = sample_valid[:, None].expand(-1, variants).clone()
    variant_valid &= output["learned_gate_valid"].reshape(batch_size, variants)
    return (
        _variant_from_weight(
            weight, batch, variant_valid=variant_valid,
        ),
        variant_valid.any(dim=1),
    )


@torch.inference_mode()
def _dose_variant(
    model: PrequentialTemporalReliabilityFusion,
    batch: dict[str, torch.Tensor],
    *,
    dose: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    changed = deterministic_endpoint_dose_temporal_break(
        batch["event_feature"], batch["event_mask"], dose=dose,
    )
    batch_size, variants = changed["event_feature"].shape[:2]
    replicated = _replicate_temporal_variants(
        batch, changed["event_feature"], changed["event_mask"],
    )
    output = model(**_model_kwargs(replicated))
    available = changed["break_valid"].unsqueeze(1)
    damaged = changed["damaged_role_mask"].unsqueeze(0)
    effective_count = (available & damaged).sum(dim=-1)
    target_count = torch.minimum(
        torch.full_like(effective_count[:, :1], dose),
        available.sum(dim=-1),
    )
    variant_valid = (
        (target_count > 0) & (effective_count == target_count)
    )
    raw_weight = output["q0_weight"].reshape(batch_size, variants)
    variant_valid &= output["learned_gate_valid"].reshape(batch_size, variants)
    return (
        _variant_from_weight(
            raw_weight, batch, variant_valid=variant_valid,
        ),
        variant_valid.any(dim=1),
    )


def _weighted_anonymous_auc(
    score: torch.Tensor,
    label: torch.Tensor,
    valid: torch.Tensor,
) -> float:
    """Weighted Mann--Whitney AUC with equal mass per recipient sample."""
    if score.ndim != 2 or valid.shape != score.shape:
        raise ValueError("A2 anonymous AUC score/validity shape differs")
    if label.shape != (score.shape[0],):
        raise ValueError("A2 anonymous AUC label shape differs")
    count = valid.sum(dim=1)
    row_weight = count.clamp_min(1).to(score.dtype).reciprocal()
    weights = row_weight[:, None].expand_as(score)[valid].detach().cpu().numpy()
    values = score[valid].detach().cpu().numpy()
    targets = (
        label[:, None].expand_as(valid)[valid].to(torch.bool)
        .detach().cpu().numpy()
    )
    weights = weights.astype(np.float64, copy=False)
    values = values.astype(np.float64, copy=False)
    targets = targets.astype(np.bool_, copy=False)
    positive_total = float(weights[targets].sum())
    negative_total = float(weights[~targets].sum())
    if positive_total <= 0.0 or negative_total <= 0.0:
        return 0.5
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    targets = targets[order]
    weights = weights[order]
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    positive = np.add.reduceat(weights * targets, starts)
    negative = np.add.reduceat(weights * (~targets), starts)
    negative_before = np.cumsum(negative) - negative
    numerator = float((positive * (negative_before + 0.5 * negative)).sum())
    return numerator / (positive_total * negative_total)


def _variant_metrics(
    value: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    common: torch.Tensor,
    preference: torch.Tensor,
) -> dict[str, Any]:
    """Score anonymous interventions as expected loss, never ensemble loss."""
    weight = value["weight"]
    velocity = value["velocity"]
    if weight.ndim == 1:
        if velocity.shape != (weight.shape[0], 3):
            raise ValueError("A2 scalar variant velocity shape differs")
        selected_weight = weight[common]
        selected_error = torch.linalg.vector_norm(
            velocity[common] - batch["target_velocity"][common], dim=-1,
        )
        coefficient_mae = float((
            selected_weight - batch["weight"][common]
        ).abs().mean())
        auc = _binary_auc(selected_weight.cpu(), preference[common].cpu())
        variant_count = 1
        valid_variant_rows = int(common.sum())
    elif weight.ndim == 2:
        if velocity.shape != (*weight.shape, 3):
            raise ValueError("A2 anonymous variant velocity shape differs")
        variant_valid = value.get("variant_valid")
        if variant_valid is None or variant_valid.shape != weight.shape:
            raise ValueError("A2 anonymous variant validity is missing")
        scored = variant_valid & common[:, None]
        count = scored.sum(dim=1)
        if bool(torch.any(common & (count == 0))):
            raise RuntimeError("A2 common row has no valid anonymous variant")
        coefficient_error = (
            weight - batch["weight"][:, None]
        ).abs()
        sample_coefficient_error = (
            coefficient_error * scored.to(coefficient_error.dtype)
        ).sum(dim=1) / count.clamp_min(1).to(coefficient_error.dtype)
        variant_error = torch.linalg.vector_norm(
            velocity - batch["target_velocity"][:, None], dim=-1,
        )
        selected_error = (
            variant_error * scored.to(variant_error.dtype)
        ).sum(dim=1) / count.clamp_min(1).to(variant_error.dtype)
        selected_error = selected_error[common]
        selected_weight = (
            weight * scored.to(weight.dtype)
        ).sum(dim=1) / count.clamp_min(1).to(weight.dtype)
        selected_weight = selected_weight[common]
        coefficient_mae = float(sample_coefficient_error[common].mean())
        auc = _weighted_anonymous_auc(weight, preference, scored)
        variant_count = int(weight.shape[1])
        valid_variant_rows = int(scored.sum())
    else:
        raise ValueError("A2 variant weight rank differs")
    return {
        "component_preference_auc": auc,
        "coefficient_mae": coefficient_mae,
        "error_mps": _distribution(selected_error),
        "q0_weight": _distribution(selected_weight),
        "anonymous_variant_count": variant_count,
        "valid_variant_row_count": valid_variant_rows,
        "anonymous_metric_definition": (
            "per-sample-expected-loss-and-weighted-flattened-auc"
            if weight.ndim == 2 else "single-prediction"
        ),
    }


@torch.inference_mode()
def _evaluate_head(
    model: PrequentialTemporalReliabilityFusion,
    arms: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    result: dict[str, Any] = {"arms": {}}
    predictions: dict[str, dict[str, torch.Tensor]] = {}
    for arm_name, arm in arms.items():
        batch = {name: value.to(device) for name, value in arm.items()}
        full_output = model(**_model_kwargs(batch))
        base_valid = _common(batch, full_output)
        variants: dict[str, dict[str, torch.Tensor]] = {
            "full": {
                "weight": full_output["q0_weight"],
                "velocity": full_output["velocity_mps"],
            },
            "local_disabled": _variant_from_weight(
                model(**_model_kwargs(batch), local_disabled=True)["q0_weight"],
                batch,
            ),
            "temporal_collapsed": _variant_from_weight(
                model(
                    **_model_kwargs(batch), temporal_collapsed=True,
                )["q0_weight"], batch,
            ),
            "pair_message_disabled": _variant_from_weight(
                model(
                    **_model_kwargs(batch), pair_message_disabled=True,
                )["q0_weight"], batch,
            ),
        }
        sample_valid: dict[str, torch.Tensor] = {
            name: base_valid.clone() for name in (
                "full", "local_disabled", "temporal_collapsed",
                "pair_message_disabled",
            )
        }
        association_batch, association_valid = _counterfactual_input(
            batch, kind="association",
        )
        output = model(**_model_kwargs(association_batch))
        variants["association_break"] = _variant_from_weight(
            output["q0_weight"], batch,
        )
        sample_valid["association_break"] = (
            base_valid & association_valid & output["learned_gate_valid"]
        )
        continuity_value, continuity_valid = _continuity_variant(model, batch)
        variants["continuity_break"] = continuity_value
        sample_valid["continuity_break"] = base_valid & continuity_valid
        dose_valid: dict[int, torch.Tensor] = {}
        for dose in DOSE_VALUES:
            value, valid = _dose_variant(model, batch, dose=dose)
            variants[f"dose{dose}"] = value
            dose_valid[dose] = base_valid & valid
        dose_common = base_valid.clone()
        for valid in dose_valid.values():
            dose_common &= valid
        for dose in DOSE_VALUES:
            sample_valid[f"dose{dose}"] = dose_common
        q0_error = torch.linalg.vector_norm(
            batch["q0_velocity"] - batch["target_velocity"], dim=-1,
        )
        history_error = torch.linalg.vector_norm(
            batch["history_velocity"] - batch["target_velocity"], dim=-1,
        )
        oracle_velocity = _mixture_velocity(
            batch["weight"], batch["q0_velocity"], batch["history_velocity"],
        )
        predictions[arm_name] = {
            "weight": full_output["q0_weight"].detach().cpu(),
            "recipient_global": arm["recipient_global"],
            "valid": base_valid.detach().cpu(),
            "motion_class": arm["motion_class"],
        }
        arm_result: dict[str, Any] = {
            "full_count": int(base_valid.numel()),
            "common_count": int(base_valid.sum()),
            "common_fraction": float(base_valid.double().mean()),
            "state_supported_count": int(full_output["state_supported"].sum()),
            "state_supported_fraction": float(
                full_output["state_supported"].double().mean()
            ),
            "fallback_count": int(full_output["fallback_used"].sum()),
            "groups": {},
        }
        for group, group_mask in {
            "overall": torch.ones_like(base_valid),
            "rotation": batch["motion_class"] == 2,
            "combined": batch["motion_class"] == 3,
        }.items():
            common = base_valid & group_mask
            if not bool(common.any()):
                raise RuntimeError(f"A2 empty {arm_name}/{group} common rows")
            preference = q0_error[common] < history_error[common]
            group_result: dict[str, Any] = {
                "full_count": int(group_mask.sum()),
                "common_count": int(common.sum()),
                "common_fraction": float(
                    common.sum().double() / group_mask.sum().clamp_min(1)
                ),
                "parent_error_mps": _distribution(torch.linalg.vector_norm(
                    batch["old_velocity"][common]
                    - batch["target_velocity"][common], dim=-1,
                )),
                "oracle_error_mps": _distribution(torch.linalg.vector_norm(
                    oracle_velocity[common] - batch["target_velocity"][common],
                    dim=-1,
                )),
                "blind_error_mps": _distribution(history_error[common]),
                "q0_better_count": int(preference.sum()),
                "history_better_or_equal_count": int((~preference).sum()),
                "variants": {},
            }
            for name in VARIANT_NAMES:
                mechanism_common = sample_valid[name] & group_mask
                if not bool(mechanism_common.any()):
                    raise RuntimeError(
                        f"A2 empty {arm_name}/{group}/{name} eligible rows"
                    )
                metric = _variant_metrics(
                    variants[name], batch, mechanism_common,
                    q0_error < history_error,
                )
                metric.update({
                    "eligible_count": int(mechanism_common.sum()),
                    "eligible_fraction": float(
                        mechanism_common.sum().double()
                        / group_mask.sum().clamp_min(1)
                    ),
                })
                if name in {
                    "association_break", "continuity_break",
                    "dose1", "dose2", "dose4",
                }:
                    metric["reference_full"] = _variant_metrics(
                        variants["full"], batch, mechanism_common,
                        q0_error < history_error,
                    )
                group_result["variants"][name] = metric
            arm_result["groups"][group] = group_result
        result["arms"][arm_name] = arm_result
    intact_lookup = {
        int(value): index for index, value in enumerate(
            predictions["intact"]["recipient_global"].tolist()
        )
    }
    for arm_name in ("global", "hard"):
        reference = torch.tensor([
            intact_lookup[int(value)]
            for value in predictions[arm_name]["recipient_global"].tolist()
        ], dtype=torch.long)
        paired_valid = (
            predictions["intact"]["valid"][reference]
            & predictions[arm_name]["valid"]
        )
        motion = predictions[arm_name]["motion_class"]
        for group, group_mask in {
            "overall": torch.ones_like(paired_valid),
            "rotation": motion == 2,
            "combined": motion == 3,
        }.items():
            paired = paired_valid & group_mask
            if not bool(paired.any()):
                raise RuntimeError(
                    f"A2 empty {arm_name}/{group} paired population"
                )
            result["arms"][arm_name]["groups"][group].update({
                "paired_count": int(paired.sum()),
                "paired_fraction": float(
                    paired.sum().double() / group_mask.sum().clamp_min(1)
                ),
                "paired_weight_separation_mean": float((
                    predictions["intact"]["weight"][reference][paired]
                    - predictions[arm_name]["weight"][paired]
                ).mean()),
            })
    return result


def _recovery(
    parent: dict[str, float], oracle: dict[str, float],
    value: dict[str, float], key: str,
) -> float:
    headroom = parent[key] - oracle[key]
    return (parent[key] - value[key]) / headroom if headroom > 0 else 0.0


def _fold_gates(metrics: dict[str, Any]) -> dict[str, bool | float]:
    gates: dict[str, bool | float] = {}
    arms = metrics["arms"]
    for arm in ("global", "hard"):
        gates[f"{arm}_state_fraction_ge_0_95"] = (
            arms[arm]["state_supported_fraction"] >= 0.95
        )
        gates[f"{arm}_common_fraction_ge_0_95"] = (
            arms[arm]["common_fraction"] >= 0.95
        )
    for group in ("overall", "rotation", "combined"):
        intact = arms["intact"]["groups"][group]
        variants = intact["variants"]
        full = variants["full"]
        local_disabled = variants["local_disabled"]
        collapsed = variants["temporal_collapsed"]
        pair_disabled = variants["pair_message_disabled"]
        association = variants["association_break"]
        continuity = variants["continuity_break"]
        parent, oracle = intact["parent_error_mps"], intact["oracle_error_mps"]
        recovery_mean = _recovery(parent, oracle, full["error_mps"], "mean")
        recovery_p50 = _recovery(parent, oracle, full["error_mps"], "p50")
        local_auc_gain = (
            full["component_preference_auc"]
            - local_disabled["component_preference_auc"]
        )
        local_mae_improvement = (
            local_disabled["coefficient_mae"] - full["coefficient_mae"]
        ) / max(local_disabled["coefficient_mae"], 1e-8)
        temporal_auc_gain = (
            full["component_preference_auc"]
            - collapsed["component_preference_auc"]
        )
        temporal_mae_improvement = (
            collapsed["coefficient_mae"] - full["coefficient_mae"]
        ) / max(collapsed["coefficient_mae"], 1e-8)
        pair_auc_gain = (
            full["component_preference_auc"]
            - pair_disabled["component_preference_auc"]
        )
        association_reference = association["reference_full"]
        association_auc_drop = (
            association_reference["component_preference_auc"]
            - association["component_preference_auc"]
        )
        association_mae_degrade = (
            association["coefficient_mae"]
            - association_reference["coefficient_mae"]
        ) / max(association_reference["coefficient_mae"], 1e-8)
        continuity_reference = continuity["reference_full"]
        continuity_auc_drop = (
            continuity_reference["component_preference_auc"]
            - continuity["component_preference_auc"]
        )
        continuity_mae_degrade = (
            continuity["coefficient_mae"]
            - continuity_reference["coefficient_mae"]
        ) / max(continuity_reference["coefficient_mae"], 1e-8)
        dose = [variants[f"dose{value}"]["coefficient_mae"] for value in DOSE_VALUES]
        dose_reference = variants["dose4"]["reference_full"]
        dose4_degrade = (dose[-1] - dose_reference["coefficient_mae"]) / max(
            dose_reference["coefficient_mae"], 1e-8
        )
        prefix = f"{group}_intact"
        gates[f"{prefix}_auc"] = full["component_preference_auc"]
        gates[f"{prefix}_auc_ge_0_75"] = full["component_preference_auc"] >= 0.75
        coefficient_limit = 0.28 if group == "overall" else 0.32
        gates[f"{prefix}_coefficient_mae"] = full["coefficient_mae"]
        gates[f"{prefix}_coefficient_mae_within_limit"] = (
            full["coefficient_mae"] <= coefficient_limit
        )
        gates[f"{prefix}_recovery_mean"] = recovery_mean
        gates[f"{prefix}_recovery_p50"] = recovery_p50
        gates[f"{prefix}_recovery_mean_ge_0_30"] = recovery_mean >= 0.30
        gates[f"{prefix}_recovery_p50_ge_0_30"] = recovery_p50 >= 0.30
        for name, value in (
            ("local_auc_gain", local_auc_gain),
            ("temporal_auc_gain", temporal_auc_gain),
            ("association_auc_drop", association_auc_drop),
            ("continuity_auc_drop", continuity_auc_drop),
        ):
            gates[f"{prefix}_{name}"] = value
            gates[f"{prefix}_{name}_ge_0_03"] = value >= 0.03
        gates[f"{prefix}_pair_message_auc_gain"] = pair_auc_gain
        gates[f"{prefix}_pair_message_auc_gain_ge_0_02"] = (
            pair_auc_gain >= 0.02
        )
        for name, value in (
            ("local_mae_improvement", local_mae_improvement),
            ("temporal_mae_improvement", temporal_mae_improvement),
            ("association_mae_degrade", association_mae_degrade),
            ("continuity_mae_degrade", continuity_mae_degrade),
        ):
            gates[f"{prefix}_{name}"] = value
            gates[f"{prefix}_{name}_ge_0_10"] = value >= 0.10
        gates[f"{prefix}_dose1_mae"] = dose[0]
        gates[f"{prefix}_dose2_mae"] = dose[1]
        gates[f"{prefix}_dose4_mae"] = dose[2]
        gates[f"{prefix}_dose_nondecreasing"] = (
            dose[1] >= 0.98 * dose[0] and dose[2] >= 0.98 * dose[1]
        )
        gates[f"{prefix}_dose4_mae_degrade"] = dose4_degrade
        gates[f"{prefix}_dose4_mae_degrade_ge_0_10"] = dose4_degrade >= 0.10
        gates[f"{prefix}_association_eligible_fraction"] = (
            association["eligible_fraction"]
        )
        gates[f"{prefix}_association_eligible_fraction_ge_0_30"] = (
            association["eligible_fraction"] >= 0.30
        )
        gates[f"{prefix}_continuity_eligible_fraction"] = (
            continuity["eligible_fraction"]
        )
        gates[f"{prefix}_continuity_eligible_fraction_ge_0_95"] = (
            continuity["eligible_fraction"] >= 0.95
        )
        gates[f"{prefix}_dose_eligible_fraction"] = (
            variants["dose4"]["eligible_fraction"]
        )
        gates[f"{prefix}_dose_eligible_fraction_ge_0_30"] = (
            variants["dose4"]["eligible_fraction"] >= 0.30
        )
        gates[f"{prefix}_common_fraction_ge_0_95"] = (
            intact["common_fraction"] >= 0.95
        )
        for arm in ("global", "hard"):
            corrupt = arms[arm]["groups"][group]
            corrupt_full = corrupt["variants"]["full"]
            error = corrupt_full["error_mps"]
            blind = corrupt["blind_error_mps"]
            gates[f"{group}_{arm}_auc_ge_0_75"] = (
                corrupt_full["component_preference_auc"] >= 0.75
            )
            gates[f"{group}_{arm}_common_fraction_ge_0_95"] = (
                corrupt["common_fraction"] >= 0.95
            )
            gates[f"{group}_{arm}_paired_fraction_ge_0_95"] = (
                corrupt["paired_fraction"] >= 0.95
            )
            gates[f"{group}_{arm}_weight_separation_ge_0_15"] = (
                corrupt["paired_weight_separation_mean"] >= 0.15
            )
            gates[f"{group}_{arm}_mean_within_blind_1_02"] = (
                error["mean"] <= 1.02 * blind["mean"]
            )
            gates[f"{group}_{arm}_p50_within_blind_1_02"] = (
                error["p50"] <= 1.02 * blind["p50"]
            )
            gates[f"{group}_{arm}_p95_within_blind_1_10"] = (
                error["p95"] <= 1.10 * blind["p95"]
            )
    return gates


def _cross_fold_gates(cv: list[dict[str, Any]]) -> dict[str, bool | float]:
    if len(cv) != FIXED_FOLDS:
        raise ValueError("A2 cross-fold gate needs two folds")
    gates: dict[str, bool | float] = {}
    for group in ("overall", "rotation", "combined"):
        left = cv[0]["metrics"]["arms"]["intact"]["groups"][group]
        right = cv[1]["metrics"]["arms"]["intact"]["groups"][group]
        lv, rv = left["variants"], right["variants"]
        lp, rp = left["parent_error_mps"], right["parent_error_mps"]
        lo, ro = left["oracle_error_mps"], right["oracle_error_mps"]
        values = {
            "auc": abs(
                lv["full"]["component_preference_auc"]
                - rv["full"]["component_preference_auc"]
            ),
            "mae": abs(
                lv["full"]["coefficient_mae"] - rv["full"]["coefficient_mae"]
            ),
            "recovery_mean": abs(
                _recovery(lp, lo, lv["full"]["error_mps"], "mean")
                - _recovery(rp, ro, rv["full"]["error_mps"], "mean")
            ),
            "recovery_p50": abs(
                _recovery(lp, lo, lv["full"]["error_mps"], "p50")
                - _recovery(rp, ro, rv["full"]["error_mps"], "p50")
            ),
        }
        mechanism: dict[str, tuple[float, float]] = {
            "local_auc_gain": (
                lv["full"]["component_preference_auc"]
                - lv["local_disabled"]["component_preference_auc"],
                rv["full"]["component_preference_auc"]
                - rv["local_disabled"]["component_preference_auc"],
            ),
            "local_mae_improvement": (
                (lv["local_disabled"]["coefficient_mae"]
                 - lv["full"]["coefficient_mae"])
                / max(lv["local_disabled"]["coefficient_mae"], 1e-8),
                (rv["local_disabled"]["coefficient_mae"]
                 - rv["full"]["coefficient_mae"])
                / max(rv["local_disabled"]["coefficient_mae"], 1e-8),
            ),
            "temporal_auc_gain": (
                lv["full"]["component_preference_auc"]
                - lv["temporal_collapsed"]["component_preference_auc"],
                rv["full"]["component_preference_auc"]
                - rv["temporal_collapsed"]["component_preference_auc"],
            ),
            "temporal_mae_improvement": (
                (lv["temporal_collapsed"]["coefficient_mae"]
                 - lv["full"]["coefficient_mae"])
                / max(lv["temporal_collapsed"]["coefficient_mae"], 1e-8),
                (rv["temporal_collapsed"]["coefficient_mae"]
                 - rv["full"]["coefficient_mae"])
                / max(rv["temporal_collapsed"]["coefficient_mae"], 1e-8),
            ),
            "pair_message_auc_gain": (
                lv["full"]["component_preference_auc"]
                - lv["pair_message_disabled"]["component_preference_auc"],
                rv["full"]["component_preference_auc"]
                - rv["pair_message_disabled"]["component_preference_auc"],
            ),
            "association_auc_drop": (
                lv["association_break"]["reference_full"][
                    "component_preference_auc"
                ]
                - lv["association_break"]["component_preference_auc"],
                rv["association_break"]["reference_full"][
                    "component_preference_auc"
                ]
                - rv["association_break"]["component_preference_auc"],
            ),
            "association_mae_degrade": (
                (lv["association_break"]["coefficient_mae"]
                 - lv["association_break"]["reference_full"][
                     "coefficient_mae"
                 ])
                / max(lv["association_break"]["reference_full"][
                    "coefficient_mae"
                ], 1e-8),
                (rv["association_break"]["coefficient_mae"]
                 - rv["association_break"]["reference_full"][
                     "coefficient_mae"
                 ])
                / max(rv["association_break"]["reference_full"][
                    "coefficient_mae"
                ], 1e-8),
            ),
            "continuity_auc_drop": (
                lv["continuity_break"]["reference_full"][
                    "component_preference_auc"
                ]
                - lv["continuity_break"]["component_preference_auc"],
                rv["continuity_break"]["reference_full"][
                    "component_preference_auc"
                ]
                - rv["continuity_break"]["component_preference_auc"],
            ),
            "continuity_mae_degrade": (
                (lv["continuity_break"]["coefficient_mae"]
                 - lv["continuity_break"]["reference_full"][
                     "coefficient_mae"
                 ])
                / max(lv["continuity_break"]["reference_full"][
                    "coefficient_mae"
                ], 1e-8),
                (rv["continuity_break"]["coefficient_mae"]
                 - rv["continuity_break"]["reference_full"][
                     "coefficient_mae"
                 ])
                / max(rv["continuity_break"]["reference_full"][
                    "coefficient_mae"
                ], 1e-8),
            ),
        }
        limits = {
            "auc": 0.10, "mae": 0.05,
            "recovery_mean": 0.15, "recovery_p50": 0.15,
        }
        for name, value in values.items():
            gates[f"{group}_{name}_fold_delta"] = value
            gates[f"{group}_{name}_fold_delta_le_{limits[name]:.2f}"] = (
                value <= limits[name]
            )
        for name, (lhs, rhs) in mechanism.items():
            delta = abs(lhs - rhs)
            gates[f"{group}_{name}_fold_delta"] = delta
            gates[f"{group}_{name}_fold_delta_le_0_05"] = delta <= 0.05
    for arm in ("global", "hard"):
        for group in ("overall", "rotation", "combined"):
            left = cv[0]["metrics"]["arms"][arm]["groups"][group]
            right = cv[1]["metrics"]["arms"][arm]["groups"][group]
            auc_delta = abs(
                left["variants"]["full"]["component_preference_auc"]
                - right["variants"]["full"]["component_preference_auc"]
            )
            separation_delta = abs(
                left["paired_weight_separation_mean"]
                - right["paired_weight_separation_mean"]
            )
            gates[f"{arm}_{group}_auc_fold_delta"] = auc_delta
            gates[f"{arm}_{group}_auc_fold_delta_le_0_10"] = auc_delta <= 0.10
            gates[f"{arm}_{group}_separation_fold_delta"] = separation_delta
            gates[f"{arm}_{group}_separation_fold_delta_le_0_05"] = (
                separation_delta <= 0.05
            )
    return gates


def _cv_passed(
    cv: list[dict[str, Any]], cross: dict[str, bool | float],
) -> bool:
    return (
        len(cv) == FIXED_FOLDS
        and all(item.get("passed") is True for item in cv)
        and _all_boolean_gates_pass(cross)
    )


def validate_a2_artifacts(output: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail-closed loader for a completed passed or failed A2 screen."""
    root = Path(output).resolve()
    result_path = root / "screen_result.json"
    state_path = root / "run_state.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    run_state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        result.get("schema_version") != RUN_SCHEMA
        or run_state.get("schema_version") != RUN_SCHEMA
        or result.get("status") not in {"passed", "failed"}
        or run_state.get("status") != result.get("status")
        or result.get("diagnostic_only") is not True
        or result.get("truth_omega_feature_construction") is not True
        or result.get("train_only") is not True
        or result.get("formal_v15") is not False
        or result.get("validation_accessed") is not False
        or result.get("test_accessed") is not False
        or result.get("future_modules_loaded") is not False
        or result.get("authorized_formal_two_stage") is not False
        or run_state.get("validation_accessed") is not False
        or run_state.get("test_accessed") is not False
        or run_state.get("future_modules_loaded") is not False
    ):
        raise ValueError("A2 artifact status/scope differs")
    passed = result["status"] == "passed"
    if (
        result.get("authorized_a2_train_counterfactual") is not passed
        or run_state.get("authorized_a2_train_counterfactual") is not passed
    ):
        raise ValueError("A2 authorization differs")
    contract = result.get("experiment_contract")
    contract_sha = _json_sha256(contract)
    if (
        not isinstance(contract, dict)
        or contract.get("schema_version") != RUN_SCHEMA
        or contract.get("updates") != FIXED_UPDATES
        or contract.get("width") != FIXED_WIDTH
        or contract.get("seed") != 20260730
        or contract.get("batch_size") != FIXED_BATCH_SIZE
        or contract.get("expert_batch_size") != FIXED_EXPERT_BATCH_SIZE
        or contract.get("folds") != FIXED_FOLDS
        or contract.get("block_size") != BLOCK_SIZE
        or contract.get("minimum_fit_events") != MINIMUM_FIT_EVENTS
        or contract.get("event_width") != EVENT_WIDTH
        or contract.get("pair_width") != PAIR_WIDTH
        or contract.get("global_width") != GLOBAL_WIDTH
        or contract.get("hard_map_policy") != HARD_MAP_POLICY
        or contract.get("arm_weight") != ARM_WEIGHT
        or contract.get("local_auxiliary_weight") != 0.50
        or contract.get("paired_corrupt_delta_weight") != 0.25
        or contract.get("truth_velocity_loss_only") is not True
        or contract.get("temporal_association_static_prefix_width") != 3
        or contract.get("leave_block_out_scope") != "profile_fit_masks_only"
        or contract.get("leave_block_out_requires_complete_block") is not True
        or contract.get("crossfit_support_unit") != "distinct_event_exposure"
        or contract.get("anonymous_intervention_metric") != (
            "per-sample-expected-loss-and-weighted-flattened-auc"
        )
        or contract.get("pair_message_auc_gain_gate") != 0.02
        or contract.get("corruption_selection_gates") != "sealed-a1"
        or contract.get("dose_semantics") != (
            "saturating-observable-temporal-roles"
        )
        or contract.get("endpoint_features_use_original_events") is not True
        or contract.get("leave_block_out_profile_refits") != ["q0", "history"]
        or contract.get("forward_excludes") != [
            "session", "motion_class", "physical_armor_id", "truth_velocity",
            "future", "absolute_pose",
        ]
        or contract.get("a1_r2_result_sha256") != A1_R2_RESULT_SHA256
        or result.get("experiment_contract_sha256") != contract_sha
        or run_state.get("experiment_contract_sha256") != contract_sha
    ):
        raise ValueError("A2 fixed experiment contract differs")
    if {
        name: sha256_file(path) for name, path in _source_paths().items()
    } != contract.get("source_sha256"):
        raise ValueError("A2 source binding differs")
    if run_state.get("screen_result_sha256") != sha256_file(result_path):
        raise ValueError("A2 result/run-state binding differs")
    a1_path = Path(result["a1_r2"]["result"]).resolve()
    if (
        sha256_file(a1_path) != result["a1_r2"].get("sha256")
        or result["a1_r2"].get("sha256") != A1_R2_RESULT_SHA256
    ):
        raise ValueError("A2 A1-R2 binding differs")
    validate_a1_r2_rejection(a1_path)
    parent_path = Path(result["parent"]["checkpoint"]).resolve()
    parent_sha = sha256_file(parent_path)
    if (
        parent_sha != contract.get("parent_checkpoint_sha256")
        or parent_sha != result["parent"].get("sha256")
    ):
        raise ValueError("A2 parent checkpoint differs")
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent_state_sha = state_dict_sha256(parent_payload["model"])
    if (
        parent_state_sha != contract.get("parent_model_state_dict_sha256")
        or parent_state_sha != result["parent"].get("model_state_dict_sha256")
    ):
        raise ValueError("A2 parent model state differs")
    for path_key, sha_key in (
        ("dataset_manifest_path", "dataset_manifest_sha256"),
        ("truth_manifest_path", "truth_manifest_sha256"),
    ):
        if sha256_file(Path(contract[path_key]).resolve()) != contract.get(sha_key):
            raise ValueError(f"A2 {path_key} differs")
    if (
        result.get("dataset_manifest_sha256")
        != contract.get("dataset_manifest_sha256")
        or result.get("truth_manifest_sha256")
        != contract.get("truth_manifest_sha256")
    ):
        raise ValueError("A2 data result binding differs")
    frozen_paths = contract["frozen_checkpoint_paths"]
    if {
        name: sha256_file(Path(frozen_paths[name]).resolve())
        for name in ("mapper", "s", "h")
    } != contract.get("frozen_checkpoint_sha256"):
        raise ValueError("A2 frozen files differ")
    mapper, _ = load_frozen_pnp_mapper(Path(frozen_paths["mapper"]).resolve())
    s_model, _ = load_frozen_v19(Path(frozen_paths["s"]).resolve())
    h_model, _ = load_frozen_hypothesis_adapter(
        Path(frozen_paths["h"]).resolve(), allow_diagnostic=True,
    )
    if {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    } != contract.get("frozen_state_dict_sha256"):
        raise ValueError("A2 frozen model states differ")
    cv = result.get("cv")
    if not isinstance(cv, list) or len(cv) != FIXED_FOLDS:
        raise ValueError("A2 fold count differs")
    for item in cv:
        gates = _fold_gates(item["metrics"])
        if item.get("gates") != gates or item.get("passed") is not (
            _all_boolean_gates_pass(gates)
        ):
            raise ValueError("A2 recomputed fold gates differ")
        for maps in (item["train_maps"], item["heldout_maps"]):
            _validate_corruption_manifest(maps["global"], kind="global")
            _validate_corruption_manifest(maps["hard"], kind="hard")
    cross = _cross_fold_gates(cv)
    if result.get("cross_fold_gates") != cross or passed is not _cv_passed(cv, cross):
        raise ValueError("A2 recomputed cross-fold gates differ")
    checkpoint_path = Path(result["fold_checkpoint"]).resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    if (
        checkpoint_path.parent != root
        or checkpoint_sha != result.get("fold_checkpoint_sha256")
        or checkpoint_sha != run_state.get("fold_checkpoint_sha256")
    ):
        raise ValueError("A2 checkpoint binding differs")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    states = payload.get("fold_models")
    state_hashes = payload.get("fold_state_dict_sha256")
    if (
        payload.get("schema_version") != RUN_SCHEMA
        or payload.get("experiment_contract") != contract
        or payload.get("experiment_contract_sha256") != contract_sha
        or payload.get("a1_r2_result_sha256") != A1_R2_RESULT_SHA256
        or not isinstance(states, list) or len(states) != FIXED_FOLDS
        or state_hashes != [state_dict_sha256(state) for state in states]
        or result.get("fold_state_dict_sha256") != state_hashes
    ):
        raise ValueError("A2 checkpoint payload differs")
    for state in states:
        model = PrequentialTemporalReliabilityFusion(
            event_width=EVENT_WIDTH, pair_width=PAIR_WIDTH,
            global_width=GLOBAL_WIDTH, width=FIXED_WIDTH,
        )
        model.load_state_dict(state, strict=True)
        if not bool(model.normalizer_fitted):
            raise ValueError("A2 checkpoint normalizer is not fitted")
        for name in ("event", "pair", "global"):
            mean, scale = getattr(model, f"{name}_mean"), getattr(model, f"{name}_scale")
            if (
                not bool(torch.isfinite(mean).all())
                or not bool(torch.isfinite(scale).all())
                or not bool((scale > 0).all())
            ):
                raise ValueError("A2 checkpoint normalizer is invalid")
    return payload, result


def _seal_preflight_failure(
    output: Path, *, contract_sha: str, details: dict[str, Any],
) -> dict[str, Any]:
    result = {
        **details, "status": "preflight_failed",
        "validation_accessed": False, "test_accessed": False,
        "future_modules_loaded": False,
        "authorized_a2_train_counterfactual": False,
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.seed != 20260730 or args.updates != FIXED_UPDATES
        or args.width != FIXED_WIDTH or args.batch_size != FIXED_BATCH_SIZE
        or args.expert_batch_size != FIXED_EXPERT_BATCH_SIZE
    ):
        raise ValueError("V15-A2 budget/architecture is fixed")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V15-A2 requires local CUDA")
    git_start = _git_state()
    if git_start["git_commit"] == "unknown" or git_start["worktree_dirty"]:
        raise RuntimeError("V15-A2 requires a clean known git commit")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing A2 output: {output}")
    output.mkdir(parents=True)
    _atomic_json(output / "run_state.json", {
        "schema_version": RUN_SCHEMA, "status": "initializing",
        "validation_accessed": False, "test_accessed": False,
        "future_modules_loaded": False,
    })
    a1_path = Path(args.a1_r2_result).resolve()
    a1 = validate_a1_r2_rejection(a1_path)
    if a1_path.is_dir():
        a1_path = a1_path / "screen_result.json"
    parent_checkpoint = Path(args.parent_checkpoint).resolve()
    if parent_checkpoint != Path(a1["parent"]["checkpoint"]).resolve():
        raise ValueError("A2 parent differs from A1-R2")
    parent_root = parent_checkpoint.parent.parent
    parent_manifest_path = parent_root / "run_manifest.json"
    parent_result_path = parent_root / "screen_result.json"
    parent_manifest_sha = sha256_file(parent_manifest_path)
    parent_result_sha = sha256_file(parent_result_path)
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
    parent_sha = sha256_file(parent_checkpoint)
    parent_contract = _validate_checkpoint_manifest_identity(
        parent_payload, parent_manifest, parent_sha,
    )
    parent_args = parent_contract["args"]
    dataset_path = Path(parent_args["dataset"]).resolve()
    truth_path = Path(parent_args["truth_history"]).resolve()
    mapper_path = Path(parent_args["mapper_checkpoint"]).resolve()
    s_path = Path(parent_args["s_checkpoint"]).resolve()
    h_path = Path(parent_args["h_checkpoint"]).resolve()
    _seed(args.seed)
    train_dataset = _load_train_dataset(dataset_path)
    dataset_manifest_path = dataset_path / "dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("test_accessed") is not False:
        raise ValueError("A2 paired dataset accessed test")
    truth_sha = dataset_manifest["truth_history_manifest_sha256"]
    train_truth = SplitScopedTruthIndex(
        truth_path, split="train", expected_manifest_sha256=truth_sha,
    )
    train_truth.attach(train_dataset)
    mapper, _ = load_frozen_pnp_mapper(mapper_path)
    s_model, _ = load_frozen_v19(s_path)
    h_model, _ = load_frozen_hypothesis_adapter(h_path, allow_diagnostic=True)
    for frozen in (mapper, s_model, h_model):
        frozen.to(device).eval().requires_grad_(False)
    frozen_files = {
        "mapper": sha256_file(mapper_path), "s": sha256_file(s_path),
        "h": sha256_file(h_path),
    }
    frozen_states = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    }
    if (
        frozen_files != parent_contract["frozen_checkpoint_sha256"]
        or frozen_states != parent_contract["frozen_state_dict_sha256"]
    ):
        raise ValueError("A2 frozen Mapper/S/H differs")
    a1_contract = a1["experiment_contract"]
    if (
        sha256_file(dataset_path / "dataset_manifest.json")
        != a1_contract.get("dataset_manifest_sha256")
        or truth_sha != a1_contract.get("truth_manifest_sha256")
        or frozen_files != a1_contract.get("frozen_checkpoint_sha256")
        or frozen_states != a1_contract.get("frozen_state_dict_sha256")
    ):
        raise ValueError("A2 data/frozen dependencies differ from A1-R2")
    base = CenterPriorProfiledTwistScreen(
        width=int(parent_args["width"]),
        center_precision=float(parent_args["center_precision"]),
    ).to(device)
    base.load_state_dict(parent_payload["model"], strict=True)
    base.eval().requires_grad_(False)
    base_sha = state_dict_sha256(base.state_dict())
    source_paths = _source_paths()
    source_sha = {name: sha256_file(path) for name, path in source_paths.items()}
    contract = {
        "schema_version": RUN_SCHEMA, "diagnostic_only": True,
        "truth_omega_feature_construction": True, "train_only": True,
        "a1_r2_result_sha256": A1_R2_RESULT_SHA256,
        "parent_checkpoint_sha256": parent_sha,
        "parent_model_state_dict_sha256": base_sha,
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "truth_manifest_path": str(truth_path / "dataset_manifest.json"),
        "truth_manifest_sha256": truth_sha,
        "frozen_checkpoint_paths": {
            "mapper": str(mapper_path), "s": str(s_path), "h": str(h_path),
        },
        "frozen_checkpoint_sha256": frozen_files,
        "frozen_state_dict_sha256": frozen_states,
        "source_sha256": source_sha, "git_commit": git_start["git_commit"],
        "folds": FIXED_FOLDS, "updates": args.updates,
        "width": args.width, "seed": args.seed,
        "batch_size": args.batch_size,
        "expert_batch_size": args.expert_batch_size,
        "block_size": BLOCK_SIZE, "minimum_fit_events": MINIMUM_FIT_EVENTS,
        "event_width": EVENT_WIDTH, "pair_width": PAIR_WIDTH,
        "global_width": GLOBAL_WIDTH, "hard_map_policy": HARD_MAP_POLICY,
        "arm_weight": ARM_WEIGHT,
        "local_auxiliary_weight": 0.50,
        "paired_corrupt_delta_weight": 0.25,
        "truth_velocity_loss_only": True,
        "temporal_association_static_prefix_width": 3,
        "leave_block_out_scope": "profile_fit_masks_only",
        "leave_block_out_requires_complete_block": True,
        "crossfit_support_unit": "distinct_event_exposure",
        "anonymous_intervention_metric": (
            "per-sample-expected-loss-and-weighted-flattened-auc"
        ),
        "pair_message_auc_gain_gate": 0.02,
        "corruption_selection_gates": "sealed-a1",
        "dose_semantics": "saturating-observable-temporal-roles",
        "endpoint_features_use_original_events": True,
        "leave_block_out_profile_refits": ["q0", "history"],
        "forward_excludes": [
            "session", "motion_class", "physical_armor_id", "truth_velocity",
            "future", "absolute_pose",
        ],
    }
    contract_sha = _json_sha256(contract)
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
    motion = prepared["motion_class"].numpy().astype(np.int64)
    support = prepared["q0_supported"].sum(dim=-1).numpy().astype(np.int64)
    strata = [(int(motion[i]), int(support[i])) for i in range(len(motion))]
    folds = _fold_assignments(sessions, folds=FIXED_FOLDS)
    plans: list[dict[str, Any]] = []
    preflight: list[dict[str, Any]] = []
    try:
        for fold, heldout_sessions in enumerate(folds):
            heldout = np.asarray([
                i for i, session in enumerate(sessions) if session in heldout_sessions
            ], dtype=np.int64)
            training = np.asarray([
                i for i, session in enumerate(sessions) if session not in heldout_sessions
            ], dtype=np.int64)
            audit: dict[str, Any] = {
                "fold": fold, "status": "running",
                "train_count": int(training.size),
                "heldout_count": int(heldout.size),
            }
            preflight.append(audit)
            train_indices, train_maps = _build_corruption_maps(
                training, keys, sessions, strata, motion,
                domain=f"train/a2/cv{fold}/fit",
            )
            heldout_indices, heldout_maps = _build_corruption_maps(
                heldout, keys, sessions, strata, motion,
                domain=f"train/a2/cv{fold}/heldout",
            )
            audit["train_maps"] = train_maps
            audit["heldout_maps"] = heldout_maps
            audit["train_blocks"] = _block_availability(
                prepared["history_event_mask"][torch.from_numpy(training)],
                prepared["motion_class"][torch.from_numpy(training)],
            )
            audit["heldout_blocks"] = _block_availability(
                prepared["history_event_mask"][torch.from_numpy(heldout)],
                prepared["motion_class"][torch.from_numpy(heldout)],
            )
            audit["status"] = "passed"
            plans.append({
                "fold": fold, "heldout_sessions": heldout_sessions,
                "training": training, "heldout": heldout,
                "train_indices": train_indices,
                "heldout_indices": heldout_indices,
                "train_maps": train_maps, "heldout_maps": heldout_maps,
            })
    except RuntimeError as error:
        preflight[-1]["status"] = "failed"
        preflight[-1]["error"] = str(error)
        return _seal_preflight_failure(output, contract_sha=contract_sha, details={
            "schema_version": RUN_SCHEMA, "diagnostic_only": True,
            "truth_omega_feature_construction": True, "formal_v15": False,
            "train_only": True, "git": git_start,
            "experiment_contract": contract,
            "experiment_contract_sha256": contract_sha,
            "a1_r2": {"result": str(a1_path), "sha256": A1_R2_RESULT_SHA256},
            "parent": {"checkpoint": str(parent_checkpoint), "sha256": parent_sha},
            "dataset_manifest_sha256": contract["dataset_manifest_sha256"],
            "truth_manifest_sha256": truth_sha,
            "metadata_preflight": preflight, "preflight_error": str(error),
        })
    _atomic_json(output / "run_state.json", {
        "schema_version": RUN_SCHEMA, "status": "expert_preflight",
        "validation_accessed": False, "test_accessed": False,
        "future_modules_loaded": False,
        "experiment_contract_sha256": contract_sha,
    })
    intact_full = _extract_arm(
        base, prepared, all_indices, all_indices, device,
        batch_size=args.expert_batch_size,
    )
    cv: list[dict[str, Any]] = []
    fold_models: list[dict[str, torch.Tensor]] = []
    try:
        for plan in plans:
            train_arms = {
                "intact": _slice_arm(intact_full, plan["training"]),
                "global": _extract_arm(
                    base, prepared, *plan["train_indices"]["global"], device,
                    batch_size=args.expert_batch_size,
                ),
                "hard": _extract_arm(
                    base, prepared, *plan["train_indices"]["hard"], device,
                    batch_size=args.expert_batch_size,
                ),
            }
            heldout_arms = {
                "intact": _slice_arm(intact_full, plan["heldout"]),
                "global": _extract_arm(
                    base, prepared, *plan["heldout_indices"]["global"], device,
                    batch_size=args.expert_batch_size,
                ),
                "hard": _extract_arm(
                    base, prepared, *plan["heldout_indices"]["hard"], device,
                    batch_size=args.expert_batch_size,
                ),
            }
            for domain, arms in (("fit", train_arms), ("heldout", heldout_arms)):
                for arm_name, arm in arms.items():
                    for group, mask in {
                        "overall": torch.ones_like(arm["crossfit_supported"]),
                        "rotation": arm["motion_class"] == 2,
                        "combined": arm["motion_class"] == 3,
                    }.items():
                        total = int(mask.sum())
                        valid = int((mask & arm["crossfit_supported"]).sum())
                        if total == 0 or valid < math.ceil(0.80 * total):
                            raise RuntimeError(
                                f"A2 {domain}/{arm_name}/{group} crossfit support insufficient"
                            )
            head, logs = _train_head(
                train_arms, device=device,
                seed=args.seed + 100 + int(plan["fold"]), updates=args.updates,
            )
            metrics = _evaluate_head(head, heldout_arms, device)
            gates = _fold_gates(metrics)
            passed = _all_boolean_gates_pass(gates)
            cv.append({
                "fold": int(plan["fold"]), "passed": passed,
                "heldout_sessions": sorted(plan["heldout_sessions"]),
                "train_count": int(plan["training"].size),
                "heldout_count": int(plan["heldout"].size),
                "train_maps": plan["train_maps"],
                "heldout_maps": plan["heldout_maps"],
                "logs": logs, "metrics": metrics, "gates": gates,
            })
            fold_models.append({
                name: value.detach().cpu() for name, value in head.state_dict().items()
            })
    except RuntimeError as error:
        return _seal_preflight_failure(output, contract_sha=contract_sha, details={
            "schema_version": RUN_SCHEMA, "diagnostic_only": True,
            "truth_omega_feature_construction": True, "formal_v15": False,
            "train_only": True, "git": git_start,
            "experiment_contract": contract,
            "experiment_contract_sha256": contract_sha,
            "a1_r2": {"result": str(a1_path), "sha256": A1_R2_RESULT_SHA256},
            "parent": {"checkpoint": str(parent_checkpoint), "sha256": parent_sha},
            "dataset_manifest_sha256": contract["dataset_manifest_sha256"],
            "truth_manifest_sha256": truth_sha,
            "metadata_preflight": preflight,
            "preflight_error": str(error), "partial_cv": cv,
        })
    cross = _cross_fold_gates(cv)
    passed = _cv_passed(cv, cross)
    if state_dict_sha256(base.state_dict()) != base_sha:
        raise RuntimeError("A2 changed frozen V14 parent")
    if {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    } != frozen_states:
        raise RuntimeError("A2 changed frozen Mapper/S/H")
    if {
        "mapper": sha256_file(mapper_path), "s": sha256_file(s_path),
        "h": sha256_file(h_path),
    } != frozen_files:
        raise RuntimeError("A2 frozen checkpoint files changed")
    if (
        sha256_file(parent_checkpoint) != parent_sha
        or sha256_file(parent_manifest_path) != parent_manifest_sha
        or sha256_file(parent_result_path) != parent_result_sha
    ):
        raise RuntimeError("A2 parent artifacts changed")
    if sha256_file(dataset_manifest_path) != contract["dataset_manifest_sha256"]:
        raise RuntimeError("A2 paired manifest changed")
    assert_manifest_split_shards_unchanged(
        dataset_path, dataset_manifest, split="train", label="A2 paired dataset",
    )
    train_truth.assert_unchanged()
    if {name: sha256_file(path) for name, path in source_paths.items()} != source_sha:
        raise RuntimeError("A2 implementation source changed")
    if _git_state() != git_start:
        raise RuntimeError("A2 git state changed during run")
    checkpoint_path = output / "fold-heads.pt"
    checkpoint = {
        "schema_version": RUN_SCHEMA, "diagnostic_only": True,
        "truth_omega_feature_construction": True,
        "experiment_contract": contract,
        "experiment_contract_sha256": contract_sha,
        "a1_r2_result_sha256": A1_R2_RESULT_SHA256,
        "fold_models": fold_models,
        "fold_state_dict_sha256": [
            state_dict_sha256(state) for state in fold_models
        ],
    }
    _atomic_checkpoint(checkpoint_path, checkpoint)
    result = {
        "schema_version": RUN_SCHEMA, "status": "passed" if passed else "failed",
        "diagnostic_only": True, "truth_omega_feature_construction": True,
        "formal_v15": False, "train_only": True,
        "validation_accessed": False, "test_accessed": False,
        "future_modules_loaded": False,
        "authorized_a2_train_counterfactual": passed,
        "authorized_formal_two_stage": False,
        "git": git_start, "experiment_contract": contract,
        "experiment_contract_sha256": contract_sha,
        "a1_r2": {"result": str(a1_path), "sha256": A1_R2_RESULT_SHA256},
        "parent": {
            "checkpoint": str(parent_checkpoint), "sha256": parent_sha,
            "model_state_dict_sha256": base_sha,
        },
        "dataset_manifest_sha256": contract["dataset_manifest_sha256"],
        "truth_manifest_sha256": truth_sha,
        "metadata_preflight": preflight,
        "cv": cv, "cross_fold_gates": cross,
        "fold_checkpoint": str(checkpoint_path),
        "fold_checkpoint_sha256": sha256_file(checkpoint_path),
        "fold_state_dict_sha256": checkpoint["fold_state_dict_sha256"],
    }
    _atomic_json(output / "screen_result.json", result)
    _atomic_json(output / "run_state.json", {
        "schema_version": RUN_SCHEMA, "status": result["status"],
        "validation_accessed": False, "test_accessed": False,
        "future_modules_loaded": False,
        "authorized_a2_train_counterfactual": passed,
        "experiment_contract_sha256": contract_sha,
        "screen_result_sha256": sha256_file(output / "screen_result.json"),
        "fold_checkpoint_sha256": result["fold_checkpoint_sha256"],
    })
    validate_a2_artifacts(output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-r2-result", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--updates", type=int, default=FIXED_UPDATES)
    parser.add_argument("--width", type=int, default=FIXED_WIDTH)
    parser.add_argument("--batch-size", type=int, default=FIXED_BATCH_SIZE)
    parser.add_argument(
        "--expert-batch-size", type=int, default=FIXED_EXPERT_BATCH_SIZE,
    )
    args = parser.parse_args()
    if (
        args.seed != 20260730 or args.updates != FIXED_UPDATES
        or args.width != FIXED_WIDTH or args.batch_size != FIXED_BATCH_SIZE
        or args.expert_batch_size != FIXED_EXPERT_BATCH_SIZE
    ):
        raise ValueError("V15-A2 budget/architecture is fixed")
    return args


def main() -> None:
    result = run(parse_args())
    print(json.dumps({
        "status": result["status"],
        "authorized_a2_train_counterfactual": result[
            "authorized_a2_train_counterfactual"
        ],
        "cross_fold_gates": result.get("cross_fold_gates"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
