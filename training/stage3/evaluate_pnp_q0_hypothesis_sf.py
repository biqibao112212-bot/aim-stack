"""Evaluate A3 mapper -> frozen S -> H -> frozen F on combined/common data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cyclic_future_foundation import load_frozen_v19
from .observable_future_pnp_ab import (
    ObservableFuturePnPSFDataset,
    canonicalize_direction_keep_c4,
    f_forward,
    load_observable_f_checkpoint,
    state_dict_sha256,
)
from .pnp_observation_mapper import compose_corrected_observations
from .pnp_q0_hypothesis_adapter import (
    compose_hypothesis_for_f,
    hypothesis_forward,
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .train_pnp_q0_hypothesis_adapter import _s_forward


EVALUATION_SCHEMA = "stage3-pnp-q0-hypothesis-sf-evaluation-v1"
def _stats(parts: list[np.ndarray]) -> dict[str, float | int | None]:
    if not parts:
        return {
            "count": 0, "mean_m": None, "p50_m": None,
            "p95_m": None, "p99_m": None, "max_m": None,
        }
    merged = np.concatenate(parts).astype(np.float64, copy=False)
    if not merged.size or not np.isfinite(merged).all():
        raise ValueError("A3 evaluation metric is empty or non-finite")
    return {
        "count": int(merged.size), "mean_m": float(merged.mean()),
        "p50_m": float(np.quantile(merged, 0.50)),
        "p95_m": float(np.quantile(merged, 0.95)),
        "p99_m": float(np.quantile(merged, 0.99)),
        "max_m": float(merged.max()),
    }


def _collector() -> dict[str, Any]:
    return {
        "conditional": [], "hard": [], "correct": [], "current": [],
        "roles": {
            name: {"conditional": [], "hard": [], "correct": []}
            for name in ("observed", "warm_adjacent", "warm_opposite", "true_cold")
        },
        "steps": {}, "horizons": {
            "short_le_0.2": {"conditional": [], "hard": [], "correct": []},
            "medium_0.2_0.4": {"conditional": [], "hard": [], "correct": []},
            "long_gt_0.4": {"conditional": [], "hard": [], "correct": []},
        },
    }


def _append_slice(
    target: dict[str, list[np.ndarray]],
    mask: torch.Tensor,
    conditional: torch.Tensor,
    hard: torch.Tensor,
    correct: torch.Tensor,
) -> None:
    if bool(mask.any()):
        target["conditional"].append(conditional[mask].cpu().numpy())
        target["hard"].append(hard[mask].cpu().numpy())
        target["correct"].append(correct[mask].cpu().numpy())


def _add_prediction(
    collector: dict[str, Any],
    prediction: dict[str, torch.Tensor],
    current: torch.Tensor,
    batch: dict[str, torch.Tensor],
    candidate_support_class: torch.Tensor,
) -> None:
    candidate_step = batch["candidate_step"].to(torch.long)
    target_step = batch["target_switch_count"].to(torch.long)
    tau = batch["tau_s"]
    mask = batch["target_query_mask"].to(torch.bool) & (tau > 0)
    matches = candidate_step[:, None, :] == target_step[:, :, None]
    if bool(torch.any(mask & (matches.sum(dim=-1) != 1))):
        raise ValueError("A3 target step does not have one candidate")
    true_row = matches.to(torch.long).argmax(dim=-1)
    true_gather = true_row[:, :, None, None].expand(-1, -1, 1, 3)
    hard_row = prediction["selected_candidate_row"]
    hard_gather = hard_row[:, :, None, None].expand(-1, -1, 1, 3)
    truth = batch["current_position_m"][:, None, :] + batch["target_visible_delta_m"]
    conditional = torch.linalg.vector_norm(
        prediction["conditional_position_m"].gather(2, true_gather).squeeze(2) - truth,
        dim=-1,
    )
    hard = torch.linalg.vector_norm(
        prediction["conditional_position_m"].gather(2, hard_gather).squeeze(2) - truth,
        dim=-1,
    )
    selected_step = candidate_step.gather(1, hard_row)
    correct = selected_step == target_step
    collector["conditional"].append(conditional[mask].cpu().numpy())
    collector["hard"].append(hard[mask].cpu().numpy())
    collector["correct"].append(correct[mask].cpu().numpy())
    collector["current"].append(torch.linalg.vector_norm(
        current - batch["current_position_m"], dim=-1
    ).cpu().numpy())

    query_role = candidate_support_class.gather(1, true_row)
    for code, name in enumerate(("observed", "warm_adjacent", "warm_opposite", "true_cold")):
        _append_slice(
            collector["roles"][name], mask & (query_role == code),
            conditional, hard, correct,
        )
    for step in target_step[mask].unique().tolist():
        key = str(int(step))
        if key not in collector["steps"]:
            collector["steps"][key] = {"conditional": [], "hard": [], "correct": []}
        _append_slice(
            collector["steps"][key], mask & (target_step == int(step)),
            conditional, hard, correct,
        )
    horizon_masks = {
        "short_le_0.2": tau <= 0.2,
        "medium_0.2_0.4": (tau > 0.2) & (tau <= 0.4),
        "long_gt_0.4": tau > 0.4,
    }
    for name, horizon_mask in horizon_masks.items():
        _append_slice(
            collector["horizons"][name], mask & horizon_mask,
            conditional, hard, correct,
        )


def _finish_slice(value: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    correct = np.concatenate(value["correct"]) if value["correct"] else np.empty(0)
    return {
        "conditional_position": _stats(value["conditional"]),
        "hard_position": _stats(value["hard"]),
        "switch_accuracy": float(correct.mean()) if correct.size else None,
    }


def _finish(collector: dict[str, Any]) -> dict[str, Any]:
    correct = np.concatenate(collector["correct"])
    return {
        "conditional_position": _stats(collector["conditional"]),
        "hard_position": _stats(collector["hard"]),
        "switch_accuracy": float(correct.mean()),
        "current_position": _stats(collector["current"]),
        "by_true_role": {
            name: _finish_slice(value) for name, value in collector["roles"].items()
        },
        "by_signed_step": {
            name: _finish_slice(value)
            for name, value in sorted(collector["steps"].items(), key=lambda item: int(item[0]))
        },
        "by_horizon": {
            name: _finish_slice(value) for name, value in collector["horizons"].items()
        },
    }


def _q0_collector() -> dict[str, list[np.ndarray] | int]:
    return {
        "observed": [], "warm_adjacent": [], "warm_opposite": [],
        "supported": [], "true_cold_count": 0,
    }


def _add_q0(
    collector: dict[str, list[np.ndarray] | int],
    h_output: dict[str, torch.Tensor],
    truth_q0: torch.Tensor,
) -> None:
    error = torch.linalg.vector_norm(h_output["q0_m"] - truth_q0, dim=-1)
    supported = h_output["evidence_supported"].to(torch.bool)
    assert isinstance(collector["supported"], list)
    collector["supported"].append(error[supported].cpu().numpy())
    for name in ("observed", "warm_adjacent", "warm_opposite"):
        mask = h_output[name].to(torch.bool)
        assert isinstance(collector[name], list)
        if bool(mask.any()):
            collector[name].append(error[mask].cpu().numpy())
    collector["true_cold_count"] = int(collector["true_cold_count"]) + int(
        h_output["true_cold"].sum()
    )


def _finish_q0(collector: dict[str, list[np.ndarray] | int]) -> dict[str, Any]:
    return {
        name: _stats(value) for name, value in collector.items()
        if name != "true_cold_count" and isinstance(value, list)
    } | {
        "true_cold": {
            "count": int(collector["true_cold_count"]),
            "coordinate_metric_count": 0,
        }
    }


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Path:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite A3 evaluation: {output}")
    device = torch.device(args.device)
    dataset = ObservableFuturePnPSFDataset(args.dataset, "validation", motion_class=3)
    mapper, mapper_provenance = load_frozen_pnp_mapper(args.mapper_checkpoint)
    s_model, s_provenance = load_frozen_v19(args.s_checkpoint)
    h_model, h_provenance = load_frozen_hypothesis_adapter(
        args.h_checkpoint, allow_diagnostic=args.allow_diagnostic_h
    )
    f_model, f_provenance = load_observable_f_checkpoint(args.f_checkpoint)
    if mapper_provenance["provenance"]["dataset_manifest_sha256"] != dataset.manifest_sha256:
        raise ValueError("A3 mapper and evaluation datasets differ")
    if (
        mapper_provenance["provenance"]["frozen_s"]["state_dict_sha256"]
        != s_provenance["state_dict_sha256"]
    ):
        raise ValueError("A3 mapper and supplied frozen S checkpoints differ")
    h_frozen = h_provenance["provenance"]
    if h_frozen["dataset_manifest_sha256"] != dataset.manifest_sha256:
        raise ValueError("A3 H and evaluation datasets differ")
    expected = {
        "frozen_mapper": mapper_provenance["state_dict_sha256"],
        "frozen_s": s_provenance["state_dict_sha256"],
        "frozen_f": f_provenance["state_dict_sha256"],
    }
    compatibility_mismatches: dict[str, dict[str, str]] = {}
    for name, state_sha in expected.items():
        if h_frozen[name]["state_dict_sha256"] != state_sha:
            compatibility_mismatches[name] = {
                "h_expected": h_frozen[name]["state_dict_sha256"],
                "loaded": state_sha,
            }
    if compatibility_mismatches:
        allowed = (
            set(compatibility_mismatches) == {"frozen_mapper"}
            and args.allow_mapper_h_mismatch
            and bool(h_provenance["provenance"]["diagnostic_only"])
        )
        if not allowed:
            raise ValueError(
                f"A3 H provenance mismatch: {sorted(compatibility_mismatches)}"
            )
    for model in (mapper, s_model, h_model, f_model):
        model.eval().requires_grad_(False).to(device)
    before_hash = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
        "f": state_dict_sha256(f_model.state_dict()),
    }

    pipeline_names = (
        "mapper_s0_f", "mapper_s0_h_f", "clean_s0_f", "clean_s0_h_f",
        "cf_mapped_history_only", "cf_mapped_current_only",
        "cf_mapped_candidate_only", "cf_mapped_history_current",
        "cf_mapped_geometry",
        "oracle_truth_s_f",
    )
    collectors = {name: _collector() for name in pipeline_names}
    mapped_q0 = _q0_collector()
    clean_q0 = _q0_collector()
    observed_identity_max = 0.0
    for start in range(0, len(dataset), args.batch_size):
        end = min(start + args.batch_size, len(dataset))
        batch = {
            name: value[start:end].to(device) for name, value in dataset.tensors.items()
        }
        canonicalize_direction_keep_c4(batch, dataset.pair_ids[start:end])
        if not torch.equal(batch["candidate_step"], batch["pnp_candidate_step"]):
            raise ValueError(
                "A3 evaluation requires clean/PnP candidate-step identity"
            )
        mapped = mapper(
            batch["pnp_s_obs_m"], batch["pnp_s_obs_mask"],
            batch["pnp_s_event_time_s"], batch["pnp_s_event_mask"],
        )
        mapped_s = _s_forward(
            s_model, mapped["corrected_obs_m"], batch["pnp_s_obs_mask"],
            batch["pnp_s_primary_mask"], batch["pnp_s_event_mask"],
            batch["pnp_s_event_time_s"], batch["pnp_s_switch_step"],
        )
        mapped_h = hypothesis_forward(h_model, mapped_s)
        observed_identity_max = max(observed_identity_max, float((
            mapped_h["q0_m"][mapped_h["observed"]]
            - mapped_s["q0_m"][mapped_h["observed"]]
        ).abs().max()))
        mapped_s_compose = compose_corrected_observations(
            mapped_s, mapped["corrected_obs_m"], batch["pnp_s_obs_mask"],
            batch["pnp_s_primary_mask"], batch["pnp_candidate_step"],
        )
        mapped_h_compose = compose_hypothesis_for_f(
            mapped_h, mapped_s["primary_index"], mapped["corrected_obs_m"],
            batch["pnp_s_obs_mask"], batch["pnp_s_primary_mask"],
            batch["pnp_candidate_step"],
        )
        for name, composed in (
            ("mapper_s0_f", mapped_s_compose),
            ("mapper_s0_h_f", mapped_h_compose),
        ):
            prediction = f_forward(
                f_model, batch, prefix="pnp_",
                history_position_rel_m=composed["history_position_rel_m"],
                current_position_m=composed["current_position_m"],
                candidate_relation_m=composed["candidate_relation_m"],
                candidate_confidence=composed["candidate_confidence"],
                detach_observation_inputs=False,
            )
            _add_prediction(
                collectors[name], prediction, composed["current_position_m"], batch,
                mapped_h_compose["candidate_support_class"],
            )
        _add_q0(mapped_q0, mapped_h, batch["pnp_s_truth_q0_m"])

        clean_s = _s_forward(
            s_model, batch["clean_s_obs_m"], batch["clean_s_obs_mask"],
            batch["clean_s_primary_mask"], batch["clean_s_event_mask"],
            batch["clean_s_event_time_s"], batch["clean_s_switch_step"],
        )
        clean_h = hypothesis_forward(h_model, clean_s)
        clean_s_compose = compose_corrected_observations(
            clean_s, batch["clean_s_obs_m"], batch["clean_s_obs_mask"],
            batch["clean_s_primary_mask"], batch["candidate_step"],
        )
        clean_h_compose = compose_hypothesis_for_f(
            clean_h, clean_s["primary_index"], batch["clean_s_obs_m"],
            batch["clean_s_obs_mask"], batch["clean_s_primary_mask"],
            batch["candidate_step"],
        )
        for name, composed in (
            ("clean_s0_f", clean_s_compose),
            ("clean_s0_h_f", clean_h_compose),
        ):
            prediction = f_forward(
                f_model, batch,
                history_position_rel_m=composed["history_position_rel_m"],
                current_position_m=composed["current_position_m"],
                candidate_relation_m=composed["candidate_relation_m"],
                candidate_confidence=composed["candidate_confidence"],
                detach_observation_inputs=False,
            )
            _add_prediction(
                collectors[name], prediction, composed["current_position_m"], batch,
                clean_h_compose["candidate_support_class"],
            )
        _add_q0(clean_q0, clean_h, batch["pnp_s_truth_q0_m"])

        counterfactuals = (
            (
                "cf_mapped_history_only", "pnp_",
                mapped_h_compose["history_position_rel_m"],
                clean_h_compose["current_position_m"],
                clean_h_compose["candidate_relation_m"],
                clean_h_compose["candidate_confidence"],
            ),
            (
                "cf_mapped_current_only", "",
                clean_h_compose["history_position_rel_m"],
                mapped_h_compose["current_position_m"],
                clean_h_compose["candidate_relation_m"],
                clean_h_compose["candidate_confidence"],
            ),
            (
                "cf_mapped_candidate_only", "",
                clean_h_compose["history_position_rel_m"],
                clean_h_compose["current_position_m"],
                mapped_h_compose["candidate_relation_m"],
                mapped_h_compose["candidate_confidence"],
            ),
            (
                "cf_mapped_history_current", "pnp_",
                mapped_h_compose["history_position_rel_m"],
                mapped_h_compose["current_position_m"],
                clean_h_compose["candidate_relation_m"],
                clean_h_compose["candidate_confidence"],
            ),
            (
                "cf_mapped_geometry", "",
                clean_h_compose["history_position_rel_m"],
                mapped_h_compose["current_position_m"],
                mapped_h_compose["candidate_relation_m"],
                mapped_h_compose["candidate_confidence"],
            ),
        )
        for (
            name, prefix, history_position, current_position,
            candidate_relation, candidate_confidence,
        ) in counterfactuals:
            prediction = f_forward(
                f_model, batch, prefix=prefix,
                history_position_rel_m=history_position,
                current_position_m=current_position,
                candidate_relation_m=candidate_relation,
                candidate_confidence=candidate_confidence,
                detach_observation_inputs=False,
            )
            _add_prediction(
                collectors[name], prediction, current_position, batch,
                mapped_h_compose["candidate_support_class"],
            )

        oracle = f_forward(f_model, batch)
        _add_prediction(
            collectors["oracle_truth_s_f"], oracle, batch["current_position_m"], batch,
            clean_h_compose["candidate_support_class"],
        )

    unchanged = {
        "mapper": state_dict_sha256(mapper.state_dict()) == before_hash["mapper"],
        "s": state_dict_sha256(s_model.state_dict()) == before_hash["s"],
        "h": state_dict_sha256(h_model.state_dict()) == before_hash["h"],
        "f": state_dict_sha256(f_model.state_dict()) == before_hash["f"],
    }
    if not all(unchanged.values()):
        raise RuntimeError("A3 evaluation changed a frozen state hash")
    result = {
        "schema_version": EVALUATION_SCHEMA,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "validation_sample_count": len(dataset),
        "eligible_query_contract": "target_query_mask and tau_s > 0",
        "direction_contract": "reflection removed; window-local C4 origin retained",
        "mapper": mapper_provenance,
        "frozen_s": s_provenance,
        "hypothesis_adapter": h_provenance,
        "frozen_f": f_provenance,
        "mapped_h_q0": _finish_q0(mapped_q0),
        "clean_h_q0": _finish_q0(clean_q0),
        "observed_identity_max_abs_m": observed_identity_max,
        "pipelines": {name: _finish(value) for name, value in collectors.items()},
        "state_hashes_before": before_hash,
        "state_hashes_unchanged": unchanged,
        "true_cold_coordinate_metric_count": 0,
        "test_accessed": False,
        "oracle_association": True,
        "deployable_pipeline": False,
        "diagnostic_only": bool(
            h_provenance["provenance"]["diagnostic_only"]
        ) or bool(compatibility_mismatches),
        "compatibility_mismatches": compatibility_mismatches,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mapper-checkpoint", required=True)
    parser.add_argument("--s-checkpoint", required=True)
    parser.add_argument("--h-checkpoint", required=True)
    parser.add_argument("--f-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-diagnostic-h", action="store_true")
    parser.add_argument("--allow-mapper-h-mismatch", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    print(evaluate(parser.parse_args()))


if __name__ == "__main__":
    main()
