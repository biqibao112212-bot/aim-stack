"""Evaluate frozen clean observable-F checkpoints on paired real PnP inputs.

Conditional errors and their paired clean->PnP deltas are the primary result.
Hard routing and route flips are diagnostics only.  Test shards are forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .observable_future_dataset import EXPERT_TO_MOTION_CLASS
from .observable_future_model import AnonymousCandidateFutureExpert
from .observable_future_pnp_upper_bound import SCHEMA_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _instantiate_checkpoint(
    checkpoint_path: Path,
    expert: str,
    device: torch.device,
    expected_sha256: str,
) -> tuple[AnonymousCandidateFutureExpert, dict[str, Any], str, str]:
    checkpoint_sha256 = _sha256(checkpoint_path)
    if expected_sha256 and checkpoint_sha256 != expected_sha256:
        raise ValueError("frozen F checkpoint SHA-256 differs from the declared value")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config")
    if not isinstance(config, dict) or config.get("expert") != expert:
        raise ValueError("checkpoint expert/configuration mismatch")
    model = AnonymousCandidateFutureExpert(
        expert,
        channels=int(config["channels"]),
        dropout=float(config["dropout"]),
        position_scale_m=float(config["position_scale_m"]),
        history_scale_s=float(config["history_scale_s"]),
        trained_horizon_s=float(config["trained_horizon_s"]),
        maximum_absolute_step=int(config["maximum_absolute_step"]),
        trajectory_rank=int(config["trajectory_rank"]),
    )
    if model.config != config:
        raise ValueError("runtime model config differs from frozen checkpoint")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen F must remain eval-only with gradients disabled")
    return model, checkpoint, checkpoint_sha256, _module_state_sha256(model)


def _tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).to(device)


def _forward(
    model: AnonymousCandidateFutureExpert,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    *,
    prefix: str,
) -> dict[str, torch.Tensor]:
    names = (
        "history_position_rel_m", "history_time_s", "history_dt_s",
        "history_switch_step", "history_mask", "current_position_m",
        "candidate_relation_m", "candidate_step", "candidate_mask",
        "candidate_confidence", "tau_s",
    )
    inputs = [_tensor(arrays[f"{prefix}{name}"][indices], device) for name in names]
    with torch.no_grad():
        return model(*inputs)


def _stats(values: list[np.ndarray] | np.ndarray) -> dict[str, float | int]:
    if isinstance(values, list):
        array = np.concatenate(values).astype(np.float64, copy=False) if values else np.empty(0)
    else:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"count": 0}
    if not np.isfinite(array).all():
        raise ValueError("metric vector contains non-finite values")
    return {
        "count": int(array.size),
        "mean_m": float(array.mean()),
        "p50_m": float(np.quantile(array, 0.50)),
        "p95_m": float(np.quantile(array, 0.95)),
        "p99_m": float(np.quantile(array, 0.99)),
        "max_m": float(array.max()),
    }


def _bootstrap_mean_ci(
    values: np.ndarray, *, seed: int, replicates: int
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"sample_count": 0, "replicates": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 128):
        count = min(128, replicates - start)
        indices = rng.integers(0, array.size, size=(count, array.size))
        means[start:start + count] = array[indices].mean(axis=1)
    return {
        "sample_count": int(array.size),
        "replicates": int(replicates),
        "mean_m": float(array.mean()),
        "ci95_low_m": float(np.quantile(means, 0.025)),
        "ci95_high_m": float(np.quantile(means, 0.975)),
    }


def _append_full_clean_metrics(
    collectors: dict[str, list[np.ndarray]],
    prediction: dict[str, torch.Tensor],
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
) -> None:
    candidate_step = _tensor(arrays["candidate_step"][indices], device)
    target_step = _tensor(arrays["target_switch_count"][indices], device)
    target_mask = _tensor(arrays["target_query_mask"][indices], device).bool()
    tau = _tensor(arrays["tau_s"][indices], device)
    target_delta = _tensor(arrays["target_visible_delta_m"][indices], device)
    matches = candidate_step[:, None, :] == target_step[:, :, None]
    true_row = matches.to(torch.long).argmax(dim=-1)
    eligible = target_mask & (tau > 0)
    gather_true = true_row[:, :, None, None].expand(-1, -1, 1, 3)
    conditional = prediction["conditional_delta_m"].gather(2, gather_true).squeeze(2)
    selected = prediction["selected_candidate_row"]
    gather_hard = selected[:, :, None, None].expand(-1, -1, 1, 3)
    hard = prediction["conditional_delta_m"].gather(2, gather_hard).squeeze(2)
    conditional_error = torch.linalg.vector_norm(conditional - target_delta, dim=-1)
    hard_error = torch.linalg.vector_norm(hard - target_delta, dim=-1)
    selected_step = candidate_step.gather(1, selected)
    collectors["conditional"].append(conditional_error[eligible].cpu().numpy())
    collectors["hard"].append(hard_error[eligible].cpu().numpy())
    collectors["route_correct"].append(
        (selected_step[eligible] == target_step[eligible]).cpu().numpy()
    )


def evaluate(args: argparse.Namespace) -> Path:
    dataset_dir = Path(args.dataset).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite PnP evaluation: {output_dir}")
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("paired PnP dataset schema mismatch")
    if not bool(manifest.get("qualification_passed", False)):
        raise ValueError("formal evaluation requires the complete qualified paired dataset")
    if bool(manifest.get("test_accessed", True)):
        raise ValueError("paired PnP dataset accessed test")
    if manifest.get("deployable_pipeline") is not False:
        raise ValueError("this evaluator only accepts the declared non-deployable upper bound")
    if args.expert not in EXPERT_TO_MOTION_CLASS:
        raise ValueError("expert must be translation, rotation, or combined")
    motion_class = EXPERT_TO_MOTION_CLASS[args.expert]
    device = torch.device(args.device)
    model, checkpoint, checkpoint_sha, state_before = _instantiate_checkpoint(
        checkpoint_path, args.expert, device, args.expected_checkpoint_sha256
    )
    provenance = checkpoint.get("provenance", {})
    if bool(provenance.get("test_accessed", True)):
        raise ValueError("frozen F checkpoint accessed test")
    if str(provenance.get("dataset_manifest_sha256")) != str(
        manifest["clean_dataset_manifest_sha256"]
    ):
        raise ValueError("frozen F checkpoint is not bound to the paired clean source")

    full_clean = {"conditional": [], "hard": [], "route_correct": []}
    paired: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "clean_conditional", "pnp_conditional", "conditional_delta",
            "conditional_response_drift", "clean_hard", "pnp_hard", "hard_delta",
            "pnp_fixed_clean_route", "fixed_route_delta", "sample_mean_conditional_delta",
        )
    }
    route_buckets = {
        "clean_correct_pnp_correct": 0,
        "clean_correct_pnp_wrong": 0,
        "clean_wrong_pnp_correct": 0,
        "clean_wrong_pnp_wrong": 0,
    }
    coverage = {
        "expert_clean_sample_count": 0,
        "expert_pnp_forward_usable_count": 0,
        "expert_q0_associated_count": 0,
        "expert_full_history_associated_count": 0,
        "expert_history_event_count": 0,
        "expert_history_event_associated_count": 0,
        "expert_history_event_ambiguous_count": 0,
    }
    q0_anchor_error_all: list[np.ndarray] = []
    q0_anchor_error_usable: list[np.ndarray] = []

    validation_shards = [
        item for item in manifest["shards"] if item["split"] == "validation"
    ]
    if not validation_shards:
        raise ValueError("paired PnP dataset has no validation shards")
    for item in validation_shards:
        shard_path = dataset_dir / Path(str(item["path"]).replace("\\", "/"))
        if _sha256(shard_path) != str(item["sha256"]):
            raise ValueError(f"paired PnP shard hash mismatch: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        expert_indices = np.flatnonzero(arrays["motion_class"] == motion_class)
        if expert_indices.size == 0:
            continue
        coverage["expert_clean_sample_count"] += int(expert_indices.size)
        coverage["expert_q0_associated_count"] += int(
            arrays["pnp_q0_associated"][expert_indices].sum()
        )
        coverage["expert_full_history_associated_count"] += int(
            arrays["pnp_full_history_associated"][expert_indices].sum()
        )
        association_mask = arrays["pnp_history_associated_mask"][expert_indices]
        ambiguous_mask = arrays["pnp_history_ambiguous_mask"][expert_indices]
        coverage["expert_history_event_count"] += int(association_mask.size)
        coverage["expert_history_event_associated_count"] += int(
            association_mask.sum()
        )
        coverage["expert_history_event_ambiguous_count"] += int(
            ambiguous_mask.sum()
        )
        q0_mask = arrays["pnp_q0_associated"][expert_indices].astype(np.bool_)
        if q0_mask.any():
            q0_anchor_error_all.append(
                arrays["pnp_q0_anchor_error_m"][expert_indices[q0_mask]]
            )

        for start in range(0, expert_indices.size, args.batch_size):
            indices = expert_indices[start:start + args.batch_size]
            clean_prediction = _forward(model, arrays, indices, device, prefix="")
            _append_full_clean_metrics(
                full_clean, clean_prediction, arrays, indices, device
            )

        usable_indices = expert_indices[
            arrays["pnp_forward_usable"][expert_indices].astype(np.bool_)
        ]
        coverage["expert_pnp_forward_usable_count"] += int(usable_indices.size)
        if usable_indices.size:
            q0_anchor_error_usable.append(
                arrays["pnp_q0_anchor_error_m"][usable_indices]
            )
        for start in range(0, usable_indices.size, args.batch_size):
            indices = usable_indices[start:start + args.batch_size]
            clean_prediction = _forward(model, arrays, indices, device, prefix="")
            pnp_prediction = _forward(model, arrays, indices, device, prefix="pnp_")
            candidate_step = _tensor(arrays["candidate_step"][indices], device)
            target_step = _tensor(arrays["target_switch_count"][indices], device)
            target_mask = _tensor(arrays["target_query_mask"][indices], device).bool()
            tau = _tensor(arrays["tau_s"][indices], device)
            clean_current = _tensor(arrays["current_position_m"][indices], device)
            pnp_current = _tensor(arrays["pnp_current_position_m"][indices], device)
            target_delta = _tensor(arrays["target_visible_delta_m"][indices], device)
            target_absolute = clean_current[:, None, :] + target_delta
            expected_pnp_delta = target_absolute - pnp_current[:, None, :]
            stored_pnp_delta = _tensor(
                arrays["pnp_target_visible_delta_m"][indices], device
            )
            if not torch.equal(expected_pnp_delta, stored_pnp_delta):
                raise ValueError("stored PnP target delta does not preserve paired truth")
            matches = candidate_step[:, None, :] == target_step[:, :, None]
            if bool(torch.any(target_mask & (matches.sum(dim=-1) != 1))):
                raise ValueError("paired target candidate is missing or duplicated")
            true_row = matches.to(torch.long).argmax(dim=-1)
            eligible = target_mask & (tau > 0)
            gather_true = true_row[:, :, None, None].expand(-1, -1, 1, 3)
            clean_cond_abs = (
                clean_current[:, None, :]
                + clean_prediction["conditional_delta_m"].gather(
                    2, gather_true
                ).squeeze(2)
            )
            pnp_cond_abs = (
                pnp_current[:, None, :]
                + pnp_prediction["conditional_delta_m"].gather(
                    2, gather_true
                ).squeeze(2)
            )
            clean_selected = clean_prediction["selected_candidate_row"]
            pnp_selected = pnp_prediction["selected_candidate_row"]
            gather_clean_hard = clean_selected[:, :, None, None].expand(-1, -1, 1, 3)
            gather_pnp_hard = pnp_selected[:, :, None, None].expand(-1, -1, 1, 3)
            clean_hard_abs = (
                clean_current[:, None, :]
                + clean_prediction["conditional_delta_m"].gather(
                    2, gather_clean_hard
                ).squeeze(2)
            )
            pnp_hard_abs = (
                pnp_current[:, None, :]
                + pnp_prediction["conditional_delta_m"].gather(
                    2, gather_pnp_hard
                ).squeeze(2)
            )
            pnp_fixed_abs = (
                pnp_current[:, None, :]
                + pnp_prediction["conditional_delta_m"].gather(
                    2, gather_clean_hard
                ).squeeze(2)
            )
            clean_cond_error = torch.linalg.vector_norm(
                clean_cond_abs - target_absolute, dim=-1
            )
            pnp_cond_error = torch.linalg.vector_norm(
                pnp_cond_abs - target_absolute, dim=-1
            )
            clean_hard_error = torch.linalg.vector_norm(
                clean_hard_abs - target_absolute, dim=-1
            )
            pnp_hard_error = torch.linalg.vector_norm(
                pnp_hard_abs - target_absolute, dim=-1
            )
            pnp_fixed_error = torch.linalg.vector_norm(
                pnp_fixed_abs - target_absolute, dim=-1
            )
            response_drift = torch.linalg.vector_norm(
                pnp_cond_abs - clean_cond_abs, dim=-1
            )
            for name, value in (
                ("clean_conditional", clean_cond_error),
                ("pnp_conditional", pnp_cond_error),
                ("conditional_delta", pnp_cond_error - clean_cond_error),
                ("conditional_response_drift", response_drift),
                ("clean_hard", clean_hard_error),
                ("pnp_hard", pnp_hard_error),
                ("hard_delta", pnp_hard_error - clean_hard_error),
                ("pnp_fixed_clean_route", pnp_fixed_error),
                ("fixed_route_delta", pnp_fixed_error - clean_hard_error),
            ):
                paired[name].append(value[eligible].cpu().numpy())
            sample_sum = torch.where(
                eligible, pnp_cond_error - clean_cond_error,
                torch.zeros_like(pnp_cond_error),
            ).sum(dim=1)
            sample_count = eligible.sum(dim=1)
            valid_sample = sample_count > 0
            paired["sample_mean_conditional_delta"].append(
                (sample_sum[valid_sample] / sample_count[valid_sample]).cpu().numpy()
            )

            clean_selected_step = candidate_step.gather(1, clean_selected)
            pnp_selected_step = candidate_step.gather(1, pnp_selected)
            clean_correct = clean_selected_step == target_step
            pnp_correct = pnp_selected_step == target_step
            for clean_value, pnp_value, name in (
                (True, True, "clean_correct_pnp_correct"),
                (True, False, "clean_correct_pnp_wrong"),
                (False, True, "clean_wrong_pnp_correct"),
                (False, False, "clean_wrong_pnp_wrong"),
            ):
                route_buckets[name] += int((
                    eligible & (clean_correct == clean_value)
                    & (pnp_correct == pnp_value)
                ).sum())

            zero = tau == 0
            if not torch.equal(
                pnp_prediction["delta_m"][zero],
                torch.zeros_like(pnp_prediction["delta_m"][zero]),
            ):
                raise RuntimeError("PnP replay violated structural tau-zero identity")

    if coverage["expert_clean_sample_count"] == 0:
        raise ValueError("validation contains no samples for the requested expert")
    if coverage["expert_pnp_forward_usable_count"] == 0:
        raise ValueError("validation contains no strict PnP upper-bound samples")
    state_after = _module_state_sha256(model)
    if state_after != state_before:
        raise RuntimeError("frozen F state changed during PnP evaluation")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("frozen F accumulated gradients during evaluation")

    full_route = np.concatenate(full_clean["route_correct"]).astype(np.float64)
    sample_delta = np.concatenate(paired["sample_mean_conditional_delta"])
    coverage["expert_pnp_forward_usable_fraction"] = (
        coverage["expert_pnp_forward_usable_count"]
        / coverage["expert_clean_sample_count"]
    )
    coverage["expert_q0_associated_fraction"] = (
        coverage["expert_q0_associated_count"]
        / coverage["expert_clean_sample_count"]
    )
    coverage["expert_full_history_associated_fraction"] = (
        coverage["expert_full_history_associated_count"]
        / coverage["expert_clean_sample_count"]
    )
    coverage["expert_history_event_associated_fraction"] = (
        coverage["expert_history_event_associated_count"]
        / coverage["expert_history_event_count"]
    )
    report = {
        "schema_version": "stage3-observable-future-pnp-upper-bound-evaluation-v1",
        "status": "completed_diagnostic",
        "expert": args.expert,
        "primary_metric": "paired conditional absolute error at tau>0",
        "hard_routing_role": "diagnostic_only",
        "deployable_pipeline": False,
        "test_accessed": False,
        "dataset": str(dataset_dir),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "clean_dataset_manifest_sha256": manifest["clean_dataset_manifest_sha256"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_update": int(checkpoint.get("update", -1)),
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "model_state_unchanged": state_before == state_after,
        "coverage": coverage,
        "full_clean_validation_tau_positive": {
            "conditional": _stats(full_clean["conditional"]),
            "hard": _stats(full_clean["hard"]),
            "switch_accuracy": float(full_route.mean()),
        },
        "paired_strict_validation_tau_positive": {
            "clean_conditional": _stats(paired["clean_conditional"]),
            "pnp_conditional": _stats(paired["pnp_conditional"]),
            "paired_conditional_delta": _stats(paired["conditional_delta"]),
            "conditional_response_drift": _stats(
                paired["conditional_response_drift"]
            ),
            "paired_sample_mean_delta_bootstrap_ci95": _bootstrap_mean_ci(
                sample_delta, seed=args.bootstrap_seed,
                replicates=args.bootstrap_replicates,
            ),
            "clean_hard_diagnostic": _stats(paired["clean_hard"]),
            "pnp_hard_diagnostic": _stats(paired["pnp_hard"]),
            "paired_hard_delta_diagnostic": _stats(paired["hard_delta"]),
            "pnp_fixed_clean_route_diagnostic": _stats(
                paired["pnp_fixed_clean_route"]
            ),
            "paired_fixed_route_delta_diagnostic": _stats(
                paired["fixed_route_delta"]
            ),
            "route_correctness_2x2": route_buckets,
        },
        "q0_anchor_error": {
            "all_q0_associated": _stats(q0_anchor_error_all),
            "strict_usable": _stats(q0_anchor_error_usable),
        },
        "interpretation_boundary": (
            "real PnP xyz with oracle past association, oracle history switch, "
            "and truth-S candidates; this is an optimistic noise upper bound, "
            "not an end-to-end PnP/S deployment result"
        ),
    }
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "evaluation.json", report)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expert", required=True, choices=sorted(EXPERT_TO_MOTION_CLASS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_replicates < 100:
        parser.error("PnP evaluation arguments are invalid")
    print(evaluate(args))


if __name__ == "__main__":
    main()

