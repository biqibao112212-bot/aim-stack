"""Train the frozen fixed-slot neural physical A/B experiment.

A infers one shared center/velocity/yaw/yaw-rate state and propagates it with
the frozen rigid equation. B directly infers center/yaw for each query tau and
has no shared velocity/yaw-rate state. Both receive identical fixed-slot clean
histories and use exactly the same decoded-position objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from .causal_physical_dataset import CausalPhysicalShardDataset
from .causal_physical_state_model import (
    ExplicitStatePhysicalPredictor,
    ImplicitQueryPhysicalPredictor,
    trainable_parameter_count,
)
from .physical_loss import causal_physical_state_loss
from .physical_metrics import fixed_slot_physical_batch_errors, summary
from .pnp_state_targets import (
    _query_pose_from_fixed_truth,
    decoded_trajectory_state,
    truth_trajectory_targets,
)


PhysicalModel = Union[ExplicitStatePhysicalPredictor, ImplicitQueryPhysicalPredictor]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_geometry(dataset: Path) -> tuple[torch.Tensor, dict[str, object], str]:
    path = dataset / "geometry_template.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    armor = sorted(payload["armors"], key=lambda item: int(item["relative_slot"]))
    geometry = torch.tensor(
        [item["relative_position_m"] for item in armor], dtype=torch.float32
    )
    if geometry.shape != (4, 3):
        raise ValueError("geometry template must contain four xyz armor positions")
    return geometry, payload, _sha256(path)


def _load_selection(
    path_value: str, manifest_path: Path,
) -> tuple[list[str] | None, list[str] | None, dict[str, object] | None]:
    if not path_value:
        return None, None, None
    path = Path(path_value).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "stage3-causal-physical-state-ab-selection-v1":
        raise ValueError("unsupported causal physical state A/B selection schema")
    if str(payload.get("dataset_manifest_sha256", "")) != _sha256(manifest_path):
        raise ValueError("selection is not bound to this causal physical dataset")
    train = [str(value) for value in payload.get("train", ())]
    validation = [str(value) for value in payload.get("validation", ())]
    test = [str(value) for value in payload.get("test", ())]
    validation_source_split = str(
        payload.get("validation_source_split", "validation")
    )
    if validation_source_split not in {"train", "validation"}:
        raise ValueError("selection validation source must be train or validation")
    if not train or not validation:
        raise ValueError("selection requires non-empty train and validation sessions")
    if test:
        raise ValueError("causal physical state A/B selection must keep test empty")
    if validation_source_split == "validation" and set(train) & set(validation):
        raise ValueError("pilot train and validation sessions must be disjoint")
    if validation_source_split == "train" and train != validation:
        raise ValueError("capacity validation must reuse the exact train sessions")
    return train, validation, {
        "path": str(path), "sha256": _sha256(path),
        "purpose": str(payload.get("purpose", "")),
        "train": train, "validation": validation, "test": [],
        "validation_source_split": validation_source_split,
        "coverage": payload.get("coverage", {}),
    }


def _validate_history_contract(
    manifest: dict[str, object], history_events: int,
) -> None:
    identity = manifest.get("identity_contract", {})
    if not isinstance(identity, dict):
        raise ValueError("causal physical dataset identity contract is missing")
    fit_events = int(identity.get("constant_motion_fit_events", 0))
    minimum_events = int(identity.get("minimum_events_before_prediction", 0))
    if fit_events < history_events or minimum_events < history_events:
        raise ValueError(
            "dataset must certify constant motion across every consumed history "
            f"event: fit={fit_events}, minimum={minimum_events}, "
            f"model_history={history_events}"
        )


def _audit_dataset_contract(
    dataset: CausalPhysicalShardDataset, geometry: torch.Tensor,
    history_events: int, minimum_coverage: float,
    required_motion_classes: set[int] | None = None,
) -> dict[str, object]:
    """Recheck row-level history and supervision eligibility before training."""
    total = 0
    eligible_total = 0
    class_counts: dict[int, list[int]] = {}
    maximum_center_residual_m = 0.0
    maximum_yaw_residual_rad = 0.0
    loader = DataLoader(dataset, batch_size=1024, num_workers=0)
    with torch.no_grad():
        for batch in loader:
            event_mask = batch["event_mask"][:, -history_events:]
            armor_mask = batch["obs_mask"][:, -history_events:]
            if not bool(event_mask.all()) or not bool(armor_mask.all()):
                raise ValueError(
                    "every consumed history event must contain all four fixed slots"
                )
            history_time = batch["event_time_s"][:, -history_events:].float()
            history_dt = history_time[:, 1:] - history_time[:, :-1]
            if not bool((history_dt > 0).all()):
                raise ValueError("consumed history timestamps must be strictly increasing")
            if float(history_dt.max()) * 15.0 >= torch.pi:
                raise ValueError(
                    "history sampling interval can alias the allowed yaw-rate range"
                )
            history_position = batch["history_position_m"][:, -history_events:]
            history_center, history_phase = _query_pose_from_fixed_truth(
                history_position, geometry,
            )
            mean_time = history_time.mean(dim=1, keepdim=True)
            centered_time = history_time - mean_time
            denominator = centered_time.square().sum(dim=1).clamp_min(1e-8)
            mean_center = history_center.mean(dim=1, keepdim=True)
            velocity = (
                centered_time[:, :, None] * (history_center - mean_center)
            ).sum(dim=1) / denominator[:, None]
            fitted_center = mean_center + centered_time[:, :, None] * velocity[:, None]
            center_residual = torch.linalg.vector_norm(
                history_center - fitted_center, dim=-1
            ).amax(dim=1)
            phase_delta = torch.atan2(
                history_phase[:, :-1, 0] * history_phase[:, 1:, 1]
                - history_phase[:, :-1, 1] * history_phase[:, 1:, 0],
                history_phase[:, :-1, 0] * history_phase[:, 1:, 0]
                + history_phase[:, :-1, 1] * history_phase[:, 1:, 1],
            )
            unwrapped_phase = torch.cat((
                torch.zeros_like(phase_delta[:, :1]), phase_delta.cumsum(dim=1),
            ), dim=1)
            mean_phase = unwrapped_phase.mean(dim=1, keepdim=True)
            omega = (
                centered_time * (unwrapped_phase - mean_phase)
            ).sum(dim=1) / denominator
            fitted_phase = mean_phase + centered_time * omega[:, None]
            yaw_residual = (unwrapped_phase - fitted_phase).abs().amax(dim=1)
            maximum_center_residual_m = max(
                maximum_center_residual_m, float(center_residual.max())
            )
            maximum_yaw_residual_rad = max(
                maximum_yaw_residual_rad, float(yaw_residual.max())
            )
            if bool((center_residual > 1e-4).any()) or bool((yaw_residual > 1e-4).any()):
                raise ValueError(
                    "consumed history violates the constant-twist residual contract"
                )

            truth = truth_trajectory_targets(
                batch["future_position"][:, :4], batch["tau"][:, :4],
                geometry, rule_queries=4,
            )
            eligible = batch["rule_query"][:, :4].all(dim=1) & truth["constant_motion"]
            total += int(eligible.numel())
            eligible_total += int(eligible.sum())
            for motion_class in batch["motion_class"].unique():
                index = int(motion_class)
                mask = batch["motion_class"] == motion_class
                values = class_counts.setdefault(index, [0, 0])
                values[0] += int(mask.sum())
                values[1] += int((mask & eligible).sum())
    if total != len(dataset) or total <= 0:
        raise ValueError("dataset audit sample count does not match the manifest")
    coverage = eligible_total / total
    class_coverage = {
        str(index): {
            "sample_count": values[0], "eligible_count": values[1],
            "coverage": values[1] / values[0],
        }
        for index, values in sorted(class_counts.items())
    }
    if (
        required_motion_classes is not None
        and set(class_counts) != required_motion_classes
    ):
        raise ValueError(
            "formal full training requires every registered motion class"
        )
    if coverage < minimum_coverage or any(
        float(values["coverage"]) < minimum_coverage
        for values in class_coverage.values()
    ):
        raise ValueError(
            "trajectory supervision coverage is below the configured minimum"
        )
    return {
        "sample_count": total, "eligible_count": eligible_total,
        "coverage": coverage, "motion_class": class_coverage,
        "minimum_consumed_events": history_events,
        "maximum_center_fit_residual_m": maximum_center_residual_m,
        "maximum_yaw_fit_residual_rad": maximum_yaw_residual_rad,
    }


def _git_state() -> dict[str, object]:
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(repo), "status", "--short"], text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    return {"git_commit": commit, "worktree_dirty": dirty}


def _capture_rng(device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    )


def _restore_rng(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def _to_device(raw: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in raw.items()}


def _train_one(
    label: str, model: PhysicalModel, batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
    device: torch.device, args: argparse.Namespace,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = device.type == "cuda" and args.amp != "off"
    amp_dtype = torch.float16 if args.amp == "float16" else torch.bfloat16
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        future = model(
            batch["obs"], batch["obs_mask"], batch["event_mask"],
            batch["event_time_s"], batch["tau"],
        )
        total, state_metrics = causal_physical_state_loss(
            future, batch["future_position"], batch["tau"], batch["rule_query"],
            model.decoder.geometry, huber_beta_m=args.huber_beta_m,
            reference_horizon_s=args.reference_horizon_s,
        )
    if float(total.detach().cpu()) <= args.minimum_update_loss:
        return {
            "objective": float(total.detach().cpu()),
            **{name: float(value.detach().cpu()) for name, value in state_metrics.items()},
            "gradient_norm": 0.0,
        }
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf")
    )
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError(f"non-finite gradient norm for {label}")
    scaler.step(optimizer)
    scaler.update()
    return {
        "objective": float(total.detach().cpu()),
        **{name: float(value.detach().cpu()) for name, value in state_metrics.items()},
        "gradient_norm": float(gradient_norm.detach().cpu()),
    }


def _train_epoch(
    models: dict[str, PhysicalModel], loader: DataLoader,
    optimizers: dict[str, torch.optim.Optimizer],
    scalers: dict[str, torch.amp.GradScaler], device: torch.device,
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    for model in models.values():
        model.train()
    totals = {label: {} for label in models}
    count = 0
    for raw in loader:
        batch = _to_device(raw, device)
        batch_count = int(batch["obs"].shape[0])
        shared_rng = _capture_rng(device)
        after_a = None
        for index, (label, model) in enumerate(models.items()):
            if index:
                _restore_rng(shared_rng)
            values = _train_one(
                label, model, batch, optimizers[label], scalers[label],
                device, args,
            )
            if index == 0:
                after_a = _capture_rng(device)
            for key, value in values.items():
                totals[label][key] = totals[label].get(key, 0.0) + value * batch_count
        if after_a is not None:
            _restore_rng(after_a)
        count += batch_count
    return {
        label: {key: value / max(count, 1) for key, value in values.items()}
        for label, values in totals.items()
    }


def _validate(
    model: PhysicalModel, loader: DataLoader,
    device: torch.device, args: argparse.Namespace,
) -> dict[str, object]:
    model.eval()
    arrays: dict[str, list[np.ndarray]] = {}
    tau_parts: list[np.ndarray] = []
    rule_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    motion_parts: list[np.ndarray] = []
    state_parts: dict[str, list[np.ndarray]] = {}
    state_motion_parts: list[np.ndarray] = []
    eligibility_parts: list[np.ndarray] = []
    gate_parts: dict[str, list[np.ndarray]] = {}
    factor_parts: list[np.ndarray] = []
    router_parts: dict[str, list[np.ndarray]] = {}
    raw_expert_parts: dict[str, list[np.ndarray]] = {}
    amp_enabled = device.type == "cuda" and args.amp != "off"
    amp_dtype = torch.float16 if args.amp == "float16" else torch.bfloat16
    with torch.no_grad():
        for raw in loader:
            batch = _to_device(raw, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                model_output = model(
                    batch["obs"], batch["obs_mask"], batch["event_mask"],
                    batch["event_time_s"], batch["tau"],
                )
                prediction = model_output["position_mean"]
            errors = fixed_slot_physical_batch_errors(
                prediction.float(), batch["future_position"]
            )
            for name, value in errors.items():
                arrays.setdefault(name, []).append(value.detach().cpu().numpy())
            tau_parts.append(batch["tau"].detach().cpu().numpy())
            rule_parts.append(batch["rule_query"].detach().cpu().numpy())
            distance_parts.append(batch["distance_m"].detach().cpu().numpy())
            motion_parts.append(batch["motion_class"].detach().cpu().numpy())
            truth_state = truth_trajectory_targets(
                batch["future_position"][:, :4], batch["tau"][:, :4],
                model.decoder.geometry,
            )
            predicted_state = decoded_trajectory_state(
                {"position_mean": prediction[:, :4].float()},
                batch["tau"][:, :4], model.decoder.geometry,
            )
            eligible = (
                batch["rule_query"][:, :4].all(dim=1)
                & truth_state["constant_motion"]
            )
            eligibility_parts.append(eligible.detach().cpu().numpy())
            truth_speed = torch.linalg.vector_norm(
                truth_state["velocity"], dim=-1
            )
            truth_abs_omega = truth_state["omega"].abs()
            move_negative = float(getattr(args, "move_negative_mps", 0.01))
            move_positive = float(getattr(args, "move_positive_mps", 0.10))
            rotate_negative = float(
                getattr(args, "rotate_negative_rad_s", 0.05)
            )
            rotate_positive = float(
                getattr(args, "rotate_positive_rad_s", 0.20)
            )
            move_is_positive = truth_speed >= move_positive
            move_is_negative = truth_speed <= move_negative
            rotate_is_positive = truth_abs_omega >= rotate_positive
            rotate_is_negative = truth_abs_omega <= rotate_negative
            factor_valid = eligible & (move_is_positive | move_is_negative) & (
                rotate_is_positive | rotate_is_negative
            )
            factor_class = (
                move_is_positive.to(torch.long)
                + 2 * rotate_is_positive.to(torch.long)
            )
            factor_class = torch.where(
                factor_valid, factor_class, torch.full_like(factor_class, -1)
            )
            factor_parts.append(factor_class.detach().cpu().numpy())
            if "move_logit" in model_output and "rotate_logit" in model_output:
                values = {
                    "move_probability": torch.sigmoid(model_output["move_logit"]),
                    "move_positive": eligible & (truth_speed >= move_positive),
                    "move_negative": eligible & (truth_speed <= move_negative),
                    "rotate_probability": torch.sigmoid(model_output["rotate_logit"]),
                    "rotate_positive": eligible & (truth_abs_omega >= rotate_positive),
                    "rotate_negative": eligible & (truth_abs_omega <= rotate_negative),
                }
                for name, value in values.items():
                    gate_parts.setdefault(name, []).append(
                        value.detach().float().cpu().numpy()
                    )
            if "router_probability" in model_output and "route_index" in model_output:
                router_values = {
                    "probability": model_output["router_probability"],
                    "prediction": model_output["route_index"],
                    "truth": factor_class,
                    "valid": factor_valid,
                    "dataset_motion_class": batch["motion_class"],
                }
                for name, value in router_values.items():
                    router_parts.setdefault(name, []).append(
                        value.detach().cpu().numpy()
                    )
                raw_values = {
                    "translation_velocity": model_output["translation_velocity"],
                    "rotation_omega": model_output["rotation_omega"],
                    "combined_velocity": model_output["combined_velocity"],
                    "combined_omega": model_output["combined_omega"],
                    "truth_velocity": truth_state["velocity"],
                    "truth_omega": truth_state["omega"],
                    "factor_class": factor_class,
                }
                for name, value in raw_values.items():
                    raw_expert_parts.setdefault(name, []).append(
                        value.detach().float().cpu().numpy()
                    )
            if bool(eligible.any()):
                predicted_velocity = predicted_state["velocity"][eligible]
                truth_velocity = truth_state["velocity"][eligible]
                predicted_omega = predicted_state["omega"][eligible]
                truth_omega = truth_state["omega"][eligible]
                speed_product = (
                    torch.linalg.vector_norm(predicted_velocity, dim=-1)
                    * torch.linalg.vector_norm(truth_velocity, dim=-1)
                )
                direction = (
                    (predicted_velocity * truth_velocity).sum(dim=-1)
                    / speed_product.clamp_min(1e-8)
                )
                values = {
                    "velocity_error_mps": torch.linalg.vector_norm(
                        predicted_velocity - truth_velocity, dim=-1
                    ),
                    "omega_error_rad_s": (predicted_omega - truth_omega).abs(),
                    "velocity_direction_cosine": direction,
                    "predicted_speed_mps": torch.linalg.vector_norm(
                        predicted_velocity, dim=-1
                    ),
                    "truth_speed_mps": torch.linalg.vector_norm(
                        truth_velocity, dim=-1
                    ),
                    "predicted_omega_rad_s": predicted_omega,
                    "truth_omega_rad_s": truth_omega,
                }
                for name, value in values.items():
                    state_parts.setdefault(name, []).append(
                        value.detach().cpu().numpy()
                    )
                state_motion_parts.append(
                    batch["motion_class"][eligible].detach().cpu().numpy()
                )
    merged = {name: np.concatenate(parts, axis=0) for name, parts in arrays.items()}
    tau = np.concatenate(tau_parts, axis=0)
    rule = np.concatenate(rule_parts, axis=0).astype(np.bool_, copy=False)
    distance = np.concatenate(distance_parts, axis=0)
    motion_class = np.concatenate(motion_parts, axis=0)
    state_values = {
        name: np.concatenate(parts, axis=0) for name, parts in state_parts.items()
    }
    state_motion = (
        np.concatenate(state_motion_parts, axis=0)
        if state_motion_parts else np.empty((0,), dtype=np.int64)
    )
    trajectory_eligible = np.concatenate(eligibility_parts, axis=0).astype(
        np.bool_, copy=False
    )
    gate_values = {
        name: np.concatenate(parts, axis=0) for name, parts in gate_parts.items()
    }
    factor_class = np.concatenate(factor_parts, axis=0)
    router_values = {
        name: np.concatenate(parts, axis=0) for name, parts in router_parts.items()
    }
    raw_expert_values = {
        name: np.concatenate(parts, axis=0)
        for name, parts in raw_expert_parts.items()
    }

    def gate_diagnostics(prefix: str) -> dict[str, float | int | None] | None:
        probability = gate_values.get(f"{prefix}_probability")
        positive = gate_values.get(f"{prefix}_positive")
        negative = gate_values.get(f"{prefix}_negative")
        if probability is None or positive is None or negative is None:
            return None
        positive = positive.astype(np.bool_, copy=False)
        negative = negative.astype(np.bool_, copy=False)
        predicted = probability >= 0.5
        return {
            "positive_count": int(positive.sum()),
            "negative_count": int(negative.sum()),
            "positive_recall": (
                float(predicted[positive].mean()) if np.any(positive) else None
            ),
            "negative_false_positive_rate": (
                float(predicted[negative].mean()) if np.any(negative) else None
            ),
            "positive_probability_mean": (
                float(probability[positive].mean()) if np.any(positive) else None
            ),
            "negative_probability_mean": (
                float(probability[negative].mean()) if np.any(negative) else None
            ),
        }

    def state_diagnostics(mask: np.ndarray) -> dict[str, object]:
        if not state_values or not np.any(mask):
            return {"sample_count": 0}
        def state_summary(values: np.ndarray) -> dict[str, float | int]:
            return {
                "count": int(values.size), "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "max": float(np.max(values)),
            }
        truth_speed = state_values["truth_speed_mps"][mask]
        predicted_speed = state_values["predicted_speed_mps"][mask]
        truth_omega = state_values["truth_omega_rad_s"][mask]
        predicted_omega = state_values["predicted_omega_rad_s"][mask]
        moving = truth_speed >= 0.05
        rotating = np.abs(truth_omega) >= 0.5
        return {
            "sample_count": int(mask.sum()),
            "velocity_error_mps": state_summary(
                state_values["velocity_error_mps"][mask]
            ),
            "omega_error_rad_s": state_summary(
                state_values["omega_error_rad_s"][mask]
            ),
            "moving_direction_cosine_median": (
                float(np.median(state_values["velocity_direction_cosine"][mask][moving]))
                if np.any(moving) else None
            ),
            "moving_speed_ratio_median": (
                float(np.median(predicted_speed[moving] / truth_speed[moving]))
                if np.any(moving) else None
            ),
            "rotating_sign_accuracy": (
                float(np.mean(np.sign(predicted_omega[rotating]) == np.sign(truth_omega[rotating])))
                if np.any(rotating) else None
            ),
            "rotating_abs_omega_ratio_median": (
                float(np.median(
                    np.abs(predicted_omega[rotating]) / np.abs(truth_omega[rotating])
                )) if np.any(rotating) else None
            ),
        }

    def raw_motion_summary(
        velocity_name: str | None, omega_name: str | None, route: int,
    ) -> dict[str, object]:
        if not raw_expert_values:
            return {"sample_count": 0}
        mask = raw_expert_values["factor_class"] == route
        if not np.any(mask):
            return {"sample_count": 0}

        def values_summary(values: np.ndarray) -> dict[str, float | int]:
            return {
                "count": int(values.size), "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "max": float(np.max(values)),
            }

        result: dict[str, object] = {"sample_count": int(mask.sum())}
        if velocity_name is not None:
            predicted = raw_expert_values[velocity_name][mask]
            truth = raw_expert_values["truth_velocity"][mask]
            predicted_speed = np.linalg.norm(predicted, axis=-1)
            truth_speed = np.linalg.norm(truth, axis=-1)
            error = np.linalg.norm(predicted - truth, axis=-1)
            direction = np.sum(predicted * truth, axis=-1) / np.maximum(
                predicted_speed * truth_speed, 1e-8
            )
            result["velocity_error_mps"] = values_summary(error)
            result["velocity_direction_cosine_median"] = float(
                np.median(direction)
            )
            result["speed_ratio_median"] = float(
                np.median(predicted_speed / np.maximum(truth_speed, 1e-8))
            )
        if omega_name is not None:
            predicted = raw_expert_values[omega_name][mask]
            truth = raw_expert_values["truth_omega"][mask]
            result["omega_error_rad_s"] = values_summary(
                np.abs(predicted - truth)
            )
            result["omega_sign_accuracy"] = float(
                np.mean(np.sign(predicted) == np.sign(truth))
            )
            result["abs_omega_ratio_median"] = float(
                np.median(np.abs(predicted) / np.maximum(np.abs(truth), 1e-8))
            )
        return result

    def router_diagnostics() -> dict[str, object] | None:
        if not router_values:
            return None
        valid = router_values["valid"].astype(np.bool_, copy=False)
        truth = router_values["truth"].astype(np.int64, copy=False)[valid]
        predicted = router_values["prediction"].astype(np.int64, copy=False)[valid]
        dataset_class = router_values["dataset_motion_class"].astype(
            np.int64, copy=False
        )[valid]
        confusion = np.zeros((4, 4), dtype=np.int64)
        np.add.at(confusion, (truth, predicted), 1)
        labels = ("stationary", "translation", "rotation", "combined")
        per_class: dict[str, object] = {}
        recalls = []
        for index, label in enumerate(labels):
            support = int(confusion[index].sum())
            predicted_count = int(confusion[:, index].sum())
            true_positive = int(confusion[index, index])
            negatives = int(confusion.sum() - support)
            recall = true_positive / support if support else None
            precision = true_positive / predicted_count if predicted_count else None
            false_positive = predicted_count - true_positive
            fpr = false_positive / negatives if negatives else None
            if recall is not None:
                recalls.append(recall)
            per_class[label] = {
                "support": support, "predicted_count": predicted_count,
                "precision": precision, "recall": recall,
                "false_positive_rate": fpr,
            }
        move_truth = np.isin(truth, (1, 3))
        move_predicted = np.isin(predicted, (1, 3))
        rotate_truth = np.isin(truth, (2, 3))
        rotate_predicted = np.isin(predicted, (2, 3))

        def factor_gate(
            factor_truth: np.ndarray, factor_predicted: np.ndarray,
        ) -> dict[str, float | int | None]:
            positive = int(factor_truth.sum())
            negative = int((~factor_truth).sum())
            return {
                "positive_count": positive, "negative_count": negative,
                "positive_recall": (
                    float(factor_predicted[factor_truth].mean())
                    if positive else None
                ),
                "negative_false_positive_rate": (
                    float(factor_predicted[~factor_truth].mean())
                    if negative else None
                ),
            }

        dataset_cross_tab = np.zeros((4, 4), dtype=np.int64)
        np.add.at(dataset_cross_tab, (truth, dataset_class), 1)
        eligible_count = int(trajectory_eligible.sum())
        return {
            "labels": list(labels),
            "confusion_true_rows_predicted_columns": confusion.tolist(),
            "per_class": per_class,
            "valid_count": int(valid.sum()),
            "eligible_count": eligible_count,
            "valid_over_eligible": (
                float(valid.sum() / eligible_count) if eligible_count else 0.0
            ),
            "accuracy": float(np.mean(truth == predicted)) if truth.size else None,
            "macro_recall": float(np.mean(recalls)) if recalls else None,
            "move_from_hard_route": factor_gate(move_truth, move_predicted),
            "rotate_from_hard_route": factor_gate(rotate_truth, rotate_predicted),
            "factor_vs_dataset_motion_class_agreement": (
                float(np.mean(truth == dataset_class)) if truth.size else None
            ),
            "factor_true_rows_dataset_class_columns": dataset_cross_tab.tolist(),
        }

    queries: list[dict[str, object]] = []
    for query in range(tau.shape[1]):
        active = rule[:, query]
        queries.append({
            "query_index": query,
            "tau_s": {"median": float(np.median(tau[:, query]))},
            "absolute": summary(merged["absolute_pg_m"][:, query]),
            "motion_delta": summary(merged["motion_delta_m"][:, query]),
            "rule": {
                "absolute": summary(merged["absolute_pg_m"][active, query]),
                "motion_delta": summary(merged["motion_delta_m"][active, query]),
            },
            "future_event": {
                "absolute": summary(merged["absolute_pg_m"][~active, query]),
                "motion_delta": summary(merged["motion_delta_m"][~active, query]),
            },
            "trajectory_eligible": {
                "absolute": summary(
                    merged["absolute_pg_m"][trajectory_eligible, query]
                ),
                "motion_delta": summary(
                    merged["motion_delta_m"][trajectory_eligible, query]
                ),
            },
        })

    def stratum(mask: np.ndarray) -> dict[str, object]:
        return {
            "sample_count": int(mask.sum()),
            "state_q0": summary(merged["state_q0_m"][mask]),
            "q3_rule_motion": summary(
                merged["motion_delta_m"][mask & rule[:, 3], 3]
            ),
            "q3_trajectory_eligible_motion": summary(
                merged["motion_delta_m"][mask & trajectory_eligible, 3]
            ),
        }

    motion_labels = {0: "stationary", 1: "linear", 2: "spin", 3: "linear_and_spin"}
    factor_labels = {
        0: "stationary", 1: "translation", 2: "rotation", 3: "combined",
    }
    all_state = np.ones(state_motion.shape, dtype=np.bool_)
    result = {
        "sample_count": int(tau.shape[0]),
        "slot_policy": "fixed-causal-slots; no permutation search",
        "state_q0": summary(merged["state_q0_m"]),
        "trajectory_eligible_state_q0": summary(
            merged["state_q0_m"][trajectory_eligible]
        ),
        "trajectory_supervision": {
            "eligible_count": int(trajectory_eligible.sum()),
            "coverage": float(trajectory_eligible.mean()),
        },
        "rigid_residual": summary(merged["rigid_residual_m"].reshape(-1)),
        "queries": queries,
        "rule_query_fraction": float(rule.mean()),
        "trajectory_state_diagnostics": {
            "overall": state_diagnostics(all_state),
            "motion_class": {
                label: state_diagnostics(state_motion == index)
                for index, label in motion_labels.items()
            },
            "policy": "reparsed identically from both arms' decoded q0..q3 positions",
        },
        "strata": {
            "motion_class": {
                label: stratum(motion_class == index)
                for index, label in motion_labels.items()
            },
            "motion_factor": {
                label: stratum(factor_class == index)
                for index, label in factor_labels.items()
            },
            "distance": {
                "near_lt_3m": stratum(distance < 3.0),
                "mid_3_to_5m": stratum((distance >= 3.0) & (distance < 5.0)),
                "far_ge_5m": stratum(distance >= 5.0),
            },
        },
    }
    move_gate = gate_diagnostics("move")
    rotate_gate = gate_diagnostics("rotate")
    if move_gate is not None and rotate_gate is not None:
        result["expert_gate_diagnostics"] = {
            "move": move_gate, "rotate": rotate_gate,
            "threshold": 0.5,
            "dead_band": {
                "move_mps": [
                    float(getattr(args, "move_negative_mps", 0.01)),
                    float(getattr(args, "move_positive_mps", 0.10)),
                ],
                "rotate_rad_s": [
                    float(getattr(args, "rotate_negative_rad_s", 0.05)),
                    float(getattr(args, "rotate_positive_rad_s", 0.20)),
                ],
            },
        }
    router = router_diagnostics()
    if router is not None:
        result["router_diagnostics"] = router
        result["raw_expert_diagnostics"] = {
            "translation": raw_motion_summary(
                "translation_velocity", None, 1,
            ),
            "rotation": raw_motion_summary(None, "rotation_omega", 2),
            "combined": raw_motion_summary(
                "combined_velocity", "combined_omega", 3,
            ),
            "policy": (
                "raw specialist outputs before hard routing; truth-derived "
                "factor labels on trajectory-eligible non-dead-band samples"
            ),
        }
    return result


def _selection_tuple(metrics: dict[str, object]) -> tuple[float, ...]:
    queries = metrics["queries"]  # type: ignore[assignment]
    headline = queries[1:4]
    return (
        max(float(item["trajectory_eligible"]["motion_delta"]["p95_m"]) for item in headline),
        float(metrics["trajectory_eligible_state_q0"]["p95_m"]),  # type: ignore[index]
        max(float(item["trajectory_eligible"]["absolute"]["p95_m"]) for item in headline),
        max(float(item["trajectory_eligible"]["motion_delta"]["median_m"]) for item in headline),
    )


def _checkpoint(
    path: Path, model: PhysicalModel, label: str, epoch: int,
    metrics: dict[str, object], provenance: dict[str, object], role: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    payload: dict[str, object] = {
        "model": model.state_dict(), "model_class": model.__class__.__name__,
        "model_config": model.config(), "label": label, "epoch": epoch,
        "checkpoint_role": role, "validation": metrics,
        "selection_tuple": _selection_tuple(metrics), "provenance": provenance,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def train(args: argparse.Namespace) -> Path:
    _seed(args.seed)
    git_state = _git_state()
    if bool(git_state["worktree_dirty"]) and not args.allow_dirty_worktree:
        raise ValueError(
            "official causal physical training requires a clean worktree; "
            "use --allow-dirty-worktree only for explicitly exploratory runs"
        )
    dataset = Path(args.dataset).resolve()
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "stage3-causal-physical-v1":
        raise ValueError("causal physical A/B requires stage3-causal-physical-v1")
    if not bool(manifest.get("qualification_passed", False)):
        raise ValueError("causal physical A/B requires a qualified dataset")
    if bool(manifest.get("test_accessed", True)):
        raise ValueError("causal physical A/B refuses a test-accessed dataset")
    identity = manifest.get("identity_contract", {})
    if identity.get("policy") != "causal-cyclic-fixed-slots-v1":
        raise ValueError("causal physical A/B requires persistent cyclic slots")
    if bool(identity.get("permutation_search", True)):
        raise ValueError("causal physical A/B forbids permutation association")
    _validate_history_contract(manifest, args.history_events)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training output: {output}")
    geometry, geometry_payload, geometry_sha256 = _load_geometry(dataset)
    train_sessions, validation_sessions, selection_record = _load_selection(
        args.selection, manifest_path
    )
    if args.validation_on_train and selection_record is not None:
        raise ValueError("validation-on-train cannot use a held-out pilot selection")

    train_ds = CausalPhysicalShardDataset(
        dataset, "train", seed=args.seed, shuffle=not args.validation_on_train,
        sample_limit=args.train_sample_limit, session_ids=train_sessions,
    )
    validation_split = (
        "train" if args.validation_on_train
        or selection_record is not None
        and selection_record["validation_source_split"] == "train"
        else "validation"
    )
    validation_ds = CausalPhysicalShardDataset(
        dataset, validation_split, seed=args.seed, shuffle=False,
        sample_limit=args.validation_sample_limit,
        session_ids=(
            train_sessions if validation_split == "train" else validation_sessions
        ),
    )
    dataset_qualification = {
        "train": _audit_dataset_contract(
            train_ds, geometry, args.history_events,
            args.minimum_supervision_coverage,
            {0, 1, 2, 3} if selection_record is None else None,
        ),
        "validation": _audit_dataset_contract(
            validation_ds, geometry, args.history_events,
            args.minimum_supervision_coverage,
            {0, 1, 2, 3} if selection_record is None else None,
        ),
        "minimum_required_coverage": args.minimum_supervision_coverage,
    }
    output.mkdir(parents=True)
    loader_options = {
        "batch_size": args.batch_size, "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, **loader_options)
    validation_loader = DataLoader(validation_ds, **loader_options)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    common = {
        "geometry": geometry,
        "input_features": 5,
        "channels": args.channels,
        "dropout": args.dropout,
        "history_events": args.history_events,
        "position_mean": torch.from_numpy(train_ds.mean),
        "position_std": torch.from_numpy(train_ds.std),
    }
    model_a = ExplicitStatePhysicalPredictor(
        **common,
    )
    model_b = ImplicitQueryPhysicalPredictor(
        **common,
    )
    model_b.encoder.load_state_dict(model_a.encoder.state_dict())
    encoder_sha256 = _state_dict_sha256(model_a.encoder.state_dict())
    if encoder_sha256 != _state_dict_sha256(model_b.encoder.state_dict()):
        raise RuntimeError("paired A/B models do not share encoder initialization")
    parameter_counts = {
        "A_explicit_state": trainable_parameter_count(model_a),
        "B_implicit_query": trainable_parameter_count(model_b),
    }
    parameter_gap = abs(parameter_counts["A_explicit_state"] - parameter_counts["B_implicit_query"])
    parameter_gap /= max(parameter_counts.values())
    if parameter_gap >= 0.01:
        raise RuntimeError("paired A/B trainable parameter gap must stay below one percent")
    models = {
        "A_explicit_state": model_a.to(device),
        "B_implicit_query": model_b.to(device),
    }
    optimizers = {
        label: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for label, model in models.items()
    }
    schedulers = {
        label: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max(1, args.epochs), eta_min=args.lr * 0.02
        ) for label, optimizer in optimizers.items()
    }
    scalers = {
        label: torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and args.amp == "float16"
        ) for label in models
    }
    source_names = (
        "train_causal_physical_ab.py", "causal_physical_dataset.py",
        "causal_physical_state_model.py", "physical_model.py",
        "physical_loss.py", "physical_metrics.py", "pnp_state_targets.py",
    )
    source_root = Path(__file__).resolve().parent
    provenance: dict[str, object] = {
        "schema_version": "stage3-causal-physical-state-ab-run-v1",
        "dataset": str(dataset), "dataset_manifest_sha256": _sha256(manifest_path),
        "geometry_template_sha256": geometry_sha256,
        "geometry_hash": geometry_payload.get("geometry_hash"),
        "test_accessed": False, "validation_split": validation_split,
        "dataset_qualification": dataset_qualification,
        "session_selection": selection_record,
        "slot_policy": "fixed causal cyclic slots; no permutation search",
        "input_allowlist": ["normalized exact xyz", "cyclic slot sin/cos", "event mask", "real event time", "tau", "train-only normalization"],
        "forbidden_predictor_inputs": ["center", "velocity", "yaw", "yaw_rate", "motion_class", "rule_query", "future truth"],
        "paired_contract": {
            "shared_encoder_initial_sha256": encoder_sha256,
            "trainable_parameters": parameter_counts,
            "relative_parameter_gap": parameter_gap,
            "same_batches_rng_optimizer_scheduler_amp": True,
        },
        "objectives": {
            "A_explicit_state": "common decoded-position objective",
            "B_implicit_query": "common decoded-position objective",
            "formula": (
                "center0_m + reference_horizon*velocity_mps + "
                "geometry_radius*phase0 + geometry_radius*reference_horizon*omega + "
                "constant-twist center/phase consistency"
            ),
            "reference_horizon_s": args.reference_horizon_s,
            "label_policy": "future truth is loss-only; state reparsed identically from both arms",
        },
        "architecture_contract": {
            "A": "neural center0/velocity/phase0/omega then frozen constant twist",
            "B": "neural per-query center/phase; no velocity or omega output",
            "analytic_state_recovery": False,
            "history_events": args.history_events,
            "evaluation_only_state_reparse": (
                "closed-form velocity/yaw-rate diagnostics are applied equally "
                "to decoded A/B and truth positions; never forward inputs or losses"
            ),
        },
        "config": vars(args),
        "source_sha256": {name: _sha256(source_root / name) for name in source_names},
        "environment": {
            "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "device": str(device), "amp": args.amp,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        **git_state,
    }
    _write_json(output / "run_manifest.json", provenance)

    validation = {label: _validate(model, validation_loader, device, args) for label, model in models.items()}
    history: list[dict[str, object]] = [{
        "epoch": 0, "validation": validation,
        "selection_tuple": {label: _selection_tuple(value) for label, value in validation.items()},
    }]
    history_path = output / "stage3-causal-physical-ab-history.json"
    _write_json(history_path, history)
    initial_selection = {
        label: _selection_tuple(value) for label, value in validation.items()
    }
    best = {label: (float("inf"),) * 4 for label in models}
    best_epoch = {label: -1 for label in models}
    best_paths = {
        label: output / f"stage3-causal-physical-{label}-seed{args.seed}-best.pt"
        for label in models
    }
    for label, model in models.items():
        _checkpoint(
            output / f"stage3-causal-physical-{label}-seed{args.seed}-initial.pt",
            model, label, 0, validation[label], provenance, "initial",
        )
    stale = {label: 0 for label in models}
    stop_reason = "epoch_limit"
    started = time.monotonic()
    epochs_completed = 0
    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        train_metrics = _train_epoch(
            models, train_loader, optimizers, scalers, device, args
        )
        for scheduler in schedulers.values():
            scheduler.step()
        validation = {
            label: _validate(model, validation_loader, device, args)
            for label, model in models.items()
        }
        epochs_completed = epoch
        record = {
            "epoch": epoch, "train": train_metrics, "validation": validation,
            "selection_tuple": {label: _selection_tuple(value) for label, value in validation.items()},
            "lr": {label: optimizer.param_groups[0]["lr"] for label, optimizer in optimizers.items()},
        }
        history.append(record)
        _write_json(history_path, history)
        print(json.dumps(record, sort_keys=True), flush=True)
        for label, model in models.items():
            selection = _selection_tuple(validation[label])
            improved = selection < best[label]
            if improved:
                best[label] = selection
                best_epoch[label] = epoch
                stale[label] = 0
                _checkpoint(
                    best_paths[label], model, label, epoch, validation[label], provenance,
                    "best", optimizers[label], schedulers[label], scalers[label],
                )
            elif epoch > args.early_stopping_warmup:
                stale[label] += 1
            else:
                stale[label] = 0
        if (
            args.patience > 0 and epoch > args.early_stopping_warmup
            and all(value >= args.patience for value in stale.values())
        ):
            stop_reason = "both_models_early_stopping"
            break
        if args.max_wall_minutes > 0 and time.monotonic() - started >= args.max_wall_minutes * 60:
            stop_reason = "wall_time_limit"
            break
    for label, model in models.items():
        _checkpoint(
            output / f"stage3-causal-physical-{label}-seed{args.seed}-last.pt",
            model, label, epochs_completed, validation[label], provenance, "last",
            optimizers[label], schedulers[label], scalers[label],
        )
    final = {
        **provenance, "status": "complete", "stop_reason": stop_reason,
        "initial_selection_tuple": initial_selection,
        "epochs_completed": epochs_completed,
        "best": {
            label: {
                "path": path.name, "sha256": _sha256(path),
                "selection_tuple": best[label], "epoch": best_epoch[label],
                "trained_checkpoint": best_epoch[label] > 0,
            }
            for label, path in best_paths.items()
        },
    }
    _write_json(output / "run_manifest.json", final)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--patience", type=int, default=0,
        help="epochs without improvement after warmup; zero disables early stopping",
    )
    parser.add_argument("--early-stopping-warmup", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--history-events", type=int, default=32)
    parser.add_argument("--huber-beta-m", type=float, default=0.005)
    parser.add_argument("--reference-horizon-s", type=float, default=0.5)
    parser.add_argument("--minimum-supervision-coverage", type=float, default=0.85)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument(
        "--minimum-update-loss", type=float, default=1e-10,
        help="skip numerically exact batches so Adam cannot push a perfect zero-motion prior away",
    )
    parser.add_argument("--device", default="")
    parser.add_argument("--amp", choices=("off", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--train-sample-limit", type=int, default=0)
    parser.add_argument("--validation-sample-limit", type=int, default=0)
    parser.add_argument("--validation-on-train", action="store_true")
    parser.add_argument("--selection", default="")
    parser.add_argument("--max-wall-minutes", type=float, default=0.0)
    parser.add_argument(
        "--allow-dirty-worktree", action="store_true",
        help="permit an explicitly exploratory, non-release training run",
    )
    args = parser.parse_args()
    positive = (
        args.epochs, args.batch_size, args.lr,
        args.channels, args.huber_beta_m, args.reference_horizon_s,
    )
    if any(value <= 0 for value in positive):
        parser.error("causal physical A/B arguments must be positive")
    if args.weight_decay < 0:
        parser.error("weight-decay cannot be negative")
    if args.patience < 0 or args.early_stopping_warmup < 0:
        parser.error("patience and early-stopping-warmup cannot be negative")
    if args.channels % 4:
        parser.error("channels must be divisible by four")
    if args.train_sample_limit < 0 or args.validation_sample_limit < 0:
        parser.error("sample limits cannot be negative")
    if args.minimum_update_loss < 0:
        parser.error("minimum-update-loss cannot be negative")
    if args.grad_clip < 0:
        parser.error("grad-clip cannot be negative")
    if not 8 <= args.history_events <= 200:
        parser.error("history-events must be within [8,200]")
    if not 0 < args.minimum_supervision_coverage <= 1:
        parser.error("minimum-supervision-coverage must be within (0,1]")
    if args.validation_on_train and (
        args.train_sample_limit <= 0 or args.validation_sample_limit <= 0
    ):
        parser.error("validation-on-train requires bounded sample limits")
    print(train(args))


if __name__ == "__main__":
    main()
