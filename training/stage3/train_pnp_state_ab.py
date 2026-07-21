"""Train a fair PnP-history explicit-state versus implicit-query-pose A/B.

Both arms consume the same train/validation PnP history and predict the same
unordered rigid four-armor set. A emits one constant-twist state. B emits an
independent rigid pose for every query tau. No future observation, exact state,
motion class, test shard, truth slot, or permutation search is available to
either predictor.
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

import numpy as np
import torch
from torch.utils.data import DataLoader

from .physical_metrics import summary
from .pnp_state_loss import pnp_state_constrained_loss
from .pnp_state_metrics import SET_POLICY, pnp_state_set_batch_errors
from .pnp_state_model import (
    ExplicitStatePnPAdapter,
    ImplicitQueryPosePredictor,
    trainable_parameter_count,
)
from .shard_dataset import Stage3ShardDataset
from .pnp_state_targets import (
    decoded_trajectory_state,
    geometry_c4_asymmetry_m,
    truth_trajectory_targets,
)


LABEL_A = "A_explicit_state"
LABEL_B = "B_implicit_query_pose"


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


def _load_session_selection(
    path_value: str, manifest_path: Path,
) -> tuple[list[str] | None, list[str] | None, dict[str, object] | None]:
    if not path_value:
        return None, None, None
    path = Path(path_value).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "stage3-pnp-state-pilot-selection-v1":
        raise ValueError("unsupported PnP state pilot selection schema")
    if payload.get("dataset_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("pilot selection does not match the dataset manifest")
    if payload.get("validation_source_split") != "validation":
        raise ValueError("pilot validation sessions must come from validation")
    if payload.get("test") != []:
        raise ValueError("pilot selection must keep test empty")

    selections: dict[str, list[str]] = {}
    for split in ("train", "validation"):
        values = payload.get(split)
        if not isinstance(values, list) or not values:
            raise ValueError(f"pilot selection has no non-empty {split} session list")
        selected = [str(value) for value in values]
        if len(selected) != len(set(selected)):
            raise ValueError(f"pilot selection contains duplicate {split} sessions")
        selections[split] = selected
    if set(selections["train"]) & set(selections["validation"]):
        raise ValueError("pilot train and validation session lists must be disjoint")
    record = {
        "path": str(path),
        "sha256": _sha256(path),
        "purpose": str(payload.get("purpose", "")),
        "train": selections["train"],
        "validation": selections["validation"],
        "test": [],
    }
    return selections["train"], selections["validation"], record


def _load_geometry_and_center_prior(
    dataset: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
    geometry_path = dataset / "geometry_template.json"
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    armor = sorted(payload["armors"], key=lambda item: int(item["relative_slot"]))
    geometry = torch.tensor(
        [item["relative_position_m"] for item in armor], dtype=torch.float32
    )
    if tuple(geometry.shape) != (4, 3):
        raise ValueError("geometry template must contain four relative xyz slots")
    manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    normalization_path = dataset / str(manifest["normalization"])
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    observation_mean = torch.tensor(
        normalization["obs_xyz"]["mean"], dtype=torch.float32
    )
    observation_std = torch.tensor(
        normalization["obs_xyz"]["std"], dtype=torch.float32
    )
    # Only train-split normalization is used. The geometry mean converts the
    # typical armor position into a useful target-center reference.
    center_reference = observation_mean - geometry.mean(dim=0)
    center_scale = observation_std.clamp_min(0.25)
    return (
        geometry,
        center_reference,
        center_scale,
        _sha256(geometry_path),
        _sha256(normalization_path),
    )


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "train_pnp_state_ab.py", "pnp_state_model.py", "pnp_state_loss.py",
        "pnp_state_metrics.py", "pnp_state_targets.py", "physical_model.py", "physical_metrics.py",
        "shard_dataset.py",
    )
    return {name: _sha256(root / name) for name in names}


def _capture_rng(device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    return torch.get_rng_state(), (
        torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    )


def _restore_rng(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def _to_device(raw: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    # The explicit allowlist prevents v4 future-observation labels and masks
    # from ever entering the predictor or optimizer path.
    names = (
        "obs", "obs_mask", "event_mask", "event_time_s", "tau",
        "future_position", "motion_class",
    )
    return {name: raw[name].to(device, non_blocking=True) for name in names}


def _trajectory_consistency(
    center: torch.Tensor, phase: torch.Tensor, tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMS residual on q0/0.1/0.2 after one constant-twist fit per sample.

    The first three headline queries stay below the pi/15-rad/s unwrap alias
    boundary. The 0.5-second and randomly ordered queries are deliberately not
    used for this diagnostic.
    """
    if center.shape[1] < 3 or phase.shape[1] < 3 or tau.shape[1] < 3:
        raise ValueError("trajectory consistency requires q0, q1, and q2")
    center, phase, tau = center[:, :3], phase[:, :3], tau[:, :3]
    order = tau.argsort(dim=1)
    center_order = order[:, :, None].expand(-1, -1, 3)
    phase_order = order[:, :, None].expand(-1, -1, 2)
    t = tau.gather(1, order)
    c = center.gather(1, center_order)
    p = phase.gather(1, phase_order)
    yaw_raw = torch.atan2(p[..., 1], p[..., 0])
    delta = torch.atan2(
        torch.sin(yaw_raw[:, 1:] - yaw_raw[:, :-1]),
        torch.cos(yaw_raw[:, 1:] - yaw_raw[:, :-1]),
    )
    yaw = torch.cat((yaw_raw[:, :1], yaw_raw[:, :1] + delta.cumsum(dim=1)), dim=1)
    t_mean = t.mean(dim=1, keepdim=True)
    centered_t = t - t_mean
    denominator = centered_t.square().sum(dim=1, keepdim=True).clamp_min(1e-12)
    c_mean = c.mean(dim=1, keepdim=True)
    c_slope = (centered_t[:, :, None] * (c - c_mean)).sum(dim=1) / denominator
    c_fit = c_mean + centered_t[:, :, None] * c_slope[:, None, :]
    yaw_mean = yaw.mean(dim=1, keepdim=True)
    yaw_slope = (centered_t * (yaw - yaw_mean)).sum(dim=1, keepdim=True) / denominator
    yaw_fit = yaw_mean + centered_t * yaw_slope
    center_rms = torch.sqrt((c - c_fit).square().sum(dim=-1).mean(dim=1))
    yaw_residual = torch.atan2(torch.sin(yaw - yaw_fit), torch.cos(yaw - yaw_fit))
    yaw_rms = torch.sqrt(yaw_residual.square().mean(dim=1))
    return center_rms, yaw_rms


def _train_one(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = device.type == "cuda" and args.amp != "off"
    amp_dtype = torch.float16 if args.amp == "float16" else torch.bfloat16
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        output = model(
            batch["obs"], batch["obs_mask"], batch["event_mask"],
            batch["event_time_s"], batch["tau"],
        )
        loss, parts = pnp_state_constrained_loss(
            output, batch["future_position"], batch["tau"], model.decoder.geometry,
            huber_beta_m=args.huber_beta_m,
        )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    return {
        "objective": float(loss.detach().cpu()),
        **{name: float(value.detach().cpu()) for name, value in parts.items()},
        "gradient_norm": float(gradient_norm.detach().cpu()),
    }


def _train_epoch_pair(
    models: dict[str, torch.nn.Module],
    loader: DataLoader,
    optimizers: dict[str, torch.optim.Optimizer],
    scalers: dict[str, torch.amp.GradScaler],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    for model in models.values():
        model.train()
    totals: dict[str, dict[str, float]] = {label: {} for label in models}
    count = 0
    for raw in loader:
        batch = _to_device(raw, device)
        batch_count = int(batch["obs"].shape[0])
        shared_rng = _capture_rng(device)
        for index, (label, model) in enumerate(models.items()):
            if index:
                _restore_rng(shared_rng)
            values = _train_one(
                model, batch, optimizers[label], scalers[label], device, args
            )
            for name, value in values.items():
                totals[label][name] = totals[label].get(name, 0.0) + value * batch_count
        count += batch_count
    return {
        label: {name: value / max(count, 1) for name, value in values.items()}
        for label, values in totals.items()
    }


def _validate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    model.eval()
    arrays: dict[str, list[np.ndarray]] = {}
    all_absolute_parts: list[np.ndarray] = []
    tau_parts: list[np.ndarray] = []
    motion_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    latest_visible_parts: list[np.ndarray] = []
    input_sample_count = 0
    qualified_by_motion = np.zeros(4, dtype=np.int64)
    input_by_motion = np.zeros(4, dtype=np.int64)
    amp_enabled = device.type == "cuda" and args.amp != "off"
    amp_dtype = torch.float16 if args.amp == "float16" else torch.bfloat16
    with torch.no_grad():
        for raw in loader:
            batch = _to_device(raw, device)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                output = model(
                    batch["obs"], batch["obs_mask"], batch["event_mask"],
                    batch["event_time_s"], batch["tau"],
                )
            truth_state = truth_trajectory_targets(
                batch["future_position"], batch["tau"], model.decoder.geometry
            )
            eligible = truth_state["constant_motion"]
            all_error = pnp_state_set_batch_errors(
                output["position_mean"].float(), batch["future_position"]
            )
            all_absolute_parts.append(
                all_error["absolute_set_m"].detach().cpu().numpy()
            )
            input_sample_count += int(eligible.numel())
            for motion_index in range(4):
                class_mask = batch["motion_class"] == motion_index
                input_by_motion[motion_index] += int(class_mask.sum().item())
                qualified_by_motion[motion_index] += int((class_mask & eligible).sum().item())
            if not bool(eligible.any()):
                continue
            error = pnp_state_set_batch_errors(
                output["position_mean"][eligible].float(),
                batch["future_position"][eligible],
            )
            center_rms, yaw_rms = _trajectory_consistency(
                output["query_center"][eligible].float(),
                output["query_phase"][eligible].float(), batch["tau"][eligible],
            )
            predicted_state = decoded_trajectory_state(
                output, batch["tau"], model.decoder.geometry
            )
            arrays.setdefault("velocity_error_mps", []).append(
                torch.linalg.vector_norm(
                    predicted_state["velocity"][eligible]
                    - truth_state["velocity"][eligible], dim=-1,
                ).detach().cpu().numpy()
            )
            arrays.setdefault("omega_error_rad_s", []).append(
                (
                    predicted_state["omega"][eligible]
                    - truth_state["omega"][eligible]
                ).abs().detach().cpu().numpy()
            )
            for name, value in error.items():
                arrays.setdefault(name, []).append(value.detach().cpu().numpy())
            arrays.setdefault("constant_twist_center_rms_m", []).append(
                center_rms.detach().cpu().numpy()
            )
            arrays.setdefault("constant_twist_yaw_rms_rad", []).append(
                yaw_rms.detach().cpu().numpy()
            )
            if "velocity" in output:
                arrays.setdefault("shared_speed_mps", []).append(
                    torch.linalg.vector_norm(output["velocity"][eligible].float(), dim=-1)
                    .detach().cpu().numpy()
                )
            if "omega" in output:
                arrays.setdefault("shared_abs_omega_rad_s", []).append(
                    output["omega"][eligible].float().abs().detach().cpu().numpy()
                )
            tau_parts.append(batch["tau"][eligible].detach().cpu().numpy())
            motion_parts.append(batch["motion_class"][eligible].detach().cpu().numpy())
            distance_parts.append(torch.linalg.vector_norm(
                batch["future_position"][eligible, 0].mean(dim=1), dim=-1
            ).detach().cpu().numpy())
            event_indices = torch.arange(
                batch["event_mask"].shape[1], device=device
            )[None, :].expand(batch["event_mask"].shape[0], -1)
            last = torch.where(
                batch["event_mask"], event_indices, torch.full_like(event_indices, -1)
            ).amax(dim=1).clamp_min(0)
            visible = batch["obs_mask"].sum(dim=2).gather(1, last[:, None]).squeeze(1)
            latest_visible_parts.append(visible[eligible].detach().cpu().numpy())
    merged = {name: np.concatenate(parts) for name, parts in arrays.items()}
    tau = np.concatenate(tau_parts)
    motion_class = np.concatenate(motion_parts)
    distance = np.concatenate(distance_parts)
    latest_visible = np.concatenate(latest_visible_parts)
    all_absolute = np.concatenate(all_absolute_parts)

    queries = [{
        "query_index": query,
        "tau_s": {"median": float(np.median(tau[:, query]))},
        "absolute": summary(merged["absolute_set_m"][:, query]),
        "hausdorff": summary(merged["hausdorff_set_m"][:, query]),
        "centroid": summary(merged["centroid_m"][:, query]),
        "centroid_motion": (
            {"count": 0} if query == 0
            else summary(merged["centroid_motion_m"][:, query - 1])
        ),
    } for query in range(tau.shape[1])]

    def stratum(mask: np.ndarray) -> dict[str, object]:
        return {
            "sample_count": int(mask.sum()),
            "queries": [{
                "query_index": query,
                "absolute": summary(merged["absolute_set_m"][mask, query]),
                "centroid_motion": (
                    {"count": 0} if query == 0 else
                    summary(merged["centroid_motion_m"][mask, query - 1])
                ),
            } for query in range(min(4, tau.shape[1]))],
        }

    motion_names = {0: "stationary", 1: "linear", 2: "spin", 3: "linear_and_spin"}
    qualification_fractions = np.divide(
        qualified_by_motion, np.maximum(input_by_motion, 1)
    )
    if tau.shape[0] / max(input_sample_count, 1) < 0.75:
        raise ValueError("constant-motion qualification coverage is below 75 percent")
    if np.any((input_by_motion > 0) & (qualification_fractions < 0.75)):
        raise ValueError("a motion-class qualification coverage is below 75 percent")
    return {
        "sample_count": int(tau.shape[0]),
        "input_sample_count": input_sample_count,
        "constant_motion_qualification": {
            "qualified": int(tau.shape[0]),
            "excluded": int(input_sample_count - tau.shape[0]),
            "fraction": float(tau.shape[0] / max(input_sample_count, 1)),
            "by_motion_class": {
                motion_names[index]: {
                    "input": int(input_by_motion[index]),
                    "qualified": int(qualified_by_motion[index]),
                } for index in range(4)
            },
        },
        "set_policy": SET_POLICY,
        "all_input_queries": [{
            "query_index": query,
            "absolute": summary(all_absolute[:, query]),
        } for query in range(min(4, all_absolute.shape[1]))],
        "state_q0": summary(merged["state_q0_m"]),
        "rigid_shape": summary(merged["rigid_shape_m"].reshape(-1)),
        "constant_twist_center_rms": summary(merged["constant_twist_center_rms_m"]),
        "constant_twist_yaw_rms_rad": summary(merged["constant_twist_yaw_rms_rad"]),
        "trajectory_state_error": {
            "velocity_mps": summary(merged["velocity_error_mps"]),
            "omega_rad_s": summary(merged["omega_error_rad_s"]),
        },
        "shared_state": (
            {
                "speed_mps": summary(merged["shared_speed_mps"]),
                "abs_omega_rad_s": summary(merged["shared_abs_omega_rad_s"]),
            }
            if "shared_speed_mps" in merged else None
        ),
        "queries": queries,
        "strata": {
            "motion_class": {
                name: stratum(motion_class == index)
                for index, name in motion_names.items()
            },
            "distance": {
                "near_lt_3m": stratum(distance < 3.0),
                "mid_3_to_5m": stratum((distance >= 3.0) & (distance < 5.0)),
                "far_ge_5m": stratum(distance >= 5.0),
            },
            "latest_visible_count": {
                str(count): stratum(latest_visible == count) for count in range(1, 5)
            },
        },
    }


def _selection_tuple(metrics: dict[str, object]) -> tuple[float, ...]:
    queries = metrics["queries"]  # type: ignore[assignment]
    return (
        max(float(query["absolute"]["p95_m"]) for query in queries[:4]),
        float(metrics["state_q0"]["p95_m"]),  # type: ignore[index]
        max(float(query["centroid_motion"]["p95_m"]) for query in queries[1:4]),
        max(float(query["absolute"]["median_m"]) for query in queries[:4]),
    )


def _checkpoint(
    path: Path, model: torch.nn.Module, label: str, epoch: int,
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
    dataset = Path(args.dataset).resolve()
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "stage3-dataset-v4-observation":
        raise ValueError("PnP state A/B requires qualified stage3-dataset-v4-observation")
    if not bool(manifest.get("qualification_passed", False)):
        raise ValueError("PnP state A/B refuses an unqualified dataset")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training output: {output}")
    output.mkdir(parents=True)
    geometry, center_reference, center_scale, geometry_sha, normalization_sha = (
        _load_geometry_and_center_prior(dataset)
    )
    geometry_c4_asymmetry = geometry_c4_asymmetry_m(geometry)
    if geometry_c4_asymmetry <= 0.005:
        raise ValueError("geometry does not support identifiable full relative yaw")
    train_sessions, validation_sessions, selection_record = _load_session_selection(
        args.selection, manifest_path
    )
    if args.validation_on_train and selection_record is not None:
        raise ValueError("session selection cannot be combined with validation-on-train")
    train_ds = Stage3ShardDataset(
        dataset, "train", augment=not args.no_augment, seed=args.seed,
        shuffle=not args.validation_on_train, sample_limit=args.train_sample_limit,
        session_ids=train_sessions,
    )
    validation_split = "train" if args.validation_on_train else "validation"
    validation_ds = Stage3ShardDataset(
        dataset, validation_split, augment=False, seed=args.seed,
        shuffle=False, sample_limit=args.validation_sample_limit,
        session_ids=validation_sessions,
    )
    loader_options = {
        "batch_size": args.batch_size, "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, **loader_options)
    validation_loader = DataLoader(validation_ds, **loader_options)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_a = ExplicitStatePnPAdapter(
        geometry, input_features=7, channels=args.channels, dropout=args.dropout,
        center_reference=center_reference, center_scale=center_scale,
    )
    model_b = ImplicitQueryPosePredictor(
        geometry, input_features=7, channels=args.channels, dropout=args.dropout,
        center_reference=center_reference, center_scale=center_scale,
    )
    model_b.encoder.load_state_dict(model_a.encoder.state_dict())
    encoder_sha = _state_dict_sha256(model_a.encoder.state_dict())
    if encoder_sha != _state_dict_sha256(model_b.encoder.state_dict()):
        raise RuntimeError("paired encoders do not share exact initialization")
    counts = {
        LABEL_A: trainable_parameter_count(model_a),
        LABEL_B: trainable_parameter_count(model_b),
    }
    head_counts = {
        LABEL_A: trainable_parameter_count(model_a.state_head),
        LABEL_B: trainable_parameter_count(model_b.query_head),
    }
    relative_parameter_gap = abs(counts[LABEL_A] - counts[LABEL_B]) / max(counts.values())
    if relative_parameter_gap > 0.01:
        raise RuntimeError("paired trainable parameter counts differ by more than one percent")
    models: dict[str, torch.nn.Module] = {
        LABEL_A: model_a.to(device), LABEL_B: model_b.to(device),
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
    provenance: dict[str, object] = {
        "schema_version": "stage3-pnp-state-ab-run-v2",
        "dataset": str(dataset),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "dataset_schema": manifest["schema_version"],
        "geometry_template_sha256": geometry_sha,
        "normalization_sha256": normalization_sha,
        "test_accessed": False,
        "validation_split": validation_split,
        "session_selection": selection_record,
        "diagnostic_only": bool(args.validation_on_train),
        "set_policy": SET_POLICY,
        "predictor_input_allowlist": [
            "normalized PnP xyz", "PnP sin/cos yaw", "reprojection RMS",
            "valid-candidate fraction", "candidate/event masks", "real event time", "tau",
        ],
        "forbidden_predictor_inputs": [
            "future observation", "future truth", "exact center", "exact velocity",
            "exact yaw", "exact yaw rate", "motion class", "test shard",
        ],
        "objective": (
            "shared unordered-set position plus trajectory-derived center/delta/"
            "velocity/relative-yaw/omega on constant-motion physical truth"
        ),
        "physical_target_policy": {
            "future_truth_is_training_and_evaluation_label_only": True,
            "state_extraction_is_applied_equally_to_decoded_A_and_B_trajectories": True,
            "constant_motion_center_tolerance_m": 0.001,
            "constant_motion_yaw_tolerance_rad": 0.001,
            "rule_queries": 4,
            "q1_alias_guard_max_abs_omega_rad_s": 15.0,
            "minimum_geometry_c4_asymmetry_m": 0.005,
            "actual_geometry_c4_asymmetry_m": geometry_c4_asymmetry,
            "minimum_overall_and_per_class_qualification_fraction": 0.75,
            "qualification_name": "query_constant_twist_fit",
        },
        "checkpoint_selection": (
            "lexicographic max(q0..q3 absolute P95), q0 P95, "
            "max(q1..q3 centroid-motion P95), max(q0..q3 absolute median)"
        ),
        "paired_contract": {
            "shared_encoder_initial_sha256": encoder_sha,
            "same_batch_augmentation_dropout_rng": True,
            "same_optimizer_scheduler_amp_clip_budget": True,
            "trainable_parameters": counts,
            "head_parameters": head_counts,
            "relative_parameter_gap": relative_parameter_gap,
            "head_execution": {
                LABEL_A: "once per sample",
                LABEL_B: "once per query; compute and function freedom scale with Q",
            },
        },
        "config": vars(args),
        "source_sha256": _source_hashes(),
        "environment": {
            "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "amp": args.amp,
        },
        **_git_state(),
    }
    _write_json(output / "run_manifest.json", provenance)
    validation = {
        label: _validate(model, validation_loader, device, args)
        for label, model in models.items()
    }
    history: list[dict[str, object]] = [{
        "epoch": 0, "validation": validation,
        "selection_tuple": {
            label: _selection_tuple(metrics) for label, metrics in validation.items()
        },
    }]
    history_path = output / "stage3-pnp-state-ab-history.json"
    _write_json(history_path, history)
    best = {label: _selection_tuple(validation[label]) for label in models}
    best_epoch = {label: 0 for label in models}
    stale = {label: 0 for label in models}
    best_paths = {
        label: output / f"stage3-pnp-{label}-seed{args.seed}-best.pt"
        for label in models
    }
    for label, model in models.items():
        _checkpoint(
            best_paths[label], model, label, 0, validation[label], provenance,
            "untrained_initial_baseline",
        )
    started = time.monotonic()
    stop_reason = "epochs_completed"
    epochs_completed = 0
    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        train_metrics = _train_epoch_pair(
            models, train_loader, optimizers, scalers, device, args
        )
        validation = {
            label: _validate(model, validation_loader, device, args)
            for label, model in models.items()
        }
        for scheduler in schedulers.values():
            scheduler.step()
        record: dict[str, object] = {
            "epoch": epoch + 1, "train": train_metrics, "validation": validation,
            "selection_tuple": {
                label: _selection_tuple(metrics) for label, metrics in validation.items()
            },
            "lr": {label: scheduler.get_last_lr()[0] for label, scheduler in schedulers.items()},
        }
        history.append(record)
        _write_json(history_path, history)
        print(json.dumps({
            "epoch": epoch + 1,
            "train_objective": {
                label: value["objective"] for label, value in train_metrics.items()
            },
            "selection_tuple": record["selection_tuple"],
        }, sort_keys=True), flush=True)
        epochs_completed = epoch + 1
        for label, model in models.items():
            score = _selection_tuple(validation[label])
            if not all(np.isfinite(score)):
                raise FloatingPointError(f"{label} produced non-finite validation metrics")
            if score < best[label]:
                best[label] = score
                best_epoch[label] = epoch + 1
                stale[label] = 0
                _checkpoint(
                    best_paths[label], model, label, epoch + 1,
                    validation[label], provenance, "best",
                )
            else:
                stale[label] += 1
        if all(value >= args.patience for value in stale.values()):
            stop_reason = "both_models_early_stopping"
            break
        if args.max_wall_minutes > 0 and time.monotonic() - started >= args.max_wall_minutes * 60:
            stop_reason = "wall_time_limit"
            break
    for label, model in models.items():
        _checkpoint(
            output / f"stage3-pnp-{label}-seed{args.seed}-last.pt",
            model, label, epochs_completed, validation[label], provenance, "last",
            optimizers[label], schedulers[label], scalers[label],
        )
    final = {
        **provenance, "status": "complete", "stop_reason": stop_reason,
        "epochs_completed": epochs_completed,
        "best": {
            label: {
                "path": path.name, "selection_tuple": list(best[label]),
                "sha256": _sha256(path), "epoch": best_epoch[label],
                "trained_checkpoint": best_epoch[label] > 0,
            } for label, path in best_paths.items()
        },
        "qualified_training_candidate": bool(
            validation_split == "validation"
            and all(epoch > 0 for epoch in best_epoch.values())
            and stop_reason != "wall_time_limit"
        ),
        "history": history_path.name, "history_sha256": _sha256(history_path),
    }
    _write_json(output / "run_manifest.json", final)
    print(str(output), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta-m", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", choices=("off", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--validation-on-train", action="store_true")
    parser.add_argument(
        "--selection", default="",
        help="hashed train/validation session selection JSON for a dynamic pilot",
    )
    parser.add_argument("--train-sample-limit", type=int, default=0)
    parser.add_argument("--validation-sample-limit", type=int, default=0)
    parser.add_argument("--max-wall-minutes", type=float, default=0.0)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("epochs, patience, and batch-size must be positive")
    if args.validation_on_train and (
        args.train_sample_limit <= 0 or args.validation_sample_limit <= 0
    ):
        parser.error(
            "--validation-on-train is diagnostic-only and requires positive "
            "train and validation sample limits"
        )
    train(args)


if __name__ == "__main__":
    main()
