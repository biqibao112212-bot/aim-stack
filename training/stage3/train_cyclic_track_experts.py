"""Train clean-physics C4-equivariant independent armor-track experts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cyclic_track_dataset import CyclicTrackPhysicalDataset
from .cyclic_track_loss import cyclic_track_expert_loss
from .cyclic_track_model import CyclicTrackExpertSystem, ROUTE_NAMES
from .train_causal_physical_ab import (
    _git_state,
    _seed,
    _sha256,
    _state_dict_sha256,
    _to_device,
    _write_json,
)


SELECTION_QUERY_INDEX = 3


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0, "mean_m": None, "median_m": None,
            "p90_m": None, "p95_m": None, "p99_m": None, "max_m": None,
        }
    return {
        "count": int(values.size),
        "mean_m": float(values.mean()),
        "median_m": float(np.median(values)),
        "p90_m": float(np.quantile(values, 0.90)),
        "p95_m": float(np.quantile(values, 0.95)),
        "p99_m": float(np.quantile(values, 0.99)),
        "max_m": float(values.max()),
    }


def _role_masks(primary: np.ndarray, visible: np.ndarray) -> dict[str, np.ndarray]:
    batch = primary.shape[0]
    row = np.arange(batch)
    primary_mask = np.zeros((batch, 4), dtype=np.bool_)
    clockwise = np.zeros_like(primary_mask)
    counterclockwise = np.zeros_like(primary_mask)
    opposite = np.zeros_like(primary_mask)
    primary_mask[row, primary] = True
    clockwise[row, (primary + 1) % 4] = True
    counterclockwise[row, (primary - 1) % 4] = True
    opposite[row, (primary + 2) % 4] = True
    adjacent = clockwise | counterclockwise
    return {
        "all_tracks": np.ones_like(primary_mask),
        "current_visible": visible,
        "primary": primary_mask,
        "adjacent_hidden": adjacent & ~visible,
        "clockwise_hidden": clockwise & ~visible,
        "counterclockwise_hidden": counterclockwise & ~visible,
        "opposite_hidden": opposite & ~visible,
    }


def _router_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    per_class: dict[str, object] = {}
    recalls = []
    for route, name in enumerate(ROUTE_NAMES):
        support = int(confusion[route].sum())
        predicted_count = int(confusion[:, route].sum())
        true_positive = int(confusion[route, route])
        recall = true_positive / support if support else None
        precision = true_positive / predicted_count if predicted_count else None
        negatives = int(confusion.sum() - support)
        false_positive = predicted_count - true_positive
        fpr = false_positive / negatives if negatives else None
        if recall is not None:
            recalls.append(recall)
        per_class[name] = {
            "support": support, "predicted_count": predicted_count,
            "recall": recall, "precision": precision,
            "false_positive_rate": fpr,
        }
    return {
        "confusion_true_rows_predicted_columns": confusion.tolist(),
        "accuracy": float(np.mean(truth == predicted)),
        "macro_recall": float(np.mean(recalls)),
        "per_class": per_class,
    }


def _equivariance_audit(
    model: CyclicTrackExpertSystem,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    reference = model(
        batch["obs"], batch["obs_mask"], batch["primary_mask"],
        batch["event_mask"], batch["event_time_s"], batch["switch_step"],
        batch["tau"],
    )
    expert_max = router_max = hard_max = 0.0
    for shift in (1, 2, 3):
        shifted = dict(batch)
        for name in ("obs", "obs_mask", "primary_mask"):
            shifted[name] = torch.roll(batch[name], shifts=shift, dims=2)
        output = model(
            shifted["obs"], shifted["obs_mask"], shifted["primary_mask"],
            shifted["event_mask"], shifted["event_time_s"],
            shifted["switch_step"], shifted["tau"],
        )
        expert_max = max(expert_max, float((
            output["expert_position"]
            - torch.roll(reference["expert_position"], shifts=shift, dims=3)
        ).abs().max().detach().cpu()))
        hard_max = max(hard_max, float((
            output["position_mean"]
            - torch.roll(reference["position_mean"], shifts=shift, dims=2)
        ).abs().max().detach().cpu()))
        router_max = max(router_max, float((
            output["router_logit"] - reference["router_logit"]
        ).abs().max().detach().cpu()))
    return {
        "expert_position_max_abs_m": expert_max,
        "hard_position_max_abs_m": hard_max,
        "router_logit_max_abs": router_max,
    }


def _validate(
    model: CyclicTrackExpertSystem,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    model.eval()
    hard_errors = []
    hard_delta_errors = []
    oracle_errors = []
    oracle_delta_errors = []
    truths = []
    predictions = []
    primary_parts = []
    visible_parts = []
    visible_count_parts = []
    switch_parts = []
    tau_parts = []
    rule_parts = []
    rigid_parts = []
    equivariance: dict[str, float] | None = None
    amp_enabled = device.type == "cuda" and args.amp != "off"
    amp_dtype = torch.float16 if args.amp == "float16" else torch.bfloat16
    with torch.no_grad():
        for raw in loader:
            batch = _to_device(raw, device)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled,
            ):
                output = model(
                    batch["obs"], batch["obs_mask"], batch["primary_mask"],
                    batch["event_mask"], batch["event_time_s"],
                    batch["switch_step"], batch["tau"],
                )
            hard = output["position_mean"].float()
            target = batch["future_position"].float()
            labels = batch["motion_class"]
            oracle = output["expert_position"].float()[
                torch.arange(labels.shape[0], device=device), labels
            ]
            hard_errors.append(torch.linalg.vector_norm(hard - target, dim=-1).cpu().numpy())
            oracle_errors.append(torch.linalg.vector_norm(oracle - target, dim=-1).cpu().numpy())
            hard_delta_errors.append(torch.linalg.vector_norm(
                (hard - hard[:, :1]) - (target - target[:, :1]), dim=-1,
            ).cpu().numpy())
            oracle_delta_errors.append(torch.linalg.vector_norm(
                (oracle - oracle[:, :1]) - (target - target[:, :1]), dim=-1,
            ).cpu().numpy())
            pair = hard[:, :, :, None] - hard[:, :, None, :]
            pair_distance = torch.linalg.vector_norm(pair, dim=-1)
            rigid_parts.append((pair_distance - pair_distance[:, :1]).abs().cpu().numpy())
            truths.append(labels.cpu().numpy())
            predictions.append(output["route_index"].cpu().numpy())
            primary_parts.append(batch["current_primary_index"].cpu().numpy())
            visible_parts.append(batch["current_visible_mask"].cpu().numpy())
            visible_count_parts.append(batch["current_visible_count"].cpu().numpy())
            recent_switch = (batch["switch_step"][:, -args.history_events:] != 0).any(dim=1)
            switch_parts.append(recent_switch.cpu().numpy())
            tau_parts.append(batch["tau"].cpu().numpy())
            rule_parts.append(batch["rule_query"].cpu().numpy())
            if equivariance is None:
                audit_batch = {
                    name: value[: min(8, value.shape[0])]
                    for name, value in batch.items()
                    if isinstance(value, torch.Tensor)
                }
                with torch.autocast(device_type=device.type, enabled=False):
                    equivariance = _equivariance_audit(model, audit_batch)

    hard_error = np.concatenate(hard_errors)
    hard_delta = np.concatenate(hard_delta_errors)
    oracle_error = np.concatenate(oracle_errors)
    oracle_delta = np.concatenate(oracle_delta_errors)
    truth = np.concatenate(truths).astype(np.int64)
    route = np.concatenate(predictions).astype(np.int64)
    primary = np.concatenate(primary_parts).astype(np.int64)
    visible = np.concatenate(visible_parts).astype(np.bool_)
    visible_count = np.concatenate(visible_count_parts).astype(np.int64)
    recent_switch = np.concatenate(switch_parts).astype(np.bool_)
    tau = np.concatenate(tau_parts)
    rule = np.concatenate(rule_parts).astype(np.bool_, copy=False)
    rigid = np.concatenate(rigid_parts)
    roles = _role_masks(primary, visible)

    def query_summary(
        absolute: np.ndarray, delta: np.ndarray, query: int,
        sample_mask: np.ndarray,
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for role_name, role_mask in roles.items():
            selected = sample_mask[:, None] & role_mask
            result[role_name] = {
                "absolute": _summary(
                    absolute[:, query][selected & rule[:, query, None]]
                ),
                "motion_delta": _summary(
                    delta[:, query][selected & rule[:, query, None]]
                ),
            }
        return result

    def prediction_metrics(absolute: np.ndarray, delta: np.ndarray) -> dict[str, object]:
        all_samples = np.ones(truth.shape, dtype=np.bool_)
        queries = []
        for query in range(absolute.shape[1]):
            queries.append({
                "query_index": query,
                "tau_s_median": float(np.median(tau[:, query])),
                "eligible_sample_count": int(rule[:, query].sum()),
                "roles": query_summary(absolute, delta, query, all_samples),
            })
        selection_query = SELECTION_QUERY_INDEX
        by_motion = {
            name: query_summary(
                absolute, delta, selection_query, truth == route_index
            )
            for route_index, name in enumerate(ROUTE_NAMES)
        }
        by_visibility = {
            f"{count}_visible": query_summary(
                absolute, delta, selection_query, visible_count == count
            ) for count in (1, 2)
        }
        by_switch = {
            "recent_switch": query_summary(
                absolute, delta, selection_query, recent_switch
            ),
            "no_recent_switch": query_summary(
                absolute, delta, selection_query, ~recent_switch
            ),
        }
        return {
            "queries": queries,
            "selection_query": {
                "query_index": selection_query,
                "tau_s_median": float(np.median(tau[:, selection_query])),
                "eligible_sample_count": int(rule[:, selection_query].sum()),
                "by_motion": by_motion,
                "by_visibility": by_visibility,
                "by_switch": by_switch,
            },
        }

    upper = np.triu(np.ones((4, 4), dtype=np.bool_), k=1)
    return {
        "sample_count": int(truth.size),
        "hard_routed": prediction_metrics(hard_error, hard_delta),
        "oracle_correct_expert": prediction_metrics(oracle_error, oracle_delta),
        "router": _router_metrics(truth, route),
        "self_rigid_pair_distance_drift": _summary(
            rigid[..., upper][rule]
        ),
        "cyclic_equivariance": equivariance,
        "visibility": {
            "one_visible_count": int((visible_count == 1).sum()),
            "two_visible_count": int((visible_count == 2).sum()),
            "recent_switch_count": int(recent_switch.sum()),
        },
    }


def _metric_p95(role: dict[str, object]) -> float:
    value = role["absolute"]["p95_m"]  # type: ignore[index]
    return float("inf") if value is None else float(value)


def _selection_tuple(metrics: dict[str, object]) -> tuple[float, ...]:
    hard_selection = metrics["hard_routed"]["selection_query"]  # type: ignore[index]
    by_motion = hard_selection["by_motion"]  # type: ignore[index]
    rotation_combined = [
        _metric_p95(by_motion[name]["adjacent_hidden"])
        for name in ("rotation", "combined")
    ]
    all_classes = [
        _metric_p95(by_motion[name]["adjacent_hidden"])
        for name in ROUTE_NAMES
    ]
    selection_all = metrics["hard_routed"]["queries"][SELECTION_QUERY_INDEX]["roles"]  # type: ignore[index]
    q0 = metrics["hard_routed"]["queries"][0]["roles"]  # type: ignore[index]
    router = metrics["router"]  # type: ignore[index]
    return (
        max(rotation_combined), max(all_classes),
        _metric_p95(selection_all["all_tracks"]),
        _metric_p95(q0["adjacent_hidden"]),
        1.0 - float(router["macro_recall"]),
    )


def _audit_dataset(dataset: CyclicTrackPhysicalDataset, batch_size: int) -> dict[str, object]:
    class_count = np.zeros(4, dtype=np.int64)
    visible_count = np.zeros(3, dtype=np.int64)
    switch_count = 0
    eligible_query_count = None
    eligible_selection_by_class = np.zeros(4, dtype=np.int64)
    sample_count = 0
    for batch in DataLoader(dataset, batch_size=batch_size, num_workers=0):
        labels = batch["motion_class"].numpy()
        for route in range(4):
            class_count[route] += int((labels == route).sum())
        counts = batch["obs_mask"].sum(dim=-1)
        active = batch["event_mask"]
        if bool(torch.any(counts[active] < 1)) or bool(torch.any(counts[active] > 2)):
            raise ValueError("dataset visibility audit failed")
        if not torch.equal(active, batch["obs_mask"].any(dim=-1)):
            raise ValueError("event_mask must equal obs_mask.any(-1)")
        for count in (1, 2):
            visible_count[count] += int((counts[active] == count).sum())
        switch_count += int((batch["switch_step"] != 0).sum())
        query_count = batch["rule_query"].sum(dim=0).numpy().astype(np.int64)
        eligible_query_count = (
            query_count if eligible_query_count is None
            else eligible_query_count + query_count
        )
        selection_rule = batch["rule_query"][:, SELECTION_QUERY_INDEX].numpy()
        for route in range(4):
            eligible_selection_by_class[route] += int(
                ((labels == route) & selection_rule).sum()
            )
        sample_count += int(labels.size)
    if np.any(class_count == 0):
        raise ValueError(f"all four motion classes are required: {class_count.tolist()}")
    if visible_count[1] == 0 or visible_count[2] == 0:
        raise ValueError("both one-visible and two-visible events are required")
    if eligible_query_count is None or np.any(eligible_query_count == 0):
        raise ValueError("every query requires eligible constant-motion supervision")
    if np.any(eligible_selection_by_class == 0):
        raise ValueError(
            "selection query requires every motion class: "
            f"{eligible_selection_by_class.tolist()}"
        )
    return {
        "sample_count": sample_count,
        "motion_class_count": dict(zip(ROUTE_NAMES, class_count.tolist())),
        "one_visible_event_count": int(visible_count[1]),
        "two_visible_event_count": int(visible_count[2]),
        "switch_event_count": switch_count,
        "eligible_query_count": eligible_query_count.tolist(),
        "eligible_selection_query_by_motion_class": dict(zip(
            ROUTE_NAMES, eligible_selection_by_class.tolist()
        )),
        "virtual_contract": dataset.virtual_contract,
    }


def _train_epoch(
    model: CyclicTrackExpertSystem,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    amp_enabled = device.type == "cuda" and args.amp != "off"
    amp_dtype = torch.float16 if args.amp == "float16" else torch.bfloat16
    for raw in loader:
        batch = _to_device(raw, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled,
        ):
            output = model(
                batch["obs"], batch["obs_mask"], batch["primary_mask"],
                batch["event_mask"], batch["event_time_s"],
                batch["switch_step"], batch["tau"],
            )
            total, parts = cyclic_track_expert_loss(
                output, batch["future_position"], batch["motion_class"],
                batch["rule_query"],
                batch["current_primary_index"], batch["current_visible_mask"],
                huber_beta_m=args.huber_beta_m,
                motion_delta_weight=args.motion_delta_weight,
                rigid_weight=args.rigid_weight,
                router_weight=args.router_weight,
                opposite_weight=args.opposite_weight,
            )
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite cyclic-track objective")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf")
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite cyclic-track gradient")
        scaler.step(optimizer)
        scaler.update()
        batch_count = int(batch["obs"].shape[0])
        values = {
            "objective": float(total.detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            **{name: float(value.detach().cpu()) for name, value in parts.items()},
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise FloatingPointError("non-finite cyclic-track train metric")
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + batch_count * value
        count += batch_count
    return {name: value / max(count, 1) for name, value in totals.items()}


def _checkpoint(
    path: Path, model: CyclicTrackExpertSystem, epoch: int,
    metrics: dict[str, object], provenance: dict[str, object], role: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    payload: dict[str, object] = {
        "model": model.state_dict(), "model_class": model.__class__.__name__,
        "model_config": model.config(), "label": "cyclic_track_expert_system",
        "epoch": epoch, "checkpoint_role": role, "validation": metrics,
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
        raise ValueError("official cyclic-track training requires a clean worktree")
    dataset_path = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    resume = bool(args.resume)
    existing_manifest: dict[str, object] | None = None
    existing_history: list[dict[str, object]] | None = None
    resume_payload: dict[str, object] | None = None
    resume_epoch = 0
    if output.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite cyclic-track output: {output}")
    if resume:
        if not output.exists():
            raise FileNotFoundError(f"resume output does not exist: {output}")
        manifest_path = output / "run_manifest.json"
        history_path = output / "stage3-cyclic-track-experts-history.json"
        if not manifest_path.exists() or not history_path.exists():
            raise ValueError("resume output is missing manifest or history")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_history = json.loads(history_path.read_text(encoding="utf-8"))
        if existing_manifest.get("schema_version") != "stage3-cyclic-track-experts-run-v1":
            raise ValueError("resume manifest schema mismatch")
        if existing_manifest.get("status") != "training":
            raise ValueError("only an interrupted training run may be resumed")
        if bool(existing_manifest.get("test_accessed", True)):
            raise ValueError("refusing to resume a test-accessed run")
        if not existing_history:
            raise ValueError("resume history is empty")
        resume_path = (
            Path(args.resume_checkpoint)
            if args.resume_checkpoint else
            output / f"stage3-cyclic-track-experts-seed{args.seed}-best.pt"
        )
        if not resume_path.is_absolute():
            resume_path = output / resume_path
        resume_path = resume_path.resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        resume_payload = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        resume_epoch = int(resume_payload.get("epoch", -1))
        if resume_epoch < 1 or resume_epoch >= args.epochs:
            raise ValueError(
                f"resume checkpoint epoch {resume_epoch} is outside [1,{args.epochs - 1}]"
            )
        if not isinstance(resume_payload.get("validation"), dict):
            raise ValueError("resume checkpoint is missing validation metrics")
        if not any(int(item.get("epoch", -1)) == resume_epoch for item in existing_history):
            raise ValueError("resume checkpoint epoch is absent from history")
        backup_path = output / (
            "stage3-cyclic-track-experts-history-"
            f"pre-resume-epoch{existing_history[-1].get('epoch', 'unknown')}.json"
        )
        if backup_path.exists():
            raise FileExistsError(f"resume backup already exists: {backup_path}")
        shutil.copy2(history_path, backup_path)
    train_ds = CyclicTrackPhysicalDataset(
        dataset_path, "train", seed=args.seed, shuffle=True,
        sample_limit=args.train_sample_limit,
        secondary_gap_ratio=args.secondary_gap_ratio,
        augment_cyclic_origin=True, augment_direction=True,
    )
    validation_ds = CyclicTrackPhysicalDataset(
        dataset_path, "validation", seed=args.seed,
        shuffle=args.validation_sample_limit > 0,
        sample_limit=args.validation_sample_limit,
        secondary_gap_ratio=args.secondary_gap_ratio,
    )
    qualification = {
        "train": _audit_dataset(train_ds, args.audit_batch_size),
        "validation": _audit_dataset(validation_ds, args.audit_batch_size),
    }
    train_ds.set_epoch(0)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CyclicTrackExpertSystem(
        torch.from_numpy(train_ds.mean), torch.from_numpy(train_ds.std),
        channels=args.channels, dropout=args.dropout,
        history_events=args.history_events,
    ).to(device)
    initial_state_sha256 = _state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(1, args.epochs), eta_min=args.lr * 0.02
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.amp == "float16"
    )
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model"], strict=True)
        if "optimizer" not in resume_payload or "scheduler" not in resume_payload:
            raise ValueError("resume checkpoint is missing optimizer/scheduler state")
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        if "scaler" in resume_payload:
            scaler.load_state_dict(resume_payload["scaler"])
    loader_options = {
        "batch_size": args.batch_size, "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, **loader_options)
    validation_loader = DataLoader(validation_ds, **loader_options)
    source_root = Path(__file__).resolve().parent
    source_names = (
        "train_cyclic_track_experts.py", "cyclic_track_dataset.py",
        "cyclic_track_model.py", "cyclic_track_loss.py", "model.py",
    )
    provenance: dict[str, object] = {
        "schema_version": "stage3-cyclic-track-experts-run-v1",
        "status": "training", "dataset": str(dataset_path),
        "dataset_manifest_sha256": train_ds.manifest_sha256,
        "dataset_qualification": qualification, "test_accessed": False,
        "input_allowlist": [
            "masked normalized physical xyz", "visibility mask",
            "primary mask", "event mask", "real causal event time",
            "tracker switch step", "query tau",
        ],
        "forbidden_predictor_inputs": [
            "future truth", "motion class", "PnP", "slot id feature",
            "fixed radius", "fixed height", "center", "phase", "geometry template",
        ],
        "architecture_contract": {
            "track_labels": "temporary cyclic state handles only",
            "equivariance": "C4 roll-equivariant raw experts and hard output",
            "router": "independent C4-invariant four-class router",
            "experts": list(ROUTE_NAMES),
            "combined_is_independent": True,
            "fixed_geometry_decoder": False,
        },
        "objective_contract": {
            "formula": (
                "balanced matching-expert direct-position Huber + "
                f"{args.motion_delta_weight}*motion-delta Huber + "
                f"{args.rigid_weight}*self-q0-pair-distance drift Huber + "
                f"{args.router_weight}*balanced four-class CE"
            ),
            "cyclic_shift_minimum_search": False,
            "opposite_track_weight": args.opposite_weight,
            "future_truth_role": "loss and validation only",
        },
        "selection_contract": (
            "query index 3 (nominal 0.5 s): worst rotation/combined adjacent-hidden "
            "P95, worst all-class adjacent-hidden P95, all-track P95; adjacent-hidden q0 "
            "P95, router macro-recall penalty"
        ),
        "config": vars(args),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial_state_sha256": initial_state_sha256,
        "source_sha256": {name: _sha256(source_root / name) for name in source_names},
        "environment": {
            "python": sys.version, "numpy": np.__version__,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "device": str(device), "amp": args.amp,
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
        },
        **git_state,
    }
    if existing_manifest is not None:
        provenance["resumed_from"] = {
            "manifest_git_commit": existing_manifest.get("git_commit"),
            "manifest_sha256": _sha256(output / "run_manifest.json"),
            "checkpoint": resume_path.name if resume_payload is not None else None,
            "checkpoint_epoch": resume_epoch,
            "history_backup": backup_path.name,
        }
        provenance["initial_state_sha256"] = existing_manifest.get(
            "initial_state_sha256", initial_state_sha256
        )
    _write_json(output / "run_manifest.json", provenance)

    if resume_payload is not None:
        validation = resume_payload["validation"]
        assert existing_history is not None
        history = [
            item for item in existing_history
            if int(item.get("epoch", -1)) <= resume_epoch
        ]
        _write_json(
            output / "stage3-cyclic-track-experts-history.json", history
        )
    else:
        validation = _validate(model, validation_loader, device, args)
        history = [{
            "epoch": 0, "validation": validation,
            "selection_tuple": _selection_tuple(validation),
        }]
    history_path = output / "stage3-cyclic-track-experts-history.json"
    _write_json(history_path, history)
    initial_path = output / f"stage3-cyclic-track-experts-seed{args.seed}-initial.pt"
    if not resume:
        _checkpoint(initial_path, model, 0, validation, provenance, "initial")
    selection_history = [
        item for item in history if item.get("selection_tuple") is not None
    ]
    best_record = min(
        selection_history,
        key=lambda item: tuple(item["selection_tuple"]),
    )
    best = tuple(best_record["selection_tuple"])
    best_epoch = int(best_record["epoch"])
    best_path = output / f"stage3-cyclic-track-experts-seed{args.seed}-best.pt"
    if not resume:
        _checkpoint(best_path, model, 0, validation, provenance, "best")
    milestones = {20, 50, 100, 150, 200, 250, args.epochs}
    started = time.monotonic()
    epochs_completed = resume_epoch if resume else 0
    stop_reason = "epoch_limit"
    for epoch in range((resume_epoch + 1) if resume else 1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        train_metrics = _train_epoch(
            model, train_loader, optimizer, scaler, device, args
        )
        scheduler.step()
        validate_now = (
            epoch == 1 or epoch % args.validation_interval == 0
            or epoch == args.epochs or epoch in milestones
        )
        record: dict[str, object] = {
            "epoch": epoch, "train": train_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if validate_now:
            validation = _validate(model, validation_loader, device, args)
            selection = _selection_tuple(validation)
            record["validation"] = validation
            record["selection_tuple"] = selection
            if selection < best:
                best = selection
                best_epoch = epoch
                _checkpoint(
                    best_path, model, epoch, validation, provenance, "best",
                    optimizer, scheduler, scaler,
                )
        history.append(record)
        _write_json(history_path, history)
        concise = {
            "epoch": epoch, "objective": train_metrics["objective"],
            "router_loss": train_metrics["router"],
            "validated": validate_now,
        }
        if validate_now:
            concise.update({
                "selection_tuple": record["selection_tuple"],
                "router_macro_recall": validation["router"]["macro_recall"],
            })
        print(json.dumps(concise, sort_keys=True), flush=True)
        if epoch in milestones:
            _checkpoint(
                output / f"stage3-cyclic-track-experts-seed{args.seed}-epoch{epoch:03d}.pt",
                model, epoch, validation, provenance, "milestone",
                optimizer, scheduler, scaler,
            )
        epochs_completed = epoch
        if args.max_wall_minutes > 0 and (
            time.monotonic() - started >= args.max_wall_minutes * 60
        ):
            stop_reason = "wall_time_limit"
            break
    last_path = output / f"stage3-cyclic-track-experts-seed{args.seed}-last.pt"
    _checkpoint(
        last_path, model, epochs_completed, validation, provenance, "last",
        optimizer, scheduler, scaler,
    )
    final = {
        **provenance, "status": "complete", "stop_reason": stop_reason,
        "epochs_completed": epochs_completed,
        "best": {
            "path": best_path.name, "sha256": _sha256(best_path),
            "epoch": best_epoch, "selection_tuple": best,
            "trained_checkpoint": best_epoch > 0,
        },
        "last": {"path": last_path.name, "sha256": _sha256(last_path)},
    }
    _write_json(output / "run_manifest.json", final)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--audit-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--history-events", type=int, default=32)
    parser.add_argument("--secondary-gap-ratio", type=float, default=0.25)
    parser.add_argument("--huber-beta-m", type=float, default=0.01)
    parser.add_argument("--motion-delta-weight", type=float, default=0.5)
    parser.add_argument("--rigid-weight", type=float, default=0.05)
    parser.add_argument("--router-weight", type=float, default=0.1)
    parser.add_argument("--opposite-weight", type=float, default=0.25)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--amp", choices=("off", "bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--train-sample-limit", type=int, default=0)
    parser.add_argument("--validation-sample-limit", type=int, default=0)
    parser.add_argument("--max-wall-minutes", type=float, default=0.0)
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", default="")
    args = parser.parse_args()
    positive = (
        args.epochs, args.batch_size, args.audit_batch_size, args.lr,
        args.channels, args.huber_beta_m, args.motion_delta_weight,
        args.rigid_weight, args.router_weight, args.opposite_weight,
        args.validation_interval, args.grad_clip,
    )
    if any(value <= 0 for value in positive):
        parser.error("cyclic-track training arguments must be positive")
    if args.channels % 4:
        parser.error("channels must be divisible by four")
    if not 8 <= args.history_events <= 200:
        parser.error("history-events must be within [8,200]")
    if not 0 <= args.secondary_gap_ratio <= 1:
        parser.error("secondary-gap-ratio must be within [0,1]")
    if args.opposite_weight > 1 or args.weight_decay < 0:
        parser.error("invalid loss/optimizer weight")
    if args.train_sample_limit < 0 or args.validation_sample_limit < 0:
        parser.error("sample limits cannot be negative")
    print(train(args))


if __name__ == "__main__":
    main()
