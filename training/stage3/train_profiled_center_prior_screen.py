"""Fixed, resumable V14-B0 screen for the anonymous q0 center prior.

Only the center prior is learned.  Frozen Mapper/S/H produce the eight-field
state input, physical center and twist are loss-only labels, truth omega is used
only to isolate the center/velocity mechanism, and no future module is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate

from .center_offset_supervision import (
    CENTER_OFFSET_TARGET_FIELD,
    CenterTruthIndex,
    attach_center_offset_after_frozen_upstream,
)
from .cyclic_future_foundation import load_frozen_v19
from .motion_truth_supervision import MOTION_TARGET_FIELD, MotionTruthIndex
from .observable_future_pnp_ab import sha256_file, state_dict_sha256
from .pnp_q0_hypothesis_adapter import (
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .profiled_center_twist_future import (
    CENTER_TWIST_FORWARD_FIELDS,
    CenterPriorProfiledTwistScreen,
    anonymous_center_prior_loss,
)
from .train_anonymous_vehicle_motion import (
    _dataset, _json_sha256, _restore_rng_state, _rng_state,
    _set_phase_lr, _validate_bindings, frozen_upstream_batch,
)
from .train_causal_physical_ab import _git_state, _seed, _to_device
from .train_increment_invariant_anonymous_future import HierarchicalSessionHistorySampler
from .train_stable_motion_bottleneck_future import motion_state_cells
from .train_pnp_window_mapper_distillation import _atomic_checkpoint, _atomic_json


RUN_SCHEMA = "stage3-profiled-center-prior-truth-omega-screen-v14-b0"
FIXED_UPDATES = 100
CHECKPOINT_INTERVAL = 25


def _distribution(values: list[np.ndarray]) -> dict[str, float | int]:
    data = np.concatenate(values).astype(np.float64, copy=False)
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError("center screen distribution is empty or non-finite")
    return {
        "count": int(data.size), "mean": float(data.mean()),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "coverage_le_0_75": float(np.mean(data <= 0.75)),
    }


def _assert_manifest_shards_unchanged(
    root: Path, manifest: dict[str, Any], *, label: str,
) -> None:
    for item in manifest.get("shards", []):
        path = root / Path(str(item["path"]).replace("\\", "/"))
        if sha256_file(path) != str(item["sha256"]):
            raise RuntimeError(f"{label} shard changed during training: {path}")


def _global_half_cycle_derangement(
    sample_count: int, device: torch.device,
) -> tuple[torch.Tensor, str]:
    if sample_count < 2:
        raise ValueError("counterfactual validation requires at least two windows")
    shift = sample_count // 2
    index = torch.remainder(
        torch.arange(sample_count, device=device) + shift, sample_count,
    )
    if bool(torch.any(index == torch.arange(sample_count, device=device))):
        raise RuntimeError("counterfactual q0 permutation has a fixed point")
    digest = hashlib.sha256(
        index.cpu().numpy().astype("<i8", copy=False).tobytes()
    ).hexdigest()
    return index, digest


def _prepare(
    mapper: torch.nn.Module, s_model: torch.nn.Module,
    h_model: torch.nn.Module, raw: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    batch = frozen_upstream_batch(mapper, s_model, h_model, raw)
    attach_center_offset_after_frozen_upstream(batch, raw)
    if MOTION_TARGET_FIELD not in raw:
        raise ValueError("motion target missing after exact truth join")
    batch[MOTION_TARGET_FIELD] = raw[MOTION_TARGET_FIELD].detach()
    return batch


@torch.inference_mode()
def evaluate_loaded(
    model: CenterPriorProfiledTwistScreen, loader: DataLoader,
    mapper: torch.nn.Module, s_model: torch.nn.Module,
    h_model: torch.nn.Module, device: torch.device,
) -> dict[str, Any]:
    model.eval()
    groups = ("overall", "rotation", "combined")
    cached: dict[str, list[torch.Tensor]] = {
        name: [] for name in (
            *CENTER_TWIST_FORWARD_FIELDS, MOTION_TARGET_FIELD,
            CENTER_OFFSET_TARGET_FIELD, "motion_class",
        )
    }
    for raw_cpu in loader:
        raw = _to_device(raw_cpu, device)
        batch = _prepare(mapper, s_model, h_model, raw)
        for name in cached:
            cached[name].append(batch[name])
    full = {name: torch.cat(values, dim=0) for name, values in cached.items()}
    sample_count = full[MOTION_TARGET_FIELD].shape[0]
    shift = sample_count // 2
    donor_index, donor_sha256 = _global_half_cycle_derangement(
        sample_count, device,
    )

    center = {group: [] for group in groups}
    velocity = {group: [] for group in groups}
    yaw = {group: [] for group in groups}
    solver = {
        branch: {
            group: {"profile": 0, "fallback": 0, "state": 0, "count": 0}
            for group in groups
        }
        for branch in ("intact", "history_blind", "q0_shuffled")
    }
    support_histogram = {str(count): 0 for count in range(5)}
    counterfactual = {
        branch: {group: [] for group in groups}
        for branch in ("intact", "history_blind", "q0_shuffled")
    }
    component_weight = {
        branch: [] for branch in ("intact", "q0_shuffled")
    }
    component_energy_gap = {
        branch: [] for branch in ("intact", "q0_shuffled")
    }
    evaluation_batch_size = int(loader.batch_size or sample_count)
    for start in range(0, sample_count, evaluation_batch_size):
        stop = min(start + evaluation_batch_size, sample_count)
        batch = {name: value[start:stop] for name, value in full.items()}
        target = batch[MOTION_TARGET_FIELD]
        state_input = {name: batch[name] for name in CENTER_TWIST_FORWARD_FIELDS}
        prediction = model.forward_at_omega(state_input, target[:, 3])
        blind = model.forward_at_omega(
            state_input, target[:, 3], use_q0_prior=False,
        )
        shuffled_input = dict(state_input)
        donor = donor_index[start:stop]
        shuffled_input["q0_relation_m"] = full["q0_relation_m"][donor]
        shuffled_input["q0_supported"] = full["q0_supported"][donor]
        shuffled = model.forward_at_omega(shuffled_input, target[:, 3])
        center_error = torch.linalg.vector_norm(
            prediction["center_offset_m"] - batch[CENTER_OFFSET_TARGET_FIELD], dim=-1,
        )
        velocity_error = torch.linalg.vector_norm(
            prediction["velocity_mps"] - target[:, :3], dim=-1,
        )
        yaw_error = (prediction["yaw_rate_rad_s"] - target[:, 3]).abs()
        masks = {
            "overall": torch.ones_like(target[:, 3], dtype=torch.bool),
            "rotation": batch["motion_class"] == 2,
            "combined": batch["motion_class"] == 3,
        }
        for name, mask in masks.items():
            center[name].append(center_error[mask].cpu().numpy())
            velocity[name].append(velocity_error[mask].cpu().numpy())
            yaw[name].append(yaw_error[mask].cpu().numpy())
            for branch, branch_prediction in (
                ("intact", prediction), ("history_blind", blind),
                ("q0_shuffled", shuffled),
            ):
                count = int(mask.sum())
                coverage = solver[branch][name]
                coverage["count"] += count
                coverage["profile"] += int(
                    branch_prediction["profile_supported"][mask].sum()
                )
                coverage["fallback"] += int((
                    ~branch_prediction["profile_supported"]
                    & branch_prediction["fallback_supported"]
                )[mask].sum())
                coverage["state"] += int(
                    branch_prediction["state_supported"][mask].sum()
                )
                branch_error = torch.linalg.vector_norm(
                    branch_prediction["velocity_mps"] - target[:, :3], dim=-1,
                )
                common_supported = (
                    prediction["state_supported"] & blind["state_supported"]
                    & shuffled["state_supported"]
                )
                counterfactual[branch][name].append(
                    branch_error[mask & common_supported].cpu().numpy()
                )
        component_weight["intact"].append(
            prediction["q0_component_weight"].cpu().numpy()
        )
        component_weight["q0_shuffled"].append(
            shuffled["q0_component_weight"].cpu().numpy()
        )
        intact_energy_valid = (
            prediction["q0_component_state_supported"]
            & prediction["history_component_state_supported"]
        )
        shuffled_energy_valid = (
            shuffled["q0_component_state_supported"]
            & shuffled["history_component_state_supported"]
        )
        component_energy_gap["intact"].append((
            prediction["q0_profile_energy"]
            - prediction["history_profile_energy"]
        )[intact_energy_valid].cpu().numpy())
        component_energy_gap["q0_shuffled"].append((
            shuffled["q0_profile_energy"]
            - shuffled["history_profile_energy"]
        )[shuffled_energy_valid].cpu().numpy())
        for count in range(5):
            support_histogram[str(count)] += int(
                (prediction["center_support_count"] == count).sum()
            )
    return {
        "center_error_m": {name: _distribution(values) for name, values in center.items()},
        "truth_omega_velocity_error_mps": {
            name: _distribution(values) for name, values in velocity.items()
        },
        "truth_omega_reproduction_error_rad_s": {
            name: _distribution(values) for name, values in yaw.items()
        },
        "support_histogram": support_histogram,
        "solver_coverage": {
            branch: {
                name: {
                    "count": values["count"],
                    "profile_fraction": values["profile"] / values["count"],
                    "fallback_fraction": values["fallback"] / values["count"],
                    "state_supported_fraction": values["state"] / values["count"],
                }
                for name, values in branch_groups.items()
            }
            for branch, branch_groups in solver.items()
        },
        "h_counterfactual_velocity_error_mps": {
            branch: {
                name: _distribution(values) for name, values in branch_groups.items()
            }
            for branch, branch_groups in counterfactual.items()
        },
        "q0_component_weight": {
            branch: _distribution(values)
            for branch, values in component_weight.items()
        },
        "q0_minus_history_profile_energy": {
            branch: _distribution(values)
            for branch, values in component_energy_gap.items()
        },
        "q0_shuffled_permutation": {
            "schema": "global-half-cycle-derangement-v1",
            "sample_count": sample_count,
            "shift": shift,
            "index_sha256": donor_sha256,
            "fixed_point_count": 0,
            "independent_of_validation_batch_size": True,
        },
    }


def _contract(args: argparse.Namespace, provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA,
        "args": {
            key: value for key, value in vars(args).items()
            if key not in {"resume_checkpoint", "stop_after_update"}
        },
        "fixed_updates": FIXED_UPDATES,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "forward_fields": CENTER_TWIST_FORWARD_FIELDS,
        "forward_fields_sha256": hashlib.sha256(
            "\n".join(CENTER_TWIST_FORWARD_FIELDS).encode("utf-8")
        ).hexdigest(),
        "truth_center_forward_input": False,
        "truth_velocity_forward_input": False,
        "truth_yaw_rate_forward_input": True,
        "truth_yaw_rate_forward_semantics": (
            "loss-only oracle mechanism slice; forbidden in free-omega inference"
        ),
        "truth_omega_mechanism_slice": True,
        "physical_id_input": False,
        "q0_confidence_input": False,
        "q0_geometry_carrier": "all four finite anonymous H hypotheses",
        "q0_support_semantics": "calibration and all-unsupported gate; no role deletion",
        "future_modules_loaded": False,
        "profile_mechanism": {
            "center_precision": args.center_precision,
            "history_center_precision": 0.01,
            "q0_endpoint_precision": 10.0,
            "q_ridge": 1e-5,
            "minimum_velocity_information_s2": 1e-4,
            "maximum_velocity_condition": 1e6,
            "minimum_time_span_s": 1e-3,
            "component_selector": (
                "learned invariant gate over support, center uncertainty, "
                "XY/Z profile energies and Schur information"
            ),
            "component_gate_truth_supervision": (
                "soft responsibility from q0-vs-history truth-velocity error"
            ),
        },
        "dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
        "truth_manifest_sha256": provenance["truth_manifest_sha256"],
        "motion_label_sha256": provenance["motion_truth"]["label_sha256"],
        "center_label_sha256": provenance["center_truth"]["label_sha256"],
        "source_sha256": provenance["source_sha256"],
        "frozen_checkpoint_sha256": provenance["frozen_checkpoint_sha256"],
        "frozen_state_dict_sha256": provenance["frozen_initial_state_dict_sha256"],
        "frozen_binding": provenance["binding"],
        "git": provenance["git"],
    }


def train(args: argparse.Namespace) -> Path:
    if args.updates != FIXED_UPDATES:
        raise ValueError(f"V14-B0 requires exactly {FIXED_UPDATES} updates")
    if (
        args.stop_after_update not in {0, *range(
            CHECKPOINT_INTERVAL, FIXED_UPDATES, CHECKPOINT_INTERVAL,
        )}
    ):
        raise ValueError("stop-after-update must be zero or a nonfinal checkpoint")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V14-B0 requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    git_state = _git_state()
    if git_state.get("git_commit") == "unknown" or bool(git_state.get("worktree_dirty")):
        raise RuntimeError("formal V14-B0 requires a known clean git commit")
    output = Path(args.output).resolve()
    resume = Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None
    if resume is None:
        if output.exists():
            raise FileExistsError(f"refusing existing output: {output}")
        output.mkdir(parents=True)
    elif not output.is_dir() or not resume.is_file():
        raise ValueError("resume requires an existing run and checkpoint")
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    _seed(args.seed)
    dataset_path = Path(args.dataset).resolve()
    manifest_path = dataset_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("test_accessed", True)):
        raise ValueError("V14-B0 refuses a dataset that accessed test")
    expected_truth_sha = manifest.get("truth_history_manifest_sha256")
    if not isinstance(expected_truth_sha, str):
        raise ValueError("paired dataset lacks truth-history binding")
    train_dataset = _dataset(dataset_path, "train", sample_limit=0)
    validation_dataset = _dataset(dataset_path, "validation", sample_limit=0)
    motion_truth = MotionTruthIndex(
        args.truth_history, expected_manifest_sha256=expected_truth_sha,
    )
    center_truth = CenterTruthIndex(
        args.truth_history, expected_manifest_sha256=expected_truth_sha,
    )
    motion_joins = {
        "train": motion_truth.attach(train_dataset, "train"),
        "validation": motion_truth.attach(validation_dataset, "validation"),
    }
    center_joins = {
        "train": center_truth.attach(train_dataset, "train"),
        "validation": center_truth.attach(validation_dataset, "validation"),
    }
    for split in ("train", "validation"):
        if motion_joins[split]["joined_key_set_sha256"] != center_joins[split]["joined_key_set_sha256"]:
            raise ValueError(f"motion and center joins differ for {split}")

    mapper, mapper_info = load_frozen_pnp_mapper(args.mapper_checkpoint)
    s_model, s_info = load_frozen_v19(args.s_checkpoint)
    h_model, h_info = load_frozen_hypothesis_adapter(
        args.h_checkpoint, allow_diagnostic=True,
    )
    binding = _validate_bindings(
        sha256_file(manifest_path), mapper_info, s_info, h_info,
    )
    for frozen in (mapper, s_model, h_model):
        frozen.to(device).eval().requires_grad_(False)
    frozen_initial = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    }
    frozen_checkpoint_sha256 = {
        "mapper": sha256_file(args.mapper_checkpoint),
        "s": sha256_file(args.s_checkpoint),
        "h": sha256_file(args.h_checkpoint),
    }

    model = CenterPriorProfiledTwistScreen(
        width=args.width, center_precision=args.center_precision,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    sampler = HierarchicalSessionHistorySampler(
        motion_state_cells(train_dataset), seed=args.seed + 1,
    )
    prefix_generator = torch.Generator().manual_seed(args.seed + 2)
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.validation_batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )
    source_paths = {
        "runner": Path(__file__).resolve(),
        "model": Path(__file__).with_name("profiled_center_twist_future.py").resolve(),
        "center_truth": Path(__file__).with_name("center_offset_supervision.py").resolve(),
        "motion_truth": Path(__file__).with_name("motion_truth_supervision.py").resolve(),
        "frozen_upstream_batch": Path(__file__).with_name("train_anonymous_vehicle_motion.py").resolve(),
        "sampler": Path(__file__).with_name("train_increment_invariant_anonymous_future.py").resolve(),
        "sampler_cells": Path(__file__).with_name("train_stable_motion_bottleneck_future.py").resolve(),
        "paired_dataset": Path(__file__).with_name("observable_future_pnp_ab.py").resolve(),
        "pnp_h_loader": Path(__file__).with_name("pnp_q0_hypothesis_adapter.py").resolve(),
        "s_loader": Path(__file__).with_name("cyclic_future_foundation.py").resolve(),
        "runtime_helpers": Path(__file__).with_name("train_causal_physical_ab.py").resolve(),
    }
    provenance = {
        "diagnostic_only": True, "test_accessed": False,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "truth_manifest_sha256": expected_truth_sha,
        "motion_truth": motion_truth.provenance,
        "center_truth": center_truth.provenance,
        "motion_joins": motion_joins, "center_joins": center_joins,
        "binding": binding,
        "frozen_initial_state_dict_sha256": frozen_initial,
        "frozen_checkpoint_sha256": frozen_checkpoint_sha256,
        "source_paths": {key: str(value) for key, value in source_paths.items()},
        "source_sha256": {key: sha256_file(value) for key, value in source_paths.items()},
        "git": git_state,
        "runtime": {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
    }
    contract = _contract(args, provenance)
    contract_sha256 = _json_sha256(contract)
    validation_history: list[dict[str, Any]] = []
    checkpoint_ledger: list[dict[str, Any]] = []
    global_update = 0
    if resume is not None:
        if resume.parent.resolve() != checkpoint_dir.resolve():
            raise ValueError("resume checkpoint must belong to this run/checkpoints")
        existing_manifest_path = output / "run_manifest.json"
        if not existing_manifest_path.is_file():
            raise ValueError("resume run manifest is missing")
        existing_manifest = json.loads(
            existing_manifest_path.read_text(encoding="utf-8")
        )
        if existing_manifest.get("schema_version") != RUN_SCHEMA:
            raise ValueError("resume run manifest schema differs")
        if existing_manifest.get("status") != "running":
            raise ValueError("only an interrupted running V14-B0 run may resume")
        if existing_manifest.get("contract_sha256") != contract_sha256:
            raise ValueError("resume run manifest contract differs")
        checkpoint_ledger = list(existing_manifest.get("checkpoint_ledger", []))
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != RUN_SCHEMA:
            raise ValueError("resume schema differs")
        if payload.get("contract_sha256") != contract_sha256:
            raise ValueError("resume contract differs")
        global_update = int(payload["global_update"])
        if not 0 <= global_update <= FIXED_UPDATES:
            raise ValueError("resume update is outside the fixed schedule")
        expected_name = f"checkpoint-update-{global_update:06d}.pt"
        if resume.name != expected_name:
            raise ValueError("resume checkpoint filename/update differ")
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-update-*.pt"))
        if not checkpoints or checkpoints[-1].resolve() != resume.resolve():
            raise ValueError("resume must use the latest checkpoint")
        checkpoint_sha = sha256_file(resume)
        manifest_update = int(existing_manifest.get("global_update", -1))
        declared = [
            item for item in checkpoint_ledger
            if int(item.get("global_update", -1)) == global_update
        ]
        if manifest_update == global_update:
            if len(declared) != 1 or declared[0].get("sha256") != checkpoint_sha:
                raise ValueError("resume checkpoint ledger/hash differs")
            if payload.get("validation_history") != existing_manifest.get(
                "validation_history"
            ):
                raise ValueError("resume checkpoint/manifest validation differs")
        elif (
            global_update == manifest_update + CHECKPOINT_INTERVAL
            and global_update % CHECKPOINT_INTERVAL == 0
            and not declared
            and payload.get("validation_history") == existing_manifest.get(
                "validation_history"
            )
        ):
            # Atomic checkpoint creation can win the race against the manifest
            # update.  Adopt exactly one next-boundary orphan after fully
            # validating its contract, frozen inputs and payload below.
            checkpoint_ledger.append({
                "global_update": global_update, "path": str(resume),
                "sha256": checkpoint_sha, "kind": "training_boundary",
                "adopted_after_manifest_crash": True,
            })
        else:
            raise ValueError("resume manifest/checkpoint progress differs")
        if payload.get("frozen_state_dict_sha256") != frozen_initial:
            raise ValueError("resume frozen upstream state differs")
        if payload.get("frozen_checkpoint_sha256") != frozen_checkpoint_sha256:
            raise ValueError("resume frozen upstream checkpoints differ")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        validation_history = list(payload["validation_history"])
        _restore_rng_state(payload["rng"], sampler, prefix_generator)
    else:
        validation_history.append({
            "global_update": 0,
            "metrics": evaluate_loaded(
                model, validation_loader, mapper, s_model, h_model, device,
            ),
        })

    manifest_payload = {
        "schema_version": RUN_SCHEMA, "status": "running",
        "contract": contract, "contract_sha256": contract_sha256,
        "provenance": provenance, "global_update": global_update,
        "validation_history": validation_history,
        "checkpoint_ledger": checkpoint_ledger,
    }
    _atomic_json(output / "run_manifest.json", manifest_payload)
    started = time.time()
    model.train()
    while global_update < FIXED_UPDATES:
        next_update = global_update + 1
        indices = sampler.draw(args.batch_size)
        raw_cpu = default_collate([train_dataset[index] for index in indices])
        raw = _to_device(raw_cpu, device)
        batch = _prepare(mapper, s_model, h_model, raw)
        target_state = batch[MOTION_TARGET_FIELD]
        state_input = {name: batch[name] for name in CENTER_TWIST_FORWARD_FIELDS}
        prediction = model.forward_at_omega(state_input, target_state[:, 3])
        shuffled_input = dict(state_input)
        shuffled_input["q0_relation_m"] = torch.roll(
            state_input["q0_relation_m"], 1, dims=0,
        )
        shuffled_input["q0_supported"] = torch.roll(
            state_input["q0_supported"], 1, dims=0,
        )
        shuffled = model.forward_at_omega(shuffled_input, target_state[:, 3])
        center_nll, center_components = anonymous_center_prior_loss(
            prediction, batch[CENTER_OFFSET_TARGET_FIELD],
        )
        center_huber = F.smooth_l1_loss(
            prediction["center_offset_m"], batch[CENTER_OFFSET_TARGET_FIELD],
            beta=0.05,
        )
        velocity_huber = F.smooth_l1_loss(
            prediction["velocity_mps"], target_state[:, :3], beta=0.25,
        )
        shuffled_velocity_huber = F.smooth_l1_loss(
            shuffled["velocity_mps"], target_state[:, :3], beta=0.25,
        )
        def gate_target(value: dict[str, torch.Tensor]) -> torch.Tensor:
            q0_error = torch.linalg.vector_norm(
                value["q0_component_velocity_mps"] - target_state[:, :3], dim=-1,
            )
            history_error = torch.linalg.vector_norm(
                value["history_component_velocity_mps"] - target_state[:, :3], dim=-1,
            )
            return torch.sigmoid((history_error - q0_error) / 0.20).detach()

        gate_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(
                prediction["q0_component_logit"], gate_target(prediction),
            )
            + F.binary_cross_entropy_with_logits(
                shuffled["q0_component_logit"], gate_target(shuffled),
            )
        )
        objective = (
            4.0 * center_huber + velocity_huber
            + 0.25 * shuffled_velocity_huber + 0.20 * gate_loss
            + 0.01 * center_nll
        )
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        nonfinite = [
            name for name, parameter in model.named_parameters()
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        ]
        if nonfinite:
            raise RuntimeError("non-finite center gradients: " + ", ".join(nonfinite))
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
        lr = _set_phase_lr(
            optimizer, base_lr=args.learning_rate,
            stage_update=next_update, stage_total=FIXED_UPDATES,
        )
        optimizer.step()
        if any(not bool(torch.isfinite(value).all()) for value in model.parameters()):
            raise RuntimeError("center optimizer produced non-finite parameters")
        global_update = next_update
        if global_update % args.log_interval == 0:
            print(json.dumps({
                "global_update": global_update,
                "objective": float(objective.detach()),
                "center_huber": float(center_huber.detach()),
                "center_nll": float(center_components["center_nll"].detach()),
                "velocity_huber": float(velocity_huber.detach()),
                "shuffled_velocity_huber": float(shuffled_velocity_huber.detach()),
                "gate_loss": float(gate_loss.detach()),
                "learning_rate": lr, "elapsed_s": time.time() - started,
            }, sort_keys=True), flush=True)
        if global_update % CHECKPOINT_INTERVAL == 0:
            checkpoint = checkpoint_dir / f"checkpoint-update-{global_update:06d}.pt"
            _atomic_checkpoint(checkpoint, {
                "schema_version": RUN_SCHEMA,
                "checkpoint_kind": "training_boundary",
                "final_validation_completed": False,
                "contract_sha256": contract_sha256,
                "global_update": global_update,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": _rng_state(sampler, prefix_generator),
                "validation_history": validation_history,
                "frozen_state_dict_sha256": frozen_initial,
                "frozen_checkpoint_sha256": frozen_checkpoint_sha256,
            })
            checkpoint_sha = sha256_file(checkpoint)
            checkpoint_ledger.append({
                "global_update": global_update, "path": str(checkpoint),
                "sha256": checkpoint_sha, "kind": "training_boundary",
                "adopted_after_manifest_crash": False,
            })
            manifest_payload["global_update"] = global_update
            manifest_payload["checkpoint_ledger"] = checkpoint_ledger
            _atomic_json(output / "run_manifest.json", manifest_payload)
            if global_update == args.stop_after_update:
                print(json.dumps({
                    "status": "interrupted_at_requested_checkpoint",
                    "global_update": global_update,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                }, sort_keys=True), flush=True)
                return output

    model.eval()
    validation_history.append({
        "global_update": global_update,
        "metrics": evaluate_loaded(
            model, validation_loader, mapper, s_model, h_model, device,
        ),
    })
    frozen_final = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    }
    if frozen_final != frozen_initial:
        raise RuntimeError("V14-B0 modified frozen upstream")
    if _git_state() != git_state:
        raise RuntimeError("git commit/worktree changed during V14-B0")
    if sha256_file(manifest_path) != provenance["dataset_manifest_sha256"]:
        raise RuntimeError("paired dataset manifest changed during V14-B0")
    truth_manifest_path = Path(args.truth_history).resolve() / "dataset_manifest.json"
    if sha256_file(truth_manifest_path) != expected_truth_sha:
        raise RuntimeError("truth manifest changed during V14-B0")
    _assert_manifest_shards_unchanged(dataset_path, manifest, label="paired dataset")
    _assert_manifest_shards_unchanged(
        Path(args.truth_history).resolve(), motion_truth.manifest, label="truth",
    )
    if {
        name: sha256_file(path) for name, path in source_paths.items()
    } != provenance["source_sha256"]:
        raise RuntimeError("bound source changed during V14-B0")
    if {
        "mapper": sha256_file(args.mapper_checkpoint),
        "s": sha256_file(args.s_checkpoint),
        "h": sha256_file(args.h_checkpoint),
    } != frozen_checkpoint_sha256:
        raise RuntimeError("frozen checkpoint file changed during V14-B0")
    final_checkpoint = checkpoint_dir / f"checkpoint-update-{global_update:06d}.pt"
    if not final_checkpoint.is_file():
        raise RuntimeError("fixed update-100 training checkpoint is missing")
    matching_final_ledger = [
        item for item in checkpoint_ledger
        if int(item.get("global_update", -1)) == global_update
    ]
    if (
        len(matching_final_ledger) != 1
        or matching_final_ledger[0].get("sha256") != sha256_file(final_checkpoint)
    ):
        raise RuntimeError("fixed update-100 checkpoint ledger differs")
    initial = validation_history[0]["metrics"]
    final = validation_history[-1]["metrics"]
    initial_velocity = initial["truth_omega_velocity_error_mps"]["overall"]
    final_velocity = final["truth_omega_velocity_error_mps"]["overall"]
    initial_center = initial["center_error_m"]["overall"]
    final_center = final["center_error_m"]["overall"]
    initial_solver = initial["solver_coverage"]["intact"]["overall"]
    final_solver = final["solver_coverage"]["intact"]["overall"]
    final_blind = final["h_counterfactual_velocity_error_mps"][
        "history_blind"
    ]["overall"]
    final_shuffled = final["h_counterfactual_velocity_error_mps"][
        "q0_shuffled"
    ]["overall"]
    final_intact = final["h_counterfactual_velocity_error_mps"][
        "intact"
    ]["overall"]
    final_intact_weight = final["q0_component_weight"]["intact"]
    final_shuffled_weight = final["q0_component_weight"]["q0_shuffled"]
    final_blind_solver = final["solver_coverage"]["history_blind"]["overall"]
    final_shuffled_solver = final["solver_coverage"]["q0_shuffled"]["overall"]
    gates = {
        "initial_state_supported_complete": (
            initial_solver["state_supported_fraction"] == 1.0
        ),
        "final_state_supported_complete": (
            final_solver["state_supported_fraction"] == 1.0
        ),
        "final_blind_state_supported_complete": (
            final_blind_solver["state_supported_fraction"] == 1.0
        ),
        "final_shuffled_state_supported_complete": (
            final_shuffled_solver["state_supported_fraction"] == 1.0
        ),
        "initial_profile_coverage_ge_0_95": (
            initial_solver["profile_fraction"] >= 0.95
        ),
        "final_profile_coverage_ge_0_95": (
            final_solver["profile_fraction"] >= 0.95
        ),
        "velocity_mean_improves_2pct": (
            final_velocity["mean"] <= 0.98 * initial_velocity["mean"]
        ),
        "velocity_p50_improves_2pct": (
            final_velocity["p50"] <= 0.98 * initial_velocity["p50"]
        ),
        "center_mean_improves_2pct": (
            final_center["mean"] <= 0.98 * initial_center["mean"]
        ),
        "center_p50_improves_2pct": (
            final_center["p50"] <= 0.98 * initial_center["p50"]
        ),
        "shuffled_h_mean_within_2pct_of_blind": (
            final_shuffled["mean"] <= 1.02 * final_blind["mean"]
        ),
        "shuffled_h_p50_within_2pct_of_blind": (
            final_shuffled["p50"] <= 1.02 * final_blind["p50"]
        ),
        "intact_h_mean_beats_blind_5pct": (
            final_intact["mean"] <= 0.95 * final_blind["mean"]
        ),
        "intact_h_p50_beats_blind_2pct": (
            final_intact["p50"] <= 0.98 * final_blind["p50"]
        ),
        "intact_q0_weight_exceeds_shuffled_by_0_10": (
            final_intact_weight["mean"]
            >= final_shuffled_weight["mean"] + 0.10
        ),
    }
    passed = all(gates.values())
    result = {
        "schema_version": RUN_SCHEMA,
        "status": "passed" if passed else "failed",
        "test_accessed": False, "global_update": global_update,
        "contract_sha256": contract_sha256,
        "checkpoint": str(final_checkpoint),
        "checkpoint_sha256": sha256_file(final_checkpoint),
        "initial_metrics": initial, "final_metrics": final,
        "body_change": {
            "velocity_mean_fraction": final_velocity["mean"] / initial_velocity["mean"],
            "velocity_p50_fraction": final_velocity["p50"] / initial_velocity["p50"],
            "center_mean_fraction": final_center["mean"] / initial_center["mean"],
            "center_p50_fraction": final_center["p50"] / initial_center["p50"],
        },
        "gates": gates,
        "authorized_free_omega_continuation": passed,
        "frozen_final_state_dict_sha256": frozen_final,
        "elapsed_s": time.time() - started,
    }
    _atomic_json(output / "screen_result.json", result)
    manifest_payload.update({
        "status": "completed", "global_update": global_update,
        "validation_history": validation_history,
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
        "screen_result_sha256": sha256_file(output / "screen_result.json"),
    })
    _atomic_json(output / "run_manifest.json", manifest_payload)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--truth-history", required=True)
    parser.add_argument("--mapper-checkpoint", required=True)
    parser.add_argument("--s-checkpoint", required=True)
    parser.add_argument("--h-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--stop-after-update", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--updates", type=int, default=FIXED_UPDATES)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--validation-batch-size", type=int, default=96)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--center-precision", type=float, default=25.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
