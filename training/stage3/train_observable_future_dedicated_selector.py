"""Train an independent anonymous selector beside a bit-exact frozen F trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .observable_future_dataset import ObservableFutureDataset
from .observable_future_model import AnonymousCandidateFutureExpert, DYNAMIC_EXPERTS
from .train_causal_physical_ab import _git_state, _seed, _to_device, _write_json
from .train_observable_future_selector import (
    _instantiate_from_checkpoint,
    _scheduled_learning_rate,
    _selector_loss,
    _sha256,
    _validate_foundation_provenance,
    evaluate,
)


DEDICATED_SELECTOR_PARAMETER_PREFIXES = (
    "history_encoder.",
    "candidate_encoder.",
    "switch_candidate_head.",
    "switch_logit.",
)
SOURCE_FILES = (
    "observable_future_dataset.py",
    "build_observable_future_dataset.py",
    "observable_future_model.py",
    "observable_future_loss.py",
    "observable_future_selector_loss.py",
    "train_observable_future_selector.py",
    "train_observable_future_dedicated_selector.py",
    "audit_observable_future_dataset.py",
)


def _module_state_sha256(
    module: nn.Module, *, prefixes: tuple[str, ...] | None = None,
    invert: bool = False,
) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        selected = prefixes is None or name.startswith(prefixes)
        if invert:
            selected = not selected
        if not selected:
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def freeze_dedicated_selector(
    trajectory_model: AnonymousCandidateFutureExpert,
    selector_model: AnonymousCandidateFutureExpert,
) -> tuple[list[str], list[str]]:
    """Freeze the parent trajectory and whitelist the independent selector."""
    for parameter in trajectory_model.parameters():
        parameter.requires_grad_(False)
    trainable: list[str] = []
    frozen: list[str] = []
    for name, parameter in selector_model.named_parameters():
        selected = name.startswith(DEDICATED_SELECTOR_PARAMETER_PREFIXES)
        parameter.requires_grad_(selected)
        (trainable if selected else frozen).append(name)
    if not trainable or not frozen:
        raise ValueError("dedicated selector freeze did not partition parameters")
    return trainable, frozen


def _canonical_selected_row(
    logits: torch.Tensor,
    candidate_step: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Resolve exact logit ties by the unique signed step, not row order."""
    maximum = logits.amax(dim=-1, keepdim=True)
    tied = (logits == maximum) & candidate_mask.to(torch.bool)[:, None, :]
    maximum_step = (
        torch.finfo(candidate_step.dtype).max
        if candidate_step.is_floating_point()
        else torch.iinfo(candidate_step.dtype).max
    )
    tied_step = torch.where(
        tied, candidate_step[:, None, :],
        candidate_step.new_full((), maximum_step),
    )
    selected_step = tied_step.amin(dim=-1)
    matches = (
        candidate_mask.to(torch.bool)[:, None, :]
        & (candidate_step[:, None, :] == selected_step.unsqueeze(-1))
    )
    if bool(torch.any(matches.sum(dim=-1) != 1)):
        raise ValueError("canonical selector tie-break requires unique candidate steps")
    return matches.to(torch.long).argmax(dim=-1)


class FrozenTrajectoryDedicatedSelector(nn.Module):
    """Use a frozen F only for coordinates and an independent clone for logits."""

    def __init__(
        self,
        trajectory_model: AnonymousCandidateFutureExpert,
        selector_model: AnonymousCandidateFutureExpert,
    ) -> None:
        super().__init__()
        if trajectory_model.config != selector_model.config:
            raise ValueError("trajectory and selector model configs must match")
        self.trajectory_model = trajectory_model
        self.selector_model = selector_model

    def train(self, mode: bool = True) -> "FrozenTrajectoryDedicatedSelector":
        super().train(mode)
        # A generic caller must never reactivate dropout in the protected F.
        self.trajectory_model.eval()
        return self

    def forward(
        self,
        history_position_rel_m: torch.Tensor,
        history_time_s: torch.Tensor,
        history_dt_s: torch.Tensor,
        history_switch_step: torch.Tensor,
        history_mask: torch.Tensor,
        current_position_m: torch.Tensor,
        candidate_relation_m: torch.Tensor,
        candidate_step: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_confidence: torch.Tensor,
        tau_s: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        inputs = (
            history_position_rel_m, history_time_s, history_dt_s,
            history_switch_step, history_mask, current_position_m,
            candidate_relation_m, candidate_step, candidate_mask,
            candidate_confidence, tau_s,
        )
        self.trajectory_model.eval()
        with torch.no_grad():
            trajectory = self.trajectory_model(*inputs)
        selector = self.selector_model(*inputs)
        logits = selector["switch_logits"]
        probability = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        selected_row = _canonical_selected_row(
            logits, candidate_step.detach(), candidate_mask.detach()
        )
        selected_step = candidate_step.detach().gather(1, selected_row)
        gather = selected_row[:, :, None, None].expand(-1, -1, 1, 3)
        selected_delta = trajectory["conditional_delta_m"].gather(
            2, gather
        ).squeeze(2)
        tau = self.trajectory_model._expanded_tau(tau_s, logits.shape[0])
        selected_delta = torch.where(
            (tau == 0).unsqueeze(-1), torch.zeros_like(selected_delta),
            selected_delta,
        )
        current = current_position_m.detach()
        return {
            "switch_logits": logits,
            "switch_probability": probability,
            "conditional_delta_m": trajectory["conditional_delta_m"],
            "conditional_position_m": trajectory["conditional_position_m"],
            "trajectory_coefficient_m": trajectory["trajectory_coefficient_m"],
            "time_basis": trajectory["time_basis"],
            "selected_candidate_row": selected_row,
            "selected_switch_step": selected_step,
            "delta_m": selected_delta,
            "position_m": current[:, None, :] + selected_delta,
        }


def set_dedicated_selector_train_mode(
    system: FrozenTrajectoryDedicatedSelector,
) -> None:
    system.eval()
    system.selector_model.history_encoder.train()
    system.selector_model.candidate_encoder.train()
    system.selector_model.switch_candidate_head.train()
    system.selector_model.switch_logit.train()


def _checkpoint(
    path: Path, system: FrozenTrajectoryDedicatedSelector,
    optimizer: torch.optim.Optimizer, epoch: int, update: int,
    metrics: dict[str, object], provenance: dict[str, object],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite dedicated selector checkpoint: {path}")
    selector_state = {
        name: value
        for name, value in system.selector_model.state_dict().items()
        if name.startswith(DEDICATED_SELECTOR_PARAMETER_PREFIXES)
    }
    torch.save({
        "model_class": "FrozenTrajectoryDedicatedSelector",
        "trajectory_model_class": "AnonymousCandidateFutureExpert",
        "model_config": system.trajectory_model.config,
        "selector_state": selector_state,
        "selector_state_sha256": _module_state_sha256(
            system.selector_model,
            prefixes=DEDICATED_SELECTOR_PARAMETER_PREFIXES,
        ),
        "optimizer": optimizer.state_dict(),
        "training_phase": "dedicated-selector-only",
        "epoch": epoch,
        "update": update,
        "validation": metrics,
        "provenance": provenance,
    }, path)


def train(args: argparse.Namespace) -> Path:
    dataset_dir = Path(args.dataset).resolve()
    output_dir = Path(args.output).resolve()
    initial_path = Path(args.initial_checkpoint).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite dedicated selector run: {output_dir}")
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("test_accessed", True)):
        raise ValueError("dedicated selector dataset accessed test")
    dataset_manifest_sha256 = _sha256(manifest_path)
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    _validate_foundation_provenance(initial, dataset_manifest_sha256)
    trajectory_model = _instantiate_from_checkpoint(initial, args.expert)
    selector_model = _instantiate_from_checkpoint(initial, args.expert)
    if int(trajectory_model.maximum_absolute_step) != max(
        abs(int(value)) for value in manifest["candidate_steps"]
    ):
        raise ValueError("dedicated selector candidate range disagrees with dataset")
    trainable_names, selector_frozen_names = freeze_dedicated_selector(
        trajectory_model, selector_model
    )
    initial_trajectory_sha256 = _module_state_sha256(trajectory_model)
    initial_selector_trainable_sha256 = _module_state_sha256(
        selector_model, prefixes=DEDICATED_SELECTOR_PARAMETER_PREFIXES
    )
    initial_selector_frozen_sha256 = _module_state_sha256(
        selector_model, prefixes=DEDICATED_SELECTOR_PARAMETER_PREFIXES,
        invert=True,
    )
    system = FrozenTrajectoryDedicatedSelector(trajectory_model, selector_model)

    _seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    system = system.to(device)
    train_dataset = ObservableFutureDataset(
        dataset_dir, "train", args.expert, seed=args.seed, shuffle=True,
        sample_limit=args.train_limit,
    )
    validation_dataset = ObservableFutureDataset(
        dataset_dir, "validation", args.expert, seed=args.seed + 1,
        shuffle=False, sample_limit=args.validation_limit,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    probe = _to_device(next(iter(validation_loader)), device)
    system.eval()
    with torch.no_grad():
        reference = system(*(
            probe[name] for name in (
                "history_position_rel_m", "history_time_s", "history_dt_s",
                "history_switch_step", "history_mask", "current_position_m",
                "candidate_relation_m", "candidate_step", "candidate_mask",
                "candidate_confidence", "tau_s",
            )
        ))
        conditional_probe = reference["conditional_delta_m"].detach().cpu()
        coefficient_probe = reference["trajectory_coefficient_m"].detach().cpu()
        time_basis_probe = reference["time_basis"].detach().cpu()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in selector_model.parameters()
         if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    output_dir.mkdir(parents=True)
    source_dir = Path(__file__).resolve().parent
    provenance = {
        "schema_version": "stage3-observable-future-dedicated-selector-run-v1",
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "dataset_qualification": True,
        "test_accessed": False,
        "phase": "truth-S combined dedicated-selector-only",
        "evaluation_split": "validation",
        "initial_checkpoint": str(initial_path),
        "initial_checkpoint_sha256": _sha256(initial_path),
        "initial_trajectory_state_sha256": initial_trajectory_sha256,
        "initial_selector_trainable_state_sha256": initial_selector_trainable_sha256,
        "initial_selector_frozen_state_sha256": initial_selector_frozen_sha256,
        "trainable_parameter_names": trainable_names,
        "selector_frozen_parameter_names": selector_frozen_names,
        "trajectory_parameter_count": sum(p.numel() for p in trajectory_model.parameters()),
        "selector_trainable_parameter_count": sum(
            p.numel() for p in selector_model.parameters() if p.requires_grad
        ),
        "conditional_trajectory_trainable": False,
        "selector_encoder_independent": True,
        "selector_model_class": "AnonymousCandidateFutureExpert",
        "selector_clone_source": "initial_checkpoint model state",
        "candidate_tie_break": "minimum unique signed step among exact maximum logits",
        "hard_gather_source": "frozen parent conditional_delta_m",
        "tau_zero_policy": "selector step zero plus explicit hard-delta zero",
        "physical_identity_input": False,
        "frozen_S_retrained": False,
        "git": _git_state(),
        "source_sha256": {name: _sha256(source_dir / name) for name in SOURCE_FILES},
        "training_arguments": dict(vars(args)),
    }
    baseline = evaluate(system, validation_loader, device, args)
    history: list[dict[str, object]] = []
    best_tuple: tuple[float, ...] | None = None
    best_record: dict[str, object] | None = None
    update = 0
    started = time.time()
    selector_named_parameters = dict(selector_model.named_parameters())
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        set_dedicated_selector_train_mode(system)
        train_objectives: list[float] = []
        for raw in train_loader:
            batch = _to_device(raw, device)
            next_update = update + 1
            learning_rate = _scheduled_learning_rate(
                next_update, base_learning_rate=args.learning_rate,
                minimum_learning_rate=args.minimum_learning_rate,
                warmup_updates=args.warmup_updates,
                total_updates=args.max_updates,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            prediction = system(
                batch["history_position_rel_m"], batch["history_time_s"],
                batch["history_dt_s"], batch["history_switch_step"],
                batch["history_mask"], batch["current_position_m"],
                batch["candidate_relation_m"], batch["candidate_step"],
                batch["candidate_mask"], batch["candidate_confidence"],
                batch["tau_s"],
            )
            objective, _ = _selector_loss(prediction, batch, args)
            objective.backward()
            for name, parameter in trajectory_model.named_parameters():
                if parameter.grad is not None:
                    raise RuntimeError(f"frozen trajectory received gradient: {name}")
            for name in selector_frozen_names:
                if selector_named_parameters[name].grad is not None:
                    raise RuntimeError(f"unused selector trajectory head received gradient: {name}")
            if args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in selector_model.parameters() if p.requires_grad],
                    args.gradient_clip_norm,
                )
            optimizer.step()
            update = next_update
            train_objectives.append(float(objective.detach().cpu()))
            if update >= args.max_updates:
                break
        reached_limit = update >= args.max_updates
        validate_now = (
            epoch % args.validation_interval == 0
            or reached_limit or epoch == args.epochs
        )
        record: dict[str, object] = {
            "epoch": epoch, "update": update,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_objective": float(np.mean(train_objectives)),
            "validation": None,
        }
        history.append(record)
        if not validate_now:
            if reached_limit:
                break
            continue
        trajectory_sha256 = _module_state_sha256(trajectory_model)
        selector_frozen_sha256 = _module_state_sha256(
            selector_model, prefixes=DEDICATED_SELECTOR_PARAMETER_PREFIXES,
            invert=True,
        )
        if trajectory_sha256 != initial_trajectory_sha256:
            raise RuntimeError("dedicated selector changed parent trajectory state")
        if selector_frozen_sha256 != initial_selector_frozen_sha256:
            raise RuntimeError("dedicated selector changed its unused trajectory heads")
        system.eval()
        with torch.no_grad():
            current = system(
                probe["history_position_rel_m"], probe["history_time_s"],
                probe["history_dt_s"], probe["history_switch_step"],
                probe["history_mask"], probe["current_position_m"],
                probe["candidate_relation_m"], probe["candidate_step"],
                probe["candidate_mask"], probe["candidate_confidence"],
                probe["tau_s"],
            )
        if not torch.equal(current["conditional_delta_m"].cpu(), conditional_probe):
            raise RuntimeError("dedicated selector changed conditional trajectory")
        if not torch.equal(current["trajectory_coefficient_m"].cpu(), coefficient_probe):
            raise RuntimeError("dedicated selector changed trajectory coefficients")
        if not torch.equal(current["time_basis"].cpu(), time_basis_probe):
            raise RuntimeError("dedicated selector changed trajectory time basis")
        validation = evaluate(system, validation_loader, device, args)
        if validation["conditional_position"] != baseline["conditional_position"]:
            raise RuntimeError("dedicated selector changed full conditional metrics")
        validation["conditional_probe_bit_exact"] = True
        validation["trajectory_state_sha256"] = trajectory_sha256
        validation["selector_frozen_state_sha256"] = selector_frozen_sha256
        record["validation"] = validation
        hard = validation["hard_routed_position"]
        selection = (
            float(hard["p95_m"]), float(hard["p99_m"]),
            -float(validation["switch_accuracy"]),
            -float(validation["switch_macro_recall"]),
            -float(validation["switch_minimum_step_recall"]),
        )
        checkpoint_path = output_dir / f"dedicated-selector-epoch-{epoch:04d}.pt"
        _checkpoint(
            checkpoint_path, system, optimizer, epoch, update, validation,
            provenance,
        )
        checkpoint_sha256 = _sha256(checkpoint_path)
        if best_tuple is None or selection < best_tuple:
            best_tuple = selection
            best_record = {
                "path": checkpoint_path.name,
                "sha256": checkpoint_sha256,
                "epoch": epoch,
                "update": update,
                "selection_tuple": list(selection),
                "validation": validation,
            }
        _write_json(output_dir / "run_progress.json", {
            "status": "running", "baseline_validation": baseline,
            "history": history, "best": best_record,
            "elapsed_s": time.time() - started, **provenance,
        })
        print(json.dumps(record), flush=True)
        if reached_limit:
            break
    if best_record is None:
        raise RuntimeError("dedicated selector produced no validation checkpoint")
    best_validation = best_record["validation"]
    gates = {
        "conditional_probe_bit_exact": bool(best_validation["conditional_probe_bit_exact"]),
        "trajectory_state_unchanged": best_validation["trajectory_state_sha256"] == initial_trajectory_sha256,
        "selector_frozen_state_unchanged": best_validation["selector_frozen_state_sha256"] == initial_selector_frozen_sha256,
        "minimum_switch_accuracy": float(best_validation["switch_accuracy"]) >= args.minimum_switch_accuracy,
        "minimum_macro_recall": float(best_validation["switch_macro_recall"]) >= args.minimum_macro_recall,
        "minimum_step_recall": float(best_validation["switch_minimum_step_recall"]) >= args.minimum_step_recall,
        "maximum_hard_p95_m": float(best_validation["hard_routed_position"]["p95_m"]) <= args.maximum_hard_p95_m,
    }
    final = {
        "status": "complete" if all(gates.values()) else "gate_failed",
        "stop_reason": "max_updates" if update >= args.max_updates else "epoch_limit",
        "expert": args.expert,
        "model_config": trajectory_model.config,
        "baseline_validation": baseline,
        "best": best_record,
        "history": history,
        "final_trajectory_state_sha256": _module_state_sha256(trajectory_model),
        "final_selector_trainable_state_sha256": _module_state_sha256(
            selector_model, prefixes=DEDICATED_SELECTOR_PARAMETER_PREFIXES
        ),
        "final_selector_frozen_state_sha256": _module_state_sha256(
            selector_model, prefixes=DEDICATED_SELECTOR_PARAMETER_PREFIXES,
            invert=True,
        ),
        "gates": gates,
        "gate_thresholds": {
            "minimum_switch_accuracy": args.minimum_switch_accuracy,
            "minimum_macro_recall": args.minimum_macro_recall,
            "minimum_step_recall": args.minimum_step_recall,
            "maximum_hard_p95_m": args.maximum_hard_p95_m,
        },
        "elapsed_s": time.time() - started,
        **provenance,
    }
    _write_json(output_dir / "run_manifest.json", final)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--expert", choices=DYNAMIC_EXPERTS, default="combined")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=130)
    parser.add_argument("--max-updates", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-updates", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--switch-weight", type=float, default=1.0)
    parser.add_argument("--macro-balance-weight", type=float, default=0.25)
    parser.add_argument("--switch-focal-gamma", type=float, default=2.0)
    parser.add_argument("--distance-cost-weight", type=float, default=1.0)
    parser.add_argument("--distance-cost-scale-m", type=float, default=0.3)
    parser.add_argument("--distance-cost-cap", type=float, default=2.0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--minimum-switch-accuracy", type=float, default=0.95)
    parser.add_argument("--minimum-macro-recall", type=float, default=0.92)
    parser.add_argument("--minimum-step-recall", type=float, default=0.80)
    parser.add_argument("--maximum-hard-p95-m", type=float, default=0.10)
    args = parser.parse_args()
    if (
        args.batch_size < 1 or args.workers < 0 or args.epochs < 1
        or args.max_updates < 1 or args.validation_interval < 1
        or args.warmup_updates < 0
        or args.minimum_learning_rate > args.learning_rate
        or min(args.learning_rate, args.minimum_learning_rate) <= 0
        or min(args.switch_weight, args.distance_cost_weight,
               args.switch_focal_gamma, args.weight_decay,
               args.gradient_clip_norm) < 0
        or args.switch_weight + args.distance_cost_weight <= 0
        or min(args.distance_cost_scale_m, args.distance_cost_cap,
               args.maximum_hard_p95_m) <= 0
        or not 0 <= args.macro_balance_weight <= 1
        or not 0 <= args.minimum_switch_accuracy <= 1
        or not 0 <= args.minimum_macro_recall <= 1
        or not 0 <= args.minimum_step_recall <= 1
    ):
        parser.error("dedicated selector training arguments are invalid")
    print(train(args))


if __name__ == "__main__":
    main()
