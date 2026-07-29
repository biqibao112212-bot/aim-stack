"""Train a PnP-domain F while preserving clean F as a separate frozen branch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cyclic_future_foundation import load_frozen_v19
from .dual_domain_future import (
    ObservationDomain,
    assert_independent_models,
    route_future_expert,
)
from .formal_run_contract import (
    capture_formal_contract,
    configure_formal_runtime,
    load_protocol,
    require_asset_binding,
    require_compatible_contracts,
    require_exact_protocol_arguments,
    require_fixed_final_state,
    require_formal_checkpoint_manifest,
    resolve_formal_schedule,
    verify_formal_contract,
)
from .observable_future_pnp_ab import (
    ObservableFuturePnPSFDataset,
    canonicalize_direction_keep_c4,
    f_forward,
    load_observable_f_checkpoint,
    sha256_file,
    state_dict_sha256,
)
from .pnp_q0_hypothesis_adapter import (
    compose_hypothesis_for_f,
    hypothesis_forward,
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .split_audit import require_formal_split_isolation
from .train_causal_physical_ab import _git_state, _seed, _to_device
from .train_observable_future_pnp_ab import (
    _absolute_loss,
    _eval_add,
    _finish_eval,
    _new_eval_storage,
    _percentiles,
)
from .train_pnp_q0_hypothesis_adapter import _s_forward
from .train_pnp_window_mapper_distillation import (
    _atomic_checkpoint,
    _atomic_json,
    _learning_rate,
    _validate_batch_contract,
)


RUN_SCHEMA = "stage3-dual-domain-pnp-f-v1"
SELECTOR_PARAMETER_PREFIXES = ("switch_candidate_head.", "switch_logit.")
LEGACY_JOINT_B_CONDITIONAL_P95_M = 0.3440020978450775
MAPPER_BASELINE_CONDITIONAL_P95_M = 0.517807
FORMAL_SOURCE_BUNDLE = (
    "training/stage3/formalization_protocol.json",
    "training/stage3/formal_run_contract.py",
    "training/stage3/train_dual_domain_pnp_f.py",
    "training/stage3/dual_domain_future.py",
    "training/stage3/observable_future_model.py",
    "training/stage3/observable_future_loss.py",
    "training/stage3/observable_future_pnp_ab.py",
    "training/stage3/train_observable_future_pnp_ab.py",
    "training/stage3/train_pnp_window_mapper_distillation.py",
    "training/stage3/build_observable_future_pnp_sf_upper_bound_dataset.py",
    "training/stage3/pnp_observation_mapper.py",
    "training/stage3/pnp_q0_hypothesis_adapter.py",
    "training/stage3/train_pnp_q0_hypothesis_adapter.py",
    "training/stage3/cyclic_future_foundation.py",
    "training/stage3/train_causal_physical_ab.py",
    "training/stage3/split_audit.py",
    "training/stage3/schema.py",
)


def _configure_trainable_parameters(
    model: torch.nn.Module, stage: str,
) -> tuple[list[torch.nn.Parameter], list[str], list[str]]:
    if stage not in {"trajectory", "selector"}:
        raise ValueError(f"unknown dual-domain training stage: {stage}")
    trainable: list[torch.nn.Parameter] = []
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    for name, parameter in model.named_parameters():
        selector = name.startswith(SELECTOR_PARAMETER_PREFIXES)
        enabled = not selector if stage == "trajectory" else selector
        parameter.requires_grad_(enabled)
        (trainable_names if enabled else frozen_names).append(name)
        if enabled:
            trainable.append(parameter)
    if not trainable or not frozen_names:
        raise RuntimeError("dual-domain stage did not partition F parameters")
    return trainable, trainable_names, frozen_names


def _partition_hash(model: torch.nn.Module, names: list[str]) -> str:
    state = model.state_dict()
    return state_dict_sha256({name: state[name] for name in names})


def _set_stage_training_mode(model: torch.nn.Module, stage: str) -> None:
    if stage == "trajectory":
        model.train()
        model.switch_candidate_head.eval()
        model.switch_logit.eval()
        return
    if stage == "selector":
        model.eval()
        model.switch_candidate_head.train()
        model.switch_logit.train()
        return
    raise ValueError(f"unknown dual-domain training stage: {stage}")


def _hash_tensor(hasher: Any, name: str, value: torch.Tensor) -> None:
    array = value.detach().cpu().contiguous().numpy()
    hasher.update(name.encode("utf-8"))
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(str(tuple(array.shape)).encode("ascii"))
    hasher.update(array.tobytes())


@torch.no_grad()
def _pnp_composed(
    mapper: torch.nn.Module,
    s_model: torch.nn.Module,
    h_model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    mapped = mapper(
        batch["pnp_s_obs_m"], batch["pnp_s_obs_mask"],
        batch["pnp_s_event_time_s"], batch["pnp_s_event_mask"],
    )
    s_output = _s_forward(
        s_model, mapped["corrected_obs_m"], batch["pnp_s_obs_mask"],
        batch["pnp_s_primary_mask"], batch["pnp_s_event_mask"],
        batch["pnp_s_event_time_s"], batch["pnp_s_switch_step"],
    )
    h_output = hypothesis_forward(h_model, s_output)
    return compose_hypothesis_for_f(
        h_output, s_output["primary_index"], mapped["corrected_obs_m"],
        batch["pnp_s_obs_mask"], batch["pnp_s_primary_mask"],
        batch["pnp_candidate_step"], batch["pnp_s_event_mask"],
    )


def _pnp_f_forward(
    model: torch.nn.Module,
    composed: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return f_forward(
        model, batch, prefix="pnp_",
        history_position_rel_m=composed["history_position_rel_m"],
        current_position_m=composed["current_position_m"],
        candidate_relation_m=composed["candidate_relation_m"],
        candidate_confidence=composed["candidate_confidence"],
        detach_observation_inputs=True,
    )


@torch.no_grad()
def evaluate(
    pnp_f: torch.nn.Module,
    clean_f: torch.nn.Module,
    mapper: torch.nn.Module,
    s_model: torch.nn.Module,
    h_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    pnp_f = route_future_expert(
        ObservationDomain.PNP_V41, clean_f=clean_f, pnp_f=pnp_f,
    )
    clean_f = route_future_expert(
        ObservationDomain.CLEAN, clean_f=clean_f, pnp_f=pnp_f,
    )
    for model in (pnp_f, clean_f, mapper, s_model, h_model):
        model.eval()
    pnp_storage = _new_eval_storage()
    clean_storage = _new_eval_storage()
    current_error: list[np.ndarray] = []
    conditional_hasher = hashlib.sha256()
    upstream_hasher = hashlib.sha256()
    minimum_step_eligible = 0
    minimum_step_correct = 0
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        _validate_batch_contract(batch)
        composed = _pnp_composed(mapper, s_model, h_model, batch)
        prediction = _pnp_f_forward(pnp_f, composed, batch)
        _hash_tensor(
            conditional_hasher, "conditional_position_m",
            prediction["conditional_position_m"],
        )
        for name in (
            "history_position_rel_m", "current_position_m",
            "candidate_relation_m", "candidate_confidence",
        ):
            _hash_tensor(upstream_hasher, name, composed[name])
        selected_row = prediction["switch_logits"].argmax(dim=-1)
        selected_step = batch["pnp_candidate_step"].gather(1, selected_row)
        minimum_step_mask = (
            batch["target_query_mask"].to(torch.bool)
            & (batch["target_switch_count"].abs() == 1)
        )
        minimum_step_eligible += int(minimum_step_mask.sum())
        minimum_step_correct += int((
            minimum_step_mask
            & (selected_step == batch["target_switch_count"])
        ).sum())
        _eval_add(
            pnp_storage, prediction, batch,
            candidate_supported=composed["candidate_supported"],
        )
        current_error.append(torch.linalg.vector_norm(
            composed["current_position_m"] - batch["current_position_m"], dim=-1
        ).cpu().numpy())
        clean_prediction = f_forward(clean_f, batch)
        _eval_add(clean_storage, clean_prediction, batch)
    pnp_metrics = _finish_eval(pnp_storage)
    pnp_metrics["minimum_step_switch_recall"] = (
        minimum_step_correct / minimum_step_eligible
        if minimum_step_eligible else None
    )
    pnp_metrics["minimum_step_switch_count"] = minimum_step_eligible
    return {
        "pnp_domain": pnp_metrics,
        "clean_frozen_domain": _finish_eval(clean_storage),
        "current_position_error": _percentiles(current_error),
        "conditional_output_sha256": conditional_hasher.hexdigest(),
        "upstream_f_input_sha256": upstream_hasher.hexdigest(),
        "clean_branch_bit_exact_by_construction": True,
        "domain_routing": "explicit external clean versus PnP branch",
    }


@torch.no_grad()
def _assert_initial_output_equivalence(
    clean_f: torch.nn.Module,
    pnp_f: torch.nn.Module,
    mapper: torch.nn.Module,
    s_model: torch.nn.Module,
    h_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    clean_f.eval()
    pnp_f.eval()
    raw_batch = next(iter(loader))
    batch = _to_device(raw_batch, device)
    composed = _pnp_composed(mapper, s_model, h_model, batch)
    clean_output = _pnp_f_forward(clean_f, composed, batch)
    pnp_output = _pnp_f_forward(pnp_f, composed, batch)
    if clean_output.keys() != pnp_output.keys():
        raise RuntimeError("initial clean/PnP F output contracts differ")
    for name in clean_output:
        if clean_output[name].is_floating_point():
            try:
                torch.testing.assert_close(
                    clean_output[name], pnp_output[name],
                    rtol=0.0, atol=1e-6, equal_nan=True,
                )
            except AssertionError as error:
                raise RuntimeError(
                    f"PnP F initialization differs on identical input: {name}"
                ) from error
        elif not torch.equal(clean_output[name], pnp_output[name]):
            raise RuntimeError(
                f"PnP F initialization discrete output differs: {name}"
            )
    pnp_f.train()


def train(args: argparse.Namespace) -> Path:
    formal_contract: dict[str, Any] | None = None
    formal_protocol: dict[str, Any] | None = None
    formal_root_protocol: dict[str, Any] | None = None
    if args.formal_oracle:
        if any((
            args.diagnostic_only, args.allow_diagnostic_h,
            args.allow_mapper_h_mismatch,
        )):
            raise ValueError("formal-oracle F forbids diagnostic opt-ins")
        protocol_path, protocol = load_protocol(
            args.formal_protocol if args.formal_protocol else None
        )
        formal_protocol = protocol[args.stage]
        formal_root_protocol = protocol
        exact_names = (
            "seed", "batch_size", "device", "workers", "epochs",
            "learning_rate", "minimum_learning_rate", "warmup_updates",
            "weight_decay", "gradient_clip_norm",
        )
        require_exact_protocol_arguments(args, formal_protocol, exact_names)
        if args.max_updates != int(formal_protocol["schedule_total_updates"]):
            raise ValueError("formal dual-domain LR schedule differs from protocol")
        if args.stage == "trajectory":
            require_exact_protocol_arguments(args, formal_protocol, (
                "position_weight", "position_mse_weight", "rate_weight",
                "position_tail_weight", "position_tail_fraction",
                "huber_beta_m", "rate_huber_beta_mps", "rate_tau_floor_s",
                "macro_balance_weight", "position_macro_balance_weight",
                "switch_focal_gamma",
            ))
        else:
            require_exact_protocol_arguments(
                args, formal_protocol, (
                    "switch_weight", "macro_balance_weight",
                    "switch_focal_gamma",
                )
            )
        if args.train_limit or args.validation_limit:
            raise ValueError("formal-oracle F cannot limit train or validation")
        configure_formal_runtime(args.device, args.workers)
        formal_contract = capture_formal_contract(
            FORMAL_SOURCE_BUNDLE, protocol_path=protocol_path,
            requested_device=args.device, workers=args.workers,
        )
    elif not (
        args.diagnostic_only
        and args.allow_diagnostic_h
        and args.allow_mapper_h_mismatch
    ):
        raise ValueError("diagnostic dual-domain F requires explicit opt-ins")
    if args.stage == "trajectory":
        args.switch_weight = 0.0
    elif args.stage == "selector":
        args.position_weight = 0.0
        args.position_mse_weight = 0.0
        args.rate_weight = 0.0
        args.position_tail_weight = 0.0
    else:
        raise ValueError(f"unknown dual-domain training stage: {args.stage}")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_dataset = ObservableFuturePnPSFDataset(
        args.dataset, "train", motion_class=3, sample_limit=args.train_limit
    )
    validation_dataset = ObservableFuturePnPSFDataset(
        args.dataset, "validation", motion_class=3,
        sample_limit=args.validation_limit,
    )
    canonicalize_direction_keep_c4(train_dataset.tensors, train_dataset.pair_ids)
    canonicalize_direction_keep_c4(
        validation_dataset.tensors, validation_dataset.pair_ids
    )
    if args.formal_oracle:
        split_isolation = require_formal_split_isolation(
            train_dataset, validation_dataset
        )
    else:
        split_isolation = None
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=args.workers,
        pin_memory=device.type == "cuda", drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )

    mapper, mapper_provenance = load_frozen_pnp_mapper(args.mapper_checkpoint)
    s_model, s_provenance = load_frozen_v19(args.s_checkpoint)
    h_model, h_provenance = load_frozen_hypothesis_adapter(
        args.h_checkpoint, allow_diagnostic=not args.formal_oracle
    )
    pnp_initial_checkpoint = (
        args.pnp_init_checkpoint
        if args.pnp_init_checkpoint is not None else args.f_checkpoint
    )
    pnp_f, pnp_parent = load_observable_f_checkpoint(pnp_initial_checkpoint)
    clean_f, clean_f_parent = load_observable_f_checkpoint(args.f_checkpoint)
    if args.formal_oracle:
        assert (
            formal_contract is not None
            and formal_root_protocol is not None
        )
        assets = formal_root_protocol["assets"]
        if train_dataset.manifest_sha256 != assets["dataset_manifest_sha256"]:
            raise ValueError("formal dual-domain F dataset asset mismatch")
        require_asset_binding("frozen_s", s_provenance, assets["frozen_s"])
        require_asset_binding("clean_f", clean_f_parent, assets["clean_f"])
    if mapper_provenance["provenance"]["dataset_manifest_sha256"] != train_dataset.manifest_sha256:
        raise ValueError("mapper and dual-domain F datasets differ")
    if h_provenance["provenance"]["dataset_manifest_sha256"] != train_dataset.manifest_sha256:
        raise ValueError("H and dual-domain F datasets differ")
    if mapper_provenance["provenance"]["frozen_s"]["state_dict_sha256"] != s_provenance["state_dict_sha256"]:
        raise ValueError("mapper and supplied frozen S differ")
    h_frozen = h_provenance["provenance"]
    if h_frozen["frozen_s"]["state_dict_sha256"] != s_provenance["state_dict_sha256"]:
        raise ValueError("H and supplied frozen S differ")
    if h_frozen["frozen_f"]["state_dict_sha256"] != clean_f_parent["state_dict_sha256"]:
        raise ValueError("H and supplied F foundation differ")
    mapper_h_mismatch = {
        "h_expected": h_frozen["frozen_mapper"]["state_dict_sha256"],
        "loaded": mapper_provenance["state_dict_sha256"],
    }
    if args.formal_oracle:
        if mapper_h_mismatch["h_expected"] != mapper_h_mismatch["loaded"]:
            raise ValueError("formal-oracle F forbids mapper/H mismatch")
        mapper_parent = mapper_provenance["provenance"]
        if (
            mapper_parent.get("formal_oracle_evaluation") is not True
            or not isinstance(mapper_parent.get("formal_source_contract"), dict)
            or h_frozen.get("formal_oracle_evaluation") is not True
            or not isinstance(h_frozen.get("formal_source_contract"), dict)
        ):
            raise ValueError("formal-oracle F requires formal mapper and H")
        require_compatible_contracts(
            "mapper", mapper_parent["formal_source_contract"], formal_contract
        )
        require_compatible_contracts(
            "H", h_frozen["formal_source_contract"], formal_contract
        )
        h_checkpoint_path = Path(args.h_checkpoint).resolve()
        h_manifest = json.loads(
            (h_checkpoint_path.parent / "run_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        require_formal_checkpoint_manifest(
            "formal dual-domain H parent",
            h_checkpoint_path,
            h_manifest,
            expected_update=int(
                formal_root_protocol["hypothesis"]["fixed_final_update"]
            ),
            checkpoint_update=int(h_provenance["update"]),
        )
    elif mapper_h_mismatch["h_expected"] == mapper_h_mismatch["loaded"]:
        raise ValueError("diagnostic mapper/H mismatch opt-in was unnecessary")
    selector_parent_metrics: dict[str, Any] | None = None
    if args.stage == "trajectory":
        if args.pnp_init_checkpoint is not None:
            raise ValueError("trajectory stage must initialize directly from clean F")
        if pnp_parent != clean_f_parent:
            raise RuntimeError("clean and PnP F foundations differ at trajectory init")
    else:
        if args.pnp_init_checkpoint is None:
            raise ValueError("selector stage requires --pnp-init-checkpoint")
        selector_payload = torch.load(
            Path(args.pnp_init_checkpoint).resolve(), map_location="cpu",
            weights_only=False,
        )
        if selector_payload.get("schema_version") != RUN_SCHEMA:
            raise ValueError("selector parent is not a dual-domain PnP F checkpoint")
        selector_provenance = selector_payload.get("provenance", {})
        if selector_provenance.get("training_stage") != "trajectory":
            raise ValueError("selector parent must come from the trajectory stage")
        clean_parent = selector_provenance.get("clean_f_foundation", {})
        if clean_parent.get("state_dict_sha256") != clean_f_parent["state_dict_sha256"]:
            raise ValueError("selector parent and clean F foundation differ")
        selector_bindings = {
            "dataset": (
                selector_provenance.get("dataset_manifest_sha256"),
                train_dataset.manifest_sha256,
            ),
            "mapper": (
                selector_provenance.get("frozen_mapper", {}).get(
                    "state_dict_sha256"
                ),
                mapper_provenance["state_dict_sha256"],
            ),
            "s": (
                selector_provenance.get("frozen_s", {}).get(
                    "state_dict_sha256"
                ),
                s_provenance["state_dict_sha256"],
            ),
            "h": (
                selector_provenance.get("frozen_h", {}).get(
                    "state_dict_sha256"
                ),
                h_provenance["state_dict_sha256"],
            ),
            "clean_f": (
                clean_parent.get("state_dict_sha256"),
                clean_f_parent["state_dict_sha256"],
            ),
        }
        mismatched_bindings = [
            name for name, (parent, loaded) in selector_bindings.items()
            if parent != loaded
        ]
        if mismatched_bindings:
            raise ValueError(
                "selector parent provenance mismatch: "
                + ", ".join(mismatched_bindings)
            )
        if selector_payload.get("model_config") != pnp_f.config:
            raise ValueError("selector parent model config differs after reload")
        if args.formal_oracle:
            if (
                selector_provenance.get("formal_oracle_evaluation") is not True
                or selector_provenance.get("fixed_final_checkpoint") is not True
                or not isinstance(
                    selector_provenance.get("formal_source_contract"), dict
                )
            ):
                raise ValueError("formal selector requires a formal trajectory parent")
            require_compatible_contracts(
                "trajectory",
                selector_provenance["formal_source_contract"],
                formal_contract,
            )
            parent_manifest_path = Path(args.pnp_init_checkpoint).resolve().parent / "run_manifest.json"
            parent_manifest = json.loads(
                parent_manifest_path.read_text(encoding="utf-8")
            )
            require_formal_checkpoint_manifest(
                "formal selector trajectory parent",
                args.pnp_init_checkpoint,
                parent_manifest,
                expected_update=int(
                    formal_root_protocol["trajectory"]["fixed_final_update"]
                ),
                checkpoint_update=int(selector_payload["update"]),
            )
        selector_parent_metrics = selector_payload.get("validation")
        if not isinstance(selector_parent_metrics, dict):
            raise ValueError("selector parent lacks validation metrics")
        if (
            float(selector_parent_metrics["pnp_domain"]["conditional_position"]["p95_m"])
            >= LEGACY_JOINT_B_CONDITIONAL_P95_M
        ):
            raise ValueError("selector stage requires a trajectory parent that passed")

    for model in (mapper, s_model, h_model, clean_f):
        model.to(device).eval().requires_grad_(False)
    pnp_f.to(device).train()
    assert_independent_models(clean_f, pnp_f)
    initial_pnp_hash = state_dict_sha256(pnp_f.state_dict())
    if initial_pnp_hash != pnp_parent["state_dict_sha256"]:
        raise RuntimeError("PnP F did not initialize bit-exact from clean F")
    if args.stage == "trajectory":
        _assert_initial_output_equivalence(
            clean_f, pnp_f, mapper, s_model, h_model,
            validation_loader, device,
        )
    trainable_parameters, trainable_names, frozen_pnp_names = (
        _configure_trainable_parameters(pnp_f, args.stage)
    )
    selector_runtime_baseline: dict[str, Any] | None = None
    if args.stage == "selector":
        selector_runtime_baseline = evaluate(
            pnp_f, clean_f, mapper, s_model, h_model,
            validation_loader, device,
        )
        parent_p95 = float(
            selector_parent_metrics["pnp_domain"]["conditional_position"]["p95_m"]
        )
        runtime_p95 = float(
            selector_runtime_baseline["pnp_domain"]["conditional_position"]["p95_m"]
        )
        if abs(parent_p95 - runtime_p95) > 1e-6:
            raise RuntimeError("selector parent metrics are not reproducible")
    frozen_pnp_partition_before = _partition_hash(pnp_f, frozen_pnp_names)
    frozen_before = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
        "clean_f": state_dict_sha256(clean_f.state_dict()),
    }
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    epoch_updates = len(train_loader)
    planned_updates = args.epochs * epoch_updates
    total_updates = min(planned_updates, args.max_updates) if args.max_updates > 0 else planned_updates
    stop_update = args.max_updates
    if args.formal_oracle:
        assert formal_protocol is not None
        total_updates, stop_update = resolve_formal_schedule(
            args.max_updates, planned_updates, formal_protocol
        )
    if total_updates <= 0:
        raise ValueError("dual-domain F requires at least one update")
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    update = 0
    stop = False
    stop_reason = "epoch_limit"
    stagnant_validations = 0
    previous_best_conditional = float("inf")
    started = time.time()
    git = _git_state()
    source_path = Path(__file__).resolve()
    for epoch in range(1, args.epochs + 1):
        _set_stage_training_mode(pnp_f, args.stage)
        sums: dict[str, float] = {}
        batches = 0
        for raw_batch in train_loader:
            update += 1
            lr = _learning_rate(
                args.learning_rate, update, total_updates,
                args.warmup_updates, args.minimum_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = _to_device(raw_batch, device)
            _validate_batch_contract(batch)
            with torch.no_grad():
                composed = _pnp_composed(mapper, s_model, h_model, batch)
            optimizer.zero_grad(set_to_none=True)
            prediction = _pnp_f_forward(pnp_f, composed, batch)
            objective, components = _absolute_loss(prediction, batch, args)
            objective.backward()
            if not any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                and bool(torch.any(parameter.grad != 0))
                for parameter in trainable_parameters
            ):
                raise RuntimeError("PnP F received no finite nonzero gradient")
            if any(parameter.grad is not None for model in (mapper, s_model, h_model, clean_f) for parameter in model.parameters()):
                raise RuntimeError("a frozen dual-domain component accumulated gradients")
            if any(
                parameter.grad is not None
                for name, parameter in pnp_f.named_parameters()
                if name in frozen_pnp_names
            ):
                raise RuntimeError("a frozen PnP F stage partition accumulated gradients")
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, args.gradient_clip_norm
            )
            optimizer.step()
            for name, value in components.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach())
            batches += 1
            if stop_update > 0 and update >= stop_update:
                stop = True
                stop_reason = (
                    "fixed_final_update" if args.formal_oracle
                    else "max_updates"
                )
                break

        validate_now = (
            (stop or epoch == args.epochs)
            if args.formal_oracle else
            (epoch <= 3 or epoch % args.validation_interval == 0
             or stop or epoch == args.epochs)
        )
        if validate_now:
            metrics = evaluate(
                pnp_f, clean_f, mapper, s_model, h_model,
                validation_loader, device,
            )
            pnp_metrics = metrics["pnp_domain"]
            if args.stage == "trajectory":
                selection = (
                    float(pnp_metrics["conditional_position"]["p95_m"]),
                    float(pnp_metrics["conditional_position"]["p99_m"]),
                )
            else:
                selection = (
                    float(pnp_metrics["hard_routed_position"]["p95_m"]),
                    -float(pnp_metrics["switch_accuracy"]),
                    -float(pnp_metrics["minimum_step_switch_recall"] or 0.0),
                )
            checkpoint_name = f"epoch-{epoch:04d}-update-{update:06d}.pt"
            checkpoint_path = output / checkpoint_name
            provenance = {
                "dataset_manifest_path": str(
                    Path(args.dataset).resolve() / "dataset_manifest.json"
                ),
                "dataset_manifest_sha256": train_dataset.manifest_sha256,
                "domain_architecture": {
                    "clean": "independent frozen F branch",
                    "pnp": "independent PnP-trained F branch",
                    "routing": "explicit external observation-domain routing",
                    "shared_trainable_parameters": False,
                },
                "training_stage": args.stage,
                "formal_oracle_evaluation": bool(args.formal_oracle),
                "fixed_final_checkpoint": bool(args.formal_oracle),
                "full_chain_provenance_clean": False,
                "formal_source_contract": formal_contract,
                "train_split_audit": train_dataset.split_audit,
                "validation_split_audit": validation_dataset.split_audit,
                "split_isolation": split_isolation,
                "frozen_mapper": mapper_provenance,
                "frozen_s": s_provenance,
                "frozen_h": h_provenance,
                "clean_f_foundation": clean_f_parent,
                "pnp_f_initialization": pnp_parent,
                "selector_runtime_baseline": selector_runtime_baseline,
                "trainable_parameter_names": trainable_names,
                "frozen_pnp_parameter_names": frozen_pnp_names,
                "mapper_h_compatibility_mismatch": mapper_h_mismatch,
                "anonymous_candidate_contract_retained": True,
                "physical_id_input": False,
                "motion_class_forward_input": False,
                "diagnostic_only": not args.formal_oracle,
                "diagnostic_reasons": (
                    [
                        "oracle_association", "legacy_h_diagnostic_provenance",
                        "mapper_h_provenance_mismatch", "dirty_training_source",
                    ] if not args.formal_oracle else []
                ),
                "oracle_association": True,
                "deployable_pipeline": False,
                "test_accessed": False,
                "source_path": str(source_path),
                "source_sha256": sha256_file(source_path),
                "git": git,
            }
            checkpoint = {
                "schema_version": RUN_SCHEMA,
                "model_class": type(pnp_f).__name__,
                "model_config": pnp_f.config,
                "model": pnp_f.state_dict(),
                "epoch": epoch, "update": update,
                "validation": metrics, "selection": selection,
                "provenance": provenance,
            }
            _atomic_checkpoint(checkpoint_path, checkpoint)
            item = {
                "epoch": epoch, "update": update, "learning_rate": lr,
                "train": {
                    name: value / max(batches, 1) for name, value in sums.items()
                },
                "validation": metrics, "selection": selection,
                "checkpoint": checkpoint_name,
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
            history.append(item)
            if args.formal_oracle or best is None or selection < tuple(best["selection"]):
                best = {
                    "epoch": epoch, "update": update,
                    "path": checkpoint_name,
                    "sha256": item["checkpoint_sha256"],
                    "selection": selection, "validation": metrics,
                }
            progress = {
                "schema_version": RUN_SCHEMA,
                "status": "running" if not stop else "complete",
                "epoch": epoch, "update": update, "best": best,
                "history": history, "elapsed_s": time.time() - started,
                "train_sample_count": len(train_dataset),
                "validation_sample_count": len(validation_dataset),
                "training_arguments": vars(args),
                "model_config": pnp_f.config,
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in trainable_parameters
                ),
                "trainable_parameter_names": trainable_names,
                "frozen_pnp_parameter_names": frozen_pnp_names,
                "initial_pnp_f_state_dict_sha256": initial_pnp_hash,
                "provenance": provenance,
            }
            _atomic_json(output / "run_progress.json", progress)
            if (
                not args.formal_oracle
                and args.stage == "trajectory" and update >= 2000
            ):
                conditional_p95 = float(
                    pnp_metrics["conditional_position"]["p95_m"]
                )
                improvement = (
                    (previous_best_conditional - conditional_p95)
                    / previous_best_conditional
                    if np.isfinite(previous_best_conditional) else float("inf")
                )
                if conditional_p95 < previous_best_conditional:
                    previous_best_conditional = conditional_p95
                stagnant_validations = (
                    stagnant_validations + 1 if improvement < 0.01 else 0
                )
                if conditional_p95 >= 0.8 * MAPPER_BASELINE_CONDITIONAL_P95_M:
                    stop = True
                    stop_reason = "trajectory_failed_20pct_improvement_by_2000"
                elif stagnant_validations >= 4:
                    stop = True
                    stop_reason = "trajectory_plateau_four_validations"
        if stop:
            break
    if best is None:
        raise RuntimeError("dual-domain F produced no checkpoint")
    frozen_unchanged = {
        "mapper": state_dict_sha256(mapper.state_dict()) == frozen_before["mapper"],
        "s": state_dict_sha256(s_model.state_dict()) == frozen_before["s"],
        "h": state_dict_sha256(h_model.state_dict()) == frozen_before["h"],
        "clean_f": state_dict_sha256(clean_f.state_dict()) == frozen_before["clean_f"],
        "pnp_stage_partition": (
            _partition_hash(pnp_f, frozen_pnp_names)
            == frozen_pnp_partition_before
        ),
    }
    if not all(frozen_unchanged.values()):
        raise RuntimeError("dual-domain F changed a frozen component")
    if args.formal_oracle:
        assert formal_contract is not None and formal_protocol is not None
        require_fixed_final_state(
            f"formal {args.stage}", formal_protocol,
            update=update, stop_reason=stop_reason,
            history=history, best=best,
        )
        verify_formal_contract(formal_contract)
    manifest = json.loads(
        (output / "run_progress.json").read_text(encoding="utf-8")
    )
    manifest.update({
        "status": "complete",
        "stop_reason": stop_reason,
        "elapsed_s": time.time() - started,
        "frozen_state_hashes_unchanged": frozen_unchanged,
    })
    best_metrics = best["validation"]
    if args.stage == "trajectory":
        manifest["gate"] = {
            "beats_legacy_joint_b_344mm": (
                float(best_metrics["pnp_domain"]["conditional_position"]["p95_m"])
                < LEGACY_JOINT_B_CONDITIONAL_P95_M
            ),
            "current_p95_le_150mm": (
                float(best_metrics["current_position_error"]["p95_m"])
                <= 0.150
            ),
            "clean_and_selector_partitions_frozen": all(
                frozen_unchanged.values()
            ),
        }
        manifest["strong_gate_conditional_p95_lt_250mm"] = (
            float(best_metrics["pnp_domain"]["conditional_position"]["p95_m"])
            < 0.250
        )
    else:
        assert selector_runtime_baseline is not None
        before_pnp = selector_runtime_baseline["pnp_domain"]
        after_pnp = best_metrics["pnp_domain"]
        manifest["gate"] = {
            "conditional_output_bit_exact": (
                best_metrics["conditional_output_sha256"]
                == selector_runtime_baseline["conditional_output_sha256"]
            ),
            "upstream_input_bit_exact": (
                best_metrics["upstream_f_input_sha256"]
                == selector_runtime_baseline["upstream_f_input_sha256"]
            ),
            "hard_p95_improved": (
                float(after_pnp["hard_routed_position"]["p95_m"])
                < float(before_pnp["hard_routed_position"]["p95_m"])
            ),
            "switch_accuracy_improved": (
                float(after_pnp["switch_accuracy"])
                > float(before_pnp["switch_accuracy"])
            ),
            "minimum_step_recall_improved": (
                float(after_pnp["minimum_step_switch_recall"] or 0.0)
                > float(before_pnp["minimum_step_switch_recall"] or 0.0)
            ),
            "clean_and_trajectory_partitions_frozen": all(
                frozen_unchanged.values()
            ),
        }
    if args.formal_oracle:
        gates = load_protocol(
            args.formal_protocol if args.formal_protocol else None
        )[1]["presealed_gates"]
        pnp = best_metrics["pnp_domain"]
        clean = best_metrics["clean_frozen_domain"]
        formal_gates = {
            "pnp_conditional_p95": (
                float(pnp["conditional_position"]["p95_m"])
                <= gates["pnp_conditional_p95_m_max"]
            ),
            "pnp_conditional_p99": (
                float(pnp["conditional_position"]["p99_m"])
                <= gates["pnp_conditional_p99_m_max"]
            ),
            "current_p95": (
                float(best_metrics["current_position_error"]["p95_m"])
                <= gates["current_p95_m_max"]
            ),
            "clean_conditional_p95": (
                float(clean["conditional_position"]["p95_m"])
                <= gates["clean_conditional_p95_m_max"]
            ),
            "clean_hard_p95": (
                float(clean["hard_routed_position"]["p95_m"])
                <= gates["clean_hard_p95_m_max"]
            ),
            "clean_switch_accuracy": (
                float(clean["switch_accuracy"])
                >= gates["clean_switch_accuracy_min"]
            ),
            "frozen_components": all(frozen_unchanged.values()),
        }
        if args.stage == "selector":
            formal_gates.update({
                "selector_hard_p95": (
                    float(pnp["hard_routed_position"]["p95_m"])
                    <= gates["selector_hard_p95_m_max"]
                ),
                "selector_hard_p99": (
                    float(pnp["hard_routed_position"]["p99_m"])
                    <= gates["selector_hard_p99_m_max"]
                ),
                "switch_accuracy": (
                    float(pnp["switch_accuracy"])
                    >= gates["switch_accuracy_min"]
                ),
                "minimum_step_recall": (
                    float(pnp["minimum_step_switch_recall"] or 0.0)
                    >= gates["minimum_step_recall_min"]
                ),
                "conditional_output_bit_exact": bool(
                    manifest["gate"]["conditional_output_bit_exact"]
                ),
                "upstream_input_bit_exact": bool(
                    manifest["gate"]["upstream_input_bit_exact"]
                ),
            })
        manifest["formal_gates"] = formal_gates
        manifest["formal_gate_passed"] = all(formal_gates.values())
    manifest["gate_passed"] = all(manifest["gate"].values())
    _atomic_json(output / "run_manifest.json", manifest)
    return output / "run_manifest.json"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", required=True)
    result.add_argument("--mapper-checkpoint", required=True)
    result.add_argument("--s-checkpoint", required=True)
    result.add_argument("--h-checkpoint", required=True)
    result.add_argument("--f-checkpoint", required=True)
    result.add_argument(
        "--pnp-init-checkpoint",
        help="trajectory checkpoint used only to initialize selector stage",
    )
    result.add_argument("--output", required=True)
    result.add_argument(
        "--stage", choices=("trajectory", "selector"), default="trajectory",
    )
    result.add_argument("--diagnostic-only", action="store_true")
    result.add_argument("--allow-diagnostic-h", action="store_true")
    result.add_argument("--allow-mapper-h-mismatch", action="store_true")
    result.add_argument("--formal-oracle", action="store_true")
    result.add_argument("--formal-protocol", default="")
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int, default=20260727)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--max-updates", type=int, default=10000)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    result.add_argument("--warmup-updates", type=int, default=500)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--gradient-clip-norm", type=float, default=1.0)
    result.add_argument("--huber-beta-m", type=float, default=0.01)
    result.add_argument("--switch-weight", type=float, default=1.0)
    result.add_argument("--position-weight", type=float, default=50.0)
    result.add_argument("--position-mse-weight", type=float, default=200.0)
    result.add_argument("--rate-weight", type=float, default=0.005)
    result.add_argument("--rate-huber-beta-mps", type=float, default=0.02)
    result.add_argument("--rate-tau-floor-s", type=float, default=0.05)
    result.add_argument("--position-tail-weight", type=float, default=0.2)
    result.add_argument("--position-tail-fraction", type=float, default=0.1)
    result.add_argument("--macro-balance-weight", type=float, default=0.25)
    result.add_argument("--position-macro-balance-weight", type=float, default=0.25)
    result.add_argument("--switch-focal-gamma", type=float, default=2.0)
    result.add_argument("--validation-interval", type=int, default=5)
    result.add_argument("--train-limit", type=int, default=0)
    result.add_argument("--validation-limit", type=int, default=0)
    return result


def main() -> None:
    args = parser().parse_args()
    if min(
        args.learning_rate, args.minimum_learning_rate,
        args.gradient_clip_norm, args.huber_beta_m,
        args.rate_huber_beta_mps, args.rate_tau_floor_s,
    ) <= 0:
        raise ValueError("positive dual-domain F configuration required")
    if min(
        args.switch_weight, args.position_weight, args.position_mse_weight,
        args.rate_weight, args.position_tail_weight,
    ) < 0:
        raise ValueError("dual-domain F weights cannot be negative")
    if not 0 < args.position_tail_fraction <= 1:
        raise ValueError("dual-domain F tail fraction must be within (0,1]")
    print(train(args))


if __name__ == "__main__":
    main()
