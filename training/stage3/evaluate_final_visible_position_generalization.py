"""Evaluate a frozen final-position refiner on disjoint newly collected data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .cyclic_future_foundation import load_frozen_v19
from .final_visible_position_refiner import (
    FinalVisiblePositionRefiner,
    FrozenVisibleSystemFinalRefiner,
)
from .observable_future_pnp_ab import (
    ObservableFuturePnPSFDataset,
    canonicalize_direction_keep_c4,
    sha256_file,
    state_dict_sha256,
)
from .pnp_q0_hypothesis_adapter import (
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .train_causal_physical_ab import _git_state
from .train_final_visible_position_refiner import (
    RUN_SCHEMA,
    evaluate,
    load_frozen_v66,
)
from .train_joint_visible_future import _build_cache
from .train_pnp_window_mapper_distillation import _atomic_json


EVALUATION_SCHEMA = "stage3-final-position-disjoint-generalization-v1"
MOTION_CLASSES = {2: "rotation", 3: "combined"}


def _manifest_sessions(manifest: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for shard in manifest.get("shards", []):
        result.update(str(value) for value in shard.get("session_ids", []))
    return result


def _load_system(
    refiner_checkpoint: str | Path,
    trajectory_checkpoint: str | Path,
    selector_checkpoint: str | Path,
) -> tuple[FrozenVisibleSystemFinalRefiner, dict[str, Any]]:
    path = Path(refiner_checkpoint).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != RUN_SCHEMA:
        raise ValueError("generalization evaluation requires a V67 checkpoint")
    provenance = payload.get("provenance", {})
    parent_dataset_sha = provenance.get(
        "dataset_manifest_sha256",
    )
    if not parent_dataset_sha:
        raise ValueError("V67 checkpoint has no parent dataset binding")
    base, base_provenance = load_frozen_v66(
        trajectory_checkpoint, selector_checkpoint,
        dataset_manifest_sha256=str(parent_dataset_sha),
    )
    config = payload.get("model_config", {}).get("refiner", {})
    refiner = FinalVisiblePositionRefiner(
        frozen_context_features=int(config["frozen_context_features"]),
        channels=int(config["channels"]),
        dropout=float(config["dropout"]),
        position_scale_m=float(config["position_scale_m"]),
        trained_horizon_s=float(config["trained_horizon_s"]),
        maximum_absolute_step=int(config["maximum_absolute_step"]),
        maximum_residual_m=float(config["maximum_residual_m"]),
    )
    system = FrozenVisibleSystemFinalRefiner(base, refiner)
    if system.config != payload.get("model_config"):
        raise ValueError("V67 checkpoint config is not exactly reproducible")
    if (
        state_dict_sha256(system.base.state_dict())
        != payload.get("frozen_base_state_dict_sha256")
    ):
        raise ValueError("V67 frozen base binding differs")
    system.refiner.load_state_dict(payload["refiner"], strict=True)
    refiner_hash = state_dict_sha256(system.refiner.state_dict())
    if refiner_hash != payload.get("refiner_state_dict_sha256"):
        raise ValueError("V67 refiner state hash differs")
    system.eval().requires_grad_(False)
    return system, {
        "path": str(path),
        "sha256": sha256_file(path),
        "epoch": int(payload["epoch"]),
        "update": int(payload["update"]),
        "refiner_state_dict_sha256": refiner_hash,
        "frozen_base_state_dict_sha256": state_dict_sha256(
            system.base.state_dict(),
        ),
        "provenance": provenance,
        "v66": base_provenance,
    }


def _datasets_for_motion(
    dataset_path: Path, motion_class: int,
) -> tuple[list[Dataset], dict[str, Any]]:
    datasets: list[Dataset] = []
    split_counts: dict[str, int] = {}
    session_ids: set[str] = set()
    for split in ("train", "validation"):
        try:
            dataset = ObservableFuturePnPSFDataset(
                dataset_path, split, motion_class=motion_class,
                allow_diagnostic=False,
            )
        except ValueError as error:
            if "has no common usable samples" not in str(error):
                raise
            split_counts[split] = 0
            continue
        canonicalize_direction_keep_c4(dataset.tensors, dataset.pair_ids)
        datasets.append(dataset)
        split_counts[split] = len(dataset)
        session_ids.update(dataset.session_ids)
    if not datasets:
        raise ValueError(
            f"new dataset has no common usable motion class {motion_class}"
        )
    return datasets, {
        "motion_class": motion_class,
        "motion_name": MOTION_CLASSES[motion_class],
        "split_sample_counts": split_counts,
        "sample_count": sum(split_counts.values()),
        "eligible_session_count": len(session_ids),
        "eligible_session_ids": sorted(session_ids),
    }


def _source_coverage(
    sf_manifest: dict[str, Any], raw_manifest_path: Path,
) -> dict[str, Any]:
    records = [
        json.loads(line) for line in raw_manifest_path.read_text(
            encoding="utf-8",
        ).splitlines() if line.strip()
    ]
    mode_by_session = {
        str(record["session_id"]): str(record["mode"]) for record in records
    }
    grouped: dict[str, dict[str, int]] = {}
    for shard in sf_manifest["shards"]:
        session_id = str(shard["session_ids"][0])
        mode = mode_by_session[session_id]
        values = grouped.setdefault(mode, {
            "session_count": 0,
            "eligible_session_count": 0,
            "physical_window_count": 0,
            "pnp_sf_common_usable_count": 0,
        })
        common = int(shard["pnp_sf_common_usable_count"])
        values["session_count"] += 1
        values["eligible_session_count"] += int(common > 0)
        values["physical_window_count"] += int(shard["sample_count"])
        values["pnp_sf_common_usable_count"] += common
    for values in grouped.values():
        values["pnp_sf_common_usable_fraction"] = (
            values["pnp_sf_common_usable_count"]
            / values["physical_window_count"]
        )
    return {
        "train_validation_only": True,
        "test_accessed": False,
        "overall": {
            "session_count": sum(
                int(values["session_count"]) for values in grouped.values()
            ),
            "eligible_session_count": sum(
                int(values["eligible_session_count"])
                for values in grouped.values()
            ),
            "physical_window_count": int(sf_manifest["sample_count"]),
            "pnp_sf_common_usable_count": int(
                sf_manifest["pnp_sf_common_usable_count"]
            ),
            "pnp_sf_common_usable_fraction": float(
                sf_manifest["pnp_sf_common_usable_fraction"]
            ),
        },
        "by_motion": grouped,
    }


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite generalization evaluation: {output}"
        )
    output.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")

    dataset_path = Path(args.dataset).resolve()
    dataset_manifest_path = dataset_path / "dataset_manifest.json"
    dataset_manifest = json.loads(
        dataset_manifest_path.read_text(encoding="utf-8"),
    )
    if not bool(dataset_manifest.get("qualification_passed", False)):
        raise ValueError("new generalization dataset is not qualified")
    if bool(dataset_manifest.get("test_accessed", True)):
        raise ValueError("new generalization dataset accessed test")

    system, system_provenance = _load_system(
        args.refiner_checkpoint, args.trajectory_checkpoint,
        args.selector_checkpoint,
    )
    original_manifest_path = Path(
        str(system_provenance["provenance"]["dataset_manifest_path"]),
    ).resolve()
    original_manifest = json.loads(
        original_manifest_path.read_text(encoding="utf-8"),
    )
    original_sessions = _manifest_sessions(original_manifest)
    new_sessions = _manifest_sessions(dataset_manifest)
    overlap = sorted(original_sessions & new_sessions)
    if overlap:
        raise ValueError(
            "generalization sessions overlap the original V67 dataset: "
            + ", ".join(overlap)
        )
    if sha256_file(dataset_manifest_path) == sha256_file(original_manifest_path):
        raise ValueError("generalization and original dataset manifests match")

    mapper, mapper_provenance = load_frozen_pnp_mapper(
        args.mapper_checkpoint,
    )
    s_model, s_provenance = load_frozen_v19(args.s_checkpoint)
    h_model, h_provenance = load_frozen_hypothesis_adapter(
        args.h_checkpoint, allow_diagnostic=True,
    )
    expected = system_provenance["provenance"]
    bindings = {
        "mapper": (
            expected.get("mapper", {}).get("state_dict_sha256"),
            mapper_provenance["state_dict_sha256"],
        ),
        "s": (
            expected.get("s", {}).get("state_dict_sha256"),
            s_provenance["state_dict_sha256"],
        ),
        "h": (
            expected.get("h", {}).get("state_dict_sha256"),
            h_provenance["state_dict_sha256"],
        ),
    }
    mismatch = [name for name, values in bindings.items() if values[0] != values[1]]
    if mismatch:
        raise ValueError(
            "generalization upstream binding differs: " + ", ".join(mismatch)
        )
    for frozen in (mapper, s_model, h_model, system):
        frozen.to(device).eval().requires_grad_(False)

    started = time.time()
    initial_system_hash = state_dict_sha256(system.state_dict())
    all_source_datasets: list[Dataset] = []
    per_motion: dict[str, Any] = {}
    cache_manifests: dict[str, Any] = {}
    for motion_class, motion_name in MOTION_CLASSES.items():
        datasets, dataset_audit = _datasets_for_motion(
            dataset_path, motion_class,
        )
        all_source_datasets.extend(datasets)
        source: Dataset = (
            datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        )
        cache, cache_manifest = _build_cache(
            source, mapper, s_model, h_model,
            device=device, batch_size=args.cache_batch_size,
        )
        loader = DataLoader(
            cache, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=device.type == "cuda",
        )
        metrics, _ = evaluate(system, loader, device)
        per_motion[motion_name] = {
            "dataset": dataset_audit,
            "metrics": metrics,
        }
        cache_manifests[motion_name] = cache_manifest

    all_source: Dataset = ConcatDataset(all_source_datasets)
    all_cache, all_cache_manifest = _build_cache(
        all_source, mapper, s_model, h_model,
        device=device, batch_size=args.cache_batch_size,
    )
    all_loader = DataLoader(
        all_cache, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    overall_metrics, queries = evaluate(
        system, all_loader, device, collect_queries=True,
    )
    assert queries is not None
    np.savez_compressed(output / "generalization_queries.npz", **queries)

    raw_manifest_path = Path(args.raw_manifest).resolve()
    source_coverage = _source_coverage(
        dataset_manifest, raw_manifest_path,
    )
    reference_path = Path(args.reference_manifest).resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    old_combined = reference["final_validation"]["final_position"]
    new_combined = per_motion["combined"]["metrics"]["final_position"]
    combined_gap = {
        name: float(new_combined[name] - old_combined[name])
        for name in ("mean_m", "p50_m", "p95_m", "p99_m")
    }
    final_system_hash = state_dict_sha256(system.state_dict())
    result = {
        "schema_version": EVALUATION_SCHEMA,
        "status": "complete",
        "evaluation_only": True,
        "model_updated": False,
        "elapsed_s": time.time() - started,
        "test_accessed": False,
        "new_dataset": {
            "path": str(dataset_path),
            "manifest_path": str(dataset_manifest_path),
            "manifest_sha256": sha256_file(dataset_manifest_path),
            "raw_manifest_path": str(raw_manifest_path),
            "raw_manifest_sha256": sha256_file(raw_manifest_path),
            "original_v67_manifest_sha256": sha256_file(
                original_manifest_path,
            ),
            "session_overlap_count": len(overlap),
            "session_overlap": overlap,
            "source_coverage": source_coverage,
        },
        "model": system_provenance,
        "upstream": {
            "mapper": mapper_provenance,
            "s": s_provenance,
            "h": h_provenance,
        },
        "cache": {
            "per_motion": cache_manifests,
            "overall": all_cache_manifest,
        },
        "per_motion": per_motion,
        "overall": overall_metrics,
        "old_validation_reference": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
            "combined_final_position": old_combined,
        },
        "new_combined_minus_old_validation_m": combined_gap,
        "queries": str(output / "generalization_queries.npz"),
        "frozen_system_initial_state_dict_sha256": initial_system_hash,
        "frozen_system_final_state_dict_sha256": final_system_hash,
        "frozen_system_verified_unchanged": initial_system_hash == final_system_hash,
        "git": _git_state(),
        "source_path": str(Path(__file__).resolve()),
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    _atomic_json(output / "generalization_manifest.json", result)
    return output / "generalization_manifest.json"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", required=True)
    result.add_argument("--raw-manifest", required=True)
    result.add_argument("--mapper-checkpoint", required=True)
    result.add_argument("--s-checkpoint", required=True)
    result.add_argument("--h-checkpoint", required=True)
    result.add_argument("--trajectory-checkpoint", required=True)
    result.add_argument("--selector-checkpoint", required=True)
    result.add_argument("--refiner-checkpoint", required=True)
    result.add_argument("--reference-manifest", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--device", default="cuda")
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--cache-batch-size", type=int, default=256)
    return result


def main() -> None:
    args = parser().parse_args()
    if min(args.batch_size, args.cache_batch_size) <= 0:
        raise ValueError("generalization evaluation batch sizes must be positive")
    print(run(args))


if __name__ == "__main__":
    main()
