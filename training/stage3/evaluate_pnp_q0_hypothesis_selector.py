"""Strict held-out evaluation for a trained A3 PnP switch-only selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .evaluate_pnp_q0_hypothesis_sf import canonicalize_direction_keep_c4
from .observable_future_pnp_ab import (
    ObservableFuturePnPSFDataset,
    state_dict_sha256,
)
from .train_causal_physical_ab import _to_device
from .train_observable_future_dedicated_selector import _module_state_sha256
from .train_pnp_q0_hypothesis_selector import (
    PNP_SELECTOR_TRAINABLE_PREFIXES,
    evaluate_selector,
    load_frozen_pnp_selector,
)


EVALUATION_SCHEMA = "stage3-pnp-q0-hypothesis-selector-evaluation-v1"


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Path:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite selector evaluation: {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    system, mapper, s_model, h_model, checkpoint = load_frozen_pnp_selector(
        args.selector_checkpoint
    )
    dataset = ObservableFuturePnPSFDataset(
        args.dataset, "validation", motion_class=3
    )
    canonicalize_direction_keep_c4(dataset.tensors, dataset.pair_ids)
    provenance = checkpoint["provenance"]
    if dataset.manifest_sha256 != provenance["dataset_manifest_sha256"]:
        raise ValueError("selector and held-out dataset manifests differ")
    if not torch.equal(
        dataset.tensors["candidate_step"],
        dataset.tensors["pnp_candidate_step"],
    ):
        raise ValueError("selector evaluation requires clean/PnP step identity")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    for model in (system, mapper, s_model, h_model):
        model.eval().to(device)
    before = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
        "trajectory_f": _module_state_sha256(system.trajectory_model),
        "selector_frozen": _module_state_sha256(
            system.selector_model,
            prefixes=PNP_SELECTOR_TRAINABLE_PREFIXES,
            invert=True,
        ),
        "selector_switch_heads": _module_state_sha256(
            system.selector_model, prefixes=PNP_SELECTOR_TRAINABLE_PREFIXES
        ),
    }
    metrics = evaluate_selector(
        system, mapper, s_model, h_model, loader, device
    )
    after = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
        "trajectory_f": _module_state_sha256(system.trajectory_model),
        "selector_frozen": _module_state_sha256(
            system.selector_model,
            prefixes=PNP_SELECTOR_TRAINABLE_PREFIXES,
            invert=True,
        ),
        "selector_switch_heads": _module_state_sha256(
            system.selector_model, prefixes=PNP_SELECTOR_TRAINABLE_PREFIXES
        ),
    }
    if after != before:
        raise RuntimeError("selector evaluation changed a frozen state hash")
    if before["selector_switch_heads"] != checkpoint["selector_state_sha256"]:
        raise RuntimeError("selector evaluation loaded the wrong switch-head state")
    saved_validation = checkpoint.get("validation") or {}
    if metrics["conditional_position"] != saved_validation.get(
        "conditional_position"
    ):
        raise RuntimeError("selector reload changed frozen conditional metrics")
    result = {
        "schema_version": EVALUATION_SCHEMA,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "validation_sample_count": len(dataset),
        "selector_checkpoint": checkpoint,
        "metrics": metrics,
        "state_hashes_before": before,
        "state_hashes_unchanged": True,
        "conditional_metrics_reload_exact": True,
        "test_accessed": provenance["test_accessed"],
        "oracle_association": provenance["oracle_association"],
        "deployable_pipeline": provenance["deployable_pipeline"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--selector-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    print(evaluate(parser.parse_args()))


if __name__ == "__main__":
    main()
