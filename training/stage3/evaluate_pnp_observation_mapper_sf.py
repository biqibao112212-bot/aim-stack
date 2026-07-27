"""Evaluate the true-A observation mapper through frozen S and frozen F."""

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
    f_forward,
    load_observable_f_checkpoint,
    sha256_file,
    state_dict_sha256,
)
from .pnp_observation_mapper import (
    CausalPnPObservationMapper,
    compose_corrected_observations,
)


EVALUATION_SCHEMA = "stage3-pnp-observation-mapper-sf-evaluation-v1"


def _stats(values: list[np.ndarray]) -> dict[str, float | int]:
    merged = np.concatenate(values).astype(np.float64, copy=False)
    if not merged.size or not np.isfinite(merged).all():
        raise ValueError("true-A evaluation metric is empty or non-finite")
    return {
        "count": int(merged.size),
        "mean_m": float(merged.mean()),
        "p50_m": float(np.quantile(merged, 0.50)),
        "p95_m": float(np.quantile(merged, 0.95)),
        "p99_m": float(np.quantile(merged, 0.99)),
        "max_m": float(merged.max()),
    }


def _load_mapper(path: Path) -> tuple[CausalPnPObservationMapper, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != "stage3-pnp-observation-mapper-run-v1"
        or payload.get("model_class") != "CausalPnPObservationMapper"
    ):
        raise ValueError("PnP observation mapper checkpoint schema mismatch")
    config = payload["model_config"]
    model = CausalPnPObservationMapper(
        torch.tensor(config["position_mean"], dtype=torch.float32),
        torch.tensor(config["position_std"], dtype=torch.float32),
        channels=int(config["channels"]), dropout=float(config["dropout"]),
        history_events=int(config["history_events"]),
        history_scale_s=float(config["history_scale_s"]),
    )
    if model.config != config:
        raise ValueError("PnP mapper checkpoint configuration is not reproducible")
    model.load_state_dict(payload["model"], strict=True)
    return model, {
        "path": str(path), "sha256": sha256_file(path),
        "state_dict_sha256": state_dict_sha256(model.state_dict()),
        "epoch": int(payload["epoch"]), "update": int(payload["update"]),
        "provenance": payload["provenance"],
    }


def _restore_original_orientation(
    batch: dict[str, torch.Tensor], pair_ids: tuple[str, ...]
) -> None:
    slot_fields = {
        "pnp_s_obs_m": 2, "pnp_s_obs_mask": 2, "pnp_s_primary_mask": 2,
        "pnp_s_truth_q0_m": 1, "clean_s_obs_m": 2,
        "clean_s_obs_mask": 2, "clean_s_primary_mask": 2,
    }
    signed_fields = (
        "history_switch_step", "pnp_history_switch_step",
        "candidate_step", "pnp_candidate_step", "target_switch_count",
        "pnp_s_switch_step", "clean_s_switch_step",
    )
    reverse_index = torch.tensor((0, 3, 2, 1), device=next(iter(batch.values())).device)
    for row, pair_id in enumerate(pair_ids):
        value = int(pair_id[:16], 16)
        shift = value % 4
        reverse = bool((value >> 2) & 1)
        for name, batch_axis in slot_fields.items():
            item = batch[name][row]
            item_axis = batch_axis - 1
            if reverse:
                item = torch.index_select(item, item_axis, reverse_index)
            batch[name][row] = torch.roll(item, -shift, dims=item_axis)
        if reverse:
            for name in signed_fields:
                batch[name][row].mul_(-1)


def _s_forward(
    model: torch.nn.Module,
    obs_m: torch.Tensor,
    obs_mask: torch.Tensor,
    primary_mask: torch.Tensor,
    event_mask: torch.Tensor,
    event_time_s: torch.Tensor,
    switch_step: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mask = obs_mask.to(torch.bool)
    normalized = (obs_m - model.position_mean) / model.position_std
    normalized = torch.where(mask.unsqueeze(-1), normalized, torch.zeros_like(normalized))
    return model(
        normalized, mask, primary_mask.to(torch.bool), event_mask.to(torch.bool),
        event_time_s, switch_step,
    )


def _new_collector() -> dict[str, list[np.ndarray] | int]:
    return {"conditional": [], "hard": [], "correct": [], "current": []}


def _add_prediction(
    collector: dict[str, list[np.ndarray] | int],
    prediction: dict[str, torch.Tensor],
    current: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> None:
    candidate_step = batch["candidate_step"].to(torch.long)
    target_step = batch["target_switch_count"].to(torch.long)
    mask = batch["target_query_mask"].to(torch.bool) & (batch["tau_s"] > 0)
    matches = candidate_step[:, None, :] == target_step[:, :, None]
    true_row = matches.to(torch.long).argmax(dim=-1)
    true_gather = true_row[:, :, None, None].expand(-1, -1, 1, 3)
    hard_row = prediction["selected_candidate_row"]
    hard_gather = hard_row[:, :, None, None].expand(-1, -1, 1, 3)
    truth = batch["current_position_m"][:, None, :] + batch["target_visible_delta_m"]
    conditional_error = torch.linalg.vector_norm(
        prediction["conditional_position_m"].gather(2, true_gather).squeeze(2)
        - truth, dim=-1,
    )
    hard_error = torch.linalg.vector_norm(
        prediction["conditional_position_m"].gather(2, hard_gather).squeeze(2)
        - truth, dim=-1,
    )
    selected_step = candidate_step.gather(1, hard_row)
    assert isinstance(collector["conditional"], list)
    assert isinstance(collector["hard"], list)
    assert isinstance(collector["correct"], list)
    assert isinstance(collector["current"], list)
    collector["conditional"].append(conditional_error[mask].cpu().numpy())
    collector["hard"].append(hard_error[mask].cpu().numpy())
    collector["correct"].append((selected_step[mask] == target_step[mask]).cpu().numpy())
    collector["current"].append(
        torch.linalg.vector_norm(
            current - batch["current_position_m"], dim=-1
        ).cpu().numpy()
    )


def _finish(collector: dict[str, list[np.ndarray] | int]) -> dict[str, Any]:
    conditional = collector["conditional"]
    hard = collector["hard"]
    correct = collector["correct"]
    current = collector["current"]
    assert isinstance(conditional, list) and isinstance(hard, list)
    assert isinstance(correct, list) and isinstance(current, list)
    return {
        "conditional_position": _stats(conditional),
        "hard_position": _stats(hard),
        "switch_accuracy": float(np.concatenate(correct).mean()),
        "current_position": _stats(current),
    }


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Path:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite true-A evaluation: {output}")
    device = torch.device(args.device)
    dataset = ObservableFuturePnPSFDataset(
        args.dataset, "validation", motion_class=3
    )
    mapper, mapper_provenance = _load_mapper(Path(args.mapper_checkpoint).resolve())
    s_model, s_provenance = load_frozen_v19(args.s_checkpoint)
    f_model, f_provenance = load_observable_f_checkpoint(args.f_checkpoint)
    if (
        mapper_provenance["provenance"]["dataset_manifest_sha256"]
        != dataset.manifest_sha256
    ):
        raise ValueError("PnP mapper and S/F evaluation datasets differ")
    for model in (mapper, s_model, f_model):
        model.eval().requires_grad_(False).to(device)
    s_state_before = state_dict_sha256(s_model.state_dict())
    f_state_before = state_dict_sha256(f_model.state_dict())

    collectors = {
        name: _new_collector()
        for name in ("raw_pnp_frozen_sf", "mapper_frozen_sf", "clean_frozen_sf", "oracle_clean_f")
    }
    mapping_raw: list[np.ndarray] = []
    mapping_corrected: list[np.ndarray] = []
    batch_size = args.batch_size
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch = {name: value[start:end].to(device) for name, value in dataset.tensors.items()}
        _restore_original_orientation(batch, dataset.pair_ids[start:end])
        mapped = mapper(
            batch["pnp_s_obs_m"], batch["pnp_s_obs_mask"],
            batch["pnp_s_event_time_s"], batch["pnp_s_event_mask"],
        )
        direct_mask = batch["pnp_s_obs_mask"].to(torch.bool)
        mapping_raw.append(torch.linalg.vector_norm(
            batch["pnp_s_obs_m"] - batch["clean_s_obs_m"], dim=-1
        )[direct_mask].cpu().numpy())
        mapping_corrected.append(torch.linalg.vector_norm(
            mapped["corrected_obs_m"] - batch["clean_s_obs_m"], dim=-1
        )[direct_mask].cpu().numpy())

        for label, obs, obs_mask, primary_mask, event_time, switch in (
            ("raw_pnp_frozen_sf", batch["pnp_s_obs_m"], batch["pnp_s_obs_mask"],
             batch["pnp_s_primary_mask"], batch["pnp_s_event_time_s"], batch["pnp_s_switch_step"]),
            ("mapper_frozen_sf", mapped["corrected_obs_m"], batch["pnp_s_obs_mask"],
             batch["pnp_s_primary_mask"], batch["pnp_s_event_time_s"], batch["pnp_s_switch_step"]),
            ("clean_frozen_sf", batch["clean_s_obs_m"], batch["clean_s_obs_mask"],
             batch["clean_s_primary_mask"], batch["clean_s_event_time_s"], batch["clean_s_switch_step"]),
        ):
            s_output = _s_forward(
                s_model, obs, obs_mask, primary_mask,
                batch["pnp_s_event_mask"], event_time, switch,
            )
            composed = compose_corrected_observations(
                s_output, obs, obs_mask, primary_mask, batch["candidate_step"]
            )
            prediction = f_forward(
                f_model, batch,
                history_position_rel_m=composed["history_position_rel_m"],
                current_position_m=composed["current_position_m"],
                candidate_relation_m=composed["candidate_relation_m"],
                candidate_confidence=composed["candidate_confidence"],
                detach_observation_inputs=False,
            )
            _add_prediction(
                collectors[label], prediction, composed["current_position_m"], batch
            )

        oracle_prediction = f_forward(f_model, batch)
        _add_prediction(
            collectors["oracle_clean_f"], oracle_prediction,
            batch["current_position_m"], batch,
        )

    result = {
        "schema_version": EVALUATION_SCHEMA,
        "mapper": mapper_provenance,
        "frozen_s": s_provenance,
        "frozen_f": f_provenance,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "validation_sample_count": len(dataset),
        "eligible_query_contract": "target_query_mask and tau_s > 0",
        "direct_mapping": {
            "raw": _stats(mapping_raw),
            "corrected": _stats(mapping_corrected),
        },
        "pipelines": {name: _finish(value) for name, value in collectors.items()},
        "frozen_s_verified_unchanged": (
            state_dict_sha256(s_model.state_dict()) == s_state_before
        ),
        "frozen_f_verified_unchanged": (
            state_dict_sha256(f_model.state_dict()) == f_state_before
        ),
        "test_accessed": False,
        "oracle_association": True,
        "deployable_pipeline": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mapper-checkpoint", required=True)
    parser.add_argument("--s-checkpoint", required=True)
    parser.add_argument("--f-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    print(evaluate(parser.parse_args()))


if __name__ == "__main__":
    main()
