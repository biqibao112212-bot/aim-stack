"""Shared dataset and differentiable composition for the paired PnP A/B run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .build_observable_future_pnp_sf_upper_bound_dataset import (
    LEGACY_SCHEMA_VERSION,
    SCHEMA_VERSION,
)
from .observable_future_model import (
    AnonymousCandidateFutureExpert,
    MaskedCausalResidualBlock,
)
from .split_audit import build_split_audit


TENSOR_FIELDS = (
    "history_position_rel_m", "history_time_s", "history_dt_s",
    "history_switch_step", "history_mask", "current_position_m",
    "candidate_relation_m", "candidate_step", "candidate_mask",
    "candidate_confidence", "tau_s", "target_switch_count",
    "target_visible_delta_m", "target_query_mask", "pnp_history_position_rel_m",
    "pnp_history_time_s", "pnp_history_dt_s", "pnp_history_switch_step",
    "pnp_history_mask", "pnp_current_position_m", "pnp_candidate_relation_m",
    "pnp_candidate_step", "pnp_candidate_mask", "pnp_candidate_confidence",
    "pnp_tau_s", "pnp_s_obs_m", "pnp_s_obs_mask", "pnp_s_primary_mask",
    "pnp_s_event_mask", "pnp_s_event_time_s", "pnp_s_switch_step",
    "pnp_s_truth_q0_m", "clean_s_obs_m", "clean_s_obs_mask",
    "clean_s_primary_mask", "clean_s_event_mask", "clean_s_event_time_s",
    "clean_s_switch_step", "motion_class",
)

_DIRECTION_SLOT_FIELDS = {
    "pnp_s_obs_m": 2, "pnp_s_obs_mask": 2, "pnp_s_primary_mask": 2,
    "pnp_s_truth_q0_m": 1, "clean_s_obs_m": 2,
    "clean_s_obs_mask": 2, "clean_s_primary_mask": 2,
}
_DIRECTION_SIGNED_FIELDS = (
    "history_switch_step", "pnp_history_switch_step",
    "candidate_step", "pnp_candidate_step", "target_switch_count",
    "pnp_s_switch_step", "clean_s_switch_step",
)


def canonicalize_direction_keep_c4(
    batch: dict[str, torch.Tensor], pair_ids: tuple[str, ...]
) -> None:
    """Undo only reflection while retaining each window-local C4 origin."""
    if not batch:
        raise ValueError("direction canonicalization requires tensor fields")
    batch_size = int(next(iter(batch.values())).shape[0])
    if len(pair_ids) != batch_size:
        raise ValueError("direction pair IDs do not match batch size")
    reverse_index = torch.tensor(
        (0, 3, 2, 1), device=next(iter(batch.values())).device
    )
    for row, pair_id in enumerate(pair_ids):
        reverse = bool((int(pair_id[:16], 16) >> 2) & 1)
        if not reverse:
            continue
        for name, batch_axis in _DIRECTION_SLOT_FIELDS.items():
            if name not in batch:
                continue
            item_axis = batch_axis - 1
            batch[name][row] = torch.index_select(
                batch[name][row], item_axis, reverse_index
            )
        for name in _DIRECTION_SIGNED_FIELDS:
            if name in batch:
                batch[name][row].mul_(-1)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class ObservableFuturePnPSFDataset(Dataset):
    """Materialize one motion-class/common-coverage split exactly once."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        *,
        motion_class: int = 3,
        sample_limit: int = 0,
        allow_diagnostic: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        if split not in {"train", "validation"}:
            raise ValueError("paired PnP A/B permits train or validation only")
        if motion_class not in {1, 2, 3}:
            raise ValueError("motion_class must be translation, rotation, or combined")
        manifest_path = self.dataset_dir / "dataset_manifest.json"
        self.manifest_sha256 = sha256_file(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") not in {
            SCHEMA_VERSION, LEGACY_SCHEMA_VERSION,
        }:
            raise ValueError("paired PnP S/F dataset schema mismatch")
        if bool(self.manifest.get("test_accessed", True)):
            raise ValueError("paired PnP S/F dataset accessed test")
        if not bool(self.manifest.get("qualification_passed", False)) and not allow_diagnostic:
            raise ValueError("formal PnP A/B requires a qualified complete dataset")
        arrays_by_name: dict[str, list[np.ndarray]] = {name: [] for name in TENSOR_FIELDS}
        pair_ids: list[np.ndarray] = []
        session_ids: list[np.ndarray] = []
        t0_values: list[np.ndarray] = []
        split_shards: set[str] = set()
        for item in self.manifest["shards"]:
            if str(item["split"]) != split:
                continue
            path = self.dataset_dir / Path(str(item["path"]).replace("\\", "/"))
            if sha256_file(path) != str(item["sha256"]):
                raise ValueError(f"paired PnP S/F shard hash mismatch: {path}")
            split_shards.add(f"{item['path']}\x1f{item['sha256']}")
            with np.load(path, allow_pickle=False) as loaded:
                missing = set(TENSOR_FIELDS + ("pair_id", "session_id", "t0_ns")) - set(loaded.files)
                if missing:
                    raise ValueError(f"paired PnP S/F shard fields missing: {sorted(missing)}")
                keep = (
                    (loaded["motion_class"] == int(motion_class))
                    & loaded["pnp_sf_common_usable"].astype(np.bool_)
                )
                if bool(keep.any()):
                    for name in TENSOR_FIELDS:
                        arrays_by_name[name].append(loaded[name][keep].copy())
                    pair_ids.append(loaded["pair_id"][keep].copy())
                    session_ids.append(loaded["session_id"][keep].copy())
                    t0_values.append(loaded["t0_ns"][keep].copy())
        if not pair_ids:
            raise ValueError(f"paired PnP S/F {split} has no common usable samples")
        numpy_arrays = {
            name: np.concatenate(values, axis=0)
            for name, values in arrays_by_name.items()
        }
        pair_array = np.concatenate(pair_ids, axis=0)
        session_array = np.concatenate(session_ids, axis=0)
        t0_array = np.concatenate(t0_values, axis=0)
        if sample_limit > 0:
            numpy_arrays = {
                name: value[:sample_limit] for name, value in numpy_arrays.items()
            }
            pair_array = pair_array[:sample_limit]
            session_array = session_array[:sample_limit]
            t0_array = t0_array[:sample_limit]
        self.tensors = {
            name: torch.from_numpy(np.ascontiguousarray(value))
            for name, value in numpy_arrays.items()
        }
        self.pair_ids = tuple(str(value) for value in pair_array)
        self.t0_ns = tuple(int(value) for value in t0_array)
        # Kept as metadata rather than a forward tensor.  Diagnostic samplers
        # may balance sessions without ever exposing session identity to a
        # model or changing the public training sample schema.
        self.session_ids = tuple(str(value) for value in session_array)
        self.motion_class = int(motion_class)
        self.split = split
        (
            self.split_audit,
            self.session_set,
            self.sample_key_set,
        ) = build_split_audit(
            split=split,
            session_ids=session_array,
            t0_ns=t0_array,
            pair_ids=pair_array,
            shard_tokens=split_shards,
            sample_limit=sample_limit,
            motion_class=motion_class,
            sample_strategy="full_split" if sample_limit <= 0 else "head_slice",
        )

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: value[index] for name, value in self.tensors.items()}


class CausalSelectedPnPAdapter(nn.Module):
    """Denoise the anonymous selected stream without a geometry decoder."""

    def __init__(self, channels: int = 96, dropout: float = 0.05) -> None:
        super().__init__()
        if channels < 16:
            raise ValueError("adapter channels must be at least 16")
        self.channels = int(channels)
        self.dropout = float(dropout)
        # rel xyz, time, dt, switch, current xyz, valid marker.
        self.projection = nn.Sequential(
            nn.Linear(10, channels), nn.LayerNorm(channels), nn.SiLU(),
        )
        self.temporal = nn.ModuleList(
            MaskedCausalResidualBlock(channels, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16)
        )
        self.history_residual_head = nn.Sequential(
            nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, 3),
        )
        self.current_residual_head = nn.Sequential(
            nn.Linear(2 * channels + 3, channels), nn.SiLU(), nn.Linear(channels, 3),
        )
        for head in (self.history_residual_head[-1], self.current_residual_head[-1]):
            assert isinstance(head, nn.Linear)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "family": "causal-selected-pnp-to-clean-observable-adapter-v1",
            "channels": self.channels,
            "dropout": self.dropout,
            "geometry_decoder": False,
            "physical_identity_input": False,
        }

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
        candidate_confidence: torch.Tensor,
        *,
        bypass: bool = False,
    ) -> dict[str, torch.Tensor]:
        if bypass:
            zero_history = torch.zeros_like(history_position_rel_m)
            zero_current = torch.zeros_like(current_position_m)
            return {
                "history_position_rel_m": history_position_rel_m,
                "current_position_m": current_position_m,
                "candidate_relation_m": candidate_relation_m,
                "candidate_confidence": candidate_confidence,
                "history_residual_m": zero_history,
                "current_residual_m": zero_current,
            }
        mask = history_mask.to(torch.bool)
        current_feature = current_position_m[:, None, :].expand(
            -1, history_position_rel_m.shape[1], -1
        )
        feature = torch.cat((
            history_position_rel_m,
            history_time_s.unsqueeze(-1),
            history_dt_s.unsqueeze(-1),
            history_switch_step.to(history_position_rel_m.dtype).unsqueeze(-1),
            current_feature,
            mask.to(history_position_rel_m.dtype).unsqueeze(-1),
        ), dim=-1)
        sequence = self.projection(feature)
        sequence = torch.where(mask.unsqueeze(-1), sequence, torch.zeros_like(sequence))
        sequence_cf = sequence.transpose(1, 2)
        for block in self.temporal:
            sequence_cf = block(sequence_cf, mask)
        sequence = sequence_cf.transpose(1, 2)
        indices = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
        last_index = torch.where(mask, indices, torch.full_like(indices, -1)).amax(dim=1)
        if bool(torch.any(last_index < 0)):
            raise ValueError("adapter history has no valid event")
        row = torch.arange(mask.shape[0], device=mask.device)
        last = sequence[row, last_index]
        mean = sequence.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        current_residual = self.current_residual_head(
            torch.cat((last, mean, current_position_m), dim=-1)
        )
        history_residual = self.history_residual_head(sequence)
        history_residual = torch.where(
            mask.unsqueeze(-1), history_residual, torch.zeros_like(history_residual)
        )
        corrected_current = current_position_m + current_residual
        noisy_absolute = current_position_m[:, None, :] + history_position_rel_m
        corrected_absolute = noisy_absolute + history_residual
        corrected_absolute = corrected_absolute.clone()
        corrected_absolute[row, last_index] = corrected_current
        corrected_history = corrected_absolute - corrected_current[:, None, :]
        corrected_history[row, last_index] = 0.0

        candidate_absolute = current_position_m[:, None, :] + candidate_relation_m
        current_role = torch.remainder(candidate_step.to(torch.long), 4) == 0
        candidate_absolute = torch.where(
            current_role.unsqueeze(-1), corrected_current[:, None, :], candidate_absolute
        )
        corrected_relation = candidate_absolute - corrected_current[:, None, :]
        corrected_relation = torch.where(
            current_role.unsqueeze(-1), torch.zeros_like(corrected_relation),
            corrected_relation,
        )
        return {
            "history_position_rel_m": corrected_history,
            "current_position_m": corrected_current,
            "candidate_relation_m": corrected_relation,
            "candidate_confidence": candidate_confidence,
            "history_residual_m": history_residual,
            "current_residual_m": current_residual,
        }


def load_observable_f_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[AnonymousCandidateFutureExpert, dict[str, Any]]:
    path = Path(checkpoint_path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("model_config", {})
    if payload.get("model_class") != "AnonymousCandidateFutureExpert":
        raise ValueError("F checkpoint model class mismatch")
    model = AnonymousCandidateFutureExpert(
        str(config["expert"]), channels=int(config["channels"]),
        dropout=float(config["dropout"]),
        position_scale_m=float(config["position_scale_m"]),
        history_scale_s=float(config["history_scale_s"]),
        trained_horizon_s=float(config["trained_horizon_s"]),
        maximum_absolute_step=int(config["maximum_absolute_step"]),
        trajectory_rank=int(config["trajectory_rank"]),
    )
    if model.config != config:
        raise ValueError("F checkpoint config is not exactly reproducible")
    model.load_state_dict(payload["model"], strict=True)
    return model, {
        "path": str(path),
        "sha256": sha256_file(path),
        "state_dict_sha256": state_dict_sha256(model.state_dict()),
        "epoch": int(payload["epoch"]),
        "update": int(payload["update"]),
    }


def f_forward(
    model: AnonymousCandidateFutureExpert,
    batch: dict[str, torch.Tensor],
    *,
    prefix: str = "",
    current_position_m: torch.Tensor | None = None,
    candidate_relation_m: torch.Tensor | None = None,
    candidate_confidence: torch.Tensor | None = None,
    history_position_rel_m: torch.Tensor | None = None,
    detach_observation_inputs: bool = True,
) -> dict[str, torch.Tensor]:
    return model(
        batch[f"{prefix}history_position_rel_m"]
        if history_position_rel_m is None else history_position_rel_m,
        batch[f"{prefix}history_time_s"], batch[f"{prefix}history_dt_s"],
        batch[f"{prefix}history_switch_step"], batch[f"{prefix}history_mask"],
        batch[f"{prefix}current_position_m"]
        if current_position_m is None else current_position_m,
        batch[f"{prefix}candidate_relation_m"]
        if candidate_relation_m is None else candidate_relation_m,
        batch[f"{prefix}candidate_step"], batch[f"{prefix}candidate_mask"],
        batch[f"{prefix}candidate_confidence"]
        if candidate_confidence is None else candidate_confidence,
        batch[f"{prefix}tau_s"],
        detach_observation_inputs=detach_observation_inputs,
    )


def sf_compose(
    s_output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    q0 = s_output["q0_m"]
    primary = s_output["primary_index"].to(torch.long)
    row = torch.arange(q0.shape[0], device=q0.device)
    current = q0[row, primary]
    step = batch["candidate_step"].to(torch.long)
    handle = torch.remainder(primary[:, None] + step, 4)
    gather = handle.unsqueeze(-1).expand(-1, -1, 3)
    candidate_absolute = q0.gather(1, gather)
    relation = candidate_absolute - current[:, None, :]
    current_role = torch.remainder(step, 4) == 0
    relation = torch.where(current_role.unsqueeze(-1), torch.zeros_like(relation), relation)
    confidence = s_output["confidence"].gather(1, handle)
    valid = s_output["q0_valid"].gather(1, handle)
    pnp_absolute = (
        batch["pnp_current_position_m"][:, None, :]
        + batch["pnp_history_position_rel_m"]
    )
    history_relative = pnp_absolute - current[:, None, :]
    return {
        "current_position_m": current,
        "candidate_relation_m": relation,
        "candidate_confidence": confidence,
        "candidate_supported": valid,
        "history_position_rel_m": history_relative,
    }
