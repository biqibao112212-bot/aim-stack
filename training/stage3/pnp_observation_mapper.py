"""Small causal PnP-to-clean observation mapper.

The mapper has one responsibility: correct oracle-associated PnP XYZ samples
to their paired clean physical observations.  It never receives a physical
slot ID, primary label, switch label, motion class, future target, or truth
state.  The four handle axis is processed with shared weights and invariant
pooling, so a window-local C4 relabel only relabels the output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .build_observable_future_pnp_sf_upper_bound_dataset import SCHEMA_VERSION
from .observable_future_model import MaskedCausalResidualBlock
from .observable_future_pnp_ab import sha256_file
from .split_audit import build_split_audit


INPUT_FIELDS = (
    "pnp_s_obs_m",
    "pnp_s_obs_mask",
    "pnp_s_event_time_s",
    "pnp_s_event_mask",
)
LABEL_FIELDS = ("clean_s_obs_m", "pnp_s_primary_mask")


class PnPObservationMappingDataset(Dataset):
    """Read only the allowlisted paired observation fields from train/val."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        *,
        sample_limit: int = 0,
        motion_class: int | None = None,
        require_common: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        if split not in {"train", "validation"}:
            raise ValueError("PnP mapping permits train or validation only")
        manifest_path = self.dataset_dir / "dataset_manifest.json"
        self.manifest_sha256 = sha256_file(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("PnP mapping dataset schema mismatch")
        if bool(self.manifest.get("test_accessed", True)):
            raise ValueError("PnP mapping dataset accessed test")
        if not bool(self.manifest.get("qualification_passed", False)):
            raise ValueError("PnP mapping requires the qualified paired dataset")
        if self.manifest.get("oracle_association") is not True:
            raise ValueError("PnP mapping view requires declared oracle association")
        if self.manifest.get("deployable_pipeline") is not False:
            raise ValueError("PnP mapping view must remain a non-deployable upper bound")

        names = INPUT_FIELDS + LABEL_FIELDS
        metadata_names = ("session_id", "t0_ns", "pair_id")
        arrays: dict[str, list[np.ndarray]] = {name: [] for name in names}
        metadata: dict[str, list[np.ndarray]] = {
            name: [] for name in metadata_names
        }
        split_shards: set[str] = set()
        for item in self.manifest["shards"]:
            if str(item["split"]) != split:
                continue
            path = self.dataset_dir / Path(str(item["path"]).replace("\\", "/"))
            if sha256_file(path) != str(item["sha256"]):
                raise ValueError(f"PnP mapping shard hash mismatch: {path}")
            split_shards.add(f"{item['path']}\x1f{item['sha256']}")
            with np.load(path, allow_pickle=False) as loaded:
                missing = set(names + metadata_names) - set(loaded.files)
                if missing:
                    raise ValueError(
                        f"PnP mapping fields missing: {sorted(missing)}"
                    )
                obs_mask = loaded["pnp_s_obs_mask"].astype(np.bool_, copy=False)
                clean_mask = loaded["clean_s_obs_mask"].astype(np.bool_, copy=False)
                if np.any(obs_mask & ~clean_mask):
                    raise ValueError("PnP mapping label is missing for a valid input")
                keep = obs_mask.any(axis=(1, 2))
                if motion_class is not None:
                    if motion_class not in {0, 1, 2, 3}:
                        raise ValueError("PnP mapping motion class must be 0..3")
                    if "motion_class" not in loaded.files:
                        raise ValueError("PnP mapping shard has no motion class")
                    keep &= loaded["motion_class"] == int(motion_class)
                if require_common:
                    if "pnp_sf_common_usable" not in loaded.files:
                        raise ValueError("PnP mapping shard has no common-usable flag")
                    keep &= loaded["pnp_sf_common_usable"].astype(np.bool_)
                if bool(keep.any()):
                    for name in names:
                        arrays[name].append(loaded[name][keep].copy())
                    for name in metadata_names:
                        metadata[name].append(loaded[name][keep].copy())
        if not arrays["pnp_s_obs_m"]:
            raise ValueError(f"PnP mapping {split} has no paired observations")
        merged = {
            name: np.concatenate(values, axis=0)
            for name, values in arrays.items()
        }
        merged_metadata = {
            name: np.concatenate(values, axis=0)
            for name, values in metadata.items()
        }
        if sample_limit > 0:
            merged = {name: value[:sample_limit] for name, value in merged.items()}
            merged_metadata = {
                name: value[:sample_limit]
                for name, value in merged_metadata.items()
            }
        self.tensors = {
            name: torch.from_numpy(np.ascontiguousarray(value))
            for name, value in merged.items()
        }
        self.split = split
        self.motion_class = motion_class
        self.require_common = bool(require_common)
        (
            self.split_audit,
            self.session_set,
            self.sample_key_set,
        ) = build_split_audit(
            split=split,
            session_ids=merged_metadata["session_id"],
            t0_ns=merged_metadata["t0_ns"],
            pair_ids=merged_metadata["pair_id"],
            shard_tokens=split_shards,
            sample_limit=sample_limit,
            motion_class=motion_class,
            sample_strategy="full_split" if sample_limit <= 0 else "head_slice",
        )

    def __len__(self) -> int:
        return int(self.tensors["pnp_s_obs_m"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: value[index] for name, value in self.tensors.items()}


class CausalPnPObservationMapper(nn.Module):
    """Map sparse anonymous PnP observations to clean physical XYZ samples."""

    model_family = "causal-c4-equivariant-pnp-observation-mapper-v1"

    def __init__(
        self,
        position_mean: torch.Tensor,
        position_std: torch.Tensor,
        *,
        channels: int = 48,
        dropout: float = 0.05,
        history_events: int = 32,
        history_scale_s: float = 0.32,
    ) -> None:
        super().__init__()
        if tuple(position_mean.shape) != (3,) or tuple(position_std.shape) != (3,):
            raise ValueError("position mean/std must have shape [3]")
        if bool(torch.any(position_std <= 0)):
            raise ValueError("position std must be positive")
        if channels < 16 or history_events < 2 or history_scale_s <= 0:
            raise ValueError("invalid PnP mapper configuration")
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.history_events = int(history_events)
        self.history_scale_s = float(history_scale_s)
        self.register_buffer("position_mean", position_mean.detach().float().clone())
        self.register_buffer("position_std", position_std.detach().float().clone())

        # C4-invariant per-event summary: mean/min/max XYZ, visible fraction,
        # relative time, dt, and event-valid marker.
        self.event_projection = nn.Sequential(
            nn.Linear(13, channels), nn.LayerNorm(channels), nn.SiLU(),
        )
        self.temporal = nn.ModuleList(
            MaskedCausalResidualBlock(channels, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16)
        )
        # Shared local head.  Explicit quadratic XYZ terms retain the useful
        # tiny polynomial baseline while causal context handles temporal jumps.
        # xyz, xyz^2, xy/xz/yz, xyz-event_mean, observed marker = 13.
        self.residual_head = nn.Sequential(
            nn.Linear(channels + 13, channels), nn.SiLU(),
            nn.Linear(channels, 3),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "family": self.model_family,
            "channels": self.channels,
            "dropout": self.dropout,
            "history_events": self.history_events,
            "history_scale_s": self.history_scale_s,
            "position_mean": self.position_mean.detach().cpu().tolist(),
            "position_std": self.position_std.detach().cpu().tolist(),
            "physical_id_input": False,
            "primary_or_switch_input": False,
            "future_or_motion_class_input": False,
            "handle_symmetry": "shared local head plus invariant pooling",
        }

    def forward(
        self,
        pnp_obs_m: torch.Tensor,
        pnp_obs_mask: torch.Tensor,
        event_time_s: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if pnp_obs_m.ndim != 4 or pnp_obs_m.shape[2:] != (4, 3):
            raise ValueError("PnP observations must have shape [B,T,4,3]")
        batch, events = pnp_obs_m.shape[:2]
        if events != self.history_events:
            raise ValueError("PnP mapper history length differs from configuration")
        if pnp_obs_mask.shape != (batch, events, 4):
            raise ValueError("PnP observation mask must have shape [B,T,4]")
        if event_time_s.shape != (batch, events) or event_mask.shape != (batch, events):
            raise ValueError("PnP event inputs must have shape [B,T]")
        mask = pnp_obs_mask.to(torch.bool)
        observed_event = mask.any(dim=-1)
        declared_event = event_mask.to(torch.bool)
        if bool(torch.any(observed_event & ~declared_event)):
            raise ValueError("observed PnP handle belongs to an invalid event")
        valid_event = declared_event & observed_event
        finite = torch.isfinite(pnp_obs_m).all(dim=-1)
        if bool(torch.any(mask & ~finite)):
            raise ValueError("valid PnP observations must be finite")
        if bool(torch.any(valid_event & ~torch.isfinite(event_time_s))):
            raise ValueError("valid PnP event times must be finite")
        if bool(torch.any(valid_event.sum(dim=1) < 1)):
            raise ValueError("PnP mapper requires at least one observed event")

        mean = self.position_mean.to(pnp_obs_m.dtype)
        std = self.position_std.to(pnp_obs_m.dtype)
        normalized = (pnp_obs_m - mean) / std
        clean = torch.where(mask.unsqueeze(-1), normalized, torch.zeros_like(normalized))
        count = mask.sum(dim=2, keepdim=True).clamp_min(1)
        event_mean = clean.sum(dim=2) / count.to(clean.dtype)
        positive = torch.where(
            mask.unsqueeze(-1), normalized, torch.full_like(normalized, torch.inf)
        )
        negative = torch.where(
            mask.unsqueeze(-1), normalized, torch.full_like(normalized, -torch.inf)
        )
        event_min = positive.amin(dim=2)
        event_max = negative.amax(dim=2)
        event_min = torch.where(valid_event.unsqueeze(-1), event_min, torch.zeros_like(event_min))
        event_max = torch.where(valid_event.unsqueeze(-1), event_max, torch.zeros_like(event_max))
        event_mean = torch.where(valid_event.unsqueeze(-1), event_mean, torch.zeros_like(event_mean))
        clean_time = torch.where(valid_event, event_time_s, torch.zeros_like(event_time_s))
        dt = torch.zeros_like(clean_time)
        dt[:, 1:] = clean_time[:, 1:] - clean_time[:, :-1]
        dt = torch.where(valid_event, dt, torch.zeros_like(dt))
        event_feature = torch.cat((
            event_mean, event_min, event_max,
            (count.squeeze(-1).to(clean.dtype) / 4.0).unsqueeze(-1),
            (clean_time / self.history_scale_s).unsqueeze(-1),
            (dt / self.history_scale_s).unsqueeze(-1),
            valid_event.to(clean.dtype).unsqueeze(-1),
        ), dim=-1)
        context = self.event_projection(event_feature)
        context = torch.where(valid_event.unsqueeze(-1), context, torch.zeros_like(context))
        context_cf = context.transpose(1, 2)
        for block in self.temporal:
            context_cf = block(context_cf, valid_event)
        context = context_cf.transpose(1, 2)

        x, y, z = clean.unbind(dim=-1)
        cross = torch.stack((x * y, x * z, y * z), dim=-1)
        local_feature = torch.cat((
            clean, clean.square(), cross,
            clean - event_mean.unsqueeze(2),
            mask.to(clean.dtype).unsqueeze(-1),
        ), dim=-1)
        expanded_context = context.unsqueeze(2).expand(-1, -1, 4, -1)
        residual_normalized = self.residual_head(torch.cat((
            expanded_context, local_feature,
        ), dim=-1))
        residual_normalized = torch.where(
            mask.unsqueeze(-1), residual_normalized,
            torch.zeros_like(residual_normalized),
        )
        residual_m = residual_normalized * std
        corrected = torch.where(
            mask.unsqueeze(-1), pnp_obs_m + residual_m,
            torch.zeros_like(pnp_obs_m),
        )
        return {
            "corrected_obs_m": corrected,
            "residual_m": residual_m,
            "obs_mask": mask,
            "event_context": context,
        }


class WindowPnPObservationMapper(nn.Module):
    """Denoise an entirely observed past window without reading beyond q0."""

    model_family = "window-causal-c4-equivariant-pnp-observation-mapper-v2"

    def __init__(
        self,
        position_mean: torch.Tensor,
        position_std: torch.Tensor,
        *,
        channels: int = 48,
        dropout: float = 0.05,
        history_events: int = 32,
        history_scale_s: float = 0.32,
    ) -> None:
        super().__init__()
        if tuple(position_mean.shape) != (3,) or tuple(position_std.shape) != (3,):
            raise ValueError("position mean/std must have shape [3]")
        if bool(torch.any(position_std <= 0)):
            raise ValueError("position std must be positive")
        if channels < 16 or channels % 2 or history_events < 2:
            raise ValueError("invalid window PnP mapper configuration")
        if history_scale_s <= 0:
            raise ValueError("window PnP mapper history scale must be positive")
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.history_events = int(history_events)
        self.history_scale_s = float(history_scale_s)
        self.register_buffer("position_mean", position_mean.detach().float().clone())
        self.register_buffer("position_std", position_std.detach().float().clone())
        self.handle_projection = nn.Sequential(
            nn.Linear(17, channels), nn.LayerNorm(channels), nn.SiLU(),
        )
        self.window_smoother = nn.GRU(
            channels, channels // 2, num_layers=2, batch_first=True,
            dropout=dropout, bidirectional=True,
        )
        self.residual_head = nn.Sequential(
            nn.Linear(channels + 13, channels), nn.SiLU(),
            nn.Linear(channels, 3),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "family": self.model_family,
            "channels": self.channels,
            "dropout": self.dropout,
            "history_events": self.history_events,
            "history_scale_s": self.history_scale_s,
            "position_mean": self.position_mean.detach().cpu().tolist(),
            "position_std": self.position_std.detach().cpu().tolist(),
            "physical_id_input": False,
            "primary_or_switch_input": False,
            "future_or_motion_class_input": False,
            "handle_symmetry": "one shared bidirectional past-window smoother",
            "window_causality": "all valid timestamps are <= q0; no future field",
            "per_event_online_causality": False,
        }

    def forward(
        self,
        pnp_obs_m: torch.Tensor,
        pnp_obs_mask: torch.Tensor,
        event_time_s: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if pnp_obs_m.ndim != 4 or pnp_obs_m.shape[2:] != (4, 3):
            raise ValueError("PnP observations must have shape [B,T,4,3]")
        batch, events = pnp_obs_m.shape[:2]
        if events != self.history_events:
            raise ValueError("window PnP history length differs from configuration")
        if pnp_obs_mask.shape != (batch, events, 4):
            raise ValueError("PnP observation mask must have shape [B,T,4]")
        if event_time_s.shape != (batch, events) or event_mask.shape != (batch, events):
            raise ValueError("PnP event inputs must have shape [B,T]")
        mask = pnp_obs_mask.to(torch.bool)
        observed_event = mask.any(dim=-1)
        declared_event = event_mask.to(torch.bool)
        if bool(torch.any(observed_event & ~declared_event)):
            raise ValueError("observed PnP handle belongs to an invalid event")
        valid_event = declared_event & observed_event
        if bool(torch.any(mask & ~torch.isfinite(pnp_obs_m).all(dim=-1))):
            raise ValueError("valid PnP observations must be finite")
        if bool(torch.any(valid_event & ~torch.isfinite(event_time_s))):
            raise ValueError("valid PnP event times must be finite")
        if bool(torch.any(valid_event & (event_time_s > 1e-6))):
            raise ValueError("window PnP mapper forbids observations after q0")
        pair_valid = valid_event[:, :, None] & valid_event[:, None, :]
        ordered_pair = torch.triu(
            torch.ones(events, events, dtype=torch.bool, device=event_time_s.device),
            diagonal=1,
        )
        nonincreasing = event_time_s[:, :, None] >= event_time_s[:, None, :]
        if bool(torch.any(pair_valid & ordered_pair & nonincreasing)):
            raise ValueError("window PnP event times must be strictly increasing")
        if bool(torch.any(valid_event.sum(dim=1) < 2)):
            raise ValueError("window PnP mapper requires two past-window events")

        mean = self.position_mean.to(pnp_obs_m.dtype)
        std = self.position_std.to(pnp_obs_m.dtype)
        normalized = (pnp_obs_m - mean) / std
        clean = torch.where(mask.unsqueeze(-1), normalized, torch.zeros_like(normalized))
        count = mask.sum(dim=2, keepdim=True).clamp_min(1)
        event_mean = clean.sum(dim=2) / count.to(clean.dtype)
        event_mean = torch.where(
            valid_event.unsqueeze(-1), event_mean, torch.zeros_like(event_mean)
        )
        clean_time = torch.where(valid_event, event_time_s, torch.zeros_like(event_time_s))
        dt = torch.zeros_like(clean_time)
        dt[:, 1:] = clean_time[:, 1:] - clean_time[:, :-1]
        dt = torch.where(valid_event, dt, torch.zeros_like(dt))
        visibility = count.squeeze(-1).to(clean.dtype) / 4.0
        token = torch.cat((
            clean, clean.square(),
            event_mean[:, :, None, :].expand(-1, -1, 4, -1),
            clean - event_mean[:, :, None, :],
            visibility[:, :, None, None].expand(-1, -1, 4, -1),
            (clean_time / self.history_scale_s)[:, :, None, None].expand(-1, -1, 4, -1),
            (dt / self.history_scale_s)[:, :, None, None].expand(-1, -1, 4, -1),
            mask.to(clean.dtype).unsqueeze(-1),
            valid_event.to(clean.dtype)[:, :, None, None].expand(-1, -1, 4, -1),
        ), dim=-1)
        projected = self.handle_projection(token)
        handle_sequence = projected.permute(0, 2, 1, 3).reshape(
            batch * 4, events, self.channels
        )
        smoothed, _ = self.window_smoother(handle_sequence)
        smoothed = smoothed.reshape(batch, 4, events, self.channels).permute(0, 2, 1, 3)

        x, y, z = clean.unbind(dim=-1)
        cross = torch.stack((x * y, x * z, y * z), dim=-1)
        local_feature = torch.cat((
            clean, clean.square(), cross,
            clean - event_mean.unsqueeze(2),
            mask.to(clean.dtype).unsqueeze(-1),
        ), dim=-1)
        residual_normalized = self.residual_head(torch.cat((
            smoothed, local_feature,
        ), dim=-1))
        residual_normalized = torch.where(
            mask.unsqueeze(-1), residual_normalized,
            torch.zeros_like(residual_normalized),
        )
        residual_m = residual_normalized * std
        corrected = torch.where(
            mask.unsqueeze(-1), pnp_obs_m + residual_m,
            torch.zeros_like(pnp_obs_m),
        )
        return {
            "corrected_obs_m": corrected,
            "residual_m": residual_m,
            "obs_mask": mask,
            "event_context": smoothed.mean(dim=2),
        }


class AnchoredWindowPnPObservationMapper(nn.Module):
    """Preserve an accepted q0 mapper while smoothing only the closed past."""

    model_family = "q0-anchored-window-pnp-observation-mapper-v3"

    def __init__(
        self,
        anchor_mapper: CausalPnPObservationMapper,
        window_smoother: WindowPnPObservationMapper,
    ) -> None:
        super().__init__()
        if anchor_mapper.config["position_mean"] != window_smoother.config["position_mean"]:
            raise ValueError("anchored mapper position means differ")
        if anchor_mapper.config["position_std"] != window_smoother.config["position_std"]:
            raise ValueError("anchored mapper position scales differ")
        if anchor_mapper.history_events != window_smoother.history_events:
            raise ValueError("anchored mapper history lengths differ")
        self.anchor_mapper = anchor_mapper
        self.window_smoother = window_smoother
        self.anchor_mapper.eval().requires_grad_(False)

    @property
    def position_mean(self) -> torch.Tensor:
        return self.window_smoother.position_mean

    @property
    def position_std(self) -> torch.Tensor:
        return self.window_smoother.position_std

    @property
    def config(self) -> dict[str, Any]:
        return {
            "family": self.model_family,
            "anchor_config": self.anchor_mapper.config,
            "window_config": self.window_smoother.config,
            "physical_id_input": False,
            "primary_or_switch_input": False,
            "future_or_motion_class_input": False,
            "q0_contract": "bit-exact frozen anchor mapper output",
            "past_contract": "trainable bidirectional closed-window smoother",
        }

    def train(self, mode: bool = True) -> "AnchoredWindowPnPObservationMapper":
        super().train(mode)
        self.anchor_mapper.eval()
        return self

    def forward(
        self,
        pnp_obs_m: torch.Tensor,
        pnp_obs_mask: torch.Tensor,
        event_time_s: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        anchor = self.anchor_mapper(
            pnp_obs_m, pnp_obs_mask, event_time_s, event_mask
        )
        window = self.window_smoother(
            pnp_obs_m, pnp_obs_mask, event_time_s, event_mask
        )
        valid_event = event_mask.to(torch.bool) & pnp_obs_mask.to(torch.bool).any(dim=2)
        q0_event = valid_event & event_time_s.abs().le(1e-6)
        if bool(torch.any(q0_event.sum(dim=1) != 1)):
            raise ValueError("anchored mapper requires exactly one q0 event")
        q0_mask = q0_event[:, :, None, None]
        corrected = torch.where(
            q0_mask, anchor["corrected_obs_m"], window["corrected_obs_m"]
        )
        mask = pnp_obs_mask.to(torch.bool)
        residual = torch.where(
            mask.unsqueeze(-1), corrected - pnp_obs_m,
            torch.zeros_like(pnp_obs_m),
        )
        return {
            "corrected_obs_m": corrected,
            "residual_m": residual,
            "obs_mask": mask,
            "event_context": window["event_context"],
            "anchor_corrected_obs_m": anchor["corrected_obs_m"],
            "window_corrected_obs_m": window["corrected_obs_m"],
        }


class AlignedAnchoredWindowPnPObservationMapper(
    AnchoredWindowPnPObservationMapper
):
    """Align the smoothed past frame to the frozen q0 anchor frame."""

    model_family = "q0-aligned-anchored-window-pnp-observation-mapper-v4"

    @property
    def config(self) -> dict[str, Any]:
        config = super().config
        config["family"] = self.model_family
        config["past_alignment"] = (
            "C4-invariant mean anchor-minus-window correction at observed q0"
        )
        return config

    def forward(
        self,
        pnp_obs_m: torch.Tensor,
        pnp_obs_mask: torch.Tensor,
        event_time_s: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        anchor = self.anchor_mapper(
            pnp_obs_m, pnp_obs_mask, event_time_s, event_mask
        )
        window = self.window_smoother(
            pnp_obs_m, pnp_obs_mask, event_time_s, event_mask
        )
        event_valid = event_mask.to(torch.bool) & pnp_obs_mask.to(torch.bool).any(dim=2)
        q0_event = event_valid & event_time_s.abs().le(1e-6)
        if bool(torch.any(q0_event.sum(dim=1) != 1)):
            raise ValueError("aligned anchored mapper requires exactly one q0 event")
        q0_handle_mask = q0_event[:, :, None] & pnp_obs_mask.to(torch.bool)
        alignment_sum = torch.where(
            q0_handle_mask.unsqueeze(-1),
            anchor["corrected_obs_m"] - window["corrected_obs_m"],
            torch.zeros_like(pnp_obs_m),
        ).sum(dim=(1, 2))
        alignment_count = q0_handle_mask.sum(dim=(1, 2)).clamp_min(1)
        alignment = alignment_sum / alignment_count[:, None].to(alignment_sum.dtype)
        aligned_window = torch.where(
            pnp_obs_mask.to(torch.bool).unsqueeze(-1),
            window["corrected_obs_m"] + alignment[:, None, None, :],
            torch.zeros_like(pnp_obs_m),
        )
        corrected = torch.where(
            q0_event[:, :, None, None],
            anchor["corrected_obs_m"], aligned_window,
        )
        mask = pnp_obs_mask.to(torch.bool)
        residual = torch.where(
            mask.unsqueeze(-1), corrected - pnp_obs_m,
            torch.zeros_like(pnp_obs_m),
        )
        return {
            "corrected_obs_m": corrected,
            "residual_m": residual,
            "obs_mask": mask,
            "event_context": window["event_context"],
            "anchor_corrected_obs_m": anchor["corrected_obs_m"],
            "window_corrected_obs_m": window["corrected_obs_m"],
            "past_alignment_m": alignment,
        }


def compose_corrected_observations(
    s_output: dict[str, torch.Tensor],
    corrected_obs_m: torch.Tensor,
    obs_mask: torch.Tensor,
    primary_mask: torch.Tensor,
    candidate_step: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compose frozen-F inputs without falling back to the raw PnP stream."""
    if corrected_obs_m.ndim != 4 or corrected_obs_m.shape[2:] != (4, 3):
        raise ValueError("corrected observations must have shape [B,T,4,3]")
    if obs_mask.shape != corrected_obs_m.shape[:-1]:
        raise ValueError("corrected observation mask shape mismatch")
    if primary_mask.shape != corrected_obs_m.shape[:-1]:
        raise ValueError("corrected primary mask shape mismatch")
    primary_observed = primary_mask.to(torch.bool) & obs_mask.to(torch.bool)
    if bool(torch.any(primary_observed.sum(dim=2) != 1)):
        raise ValueError("every frozen-F history event needs one corrected primary")
    selected_absolute = (
        corrected_obs_m * primary_observed.unsqueeze(-1).to(corrected_obs_m.dtype)
    ).sum(dim=2)

    q0 = s_output["q0_m"]
    primary = s_output["primary_index"].to(torch.long)
    row = torch.arange(q0.shape[0], device=q0.device)
    current = q0[row, primary]
    step = candidate_step.to(torch.long)
    handle = torch.remainder(primary[:, None] + step, 4)
    gather = handle.unsqueeze(-1).expand(-1, -1, 3)
    candidate_absolute = q0.gather(1, gather)
    relation = candidate_absolute - current[:, None, :]
    current_role = torch.remainder(step, 4) == 0
    relation = torch.where(
        current_role.unsqueeze(-1), torch.zeros_like(relation), relation
    )
    confidence = s_output["confidence"].gather(1, handle)
    valid = s_output["q0_valid"].gather(1, handle)
    return {
        "current_position_m": current,
        "candidate_relation_m": relation,
        "candidate_confidence": confidence,
        "candidate_supported": valid,
        "history_position_rel_m": selected_absolute - current[:, None, :],
        "selected_history_absolute_m": selected_absolute,
    }
