"""C4-equivariant q0 hypothesis adaptation after the frozen V19 S layer.

The adapter refines anonymous, window-local cyclic handles.  It never receives
future labels, a motion class, a pair/session identifier, a permanent armor ID,
or a candidate step.  Current visible q0 observations are preserved bit-exact.
Warm handles are supervised from their causal history; never-seen cold handles
receive a finite shared structural prior but no coordinate supervision.
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
from .cyclic_track_model import CyclicMessageBlock
from .observable_future_pnp_ab import sha256_file, state_dict_sha256
from .pnp_observation_mapper import (
    AlignedAnchoredWindowPnPObservationMapper,
    AnchoredWindowPnPObservationMapper,
    CausalPnPObservationMapper,
    WindowPnPObservationMapper,
)
from .split_audit import build_split_audit


H_INPUT_FIELDS = (
    "pnp_s_obs_m",
    "pnp_s_obs_mask",
    "pnp_s_primary_mask",
    "pnp_s_event_mask",
    "pnp_s_event_time_s",
    "pnp_s_switch_step",
)
H_LABEL_FIELDS = ("pnp_s_truth_q0_m",)
H_CLEAN_CONTROL_FIELDS = (
    "clean_s_obs_m",
    "clean_s_obs_mask",
    "clean_s_primary_mask",
    "clean_s_event_mask",
    "clean_s_event_time_s",
    "clean_s_switch_step",
)
_INTERNAL_FIELDS = (
    "motion_class",
    "pnp_sf_common_usable",
    "pnp_s_direction_sign",
    "pnp_s_window_shift",
    "session_id",
    "t0_ns",
    "pair_id",
)


def _undo_reflection_keep_c4_origin(
    arrays: dict[str, np.ndarray], direction_sign: np.ndarray
) -> None:
    """Remove reflection only; retain the pre-baked window-local C4 roll."""
    reverse_rows = np.flatnonzero(direction_sign.astype(np.int64) == -1)
    if not reverse_rows.size:
        return
    reverse_index = np.asarray((0, 3, 2, 1), dtype=np.int64)
    for name in (
        "pnp_s_obs_m", "pnp_s_obs_mask", "pnp_s_primary_mask",
        "clean_s_obs_m", "clean_s_obs_mask", "clean_s_primary_mask",
    ):
        value = arrays[name]
        value[reverse_rows] = np.take(value[reverse_rows], reverse_index, axis=2)
    value = arrays["pnp_s_truth_q0_m"]
    value[reverse_rows] = np.take(value[reverse_rows], reverse_index, axis=1)
    arrays["pnp_s_switch_step"][reverse_rows] *= -1
    arrays["clean_s_switch_step"][reverse_rows] *= -1


class PnPQ0HypothesisDataset(Dataset):
    """Strict all-motion/common view for H, with no future fields exposed."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        *,
        sample_limit: int = 0,
        motion_class: int | None = None,
        canonical_direction: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        if split not in {"train", "validation"}:
            raise ValueError("H dataset permits train or validation only")
        if motion_class is not None and motion_class not in {0, 1, 2, 3}:
            raise ValueError("H motion-class filter must be 0..3")
        if not canonical_direction:
            raise ValueError("A3 forbids direction-reflection augmentation")
        manifest_path = self.dataset_dir / "dataset_manifest.json"
        self.manifest_sha256 = sha256_file(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("H dataset schema mismatch")
        if bool(self.manifest.get("test_accessed", True)):
            raise ValueError("H dataset accessed test")
        if not bool(self.manifest.get("qualification_passed", False)):
            raise ValueError("H requires the qualified paired dataset")
        if self.manifest.get("oracle_association") is not True:
            raise ValueError("H dataset must declare oracle association")
        if self.manifest.get("deployable_pipeline") is not False:
            raise ValueError("H dataset must remain a non-deployable upper bound")

        names = (
            H_INPUT_FIELDS + H_LABEL_FIELDS
            + H_CLEAN_CONTROL_FIELDS + _INTERNAL_FIELDS
        )
        values: dict[str, list[np.ndarray]] = {name: [] for name in names}
        for item in self.manifest["shards"]:
            if str(item["split"]) != split:
                continue
            path = self.dataset_dir / Path(str(item["path"]).replace("\\", "/"))
            if sha256_file(path) != str(item["sha256"]):
                raise ValueError(f"H shard hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as loaded:
                missing = set(names) - set(loaded.files)
                if missing:
                    raise ValueError(f"H fields missing: {sorted(missing)}")
                keep = loaded["pnp_sf_common_usable"].astype(np.bool_)
                if motion_class is not None:
                    keep &= loaded["motion_class"] == int(motion_class)
                if bool(keep.any()):
                    selected = {name: loaded[name][keep].copy() for name in names}
                    _undo_reflection_keep_c4_origin(
                        selected, selected["pnp_s_direction_sign"]
                    )
                    for name in names:
                        values[name].append(selected[name])
        if not values["pnp_s_obs_m"]:
            raise ValueError(f"H {split} split has no common observations")
        merged = {
            name: np.concatenate(parts, axis=0) for name, parts in values.items()
        }
        if sample_limit > 0 and sample_limit < len(merged["pnp_s_obs_m"]):
            # A capacity view must span the full split.  Shards/session rows are
            # ordered, so a head slice can contain almost no warm hypotheses.
            indices = np.linspace(
                0, len(merged["pnp_s_obs_m"]) - 1,
                num=sample_limit, dtype=np.int64,
            )
            merged = {name: value[indices] for name, value in merged.items()}
        if not np.all(merged["pnp_s_direction_sign"] == np.asarray((-1, 1))[
            (merged["pnp_s_direction_sign"] > 0).astype(np.int64)
        ]):
            raise ValueError("invalid direction sign")
        self.c4_origin_counts = tuple(
            int((merged["pnp_s_window_shift"] == shift).sum()) for shift in range(4)
        )
        self.source_direction_counts = {
            str(sign): int((merged["pnp_s_direction_sign"] == sign).sum())
            for sign in (-1, 1)
        }
        split_shards = {
            f"{item['path']}\x1f{item['sha256']}"
            for item in self.manifest["shards"] if str(item["split"]) == split
        }
        sample_strategy = (
            "full_split" if sample_limit <= 0
            else "deterministic_even_span_without_replacement"
        )
        (
            self.split_audit,
            self.session_set,
            self.sample_key_set,
        ) = build_split_audit(
            split=split,
            session_ids=merged["session_id"],
            t0_ns=merged["t0_ns"],
            pair_ids=merged["pair_id"],
            shard_tokens=split_shards,
            sample_limit=sample_limit,
            motion_class=motion_class,
            sample_strategy=sample_strategy,
        )
        self.session_ids = tuple(str(value) for value in merged["session_id"])
        exposed = H_INPUT_FIELDS + H_LABEL_FIELDS + H_CLEAN_CONTROL_FIELDS
        self.tensors = {
            name: torch.from_numpy(np.ascontiguousarray(merged[name]))
            for name in exposed
        }
        self.split = split
        self.motion_class = motion_class
        self.canonical_direction = True
        self.sample_strategy = sample_strategy

    def __len__(self) -> int:
        return int(self.tensors["pnp_s_obs_m"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: value[index] for name, value in self.tensors.items()}


def role_masks(
    current_primary: torch.Tensor,
    current_visible: torch.Tensor,
    warm_hidden: torch.Tensor,
    cold: torch.Tensor,
    adjacent: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Partition each anonymous handle from causal support only."""
    for value in (current_primary, current_visible, warm_hidden, cold, adjacent):
        if value.ndim != 2 or value.shape[1] != 4:
            raise ValueError("H role masks must have shape [B,4]")
    primary = current_primary.to(torch.bool)
    observed = current_visible.to(torch.bool)
    warm = warm_hidden.to(torch.bool)
    cold_mask = cold.to(torch.bool)
    adjacent_mask = adjacent.to(torch.bool)
    opposite = torch.roll(primary, shifts=2, dims=1)
    warm_adjacent = warm & adjacent_mask
    warm_opposite = warm & opposite
    if bool(torch.any(warm & ~(warm_adjacent | warm_opposite))):
        raise ValueError("warm handles are not partitioned into adjacent/opposite")
    partition = torch.stack((observed, warm_adjacent, warm_opposite, cold_mask))
    if bool(torch.any(partition.sum(dim=0) != 1)):
        raise ValueError("H support classes must partition all four handles")
    return {
        "observed": observed,
        "warm_adjacent": warm_adjacent,
        "warm_opposite": warm_opposite,
        "true_cold": cold_mask,
        "evidence_supported": ~cold_mask,
        "opposite": opposite,
    }


class C4Q0HypothesisAdapter(nn.Module):
    """Refine only warm q0 hypotheses while preserving observed/cold rows."""

    model_family = "c4-q0-hypothesis-adapter-v2"

    def __init__(
        self,
        position_mean: torch.Tensor,
        position_std: torch.Tensor,
        *,
        channels: int = 64,
        dropout: float = 0.05,
        message_layers: int = 2,
        age_scale_s: float = 0.32,
    ) -> None:
        super().__init__()
        if tuple(position_mean.shape) != (3,) or tuple(position_std.shape) != (3,):
            raise ValueError("H position normalization must have shape [3]")
        if bool(torch.any(position_std <= 0)):
            raise ValueError("H position std must be positive")
        if channels < 16 or message_layers < 2 or age_scale_s <= 0:
            raise ValueError("invalid H configuration")
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.message_layers = int(message_layers)
        self.age_scale_s = float(age_scale_s)
        self.register_buffer("position_mean", position_mean.detach().float().clone())
        self.register_buffer("position_std", position_std.detach().float().clone())

        # anchor-relative q0/direct q0 (6), outgoing/incoming edge (6),
        # q0/out/in sigma (3), age (1), four support classes (4),
        # anchor-composed and outgoing/incoming edge support (3) = 23.
        self.projection = nn.Sequential(
            nn.Linear(23, channels), nn.LayerNorm(channels), nn.SiLU(),
        )
        self.messages = nn.ModuleList(
            CyclicMessageBlock(channels, dropout) for _ in range(message_layers)
        )
        self.edge_residual_head = nn.Sequential(
            nn.Linear(2 * channels, channels), nn.SiLU(), nn.Linear(channels, 3),
        )
        self.hidden_residual_head = nn.Sequential(
            nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, 3),
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, 3),
        )
        self.support_head = nn.Sequential(
            nn.Linear(channels, channels // 2), nn.SiLU(),
            nn.Linear(channels // 2, 1),
        )
        for head in (self.edge_residual_head[-1], self.hidden_residual_head[-1]):
            assert isinstance(head, nn.Linear)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        assert isinstance(self.sigma_head[-1], nn.Linear)
        nn.init.zeros_(self.sigma_head[-1].weight)
        nn.init.constant_(self.sigma_head[-1].bias, -2.0)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "family": self.model_family,
            "channels": self.channels,
            "dropout": self.dropout,
            "message_layers": self.message_layers,
            "age_scale_s": self.age_scale_s,
            "position_mean": self.position_mean.detach().cpu().tolist(),
            "position_std": self.position_std.detach().cpu().tolist(),
            "physical_id_input": False,
            "future_switch_motion_input": False,
            "candidate_step_input": False,
            "observed_q0_identity": True,
            "cold_q0_identity": True,
            "cold_coordinate_supervision": False,
            "f_confidence_contract": "preserve frozen S confidence",
            "handle_symmetry": "shared heads and directed circular messages",
            "direction_reflection_augmentation": False,
        }

    def forward(
        self,
        q0_m: torch.Tensor,
        direct_q0_m: torch.Tensor,
        q0_sigma_m: torch.Tensor,
        confidence: torch.Tensor,
        edge0_m: torch.Tensor,
        edge0_sigma_m: torch.Tensor,
        edge0_supported: torch.Tensor,
        age_s: torch.Tensor,
        current_primary: torch.Tensor,
        current_visible: torch.Tensor,
        warm_hidden: torch.Tensor,
        cold: torch.Tensor,
        adjacent: torch.Tensor,
        anchor_composed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if q0_m.ndim != 3 or q0_m.shape[1:] != (4, 3):
            raise ValueError("H q0 input must have shape [B,4,3]")
        batch = q0_m.shape[0]
        if direct_q0_m.shape != q0_m.shape or edge0_m.shape != q0_m.shape:
            raise ValueError("H position/edge inputs must have shape [B,4,3]")
        if q0_sigma_m.shape != (batch, 4, 1) or edge0_sigma_m.shape != (batch, 4, 1):
            raise ValueError("H input sigmas must have shape [B,4,1]")
        if age_s.shape != (batch, 4):
            raise ValueError("H age must have shape [B,4]")
        roles = role_masks(
            current_primary, current_visible, warm_hidden, cold, adjacent
        )
        primary = current_primary.to(torch.bool)
        if bool(torch.any(primary.sum(dim=1) != 1)):
            raise ValueError("H requires one current primary")
        row = torch.arange(batch, device=q0_m.device)
        primary_index = primary.to(torch.long).argmax(dim=1)
        anchor = q0_m[row, primary_index]
        std = self.position_std.to(q0_m.dtype)
        scale = std.mean()
        relative_q0 = (q0_m - anchor[:, None]) / std
        relative_direct = (direct_q0_m - anchor[:, None]) / std
        outgoing = edge0_m / std
        incoming = torch.roll(edge0_m, shifts=1, dims=1) / std
        outgoing_sigma = edge0_sigma_m.squeeze(-1) / scale
        incoming_sigma = torch.roll(edge0_sigma_m, shifts=1, dims=1).squeeze(-1) / scale
        age = torch.where(torch.isfinite(age_s), age_s, torch.full_like(age_s, 10.0))
        age = age.clamp(0.0, 10.0 * self.age_scale_s) / self.age_scale_s
        support_onehot = torch.stack((
            roles["observed"], roles["warm_adjacent"],
            roles["warm_opposite"], roles["true_cold"],
        ), dim=-1).to(q0_m.dtype)
        outgoing_supported = edge0_supported.to(torch.bool)
        incoming_supported = torch.roll(outgoing_supported, shifts=1, dims=1)
        feature = torch.cat((
            relative_q0, relative_direct, outgoing, incoming,
            q0_sigma_m / scale,
            outgoing_sigma.unsqueeze(-1), incoming_sigma.unsqueeze(-1),
            age.unsqueeze(-1), support_onehot,
            anchor_composed.to(q0_m.dtype).unsqueeze(-1),
            outgoing_supported.to(q0_m.dtype).unsqueeze(-1),
            incoming_supported.to(q0_m.dtype).unsqueeze(-1),
        ), dim=-1)
        state = self.projection(feature)
        for block in self.messages:
            state = block(state)

        next_state = torch.roll(state, shifts=-1, dims=1)
        edge_residual_m = self.edge_residual_head(
            torch.cat((state, next_state), dim=-1)
        ) * std
        corrected_edge_m = edge0_m + edge_residual_m
        observed_q0_m = q0_m
        anchor_q0 = observed_q0_m[row, primary_index]
        edge_at_primary = corrected_edge_m[row, primary_index]
        edge_after_primary = corrected_edge_m[
            row, torch.remainder(primary_index + 1, 4)
        ]
        edge_before_primary = corrected_edge_m[
            row, torch.remainder(primary_index - 1, 4)
        ]
        edge_two_before = corrected_edge_m[
            row, torch.remainder(primary_index - 2, 4)
        ]
        clockwise_adjacent = anchor_q0[:, None] + edge_at_primary[:, None]
        counterclockwise_adjacent = anchor_q0[:, None] - corrected_edge_m
        clockwise_opposite = anchor_q0 + edge_at_primary + edge_after_primary
        counterclockwise_opposite = anchor_q0 - edge_before_primary - edge_two_before
        opposite_prior = 0.5 * (clockwise_opposite + counterclockwise_opposite)

        handle_index = torch.arange(4, device=q0_m.device)[None]
        relative_handle = torch.remainder(handle_index - primary_index[:, None], 4)
        structural = torch.where(
            (relative_handle == 1).unsqueeze(-1),
            clockwise_adjacent.expand(-1, 4, -1),
            counterclockwise_adjacent,
        )
        structural = torch.where(
            (relative_handle == 2).unsqueeze(-1),
            opposite_prior[:, None].expand(-1, 4, -1), structural,
        )
        structural = torch.where(
            (relative_handle == 0).unsqueeze(-1), observed_q0_m, structural
        )
        hidden_residual_m = self.hidden_residual_head(state) * std
        hidden_hypothesis = structural + hidden_residual_m
        preserve_q0 = roles["observed"] | roles["true_cold"]
        refined_q0_m = torch.where(
            preserve_q0.unsqueeze(-1), observed_q0_m, hidden_hypothesis
        )

        learned_sigma_m = (
            torch.nn.functional.softplus(self.sigma_head(state)) + 1e-4
        ) * scale
        hypothesis_sigma_m = torch.where(
            roles["true_cold"].unsqueeze(-1),
            torch.maximum(learned_sigma_m, torch.ones_like(learned_sigma_m)),
            learned_sigma_m,
        )
        support_logits = self.support_head(state).squeeze(-1)
        support_probability = torch.sigmoid(support_logits)
        if torch.any(roles["true_cold"] & (confidence != 0)):
            raise ValueError(
                "V19 contract violation: true-cold confidence must already be zero"
            )
        confidence_for_f = confidence
        support_class = torch.zeros_like(primary_index[:, None].expand(-1, 4))
        support_class = torch.where(
            roles["warm_adjacent"], torch.ones_like(support_class), support_class
        )
        support_class = torch.where(
            roles["warm_opposite"], torch.full_like(support_class, 2), support_class
        )
        support_class = torch.where(
            roles["true_cold"], torch.full_like(support_class, 3), support_class
        )
        return {
            "q0_m": refined_q0_m,
            "edge0_m": corrected_edge_m,
            "edge_residual_m": edge_residual_m,
            "hidden_residual_m": hidden_residual_m,
            "hypothesis_sigma_m": hypothesis_sigma_m,
            "support_logits": support_logits,
            "support_probability": support_probability,
            "confidence_for_f": confidence_for_f,
            "support_class": support_class,
            "evidence_supported": roles["evidence_supported"],
            "observed": roles["observed"],
            "warm_adjacent": roles["warm_adjacent"],
            "warm_opposite": roles["warm_opposite"],
            "true_cold": roles["true_cold"],
            "refined_mask": roles["warm_adjacent"] | roles["warm_opposite"],
        }


def hypothesis_forward(
    model: C4Q0HypothesisAdapter,
    s_output: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Explicit S-to-H field adapter; labels and future fields cannot enter."""
    return model(
        s_output["q0_m"], s_output["direct_q0_m"],
        s_output["q0_sigma_m"], s_output["confidence"],
        s_output["edge0_m"], s_output["edge0_sigma_m"],
        s_output["edge0_supported"], s_output["age_s"],
        s_output["current_primary"], s_output["current_visible"],
        s_output["warm_hidden"], s_output["cold"],
        s_output["adjacent"], s_output["anchor_composed"],
    )


def compose_hypothesis_for_f(
    h_output: dict[str, torch.Tensor],
    primary_index: torch.Tensor,
    corrected_obs_m: torch.Tensor,
    obs_mask: torch.Tensor,
    primary_mask: torch.Tensor,
    candidate_step: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build F inputs solely from corrected observations and H hypotheses."""
    if corrected_obs_m.ndim != 4 or corrected_obs_m.shape[2:] != (4, 3):
        raise ValueError("H compose observations must have shape [B,T,4,3]")
    primary_observed = primary_mask.to(torch.bool) & obs_mask.to(torch.bool)
    if bool(torch.any(primary_observed.sum(dim=2) != 1)):
        raise ValueError("H compose requires one corrected primary per event")
    selected_absolute = (
        corrected_obs_m * primary_observed.unsqueeze(-1).to(corrected_obs_m.dtype)
    ).sum(dim=2)
    q0 = h_output["q0_m"]
    primary = primary_index.to(torch.long)
    row = torch.arange(q0.shape[0], device=q0.device)
    current = q0[row, primary]
    step = candidate_step.to(torch.long)
    handle = torch.remainder(primary[:, None] + step, 4)
    gather = handle.unsqueeze(-1).expand(-1, -1, 3)
    relation = q0.gather(1, gather) - current[:, None]
    current_role = torch.remainder(step, 4) == 0
    relation = torch.where(
        current_role.unsqueeze(-1), torch.zeros_like(relation), relation
    )
    confidence = h_output["confidence_for_f"].gather(1, handle)
    supported = h_output["evidence_supported"].gather(1, handle)
    support_class = h_output["support_class"].gather(1, handle)
    return {
        "current_position_m": current,
        "candidate_relation_m": relation,
        "candidate_confidence": confidence,
        "candidate_supported": supported,
        "candidate_support_class": support_class,
        "candidate_handle": handle,
        "history_position_rel_m": selected_absolute - current[:, None],
        "selected_history_absolute_m": selected_absolute,
    }


def load_frozen_pnp_mapper(
    checkpoint_path: str | Path,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    path = Path(checkpoint_path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != "stage3-pnp-observation-mapper-run-v1"
        or payload.get("model_class") not in {
            "CausalPnPObservationMapper", "WindowPnPObservationMapper",
            "AnchoredWindowPnPObservationMapper",
            "AlignedAnchoredWindowPnPObservationMapper",
        }
    ):
        raise ValueError("PnP mapper checkpoint schema mismatch")
    config = payload["model_config"]
    family = config.get("family")
    if family == CausalPnPObservationMapper.model_family:
        model_class = CausalPnPObservationMapper
    elif family == WindowPnPObservationMapper.model_family:
        model_class = WindowPnPObservationMapper
    elif family in {
        AnchoredWindowPnPObservationMapper.model_family,
        AlignedAnchoredWindowPnPObservationMapper.model_family,
    }:
        anchor_config = config["anchor_config"]
        window_config = config["window_config"]
        anchor = CausalPnPObservationMapper(
            torch.tensor(anchor_config["position_mean"], dtype=torch.float32),
            torch.tensor(anchor_config["position_std"], dtype=torch.float32),
            channels=int(anchor_config["channels"]),
            dropout=float(anchor_config["dropout"]),
            history_events=int(anchor_config["history_events"]),
            history_scale_s=float(anchor_config["history_scale_s"]),
        )
        window = WindowPnPObservationMapper(
            torch.tensor(window_config["position_mean"], dtype=torch.float32),
            torch.tensor(window_config["position_std"], dtype=torch.float32),
            channels=int(window_config["channels"]),
            dropout=float(window_config["dropout"]),
            history_events=int(window_config["history_events"]),
            history_scale_s=float(window_config["history_scale_s"]),
        )
        wrapper_class = (
            AlignedAnchoredWindowPnPObservationMapper
            if family == AlignedAnchoredWindowPnPObservationMapper.model_family
            else AnchoredWindowPnPObservationMapper
        )
        model = wrapper_class(anchor, window)
        model_class = None
    else:
        raise ValueError(f"unsupported PnP mapper family: {family}")
    if model_class is not None:
        model = model_class(
            torch.tensor(config["position_mean"], dtype=torch.float32),
            torch.tensor(config["position_std"], dtype=torch.float32),
            channels=int(config["channels"]), dropout=float(config["dropout"]),
            history_events=int(config["history_events"]),
            history_scale_s=float(config["history_scale_s"]),
        )
    if model.config != config:
        raise ValueError("PnP mapper checkpoint config is not reproducible")
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    return model, {
        "path": str(path),
        "sha256": sha256_file(path),
        "state_dict_sha256": state_dict_sha256(model.state_dict()),
        "epoch": int(payload["epoch"]),
        "update": int(payload["update"]),
        "provenance": payload["provenance"],
    }


def load_frozen_hypothesis_adapter(
    checkpoint_path: str | Path,
    *,
    allow_diagnostic: bool = False,
) -> tuple[C4Q0HypothesisAdapter, dict[str, Any]]:
    path = Path(checkpoint_path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != "stage3-pnp-q0-hypothesis-adapter-run-v1"
        or payload.get("model_class") != "C4Q0HypothesisAdapter"
    ):
        raise ValueError("H checkpoint schema mismatch")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("H checkpoint provenance is missing")
    validation_from_train = provenance.get("validation_from_train")
    diagnostic_only = provenance.get("diagnostic_only")
    if not isinstance(validation_from_train, bool) or not isinstance(
        diagnostic_only, bool
    ):
        raise ValueError(
            "H checkpoint lacks formal/diagnostic split provenance"
        )
    if validation_from_train and not diagnostic_only:
        raise ValueError("H train-sourced validation must be diagnostic-only")
    if diagnostic_only and not allow_diagnostic:
        raise ValueError("diagnostic H checkpoint is forbidden in formal use")
    formal_oracle = provenance.get("formal_oracle_evaluation") is True
    if formal_oracle and (
        diagnostic_only
        or provenance.get("fixed_final_checkpoint") is not True
        or not isinstance(provenance.get("formal_source_contract"), dict)
    ):
        raise ValueError("formal-oracle H provenance is incomplete")
    expected_contract = {
        "test_accessed": False,
        "oracle_association": True,
        "deployable_pipeline": False,
        "optimizer_only_h": True,
        "cold_coordinate_supervision": False,
    }
    for name, expected in expected_contract.items():
        if provenance.get(name) is not expected:
            raise ValueError(f"H checkpoint contract mismatch: {name}")
    if not diagnostic_only:
        source_sha = provenance.get("source_sha256")
        train_audit = provenance.get("train_split_audit")
        validation_audit = provenance.get("validation_split_audit")
        if (
            not isinstance(source_sha, dict) or not source_sha
            or not isinstance(train_audit, dict)
            or not isinstance(validation_audit, dict)
            or train_audit.get("split") != "train"
            or validation_audit.get("split") != "validation"
            or train_audit.get("sample_strategy") != "full_split"
            or validation_audit.get("sample_strategy") != "full_split"
            or train_audit.get("sample_limit") != 0
            or validation_audit.get("sample_limit") != 0
            or train_audit.get("motion_class") is not None
            or validation_audit.get("motion_class") is not None
            or train_audit.get("duplicate_sample_key_count") != 0
            or validation_audit.get("duplicate_sample_key_count") != 0
            or provenance.get("train_validation_session_overlap_count") != 0
        ):
            raise ValueError("formal H source/split audit provenance is invalid")
        manifest_path = path.parent / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("formal H requires final run_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        best = manifest.get("best", {})
        if (
            manifest.get("schema_version")
            != "stage3-pnp-q0-hypothesis-adapter-run-v1"
            or manifest.get("status") != "complete"
            or best.get("path") != path.name
            or best.get("sha256") != sha256_file(path)
            or best.get("epoch") != payload.get("epoch")
            or best.get("update") != payload.get("update")
            or best.get("selection") != list(payload.get("selection", ()))
            or best.get("validation") != payload.get("validation")
            or manifest.get("source_sha256") != source_sha
        ):
            raise ValueError("formal H is not the manifest-declared best")
        if formal_oracle and (
            manifest.get("formal_gate_passed") is not True
            or manifest.get("update") != payload.get("update")
        ):
            raise ValueError("formal-oracle H did not pass its fixed-final gate")
    config = payload["model_config"]
    model = C4Q0HypothesisAdapter(
        torch.tensor(config["position_mean"], dtype=torch.float32),
        torch.tensor(config["position_std"], dtype=torch.float32),
        channels=int(config["channels"]), dropout=float(config["dropout"]),
        message_layers=int(config["message_layers"]),
        age_scale_s=float(config["age_scale_s"]),
    )
    if model.config != config:
        raise ValueError("H checkpoint config is not reproducible")
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    return model, {
        "path": str(path), "sha256": sha256_file(path),
        "state_dict_sha256": state_dict_sha256(model.state_dict()),
        "epoch": int(payload["epoch"]), "update": int(payload["update"]),
        "provenance": provenance,
        "validation": payload.get("validation"),
    }


def roll_s_output_c4(
    s_output: dict[str, torch.Tensor], shift: int
) -> dict[str, torch.Tensor]:
    """Roll every V19 handle/edge tensor while preserving scalar event state."""
    rolled: dict[str, torch.Tensor] = {}
    for name, value in s_output.items():
        if value.ndim >= 2 and value.shape[1] == 4:
            rolled[name] = torch.roll(value, shifts=shift, dims=1)
        elif name == "primary_index":
            rolled[name] = torch.remainder(value + int(shift), 4)
        else:
            rolled[name] = value
    return rolled
