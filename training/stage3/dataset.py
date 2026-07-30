"""Raw-session join and irregular event-sequence tensorization for Stage 3.

The converter never uses detector number, tracker id, or legacy telemetry. It
joins the dedicated pre-tracker and exact-exposure streams on the full key and
rejects ambiguous windows before writing training tensors.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
import json
import hashlib
import math
import re

import numpy as np

from .schema import (
    ObservationFrame,
    TruthFrame,
    iter_json_records,
    _quat_rotate,
)

MAX_ARMORS = 4
EVENT_COUNT = 200
FORMAL_MOTION_HISTORY_EVENT_LIMIT = 32
MIN_HISTORY_S = 0.2
MAX_LATEST_AGE_S = 0.05
QUERY_TAUS_S = (0.0, 0.1, 0.2, 0.5)
MOTION_CLASSES = {"stationary": 0, "linear": 1, "spin": 2, "linear_and_spin": 3}
STAGE3_OBSERVATION_V1_HISTORICAL_H_M = 0.07
MOTION_SEGMENT_TIME_GUARD_NS = 2_000


@dataclass(frozen=True)
class CameraGimbalExtrinsic:
    rotation: np.ndarray  # ^G R_C, [3,3]
    translation_m: np.ndarray  # ^G t_C, [3]

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation_m, dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("camera/gimbal extrinsic must have shapes [3,3] and [3]")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise ValueError("camera/gimbal extrinsic must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-5
        ):
            raise ValueError("camera/gimbal rotation must be a proper orthonormal matrix")
        if np.linalg.norm(translation) > 5.0:
            raise ValueError("camera/gimbal translation must be expressed in metres")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation_m", translation)


def load_camera_gimbal_extrinsic(path: str | Path) -> CameraGimbalExtrinsic:
    text = Path(path).resolve().read_text(encoding="utf-8")

    def matrix_values(key: str, expected: int) -> np.ndarray:
        match = re.search(
            rf"(?ms)^{re.escape(key)}:\s*!!opencv-matrix\s*.*?^\s*data:\s*\[([^]]+)\]",
            text,
        )
        if match is None:
            raise ValueError(f"calibration is missing {key}")
        values = np.fromstring(match.group(1), sep=",", dtype=np.float64)
        if values.size != expected:
            raise ValueError(f"{key} must contain {expected} numeric values")
        return values

    rotation = matrix_values("R_CAMERA2GIMBAL", 9).reshape(3, 3)
    translation = matrix_values("T_CAMERA2GIMBAL", 3)
    return CameraGimbalExtrinsic(rotation, translation)


@dataclass(frozen=True)
class TensorSample:
    session_id: str
    t0_ns: int
    obs: np.ndarray  # [T,4,5]
    obs_mask: np.ndarray  # [T,4]
    event_mask: np.ndarray  # [T]
    event_time_s: np.ndarray  # [T], real time relative to t0
    tau: np.ndarray  # [Q], effective truth time relative to t0
    tau_requested: np.ndarray  # [Q], requested query time before truth matching
    future_timestamp_ns: np.ndarray  # [Q], exact matched truth timestamps
    future_position: np.ndarray  # [Q,4,3]
    future_normal: np.ndarray  # [Q,4,3]
    motion_class: int
    motion_command_epoch: int
    motion_segment_start_ns: int
    motion_segment_end_ns: int
    history_start_ns: int
    future_end_ns: int


def _key(record: Mapping[str, object]) -> tuple[str, int, int, int]:
    return (
        str(record["session_id"]),
        int(record["producer_epoch"]),
        int(record["frame_seq"]),
        int(record["timestamp_ns"]),
    )


def _camera_to_tracker_rotation(gimbal_pitch_deg: float, gimbal_yaw_deg: float) -> np.ndarray:
    pitch = math.radians(gimbal_pitch_deg)
    yaw = math.radians(gimbal_yaw_deg)
    stabilizing_pitch = -pitch
    pitch_rotation = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(stabilizing_pitch), -math.sin(stabilizing_pitch)],
        [0.0, math.sin(stabilizing_pitch), math.cos(stabilizing_pitch)],
    ])
    yaw_rotation = np.asarray([
        [math.cos(yaw), 0.0, math.sin(yaw)],
        [0.0, 1.0, 0.0],
        [-math.sin(yaw), 0.0, math.cos(yaw)],
    ])
    axis_permutation = np.asarray([
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ])
    return axis_permutation @ yaw_rotation @ pitch_rotation


def _observation_position_v3(
    frame: ObservationFrame,
    position_m: tuple[float, float, float],
    extrinsic: CameraGimbalExtrinsic,
) -> np.ndarray:
    position = np.asarray(position_m, dtype=np.float64)
    if frame.schema_version == "stage3-observation-v2":
        if frame.R_camera2gimbal is None or frame.t_camera2gimbal_m is None:
            raise ValueError("v2 observation has no R/T audit record")
        if not np.allclose(np.asarray(frame.R_camera2gimbal).reshape(3, 3), extrinsic.rotation, atol=1e-9) or not np.allclose(
            np.asarray(frame.t_camera2gimbal_m), extrinsic.translation_m, atol=1e-9
        ):
            raise ValueError("observation calibration differs from dataset calibration")
        return position

    # Historical v1 only stored the old derived tracker position:
    # p_T_old = R_TC (p_C - [0,H,0]). Recover raw p_C, then apply the new
    # calibrated rigid transform. The named constant exists only for immutable
    # v1 migration; it is not a runtime model parameter.
    R_tracker_camera = _camera_to_tracker_rotation(
        frame.gimbal_pitch_deg, frame.gimbal_yaw_deg
    )
    camera_point = R_tracker_camera.T @ position + np.asarray(
        [0.0, STAGE3_OBSERVATION_V1_HISTORICAL_H_M, 0.0]
    )
    R_tracker_gimbal = R_tracker_camera @ extrinsic.rotation.T
    return R_tracker_gimbal @ (
        extrinsic.rotation @ camera_point + extrinsic.translation_m
    )


def _world_to_tracker(
    point_world: tuple[float, float, float],
    tracker_origin_world: tuple[float, float, float],
    chassis_quaternion_world_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    # Formal capture records the exact exposure gimbal pivot as tracker origin.
    # Labels therefore do not reconstruct an origin through camera conventions.
    delta = tuple(a - b for a, b in zip(point_world, tracker_origin_world))
    q = chassis_quaternion_world_wxyz
    return _quat_rotate((q[0], -q[1], -q[2], -q[3]), delta)


def _motion_class(manifest: Mapping[str, object] | None, truth: TruthFrame) -> int:
    if manifest is not None:
        mode = str(manifest.get("mode", ""))
        if mode in MOTION_CLASSES:
            return MOTION_CLASSES[mode]
    speed = math.sqrt(sum(v * v for v in truth.velocity_world_mps))
    angular = abs(truth.yaw_rate_rad_s)
    if speed < 0.02 and angular < 0.02:
        return 0
    if angular < 0.02:
        return 1
    if speed < 0.02:
        return 2
    return 3


def _constant_truth_window(
    truths: list[TruthFrame], start_ns: int, end_ns: int,
) -> bool:
    """Require one exact constant physical state across the full model window."""
    frames = [frame for frame in truths if start_ns <= frame.timestamp_ns <= end_ns]
    if not frames or frames[0].timestamp_ns > start_ns or frames[-1].timestamp_ns < end_ns:
        return False
    first = frames[0]
    if first.target_origin_world_m is None:
        return False
    timestamps = np.asarray([frame.timestamp_ns for frame in frames], dtype=np.int64)
    velocities = np.asarray([frame.velocity_world_mps for frame in frames], dtype=np.float64)
    yaw_rates = np.asarray([frame.yaw_rate_rad_s for frame in frames], dtype=np.float64)
    origins = [frame.target_origin_world_m for frame in frames]
    if any(origin is None for origin in origins):
        return False
    origins_array = np.asarray(origins, dtype=np.float64)
    yaws = np.asarray([frame.yaw_rad for frame in frames], dtype=np.float64)
    time_s = (timestamps - timestamps[0]).astype(np.float64) / 1e9
    expected_origin = origins_array[0] + velocities[0] * time_s[:, None]
    expected_yaw = yaws[0] + yaw_rates[0] * time_s
    yaw_delta = yaws - expected_yaw
    return bool(
        all(frame.producer_epoch == first.producer_epoch for frame in frames)
        and all(frame.target_id == first.target_id for frame in frames)
        and all(str(frame.geometry_hash) == str(first.geometry_hash) for frame in frames)
        and np.linalg.norm(velocities - velocities[0], axis=1).max(initial=0.0) <= 1e-6
        and np.abs(yaw_rates - yaw_rates[0]).max(initial=0.0) <= 1e-6
        and np.linalg.norm(origins_array - expected_origin, axis=1).max(initial=0.0) <= 1e-4
        and np.abs(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))).max(initial=0.0) <= 1e-4
    )


def load_manifests(paths: Iterable[str | Path]) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for path in paths:
        for record in iter_json_records(path):
            session_id = str(record.get("session_id", ""))
            if not session_id:
                raise ValueError(f"manifest record in {path} has no session_id")
            if session_id in result:
                raise ValueError(f"duplicate session manifest: {session_id}")
            result[session_id] = record
    return result


def _load_observations(path: str | Path) -> dict[str, list[ObservationFrame]]:
    result: dict[str, list[ObservationFrame]] = {}
    for record in iter_json_records(path):
        frame = ObservationFrame.from_mapping(record)
        result.setdefault(frame.session_id, []).append(frame)
    for frames in result.values():
        frames.sort(key=lambda frame: (frame.producer_epoch, frame.timestamp_ns, frame.frame_seq))
    return result


def _load_truth(path: str | Path) -> dict[tuple[str, int, int, int], TruthFrame]:
    result: dict[tuple[str, int, int, int], TruthFrame] = {}
    geometry_by_session: dict[str, str] = {}
    for record in iter_json_records(path):
        if not bool(record.get("has_exact_exposure_truth", False)):
            continue
        truth = TruthFrame.from_mapping(record)
        if truth.geometry_hash is None:
            raise ValueError(f"truth record has no geometry_hash: {truth.session_id}/{truth.frame_seq}")
        previous_geometry = geometry_by_session.setdefault(truth.session_id, truth.geometry_hash)
        if previous_geometry != truth.geometry_hash:
            raise ValueError(f"geometry drift in session {truth.session_id}")
        key = (truth.session_id, truth.producer_epoch, truth.frame_seq, truth.timestamp_ns)
        if key in result:
            raise ValueError(f"duplicate exact truth key: {key}")
        result[key] = truth
    return result


def _nearest_truth(
    truths: list[TruthFrame], timestamp_ns: int, max_error_ns: int = 25_000_000,
    producer_epoch: int | None = None,
) -> TruthFrame | None:
    if producer_epoch is not None:
        truths = [frame for frame in truths if frame.producer_epoch == producer_epoch]
    if not truths:
        return None
    timestamps = np.fromiter((frame.timestamp_ns for frame in truths), dtype=np.int64)
    index = int(np.searchsorted(timestamps, timestamp_ns))
    candidates = [min(index, len(timestamps) - 1)]
    if index > 0:
        candidates.append(index - 1)
    best = min(candidates, key=lambda i: abs(int(timestamps[i]) - timestamp_ns))
    if abs(int(timestamps[best]) - timestamp_ns) > max_error_ns:
        return None
    return truths[best]


def _make_sample(
    anchor: ObservationFrame,
    observations: list[ObservationFrame],
    truths: list[TruthFrame],
    manifest: Mapping[str, object] | None,
    extrinsic: CameraGimbalExtrinsic,
    rng: np.random.Generator | None = None,
    query_taus: tuple[float, ...] = QUERY_TAUS_S,
    diagnostics: dict[str, object] | None = None,
    truth_by_key: Mapping[tuple[str, int, int, int], TruthFrame] | None = None,
    inputs_epoch_filtered: bool = False,
    motion_segment: Mapping[str, object] | None = None,
) -> TensorSample | None:
    rejection_counts: Counter[str] | None = None
    if diagnostics is not None:
        rejection_counts = diagnostics.setdefault("rejection_counts", Counter())  # type: ignore[assignment]

    def reject(reason: str) -> None:
        if rejection_counts is not None:
            rejection_counts[reason] += 1

    if len(anchor.armors) > MAX_ARMORS:
        reject("anchor_more_than_four_candidates")
        return None
    anchor_truth = (
        truth_by_key.get(anchor.key) if truth_by_key is not None else next((
            truth for truth in truths
            if (truth.session_id, truth.producer_epoch, truth.frame_seq, truth.timestamp_ns) == anchor.key
        ), None)
    )
    if anchor_truth is None:
        reject("missing_anchor_truth")
        return None
    tracker_quaternion = anchor_truth.chassis_quaternion_world_wxyz
    t0 = anchor.timestamp_ns
    segment_start_ns = (
        int(motion_segment["start_timestamp_ns"])
        if motion_segment is not None else -1
    )
    segment_end_ns = (
        int(motion_segment["end_timestamp_ns"])
        if motion_segment is not None else 2**63 - 1
    )
    if motion_segment is not None and not segment_start_ns <= t0 < segment_end_ns:
        reject("anchor_outside_motion_segment")
        return None
    eligible = sorted((
        frame for frame in observations
        if frame.timestamp_ns <= t0 and
        (motion_segment is None or frame.timestamp_ns >= segment_start_ns) and
        (inputs_epoch_filtered or frame.producer_epoch == anchor.producer_epoch)
    ), key=lambda frame: (frame.timestamp_ns, frame.frame_seq))

    def valid_armors(frame: ObservationFrame):
        return [
            armor for armor in frame.armors
            if armor.valid and all(math.isfinite(value) for value in (*armor.position_m, armor.yaw_rad))
        ]

    valid_events = [frame for frame in eligible if valid_armors(frame)]
    history_event_limit = (
        FORMAL_MOTION_HISTORY_EVENT_LIMIT
        if motion_segment is not None else EVENT_COUNT
    )
    selected = valid_events[-history_event_limit:]
    if not selected:
        reject("no_valid_observation_events")
        return None
    contributing_start_ns = (
        selected[0].timestamp_ns
        if motion_segment is not None or len(selected) == history_event_limit
        else eligible[0].timestamp_ns
    )
    if any(
        len(frame.armors) > MAX_ARMORS
        for frame in eligible
        if contributing_start_ns <= frame.timestamp_ns <= t0
    ):
        reject("history_more_than_four_candidates")
        return None

    obs = np.zeros((EVENT_COUNT, MAX_ARMORS, 5), dtype=np.float32)
    obs_mask = np.zeros((EVENT_COUNT, MAX_ARMORS), dtype=np.bool_)
    event_mask = np.zeros((EVENT_COUNT,), dtype=np.bool_)
    event_time_s = np.zeros((EVENT_COUNT,), dtype=np.float32)
    destination_start = EVENT_COUNT - len(selected)
    for event_index, frame in enumerate(selected, destination_start):
        event_time_s[event_index] = (frame.timestamp_ns - t0) / 1e9
        for armor_index, armor in enumerate(frame.armors):
            values = (*armor.position_m, math.sin(armor.yaw_rad), math.cos(armor.yaw_rad))
            if armor.valid and all(math.isfinite(value) for value in values):
                obs[event_index, armor_index, :3] = _observation_position_v3(
                    frame, armor.position_m, extrinsic
                )
                obs[event_index, armor_index, 3:] = values[3:]
                obs_mask[event_index, armor_index] = True
        event_mask[event_index] = obs_mask[event_index].any()
    recent_cutoff = t0 - int(MIN_HISTORY_S * 1e9)
    recent_valid = [
        frame for frame in selected if frame.timestamp_ns >= recent_cutoff
    ]
    latest_valid_timestamp = max((frame.timestamp_ns for frame in recent_valid), default=0)
    if len(recent_valid) < 8:
        reject("insufficient_recent_valid_observations")
        return None
    if latest_valid_timestamp == 0 or t0 - latest_valid_timestamp > int(MAX_LATEST_AGE_S * 1e9):
        reject("latest_valid_observation_too_old")
        return None
    future_position = np.zeros((len(query_taus), MAX_ARMORS, 3), dtype=np.float32)
    future_normal = np.zeros_like(future_position)
    effective_taus = np.zeros((len(query_taus),), dtype=np.float32)
    future_timestamps = np.zeros((len(query_taus),), dtype=np.int64)
    for query_index, tau in enumerate(query_taus):
        future = _nearest_truth(
            truths, t0 + int(tau * 1e9),
            producer_epoch=None if inputs_epoch_filtered else anchor.producer_epoch,
        )
        if future is None:
            reject("missing_future_truth")
            return None
        if motion_segment is not None and future.timestamp_ns >= segment_end_ns:
            reject("cross_motion_segment_future")
            return None
        if tau >= 0.0 and future.timestamp_ns < t0:
            reject("future_truth_precedes_anchor")
            return None
        if len(future.armors) != MAX_ARMORS:
            reject("invalid_future_armor_count")
            return None
        effective_taus[query_index] = (future.timestamp_ns - t0) / 1e9
        future_timestamps[query_index] = future.timestamp_ns
        for armor_index, armor in enumerate(future.armors):
            future_position[query_index, armor_index] = _world_to_tracker(
                armor.position_world_m,
                anchor_truth.gimbal_origin_world_m,
                tracker_quaternion,
            )
            future_normal[query_index, armor_index] = _quat_rotate(
                (tracker_quaternion[0], -tracker_quaternion[1],
                 -tracker_quaternion[2], -tracker_quaternion[3]),
                armor.outward_normal_world,
            )
            normal_norm = np.linalg.norm(future_normal[query_index, armor_index])
            if normal_norm < 1e-6:
                reject("invalid_future_normal")
                return None
            future_normal[query_index, armor_index] /= normal_norm
    history_start_ns = int(selected[0].timestamp_ns)
    future_end_ns = int(future_timestamps.max(initial=t0))
    if motion_segment is not None:
        if history_start_ns < segment_start_ns:
            reject("cross_motion_segment_history")
            return None
        if history_start_ns - segment_start_ns < MOTION_SEGMENT_TIME_GUARD_NS:
            reject("history_too_close_to_motion_segment_start")
            return None
        if segment_end_ns - 1 - future_end_ns < MOTION_SEGMENT_TIME_GUARD_NS:
            reject("future_too_close_to_motion_segment_end")
            return None
        if not _constant_truth_window(truths, history_start_ns, future_end_ns):
            reject("nonconstant_full_motion_window")
            return None
    if rng is not None:
        original_obs_mask = obs_mask.copy()
        original_event_mask = event_mask.copy()
        original_event_time_s = event_time_s.copy()
        for slot in range(EVENT_COUNT):
            for armor_index in range(MAX_ARMORS):
                if obs_mask[slot, armor_index] and rng.random() < 0.05:
                    obs_mask[slot, armor_index] = False
        if rng.random() < 0.30:
            valid_indices = np.flatnonzero(event_mask)
            if len(valid_indices) >= 2:
                length = min(int(rng.integers(2, 11)), len(valid_indices))
                start = int(rng.integers(0, len(valid_indices) - length + 1))
                drop = valid_indices[start:start + length]
                obs_mask[drop] = False
        keep = np.flatnonzero(obs_mask.any(axis=1))
        compact_obs = np.zeros_like(obs)
        compact_obs_mask = np.zeros_like(obs_mask)
        compact_event_mask = np.zeros_like(event_mask)
        compact_event_time_s = np.zeros_like(event_time_s)
        compact_start = EVENT_COUNT - len(keep)
        compact_obs[compact_start:] = obs[keep]
        compact_obs_mask[compact_start:] = obs_mask[keep]
        compact_event_mask[compact_start:] = True
        compact_event_time_s[compact_start:] = event_time_s[keep]
        recent_count = np.count_nonzero(
            compact_event_mask & (compact_event_time_s >= -MIN_HISTORY_S)
        )
        if recent_count >= 8 and compact_event_time_s[-1] >= -MAX_LATEST_AGE_S:
            obs, obs_mask = compact_obs, compact_obs_mask
            event_mask, event_time_s = compact_event_mask, compact_event_time_s
        else:
            obs_mask = original_obs_mask
            event_mask = original_event_mask
            event_time_s = original_event_time_s
    if diagnostics is not None and motion_segment is not None:
        admitted = diagnostics.setdefault("admitted_motion_command_epochs", Counter())
        assert isinstance(admitted, Counter)
        admitted[int(motion_segment["motion_command_epoch"])] += 1
        history_margin = history_start_ns - segment_start_ns
        future_margin = segment_end_ns - 1 - future_end_ns
        diagnostics["minimum_history_margin_to_segment_start_ns"] = min(
            int(diagnostics.get("minimum_history_margin_to_segment_start_ns", history_margin)),
            history_margin,
        )
        diagnostics["minimum_future_margin_to_segment_end_ns"] = min(
            int(diagnostics.get("minimum_future_margin_to_segment_end_ns", future_margin)),
            future_margin,
        )
    return TensorSample(
        session_id=anchor.session_id,
        t0_ns=t0,
        obs=obs,
        obs_mask=obs_mask,
        event_mask=event_mask,
        event_time_s=event_time_s,
        tau=effective_taus,
        tau_requested=np.asarray(query_taus, dtype=np.float32),
        future_timestamp_ns=future_timestamps,
        future_position=future_position,
        future_normal=future_normal,
        motion_class=_motion_class(
            motion_segment if motion_segment is not None else manifest,
            anchor_truth,
        ),
        motion_command_epoch=(
            int(motion_segment["motion_command_epoch"])
            if motion_segment is not None else -1
        ),
        motion_segment_start_ns=segment_start_ns,
        motion_segment_end_ns=segment_end_ns,
        history_start_ns=history_start_ns,
        future_end_ns=future_end_ns,
    )


def build_samples(
    observations_path: str | Path,
    truth_path: str | Path,
    manifests: Mapping[str, Mapping[str, object]] | None = None,
    extrinsic: CameraGimbalExtrinsic | None = None,
    anchor_hz: float = 20.0,
    augment: bool = False,
    seed: int = 0,
    diagnostics: dict[str, object] | None = None,
    min_history_timestamp_ns: int | None = None,
    preloaded_observations: Iterable[ObservationFrame] | None = None,
    preloaded_truth: Iterable[TruthFrame] | None = None,
    motion_segments: Mapping[str, tuple[Mapping[str, object], ...]] | None = None,
) -> list[TensorSample]:
    if extrinsic is None:
        raise ValueError("Stage-3 v3 tensorization requires calibrated camera/gimbal R/T")
    if preloaded_observations is None:
        observations = _load_observations(observations_path)
    else:
        observations: dict[str, list[ObservationFrame]] = {}
        for frame in preloaded_observations:
            observations.setdefault(frame.session_id, []).append(frame)
        for frames in observations.values():
            frames.sort(key=lambda frame: (frame.producer_epoch, frame.timestamp_ns, frame.frame_seq))
    if preloaded_truth is None:
        truth_by_key = _load_truth(truth_path)
    else:
        truth_by_key: dict[tuple[str, int, int, int], TruthFrame] = {}
        for truth in preloaded_truth:
            key = (truth.session_id, truth.producer_epoch, truth.frame_seq, truth.timestamp_ns)
            if key in truth_by_key:
                raise ValueError(f"duplicate exact truth key: {key}")
            truth_by_key[key] = truth
    truth_by_session: dict[str, list[TruthFrame]] = {}
    for truth in truth_by_key.values():
        truth_by_session.setdefault(truth.session_id, []).append(truth)
    for frames in truth_by_session.values():
        frames.sort(key=lambda frame: frame.timestamp_ns)
    rng = np.random.default_rng(seed) if augment else None
    step_ns = int(1e9 / anchor_hz)
    result: list[TensorSample] = []
    for session_id, frames in observations.items():
        truths = truth_by_session.get(session_id, [])
        if not truths:
            continue
        observations_by_epoch: dict[int, list[ObservationFrame]] = {}
        truths_by_epoch: dict[int, list[TruthFrame]] = {}
        for frame in frames:
            observations_by_epoch.setdefault(frame.producer_epoch, []).append(frame)
        for truth in truths:
            truths_by_epoch.setdefault(truth.producer_epoch, []).append(truth)
        for epoch, epoch_frames in observations_by_epoch.items():
            epoch_truths = truths_by_epoch.get(epoch, [])
            if not epoch_truths:
                continue
            last_anchor = -10**30
            for anchor_index, anchor in enumerate(epoch_frames):
                if anchor.timestamp_ns - last_anchor < step_ns:
                    continue
                segment = None
                if motion_segments is not None and session_id in motion_segments:
                    segment = next((
                        value for value in motion_segments[session_id]
                        if int(value["start_timestamp_ns"]) <= anchor.timestamp_ns
                        < int(value["end_timestamp_ns"])
                    ), None)
                    if segment is None:
                        if diagnostics is not None:
                            counts = diagnostics.setdefault("rejection_counts", Counter())
                            assert isinstance(counts, Counter)
                            counts["anchor_outside_motion_segment"] += 1
                        continue
                history_event_limit = (
                    FORMAL_MOTION_HISTORY_EVENT_LIMIT
                    if segment is not None else EVENT_COUNT
                )
                history_frames = epoch_frames[:anchor_index + 1]
                if segment is not None:
                    segment_start_ns = int(segment["start_timestamp_ns"])
                    history_frames = [
                        frame for frame in history_frames
                        if frame.timestamp_ns >= segment_start_ns
                    ]
                valid_history = [
                    frame for frame in history_frames
                    if any(
                        armor.valid and all(math.isfinite(value) for value in (*armor.position_m, armor.yaw_rad))
                        for armor in frame.armors
                    )
                ][-history_event_limit:]
                history_start_ns = (
                    (
                        valid_history[0].timestamp_ns
                        if valid_history else history_frames[0].timestamp_ns
                    )
                    if segment is not None
                    else (
                        valid_history[0].timestamp_ns
                        if len(valid_history) == history_event_limit
                        else epoch_frames[0].timestamp_ns
                    )
                )
                if min_history_timestamp_ns is not None and history_start_ns < min_history_timestamp_ns:
                    if diagnostics is not None:
                        counts = diagnostics.setdefault("rejection_counts", Counter())
                        assert isinstance(counts, Counter)
                        counts["ego_unstable_history"] += 1
                    continue
                key = anchor.key
                if key not in truth_by_key:
                    if diagnostics is not None:
                        counts = diagnostics.setdefault("rejection_counts", Counter())
                        assert isinstance(counts, Counter)
                        counts["missing_exact_anchor_truth"] += 1
                    continue
                query_seed = int.from_bytes(hashlib.sha256(
                    f"{seed}:{anchor.session_id}:{anchor.producer_epoch}:{anchor.frame_seq}:{anchor.timestamp_ns}".encode("utf-8")
                ).digest()[:8], "little")
                query_rng = np.random.default_rng(query_seed)
                random_taus = tuple(float(value) for value in query_rng.uniform(0.0, 0.5, 4))
                query_taus = QUERY_TAUS_S + random_taus
                sample = _make_sample(
                    anchor, epoch_frames[:anchor_index + 1], epoch_truths,
                    (manifests or {}).get(session_id), extrinsic, rng, query_taus, diagnostics,
                    truth_by_key=truth_by_key, inputs_epoch_filtered=True,
                    motion_segment=segment,
                )
                if sample is not None:
                    result.append(sample)
                    last_anchor = anchor.timestamp_ns
    if diagnostics is not None:
        diagnostics["sample_count"] = len(result)
        if result:
            tau_errors = np.concatenate([
                np.abs(sample.tau - sample.tau_requested) for sample in result
            ])
            diagnostics["tau_error_ms"] = {
                "max": float(tau_errors.max() * 1000.0),
                "p50": float(np.quantile(tau_errors, 0.50) * 1000.0),
                "p95": float(np.quantile(tau_errors, 0.95) * 1000.0),
                "p99": float(np.quantile(tau_errors, 0.99) * 1000.0),
            }
    return result


def samples_to_arrays(samples: list[TensorSample]) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("no valid samples were built")
    arrays = {
        "obs": np.stack([sample.obs for sample in samples]),
        "obs_mask": np.stack([sample.obs_mask for sample in samples]),
        "event_mask": np.stack([sample.event_mask for sample in samples]),
        "event_time_s": np.stack([sample.event_time_s for sample in samples]),
        "tau": np.stack([sample.tau for sample in samples]),
        "tau_requested": np.stack([sample.tau_requested for sample in samples]),
        "tau_error_s": np.stack([sample.tau - sample.tau_requested for sample in samples]),
        "future_timestamp_ns": np.stack([sample.future_timestamp_ns for sample in samples]),
        "future_position": np.stack([sample.future_position for sample in samples]),
        "future_normal": np.stack([sample.future_normal for sample in samples]),
        "motion_class": np.asarray([sample.motion_class for sample in samples], dtype=np.int64),
        "session_id": np.asarray([sample.session_id for sample in samples]),
        "t0_ns": np.asarray([sample.t0_ns for sample in samples], dtype=np.int64),
    }
    segment_audited = np.asarray(
        [sample.motion_command_epoch >= 0 for sample in samples], dtype=np.bool_
    )
    if segment_audited.any() and not segment_audited.all():
        raise ValueError("one shard cannot mix ACK-audited and legacy samples")
    if segment_audited.all():
        arrays.update({
            "motion_command_epoch": np.asarray(
                [sample.motion_command_epoch for sample in samples], dtype=np.int64
            ),
            "motion_segment_start_ns": np.asarray(
                [sample.motion_segment_start_ns for sample in samples], dtype=np.int64
            ),
            "motion_segment_end_ns": np.asarray(
                [sample.motion_segment_end_ns for sample in samples], dtype=np.int64
            ),
            "history_start_ns": np.asarray(
                [sample.history_start_ns for sample in samples], dtype=np.int64
            ),
            "future_end_ns": np.asarray(
                [sample.future_end_ns for sample in samples], dtype=np.int64
            ),
            "window_constant_motion": np.ones((len(samples),), dtype=np.bool_),
        })
    return arrays
