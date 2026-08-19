#!/usr/bin/env python3
"""Truth-free causal camera-ray observer for unordered armor candidates.

The observer deliberately does not resolve a physical armor slot.  It turns a
stream of complete frame events into short-lived anonymous ray handles while
making missing, stale, ambiguous, and invalid-stream states explicit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class ObserverConfig:
    history_window_s: float = 0.2
    min_history_events: int = 8
    freshness_s: float = 0.05
    continuity_gate_deg: float = 2.0
    ambiguity_margin_deg: float = 0.25
    max_candidates: int = 4
    max_history_events: int = 200


@dataclass
class _RayTrack:
    handle_id: int
    history: list[tuple[int, float, float, float]] = field(default_factory=list)

    def predict(self, timestamp_ns: int) -> tuple[float, float]:
        _last_ns, last_u, last_v, _depth = self.history[-1]
        if len(self.history) < 2:
            return last_u, last_v
        previous_ns, previous_u, previous_v, _ = self.history[-2]
        dt_s = (self.history[-1][0] - previous_ns) * 1e-9
        if dt_s <= 0.0:
            return last_u, last_v
        future_s = min(max((timestamp_ns - self.history[-1][0]) * 1e-9, 0.0), 0.35)
        du_dt = wrap_degrees(last_u - previous_u) / dt_s
        dv_dt = (last_v - previous_v) / dt_s
        return last_u + du_dt * future_s, last_v + dv_dt * future_s

    def append(
        self,
        timestamp_ns: int,
        u_deg: float,
        v_deg: float,
        depth_m: float,
        limit: int,
    ) -> None:
        self.history.append((timestamp_ns, u_deg, v_deg, depth_m))
        self.history = self.history[-limit:]


class CausalRayObserver:
    """Stateful observer satisfying the conservative observer-v1 boundary."""

    def __init__(self, config: ObserverConfig | None = None) -> None:
        self.config = config or ObserverConfig()
        self._session_id: str | None = None
        self._producer_epoch: int | None = None
        self._last_frame_seq: int | None = None
        self._last_timestamp_ns: int | None = None
        self._last_valid_timestamp_ns: int | None = None
        self._tracks: dict[int, _RayTrack] = {}
        self._next_handle_id = 0
        self._reacquiring = False

    def reset(self) -> None:
        self._session_id = None
        self._producer_epoch = None
        self._last_frame_seq = None
        self._last_timestamp_ns = None
        self._last_valid_timestamp_ns = None
        self._tracks.clear()
        self._next_handle_id = 0
        self._reacquiring = False

    def _clear_dynamic_state(self, reacquiring: bool) -> None:
        self._tracks.clear()
        self._reacquiring = reacquiring

    @staticmethod
    def _ray_candidate(candidate: dict[str, Any]) -> dict[str, Any] | None:
        if candidate.get("valid") is False:
            return None
        tvec = candidate.get("camera_tvec_m")
        if not isinstance(tvec, (list, tuple)) or len(tvec) != 3:
            return None
        try:
            right, down, forward = (float(value) for value in tvec)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (right, down, forward)) or forward <= 0.0:
            return None
        u_deg = math.degrees(math.atan2(right, forward))
        v_deg = math.degrees(math.atan2(down, forward))
        yaw = candidate.get("yaw_absolute_rad")
        yaw_rad = None
        if yaw is not None:
            try:
                value = float(yaw)
                yaw_rad = value if math.isfinite(value) else None
            except (TypeError, ValueError):
                pass
        return {
            "u_deg": u_deg,
            "v_deg": v_deg,
            "depth_m": forward,
            "raw_camera_tvec_m": [right, down, forward],
            "observed_yaw_rad": yaw_rad,
            "measurement_quality_features": {
                "detector_confidence": candidate.get("detector_confidence"),
                "reprojection_rms_px": candidate.get("reprojection_rms_px"),
                "reprojection_max_px": candidate.get("reprojection_max_px"),
                "corner_source": candidate.get("corner_source"),
            },
        }

    @staticmethod
    def _candidate_key(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
        yaw = candidate["observed_yaw_rad"]
        return (
            round(float(candidate["u_deg"]), 12),
            round(float(candidate["v_deg"]), 12),
            round(float(candidate["depth_m"]), 12),
            round(float(yaw), 12) if yaw is not None else float("inf"),
        )

    def _base_output(self, frame: dict[str, Any]) -> dict[str, Any]:
        timestamp_ns = int(frame["capture_timestamp_ns"])
        age_ns = (
            None
            if self._last_valid_timestamp_ns is None
            else max(timestamp_ns - self._last_valid_timestamp_ns, 0)
        )
        return {
            "schema_version": "autoaim-causal-ray-observer-output-v1",
            "observer_status": "NO_DATA",
            "status_reason": [],
            "observation_timestamp_ns": timestamp_ns,
            "age_ns": age_ns,
            "frame_availability": False,
            "candidate_count": len(frame.get("candidates", [])),
            "valid_candidate_count": 0,
            "anonymous_handles": [],
            "set_ambiguity_status": "none",
            "camera_frame_only": True,
            "physical_identity_resolved": False,
            "world_state_valid": False,
            "prediction_valid": False,
            "fire_control_valid": False,
            "uncertainty_status": "uncalibrated",
            "downstream_eligibility": "none",
        }

    def _invalid(self, frame: dict[str, Any], reason: str) -> dict[str, Any]:
        output = self._base_output(frame)
        output["observer_status"] = "INVALID_STREAM"
        output["status_reason"] = [reason]
        self.reset()
        return output

    def _validate_stream(self, frame: dict[str, Any]) -> str | None:
        required = ("session_id", "producer_epoch", "frame_seq", "capture_timestamp_ns")
        if any(key not in frame for key in required):
            return "missing_stream_identity"
        if frame.get("observation_sink_status", "ok") != "ok":
            return "observation_sink_failure"
        session_id = str(frame["session_id"])
        producer_epoch = int(frame["producer_epoch"])
        frame_seq = int(frame["frame_seq"])
        timestamp_ns = int(frame["capture_timestamp_ns"])
        if self._session_id is None:
            self._session_id = session_id
            self._producer_epoch = producer_epoch
        elif session_id != self._session_id or producer_epoch != self._producer_epoch:
            return "session_or_epoch_change"
        if self._last_frame_seq is not None and frame_seq <= self._last_frame_seq:
            return "frame_sequence_duplicate_or_regression"
        if self._last_timestamp_ns is not None and timestamp_ns <= self._last_timestamp_ns:
            return "timestamp_duplicate_or_regression"
        self._last_frame_seq = frame_seq
        self._last_timestamp_ns = timestamp_ns
        return None

    def _association_cost(
        self, track: _RayTrack, candidate: dict[str, Any], timestamp_ns: int
    ) -> float:
        predicted_u, predicted_v = track.predict(timestamp_ns)
        return math.hypot(
            wrap_degrees(float(candidate["u_deg"]) - predicted_u),
            float(candidate["v_deg"]) - predicted_v,
        )

    def _associate(
        self, candidates: list[dict[str, Any]], timestamp_ns: int
    ) -> tuple[list[tuple[dict[str, Any], int]], str | None]:
        if not self._tracks:
            assignments = []
            for candidate in candidates:
                handle_id = self._next_handle_id
                self._next_handle_id += 1
                assignments.append((candidate, handle_id))
            return assignments, None

        ranked_by_candidate: list[list[tuple[float, int]]] = []
        for candidate in candidates:
            ranked = sorted(
                (
                    self._association_cost(track, candidate, timestamp_ns),
                    handle_id,
                )
                for handle_id, track in self._tracks.items()
            )
            ranked_by_candidate.append(ranked)

        proposed: list[int | None] = []
        for ranked in ranked_by_candidate:
            within = [item for item in ranked if item[0] <= self.config.continuity_gate_deg]
            if len(within) >= 2 and within[1][0] - within[0][0] <= self.config.ambiguity_margin_deg:
                return [], "close_assignment_cost"
            proposed.append(within[0][1] if within else None)
        chosen = [value for value in proposed if value is not None]
        if len(chosen) != len(set(chosen)):
            return [], "competing_candidates_for_handle"
        free_capacity = self.config.max_candidates - len(self._tracks)
        if sum(value is None for value in proposed) > free_capacity:
            return [], "anonymous_handle_capacity"
        assignments = []
        for candidate, handle_id in zip(candidates, proposed):
            if handle_id is None:
                handle_id = self._next_handle_id
                self._next_handle_id += 1
            assignments.append((candidate, handle_id))
        return assignments, None

    def _handle_output(
        self,
        track: _RayTrack,
        candidate: dict[str, Any],
        timestamp_ns: int,
        continuity_residual_deg: float | None,
    ) -> dict[str, Any]:
        cutoff_ns = timestamp_ns - int(self.config.history_window_s * 1e9)
        recent = [row for row in track.history if row[0] >= cutoff_ns]
        timestamps = np.asarray([(row[0] - timestamp_ns) * 1e-9 for row in recent], dtype=float)
        u_values = np.asarray([row[1] for row in recent], dtype=float)
        v_values = np.asarray([row[2] for row in recent], dtype=float)
        du_dt = dv_dt = None
        if len(recent) >= 2 and float(np.ptp(timestamps)) > 0.0:
            centered = timestamps - float(np.mean(timestamps))
            denominator = float(np.dot(centered, centered))
            du_unwrapped = np.degrees(np.unwrap(np.radians(u_values)))
            du_dt = float(np.dot(centered, du_unwrapped - np.mean(du_unwrapped)) / denominator)
            dv_dt = float(np.dot(centered, v_values - np.mean(v_values)) / denominator)
        gaps = [
            (later[0] - earlier[0]) * 1e-9
            for earlier, later in zip(recent, recent[1:])
        ]
        qualified = len(recent) >= self.config.min_history_events
        return {
            "ephemeral_handle_id": track.handle_id,
            "u_deg": float(candidate["u_deg"]),
            "v_deg": float(candidate["v_deg"]),
            "du_dt_deg_s": du_dt if qualified else None,
            "dv_dt_deg_s": dv_dt if qualified else None,
            "raw_camera_tvec_m": candidate["raw_camera_tvec_m"],
            "depth_status": "raw_uncalibrated",
            "history_event_count": len(recent),
            "history_span_s": (recent[-1][0] - recent[0][0]) * 1e-9 if recent else 0.0,
            "max_gap_s": max(gaps) if gaps else 0.0,
            "last_gap_s": gaps[-1] if gaps else 0.0,
            "local_continuity_residual_deg": continuity_residual_deg,
            "measurement_quality_features": candidate["measurement_quality_features"],
            "angular_uncertainty_interval": None,
            "qualified": qualified,
        }

    def update(self, frame: dict[str, Any]) -> dict[str, Any]:
        stream_error = self._validate_stream(frame)
        if stream_error is not None:
            return self._invalid(frame, stream_error)
        timestamp_ns = int(frame["capture_timestamp_ns"])
        output = self._base_output(frame)
        raw_candidates = frame.get("candidates", [])
        if not isinstance(raw_candidates, list):
            return self._invalid(frame, "candidate_set_not_a_list")
        candidates = [candidate for raw in raw_candidates if (candidate := self._ray_candidate(raw))]
        candidates.sort(key=self._candidate_key)
        output["valid_candidate_count"] = len(candidates)

        if len(candidates) > self.config.max_candidates:
            self._clear_dynamic_state(reacquiring=True)
            output.update(
                observer_status="AMBIGUOUS_SET",
                status_reason=["candidate_count_exceeds_maximum"],
                set_ambiguity_status="too_many_candidates",
            )
            return output
        keys = [self._candidate_key(candidate) for candidate in candidates]
        if len(keys) != len(set(keys)):
            self._clear_dynamic_state(reacquiring=True)
            output.update(
                observer_status="AMBIGUOUS_SET",
                status_reason=["indistinguishable_duplicate_candidates"],
                set_ambiguity_status="duplicate_geometry",
            )
            return output
        if not candidates:
            output["status_reason"] = ["no_valid_candidate"]
            if self._last_valid_timestamp_ns is None:
                output["observer_status"] = "NO_DATA" if not self._reacquiring else "REACQUIRING"
                return output
            age_s = (timestamp_ns - self._last_valid_timestamp_ns) * 1e-9
            output["age_ns"] = timestamp_ns - self._last_valid_timestamp_ns
            if age_s > self.config.freshness_s:
                self._clear_dynamic_state(reacquiring=True)
                output["observer_status"] = "STALE"
                output["status_reason"] = ["latest_valid_event_too_old"]
            else:
                output["observer_status"] = "ACQUIRING"
                output["status_reason"] = ["current_frame_missing"]
            return output

        if (
            self._last_valid_timestamp_ns is not None
            and (timestamp_ns - self._last_valid_timestamp_ns) * 1e-9 > self.config.freshness_s
        ):
            self._clear_dynamic_state(reacquiring=True)
        prior_predictions = {
            handle_id: track.predict(timestamp_ns) for handle_id, track in self._tracks.items()
        }
        assignments, ambiguity = self._associate(candidates, timestamp_ns)
        if ambiguity is not None:
            self._clear_dynamic_state(reacquiring=True)
            output.update(
                observer_status="AMBIGUOUS_SET",
                status_reason=[ambiguity],
                set_ambiguity_status="assignment_ambiguous",
            )
            return output

        handles = []
        for candidate, handle_id in assignments:
            track = self._tracks.get(handle_id)
            residual = None
            if track is None:
                track = _RayTrack(handle_id)
                self._tracks[handle_id] = track
            else:
                predicted_u, predicted_v = prior_predictions[handle_id]
                residual = math.hypot(
                    wrap_degrees(float(candidate["u_deg"]) - predicted_u),
                    float(candidate["v_deg"]) - predicted_v,
                )
            track.append(
                timestamp_ns,
                float(candidate["u_deg"]),
                float(candidate["v_deg"]),
                float(candidate["depth_m"]),
                self.config.max_history_events,
            )
            handles.append(self._handle_output(track, candidate, timestamp_ns, residual))
        self._last_valid_timestamp_ns = timestamp_ns
        output["age_ns"] = 0
        output["frame_availability"] = True
        output["anonymous_handles"] = sorted(handles, key=lambda row: row["ephemeral_handle_id"])
        if any(handle["qualified"] for handle in handles):
            output["observer_status"] = "OBSERVED_ANONYMOUS"
            output["downstream_eligibility"] = "anonymous_current_state_only"
            output["status_reason"] = ["qualified_anonymous_ray"]
            self._reacquiring = False
        elif self._reacquiring:
            output["observer_status"] = "REACQUIRING"
            output["status_reason"] = ["history_rebuilding_after_reset"]
        else:
            output["observer_status"] = "ACQUIRING"
            output["status_reason"] = ["insufficient_recent_history"]
        return output
