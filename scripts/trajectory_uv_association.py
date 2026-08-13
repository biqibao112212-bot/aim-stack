#!/usr/bin/env python3
"""Serializable same-armor pair scorer and causal four-track associator."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = (
    "dt_s",
    "log1p_dt",
    "delta_u_deg",
    "delta_v_deg",
    "distance_deg",
    "predicted_residual_u_deg",
    "predicted_residual_v_deg",
    "predicted_residual_deg",
    "track_velocity_u_deg_s",
    "track_velocity_v_deg_s",
    "implied_velocity_u_deg_s",
    "implied_velocity_v_deg_s",
    "velocity_change_deg_s",
    "track_u_deg",
    "track_v_deg",
    "detection_u_deg",
    "detection_v_deg",
)


def wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def pair_feature(
    history: list[tuple[float, float, float]],
    timestamp_s: float,
    u_deg: float,
    v_deg: float,
) -> np.ndarray:
    if not history:
        raise ValueError("pair feature requires an existing track")
    last_t, last_u, last_v = history[-1]
    dt = max(float(timestamp_s) - last_t, 1e-5)
    delta_u = wrap_degrees(float(u_deg) - last_u)
    delta_v = float(v_deg) - last_v
    if len(history) >= 2:
        previous_t, previous_u, previous_v = history[-2]
        previous_dt = max(last_t - previous_t, 1e-5)
        velocity_u = wrap_degrees(last_u - previous_u) / previous_dt
        velocity_v = (last_v - previous_v) / previous_dt
    else:
        velocity_u = 0.0
        velocity_v = 0.0
    prediction_dt = min(dt, 0.35)
    predicted_u = last_u + velocity_u * prediction_dt
    predicted_v = last_v + velocity_v * prediction_dt
    residual_u = wrap_degrees(float(u_deg) - predicted_u)
    residual_v = float(v_deg) - predicted_v
    implied_u = delta_u / dt
    implied_v = delta_v / dt
    return np.asarray(
        [
            dt,
            math.log1p(dt),
            delta_u,
            delta_v,
            math.hypot(delta_u, delta_v),
            residual_u,
            residual_v,
            math.hypot(residual_u, residual_v),
            velocity_u,
            velocity_v,
            implied_u,
            implied_v,
            math.hypot(implied_u - velocity_u, implied_v - velocity_v),
            last_u,
            last_v,
            float(u_deg),
            float(v_deg),
        ],
        dtype=float,
    )


class PairLogisticModel:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        metadata: dict[str, Any] | None = None,
        max_examples: int = 250000,
    ) -> "PairLogisticModel":
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError("invalid pair-feature matrix")
        if len(np.unique(labels)) != 2:
            raise ValueError("pair model requires positive and negative examples")
        if len(features) > max_examples:
            # Deterministic, order-independent enough for the stable input sort.
            indices = np.linspace(0, len(features) - 1, max_examples).astype(int)
            features = features[indices]
            labels = labels[indices]
        median = np.nanmedian(features, axis=0)
        features = np.where(np.isfinite(features), features, median)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=17,
        )
        classifier.fit(scaled, labels)
        probabilities = classifier.predict_proba(scaled)[:, 1]
        payload = {
            "schema_version": 1,
            "kind": "causal_uv_same_armor_pair_logistic",
            "runtime_input_contract": ["timestamp_s", "u_deg", "v_deg"],
            "truth_is_runtime_input": False,
            "feature_names": list(FEATURE_NAMES),
            "feature_median": median.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "coef": classifier.coef_[0].tolist(),
            "intercept": float(classifier.intercept_[0]),
            "training_examples": len(features),
            "positive_fraction": float(np.mean(labels)),
            "training_probability_p05": float(np.percentile(probabilities, 5)),
            "training_probability_p50": float(np.percentile(probabilities, 50)),
            "training_probability_p95": float(np.percentile(probabilities, 95)),
            "metadata": dict(metadata or {}),
        }
        return cls(payload)

    def probability(self, feature: np.ndarray) -> float:
        values = np.asarray(feature, dtype=float)
        if values.shape != (len(FEATURE_NAMES),):
            raise ValueError("invalid pair feature")
        median = np.asarray(self.payload["feature_median"], dtype=float)
        values = np.where(np.isfinite(values), values, median)
        scaled = (values - np.asarray(self.payload["scaler_mean"], dtype=float)) / np.asarray(
            self.payload["scaler_scale"], dtype=float
        )
        logit = float(np.dot(np.asarray(self.payload["coef"], dtype=float), scaled)) + float(
            self.payload["intercept"]
        )
        if logit >= 0.0:
            return 1.0 / (1.0 + math.exp(-logit))
        exponential = math.exp(logit)
        return exponential / (1.0 + exponential)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PairLogisticModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("truth_is_runtime_input") is not False:
            raise ValueError("invalid pair-model contract")
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("pair-model feature contract mismatch")
        return cls(payload)


@dataclass
class TrackState:
    history: list[tuple[float, float, float]]


class LearnedCausalAssociator:
    def __init__(
        self,
        model: PairLogisticModel,
        birth_probability: float = 0.5,
        drop_probability: float = 0.05,
        geometry_gate_deg: float = 25.0,
        max_tracks: int = 4,
    ) -> None:
        if not 0.0 < drop_probability < birth_probability < 1.0:
            raise ValueError("require 0 < drop_probability < birth_probability < 1")
        self.model = model
        self.birth_probability = birth_probability
        self.drop_probability = drop_probability
        self.geometry_gate_deg = geometry_gate_deg
        self.tracks: list[TrackState | None] = [None] * max_tracks

    def _probability(self, track: TrackState, detection: dict[str, float]) -> float:
        feature = pair_feature(
            track.history,
            float(detection["t_s"]),
            math.degrees(float(detection["u"])),
            math.degrees(float(detection["v"])),
        )
        if float(feature[4]) > self.geometry_gate_deg:
            return 0.0
        return self.model.probability(feature)

    def _update(self, track_id: int, detection: dict[str, float]) -> None:
        observation = (
            float(detection["t_s"]),
            math.degrees(float(detection["u"])),
            math.degrees(float(detection["v"])),
        )
        track = self.tracks[track_id]
        if track is None:
            self.tracks[track_id] = TrackState([observation])
        else:
            if observation[0] <= track.history[-1][0]:
                raise ValueError("association timestamps must increase")
            track.history.append(observation)
            track.history = track.history[-2:]

    def update(self, detections: list[dict[str, float]]) -> list[tuple[dict[str, float], int | None]]:
        if not detections:
            return []
        active = [index for index, track in enumerate(self.tracks) if track is not None]
        free = [index for index, track in enumerate(self.tracks) if track is None]
        track_columns = active + free
        drop_offset = len(track_columns)
        large = 1e6
        costs = np.full((len(detections), len(track_columns) + len(detections)), large)
        for row, detection in enumerate(detections):
            for column, track_id in enumerate(track_columns):
                probability = (
                    self.birth_probability
                    if track_id in free
                    else self._probability(self.tracks[track_id], detection)
                )
                if probability > 0.0:
                    costs[row, column] = -math.log(max(probability, 1e-9))
            costs[row, drop_offset + row] = -math.log(self.drop_probability)
        row_indices, column_indices = linear_sum_assignment(costs)
        choices: list[int | None] = [None] * len(detections)
        for row, column in zip(row_indices, column_indices):
            if column < len(track_columns) and costs[row, column] < large:
                choices[row] = track_columns[column]
        result = []
        for detection, track_id in zip(detections, choices):
            if track_id is not None:
                self._update(track_id, detection)
            result.append((detection, track_id))
        return result


class CyclicSegmentAssociator:
    """Causal segment tracker with modulo-four birth ordering.

    Short gaps use local CV continuity.  A detection that cannot be matched to
    a recently visible segment starts the next cyclic ID.  The absolute ID and
    birth direction are arbitrary; scoring remains invariant to one global
    permutation.
    """

    def __init__(
        self,
        continuity_gate_deg: float = 1.0,
        reacquire_timeout_s: float = 0.5,
        max_tracks: int = 4,
        birth_descending_u: bool = False,
    ) -> None:
        self.continuity_gate_deg = continuity_gate_deg
        self.reacquire_timeout_s = reacquire_timeout_s
        self.max_tracks = max_tracks
        self.birth_descending_u = birth_descending_u
        self.tracks: list[TrackState | None] = [None] * max_tracks
        self.next_birth_id = 0

    def _prediction_cost(self, track: TrackState, detection: dict[str, float]) -> float:
        timestamp_s = float(detection["t_s"])
        dt = timestamp_s - track.history[-1][0]
        if dt <= 0.0 or dt > self.reacquire_timeout_s:
            return float("inf")
        last_t, last_u, last_v = track.history[-1]
        velocity_u = velocity_v = 0.0
        if len(track.history) >= 2:
            previous_t, previous_u, previous_v = track.history[-2]
            previous_dt = max(last_t - previous_t, 1e-5)
            velocity_u = wrap_degrees(last_u - previous_u) / previous_dt
            velocity_v = (last_v - previous_v) / previous_dt
        prediction_dt = min(dt, 0.35)
        predicted_u = last_u + velocity_u * prediction_dt
        predicted_v = last_v + velocity_v * prediction_dt
        detected_u = math.degrees(float(detection["u"]))
        detected_v = math.degrees(float(detection["v"]))
        return math.hypot(
            wrap_degrees(detected_u - predicted_u), detected_v - predicted_v
        )

    def _write_track(self, track_id: int, detection: dict[str, float], reset: bool) -> None:
        observation = (
            float(detection["t_s"]),
            math.degrees(float(detection["u"])),
            math.degrees(float(detection["v"])),
        )
        if reset or self.tracks[track_id] is None:
            self.tracks[track_id] = TrackState([observation])
        else:
            track = self.tracks[track_id]
            if observation[0] <= track.history[-1][0]:
                raise ValueError("cyclic association timestamps must increase")
            track.history.append(observation)
            track.history = track.history[-2:]

    def _birth_id(self, timestamp_s: float, reserved: set[int]) -> int | None:
        for offset in range(self.max_tracks):
            candidate = (self.next_birth_id + offset) % self.max_tracks
            track = self.tracks[candidate]
            recently_active = (
                track is not None
                and timestamp_s - track.history[-1][0] <= self.reacquire_timeout_s
            )
            if candidate not in reserved and not recently_active:
                self.next_birth_id = (candidate + 1) % self.max_tracks
                return candidate
        return None

    def update(self, detections: list[dict[str, float]]) -> list[tuple[dict[str, float], int | None]]:
        if not detections:
            return []
        candidate_pairs = []
        for detection_index, detection in enumerate(detections):
            for track_id, track in enumerate(self.tracks):
                if track is None:
                    continue
                cost = self._prediction_cost(track, detection)
                if cost <= self.continuity_gate_deg:
                    candidate_pairs.append((cost, detection_index, track_id))
        choices: list[int | None] = [None] * len(detections)
        match_costs: dict[int, float] = {}
        used_tracks: set[int] = set()
        for cost, detection_index, track_id in sorted(candidate_pairs):
            if choices[detection_index] is None and track_id not in used_tracks:
                choices[detection_index] = track_id
                match_costs[detection_index] = cost
                used_tracks.add(track_id)
        unmatched = [index for index, choice in enumerate(choices) if choice is None]
        unmatched.sort(
            key=lambda index: float(detections[index]["u"]),
            reverse=self.birth_descending_u,
        )
        for detection_index in unmatched:
            track_id = self._birth_id(float(detections[detection_index]["t_s"]), used_tracks)
            choices[detection_index] = track_id
            if track_id is not None:
                detections[detection_index]["association_mode"] = "cyclic_birth"
                detections[detection_index]["association_confidence"] = 0.25
                used_tracks.add(track_id)
                self._write_track(track_id, detections[detection_index], reset=True)
            else:
                detections[detection_index]["association_mode"] = "capacity_drop"
                detections[detection_index]["association_confidence"] = 0.0
        for detection_index, track_id in enumerate(choices):
            if detection_index in unmatched:
                continue
            if track_id is not None:
                cost = match_costs[detection_index]
                detections[detection_index]["association_mode"] = "continuity"
                detections[detection_index]["association_confidence"] = math.exp(
                    -cost / max(self.continuity_gate_deg, 1e-6)
                )
                self._write_track(int(track_id), detections[detection_index], reset=False)
        return list(zip(detections, choices))


def build_pair_training_rows(observed_rows: Iterable[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in observed_rows:
        by_run.setdefault(str(row["run"]), []).append(row)
    features: list[np.ndarray] = []
    labels: list[int] = []
    for rows in by_run.values():
        events: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            events.setdefault(int(row["timestamp_ns"]), []).append(row)
        histories: dict[int, list[tuple[float, float, float]]] = {}
        first_ns = min(events)
        for timestamp_ns, detections in sorted(events.items()):
            timestamp_s = (timestamp_ns - first_ns) * 1e-9
            for detection in detections:
                detection_slot = int(detection["slot"])
                for track_slot, history in histories.items():
                    features.append(
                        pair_feature(
                            history,
                            timestamp_s,
                            float(detection["u_deg"]),
                            float(detection["v_deg"]),
                        )
                    )
                    labels.append(int(track_slot == detection_slot))
            for detection in detections:
                slot = int(detection["slot"])
                history = histories.setdefault(slot, [])
                history.append(
                    (
                        timestamp_s,
                        float(detection["u_deg"]),
                        float(detection["v_deg"]),
                    )
                )
                histories[slot] = history[-2:]
    return np.vstack(features), np.asarray(labels, dtype=int)
