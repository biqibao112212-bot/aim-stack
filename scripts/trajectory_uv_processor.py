#!/usr/bin/env python3
"""Serializable causal u/v trajectory processor selected by Decision 215.

Runtime inputs are timestamped observed image-ray angles only.  Truth is used
outside this module to train residual targets and to score replay output.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


SCHEMA_VERSION = 1


def _finite(value: float, default: float = 0.0) -> float:
    converted = float(value)
    return converted if math.isfinite(converted) else default


def unwrap_degrees(values: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(values.astype(float))))


def polynomial_prediction(
    times: np.ndarray, values: np.ndarray, horizon_s: float, degree: int = 1
) -> float:
    local = times - times[-1]
    design = np.column_stack([local**power for power in range(degree + 1)])
    weights = np.linalg.lstsq(design, values, rcond=None)[0]
    return float(sum(weights[power] * horizon_s**power for power in range(degree + 1)))


def angular_error_deg(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    def rays(values: np.ndarray) -> np.ndarray:
        tangent_u = np.tan(np.radians(values[:, 0]))
        tangent_v = np.tan(np.radians(values[:, 1]))
        result = np.column_stack([tangent_u, tangent_v, np.ones(len(values))])
        return result / np.linalg.norm(result, axis=1, keepdims=True)

    dots = np.sum(rays(predicted) * rays(actual), axis=1)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def make_common_feature(
    times: np.ndarray, u: np.ndarray, v: np.ndarray, horizon_s: float
) -> np.ndarray:
    """Decision-215 feature contract: causal timestamps and observed u/v only."""
    relative_time = times - times[-1]
    return np.concatenate(
        [relative_time, u - u[-1], v - v[-1], np.asarray([u[-1], v[-1], horizon_s])]
    )


def cv_forecast(
    times: np.ndarray, u: np.ndarray, v: np.ndarray, horizon_s: float
) -> np.ndarray:
    return np.asarray(
        [
            polynomial_prediction(times, u, horizon_s),
            polynomial_prediction(times, v, horizon_s),
        ]
    )


def kalman_cv_forecast(
    times: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    horizon_s: float,
    process_scale: float,
    measurement_variance: float,
) -> np.ndarray:
    def one(values: np.ndarray) -> float:
        state = np.asarray([values[0], 0.0], dtype=float)
        covariance = np.diag([max(measurement_variance, 1e-9), 100.0])
        observation = np.asarray([[1.0, 0.0]])
        for index in range(1, len(values)):
            dt = max(float(times[index] - times[index - 1]), 1e-5)
            transition = np.asarray([[1.0, dt], [0.0, 1.0]])
            process = process_scale * np.asarray(
                [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]]
            )
            state = transition.dot(state)
            covariance = transition.dot(covariance).dot(transition.T) + process
            innovation = values[index] - float(observation.dot(state))
            innovation_variance = (
                float(observation.dot(covariance).dot(observation.T))
                + measurement_variance
            )
            gain = covariance.dot(observation.T)[:, 0] / max(
                innovation_variance, 1e-12
            )
            state = state + gain * innovation
            covariance = (np.eye(2) - np.outer(gain, observation[0])).dot(covariance)
        return float(state[0] + horizon_s * state[1])

    return np.asarray([one(u), one(v)])


@dataclass(frozen=True)
class ProcessorConfig:
    history_size: int = 16
    max_history_span_s: float = 0.75
    max_consecutive_gap_s: float = 0.12
    kalman_min_samples: int = 3
    kalman_process_scale: float = 0.01
    kalman_measurement_variance: float = 0.1
    enable_kalman_fallback: bool = False
    ridge_alpha: float = 10.0
    max_train_examples_per_horizon: int = 20000
    max_calibration_examples_per_horizon: int = 4000


@dataclass
class Prediction:
    u_deg: float
    v_deg: float
    method: str
    confidence: float
    uncertainty_p90_deg: float
    history_samples: int
    history_span_s: float
    max_gap_s: float
    detector_coverage: float
    residual_correction_deg: float
    residual_scale_p90_deg: float
    confidence_span: float
    confidence_gap: float
    confidence_coverage: float
    confidence_residual: float


class UvRidgeModel:
    """JSON-serializable per-horizon Ridge residual model."""

    def __init__(self, config: ProcessorConfig, horizons: dict[str, dict[str, Any]], metadata: dict[str, Any]):
        self.config = config
        self.horizons = horizons
        self.metadata = metadata

    @staticmethod
    def _key(horizon_s: float) -> str:
        return f"{float(horizon_s):.6f}"

    @classmethod
    def fit(
        cls,
        examples: Iterable[dict[str, Any]],
        config: ProcessorConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "UvRidgeModel":
        config = config or ProcessorConfig()
        grouped: dict[str, list[dict[str, Any]]] = {}
        all_examples = list(examples)
        for item in all_examples:
            grouped.setdefault(cls._key(float(item["horizon_s"])), []).append(item)
        models: dict[str, dict[str, Any]] = {}
        for key, items in sorted(grouped.items()):
            ordered = sorted(items, key=lambda item: str(item.get("example_id", "")))
            if len(ordered) > config.max_train_examples_per_horizon:
                indices = np.linspace(
                    0, len(ordered) - 1, config.max_train_examples_per_horizon
                ).astype(int)
                ordered = [ordered[index] for index in indices]
            x = np.vstack([np.asarray(item["feature_uv"], dtype=float) for item in ordered])
            y = np.vstack([np.asarray(item["residual_to_cv"], dtype=float) for item in ordered])
            median = np.nanmedian(x, axis=0)
            x = np.where(np.isfinite(x), x, median)
            scaler = StandardScaler()
            scaled = scaler.fit_transform(x)
            ridge = Ridge(alpha=config.ridge_alpha)
            ridge.fit(scaled, y)
            calibration = ordered
            if len(calibration) > config.max_calibration_examples_per_horizon:
                indices = np.linspace(
                    0,
                    len(calibration) - 1,
                    config.max_calibration_examples_per_horizon,
                ).astype(int)
                calibration = [calibration[index] for index in indices]
            calibration_x = np.vstack(
                [np.asarray(item["feature_uv"], dtype=float) for item in calibration]
            )
            calibration_x = np.where(np.isfinite(calibration_x), calibration_x, median)
            correction = ridge.predict(scaler.transform(calibration_x))
            ridge_prediction = np.vstack(
                [np.asarray(item["cv"], dtype=float) for item in calibration]
            ) + correction
            actual = np.vstack(
                [np.asarray(item["actual"], dtype=float) for item in calibration]
            )
            ridge_error = angular_error_deg(ridge_prediction, actual)
            hold_prediction = np.vstack(
                [np.asarray(item["hold"], dtype=float) for item in calibration]
            )
            hold_error = angular_error_deg(hold_prediction, actual)
            kalman_prediction = np.vstack(
                [
                    kalman_cv_forecast(
                        np.asarray(item["times"], dtype=float),
                        np.asarray(item["history_u"], dtype=float),
                        np.asarray(item["history_v"], dtype=float),
                        float(item["horizon_s"]),
                        config.kalman_process_scale,
                        config.kalman_measurement_variance,
                    )
                    for item in calibration
                ]
            )
            kalman_error = angular_error_deg(kalman_prediction, actual)
            spans = np.asarray(
                [float(item["times"][-1] - item["times"][0]) for item in ordered]
            )
            gaps = np.asarray(
                [float(np.max(np.diff(item["times"]))) for item in ordered]
            )
            intervals = np.concatenate(
                [np.diff(np.asarray(item["times"], dtype=float)) for item in ordered]
            )
            example_ids = "\n".join(str(item.get("example_id", "")) for item in ordered)
            models[key] = {
                "horizon_s": float(key),
                "feature_size": int(x.shape[1]),
                "feature_median": median.tolist(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "coef": ridge.coef_.tolist(),
                "intercept": ridge.intercept_.tolist(),
                "training_examples": len(ordered),
                "calibration_examples": len(calibration),
                "training_example_hash": hashlib.sha256(example_ids.encode("utf-8")).hexdigest(),
                "median_interval_s": float(np.median(intervals)),
                "history_span_p50_s": float(np.percentile(spans, 50)),
                "history_gap_p90_s": float(np.percentile(gaps, 90)),
                "ridge_error_p50_deg": float(np.percentile(ridge_error, 50)),
                "ridge_error_p90_deg": float(np.percentile(ridge_error, 90)),
                "ridge_error_p95_deg": float(np.percentile(ridge_error, 95)),
                "kalman_error_p90_deg": float(np.percentile(kalman_error, 90)),
                "hold_error_p90_deg": float(np.percentile(hold_error, 90)),
            }
        if not models:
            raise ValueError("cannot fit an empty trajectory model")
        return cls(config, models, dict(metadata or {}))

    def predict_residual(self, feature: np.ndarray, horizon_s: float) -> tuple[np.ndarray, dict[str, Any]]:
        key = self._key(horizon_s)
        if key not in self.horizons:
            raise KeyError(f"unsupported horizon: {horizon_s}")
        model = self.horizons[key]
        values = np.asarray(feature, dtype=float)
        if values.shape != (int(model["feature_size"]),):
            raise ValueError(
                f"feature shape {values.shape} does not match {(int(model['feature_size']),)}"
            )
        median = np.asarray(model["feature_median"], dtype=float)
        values = np.where(np.isfinite(values), values, median)
        scaled = (values - np.asarray(model["scaler_mean"], dtype=float)) / np.asarray(
            model["scaler_scale"], dtype=float
        )
        residual = np.asarray(model["coef"], dtype=float).dot(scaled) + np.asarray(
            model["intercept"], dtype=float
        )
        return residual, model

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "causal_uv_cv_ridge_residual",
            "runtime_input_contract": ["timestamp_s", "u_deg", "v_deg"],
            "truth_is_runtime_input": False,
            "config": asdict(self.config),
            "horizons": self.horizons,
            "metadata": self.metadata,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "UvRidgeModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported model schema: {payload.get('schema_version')}")
        if payload.get("truth_is_runtime_input") is not False:
            raise ValueError("runtime model must explicitly exclude truth")
        return cls(
            ProcessorConfig(**payload["config"]),
            dict(payload["horizons"]),
            dict(payload.get("metadata", {})),
        )


class CausalUvRidgeProcessor:
    """Stateful per-track processor with Ridge, Kalman and hold modes."""

    def __init__(self, model: UvRidgeModel):
        self.model = model
        self.config = model.config
        self.history: list[tuple[float, float, float]] = []
        self.last_preceding_gap_s = float("nan")

    def reset(self) -> None:
        self.history.clear()
        self.last_preceding_gap_s = float("nan")

    def update(self, timestamp_s: float, u_deg: float, v_deg: float) -> float:
        timestamp_s = _finite(timestamp_s)
        u_deg = _finite(u_deg)
        v_deg = _finite(v_deg)
        if self.history and timestamp_s <= self.history[-1][0]:
            raise ValueError("processor timestamps must be strictly increasing")
        preceding_gap = (
            float("nan") if not self.history else timestamp_s - self.history[-1][0]
        )
        if self.history:
            previous_u = self.history[-1][1]
            delta = (u_deg - previous_u + 180.0) % 360.0 - 180.0
            u_deg = previous_u + delta
        self.history.append((timestamp_s, u_deg, v_deg))
        keep = max(self.config.history_size * 4, self.config.history_size + 1)
        self.history = self.history[-keep:]
        self.last_preceding_gap_s = preceding_gap
        return preceding_gap

    def _continuous_tail(self) -> list[tuple[float, float, float]]:
        if not self.history:
            return []
        start = 0
        for index in range(1, len(self.history)):
            if self.history[index][0] - self.history[index - 1][0] > self.config.max_consecutive_gap_s:
                start = index
        tail = self.history[start:]
        while len(tail) > 1 and tail[-1][0] - tail[0][0] > self.config.max_history_span_s:
            tail = tail[1:]
        return tail[-self.config.history_size :]

    def predict(self, horizon_s: float) -> Prediction:
        tail = self._continuous_tail()
        if not tail:
            raise RuntimeError("prediction requires at least one observation")
        times = np.asarray([row[0] for row in tail], dtype=float)
        u = np.asarray([row[1] for row in tail], dtype=float)
        v = np.asarray([row[2] for row in tail], dtype=float)
        span = float(times[-1] - times[0]) if len(times) > 1 else 0.0
        max_gap = float(np.max(np.diff(times))) if len(times) > 1 else float("inf")
        key = self.model._key(horizon_s)
        horizon_model = self.model.horizons.get(key)
        residual = np.zeros(2, dtype=float)
        if len(tail) >= self.config.history_size and horizon_model is not None:
            selected_times = times[-self.config.history_size :]
            selected_u = u[-self.config.history_size :]
            selected_v = v[-self.config.history_size :]
            base = cv_forecast(selected_times, selected_u, selected_v, horizon_s)
            feature = make_common_feature(selected_times, selected_u, selected_v, horizon_s)
            residual, horizon_model = self.model.predict_residual(feature, horizon_s)
            predicted = base + residual
            method = "ridge_uv_residual"
            times, u, v = selected_times, selected_u, selected_v
            span = float(times[-1] - times[0])
            max_gap = float(np.max(np.diff(times)))
            residual_scale = float(horizon_model["ridge_error_p90_deg"])
            method_prior = 1.0
        elif (
            self.config.enable_kalman_fallback
            and len(tail) >= self.config.kalman_min_samples
        ):
            predicted = kalman_cv_forecast(
                times,
                u,
                v,
                horizon_s,
                self.config.kalman_process_scale,
                self.config.kalman_measurement_variance,
            )
            method = "kalman_fallback"
            residual_scale = float(
                horizon_model["kalman_error_p90_deg"] if horizon_model else 1.0
            )
            method_prior = 0.55
        else:
            predicted = np.asarray([u[-1], v[-1]])
            method = "hold_fallback"
            residual_scale = float(
                horizon_model["hold_error_p90_deg"] if horizon_model else 2.0
            )
            method_prior = 0.2
        if horizon_model:
            median_interval = max(float(horizon_model["median_interval_s"]), 1e-6)
            target_span = max(float(horizon_model["history_span_p50_s"]), median_interval)
            target_gap = max(float(horizon_model["history_gap_p90_s"]), median_interval)
        else:
            median_interval = 1.0 / 120.0
            target_span = median_interval * max(self.config.history_size - 1, 1)
            target_gap = median_interval * 2.0
        expected_intervals = max(span / median_interval, 1.0)
        coverage = min(1.0, max(len(times) - 1, 0) / expected_intervals)
        span_score = min(1.0, span / target_span) if target_span > 0.0 else 0.0
        gap_score = 0.0 if not math.isfinite(max_gap) else math.exp(-max_gap / target_gap)
        coverage_score = math.sqrt(max(coverage, 0.0))
        residual_magnitude = float(np.linalg.norm(residual))
        residual_score = math.exp(-residual_magnitude / max(3.0 * residual_scale, 1e-6))
        confidence = float(
            np.clip(
                method_prior
                * (0.25 + 0.75 * span_score)
                * (0.25 + 0.75 * gap_score)
                * (0.25 + 0.75 * coverage_score)
                * (0.4 + 0.6 * residual_score),
                0.0,
                1.0,
            )
        )
        uncertainty = residual_scale / max(confidence, 0.05)
        return Prediction(
            u_deg=float(predicted[0]),
            v_deg=float(predicted[1]),
            method=method,
            confidence=confidence,
            uncertainty_p90_deg=float(uncertainty),
            history_samples=len(times),
            history_span_s=span,
            max_gap_s=max_gap,
            detector_coverage=coverage,
            residual_correction_deg=residual_magnitude,
            residual_scale_p90_deg=residual_scale,
            confidence_span=span_score,
            confidence_gap=gap_score,
            confidence_coverage=coverage_score,
            confidence_residual=residual_score,
        )
