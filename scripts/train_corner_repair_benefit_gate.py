#!/usr/bin/env python3
"""Train a truth-free secondary gate for frozen corner-repair proposals.

Truth is joined only after all runtime-observable features have been built.  The
gate is deliberately reject-only: it can veto a proposal accepted by the frozen
v3 repairer, but it can never activate a proposal that v3 rejected.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import cv2
from scipy.optimize import linear_sum_assignment
from torch import nn

from replay_linux_corner_repair_pnp_candidates import (
    CORNER_ORDER,
    LARGE_POINTS_MM,
    SMALL_POINTS_MM,
    calibration,
    solve_ippe_candidates,
)
from score_linux_repair_observer_validation import (
    error_components,
    exact_camera_tvec,
    key,
    label_matrix,
    read_jsonl,
    sha256,
)
from training.stage3.corner_residual_network import observable_features


SCHEMA = "aim-stack.corner-repair-benefit-gate/1"
FEATURE_NAMES = (
    *(f"raw_geometry_{index}" for index in range(15)),
    *(f"proposal_delta_over_quad_scale_{index}" for index in range(8)),
    "detector_confidence",
    "v3_reliability_probability",
    "proposal_correction_rms_px",
    "proposal_correction_max_px",
    "frame_candidate_count",
    "detector_type_large",
    "raw_pnp_valid",
    "raw_ray_u_rad",
    "raw_ray_v_rad",
    "raw_log_range_m",
    "raw_yaw_sin",
    "raw_yaw_cos",
    "raw_reprojection_rms_px",
    "raw_reprojection_max_px",
    "raw_pnp_solution_count",
    "raw_ippe_reprojection_gap_px",
    "raw_ippe_tvec_separation_m",
    "proposal_pnp_valid",
    "proposal_ray_shift_rad",
    "proposal_radial_shift_m",
    "proposal_transverse_shift_m",
    "proposal_log_range_ratio",
    "proposal_yaw_jump_rad",
    "proposal_reprojection_rms_px",
    "proposal_reprojection_delta_px",
    "proposal_pnp_solution_count",
    "proposal_ippe_reprojection_gap_px",
    "proposal_ippe_tvec_separation_m",
    "proposal_solver_branch_changed",
    "proposal_width_ratio",
    "proposal_height_ratio",
    "proposal_area_ratio",
    "proposal_aspect_ratio_ratio",
    "proposal_common_translation_over_scale",
    "proposal_shape_change_over_scale",
    "past_match_valid",
    "past_frame_dt_s",
    "past_ray_u_velocity_rad_s",
    "past_ray_v_velocity_rad_s",
    "past_ray_speed_rad_s",
    "past_frame_candidate_count",
    "raw_history_continuity_residual_rad",
    "proposal_history_continuity_residual_rad",
    "proposal_minus_raw_continuity_residual_rad",
)


def corners(candidate: dict[str, Any], field: str) -> np.ndarray:
    values = candidate["repair"][field]
    result = np.asarray([values[name] for name in CORNER_ORDER], dtype=np.float64)
    if result.shape != (4, 2) or not np.isfinite(result).all():
        raise ValueError(f"invalid {field}")
    return result


def selected(group: dict[str, Any]) -> dict[str, Any] | None:
    return next((item for item in group["candidates"] if item.get("selected")), None)


def ray_angles(tvec: np.ndarray) -> tuple[float, float]:
    x, y, z = (float(value) for value in tvec)
    return math.atan2(x, z), math.atan2(y, math.hypot(x, z))


def wrap_radians(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def runtime_feature_rows(
    frames: list[dict[str, Any]], matrix: np.ndarray, distortion: np.ndarray
) -> dict[tuple[int, int, int, int], np.ndarray]:
    """Build features without opening labels or session planning metadata."""
    output: dict[tuple[int, int, int, int], np.ndarray] = {}
    previous_timestamp_ns: int | None = None
    previous_tracks: list[tuple[float, float, float, float]] = []
    for frame in frames:
        timestamp_ns = int(frame["timestamp_ns"])
        dt = (
            None
            if previous_timestamp_ns is None
            else (timestamp_ns - previous_timestamp_ns) / 1.0e9
        )
        current_tracks: list[tuple[float, float, float, float]] = []
        for candidate in frame["candidates"]:
            raw = corners(candidate, "raw_corners_px")
            proposed = corners(candidate, "model_proposed_corners_px")
            delta = proposed - raw
            raw_area = max(abs(float(cv2.contourArea(raw.astype(np.float32)))), 1.0)
            proposed_area = max(abs(float(cv2.contourArea(proposed.astype(np.float32)))), 1.0)
            quad_scale = math.sqrt(raw_area)
            raw_left = 0.5 * (raw[0] + raw[1])
            raw_right = 0.5 * (raw[2] + raw[3])
            proposed_left = 0.5 * (proposed[0] + proposed[1])
            proposed_right = 0.5 * (proposed[2] + proposed[3])
            raw_width = max(float(np.linalg.norm(raw_right - raw_left)), 1.0e-6)
            proposed_width = max(float(np.linalg.norm(proposed_right - proposed_left)), 1.0e-6)
            raw_height = max(
                0.5 * (
                    float(np.linalg.norm(raw[0] - raw[1]))
                    + float(np.linalg.norm(raw[3] - raw[2]))
                ),
                1.0e-6,
            )
            proposed_height = max(
                0.5 * (
                    float(np.linalg.norm(proposed[0] - proposed[1]))
                    + float(np.linalg.norm(proposed[3] - proposed[2]))
                ),
                1.0e-6,
            )
            raw_solution = selected(candidate["raw_pnp"])
            raw_valid = raw_solution is not None
            if raw_solution is None:
                ray_u = ray_v = log_range = yaw_sin = yaw_cos = 0.0
                reprojection_rms = reprojection_max = 0.0
            else:
                tvec = np.asarray(raw_solution["tvec_m"], dtype=np.float64)
                ray_u, ray_v = ray_angles(tvec)
                log_range = math.log(max(float(np.linalg.norm(tvec)), 1.0e-3))
                yaw = float(raw_solution["observed_yaw_rad"])
                yaw_sin, yaw_cos = math.sin(yaw), math.cos(yaw)
                reprojection_rms = float(raw_solution["reprojection_rms_px"])
                reprojection_max = float(raw_solution["reprojection_max_px"])
            solutions = candidate["raw_pnp"]["candidates"]
            if len(solutions) >= 2:
                first, second = solutions[0], solutions[1]
                reprojection_gap = max(
                    float(second["reprojection_rms_px"]) - float(first["reprojection_rms_px"]),
                    0.0,
                )
                tvec_separation = float(
                    np.linalg.norm(
                        np.asarray(second["tvec_m"], dtype=np.float64)
                        - np.asarray(first["tvec_m"], dtype=np.float64)
                    )
                )
            else:
                reprojection_gap = tvec_separation = 0.0

            object_points = (
                LARGE_POINTS_MM if candidate["detector_type"] == "large" else SMALL_POINTS_MM
            )
            proposal_solutions, _ = solve_ippe_candidates(
                proposed.astype(np.float32), matrix, distortion, object_points
            )
            proposal_solution = next(
                (item for item in proposal_solutions if item["selected"]), None
            )
            proposal_valid = proposal_solution is not None
            proposal_u = proposal_v = 0.0
            proposal_ray_shift = radial_shift = transverse_shift = log_range_ratio = 0.0
            yaw_jump = proposal_reprojection = proposal_reprojection_delta = 0.0
            solver_branch_changed = 0.0
            if raw_solution is not None and proposal_solution is not None:
                raw_tvec = np.asarray(raw_solution["tvec_m"], dtype=np.float64)
                proposal_tvec_value = np.asarray(proposal_solution["tvec_m"], dtype=np.float64)
                proposal_u, proposal_v = ray_angles(proposal_tvec_value)
                proposal_ray_shift = math.hypot(
                    wrap_radians(proposal_u - ray_u), proposal_v - ray_v
                )
                displacement = proposal_tvec_value - raw_tvec
                raw_los = raw_tvec / max(float(np.linalg.norm(raw_tvec)), 1.0e-9)
                radial_shift = float(np.dot(displacement, raw_los))
                transverse_shift = float(
                    np.linalg.norm(displacement - radial_shift * raw_los)
                )
                log_range_ratio = math.log(
                    max(float(np.linalg.norm(proposal_tvec_value)), 1.0e-6)
                    / max(float(np.linalg.norm(raw_tvec)), 1.0e-6)
                )
                yaw_jump = wrap_radians(
                    float(proposal_solution["observed_yaw_rad"])
                    - float(raw_solution["observed_yaw_rad"])
                )
                proposal_reprojection = float(proposal_solution["reprojection_rms_px"])
                proposal_reprojection_delta = proposal_reprojection - reprojection_rms
                solver_branch_changed = float(
                    int(proposal_solution["solver_solution_index"])
                    != int(raw_solution["solver_solution_index"])
                )
            if len(proposal_solutions) >= 2:
                proposal_reprojection_gap = max(
                    float(proposal_solutions[1]["reprojection_rms_px"])
                    - float(proposal_solutions[0]["reprojection_rms_px"]),
                    0.0,
                )
                proposal_tvec_separation = float(
                    np.linalg.norm(
                        np.asarray(proposal_solutions[1]["tvec_m"], dtype=np.float64)
                        - np.asarray(proposal_solutions[0]["tvec_m"], dtype=np.float64)
                    )
                )
            else:
                proposal_reprojection_gap = proposal_tvec_separation = 0.0

            past_valid = False
            past_dt = du_dt = dv_dt = speed = 0.0
            raw_continuity = proposal_continuity = continuity_delta = 0.0
            if (
                raw_valid
                and dt is not None
                and 0.0 < dt <= 0.2
                and previous_tracks
            ):
                finite_previous = [item for item in previous_tracks if np.isfinite(item).all()]
                if finite_previous:
                    distances = [
                        math.hypot(
                            wrap_radians(ray_u - (old_u + old_du * dt)),
                            ray_v - (old_v + old_dv * dt),
                        )
                        for old_u, old_v, old_du, old_dv in finite_previous
                    ]
                    ordered_distances = sorted((value, index) for index, value in enumerate(distances))
                    nearest_distance, nearest_index = ordered_distances[0]
                    assignment_margin_ok = (
                        len(ordered_distances) == 1
                        or ordered_distances[1][0] - nearest_distance > math.radians(0.25)
                    )
                    if nearest_distance <= math.radians(5.0) and assignment_margin_ok:
                        old_u, old_v, old_du, old_dv = finite_previous[nearest_index]
                        past_valid = True
                        past_dt = dt
                        du_dt = float(np.clip(wrap_radians(ray_u - old_u) / dt, -10.0, 10.0))
                        dv_dt = float(np.clip((ray_v - old_v) / dt, -10.0, 10.0))
                        speed = math.hypot(du_dt, dv_dt)
                        predicted_u = old_u + old_du * dt
                        predicted_v = old_v + old_dv * dt
                        raw_continuity = math.hypot(
                            wrap_radians(ray_u - predicted_u), ray_v - predicted_v
                        )
                        if proposal_valid:
                            proposal_continuity = math.hypot(
                                wrap_radians(proposal_u - predicted_u),
                                proposal_v - predicted_v,
                            )
                            continuity_delta = proposal_continuity - raw_continuity

            current_tracks.append(
                (
                    ray_u if raw_valid else math.nan,
                    ray_v if raw_valid else math.nan,
                    du_dt if past_valid else 0.0,
                    dv_dt if past_valid else 0.0,
                )
            )

            count = len(frame["candidates"])
            repair = candidate["repair"]
            scalars = np.asarray(
                [
                    float(candidate["detector_confidence"]),
                    float(repair["reliability_probability"]),
                    float(repair["predicted_correction_rms_px"]),
                    float(repair["predicted_correction_max_px"]),
                    float(count),
                    float(candidate["detector_type"] == "large"),
                    float(raw_valid),
                    ray_u,
                    ray_v,
                    log_range,
                    yaw_sin,
                    yaw_cos,
                    reprojection_rms,
                    reprojection_max,
                    float(len(solutions)),
                    reprojection_gap,
                    tvec_separation,
                    float(proposal_valid),
                    proposal_ray_shift,
                    radial_shift,
                    transverse_shift,
                    log_range_ratio,
                    yaw_jump,
                    proposal_reprojection,
                    proposal_reprojection_delta,
                    float(len(proposal_solutions)),
                    proposal_reprojection_gap,
                    proposal_tvec_separation,
                    solver_branch_changed,
                    proposed_width / raw_width,
                    proposed_height / raw_height,
                    proposed_area / raw_area,
                    (proposed_width / proposed_height) / (raw_width / raw_height),
                    float(np.linalg.norm(delta.mean(axis=0)) / quad_scale),
                    float(np.sqrt(np.mean(np.square(delta - delta.mean(axis=0)))) / quad_scale),
                    float(past_valid),
                    past_dt,
                    du_dt,
                    dv_dt,
                    speed,
                    float(len(previous_tracks)),
                    raw_continuity,
                    proposal_continuity,
                    continuity_delta,
                ],
                dtype=np.float32,
            )
            values = np.concatenate(
                (
                    observable_features(raw).astype(np.float32),
                    (delta / quad_scale).reshape(-1).astype(np.float32),
                    scalars,
                )
            )
            if len(values) != len(FEATURE_NAMES) or not np.isfinite(values).all():
                raise ValueError("benefit-gate runtime feature contract failed")
            identity = (*key(frame), int(candidate["observation_id"]))
            if identity in output:
                raise ValueError(f"duplicate candidate identity: {identity}")
            output[identity] = values
        previous_timestamp_ns = timestamp_ns
        previous_tracks = current_tracks
    return output


def proposal_tvec(
    candidate: dict[str, Any], matrix: np.ndarray, distortion: np.ndarray
) -> np.ndarray | None:
    object_points = (
        LARGE_POINTS_MM if candidate["detector_type"] == "large" else SMALL_POINTS_MM
    )
    proposals, _ = solve_ippe_candidates(
        corners(candidate, "model_proposed_corners_px").astype(np.float32),
        matrix,
        distortion,
        object_points,
    )
    item = next((value for value in proposals if value["selected"]), None)
    return None if item is None else np.asarray(item["tvec_m"], dtype=np.float64)


@dataclass
class Sample:
    session: str
    split: str
    mode: str
    identity: tuple[int, int, int, int]
    features: np.ndarray
    old_applied: bool
    outcome: str
    raw: dict[str, float]
    proposed: dict[str, float]

    @property
    def benefit(self) -> bool:
        return self.outcome == "BENEFIT"


def load_session_samples(session: Path, match_gate_px: float) -> list[Sample]:
    result = json.loads((session / "session-result.json").read_text(encoding="utf-8"))
    pnp_path = session / "repair-pnp-complete-candidates-v1.jsonl"
    frames = read_jsonl(pnp_path)
    pnp_manifest = json.loads(
        (session / "repair-pnp-complete-candidates-v1-manifest.json").read_text(encoding="utf-8")
    )
    calibration_path = Path(pnp_manifest["camera_calibration"]).resolve(strict=True)
    if pnp_manifest["camera_calibration_sha256"] != sha256(calibration_path):
        raise ValueError("camera calibration hash mismatch")
    matrix, distortion, _ = calibration(calibration_path)
    features = runtime_feature_rows(frames, matrix, distortion)
    labels: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for label in read_jsonl(session / "exact-corners.jsonl"):
        if int(label["frame_seq"]) >= int(result["first_eligible_frame_seq"]):
            labels[key(label)].append(label)
    output: list[Sample] = []
    for frame in frames:
        frame_labels = labels.get(key(frame), [])
        candidates = frame["candidates"]
        if not frame_labels or not candidates:
            continue
        costs = np.zeros((len(candidates), len(frame_labels)), dtype=np.float64)
        for candidate_index, candidate in enumerate(candidates):
            raw = corners(candidate, "raw_corners_px")
            for label_index, label in enumerate(frame_labels):
                exact = np.asarray(label["exact_corners_px"], dtype=np.float64)
                costs[candidate_index, label_index] = float(np.sqrt(np.mean(np.square(raw - exact))))
        rows, columns = linear_sum_assignment(costs)
        for candidate_index, label_index in zip(rows, columns):
            if costs[candidate_index, label_index] > match_gate_px:
                continue
            candidate = candidates[int(candidate_index)]
            raw_item = selected(candidate["raw_pnp"])
            if raw_item is None:
                continue
            label = frame_labels[int(label_index)]
            truth = exact_camera_tvec(label)
            raw_error = error_components(np.asarray(raw_item["tvec_m"]), truth)
            matrix, distortion = label_matrix(label)
            proposed_point = proposal_tvec(candidate, matrix, distortion)
            if proposed_point is None:
                proposed_error = {name: math.inf for name in raw_error}
            else:
                proposed_error = error_components(proposed_point, truth)
            transverse_delta = (
                raw_error["transverse_error_mm"] - proposed_error["transverse_error_mm"]
            )
            safe_angle = (
                proposed_error["angular_error_deg"]
                <= raw_error["angular_error_deg"] + 0.02
            )
            safe_radial = (
                proposed_error["radial_error_abs_mm"]
                <= raw_error["radial_error_abs_mm"] * 1.02
            )
            if transverse_delta >= 0.5 and safe_angle and safe_radial:
                outcome = "BENEFIT"
            elif transverse_delta <= -0.5 or not safe_angle or not safe_radial:
                outcome = "HARM"
            else:
                outcome = "UNCERTAIN"
            identity = (*key(frame), int(candidate["observation_id"]))
            output.append(
                Sample(
                    session=session.name,
                    split=str(result["planned"]["split"]),
                    mode=str(result["planned"]["mode"]),
                    identity=identity,
                    features=features[identity],
                    old_applied=bool(candidate["repair"]["applied"]),
                    outcome=outcome,
                    raw=raw_error,
                    proposed=proposed_error,
                )
            )
    return output


class BenefitGate(nn.Module):
    def __init__(self, dimensions: int, architecture: str) -> None:
        super().__init__()
        if architecture == "linear":
            self.network = nn.Linear(dimensions, 1)
        elif architecture == "mlp":
            self.network = nn.Sequential(
                nn.Linear(dimensions, 32),
                nn.SiLU(),
                nn.Linear(32, 16),
                nn.SiLU(),
                nn.Linear(16, 1),
            )
        else:
            raise ValueError(f"unknown architecture: {architecture}")

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).reshape(-1)


def deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def fit_model(
    samples: list[Sample], architecture: str, seed: int, epochs: int
) -> tuple[BenefitGate, np.ndarray, np.ndarray]:
    eligible = [
        sample
        for sample in samples
        if sample.old_applied and sample.outcome != "UNCERTAIN"
    ]
    if not eligible or len({sample.benefit for sample in eligible}) < 2:
        raise ValueError("training fold lacks both benefit classes")
    x = np.asarray([sample.features for sample in eligible], dtype=np.float32)
    y = np.asarray([sample.benefit for sample in eligible], dtype=np.float32)
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1.0e-6)
    normalized = (x - mean) / std
    session_counts: dict[str, int] = defaultdict(int)
    for sample in eligible:
        session_counts[sample.session] += 1
    weights = np.asarray([1.0 / session_counts[sample.session] for sample in eligible], dtype=np.float32)
    weights *= len(weights) / weights.sum()
    positives = max(float(y.sum()), 1.0)
    negatives = max(float(len(y) - y.sum()), 1.0)
    class_weights = np.where(y > 0.5, negatives / positives, 1.0).astype(np.float32)
    weights *= class_weights
    deterministic_seed(seed)
    model = BenefitGate(x.shape[1], architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    tensor_x = torch.from_numpy(normalized.astype(np.float32))
    tensor_y = torch.from_numpy(y)
    tensor_w = torch.from_numpy(weights)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss_values = nn.functional.binary_cross_entropy_with_logits(
            model(tensor_x), tensor_y, reduction="none"
        )
        loss = torch.sum(loss_values * tensor_w) / torch.sum(tensor_w)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss - 1.0e-7:
            best_loss = value
            best_state = {name: item.detach().clone() for name, item in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 40:
            break
    if best_state is None:
        raise AssertionError("model training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, mean.astype(np.float32), std.astype(np.float32)


def predict(model: BenefitGate, mean: np.ndarray, std: np.ndarray, samples: list[Sample]) -> np.ndarray:
    x = np.asarray([sample.features for sample in samples], dtype=np.float32)
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy((x - mean) / std))).numpy()


def percentile(values: Iterable[float], q: float) -> float | None:
    values = list(values)
    return None if not values else float(np.percentile(np.asarray(values), q))


def gate_metrics(samples: list[Sample], probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    by_session: dict[str, list[tuple[Sample, bool]]] = defaultdict(list)
    for sample, probability in zip(samples, probabilities):
        accept = bool(sample.old_applied and probability >= threshold)
        by_session[sample.session].append((sample, accept))
    sessions = []
    for name, rows in sorted(by_session.items()):
        raw_angular = [sample.raw["angular_error_deg"] for sample, _ in rows]
        raw_radial = [sample.raw["radial_error_abs_mm"] for sample, _ in rows]
        raw_transverse = [sample.raw["transverse_error_mm"] for sample, _ in rows]
        gated_angular = [
            (sample.proposed if accept else sample.raw)["angular_error_deg"]
            for sample, accept in rows
        ]
        gated_radial = [
            (sample.proposed if accept else sample.raw)["radial_error_abs_mm"]
            for sample, accept in rows
        ]
        gated_transverse = [
            (sample.proposed if accept else sample.raw)["transverse_error_mm"]
            for sample, accept in rows
        ]
        raw_a95, gated_a95 = percentile(raw_angular, 95), percentile(gated_angular, 95)
        raw_r95, gated_r95 = percentile(raw_radial, 95), percentile(gated_radial, 95)
        raw_t95, gated_t95 = percentile(raw_transverse, 95), percentile(gated_transverse, 95)
        sessions.append(
            {
                "session": name,
                "mode": rows[0][0].mode,
                "samples": len(rows),
                "old_applied": sum(sample.old_applied for sample, _ in rows),
                "gate_applied": sum(accept for _, accept in rows),
                "gate_precision": (
                    sum(sample.benefit and accept for sample, accept in rows)
                    / max(sum(accept for _, accept in rows), 1)
                ),
                "raw_angular_p95_deg": raw_a95,
                "gated_angular_p95_deg": gated_a95,
                "raw_radial_p95_mm": raw_r95,
                "gated_radial_p95_mm": gated_r95,
                "raw_transverse_p95_mm": raw_t95,
                "gated_transverse_p95_mm": gated_t95,
                "angular_noninferior": gated_a95 <= raw_a95 + 0.02,
                "radial_noninferior": gated_r95 <= raw_r95 * 1.02,
                "transverse_noninferior": gated_t95 <= raw_t95,
                "transverse_improvement_fraction": (raw_t95 - gated_t95) / max(raw_t95, 1.0e-9),
            }
        )
    feasible = all(
        row["angular_noninferior"] and row["radial_noninferior"] and row["transverse_noninferior"]
        for row in sessions
    )
    applied_rows = [
        (sample, bool(sample.old_applied and probability >= threshold))
        for sample, probability in zip(samples, probabilities)
    ]
    applied = sum(accept for _, accept in applied_rows)
    benefit_applied = sum(sample.outcome == "BENEFIT" and accept for sample, accept in applied_rows)
    harm_applied = sum(sample.outcome == "HARM" and accept for sample, accept in applied_rows)
    uncertain_applied = sum(sample.outcome == "UNCERTAIN" and accept for sample, accept in applied_rows)
    macro_improvement = float(
        np.mean([row["transverse_improvement_fraction"] for row in sessions])
    )
    result = {
        "threshold": threshold,
        "sessions": sessions,
        "feasible": feasible,
        "session_macro_transverse_improvement_fraction": macro_improvement,
        "applied": applied,
        "benefit_applied": benefit_applied,
        "harm_applied": harm_applied,
        "uncertain_applied": uncertain_applied,
        "benefit_precision": benefit_applied / max(applied, 1),
        "harm_apply_fraction": harm_applied / max(applied, 1),
    }
    result["deployment_candidate"] = bool(
        feasible
        and applied >= 10
        and result["benefit_precision"] >= 0.8
        and result["harm_apply_fraction"] <= 0.05
        and macro_improvement >= 0.01
    )
    return result


def choose_threshold(samples: list[Sample], probabilities: np.ndarray) -> dict[str, Any]:
    candidates = [gate_metrics(samples, probabilities, float(value)) for value in np.linspace(0.05, 0.99, 95)]
    reject_all = gate_metrics(samples, probabilities, 1.01)
    deployment_candidates = [row for row in candidates if row["deployment_candidate"]]
    if not deployment_candidates:
        return reject_all
    return max(
        deployment_candidates,
        key=lambda row: (
            row["session_macro_transverse_improvement_fraction"],
            row["benefit_precision"],
            -row["applied"],
            row["threshold"],
        ),
    )


def sha256_text(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--match-gate-px", type=float, default=25.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", action="append", type=int, default=[4201, 4202, 4203])
    args = parser.parse_args()
    output = args.output_dir.resolve()
    model_dir = args.model_dir.resolve()
    if output.exists() or model_dir.exists():
        raise FileExistsError("refusing to overwrite benefit-gate evidence or model")
    output.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    sessions: list[Path] = []
    for root in args.collection_root:
        sessions.extend(
            sorted(
                result.parent
                for result in root.resolve(strict=True).glob("*/session-result.json")
                if (result.parent / "repair-pnp-complete-candidates-v1.jsonl").exists()
            )
        )
    all_samples = [sample for session in sessions for sample in load_session_samples(session, args.match_gate_px)]
    development = [sample for sample in all_samples if sample.split == "train"]
    validation = [sample for sample in all_samples if sample.split == "validation"]
    if len({sample.session for sample in development}) < 3:
        raise ValueError("session-disjoint cross-validation requires at least three development sessions")

    comparisons: dict[str, Any] = {
        "reject_all": {
            "development": gate_metrics(development, np.zeros(len(development)), 1.01),
            "validation_diagnostic": gate_metrics(validation, np.zeros(len(validation)), 1.01),
        }
    }
    architecture_artifacts = {}
    for architecture in ("linear", "mlp"):
        oof_seed_probabilities = []
        final_seed_probabilities = []
        final_models = []
        for seed in args.seed:
            oof = np.zeros(len(development), dtype=np.float32)
            for held_session in sorted({sample.session for sample in development}):
                train = [sample for sample in development if sample.session != held_session]
                held_indices = [index for index, sample in enumerate(development) if sample.session == held_session]
                held = [development[index] for index in held_indices]
                model, mean, std = fit_model(train, architecture, seed, args.epochs)
                oof[held_indices] = predict(model, mean, std, held)
            oof_seed_probabilities.append(oof)
            model, mean, std = fit_model(development, architecture, seed, args.epochs)
            final_seed_probabilities.append(predict(model, mean, std, validation))
            final_models.append((seed, model, mean, std))
        oof_ensemble = np.mean(np.asarray(oof_seed_probabilities), axis=0)
        validation_ensemble = np.mean(np.asarray(final_seed_probabilities), axis=0)
        selected = choose_threshold(development, oof_ensemble)
        validation_metrics = gate_metrics(validation, validation_ensemble, float(selected["threshold"]))
        comparisons[architecture] = {
            "development_oof": selected,
            "validation_diagnostic": validation_metrics,
            "seed_count": len(args.seed),
        }
        checkpoints = []
        for seed, model, mean, std in final_models:
            checkpoint_path = model_dir / f"{architecture}-seed{seed}.pt"
            torch.save(
                {
                    "schema_version": SCHEMA,
                    "architecture": architecture,
                    "feature_names": list(FEATURE_NAMES),
                    "feature_mean": mean,
                    "feature_std": std,
                    "state_dict": model.state_dict(),
                    "seed": seed,
                    "threshold": float(selected["threshold"]),
                    "reject_only": True,
                    "truth_runtime_input": False,
                    "motion_mode_runtime_input": False,
                },
                checkpoint_path,
            )
            checkpoints.append(
                {"file": checkpoint_path.name, "sha256": sha256(checkpoint_path), "bytes": checkpoint_path.stat().st_size}
            )
        architecture_artifacts[architecture] = checkpoints

    eligible_architectures = [
        name
        for name in ("linear", "mlp")
        if comparisons[name]["development_oof"]["deployment_candidate"]
    ]
    selected_architecture = (
        max(
            eligible_architectures,
            key=lambda name: comparisons[name]["development_oof"]["session_macro_transverse_improvement_fraction"],
        )
        if eligible_architectures
        else "reject_all"
    )
    rows_path = output / "benefit_gate_samples.csv.gz"
    with gzip.open(rows_path, "xt", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "session", "split", "mode", "producer_epoch", "frame_seq", "timestamp_ns", "observation_id",
            "old_applied", "outcome", "benefit", *FEATURE_NAMES,
            "raw_angular_error_deg", "raw_radial_error_abs_mm", "raw_transverse_error_mm",
            "proposed_angular_error_deg", "proposed_radial_error_abs_mm", "proposed_transverse_error_mm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in all_samples:
            row = {
                "session": sample.session, "split": sample.split, "mode": sample.mode,
                "producer_epoch": sample.identity[0], "frame_seq": sample.identity[1],
                "timestamp_ns": sample.identity[2], "observation_id": sample.identity[3],
                "old_applied": int(sample.old_applied), "outcome": sample.outcome,
                "benefit": int(sample.benefit),
                **{name: float(value) for name, value in zip(FEATURE_NAMES, sample.features)},
                "raw_angular_error_deg": sample.raw["angular_error_deg"],
                "raw_radial_error_abs_mm": sample.raw["radial_error_abs_mm"],
                "raw_transverse_error_mm": sample.raw["transverse_error_mm"],
                "proposed_angular_error_deg": sample.proposed["angular_error_deg"],
                "proposed_radial_error_abs_mm": sample.proposed["radial_error_abs_mm"],
                "proposed_transverse_error_mm": sample.proposed["transverse_error_mm"],
            }
            writer.writerow(row)
    registry = {
        "schema_version": SCHEMA,
        "claim": "method-development evidence only; current validation was used diagnostically; sealed test remains closed",
        "feature_contract": {
            "dimension": len(FEATURE_NAMES),
            "names": list(FEATURE_NAMES),
            "sha256": sha256_text(FEATURE_NAMES),
            "truth_runtime_input": False,
            "future_runtime_input": False,
            "motion_mode_runtime_input": False,
            "past_window_max_seconds": 0.2,
        },
        "label_contract": {
            "truth_used_offline_only": True,
            "classes": ["BENEFIT", "HARM", "UNCERTAIN"],
            "uncertain_training_weight": 0.0,
            "exploratory_transverse_margin_mm": 0.5,
            "margin_status": "temporary until repeated-pose measurement noise is collected",
            "angular_pointwise_tolerance_deg": 0.02,
            "radial_pointwise_factor": 1.02,
        },
        "policy": "final_apply = frozen_v3_apply AND benefit_gate_probability >= threshold",
        "selection_gate": {
            "per_session_angular_p95_noninferiority_deg": 0.02,
            "per_session_radial_p95_noninferiority_factor": 1.02,
            "per_session_transverse_p95_noninferiority": True,
            "minimum_oof_applied": 10,
            "minimum_oof_benefit_precision": 0.8,
            "maximum_oof_harm_apply_fraction": 0.05,
            "minimum_session_macro_transverse_p95_improvement_fraction": 0.01,
            "fallback": "reject_all",
        },
        "sessions": len(sessions),
        "development_samples": len(development),
        "validation_diagnostic_samples": len(validation),
        "selected_by_development_oof": selected_architecture,
        "fresh_validation_required": True,
        "sealed_test_access_authorized": False,
        "comparisons": comparisons,
        "model_artifacts": architecture_artifacts,
        "sample_artifact": {"file": rows_path.name, "sha256": sha256(rows_path), "bytes": rows_path.stat().st_size},
    }
    registry_path = output / "benefit_gate_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    (output / "retention_manifest.json").write_text(
        json.dumps(
            {
                "classification": "long_term_private_method_development_evidence",
                "deletion_allowed": False,
                "artifacts": {
                    path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                    for path in output.iterdir() if path.is_file() and path.name != "retention_manifest.json"
                },
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
