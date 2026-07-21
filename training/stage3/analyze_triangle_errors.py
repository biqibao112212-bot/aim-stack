"""Four-way Stage-3 error attribution on the exact validation exposures.

The four comparisons are deliberately kept separate:

* current observation ``O(t0)`` against truth ``G(t0)``;
* future observation ``O(t1)`` against truth ``G(t1)``;
* model prediction ``P(t1)`` against truth ``G(t1)``;
* model prediction ``P(t1)`` against future observation ``O(t1)``.

Future observations are joined by the exact ``future_timestamp_ns`` saved in
the v3 shard.  Missing exact frames are reported as coverage gaps; no nearest
frame or interpolation is used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .dataset import _observation_position_v3, load_camera_gimbal_extrinsic
from .evaluate_v2 import HORIZON_BINS, _horizon_name, _sha256
from .model import Stage3TCN
from .schema import ObservationFrame, iter_json_records


PERMUTATIONS = tuple(itertools.permutations(range(4)))


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    values = list(values)
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_m": float(array.mean()),
        "median_m": float(np.quantile(array, 0.50)),
        "p90_m": float(np.quantile(array, 0.90)),
        "p95_m": float(np.quantile(array, 0.95)),
        "p99_m": float(np.quantile(array, 0.99)),
    }


def _aligned_prediction(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-query errors and the single permutation used for the sample."""
    best_score = float("inf")
    best_errors: np.ndarray | None = None
    best_perm: np.ndarray | None = None
    for permutation in PERMUTATIONS:
        aligned = prediction[:, list(permutation), :]
        errors = np.linalg.norm(aligned - target, axis=-1).mean(axis=-1)
        score = float(errors.mean())
        if score < best_score:
            best_score = score
            best_errors = errors
            best_perm = np.asarray(permutation, dtype=np.int64)
    assert best_errors is not None and best_perm is not None
    return best_errors, best_perm


def _match_observation_to_truth(
    observation: np.ndarray, truth: np.ndarray
) -> tuple[float, dict[int, int]] | None:
    """Minimum-cost injective match, mapping truth slot -> observation row."""
    count = int(len(observation))
    if count < 1 or count > 4:
        return None
    distances = np.linalg.norm(observation[:, None, :] - truth[None, :, :], axis=-1)
    best_cost = float("inf")
    best: dict[int, int] | None = None
    for truth_slots in itertools.permutations(range(4), count):
        cost = float(sum(distances[row, slot] for row, slot in enumerate(truth_slots)))
        if cost < best_cost:
            best_cost = cost
            best = {int(slot): int(row) for row, slot in enumerate(truth_slots)}
    assert best is not None
    return best_cost / count, best


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denominator == 0.0 else float(np.dot(a, b) / denominator)


def _rank(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        value = values[order[start]]
        while end < len(values) and values[order[end]] == value:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    return _pearson(_rank(x).tolist(), _rank(y).tolist())


def _band(value: float, thresholds: tuple[float, ...], names: tuple[str, ...]) -> str:
    for threshold, name in zip(thresholds, names):
        if value < threshold:
            return name
    return names[-1]


def _speed_band(value: float) -> str:
    return _band(value, (0.02, 1.0, 2.0), ("zero", "low", "medium", "high"))


def _omega_band(value: float) -> str:
    return _band(abs(value), (0.02, 5.0, 10.0), ("zero", "low", "medium", "high"))


class Collectors:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self.sessions: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def add(self, metric: str, value: float, session_id: str, tags: Iterable[str]) -> None:
        self.values[metric]["overall"].append(float(value))
        self.values[metric][f"session:{session_id}"].append(float(value))
        self.sessions[session_id][metric].append(float(value))
        for tag in tags:
            self.values[metric][tag].append(float(value))


def _resolve_source(path: str, raw_root: Path) -> Path:
    candidate = Path(path).resolve()
    if candidate.is_file():
        return candidate
    # Keep the artifact paths authoritative, but make the report usable when
    # it was copied between machines and the absolute drive prefix changed.
    matches = list(raw_root.glob(f"*/run-*/{Path(path).name}"))
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError(f"canonical source is unavailable: {path}")


def _load_observation_index(path: Path, extrinsic: Any) -> dict[int, list[np.ndarray]]:
    result: dict[int, list[np.ndarray]] = {}
    for record in iter_json_records(path):
        frame = ObservationFrame.from_mapping(record)
        if frame.timestamp_ns in result:
            raise ValueError(f"duplicate observation timestamp in {path}: {frame.timestamp_ns}")
        positions: list[np.ndarray] = []
        for armor in frame.armors:
            if not armor.valid or not all(math.isfinite(v) for v in (*armor.position_m, armor.yaw_rad)):
                continue
            positions.append(_observation_position_v3(frame, armor.position_m, extrinsic))
        result[frame.timestamp_ns] = positions
    return result


def _tags(mode: str, distance: float, speed: float, omega: float, tau: float, visible: int | None = None) -> list[str]:
    tags = [
        f"mode:{mode}",
        f"distance:{'near' if distance < 3.0 else ('mid' if distance < 5.0 else 'far')}",
        f"speed:{_speed_band(speed)}",
        f"omega:{_omega_band(omega)}",
        f"horizon:{_horizon_name(tau)}",
    ]
    if visible is not None:
        tags.append(f"visible:{visible}")
    return tags


def analyze(args: argparse.Namespace) -> Path:
    dataset = Path(args.dataset).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite analysis output: {output}")
    if args.split == "test" and not args.allow_test:
        raise ValueError("test analysis requires explicit --allow-test")
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "stage3-dataset-v3" or not manifest.get("qualification_passed"):
        raise ValueError("analysis requires a qualified stage3-dataset-v3")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    provenance = checkpoint.get("provenance", {})
    if provenance.get("schema_version") != "stage3-training-run-v1":
        raise ValueError("checkpoint has no accepted Stage-3 provenance")
    if provenance.get("dataset_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("checkpoint does not match dataset manifest")
    config = checkpoint["model_config"]
    model = Stage3TCN(channels=int(config.get("channels", 64)), dropout=float(config.get("dropout", 0.1)))
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()

    def artifact(name: str) -> Path:
        path = dataset / str(manifest[name])
        if _sha256(path) != manifest.get("artifact_sha256", {}).get(name):
            raise ValueError(f"dataset artifact hash mismatch: {name}")
        return path

    normalization = json.loads(artifact("normalization").read_text(encoding="utf-8"))
    obs_mean = np.asarray(normalization["obs_xyz"]["mean"], dtype=np.float32)
    obs_std = np.asarray(normalization["obs_xyz"]["std"], dtype=np.float32)
    qualification = json.loads(artifact("qualification_report").read_text(encoding="utf-8"))
    sources_path = dataset / str(manifest["canonical_sources"])
    if _sha256(sources_path) != manifest.get("artifact_sha256", {}).get("canonical_sources"):
        raise ValueError("canonical_sources hash mismatch")
    sources = {str(item["session_id"]): item for item in iter_json_records(sources_path)}
    session_meta = {
        str(item["session_id"]): item["manifest"] for item in qualification["sessions"]
    }
    extrinsic_path = Path(str(manifest["camera_gimbal_extrinsic_yaml"])).resolve()
    if _sha256(extrinsic_path) != str(manifest.get("camera_gimbal_extrinsic_sha256", "")):
        raise ValueError("camera/gimbal extrinsic hash mismatch")
    extrinsic = load_camera_gimbal_extrinsic(extrinsic_path)

    collectors = Collectors()
    coverage = Counter()
    pairs = defaultdict(list)
    pair_slices: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    coverage_slices: dict[str, Counter] = defaultdict(Counter)
    sample_current: list[float] = []
    sample_prediction: list[float] = []
    sample_sessions: list[str] = []
    source_cache: dict[str, dict[int, list[np.ndarray]]] = {}
    session_query_count = Counter()
    observed_sessions: set[str] = set()
    sample_count = 0
    shards = [item for item in manifest["shards"] if item["split"] == args.split]
    for shard in shards:
        shard_path = dataset / str(shard["path"])
        if _sha256(shard_path) != str(shard["sha256"]):
            raise ValueError(f"shard hash mismatch: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        for start in range(0, len(arrays["session_id"]), args.batch_size):
            indices = np.arange(start, min(start + args.batch_size, len(arrays["session_id"])))
            raw = arrays["obs"][indices].astype(np.float32, copy=True)
            normalized = raw.copy()
            normalized[..., :3] = (normalized[..., :3] - obs_mean) / obs_std
            with torch.no_grad():
                output_batch = model(
                    torch.from_numpy(normalized).to(device),
                    torch.from_numpy(arrays["obs_mask"][indices]).to(device),
                    torch.from_numpy(arrays["event_mask"][indices]).to(device),
                    torch.from_numpy(arrays["event_time_s"][indices]).to(device),
                    torch.from_numpy(arrays["tau"][indices].astype(np.float32)).to(device),
                )
            predictions = output_batch["position_mean"].cpu().numpy()
            for local, index in enumerate(indices):
                session_id = str(arrays["session_id"][index])
                observed_sessions.add(session_id)
                sample_count += 1
                meta = session_meta[session_id]
                mode = str(meta.get("mode", "unknown"))
                distance = float(meta.get("distance_m", float("nan")))
                speed = float(meta.get("linear_speed_mps", 0.0))
                omega = float(meta.get("spin_rad_s", 0.0))
                truth = arrays["future_position"][index].astype(np.float64)
                tau = arrays["tau"][index].astype(np.float64)
                pred = predictions[local].astype(np.float64)
                pred_errors, permutation = _aligned_prediction(pred, truth)
                aligned_pred = pred[:, permutation, :]
                session_query_count[session_id] += len(tau)
                for query_index in range(len(tau)):
                    collectors.add(
                        "prediction_vs_truth",
                        float(pred_errors[query_index]),
                        session_id,
                        _tags(mode, distance, speed, omega, float(tau[query_index])),
                    )
                if session_id not in source_cache:
                    source = sources.get(session_id)
                    if source is None:
                        raise ValueError(f"no canonical source for {session_id}")
                    source_cache[session_id] = _load_observation_index(
                        _resolve_source(str(source["observations"]), dataset.parent.parent.parent / "autoaim-stage3-v1"),
                        extrinsic,
                    )
                observation_index = source_cache[session_id]
                current_value: float | None = None
                for query, timestamp in enumerate(arrays["future_timestamp_ns"][index]):
                    timestamp = int(timestamp)
                    base_tags = _tags(mode, distance, speed, omega, float(tau[query]))
                    slice_keys = ["overall", *base_tags]
                    for key in slice_keys:
                        coverage_slices[key]["total_queries"] += 1
                    coverage["total_queries"] += 1
                    visible = observation_index.get(timestamp)
                    if visible is None:
                        coverage["missing_exact_frame"] += 1
                        for key in slice_keys:
                            coverage_slices[key]["missing_exact_frame"] += 1
                        continue
                    coverage["exact_frame"] += 1
                    for key in slice_keys:
                        coverage_slices[key]["exact_frame"] += 1
                    if len(visible) > 4:
                        coverage["too_many_valid_candidates"] += 1
                        for key in slice_keys:
                            coverage_slices[key]["too_many_valid_candidates"] += 1
                        continue
                    if not visible:
                        coverage["zero_valid_candidates"] += 1
                        for key in slice_keys:
                            coverage_slices[key]["zero_valid_candidates"] += 1
                        continue
                    coverage["usable_exact_query"] += 1
                    for key in slice_keys:
                        coverage_slices[key]["usable_exact_query"] += 1
                    truth_query = truth[query]
                    matched = _match_observation_to_truth(np.asarray(visible), truth_query)
                    if matched is None:
                        continue
                    obs_error, assignment = matched
                    tags = _tags(mode, distance, speed, omega, float(tau[query]), len(visible))
                    collectors.add("future_observation_vs_truth", obs_error, session_id, tags)
                    pred_obs = float(np.mean([
                        np.linalg.norm(aligned_pred[query, slot] - np.asarray(visible)[row])
                        for slot, row in assignment.items()
                    ]))
                    collectors.add("prediction_vs_future_observation", pred_obs, session_id, tags)
                    pairs["future_observation_vs_truth"].append(float(obs_error))
                    pairs["prediction_vs_truth"].append(float(pred_errors[query]))
                    pairs["prediction_vs_future_observation"].append(pred_obs)
                    for key in [*slice_keys, f"visible:{len(visible)}"]:
                        pair_slices[key]["future_observation_vs_truth"].append(float(obs_error))
                        pair_slices[key]["prediction_vs_truth"].append(float(pred_errors[query]))
                        pair_slices[key]["prediction_vs_future_observation"].append(pred_obs)
                    if query == 0:
                        current_value = float(obs_error)
                if current_value is not None:
                    collectors.add("current_observation_vs_truth", current_value, session_id, _tags(mode, distance, speed, omega, 0.0))
                    sample_current.append(current_value)
                    sample_prediction.append(float(np.mean(pred_errors)))
                    sample_sessions.append(session_id)

    if not observed_sessions:
        raise ValueError("analysis selected no samples")
    metric_summaries = {
        metric: {group: _summary(values) for group, values in groups.items()}
        for metric, groups in collectors.values.items()
    }
    session_summaries = {
        session: {metric: _summary(values) for metric, values in metrics.items()}
        for session, metrics in collectors.sessions.items()
    }
    future_obs = pairs["future_observation_vs_truth"]
    pred_truth = pairs["prediction_vs_truth"]
    pred_obs = pairs["prediction_vs_future_observation"]
    med_obs = float(np.median(future_obs)) if future_obs else float("nan")
    med_pred = float(np.median(pred_truth)) if pred_truth else float("nan")
    attribution = {
        "paired_query_count": len(pred_truth),
        "correlation_future_observation_vs_prediction_truth": {
            "pearson": _pearson(future_obs, pred_truth),
            "spearman": _spearman(future_obs, pred_truth),
        },
        "correlation_future_observation_vs_prediction_observation": {
            "pearson": _pearson(future_obs, pred_obs),
            "spearman": _spearman(future_obs, pred_obs),
        },
        "current_observation_vs_sample_prediction_truth": {
            "pearson": _pearson(sample_current, sample_prediction),
            "spearman": _spearman(sample_current, sample_prediction),
            "paired_sample_count": len(sample_current),
        },
        "prediction_beats_future_observation_fraction": (
            float(np.mean(np.asarray(pred_truth) < np.asarray(future_obs))) if future_obs else None
        ),
        "mean_observation_minus_prediction_truth_m": (
            float(np.mean(np.asarray(future_obs) - np.asarray(pred_truth))) if future_obs else None
        ),
        "prediction_is_closer_to_future_observation_fraction": (
            float(np.mean(np.asarray(pred_obs) < np.asarray(pred_truth))) if pred_truth else None
        ),
        "quadrant_thresholds_m": {"future_observation_median": med_obs, "prediction_truth_median": med_pred},
        "quadrants": {},
    }
    if future_obs:
        for name, condition in (
            ("low_observation_low_prediction", (np.asarray(future_obs) <= med_obs) & (np.asarray(pred_truth) <= med_pred)),
            ("high_observation_low_prediction", (np.asarray(future_obs) > med_obs) & (np.asarray(pred_truth) <= med_pred)),
            ("low_observation_high_prediction", (np.asarray(future_obs) <= med_obs) & (np.asarray(pred_truth) > med_pred)),
            ("high_observation_high_prediction", (np.asarray(future_obs) > med_obs) & (np.asarray(pred_truth) > med_pred)),
        ):
            attribution["quadrants"][name] = {"count": int(condition.sum()), "fraction": float(condition.mean())}

    def pair_attribution(values: Mapping[str, list[float]]) -> dict[str, Any]:
        observation = np.asarray(values["future_observation_vs_truth"], dtype=np.float64)
        prediction = np.asarray(values["prediction_vs_truth"], dtype=np.float64)
        prediction_observation = np.asarray(values["prediction_vs_future_observation"], dtype=np.float64)
        if not len(observation):
            return {"count": 0}
        return {
            "count": int(len(observation)),
            "future_observation_vs_truth": _summary(observation.tolist()),
            "prediction_vs_truth": _summary(prediction.tolist()),
            "prediction_vs_future_observation": _summary(prediction_observation.tolist()),
            "prediction_beats_future_observation_fraction": float(np.mean(prediction < observation)),
            "mean_observation_minus_prediction_truth_m": float(np.mean(observation - prediction)),
            "prediction_is_closer_to_future_observation_fraction": float(np.mean(prediction_observation < prediction)),
        }

    attribution["slices"] = {
        key: pair_attribution(values) for key, values in pair_slices.items()
    }
    report = {
        "schema_version": "stage3-four-way-error-analysis-v1",
        "split": args.split,
        "sample_count": sample_count,
        "query_count": int(coverage["total_queries"]),
        "metrics": metric_summaries,
        "sessions": session_summaries,
        "coverage": dict(coverage),
        "coverage_rates": {
            "exact_frame": coverage["exact_frame"] / max(coverage["total_queries"], 1),
            "usable_exact_query": coverage["usable_exact_query"] / max(coverage["total_queries"], 1),
        },
        "coverage_by_slice": {
            key: {
                **dict(counts),
                "exact_frame_rate": counts["exact_frame"] / max(counts["total_queries"], 1),
                "usable_exact_query_rate": counts["usable_exact_query"] / max(counts["total_queries"], 1),
            }
            for key, counts in coverage_slices.items()
        },
        "attribution": attribution,
        "exact_timestamp_contract": {
            "timestamp_field": "future_timestamp_ns",
            "nearest_or_interpolated_observation_used": False,
            "matching": "minimum-cost-injective-observation-to-four-truth-slots",
        },
        "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path), "provenance": provenance},
        "dataset_manifest_sha256": _sha256(manifest_path),
        "calibration": {"path": str(extrinsic_path), "sha256": _sha256(extrinsic_path)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"query_count": report["query_count"], "coverage": report["coverage_rates"], "metrics": {k: v.get("overall", {}) for k, v in metric_summaries.items()}}, indent=2, sort_keys=True))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=256)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
