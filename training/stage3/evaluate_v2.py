"""Validation-first evaluation for qualified Stage-3 v2 shards."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baselines import load_geometry_template, rigid_constant_velocity_yaw_rate
from .model import Stage3TCN


PERMUTATIONS = tuple(itertools.permutations(range(4)))
HORIZON_BINS = (
    ("0_50ms", 0.0, 0.05),
    ("50_100ms", 0.05, 0.10),
    ("100_200ms", 0.10, 0.20),
    ("200_350ms", 0.20, 0.35),
    ("350_525ms", 0.35, 0.525001),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aligned_query_errors(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    best_error = float("inf")
    best: np.ndarray | None = None
    for permutation in PERMUTATIONS:
        aligned = prediction[:, list(permutation), :]
        errors = np.linalg.norm(aligned - target, axis=-1).mean(axis=-1)
        score = float(errors.mean())
        if score < best_error:
            best_error = score
            best = errors
    assert best is not None
    return best


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean_m": float(array.mean()),
        "median_m": float(np.quantile(array, 0.50)),
        "p90_m": float(np.quantile(array, 0.90)),
        "p95_m": float(np.quantile(array, 0.95)),
        "p99_m": float(np.quantile(array, 0.99)),
    }


def _horizon_name(tau: float) -> str:
    for name, low, high in HORIZON_BINS:
        if low <= tau <= high:
            return name
    return "out_of_domain"


def evaluate(args: argparse.Namespace) -> Path:
    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    if args.split == "test" and not args.allow_test:
        raise ValueError("test evaluation requires explicit --allow-test")
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "stage3-dataset-v3" or not manifest.get("qualification_passed"):
        raise ValueError("evaluation requires a qualified stage3-dataset-v3")
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8")) if args.selection else {}
    if selection and selection.get("dataset_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("selection does not match dataset manifest")
    wanted = set(str(value) for value in selection.get(args.split, ())) if selection else None
    if selection and not wanted:
        raise ValueError(f"selection has no {args.split} sessions")
    source_split = args.split
    if selection and args.split == "validation":
        source_split = str(selection.get("validation_source_split", "validation"))
        if source_split not in {"train", "validation"}:
            raise ValueError("invalid validation_source_split in selection")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    provenance = checkpoint.get("provenance", {})
    if provenance.get("schema_version") != "stage3-training-run-v1":
        raise ValueError("checkpoint has no accepted Stage-3 training provenance")
    if provenance.get("dataset_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("checkpoint does not match dataset manifest")
    if provenance.get("selection_sha256") and not args.selection:
        raise ValueError("checkpoint was selection-trained; matching --selection is required")
    if args.selection and provenance.get("selection_sha256") != _sha256(Path(args.selection)):
        raise ValueError("checkpoint does not match evaluation selection")
    if "model" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError("raw or legacy checkpoint is not accepted by v3 evaluation")
    model_config = checkpoint["model_config"]
    model = Stage3TCN(
        channels=int(model_config.get("channels", 64)),
        dropout=float(model_config.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    artifact_paths = {
        "normalization": dataset / manifest["normalization"],
        "geometry_template": dataset / manifest["geometry_template"],
        "qualification_report": dataset / manifest["qualification_report"],
    }
    for name, path in artifact_paths.items():
        if _sha256(path) != manifest.get("artifact_sha256", {}).get(name):
            raise ValueError(f"dataset artifact hash mismatch: {name}")
    normalization = json.loads(artifact_paths["normalization"].read_text(encoding="utf-8"))
    obs_mean = np.asarray(normalization["obs_xyz"]["mean"], dtype=np.float32)
    obs_std = np.asarray(normalization["obs_xyz"]["std"], dtype=np.float32)
    geometry = load_geometry_template(artifact_paths["geometry_template"])
    qualification = json.loads(artifact_paths["qualification_report"].read_text(encoding="utf-8"))
    mode_by_session = {
        str(item["session_id"]): str(item["manifest"]["mode"])
        for item in qualification["sessions"]
    }

    values: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in ("learned", "rigid_static", "rigid_cv_yaw_rate")
    }
    per_session: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    baseline_status = Counter()
    motion_correct = motion_total = 0
    sample_count = 0
    observed_sessions: set[str] = set()
    baseline_valid = Counter()
    shards = [item for item in manifest["shards"] if item["split"] == source_split]
    for shard in shards:
        if wanted is not None and not wanted.intersection(str(value) for value in shard["session_ids"]):
            continue
        shard_path = dataset / shard["path"]
        if _sha256(shard_path) != str(shard.get("sha256", "")):
            raise ValueError(f"shard hash mismatch: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        indices = np.arange(len(arrays["session_id"]))
        if wanted is not None:
            indices = np.asarray([
                index for index in indices if str(arrays["session_id"][index]) in wanted
            ], dtype=np.int64)
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            obs_raw = arrays["obs"][batch_indices].astype(np.float32, copy=True)
            obs_input = obs_raw.copy()
            obs_input[..., :3] = (obs_input[..., :3] - obs_mean) / obs_std
            with torch.no_grad():
                output_batch = model(
                    torch.from_numpy(obs_input).to(device),
                    torch.from_numpy(arrays["obs_mask"][batch_indices]).to(device),
                    torch.from_numpy(arrays["event_mask"][batch_indices]).to(device),
                    torch.from_numpy(arrays["event_time_s"][batch_indices]).to(device),
                    torch.from_numpy(arrays["tau"][batch_indices].astype(np.float32)).to(device),
                )
            learned_batch = output_batch["position_mean"].cpu().numpy()
            motion_batch = output_batch["motion_logits"].argmax(dim=-1).cpu().numpy()
            for local, index in enumerate(batch_indices):
                session_id = str(arrays["session_id"][index])
                observed_sessions.add(session_id)
                target = arrays["future_position"][index]
                tau = arrays["tau"][index]
                static_pred, _, static_status = rigid_constant_velocity_yaw_rate(
                    obs_raw[local], arrays["obs_mask"][index], arrays["event_mask"][index],
                    arrays["event_time_s"][index], tau, geometry, static=True,
                )
                rigid_pred, _, rigid_status = rigid_constant_velocity_yaw_rate(
                    obs_raw[local], arrays["obs_mask"][index], arrays["event_mask"][index],
                    arrays["event_time_s"][index], tau, geometry, static=False,
                )
                baseline_status[f"static_valid={static_status['valid']}"] += 1
                baseline_status[f"rigid_valid={rigid_status['valid']}"] += 1
                predictions = {"learned": learned_batch[local]}
                if bool(static_status["valid"]):
                    predictions["rigid_static"] = static_pred
                    baseline_valid["rigid_static"] += 1
                if bool(rigid_status["valid"]):
                    predictions["rigid_cv_yaw_rate"] = rigid_pred
                    baseline_valid["rigid_cv_yaw_rate"] += 1
                for name, prediction in predictions.items():
                    errors = _aligned_query_errors(prediction, target)
                    for query_index, error in enumerate(errors):
                        horizon = _horizon_name(float(tau[query_index]))
                        values[name]["overall"].append(float(error))
                        values[name][f"query:{query_index}"].append(float(error))
                        values[name][f"horizon:{horizon}"].append(float(error))
                        values[name][f"mode:{mode_by_session[session_id]}"].append(float(error))
                        values[name][
                            f"mode_horizon:{mode_by_session[session_id]}:{horizon}"
                        ].append(float(error))
                        per_session[session_id][name].append(float(error))
                motion_correct += int(int(motion_batch[local]) == int(arrays["motion_class"][index]))
                motion_total += 1
                sample_count += 1

    if sample_count == 0:
        raise ValueError("evaluation selected no samples")
    if wanted is not None and observed_sessions != wanted:
        raise ValueError(
            "evaluation selection did not exactly match observed sessions; "
            f"missing={sorted(wanted - observed_sessions)} "
            f"extra={sorted(observed_sessions - wanted)}"
        )
    if any(baseline_valid[name] != sample_count for name in ("rigid_static", "rigid_cv_yaw_rate")):
        raise ValueError(
            "acceptance evaluation requires 100% valid paired baselines; "
            f"coverage={dict(baseline_valid)} samples={sample_count}"
        )
    report = {
        "schema_version": "stage3-evaluation-v2",
        "split": args.split,
        "source_split": source_split,
        "test_accessed": args.split == "test",
        "sample_count": sample_count,
        "motion_accuracy": motion_correct / max(motion_total, 1),
        "baseline_status": dict(baseline_status),
        "baseline_valid_coverage": {
            name: baseline_valid[name] / sample_count
            for name in ("rigid_static", "rigid_cv_yaw_rate")
        },
        "metrics": {
            model_name: {group: _summary(group_values) for group, group_values in groups.items()}
            for model_name, groups in values.items()
        },
        "session_equal_weight": {
            model_name: _summary([
                float(np.median(groups[model_name])) for groups in per_session.values()
                if groups.get(model_name)
            ]) for model_name in values
        },
        "sessions": {
            session_id: {
                model_name: _summary(model_values)
                for model_name, model_values in groups.items()
            } for session_id, groups in per_session.items()
        },
        "checkpoint_provenance": provenance,
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": _sha256(Path(args.checkpoint)),
            "role": checkpoint.get("checkpoint_role"),
            "epoch": checkpoint.get("epoch"),
            "monitor": checkpoint.get("monitor"),
            "monitor_value": checkpoint.get("monitor_value"),
            "stop_reason": checkpoint.get("stop_reason"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "sample_count": sample_count,
        "learned": report["metrics"]["learned"]["overall"],
        "rigid_cv_yaw_rate": report["metrics"]["rigid_cv_yaw_rate"]["overall"],
    }, indent=2, sort_keys=True))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection", default="")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=256)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
