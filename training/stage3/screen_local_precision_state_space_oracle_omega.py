"""Train-only state-space ceiling for local-precision profiled twist.

This screen trains no network.  On the two sealed train-session folds it uses
truth omega only as the declared fixed-omega mechanism slice and truth velocity
only inside a three-start, 32-step local-precision oracle evaluator.  Mapper,
S, H, V14, validation, test, future modules, and plotting remain frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import torch

from .cyclic_future_foundation import load_frozen_v19
from .evaluate_profiled_center_twist_mechanism import (
    _validate_checkpoint_manifest_identity,
)
from .frozen_expert_reliability_fusion import oracle_projection_coefficient
from .locally_weighted_profiled_twist import LocallyWeightedProfiledTwistAtOmega
from .loss_only_local_precision_oracle import (
    assert_forward_cache_truth_free,
    optimize_loss_only_local_precision_oracle,
)
from .motion_truth_supervision import MOTION_TARGET_FIELD
from .observable_future_pnp_ab import sha256_file, state_dict_sha256
from .pnp_q0_hypothesis_adapter import (
    load_frozen_hypothesis_adapter,
    load_frozen_pnp_mapper,
)
from .profiled_center_twist_future import (
    CENTER_TWIST_FORWARD_FIELDS,
    CenterPriorProfiledTwistScreen,
)
from .screen_frozen_expert_reliability_oracle_omega import (
    _cache_prepared_state,
    _combined_metadata,
    _distribution,
    _fold_assignments,
)
from .screen_prequential_temporal_reliability_oracle_omega import (
    validate_a2_artifacts,
)
from .split_scoped_truth_supervision import (
    SplitScopedTruthIndex,
    assert_manifest_split_shards_unchanged,
)
from .train_causal_physical_ab import _git_state, _seed
from .train_pnp_window_mapper_distillation import _atomic_json
from .screen_endpoint_token_reliability_oracle_omega import _load_train_dataset


RUN_SCHEMA = "stage3-local-precision-state-space-oracle-omega-v15-a3-p0"
V93_RESULT_SHA256 = (
    "8f34bf7121827c5e744b617c3c9648a578599cdf4c7329e635d2db0ac0d57561"
)
FIXED_FOLDS = 2
FIXED_SEED = 20260730
FIXED_ORACLE_STEPS = 32
FIXED_ORACLE_CHUNK_SIZE = 32
RECOVERY_GATE = 0.40
MIN_PARENT_PROJECTION_HEADROOM_MPS = 0.02
MIN_PARENT_LOCAL_IMPROVEMENT_MPS = 0.01
CHECKPOINT_PREFIX = "oracle-resume-g"
CONTRACT_NAME = "experiment_contract.json"
SIDECAR_DIRECTORY = "oracle-sidecar"


def _source_paths() -> dict[str, Path]:
    here = Path(__file__).resolve()
    return {
        "runner": here,
        "weighted_profile": here.with_name(
            "locally_weighted_profiled_twist.py"
        ).resolve(),
        "loss_only_oracle": here.with_name(
            "loss_only_local_precision_oracle.py"
        ).resolve(),
        "prequential_local_precision_irls": here.with_name(
            "prequential_local_precision_irls.py"
        ).resolve(),
        "a2_runner": here.with_name(
            "screen_prequential_temporal_reliability_oracle_omega.py"
        ).resolve(),
        "v14_profile": here.with_name("profiled_center_twist_future.py").resolve(),
    }


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _indices_sha256(indices: np.ndarray | torch.Tensor) -> str:
    if isinstance(indices, torch.Tensor):
        array = indices.detach().cpu().numpy()
    else:
        array = np.asarray(indices)
    canonical = np.ascontiguousarray(array, dtype=np.int64)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    return hashlib.sha256(array.tobytes()).hexdigest()


def validate_v93_failure(
    path: str | Path,
    *,
    validator: Callable[[str | Path], tuple[dict[str, Any], dict[str, Any]]] = (
        validate_a2_artifacts
    ),
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """First strict-load A2, then accept only the sealed v93 failed result."""
    root = Path(path).resolve()
    payload, result = validator(root)
    result_path = root / "screen_result.json"
    if sha256_file(result_path) != V93_RESULT_SHA256:
        raise ValueError("state-space oracle requires the sealed v93 result hash")
    if (
        result.get("status") != "failed"
        or result.get("validation_accessed") is not False
        or result.get("test_accessed") is not False
        or result.get("future_modules_loaded") is not False
        or result.get("train_only") is not True
        or result.get("authorized_a2_train_counterfactual") is not False
    ):
        raise ValueError("sealed v93 failure status/scope differs")
    return payload, result, result_path


def _anchor_availability(center_supported: torch.Tensor) -> torch.Tensor:
    if center_supported.ndim != 1 or center_supported.dtype != torch.bool:
        raise ValueError("center support must have shape [B] and boolean dtype")
    return center_supported[:, None].expand(-1, 4).clone()


def _forward_cache(
    state: dict[str, torch.Tensor],
    omega_rad_s: torch.Tensor,
    center_prior: dict[str, torch.Tensor],
) -> dict[str, Any]:
    missing = set(CENTER_TWIST_FORWARD_FIELDS) - set(state)
    if missing:
        raise ValueError(f"oracle forward state fields missing: {sorted(missing)}")
    cache: dict[str, Any] = {
        **{name: state[name] for name in CENTER_TWIST_FORWARD_FIELDS},
        "omega_rad_s": omega_rad_s,
        "center_prior": center_prior,
    }
    assert_forward_cache_truth_free(cache)
    return cache


def _repeat_forward_cache(cache: dict[str, Any], repeats: int) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("forward-cache repeat count must be positive")
    result: dict[str, Any] = {}
    for name, value in cache.items():
        if name == "center_prior":
            result[name] = {
                key: tensor.repeat(repeats, *([1] * (tensor.ndim - 1)))
                for key, tensor in value.items()
            }
        else:
            result[name] = value.repeat(repeats, *([1] * (value.ndim - 1)))
    assert_forward_cache_truth_free(result)
    return result


def _build_oracle_solve(
    weighted: LocallyWeightedProfiledTwistAtOmega,
    cache: dict[str, Any],
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
    repeated = _repeat_forward_cache(cache, 3)

    def solve(
        observation_log_precision: torch.Tensor,
        anchor_alpha: torch.Tensor,
        center_alpha: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        result = weighted(
            repeated["history_obs_rel_m"], repeated["history_obs_mask"],
            repeated["history_event_mask"], repeated["history_time_s"],
            repeated["omega_rad_s"], repeated["center_prior"],
            observation_log_precision=observation_log_precision,
            anchor_alpha=anchor_alpha, center_alpha=center_alpha,
            use_q0_prior=True,
        )
        # The oracle optimizes only the profiled local state.  V14's fallback
        # is a reporting fallback, not an implicit third velocity expert.
        return {**result, "state_supported": result["profile_supported"]}

    return solve


def _optimize_chunk(
    weighted: LocallyWeightedProfiledTwistAtOmega,
    cache: dict[str, Any],
    target_velocity_mps: torch.Tensor,
    *,
    optimize: Callable[..., dict[str, torch.Tensor]] = (
        optimize_loss_only_local_precision_oracle
    ),
) -> dict[str, torch.Tensor]:
    availability = _anchor_availability(
        cache["center_prior"]["center_supported"].to(torch.bool)
    )
    result = optimize(
        _build_oracle_solve(weighted, cache),
        observation_mask=(
            cache["history_obs_mask"].to(torch.bool)
            & cache["history_event_mask"].to(torch.bool).unsqueeze(-1)
        ),
        q0_supported=availability,
        target_velocity_mps=target_velocity_mps,
        steps=FIXED_ORACLE_STEPS,
    )
    return {**result, "profile_supported": result["state_supported"]}


def _slice_cache(cache: dict[str, Any], row: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in cache.items():
        if name == "center_prior":
            result[name] = {key: tensor[row:row + 1] for key, tensor in value.items()}
        else:
            result[name] = value[row:row + 1]
    return result


@torch.inference_mode()
def _verify_fp32_boundaries(
    weighted: LocallyWeightedProfiledTwistAtOmega,
    base: CenterPriorProfiledTwistScreen,
    cache: dict[str, Any],
) -> int:
    """Require batch-exact history/q0 delegation on every real FP32 row."""
    batch, events = cache["history_obs_rel_m"].shape[:2]
    prior = cache["center_prior"]
    visible = (
        cache["history_obs_mask"].to(torch.bool)
        & cache["history_event_mask"].to(torch.bool).unsqueeze(-1)
    )
    zero_logit = torch.zeros(
        batch, events, 4, dtype=torch.float32,
        device=cache["history_obs_rel_m"].device,
    )
    center_available = prior["center_supported"].to(torch.bool)
    available = _anchor_availability(center_available).to(torch.float32)
    for use_q0, anchor, center in (
        (
            False, torch.zeros_like(available),
            torch.zeros(batch, device=available.device),
        ),
        (True, available, center_available.to(torch.float32)),
    ):
        reference = base.profile(
            cache["history_obs_rel_m"], cache["history_obs_mask"],
            cache["history_event_mask"], cache["history_time_s"],
            cache["omega_rad_s"], prior, use_q0_prior=use_q0,
        )
        actual = weighted(
            cache["history_obs_rel_m"], cache["history_obs_mask"],
            cache["history_event_mask"], cache["history_time_s"],
            cache["omega_rad_s"], prior,
            observation_log_precision=zero_logit,
            anchor_alpha=anchor, center_alpha=center,
            use_q0_prior=True,
        )
        for name in (
            "velocity_mps", "profile_supported", "state_supported",
            "fallback_velocity_mps", "fallback_supported",
        ):
            if not torch.equal(actual[name], reference[name]):
                raise RuntimeError(
                    f"FP32 {('q0' if use_q0 else 'history')} boundary differs"
                )
        if not bool(actual["boundary_delegated"].all()):
            raise RuntimeError("FP32 boundary did not use frozen V14 delegate")
    if bool(torch.any(visible & ~torch.isfinite(cache["history_obs_rel_m"]).all(-1))):
        raise RuntimeError("boundary cache visible state became non-finite")
    return batch


def _recovery(
    parent: dict[str, float | int],
    projection: dict[str, float | int],
    local: dict[str, float | int],
    key: str,
) -> float:
    headroom = float(parent[key]) - float(projection[key])
    if not np.isfinite(headroom) or headroom <= 0:
        return 0.0
    return (float(parent[key]) - float(local[key])) / headroom


def _fold_gates(metrics: dict[str, Any]) -> dict[str, bool | float]:
    gates: dict[str, bool | float] = {}
    for group in ("overall", "rotation", "combined"):
        values = metrics["groups"][group]
        gates[f"{group}_local_coverage_100pct"] = (
            values["local_common_count"] == values["reference_common_count"]
            and values["reference_common_count"] > 0
        )
        gates[f"{group}_full_count_matches_a2"] = (
            values["full_count"] == values["expected_full_count"]
        )
        gates[f"{group}_reference_common_ge_a2_min"] = (
            values["reference_common_count"]
            >= values["expected_min_reference_common_count"]
        )
        for statistic in ("mean", "p50"):
            headroom = (
                float(values["parent_error_mps"][statistic])
                - float(values["projection_error_mps"][statistic])
            )
            improvement = (
                float(values["parent_error_mps"][statistic])
                - float(values["local_error_mps"][statistic])
            )
            recovery = _recovery(
                values["parent_error_mps"], values["projection_error_mps"],
                values["local_error_mps"], statistic,
            )
            gates[f"{group}_{statistic}_recovery"] = recovery
            gates[f"{group}_{statistic}_parent_projection_headroom_mps"] = headroom
            gates[f"{group}_{statistic}_parent_local_improvement_mps"] = improvement
            gates[f"{group}_{statistic}_recovery_ge_0_40"] = (
                recovery >= RECOVERY_GATE
            )
            gates[f"{group}_{statistic}_headroom_ge_0_02_mps"] = (
                headroom >= MIN_PARENT_PROJECTION_HEADROOM_MPS
            )
            gates[f"{group}_{statistic}_improvement_ge_0_01_mps"] = (
                improvement >= MIN_PARENT_LOCAL_IMPROVEMENT_MPS
            )
    return gates


def _all_boolean_gates_pass(gates: dict[str, bool | float]) -> bool:
    boolean = [value for value in gates.values() if isinstance(value, bool)]
    return bool(boolean) and all(boolean)


def _authorized(folds: list[dict[str, Any]]) -> bool:
    return (
        len(folds) == FIXED_FOLDS
        and all(item.get("passed") is True for item in folds)
    )


def _atomic_torch_replace(path: Path, payload: Any) -> None:
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}-{time.time_ns()}")
    try:
        torch.save(payload, pending)
        pending.replace(path)
    finally:
        if pending.exists():
            pending.unlink()


def _atomic_torch_create(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite accepted oracle shard: {path}")
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}-{time.time_ns()}")
    try:
        torch.save(payload, pending)
        # A hard-link publication is atomic and cannot replace a destination
        # created by a concurrent/resumed process between the check and commit.
        os.link(pending, path)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite accepted oracle shard: {path}"
        ) from exc
    finally:
        if pending.exists():
            pending.unlink()


def _rng_payload(device: torch.device) -> dict[str, Any]:
    return {
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if device.type == "cuda" else []
        ),
    }


def _restore_rng(payload: dict[str, Any], device: torch.device) -> None:
    cpu = payload.get("cpu_rng_state")
    cuda = payload.get("cuda_rng_state_all")
    if not isinstance(cpu, torch.Tensor) or not isinstance(cuda, list):
        raise ValueError("oracle resume RNG state is missing")
    torch.set_rng_state(cpu)
    if device.type == "cuda":
        if len(cuda) != torch.cuda.device_count():
            raise ValueError("oracle resume CUDA RNG device count differs")
        torch.cuda.set_rng_state_all(cuda)


def _checkpoint_payload(
    contract_sha: str,
    completed: dict[str, dict[str, Any]],
    device: torch.device,
    *,
    status: str,
    generation: int,
    rng_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rng = _rng_payload(device) if rng_source is None else {
        "cpu_rng_state": rng_source.get("cpu_rng_state"),
        "cuda_rng_state_all": rng_source.get("cuda_rng_state_all"),
    }
    if (
        not isinstance(rng["cpu_rng_state"], torch.Tensor)
        or not isinstance(rng["cuda_rng_state_all"], list)
        or not all(
            isinstance(value, torch.Tensor)
            for value in rng["cuda_rng_state_all"]
        )
    ):
        raise ValueError("oracle checkpoint RNG source differs")
    return {
        "schema_version": RUN_SCHEMA,
        "status": status,
        "generation": generation,
        "experiment_contract_sha256": contract_sha,
        "completed_shards": completed,
        **rng,
    }


def _checkpoint_path(output: Path, generation: int) -> Path:
    if generation <= 0:
        raise ValueError("oracle checkpoint generation must be positive")
    return output / f"{CHECKPOINT_PREFIX}{generation:08d}.pt"


def _next_checkpoint_generation(output: Path) -> int:
    generations = []
    for path in output.glob(f"{CHECKPOINT_PREFIX}*.pt"):
        suffix = path.stem.removeprefix(CHECKPOINT_PREFIX)
        if suffix.isdigit():
            generations.append(int(suffix))
    return max(generations, default=0) + 1


def _write_checkpoint_and_state(
    output: Path,
    contract_sha: str,
    completed: dict[str, dict[str, Any]],
    device: torch.device,
    *,
    status: str,
    result_sha: str | None = None,
    authorized: bool = False,
    generation: int | None = None,
    rng_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if generation is None:
        generation = _next_checkpoint_generation(output)
    checkpoint_path = _checkpoint_path(output, generation)
    payload = _checkpoint_payload(
        contract_sha, completed, device, status=status, generation=generation,
        rng_source=rng_source,
    )
    if checkpoint_path.exists():
        stored = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False,
        )
        if (
            stored.get("schema_version") != RUN_SCHEMA
            or stored.get("generation") != generation
            or stored.get("status") != status
            or stored.get("experiment_contract_sha256") != contract_sha
            or stored.get("completed_shards") != completed
        ):
            raise ValueError("existing oracle checkpoint generation differs")
        payload = stored
    else:
        _atomic_torch_create(checkpoint_path, payload)
    cpu_sha = _tensor_sha256(payload["cpu_rng_state"])
    cuda_sha = [_tensor_sha256(value) for value in payload["cuda_rng_state_all"]]
    run_state = {
        "schema_version": RUN_SCHEMA,
        "status": status,
        "generation": generation,
        "experiment_contract_sha256": contract_sha,
        "validation_accessed": False,
        "test_accessed": False,
        "future_modules_loaded": False,
        "authorized_train_local_precision": authorized,
        "completed_shards": completed,
        "oracle_checkpoint": str(checkpoint_path.resolve()),
        "oracle_checkpoint_sha256": sha256_file(checkpoint_path),
        "cpu_rng_state_sha256": cpu_sha,
        "cuda_rng_state_sha256": cuda_sha,
        "screen_result_sha256": result_sha,
    }
    _atomic_json(output / "run_state.json", run_state)
    return payload


def _validate_sidecar_payload(
    payload: dict[str, Any],
    expected: dict[str, Any],
    contract_sha: str,
) -> None:
    if (
        payload.get("schema_version") != RUN_SCHEMA
        or payload.get("experiment_contract_sha256") != contract_sha
        or payload.get("shard_id") != expected["shard_id"]
        or payload.get("fold") != expected["fold"]
        or payload.get("start") != expected["start"]
        or payload.get("stop") != expected["stop"]
        or payload.get("sample_indices_sha256") != expected[
            "sample_indices_sha256"
        ]
        or payload.get("raw_label_fields") != []
        or any(
            any(token in name.lower() for token in ("truth", "target", "future"))
            for name in payload
        )
    ):
        raise ValueError("oracle sidecar identity/scope differs")
    indices = payload.get("sample_indices")
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype != torch.int64
        or indices.ndim != 1
        or indices.shape[0] != expected["stop"] - expected["start"]
        or _indices_sha256(indices) != expected["sample_indices_sha256"]
    ):
        raise ValueError("oracle sidecar sample indices differ")
    count = indices.shape[0]
    tensor_fields = (
        "motion_class", "parent_error_mps", "projection_error_mps",
        "local_error_mps", "reference_common", "common",
        "local_profile_supported", "best_start", "anchor_alpha",
        "center_alpha", "observation_log_precision",
    )
    for name in tensor_fields:
        value = payload.get(name)
        if not isinstance(value, torch.Tensor) or value.shape[0] != count:
            raise ValueError(f"oracle sidecar {name} shape differs")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"oracle sidecar {name} is non-finite")
    if (
        payload["motion_class"].ndim != 1
        or payload["parent_error_mps"].ndim != 1
        or payload["projection_error_mps"].ndim != 1
        or payload["local_error_mps"].ndim != 1
        or payload["reference_common"].dtype != torch.bool
        or payload["reference_common"].ndim != 1
        or payload["common"].dtype != torch.bool
        or payload["common"].ndim != 1
        or payload["local_profile_supported"].dtype != torch.bool
        or payload["local_profile_supported"].ndim != 1
        or payload["best_start"].ndim != 1
        or payload["anchor_alpha"].shape != (count, 4)
        or payload["center_alpha"].ndim != 1
        or payload["observation_log_precision"].ndim != 3
        or payload["observation_log_precision"].shape[2] != 4
        or not torch.equal(
            payload["common"],
            payload["reference_common"] & payload["local_profile_supported"],
        )
    ):
        raise ValueError("oracle sidecar support/domain fields differ")


def _expected_shards(
    fold_indices: list[np.ndarray], *, chunk_size: int,
) -> dict[str, dict[str, Any]]:
    if chunk_size != FIXED_ORACLE_CHUNK_SIZE:
        raise ValueError("state-space oracle chunk size is fixed")
    result: dict[str, dict[str, Any]] = {}
    for fold, indices in enumerate(fold_indices):
        for start in range(0, len(indices), chunk_size):
            stop = min(start + chunk_size, len(indices))
            shard_id = f"fold{fold:02d}-{start:06d}-{stop:06d}"
            selected = indices[start:stop]
            result[shard_id] = {
                "shard_id": shard_id, "fold": fold,
                "start": start, "stop": stop,
                "sample_indices_sha256": _indices_sha256(selected),
            }
    return result


def _strict_completed_shards(
    output: Path,
    completed: dict[str, dict[str, Any]],
    expected: dict[str, dict[str, Any]],
    contract_sha: str,
) -> None:
    if not isinstance(completed, dict) or not set(completed).issubset(expected):
        raise ValueError("oracle completed-shard set differs")
    sidecar_root = (output / SIDECAR_DIRECTORY).resolve()
    for shard_id, manifest in completed.items():
        path = Path(manifest.get("path", "")).resolve()
        if (
            path.parent != sidecar_root
            or path.name != f"{shard_id}.pt"
            or sha256_file(path) != manifest.get("sha256")
        ):
            raise ValueError("accepted oracle shard changed")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _validate_sidecar_payload(payload, expected[shard_id], contract_sha)


def _initialize_or_resume(
    output: Path,
    contract: dict[str, Any],
    expected: dict[str, dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    contract_sha = _json_sha256(contract)
    contract_path = output / CONTRACT_NAME
    if not output.exists():
        output.mkdir(parents=True)
        (output / SIDECAR_DIRECTORY).mkdir()
        _atomic_json(contract_path, contract)
        completed: dict[str, dict[str, Any]] = {}
        checkpoint = _write_checkpoint_and_state(
            output, contract_sha, completed, device, status="running",
        )
        return completed, checkpoint
    if not output.is_dir() or not contract_path.exists():
        raise FileExistsError("existing oracle output has no resumable contract")
    stored_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if stored_contract != contract or _json_sha256(stored_contract) != contract_sha:
        raise ValueError("oracle resume experiment contract differs")
    state_path = output / "run_state.json"
    if not state_path.exists():
        raise ValueError("oracle resume metadata is incomplete")
    run_state = json.loads(state_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(run_state.get("oracle_checkpoint", "")).resolve()
    generation = run_state.get("generation")
    if (
        not isinstance(generation, int)
        or generation <= 0
        or checkpoint_path != _checkpoint_path(output, generation).resolve()
        or not checkpoint_path.exists()
    ):
        raise ValueError("oracle resume commit pointer differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cpu_rng = checkpoint.get("cpu_rng_state")
    cuda_rng = checkpoint.get("cuda_rng_state_all")
    if (
        run_state.get("schema_version") != RUN_SCHEMA
        or checkpoint.get("schema_version") != RUN_SCHEMA
        or checkpoint.get("generation") != generation
        or run_state.get("status") not in {"running", "passed", "failed"}
        or checkpoint.get("status") != run_state.get("status")
        or run_state.get("experiment_contract_sha256") != contract_sha
        or checkpoint.get("experiment_contract_sha256") != contract_sha
        or run_state.get("validation_accessed") is not False
        or run_state.get("test_accessed") is not False
        or run_state.get("future_modules_loaded") is not False
        or run_state.get("oracle_checkpoint_sha256") != sha256_file(checkpoint_path)
        or run_state.get("completed_shards") != checkpoint.get("completed_shards")
        or not isinstance(cpu_rng, torch.Tensor)
        or not isinstance(cuda_rng, list)
        or run_state.get("cpu_rng_state_sha256") != _tensor_sha256(cpu_rng)
        or run_state.get("cuda_rng_state_sha256")
        != [_tensor_sha256(value) for value in cuda_rng]
    ):
        raise ValueError("oracle resume state/checkpoint differs")
    completed = checkpoint.get("completed_shards")
    _strict_completed_shards(output, completed, expected, contract_sha)
    sidecar_root = output / SIDECAR_DIRECTORY
    if not sidecar_root.is_dir():
        raise ValueError("oracle sidecar directory is missing")
    # A crash may leave a complete atomically-created shard just before its
    # checkpoint update.  Strict-load and adopt it; never overwrite it.
    _restore_rng(checkpoint, device)
    adopted = False
    for path in sidecar_root.glob("*.pt"):
        shard_id = path.stem
        if shard_id in completed:
            continue
        if run_state.get("status") != "running":
            raise ValueError("completed oracle has an unaccepted sidecar")
        if shard_id not in expected:
            raise ValueError("unexpected oracle sidecar shard exists")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _validate_sidecar_payload(payload, expected[shard_id], contract_sha)
        completed[shard_id] = {
            "path": str(path.resolve()), "sha256": sha256_file(path),
        }
        adopted = True
    if adopted:
        checkpoint = _write_checkpoint_and_state(
            output, contract_sha, completed, device,
            status=run_state.get("status", "running"),
            result_sha=run_state.get("screen_result_sha256"),
            authorized=bool(run_state.get("authorized_train_local_precision", False)),
        )
    if run_state.get("status") == "running":
        recovered = _converge_recoverable_final_result(
            output, contract, expected, completed, device,
            rng_source=checkpoint,
        )
        if recovered is not None:
            _, validated = validate_local_precision_oracle_artifacts(output)
            return completed, validated
    if run_state.get("status") in {"passed", "failed"}:
        _, result = validate_local_precision_oracle_artifacts(output)
        return completed, result
    if run_state.get("status") != "running":
        raise ValueError("oracle resume status differs")
    return completed, checkpoint


def _commit_shard(
    output: Path,
    contract_sha: str,
    expected: dict[str, Any],
    payload: dict[str, Any],
    completed: dict[str, dict[str, Any]],
    device: torch.device,
) -> None:
    shard_id = expected["shard_id"]
    if shard_id in completed:
        raise FileExistsError("refusing to overwrite accepted oracle shard")
    _validate_sidecar_payload(payload, expected, contract_sha)
    path = output / SIDECAR_DIRECTORY / f"{shard_id}.pt"
    _atomic_torch_create(path, payload)
    manifest = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    # Reload before accepting it into the checkpoint.
    stored = torch.load(path, map_location="cpu", weights_only=False)
    _validate_sidecar_payload(stored, expected, contract_sha)
    if sha256_file(path) != manifest["sha256"]:
        raise RuntimeError("oracle shard changed during acceptance")
    completed[shard_id] = manifest
    _write_checkpoint_and_state(
        output, contract_sha, completed, device, status="running",
    )


def _weighted_like(base: CenterPriorProfiledTwistScreen) -> LocallyWeightedProfiledTwistAtOmega:
    profile = base.profile
    return LocallyWeightedProfiledTwistAtOmega(
        center_precision=profile.center_precision,
        history_center_precision=profile.history_center_precision,
        q_ridge=profile.q_ridge,
        q0_endpoint_precision=profile.q0_endpoint_precision,
        use_learned_center_variance=profile.use_learned_center_variance,
        minimum_velocity_information_s2=profile.minimum_velocity_information_s2,
        maximum_velocity_condition=profile.maximum_velocity_condition,
        minimum_time_span_s=profile.minimum_time_span_s,
    )


def _sidecar_payload(
    *,
    expected: dict[str, Any],
    contract_sha: str,
    sample_indices: np.ndarray,
    motion_class: torch.Tensor,
    parent_velocity: torch.Tensor,
    projection_velocity: torch.Tensor,
    projection_valid: torch.Tensor,
    local: dict[str, torch.Tensor],
    target_velocity: torch.Tensor,
    parent_supported: torch.Tensor,
) -> dict[str, Any]:
    reference_common = (
        projection_valid.to(torch.bool) & parent_supported.to(torch.bool)
    )
    common = reference_common & local["profile_supported"].to(torch.bool)
    error = lambda value: torch.linalg.vector_norm(
        value - target_velocity, dim=-1,
    ).detach().cpu()
    return {
        "schema_version": RUN_SCHEMA,
        "experiment_contract_sha256": contract_sha,
        **expected,
        "sample_indices": torch.from_numpy(sample_indices.copy()).long(),
        "raw_label_fields": [],
        "motion_class": motion_class.detach().cpu(),
        "parent_error_mps": error(parent_velocity),
        "projection_error_mps": error(projection_velocity),
        "local_error_mps": error(local["velocity_mps"]),
        "reference_common": reference_common.detach().cpu(),
        "common": common.detach().cpu(),
        "local_profile_supported": local["profile_supported"].detach().cpu(),
        "best_start": local["best_start"].detach().cpu(),
        "anchor_alpha": local["anchor_alpha"].detach().cpu(),
        "center_alpha": local["center_alpha"].detach().cpu(),
        "observation_log_precision": local[
            "observation_log_precision"
        ].detach().cpu(),
    }


def _a2_expected_reference_counts(
    a2_result: dict[str, Any],
) -> list[dict[str, dict[str, int]]]:
    cv = a2_result.get("cv")
    if not isinstance(cv, list) or len(cv) != FIXED_FOLDS:
        raise ValueError("sealed A2 fold count is missing")
    expected: list[dict[str, dict[str, int]]] = []
    for fold, item in enumerate(cv):
        if item.get("fold") != fold:
            raise ValueError("sealed A2 fold order differs")
        try:
            groups = item["metrics"]["arms"]["intact"]["groups"]
            expected.append({
                group: {
                    "full_count": int(groups[group]["full_count"]),
                    "min_reference_common_count": int(
                        groups[group]["common_count"]
                    ),
                }
                for group in ("overall", "rotation", "combined")
            })
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("sealed A2 intact reference counts are missing") from exc
    return expected


def _summarize_fold(
    payloads: list[dict[str, Any]],
    expected_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    if not payloads:
        raise ValueError("oracle fold has no sidecars")
    values = {
        name: torch.cat([payload[name] for payload in payloads], dim=0)
        for name in (
            "motion_class", "parent_error_mps", "projection_error_mps",
            "local_error_mps", "reference_common", "common",
            "local_profile_supported",
        )
    }
    reference_common = values["reference_common"].to(torch.bool)
    local_supported = values["local_profile_supported"].to(torch.bool)
    common = values["common"].to(torch.bool)
    if not torch.equal(common, reference_common & local_supported):
        raise ValueError("oracle local common domain differs")
    result: dict[str, Any] = {
        "sample_count": int(reference_common.numel()),
        "reference_common_count": int(reference_common.sum()),
        "local_common_count": int(common.sum()),
        "local_coverage_fraction": float(
            common.sum().double() / reference_common.sum().clamp_min(1)
        ),
        "local_profile_supported_count": int(
            values["local_profile_supported"].to(torch.bool).sum()
        ),
        "groups": {},
    }
    for group, group_mask in {
        "overall": torch.ones_like(common),
        "rotation": values["motion_class"] == 2,
        "combined": values["motion_class"] == 3,
    }.items():
        selected = reference_common & group_mask
        local_common = common & group_mask
        if not bool(selected.any()):
            raise ValueError(f"oracle fold has no reference-common {group} rows")
        parent = _distribution(values["parent_error_mps"][selected])
        projection = _distribution(values["projection_error_mps"][selected])
        local = _distribution(values["local_error_mps"][selected])
        result["groups"][group] = {
            "full_count": int(group_mask.sum()),
            "expected_full_count": int(expected_counts[group]["full_count"]),
            "reference_common_count": int(selected.sum()),
            "expected_min_reference_common_count": int(
                expected_counts[group]["min_reference_common_count"]
            ),
            "local_common_count": int(local_common.sum()),
            "local_coverage_fraction": float(
                local_common.sum().double() / selected.sum()
            ),
            "parent_error_mps": parent,
            "projection_error_mps": projection,
            "local_error_mps": local,
            "recovery_mean": _recovery(parent, projection, local, "mean"),
            "recovery_p50": _recovery(parent, projection, local, "p50"),
        }
    return result


def _load_fold_sidecars(
    output: Path,
    expected: dict[str, dict[str, Any]],
    completed: dict[str, dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    folds: list[list[dict[str, Any]]] = [[] for _ in range(FIXED_FOLDS)]
    if set(completed) != set(expected):
        raise ValueError("oracle sidecar set is incomplete")
    for shard_id in sorted(expected):
        payload = torch.load(
            completed[shard_id]["path"], map_location="cpu", weights_only=False,
        )
        folds[int(expected[shard_id]["fold"])].append(payload)
    return folds


def _validate_recoverable_final_result(
    output: Path,
    result: dict[str, Any],
    contract: dict[str, Any],
    expected: dict[str, dict[str, Any]],
    completed: dict[str, dict[str, Any]],
) -> tuple[str, bool, int]:
    """Validate a result written just before its final commit-pointer update."""
    contract_sha = _json_sha256(contract)
    status = result.get("status")
    authorized = result.get("authorized_train_local_precision")
    if (
        result.get("schema_version") != RUN_SCHEMA
        or status not in {"passed", "failed"}
        or not isinstance(authorized, bool)
        or result.get("diagnostic_only") is not True
        or result.get("train_only") is not True
        or result.get("network_trained") is not False
        or result.get("validation_accessed") is not False
        or result.get("test_accessed") is not False
        or result.get("future_modules_loaded") is not False
        or result.get("plots_created") is not False
        or result.get("authorized_validation") is not False
        or result.get("authorized_formal_two_stage") is not False
        or result.get("experiment_contract") != contract
        or result.get("experiment_contract_sha256") != contract_sha
        or result.get("completed_shards") != completed
        or set(completed) != set(expected)
    ):
        raise ValueError("recoverable final oracle result scope differs")
    generation = result.get("oracle_checkpoint_generation")
    if (
        not isinstance(generation, int)
        or generation <= 0
        or result.get("oracle_checkpoint")
        != str(_checkpoint_path(output, generation).resolve())
    ):
        raise ValueError("recoverable final oracle checkpoint target differs")
    expected_counts = contract.get("expected_reference_counts")
    cv = result.get("cv")
    if (
        not isinstance(expected_counts, list)
        or len(expected_counts) != FIXED_FOLDS
        or not isinstance(cv, list)
        or len(cv) != FIXED_FOLDS
    ):
        raise ValueError("recoverable final oracle fold metadata differs")
    fold_payloads = _load_fold_sidecars(output, expected, completed)
    for fold, payloads in enumerate(fold_payloads):
        metrics = _summarize_fold(payloads, expected_counts[fold])
        gates = _fold_gates(metrics)
        item = cv[fold]
        if (
            item.get("fold") != fold
            or item.get("heldout_count") != metrics["sample_count"]
            or item.get("metrics") != metrics
            or item.get("gates") != gates
            or item.get("passed") is not _all_boolean_gates_pass(gates)
        ):
            raise ValueError("recoverable final oracle fold differs")
    recomputed_authorized = _authorized(cv)
    if (
        authorized is not recomputed_authorized
        or (status == "passed") is not recomputed_authorized
    ):
        raise ValueError("recoverable final oracle authorization differs")
    return status, authorized, generation


def _converge_recoverable_final_result(
    output: Path,
    contract: dict[str, Any],
    expected: dict[str, dict[str, Any]],
    completed: dict[str, dict[str, Any]],
    device: torch.device,
    *,
    rng_source: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    result_path = output / "screen_result.json"
    if not result_path.exists():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    status, authorized, generation = _validate_recoverable_final_result(
        output, result, contract, expected, completed,
    )
    _write_checkpoint_and_state(
        output, _json_sha256(contract), completed, device,
        status=status, result_sha=sha256_file(result_path),
        authorized=authorized, generation=generation,
        rng_source=rng_source,
    )
    return result


def validate_local_precision_oracle_artifacts(
    output: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed on a completed train-only local-precision oracle screen."""
    root = Path(output).resolve()
    result_path = root / "screen_result.json"
    state_path = root / "run_state.json"
    contract_path = root / CONTRACT_NAME
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        run_state = json.loads(state_path.read_text(encoding="utf-8"))
        stored_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise ValueError("local-precision oracle artifacts are incomplete") from exc
    if run_state.get("status") == "running":
        expected_for_recovery = stored_contract.get("expected_shards")
        completed_for_recovery = result.get("completed_shards")
        if (
            not isinstance(expected_for_recovery, dict)
            or not isinstance(completed_for_recovery, dict)
        ):
            raise ValueError("recoverable final oracle manifests are missing")
        _strict_completed_shards(
            root, completed_for_recovery, expected_for_recovery,
            _json_sha256(stored_contract),
        )
        recovery_checkpoint_path = Path(
            run_state.get("oracle_checkpoint", "")
        ).resolve()
        recovery_generation = run_state.get("generation")
        if (
            not isinstance(recovery_generation, int)
            or recovery_generation <= 0
            or recovery_checkpoint_path
            != _checkpoint_path(root, recovery_generation).resolve()
            or not recovery_checkpoint_path.is_file()
        ):
            raise ValueError("recoverable final oracle commit pointer differs")
        recovery_checkpoint = torch.load(
            recovery_checkpoint_path, map_location="cpu", weights_only=False,
        )
        _converge_recoverable_final_result(
            root, stored_contract, expected_for_recovery,
            completed_for_recovery, torch.device("cpu"),
            rng_source=recovery_checkpoint,
        )
        run_state = json.loads(state_path.read_text(encoding="utf-8"))
    generation = run_state.get("generation")
    checkpoint_path = Path(run_state.get("oracle_checkpoint", "")).resolve()
    if (
        not isinstance(generation, int)
        or generation <= 0
        or checkpoint_path != _checkpoint_path(root, generation).resolve()
    ):
        raise ValueError("local-precision oracle commit pointer differs")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False,
        )
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("local-precision oracle checkpoint is missing") from exc
    status = result.get("status")
    authorized = result.get("authorized_train_local_precision")
    if (
        result.get("schema_version") != RUN_SCHEMA
        or run_state.get("schema_version") != RUN_SCHEMA
        or checkpoint.get("schema_version") != RUN_SCHEMA
        or run_state.get("generation") != generation
        or checkpoint.get("generation") != generation
        or status not in {"passed", "failed"}
        or run_state.get("status") != status
        or checkpoint.get("status") != status
        or result.get("diagnostic_only") is not True
        or result.get("train_only") is not True
        or result.get("network_trained") is not False
        or result.get("truth_omega_feature_construction") is not True
        or result.get("truth_velocity_loss_only") is not True
        or result.get("validation_accessed") is not False
        or result.get("test_accessed") is not False
        or result.get("future_modules_loaded") is not False
        or result.get("plots_created") is not False
        or result.get("authorized_validation") is not False
        or result.get("authorized_formal_two_stage") is not False
        or run_state.get("validation_accessed") is not False
        or run_state.get("test_accessed") is not False
        or run_state.get("future_modules_loaded") is not False
        or not isinstance(authorized, bool)
        or run_state.get("authorized_train_local_precision") is not authorized
    ):
        raise ValueError("local-precision oracle status/scope differs")
    contract = result.get("experiment_contract")
    if not isinstance(contract, dict):
        raise ValueError("local-precision oracle contract is missing")
    required_contract_fields = {
        "v93_result_path", "dataset_manifest_path", "truth_manifest_path",
        "parent_checkpoint_path", "frozen_checkpoint_paths",
        "expected_shards", "fold_sample_indices_sha256", "source_sha256",
        "expected_reference_counts",
    }
    if not required_contract_fields.issubset(contract):
        raise ValueError("local-precision oracle contract fields are missing")
    contract_sha = _json_sha256(contract)
    if (
        stored_contract != contract
        or result.get("experiment_contract_sha256") != contract_sha
        or run_state.get("experiment_contract_sha256") != contract_sha
        or checkpoint.get("experiment_contract_sha256") != contract_sha
        or contract.get("schema_version") != RUN_SCHEMA
        or contract.get("diagnostic_only") is not True
        or contract.get("train_only") is not True
        or contract.get("truth_omega_feature_construction") is not True
        or contract.get("truth_velocity_loss_only") is not True
        or contract.get("network_trained") is not False
        or contract.get("v93_result_sha256") != V93_RESULT_SHA256
        or contract.get("seed") != FIXED_SEED
        or contract.get("folds") != FIXED_FOLDS
        or contract.get("oracle_steps") != FIXED_ORACLE_STEPS
        or contract.get("oracle_starts") != 3
        or contract.get("oracle_chunk_size") != FIXED_ORACLE_CHUNK_SIZE
        or contract.get("recovery_gate") != RECOVERY_GATE
        or contract.get("minimum_parent_projection_headroom_mps")
        != MIN_PARENT_PROJECTION_HEADROOM_MPS
        or contract.get("minimum_parent_local_improvement_mps")
        != MIN_PARENT_LOCAL_IMPROVEMENT_MPS
        or contract.get("anchor_availability")
        != "center-supported-expanded-four-roles"
        or contract.get("forward_cache_contains_truth") is not False
        or contract.get("forward_excludes") != [
            "session", "motion_class", "physical_armor_id",
            "truth_velocity", "future", "absolute_pose",
        ]
    ):
        raise ValueError("local-precision oracle fixed contract differs")
    if {
        name: sha256_file(path) for name, path in _source_paths().items()
    } != contract.get("source_sha256"):
        raise ValueError("local-precision oracle source binding differs")
    v93_path = Path(contract["v93_result_path"]).resolve()
    _, a2_result, validated_v93_path = validate_v93_failure(v93_path.parent)
    expected_reference_counts = _a2_expected_reference_counts(a2_result)
    if (
        v93_path != validated_v93_path
        or result.get("v93") != {
            "result": str(v93_path), "sha256": V93_RESULT_SHA256,
        }
        or contract.get("v93_contract_sha256")
        != a2_result.get("experiment_contract_sha256")
        or contract.get("expected_reference_counts")
        != expected_reference_counts
    ):
        raise ValueError("local-precision oracle V93 binding differs")
    for path_key, sha_key in (
        ("dataset_manifest_path", "dataset_manifest_sha256"),
        ("truth_manifest_path", "truth_manifest_sha256"),
    ):
        path = Path(contract[path_key]).resolve()
        if sha256_file(path) != contract.get(sha_key):
            raise ValueError(f"local-precision oracle {path_key} differs")
    dataset = _load_train_dataset(
        Path(contract["dataset_manifest_path"]).resolve().parent
    )
    dataset_manifest_path = Path(contract["dataset_manifest_path"]).resolve()
    dataset_path = dataset_manifest_path.parent
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    assert_manifest_split_shards_unchanged(
        dataset_path, dataset_manifest, split="train",
        label="local-precision state-space oracle dataset validation",
    )
    truth_path = Path(contract["truth_manifest_path"]).resolve().parent
    truth_index = SplitScopedTruthIndex(
        truth_path, split="train",
        expected_manifest_sha256=contract["truth_manifest_sha256"],
    )
    truth_index.attach(dataset)
    truth_index.assert_unchanged()
    _, sessions = _combined_metadata(dataset)
    fold_sessions = _fold_assignments(sessions, folds=FIXED_FOLDS)
    fold_indices = [
        np.asarray([
            index for index, session in enumerate(sessions)
            if session in heldout_sessions
        ], dtype=np.int64)
        for heldout_sessions in fold_sessions
    ]
    recomputed_expected = _expected_shards(
        fold_indices, chunk_size=FIXED_ORACLE_CHUNK_SIZE,
    )
    if (
        contract.get("expected_shards") != recomputed_expected
        or contract.get("fold_sample_indices_sha256") != [
            _indices_sha256(indices) for indices in fold_indices
        ]
    ):
        raise ValueError("local-precision oracle fold/shard plan differs")
    parent_path = Path(contract["parent_checkpoint_path"]).resolve()
    parent_root = parent_path.parent.parent
    if (
        sha256_file(parent_path) != contract.get("parent_checkpoint_sha256")
        or sha256_file(parent_root / "run_manifest.json")
        != contract.get("parent_manifest_sha256")
        or sha256_file(parent_root / "screen_result.json")
        != contract.get("parent_result_sha256")
    ):
        raise ValueError("local-precision oracle parent files differ")
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    if state_dict_sha256(parent_payload["model"]) != contract.get(
        "parent_model_state_dict_sha256"
    ):
        raise ValueError("local-precision oracle parent model differs")
    frozen_paths = {
        name: Path(path).resolve()
        for name, path in contract["frozen_checkpoint_paths"].items()
    }
    if {
        name: sha256_file(path) for name, path in frozen_paths.items()
    } != contract.get("frozen_checkpoint_sha256"):
        raise ValueError("local-precision oracle frozen files differ")
    mapper, _ = load_frozen_pnp_mapper(frozen_paths["mapper"])
    s_model, _ = load_frozen_v19(frozen_paths["s"])
    h_model, _ = load_frozen_hypothesis_adapter(
        frozen_paths["h"], allow_diagnostic=True,
    )
    if {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    } != contract.get("frozen_state_dict_sha256"):
        raise ValueError("local-precision oracle frozen states differ")
    expected = contract.get("expected_shards")
    completed = checkpoint.get("completed_shards")
    if (
        not isinstance(expected, dict)
        or not expected
        or not isinstance(completed, dict)
        or set(completed) != set(expected)
        or run_state.get("completed_shards") != completed
        or result.get("completed_shards") != completed
        or result.get("oracle_checkpoint_generation") != generation
        or result.get("oracle_checkpoint") != str(checkpoint_path.resolve())
        or run_state.get("oracle_checkpoint") != str(checkpoint_path.resolve())
        or run_state.get("oracle_checkpoint_sha256") != sha256_file(checkpoint_path)
        or run_state.get("screen_result_sha256") != sha256_file(result_path)
    ):
        raise ValueError("local-precision oracle artifact hashes differ")
    cpu_rng = checkpoint.get("cpu_rng_state")
    cuda_rng = checkpoint.get("cuda_rng_state_all")
    if (
        not isinstance(cpu_rng, torch.Tensor)
        or not isinstance(cuda_rng, list)
        or run_state.get("cpu_rng_state_sha256") != _tensor_sha256(cpu_rng)
        or run_state.get("cuda_rng_state_sha256")
        != [_tensor_sha256(value) for value in cuda_rng]
    ):
        raise ValueError("local-precision oracle RNG binding differs")
    _strict_completed_shards(root, completed, expected, contract_sha)
    actual_sidecars = {
        path.name for path in (root / SIDECAR_DIRECTORY).glob("*.pt")
    }
    if actual_sidecars != {f"{shard_id}.pt" for shard_id in expected}:
        raise ValueError("local-precision oracle sidecar set differs")
    fold_payloads = _load_fold_sidecars(root, expected, completed)
    cv = result.get("cv")
    if not isinstance(cv, list) or len(cv) != FIXED_FOLDS:
        raise ValueError("local-precision oracle fold count differs")
    for fold, payloads in enumerate(fold_payloads):
        metrics = _summarize_fold(
            payloads, expected_reference_counts[fold],
        )
        gates = _fold_gates(metrics)
        item = cv[fold]
        if (
            item.get("fold") != fold
            or item.get("heldout_count") != metrics["sample_count"]
            or item.get("heldout_sessions") != sorted(fold_sessions[fold])
            or item.get("metrics") != metrics
            or item.get("gates") != gates
            or item.get("passed") is not _all_boolean_gates_pass(gates)
        ):
            raise ValueError("local-precision oracle recomputed fold differs")
    recomputed_authorized = _authorized(cv)
    if (
        authorized is not recomputed_authorized
        or (status == "passed") is not recomputed_authorized
    ):
        raise ValueError("local-precision oracle authorization differs")
    return checkpoint, result


def _evaluate_chunk(
    *,
    base: CenterPriorProfiledTwistScreen,
    weighted: LocallyWeightedProfiledTwistAtOmega,
    prepared: dict[str, torch.Tensor],
    sample_indices: np.ndarray,
    device: torch.device,
    expected: dict[str, Any],
    contract_sha: str,
) -> tuple[dict[str, Any], int]:
    index = torch.from_numpy(sample_indices).long()
    state = {
        name: prepared[name][index].to(device, non_blocking=True)
        for name in CENTER_TWIST_FORWARD_FIELDS
    }
    target = prepared[MOTION_TARGET_FIELD][index].to(device, non_blocking=True)
    omega = target[:, 3]
    old = base.forward_at_omega(state, omega)
    center = base.estimate_center(state["q0_relation_m"], state["q0_supported"])
    prior = {**center, "q0_relation_m": state["q0_relation_m"]}
    cache = _forward_cache(state, omega, prior)
    boundary_count = _verify_fp32_boundaries(weighted, base, cache)
    profile_args = (
        state["history_obs_rel_m"], state["history_obs_mask"],
        state["history_event_mask"], state["history_time_s"], omega, prior,
    )
    q0 = base.profile(*profile_args, use_q0_prior=True)
    history = base.profile(*profile_args, use_q0_prior=False)
    projection = oracle_projection_coefficient(
        target[:, :3], q0["velocity_mps"], history["velocity_mps"],
        q0["profile_supported"], history["profile_supported"],
    )
    projection_velocity = (
        projection["weight"].unsqueeze(-1) * q0["velocity_mps"]
        + (1.0 - projection["weight"]).unsqueeze(-1)
        * history["velocity_mps"]
    )
    local = _optimize_chunk(weighted, cache, target[:, :3])
    payload = _sidecar_payload(
        expected=expected, contract_sha=contract_sha,
        sample_indices=sample_indices,
        motion_class=prepared["motion_class"][index],
        parent_velocity=old["velocity_mps"],
        projection_velocity=projection_velocity,
        projection_valid=projection["oracle_valid"], local=local,
        target_velocity=target[:, :3],
        parent_supported=old["state_supported"],
    )
    return payload, boundary_count


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.seed != FIXED_SEED
        or args.oracle_steps != FIXED_ORACLE_STEPS
        or args.oracle_chunk_size != FIXED_ORACLE_CHUNK_SIZE
    ):
        raise ValueError("state-space oracle budget is fixed")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("state-space oracle requires CUDA")
    git_start = _git_state()
    if git_start["git_commit"] == "unknown" or git_start["worktree_dirty"]:
        raise RuntimeError("state-space oracle requires a clean known git commit")
    _, a2, a2_result_path = validate_v93_failure(args.a2_result_dir)
    expected_reference_counts = _a2_expected_reference_counts(a2)
    a2_contract = a2["experiment_contract"]
    _seed(args.seed)
    dataset_manifest_path = Path(a2_contract["dataset_manifest_path"]).resolve()
    truth_manifest_path = Path(a2_contract["truth_manifest_path"]).resolve()
    dataset_path = dataset_manifest_path.parent
    truth_path = truth_manifest_path.parent
    dataset_manifest = json.loads(
        dataset_manifest_path.read_text(encoding="utf-8")
    )
    train_dataset = _load_train_dataset(dataset_path)
    train_truth = SplitScopedTruthIndex(
        truth_path, split="train",
        expected_manifest_sha256=a2_contract["truth_manifest_sha256"],
    )
    train_truth.attach(train_dataset)
    frozen_paths = {
        name: Path(value).resolve()
        for name, value in a2_contract["frozen_checkpoint_paths"].items()
    }
    mapper, _ = load_frozen_pnp_mapper(frozen_paths["mapper"])
    s_model, _ = load_frozen_v19(frozen_paths["s"])
    h_model, _ = load_frozen_hypothesis_adapter(
        frozen_paths["h"], allow_diagnostic=True,
    )
    for frozen in (mapper, s_model, h_model):
        frozen.to(device).eval().requires_grad_(False)
    frozen_files = {name: sha256_file(path) for name, path in frozen_paths.items()}
    frozen_states = {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    }
    if (
        frozen_files != a2_contract["frozen_checkpoint_sha256"]
        or frozen_states != a2_contract["frozen_state_dict_sha256"]
    ):
        raise ValueError("state-space oracle frozen dependencies differ")
    parent_path = Path(a2["parent"]["checkpoint"]).resolve()
    parent_root = parent_path.parent.parent
    parent_manifest_path = parent_root / "run_manifest.json"
    parent_result_path = parent_root / "screen_result.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent_sha = sha256_file(parent_path)
    parent_contract = _validate_checkpoint_manifest_identity(
        parent_payload, parent_manifest, parent_sha,
    )
    parent_args = parent_contract["args"]
    base = CenterPriorProfiledTwistScreen(
        width=int(parent_args["width"]),
        center_precision=float(parent_args["center_precision"]),
    ).to(device)
    base.load_state_dict(parent_payload["model"], strict=True)
    base.eval().requires_grad_(False)
    base_state_sha = state_dict_sha256(base.state_dict())
    if base_state_sha != a2_contract["parent_model_state_dict_sha256"]:
        raise ValueError("state-space oracle V14 parent state differs")
    weighted = _weighted_like(base).to(device).eval()
    prepared = _cache_prepared_state(
        train_dataset, mapper, s_model, h_model, device,
        batch_size=int(a2_contract["expert_batch_size"]),
    )
    _, sessions = _combined_metadata(train_dataset)
    fold_sessions = _fold_assignments(sessions, folds=FIXED_FOLDS)
    fold_indices: list[np.ndarray] = []
    for fold, heldout_sessions in enumerate(fold_sessions):
        if sorted(heldout_sessions) != sorted(a2["cv"][fold]["heldout_sessions"]):
            raise ValueError("state-space oracle fold differs from A2")
        fold_indices.append(np.asarray([
            index for index, session in enumerate(sessions)
            if session in heldout_sessions
        ], dtype=np.int64))
    expected = _expected_shards(
        fold_indices, chunk_size=args.oracle_chunk_size,
    )
    source_paths = _source_paths()
    source_sha = {name: sha256_file(path) for name, path in source_paths.items()}
    contract = {
        "schema_version": RUN_SCHEMA,
        "diagnostic_only": True,
        "train_only": True,
        "truth_omega_feature_construction": True,
        "truth_velocity_loss_only": True,
        "network_trained": False,
        "v93_result_path": str(a2_result_path),
        "v93_result_sha256": V93_RESULT_SHA256,
        "v93_contract_sha256": a2["experiment_contract_sha256"],
        "parent_checkpoint_path": str(parent_path),
        "parent_checkpoint_sha256": parent_sha,
        "parent_model_state_dict_sha256": base_state_sha,
        "parent_manifest_sha256": sha256_file(parent_manifest_path),
        "parent_result_sha256": sha256_file(parent_result_path),
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "truth_manifest_path": str(truth_manifest_path),
        "truth_manifest_sha256": sha256_file(truth_manifest_path),
        "frozen_checkpoint_paths": {
            name: str(path) for name, path in frozen_paths.items()
        },
        "frozen_checkpoint_sha256": frozen_files,
        "frozen_state_dict_sha256": frozen_states,
        "source_sha256": source_sha,
        "git_commit": git_start["git_commit"],
        "seed": args.seed,
        "folds": FIXED_FOLDS,
        "oracle_steps": FIXED_ORACLE_STEPS,
        "oracle_starts": 3,
        "oracle_chunk_size": FIXED_ORACLE_CHUNK_SIZE,
        "recovery_gate": RECOVERY_GATE,
        "minimum_parent_projection_headroom_mps": (
            MIN_PARENT_PROJECTION_HEADROOM_MPS
        ),
        "minimum_parent_local_improvement_mps": (
            MIN_PARENT_LOCAL_IMPROVEMENT_MPS
        ),
        "anchor_availability": "center-supported-expanded-four-roles",
        "forward_cache_contains_truth": False,
        "forward_excludes": [
            "session", "motion_class", "physical_armor_id",
            "truth_velocity", "future", "absolute_pose",
        ],
        "fold_sample_indices_sha256": [
            _indices_sha256(indices) for indices in fold_indices
        ],
        "expected_shards": expected,
        "expected_reference_counts": expected_reference_counts,
    }
    contract_sha = _json_sha256(contract)
    output = Path(args.output).resolve()
    completed, resume = _initialize_or_resume(
        output, contract, expected, device,
    )
    if isinstance(resume, dict) and resume.get("status") in {"passed", "failed"}:
        return resume
    boundary_count = 0
    for fold, indices in enumerate(fold_indices):
        for start in range(0, len(indices), args.oracle_chunk_size):
            stop = min(start + args.oracle_chunk_size, len(indices))
            shard_id = f"fold{fold:02d}-{start:06d}-{stop:06d}"
            if shard_id in completed:
                continue
            selected = indices[start:stop]
            payload, checked = _evaluate_chunk(
                base=base, weighted=weighted, prepared=prepared,
                sample_indices=selected, device=device,
                expected=expected[shard_id], contract_sha=contract_sha,
            )
            boundary_count += checked
            _commit_shard(
                output, contract_sha, expected[shard_id], payload,
                completed, device,
            )
    fold_payloads = _load_fold_sidecars(output, expected, completed)
    cv: list[dict[str, Any]] = []
    for fold, payloads in enumerate(fold_payloads):
        metrics = _summarize_fold(
            payloads, expected_reference_counts[fold],
        )
        gates = _fold_gates(metrics)
        passed = _all_boolean_gates_pass(gates)
        cv.append({
            "fold": fold, "passed": passed,
            "heldout_sessions": sorted(fold_sessions[fold]),
            "heldout_count": int(fold_indices[fold].size),
            "metrics": metrics, "gates": gates,
        })
    authorized = _authorized(cv)
    if state_dict_sha256(base.state_dict()) != base_state_sha:
        raise RuntimeError("state-space oracle changed frozen V14")
    if {
        "mapper": state_dict_sha256(mapper.state_dict()),
        "s": state_dict_sha256(s_model.state_dict()),
        "h": state_dict_sha256(h_model.state_dict()),
    } != frozen_states:
        raise RuntimeError("state-space oracle changed frozen Mapper/S/H")
    if {name: sha256_file(path) for name, path in source_paths.items()} != source_sha:
        raise RuntimeError("state-space oracle source changed during run")
    if sha256_file(dataset_manifest_path) != contract["dataset_manifest_sha256"]:
        raise RuntimeError("state-space oracle dataset manifest changed")
    assert_manifest_split_shards_unchanged(
        dataset_path, dataset_manifest, split="train",
        label="local-precision state-space oracle dataset",
    )
    train_truth.assert_unchanged()
    if _git_state() != git_start:
        raise RuntimeError("state-space oracle git state changed during run")
    final_generation = _next_checkpoint_generation(output)
    final_checkpoint_path = _checkpoint_path(output, final_generation).resolve()
    result = {
        "schema_version": RUN_SCHEMA,
        "status": "passed" if authorized else "failed",
        "diagnostic_only": True,
        "train_only": True,
        "network_trained": False,
        "truth_omega_feature_construction": True,
        "truth_velocity_loss_only": True,
        "validation_accessed": False,
        "test_accessed": False,
        "future_modules_loaded": False,
        "plots_created": False,
        "authorized_train_local_precision": authorized,
        "authorized_validation": False,
        "authorized_formal_two_stage": False,
        "experiment_contract": contract,
        "experiment_contract_sha256": contract_sha,
        "v93": {"result": str(a2_result_path), "sha256": V93_RESULT_SHA256},
        "cv": cv,
        "completed_shards": completed,
        "fp32_boundary_samples_checked_this_process": boundary_count,
        "oracle_checkpoint_generation": final_generation,
        "oracle_checkpoint": str(final_checkpoint_path),
    }
    result_path = output / "screen_result.json"
    if result_path.exists():
        raise FileExistsError("refusing to overwrite state-space oracle result")
    _atomic_json(result_path, result)
    _write_checkpoint_and_state(
        output, contract_sha, completed, device,
        status=result["status"], result_sha=sha256_file(result_path),
        authorized=authorized, generation=final_generation,
    )
    _, validated_result = validate_local_precision_oracle_artifacts(output)
    return validated_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a2-result-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    parser.add_argument("--oracle-steps", type=int, default=FIXED_ORACLE_STEPS)
    parser.add_argument(
        "--oracle-chunk-size", type=int, default=FIXED_ORACLE_CHUNK_SIZE,
    )
    args = parser.parse_args()
    if (
        args.seed != FIXED_SEED
        or args.oracle_steps != FIXED_ORACLE_STEPS
        or args.oracle_chunk_size != FIXED_ORACLE_CHUNK_SIZE
    ):
        raise ValueError("state-space oracle budget is fixed")
    return args


def main() -> None:
    result = run(parse_args())
    print(json.dumps({
        "status": result["status"],
        "authorized_train_local_precision": result[
            "authorized_train_local_precision"
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
