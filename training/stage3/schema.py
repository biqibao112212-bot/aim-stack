"""Typed schema helpers for the stage-three estimator.

The capture side is intentionally independent from the legacy telemetry JSONL.
This module accepts the canonical JSON projection used by the first recorder
implementation and validates that it contains only pre-tracker observations.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping
import hashlib
import json
import math

SCHEMA_VERSION = "stage3-observation-v2"
LEGACY_OBSERVATION_SCHEMA_VERSION = "stage3-observation-v1"
OBSERVATION_SCHEMA_VERSIONS = {LEGACY_OBSERVATION_SCHEMA_VERSION, SCHEMA_VERSION}
TRUTH_SCHEMA_VERSION = "stage3-truth-v1"
MANIFEST_SCHEMA_VERSION = "stage3-manifest-v1"
MAX_ARMORS = 4


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nullable_finite(value: Any, name: str) -> float:
    """Keep invalid/raw recorder values representable without inventing data."""
    if value is None:
        return float("nan")
    result = float(value)
    return result if math.isfinite(result) else float("nan")


@dataclass(frozen=True)
class ArmorObservation:
    observation_index: int
    position_m: tuple[float, float, float]
    yaw_rad: float
    reprojection_rms_px: float | None
    camera_tvec_m: tuple[float, float, float] | None = None
    valid: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArmorObservation":
        position = tuple(_nullable_finite(v, "position_m") for v in value["position_m"])
        if len(position) != 3:
            raise ValueError("position_m must have three values")
        camera_tvec_raw = value.get("camera_tvec_m")
        camera_tvec = (
            None if camera_tvec_raw is None else
            tuple(_nullable_finite(v, "camera_tvec_m") for v in camera_tvec_raw)
        )
        if camera_tvec is not None and len(camera_tvec) != 3:
            raise ValueError("camera_tvec_m must have three values")
        return cls(
            observation_index=int(value.get("observation_index", 0)),
            position_m=position,
            yaw_rad=_nullable_finite(value.get("yaw_rad"), "yaw_rad"),
            reprojection_rms_px=(
                None if value.get("reprojection_rms_px") is None else
                _nullable_finite(value.get("reprojection_rms_px"), "reprojection_rms_px")
            ),
            camera_tvec_m=camera_tvec,
            valid=bool(value.get("valid", True)),
        )


@dataclass(frozen=True)
class ObservationFrame:
    session_id: str
    producer_epoch: int
    frame_seq: int
    timestamp_ns: int
    armors: tuple[ArmorObservation, ...]
    tracker_origin_world_m: tuple[float, float, float] | None = None
    tracker_quaternion_world_wxyz: tuple[float, float, float, float] | None = None
    camera_origin_world_m: tuple[float, float, float] | None = None
    camera_quaternion_world_wxyz: tuple[float, float, float, float] | None = None
    gimbal_yaw_deg: float = 0.0
    gimbal_pitch_deg: float = 0.0
    position_contract: str = ""
    camera_gimbal_extrinsic_from_config: bool = False
    R_camera2gimbal: tuple[float, ...] | None = None
    t_camera2gimbal_m: tuple[float, float, float] | None = None
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ObservationFrame":
        version = str(value.get("schema_version", SCHEMA_VERSION))
        if version not in OBSERVATION_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported observation schema: {version}")
        armors = tuple(ArmorObservation.from_mapping(v) for v in value.get("armors", ()))
        session_id = str(value["session_id"])
        if not session_id:
            raise ValueError("observation session_id is empty")
        timestamp_ns = int(value["timestamp_ns"])
        if timestamp_ns <= 0:
            raise ValueError("observation timestamp_ns must be positive")
        position_contract = str(value.get("position_contract", ""))
        R_camera2gimbal = _optional_tuple(value.get("R_camera2gimbal"), 9)
        t_camera2gimbal_m = _optional_tuple(value.get("t_camera2gimbal_m"), 3)
        extrinsic_from_config = bool(value.get("camera_gimbal_extrinsic_from_config", False))
        if version == SCHEMA_VERSION:
            if position_contract != "calibrated-camera-gimbal-extrinsic-v1":
                raise ValueError(f"unsupported position contract: {position_contract}")
            if not extrinsic_from_config or R_camera2gimbal is None or t_camera2gimbal_m is None:
                raise ValueError("v2 observation is missing its calibrated R/T audit record")
        return cls(
            session_id=session_id,
            producer_epoch=int(value["producer_epoch"]),
            frame_seq=int(value["frame_seq"]),
            timestamp_ns=timestamp_ns,
            armors=armors,
            tracker_origin_world_m=_optional_tuple(value.get("tracker_origin_world_ros_m"), 3),
            tracker_quaternion_world_wxyz=_optional_tuple(
                value.get("tracker_gimbal_quaternion_world_wxyz"), 4
            ),
            camera_origin_world_m=_optional_tuple(value.get("camera_origin_world_ros_m"), 3),
            camera_quaternion_world_wxyz=_optional_tuple(
                value.get("camera_quaternion_world_wxyz"), 4
            ),
            gimbal_yaw_deg=_finite(value.get("gimbal_yaw_deg", 0.0), "gimbal_yaw_deg"),
            gimbal_pitch_deg=_finite(value.get("gimbal_pitch_deg", 0.0), "gimbal_pitch_deg"),
            position_contract=position_contract,
            camera_gimbal_extrinsic_from_config=extrinsic_from_config,
            R_camera2gimbal=R_camera2gimbal,
            t_camera2gimbal_m=t_camera2gimbal_m,  # type: ignore[arg-type]
            schema_version=version,
        )

    @property
    def key(self) -> tuple[str, int, int, int]:
        return (self.session_id, self.producer_epoch, self.frame_seq, self.timestamp_ns)


@dataclass(frozen=True)
class TruthArmor:
    relative_slot: int
    position_world_m: tuple[float, float, float]
    outward_normal_world: tuple[float, float, float]
    relative_yaw_rad: float = 0.0


def _optional_tuple(value: Any, size: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    values = tuple(_finite(v, "tuple") for v in value)
    if len(values) != size:
        raise ValueError(f"expected {size} values")
    return values


def _quat_rotate(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = q
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return (
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    )


def _normalise_quat(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm < 1e-9:
        raise ValueError("quaternion norm is zero")
    return tuple(v / norm for v in values)  # type: ignore[return-value]


@dataclass(frozen=True)
class TruthFrame:
    session_id: str
    producer_epoch: int
    frame_seq: int
    timestamp_ns: int
    chassis_origin_world_m: tuple[float, float, float]
    chassis_quaternion_world_wxyz: tuple[float, float, float, float]
    gimbal_origin_world_m: tuple[float, float, float]
    gimbal_quaternion_world_wxyz: tuple[float, float, float, float]
    target_id: int
    velocity_world_mps: tuple[float, float, float]
    yaw_rad: float
    yaw_rate_rad_s: float
    armors: tuple[TruthArmor, ...]
    radius_even_m: float
    radius_odd_m: float
    armor_height_m: float
    camera_origin_world_m: tuple[float, float, float]
    camera_quaternion_world_wxyz: tuple[float, float, float, float]
    exposure_state_flags: int = 0
    geometry_hash: str | None = None
    schema_version: str = TRUTH_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TruthFrame":
        version = str(value.get("schema_version", TRUTH_SCHEMA_VERSION))
        if version != TRUTH_SCHEMA_VERSION:
            raise ValueError(f"unsupported truth schema: {version}")
        if not bool(value.get("has_exact_exposure_truth", False)):
            raise ValueError("truth frame does not contain exact exposure truth")
        gt = value.get("ground_truth")
        if not isinstance(gt, Mapping):
            raise ValueError("truth frame ground_truth is missing")
        targets = gt.get("targets", ())
        selected_id = int(value.get("selected_target_id", 0))
        selected_index = int(value.get("selected_target_index", -1))
        target: Mapping[str, Any] | None = None
        if 0 <= selected_index < len(targets):
            candidate = targets[selected_index]
            if isinstance(candidate, Mapping):
                if int(candidate.get("target_id", -1)) != selected_id:
                    raise ValueError("selected_target_index and selected_target_id disagree")
                target = candidate
        if target is None:
            for candidate in targets:
                if isinstance(candidate, Mapping) and int(candidate.get("target_id", -1)) == selected_id:
                    target = candidate
                    break
        if target is None:
            raise ValueError("selected target is not present in ground truth")
        if int(target.get("state_flags", 0)) & 0x7 != 0x7:
            raise ValueError("selected target is missing world state/orientation/geometry")
        exposure = value.get("exposure_state")
        if not isinstance(exposure, Mapping):
            raise ValueError("exposure_state is missing")
        exposure_flags = int(value.get("exposure_state_flags", 0))
        if exposure_flags & 0x7 != 0x7:
            raise ValueError("exact truth is missing chassis/gimbal/camera exposure pose")
        camera_origin = _optional_tuple(exposure.get("camera_position_world_m"), 3)
        camera_quat = _optional_tuple(exposure.get("camera_quaternion_world_wxyz"), 4)
        if camera_origin is None or camera_quat is None:
            raise ValueError("camera exposure transform is missing")
        camera_quat = _normalise_quat(camera_quat)  # type: ignore[assignment]
        target_pos = tuple(_finite(v, "target.position_m") for v in target["world_position_m"])
        if len(target_pos) != 3:
            raise ValueError("target world_position_m must have three values")
        target_quat = _normalise_quat(tuple(_finite(v, "target.quaternion") for v in target["world_quaternion_wxyz"]))
        armors = []
        for raw in target.get("armors", ()):
            if not isinstance(raw, Mapping):
                continue
            rel = tuple(_finite(v, "relative_position_m") for v in raw["relative_position_m"])
            normal_rel = tuple(_finite(v, "outward_normal") for v in raw["outward_normal"])
            position_world = tuple(a + b for a, b in zip(target_pos, _quat_rotate(target_quat, rel)))
            normal_world = _quat_rotate(target_quat, normal_rel)
            armors.append(TruthArmor(
                relative_slot=int(raw.get("relative_slot", len(armors))),
                position_world_m=position_world,
                outward_normal_world=normal_world,
                relative_yaw_rad=_finite(raw.get("relative_yaw_rad", 0.0), "relative_yaw_rad"),
            ))
        if len(armors) != 4:
            raise ValueError("selected target must expose four armor geometry records")
        if {armor.relative_slot for armor in armors} != {0, 1, 2, 3}:
            raise ValueError("selected target relative_slot values must be unique 0..3")
        chassis_origin = _optional_tuple(exposure.get("chassis_position_world_m"), 3)
        chassis_quaternion = _optional_tuple(exposure.get("chassis_quaternion_world_wxyz"), 4)
        gimbal_origin = _optional_tuple(exposure.get("gimbal_position_world_m"), 3)
        gimbal_quaternion = _optional_tuple(exposure.get("gimbal_quaternion_world_wxyz"), 4)
        if (
            chassis_origin is None or chassis_quaternion is None or
            gimbal_origin is None or gimbal_quaternion is None
        ):
            raise ValueError("exact truth chassis/gimbal exposure transform is missing")
        velocity = tuple(_finite(v, "velocity_mps") for v in target["world_velocity_mps"])
        if len(velocity) != 3:
            raise ValueError("target world_velocity_mps must have three values")
        session_id = str(value["session_id"])
        if not session_id:
            raise ValueError("truth session_id is empty")
        timestamp_ns = int(value["timestamp_ns"])
        if timestamp_ns <= 0:
            raise ValueError("truth timestamp_ns must be positive")
        return cls(
            session_id=session_id,
            producer_epoch=int(value["producer_epoch"]),
            frame_seq=int(value["frame_seq"]),
            timestamp_ns=timestamp_ns,
            chassis_origin_world_m=chassis_origin,
            chassis_quaternion_world_wxyz=_normalise_quat(chassis_quaternion),
            gimbal_origin_world_m=gimbal_origin,
            gimbal_quaternion_world_wxyz=_normalise_quat(gimbal_quaternion),
            target_id=int(target["target_id"]),
            velocity_world_mps=velocity,
            yaw_rad=_finite(target["world_yaw_rad"], "yaw_rad"),
            yaw_rate_rad_s=_finite(target["world_vyaw_rad_s"], "yaw_rate_rad_s"),
            armors=tuple(sorted(armors, key=lambda armor: armor.relative_slot)),
            radius_even_m=_finite(target["radius_even_m"], "radius_even_m"),
            radius_odd_m=_finite(target["radius_odd_m"], "radius_odd_m"),
            armor_height_m=_finite(target["armor_height_m"], "armor_height_m"),
            camera_origin_world_m=camera_origin,
            camera_quaternion_world_wxyz=camera_quat,
            exposure_state_flags=exposure_flags,
            geometry_hash=(None if value.get("geometry_hash") is None else str(value["geometry_hash"])),
        )


@dataclass(frozen=True)
class SessionManifest:
    session_id: str
    target_number: int
    mode: str
    distance_m: float
    initial_yaw_rad: float
    direction_deg: float
    linear_speed_mps: float
    linear_span_m: float
    spin_rad_s: float
    duration_s: float
    seed: int
    schema_version: str = MANIFEST_SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def iter_json_records(path: str | Path) -> Iterator[Mapping[str, Any]]:
    """Stream JSONL records and fail on malformed or empty lines."""

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"record at {path}:{line_number} is not an object")
            yield value


def iter_observation_frames(path: str | Path) -> Iterator[ObservationFrame]:
    for record in iter_json_records(path):
        yield ObservationFrame.from_mapping(record)


def schema_fingerprint() -> str:
    payload = {
        "observation": sorted(OBSERVATION_SCHEMA_VERSIONS),
        "truth": TRUTH_SCHEMA_VERSION,
        "max_armors": MAX_ARMORS,
        "fields": [
            "session_id",
            "producer_epoch",
            "frame_seq",
            "timestamp_ns",
            "position_m",
            "camera_tvec_m",
            "yaw_rad",
            "reprojection_rms_px",
            "valid",
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
