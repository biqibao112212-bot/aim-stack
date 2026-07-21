from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

from training.stage3.build_dataset import discover_canonical_sources, stratified_session_split
from training.stage3.baselines import rigid_constant_velocity_yaw_rate
from training.stage3.analyze_triangle_errors import _match_observation_to_truth
from training.stage3.dataset import CameraGimbalExtrinsic, _make_sample, _world_to_tracker
from training.stage3.losses import stage3_loss, stage3_observation_loss
from training.stage3.model import Stage3TCN
from training.stage3.schema import ArmorObservation, ObservationFrame, TruthArmor, TruthFrame
from training.stage3.train import _position_set_l2


EXTRINSIC = CameraGimbalExtrinsic(
    np.asarray([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
    np.asarray([0.25631080, 0.00183094, 0.09543117]),
)


def _observation(timestamp_ns: int, frame_seq: int, armor_count: int = 1, valid: bool = True) -> ObservationFrame:
    armors = tuple(
        ArmorObservation(
            observation_index=index, position_m=(3.0, 0.1 * index, 0.2),
            yaw_rad=0.1 * index, reprojection_rms_px=0.2, valid=valid,
        )
        for index in range(armor_count)
    )
    return ObservationFrame(
        session_id="s", producer_epoch=1, frame_seq=frame_seq,
        timestamp_ns=timestamp_ns, armors=armors,
        tracker_origin_world_m=(0.0, 0.0, 0.0),
        tracker_quaternion_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        camera_origin_world_m=(0.0, 0.0, 0.0),
        camera_quaternion_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        schema_version="stage3-observation-v1",
    )


def _truth(timestamp_ns: int, frame_seq: int) -> TruthFrame:
    return TruthFrame(
        session_id="s", producer_epoch=1, frame_seq=frame_seq,
        timestamp_ns=timestamp_ns,
        chassis_origin_world_m=(0.0, 0.0, 0.0),
        chassis_quaternion_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        gimbal_origin_world_m=(0.0, 0.0, 0.0),
        gimbal_quaternion_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        target_id=3, velocity_world_mps=(0.0, 0.0, 0.0),
        yaw_rad=0.0, yaw_rate_rad_s=0.0,
        armors=tuple(
            TruthArmor(index, (3.0, 0.1 * index, 0.13), (1.0, 0.0, 0.0))
            for index in range(4)
        ),
        radius_even_m=0.3, radius_odd_m=0.3, armor_height_m=0.0,
        camera_origin_world_m=(0.0, 0.0, 0.0),
        camera_quaternion_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        exposure_state_flags=7, geometry_hash="geometry",
    )


def test_observation_schema_accepts_null_quality_and_extra_raw_candidates() -> None:
    frame = ObservationFrame.from_mapping({
        "schema_version": "stage3-observation-v1",
        "session_id": "s",
        "producer_epoch": 1,
        "frame_seq": 2,
        "timestamp_ns": 3,
        "armors": [
            {"observation_index": index, "position_m": [1, 2, 3], "yaw_rad": 0.1, "reprojection_rms_px": None}
            for index in range(5)
        ],
    })
    assert len(frame.armors) == 5
    assert frame.armors[0].reprojection_rms_px is None


def test_model_and_joint_permutation_loss() -> None:
    torch.manual_seed(0)
    model = Stage3TCN()
    obs = torch.randn(2, 200, 4, 5)
    mask = torch.ones(2, 200, 4, dtype=torch.bool)
    event_mask = torch.ones(2, 200, dtype=torch.bool)
    event_time_s = torch.linspace(-1.0, 0.0, 200).expand(2, -1)
    tau = torch.rand(2, 8) * 0.5
    output = model(obs, mask, event_mask, event_time_s, tau)
    target_position = output["position_mean"].detach().clone()
    target_normal = output["normal"].detach().clone()
    target_motion = torch.tensor([0, 3])
    loss, metrics = stage3_loss(output, target_position, target_normal, target_motion)
    loss.backward()
    assert output["position_mean"].shape == (2, 8, 4, 3)
    assert output["motion_logits"].shape == (2, 4)
    assert np.isfinite(float(loss))
    assert "variance_mean" in metrics


def test_model_distinguishes_identical_values_at_different_real_times() -> None:
    torch.manual_seed(1)
    model = Stage3TCN().eval()
    obs = torch.randn(1, 200, 4, 5)
    mask = torch.ones(1, 200, 4, dtype=torch.bool)
    event_mask = torch.ones(1, 200, dtype=torch.bool)
    dense_time = torch.linspace(-1.0, 0.0, 200).unsqueeze(0)
    sparse_time = torch.linspace(-4.0, 0.0, 200).unsqueeze(0)
    tau = torch.tensor([[0.1, 0.3]])
    with torch.no_grad():
        dense = model(obs, mask, event_mask, dense_time, tau)["position_mean"]
        sparse = model(obs, mask, event_mask, sparse_time, tau)["position_mean"]
    assert not torch.allclose(dense, sparse)


def test_last_200_irregular_valid_events_are_preserved_without_grid_collision() -> None:
    t0 = 5_000_000_000
    timestamps = [t0 - 300_000_000]
    for index in range(1, 205):
        timestamps.append(timestamps[-1] + (1_000_000 if index % 2 else 2_000_000))
    shift = t0 - timestamps[-1]
    timestamps = [value + shift for value in timestamps]
    observations = [_observation(timestamp, index) for index, timestamp in enumerate(timestamps)]
    anchor = observations[-1]
    sample = _make_sample(
        anchor, observations, [_truth(t0, 204)], None, EXTRINSIC, query_taus=(0.0,)
    )
    assert sample is not None
    assert sample.event_mask.all()
    assert sample.obs_mask.any(axis=1).all()
    expected = (np.asarray(timestamps[-200:], dtype=np.float64) - t0) / 1e9
    assert np.allclose(sample.event_time_s, expected)
    assert sample.event_time_s[-1] == 0.0


def test_position_monitor_uses_one_joint_permutation() -> None:
    target = torch.arange(2 * 3 * 4 * 3, dtype=torch.float32).reshape(2, 3, 4, 3)
    prediction = target[:, :, [2, 0, 3, 1], :]
    error = _position_set_l2(prediction, target)
    assert torch.allclose(error, torch.zeros_like(error))


def test_four_way_analysis_uses_injective_unordered_observation_matching() -> None:
    truth = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    observed = truth[[2, 0, 3]] + np.asarray([0.01, -0.02, 0.03])
    result = _match_observation_to_truth(observed, truth)
    assert result is not None
    error, assignment = result
    assert set(assignment) == {0, 2, 3}
    assert np.isclose(error, np.linalg.norm([0.01, -0.02, 0.03]))
    assert _match_observation_to_truth(np.zeros((5, 3)), truth) is None


def test_observation_loss_masks_missing_future_frames_and_keeps_physical_term() -> None:
    torch.manual_seed(3)
    model = Stage3TCN(input_features=7, observation_heads=True)
    obs = torch.randn(2, 16, 4, 7)
    obs_mask = torch.ones(2, 16, 4, dtype=torch.bool)
    event_mask = torch.ones(2, 16, dtype=torch.bool)
    event_time_s = torch.linspace(-0.2, 0.0, 16).expand(2, -1)
    tau = torch.rand(2, 8) * 0.5
    output = model(obs, obs_mask, event_mask, event_time_s, tau)
    future_position = torch.randn(2, 8, 4, 3)
    future_normal = torch.nn.functional.normalize(torch.randn(2, 8, 4, 3), dim=-1)
    future_observation = torch.randn(2, 8, 4, 3)
    future_mask = torch.zeros(2, 8, 4, dtype=torch.bool)
    frame_available = torch.zeros(2, 8, dtype=torch.bool)
    ambiguous = torch.zeros(2, 8, dtype=torch.bool)
    loss, metrics = stage3_observation_loss(
        output, future_position, future_normal, torch.tensor([0, 3]),
        future_observation, future_mask, frame_available, ambiguous,
    )
    loss.backward()
    assert np.isfinite(float(loss))
    assert np.isfinite(metrics["loss"])


def test_effective_tau_uses_matched_truth_timestamp() -> None:
    t0 = 2_000_000_000
    anchor = _observation(t0, 100)
    observations = [_observation(t0 - offset * 20_000_000, 100 - offset) for offset in range(8)]
    truths = [_truth(t0, 100), _truth(t0 + 112_000_000, 101)]
    sample = _make_sample(anchor, observations, truths, None, EXTRINSIC, query_taus=(0.1,))
    assert sample is not None
    assert np.isclose(sample.tau_requested[0], 0.1)
    assert np.isclose(sample.tau[0], 0.112)
    assert sample.future_timestamp_ns[0] == t0 + 112_000_000


def test_any_history_frame_over_four_rejects_window() -> None:
    t0 = 2_000_000_000
    anchor = _observation(t0, 100)
    observations = [_observation(t0 - offset * 20_000_000, 100 - offset) for offset in range(8)]
    observations.append(_observation(t0 - 60_000_000, 90, armor_count=5))
    assert _make_sample(anchor, observations, [_truth(t0, 100)], None, EXTRINSIC, query_taus=(0.0,)) is None


def test_latest_valid_age_ignores_newer_zero_candidate_frames() -> None:
    t0 = 2_000_000_000
    anchor = _observation(t0, 100, armor_count=0)
    observations = [anchor]
    observations.extend(
        _observation(t0 - 51_000_000 - offset * 15_000_000, 90 - offset)
        for offset in range(8)
    )
    assert _make_sample(anchor, observations, [_truth(t0, 100)], None, EXTRINSIC, query_taus=(0.0,)) is None


def test_gimbal_origin_and_chassis_axes_define_label_frame_without_h() -> None:
    value = _world_to_tracker(
        (4.0, 2.0, 3.0), (1.0, 2.0, 3.0),
        (1.0, 0.0, 0.0, 0.0),
    )
    assert np.allclose(value, (3.0, 0.0, 0.0), atol=1e-8)


def test_canonical_source_discovery_ignores_extra_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    raw_root = tmp_path / "raw"
    session_id = "stage3-test-0000"
    manifest = {
        "schema_version": "stage3-manifest-v1", "session_id": session_id,
        "mode": "stationary", "distance_m": 3.0,
    }
    evidence.mkdir()
    (evidence / f".manifest-{session_id}.json").write_text(json.dumps(manifest), encoding="utf-8")
    accepted = raw_root / session_id / "run-accepted"
    accepted.mkdir(parents=True)
    observations = accepted / "observations.jsonl"
    truth = accepted / "truth.jsonl"
    observations.write_text("{}\n", encoding="utf-8")
    truth.write_text("{}\n", encoding="utf-8")
    result_dir = evidence / session_id
    result_dir.mkdir()
    (result_dir / "session_result.json").write_text(json.dumps({
        "session_id": session_id, "observations": str(observations), "truth": str(truth),
    }), encoding="utf-8")
    extra = evidence / "first-article-0000"
    extra.mkdir()
    (extra / "session_result.json").write_text("{}", encoding="utf-8")
    sources = discover_canonical_sources([manifest], evidence, raw_root)
    assert len(sources) == 1
    assert sources[0]["observations"] == str(observations.resolve())


def test_stratified_split_is_deterministic_disjoint_cover() -> None:
    records = []
    for index in range(20):
        records.append({
            "session_id": f"s{index:02d}",
            "mode": ("stationary", "linear", "spin", "linear_and_spin")[index % 4],
            "distance_m": 1.0 + (index % 3) * 2.5,
            "distance_bin": index % 3,
            "direction_sector": index % 8,
            "linear_speed_mps": float(index % 4),
            "spin_rad_s": (-1.0 if index % 2 else 1.0) * float(index % 15),
        })
    first = stratified_session_split(records, 7)
    second = stratified_session_split(records, 7)
    assert first == second
    assert [len(first[name]) for name in ("train", "validation", "test")] == [12, 4, 4]
    combined = first["train"] + first["validation"] + first["test"]
    assert len(combined) == len(set(combined)) == 20


def test_rigid_constant_velocity_yaw_rate_noiseless_combined_motion() -> None:
    geometry = {
        "position": np.asarray([[0.3, 0, 0], [0, 0.3, 0], [-0.3, 0, 0], [0, -0.3, 0]], dtype=np.float64),
        "normal": np.asarray([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=np.float64),
        "yaw": np.asarray([0.0, np.pi / 2, np.pi, -np.pi / 2], dtype=np.float64),
    }
    obs = np.zeros((200, 4, 5), dtype=np.float32)
    mask = np.zeros((200, 4), dtype=np.bool_)
    event_mask = np.zeros((200,), dtype=np.bool_)
    event_time_s = np.zeros((200,), dtype=np.float32)
    velocity = np.asarray([0.4, -0.2, 0.0])
    omega = 1.2
    for slot in range(160, 200, 5):
        time_s = -0.8 + (slot - 160) * 0.021
        theta = 0.4 + omega * time_s
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        positions = np.asarray([2.0, 0.5, 0.2]) + velocity * time_s + geometry["position"] @ rotation.T
        yaws = geometry["yaw"] + theta
        obs[slot, :, :3] = positions
        obs[slot, :, 3] = np.sin(yaws)
        obs[slot, :, 4] = np.cos(yaws)
        mask[slot] = True
        event_mask[slot] = True
        event_time_s[slot] = time_s
    tau = np.asarray([0.0, 0.1, 0.5], dtype=np.float32)
    predicted, _, status = rigid_constant_velocity_yaw_rate(
        obs, mask, event_mask, event_time_s, tau, geometry
    )
    assert status["valid"]
    expected = []
    for seconds in tau:
        theta = 0.4 + omega * float(seconds)
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        expected.append(np.asarray([2.0, 0.5, 0.2]) + velocity * float(seconds) + geometry["position"] @ rotation.T)
    expected_array = np.asarray(expected)
    best = min(
        float(np.max(np.abs(predicted[:, list(permutation)] - expected_array)))
        for permutation in itertools.permutations(range(4))
    )
    assert best < 1e-5
