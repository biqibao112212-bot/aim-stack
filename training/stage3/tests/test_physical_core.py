from __future__ import annotations

import pytest
import torch

from training.stage3.physical_loss import physical_loss
from training.stage3.physical_metrics import physical_batch_errors, q0_permutation
from training.stage3.physical_baseline import ExactStateRigidRollout, RigidTwoFrameRollout
from training.stage3.build_truth_history_dataset import _rotation_matrix
from training.stage3.physical_model import AnchoredDeltaPredictor, RigidMotionPredictor


GEOMETRY = torch.tensor([
    [0.30, 0.00, 0.02],
    [0.00, 0.25, -0.02],
    [-0.30, 0.00, 0.02],
    [0.00, -0.25, -0.02],
], dtype=torch.float32)


def _batch(batch: int = 2) -> tuple[torch.Tensor, ...]:
    obs = torch.zeros(batch, 200, 4, 5)
    obs_mask = torch.zeros(batch, 200, 4, dtype=torch.bool)
    event_mask = torch.zeros(batch, 200, dtype=torch.bool)
    event_time_s = torch.zeros(batch, 200)
    obs[:, -3:, :, :3] = GEOMETRY
    obs[:, -3:, :, 4] = 1.0
    obs_mask[:, -3:] = True
    event_mask[:, -3:] = True
    event_time_s[:, -3:] = torch.tensor([-0.04, -0.02, 0.0])
    tau = torch.tensor([0.0, 0.1, 0.2, 0.5]).expand(batch, -1).clone()
    return obs, obs_mask, event_mask, event_time_s, tau


def test_q0_permutation_is_reused_for_future_queries() -> None:
    target = GEOMETRY.view(1, 1, 4, 3).expand(1, 2, -1, -1).clone()
    prediction = target[:, :, [2, 0, 3, 1]].clone()
    prediction[:, 1, 0] = target[:, 1, 0]  # A future-only reassignment would cheat.
    aligned, _, _ = q0_permutation(prediction, target)
    assert torch.allclose(aligned[:, 0], target[:, 0])
    assert not torch.allclose(aligned[:, 1], target[:, 1])


def test_physical_metrics_are_zero_for_permuted_perfect_prediction() -> None:
    target = GEOMETRY.view(1, 1, 4, 3).expand(3, 4, -1, -1).clone()
    prediction = target[:, :, [3, 1, 0, 2]].clone()
    metrics = physical_batch_errors(prediction, target)
    for name in (
        "state_q0_m", "absolute_pg_m", "motion_delta_m",
        "centroid_q0_m", "centered_shape_q0_m", "rigid_residual_q0_m",
    ):
        assert torch.max(torch.abs(metrics[name])).item() < 1e-6


def test_translation_changes_state_not_motion_or_rigidity() -> None:
    target = GEOMETRY.view(1, 1, 4, 3).expand(1, 4, -1, -1).clone()
    offset = torch.tensor([0.2, -0.1, 0.05])
    metrics = physical_batch_errors(target + offset, target)
    expected = torch.linalg.vector_norm(offset).item()
    assert metrics["state_q0_m"].item() == pytest.approx(expected, abs=1e-6)
    assert torch.max(metrics["motion_delta_m"]).item() < 1e-6
    assert metrics["rigid_residual_q0_m"].item() < 1e-6


def test_anchored_direct_model_has_exact_zero_query_displacement() -> None:
    model = AnchoredDeltaPredictor(channels=32, dropout=0.0)
    output = model(*_batch())
    assert output["position_mean"].shape == (2, 4, 4, 3)
    assert torch.equal(output["delta"][:, 0], torch.zeros_like(output["delta"][:, 0]))
    assert torch.allclose(output["position_mean"][:, 0], output["anchor_position"])


def test_rigid_model_preserves_pairwise_geometry_at_every_query() -> None:
    model = RigidMotionPredictor(GEOMETRY, channels=32, dropout=0.0)
    output = model(*_batch())
    position = output["position_mean"]
    assert position.shape == (2, 4, 4, 3)
    assert torch.isfinite(position).all()
    pair_i, pair_j = torch.triu_indices(4, 4, offset=1)
    actual = torch.linalg.vector_norm(position[:, :, pair_i] - position[:, :, pair_j], dim=-1)
    expected = torch.linalg.vector_norm(GEOMETRY[pair_i] - GEOMETRY[pair_j], dim=-1)
    assert torch.allclose(actual, expected.view(1, 1, -1), atol=1e-5)


def test_rigid_model_preserves_true_center_geometry_without_recentering() -> None:
    offset_geometry = GEOMETRY + torch.tensor([0.008, 0.001, -0.068])
    assert torch.linalg.vector_norm(offset_geometry.mean(dim=0)).item() > 0.05
    model = RigidMotionPredictor(offset_geometry, channels=32, dropout=0.0)
    assert torch.equal(model.geometry, offset_geometry)


def test_all_missing_history_remains_finite() -> None:
    model = AnchoredDeltaPredictor(channels=32, dropout=0.0)
    obs, obs_mask, event_mask, event_time_s, tau = _batch(batch=1)
    output = model(
        torch.zeros_like(obs), torch.zeros_like(obs_mask),
        torch.zeros_like(event_mask), torch.zeros_like(event_time_s), tau,
    )
    assert torch.isfinite(output["position_mean"]).all()


def test_physical_loss_is_exactly_zero_for_perfect_prediction() -> None:
    target = GEOMETRY.view(1, 1, 4, 3).expand(2, 4, -1, -1).clone()
    tau = torch.tensor([0.0, 0.1, 0.2, 0.5]).expand(2, -1).clone()
    loss, metrics = physical_loss(target.clone(), target, tau)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)
    assert metrics["loss"] == pytest.approx(0.0, abs=1e-8)


def test_physical_loss_rejects_nonzero_query_zero() -> None:
    target = GEOMETRY.view(1, 1, 4, 3).expand(1, 4, -1, -1).clone()
    tau = torch.tensor([[0.001, 0.1, 0.2, 0.5]])
    with pytest.raises(ValueError, match="exact tau=0"):
        physical_loss(target, target, tau)


def test_physical_loss_masks_unpredictable_future_queries() -> None:
    target = GEOMETRY.view(1, 1, 4, 3).expand(1, 4, -1, -1).clone()
    prediction = target.clone()
    prediction[:, 3] += 10.0
    tau = torch.tensor([[0.0, 0.1, 0.2, 0.5]])
    mask = torch.tensor([[True, True, True, False]])
    loss, _ = physical_loss(prediction, target, tau, query_mask=mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)


def test_analytic_rollout_is_exact_for_constant_translation_and_spin() -> None:
    obs = torch.zeros(1, 6, 4, 3)
    obs_mask = torch.zeros(1, 6, 4, dtype=torch.bool)
    event_mask = torch.zeros(1, 6, dtype=torch.bool)
    event_time = torch.zeros(1, 6)
    velocity = torch.tensor([0.4, -0.2, 0.1])
    omega = 1.3
    center0 = torch.tensor([2.0, 0.5, 0.2])
    for slot, seconds in ((4, -0.02), (5, 0.0)):
        angle = omega * seconds
        c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        relative = GEOMETRY.clone()
        x = c * relative[:, 0] - s * relative[:, 1]
        y = s * relative[:, 0] + c * relative[:, 1]
        relative = torch.stack((x, y, relative[:, 2]), dim=-1)
        obs[0, slot] = center0 + velocity * seconds + relative
        obs_mask[0, slot] = True
        event_mask[0, slot] = True
        event_time[0, slot] = seconds
    tau = torch.tensor([[0.0, 0.1, 0.2, 0.5]])
    model = RigidTwoFrameRollout(torch.zeros(3), torch.ones(3))
    prediction = model(obs, obs_mask, event_mask, event_time, tau)["position_mean"]
    expected = []
    for seconds in tau[0]:
        angle = omega * seconds
        c, s = torch.cos(angle), torch.sin(angle)
        x = c * GEOMETRY[:, 0] - s * GEOMETRY[:, 1]
        y = s * GEOMETRY[:, 0] + c * GEOMETRY[:, 1]
        relative = torch.stack((x, y, GEOMETRY[:, 2]), dim=-1)
        expected.append(center0 + velocity * seconds + relative)
    assert torch.allclose(prediction[0], torch.stack(expected), atol=2e-5)


def test_exact_state_rollout_is_exact_for_constant_translation_and_spin() -> None:
    anchor = (torch.tensor([2.0, 0.5, 0.2]) + GEOMETRY).unsqueeze(0)
    velocity = torch.tensor([[0.4, -0.2, 0.1]])
    yaw_rate = torch.tensor([1.3])
    tau = torch.tensor([[0.0, 0.1, 0.2, 0.5]])
    center = torch.tensor([[2.0, 0.5, 0.2]])
    prediction = ExactStateRigidRollout()(
        anchor, center, velocity, yaw_rate, tau
    )["position_mean"]
    expected = []
    for seconds in tau[0]:
        c, s = torch.cos(yaw_rate[0] * seconds), torch.sin(yaw_rate[0] * seconds)
        x = c * GEOMETRY[:, 0] - s * GEOMETRY[:, 1]
        y = s * GEOMETRY[:, 0] + c * GEOMETRY[:, 1]
        expected.append(
            torch.tensor([2.0, 0.5, 0.2]) + velocity[0] * seconds
            + torch.stack((x, y, GEOMETRY[:, 2]), dim=-1)
        )
    assert torch.allclose(prediction[0], torch.stack(expected), atol=1e-6)


def test_world_to_anchor_row_vector_rotation_for_nonunit_quaternion_pose() -> None:
    angle = torch.tensor(0.7, dtype=torch.float64)
    quaternion = (
        float(torch.cos(angle / 2)), 0.0, 0.0, float(torch.sin(angle / 2))
    )
    rotation = torch.from_numpy(_rotation_matrix(quaternion))
    local = torch.tensor([0.4, -0.2, 0.1], dtype=torch.float64)
    world = rotation @ local
    recovered = world @ rotation
    assert torch.allclose(recovered, local, atol=1e-12)


@pytest.mark.parametrize("yaw_rate", [-1.3, 1.3])
def test_exact_state_rollout_preserves_yaw_rate_sign(yaw_rate: float) -> None:
    geometry = GEOMETRY + torch.tensor([0.008, 0.001, -0.068])
    center = torch.tensor([[2.0, 0.5, 0.2]])
    anchor = center[:, None] + geometry
    tau = torch.tensor([[0.0, 0.1]])
    output = ExactStateRigidRollout()(
        anchor, center, torch.zeros(1, 3), torch.tensor([yaw_rate]), tau
    )["position_mean"]
    before = anchor[0, 0, :2] - center[0, :2]
    after = output[0, 1, 0, :2] - center[0, :2]
    cross = before[0] * after[1] - before[1] * after[0]
    assert torch.sign(cross).item() == pytest.approx(torch.sign(torch.tensor(yaw_rate)).item())
