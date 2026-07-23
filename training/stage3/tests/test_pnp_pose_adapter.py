from __future__ import annotations

import torch

from training.stage3.pnp_pose_adapter_loss import current_pose_loss
from training.stage3.pnp_pose_adapter_model import CurrentPnPPoseAdapter
from training.stage3.train_pnp_pose_adapter import (
    _anchor_mask,
    _batch_errors,
    _prepare_batch,
    _selection_tuple,
)


GEOMETRY = torch.tensor([
    [0.008, -0.211, -0.061],
    [0.220, 0.000, -0.075],
    [0.008, 0.211, -0.061],
    [-0.204, 0.000, -0.075],
], dtype=torch.float32)
MEAN = torch.tensor([4.2, 0.0, -0.06])
STD = torch.tensor([2.2, 1.2, 0.17])


def _inputs(batch: int = 2, time: int = 8) -> tuple[torch.Tensor, ...]:
    center = torch.tensor([[4.0, 0.2, 0.1], [5.0, -0.4, 0.2]])[:batch]
    angle = torch.tensor([0.3, -0.7])[:batch]
    cosine, sine = torch.cos(angle), torch.sin(angle)
    gx, gy = GEOMETRY[:, 0], GEOMETRY[:, 1]
    rotated = torch.stack((
        cosine[:, None] * gx - sine[:, None] * gy,
        sine[:, None] * gx + cosine[:, None] * gy,
        GEOMETRY[:, 2].expand(batch, -1),
    ), dim=-1)
    current = center[:, None] + rotated
    raw_xyz = current[:, None].expand(-1, time, -1, -1).clone()
    obs = torch.zeros(batch, time, 4, 7)
    obs[..., :3] = (raw_xyz - MEAN) / STD
    armor_yaw = angle[:, None] + torch.tensor([-1.53, 0.0, 1.53, 3.14])[None]
    obs[..., 3] = torch.sin(armor_yaw)[:, None]
    obs[..., 4] = torch.cos(armor_yaw)[:, None]
    obs_mask = torch.ones(batch, time, 4, dtype=torch.bool)
    event_mask = torch.ones(batch, time, dtype=torch.bool)
    event_time = torch.linspace(-0.14, 0.0, time).expand(batch, -1).clone()
    return obs, obs_mask, event_mask, event_time, current


def _model() -> CurrentPnPPoseAdapter:
    torch.manual_seed(11)
    return CurrentPnPPoseAdapter(
        GEOMETRY, MEAN, STD, channels=32, dropout=0.0
    ).eval()


def test_candidate_permutation_cannot_change_the_pose() -> None:
    model = _model()
    obs, obs_mask, event_mask, event_time, _ = _inputs()
    expected = model(obs, obs_mask, event_mask, event_time)
    order = torch.tensor([2, 0, 3, 1])
    actual = model(obs[:, :, order], obs_mask[:, :, order], event_mask, event_time)
    assert torch.allclose(actual["center"], expected["center"], atol=1e-6)
    assert torch.allclose(actual["phase"], expected["phase"], atol=1e-6)
    assert torch.allclose(actual["position_mean"], expected["position_mean"], atol=1e-6)


def test_independent_per_event_permutations_cannot_change_the_pose() -> None:
    model = _model()
    obs, obs_mask, event_mask, event_time, _ = _inputs()
    expected = model(obs, obs_mask, event_mask, event_time)
    generator = torch.Generator().manual_seed(91)
    order = torch.stack([
        torch.randperm(4, generator=generator) for _ in range(obs.shape[1])
    ])
    gather_obs = order[None, :, :, None].expand(obs.shape[0], -1, -1, obs.shape[-1])
    gather_mask = order[None].expand(obs.shape[0], -1, -1)
    actual = model(
        obs.gather(2, gather_obs), obs_mask.gather(2, gather_mask),
        event_mask, event_time,
    )
    assert torch.allclose(actual["center"], expected["center"], atol=1e-6)
    assert torch.allclose(actual["phase"], expected["phase"], atol=1e-6)


def test_decoder_always_preserves_fixed_geometry() -> None:
    output = _model()(*_inputs()[:4])["position_mean"][:, 0]
    pair_i, pair_j = torch.triu_indices(4, 4, offset=1)
    expected = torch.linalg.vector_norm(GEOMETRY[pair_i] - GEOMETRY[pair_j], dim=-1)
    actual = torch.linalg.vector_norm(output[:, pair_i] - output[:, pair_j], dim=-1)
    assert output.dtype == torch.float32
    assert torch.allclose(actual, expected[None], atol=2e-6)


def test_masked_and_nonfinite_candidate_cannot_pollute_output() -> None:
    model = _model()
    obs, obs_mask, event_mask, event_time, _ = _inputs()
    obs_mask[:, -1, 2] = False
    clean = obs.clone()
    clean[:, -1, 2] = 0.0
    poisoned = clean.clone()
    poisoned[0, -1, 2] = torch.tensor([
        float("nan"), float("inf"), -float("inf"), 1e30, -1e30, 9.0, 9.0
    ])
    expected = model(clean, obs_mask, event_mask, event_time)
    actual = model(poisoned, obs_mask, event_mask, event_time)
    assert torch.equal(actual["position_mean"], expected["position_mean"])
    assert torch.isfinite(actual["position_mean"]).all()


def test_loss_reads_only_same_time_q0_target() -> None:
    model = _model()
    obs, obs_mask, event_mask, event_time, target = _inputs()
    output = model(obs, obs_mask, event_mask, event_time)
    loss, parts = current_pose_loss(output, target, GEOMETRY)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert set(parts) == {"center", "phase"}
    loss.backward()
    assert model.pose_head[-1].weight.grad is not None
    assert torch.isfinite(model.pose_head[-1].weight.grad).all()


def test_anchor_filter_rejects_stale_current_observations() -> None:
    obs, obs_mask, event_mask, event_time, target = _inputs()
    event_time[1, -1] = -0.03
    raw = {
        "obs": obs, "obs_mask": obs_mask, "event_mask": event_mask,
        "event_time_s": event_time,
        "tau": torch.tensor([[0.0, 0.1, 0.2, 0.5, 0.3, 0.4, 0.6, 0.7]]).expand(2, -1).clone(),
        "future_position": target[:, None].expand(-1, 8, -1, -1).clone(),
        "motion_class": torch.tensor([0, 3]),
    }
    assert _anchor_mask(raw, 1e-6).tolist() == [True, False]
    batch = _prepare_batch(raw, torch.device("cpu"), 1e-6)
    assert batch is not None and batch["target_q0"].shape == (1, 4, 3)


def test_anchor_filter_rejects_even_tolerated_future_observations() -> None:
    obs, obs_mask, event_mask, event_time, _ = _inputs()
    event_time[:, -1] = 5e-7
    raw = {"event_mask": event_mask, "event_time_s": event_time}
    assert _anchor_mask(raw, 1e-6).tolist() == [False, False]


def test_full_phase_alias_is_visible_despite_small_unordered_error() -> None:
    decoder = _model().decoder
    center = torch.tensor([[[4.0, 0.0, 0.0]]])
    truth_phase = torch.tensor([[[1.0, 0.0]]])
    alias_phase = torch.tensor([[[0.0, 1.0]]])
    truth = decoder(center, truth_phase)[:, 0]
    prediction = decoder(center, alias_phase)
    errors = _batch_errors({
        "position_mean": prediction,
        "center": center[:, 0],
        "phase": alias_phase[:, 0],
    }, truth, GEOMETRY)
    assert errors["phase_abs_deg"].item() == 90.0
    assert errors["phase_alias_index"].item() == 1
    assert errors["fixed_slot_position_m"].item() > errors["unordered_set_position_m"].item()


def test_checkpoint_selection_prioritizes_fixed_slot_p95() -> None:
    metrics = {"overall": {
        "fixed_slot_position_m": {"p95": 0.2, "p99": 0.4},
        "center_m": {"p95": 0.1}, "phase_abs_deg": {"p95": 20.0},
    }}
    assert _selection_tuple(metrics) == (0.2, 0.1, 20.0, 0.4)
