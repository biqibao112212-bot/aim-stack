from __future__ import annotations

import inspect
import math

import torch

from training.stage3.cyclic_future_loss import cyclic_future_expert_loss
from training.stage3.cyclic_future_model import (
    CyclicFutureMotionExpert,
    decode_combined,
    decode_rotation,
    decode_translation,
    planar_j,
)


def _rotate_xy(value: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    return torch.stack((
        c[..., None] * value[..., 0] - s[..., None] * value[..., 1],
        s[..., None] * value[..., 0] + c[..., None] * value[..., 1],
    ), dim=-1)


def _physical_state(dtype: torch.dtype = torch.float64):
    center = torch.tensor((2.8, -0.3, 0.12), dtype=dtype)
    velocity = torch.tensor((0.7, -0.25, 0.08), dtype=dtype)
    offset = torch.tensor((
        (0.31, 0.02, -0.04),
        (-0.03, 0.24, 0.03),
        (-0.27, -0.01, -0.02),
        (0.04, -0.34, 0.05),
    ), dtype=dtype)
    q0 = center + offset
    return center, velocity, offset, q0


def test_combined_decoder_matches_center_based_constant_twist() -> None:
    center, velocity, offset, q0 = _physical_state()
    q0 = q0.unsqueeze(0)
    primary = torch.tensor((0,))
    tau = torch.tensor(((0.0, 0.1, 0.2, 0.5),), dtype=torch.float64)
    for omega_value in (1.7, -1.7):
        omega = torch.tensor((omega_value,), dtype=torch.float64)
        primary_velocity = velocity.clone()
        primary_velocity[:2] += omega_value * planar_j(offset[0, :2])
        primary_acceleration = -omega_value ** 2 * offset[0, :2]
        actual = decode_combined(
            q0, primary, primary_velocity.unsqueeze(0),
            primary_acceleration.unsqueeze(0), omega, tau,
        )[0]
        expected = []
        for time in tau[0]:
            rotated = _rotate_xy(offset[:, :2], omega[0] * time)
            xy = center[:2] + velocity[:2] * time + rotated
            z = center[2] + velocity[2] * time + offset[:, 2]
            expected.append(torch.cat((xy, z[:, None]), dim=-1))
        expected = torch.stack(expected)
        assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_rotation_decoder_matches_fixed_center_rotation() -> None:
    center, _, offset, q0 = _physical_state()
    q0 = q0.unsqueeze(0)
    omega = torch.tensor((-2.1,), dtype=torch.float64)
    primary_velocity = omega[:, None] * planar_j(offset[0, :2]).unsqueeze(0)
    tau = torch.tensor(((0.0, 0.1, 0.5),), dtype=torch.float64)
    actual = decode_rotation(q0, torch.tensor((0,)), primary_velocity, omega, tau)[0]
    expected = []
    for time in tau[0]:
        xy = center[:2] + _rotate_xy(offset[:, :2], omega[0] * time)
        expected.append(torch.cat((xy, q0[0, :, 2:3]), dim=-1))
    assert torch.allclose(actual, torch.stack(expected), atol=1e-10, rtol=1e-10)


def test_tau_zero_is_bit_exact_q0_for_every_decoder() -> None:
    _, velocity, offset, q0 = _physical_state(torch.float32)
    q0 = q0.unsqueeze(0)
    tau = torch.zeros((1, 1))
    primary = torch.tensor((0,))
    omega = torch.tensor((3.2,))
    rotation_velocity = omega[:, None] * planar_j(offset[0, :2]).unsqueeze(0)
    combined_velocity = velocity.unsqueeze(0).clone()
    combined_velocity[:, :2] += rotation_velocity
    acceleration = -omega[:, None].square() * offset[0, :2].unsqueeze(0)
    outputs = (
        decode_translation(q0, velocity.unsqueeze(0), tau),
        decode_rotation(q0, primary, rotation_velocity, omega, tau),
        decode_combined(
            q0, primary, combined_velocity, acceleration, omega, tau,
        ),
    )
    for output in outputs:
        assert torch.equal(output[:, 0], q0)


def test_combined_small_omega_forward_and_gradient_are_finite() -> None:
    _, _, _, q0 = _physical_state(torch.float64)
    q0 = q0.unsqueeze(0)
    tau = torch.tensor(((0.0, 1e-6, 0.1, 0.5),), dtype=torch.float64)
    for value in (0.0, 1e-9, -1e-9, 1e-6, -1e-4, 1e-2):
        omega = torch.tensor((value,), dtype=torch.float64, requires_grad=True)
        velocity = torch.tensor(((0.4, -0.2, 0.03),), dtype=torch.float64, requires_grad=True)
        acceleration = torch.tensor(((0.1, -0.04),), dtype=torch.float64, requires_grad=True)
        output = decode_combined(
            q0, torch.tensor((0,)), velocity, acceleration, omega, tau,
        )
        output.square().sum().backward()
        assert torch.isfinite(output).all()
        assert torch.isfinite(omega.grad).all()
        assert torch.isfinite(velocity.grad).all()
        assert torch.isfinite(acceleration.grad).all()


def test_combined_preserves_3d_pair_distance_with_unequal_heights() -> None:
    _, velocity, offset, q0 = _physical_state()
    omega = torch.tensor((2.4,), dtype=torch.float64)
    primary_velocity = velocity.clone()
    primary_velocity[:2] += omega.item() * planar_j(offset[0, :2])
    acceleration = -omega.square()[:, None] * offset[0, :2].unsqueeze(0)
    future = decode_combined(
        q0.unsqueeze(0), torch.tensor((0,)), primary_velocity.unsqueeze(0),
        acceleration, omega, torch.tensor(((0.0, 0.13, 0.47),), dtype=torch.float64),
    )[0]
    distance = torch.cdist(future, future)
    assert torch.allclose(distance, distance[:1].expand_as(distance), atol=1e-10)


def _synthetic_batch() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    torch.manual_seed(13)
    batch, time = 2, 8
    q0 = torch.tensor((
        ((2.9, 0.1, 0.08), (3.0, 0.3, 0.11), (3.2, 0.0, 0.06), (3.0, -0.25, 0.13)),
        ((3.1, -0.2, 0.09), (3.2, 0.05, 0.12), (3.4, -0.1, 0.07), (3.15, -0.4, 0.14)),
    ))
    obs = torch.zeros(batch, time, 4, 3)
    visible = torch.zeros(batch, time, 4, dtype=torch.bool)
    primary = torch.zeros_like(visible)
    for row in range(batch):
        for event in range(time):
            obs[row, event, 0] = q0[row, 0] - torch.tensor((0.01 * (time - 1 - event), 0.0, 0.0))
            visible[row, event, 0] = True
            primary[row, event, 0] = True
            if event % 2 == 0:
                obs[row, event, 1] = q0[row, 1]
                visible[row, event, 1] = True
    current_visible = torch.tensor(((True, True, False, False),) * batch)
    anchor_composed = torch.tensor(((False, False, False, True),) * batch)
    valid = current_visible | anchor_composed
    edge = torch.roll(q0, shifts=-1, dims=1) - q0
    clockwise = torch.tensor(((False, True, False, False),) * batch)
    counter = torch.tensor(((False, False, False, True),) * batch)
    inputs = {
        "obs": obs,
        "obs_mask": visible,
        "primary_mask": primary,
        "event_mask": torch.ones(batch, time, dtype=torch.bool),
        "event_time_s": torch.linspace(-0.07, 0.0, time).repeat(batch, 1),
        "switch_step": torch.zeros(batch, time),
        "tau": torch.tensor(((0.0, 0.1, 0.2, 0.5),) * batch),
    }
    s_state = {
        "q0_m": q0,
        "q0_valid": valid,
        "q0_sigma_m": torch.full((batch, 4, 1), 0.01),
        "current_visible": current_visible,
        "q0_observed": current_visible,
        "anchor_composed": anchor_composed,
        "cold": torch.tensor(((False, False, True, False),) * batch),
        "age_s": torch.tensor(((0.0, 0.0, math.inf, 0.1),) * batch),
        "edge0_valid": torch.tensor(((True, False, False, True),) * batch),
        "edge0_sigma_m": torch.full((batch, 4, 1), 0.02),
        "edge_age_s": torch.tensor(((0.0, math.inf, math.inf, 0.1),) * batch),
        "clockwise": clockwise,
        "counterclockwise": counter,
        "primary_index": torch.zeros(batch, dtype=torch.long),
        "edge0_m": edge,
    }
    return inputs, s_state


def _model(expert: str) -> CyclicFutureMotionExpert:
    return CyclicFutureMotionExpert(
        expert,
        torch.tensor((3.0, 0.0, 0.1)),
        torch.tensor((1.0, 1.0, 0.1)),
        channels=8, dropout=0.0, history_events=8,
    )


def _forward(model, batch, state):
    return model(
        batch["obs"], batch["obs_mask"], batch["primary_mask"],
        batch["event_mask"], batch["event_time_s"], batch["switch_step"],
        batch["tau"], state,
    )


def test_forward_contract_has_no_future_truth_class_or_geometry() -> None:
    parameters = list(inspect.signature(CyclicFutureMotionExpert.forward).parameters)
    assert parameters == [
        "self", "obs", "obs_mask", "primary_mask", "event_mask",
        "event_time_s", "switch_step", "tau", "s_state", "q0_override_m",
    ]
    for expert in ("translation", "rotation", "combined"):
        config = _model(expert).config()
        assert config["q0_head"] is False
        assert config["fixed_geometry"] is False
        assert config["combined_calls_other_experts"] is False


def test_future_expert_is_exactly_c4_roll_equivariant() -> None:
    batch, state = _synthetic_batch()
    for expert in ("translation", "rotation", "combined"):
        model = _model(expert).eval()
        reference = _forward(model, batch, state)
        for shift in (1, 2, 3):
            shifted_batch = dict(batch)
            for name in ("obs", "obs_mask", "primary_mask"):
                shifted_batch[name] = torch.roll(batch[name], shifts=shift, dims=2)
            shifted_state = dict(state)
            for name, value in state.items():
                if value.ndim >= 2 and value.shape[1] == 4:
                    shifted_state[name] = torch.roll(value, shifts=shift, dims=1)
            shifted_state["primary_index"] = (state["primary_index"] + shift) % 4
            actual = _forward(model, shifted_batch, shifted_state)
            assert torch.allclose(
                actual["position_m"],
                torch.roll(reference["position_m"], shifts=shift, dims=2),
                atol=2e-6,
            )
            assert torch.equal(
                actual["future_valid"],
                torch.roll(reference["future_valid"], shifts=shift, dims=1),
            )
            assert torch.allclose(
                actual["omega_rad_s"], reference["omega_rad_s"], atol=1e-6
            )


def test_excluded_targets_do_not_change_future_loss() -> None:
    batch, state = _synthetic_batch()
    model = _model("translation").eval()
    prediction = _forward(model, batch, state)
    truth = prediction["position_m"].detach().clone()
    rule = torch.ones(truth.shape[:2], dtype=torch.bool)
    base, _ = cyclic_future_expert_loss(
        "translation", prediction, state, truth, batch["tau"], rule,
    )
    changed = truth.clone()
    changed[:, :, 2:] += 1000.0
    altered, _ = cyclic_future_expert_loss(
        "translation", prediction, state, changed, batch["tau"], rule,
    )
    assert torch.equal(base, altered)


def test_rotating_experts_exclude_cold_truth_from_loss_and_gradient() -> None:
    batch, state = _synthetic_batch()
    for expert in ("rotation", "combined"):
        template = _model(expert).eval()
        initial = {name: value.clone() for name, value in template.state_dict().items()}
        results = []
        for mutate_cold in (False, True):
            model = _model(expert).eval()
            model.load_state_dict(initial)
            prediction = _forward(model, batch, state)
            truth = prediction["position_m"].detach().clone()
            if mutate_cold:
                truth[:, :, 2] += 1000.0
            rule = torch.ones(truth.shape[:2], dtype=torch.bool)
            total, _ = cyclic_future_expert_loss(
                expert, prediction, state, truth, batch["tau"], rule,
            )
            total.backward()
            gradient = torch.cat([
                parameter.grad.reshape(-1)
                for parameter in model.parameters()
                if parameter.grad is not None
            ])
            results.append((total.detach(), gradient.detach()))
        assert torch.equal(results[0][0], results[1][0])
        assert torch.equal(results[0][1], results[1][1])


def test_motion_bounds_cover_qualified_dynamic_tail() -> None:
    for expert in ("translation", "rotation", "combined"):
        model = _model(expert)
        assert model.max_speed_mps >= 7.0
        assert model.max_omega_rad_s >= 20.0


def test_truth_q0_override_is_evaluation_only() -> None:
    batch, state = _synthetic_batch()
    model = _model("combined")
    try:
        _forward_with_override = model(
            batch["obs"], batch["obs_mask"], batch["primary_mask"],
            batch["event_mask"], batch["event_time_s"], batch["switch_step"],
            batch["tau"], state, q0_override_m=state["q0_m"],
        )
    except ValueError as error:
        assert "evaluation-only" in str(error)
    else:
        raise AssertionError("train-mode q0 override must fail closed")
    model.eval()
    output = model(
        batch["obs"], batch["obs_mask"], batch["primary_mask"],
        batch["event_mask"], batch["event_time_s"], batch["switch_step"],
        batch["tau"], state, q0_override_m=state["q0_m"],
    )
    assert output["position_m"].shape == (2, 4, 4, 3)


def test_ineligible_future_query_does_not_change_loss() -> None:
    batch, state = _synthetic_batch()
    model = _model("combined").eval()
    prediction = _forward(model, batch, state)
    truth = prediction["position_m"].detach().clone()
    rule = torch.ones(truth.shape[:2], dtype=torch.bool)
    rule[:, -1] = False
    base, _ = cyclic_future_expert_loss(
        "combined", prediction, state, truth, batch["tau"], rule,
    )
    changed = truth.clone()
    changed[:, -1] += 1000.0
    altered, _ = cyclic_future_expert_loss(
        "combined", prediction, state, changed, batch["tau"], rule,
    )
    assert torch.equal(base, altered)
