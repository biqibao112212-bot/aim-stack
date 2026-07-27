from __future__ import annotations

import math

import torch

from training.stage3.cyclic_rotation_ab_loss import cyclic_rotation_ab_loss
from training.stage3.cyclic_rotation_ab_model import (
    DirectRotationTrajectoryExpert,
    OrderedLocalRelationalMotionEncoder,
    ParametricRotationFutureExpertV2,
    UnsignedRelationalMotionEncoder,
    deterministic_rotation_direction,
)


def _case(sign: float = 1.0):
    batch, time = 2, 8
    angles = sign * torch.linspace(-0.35, 0.0, time)
    center = torch.tensor((3.0, 0.1))
    radii = torch.tensor((0.28, 0.31, 0.26, 0.33))
    offsets = torch.tensor((0.0, math.pi / 2, math.pi, 3 * math.pi / 2))
    history = torch.zeros(batch, time, 4, 3)
    for event, angle in enumerate(angles):
        phase = angle + offsets
        history[:, event, :, 0] = center[0] + radii * torch.cos(phase)
        history[:, event, :, 1] = center[1] + radii * torch.sin(phase)
        history[:, event, :, 2] = torch.tensor((0.08, 0.12, 0.06, 0.13))
    visible = torch.zeros(batch, time, 4, dtype=torch.bool)
    visible[:, :, 0] = True
    primary = visible.clone()
    q0 = history[:, -1].clone()
    current = torch.tensor(((True, False, False, False),) * batch)
    warm = torch.tensor(((False, True, False, False),) * batch)
    valid = current | warm
    inputs = {
        "obs": torch.where(visible.unsqueeze(-1), history, torch.zeros_like(history)),
        "obs_mask": visible,
        "primary_mask": primary,
        "event_mask": torch.ones(batch, time, dtype=torch.bool),
        "event_time_s": torch.linspace(-0.07, 0.0, time).repeat(batch, 1),
        "switch_step": torch.zeros(batch, time),
        "tau": torch.tensor(((0.0, 0.1, 0.2, 0.5),) * batch),
    }
    state = {
        "q0_m": q0,
        "q0_valid": valid,
        "q0_sigma_m": torch.full((batch, 4, 1), 0.01),
        "current_visible": current,
        "q0_observed": current,
        "anchor_composed": warm,
        "cold": ~valid,
        "age_s": torch.zeros(batch, 4),
        "edge0_valid": valid,
        "edge0_sigma_m": torch.full((batch, 4, 1), 0.01),
        "edge_age_s": torch.zeros(batch, 4),
        "clockwise": warm,
        "counterclockwise": torch.zeros_like(warm),
        "primary_index": torch.zeros(batch, dtype=torch.long),
    }
    return inputs, state


def _forward(model, inputs, state):
    return model(
        inputs["obs"], inputs["obs_mask"], inputs["primary_mask"],
        inputs["event_mask"], inputs["event_time_s"], inputs["switch_step"],
        inputs["tau"], state,
    )


def test_direction_is_deterministic_causal_geometry() -> None:
    mean = torch.zeros(3)
    std = torch.ones(3)
    for expected in (-1.0, 1.0):
        inputs, _ = _case(expected)
        sign, valid, source = deterministic_rotation_direction(
            inputs["obs"], inputs["obs_mask"], inputs["event_mask"],
            inputs["event_time_s"], mean, std,
        )
        assert bool(valid.all())
        assert torch.equal(sign, torch.full_like(sign, expected))
        assert torch.equal(source, torch.full_like(source, 2))


def test_direction_refuses_to_guess_without_three_observations() -> None:
    inputs, _ = _case(1.0)
    inputs["event_mask"][:, :-2] = False
    inputs["obs_mask"][:, :-2] = False
    sign, valid, source = deterministic_rotation_direction(
        inputs["obs"], inputs["obs_mask"], inputs["event_mask"],
        inputs["event_time_s"], torch.zeros(3), torch.ones(3),
    )
    assert not bool(valid.any())
    assert torch.equal(sign, torch.zeros_like(sign))
    assert torch.equal(source, torch.zeros_like(source))


def test_both_arms_use_same_nonlearned_direction_and_exact_q0() -> None:
    inputs, state = _case(-1.0)
    models = (
        ParametricRotationFutureExpertV2(
            torch.zeros(3), torch.ones(3), channels=8,
            dropout=0.0, history_events=8,
        ),
        DirectRotationTrajectoryExpert(
            torch.zeros(3), torch.ones(3), channels=8,
            dropout=0.0, history_events=8,
        ),
        ParametricRotationFutureExpertV2(
            torch.zeros(3), torch.ones(3), channels=8,
            dropout=0.0, history_events=8, relational_evidence=True,
        ),
        DirectRotationTrajectoryExpert(
            torch.zeros(3), torch.ones(3), channels=8,
            dropout=0.0, history_events=8, relational_evidence=True,
        ),
        DirectRotationTrajectoryExpert(
            torch.zeros(3), torch.ones(3), channels=8,
            dropout=0.0, history_events=8,
            local_relational_evidence=True,
        ),
    )
    for model in models:
        output = _forward(model.eval(), inputs, state)
        assert torch.equal(output["direction_sign"], torch.full((2,), -1.0))
        assert bool(output["direction_valid"].all())
        assert torch.equal(output["position_m"][:, 0], state["q0_m"])
        assert not any("direction" in name for name, _ in model.named_parameters())
        distance = torch.cdist(output["position_m"], output["position_m"])
        q0_distance = torch.cdist(state["q0_m"], state["q0_m"])
        assert torch.allclose(
            distance, q0_distance[:, None].expand_as(distance), atol=2e-6,
        )


def test_rotation_ab_models_are_c4_equivariant() -> None:
    inputs, state = _case(1.0)
    models = (
        ParametricRotationFutureExpertV2(
            torch.zeros(3), torch.ones(3), channels=8,
            dropout=0.0, history_events=8,
        ),
        DirectRotationTrajectoryExpert(
            torch.zeros(3), torch.ones(3), channels=8,
            dropout=0.0, history_events=8,
        ),
        ParametricRotationFutureExpertV2(
            torch.zeros(3), torch.ones(3), channels=8,
            dropout=0.0, history_events=8, relational_evidence=True,
        ),
        DirectRotationTrajectoryExpert(
            torch.zeros(3), torch.ones(3), channels=8,
            dropout=0.0, history_events=8, relational_evidence=True,
        ),
        DirectRotationTrajectoryExpert(
            torch.zeros(3), torch.ones(3), channels=8,
            dropout=0.0, history_events=8,
            local_relational_evidence=True,
        ),
    )
    for model in models:
        model.eval()
        reference = _forward(model, inputs, state)
        for shift in (1, 2, 3):
            shifted_inputs = dict(inputs)
            for name in ("obs", "obs_mask", "primary_mask"):
                shifted_inputs[name] = torch.roll(inputs[name], shifts=shift, dims=2)
            shifted_state = {}
            for name, value in state.items():
                shifted_state[name] = (
                    torch.roll(value, shifts=shift, dims=1)
                    if value.ndim >= 2 and value.shape[1] == 4 else value
                )
            shifted_state["primary_index"] = (state["primary_index"] + shift) % 4
            output = _forward(model, shifted_inputs, shifted_state)
            assert torch.allclose(
                output["position_m"],
                torch.roll(reference["position_m"], shifts=shift, dims=2),
                atol=2e-6,
            )
            assert torch.equal(output["direction_sign"], reference["direction_sign"])


def test_rotation_ab_loss_has_no_direction_term() -> None:
    inputs, state = _case(1.0)
    for architecture, model in (
        ("parametric_v2", ParametricRotationFutureExpertV2(
            torch.zeros(3), torch.ones(3), channels=8,
            dropout=0.0, history_events=8,
        )),
        ("direct_trajectory", DirectRotationTrajectoryExpert(
            torch.zeros(3), torch.ones(3), channels=8,
            dropout=0.0, history_events=8,
        )),
        ("parametric_relational_v3", ParametricRotationFutureExpertV2(
            torch.zeros(3), torch.ones(3), channels=8,
            dropout=0.0, history_events=8, relational_evidence=True,
        )),
        ("direct_relational_trajectory", DirectRotationTrajectoryExpert(
            torch.zeros(3), torch.ones(3), channels=8,
            dropout=0.0, history_events=8, relational_evidence=True,
        )),
        ("direct_ordered_relational_trajectory", DirectRotationTrajectoryExpert(
            torch.zeros(3), torch.ones(3), channels=8,
            dropout=0.0, history_events=8,
            local_relational_evidence=True,
        )),
    ):
        prediction = _forward(model.eval(), inputs, state)
        truth = prediction["position_m"].detach().clone()
        total, parts = cyclic_rotation_ab_loss(
            prediction, state, truth, inputs["tau"],
            torch.ones(truth.shape[:2], dtype=torch.bool),
            architecture=architecture,
        )
        assert torch.isfinite(total)
        assert "direction_loss" not in parts
        assert "omega_sign" not in parts
        assert float(parts["direction_coverage"]) == 1.0


def test_direct_rigid_projection_gradients_are_finite() -> None:
    inputs, state = _case(1.0)
    model = DirectRotationTrajectoryExpert(
        torch.zeros(3), torch.ones(3), channels=8,
        dropout=0.0, history_events=8,
    )
    prediction = _forward(model, inputs, state)
    truth = prediction["position_m"].detach() + 0.01
    total, _ = cyclic_rotation_ab_loss(
        prediction, state, truth, inputs["tau"],
        torch.ones(truth.shape[:2], dtype=torch.bool),
        architecture="direct_trajectory",
    )
    total.backward()
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_unsigned_relational_evidence_does_not_encode_direction() -> None:
    positive, _ = _case(1.0)
    negative, _ = _case(-1.0)
    encoder = UnsignedRelationalMotionEncoder(
        channels=8, dropout=0.0, history_events=8,
    ).eval()
    positive_relation, positive_edge, positive_curve = encoder(
        positive["obs"], positive["obs_mask"], positive["event_mask"],
        positive["event_time_s"], torch.zeros(3), torch.ones(3),
    )
    negative_relation, negative_edge, negative_curve = encoder(
        negative["obs"], negative["obs_mask"], negative["event_mask"],
        negative["event_time_s"], torch.zeros(3), torch.ones(3),
    )
    assert not bool(positive_edge.any())
    assert bool(positive_curve.all())
    assert torch.equal(positive_edge, negative_edge)
    assert torch.equal(positive_curve, negative_curve)
    assert torch.allclose(positive_relation, negative_relation, atol=1e-6)


def test_relational_evidence_ignores_masked_coordinates() -> None:
    inputs, _ = _case(1.0)
    encoder = UnsignedRelationalMotionEncoder(
        channels=8, dropout=0.0, history_events=8,
    ).eval()
    reference = encoder(
        inputs["obs"], inputs["obs_mask"], inputs["event_mask"],
        inputs["event_time_s"], torch.zeros(3), torch.ones(3),
    )[0]
    changed = inputs["obs"].clone()
    changed[~inputs["obs_mask"]] = 1e6
    actual = encoder(
        changed, inputs["obs_mask"], inputs["event_mask"],
        inputs["event_time_s"], torch.zeros(3), torch.ones(3),
    )[0]
    assert torch.allclose(reference, actual, atol=1e-6)


def test_direct_relational_rigid_projection_gradients_are_finite() -> None:
    inputs, state = _case(1.0)
    model = DirectRotationTrajectoryExpert(
        torch.zeros(3), torch.ones(3), channels=8,
        dropout=0.0, history_events=8, relational_evidence=True,
    )
    prediction = _forward(model, inputs, state)
    truth = prediction["position_m"].detach() + 0.01
    total, _ = cyclic_rotation_ab_loss(
        prediction, state, truth, inputs["tau"],
        torch.ones(truth.shape[:2], dtype=torch.bool),
        architecture="direct_relational_trajectory",
    )
    total.backward()
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_ordered_local_relation_is_unsigned_masked_and_c4_equivariant() -> None:
    positive, _ = _case(1.0)
    negative, _ = _case(-1.0)
    encoder = OrderedLocalRelationalMotionEncoder(
        channels=8, dropout=0.0, history_events=8,
    ).eval()
    reference = encoder(
        positive["obs"], positive["obs_mask"], positive["event_mask"],
        positive["event_time_s"], torch.zeros(3), torch.ones(3),
    )
    reversed_relation = encoder(
        negative["obs"], negative["obs_mask"], negative["event_mask"],
        negative["event_time_s"], torch.zeros(3), torch.ones(3),
    )
    relation, edge_sample, curve_sample, edge_track, curve_track = reference
    assert relation.shape == (2, 4, 8)
    assert edge_sample.shape == curve_sample.shape == (2,)
    assert edge_track.shape == curve_track.shape == (2, 4)
    assert not bool(edge_sample.any())
    assert bool(curve_sample.all())
    assert torch.equal(curve_track, torch.tensor(((True, False, False, False),) * 2))
    assert torch.equal(edge_track, torch.zeros_like(edge_track))
    assert torch.equal(relation[:, 1:], torch.zeros_like(relation[:, 1:]))
    assert torch.allclose(relation, reversed_relation[0], atol=1e-6)

    changed = positive["obs"].clone()
    changed[~positive["obs_mask"]] = 1e6
    masked = encoder(
        changed, positive["obs_mask"], positive["event_mask"],
        positive["event_time_s"], torch.zeros(3), torch.ones(3),
    )[0]
    assert torch.allclose(relation, masked, atol=1e-6)

    shifted = encoder(
        torch.roll(positive["obs"], 1, 2),
        torch.roll(positive["obs_mask"], 1, 2),
        positive["event_mask"], positive["event_time_s"],
        torch.zeros(3), torch.ones(3),
    )
    assert torch.allclose(shifted[0], torch.roll(relation, 1, 1), atol=1e-6)
    assert torch.equal(shifted[3], torch.roll(edge_track, 1, 1))
    assert torch.equal(shifted[4], torch.roll(curve_track, 1, 1))


def test_ordered_local_direct_loss_uses_relation_without_forbidden_outputs() -> None:
    inputs, state = _case(1.0)
    model = DirectRotationTrajectoryExpert(
        torch.zeros(3), torch.ones(3), channels=8,
        dropout=0.0, history_events=8,
        local_relational_evidence=True,
    )
    prediction = _forward(model, inputs, state)
    forbidden = {
        "center", "radius", "phase", "slot", "velocity", "omega",
        "acceleration",
    }
    assert not any(
        any(word in name for word in forbidden) for name in prediction
    )
    assert torch.equal(prediction["position_m"][:, 0], state["q0_m"])
    assert torch.equal(
        prediction["relation_edge_chord_ratio"][:, 0],
        torch.zeros_like(prediction["relation_edge_chord_ratio"][:, 0]),
    )
    assert torch.equal(
        prediction["relation_edge_chord_ratio"][:, :, 1:],
        torch.zeros_like(prediction["relation_edge_chord_ratio"][:, :, 1:]),
    )
    truth = prediction["position_m"].detach().clone()
    truth[:, 3, 0, 0] += 0.08
    rule = torch.ones(truth.shape[:2], dtype=torch.bool)
    total, parts = cyclic_rotation_ab_loss(
        prediction, state, truth, inputs["tau"], rule,
        architecture="direct_ordered_relational_trajectory",
        rigid_weight=0.0,
        edge_chord_weight=0.10,
        relation_probe_weight=0.05,
        q3_tail_weight=0.10,
    )
    assert torch.isfinite(total)
    assert float(parts["edge_chord"]) > 0
    assert float(parts["q3_tail"]) > 0
    total.backward()
    relation_gradients = [
        parameter.grad for name, parameter in model.named_parameters()
        if name.startswith("relation.") and parameter.grad is not None
    ]
    assert relation_gradients
    assert all(torch.isfinite(gradient).all() for gradient in relation_gradients)
    assert any(bool((gradient.abs() > 0).any()) for gradient in relation_gradients)

    q3_disabled = rule.clone()
    q3_disabled[:, 3] = False
    _, disabled_parts = cyclic_rotation_ab_loss(
        prediction, state, truth, inputs["tau"], q3_disabled,
        architecture="direct_ordered_relational_trajectory",
        rigid_weight=0.0,
        edge_chord_weight=0.10,
        relation_probe_weight=0.05,
        q3_tail_weight=0.10,
    )
    assert float(disabled_parts["q3_tail"]) == 0.0


def test_ordered_relation_probe_ignores_unsupported_edge_truth() -> None:
    inputs, state = _case(1.0)
    state = dict(state)
    state["q0_valid"] = torch.ones_like(state["q0_valid"])
    state["anchor_composed"] = ~state["current_visible"]
    model = DirectRotationTrajectoryExpert(
        torch.zeros(3), torch.ones(3), channels=8,
        dropout=0.0, history_events=8,
        local_relational_evidence=True,
    ).eval()
    prediction = _forward(model, inputs, state)
    truth = prediction["position_m"].detach().clone()
    rule = torch.ones(truth.shape[:2], dtype=torch.bool)
    _, reference = cyclic_rotation_ab_loss(
        prediction, state, truth, inputs["tau"], rule,
        architecture="direct_ordered_relational_trajectory",
        relation_probe_weight=0.05,
    )
    changed = truth.clone()
    changed[:, 1:, 2:, :2] += 100.0
    _, actual = cyclic_rotation_ab_loss(
        prediction, state, changed, inputs["tau"], rule,
        architecture="direct_ordered_relational_trajectory",
        relation_probe_weight=0.05,
    )
    assert torch.allclose(reference["relation_probe"], actual["relation_probe"])
