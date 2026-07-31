from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F

from training.stage3.endpoint_token_reliability_fusion import (
    EndpointTokenReliabilityFusion,
    endpoint_token_features,
)
from training.stage3.frozen_expert_reliability_fusion import (
    oracle_projection_coefficient,
)
from training.stage3.screen_endpoint_token_reliability_oracle_omega import (
    _model_kwargs,
)


def _rotate_xy(value: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    result = value.clone()
    result[..., :2] = value[..., :2] @ matrix.T
    return result


def _fixture() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(19)
    batch, events = 8, 6
    time = torch.linspace(-0.12, 0.0, events).repeat(batch, 1)
    omega = torch.linspace(-5.0, 5.0, batch)
    q0 = torch.randn(batch, 4, 3, generator=generator) * 0.4
    center = torch.randn(batch, 3, generator=generator) * 0.2
    velocity = torch.randn(batch, 3, generator=generator) * 0.8
    theta = omega[:, None] * time
    cosine, sine = torch.cos(theta), torch.sin(theta)
    qx = q0[:, None, :, 0]
    qy = q0[:, None, :, 1]
    rotated_q = torch.stack((
        cosine[..., None] * qx - sine[..., None] * qy,
        sine[..., None] * qx + cosine[..., None] * qy,
        q0[:, None, :, 2].expand(-1, events, -1),
    ), dim=-1)
    cx, cy = center[:, 0, None], center[:, 1, None]
    center_term = torch.stack((
        (1.0 - cosine) * cx + sine * cy,
        -sine * cx + (1.0 - cosine) * cy,
        torch.zeros_like(cosine),
    ), dim=-1).unsqueeze(2)
    observation = (
        rotated_q + center_term
        + time[:, :, None, None] * velocity[:, None, None]
    )
    observation = observation + 0.002 * torch.randn(
        observation.shape, generator=generator,
    )
    mask = torch.ones(batch, events, 4, dtype=torch.bool)
    mask[0, :2, 2] = False
    event_mask = torch.ones(batch, events, dtype=torch.bool)
    primary = torch.zeros(batch, events, 4, dtype=torch.bool)
    primary[:, :3, 0] = True
    primary[:, 3:, 1] = True
    q0_supported = torch.ones(batch, 4, dtype=torch.bool)
    pooled = torch.randn(batch, 13, generator=generator) * 0.1
    q0_velocity = velocity + 0.08 * torch.randn(
        velocity.shape, generator=generator,
    )
    history_velocity = velocity + 0.12 * torch.randn(
        velocity.shape, generator=generator,
    )
    q0_center = center + 0.02 * torch.randn(
        center.shape, generator=generator,
    )
    history_center = center + 0.03 * torch.randn(
        center.shape, generator=generator,
    )
    feature_args = {
        "history_obs_rel_m": observation,
        "history_obs_mask": mask,
        "history_event_mask": event_mask,
        "history_time_s": time,
        "history_primary_mask": primary,
        "history_switch_step": torch.arange(events).repeat(batch, 1) // 3,
        "q0_relation_m": q0,
        "q0_supported": q0_supported,
        "omega_rad_s": omega,
        "q0_velocity_mps": q0_velocity,
        "history_velocity_mps": history_velocity,
        "q0_center_offset_m": q0_center,
        "history_center_offset_m": history_center,
        "pooled_diagnostic_feature": pooled,
    }
    experts = {
        "q0_velocity_mps": q0_velocity,
        "history_velocity_mps": history_velocity,
        "q0_supported": torch.ones(batch, dtype=torch.bool),
        "history_supported": torch.ones(batch, dtype=torch.bool),
        "fallback_velocity_mps": velocity,
        "fallback_supported": torch.ones(batch, dtype=torch.bool),
    }
    return feature_args, experts


def _model(
    features: dict[str, torch.Tensor],
) -> EndpointTokenReliabilityFusion:
    torch.manual_seed(7)
    model = EndpointTokenReliabilityFusion(width=12)
    model.fit_normalizer(
        **features,
        row_mask=torch.ones(features["global_feature"].shape[0], dtype=torch.bool),
    )
    return model.eval()


def test_endpoint_features_and_head_are_joint_s4_invariant() -> None:
    args, experts = _fixture()
    features = endpoint_token_features(**args)
    model = _model(features)
    reference = model(**features, **experts)
    permutation = torch.tensor([2, 0, 3, 1])
    changed = dict(args)
    for name in (
        "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
    ):
        changed[name] = changed[name][:, :, permutation]
    changed["q0_relation_m"] = changed["q0_relation_m"][:, permutation]
    changed["q0_supported"] = changed["q0_supported"][:, permutation]
    permuted = endpoint_token_features(**changed)
    result = model(**permuted, **experts)
    assert torch.allclose(reference["q0_weight"], result["q0_weight"], atol=1e-6)
    assert torch.allclose(reference["velocity_mps"], result["velocity_mps"], atol=1e-6)


def test_endpoint_features_are_planar_o2_invariant_and_fusion_equivariant() -> None:
    args, experts = _fixture()
    features = endpoint_token_features(**args)
    model = _model(features)
    reference = model(**features, **experts)
    for matrix in (
        torch.tensor([[0.6, -0.8], [0.8, 0.6]]),
        torch.tensor([[1.0, 0.0], [0.0, -1.0]]),
    ):
        changed = dict(args)
        for name in (
            "history_obs_rel_m", "q0_relation_m", "q0_velocity_mps",
            "history_velocity_mps", "q0_center_offset_m",
            "history_center_offset_m",
        ):
            changed[name] = _rotate_xy(changed[name], matrix)
        if torch.det(matrix) < 0:
            changed["omega_rad_s"] = -changed["omega_rad_s"]
        transformed = endpoint_token_features(**changed)
        transformed_experts = dict(experts)
        transformed_experts["q0_velocity_mps"] = changed["q0_velocity_mps"]
        transformed_experts["history_velocity_mps"] = changed[
            "history_velocity_mps"
        ]
        transformed_experts["fallback_velocity_mps"] = _rotate_xy(
            experts["fallback_velocity_mps"], matrix,
        )
        result = model(**transformed, **transformed_experts)
        assert torch.allclose(
            reference["q0_weight"], result["q0_weight"], atol=2e-5,
        )
        assert torch.allclose(
            _rotate_xy(reference["velocity_mps"], matrix),
            result["velocity_mps"], atol=2e-5,
        )


def test_endpoint_features_and_fusion_obey_common_velocity_ramp() -> None:
    args, experts = _fixture()
    features = endpoint_token_features(**args)
    model = _model(features)
    reference = model(**features, **experts)
    ramp = torch.tensor([1.3, -0.7, 0.2])
    changed = dict(args)
    changed["history_obs_rel_m"] = (
        args["history_obs_rel_m"]
        + args["history_time_s"][:, :, None, None] * ramp
    )
    changed["q0_velocity_mps"] = args["q0_velocity_mps"] + ramp
    changed["history_velocity_mps"] = args["history_velocity_mps"] + ramp
    ramped_features = endpoint_token_features(**changed)
    ramped_experts = dict(experts)
    ramped_experts["q0_velocity_mps"] = experts["q0_velocity_mps"] + ramp
    ramped_experts["history_velocity_mps"] = (
        experts["history_velocity_mps"] + ramp
    )
    ramped_experts["fallback_velocity_mps"] = (
        experts["fallback_velocity_mps"] + ramp
    )
    result = model(**ramped_features, **ramped_experts)
    assert torch.allclose(reference["q0_weight"], result["q0_weight"], atol=1e-6)
    assert torch.allclose(
        reference["velocity_mps"] + ramp, result["velocity_mps"], atol=2e-6,
    )


def test_independent_role_pairing_changes_endpoint_evidence() -> None:
    args, experts = _fixture()
    reference = endpoint_token_features(**args)
    model = _model(reference)
    reference_weight = model(**reference, **experts)["q0_weight"]
    changed = dict(args)
    changed["q0_relation_m"] = args["q0_relation_m"][:, [1, 0, 3, 2]]
    broken = endpoint_token_features(**changed)
    assert not torch.allclose(
        reference["event_feature"], broken["event_feature"], atol=1e-5,
    )
    assert not torch.allclose(
        reference["pair_feature"], broken["pair_feature"], atol=1e-5,
    )
    broken_weight = model(**broken, **experts)["q0_weight"]
    assert not torch.allclose(reference_weight, broken_weight, atol=1e-6)
    pair_only = dict(reference)
    pair_only["pair_feature"] = broken["pair_feature"]
    pair_only_weight = model(**pair_only, **experts)["q0_weight"]
    assert not torch.allclose(reference_weight, pair_only_weight, atol=1e-6)
    assert torch.allclose(
        model(**reference, **experts, ablate_local=True)["q0_weight"],
        model(**pair_only, **experts, ablate_local=True)["q0_weight"],
        atol=1e-7,
    )


def test_endpoint_support_override_and_local_ablation_are_explicit() -> None:
    args, experts = _fixture()
    features = endpoint_token_features(**args)
    model = _model(features)
    ablated = model(**features, **experts, ablate_local=True)
    full = model(**features, **experts)
    assert not torch.allclose(full["q0_weight"], ablated["q0_weight"])
    q0_only = dict(experts)
    q0_only["history_supported"] = torch.zeros_like(experts["history_supported"])
    assert torch.equal(
        model(**features, **q0_only)["q0_weight"],
        torch.ones_like(experts["q0_supported"], dtype=torch.float32),
    )
    history_only = dict(experts)
    history_only["q0_supported"] = torch.zeros_like(experts["q0_supported"])
    assert torch.equal(
        model(**features, **history_only)["q0_weight"],
        torch.zeros_like(experts["q0_supported"], dtype=torch.float32),
    )
    fallback = dict(experts)
    fallback["q0_supported"] = torch.zeros_like(experts["q0_supported"])
    fallback["history_supported"] = torch.zeros_like(
        experts["history_supported"]
    )
    fallback_result = model(**features, **fallback)
    assert torch.all(fallback_result["fallback_used"])
    assert torch.all(fallback_result["state_supported"])
    assert not torch.any(fallback_result["learned_gate_valid"])
    assert torch.equal(
        fallback_result["velocity_mps"], fallback["fallback_velocity_mps"],
    )
    unsupported = dict(fallback)
    unsupported["fallback_supported"] = torch.zeros_like(
        fallback["fallback_supported"]
    )
    unsupported_result = model(**features, **unsupported)
    assert not torch.any(unsupported_result["fallback_used"])
    assert not torch.any(unsupported_result["state_supported"])
    assert not torch.any(unsupported_result["learned_gate_valid"])
    assert torch.equal(
        unsupported_result["velocity_mps"],
        torch.zeros_like(unsupported["fallback_velocity_mps"]),
    )


def test_inferred_roles_remain_visible_when_a_different_q0_role_is_supported() -> None:
    args, experts = _fixture()
    args["q0_supported"] = torch.zeros_like(args["q0_supported"])
    args["q0_supported"][:, 0] = True
    args["history_obs_mask"] = torch.zeros_like(args["history_obs_mask"])
    args["history_obs_mask"][:, :, 1:3] = True
    features = endpoint_token_features(**args)
    assert torch.all(features["event_mask"][:, :, 1:3])
    assert not torch.any(features["event_mask"][:, :, (0, 3)])
    assert torch.all(features["role_mask"][:, 1:3])
    assert not torch.any(features["role_mask"][:, (0, 3)])
    # Pair index (1,2) is the fourth of combinations(range(4), 2).
    assert torch.all(features["pair_mask"][:, 3])
    model = _model(features)
    reference = model(**features, **experts)["q0_weight"]
    permutation = torch.tensor([2, 0, 3, 1])
    changed = dict(args)
    for name in (
        "history_obs_rel_m", "history_obs_mask", "history_primary_mask",
    ):
        changed[name] = changed[name][:, :, permutation]
    changed["q0_relation_m"] = changed["q0_relation_m"][:, permutation]
    changed["q0_supported"] = changed["q0_supported"][:, permutation]
    permuted = endpoint_token_features(**changed)
    assert torch.allclose(
        reference, model(**permuted, **experts)["q0_weight"], atol=1e-6,
    )


def test_truth_velocity_is_loss_only_and_optimizer_contains_only_head() -> None:
    args, experts = _fixture()
    features = endpoint_token_features(**args)
    model = _model(features).train()
    first = model(**features, **experts)
    truth_a = torch.randn_like(experts["q0_velocity_mps"])
    truth_b = truth_a + 10.0
    oracle_a = oracle_projection_coefficient(
        truth_a, experts["q0_velocity_mps"], experts["history_velocity_mps"],
        experts["q0_supported"], experts["history_supported"],
    )
    oracle_b = oracle_projection_coefficient(
        truth_b, experts["q0_velocity_mps"], experts["history_velocity_mps"],
        experts["q0_supported"], experts["history_supported"],
    )
    second = model(**features, **experts)
    assert torch.equal(first["q0_weight"], second["q0_weight"])
    assert not torch.equal(oracle_a["weight"], oracle_b["weight"])
    loss_a = F.smooth_l1_loss(first["q0_weight"], oracle_a["weight"]) + (
        first["velocity_mps"] - truth_a
    ).square().sum(dim=-1).sqrt().mean()
    loss_b = F.smooth_l1_loss(second["q0_weight"], oracle_b["weight"]) + (
        second["velocity_mps"] - truth_b
    ).square().sum(dim=-1).sqrt().mean()
    assert not torch.equal(loss_a, loss_b)
    loss_a.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    forbidden = {"truth", "session", "motion", "future", "physical_id"}
    assert not any(
        any(token in name for token in forbidden)
        for name in inspect.signature(model.forward).parameters
    )
    mock_batch = {
        **features,
        "q0_velocity": experts["q0_velocity_mps"],
        "history_velocity": experts["history_velocity_mps"],
        "q0_supported": experts["q0_supported"],
        "history_supported": experts["history_supported"],
        "fallback_velocity": experts["fallback_velocity_mps"],
        "fallback_supported": experts["fallback_supported"],
        "target_velocity": truth_a,
        "motion_class": torch.zeros(truth_a.shape[0]),
        "session": torch.zeros(truth_a.shape[0]),
    }
    assert set(_model_kwargs(mock_batch)) == {
        "event_feature", "event_mask", "role_feature", "role_mask",
        "pair_feature", "pair_mask", "global_feature",
        "q0_velocity_mps", "history_velocity_mps", "q0_supported",
        "history_supported", "fallback_velocity_mps", "fallback_supported",
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert {
        id(parameter) for group in optimizer.param_groups
        for parameter in group["params"]
    } == {id(parameter) for parameter in model.parameters()}
