from __future__ import annotations

from itertools import combinations

import torch

from training.stage3.locally_weighted_profiled_twist import (
    LocallyWeightedProfiledTwistAtOmega,
)
from training.stage3.prequential_local_precision_irls import (
    PAIR_INDICES,
    PrequentialLocalPrecisionIRLS,
)


FORBIDDEN_FORWARD_NAMES = ("truth", "session", "class", "future")


def _rigid_fixture(batch_size: int = 2, events: int = 9):
    time_s = torch.linspace(-0.08, 0.0, events).expand(batch_size, -1).clone()
    center = torch.tensor([
        [0.18, -0.04, 0.015], [-0.11, 0.09, -0.025],
    ])[:batch_size]
    radial = torch.tensor([
        [-0.22, 0.00, 0.010], [0.00, 0.29, -0.005],
        [0.22, 0.00, -0.010], [0.00, -0.29, 0.005],
    ])
    q0 = center[:, None] + radial[None]
    truth_velocity = torch.tensor([
        [0.62, -0.31, 0.075], [-0.28, 0.47, -0.045],
    ])[:batch_size]
    omega = torch.tensor([4.5, -5.7])[:batch_size]
    observations = torch.zeros(batch_size, events, 4, 3)
    eye = torch.eye(2)
    for row in range(batch_size):
        for event in range(events):
            theta = omega[row] * time_s[row, event]
            rotation = torch.stack((
                torch.stack((torch.cos(theta), -torch.sin(theta))),
                torch.stack((torch.sin(theta), torch.cos(theta))),
            ))
            observations[row, event, :, :2] = (
                (eye - rotation) @ center[row, :2]
                + time_s[row, event] * truth_velocity[row, :2]
                + (rotation @ q0[row, :, :2].T).T
            )
            observations[row, event, :, 2] = (
                q0[row, :, 2]
                + time_s[row, event] * truth_velocity[row, 2]
            )
    generator = torch.Generator().manual_seed(391)
    observations = observations + 0.002 * torch.randn(
        observations.shape, generator=generator,
    )
    observation_mask = torch.ones(batch_size, events, 4, dtype=torch.bool)
    event_mask = torch.ones(batch_size, events, dtype=torch.bool)
    anchor_mask = torch.ones(batch_size, 4, dtype=torch.bool)
    pair_mask = torch.stack([
        observation_mask[:, :, left] & observation_mask[:, :, right]
        for left, right in PAIR_INDICES
    ], dim=2)

    planar_norm = torch.linalg.vector_norm(observations[..., :2], dim=-1)
    spatial_norm = torch.linalg.vector_norm(observations, dim=-1)
    event_feature = torch.stack((
        time_s[:, :, None].expand(-1, -1, 4),
        time_s[:, :, None].square().expand(-1, -1, 4),
        planar_norm, observations[..., 2], spatial_norm,
        planar_norm.square(),
    ), dim=-1)
    anchor_planar_norm = torch.linalg.vector_norm(q0[..., :2], dim=-1)
    anchor_feature = torch.stack((
        anchor_planar_norm, q0[..., 2],
        torch.linalg.vector_norm(q0, dim=-1),
        anchor_planar_norm.square(),
    ), dim=-1)
    pair_rows = []
    for left, right in PAIR_INDICES:
        delta = observations[:, :, left] - observations[:, :, right]
        pair_rows.append(torch.stack((
            torch.linalg.vector_norm(delta[..., :2], dim=-1),
            delta[..., 2].abs(), torch.linalg.vector_norm(delta, dim=-1),
            (observations[:, :, left, :2] * observations[:, :, right, :2]).sum(
                dim=-1,
            ),
        ), dim=-1))
    pair_feature = torch.stack(pair_rows, dim=2)
    global_feature = torch.stack((
        time_s[:, -1] - time_s[:, 0],
        observation_mask.sum(dim=(1, 2)).to(torch.float32)
        / float(events * 4),
    ), dim=-1)
    network_features = {
        "event_feature": event_feature,
        "event_mask": observation_mask,
        "anchor_feature": anchor_feature,
        "anchor_mask": anchor_mask,
        "pair_feature": pair_feature,
        "pair_mask": pair_mask,
        "global_feature": global_feature,
    }
    solver_features = {
        "history_obs_rel_m": observations,
        "history_obs_mask": observation_mask,
        "history_event_mask": event_mask,
        "history_time_s": time_s,
        "omega_rad_s": omega,
        "center_prior": {
            "center_supported": torch.ones(batch_size, dtype=torch.bool),
            "center_offset_m": center,
            "center_log_variance_xy_z": torch.full((batch_size, 2), -3.2),
            "q0_relation_m": q0,
        },
    }
    return network_features, solver_features, truth_velocity


def _network(
    features: dict[str, torch.Tensor],
) -> PrequentialLocalPrecisionIRLS:
    torch.manual_seed(811)
    model = PrequentialLocalPrecisionIRLS(
        event_width=6, anchor_width=4, pair_width=4,
        global_width=2, width=20,
    )
    model.fit_normalizer(**features)
    return model


def _solve(
    solver: LocallyWeightedProfiledTwistAtOmega,
    features: dict[str, object],
    weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return solver(
        features["history_obs_rel_m"],
        features["history_obs_mask"],
        features["history_event_mask"],
        features["history_time_s"],
        features["omega_rad_s"],
        features["center_prior"],
        observation_log_precision=weights["observation_log_precision"],
        anchor_alpha=weights["anchor_alpha"],
        center_alpha=weights["center_alpha"],
    )


def _run_two_step(
    model: PrequentialLocalPrecisionIRLS,
    solver: LocallyWeightedProfiledTwistAtOmega,
    network_features: dict[str, torch.Tensor],
    solver_features: dict[str, object],
):
    solver_steps: list[dict[str, torch.Tensor]] = []

    def callback(step: int, weights: dict[str, torch.Tensor]):
        result = _solve(solver, solver_features, weights)
        solver_steps.append(result)
        if step == 0:
            residual = result["weighted_residual_m"]
            precision = result["observation_precision"]
            residual_square = (
                precision.unsqueeze(-1) * residual.square()
            ).sum(dim=(1, 2, 3))
            fitted_coordinate_count = (
                3.0 * result["observation_precision_sum"]
            ).clamp_min(1.0)
            fit_only_scale = torch.sqrt(
                residual_square / fitted_coordinate_count
            ).clamp_min(1e-4)
            standardized_residual = torch.sqrt(
                residual.square().sum(dim=-1) + 1e-12
            ) / fit_only_scale[:, None, None]
            standardized_residual = torch.where(
                solver_features["history_obs_mask"], standardized_residual,
                torch.zeros_like(standardized_residual),
            )
            return {
                "standardized_residual": standardized_residual,
                "leverage": result["velocity_leverage"],
            }
        return result

    output = model.run_two_step_irls(callback, **network_features)
    return output, solver_steps


def _pair_permutation(permutation: torch.Tensor) -> torch.Tensor:
    lookup = {pair: index for index, pair in enumerate(combinations(range(4), 2))}
    return torch.tensor([
        lookup[tuple(sorted((int(permutation[left]), int(permutation[right]))))]
        for left, right in PAIR_INDICES
    ])


def test_real_two_step_solver_connection_backpropagates_truth_velocity_loss() -> None:
    network_features, solver_features, truth_velocity = _rigid_fixture()
    assert not any(
        forbidden in name.lower()
        for name in network_features
        for forbidden in FORBIDDEN_FORWARD_NAMES
    )
    assert not any(
        forbidden in name.lower()
        for name in solver_features
        for forbidden in FORBIDDEN_FORWARD_NAMES
    )
    assert not any(
        forbidden in name.lower()
        for name in solver_features["center_prior"]
        for forbidden in FORBIDDEN_FORWARD_NAMES
    )
    model = _network(network_features).train()
    solver = LocallyWeightedProfiledTwistAtOmega().train()
    output, solver_steps = _run_two_step(
        model, solver, network_features, solver_features,
    )
    assert len(solver_steps) == 2
    for result in solver_steps:
        assert bool(result["profile_supported"].all())
        assert bool(result["state_supported"].all())
        for name in (
            "velocity_mps", "weighted_residual_m", "velocity_leverage",
            "observation_precision",
        ):
            assert bool(torch.isfinite(result[name]).all()), name
    velocity = output["solution"]["velocity_mps"]
    loss = (velocity - truth_velocity).square().mean()
    assert bool(torch.isfinite(loss))
    loss.backward()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert parameters and all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )


@torch.inference_mode()
def test_connected_two_step_pipeline_is_s4_invariant() -> None:
    network_features, solver_features, _ = _rigid_fixture()
    model = _network(network_features).eval()
    solver = LocallyWeightedProfiledTwistAtOmega().eval()
    reference, _ = _run_two_step(
        model, solver, network_features, solver_features,
    )

    permutation = torch.tensor([2, 0, 3, 1])
    pair_order = _pair_permutation(permutation)
    changed_network = dict(network_features)
    changed_network["event_feature"] = network_features["event_feature"][:, :, permutation]
    changed_network["event_mask"] = network_features["event_mask"][:, :, permutation]
    changed_network["anchor_feature"] = network_features["anchor_feature"][:, permutation]
    changed_network["anchor_mask"] = network_features["anchor_mask"][:, permutation]
    changed_network["pair_feature"] = network_features["pair_feature"][:, :, pair_order]
    changed_network["pair_mask"] = network_features["pair_mask"][:, :, pair_order]
    changed_solver = dict(solver_features)
    changed_solver["history_obs_rel_m"] = solver_features[
        "history_obs_rel_m"
    ][:, :, permutation]
    changed_solver["history_obs_mask"] = solver_features[
        "history_obs_mask"
    ][:, :, permutation]
    changed_solver["center_prior"] = dict(solver_features["center_prior"])
    changed_solver["center_prior"]["q0_relation_m"] = solver_features[
        "center_prior"
    ]["q0_relation_m"][:, permutation]
    actual, _ = _run_two_step(
        model, solver, changed_network, changed_solver,
    )
    torch.testing.assert_close(
        actual["solution"]["velocity_mps"],
        reference["solution"]["velocity_mps"],
        atol=3e-5, rtol=3e-5,
    )


def test_all_unsupported_q0_uses_zero_alphas_and_history_profile() -> None:
    network_features, solver_features, _ = _rigid_fixture()
    network_features = dict(network_features)
    network_features["anchor_mask"] = torch.zeros_like(
        network_features["anchor_mask"],
    )
    solver_features = dict(solver_features)
    solver_features["center_prior"] = dict(solver_features["center_prior"])
    solver_features["center_prior"]["center_supported"] = torch.zeros_like(
        solver_features["center_prior"]["center_supported"],
    )
    model = _network(network_features).train()
    solver = LocallyWeightedProfiledTwistAtOmega().train()
    output, solver_steps = _run_two_step(
        model, solver, network_features, solver_features,
    )
    for weights in (output["step1"], output["step2"]):
        assert torch.equal(
            weights["anchor_alpha"], torch.zeros_like(weights["anchor_alpha"]),
        )
        assert torch.equal(
            weights["center_alpha"], torch.zeros_like(weights["center_alpha"]),
        )
    for result in solver_steps:
        assert not bool(result["q0_prior_used"].any())
        assert bool(result["profile_supported"].all())
        assert bool(result["state_supported"].all())
        assert bool(torch.isfinite(result["velocity_mps"]).all())
