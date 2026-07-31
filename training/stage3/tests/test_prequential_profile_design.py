from __future__ import annotations

from typing import Any

import pytest
import torch

from training.stage3.prequential_profile_design import (
    profiled_twist_dense_design,
)
from training.stage3.prequential_profile_diagnostics import (
    linear_gaussian_crossfit_diagnostics,
)
from training.stage3.profiled_center_twist_future import (
    ProfiledRigidTwistAtOmega,
)


def _rotation(theta: torch.Tensor) -> torch.Tensor:
    cosine, sine = torch.cos(theta), torch.sin(theta)
    return torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
        *theta.shape, 2, 2,
    )


def _synthetic_problem(
    *, batch: int = 4, events: int = 12,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(20260731)
    time = torch.linspace(-0.22, 0.0, events, dtype=dtype).expand(batch, -1).clone()
    omega = torch.tensor([2.1, -3.0, 4.2, -1.6], dtype=dtype)[:batch]
    center = torch.randn(batch, 3, dtype=dtype, generator=generator) * 0.15
    velocity = torch.randn(batch, 3, dtype=dtype, generator=generator) * 1.2
    q = torch.randn(batch, 4, 3, dtype=dtype, generator=generator) * 0.35
    rotation = _rotation(omega[:, None] * time)
    eye = torch.eye(2, dtype=dtype).view(1, 1, 2, 2)
    center_term = torch.einsum(
        "btij,bj->bti", eye - rotation, center[:, :2],
    )
    rotated_q = torch.einsum("btij,brj->btri", rotation, q[..., :2])
    planar = (
        center_term[:, :, None]
        + time[:, :, None, None] * velocity[:, None, None, :2]
        + rotated_q
    )
    vertical = (
        time[:, :, None] * velocity[:, None, 2:3]
        + q[:, None, :, 2]
    )
    observation = torch.cat((planar, vertical.unsqueeze(-1)), dim=-1)
    observation = observation + 0.002 * torch.randn(
        observation.shape, dtype=dtype, generator=generator,
    )
    mask = torch.rand(batch, events, 4, generator=generator) > 0.18
    mask[:, 0] = True
    mask[:, -1] = True
    event_mask = mask.any(dim=-1)
    center_prior = {
        "center_offset_m": center + 0.02 * torch.randn(
            center.shape, dtype=dtype, generator=generator,
        ),
        "center_supported": torch.tensor(
            [True, True, False, True], dtype=torch.bool,
        )[:batch],
        "center_log_variance_xy_z": torch.tensor(
            [[-2.0, -1.5], [-1.2, -1.0], [-0.8, -0.5], [-2.5, -2.0]],
            dtype=dtype,
        )[:batch],
        "q0_relation_m": q + 0.015 * torch.randn(
            q.shape, dtype=dtype, generator=generator,
        ),
    }
    return {
        "history_obs_rel_m": observation,
        "history_obs_mask": mask,
        "history_event_mask": event_mask,
        "history_time_s": time,
        "omega_rad_s": omega,
        "center_prior": center_prior,
    }


def _dense_solve(system: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    solutions: list[torch.Tensor] = []
    supported: list[torch.Tensor] = []
    for row in range(system["design"].shape[0]):
        selected = torch.nonzero(system["mask"][row], as_tuple=False).flatten()
        design = system["design"][row, selected]
        target = system["target"][row, selected]
        normal = (
            design.transpose(0, 1) @ design + system["prior_precision"][row]
        )
        rhs = design.transpose(0, 1) @ target + system["prior_natural"][row]
        factor, info = torch.linalg.cholesky_ex(normal)
        valid = (
            bool((info == 0).item())
            and bool((
                torch.isfinite(torch.diagonal(factor))
                & (torch.diagonal(factor) > 0)
            ).all())
        )
        if valid:
            solution = torch.cholesky_solve(rhs[:, None], factor).squeeze(-1)
            valid = bool(torch.isfinite(solution).all())
        else:
            solution = target.new_zeros(normal.shape[0])
        solutions.append(solution if valid else torch.zeros_like(solution))
        supported.append(torch.tensor(valid, device=target.device))
    return torch.stack(solutions), torch.stack(supported).to(torch.bool)


def _compare_dense_and_v14(
    problem: dict[str, Any], *, use_q0_prior: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    profile = ProfiledRigidTwistAtOmega().to(problem["history_obs_rel_m"].dtype)
    frozen = profile(**problem, use_q0_prior=use_q0_prior)
    design = profiled_twist_dense_design(
        **problem, use_q0_prior=use_q0_prior,
    )
    xy_solution, xy_supported = _dense_solve(design["xy"])
    z_solution, z_supported = _dense_solve(design["z"])
    dense_velocity = torch.cat((xy_solution[:, 2:4], z_solution[:, :1]), dim=-1)
    dense_supported = xy_supported & z_supported
    assert bool(frozen["profile_supported"].all())
    assert torch.equal(dense_supported, frozen["profile_supported"])
    torch.testing.assert_close(
        dense_velocity, frozen["velocity_mps"], atol=2e-10, rtol=2e-10,
    )
    return dense_velocity, frozen["velocity_mps"]


@pytest.mark.parametrize("use_q0_prior", [False, True])
def test_dense_xy_z_design_reproduces_frozen_v14_random_profiles(
    use_q0_prior: bool,
) -> None:
    problem = _synthetic_problem()
    design = profiled_twist_dense_design(
        **problem, use_q0_prior=use_q0_prior,
    )
    assert design["xy"]["design"].shape == (4, 12 * 4 * 2, 12)
    assert design["z"]["design"].shape == (4, 12 * 4, 5)
    assert design["xy"]["velocity_column_mask"].tolist() == (
        [False, False, True, True] + [False] * 8
    )
    assert design["z"]["velocity_column_mask"].tolist() == [
        True, False, False, False, False,
    ]
    _compare_dense_and_v14(problem, use_q0_prior=use_q0_prior)


def test_history_and_q0_prior_natural_parameters_match_v14() -> None:
    problem = _synthetic_problem(batch=2)
    history = profiled_twist_dense_design(
        **problem, use_q0_prior=False,
    )
    torch.testing.assert_close(
        torch.diagonal(history["xy"]["prior_precision"], dim1=-2, dim2=-1)[:, :2],
        torch.full((2, 2), 0.01, dtype=torch.float64),
    )
    torch.testing.assert_close(
        torch.diagonal(history["xy"]["prior_precision"], dim1=-2, dim2=-1)[:, 4:],
        torch.full((2, 8), 1e-5, dtype=torch.float64),
    )
    assert torch.count_nonzero(history["xy"]["prior_natural"]).item() == 0
    assert torch.count_nonzero(history["z"]["prior_natural"]).item() == 0

    informed = profiled_twist_dense_design(
        **problem, use_q0_prior=True,
    )
    log_variance = problem["center_prior"]["center_log_variance_xy_z"][:, 0]
    expected_center_precision = 25.0 * (
        0.15**2 * torch.exp(-log_variance)
    ).clamp(0.01 / 25.0, 40.0)
    torch.testing.assert_close(
        informed["center_precision_used"], expected_center_precision,
    )
    torch.testing.assert_close(
        informed["endpoint_precision_used"],
        torch.full((2,), 10.0, dtype=torch.float64),
    )
    expected_center_natural = (
        expected_center_precision[:, None]
        * problem["center_prior"]["center_offset_m"][:, :2]
    )
    torch.testing.assert_close(
        informed["xy"]["prior_natural"][:, :2], expected_center_natural,
    )
    torch.testing.assert_close(
        informed["xy"]["prior_natural"][:, 4:],
        (10.0 * problem["center_prior"]["q0_relation_m"][..., :2]).reshape(2, 8),
    )
    torch.testing.assert_close(
        informed["z"]["prior_natural"][:, 1:],
        10.0 * problem["center_prior"]["q0_relation_m"][..., 2],
    )


@pytest.mark.parametrize("use_q0_prior", [False, True])
def test_s4_relabelling_preserves_dense_and_v14_velocity(
    use_q0_prior: bool,
) -> None:
    problem = _synthetic_problem()
    baseline, _ = _compare_dense_and_v14(
        problem, use_q0_prior=use_q0_prior,
    )
    permutation = torch.tensor([2, 0, 3, 1])
    relabelled = dict(problem)
    relabelled["history_obs_rel_m"] = problem["history_obs_rel_m"][:, :, permutation]
    relabelled["history_obs_mask"] = problem["history_obs_mask"][:, :, permutation]
    relabelled["center_prior"] = dict(problem["center_prior"])
    relabelled["center_prior"]["q0_relation_m"] = problem["center_prior"][
        "q0_relation_m"
    ][:, permutation]
    actual, _ = _compare_dense_and_v14(
        relabelled, use_q0_prior=use_q0_prior,
    )
    torch.testing.assert_close(actual, baseline, atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize(
    "planar_transform",
    [
        torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64),
        torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    ],
)
@pytest.mark.parametrize("use_q0_prior", [False, True])
def test_o2_rotation_and_reflection_match_v14_equivariance(
    planar_transform: torch.Tensor, use_q0_prior: bool,
) -> None:
    problem = _synthetic_problem()
    baseline, _ = _compare_dense_and_v14(
        problem, use_q0_prior=use_q0_prior,
    )
    transformed = dict(problem)
    transformed_obs = problem["history_obs_rel_m"].clone()
    transformed_obs[..., :2] = (
        problem["history_obs_rel_m"][..., :2] @ planar_transform.transpose(0, 1)
    )
    transformed["history_obs_rel_m"] = transformed_obs
    determinant = torch.linalg.det(planar_transform)
    transformed["omega_rad_s"] = determinant * problem["omega_rad_s"]
    transformed["center_prior"] = dict(problem["center_prior"])
    transformed_center = problem["center_prior"]["center_offset_m"].clone()
    transformed_center[..., :2] = (
        transformed_center[..., :2] @ planar_transform.transpose(0, 1)
    )
    transformed_q = problem["center_prior"]["q0_relation_m"].clone()
    transformed_q[..., :2] = (
        transformed_q[..., :2] @ planar_transform.transpose(0, 1)
    )
    transformed["center_prior"]["center_offset_m"] = transformed_center
    transformed["center_prior"]["q0_relation_m"] = transformed_q
    actual, _ = _compare_dense_and_v14(
        transformed, use_q0_prior=use_q0_prior,
    )
    expected = baseline.clone()
    expected[..., :2] = baseline[..., :2] @ planar_transform.transpose(0, 1)
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("use_q0_prior", [False, True])
def test_common_velocity_ramp_matches_v14_equivariance(
    use_q0_prior: bool,
) -> None:
    problem = _synthetic_problem()
    baseline, _ = _compare_dense_and_v14(
        problem, use_q0_prior=use_q0_prior,
    )
    ramp = torch.tensor([0.7, -0.4, 0.25], dtype=torch.float64)
    shifted = dict(problem)
    shifted["history_obs_rel_m"] = (
        problem["history_obs_rel_m"]
        + problem["history_time_s"][:, :, None, None] * ramp
    )
    actual, _ = _compare_dense_and_v14(
        shifted, use_q0_prior=use_q0_prior,
    )
    torch.testing.assert_close(
        actual, baseline + ramp, atol=2e-10, rtol=2e-10,
    )


def test_hidden_padding_is_sanitized_and_has_no_effect() -> None:
    problem = _synthetic_problem(batch=2)
    problem["history_obs_mask"][0, 3, 1] = False
    problem["history_event_mask"][1, 5] = False
    padded = dict(problem)
    padded["history_obs_rel_m"] = problem["history_obs_rel_m"].clone()
    padded["history_obs_rel_m"][0, 3, 1] = torch.nan
    padded["history_obs_rel_m"][1, 5] = torch.nan
    padded["history_time_s"] = problem["history_time_s"].clone()
    padded["history_time_s"][1, 5] = torch.nan
    sanitized = dict(problem)
    sanitized["history_obs_rel_m"] = problem["history_obs_rel_m"].clone()
    sanitized["history_obs_rel_m"][0, 3, 1] = 0.0
    sanitized["history_obs_rel_m"][1, 5] = 0.0
    sanitized["history_time_s"] = problem["history_time_s"].clone()
    sanitized["history_time_s"][1, 5] = 0.0
    first = profiled_twist_dense_design(
        **padded, use_q0_prior=True,
    )
    second = profiled_twist_dense_design(
        **sanitized, use_q0_prior=True,
    )
    for axis in ("xy", "z"):
        for name in ("design", "target", "mask", "prior_precision", "prior_natural"):
            assert torch.equal(first[axis][name], second[axis][name]), (axis, name)
        assert bool(torch.isfinite(first[axis]["design"]).all())
        assert bool(torch.isfinite(first[axis]["target"]).all())
        assert torch.count_nonzero(
            first[axis]["design"][~first[axis]["mask"]]
        ).item() == 0
        assert torch.count_nonzero(
            first[axis]["target"][~first[axis]["mask"]]
        ).item() == 0


def test_dense_design_is_directly_consumable_by_crossfit_diagnostics() -> None:
    problem = _synthetic_problem(batch=2)
    systems = profiled_twist_dense_design(
        **problem, use_q0_prior=True,
    )
    for axis, coordinates_per_event in (("xy", 8), ("z", 4)):
        system = systems[axis]
        row = torch.arange(system["mask"].shape[1])
        event = row // coordinates_per_event
        fit_selector = event < 8
        fit_mask = system["mask"] & fit_selector.unsqueeze(0)
        heldout_mask = system["mask"] & ~fit_selector.unsqueeze(0)
        result = linear_gaussian_crossfit_diagnostics(
            fit_design=system["design"], fit_target=system["target"],
            fit_mask=fit_mask,
            heldout_design=system["design"], heldout_target=system["target"],
            heldout_mask=heldout_mask,
            prior_precision=system["prior_precision"],
            prior_natural=system["prior_natural"],
            velocity_column_mask=system["velocity_column_mask"],
        )
        assert result["fit_supported"].tolist() == [True, True]
        assert result["heldout_joint_valid"].tolist() == [True, True]
        assert result["velocity_covariance"].shape[-1] == (2 if axis == "xy" else 1)
