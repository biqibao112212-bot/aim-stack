from __future__ import annotations

import torch

from training.stage3.loss_only_local_precision_oracle import (
    assert_forward_cache_truth_free,
    optimize_loss_only_local_precision_oracle,
)


def _toy_solve(target: torch.Tensor):
    batch = target.shape[0]

    def solve(
        logits: torch.Tensor, anchor: torch.Tensor, center: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        repeated = target.repeat(3, 1)
        visible_mean = logits.mean(dim=(1, 2))
        velocity = torch.stack((
            visible_mean,
            anchor.mean(dim=-1),
            center,
        ), dim=-1)
        # Keep a harmless dependency on every variable in every coordinate.
        velocity = velocity + 0.0 * repeated
        return {
            "velocity_mps": velocity,
            "state_supported": torch.ones(3 * batch, dtype=torch.bool),
        }

    return solve


def test_oracle_runs_exact_budget_and_selects_per_sample_solution() -> None:
    target = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    result = optimize_loss_only_local_precision_oracle(
        _toy_solve(target),
        observation_mask=torch.ones(2, 3, 4, dtype=torch.bool),
        q0_supported=torch.ones(2, 4, dtype=torch.bool),
        target_velocity_mps=target,
    )
    assert int(result["steps"]) == 32
    assert bool(((result["best_start"] >= 0) & (result["best_start"] < 3)).all())
    assert bool(result["state_supported"].all())
    assert float(result["loss"].max()) < 1e-7


def test_unsupported_q0_alphas_are_forced_to_history() -> None:
    target = torch.zeros(1, 3)
    support = torch.tensor([[True, False, True, False]])
    result = optimize_loss_only_local_precision_oracle(
        _toy_solve(target),
        observation_mask=torch.ones(1, 2, 4, dtype=torch.bool),
        q0_supported=support,
        target_velocity_mps=target,
    )
    assert torch.equal(
        result["anchor_alpha"][~support], torch.zeros(2),
    )


def test_all_q0_unsupported_is_a_legal_observation_only_oracle() -> None:
    target = torch.tensor([[0.25, 0.0, 0.0]])

    def observation_only(
        logits: torch.Tensor, anchor: torch.Tensor, center: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del anchor, center
        velocity = torch.stack((
            logits[:, 0, 0], torch.zeros_like(logits[:, 0, 0]),
            torch.zeros_like(logits[:, 0, 0]),
        ), dim=-1)
        return {
            "velocity_mps": velocity,
            "state_supported": torch.ones(logits.shape[0], dtype=torch.bool),
        }

    result = optimize_loss_only_local_precision_oracle(
        observation_only,
        observation_mask=torch.ones(1, 2, 4, dtype=torch.bool),
        q0_supported=torch.zeros(1, 4, dtype=torch.bool),
        target_velocity_mps=target,
    )
    assert torch.count_nonzero(result["anchor_alpha"]) == 0
    assert torch.count_nonzero(result["center_alpha"]) == 0
    assert bool(result["state_supported"].all())


def test_oracle_uses_exact_deployable_projection_and_masks_padding() -> None:
    seen: list[torch.Tensor] = []

    def record(
        logits: torch.Tensor, anchor: torch.Tensor, center: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del anchor, center
        seen.append(logits.detach().clone())
        zero = logits.sum(dim=(1, 2)) * 0.0
        return {
            "velocity_mps": torch.stack((zero, zero, zero), dim=-1),
            "state_supported": torch.ones(logits.shape[0], dtype=torch.bool),
        }

    mask = torch.tensor([[[True, True, False, False], [True, False, False, False]]])
    result = optimize_loss_only_local_precision_oracle(
        record, observation_mask=mask,
        q0_supported=torch.zeros(1, 4, dtype=torch.bool),
        target_velocity_mps=torch.zeros(1, 3),
    )
    assert len(seen) == 33
    for value in seen:
        repeated_mask = mask.repeat(3, 1, 1)
        assert torch.count_nonzero(value[~repeated_mask]) == 0
        shifted = value.clone()
        shifted[repeated_mask] += 11.0
        from training.stage3.prequential_local_precision_irls import (
            project_visible_observation_log_precision,
        )
        assert torch.equal(
            project_visible_observation_log_precision(
                shifted, repeated_mask, 2.0,
            ),
            project_visible_observation_log_precision(
                value, repeated_mask, 2.0,
            ),
        )
    assert torch.count_nonzero(
        result["observation_log_precision"][~mask]
    ) == 0


def test_oracle_is_exactly_independent_of_evaluator_batch_composition() -> None:
    first_target = torch.tensor([[0.0, 0.2, 0.7]])
    second_target = torch.tensor([[0.0, 0.8, 0.1]])

    def run(target: torch.Tensor) -> dict[str, torch.Tensor]:
        return optimize_loss_only_local_precision_oracle(
            _toy_solve(target),
            observation_mask=torch.ones(target.shape[0], 3, 4, dtype=torch.bool),
            q0_supported=torch.ones(target.shape[0], 4, dtype=torch.bool),
            target_velocity_mps=target,
        )

    alone = run(first_target)
    together = run(torch.cat((first_target, second_target), dim=0))
    for name in (
        "velocity_mps", "observation_log_precision", "anchor_alpha",
        "center_alpha", "loss", "best_start", "state_supported",
    ):
        assert torch.equal(alone[name][0], together[name][0]), name


def test_oracle_does_not_clamp_the_preprojection_logit_carrier() -> None:
    import inspect
    from training.stage3.loss_only_local_precision_oracle import (
        optimize_loss_only_local_precision_oracle as implementation,
    )

    source = inspect.getsource(implementation)
    assert "raw_logits.clamp_" not in source


def test_truth_free_cache_guard_rejects_loss_sidecar_fields() -> None:
    assert_forward_cache_truth_free({"event_feature": torch.zeros(1)})
    for key in ("truth_velocity", "oracle_alpha", "session_id", "motion_class"):
        try:
            assert_forward_cache_truth_free({key: torch.zeros(1)})
        except ValueError:
            pass
        else:
            raise AssertionError(f"cache guard accepted {key}")
