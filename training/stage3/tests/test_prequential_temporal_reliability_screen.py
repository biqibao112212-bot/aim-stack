from __future__ import annotations

import inspect
from copy import deepcopy
from itertools import permutations

import torch

from training.stage3 import screen_prequential_temporal_reliability_oracle_omega as screen
from training.stage3.prequential_endpoint_innovation import (
    deterministic_endpoint_dose_temporal_break,
    same_time_unordered_pair_temporal_features,
)


class _InvariantTemporalProbe(torch.nn.Module):
    """Small anonymous probe used to audit the screen's variant averaging."""

    def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = kwargs["event_feature"]
        mask = kwargs["event_mask"].to(feature.dtype)
        time_weight = torch.arange(
            feature.shape[1], dtype=feature.dtype, device=feature.device,
        )[None, :, None]
        role_score = (feature[..., 3] * mask * time_weight).sum(dim=1)
        score = (
            role_score.square().sum(dim=-1)
            + 0.1 * role_score.prod(dim=-1)
        )
        return {
            "q0_weight": torch.sigmoid(0.01 * score),
            "learned_gate_valid": torch.ones(
                feature.shape[0], dtype=torch.bool, device=feature.device,
            ),
        }


class _ScreenProbe(_InvariantTemporalProbe):
    def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        probe = super().forward(**kwargs)
        weight = probe["q0_weight"]
        q0 = kwargs["q0_velocity_mps"]
        history = kwargs["history_velocity_mps"]
        supported = kwargs["q0_supported"] | kwargs["history_supported"]
        return {
            **probe,
            "velocity_mps": (
                weight[:, None] * q0 + (1.0 - weight[:, None]) * history
            ),
            "state_supported": supported,
            "fallback_used": torch.zeros_like(supported),
        }


def _batch(event_feature: torch.Tensor, event_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    pair = same_time_unordered_pair_temporal_features(event_feature, event_mask)
    batch_size = event_feature.shape[0]
    zeros3 = torch.zeros(batch_size, 3, dtype=event_feature.dtype)
    return {
        "event_feature": event_feature,
        "event_mask": event_mask,
        "event_pair_feature": pair["pair_feature"],
        "pair_mask": pair["pair_mask"],
        "global_feature": torch.zeros(batch_size, 11, dtype=event_feature.dtype),
        "q0_velocity": zeros3.clone(),
        "history_velocity": zeros3.clone(),
        "q0_supported": torch.ones(batch_size, dtype=torch.bool),
        "history_supported": torch.ones(batch_size, dtype=torch.bool),
        "fallback_velocity": zeros3.clone(),
        "fallback_supported": torch.ones(batch_size, dtype=torch.bool),
    }


def test_dose_retains_only_anonymous_subsets_with_valid_donors() -> None:
    event = torch.zeros(1, 5, 4, 11)
    event[..., 3] = torch.arange(5, dtype=torch.float32)[None, :, None]
    event[..., 3] += torch.arange(4, dtype=torch.float32)[None, None, :] * 10.0
    mask = torch.zeros(1, 5, 4, dtype=torch.bool)
    mask[:, :, :2] = True
    mask[:, :1, 2:] = True
    batch = _batch(event, mask)
    model = _InvariantTemporalProbe()

    value, valid = screen._dose_variant(model, batch, dose=1)

    changed = deterministic_endpoint_dose_temporal_break(event, mask, dose=1)
    replicated = screen._replicate_temporal_variants(
        batch, changed["event_feature"], changed["event_mask"],
    )
    raw = model(**screen._model_kwargs(replicated))["q0_weight"].reshape(1, 4)
    subset_valid = (
        ~changed["damaged_role_mask"].unsqueeze(0)
        | changed["break_valid"].unsqueeze(1)
    ).all(dim=-1)
    assert bool(valid.item())
    assert int(subset_valid.sum()) == 2
    torch.testing.assert_close(value["weight"], raw)
    assert torch.equal(value["variant_valid"], subset_valid)


def test_dose_saturates_at_all_observable_temporal_roles() -> None:
    event = torch.randn(1, 5, 4, 11)
    mask = torch.zeros(1, 5, 4, dtype=torch.bool)
    mask[:, :, :2] = True
    mask[:, :1, 2:] = True
    value, valid = screen._dose_variant(
        _InvariantTemporalProbe(), _batch(event, mask), dose=4,
    )
    assert bool(valid.item())
    assert value["variant_valid"].tolist() == [[True]]


def test_continuity_group_average_is_independent_of_anonymous_role_labels() -> None:
    generator = torch.Generator().manual_seed(20260730)
    event = torch.randn(2, 6, 4, 11, generator=generator)
    mask = torch.ones(2, 6, 4, dtype=torch.bool)
    batch = _batch(event, mask)
    model = _InvariantTemporalProbe()
    reference, reference_valid = screen._continuity_variant(model, batch)

    for values in list(permutations(range(4)))[::5]:
        order = torch.tensor(values, dtype=torch.long)
        changed = _batch(event[:, :, order], mask[:, :, order])
        value, valid = screen._continuity_variant(model, changed)
        torch.testing.assert_close(
            value["weight"].sort(dim=1).values,
            reference["weight"].sort(dim=1).values,
        )
        assert torch.equal(valid, reference_valid)


def test_anonymous_variant_metrics_use_expected_loss_not_ensemble_loss() -> None:
    batch = {
        "weight": torch.tensor([0.5, 0.5]),
        "target_velocity": torch.zeros(2, 3),
    }
    value = {
        "weight": torch.tensor([[0.1, 0.9], [0.2, 0.8]]),
        "velocity": torch.tensor([
            [[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0]],
            [[0.3, 0.0, 0.0], [-0.3, 0.0, 0.0]],
        ]),
        "variant_valid": torch.ones(2, 2, dtype=torch.bool),
    }
    metrics = screen._variant_metrics(
        value, batch, torch.ones(2, dtype=torch.bool),
        torch.tensor([False, True]),
    )
    # Averaging predictions first would give zero coefficient and velocity
    # error.  The intended estimand averages each anonymous intervention loss.
    assert abs(metrics["coefficient_mae"] - 0.35) < 1e-6
    assert abs(metrics["error_mps"]["mean"] - 0.35) < 1e-6
    assert metrics["valid_variant_row_count"] == 4


def test_weighted_anonymous_auc_survives_independent_row_relabelling() -> None:
    score = torch.tensor([
        [0.1, 0.3, 0.2], [0.8, 0.6, 0.7],
        [0.4, 0.5, 0.2], [0.9, 0.7, 0.8],
    ])
    label = torch.tensor([False, True, False, True])
    valid = torch.tensor([
        [True, True, False], [True, True, True],
        [False, True, True], [True, False, True],
    ])
    reference = screen._weighted_anonymous_auc(score, label, valid)
    relabelled_score = score.clone()
    relabelled_valid = valid.clone()
    orders = ([1, 0, 2], [2, 0, 1], [2, 1, 0], [0, 2, 1])
    for row, order in enumerate(orders):
        relabelled_score[row] = score[row, list(order)]
        relabelled_valid[row] = valid[row, list(order)]
    assert screen._weighted_anonymous_auc(
        relabelled_score, label, relabelled_valid,
    ) == reference


def test_crossfit_support_counts_distinct_exposures_not_role_tokens() -> None:
    mask = torch.zeros(2, 3, 4, dtype=torch.bool)
    mask[0, 0, :2] = True
    mask[1, 0, 0] = True
    mask[1, 2, 3] = True
    assert screen._crossfit_temporal_supported(mask).tolist() == [False, True]


def test_evaluation_keeps_base_population_and_uses_mechanism_domains() -> None:
    event = torch.randn(4, 5, 4, 11)
    mask = torch.zeros(4, 5, 4, dtype=torch.bool)
    mask[:, :, 0] = True
    mask[:, :1, 1:] = True
    arm = _batch(event, mask)
    arm.update({
        "target_velocity": torch.zeros(4, 3),
        "old_velocity": torch.ones(4, 3),
        "weight": torch.full((4,), 0.5),
        "oracle_valid": torch.ones(4, dtype=torch.bool),
        "crossfit_supported": torch.ones(4, dtype=torch.bool),
        "motion_class": torch.tensor([2, 2, 3, 3]),
        "recipient_global": torch.arange(4),
        "donor_global": torch.arange(4),
    })
    arm["q0_velocity"] = torch.ones(4, 3)
    arm["history_velocity"] = torch.zeros(4, 3)
    metrics = screen._evaluate_head(
        _ScreenProbe(), {"intact": arm, "global": arm, "hard": arm},
        torch.device("cpu"),
    )
    intact = metrics["arms"]["intact"]
    overall = intact["groups"]["overall"]["variants"]
    assert intact["common_count"] == 4
    assert overall["association_break"]["eligible_count"] == 4
    assert overall["dose4"]["eligible_count"] == 4
    assert overall["dose4"]["valid_variant_row_count"] == 4


def test_runner_contract_locks_profile_only_crossfit_and_static_prefix() -> None:
    source = inspect.getsource(screen)
    assert '"temporal_association_static_prefix_width": 3' in source
    assert '"leave_block_out_scope": "profile_fit_masks_only"' in source
    assert '"leave_block_out_requires_complete_block": True' in source
    assert '"crossfit_support_unit": "distinct_event_exposure"' in source
    assert (
        '"per-sample-expected-loss-and-weighted-flattened-auc"' in source
    )
    assert '"pair_message_auc_gain_gate": 0.02' in source
    assert '"corruption_selection_gates": "sealed-a1"' in source
    assert '"dose_semantics": "saturating-observable-temporal-roles"' in source
    assert '"endpoint_features_use_original_events": True' in source
    assert '"leave_block_out_profile_refits": ["q0", "history"]' in source


def test_local_auxiliary_has_no_extra_velocity_regression() -> None:
    source = inspect.getsource(screen._train_head)
    local_block = source.split('if name == "intact":', 1)[1].split(
        "predictions[name]", 1,
    )[0]
    assert "local_coefficient" in local_block
    assert "local_preference" in local_block
    assert "local_velocity" not in local_block
    assert "vector_huber_loss" not in local_block


def _gate_metrics() -> dict[str, object]:
    def distribution(mean: float, p50: float, p95: float) -> dict[str, float]:
        return {"mean": mean, "p50": p50, "p95": p95}

    def variants() -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for name in screen.VARIANT_NAMES:
            result[name] = {
                "component_preference_auc": 0.60,
                "coefficient_mae": 0.30,
                "error_mps": distribution(0.80, 0.60, 1.30),
            }
        result["full"].update({
            "component_preference_auc": 0.82,
            "coefficient_mae": 0.20,
            "error_mps": distribution(0.65, 0.45, 1.00),
        })
        reference = {
            "component_preference_auc": 0.82,
            "coefficient_mae": 0.20,
            "error_mps": distribution(0.65, 0.45, 1.00),
        }
        for name in (
            "association_break", "continuity_break",
            "dose1", "dose2", "dose4",
        ):
            result[name]["reference_full"] = deepcopy(reference)
            result[name]["eligible_fraction"] = 1.0
        result["dose1"]["coefficient_mae"] = 0.24
        result["dose2"]["coefficient_mae"] = 0.27
        result["dose4"]["coefficient_mae"] = 0.31
        return result

    def group(*, corrupt: bool) -> dict[str, object]:
        result: dict[str, object] = {
            "common_fraction": 1.0,
            "parent_error_mps": distribution(1.00, 0.80, 1.50),
            "oracle_error_mps": distribution(0.50, 0.30, 0.80),
            "blind_error_mps": distribution(0.80, 0.60, 1.30),
            "variants": variants(),
        }
        if corrupt:
            result.update({
                "paired_fraction": 1.0,
                "paired_weight_separation_mean": 0.20,
            })
        return result

    arms: dict[str, object] = {}
    for arm in ("intact", "global", "hard"):
        arms[arm] = {
            "state_supported_fraction": 1.0,
            "common_fraction": 1.0,
            "groups": {
                name: group(corrupt=arm != "intact")
                for name in ("overall", "rotation", "combined")
            },
        }
    return {"arms": arms}


def test_fold_gates_require_pair_messages_and_corruption_selection() -> None:
    metrics = _gate_metrics()
    gates = screen._fold_gates(metrics)
    assert gates["overall_intact_pair_message_auc_gain_ge_0_02"] is True
    assert gates["overall_global_auc_ge_0_75"] is True
    assert gates["overall_hard_paired_fraction_ge_0_95"] is True
    assert gates["overall_hard_weight_separation_ge_0_15"] is True

    broken = deepcopy(metrics)
    broken["arms"]["intact"]["groups"]["overall"]["variants"][
        "pair_message_disabled"
    ]["component_preference_auc"] = 0.82
    broken["arms"]["global"]["groups"]["overall"]["variants"]["full"][
        "component_preference_auc"
    ] = 0.50
    broken["arms"]["hard"]["groups"]["overall"][
        "paired_weight_separation_mean"
    ] = 0.0
    gates = screen._fold_gates(broken)
    assert gates["overall_intact_pair_message_auc_gain_ge_0_02"] is False
    assert gates["overall_global_auc_ge_0_75"] is False
    assert gates["overall_hard_weight_separation_ge_0_15"] is False


def test_cross_fold_gates_bind_pair_and_corruption_stability() -> None:
    metrics = _gate_metrics()
    gates = screen._cross_fold_gates([
        {"metrics": deepcopy(metrics)}, {"metrics": deepcopy(metrics)},
    ])
    assert gates["overall_pair_message_auc_gain_fold_delta_le_0_05"] is True
    assert gates["global_overall_auc_fold_delta_le_0_10"] is True
    assert gates["hard_combined_separation_fold_delta_le_0_05"] is True
