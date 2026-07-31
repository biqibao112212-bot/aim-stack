from __future__ import annotations

import inspect

import torch

from training.stage3.prequential_endpoint_innovation import (
    UNORDERED_PAIR_INDICES,
    combine_blockwise_endpoint_event_features,
    conjugate_role_donor_index,
    deterministic_endpoint_dose_temporal_break,
    deterministic_per_event_s4_continuity_break,
    deterministic_temporal_association_break,
    fixed_history_leave_block_out_masks,
    same_time_unordered_pair_temporal_features,
)


def _fixture() -> tuple[torch.Tensor, torch.Tensor]:
    feature = torch.arange(1 * 5 * 4 * 7, dtype=torch.float32).reshape(
        1, 5, 4, 7,
    )
    mask = torch.ones(1, 5, 4, dtype=torch.bool)
    mask[:, 1, 3] = False
    return feature, mask


def _sorted_rows(value: torch.Tensor) -> torch.Tensor:
    key = value.to(torch.float64) @ torch.linspace(
        1.0, 2.0, value.shape[-1], dtype=torch.float64,
    )
    return value[torch.argsort(key)]


def test_fixed_leave_block_out_masks_are_disjoint_and_enforce_fit_support() -> None:
    event_mask = torch.tensor([
        [True, True, False, True, True, True],
        [False, True, True, True, False, False],
    ])
    result = fixed_history_leave_block_out_masks(
        event_mask, block_size=2, minimum_fit_events=2,
    )
    fit = result["fit_event_mask"]
    heldout = result["heldout_event_mask"]
    assert fit.shape == heldout.shape == (2, 3, 6)
    assert not torch.any(fit & heldout)
    assert torch.all(
        fit.sum(dim=-1)[result["block_valid"]] >= 2
    )
    assert torch.all(heldout.sum(dim=1) <= 1)
    # Five active events expose only two complete two-event blocks; the final
    # singleton stays uncovered.  The three-event row cannot score its first
    # complete block because that would leave only one fit event.
    first_coverage = result["heldout_coverage_mask"][0, event_mask[0]]
    assert first_coverage.tolist() == [True, True, True, True, False]
    assert result["heldout_coverage_mask"][1].sum() == 0
    assert result["heldout_block_index"][1, 3] == -1
    assert torch.all(result["heldout_block_index"][1, [0, 1, 2, 4, 5]] == -1)


def test_blockwise_features_select_only_the_unique_heldout_source() -> None:
    event_mask = torch.tensor([[True, True, True, True]])
    plan = fixed_history_leave_block_out_masks(
        event_mask, block_size=1, minimum_fit_events=2,
    )
    blocks = plan["heldout_event_mask"].shape[1]
    feature = torch.full((1, blocks, 4, 4, 2), torch.nan)
    role_mask = torch.ones(1, blocks, 4, 4, dtype=torch.bool)
    role_mask[:, :, :, 3] = False
    for block in range(blocks):
        selected = plan["heldout_event_mask"][0, block]
        feature[0, block, selected] = float(10 + block)
    result = combine_blockwise_endpoint_event_features(
        feature, plan["heldout_event_mask"],
        block_event_role_mask=role_mask,
    )
    assert result["event_feature"].shape == (1, 4, 4, 2)
    assert torch.isfinite(result["event_feature"]).all()
    assert torch.all(result["event_mask"][..., :3])
    assert not torch.any(result["event_mask"][..., 3])
    for event in range(4):
        assert torch.all(result["event_feature"][0, event, :3] == 10 + event)
    assert torch.all(result["event_feature"][..., 3, :] == 0)


def test_same_time_pair_tokens_are_symmetric_and_s4_covariant() -> None:
    feature, mask = _fixture()
    reference = same_time_unordered_pair_temporal_features(feature, mask)
    permutation = torch.tensor([2, 0, 3, 1])
    changed = same_time_unordered_pair_temporal_features(
        feature[:, :, permutation], mask[:, :, permutation],
    )
    original_pair_for_set = {
        frozenset(pair): index for index, pair in enumerate(UNORDERED_PAIR_INDICES)
    }
    for new_pair_index, (left, right) in enumerate(UNORDERED_PAIR_INDICES):
        old_pair_index = original_pair_for_set[frozenset((
            int(permutation[left]), int(permutation[right]),
        ))]
        torch.testing.assert_close(
            changed["pair_feature"][:, :, new_pair_index],
            reference["pair_feature"][:, :, old_pair_index],
        )
        assert torch.equal(
            changed["pair_mask"][:, :, new_pair_index],
            reference["pair_mask"][:, :, old_pair_index],
        )
    # The helper sees only invariant scalars: a hypothetical O(2), reflection,
    # or common-ramp transform leaves its inputs and therefore outputs exact.
    repeated = same_time_unordered_pair_temporal_features(
        feature.clone(), mask.clone(),
    )
    assert torch.equal(reference["pair_feature"], repeated["pair_feature"])


def test_temporal_break_is_per_role_and_preserves_prefix_mask_and_residual_marginals() -> None:
    feature, mask = _fixture()
    result = deterministic_temporal_association_break(feature, mask)
    donor = result["donor_event_index"]
    assert donor.shape == (1, 5, 4)
    assert torch.all(result["break_valid"])
    assert torch.equal(result["event_mask"], mask)
    assert torch.equal(result["event_feature"][..., :3], feature[..., :3])
    event = torch.arange(feature.shape[1]).view(1, -1, 1)
    assert torch.all(donor[mask] != event.expand_as(donor)[mask])
    assert not torch.equal(donor[0, :, 3], donor[0, :, 0])
    for role in range(4):
        visible = mask[0, :, role]
        torch.testing.assert_close(
            _sorted_rows(result["event_feature"][0, visible, role, 3:]),
            _sorted_rows(feature[0, visible, role, 3:]),
        )
    assert not torch.equal(result["event_feature"][..., 3:], feature[..., 3:])
    permutation = torch.tensor([2, 0, 3, 1])
    relabelled = deterministic_temporal_association_break(
        feature[:, :, permutation], mask[:, :, permutation],
    )
    torch.testing.assert_close(
        relabelled["event_feature"],
        result["event_feature"][:, :, permutation],
    )
    assert torch.equal(
        relabelled["event_mask"], result["event_mask"][:, :, permutation],
    )
    assert torch.equal(
        relabelled["donor_event_index"], donor[:, :, permutation],
    )


def test_per_event_s4_break_preserves_each_static_set_and_covaries_with_plan() -> None:
    feature, mask = _fixture()
    result = deterministic_per_event_s4_continuity_break(feature, mask)
    donor = result["donor_role_index"]
    role = torch.arange(4).view(1, 1, 4)
    assert torch.all(donor[result["break_valid"]] != role.expand_as(donor)[
        result["break_valid"]
    ])
    for event in range(feature.shape[1]):
        torch.testing.assert_close(
            _sorted_rows(result["event_feature"][0, event]),
            _sorted_rows(feature[0, event]),
        )
    before_pair = same_time_unordered_pair_temporal_features(feature, mask)
    after_pair = same_time_unordered_pair_temporal_features(
        result["event_feature"], result["event_mask"],
    )
    for event in range(feature.shape[1]):
        torch.testing.assert_close(
            _sorted_rows(after_pair["pair_feature"][0, event]),
            _sorted_rows(before_pair["pair_feature"][0, event]),
        )
    permutation = torch.tensor([2, 0, 3, 1])
    conjugated = conjugate_role_donor_index(donor, permutation)
    relabelled = deterministic_per_event_s4_continuity_break(
        feature[:, :, permutation], mask[:, :, permutation],
        role_donor_index=conjugated,
    )
    torch.testing.assert_close(
        relabelled["event_feature"],
        result["event_feature"][:, :, permutation],
    )
    assert torch.equal(
        relabelled["event_mask"], result["event_mask"][:, :, permutation],
    )


def test_dose_break_enumerates_anonymous_subsets_and_preserves_stream_marginals() -> None:
    feature, mask = _fixture()
    association = deterministic_temporal_association_break(feature, mask)
    for dose, variants in ((1, 4), (2, 6), (4, 1)):
        result = deterministic_endpoint_dose_temporal_break(
            feature, mask, dose=dose,
            donor_event_index=association["donor_event_index"],
        )
        assert torch.equal(
            result["donor_event_index"], association["donor_event_index"],
        )
        assert result["event_feature"].shape[:2] == (1, variants)
        assert torch.all(result["damaged_role_mask"].sum(dim=-1) == dose)
        assert torch.equal(
            result["event_feature"][..., :3],
            feature[:, None, ..., :3].expand(-1, variants, -1, -1, -1),
        )
        assert torch.equal(
            result["event_mask"],
            mask[:, None].expand(-1, variants, -1, -1),
        )
        event = torch.arange(feature.shape[1]).view(1, -1, 1)
        assert torch.all(
            result["donor_event_index"][mask]
            != event.expand_as(result["donor_event_index"])[mask]
        )
        for variant in range(variants):
            damaged = result["damaged_role_mask"][variant]
            for role in range(4):
                if damaged[role]:
                    visible = mask[0, :, role]
                    torch.testing.assert_close(
                        _sorted_rows(result[
                            "event_feature"
                        ][0, variant, visible, role, 3:]),
                        _sorted_rows(feature[0, visible, role, 3:]),
                    )
                else:
                    torch.testing.assert_close(
                        result["event_feature"][0, variant, :, role],
                        feature[0, :, role],
                    )
                    assert torch.equal(
                        result["event_mask"][0, variant, :, role],
                        mask[0, :, role],
                    )

    # Dose-two variants form an S4-covariant set: relabelling changes only the
    # variant whose damaged-role mask names the corresponding anonymous subset.
    reference = deterministic_endpoint_dose_temporal_break(
        feature, mask, dose=2,
    )
    permutation = torch.tensor([2, 0, 3, 1])
    changed = deterministic_endpoint_dose_temporal_break(
        feature[:, :, permutation], mask[:, :, permutation], dose=2,
    )
    assert torch.equal(
        changed["donor_event_index"],
        reference["donor_event_index"][:, :, permutation],
    )
    for old_variant, old_damage in enumerate(reference["damaged_role_mask"]):
        new_damage = old_damage[permutation]
        match = torch.nonzero(
            torch.all(changed["damaged_role_mask"] == new_damage, dim=-1),
            as_tuple=False,
        ).flatten()
        assert match.numel() == 1
        new_variant = int(match.item())
        torch.testing.assert_close(
            changed["event_feature"][:, new_variant],
            reference["event_feature"][:, old_variant, :, permutation],
        )
        assert torch.equal(
            changed["event_mask"][:, new_variant],
            reference["event_mask"][:, old_variant, :, permutation],
        )


def test_head_feature_apis_expose_no_forbidden_semantic_inputs() -> None:
    forbidden = {"identity", "session", "motion", "truth", "future", "physical"}
    for function in (
        fixed_history_leave_block_out_masks,
        combine_blockwise_endpoint_event_features,
        same_time_unordered_pair_temporal_features,
    ):
        assert not any(
            any(token in name for token in forbidden)
            for name in inspect.signature(function).parameters
        )
