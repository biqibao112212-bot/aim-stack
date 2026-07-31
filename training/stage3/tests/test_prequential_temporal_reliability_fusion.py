from __future__ import annotations

import torch

from training.stage3.prequential_temporal_reliability_fusion import (
    PAIR_INDICES,
    PrequentialTemporalReliabilityFusion,
)


def _fixture(*, batch: int = 5, events: int = 6) -> dict[str, torch.Tensor]:
    torch.manual_seed(1701)
    event_mask = torch.ones(batch, events, 4, dtype=torch.bool)
    event_mask[0, 0, 2:] = False
    pair_mask = torch.stack([
        event_mask[:, :, left] & event_mask[:, :, right]
        for left, right in PAIR_INDICES
    ], dim=-1)
    return {
        "event_feature": torch.randn(batch, events, 4, 7),
        "event_mask": event_mask,
        "event_pair_feature": torch.randn(batch, events, 6, 5),
        "pair_mask": pair_mask,
        "global_feature": torch.randn(batch, 4),
        "q0_velocity_mps": torch.randn(batch, 3),
        "history_velocity_mps": torch.randn(batch, 3),
        "q0_supported": torch.ones(batch, dtype=torch.bool),
        "history_supported": torch.ones(batch, dtype=torch.bool),
        "fallback_velocity_mps": torch.randn(batch, 3),
        "fallback_supported": torch.ones(batch, dtype=torch.bool),
    }


def _model(data: dict[str, torch.Tensor]) -> PrequentialTemporalReliabilityFusion:
    torch.manual_seed(29)
    model = PrequentialTemporalReliabilityFusion(
        event_width=7, pair_width=5, global_width=4, width=16,
    )
    model.fit_normalizer(
        event_feature=data["event_feature"],
        event_mask=data["event_mask"],
        event_pair_feature=data["event_pair_feature"],
        pair_mask=data["pair_mask"],
        global_feature=data["global_feature"],
        row_mask=torch.ones(data["global_feature"].shape[0], dtype=torch.bool),
    )
    return model.eval()


def _pair_permutation(permutation: torch.Tensor) -> torch.Tensor:
    lookup = {pair: index for index, pair in enumerate(PAIR_INDICES)}
    result = []
    for left, right in PAIR_INDICES:
        mapped = tuple(sorted((
            int(permutation[left]), int(permutation[right]),
        )))
        result.append(lookup[mapped])
    return torch.tensor(result)


def _permuted(
    data: dict[str, torch.Tensor], permutation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    inverse = torch.argsort(permutation)
    pair_inverse = torch.argsort(_pair_permutation(permutation))
    result = dict(data)
    result["event_feature"] = data["event_feature"][:, :, inverse]
    result["event_mask"] = data["event_mask"][:, :, inverse]
    result["event_pair_feature"] = data["event_pair_feature"][:, :, pair_inverse]
    result["pair_mask"] = data["pair_mask"][:, :, pair_inverse]
    return result


def test_scalar_features_and_velocity_are_o2_equivariant() -> None:
    data = _fixture()
    model = _model(data)
    reference = model(**data)
    matrix, _ = torch.linalg.qr(torch.randn(3, 3))
    rotated = dict(data)
    for name in (
        "q0_velocity_mps", "history_velocity_mps", "fallback_velocity_mps",
    ):
        rotated[name] = data[name] @ matrix
    actual = model(**rotated)
    assert torch.allclose(actual["q0_weight"], reference["q0_weight"], atol=1e-6)
    assert torch.allclose(
        actual["velocity_mps"], reference["velocity_mps"] @ matrix,
        atol=2e-6,
    )


def test_arbitrary_synchronized_s4_permutation_is_invariant() -> None:
    data = _fixture()
    model = _model(data)
    reference = model(**data)
    for permutation in (
        torch.tensor([2, 0, 3, 1]),
        torch.tensor([3, 2, 1, 0]),
        torch.tensor([1, 3, 0, 2]),
    ):
        actual = model(**_permuted(data, permutation))
        assert torch.allclose(actual["full_logit"], reference["full_logit"], atol=2e-6)
        assert torch.allclose(actual["velocity_mps"], reference["velocity_mps"], atol=2e-6)


def test_breaking_per_event_s4_continuity_changes_recurrent_output() -> None:
    data = _fixture(batch=3)
    model = _model(data)
    reference = model(**data)["full_logit"]
    changed = dict(data)
    permutation = torch.tensor([2, 0, 3, 1])
    # Here ``permutation`` maps the new one-event roles to their old roles;
    # apply the same map to that event's unordered pair tokens.
    pair_permutation = _pair_permutation(permutation)
    changed["event_feature"] = data["event_feature"].clone()
    changed["event_mask"] = data["event_mask"].clone()
    changed["event_pair_feature"] = data["event_pair_feature"].clone()
    changed["pair_mask"] = data["pair_mask"].clone()
    changed["event_feature"][:, 2] = data["event_feature"][:, 2, permutation]
    changed["event_mask"][:, 2] = data["event_mask"][:, 2, permutation]
    changed["event_pair_feature"][:, 2] = data[
        "event_pair_feature"
    ][:, 2, pair_permutation]
    changed["pair_mask"][:, 2] = data["pair_mask"][:, 2, pair_permutation]
    actual = model(**changed)["full_logit"]
    assert not torch.equal(actual, reference)


def test_common_velocity_ramp_passes_through_exactly() -> None:
    data = _fixture()
    model = _model(data)
    reference = model(**data)
    ramp = torch.randn(data["global_feature"].shape[0], 3)
    shifted = dict(data)
    for name in (
        "q0_velocity_mps", "history_velocity_mps", "fallback_velocity_mps",
    ):
        shifted[name] = data[name] + ramp
    actual = model(**shifted)
    assert torch.allclose(actual["q0_weight"], reference["q0_weight"], atol=1e-7)
    assert torch.allclose(
        actual["velocity_mps"], reference["velocity_mps"] + ramp,
        atol=2e-6,
    )


def test_ablation_branches_are_independent() -> None:
    data = _fixture()
    model = _model(data)
    changed_local = dict(data)
    changed_local["event_feature"] = data["event_feature"] + 100.0
    changed_local["event_pair_feature"] = data["event_pair_feature"] - 100.0
    disabled_reference = model(**data, local_disabled=True)
    disabled_actual = model(**changed_local, local_disabled=True)
    assert torch.equal(
        disabled_reference["q0_weight"], disabled_actual["q0_weight"],
    )

    changed_pair = dict(data)
    changed_pair["event_pair_feature"] = data["event_pair_feature"] * 20.0
    pair_reference = model(**data, pair_message_disabled=True)
    pair_actual = model(**changed_pair, pair_message_disabled=True)
    assert torch.equal(pair_reference["q0_weight"], pair_actual["q0_weight"])

    reversed_data = dict(data)
    for name in ("event_feature", "event_mask", "event_pair_feature", "pair_mask"):
        reversed_data[name] = torch.flip(data[name], dims=(1,))
    recurrent = model(**reversed_data)["q0_weight"]
    assert not torch.equal(recurrent, model(**data)["q0_weight"])
    collapsed_reference = model(**data, temporal_collapsed=True)["q0_weight"]
    collapsed_actual = model(
        **reversed_data, temporal_collapsed=True,
    )["q0_weight"]
    assert torch.allclose(collapsed_reference, collapsed_actual, atol=2e-6)


def test_real_backward_is_finite_and_support_overrides_are_exact() -> None:
    data = _fixture()
    data["q0_supported"] = torch.tensor([True, False, False, True, True])
    data["history_supported"] = torch.tensor([False, True, False, True, True])
    data["fallback_supported"] = torch.tensor([False, False, True, False, False])
    model = _model(data).train()
    output = model(**data)
    assert output["q0_weight"][0] == 1
    assert output["q0_weight"][1] == 0
    assert output["fallback_used"][2]
    assert torch.equal(output["velocity_mps"][0], data["q0_velocity_mps"][0])
    assert torch.equal(
        output["velocity_mps"][1], data["history_velocity_mps"][1],
    )
    assert torch.equal(
        output["velocity_mps"][2], data["fallback_velocity_mps"][2],
    )
    loss = (
        output["velocity_mps"].square().mean()
        + output["raw_q0_weight"].square().mean()
        + output["raw_local_q0_weight"].square().mean()
        + output["raw_global_only_q0_weight"].square().mean()
    )
    loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable and all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in trainable
    )


def test_state_dict_strict_round_trip_preserves_outputs() -> None:
    data = _fixture()
    model = _model(data)
    reference = model(**data)
    restored = PrequentialTemporalReliabilityFusion(
        event_width=7, pair_width=5, global_width=4, width=16,
    )
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.eval()
    actual = restored(**data)
    for name in (
        "velocity_mps", "local_logit", "global_only_logit", "full_logit",
        "local_q0_weight", "global_only_q0_weight", "q0_weight",
    ):
        assert torch.equal(reference[name], actual[name])
