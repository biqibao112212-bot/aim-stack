from __future__ import annotations

import pytest
import torch

from training.stage3.prequential_local_precision_irls import (
    PAIR_INDICES,
    PrequentialLocalPrecisionIRLS,
    project_visible_observation_log_precision,
)


def _fixture(batch: int = 4, events: int = 5) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(194)
    event_mask = torch.ones(batch, events, 4, dtype=torch.bool)
    event_mask[0, 0, 2:] = False
    anchor_mask = torch.ones(batch, 4, dtype=torch.bool)
    anchor_mask[0, 3] = False
    pair_mask = torch.stack([
        event_mask[:, :, left] & event_mask[:, :, right]
        for left, right in PAIR_INDICES
    ], dim=2)
    return {
        "event_feature": torch.randn(batch, events, 4, 7, generator=generator),
        "event_mask": event_mask,
        "anchor_feature": torch.randn(batch, 4, 5, generator=generator),
        "anchor_mask": anchor_mask,
        "pair_feature": torch.randn(batch, events, 6, 9, generator=generator),
        "pair_mask": pair_mask,
        "global_feature": torch.randn(batch, 3, generator=generator),
    }


def _model(data: dict[str, torch.Tensor]) -> PrequentialLocalPrecisionIRLS:
    torch.manual_seed(77)
    model = PrequentialLocalPrecisionIRLS(
        event_width=7, anchor_width=5, pair_width=9,
        global_width=3, width=16,
    )
    model.fit_normalizer(**data)
    return model


def _pair_permutation(permutation: torch.Tensor) -> torch.Tensor:
    lookup = {pair: index for index, pair in enumerate(PAIR_INDICES)}
    result = []
    for left, right in PAIR_INDICES:
        mapped = tuple(sorted((int(permutation[left]), int(permutation[right]))))
        result.append(lookup[mapped])
    return torch.tensor(result)


def test_s4_equivariance_and_no_global_output() -> None:
    data = _fixture()
    model = _model(data).eval()
    reference = model(**data)
    assert set(reference) == {
        "observation_log_precision", "anchor_alpha", "center_alpha",
    }
    permutation = torch.tensor([2, 0, 3, 1])
    pair_order = _pair_permutation(permutation)
    changed = dict(data)
    changed["event_feature"] = data["event_feature"][:, :, permutation]
    changed["event_mask"] = data["event_mask"][:, :, permutation]
    changed["anchor_feature"] = data["anchor_feature"][:, permutation]
    changed["anchor_mask"] = data["anchor_mask"][:, permutation]
    changed["pair_feature"] = data["pair_feature"][:, :, pair_order]
    changed["pair_mask"] = data["pair_mask"][:, :, pair_order]
    actual = model(**changed)
    torch.testing.assert_close(
        actual["observation_log_precision"],
        reference["observation_log_precision"][:, :, permutation],
        rtol=0.0, atol=2e-6,
    )
    torch.testing.assert_close(
        actual["anchor_alpha"], reference["anchor_alpha"][:, permutation],
        rtol=0.0, atol=2e-6,
    )
    torch.testing.assert_close(actual["center_alpha"], reference["center_alpha"])
    assert not any("embedding" in name for name, _ in model.named_parameters())


def test_masks_normalizer_and_neutral_precision_are_exact() -> None:
    data = _fixture()
    # Invisible sentinels must not enter fitted statistics.
    data["event_feature"][~data["event_mask"]] = 1e9
    data["anchor_feature"][~data["anchor_mask"]] = -1e9
    data["pair_feature"][~data["pair_mask"]] = 1e8
    model = _model(data)
    assert float(model.event_mean.abs().max()) < 1e3
    assert float(model.anchor_mean.abs().max()) < 1e3
    assert float(model.pair_mean.abs().max()) < 1e3
    with torch.no_grad():
        model.observation_head.weight.zero_()
    output = model(**data)
    assert torch.equal(
        output["observation_log_precision"],
        torch.zeros_like(output["observation_log_precision"]),
    )
    assert torch.equal(
        output["anchor_alpha"][~data["anchor_mask"]],
        torch.zeros_like(output["anchor_alpha"][~data["anchor_mask"]]),
    )
    visible_weight = torch.exp(
        output["observation_log_precision"][data["event_mask"]]
    )
    assert torch.equal(visible_weight, torch.ones_like(visible_weight))


def test_public_projection_ignores_invisible_values_and_common_offset() -> None:
    raw = torch.tensor([[[1.0, 2.0, float("nan"), 4.0], [3.0, 5.0, 7.0, 9.0]]])
    mask = torch.tensor([[[True, True, False, False], [True, False, False, False]]])
    reference = project_visible_observation_log_precision(
        raw=raw, mask=mask, max_abs=2.0,
    )
    shifted = raw.clone()
    shifted[mask] += 17.0
    actual = project_visible_observation_log_precision(shifted, mask, 2.0)
    assert torch.equal(actual, reference)
    assert torch.equal(actual[~mask], torch.zeros_like(actual[~mask]))


def test_global_only_modulates_nonzero_local_contrasts() -> None:
    data = _fixture()
    model = _model(data).eval()
    neutral = dict(data)
    neutral["event_feature"] = model.event_mean.view(1, 1, 1, -1).expand_as(
        data["event_feature"]
    ).clone()
    neutral["anchor_feature"] = model.anchor_mean.view(1, 1, -1).expand_as(
        data["anchor_feature"]
    ).clone()
    neutral["pair_feature"] = model.pair_mean.view(1, 1, 1, -1).expand_as(
        data["pair_feature"]
    ).clone()
    first = model(**neutral)
    changed = dict(neutral)
    changed["global_feature"] = neutral["global_feature"] * 13.0 + 7.0
    second = model(**changed)
    for name in ("observation_log_precision", "anchor_alpha", "center_alpha"):
        assert torch.equal(first[name], second[name])
    assert torch.equal(
        first["observation_log_precision"],
        torch.zeros_like(first["observation_log_precision"]),
    )
    assert torch.equal(
        first["anchor_alpha"][data["anchor_mask"]],
        torch.full_like(first["anchor_alpha"][data["anchor_mask"]], 0.5),
    )
    assert torch.equal(first["center_alpha"], torch.full_like(first["center_alpha"], 0.5))

    nonneutral = dict(data)
    global_feature = data["global_feature"].clone().requires_grad_()
    nonneutral["global_feature"] = global_feature
    output = model(**nonneutral)
    objective = (
        output["observation_log_precision"].square().sum()
        + output["anchor_alpha"].sum() + output["center_alpha"].sum()
    )
    gradient = torch.autograd.grad(objective, global_feature)[0]
    assert bool(torch.isfinite(gradient).all())
    assert bool((gradient.abs() > 0).any())


def test_missing_q0_support_forces_all_prior_alphas_to_history() -> None:
    data = _fixture()
    data["anchor_mask"][0].zero_()
    model = _model(data)
    output = model(**data)
    assert torch.equal(
        output["anchor_alpha"][0], torch.zeros_like(output["anchor_alpha"][0]),
    )
    assert output["center_alpha"][0].item() == 0.0


def test_two_irls_steps_share_parameters_and_accept_residual_leverage() -> None:
    data = _fixture()
    model = _model(data)
    calls: list[tuple[int, dict[str, torch.Tensor]]] = []
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())

    def solver(step: int, weights: dict[str, torch.Tensor]):
        calls.append((step, weights))
        assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
        if step == 0:
            return {
                "standardized_residual": torch.randn_like(
                    weights["observation_log_precision"]
                ),
                "leverage": torch.rand_like(
                    weights["observation_log_precision"]
                ),
            }
        return {"accepted": True}

    result = model.run_two_step_irls(solver, **data)
    assert [step for step, _ in calls] == [0, 1]
    assert result["solution"] == {"accepted": True}
    assert not torch.equal(
        result["step1"]["observation_log_precision"],
        result["step2"]["observation_log_precision"],
    )


def test_all_trainable_parameters_receive_finite_gradients() -> None:
    data = _fixture()
    model = _model(data).train()
    residual = torch.randn_like(data["event_mask"], dtype=torch.float32)
    leverage = torch.rand_like(residual)
    output = model(
        **data, current_standardized_residual=residual,
        current_leverage=leverage,
    )
    loss = (
        output["observation_log_precision"][data["event_mask"]].square().mean()
        + output["anchor_alpha"][data["anchor_mask"]].mean()
        + output["center_alpha"].mean()
    )
    loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable and all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in trainable
    )


def test_invisible_nonfinite_padding_has_finite_forward_and_backward() -> None:
    clean = _fixture()
    reference_model = _model(clean).train()
    reference = reference_model(**clean)

    padded = {name: value.clone() for name, value in clean.items()}
    padded["event_feature"][~padded["event_mask"]] = float("nan")
    padded["anchor_feature"][~padded["anchor_mask"]] = float("inf")
    padded["pair_feature"][~padded["pair_mask"]] = -float("inf")
    model = _model(padded).train()
    output = model(**padded)
    for name in ("observation_log_precision", "anchor_alpha", "center_alpha"):
        assert bool(torch.isfinite(output[name]).all())
        torch.testing.assert_close(output[name], reference[name], rtol=0.0, atol=0.0)

    loss = (
        output["observation_log_precision"][padded["event_mask"]].square().mean()
        + output["anchor_alpha"][padded["anchor_mask"]].mean()
        + output["center_alpha"].mean()
    )
    loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable and all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in trainable
    )


@pytest.mark.parametrize(
    ("feature_name", "index", "value", "message"),
    (
        ("event_feature", (0, 1, 0, 0), float("nan"), "event feature"),
        ("anchor_feature", (0, 0, 0), float("inf"), "anchor feature"),
        ("pair_feature", (0, 1, 0, 0), -float("inf"), "pair feature"),
    ),
)
def test_visible_nonfinite_local_feature_is_rejected(
    feature_name: str,
    index: tuple[int, ...],
    value: float,
    message: str,
) -> None:
    data = _fixture()
    model = _model(data)
    bad = {name: tensor.clone() for name, tensor in data.items()}
    bad[feature_name][index] = value
    with pytest.raises(ValueError, match=f"visible {message} is non-finite"):
        model(**bad)
