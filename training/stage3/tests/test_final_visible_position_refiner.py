from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip("torch")

from training.stage3.final_visible_position_refiner import (
    FinalVisiblePositionRefiner,
    FrozenVisibleSystemFinalRefiner,
    final_visible_position_loss,
)
from training.stage3.observable_future_model import AnonymousCandidateFutureExpert
from training.stage3.ordinal_visible_selector import (
    OrdinalVisibleFutureModel,
    OrdinalVisibleProgressSelector,
)


def _batch(batch_size: int = 3, device: str = "cpu") -> dict[str, torch.Tensor]:
    history_time = torch.linspace(
        -0.35, 0.0, 8, device=device,
    ).repeat(batch_size, 1)
    history_rel = torch.zeros(batch_size, 8, 3, device=device)
    history_rel[..., 0] = 0.2 * history_time
    history_rel[..., 1] = -0.1 * history_time
    history_dt = torch.zeros(batch_size, 8, device=device)
    history_dt[:, 1:] = history_time[:, 1:] - history_time[:, :-1]
    steps = torch.arange(
        -6, 7, dtype=torch.float32, device=device,
    ).repeat(batch_size, 1)
    relation = torch.stack(
        (0.18 * steps, -0.08 * steps, 0.02 * steps), dim=-1,
    )
    relation[:, 6] = 0.0
    tau = torch.tensor(
        [0.0, 0.05, 0.12, 0.22, 0.36, 0.5], device=device,
    )
    target = torch.tensor(
        [0, 0, 0, 1, 1, 2], device=device,
    ).repeat(batch_size, 1)
    target[1] *= -1
    target[2] = 0
    target_visible = torch.zeros(batch_size, tau.numel(), 3, device=device)
    target_visible[..., 0] = target.to(torch.float32) * 0.18
    return {
        "history_position_rel_m": history_rel,
        "history_time_s": history_time,
        "history_dt_s": history_dt,
        "history_switch_step": torch.zeros(batch_size, 8, device=device),
        "history_mask": torch.ones(
            batch_size, 8, dtype=torch.bool, device=device,
        ),
        "current_position_m": torch.tensor(
            [[4.0, 1.0, 0.2]], device=device,
        ).repeat(batch_size, 1),
        "candidate_relation_m": relation,
        "candidate_step": steps,
        "candidate_mask": torch.ones(
            batch_size, 13, dtype=torch.bool, device=device,
        ),
        "candidate_confidence": torch.ones(batch_size, 13, device=device),
        "tau_s": tau,
        "target_switch_count": target,
        "target_visible_delta_m": target_visible,
        "target_query_mask": torch.ones(
            batch_size, tau.numel(), dtype=torch.bool, device=device,
        ),
    }


def _system(device: str = "cpu") -> FrozenVisibleSystemFinalRefiner:
    trajectory = AnonymousCandidateFutureExpert(
        "combined", channels=16, dropout=0.0,
        trained_horizon_s=0.5, maximum_absolute_step=6,
    ).to(device).eval()
    selector = OrdinalVisibleProgressSelector(
        frozen_context_features=4 * trajectory.channels + 15,
        channels=32, dropout=0.0,
        trained_horizon_s=trajectory.trained_horizon_s,
        maximum_absolute_step=trajectory.maximum_absolute_step,
    ).to(device)
    base = OrdinalVisibleFutureModel(trajectory, selector)
    refiner = FinalVisiblePositionRefiner(
        frozen_context_features=4 * trajectory.channels + 15,
        channels=32, dropout=0.0,
        trained_horizon_s=trajectory.trained_horizon_s,
        maximum_absolute_step=trajectory.maximum_absolute_step,
    )
    return FrozenVisibleSystemFinalRefiner(base, refiner).to(device)


def _state_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def test_zero_initialization_is_bit_exact_to_frozen_v66() -> None:
    system = _system().eval()
    batch = _batch()
    with torch.no_grad():
        expected = system.base(batch)["position_m"]
        prediction = system(batch)
    assert torch.equal(prediction["position_residual_m"], torch.zeros_like(expected))
    assert torch.equal(prediction["position_m"], expected)


def test_training_changes_only_the_final_refiner() -> None:
    system = _system().train()
    batch = _batch()
    frozen_before = _state_sha(system.base)
    optimizer = torch.optim.AdamW(system.refiner.parameters(), lr=1e-3)
    prediction = system(batch)
    objective, _ = final_visible_position_loss(prediction, batch)
    objective.backward()
    gradients = [
        parameter.grad for parameter in system.refiner.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert any(bool(torch.any(value != 0)) for value in gradients)
    assert all(parameter.grad is None for parameter in system.base.parameters())
    optimizer.step()
    assert _state_sha(system.base) == frozen_before
    assert any(
        bool(torch.any(parameter.detach() != 0))
        for parameter in system.refiner.residual[-1].parameters()
    )


def test_query_permutation_only_reorders_final_outputs() -> None:
    system = _system().eval()
    batch = _batch()
    with torch.no_grad():
        baseline = system(batch)
    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    changed = dict(batch)
    changed["tau_s"] = batch["tau_s"][permutation]
    with torch.no_grad():
        result = system(changed)
    for key in (
        "position_m", "position_residual_m", "unrefined_position_m",
        "selected_probability", "route_entropy",
    ):
        torch.testing.assert_close(result[key], baseline[key][:, permutation])


def test_candidate_permutation_preserves_final_position() -> None:
    system = _system().eval()
    batch = _batch()
    with torch.no_grad():
        baseline = system(batch)["position_m"]
    permutation = torch.tensor([12, 8, 2, 3, 10, 6, 1, 11, 0, 5, 9, 4, 7])
    changed = dict(batch)
    for key in (
        "candidate_relation_m", "candidate_step", "candidate_mask",
        "candidate_confidence",
    ):
        changed[key] = batch[key][:, permutation]
    with torch.no_grad():
        result = system(changed)["position_m"]
    torch.testing.assert_close(result, baseline)


def test_config_excludes_truth_identity_and_decoder_shortcuts() -> None:
    config = _system().refiner.config
    assert config["future_truth_forward_input"] is False
    assert config["physical_id_input"] is False
    assert config["motion_class_forward_input"] is False
    assert config["candidate_wise_head"] is False
    assert config["physics_decoder"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_forward_backward_is_finite() -> None:
    system = _system("cuda").train()
    batch = _batch(device="cuda")
    prediction = system(batch)
    objective, _ = final_visible_position_loss(prediction, batch)
    assert bool(torch.isfinite(objective))
    objective.backward()
    gradients = [
        parameter.grad for parameter in system.refiner.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(value).all()) for value in gradients)
