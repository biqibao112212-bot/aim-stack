from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from training.stage3.causal_physical_state_model import (
    FactorizedExpertPhysicalPredictor,
)
from training.stage3.factorized_expert_loss import factorized_expert_loss
from training.stage3.pnp_state_targets import truth_trajectory_targets
from training.stage3.train_causal_physical_ab import _validate
from training.stage3.train_factorized_motion_experts import (
    _expert_selection_tuple,
    _rotation_augmented_batch,
)


GEOMETRY = torch.tensor([
    [0.01, -0.21, -0.061], [0.22, 0.00, -0.075],
    [0.01, 0.21, -0.061], [-0.20, 0.00, -0.075],
], dtype=torch.float32)
MEAN = torch.tensor([4.0, 0.0, -0.07])
STD = torch.tensor([2.0, 1.0, 0.1])


def _model() -> FactorizedExpertPhysicalPredictor:
    torch.manual_seed(31)
    return FactorizedExpertPhysicalPredictor(
        geometry=GEOMETRY, position_mean=MEAN, position_std=STD,
        channels=16, dropout=0.0, history_events=8,
    )


def _decode(
    model: FactorizedExpertPhysicalPredictor, time: torch.Tensor, *,
    velocity: torch.Tensor = torch.tensor([0.8, -0.3, 0.0]),
    omega: float = 1.2,
) -> torch.Tensor:
    center0 = torch.tensor([4.0, 0.2, 0.0])
    center = center0 + time[:, None] * velocity
    angle = 0.4 + time * omega
    phase = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
    return model.decoder(center[None], phase[None])[0]


def _batch() -> dict[str, torch.Tensor]:
    model = _model()
    history_time = torch.linspace(-0.14, 0.0, 8)
    tau = torch.tensor([0.0, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9])
    history = _decode(model, history_time)
    future = _decode(model, tau)
    mask = torch.ones(1, 8, 4, dtype=torch.bool)
    slot = torch.zeros(1, 8, 4, 2)
    obs = torch.cat((((history[None] - MEAN) / STD), slot), dim=-1)
    return {
        "obs": obs,
        "history_position_m": history[None],
        "obs_mask": mask,
        "event_mask": torch.ones(1, 8, dtype=torch.bool),
        "event_time_s": history_time[None],
        "tau": tau[None],
        "future_position": future[None],
        "motion_class": torch.tensor([3]),
        "rule_query": torch.ones(1, 8, dtype=torch.bool),
        "distance_m": torch.tensor([4.0]),
    }


def test_factorized_forward_has_independent_experts_and_hard_rigid_decode() -> None:
    model = _model().eval()
    batch = _batch()
    output = model(
        batch["obs"], batch["obs_mask"], batch["event_mask"],
        batch["event_time_s"], batch["tau"],
    )
    assert output["velocity_expert"].shape == (1, 3)
    assert output["omega_expert"].shape == (1,)
    assert output["move_logit"].shape == output["rotate_logit"].shape == (1,)
    assert torch.equal(output["velocity"], (
        output["move_active"].float()[:, None] * output["velocity_expert"]
    ))
    assert torch.equal(output["omega"], (
        output["rotate_active"].float() * output["omega_expert"]
    ))
    pair_i, pair_j = torch.triu_indices(4, 4, offset=1)
    expected = torch.linalg.vector_norm(
        GEOMETRY[pair_i] - GEOMETRY[pair_j], dim=-1
    )
    actual = torch.linalg.vector_norm(
        output["position_mean"][:, :, pair_i]
        - output["position_mean"][:, :, pair_j], dim=-1,
    )
    assert torch.allclose(actual, expected.view(1, 1, -1), atol=2e-6)


def test_closed_gate_cannot_silence_positive_expert_supervision() -> None:
    model = _model()
    batch = _batch()
    target = batch["future_position"][:, :4]
    tau = batch["tau"][:, :4]
    truth = truth_trajectory_targets(target, tau, GEOMETRY, rule_queries=4)
    velocity_expert = torch.zeros(1, 3, requires_grad=True)
    omega_expert = torch.zeros(1, requires_grad=True)
    move_logit = torch.tensor([-20.0], requires_grad=True)
    rotate_logit = torch.tensor([-20.0], requires_grad=True)
    prediction = {
        "position_mean": target.clone(),
        "center0": truth["center0"].clone().requires_grad_(),
        "phase0": truth["phase0"].clone().requires_grad_(),
        "velocity_expert": velocity_expert,
        "move_logit": move_logit,
        "velocity": torch.zeros_like(velocity_expert),
        "omega_expert": omega_expert,
        "rotate_logit": rotate_logit,
        "omega": torch.zeros_like(omega_expert),
    }
    loss, parts = factorized_expert_loss(
        prediction, target, tau, torch.ones_like(tau, dtype=torch.bool), GEOMETRY,
    )
    loss.backward()
    assert float(parts["velocity_expert"]) > 0
    assert float(parts["omega_expert"]) > 0
    assert float(velocity_expert.grad.abs().max()) > 0
    assert float(omega_expert.grad.abs().max()) > 0
    assert float(move_logit.grad) < 0
    assert float(rotate_logit.grad) < 0


def test_factorized_loss_does_not_require_gated_state_outputs() -> None:
    batch = _batch()
    model = _model()
    prediction = model(
        batch["obs"], batch["obs_mask"], batch["event_mask"],
        batch["event_time_s"], batch["tau"],
    )
    prediction.pop("velocity")
    prediction.pop("omega")
    loss, parts = factorized_expert_loss(
        prediction, batch["future_position"][:, :4], batch["tau"][:, :4],
        batch["rule_query"][:, :4], GEOMETRY,
    )
    assert torch.isfinite(loss)
    assert set(parts) >= {
        "center0", "phase0", "velocity_expert", "omega_expert",
        "move_gate", "rotate_gate",
    }


def test_rotation_augmentation_preserves_motion_and_geometry() -> None:
    batch = _batch()
    augmented = _rotation_augmented_batch(
        batch, GEOMETRY, MEAN, STD,
        torch.tensor([torch.pi / 2]), torch.tensor([[0.1, -0.2]]),
    )
    before = truth_trajectory_targets(
        batch["future_position"][:, :4], batch["tau"][:, :4],
        GEOMETRY, rule_queries=4,
    )
    after = truth_trajectory_targets(
        augmented["future_position"][:, :4], batch["tau"][:, :4],
        GEOMETRY, rule_queries=4,
    )
    assert torch.allclose(
        torch.linalg.vector_norm(before["velocity"], dim=-1),
        torch.linalg.vector_norm(after["velocity"], dim=-1), atol=1e-5,
    )
    assert torch.allclose(before["omega"], after["omega"], atol=1e-5)
    assert torch.allclose(
        after["center0"][:, :2], before["center0"][:, :2] + torch.tensor([[0.1, -0.2]]),
        atol=1e-5,
    )
    assert not torch.allclose(before["velocity"], after["velocity"])
    assert torch.isfinite(augmented["obs"]).all()


def test_augmentation_forward_inputs_do_not_depend_on_future_truth() -> None:
    batch = _batch()
    changed_future = {key: value.clone() for key, value in batch.items()}
    changed_future["future_position"][..., :2] += torch.tensor([1.7, -0.9])
    kwargs = {
        "geometry": GEOMETRY, "position_mean": MEAN, "position_std": STD,
        "angle": torch.tensor([0.7]),
        "translation_xy": torch.tensor([[0.1, -0.2]]),
    }
    original = _rotation_augmented_batch(batch, **kwargs)
    changed = _rotation_augmented_batch(changed_future, **kwargs)
    assert torch.equal(original["obs"], changed["obs"])
    assert torch.equal(
        original["history_position_m"], changed["history_position_m"]
    )
    assert not torch.equal(original["future_position"], changed["future_position"])


def test_validation_reports_gate_recall_and_false_positive_rate() -> None:
    class OneBatch(Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            return {key: value[0] for key, value in _batch().items()}

    report = _validate(
        _model().eval(), DataLoader(OneBatch(), batch_size=1),
        torch.device("cpu"), SimpleNamespace(amp="off"),
    )
    gates = report["expert_gate_diagnostics"]
    assert gates["move"]["positive_count"] == 1
    assert gates["rotate"]["positive_count"] == 1


def test_expert_selection_prioritizes_worst_dynamic_class() -> None:
    metrics = {
        "strata": {"motion_class": {
            "linear": {"q3_trajectory_eligible_motion": {"p95_m": 0.3, "median_m": 0.1}},
            "spin": {"q3_trajectory_eligible_motion": {"p95_m": 0.2, "median_m": 0.05}},
            "linear_and_spin": {"q3_trajectory_eligible_motion": {"p95_m": 0.4, "median_m": 0.2}},
        }},
        "queries": [{}, {}, {}, {"trajectory_eligible": {
            "motion_delta": {"p95_m": 0.35},
        }}],
        "trajectory_eligible_state_q0": {"p95_m": 0.08},
    }
    assert _expert_selection_tuple(metrics) == (0.4, 0.35, 0.08, 0.2)


def test_factorized_forward_has_no_truth_or_motion_class_argument() -> None:
    parameters = inspect.signature(FactorizedExpertPhysicalPredictor.forward).parameters
    assert "future" not in parameters
    assert "motion_class" not in parameters
    assert "velocity" not in parameters
