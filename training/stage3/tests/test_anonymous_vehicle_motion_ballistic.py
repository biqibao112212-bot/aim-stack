from __future__ import annotations

import numpy as np
import pytest
import torch

from training.stage3.anonymous_vehicle_motion import AnonymousVehicleFutureModel
from training.stage3.anonymous_vehicle_motion_v2 import (
    VisibilityAwareAnonymousVehicleFutureModel,
)
from training.stage3.continuous_invariant_anonymous_future import (
    ContinuousInvariantAnonymousFutureModel,
)
from training.stage3.evaluate_anonymous_vehicle_motion_ballistic import (
    OPPOSITE_SOURCE_FAILURE,
    _canonical_ballistic_label,
    _distance_rows,
    _load_motion_checkpoint,
    _pair_reverse_flag,
    _selection_diagnostics,
)
from training.stage3.evaluate_final_visible_position_ballistic import TruthState
from training.stage3.observable_future_pnp_ab import state_dict_sha256
from training.stage3.train_anonymous_vehicle_motion import RUN_SCHEMA
from training.stage3.train_anonymous_vehicle_motion_v2 import RUN_SCHEMA as V2_RUN_SCHEMA
from training.stage3.train_continuous_invariant_anonymous_future import (
    RUN_SCHEMA as V3_RUN_SCHEMA,
)
from training.stage3.train_visibility_aware_expert_router import (
    RUN_SCHEMA as ROUTER_RUN_SCHEMA,
)


def test_selection_diagnostics_separates_exact_step_from_modulo4_role() -> None:
    queries = {
        "predicted_switch_count": np.asarray([0, 5, -3, 2], dtype=np.int64),
        "target_switch_count": np.asarray([0, 1, 1, -2], dtype=np.int64),
        "hard_error_m": np.asarray([0.1, 0.2, 0.5, 0.3], dtype=np.float32),
        "conditional_error_m": np.asarray([0.1, 0.2, 0.2, 0.4], dtype=np.float32),
    }
    metrics = _selection_diagnostics(queries)
    assert metrics["exact_signed_step_accuracy"] == 0.25
    assert metrics["modulo4_physical_role_accuracy"] == 1.0
    assert metrics["exact_wrong_count"] == 3
    assert metrics["exact_wrong_but_same_role_count"] == 3
    assert metrics["wrong_role_count"] == 0


def test_distance_rows_report_hard_conditional_and_both_selection_definitions() -> None:
    queries = {
        "truth_distance_m": np.asarray([2.2, 2.8, 6.2], dtype=np.float32),
        "hard_error_m": np.asarray([0.05, 0.15, 0.30], dtype=np.float32),
        "conditional_error_m": np.asarray([0.04, 0.10, 0.20], dtype=np.float32),
        "predicted_switch_count": np.asarray([0, 5, 2], dtype=np.int64),
        "target_switch_count": np.asarray([0, 1, 1], dtype=np.int64),
    }
    rows = _distance_rows(queries)
    two_to_three = next(row for row in rows if row["distance_bin_m"] == "[2,3)")
    overall = next(row for row in rows if row["distance_bin_m"] == "[1,7) overall")
    assert two_to_three["count"] == 2
    assert np.isclose(two_to_three["hard_p50_mm"], 100.0)
    assert np.isclose(two_to_three["conditional_p50_mm"], 70.0)
    assert np.isclose(two_to_three["exact_step_accuracy"], 0.5)
    assert np.isclose(two_to_three["modulo4_role_accuracy"], 1.0)
    assert overall["count"] == 3


def test_v3_selection_diagnostics_reports_role_without_fake_signed_step() -> None:
    queries = {
        "predicted_switch_count": np.asarray([0, 1, 2], dtype=np.int64),
        "predicted_role": np.asarray([0, 1, 2], dtype=np.int64),
        "signed_step_available": np.zeros(3, dtype=np.bool_),
        "target_switch_count": np.asarray([0, 5, -1], dtype=np.int64),
        "hard_error_m": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "conditional_error_m": np.asarray([0.1, 0.2, 0.2], dtype=np.float32),
    }
    metrics = _selection_diagnostics(queries)
    assert metrics["exact_signed_step_accuracy"] is None
    assert metrics["exact_signed_step_available"] is False
    assert metrics["modulo4_physical_role_accuracy"] == pytest.approx(2 / 3)


def test_pair_reverse_flag_is_audit_only() -> None:
    assert not _pair_reverse_flag("0000000000000000-rest")
    assert _pair_reverse_flag("0000000000000004-rest")


def test_canonical_ballistic_label_never_applies_a_second_reflection(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_label(*args, **kwargs):
        captured.update(kwargs)
        return {"flight_time_s": 0.1}

    monkeypatch.setattr(
        "training.stage3.evaluate_anonymous_vehicle_motion_ballistic._ballistic_label",
        fake_label,
    )
    state = TruthState(
        history_position_m=np.zeros((1, 4, 3), dtype=np.float32),
        event_mask=np.ones(1, dtype=np.bool_),
        event_time_s=np.zeros(1, dtype=np.float32),
        q0_position_m=np.zeros((4, 3), dtype=np.float32),
        center_m=np.zeros(3, dtype=np.float32),
        velocity_mps=np.zeros(3, dtype=np.float32),
        yaw_rate_rad_s=0.0,
    )
    label, reason = _canonical_ballistic_label(
        state, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
        bullet_speed_mps=22.0, dense_step_s=0.001,
    )
    assert label is not None and reason is None
    assert captured["reverse_direction"] is False


def test_canonical_ballistic_label_fails_closed_only_for_opposite_jump(monkeypatch) -> None:
    state = TruthState(
        history_position_m=np.zeros((1, 4, 3), dtype=np.float32),
        event_mask=np.ones(1, dtype=np.bool_),
        event_time_s=np.zeros(1, dtype=np.float32),
        q0_position_m=np.zeros((4, 3), dtype=np.float32),
        center_m=np.zeros(3, dtype=np.float32),
        velocity_mps=np.zeros(3, dtype=np.float32),
        yaw_rate_rad_s=0.0,
    )
    monkeypatch.setattr(
        "training.stage3.evaluate_anonymous_vehicle_motion_ballistic._ballistic_label",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError(OPPOSITE_SOURCE_FAILURE)),
    )
    label, reason = _canonical_ballistic_label(
        state, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
        bullet_speed_mps=22.0, dense_step_s=0.001,
    )
    assert label is None and reason == "opposite_source_jump"

    monkeypatch.setattr(
        "training.stage3.evaluate_anonymous_vehicle_motion_ballistic._ballistic_label",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("unexpected")),
    )
    with pytest.raises(ValueError, match="unexpected"):
        _canonical_ballistic_label(
            state, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
            bullet_speed_mps=22.0, dense_step_s=0.001,
        )


def test_motion_checkpoint_loader_requires_update_2100(tmp_path) -> None:
    model = AnonymousVehicleFutureModel(channels=32, dropout=0.0, message_layers=2)
    payload = {
        "schema_version": RUN_SCHEMA,
        "fixed_endpoint": True,
        "checkpoint_role": "fixed_final_endpoint",
        "progress": {"global_update": 2100},
        "provenance": {
            "oracle_association": True,
            "deployable_pipeline": False,
            "test_accessed": False,
        },
        "model_config": model.config,
        "model": model.state_dict(),
        "model_state_dict_sha256": state_dict_sha256(model.state_dict()),
    }
    accepted = tmp_path / "accepted.pt"
    torch.save(payload, accepted)
    loaded, info = _load_motion_checkpoint(accepted)
    assert info["global_update"] == 2100
    assert state_dict_sha256(loaded.state_dict()) == payload["model_state_dict_sha256"]

    payload["progress"]["global_update"] = 2099
    rejected = tmp_path / "rejected.pt"
    torch.save(payload, rejected)
    with pytest.raises(ValueError, match="update 2100"):
        _load_motion_checkpoint(rejected)


def test_motion_checkpoint_loader_accepts_v2_only_at_update_2400(tmp_path) -> None:
    model = VisibilityAwareAnonymousVehicleFutureModel(
        channels=32, dropout=0.0, message_layers=2,
        basis_count=6, latent_experts=3,
    )
    payload = {
        "schema_version": V2_RUN_SCHEMA,
        "fixed_endpoint": True,
        "checkpoint_role": "fixed_final_endpoint",
        "progress": {"global_update": 2400},
        "provenance": {
            "oracle_association": True,
            "deployable_pipeline": False,
            "test_accessed": False,
        },
        "model_config": model.config,
        "model": model.state_dict(),
        "model_state_dict_sha256": state_dict_sha256(model.state_dict()),
    }
    accepted = tmp_path / "accepted-v2.pt"
    torch.save(payload, accepted)
    loaded, info = _load_motion_checkpoint(accepted)
    assert info["global_update"] == 2400
    assert info["model_version"] == "v2"
    assert state_dict_sha256(loaded.state_dict()) == payload["model_state_dict_sha256"]

    payload["progress"]["global_update"] = 2100
    rejected = tmp_path / "rejected-v2.pt"
    torch.save(payload, rejected)
    with pytest.raises(ValueError, match="update 2400"):
        _load_motion_checkpoint(rejected)


def test_motion_checkpoint_loader_accepts_v3_only_at_update_2100(tmp_path) -> None:
    model = ContinuousInvariantAnonymousFutureModel(
        channels=32, dropout=0.0, message_layers=2, basis_count=6,
    )
    payload = {
        "schema_version": V3_RUN_SCHEMA,
        "fixed_endpoint": True,
        "checkpoint_role": "fixed_final_endpoint",
        "progress": {"global_update": 2100},
        "provenance": {
            "oracle_association": True,
            "deployable_pipeline": False,
            "test_accessed": False,
        },
        "model_config": model.config,
        "model": model.state_dict(),
        "model_state_dict_sha256": state_dict_sha256(model.state_dict()),
    }
    accepted = tmp_path / "accepted-v3.pt"
    torch.save(payload, accepted)
    loaded, info = _load_motion_checkpoint(accepted)
    assert info["global_update"] == 2100
    assert info["model_version"] == "v3"
    assert state_dict_sha256(loaded.state_dict()) == payload["model_state_dict_sha256"]

    payload["progress"]["global_update"] = 2099
    rejected = tmp_path / "rejected-v3.pt"
    torch.save(payload, rejected)
    with pytest.raises(ValueError, match="update 2100"):
        _load_motion_checkpoint(rejected)


def test_motion_checkpoint_loader_accepts_hard_router_only_at_update_600(
    tmp_path,
) -> None:
    model = VisibilityAwareAnonymousVehicleFutureModel(
        channels=32, dropout=0.0, message_layers=2,
        basis_count=6, latent_experts=3,
    )
    payload = {
        "schema_version": ROUTER_RUN_SCHEMA,
        "fixed_endpoint": True,
        "checkpoint_role": "fixed_final_endpoint",
        "progress": {"global_update": 600},
        "provenance": {
            "oracle_association": True,
            "deployable_pipeline": False,
            "test_accessed": False,
        },
        "model_config": model.config,
        "model": model.state_dict(),
        "model_state_dict_sha256": state_dict_sha256(model.state_dict()),
    }
    accepted = tmp_path / "accepted-router.pt"
    torch.save(payload, accepted)
    loaded, info = _load_motion_checkpoint(accepted)
    assert info["global_update"] == 600
    assert info["model_version"] == "v2_hard_router"
    assert state_dict_sha256(loaded.state_dict()) == payload["model_state_dict_sha256"]

    payload["progress"]["global_update"] = 599
    rejected = tmp_path / "rejected-router.pt"
    torch.save(payload, rejected)
    with pytest.raises(ValueError, match="update 600"):
        _load_motion_checkpoint(rejected)
