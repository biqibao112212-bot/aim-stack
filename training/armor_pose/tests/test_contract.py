from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch

from training.armor_pose.data import ArmorPoseDataset, FORBIDDEN_ONLINE_TOKENS, load_development_pack
from training.armor_pose.dense_correspondence_head import (
    DenseCorrespondenceNet, LegacyDenseCorrespondenceNet, build_dense_correspondence_net,
)
from training.armor_pose.gpu_pnp import solve_weighted_planar_pnp
from training.armor_pose.labels import CANONICAL_CORNER_UV, canonical_uv_to_object_points
from training.armor_pose.sparse_prob_head import ProbabilisticCornerNet
from training.corner_pnp.pnp import NOMINAL_OBJECT_POINTS_M


TRAIN_PACK = Path(
    "/home/potato/Projects/仿真/runtime/autoaim-b-corner-pnp-20260824-v1/packs/"
    "train-v12-pilot-calibrated-stable.npz"
)


def test_canonical_uv_preserves_production_corner_order() -> None:
    points = canonical_uv_to_object_points(CANONICAL_CORNER_UV)
    assert torch.allclose(points, torch.as_tensor(NOMINAL_OBJECT_POINTS_M, dtype=torch.float32), atol=1.0e-7)


def test_dataset_separates_online_inputs_from_supervision() -> None:
    dataset = ArmorPoseDataset(load_development_pack(TRAIN_PACK, expected_split="train"), maximum_samples=1)
    item = dataset[0]
    assert set(item["online"]) == {
        "patch_rgb", "detector_geometry", "raw_corners_px", "raw_corners_patch_px",
        "patch_to_image_h", "raw_scale_px", "intrinsics",
    }
    assert not any(any(token in name for token in FORBIDDEN_ONLINE_TOKENS) for name in item["online"])
    assert "target_corners_px" in item["supervision"]


def test_online_signatures_do_not_accept_reference_or_target() -> None:
    forbidden = ("target", "truth", "reference", "exact", "motion", "session", "history")
    for function in (ProbabilisticCornerNet.forward, DenseCorrespondenceNet.forward,
                     LegacyDenseCorrespondenceNet.forward, solve_weighted_planar_pnp):
        names = tuple(inspect.signature(function).parameters)
        assert not any(token in name for token in forbidden for name in names)


def test_dense_checkpoint_config_reconstructs_legacy_and_projective_versions() -> None:
    for original_type in (LegacyDenseCorrespondenceNet, DenseCorrespondenceNet):
        original = original_type()
        config = original.config
        if original_type is LegacyDenseCorrespondenceNet:
            # Completed V19 checkpoints predate the explicit architecture key;
            # their frozen family is sufficient to reconstruct them.
            config = {key: value for key, value in config.items() if key != "architecture"}
        reconstructed = build_dense_correspondence_net(model_config=config)
        reconstructed.load_state_dict(original.state_dict(), strict=True)
        assert reconstructed.config["family"] == original.config["family"]
        assert reconstructed.config["architecture"] == original.config["architecture"]


def test_v19_plan_never_names_sealed_pack_as_development_input() -> None:
    import json
    plan = json.loads((Path(__file__).parents[1] / "experiment_plan_v19.json").read_text())
    development = plan["development_data"]
    assert "test-v15" not in development["train_pack"]
    assert "test-v18" not in development["train_pack"]
    assert development["exploratory_only"] is True
