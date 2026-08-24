from __future__ import annotations

import torch

from training.armor_pose.dense_correspondence_head import DenseCorrespondenceNet, stratified_correspondences
from training.armor_pose.gpu_pnp import _so3_exp, project_points, solve_weighted_planar_pnp
from training.armor_pose.fusion import SparseDensePoseNet
from training.armor_pose.labels import dense_surface_labels
from training.armor_pose.sparse_prob_head import ProbabilisticCornerNet
from training.corner_pnp.pnp import NOMINAL_OBJECT_POINTS_M


def _cuda() -> torch.device:
    assert torch.cuda.is_available(), "V19 tests fail closed when CUDA is unavailable"
    return torch.device("cuda")


def _online(batch: int = 2) -> dict[str, torch.Tensor]:
    device = _cuda()
    raw_patch = torch.tensor(
        [[[31.0, 48.0], [31.0, 16.0], [96.0, 16.0], [96.0, 48.0]]], device=device
    ).expand(batch, -1, -1).clone()
    raw_full = raw_patch + raw_patch.new_tensor([300.0, 200.0])
    transform = torch.eye(3, device=device)[None].expand(batch, -1, -1).clone()
    transform[:, :2, 2] = transform.new_tensor([300.0, 200.0])
    return {
        "patch": torch.rand(batch, 3, 64, 128, device=device),
        "geometry": torch.randn(batch, 15, device=device),
        "raw_patch": raw_patch,
        "raw_full": raw_full,
        "transform": transform,
        "scale": torch.full((batch,), 45.0, device=device),
        "intrinsics": torch.tensor([[1200.0, 1200.0, 720.0, 540.0]], device=device).expand(batch, -1).clone(),
    }


def test_sparse_zero_initialization_preserves_raw_and_covariance_is_spd_cuda() -> None:
    value = _online()
    model = ProbabilisticCornerNet().to(_cuda()).eval()
    prediction = model(value["patch"], value["geometry"], value["raw_full"], value["raw_patch"],
                       value["transform"], value["scale"])
    assert prediction.image_mean.is_cuda
    assert torch.allclose(prediction.image_mean, value["raw_full"], atol=3.0e-5)
    assert torch.all(torch.linalg.cholesky_ex(prediction.image_covariance).info == 0)
    assert torch.isfinite(prediction.image_covariance).all()


def test_dense_support_uv_and_sampling_stay_on_cuda() -> None:
    value = _online()
    support, uv, edge = dense_surface_labels(value["raw_patch"])
    assert support.is_cuda and uv.is_cuda and edge.is_cuda
    assert support.sum() > 0 and edge.max() <= 1.0
    model = DenseCorrespondenceNet().to(_cuda())
    prediction = model(value["patch"], value["geometry"])
    correspondences = stratified_correspondences(prediction, value["transform"], count=64)
    assert correspondences.image_points.shape == (2, 64, 2)
    assert correspondences.object_points.shape == (2, 64, 3)
    assert correspondences.weights.is_cuda
    assert torch.isfinite(correspondences.object_points).all()


def test_truth_free_gpu_pnp_recovers_synthetic_translation() -> None:
    device = _cuda()
    object_points = torch.as_tensor(NOMINAL_OBJECT_POINTS_M, dtype=torch.float32, device=device)[None]
    intrinsics = torch.tensor([[1200.0, 1180.0, 720.0, 540.0]], device=device)
    rotation = _so3_exp(torch.tensor([[0.10, -0.20, 0.05]], device=device))
    translation = torch.tensor([[0.20, -0.10, 4.0]], device=device)
    image_points, _ = project_points(rotation, translation, object_points, intrinsics)
    result = solve_weighted_planar_pnp(image_points, object_points, intrinsics, iterations=12)
    assert result.translation_m.is_cuda
    assert result.valid[0, 0]
    assert 1.0e3 * torch.linalg.vector_norm(result.translation_m[0, 0] - translation[0]) < 0.1
    assert result.reprojection_rms_px[0, 0] < 1.0e-3


def test_covariance_downweights_a_corrupted_corner_cuda() -> None:
    device = _cuda()
    object_points = torch.as_tensor(NOMINAL_OBJECT_POINTS_M, dtype=torch.float32, device=device)[None]
    intrinsics = torch.tensor([[1150.0, 1150.0, 720.0, 540.0]], device=device)
    rotation = _so3_exp(torch.tensor([[0.05, -0.15, 0.02]], device=device))
    translation = torch.tensor([[0.1, 0.03, 3.5]], device=device)
    image_points, _ = project_points(rotation, translation, object_points, intrinsics)
    corrupted = image_points.clone()
    corrupted[:, 0] += corrupted.new_tensor([8.0, -6.0])
    identity = torch.eye(2, device=device)[None, None].expand(1, 4, -1, -1).clone()
    uniform = solve_weighted_planar_pnp(corrupted, object_points, intrinsics, covariance=identity)
    uncertain = identity.clone()
    uncertain[:, 0] *= 400.0
    weighted = solve_weighted_planar_pnp(corrupted, object_points, intrinsics, covariance=uncertain)
    error_uniform = torch.linalg.vector_norm(uniform.translation_m[0, 0] - translation[0])
    error_weighted = torch.linalg.vector_norm(weighted.translation_m[0, 0] - translation[0])
    assert weighted.valid[0, 0]
    assert error_weighted < error_uniform


def test_sparse_dense_fusion_has_truth_free_cuda_forward() -> None:
    value = _online(batch=1)
    model = SparseDensePoseNet(dense_count=32).to(_cuda()).eval()
    output = model(
        value["patch"], value["geometry"], value["raw_full"], value["raw_patch"],
        value["transform"], value["scale"], value["intrinsics"],
    )
    assert output.image_points.shape == (1, 36, 2)
    assert output.object_points.shape == (1, 36, 3)
    assert output.weights.is_cuda and output.pnp.translation_m.is_cuda
    assert torch.isfinite(output.weights).all()
