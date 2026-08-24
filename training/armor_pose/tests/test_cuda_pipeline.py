from __future__ import annotations

from dataclasses import fields

import torch

from training.armor_pose.dense_correspondence_head import (
    DenseCorrespondenceNet, SpatialBinCornerTail, stratified_correspondences,
)
from training.armor_pose.gpu_pnp import (
    ObservablePnPInitialization,
    _so3_exp,
    observable_initialization_from_result,
    project_points,
    solve_weighted_planar_pnp,
)
from training.armor_pose.fusion import SparseDensePoseNet
from training.armor_pose.labels import (
    CANONICAL_CORNER_UV, canonical_uv_to_object_points, dense_surface_labels,
    map_points_homography, pixel_to_uv_homography,
)
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
    prediction = model(value["patch"], value["geometry"], value["raw_patch"])
    correspondences = stratified_correspondences(prediction, value["transform"], count=64)
    assert correspondences.image_points.shape == (2, 64, 2)
    assert correspondences.object_points.shape == (2, 64, 3)
    assert correspondences.weights.is_cuda
    assert torch.isfinite(correspondences.object_points).all()


def test_dense_projective_head_zero_initialization_preserves_raw_patch_cuda() -> None:
    value = _online()
    model = DenseCorrespondenceNet().to(_cuda()).eval()
    prediction = model(value["patch"], value["geometry"], value["raw_patch"])
    assert prediction.corner_heatmap_logits is not None
    assert prediction.local_corners_patch is not None
    assert prediction.tail_corners_patch is not None
    assert torch.allclose(prediction.local_corners_patch, value["raw_patch"], atol=3.0e-5)
    assert torch.allclose(prediction.tail_corners_patch, value["raw_patch"], atol=1.0e-7)
    assert torch.allclose(prediction.predicted_corners_patch, value["raw_patch"], atol=3.0e-5)
    assert model.config["nonprojective_uv_residual_enabled"] is False
    assert model.config["nonprojective_uv_residual_scale"] == 0.0


def test_spatial_bin_tail_responds_to_translation_with_equal_global_average_cuda() -> None:
    head = SpatialBinCornerTail(1, hidden=1).to(_cuda()).eval()
    with torch.no_grad():
        first, last = head.mlp[1], head.mlp[3]
        first.weight.zero_()
        first.bias.zero_()
        last.weight.zero_()
        last.bias.zero_()
        first.weight[0, 0] = 1.0
        last.weight[0, 0] = 1.0
    left = torch.zeros(1, 1, 64, 128, device=_cuda())
    right = torch.zeros_like(left)
    left[:, :, :16, :16] = 1.0
    right[:, :, -16:, -16:] = 1.0
    assert torch.allclose(left.mean(dim=(-2, -1)), right.mean(dim=(-2, -1)))
    left_output = head(left)
    right_output = head(right)
    assert left_output[0, 0, 0] > 0.0
    assert right_output[0, 0, 0] == 0.0


def test_dense_uv_is_strictly_induced_by_its_four_predicted_corners_cuda() -> None:
    value = _online(batch=1)
    model = DenseCorrespondenceNet().to(_cuda()).eval()
    prediction = model(value["patch"], value["geometry"], value["raw_patch"])
    transform = pixel_to_uv_homography(prediction.predicted_corners_patch)
    mapped = map_points_homography(prediction.predicted_corners_patch, transform)
    expected = CANONICAL_CORNER_UV.to(device=_cuda())[None]
    assert torch.allclose(mapped, expected, atol=2.0e-4, rtol=0.0)


def test_dense_projective_head_cuda_gradients_are_finite() -> None:
    value = _online(batch=1)
    model = DenseCorrespondenceNet().to(_cuda()).train()
    prediction = model(value["patch"], value["geometry"], value["raw_patch"])
    assert prediction.in_context_logits is not None
    objective = (
        prediction.support_logits.mean() + prediction.canonical_uv.square().mean()
        + prediction.log_variance.square().mean() + prediction.edge_distance.mean()
        + prediction.predicted_corners_patch.square().mean()
        + prediction.in_context_logits.square().mean()
    )
    objective.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


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


def test_observable_raw_initialization_keeps_same_homography_dense_solve_in_its_pose_basin_cuda() -> None:
    device = _cuda()
    rotation_vectors = torch.tensor([
        [0.00, 0.00, 0.00], [0.04, -0.03, 0.01], [-0.08, 0.06, -0.02],
        [0.18, -0.14, 0.03], [-0.24, -0.18, 0.04], [0.31, 0.22, -0.06],
        [-0.42, 0.27, 0.08], [0.48, -0.36, -0.09], [0.56, 0.42, 0.10],
        [-0.61, -0.38, -0.12], [0.67, -0.48, 0.14], [-0.70, 0.53, -0.15],
    ], device=device)
    batch = rotation_vectors.shape[0]
    rotation = _so3_exp(rotation_vectors)
    translation = torch.stack((
        torch.linspace(-0.45, 0.45, batch, device=device),
        torch.linspace(0.28, -0.28, batch, device=device),
        torch.linspace(2.0, 8.0, batch, device=device),
    ), dim=1)
    intrinsics = torch.tensor([[1303.6753, 1303.6753, 720.0, 540.0]], device=device).expand(batch, -1).clone()
    raw_object = torch.as_tensor(NOMINAL_OBJECT_POINTS_M, dtype=torch.float32, device=device)[None].expand(batch, -1, -1)
    axis = torch.linspace(-0.92, 0.92, 8, device=device)
    uv_y, uv_x = torch.meshgrid(axis, axis, indexing="ij")
    dense_uv = torch.stack((uv_x, uv_y), dim=-1).reshape(1, 64, 2).expand(batch, -1, -1)
    dense_object = canonical_uv_to_object_points(dense_uv)
    raw_image, _ = project_points(rotation, translation, raw_object, intrinsics)
    raw_image = raw_image.detach().requires_grad_()
    dense_image, _ = project_points(rotation, translation, dense_object, intrinsics)

    raw_result = solve_weighted_planar_pnp(raw_image, raw_object, intrinsics, iterations=12)
    assert raw_result.rotation_camera_from_pnp.requires_grad
    assert raw_result.translation_m.requires_grad
    initialization = observable_initialization_from_result(raw_result)
    assert isinstance(initialization, ObservablePnPInitialization)
    assert initialization.rotation_camera_from_pnp.is_cuda
    assert initialization.translation_m.is_cuda and initialization.valid.is_cuda
    assert not initialization.rotation_camera_from_pnp.requires_grad
    assert not initialization.translation_m.requires_grad
    field_names = {item.name for item in fields(initialization)}
    assert not field_names.intersection({"target", "reference", "truth", "label", "target_pose", "reference_pose"})

    dense_result = solve_weighted_planar_pnp(
        dense_image, dense_object, intrinsics, iterations=12, initialization=initialization,
    )
    assert raw_result.valid[:, 0].all() and dense_result.valid[:, 0].all()
    raw_dense_distance = torch.cdist(dense_result.translation_m, raw_result.translation_m).amin(dim=(1, 2))
    truth_distance = torch.linalg.vector_norm(dense_result.translation_m - translation[:, None], dim=-1).amin(dim=1)
    # Both sets sample one exact observable homography; a selected dense mode
    # must not jump into a metre-away planar basin.
    assert torch.quantile(raw_dense_distance, 0.9) < 2.0e-4
    assert raw_dense_distance.max() < 1.0e-3
    assert truth_distance.max() * 1.0e3 < 0.2


def test_invalid_observable_initialization_falls_back_to_own_dlt_and_degenerate_dlt_fails_closed_cuda() -> None:
    device = _cuda()
    object_points = torch.as_tensor(NOMINAL_OBJECT_POINTS_M, dtype=torch.float32, device=device)[None]
    intrinsics = torch.tensor([[1200.0, 1180.0, 720.0, 540.0]], device=device)
    rotation = _so3_exp(torch.tensor([[0.12, -0.28, 0.04]], device=device))
    translation = torch.tensor([[0.18, -0.06, 4.5]], device=device)
    image_points, _ = project_points(rotation, translation, object_points, intrinsics)
    raw_result = solve_weighted_planar_pnp(image_points, object_points, intrinsics, iterations=12)
    raw_result.valid.zero_()
    invalid_initialization = observable_initialization_from_result(raw_result)

    default_result = solve_weighted_planar_pnp(image_points, object_points, intrinsics, iterations=12)
    explicit_none_result = solve_weighted_planar_pnp(
        image_points, object_points, intrinsics, iterations=12, initialization=None,
    )
    fallback_result = solve_weighted_planar_pnp(
        image_points, object_points, intrinsics, iterations=12, initialization=invalid_initialization,
    )
    assert default_result.valid[0, 0] and fallback_result.valid[0, 0]
    assert torch.equal(default_result.valid, explicit_none_result.valid)
    assert torch.allclose(default_result.translation_m, explicit_none_result.translation_m, atol=0.0, rtol=0.0)
    assert torch.allclose(default_result.translation_m[:, 0], fallback_result.translation_m[:, 0], atol=2.0e-5)

    degenerate_object = object_points[:, :1].expand(-1, 4, -1).clone()
    degenerate_image = image_points[:, :1].expand(-1, 4, -1).clone()
    failed = solve_weighted_planar_pnp(
        degenerate_image, degenerate_object, intrinsics, initialization=invalid_initialization,
    )
    assert not failed.valid.any()


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
