"""Clean CUDA-first, truth-free weighted planar PnP MAP solver.

This module intentionally contains no OpenCV, NumPy conversion, CPU fallback,
or labelled-pose initialization.  It is the online geometry kernel; target pose
is accepted only by separate training losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GpuPnPResult:
    rotation_camera_from_pnp: torch.Tensor  # [B,M,3,3]
    translation_m: torch.Tensor  # [B,M,3]
    covariance_local: torch.Tensor  # [B,M,6,6]
    objective: torch.Tensor  # [B,M]
    mode_log_weight: torch.Tensor  # [B,M]
    valid: torch.Tensor  # [B,M]
    condition: torch.Tensor  # [B,M]
    reprojection_rms_px: torch.Tensor  # [B,M]


@dataclass(frozen=True)
class ObservablePnPInitialization:
    """Detached pose seeds produced by an online-observable GPU-PnP solve.

    This is deliberately not a target/reference-pose argument.  Construct it
    with :func:`observable_initialization_from_result` from a solve over the
    same frame's detector corners (or another explicitly online correspondence
    set), then pass it to a denser solve.  The receiving solve re-refines and
    re-scores every seed against its own correspondences; it never averages
    planar modes or treats the seed as a pose label.
    """

    rotation_camera_from_pnp: torch.Tensor  # [B,S,3,3]
    translation_m: torch.Tensor  # [B,S,3]
    valid: torch.Tensor  # [B,S]
    provenance: str = "online-observable-gpu-pnp"


def observable_initialization_from_result(result: GpuPnPResult) -> ObservablePnPInitialization:
    """Stop gradients through discrete mode selection and expose its top modes as seeds."""
    return ObservablePnPInitialization(
        rotation_camera_from_pnp=result.rotation_camera_from_pnp.detach(),
        translation_m=result.translation_m.detach(),
        valid=result.valid.detach(),
    )


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*vector.shape[:-1], 3, 3)


def _so3_exp(rotation_vector: torch.Tensor) -> torch.Tensor:
    theta2 = rotation_vector.square().sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1.0e-16)
    a = torch.where(theta2 < 1.0e-8, 1.0 - theta2 / 6.0, torch.sin(theta) / theta)
    b = torch.where(theta2 < 1.0e-8, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / theta2)
    cross = _skew(rotation_vector)
    identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    return identity + a[..., None] * cross + b[..., None] * (cross @ cross)


def project_points(rotation: torch.Tensor, translation_m: torch.Tensor,
                   object_points: torch.Tensor, intrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project arbitrary batched correspondences without hiding negative depth."""
    camera = object_points @ rotation.transpose(-1, -2) + translation_m[:, None]
    x, y, z = camera.unbind(dim=-1)
    safe_z = torch.where(z.abs() >= 1.0e-7, z, torch.where(z >= 0, z.new_tensor(1.0e-7), z.new_tensor(-1.0e-7)))
    fx, fy, cx, cy = intrinsics.unbind(dim=-1)
    pixels = torch.stack((fx[:, None] * x / safe_z + cx[:, None],
                          fy[:, None] * y / safe_z + cy[:, None]), dim=-1)
    return pixels, camera


def _validate_inputs(image_points: torch.Tensor, object_points: torch.Tensor, intrinsics: torch.Tensor,
                     weights: torch.Tensor | None, covariance: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    if image_points.ndim != 3 or image_points.shape[-1] != 2:
        raise ValueError("image_points must be [B,N,2]")
    if object_points.shape != (*image_points.shape[:2], 3) or intrinsics.shape != (image_points.shape[0], 4):
        raise ValueError("object point or intrinsics contract changed")
    if image_points.shape[1] < 4:
        raise ValueError("PnP requires at least four correspondences")
    if weights is None:
        weights = torch.ones(image_points.shape[:2], dtype=image_points.dtype, device=image_points.device)
    if weights.shape != image_points.shape[:2]:
        raise ValueError("weights must be [B,N]")
    if covariance is None:
        identity = torch.eye(2, dtype=image_points.dtype, device=image_points.device)
        covariance = identity.expand(*image_points.shape[:2], 2, 2)
    if covariance.shape != (*image_points.shape[:2], 2, 2):
        raise ValueError("covariance must be [B,N,2,2]")
    if not image_points.is_floating_point() or not object_points.is_floating_point():
        raise TypeError("PnP tensors must be floating point")
    if image_points.device != object_points.device or image_points.device != intrinsics.device:
        raise ValueError("all PnP tensors must be on one device")
    return weights, covariance


def _planar_dlt(image_points: torch.Tensor, object_points: torch.Tensor, intrinsics: torch.Tensor,
                scalar_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted planar DLT in normalized image and fixed nominal-plane coordinates."""
    fx, fy, cx, cy = intrinsics.unbind(dim=-1)
    normalized_image = torch.stack(((image_points[..., 0] - cx[:, None]) / fx[:, None],
                                    (image_points[..., 1] - cy[:, None]) / fy[:, None]), dim=-1)
    xy = object_points[..., :2]
    object_scale = image_points.new_tensor([0.0675, 0.0275])
    xy_normalized = xy / object_scale
    x, y = xy_normalized.unbind(dim=-1)
    u, v = normalized_image.unbind(dim=-1)
    one, zero = torch.ones_like(x), torch.zeros_like(x)
    row_u = torch.stack((x, y, one, zero, zero, zero, -u * x, -u * y), dim=-1)
    row_v = torch.stack((zero, zero, zero, x, y, one, -v * x, -v * y), dim=-1)
    matrix = torch.stack((row_u, row_v), dim=2).reshape(image_points.shape[0], -1, 8)
    right = torch.stack((u, v), dim=2).reshape(image_points.shape[0], -1, 1)
    equation_weights = scalar_weights[:, :, None].expand(-1, -1, 2).reshape(image_points.shape[0], -1, 1)
    square_root_weight = equation_weights.clamp_min(1.0e-12).sqrt()
    weighted_matrix = square_root_weight * matrix
    weighted_right = square_root_weight * right
    # Do not form normal equations here: their squared condition number caused
    # centimetre-scale depth tails even for perfect dense correspondences.
    solution = torch.linalg.lstsq(weighted_matrix, weighted_right).solution.squeeze(-1)
    homography_normalized = torch.cat((solution, torch.ones_like(solution[:, :1])), dim=1).reshape(-1, 3, 3)
    homography = homography_normalized.clone()
    homography[:, :, 0] = homography_normalized[:, :, 0] / object_scale[0]
    homography[:, :, 1] = homography_normalized[:, :, 1] / object_scale[1]
    h1, h2, h3 = homography[:, :, 0], homography[:, :, 1], homography[:, :, 2]
    sign = torch.where(h3[:, 2:3] >= 0, torch.ones_like(h3[:, 2:3]), -torch.ones_like(h3[:, 2:3]))
    h1, h2, h3 = sign * h1, sign * h2, sign * h3
    scale = 2.0 / (torch.linalg.vector_norm(h1, dim=1) + torch.linalg.vector_norm(h2, dim=1)).clamp_min(1.0e-8)
    r1 = torch.nn.functional.normalize(h1, dim=1)
    r2_raw = h2 - (r1 * h2).sum(dim=1, keepdim=True) * r1
    r2 = torch.nn.functional.normalize(r2_raw, dim=1)
    r3 = torch.linalg.cross(r1, r2, dim=1)
    rotation = torch.stack((r1, r2, r3), dim=-1)
    translation = scale[:, None] * h3
    condition = torch.linalg.cond(weighted_matrix)
    return rotation, translation, condition


def _whiten(residual: torch.Tensor, jacobian: torch.Tensor, covariance: torch.Tensor,
            weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    identity = torch.eye(2, dtype=covariance.dtype, device=covariance.device)
    scaled_covariance = covariance / weights.clamp_min(1.0e-6)[..., None, None]
    cholesky, info = torch.linalg.cholesky_ex(scaled_covariance + 1.0e-5 * identity)
    safe_cholesky = torch.where((info == 0)[..., None, None], cholesky, identity)
    white_residual = torch.linalg.solve_triangular(safe_cholesky, residual[..., None], upper=False).squeeze(-1)
    white_jacobian = torch.linalg.solve_triangular(safe_cholesky, jacobian, upper=False)
    return white_residual, white_jacobian, info


def _linearize(rotation: torch.Tensor, translation: torch.Tensor, image_points: torch.Tensor,
               object_points: torch.Tensor, intrinsics: torch.Tensor, weights: torch.Tensor,
               covariance: torch.Tensor, translation_scale_m: float) -> tuple[torch.Tensor, ...]:
    projected, camera = project_points(rotation, translation, object_points, intrinsics)
    x, y, z = camera.unbind(dim=-1)
    safe_z = torch.where(z.abs() >= 1.0e-7, z, torch.where(z >= 0, z.new_tensor(1.0e-7), z.new_tensor(-1.0e-7)))
    fx, fy, _, _ = intrinsics.unbind(dim=-1)
    zero = torch.zeros_like(z)
    du = torch.stack((fx[:, None] / safe_z, zero, -fx[:, None] * x / safe_z.square()), dim=-1)
    dv = torch.stack((zero, fy[:, None] / safe_z, -fy[:, None] * y / safe_z.square()), dim=-1)
    image_jacobian = torch.stack((du, dv), dim=-2)
    identity3 = torch.eye(3, dtype=camera.dtype, device=camera.device)
    camera_delta = torch.cat((-_skew(camera),
                              identity3.view(1, 1, 3, 3).expand(*camera.shape[:2], -1, -1) * translation_scale_m), dim=-1)
    jacobian = image_jacobian @ camera_delta
    residual = image_points - projected
    white_residual, white_jacobian, covariance_info = _whiten(residual, jacobian, covariance, weights)
    normal = torch.einsum("bnki,bnkj->bij", white_jacobian, white_jacobian)
    rhs = torch.einsum("bnki,bnk->bi", white_jacobian, white_residual)
    objective = white_residual.square().sum(dim=(1, 2))
    return normal, rhs, objective, projected, camera, covariance_info


def _refine(rotation: torch.Tensor, translation: torch.Tensor, image_points: torch.Tensor,
            object_points: torch.Tensor, intrinsics: torch.Tensor, weights: torch.Tensor,
            covariance: torch.Tensor, *, iterations: int, damping: float,
            translation_scale_m: float) -> tuple[torch.Tensor, ...]:
    identity6 = torch.eye(6, dtype=rotation.dtype, device=rotation.device).expand(rotation.shape[0], -1, -1)
    accepted = torch.zeros(rotation.shape[0], dtype=torch.bool, device=rotation.device)
    for _ in range(iterations):
        normal, rhs, objective, _, _, _ = _linearize(
            rotation, translation, image_points, object_points, intrinsics, weights, covariance, translation_scale_m
        )
        step = torch.linalg.solve(normal + damping * identity6, rhs[..., None]).squeeze(-1)
        candidate_rotation = _so3_exp(step[:, :3]) @ rotation
        candidate_translation = (_so3_exp(step[:, :3]) @ translation[..., None]).squeeze(-1) + step[:, 3:] * translation_scale_m
        _, _, candidate_objective, _, candidate_camera, _ = _linearize(
            candidate_rotation, candidate_translation, image_points, object_points, intrinsics, weights, covariance,
            translation_scale_m,
        )
        take = torch.isfinite(candidate_objective) & (candidate_objective <= objective) & (candidate_camera[..., 2] > 1.0e-5).all(dim=1)
        rotation = torch.where(take[:, None, None], candidate_rotation, rotation)
        translation = torch.where(take[:, None], candidate_translation, translation)
        accepted |= take
    normal, _, objective, projected, camera, covariance_info = _linearize(
        rotation, translation, image_points, object_points, intrinsics, weights, covariance, translation_scale_m
    )
    condition = torch.linalg.cond(normal + damping * identity6)
    covariance_local = torch.linalg.solve(normal + damping * identity6, identity6)
    reprojection = torch.sqrt((projected - image_points).square().mean(dim=(1, 2)))
    valid = (
        accepted
        & torch.isfinite(rotation).all(dim=(1, 2))
        & torch.isfinite(translation).all(dim=1)
        & torch.isfinite(condition)
        & (condition < 1.0e10)
        & (camera[..., 2] > 1.0e-5).all(dim=1)
        & (covariance_info == 0).all(dim=1)
    )
    return rotation, translation, covariance_local, objective, valid, condition, reprojection


def _validated_observable_seeds(initialization: ObservablePnPInitialization | None, *, batch: int,
                                device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...] | None:
    if initialization is None:
        return None
    if not isinstance(initialization, ObservablePnPInitialization):
        raise TypeError("initialization must be ObservablePnPInitialization or None")
    if initialization.provenance != "online-observable-gpu-pnp":
        raise ValueError("PnP initialization provenance is not online-observable")
    rotation = initialization.rotation_camera_from_pnp
    translation = initialization.translation_m
    valid = initialization.valid
    if rotation.ndim != 4 or rotation.shape[0] != batch or rotation.shape[2:] != (3, 3):
        raise ValueError("observable initialization rotation must be [B,S,3,3]")
    if translation.shape != (*rotation.shape[:2], 3) or valid.shape != rotation.shape[:2]:
        raise ValueError("observable initialization translation/valid contract changed")
    if rotation.device != device or translation.device != device or valid.device != device:
        raise ValueError("observable initialization must remain on the PnP device")
    if not rotation.is_floating_point() or not translation.is_floating_point():
        raise TypeError("observable initialization pose tensors must be floating point")
    if rotation.requires_grad or translation.requires_grad:
        raise ValueError("observable initialization must be detached from its source solve")
    finite = torch.isfinite(rotation).all(dim=(2, 3)) & torch.isfinite(translation).all(dim=2)
    valid = valid.to(dtype=torch.bool) & finite
    return rotation.to(dtype=dtype), translation.to(dtype=dtype), valid


def _select_diverse_modes(rotation: torch.Tensor, translation: torch.Tensor,
                          ranked_objective: torch.Tensor) -> torch.Tensor:
    """Select MAP first and a genuinely separate planar basin second when available."""
    batch, proposals = ranked_objective.shape
    best = ranked_objective.argmin(dim=1)
    batch_index = torch.arange(batch, device=ranked_objective.device)
    best_rotation = rotation[batch_index, best]
    best_translation = translation[batch_index, best]
    relative = rotation @ best_rotation[:, None].transpose(-1, -2)
    trace = torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1)
    rotation_distance = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
    translation_distance = torch.linalg.vector_norm(translation - best_translation[:, None], dim=-1)
    separate = (rotation_distance > 2.0e-2) | (translation_distance > 5.0e-3)
    separate[batch_index, best] = False
    separate_objective = torch.where(separate, ranked_objective, torch.full_like(ranked_objective, torch.inf))
    second_separate = separate_objective.argmin(dim=1)
    has_separate = torch.isfinite(separate_objective[batch_index, second_separate])
    fallback_objective = ranked_objective.clone()
    fallback_objective[batch_index, best] = torch.inf
    second_fallback = fallback_objective.argmin(dim=1)
    second = torch.where(has_separate, second_separate, second_fallback)
    return torch.stack((best, second), dim=1)


def solve_weighted_planar_pnp(image_points: torch.Tensor, object_points: torch.Tensor,
                              intrinsics: torch.Tensor, *, weights: torch.Tensor | None = None,
                              covariance: torch.Tensor | None = None, modes: int = 2,
                              iterations: int = 8,
                              initialization: ObservablePnPInitialization | None = None) -> GpuPnPResult:
    """Solve planar PnP from observable correspondences and return top MAP modes.

    ``initialization`` may carry detached modes from another same-frame,
    online-observable GPU solve, most usefully the raw four detector corners.
    They are seeds only: all candidates are refined and ranked solely by this
    call's ``image_points``/``object_points``/uncertainty.  No target or
    reference pose is accepted by this online kernel.
    """
    weights, covariance = _validate_inputs(image_points, object_points, intrinsics, weights, covariance)
    if modes != 2:
        raise ValueError("V19 pre-registers exactly two retained planar modes")
    original_dtype = image_points.dtype
    # Consumer GPUs have slow FP64 throughput, but these matrices are tiny and
    # planar depth is condition-sensitive.  Keeping geometry on the same CUDA
    # device in FP64 removes the perfect-correspondence tail without a CPU hop.
    geometry_dtype = torch.float64 if image_points.device.type == "cuda" else original_dtype
    image_points = image_points.to(dtype=geometry_dtype)
    object_points = object_points.to(dtype=geometry_dtype)
    intrinsics = intrinsics.to(dtype=geometry_dtype)
    weights = weights.to(dtype=geometry_dtype)
    covariance = covariance.to(dtype=geometry_dtype)
    finite_input = (
        torch.isfinite(image_points).all(dim=(1, 2))
        & torch.isfinite(object_points).all(dim=(1, 2))
        & torch.isfinite(intrinsics).all(dim=1)
        & torch.isfinite(weights).all(dim=1)
        & torch.isfinite(covariance).all(dim=(1, 2, 3))
        & (intrinsics[:, :2] > 0).all(dim=1)
        & ((weights > 1.0e-6).sum(dim=1) >= 4)
    )
    branch_weights = weights.clamp_min(0.0)
    branch_weights = branch_weights / branch_weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6) * image_points.shape[1]
    scalar_covariance_weight = 2.0 / torch.diagonal(covariance, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(1.0e-6)
    dlt_weight = branch_weights * scalar_covariance_weight
    base_rotation, base_translation, dlt_condition = _planar_dlt(image_points, object_points, intrinsics, dlt_weight)

    # Fixed observable-only multi-starts.  They create deterministic basins for
    # weak-perspective planar ambiguity without consulting a labelled pose.
    perturbations = image_points.new_tensor([
        [0.0, 0.0, 0.0], [0.18, 0.0, 0.0], [-0.18, 0.0, 0.0],
        [0.0, 0.18, 0.0], [0.0, -0.18, 0.0], [0.32, 0.18, 0.0],
        [-0.32, 0.18, 0.0], [0.32, -0.18, 0.0], [-0.32, -0.18, 0.0],
    ])
    batch, proposals = image_points.shape[0], perturbations.shape[0]
    seed_rotation = _so3_exp(perturbations)[None] @ base_rotation[:, None]
    seed_translation = base_translation[:, None].expand(-1, proposals, -1)
    dlt_seed_valid = torch.isfinite(dlt_condition) & (dlt_condition < 1.0e12)
    seed_enabled = dlt_seed_valid[:, None].expand(-1, proposals)
    observable_seeds = _validated_observable_seeds(
        initialization, batch=batch, device=image_points.device, dtype=geometry_dtype,
    )
    if observable_seeds is not None:
        observable_rotation, observable_translation, observable_valid = observable_seeds
        seed_rotation = torch.cat((seed_rotation, observable_rotation), dim=1)
        seed_translation = torch.cat((seed_translation, observable_translation), dim=1)
        seed_enabled = torch.cat((seed_enabled, observable_valid), dim=1)
        proposals = seed_rotation.shape[1]
    repeat = lambda value: value[:, None].expand(-1, proposals, *value.shape[1:]).reshape(batch * proposals, *value.shape[1:])
    refined = _refine(
        seed_rotation.reshape(batch * proposals, 3, 3), seed_translation.reshape(batch * proposals, 3),
        repeat(image_points), repeat(object_points), repeat(intrinsics), repeat(branch_weights), repeat(covariance),
        iterations=iterations, damping=1.0e-3, translation_scale_m=0.10,
    )
    rotation, translation, covariance_local, objective, valid, condition, reprojection = (
        value.reshape(batch, proposals, *value.shape[1:]) for value in refined
    )
    valid = valid & seed_enabled & finite_input[:, None]
    ranked_objective = torch.where(valid, objective, torch.full_like(objective, torch.inf))
    # Preserve the V19 no-initialization ranking exactly.  Observable seeds add
    # repeated refinements around the raw modes, so only that path deduplicates
    # near-identical proposals before retaining the two planar basins.
    selected = (
        ranked_objective.topk(k=modes, dim=1, largest=False).indices
        if initialization is None
        else _select_diverse_modes(rotation, translation, ranked_objective)
    )
    gather = lambda value: value.gather(1, selected.reshape(batch, modes, *([1] * (value.ndim - 2))).expand(batch, modes, *value.shape[2:]))
    selected_rotation = gather(rotation)
    selected_translation = gather(translation)
    selected_covariance = gather(covariance_local)
    selected_objective = gather(objective)
    selected_valid = gather(valid)
    selected_condition = gather(condition)
    selected_reprojection = gather(reprojection)
    log_mass = torch.where(selected_valid, -0.5 * selected_objective, torch.full_like(selected_objective, -torch.inf))
    mode_log_weight = log_mass - torch.logsumexp(log_mass, dim=1, keepdim=True)
    mode_log_weight = torch.where(selected_valid, mode_log_weight, torch.full_like(mode_log_weight, -torch.inf))
    return GpuPnPResult(
        rotation_camera_from_pnp=selected_rotation.to(dtype=original_dtype),
        translation_m=selected_translation.to(dtype=original_dtype),
        covariance_local=selected_covariance.to(dtype=original_dtype),
        objective=selected_objective.to(dtype=original_dtype),
        mode_log_weight=mode_log_weight.to(dtype=original_dtype),
        valid=selected_valid,
        condition=selected_condition.to(dtype=original_dtype),
        reprojection_rms_px=selected_reprojection.to(dtype=original_dtype),
    )
