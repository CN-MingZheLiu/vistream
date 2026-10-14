"""Geometry helpers for aligning independently inferred temporal windows."""

from __future__ import annotations

import torch

from lingbot_map.utils.rotation import mat_to_quat, quat_to_mat


def w2c_pose_encoding_to_c2w(
    pose_enc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode W2C pose encodings into C2W rotations and camera centers."""
    work = pose_enc.to(torch.float32)
    rotation_cw = quat_to_mat(work[..., 3:7])
    translation_cw = work[..., :3]
    rotation_wc = rotation_cw.transpose(-1, -2)
    center_w = -torch.matmul(rotation_wc, translation_cw.unsqueeze(-1)).squeeze(-1)
    return rotation_wc, center_w


def apply_sim3_to_w2c_pose_encoding(
    pose_enc: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Apply ``p_out = scale * rotation @ p_in + translation`` to W2C poses."""
    rotation_wc, center_w = w2c_pose_encoding_to_c2w(pose_enc)
    rotation = rotation.to(device=pose_enc.device, dtype=torch.float32)
    translation = translation.to(device=pose_enc.device, dtype=torch.float32)
    scale = scale.to(device=pose_enc.device, dtype=torch.float32)

    rotation_wc_out = torch.matmul(rotation[:, None], rotation_wc)
    center_w_out = (
        scale[:, None, None]
        * torch.matmul(rotation[:, None], center_w.unsqueeze(-1)).squeeze(-1)
        + translation[:, None]
    )
    rotation_cw_out = rotation_wc_out.transpose(-1, -2)
    translation_cw_out = -torch.matmul(
        rotation_cw_out, center_w_out.unsqueeze(-1)
    ).squeeze(-1)

    result = pose_enc.clone()
    result[..., :3] = translation_cw_out.to(result.dtype)
    result[..., 3:7] = mat_to_quat(rotation_cw_out).to(result.dtype)
    return result


def _project_to_rotation(matrix: torch.Tensor) -> torch.Tensor:
    """Project a batch of 3x3 matrices onto SO(3)."""
    u, _, vh = torch.linalg.svd(matrix)
    rotation = torch.matmul(u, vh)
    negative = torch.linalg.det(rotation) < 0
    if negative.any():
        u = u.clone()
        u[negative, :, -1] *= -1
        rotation = torch.matmul(u, vh)
    return rotation


def _robust_rotation_mean(candidates: torch.Tensor) -> torch.Tensor:
    """Huber-weighted chordal mean for rotations shaped [B, N, 3, 3]."""
    initial = _project_to_rotation(candidates.sum(dim=1))
    relative = torch.matmul(initial[:, None].transpose(-1, -2), candidates)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    median = angle.median(dim=1).values
    mad = (angle - median[:, None]).abs().median(dim=1).values
    cutoff = torch.maximum(
        median + 3.0 * 1.4826 * mad,
        torch.full_like(median, torch.deg2rad(torch.tensor(1.0)).item()),
    )
    weights = torch.clamp(cutoff[:, None] / angle.clamp_min(1e-6), max=1.0)
    weighted = (candidates * weights[..., None, None]).sum(dim=1)
    return _project_to_rotation(weighted)


def _median_positive_ratio(
    anchor: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-batch log-median positive ratios and their validity mask."""
    batch_size = anchor.shape[0]
    anchor = anchor.to(torch.float32).reshape(batch_size, -1)
    target = target.to(torch.float32).reshape(batch_size, -1)
    valid = (
        torch.isfinite(anchor)
        & torch.isfinite(target)
        & (anchor > 1e-6)
        & (target > 1e-6)
    )
    scales = torch.ones(batch_size, device=anchor.device, dtype=torch.float32)
    has_scale = torch.zeros(batch_size, device=anchor.device, dtype=torch.bool)
    for batch_idx in range(batch_size):
        if valid[batch_idx].any():
            log_ratio = (
                torch.log(anchor[batch_idx, valid[batch_idx]])
                - torch.log(target[batch_idx, valid[batch_idx]])
            )
            scales[batch_idx] = torch.exp(log_ratio.median())
            has_scale[batch_idx] = True
    return scales, has_scale


def _center_distance_scale(
    anchor_centers: torch.Tensor,
    target_centers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate scale from robust ratios of all pairwise camera baselines."""
    batch_size, num_frames, _ = anchor_centers.shape
    scales = torch.ones(batch_size, device=anchor_centers.device, dtype=torch.float32)
    valid_scale = torch.zeros(batch_size, device=anchor_centers.device, dtype=torch.bool)
    if num_frames < 2:
        return scales, valid_scale

    for batch_idx in range(batch_size):
        anchor_dist = torch.pdist(anchor_centers[batch_idx])
        target_dist = torch.pdist(target_centers[batch_idx])
        finite = (
            torch.isfinite(anchor_dist)
            & torch.isfinite(target_dist)
            & (anchor_dist > 1e-4)
            & (target_dist > 1e-4)
        )
        if finite.any():
            log_ratio = torch.log(anchor_dist[finite]) - torch.log(target_dist[finite])
            scales[batch_idx] = torch.exp(log_ratio.median())
            valid_scale[batch_idx] = True
    return scales, valid_scale


def estimate_robust_window_sim3(
    anchor_pose_enc: torch.Tensor,
    target_pose_enc: torch.Tensor,
    anchor_depth: torch.Tensor | None = None,
    target_depth: torch.Tensor | None = None,
    scale_mode: str = "hybrid",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate a robust Sim(3) mapping target-window coordinates to anchor."""
    if anchor_pose_enc.shape != target_pose_enc.shape:
        raise ValueError(
            "Aligned pose tensors must have matching shapes, got "
            f"{tuple(anchor_pose_enc.shape)} and {tuple(target_pose_enc.shape)}"
        )
    if scale_mode not in {"median", "depth", "centers", "hybrid"}:
        raise ValueError(f"Unsupported window scale mode: {scale_mode}")

    anchor_rot, anchor_center = w2c_pose_encoding_to_c2w(anchor_pose_enc)
    target_rot, target_center = w2c_pose_encoding_to_c2w(target_pose_enc)
    rotation_candidates = torch.matmul(anchor_rot, target_rot.transpose(-1, -2))
    rotation = _robust_rotation_mean(rotation_candidates)

    center_scale, center_valid = _center_distance_scale(anchor_center, target_center)
    depth_scale = torch.ones_like(center_scale)
    depth_valid = torch.zeros_like(center_valid)
    if anchor_depth is not None and target_depth is not None:
        depth_scale, depth_valid = _median_positive_ratio(anchor_depth, target_depth)

    if scale_mode in {"median", "depth"}:
        scale = torch.where(depth_valid, depth_scale, center_scale)
    elif scale_mode == "centers":
        scale = torch.where(center_valid, center_scale, depth_scale)
    else:
        both = depth_valid & center_valid
        log_disagreement = (torch.log(depth_scale) - torch.log(center_scale)).abs()
        consistent = both & (log_disagreement < torch.log(torch.tensor(1.5)).item())
        blended = torch.exp(0.75 * torch.log(depth_scale) + 0.25 * torch.log(center_scale))
        scale = torch.where(consistent, blended, depth_scale)
        scale = torch.where(depth_valid, scale, center_scale)

    scale = torch.where(depth_valid | center_valid, scale, torch.ones_like(scale))
    scale = scale.clamp(min=0.1, max=10.0)

    rotated_target = torch.matmul(
        rotation[:, None], target_center.unsqueeze(-1)
    ).squeeze(-1)
    translation_candidates = anchor_center - scale[:, None, None] * rotated_target
    translation = translation_candidates.median(dim=1).values

    return (
        scale.to(anchor_pose_enc.dtype),
        rotation.to(anchor_pose_enc.dtype),
        translation.to(anchor_pose_enc.dtype),
    )
