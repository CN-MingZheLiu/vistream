"""Navigation-priority training losses for VIStream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from .geometry.lie import invert_se3, se3_log


@dataclass(frozen=True)
class LossWeights:
    relative_translation: float = 1.0
    relative_rotation: float = 0.7
    anchored_absolute: float = 0.5
    velocity: float = 0.2
    long_drift: float = 0.6
    imu_consistency: float = 0.05
    depth: float = 0.25
    depth_gradient: float = 0.10
    world_points: float = 0.05
    residual_regularizer: float = 1e-4
    oracle_gate: float = 0.2
    safe_hinge: float = 0.5


def _relative(poses: torch.Tensor, stride: int = 1) -> torch.Tensor:
    return invert_se3(poses[:, :-stride]) @ poses[:, stride:]


def _smooth_l1_norm(values: torch.Tensor, beta: float = 0.05) -> torch.Tensor:
    return F.smooth_l1_loss(values, torch.zeros_like(values), beta=beta)


def _smooth_l1_per_sample(values: torch.Tensor, beta: float = 0.05) -> torch.Tensor:
    loss = F.smooth_l1_loss(values, torch.zeros_like(values), beta=beta, reduction="none")
    return loss.reshape(loss.shape[0], -1).mean(dim=-1)


def _depth_gradients(depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return depth[..., 1:, :] - depth[..., :-1, :], depth[..., :, 1:] - depth[..., :, :-1]


def _unproject_depth(
    depth: torch.Tensor, intrinsics: torch.Tensor, camera_to_world: torch.Tensor
) -> torch.Tensor:
    batch, frames, _, height, width = depth.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    pixels = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    rays = torch.einsum("bsij,hwj->bshwi", torch.linalg.inv(intrinsics), pixels)
    points = rays * depth.permute(0, 1, 3, 4, 2)
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    return torch.einsum("bsij,bshwj->bshwi", camera_to_world, homogeneous)[..., :3]


def vistream_losses(
    predictions: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    weights: LossWeights = LossWeights(),
    include_per_sample: bool = False,
) -> dict[str, torch.Tensor]:
    predicted_pose = predictions["camera_to_world"].float()
    target_pose = batch["poses"].to(predicted_pose).float()
    if predicted_pose.shape != target_pose.shape:
        raise ValueError("predicted and ground-truth poses must have equal shape")

    predicted_relative = _relative(predicted_pose)
    target_relative = _relative(target_pose)
    relative_error = se3_log(invert_se3(target_relative) @ predicted_relative)
    relative_translation = _smooth_l1_norm(relative_error[..., :3])
    relative_rotation = _smooth_l1_norm(relative_error[..., 3:], beta=0.02)
    relative_translation_per_sample = _smooth_l1_per_sample(relative_error[..., :3])
    relative_rotation_per_sample = _smooth_l1_per_sample(relative_error[..., 3:], beta=0.02)

    anchor = target_pose[:, :1] @ invert_se3(predicted_pose[:, :1])
    anchored = anchor @ predicted_pose
    absolute_error = se3_log(invert_se3(target_pose) @ anchored)
    anchored_absolute = _smooth_l1_norm(absolute_error[..., :3])
    anchored_absolute_per_sample = _smooth_l1_per_sample(absolute_error[..., :3])

    camera_dt = batch["camera_dt"].to(predicted_pose).clamp_min(1e-4)
    predicted_velocity = (
        anchored[:, 1:, :3, 3] - anchored[:, :-1, :3, 3]
    ) / camera_dt[:, 1:].unsqueeze(-1)
    target_velocity = (
        target_pose[:, 1:, :3, 3] - target_pose[:, :-1, :3, 3]
    ) / camera_dt[:, 1:].unsqueeze(-1)
    velocity = F.smooth_l1_loss(predicted_velocity, target_velocity, beta=0.1)
    velocity_per_sample = _smooth_l1_per_sample(predicted_velocity - target_velocity, beta=0.1)

    drift_terms: list[torch.Tensor] = []
    for stride in (4, 8, 16, 32, 64, 128):
        if predicted_pose.shape[1] > stride:
            drift_error = se3_log(
                invert_se3(_relative(target_pose, stride)) @ _relative(predicted_pose, stride)
            )
            drift_terms.append(
                _smooth_l1_norm(drift_error[..., :3])
                + 0.5 * _smooth_l1_norm(drift_error[..., 3:], beta=0.02)
            )
    long_drift = torch.stack(drift_terms).mean() if drift_terms else relative_translation.new_zeros(())
    drift_per_sample_terms: list[torch.Tensor] = []
    for stride in (4, 8, 16, 32, 64, 128):
        if predicted_pose.shape[1] > stride:
            drift_error = se3_log(
                invert_se3(_relative(target_pose, stride)) @ _relative(predicted_pose, stride)
            )
            drift_per_sample_terms.append(
                _smooth_l1_per_sample(drift_error[..., :3])
                + 0.5 * _smooth_l1_per_sample(drift_error[..., 3:], beta=0.02)
            )
    long_drift_per_sample = (
        torch.stack(drift_per_sample_terms).mean(dim=0)
        if drift_per_sample_terms
        else relative_translation_per_sample.new_zeros(relative_translation_per_sample.shape)
    )

    imu_delta = predictions["imu_delta_transform"].float()
    imu_error = se3_log(invert_se3(imu_delta[:, 1:]) @ predicted_relative)
    reliability = predictions["camera_gate"][:, 1:].detach().float()
    # A six-axis IMU observes angular increments and changes in velocity, but
    # not metric translation when initial velocity is unknown. Penalizing the
    # translational part here previously pulled vehicle motion toward the
    # zero-velocity preintegration result. Use only the observable rotation.
    imu_rotation_error = imu_error[..., 3:]
    imu_consistency = (
        F.smooth_l1_loss(
            imu_rotation_error,
            torch.zeros_like(imu_rotation_error),
            beta=0.02,
            reduction="none",
        )
        * reliability
    ).mean()
    imu_consistency_per_sample = (
        F.smooth_l1_loss(
            imu_rotation_error,
            torch.zeros_like(imu_rotation_error),
            beta=0.02,
            reduction="none",
        )
        * reliability
    ).reshape(imu_rotation_error.shape[0], -1).mean(dim=-1)

    navigation = (
        weights.relative_translation * relative_translation
        + weights.relative_rotation * relative_rotation
        + weights.anchored_absolute * anchored_absolute
        + weights.velocity * velocity
        + weights.long_drift * long_drift
        + weights.imu_consistency * imu_consistency
    )
    navigation_per_sample = (
        weights.relative_translation * relative_translation_per_sample
        + weights.relative_rotation * relative_rotation_per_sample
        + weights.anchored_absolute * anchored_absolute_per_sample
        + weights.velocity * velocity_per_sample
        + weights.long_drift * long_drift_per_sample
        + weights.imu_consistency * imu_consistency_per_sample
    )

    zero = navigation.new_zeros(())
    depth_loss = zero
    depth_gradient = zero
    world_points = zero
    if "depth" in predictions and batch.get("depth") is not None and batch["depth"].numel() > 0:
        predicted_depth = predictions["depth"].permute(0, 1, 4, 2, 3).float()
        target_depth = batch["depth"].to(predicted_depth).float()
        valid = batch["depth_mask"].to(predicted_depth.device)
        valid = valid & torch.isfinite(predicted_depth) & (predicted_depth > 0)
        if valid.any():
            log_error = torch.log(predicted_depth.clamp_min(1e-3)) - torch.log(
                target_depth.clamp_min(1e-3)
            )
            depth_loss = F.smooth_l1_loss(
                log_error[valid], torch.zeros_like(log_error[valid]), beta=0.05
            )
            pred_dy, pred_dx = _depth_gradients(torch.log(predicted_depth.clamp_min(1e-3)))
            gt_dy, gt_dx = _depth_gradients(torch.log(target_depth.clamp_min(1e-3)))
            mask_dy = valid[..., 1:, :] & valid[..., :-1, :]
            mask_dx = valid[..., :, 1:] & valid[..., :, :-1]
            gradient_values = []
            if mask_dy.any():
                gradient_values.append((pred_dy - gt_dy).abs()[mask_dy].mean())
            if mask_dx.any():
                gradient_values.append((pred_dx - gt_dx).abs()[mask_dx].mean())
            if gradient_values:
                depth_gradient = torch.stack(gradient_values).mean()

        if "world_points" in predictions and "gt_intrinsics" in batch:
            gt_points = _unproject_depth(
                target_depth,
                batch["gt_intrinsics"].to(target_depth),
                target_pose,
            )
            predicted_points = predictions["world_points"].float()
            point_mask = batch["depth_mask"].squeeze(2).to(predicted_points.device)
            point_mask = point_mask & torch.isfinite(predicted_points).all(dim=-1)
            if point_mask.any():
                world_points = F.smooth_l1_loss(
                    predicted_points[point_mask], gt_points[point_mask], beta=0.1
                )

    residual_regularizer = (
        predictions["camera_residual_rms"].square()
        + predictions["geometry_residual_rms"].square()
    ).mean()
    geometry = (
        weights.depth * depth_loss
        + weights.depth_gradient * depth_gradient
        + weights.world_points * world_points
        + weights.residual_regularizer * residual_regularizer
    )
    total = navigation + geometry
    result = {
        "total": total,
        "navigation": navigation,
        "geometry": geometry,
        "relative_translation": relative_translation,
        "relative_rotation": relative_rotation,
        "anchored_absolute": anchored_absolute,
        "velocity": velocity,
        "long_drift": long_drift,
        "imu_consistency": imu_consistency,
        "depth": depth_loss,
        "depth_gradient": depth_gradient,
        "world_points": world_points,
        "residual_regularizer": residual_regularizer,
    }
    if include_per_sample:
        result["navigation_per_sample"] = navigation_per_sample

    # The native LingBot prediction is computed under no_grad and attached by
    # VIStream.forward(return_native=True).  It supplies a per-sample oracle
    # target for the safety gate and a hinge that directly penalizes harmful
    # fusion.  The teacher is detached before it participates in the target.
    native_predictions = predictions.get("native_predictions")
    if native_predictions is not None:
        native_losses = vistream_losses(
            native_predictions, batch, weights, include_per_sample=True
        )
        native_navigation = native_losses["navigation_per_sample"].detach()
        candidate_navigation = navigation_per_sample
        target = torch.sigmoid(-(
            candidate_navigation.detach() - native_navigation
        ) / 0.1)
        safe_gate = predictions["safe_gate"].float().mean(dim=1)
        # BCE on probabilities is intentionally rejected by CUDA autocast.
        # Converting the already-bounded probability back to logits gives the
        # same objective while using the BF16-safe fused implementation.
        safe_logits = torch.logit(
            safe_gate[..., 0].clamp(1e-4, 1.0 - 1e-4)
        )
        gate_loss = F.binary_cross_entropy_with_logits(
            safe_logits, target.float()
        )
        safe_hinge = F.relu(candidate_navigation - native_navigation).mean()
        total = total + weights.oracle_gate * gate_loss + weights.safe_hinge * safe_hinge
        result["total"] = total
        result["oracle_gate"] = gate_loss
        result["safe_hinge"] = safe_hinge
        result["native_navigation"] = native_navigation.mean()
        result["navigation_degradation"] = (
            candidate_navigation - native_navigation
        ).mean()
    return result
