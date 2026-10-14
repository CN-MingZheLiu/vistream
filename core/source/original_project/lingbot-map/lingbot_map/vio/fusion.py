"""Adapters between frozen LingBot predictions, the VIO backend, and mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .backend import FixedLagVIOBackend, VIOConfig, VIOResult
from .math import inverse_transform
from .preintegration import ImuSample


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        try:
            value = value.numpy()
        except TypeError:
            value = value.float().numpy()
    return np.asarray(value)


def load_camera_imu_transform(path: str | Path, camera: str = "cam0") -> np.ndarray:
    """Read ``T_cam_imu`` from a Kalibr/TUM-VI camchain YAML file."""
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        calibration = yaml.safe_load(handle)
    try:
        transform = np.asarray(calibration[camera]["T_cam_imu"], dtype=np.float64)
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path} has no {camera}.T_cam_imu") from error
    if transform.shape != (4, 4):
        raise ValueError(f"T_cam_imu must be 4x4, got {transform.shape}")
    return transform


def visual_confidence_from_predictions(predictions: Mapping) -> np.ndarray | None:
    """Reduce LingBot's dense confidence to one robust weight per frame."""
    confidence = predictions.get("depth_conf")
    if confidence is None:
        confidence = predictions.get("world_points_conf")
    if confidence is None:
        return None
    confidence = _numpy(confidence).astype(np.float64)
    if confidence.ndim == 4 and confidence.shape[0] == 1:
        confidence = confidence[0]
    if confidence.ndim < 2:
        return confidence.reshape(-1)
    return np.nanmedian(confidence.reshape(confidence.shape[0], -1), axis=1)


def fuse_lingbot_with_imu(predictions: Mapping, frame_timestamps: Sequence[float],
                          imu_samples: Sequence[ImuSample] | np.ndarray,
                          *, config: VIOConfig | None = None,
                          transform_camera_imu: np.ndarray | None = None,
                          visual_pose_convention: str = "c2w") -> VIOResult:
    """Run VIO after LingBot inference without changing any network weight."""
    if "extrinsic" not in predictions:
        raise KeyError("predictions must contain decoded camera poses in 'extrinsic'")
    poses = _numpy(predictions["extrinsic"]).astype(np.float64)
    if poses.ndim == 4 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.shape[-2:] == (3, 4):
        homogeneous = np.repeat(np.eye(4)[None], len(poses), axis=0)
        homogeneous[:, :3, :4] = poses
        poses = homogeneous
    if visual_pose_convention == "w2c":
        poses = np.stack([inverse_transform(pose) for pose in poses])
    elif visual_pose_convention != "c2w":
        raise ValueError("visual_pose_convention must be 'c2w' or 'w2c'")
    backend = FixedLagVIOBackend(config, transform_camera_imu)
    return backend.optimize(poses, frame_timestamps, imu_samples, visual_confidence_from_predictions(predictions))


def backproject_depth(depth, intrinsics, camera_poses_c2w) -> np.ndarray:
    """Backproject unchanged per-frame depth into one world-coordinate tensor."""
    depth = _numpy(depth).astype(np.float64)
    intrinsics = _numpy(intrinsics).astype(np.float64)
    camera_poses = _numpy(camera_poses_c2w).astype(np.float64)
    if depth.ndim == 5 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.shape[-1:] == (1,):
        depth = depth[..., 0]
    if intrinsics.ndim == 4 and intrinsics.shape[0] == 1:
        intrinsics = intrinsics[0]
    if camera_poses.shape[-2:] == (3, 4):
        full = np.repeat(np.eye(4)[None], len(camera_poses), axis=0)
        full[:, :3, :4] = camera_poses
        camera_poses = full
    if not (len(depth) == len(intrinsics) == len(camera_poses)):
        raise ValueError("depth, intrinsics, and camera pose frame counts differ")
    height, width = depth.shape[1:3]
    v, u = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pixels = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float64)
    result = np.empty((*depth.shape, 3), dtype=np.float32)
    for frame in range(len(depth)):
        rays = pixels @ np.linalg.inv(intrinsics[frame]).T
        camera_points = rays * depth[frame, ..., None]
        world_points = camera_points @ camera_poses[frame, :3, :3].T + camera_poses[frame, :3, 3]
        result[frame] = world_points.astype(np.float32)
    return result


def fuse_global_point_cloud(depth, intrinsics, camera_poses_c2w, *, confidence=None,
                            confidence_threshold: float | None = None, voxel_size: float | None = None) -> np.ndarray:
    """Return a confidence-filtered, optionally voxel-downsampled global cloud."""
    world_points = backproject_depth(depth, intrinsics, camera_poses_c2w)
    flat = world_points.reshape(-1, 3)
    valid = np.isfinite(flat).all(axis=1)
    depth_values = _numpy(depth).reshape(-1)
    valid &= np.isfinite(depth_values) & (depth_values > 0)
    if confidence is not None and confidence_threshold is not None:
        valid &= _numpy(confidence).reshape(-1) >= confidence_threshold
    flat = flat[valid]
    if voxel_size is not None and voxel_size > 0 and len(flat):
        voxel = np.floor(flat / float(voxel_size)).astype(np.int64)
        _, keep = np.unique(voxel, axis=0, return_index=True)
        flat = flat[np.sort(keep)]
    return flat


def apply_vio_result(predictions: Mapping, result: VIOResult, *, build_world_points: bool = True) -> dict:
    """Create mapping outputs while preserving LingBot depth/intrinsics/confidence."""
    output = dict(predictions)
    corrected = result.camera_poses_c2w[:, :3, :4].astype(np.float32)
    output["lingbot_extrinsic"] = predictions.get("extrinsic")
    if "world_points" in predictions:
        output["lingbot_world_points"] = predictions["world_points"]
    if "world_points_conf" in predictions:
        output["lingbot_world_points_conf"] = predictions["world_points_conf"]
    output["extrinsic"] = corrected
    output["extrinsic_convention"] = "c2w"
    output["vio_extrinsic_c2w"] = corrected
    output["vio_velocity"] = result.velocities.astype(np.float32)
    output["vio_gyro_bias"] = result.gyro_biases.astype(np.float32)
    output["vio_accel_bias"] = result.accel_biases.astype(np.float32)
    output["vio_gravity"] = result.state.gravity_world.astype(np.float32)
    output["vio_scale"] = np.float32(result.state.scale)
    output["vio_diagnostics"] = result.diagnostics
    if build_world_points:
        if "depth" not in predictions or "intrinsic" not in predictions:
            raise KeyError("depth and intrinsic are required to rebuild the VIO world map")
        output["world_points"] = backproject_depth(predictions["depth"], predictions["intrinsic"], result.camera_poses_c2w)
        if "depth_conf" in predictions:
            output["world_points_conf"] = predictions["depth_conf"]
    return output
