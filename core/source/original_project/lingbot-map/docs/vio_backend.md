# Frozen LingBot-Map with a Fixed-Lag VIO Backend

## Data flow

1. LingBot-Map runs under `eval()`, `requires_grad_(False)`, and
   `torch.no_grad()`. It produces camera poses, depth, intrinsics, and dense
   confidence exactly as in RGB-only inference.
2. Every IMU sample between adjacent RGB timestamps is midpoint-preintegrated
   into delta rotation, delta velocity, delta position, a 15x15 covariance,
   and first-order gyroscope/accelerometer bias Jacobians.
3. A bounded fixed-lag optimizer uses adjacent LingBot relative poses as
   visual factors and preintegrated measurements as inertial factors. Its
   state contains `T_world_imu`, world velocity, both IMU biases, gravity, and
   a positive visual-to-metric scale.
4. Corrected `T_world_camera` poses are combined with the unchanged LingBot
   depth and intrinsics. Every depth map is backprojected directly into the
   corrected world frame to form the global point cloud.

The backend is implemented in NumPy/SciPy and has no dependency on or path
back into the neural network.

## Python API

```python
from lingbot_map.vio import (
    VIOConfig,
    apply_vio_result,
    fuse_lingbot_with_imu,
    load_camera_imu_transform,
    load_imu_csv,
    load_timestamps,
)

imu = load_imu_csv("mav0/imu0/data.csv")
timestamps = load_timestamps("mav0/cam0/data.csv")
T_cam_imu = load_camera_imu_transform("dso/camchain.yaml", "cam0")

result = fuse_lingbot_with_imu(
    predictions,                         # decoded `extrinsic` is C2W
    timestamps,
    imu,
    config=VIOConfig(window_size=10),
    transform_camera_imu=T_cam_imu,
)
corrected = apply_vio_result(predictions, result)
```

`apply_vio_result` returns a shallow copy. The objects stored under `depth`,
`depth_conf`, and `intrinsic` are preserved. The original visual pose is kept
under `lingbot_extrinsic`; `extrinsic`, `vio_extrinsic_c2w`, and
`world_points` contain the corrected mapping result.

## Coordinate conventions

- `extrinsic` passed to `fuse_lingbot_with_imu` is camera-to-world by default.
  Pass `visual_pose_convention="w2c"` for an OpenCV world-to-camera input.
- `T_cam_imu` maps IMU coordinates into camera coordinates, matching Kalibr
  and TUM-VI camchain files.
- Accelerometer data is raw specific force in `m/s^2`; gyroscope data is in
  `rad/s`; all timestamps are in seconds after loading.
- Gravity has fixed nominal magnitude `9.80665 m/s^2`, while its world-frame
  direction is estimated.

## Practical requirements

The camera and IMU clocks must be synchronized, and IMU samples must cover
the first through last selected RGB timestamp. Metric scale and gravity need
translational/rotational excitation; constant-velocity motion alone cannot
make scale observable. Supply an accurate camera-IMU transform when the
sensors are not colocated. Noise densities in `IMUNoise` should be changed to
match the sensor datasheet for production use.
