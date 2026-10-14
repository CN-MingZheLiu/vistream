#!/usr/bin/env python3
"""Evaluate frozen LingBot-Map trajectories with the fixed-lag VIO backend.

The script intentionally opens ground truth only after the VIO trajectory has
been finalized.  Cached LingBot poses are valid inputs because the visual
network is frozen and the backend consumes only its decoded camera poses.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

from evaluate_kitti_odometry import aligned_poses, position_metrics, rotation_angle_deg, umeyama
from lingbot_map.vio import FixedLagVIOBackend, IMUNoise, VIOConfig


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs/fixed_lag_vio_four_datasets_20260811"


@dataclass
class DatasetBundle:
    name: str
    sequence: str
    visual_cache: Path
    visual_c2w: np.ndarray
    timestamps: np.ndarray
    imu: np.ndarray
    transform_camera_imu: np.ndarray
    imu_noise: IMUNoise
    ground_truth_loader: Callable[[], np.ndarray]
    metadata: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("all", "core", "kitti", "oxford", "tum", "euroc"), default="all",
        help="core runs the three locally cached datasets while a EuRoC cache is being prepared.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--initialization-frames", type=int, default=100)
    parser.add_argument(
        "--window-stride", type=int, default=8,
        help="Fixed-lag advance in frames; the default leaves two overlap states.",
    )
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Use only the first N frames for a diagnostic run; omitted means full sequences.",
    )
    parser.add_argument(
        "--kitti-cache", type=Path,
        default=ROOT / "outputs/kitti_05_stride6_lingbot_map_long/trajectory.npz",
    )
    parser.add_argument(
        "--kitti-root", type=Path,
        default=Path("/data2/huangshuaixin/LingBot-Map-relocalization/kitti_odometry/dataset"),
    )
    parser.add_argument(
        "--kitti-imu", type=Path,
        default=Path("/data2/huangshuaixin/kiss_imu_data/KITTI_ODO/kitti_odo_05/imu.csv"),
    )
    parser.add_argument(
        "--kitti-imu-to-velo", type=Path,
        default=ROOT / ".cache/kitti_2011_09_30/calib_imu_to_velo.txt",
    )
    parser.add_argument(
        "--oxford-cache", type=Path,
        default=ROOT / "outputs/oxford_all_current_long_3840/keble-college-02/trajectory.npz",
    )
    parser.add_argument(
        "--oxford-root", type=Path,
        default=Path("/data2/huangshuaixin/datasets/Oxford_Spires"),
    )
    parser.add_argument(
        "--tum-cache", type=Path,
        default=ROOT / "outputs/tumvi_improved_room2_rectified/trajectory.npz",
    )
    parser.add_argument(
        "--tum-root", type=Path,
        default=Path("/data2/mingzheliu/datasets/TUM_VI/dataset-room2_512_16"),
    )
    parser.add_argument(
        "--euroc-cache", type=Path,
        default=ROOT / "outputs/euroc_mh01_lingbot_map/trajectory.npz",
    )
    parser.add_argument(
        "--euroc-root", type=Path,
        default=Path("/data1/mingzheliu/datasets/EuRoc_flat/MH_01_easy"),
    )
    return parser.parse_args()


def _poses_4x4(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(f"Expected poses [N,3,4] or [N,4,4], got {poses.shape}")
    if poses.shape[1:] == (4, 4):
        return poses.copy()
    output = np.repeat(np.eye(4, dtype=np.float64)[None], len(poses), axis=0)
    output[:, :3, :4] = poses
    return output


def _read_transform(path: Path, key: str) -> np.ndarray:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}:"):
            values = np.fromstring(line.split(":", 1)[1], sep=" ", dtype=np.float64)
            if len(values) != 12:
                raise ValueError(f"{path}:{key} does not contain 12 values")
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :4] = values.reshape(3, 4)
            return transform
    raise ValueError(f"No {key} transform in {path}")


def _read_projection(path: Path, key: str) -> np.ndarray:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}:"):
            values = np.fromstring(line.split(":", 1)[1], sep=" ", dtype=np.float64)
            return values.reshape(3, 4)
    raise ValueError(f"No {key} projection in {path}")


def _read_kitti_rigid_calibration(path: Path) -> np.ndarray:
    entries: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in {"R", "T"}:
            entries[key] = np.fromstring(value, sep=" ", dtype=np.float64)
    if entries.get("R", np.empty(0)).size != 9 or entries.get("T", np.empty(0)).size != 3:
        raise ValueError(f"Expected KITTI R/T calibration in {path}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = entries["R"].reshape(3, 3)
    transform[:3, 3] = entries["T"]
    return transform


def _kitti_camera_imu(calib_path: Path, imu_to_velo_path: Path) -> np.ndarray:
    transform_cam0_velo = _read_transform(calib_path, "Tr")
    transform_velo_imu = _read_kitti_rigid_calibration(imu_to_velo_path)
    projection2 = _read_projection(calib_path, "P2")
    camera2_translation = np.linalg.solve(projection2[:, :3], projection2[:, 3])
    transform_cam2_cam0 = np.eye(4, dtype=np.float64)
    transform_cam2_cam0[:3, 3] = camera2_translation
    return transform_cam2_cam0 @ transform_cam0_velo @ transform_velo_imu


def _noise_from_yaml(path: Path) -> IMUNoise:
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    return IMUNoise(
        gyro_noise_density=float(values["gyroscope_noise_density"]),
        accel_noise_density=float(values["accelerometer_noise_density"]),
        gyro_random_walk=float(values["gyroscope_random_walk"]),
        accel_random_walk=float(values["accelerometer_random_walk"]),
    )


def _ensure_imu_coverage(imu: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    imu = np.asarray(imu, dtype=np.float64)
    order = np.argsort(imu[:, 0], kind="stable")
    imu = imu[order]
    imu = imu[np.r_[True, np.diff(imu[:, 0]) > 0.0]]
    median_dt = float(np.median(np.diff(imu[:, 0])))
    maximum_extension = max(0.1, 20.0 * median_dt)
    start_extension = max(0.0, float(imu[0, 0] - timestamps[0]))
    end_extension = max(0.0, float(timestamps[-1] - imu[-1, 0]))
    if start_extension > maximum_extension or end_extension > maximum_extension:
        raise ValueError(
            "IMU does not cover the camera stream: "
            f"start extension={start_extension:.6f}s, end extension={end_extension:.6f}s, "
            f"allowed={maximum_extension:.6f}s"
        )
    if start_extension:
        first = imu[0].copy()
        first[0] = timestamps[0]
        imu = np.vstack((first, imu))
    if end_extension:
        last = imu[-1].copy()
        last[0] = timestamps[-1]
        imu = np.vstack((imu, last))
    return imu, {
        "median_sample_period_s": median_dt,
        "held_start_s": start_extension,
        "held_end_s": end_extension,
    }


def _limit(bundle: DatasetBundle, maximum: int | None) -> DatasetBundle:
    if maximum is None:
        return bundle
    if maximum < 2:
        raise ValueError("--max-frames must be at least 2")
    count = min(maximum, len(bundle.timestamps))
    bundle.visual_c2w = bundle.visual_c2w[:count]
    bundle.timestamps = bundle.timestamps[:count]
    original_loader = bundle.ground_truth_loader
    bundle.ground_truth_loader = lambda: original_loader()[:count]
    bundle.metadata["diagnostic_max_frames"] = count
    return bundle


def load_kitti(args: argparse.Namespace) -> DatasetBundle:
    with np.load(args.kitti_cache, allow_pickle=False) as cache:
        visual = _poses_4x4(cache["predicted_c2w"])
        timestamps = np.asarray(cache["timestamps"], dtype=np.float64)
    raw = np.loadtxt(args.kitti_imu, delimiter=",", dtype=np.float64)
    imu = np.column_stack((raw[:, 0], raw[:, 15:18], raw[:, 21:24]))
    imu, coverage = _ensure_imu_coverage(imu, timestamps)
    transform = _kitti_camera_imu(
        args.kitti_root / "sequences/05/calib.txt", args.kitti_imu_to_velo
    )

    def load_gt() -> np.ndarray:
        with np.load(args.kitti_cache, allow_pickle=False) as cache:
            return _poses_4x4(cache["ground_truth_c2w"])

    return _limit(DatasetBundle(
        name="KITTI Odometry", sequence="05", visual_cache=args.kitti_cache,
        visual_c2w=visual, timestamps=timestamps, imu=imu,
        transform_camera_imu=transform,
        imu_noise=IMUNoise(
            gyro_noise_density=1.7e-4, accel_noise_density=2.0e-3,
            gyro_random_walk=1.0e-5, accel_random_walk=2.0e-4,
        ),
        ground_truth_loader=load_gt,
        metadata={
            "camera": "rectified image_2", "imu_format": "KITTI OXTS 100 Hz export",
            "imu_source": str(args.kitti_imu), "calibration_source": str(args.kitti_imu_to_velo),
            "imu_coverage": coverage,
        },
    ), args.max_frames)


def _load_oxford_imu(path: Path, origin: float) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = float(row["secs"]) + 1e-9 * float(row["nsecs"])
            rows.append([
                timestamp - origin,
                float(row["acc_x"]), float(row["acc_y"]), float(row["acc_z"]),
                float(row["ang_vel_x"]), float(row["ang_vel_y"]), float(row["ang_vel_z"]),
            ])
    return np.asarray(rows, dtype=np.float64)


def _load_oxford_transform(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)["cam0"]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(values["C_q_CI"]).as_matrix()
    transform[:3, 3] = np.asarray(values["C_r_CI"], dtype=np.float64)
    return transform


def load_oxford(args: argparse.Namespace) -> DatasetBundle:
    with np.load(args.oxford_cache, allow_pickle=False) as cache:
        # This historical cache stores the decoder's W2C matrix under the
        # benchmark_c2w label. VIO requires the physical C2W convention.
        visual = np.linalg.inv(_poses_4x4(cache["predicted_c2w"]))
        raw_timestamps = np.asarray(cache["timestamps"], dtype=np.float64)
    origin = float(raw_timestamps[0])
    timestamps = raw_timestamps - origin
    sequence_root = args.oxford_root / "sequences/2024-03-12-keble-college-02"
    imu_path = sequence_root / "raw/imu.csv"
    imu, coverage = _ensure_imu_coverage(_load_oxford_imu(imu_path, origin), timestamps)
    calibration_path = args.oxford_root / "calibration/cam-lidar-imu.yaml"
    noise_path = args.oxford_root / "calibration/imu.yaml"

    def load_gt() -> np.ndarray:
        with np.load(args.oxford_cache, allow_pickle=False) as cache:
            return _poses_4x4(cache["ground_truth_c2w"])

    return _limit(DatasetBundle(
        name="Oxford Spires", sequence="keble-college-02", visual_cache=args.oxford_cache,
        visual_c2w=visual, timestamps=timestamps, imu=imu,
        transform_camera_imu=_load_oxford_transform(calibration_path),
        imu_noise=_noise_from_yaml(noise_path), ground_truth_loader=load_gt,
        metadata={
            "camera": "cam0 official centered rectification", "imu_source": str(imu_path),
            "visual_pose_adapter": "inverted historical benchmark_c2w cache to OpenCV C2W",
            "calibration_source": str(calibration_path), "noise_source": str(noise_path),
            "absolute_timestamp_origin_s": origin, "imu_coverage": coverage,
        },
    ), args.max_frames)


def _load_native_imu(path: Path, origin_ns: int) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float64)
    timestamp_ns = raw[:, 0].astype(np.int64)
    return np.column_stack(((timestamp_ns - origin_ns) * 1e-9, raw[:, 4:7], raw[:, 1:4]))


def _load_camchain_transform(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as handle:
        return np.asarray(yaml.safe_load(handle)["cam0"]["T_cam_imu"], dtype=np.float64)


def _interpolate_body_ground_truth(
    path: Path, timestamps_ns: np.ndarray, transform_camera_imu: np.ndarray,
    *, allow_partial: bool = False,
) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float64)
    gt_ns = raw[:, 0].astype(np.int64)
    start_gap = max(0.0, float(gt_ns[0] - timestamps_ns[0]) * 1e-9)
    end_gap = max(0.0, float(timestamps_ns[-1] - gt_ns[-1]) * 1e-9)
    if not allow_partial and (start_gap > 0.02 or end_gap > 0.02):
        raise ValueError(
            f"Camera stream exceeds ground truth by start={start_gap:.6f}s, "
            f"end={end_gap:.6f}s (20 ms allowed)"
        )
    valid = (timestamps_ns >= gt_ns[0]) & (timestamps_ns <= gt_ns[-1])
    clipped_ns = np.clip(timestamps_ns, gt_ns[0], gt_ns[-1])
    query = (clipped_ns - gt_ns[0]) * 1e-9
    source = (gt_ns - gt_ns[0]) * 1e-9
    positions = np.column_stack([
        np.interp(query, source, raw[:, axis]) for axis in (1, 2, 3)
    ])
    # TUM-VI and EuRoC both store body/IMU orientation as q_wxyz.
    body_rotations = Slerp(
        source, Rotation.from_quat(raw[:, [5, 6, 7, 4]])
    )(query).as_matrix()
    world_imu = np.repeat(np.eye(4, dtype=np.float64)[None], len(query), axis=0)
    world_imu[:, :3, :3] = body_rotations
    world_imu[:, :3, 3] = positions
    world_camera = world_imu @ np.linalg.inv(transform_camera_imu)
    if allow_partial:
        world_camera[~valid] = np.nan
    return world_camera


def load_tum(args: argparse.Namespace) -> DatasetBundle:
    with np.load(args.tum_cache, allow_pickle=False) as cache:
        visual = _poses_4x4(cache["predicted_c2w"])
        timestamps_ns = np.asarray(cache["timestamps_ns"], dtype=np.int64)
    origin_ns = int(timestamps_ns[0])
    timestamps = (timestamps_ns - origin_ns) * 1e-9
    imu_path = args.tum_root / "mav0/imu0/data.csv"
    imu, coverage = _ensure_imu_coverage(_load_native_imu(imu_path, origin_ns), timestamps)
    calibration_path = args.tum_root / "dso/camchain.yaml"
    transform = _load_camchain_transform(calibration_path)
    noise_path = args.tum_root / "dso/imu_config.yaml"

    def load_gt() -> np.ndarray:
        return _interpolate_body_ground_truth(
            args.tum_root / "mav0/mocap0/data.csv", timestamps_ns, transform
        )

    return _limit(DatasetBundle(
        name="TUM-VI", sequence="dataset-room2_512_16", visual_cache=args.tum_cache,
        visual_c2w=visual, timestamps=timestamps, imu=imu,
        transform_camera_imu=transform, imu_noise=_noise_from_yaml(noise_path),
        ground_truth_loader=load_gt,
        metadata={
            "camera": "cam0 rectified 90 degree pinhole", "imu_source": str(imu_path),
            "calibration_source": str(calibration_path), "noise_source": str(noise_path),
            "timestamp_origin_ns": origin_ns, "imu_coverage": coverage,
        },
    ), args.max_frames)


def _euroc_transform(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    # EuRoC calls this T_BS; for cam0 it is the published IMU-to-camera transform.
    transform = values["T_BS"]
    rows, cols = int(transform["rows"]), int(transform["cols"])
    matrix = np.asarray(transform["data"], dtype=np.float64)
    if rows != 4 or cols != 4 or matrix.size != rows * cols:
        raise ValueError(f"{path} has an invalid 4x4 T_BS transform")
    return matrix.reshape(rows, cols)


def load_euroc(args: argparse.Namespace) -> DatasetBundle:
    if not args.euroc_cache.is_file():
        raise FileNotFoundError(
            f"Missing EuRoC frozen visual cache: {args.euroc_cache}. "
            "Generate it with cache_euroc_visual.py first."
        )
    with np.load(args.euroc_cache, allow_pickle=False) as cache:
        visual = _poses_4x4(cache["predicted_c2w"])
        timestamps_ns = np.asarray(cache["timestamps_ns"], dtype=np.int64)
    origin_ns = int(timestamps_ns[0])
    timestamps = (timestamps_ns - origin_ns) * 1e-9
    imu_path = args.euroc_root / "mav0/imu0/data.csv"
    imu, coverage = _ensure_imu_coverage(_load_native_imu(imu_path, origin_ns), timestamps)
    sensor_path = args.euroc_root / "mav0/cam0/sensor.yaml"
    transform = _euroc_transform(sensor_path)
    imu_sensor_path = args.euroc_root / "mav0/imu0/sensor.yaml"

    def load_gt() -> np.ndarray:
        return _interpolate_body_ground_truth(
            args.euroc_root / "mav0/state_groundtruth_estimate0/data.csv",
            timestamps_ns, transform, allow_partial=True,
        )

    return _limit(DatasetBundle(
        name="EuRoC MAV", sequence="MH_01_easy", visual_cache=args.euroc_cache,
        visual_c2w=visual, timestamps=timestamps, imu=imu,
        transform_camera_imu=transform, imu_noise=_noise_from_yaml(imu_sensor_path),
        ground_truth_loader=load_gt,
        metadata={
            "camera": "cam0", "imu_source": str(imu_path),
            "calibration_source": str(sensor_path), "noise_source": str(imu_sensor_path),
            "timestamp_origin_ns": origin_ns, "imu_coverage": coverage,
        },
    ), args.max_frames)


def _time_rpe(
    predicted: np.ndarray, ground_truth: np.ndarray, timestamps: np.ndarray, delta_s: float,
) -> dict[str, float | int]:
    right = np.searchsorted(timestamps, timestamps + delta_s, side="left")
    first = np.arange(len(timestamps), dtype=np.int64)
    valid = right < len(timestamps)
    first, right = first[valid], right[valid]
    if not len(first):
        return {
            "pairs": 0, "target_delta_s": delta_s, "mean_delta_s": math.nan,
            "translation_rmse_m": math.nan, "rotation_rmse_deg": math.nan,
        }
    gt_relative = np.linalg.inv(ground_truth[first]) @ ground_truth[right]
    pred_relative = np.linalg.inv(predicted[first]) @ predicted[right]
    error = np.linalg.inv(gt_relative) @ pred_relative
    translation = np.linalg.norm(error[:, :3, 3], axis=1)
    rotation = rotation_angle_deg(error[:, :3, :3])
    return {
        "pairs": int(len(first)), "target_delta_s": delta_s,
        "mean_delta_s": float(np.mean(timestamps[right] - timestamps[first])),
        "translation_rmse_m": float(np.sqrt(np.mean(translation ** 2))),
        "translation_mean_m": float(np.mean(translation)),
        "rotation_rmse_deg": float(np.sqrt(np.mean(rotation ** 2))),
        "rotation_mean_deg": float(np.mean(rotation)),
    }


def evaluate_trajectory(poses: np.ndarray, ground_truth: np.ndarray, timestamps: np.ndarray) -> dict:
    predicted_positions = poses[:, :3, 3]
    gt_positions = ground_truth[:, :3, 3]
    se3_positions, se3_rotation, _, se3_translation = umeyama(
        predicted_positions, gt_positions, with_scale=False
    )
    sim3_positions, sim3_rotation, sim3_scale, sim3_translation = umeyama(
        predicted_positions, gt_positions, with_scale=True
    )
    se3 = aligned_poses(poses, se3_rotation, 1.0, se3_translation)
    sim3 = aligned_poses(poses, sim3_rotation, sim3_scale, sim3_translation)
    return {
        "ate_raw": position_metrics(predicted_positions, gt_positions),
        "ate_se3_aligned": position_metrics(se3_positions, gt_positions),
        "ate_sim3_aligned": position_metrics(sim3_positions, gt_positions),
        "sim3_alignment_scale": float(sim3_scale),
        "rpe_1s_se3_aligned": _time_rpe(se3, ground_truth, timestamps, 1.0),
        "rpe_1s_sim3_aligned": _time_rpe(sim3, ground_truth, timestamps, 1.0),
    }


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def run_dataset(bundle: DatasetBundle, args: argparse.Namespace) -> dict:
    output_dir = args.output_dir / bundle.name.lower().replace(" ", "_").replace("-", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = VIOConfig(
        window_size=args.window_size, window_stride=args.window_stride,
        initialization_frames=args.initialization_frames,
        max_iterations=args.max_iterations, robust_loss="linear",
        imu_noise=bundle.imu_noise,
    )
    print(
        f"[{bundle.name}] {bundle.sequence}: {len(bundle.timestamps)} frames, "
        f"{len(bundle.imu)} IMU samples; running VIO...",
        flush=True,
    )
    started = time.perf_counter()
    result = FixedLagVIOBackend(config, bundle.transform_camera_imu).optimize(
        bundle.visual_c2w, bundle.timestamps, bundle.imu
    )
    runtime = time.perf_counter() - started

    # Leakage boundary: the optimized trajectory is persisted before GT is opened.
    np.savez_compressed(
        output_dir / "trajectory_before_gt.npz",
        timestamps=bundle.timestamps,
        visual_c2w=bundle.visual_c2w,
        vio_c2w=result.camera_poses_c2w,
        velocities=result.velocities,
        gyro_biases=result.gyro_biases,
        accel_biases=result.accel_biases,
        gravity_world=result.state.gravity_world,
        recovered_scale=result.state.scale,
        transform_camera_imu=bundle.transform_camera_imu,
    )

    ground_truth = _poses_4x4(bundle.ground_truth_loader())
    if len(ground_truth) != len(bundle.timestamps):
        raise ValueError(
            f"Ground truth count {len(ground_truth)} != trajectory count {len(bundle.timestamps)}"
        )
    ground_truth_valid = np.isfinite(ground_truth).all(axis=(1, 2))
    evaluation_frames = int(np.count_nonzero(ground_truth_valid))
    if evaluation_frames < 2:
        raise ValueError(f"Ground truth has only {evaluation_frames} valid overlapping frames")
    evaluation_timestamps = bundle.timestamps[ground_truth_valid]
    visual_metrics = evaluate_trajectory(
        bundle.visual_c2w[ground_truth_valid], ground_truth[ground_truth_valid],
        evaluation_timestamps,
    )
    vio_metrics = evaluate_trajectory(
        result.camera_poses_c2w[ground_truth_valid], ground_truth[ground_truth_valid],
        evaluation_timestamps,
    )
    windows = result.diagnostics["windows"]
    successful = sum(bool(window["success"]) for window in windows)
    metrics = {
        "dataset": bundle.name, "sequence": bundle.sequence,
        "frames": len(bundle.timestamps), "evaluation_frames": evaluation_frames,
        "imu_samples": len(bundle.imu),
        "duration_s": float(bundle.timestamps[-1] - bundle.timestamps[0]),
        "visual_cache": str(bundle.visual_cache),
        "protocol": {
            "lingbot_frozen": True, "imu_enters_lingbot": False,
            "ground_truth_use": "metrics only, after trajectory_before_gt.npz was written",
            "ground_truth_coverage": "official timestamp overlap only; no extrapolation",
            "window_size": config.window_size, "window_stride": config.window_stride,
            "initialization_frames": config.initialization_frames,
            "max_iterations": config.max_iterations,
        },
        "calibration": {
            "transform_camera_imu": bundle.transform_camera_imu.tolist(),
            "imu_noise": asdict(bundle.imu_noise),
        },
        "input": bundle.metadata,
        "backend": {
            "runtime_s": runtime, "windows": len(windows), "successful_windows": successful,
            "convergence_rate": successful / max(len(windows), 1),
            "recovered_visual_to_metric_scale": float(result.state.scale),
            "gravity_world": result.state.gravity_world.tolist(),
            "gravity_magnitude": float(np.linalg.norm(result.state.gravity_world)),
            "mean_gyro_bias": result.gyro_biases.mean(axis=0).tolist(),
            "mean_accel_bias": result.accel_biases.mean(axis=0).tolist(),
        },
        "visual": visual_metrics, "vio": vio_metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, default=_json_default)
        handle.write("\n")
    np.savez_compressed(
        output_dir / "trajectory.npz", timestamps=bundle.timestamps,
        visual_c2w=bundle.visual_c2w, vio_c2w=result.camera_poses_c2w,
        ground_truth_c2w=ground_truth,
    )
    print(
        f"[{bundle.name}] done in {runtime:.1f}s: "
        f"SE3 ATE {visual_metrics['ate_se3_aligned']['rmse_m']:.3f} -> "
        f"{vio_metrics['ate_se3_aligned']['rmse_m']:.3f} m; "
        f"scale={result.state.scale:.4g}",
        flush=True,
    )
    return metrics


def write_summary(results: list[dict], output_dir: Path) -> None:
    summary_path = output_dir / "summary.json"
    merged: dict[tuple[str, str], dict] = {}
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
        for item in previous.get("sequences", []):
            merged[(item["dataset"], item["sequence"])] = item
    for item in results:
        merged[(item["dataset"], item["sequence"])] = item
    results = list(merged.values())
    summary = {
        "protocol": "one representative full sequence per dataset; no GT in VIO",
        "sequences": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)
        handle.write("\n")
    columns = [
        "dataset", "sequence", "frames", "duration_s", "runtime_s", "convergence_rate",
        "recovered_scale", "visual_se3_ate_rmse_m", "vio_se3_ate_rmse_m",
        "visual_sim3_ate_rmse_m", "vio_sim3_ate_rmse_m",
        "visual_rpe_1s_translation_rmse_m", "vio_rpe_1s_translation_rmse_m",
        "visual_rpe_1s_rotation_rmse_deg", "vio_rpe_1s_rotation_rmse_deg",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in results:
            writer.writerow({
                "dataset": item["dataset"], "sequence": item["sequence"],
                "frames": item["frames"], "duration_s": item["duration_s"],
                "runtime_s": item["backend"]["runtime_s"],
                "convergence_rate": item["backend"]["convergence_rate"],
                "recovered_scale": item["backend"]["recovered_visual_to_metric_scale"],
                "visual_se3_ate_rmse_m": item["visual"]["ate_se3_aligned"]["rmse_m"],
                "vio_se3_ate_rmse_m": item["vio"]["ate_se3_aligned"]["rmse_m"],
                "visual_sim3_ate_rmse_m": item["visual"]["ate_sim3_aligned"]["rmse_m"],
                "vio_sim3_ate_rmse_m": item["vio"]["ate_sim3_aligned"]["rmse_m"],
                "visual_rpe_1s_translation_rmse_m": item["visual"]["rpe_1s_se3_aligned"]["translation_rmse_m"],
                "vio_rpe_1s_translation_rmse_m": item["vio"]["rpe_1s_se3_aligned"]["translation_rmse_m"],
                "visual_rpe_1s_rotation_rmse_deg": item["visual"]["rpe_1s_se3_aligned"]["rotation_rmse_deg"],
                "vio_rpe_1s_rotation_rmse_deg": item["vio"]["rpe_1s_se3_aligned"]["rotation_rmse_deg"],
            })


def main() -> None:
    args = parse_args()
    if args.window_stride >= args.window_size:
        raise ValueError("--window-stride must be smaller than --window-size")
    loaders = {
        "kitti": load_kitti, "oxford": load_oxford,
        "tum": load_tum, "euroc": load_euroc,
    }
    if args.dataset == "all":
        selected = list(loaders)
    elif args.dataset == "core":
        selected = ["kitti", "oxford", "tum"]
    else:
        selected = [args.dataset]
    results = []
    for name in selected:
        results.append(run_dataset(loaders[name](args), args))
        write_summary(results, args.output_dir)


if __name__ == "__main__":
    main()
