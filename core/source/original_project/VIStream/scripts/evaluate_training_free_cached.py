#!/usr/bin/env python3
"""Evaluate training-free fusion on frozen Lingbot cache files.

Candidate trajectories are written before ground truth is read. The entry
point intentionally evaluates one audited full sequence per dataset; it does
not use metrics to select a processor, a gate, or a cache.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vistream.metrics import ate_rmse
from vistream.training_free import FusionConfig, RawIMUProcessor, TrainingFreeIMUFusion
from vistream.training_free.io import load_trajectory_npz


DEFAULT_LINGBOT_ROOT = Path("/home/mingzheliu/MZL_workplace/lingbot-map")


def _poses_4x4(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape[1:] == (4, 4):
        return poses.copy()
    if poses.shape[1:] != (3, 4):
        raise ValueError(f"expected poses [N,3,4] or [N,4,4], got {poses.shape}")
    output = np.repeat(np.eye(4)[None], len(poses), axis=0)
    output[:, :3, :4] = poses
    return output


def _read_kitti_rigid(path: Path) -> np.ndarray:
    entries: dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, values = line.split(":", 1)
        if key in {"R", "T"}:
            entries[key] = np.fromstring(values, sep=" ", dtype=np.float64)
    if entries.get("R", np.empty(0)).size != 9 or entries.get("T", np.empty(0)).size != 3:
        raise ValueError(f"invalid KITTI R/T calibration: {path}")
    output = np.eye(4)
    output[:3, :3] = entries["R"].reshape(3, 3)
    output[:3, 3] = entries["T"]
    return output


def _read_kitti_transform(path: Path, key: str) -> np.ndarray:
    for line in path.read_text().splitlines():
        if line.startswith(f"{key}:"):
            values = np.fromstring(line.split(":", 1)[1], sep=" ", dtype=np.float64)
            output = np.eye(4)
            output[:3, :4] = values.reshape(3, 4)
            return output
    raise ValueError(f"missing {key} in {path}")


def _kitti_camera_from_imu(root: Path, imu_to_velo: Path) -> np.ndarray:
    calibration = root / "sequences/05/calib.txt"
    camera0_from_velo = _read_kitti_transform(calibration, "Tr")
    projection2 = None
    for line in calibration.read_text().splitlines():
        if line.startswith("P2:"):
            projection2 = np.fromstring(line.split(":", 1)[1], sep=" ", dtype=np.float64).reshape(3, 4)
            break
    if projection2 is None:
        raise ValueError(f"missing P2 in {calibration}")
    camera2_from_camera0 = np.eye(4)
    camera2_from_camera0[:3, 3] = np.linalg.solve(projection2[:, :3], projection2[:, 3])
    return camera2_from_camera0 @ camera0_from_velo @ _read_kitti_rigid(imu_to_velo)


def _ensure_coverage(samples: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.argsort(samples[:, 0], kind="stable")]
    samples = samples[np.r_[True, np.diff(samples[:, 0]) > 0.0]]
    median_period = float(np.median(np.diff(samples[:, 0])))
    maximum_hold = max(0.1, 20.0 * median_period)
    start_gap = max(0.0, float(samples[0, 0] - timestamps[0]))
    end_gap = max(0.0, float(timestamps[-1] - samples[-1, 0]))
    if start_gap > maximum_hold or end_gap > maximum_hold:
        raise ValueError(
            f"IMU does not cover camera stream: start={start_gap:.6f}s, "
            f"end={end_gap:.6f}s, allowed={maximum_hold:.6f}s"
        )
    if start_gap:
        first = samples[0].copy()
        first[0] = timestamps[0]
        samples = np.vstack((first, samples))
    if end_gap:
        last = samples[-1].copy()
        last[0] = timestamps[-1]
        samples = np.vstack((samples, last))
    return samples


def _interpolate_gt(
    path: Path, timestamps_ns: np.ndarray, camera_from_imu: np.ndarray, partial: bool
) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float64)
    source_ns = raw[:, 0].astype(np.int64)
    valid = (timestamps_ns >= source_ns[0]) & (timestamps_ns <= source_ns[-1])
    if not partial and not np.all(valid):
        raise ValueError("ground-truth timestamps do not cover the cached visual trajectory")
    query = np.clip(timestamps_ns, source_ns[0], source_ns[-1])
    source_seconds = (source_ns - source_ns[0]) * 1e-9
    query_seconds = (query - source_ns[0]) * 1e-9
    positions = np.column_stack(
        [np.interp(query_seconds, source_seconds, raw[:, column]) for column in (1, 2, 3)]
    )
    rotation = Slerp(source_seconds, Rotation.from_quat(raw[:, [5, 6, 7, 4]]))(
        query_seconds
    ).as_matrix()
    world_imu = np.repeat(np.eye(4)[None], len(query), axis=0)
    world_imu[:, :3, :3] = rotation
    world_imu[:, :3, 3] = positions
    output = world_imu @ np.linalg.inv(camera_from_imu)
    if partial:
        output[~valid] = np.nan
    return output


def _native_imu(path: Path, origin_ns: int) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float64)
    return np.column_stack(((raw[:, 0].astype(np.int64) - origin_ns) * 1e-9, raw[:, 4:7], raw[:, 1:4]))


def _matrix_yaml(path: Path, key: str) -> np.ndarray:
    values = yaml.safe_load(path.read_text())[key]
    if isinstance(values, dict):
        values = values["data"]
    output = np.asarray(values, dtype=np.float64)
    return output.reshape(4, 4)


def _oxford_imu(path: Path, origin: float) -> np.ndarray:
    records = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp = float(row["secs"]) + 1e-9 * float(row["nsecs"])
            records.append((
                timestamp - origin,
                float(row["acc_x"]), float(row["acc_y"]), float(row["acc_z"]),
                float(row["ang_vel_x"]), float(row["ang_vel_y"]), float(row["ang_vel_z"]),
            ))
    return np.asarray(records, dtype=np.float64)


def _oxford_camera_from_imu(path: Path) -> np.ndarray:
    entry = yaml.safe_load(path.read_text())["cam0"]
    output = np.eye(4)
    output[:3, :3] = Rotation.from_quat(entry["C_q_CI"]).as_matrix()
    output[:3, 3] = np.asarray(entry["C_r_CI"], dtype=np.float64)
    return output


def _metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(target).all(axis=(1, 2))
    if np.count_nonzero(valid) < 3:
        raise ValueError("fewer than three valid ground-truth poses")
    pred = predicted[valid, :3, 3]
    gt = target[valid, :3, 3]
    se3 = ate_rmse(pred, gt, "se3")
    sim3 = ate_rmse(pred, gt, "sim3")
    se3.pop("aligned")
    sim3.pop("aligned")
    return {"frames": int(np.count_nonzero(valid)), "ate_se3": se3, "ate_sim3": sim3}


def _write_candidate(
    output: Path,
    timestamps: np.ndarray,
    visual: np.ndarray,
    fused: np.ndarray,
    diagnostics: dict[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, timestamps=timestamps, visual_c2w=visual, fused_c2w=fused)
    output.with_suffix(".diagnostics.json").write_text(
        json.dumps(
            diagnostics,
            indent=2,
            default=lambda value: (
                value.tolist()
                if isinstance(value, np.ndarray)
                else value.item()
                if isinstance(value, np.generic)
                else str(value)
            ),
        )
        + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "kitti", "oxford", "tumvi", "euroc"), default="all")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/training_free_cached")
    parser.add_argument("--lingbot-root", type=Path, default=DEFAULT_LINGBOT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/training_free_rotation_only.json")
    return parser.parse_args()


def _bundle_kitti(root: Path) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Callable[[], np.ndarray]]:
    cache = root / "outputs/kitti_05_stride6_lingbot_map_long/trajectory.npz"
    visual, timestamps = load_trajectory_npz(cache)
    imu_path = Path("/data2/huangshuaixin/kiss_imu_data/KITTI_ODO/kitti_odo_05/imu.csv")
    raw = np.loadtxt(imu_path, delimiter=",", dtype=np.float64)
    imu = _ensure_coverage(np.column_stack((raw[:, 0], raw[:, 15:18], raw[:, 21:24])), timestamps)
    imu_to_velo = root / ".cache/kitti_2011_09_30/calib_imu_to_velo.txt"
    calibration_root = Path("/data2/huangshuaixin/LingBot-Map-relocalization/kitti_odometry/dataset")
    rotation = _kitti_camera_from_imu(calibration_root, imu_to_velo)[:3, :3]
    return (
        "kitti",
        visual,
        timestamps,
        imu,
        rotation,
        lambda: _poses_4x4(np.load(cache)["ground_truth_c2w"]),
    )


def _bundle_oxford(root: Path) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Callable[[], np.ndarray]]:
    cache = root / "outputs/oxford_all_current_long_3840/keble-college-02/trajectory.npz"
    visual, raw_timestamps = load_trajectory_npz(cache)
    # Historical cache records decoder W2C under this label; convert before fusion.
    visual = np.linalg.inv(_poses_4x4(visual))
    origin = float(raw_timestamps[0])
    timestamps = raw_timestamps - origin
    dataset = Path("/data2/huangshuaixin/datasets/Oxford_Spires")
    sequence = dataset / "sequences/2024-03-12-keble-college-02"
    imu = _ensure_coverage(_oxford_imu(sequence / "raw/imu.csv", origin), timestamps)
    transform = _oxford_camera_from_imu(dataset / "calibration/cam-lidar-imu.yaml")
    return (
        "oxford",
        visual,
        timestamps,
        imu,
        transform[:3, :3],
        lambda: _poses_4x4(np.load(cache)["ground_truth_c2w"]),
    )


def _bundle_tumvi(root: Path) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Callable[[], np.ndarray]]:
    cache = root / "outputs/tumvi_improved_room2_rectified/trajectory.npz"
    visual, timestamps = load_trajectory_npz(cache)
    timestamps -= timestamps[0]
    timestamps_ns = np.load(cache)["timestamps_ns"].astype(np.int64)
    dataset = Path("/data2/mingzheliu/datasets/TUM_VI/dataset-room2_512_16")
    transform = np.asarray(
        yaml.safe_load((dataset / "dso/camchain.yaml").read_text())["cam0"]["T_cam_imu"],
        dtype=np.float64,
    )
    imu = _ensure_coverage(_native_imu(dataset / "mav0/imu0/data.csv", int(timestamps_ns[0])), timestamps)
    return (
        "tumvi",
        visual,
        timestamps,
        imu,
        transform[:3, :3],
        lambda: _interpolate_gt(
            dataset / "mav0/mocap0/data.csv", timestamps_ns, transform, True
        ),
    )


def _bundle_euroc(root: Path) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Callable[[], np.ndarray]]:
    cache = root / "outputs/euroc_mh01_lingbot_map/trajectory.npz"
    visual, timestamps = load_trajectory_npz(cache)
    timestamps -= timestamps[0]
    timestamps_ns = np.load(cache)["timestamps_ns"].astype(np.int64)
    dataset = Path("/data1/mingzheliu/datasets/EuRoc_flat/MH_01_easy")
    transform = _matrix_yaml(dataset / "mav0/cam0/sensor.yaml", "T_BS")
    imu = _ensure_coverage(_native_imu(dataset / "mav0/imu0/data.csv", int(timestamps_ns[0])), timestamps)
    return (
        "euroc",
        visual,
        timestamps,
        imu,
        transform[:3, :3],
        lambda: _interpolate_gt(
            dataset / "mav0/state_groundtruth_estimate0/data.csv",
            timestamps_ns,
            transform,
            True,
        ),
    )


def main() -> None:
    args = parse_args()
    config = FusionConfig(**json.loads(args.config.read_text()))
    loaders = {
        "kitti": _bundle_kitti,
        "oxford": _bundle_oxford,
        "tumvi": _bundle_tumvi,
        "euroc": _bundle_euroc,
    }
    selected = list(loaders) if args.dataset == "all" else [args.dataset]
    summary: dict[str, Any] = {
        "protocol": "fused candidate persisted before GT load; frozen Lingbot and training-free IMU fusion",
        "config": config.__dict__,
        "results": {},
    }
    for name in selected:
        name, visual, timestamps, imu, rotation, ground_truth = loaders[name](args.lingbot_root)
        result = TrainingFreeIMUFusion(config).fuse(visual, timestamps, imu, RawIMUProcessor(), rotation)
        candidate = args.output_dir / name / "candidate_before_gt.npz"
        _write_candidate(candidate, timestamps, _poses_4x4(visual), result.camera_poses_c2w, result.diagnostics)
        gt = ground_truth()
        summary["results"][name] = {
            "candidate": str(candidate),
            "visual": _metrics(_poses_4x4(visual), gt),
            "training_free": _metrics(result.camera_poses_c2w, gt),
            "accepted": result.diagnostics["accepted"],
            "gate_mean": result.diagnostics["gate_mean"],
        }
        print(json.dumps({name: summary["results"][name]}, default=str), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
