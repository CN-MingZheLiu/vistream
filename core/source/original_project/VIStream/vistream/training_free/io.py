"""File formats used by the training-free trajectory fusion entry point."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def timestamp_scale(values: np.ndarray, requested: str = "auto") -> float:
    """Resolve an explicit or magnitude-based timestamp unit to seconds."""
    known = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
    if requested != "auto":
        if requested not in known:
            raise ValueError("timestamp unit must be auto, s, ms, us, or ns")
        return known[requested]
    value = abs(float(np.asarray(values).reshape(-1)[0]))
    if value > 1e17:
        return 1e-9
    if value > 1e14:
        return 1e-6
    if value > 1e11:
        return 1e-3
    return 1.0


def load_trajectory_npz(
    path: str | Path,
    poses_key: str = "predicted_c2w",
    timestamps_key: str | None = None,
    timestamp_unit: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Load Lingbot poses and normalize their timestamps to seconds."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        if poses_key not in archive:
            raise KeyError(f"{path} has no {poses_key!r}; keys={archive.files}")
        if timestamps_key is None:
            candidates = ("timestamps", "timestamps_ns", "timestamp_ns")
            timestamps_key = next((key for key in candidates if key in archive), None)
        if timestamps_key is None or timestamps_key not in archive:
            raise KeyError(f"{path} has no usable timestamp key; keys={archive.files}")
        poses = np.asarray(archive[poses_key], dtype=np.float64)
        timestamps = np.asarray(archive[timestamps_key], dtype=np.float64).reshape(-1)
    if poses.ndim != 3 or poses.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(f"{path}:{poses_key} must be [N,3,4] or [N,4,4]")
    if len(timestamps) != len(poses):
        raise ValueError(f"{path} has {len(poses)} poses but {len(timestamps)} timestamps")
    unit = timestamp_unit
    if unit == "auto" and timestamps_key.endswith("_ns"):
        unit = "ns"
    timestamps *= timestamp_scale(timestamps, unit)
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{path}:{timestamps_key} must be strictly increasing")
    return poses, timestamps


def load_normalized_imu_csv(
    path: str | Path,
    order: str,
    timestamp_unit: str = "auto",
) -> np.ndarray:
    """Read a 7-column IMU CSV into ``[time, acc xyz, gyro xyz]``.

    ``gyro_first`` covers EuRoC/TUM-VI, while ``accel_first`` covers the
    normalized KITTI convention. Header/comment rows are ignored.
    """
    path = Path(path)
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or row[0].lstrip().startswith("#"):
                continue
            try:
                values = [float(value.strip()) for value in row[:7]]
            except ValueError:
                continue
            if len(values) == 7:
                rows.append(values)
    if not rows:
        raise ValueError(f"{path} contains no numeric 7-column IMU records")
    values = np.asarray(rows, dtype=np.float64)
    values[:, 0] *= timestamp_scale(values[:, 0], timestamp_unit)
    if order == "gyro_first":
        values = np.column_stack((values[:, 0], values[:, 4:7], values[:, 1:4]))
    elif order != "accel_first":
        raise ValueError("IMU order must be accel_first or gyro_first")
    sorted_indices = np.argsort(values[:, 0], kind="stable")
    values = values[sorted_indices]
    return values[np.r_[True, np.diff(values[:, 0]) > 0.0]]


def load_normalized_imu_npz(path: str | Path) -> np.ndarray:
    """Read a normalized raw IMU NPZ with ``samples`` or time/acc/gyro arrays."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        if "samples" in keys:
            samples = np.asarray(archive["samples"], dtype=np.float64)
        elif {"timestamps", "acc", "gyro"} <= keys:
            samples = np.column_stack(
                (archive["timestamps"], archive["acc"], archive["gyro"])
            )
        else:
            raise ValueError(f"{path} needs samples or timestamps+acc+gyro arrays")
    if samples.ndim != 2 or samples.shape[1] != 7:
        raise ValueError(f"{path} samples must have shape [N,7]")
    return samples
