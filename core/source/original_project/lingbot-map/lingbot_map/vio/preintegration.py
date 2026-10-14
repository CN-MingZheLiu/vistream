"""Classical IMU preintegration on the manifold (Forster-style deltas).

The implementation is deliberately independent of torch and of LingBot's
network.  IMU samples are integrated once between two RGB timestamps; the
backend later applies the first-order bias Jacobians while optimizing biases.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .math import so3_exp, so3_log, skew


@dataclass(frozen=True)
class ImuSample:
    timestamp: float
    acceleration: np.ndarray
    gyroscope: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", float(self.timestamp))
        for name in ("acceleration", "gyroscope"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,):
                raise ValueError(f"{name} must have shape (3,), got {value.shape}")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class IMUNoise:
    """Continuous-time white noise and random-walk standard deviations."""

    gyro_noise_density: float = 1.7e-4
    accel_noise_density: float = 2.0e-3
    gyro_random_walk: float = 1.0e-5
    accel_random_walk: float = 2.0e-4


@dataclass
class PreintegratedMeasurement:
    timestamp_i: float
    timestamp_j: float
    delta_rotation: np.ndarray
    delta_velocity: np.ndarray
    delta_position: np.ndarray
    covariance: np.ndarray
    jacobian_rotation_bias: np.ndarray
    jacobian_velocity_gyro_bias: np.ndarray
    jacobian_velocity_accel_bias: np.ndarray
    jacobian_position_gyro_bias: np.ndarray
    jacobian_position_accel_bias: np.ndarray
    sample_count: int
    linearization_gyro_bias: np.ndarray
    linearization_accel_bias: np.ndarray

    @property
    def dt(self) -> float:
        return self.timestamp_j - self.timestamp_i

    def bias_corrected(self, gyro_bias: np.ndarray, accel_bias: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dbg = np.asarray(gyro_bias, dtype=np.float64) - self.linearization_gyro_bias
        dba = np.asarray(accel_bias, dtype=np.float64) - self.linearization_accel_bias
        rotation = self.delta_rotation @ so3_exp(self.jacobian_rotation_bias @ dbg)
        velocity = self.delta_velocity + self.jacobian_velocity_gyro_bias @ dbg + self.jacobian_velocity_accel_bias @ dba
        position = self.delta_position + self.jacobian_position_gyro_bias @ dbg + self.jacobian_position_accel_bias @ dba
        return rotation, velocity, position


def _as_samples(samples: Iterable[ImuSample] | np.ndarray) -> list[ImuSample]:
    if isinstance(samples, np.ndarray):
        data = np.asarray(samples, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != 7:
            raise ValueError("IMU array must have shape [N, 7] = timestamp, ax, ay, az, gx, gy, gz")
        return [ImuSample(row[0], row[1:4], row[4:7]) for row in data]
    result = list(samples)
    if not all(isinstance(item, ImuSample) for item in result):
        raise TypeError("samples must contain ImuSample objects")
    return result


def _timestamp_scale(value: float) -> float:
    value = abs(float(value))
    if value > 1e17:
        return 1e-9
    if value > 1e14:
        return 1e-6
    if value > 1e11:
        return 1e-3
    return 1.0


def load_imu_csv(path: str | Path, *, timestamp_unit: str = "auto", order: str = "tumvi") -> list[ImuSample]:
    """Load EuRoC/TUM-VI, KITTI, or a headerless 7-column IMU CSV.

    ``tumvi`` means timestamp, gyro xyz, accel xyz (the EuRoC convention).
    ``accel_first`` means timestamp, accel xyz, gyro xyz.
    Timestamps are returned in seconds and samples are sorted/deduplicated.
    """
    path = Path(path)
    rows: list[list[float]] = []
    detected_timestamp_unit: str | None = None
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if row[0].lstrip().startswith("#"):
                header = row[0].lower()
                for unit in ("ns", "us", "ms"):
                    if f"[{unit}]" in header:
                        detected_timestamp_unit = unit
                        break
                continue
            try:
                values = [float(item.strip()) for item in row[:7]]
            except ValueError:
                continue
            if len(values) == 7:
                rows.append(values)
    if not rows:
        raise ValueError(f"No 7-column IMU samples found in {path}")
    data = np.asarray(rows, dtype=np.float64)
    if timestamp_unit == "auto":
        scale = ({"ns": 1e-9, "us": 1e-6, "ms": 1e-3}.get(detected_timestamp_unit)
                 if detected_timestamp_unit else _timestamp_scale(data[0, 0]))
    else:
        scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}.get(timestamp_unit)
        if scale is None:
            raise ValueError("timestamp_unit must be auto, s, ms, us, or ns")
    data[:, 0] *= scale
    if order.lower() in {"tumvi", "euroc", "gyro_first"}:
        gyro, accel = data[:, 1:4], data[:, 4:7]
    elif order.lower() in {"accel_first", "kitti"}:
        accel, gyro = data[:, 1:4], data[:, 4:7]
    else:
        raise ValueError("order must be tumvi/euroc/gyro_first or accel_first/kitti")
    order_indices = np.argsort(data[:, 0], kind="stable")
    result: list[ImuSample] = []
    last_timestamp = -np.inf
    for index in order_indices:
        timestamp = float(data[index, 0])
        if timestamp <= last_timestamp:
            continue
        result.append(ImuSample(timestamp, accel[index], gyro[index]))
        last_timestamp = timestamp
    return result


def load_timestamps(path: str | Path, *, timestamp_unit: str = "auto") -> np.ndarray:
    """Load one timestamp per line or the first column of a CSV."""
    values: list[float] = []
    detected_timestamp_unit: str | None = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.lstrip().startswith("#"):
                lowered = line.lower()
                for unit in ("ns", "us", "ms"):
                    if f"[{unit}]" in lowered:
                        detected_timestamp_unit = unit
                        break
                continue
            token = line.strip().split(",")[0].split()[0] if line.strip() else ""
            try:
                values.append(float(token))
            except (ValueError, IndexError):
                continue
    if not values:
        raise ValueError(f"No timestamps found in {path}")
    timestamps = np.asarray(values, dtype=np.float64)
    if timestamp_unit == "auto":
        scale = ({"ns": 1e-9, "us": 1e-6, "ms": 1e-3}.get(detected_timestamp_unit)
                 if detected_timestamp_unit else _timestamp_scale(timestamps[0]))
    else:
        scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}.get(timestamp_unit)
        if scale is None:
            raise ValueError("timestamp_unit must be auto, s, ms, us, or ns")
    return timestamps * scale


def _sample_values(samples: Sequence[ImuSample], timestamps: np.ndarray, attribute: str) -> np.ndarray:
    source_timestamps = np.asarray([sample.timestamp for sample in samples], dtype=np.float64)
    source_values = np.asarray([getattr(sample, attribute) for sample in samples], dtype=np.float64)
    return np.column_stack([np.interp(timestamps, source_timestamps, source_values[:, axis]) for axis in range(3)])


def _integrate(samples: Sequence[ImuSample], timestamp_i: float, timestamp_j: float,
               gyro_bias: np.ndarray, accel_bias: np.ndarray, noise: IMUNoise,
               propagate_covariance: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if timestamp_j <= timestamp_i:
        raise ValueError("IMU interval must have positive duration")
    source_timestamps = np.asarray([s.timestamp for s in samples])
    if len(source_timestamps) == 0 or source_timestamps[0] > timestamp_i or source_timestamps[-1] < timestamp_j:
        raise ValueError(f"IMU samples do not cover [{timestamp_i}, {timestamp_j}]")
    inside = source_timestamps[(source_timestamps > timestamp_i) & (source_timestamps < timestamp_j)]
    boundaries = np.concatenate(([timestamp_i], inside, [timestamp_j]))
    # Midpoint values are linearly interpolated from the raw sensor stream.
    midpoints = 0.5 * (boundaries[:-1] + boundaries[1:])
    accelerations = _sample_values(samples, midpoints, "acceleration") - accel_bias
    gyroscopes = _sample_values(samples, midpoints, "gyroscope") - gyro_bias
    delta_rotation = np.eye(3)
    delta_velocity = np.zeros(3)
    delta_position = np.zeros(3)
    covariance = np.zeros((15, 15))
    for dt, acceleration, gyroscope in zip(np.diff(boundaries), accelerations, gyroscopes):
        dt = float(dt)
        rotation_before = delta_rotation
        delta_position += delta_velocity * dt + 0.5 * (rotation_before @ acceleration) * dt * dt
        delta_velocity += (rotation_before @ acceleration) * dt
        delta_rotation = delta_rotation @ so3_exp(gyroscope * dt)

        if propagate_covariance:
            # Error state: [dtheta, dv, dp, dbg, dba]. The covariance is in
            # the local preintegration frame and weights the inertial factor.
            F = np.eye(15)
            F[:3, :3] -= skew(gyroscope) * dt
            F[:3, 9:12] = -np.eye(3) * dt
            F[3:6, :3] = -rotation_before @ skew(acceleration) * dt
            F[3:6, 12:15] = -rotation_before * dt
            F[6:9, :3] = -0.5 * rotation_before @ skew(acceleration) * dt * dt
            F[6:9, 3:6] = np.eye(3) * dt
            F[6:9, 12:15] = -0.5 * rotation_before * dt * dt
            G = np.zeros((15, 12))
            G[:3, :3] = -np.eye(3) * dt
            G[3:6, 3:6] = -rotation_before * dt
            G[6:9, 3:6] = -0.5 * rotation_before * dt * dt
            G[9:12, 6:9] = np.eye(3) * dt
            G[12:15, 9:12] = np.eye(3) * dt
            continuous_covariance = np.diag([
                noise.gyro_noise_density ** 2] * 3 + [noise.accel_noise_density ** 2] * 3 +
                [noise.gyro_random_walk ** 2] * 3 + [noise.accel_random_walk ** 2] * 3
            ) / max(dt, 1e-9)
            covariance = F @ covariance @ F.T + G @ continuous_covariance @ G.T
    return delta_rotation, delta_velocity, delta_position, covariance, len(boundaries) - 1


def preintegrate_interval(samples: Iterable[ImuSample] | np.ndarray, timestamp_i: float, timestamp_j: float,
                          gyro_bias: np.ndarray | None = None, accel_bias: np.ndarray | None = None,
                          noise: IMUNoise | None = None) -> PreintegratedMeasurement:
    samples = _as_samples(samples)
    samples = sorted(samples, key=lambda sample: sample.timestamp)
    gyro_bias = np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, dtype=np.float64)
    accel_bias = np.zeros(3) if accel_bias is None else np.asarray(accel_bias, dtype=np.float64)
    noise = IMUNoise() if noise is None else noise
    delta_rotation, delta_velocity, delta_position, covariance, sample_count = _integrate(
        samples, float(timestamp_i), float(timestamp_j), gyro_bias, accel_bias, noise
    )
    # Finite-difference Jacobians are computed once per interval. This keeps
    # the propagation code readable and is negligible beside RGB inference.
    eps_g, eps_a = 1e-5, 1e-4
    jac_r_g = np.zeros((3, 3)); jac_v_g = np.zeros((3, 3)); jac_v_a = np.zeros((3, 3))
    jac_p_g = np.zeros((3, 3)); jac_p_a = np.zeros((3, 3))
    for axis in range(3):
        perturbed = gyro_bias.copy(); perturbed[axis] += eps_g
        r, v, p, _, _ = _integrate(
            samples, timestamp_i, timestamp_j, perturbed, accel_bias, noise, False
        )
        jac_r_g[:, axis] = so3_log(delta_rotation.T @ r) / eps_g
        jac_v_g[:, axis] = (v - delta_velocity) / eps_g
        jac_p_g[:, axis] = (p - delta_position) / eps_g
        perturbed = accel_bias.copy(); perturbed[axis] += eps_a
        _, v, p, _, _ = _integrate(
            samples, timestamp_i, timestamp_j, gyro_bias, perturbed, noise, False
        )
        jac_v_a[:, axis] = (v - delta_velocity) / eps_a
        jac_p_a[:, axis] = (p - delta_position) / eps_a
    return PreintegratedMeasurement(
        float(timestamp_i), float(timestamp_j), delta_rotation, delta_velocity, delta_position,
        covariance, jac_r_g, jac_v_g, jac_v_a, jac_p_g, jac_p_a, sample_count,
        gyro_bias.copy(), accel_bias.copy(),
    )


def preintegrate_sequence(samples: Iterable[ImuSample] | np.ndarray, timestamps: Sequence[float],
                          gyro_bias: np.ndarray | None = None, accel_bias: np.ndarray | None = None,
                          noise: IMUNoise | None = None) -> list[PreintegratedMeasurement]:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be a strictly increasing 1-D array with at least two entries")
    samples = sorted(_as_samples(samples), key=lambda sample: sample.timestamp)
    imu_timestamps = np.asarray([sample.timestamp for sample in samples])
    if len(samples) < 2 or imu_timestamps[0] > timestamps[0] or imu_timestamps[-1] < timestamps[-1]:
        raise ValueError(
            f"IMU samples [{imu_timestamps[0] if len(samples) else 'empty'}, "
            f"{imu_timestamps[-1] if len(samples) else 'empty'}] do not cover camera "
            f"timestamps [{timestamps[0]}, {timestamps[-1]}]"
        )
    measurements = []
    for index in range(len(timestamps) - 1):
        # Include one sample on either side so interpolation at both camera
        # timestamp boundaries remains valid without rescanning the full log.
        begin = max(0, int(np.searchsorted(imu_timestamps, timestamps[index], side="right")) - 1)
        end = min(
            len(samples),
            int(np.searchsorted(imu_timestamps, timestamps[index + 1], side="left")) + 1,
        )
        measurements.append(
            preintegrate_interval(
                samples[begin:end], timestamps[index], timestamps[index + 1],
                gyro_bias, accel_bias, noise,
            )
        )
    return measurements
