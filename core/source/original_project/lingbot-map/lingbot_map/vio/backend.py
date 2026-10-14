"""VINS-Mono/OKVIS-style fixed-lag visual-inertial optimization backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .math import inverse_transform, make_transform, so3_exp, so3_log
from .preintegration import IMUNoise, ImuSample, PreintegratedMeasurement, preintegrate_sequence


@dataclass
class VIOConfig:
    window_size: int = 10
    window_stride: int = 1
    initialization_frames: int = 100
    max_iterations: int = 20
    robust_loss: str = "linear"
    huber_scale: float = 2.5
    gravity_magnitude: float = 9.80665
    initial_scale: float = 1.0
    visual_rotation_sigma: float = 0.08
    visual_translation_sigma: float = 0.20
    initial_velocity_sigma: float = 3.0
    initial_gyro_bias_sigma: float = 0.05
    initial_accel_bias_sigma: float = 0.5
    gravity_magnitude_sigma: float = 0.05
    log_scale_sigma: float = 0.25
    covariance_floor: float = 1e-10
    min_visual_confidence: float = 0.05
    imu_noise: IMUNoise = field(default_factory=IMUNoise)

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if self.window_stride < 1:
            raise ValueError("window_stride must be positive")
        if self.initialization_frames < 4:
            raise ValueError("initialization_frames must be at least 4")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.initial_scale <= 0:
            raise ValueError("initial_scale must be positive")


@dataclass
class VIOState:
    rotations_world_imu: np.ndarray
    positions_world_imu: np.ndarray
    velocities_world: np.ndarray
    gyro_biases: np.ndarray
    accel_biases: np.ndarray
    gravity_world: np.ndarray
    scale: float


@dataclass
class VIOResult:
    camera_poses_c2w: np.ndarray
    state: VIOState
    preintegrated_measurements: list[PreintegratedMeasurement]
    diagnostics: dict

    @property
    def velocities(self) -> np.ndarray:
        return self.state.velocities_world

    @property
    def gyro_biases(self) -> np.ndarray:
        return self.state.gyro_biases

    @property
    def accel_biases(self) -> np.ndarray:
        return self.state.accel_biases


def _poses_4x4(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(f"visual poses must have shape [N,3,4] or [N,4,4], got {poses.shape}")
    if poses.shape[1:] == (4, 4):
        return poses.copy()
    result = np.repeat(np.eye(4)[None], len(poses), axis=0)
    result[:, :3, :4] = poses
    return result


def _normalize_confidence(confidence: np.ndarray | None, count: int, minimum: float) -> np.ndarray:
    if confidence is None:
        return np.ones(count, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if len(confidence) == count - 1:
        confidence = np.concatenate(([confidence[0]], confidence))
    if len(confidence) != count:
        raise ValueError(f"visual confidence must have {count} or {count - 1} values")
    confidence = np.nan_to_num(confidence, nan=minimum, posinf=1.0, neginf=minimum)
    low, high = np.percentile(confidence, [10.0, 90.0])
    if high - low > 1e-12:
        confidence = (confidence - low) / (high - low)
    else:
        confidence = np.ones_like(confidence)
    return np.clip(confidence, minimum, 1.0)


class FixedLagVIOBackend:
    """Optimize metric IMU states while treating LingBot as a frozen sensor.

    The visual model is never called by this class. Its absolute camera poses
    are converted into adjacent relative-pose factors; IMU samples become
    preintegration factors. Each solve only includes a bounded number of
    states and carries its boundary state, gravity, and scale forward.
    """

    def __init__(self, config: VIOConfig | None = None, transform_camera_imu: np.ndarray | None = None) -> None:
        self.config = VIOConfig() if config is None else config
        self.transform_camera_imu = np.eye(4) if transform_camera_imu is None else np.asarray(transform_camera_imu, dtype=np.float64)
        if self.transform_camera_imu.shape != (4, 4):
            raise ValueError("transform_camera_imu must be a 4x4 matrix mapping IMU coordinates to camera coordinates")

    def optimize(self, visual_poses_c2w: np.ndarray, frame_timestamps: Sequence[float],
                 imu_samples: Sequence[ImuSample] | np.ndarray,
                 visual_confidence: np.ndarray | None = None) -> VIOResult:
        visual_poses = _poses_4x4(visual_poses_c2w)
        timestamps = np.asarray(frame_timestamps, dtype=np.float64)
        if len(visual_poses) != len(timestamps):
            raise ValueError("visual pose and frame timestamp counts differ")
        if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
            raise ValueError("at least two strictly increasing frame timestamps are required")
        confidence = _normalize_confidence(visual_confidence, len(timestamps), self.config.min_visual_confidence)
        measurements = preintegrate_sequence(imu_samples, timestamps, noise=self.config.imu_noise)
        state = self._initialize_state(visual_poses, timestamps, imu_samples, measurements)
        global_log_scale_prior = float(np.log(state.scale))

        window_size = min(self.config.window_size, len(timestamps))
        final_start = len(timestamps) - window_size
        starts = list(range(0, final_start + 1, self.config.window_stride))
        if starts[-1] != final_start:
            starts.append(final_start)
        window_diagnostics = []
        for start in starts:
            end = start + window_size
            state, diagnostic = self._solve_window(
                state, visual_poses[start:end], measurements[start:end - 1],
                confidence[start:end], start, global_log_scale_prior,
            )
            window_diagnostics.append({"start": start, "end": end, **diagnostic})

        # Scale is a single global state. Individual fixed-lag solves estimate
        # it against the same initialization prior, so report their robust
        # aggregate rather than the arbitrary value from the final window.
        state.scale = float(np.median([item["scale"] for item in window_diagnostics]))

        camera_poses = self._camera_poses(state)
        diagnostics = {
            "converged": all(item["success"] for item in window_diagnostics),
            "num_windows": len(window_diagnostics),
            "windows": window_diagnostics,
            "scale": state.scale,
            "gravity_world": state.gravity_world.copy(),
            "gravity_magnitude": float(np.linalg.norm(state.gravity_world)),
            "mean_gyro_bias": state.gyro_biases.mean(axis=0),
            "mean_accel_bias": state.accel_biases.mean(axis=0),
        }
        return VIOResult(camera_poses, state, measurements, diagnostics)

    def _initialize_state(self, visual_poses: np.ndarray, timestamps: np.ndarray,
                          imu_samples: Sequence[ImuSample] | np.ndarray,
                          measurements: Sequence[PreintegratedMeasurement]) -> VIOState:
        scale = self.config.initial_scale
        transform_world_imu = visual_poses @ self.transform_camera_imu
        rotations = transform_world_imu[:, :3, :3].copy()
        # Put the first visual camera at the origin while retaining the metric
        # camera-IMU lever arm in the IMU positions.
        camera_positions = scale * (visual_poses[:, :3, 3] - visual_poses[0, :3, 3])
        lever_camera = self.transform_camera_imu[:3, 3]
        positions = camera_positions + np.einsum("nij,j->ni", visual_poses[:, :3, :3], lever_camera)
        velocities = np.gradient(positions, timestamps, axis=0, edge_order=1)
        gyro_biases = np.zeros_like(positions)
        accel_biases = np.zeros_like(positions)
        if isinstance(imu_samples, np.ndarray):
            accel = np.asarray(imu_samples, dtype=np.float64)[:, 1:4]
        else:
            accel = np.asarray([sample.acceleration for sample in imu_samples], dtype=np.float64)
        mean_accel = np.median(accel[:max(1, min(len(accel), 100))], axis=0)
        norm = float(np.linalg.norm(mean_accel))
        if norm < 1e-6:
            gravity = np.array([0.0, 0.0, -self.config.gravity_magnitude])
        else:
            gravity = -rotations[0] @ mean_accel * (self.config.gravity_magnitude / norm)
        initialized = VIOState(rotations, positions, velocities, gyro_biases, accel_biases, gravity, scale)
        return self._linear_metric_initialization(
            initialized, visual_poses, measurements, timestamps
        )

    def _linear_metric_initialization(self, state: VIOState, visual_poses: np.ndarray,
                                      measurements: Sequence[PreintegratedMeasurement],
                                      timestamps: np.ndarray) -> VIOState:
        """Initialize scale, gravity, and velocity using the standard VINS linear solve."""
        total_count = len(visual_poses)
        count = min(total_count, self.config.initialization_frames)
        if count < 4:
            return state
        initialization_poses = visual_poses[:count]
        initialization_measurements = measurements[:count - 1]
        camera_positions = (
            initialization_poses[:, :3, 3] - initialization_poses[0, :3, 3]
        )
        lever = np.einsum(
            "nij,j->ni", initialization_poses[:, :3, :3],
            self.transform_camera_imu[:3, 3],
        )
        # First solve includes unconstrained gravity: x = [scale, gravity, velocities].
        columns = 4 + 3 * count
        matrix = np.zeros((6 * (count - 1), columns))
        target = np.zeros(6 * (count - 1))
        for interval, measurement in enumerate(initialization_measurements):
            row = 6 * interval
            dt = measurement.dt
            rotation = state.rotations_world_imu[interval]
            matrix[row:row + 3, 0] = camera_positions[interval + 1] - camera_positions[interval]
            matrix[row:row + 3, 1:4] = -0.5 * np.eye(3) * dt * dt
            matrix[row:row + 3, 4 + 3 * interval:4 + 3 * (interval + 1)] = -np.eye(3) * dt
            target[row:row + 3] = rotation @ measurement.delta_position - (lever[interval + 1] - lever[interval])
            matrix[row + 3:row + 6, 1:4] = -np.eye(3) * dt
            matrix[row + 3:row + 6, 4 + 3 * interval:4 + 3 * (interval + 1)] = -np.eye(3)
            matrix[row + 3:row + 6, 4 + 3 * (interval + 1):4 + 3 * (interval + 2)] = np.eye(3)
            target[row + 3:row + 6] = rotation @ measurement.delta_velocity
        solution, _, rank, singular_values = np.linalg.lstsq(matrix, target, rcond=1e-9)
        if rank < columns or singular_values[-1] <= singular_values[0] * 1e-8:
            return state
        gravity = solution[1:4]
        gravity_norm = float(np.linalg.norm(gravity))
        if gravity_norm < 1e-3:
            return state
        gravity *= self.config.gravity_magnitude / gravity_norm

        # With gravity direction fixed, solve scale and all velocities again.
        columns = 1 + 3 * count
        matrix = np.zeros((6 * (count - 1), columns))
        target = np.zeros(6 * (count - 1))
        for interval, measurement in enumerate(initialization_measurements):
            row = 6 * interval
            dt = measurement.dt
            rotation = state.rotations_world_imu[interval]
            matrix[row:row + 3, 0] = camera_positions[interval + 1] - camera_positions[interval]
            matrix[row:row + 3, 1 + 3 * interval:1 + 3 * (interval + 1)] = -np.eye(3) * dt
            target[row:row + 3] = (
                rotation @ measurement.delta_position
                - (lever[interval + 1] - lever[interval])
                + 0.5 * gravity * dt * dt
            )
            matrix[row + 3:row + 6, 1 + 3 * interval:1 + 3 * (interval + 1)] = -np.eye(3)
            matrix[row + 3:row + 6, 1 + 3 * (interval + 1):1 + 3 * (interval + 2)] = np.eye(3)
            target[row + 3:row + 6] = rotation @ measurement.delta_velocity + gravity * dt
        solution, _, rank, singular_values = np.linalg.lstsq(matrix, target, rcond=1e-9)
        candidate_scale = float(solution[0])
        if (rank < columns or singular_values[-1] <= singular_values[0] * 1e-8
                or not np.isfinite(candidate_scale) or candidate_scale <= 1e-4):
            return state
        state.scale = candidate_scale
        state.gravity_world = gravity
        all_camera_positions = visual_poses[:, :3, 3] - visual_poses[0, :3, 3]
        all_lever = np.einsum(
            "nij,j->ni", visual_poses[:, :3, :3], self.transform_camera_imu[:3, 3]
        )
        state.positions_world_imu = candidate_scale * all_camera_positions + all_lever
        state.velocities_world = np.gradient(
            state.positions_world_imu, timestamps, axis=0, edge_order=1
        )
        state.velocities_world[:count] = solution[1:].reshape(count, 3)
        return state

    def _solve_window(self, state: VIOState, visual_poses: np.ndarray,
                      measurements: Sequence[PreintegratedMeasurement], confidence: np.ndarray,
                      offset: int, global_log_scale_prior: float) -> tuple[VIOState, dict]:
        count = len(visual_poses)
        indices = np.arange(offset, offset + count)
        reference_rotations = state.rotations_world_imu[indices].copy()
        anchor_position = state.positions_world_imu[indices[0]].copy()

        # Layout: dR[1:], p[1:], v[:], bg[:], ba[:], gravity, log(scale).
        rotation_slice = slice(0, 3 * (count - 1))
        position_slice = slice(rotation_slice.stop, rotation_slice.stop + 3 * (count - 1))
        velocity_slice = slice(position_slice.stop, position_slice.stop + 3 * count)
        gyro_bias_slice = slice(velocity_slice.stop, velocity_slice.stop + 3 * count)
        accel_bias_slice = slice(gyro_bias_slice.stop, gyro_bias_slice.stop + 3 * count)
        gravity_slice = slice(accel_bias_slice.stop, accel_bias_slice.stop + 3)
        scale_index = gravity_slice.stop
        x0 = np.zeros(scale_index + 1)
        x0[position_slice] = state.positions_world_imu[indices[1:]].reshape(-1)
        x0[velocity_slice] = state.velocities_world[indices].reshape(-1)
        x0[gyro_bias_slice] = state.gyro_biases[indices].reshape(-1)
        x0[accel_bias_slice] = state.accel_biases[indices].reshape(-1)
        x0[gravity_slice] = state.gravity_world
        x0[scale_index] = np.log(state.scale)
        prior_velocity = state.velocities_world[indices[0]].copy()
        prior_gyro_bias = state.gyro_biases[indices[0]].copy()
        prior_accel_bias = state.accel_biases[indices[0]].copy()
        transform_imu_camera = inverse_transform(self.transform_camera_imu)

        def unpack(vector: np.ndarray):
            rotations = reference_rotations.copy()
            increments = vector[rotation_slice].reshape(count - 1, 3)
            for local in range(1, count):
                rotations[local] = reference_rotations[local] @ so3_exp(increments[local - 1])
            positions = np.empty((count, 3)); positions[0] = anchor_position
            positions[1:] = vector[position_slice].reshape(count - 1, 3)
            velocities = vector[velocity_slice].reshape(count, 3)
            gyro_biases = vector[gyro_bias_slice].reshape(count, 3)
            accel_biases = vector[accel_bias_slice].reshape(count, 3)
            gravity = vector[gravity_slice]
            scale = float(np.exp(np.clip(vector[scale_index], -8.0, 8.0)))
            return rotations, positions, velocities, gyro_biases, accel_biases, gravity, scale

        visual_relatives = [inverse_transform(visual_poses[i]) @ visual_poses[i + 1] for i in range(count - 1)]
        sqrt_information = [self._sqrt_information(measurement.covariance) for measurement in measurements]

        def residual(vector: np.ndarray) -> np.ndarray:
            rotations, positions, velocities, gyro_biases, accel_biases, gravity, scale = unpack(vector)
            values = [
                (velocities[0] - prior_velocity) / self.config.initial_velocity_sigma,
                (gyro_biases[0] - prior_gyro_bias) / self.config.initial_gyro_bias_sigma,
                (accel_biases[0] - prior_accel_bias) / self.config.initial_accel_bias_sigma,
                np.array([(np.linalg.norm(gravity) - self.config.gravity_magnitude) / self.config.gravity_magnitude_sigma]),
                np.array([(np.log(scale) - global_log_scale_prior) / self.config.log_scale_sigma]),
            ]
            camera_poses = []
            for local in range(count):
                transform_world_imu = make_transform(rotations[local], positions[local])
                camera_poses.append(transform_world_imu @ transform_imu_camera)
            for local, measurement in enumerate(measurements):
                dt = measurement.dt
                corrected_rotation, corrected_velocity, corrected_position = measurement.bias_corrected(
                    gyro_biases[local], accel_biases[local]
                )
                inertial = np.concatenate((
                    so3_log(corrected_rotation.T @ rotations[local].T @ rotations[local + 1]),
                    rotations[local].T @ (velocities[local + 1] - velocities[local] - gravity * dt) - corrected_velocity,
                    rotations[local].T @ (positions[local + 1] - positions[local] - velocities[local] * dt - 0.5 * gravity * dt * dt) - corrected_position,
                    gyro_biases[local + 1] - gyro_biases[local],
                    accel_biases[local + 1] - accel_biases[local],
                ))
                values.append(sqrt_information[local] @ inertial)
                predicted = inverse_transform(camera_poses[local]) @ camera_poses[local + 1]
                observed = visual_relatives[local]
                weight = np.sqrt(max(confidence[local], confidence[local + 1]))
                visual = np.concatenate((
                    so3_log(observed[:3, :3].T @ predicted[:3, :3]) / self.config.visual_rotation_sigma,
                    (predicted[:3, 3] - scale * observed[:3, 3]) / self.config.visual_translation_sigma,
                ))
                visual_norm = float(np.linalg.norm(visual) / np.sqrt(len(visual)))
                robust_weight = (
                    1.0 if visual_norm <= self.config.huber_scale
                    else np.sqrt(self.config.huber_scale / max(visual_norm, 1e-12))
                )
                values.append(weight * robust_weight * visual)
            return np.concatenate(values)

        sparsity = self._jacobian_sparsity(count, rotation_slice, position_slice, velocity_slice,
                                           gyro_bias_slice, accel_bias_slice, gravity_slice, scale_index)
        solution = least_squares(
            residual, x0, jac="2-point", jac_sparsity=sparsity,
            loss=self.config.robust_loss, f_scale=self.config.huber_scale,
            max_nfev=self.config.max_iterations, x_scale="jac",
        )
        rotations, positions, velocities, gyro_biases, accel_biases, gravity, scale = unpack(solution.x)
        updated = VIOState(
            state.rotations_world_imu.copy(), state.positions_world_imu.copy(), state.velocities_world.copy(),
            state.gyro_biases.copy(), state.accel_biases.copy(), gravity.copy(), scale,
        )
        updated.rotations_world_imu[indices] = rotations
        updated.positions_world_imu[indices] = positions
        updated.velocities_world[indices] = velocities
        updated.gyro_biases[indices] = gyro_biases
        updated.accel_biases[indices] = accel_biases
        return updated, {
            "success": bool(solution.success), "status": int(solution.status),
            "cost": float(solution.cost), "optimality": float(solution.optimality),
            "nfev": int(solution.nfev), "message": str(solution.message),
            "scale": float(scale),
        }

    def _sqrt_information(self, covariance: np.ndarray) -> np.ndarray:
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, self.config.covariance_floor)
        return np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

    @staticmethod
    def _jacobian_sparsity(count: int, rotation_slice: slice, position_slice: slice,
                           velocity_slice: slice, gyro_bias_slice: slice, accel_bias_slice: slice,
                           gravity_slice: slice, scale_index: int):
        rows = 11 + 21 * (count - 1)
        cols = scale_index + 1
        matrix = lil_matrix((rows, cols), dtype=np.int8)
        matrix[0:3, velocity_slice.start:velocity_slice.start + 3] = 1
        matrix[3:6, gyro_bias_slice.start:gyro_bias_slice.start + 3] = 1
        matrix[6:9, accel_bias_slice.start:accel_bias_slice.start + 3] = 1
        matrix[9, gravity_slice] = 1
        matrix[10, scale_index] = 1

        def state_columns(local: int, include_pose: bool = True) -> list[int]:
            columns: list[int] = []
            if include_pose and local > 0:
                columns.extend(range(rotation_slice.start + 3 * (local - 1), rotation_slice.start + 3 * local))
                columns.extend(range(position_slice.start + 3 * (local - 1), position_slice.start + 3 * local))
            columns.extend(range(velocity_slice.start + 3 * local, velocity_slice.start + 3 * (local + 1)))
            columns.extend(range(gyro_bias_slice.start + 3 * local, gyro_bias_slice.start + 3 * (local + 1)))
            columns.extend(range(accel_bias_slice.start + 3 * local, accel_bias_slice.start + 3 * (local + 1)))
            return columns

        row = 11
        for local in range(count - 1):
            imu_columns = state_columns(local) + state_columns(local + 1) + list(range(gravity_slice.start, gravity_slice.stop))
            matrix[row:row + 15, imu_columns] = 1
            row += 15
            visual_columns: list[int] = [scale_index]
            for frame in (local, local + 1):
                if frame > 0:
                    visual_columns.extend(range(rotation_slice.start + 3 * (frame - 1), rotation_slice.start + 3 * frame))
                    visual_columns.extend(range(position_slice.start + 3 * (frame - 1), position_slice.start + 3 * frame))
            matrix[row:row + 6, visual_columns] = 1
            row += 6
        return matrix.tocsr()

    def _camera_poses(self, state: VIOState) -> np.ndarray:
        transform_imu_camera = inverse_transform(self.transform_camera_imu)
        poses = np.empty((len(state.positions_world_imu), 4, 4))
        for index, (rotation, position) in enumerate(zip(state.rotations_world_imu, state.positions_world_imu)):
            poses[index] = make_transform(rotation, position) @ transform_imu_camera
        return poses
