"""Training-free, reliability-gated rotation fusion for LingBot trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .geometry import project_rotation, so3_exp, so3_log
from .imu_processor import IMUProcessingContext, IMUProcessor, IMUStream


@dataclass(frozen=True)
class FusionConfig:
    bias_window: int = 101
    innovation_soft_rad: float = 0.035
    innovation_hard_rad: float = 0.35
    visual_jerk_scale: float = 0.10
    minimum_gate: float = 0.0
    maximum_gate: float = 0.85
    gate_smoothing_frames: float = 2.0
    translation_mode: str = "step_blend"
    translation_window_frames: int = 32
    translation_direction_gain: float = 0.35
    maximum_translation_direction_rad: float = 0.08
    maximum_step_change_ratio: float = 0.02
    maximum_path_change_ratio: float = 0.025
    warmup_intervals: int = 8

    def __post_init__(self) -> None:
        if self.bias_window < 3:
            raise ValueError("bias_window must be at least 3")
        if not 0.0 <= self.minimum_gate <= self.maximum_gate <= 1.0:
            raise ValueError("gates must satisfy 0 <= minimum <= maximum <= 1")
        if self.innovation_soft_rad <= 0 or self.innovation_hard_rad <= self.innovation_soft_rad:
            raise ValueError("innovation thresholds must be positive and increasing")
        if self.maximum_step_change_ratio < 0 or self.maximum_path_change_ratio < 0:
            raise ValueError("visual-preservation budgets cannot be negative")
        if self.translation_mode not in {"none", "step_blend", "local_window"}:
            raise ValueError(
                "translation_mode must be 'none', 'step_blend', or 'local_window'"
            )
        if self.translation_window_frames < 2:
            raise ValueError("translation_window_frames must be at least 2")


@dataclass(frozen=True)
class FusionResult:
    camera_poses_c2w: np.ndarray
    interval_gates: np.ndarray
    diagnostics: dict[str, Any]


def _poses_4x4(poses: np.ndarray) -> np.ndarray:
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(f"visual poses must be [N,3,4] or [N,4,4], got {values.shape}")
    if values.shape[1:] == (4, 4):
        result = values.copy()
    else:
        result = np.repeat(np.eye(4)[None], len(values), axis=0)
        result[:, :3, :4] = values
    if len(result) < 2 or not np.isfinite(result).all():
        raise ValueError("visual trajectory must contain at least two finite poses")
    return result


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), 0.5 * weights.sum())])


def _robust_bias(errors: np.ndarray, window: int) -> np.ndarray:
    count = len(errors)
    radius = max(1, window // 2)
    output = np.zeros_like(errors)
    for index in range(count):
        local = errors[max(0, index - radius): min(count, index + radius + 1)]
        center = np.median(local, axis=0)
        distance = np.linalg.norm(local - center, axis=1)
        scale = max(1e-6, 1.4826 * float(np.median(np.abs(distance - np.median(distance)))))
        weights = 1.0 / (1.0 + (distance / (3.0 * scale)) ** 2)
        for axis in range(3):
            output[index, axis] = _weighted_median(local[:, axis], weights)
    return output


def _align_points_sim3(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align one local candidate to the visual window without ground truth."""
    source_center = source - source.mean(axis=0)
    target_center = target - target.mean(axis=0)
    covariance = target_center.T @ source_center / len(source)
    left, singular, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right) < 0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right
    variance = float(np.mean(np.sum(source_center**2, axis=1)))
    scale = float(np.sum(singular * np.diag(correction)) / max(variance, 1e-12))
    translation = target.mean(axis=0) - scale * (rotation @ source.mean(axis=0))
    return scale * (source @ rotation.T) + translation


def _integrate_gyro_intervals(
    samples: np.ndarray,
    frame_timestamps: np.ndarray,
    rotation_camera_from_imu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(samples, dtype=np.float64)
    times = values[:, 0]
    if times[0] > frame_timestamps[0] or times[-1] < frame_timestamps[-1]:
        raise ValueError(
            f"processed IMU [{times[0]}, {times[-1]}] does not cover camera "
            f"[{frame_timestamps[0]}, {frame_timestamps[-1]}]"
        )
    camera_from_imu = np.asarray(rotation_camera_from_imu, dtype=np.float64)
    if camera_from_imu.shape != (3, 3):
        raise ValueError("rotation_camera_from_imu must have shape [3,3]")
    rotations, sample_counts = [], []
    for begin, end in zip(frame_timestamps[:-1], frame_timestamps[1:]):
        inside = times[(times > begin) & (times < end)]
        boundaries = np.concatenate(([begin], inside, [end]))
        midpoints = 0.5 * (boundaries[:-1] + boundaries[1:])
        gyro = np.column_stack(
            [np.interp(midpoints, times, values[:, 4 + axis]) for axis in range(3)]
        )
        gyro = gyro @ camera_from_imu.T
        rotation = np.eye(3)
        for delta_t, angular_rate in zip(np.diff(boundaries), gyro):
            rotation = rotation @ so3_exp(angular_rate * delta_t)
        rotations.append(rotation)
        sample_counts.append(len(midpoints))
    return np.asarray(rotations), np.asarray(sample_counts, dtype=np.int64)


class TrainingFreeIMUFusion:
    """Fuse a frozen visual trajectory with arbitrary processed IMU output.

    Rotation is corrected only where IMU/visual innovation and visual temporal
    consistency support it. Translation magnitudes remain exactly visual; a
    bounded direction correction uses the fused orientation. The final path is
    rejected wholesale if it exceeds the configured visual-preservation budget.
    """

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = FusionConfig() if config is None else config

    def fuse(
        self,
        visual_poses_c2w: np.ndarray,
        frame_timestamps: np.ndarray,
        raw_imu_samples: np.ndarray,
        processor: IMUProcessor,
        rotation_camera_from_imu: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FusionResult:
        visual = _poses_4x4(visual_poses_c2w)
        timestamps = np.asarray(frame_timestamps, dtype=np.float64)
        if timestamps.shape != (len(visual),) or np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("frame_timestamps must match poses and be strictly increasing")
        camera_from_imu = (
            np.eye(3) if rotation_camera_from_imu is None
            else np.asarray(rotation_camera_from_imu, dtype=np.float64)
        )
        context = IMUProcessingContext(timestamps, camera_from_imu, metadata or {})
        processed = processor.process(np.asarray(raw_imu_samples), context)
        imu_rotations, sample_counts = self._processor_rotations(processed, context)
        if len(imu_rotations) != len(visual) - 1:
            raise ValueError("processor returned the wrong number of IMU intervals")

        visual_relative = np.linalg.inv(visual[:-1]) @ visual[1:]
        visual_rotvec = np.asarray([so3_log(item[:3, :3]) for item in visual_relative])
        imu_rotvec = np.asarray([so3_log(item) for item in imu_rotations])
        dt = np.diff(timestamps)
        bias_rate = _robust_bias((imu_rotvec - visual_rotvec) / dt[:, None], self.config.bias_window)
        unbiased_imu = np.asarray(
            [rotation @ so3_exp(-bias * delta_t) for rotation, bias, delta_t in zip(imu_rotations, bias_rate, dt)]
        )
        innovation = np.asarray(
            [so3_log(visual_r.T @ imu_r) for visual_r, imu_r in zip(visual_relative[:, :3, :3], unbiased_imu)]
        )
        innovation_norm = np.linalg.norm(innovation, axis=1)
        visual_rate = visual_rotvec / dt[:, None]
        visual_jerk = np.zeros(len(visual_rate))
        if len(visual_rate) > 1:
            visual_jerk[1:] = np.linalg.norm(np.diff(visual_rate, axis=0), axis=1)
        temporal_unreliability = visual_jerk / (visual_jerk + self.config.visual_jerk_scale)
        agreement = np.clip(
            (self.config.innovation_hard_rad - innovation_norm)
            / (self.config.innovation_hard_rad - self.config.innovation_soft_rad),
            0.0,
            1.0,
        )
        # Large innovation is usable only when the visual stream itself is
        # temporally unstable. Stable visual motion remains the hard anchor.
        gates = agreement + (1.0 - agreement) * temporal_unreliability
        if processed.interval_confidence is not None:
            confidence = np.asarray(processed.interval_confidence, dtype=np.float64).reshape(-1)
            if len(confidence) != len(gates):
                raise ValueError("processor confidence must have one value per camera interval")
            gates *= np.clip(confidence, 0.0, 1.0)
        gates = np.clip(gates, self.config.minimum_gate, self.config.maximum_gate)
        gates[: min(self.config.warmup_intervals, len(gates))] = 0.0
        if self.config.gate_smoothing_frames > 0 and len(gates) > 1:
            gates = gaussian_filter1d(gates, self.config.gate_smoothing_frames, mode="nearest")
            gates = np.clip(gates, self.config.minimum_gate, self.config.maximum_gate)

        fused = self._reconstruct(visual, unbiased_imu, gates)
        accepted, preservation = self._preservation_check(visual, fused)
        if not accepted:
            fused = visual.copy()
            gates = np.zeros_like(gates)
        diagnostics = {
            "training_free": True,
            "processor": processed.source,
            "processor_metadata": dict(processed.metadata),
            "config": asdict(self.config),
            "accepted": accepted,
            "fallback_reason": None if accepted else "visual_preservation_budget_exceeded",
            "intervals": len(gates),
            "imu_sample_counts": sample_counts,
            "estimated_gyro_bias_rate": bias_rate,
            "innovation_rad": innovation_norm,
            "visual_temporal_unreliability": temporal_unreliability,
            "gate_mean": float(gates.mean()),
            "gate_active_fraction": float(np.mean(gates > 1e-3)),
            **preservation,
        }
        return FusionResult(fused, gates, diagnostics)

    @staticmethod
    def _processor_rotations(
        processed: IMUStream, context: IMUProcessingContext
    ) -> tuple[np.ndarray, np.ndarray]:
        if processed.interval_delta_rotations is not None:
            rotations = np.asarray(processed.interval_delta_rotations, dtype=np.float64)
            return rotations, np.full(len(rotations), -1, dtype=np.int64)
        if processed.samples is None:
            raise ValueError("processor returned neither samples nor interval rotations")
        return _integrate_gyro_intervals(
            processed.samples,
            context.frame_timestamps,
            context.rotation_camera_from_imu,
        )

    def _reconstruct(
        self, visual: np.ndarray, imu_rotations: np.ndarray, gates: np.ndarray
    ) -> np.ndarray:
        result = visual.copy()
        result[0] = visual[0]
        for index, (imu_rotation, gate) in enumerate(zip(imu_rotations, gates)):
            relative = np.linalg.inv(visual[index]) @ visual[index + 1]
            visual_rotation = relative[:3, :3]
            correction = so3_log(visual_rotation.T @ imu_rotation)
            fused_relative_rotation = visual_rotation @ so3_exp(gate * correction)
            result[index + 1, :3, :3] = project_rotation(
                result[index, :3, :3] @ fused_relative_rotation
            )
        if self.config.translation_mode == "none":
            return result
        if self.config.translation_mode == "local_window":
            result[:, :3, 3] = self._local_window_positions(visual, result[:, :3, :3])
            return result
        for index, gate in enumerate(gates):
            visual_step_world = visual[index + 1, :3, 3] - visual[index, :3, 3]
            visual_step_local = visual[index, :3, :3].T @ visual_step_world
            rotated_step = result[index, :3, :3] @ visual_step_local
            blend = min(1.0, gate * self.config.translation_direction_gain)
            candidate = (1.0 - blend) * visual_step_world + blend * rotated_step
            visual_norm = float(np.linalg.norm(visual_step_world))
            candidate_norm = float(np.linalg.norm(candidate))
            if visual_norm > 1e-12 and candidate_norm > 1e-12:
                candidate *= visual_norm / candidate_norm
                angle = float(
                    np.arccos(
                        np.clip(
                            np.dot(candidate, visual_step_world) / (visual_norm**2),
                            -1.0,
                            1.0,
                        )
                    )
                )
                if angle > self.config.maximum_translation_direction_rad:
                    ratio = self.config.maximum_translation_direction_rad / angle
                    candidate = (1.0 - ratio) * visual_step_world + ratio * candidate
                    candidate *= visual_norm / max(float(np.linalg.norm(candidate)), 1e-12)
            result[index + 1, :3, 3] = result[index, :3, 3] + candidate
        return result

    def _local_window_positions(
        self, visual: np.ndarray, fused_rotations: np.ndarray
    ) -> np.ndarray:
        """Propagate corrected headings in overlapping visual-aligned windows."""
        frame_count = len(visual)
        window = min(self.config.translation_window_frames, frame_count)
        overlap = window // 2
        stride = max(1, window - overlap)
        visual_positions = visual[:, :3, 3]
        visual_steps = np.diff(visual_positions, axis=0)
        local_steps = np.einsum(
            "nij,nj->ni", visual[:-1, :3, :3].transpose(0, 2, 1), visual_steps
        )
        accumulated = np.zeros((frame_count, 3), dtype=np.float64)
        weights = np.zeros(frame_count, dtype=np.float64)
        for start in range(0, frame_count - 1, stride):
            end = min(frame_count, start + window)
            candidate = np.empty((end - start, 3), dtype=np.float64)
            candidate[0] = visual_positions[start]
            for index in range(start, end - 1):
                candidate[index - start + 1] = (
                    candidate[index - start]
                    + fused_rotations[index] @ local_steps[index]
                )
            candidate = _align_points_sim3(candidate, visual_positions[start:end])
            local_weights = np.ones(end - start, dtype=np.float64)
            taper = min(overlap, end - start - 1)
            if taper:
                ramp = np.linspace(1.0 / (taper + 1), 1.0, taper)
                if start > 0:
                    local_weights[:taper] = ramp
                if end < frame_count:
                    local_weights[-taper:] = ramp[::-1]
            accumulated[start:end] += candidate * local_weights[:, None]
            weights[start:end] += local_weights
            if end == frame_count:
                break
        if np.any(weights <= 0.0):
            raise RuntimeError("local translation windows did not cover every frame")
        return accumulated / weights[:, None]

    def _preservation_check(
        self, visual: np.ndarray, fused: np.ndarray
    ) -> tuple[bool, dict[str, float]]:
        visual_steps = np.diff(visual[:, :3, 3], axis=0)
        fused_steps = np.diff(fused[:, :3, 3], axis=0)
        scale = max(float(np.sqrt(np.mean(np.sum(visual_steps**2, axis=1)))), 1e-9)
        step_change = float(
            np.sqrt(np.mean(np.sum((fused_steps - visual_steps) ** 2, axis=1))) / scale
        )
        visual_path = float(np.linalg.norm(visual[-1, :3, 3] - visual[0, :3, 3]))
        path_scale = max(visual_path, float(np.sum(np.linalg.norm(visual_steps, axis=1))) * 0.1, 1e-9)
        path_change = float(
            np.sqrt(np.mean(np.sum((fused[:, :3, 3] - visual[:, :3, 3]) ** 2, axis=1)))
            / path_scale
        )
        numerical_tolerance = 1e-12
        accepted = (
            step_change <= self.config.maximum_step_change_ratio + numerical_tolerance
            and path_change <= self.config.maximum_path_change_ratio + numerical_tolerance
        )
        return accepted, {
            "relative_step_change": step_change,
            "relative_path_change": path_change,
        }
