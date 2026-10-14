import numpy as np

from lingbot_map.vio import FixedLagVIOBackend, ImuSample, VIOConfig


def test_backend_recovers_metric_scale_from_excited_motion() -> None:
    frame_count = 10
    frame_dt = 0.2
    timestamps = np.arange(frame_count) * frame_dt
    frequency = 3.0
    amplitude = 0.5
    metric_scale = 2.5
    metric_x = amplitude * np.sin(frequency * timestamps)
    metric_x -= metric_x[0]
    visual_poses = np.repeat(np.eye(4)[None], frame_count, axis=0)
    visual_poses[:, 0, 3] = metric_x / metric_scale

    imu_samples = []
    for timestamp in np.arange(0.0, timestamps[-1] + 0.002, 0.002):
        acceleration_x = -amplitude * frequency**2 * np.sin(frequency * timestamp)
        imu_samples.append(
            ImuSample(timestamp, np.array([acceleration_x, 0.0, 9.80665]), np.zeros(3))
        )
    config = VIOConfig(
        window_size=frame_count,
        max_iterations=8,
        visual_translation_sigma=0.04,
        log_scale_sigma=10.0,
    )
    result = FixedLagVIOBackend(config).optimize(
        visual_poses, timestamps, imu_samples
    )

    assert result.diagnostics["converged"]
    np.testing.assert_allclose(result.state.scale, metric_scale, rtol=2e-3)
    np.testing.assert_allclose(
        result.camera_poses_c2w[:, 0, 3], metric_x, atol=2e-3
    )
    np.testing.assert_allclose(
        np.linalg.norm(result.state.gravity_world), 9.80665, atol=1e-4
    )


def test_fixed_lag_backend_uses_bounded_overlapping_windows() -> None:
    frame_count = 7
    timestamps = np.arange(frame_count) * 0.1
    poses = np.repeat(np.eye(4)[None], frame_count, axis=0)
    poses[:, 0, 3] = timestamps
    samples = [
        ImuSample(timestamp, np.array([0.0, 0.0, 9.80665]), np.zeros(3))
        for timestamp in np.arange(0.0, timestamps[-1] + 0.005, 0.005)
    ]
    config = VIOConfig(window_size=4, window_stride=2, max_iterations=4)
    result = FixedLagVIOBackend(config).optimize(poses, timestamps, samples)

    assert result.diagnostics["num_windows"] == 3
    assert all(window["end"] - window["start"] == 4 for window in result.diagnostics["windows"])
    np.testing.assert_allclose(result.camera_poses_c2w[:, 0, 3], timestamps, atol=1e-5)


def test_fixed_lag_scale_uses_one_global_prior() -> None:
    frame_count = 18
    frame_dt = 0.2
    timestamps = np.arange(frame_count) * frame_dt
    frequency = 2.0
    amplitude = 0.4
    metric_scale = 1.8
    metric_x = amplitude * np.sin(frequency * timestamps)
    visual_poses = np.repeat(np.eye(4)[None], frame_count, axis=0)
    visual_poses[:, 0, 3] = metric_x / metric_scale
    samples = [
        ImuSample(
            timestamp,
            np.array([
                -amplitude * frequency**2 * np.sin(frequency * timestamp),
                0.0,
                9.80665,
            ]),
            np.zeros(3),
        )
        for timestamp in np.linspace(
            0.0, timestamps[-1], int(round(timestamps[-1] / 0.002)) + 1
        )
    ]
    config = VIOConfig(
        window_size=7,
        window_stride=5,
        initialization_frames=frame_count,
        max_iterations=8,
        visual_translation_sigma=0.04,
        log_scale_sigma=10.0,
    )

    result = FixedLagVIOBackend(config).optimize(visual_poses, timestamps, samples)

    window_scales = np.asarray([window["scale"] for window in result.diagnostics["windows"]])
    np.testing.assert_allclose(result.state.scale, np.median(window_scales), rtol=1e-12)
    np.testing.assert_allclose(result.state.scale, metric_scale, rtol=2e-2)
