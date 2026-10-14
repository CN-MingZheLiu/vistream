import numpy as np

from lingbot_map.vio import ImuSample, load_imu_csv, preintegrate_interval


def test_stationary_preintegration_and_covariance() -> None:
    gravity = 9.80665
    samples = [
        ImuSample(index * 0.01, np.array([0.0, 0.0, gravity]), np.zeros(3))
        for index in range(101)
    ]
    measurement = preintegrate_interval(samples, 0.0, 1.0)

    np.testing.assert_allclose(measurement.delta_rotation, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(measurement.delta_velocity, [0.0, 0.0, gravity], atol=1e-8)
    np.testing.assert_allclose(measurement.delta_position, [0.0, 0.0, 0.5 * gravity], atol=1e-8)
    assert measurement.sample_count == 100
    assert np.linalg.eigvalsh(measurement.covariance).min() > 0.0


def test_tumvi_csv_loader_converts_ns_and_column_order(tmp_path) -> None:
    path = tmp_path / "imu.csv"
    path.write_text(
        "#timestamp [ns],gx,gy,gz,ax,ay,az\n"
        "1000000000,1,2,3,4,5,6\n"
        "1010000000,7,8,9,10,11,12\n",
        encoding="utf-8",
    )
    samples = load_imu_csv(path)

    assert [sample.timestamp for sample in samples] == [1.0, 1.01]
    np.testing.assert_array_equal(samples[0].gyroscope, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(samples[0].acceleration, [4.0, 5.0, 6.0])
