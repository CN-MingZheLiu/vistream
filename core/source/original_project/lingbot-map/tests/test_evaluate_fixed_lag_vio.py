import numpy as np
import pytest
import yaml

from evaluate_fixed_lag_vio import _euroc_transform, _interpolate_body_ground_truth


def test_euroc_transform_reshapes_asl_yaml_matrix(tmp_path) -> None:
    expected = np.arange(16, dtype=np.float64).reshape(4, 4)
    path = tmp_path / "sensor.yaml"
    path.write_text(
        yaml.safe_dump(
            {"T_BS": {"rows": 4, "cols": 4, "data": expected.ravel().tolist()}}
        ),
        encoding="utf-8",
    )

    np.testing.assert_array_equal(_euroc_transform(path), expected)


def test_partial_ground_truth_marks_uncovered_endpoints(tmp_path) -> None:
    path = tmp_path / "data.csv"
    raw = np.array([
        [100, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [200, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ])
    np.savetxt(path, raw, delimiter=",")
    timestamps = np.array([50, 100, 150, 200, 250], dtype=np.int64)

    poses = _interpolate_body_ground_truth(
        path, timestamps, np.eye(4), allow_partial=True
    )

    assert np.isnan(poses[[0, -1]]).all()
    assert np.isfinite(poses[1:-1]).all()
    np.testing.assert_allclose(poses[2, :3, 3], [0.5, 0.0, 0.0])


def test_strict_ground_truth_rejects_large_endpoint_gap(tmp_path) -> None:
    path = tmp_path / "data.csv"
    raw = np.array([
        [1_000_000_000, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [2_000_000_000, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ])
    np.savetxt(path, raw, delimiter=",")

    with pytest.raises(ValueError, match="20 ms allowed"):
        _interpolate_body_ground_truth(
            path, np.array([0, 2_000_000_000], dtype=np.int64), np.eye(4)
        )
