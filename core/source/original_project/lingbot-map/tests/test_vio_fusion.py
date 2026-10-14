import numpy as np

from lingbot_map.vio import VIOResult, VIOState, apply_vio_result, backproject_depth


def test_backprojection_uses_corrected_camera_to_world_pose() -> None:
    depth = np.full((1, 2, 2, 1), 2.0, dtype=np.float32)
    intrinsic = np.array([[[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]])
    pose = np.eye(4)[None]
    pose[0, :3, 3] = [1.0, 2.0, 3.0]

    points = backproject_depth(depth, intrinsic, pose)

    np.testing.assert_allclose(points[0, 0, 0], [0.5, 1.5, 5.0])
    np.testing.assert_allclose(points[0, 1, 1], [1.5, 2.5, 5.0])


def test_apply_vio_preserves_lingbot_depth_intrinsics_and_confidence() -> None:
    depth = np.ones((1, 1, 1, 1), dtype=np.float32)
    confidence = np.ones((1, 1, 1), dtype=np.float32) * 2.0
    intrinsic = np.eye(3, dtype=np.float32)[None]
    original_pose = np.eye(4, dtype=np.float32)[None, :3]
    corrected_pose = np.eye(4)[None]
    corrected_pose[0, 0, 3] = 4.0
    state = VIOState(
        rotations_world_imu=np.eye(3)[None],
        positions_world_imu=np.array([[4.0, 0.0, 0.0]]),
        velocities_world=np.zeros((1, 3)),
        gyro_biases=np.zeros((1, 3)),
        accel_biases=np.zeros((1, 3)),
        gravity_world=np.array([0.0, 0.0, -9.80665]),
        scale=2.0,
    )
    result = VIOResult(corrected_pose, state, [], {"converged": True})
    predictions = {
        "depth": depth,
        "depth_conf": confidence,
        "intrinsic": intrinsic,
        "extrinsic": original_pose,
    }

    output = apply_vio_result(predictions, result)

    assert output["depth"] is depth
    assert output["depth_conf"] is confidence
    assert output["intrinsic"] is intrinsic
    np.testing.assert_allclose(output["world_points"][0, 0, 0], [4.0, 0.0, 1.0])
    assert output["extrinsic_convention"] == "c2w"
