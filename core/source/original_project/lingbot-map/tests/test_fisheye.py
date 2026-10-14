import cv2
import numpy as np
import torch

from lingbot_map.utils.fisheye import rectify_equidistant_images


def test_rectify_equidistant_images_returns_rgb_tensor(tmp_path) -> None:
    height = width = 32
    horizontal = np.linspace(0, 255, width, dtype=np.uint8)
    image = np.broadcast_to(horizontal, (height, width)).copy()
    path = tmp_path / "fisheye.png"
    assert cv2.imwrite(str(path), image)

    calibration = {
        "resolution": [width, height],
        "intrinsics": [14.0, 14.0, 15.5, 15.5],
        "distortion_coeffs": [0.0, 0.0, 0.0, 0.0],
    }
    result = rectify_equidistant_images(
        [path], calibration, target_size=28, fov_degrees=90.0, workers=1
    )

    assert result.shape == (1, 3, 28, 28)
    assert torch.isfinite(result).all()
    assert 0.0 <= float(result.min()) <= float(result.max()) <= 1.0
    torch.testing.assert_close(result[:, 0], result[:, 1])
    torch.testing.assert_close(result[:, 1], result[:, 2])
