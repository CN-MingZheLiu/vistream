import torch

from lingbot_map.models.gct_stream_window import GCTStream
from lingbot_map.utils.rotation import mat_to_quat
from lingbot_map.utils.window_alignment import (
    apply_sim3_to_w2c_pose_encoding,
    estimate_robust_window_sim3,
    w2c_pose_encoding_to_c2w,
)


def _rotation_y(angle: float) -> torch.Tensor:
    cosine = torch.cos(torch.tensor(angle))
    sine = torch.sin(torch.tensor(angle))
    return torch.tensor(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    )


def _encode_w2c(rotation_wc: torch.Tensor, center_w: torch.Tensor) -> torch.Tensor:
    rotation_cw = rotation_wc.transpose(-1, -2)
    translation_cw = -torch.matmul(rotation_cw, center_w.unsqueeze(-1)).squeeze(-1)
    fov = torch.full((*center_w.shape[:-1], 2), 1.0)
    return torch.cat([translation_cw, mat_to_quat(rotation_cw), fov], dim=-1)


def test_apply_sim3_preserves_w2c_convention() -> None:
    centers = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.2, 2.0], [2.0, -0.1, 5.0]]])
    rotations = torch.stack([_rotation_y(0.0), _rotation_y(0.1), _rotation_y(0.2)])[None]
    pose_enc = _encode_w2c(rotations, centers)
    sim_rotation = _rotation_y(-0.35)[None]
    sim_translation = torch.tensor([[3.0, -2.0, 4.0]])
    sim_scale = torch.tensor([2.5])

    warped = apply_sim3_to_w2c_pose_encoding(
        pose_enc, sim_rotation, sim_translation, sim_scale
    )
    warped_rotation, warped_center = w2c_pose_encoding_to_c2w(warped)

    expected_rotation = torch.matmul(sim_rotation[:, None], rotations)
    expected_center = (
        sim_scale[:, None, None]
        * torch.matmul(sim_rotation[:, None], centers.unsqueeze(-1)).squeeze(-1)
        + sim_translation[:, None]
    )
    torch.testing.assert_close(warped_rotation, expected_rotation, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(warped_center, expected_center, atol=1e-5, rtol=1e-5)


def test_robust_sim3_recovers_multiframe_alignment() -> None:
    target_centers = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.1, 1.8], [2.2, -0.2, 4.0],
          [3.5, 0.3, 6.5], [5.0, 0.0, 9.0], [6.8, -0.1, 12.0]]]
    )
    target_rotations = torch.stack([_rotation_y(0.03 * idx) for idx in range(6)])[None]
    sim_rotation = _rotation_y(0.42)[None]
    sim_translation = torch.tensor([[-4.0, 1.5, 7.0]])
    sim_scale = torch.tensor([1.7])
    anchor_centers = (
        sim_scale[:, None, None]
        * torch.matmul(sim_rotation[:, None], target_centers.unsqueeze(-1)).squeeze(-1)
        + sim_translation[:, None]
    )
    anchor_rotations = torch.matmul(sim_rotation[:, None], target_rotations)
    anchor_pose = _encode_w2c(anchor_rotations, anchor_centers)
    target_pose = _encode_w2c(target_rotations, target_centers)
    target_depth = torch.linspace(2.0, 20.0, 6 * 4 * 5).reshape(1, 6, 4, 5, 1)
    anchor_depth = target_depth * sim_scale.item()

    scale, rotation, translation = estimate_robust_window_sim3(
        anchor_pose, target_pose, anchor_depth, target_depth, scale_mode="hybrid"
    )

    torch.testing.assert_close(scale, sim_scale, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(rotation, sim_rotation, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(translation, sim_translation, atol=1e-4, rtol=1e-4)


def test_stitching_uses_real_frame_ids_and_prefers_new_window() -> None:
    first = {
        "pose_enc": torch.arange(6, dtype=torch.float32).reshape(1, 6, 1),
        "frame_indices": torch.arange(0, 6).reshape(1, 6),
    }
    second = {
        "pose_enc": (100 + torch.arange(5, dtype=torch.float32)).reshape(1, 5, 1),
        "frame_indices": torch.arange(4, 9).reshape(1, 5),
    }
    third = {
        "pose_enc": (200 + torch.arange(4, dtype=torch.float32)).reshape(1, 4, 1),
        "frame_indices": torch.arange(7, 11).reshape(1, 4),
    }
    model_stub = object.__new__(GCTStream)

    stitched = model_stub._stitch_windows([first, second, third], 6, 2)

    assert stitched["frame_indices"].tolist() == [list(range(11))]
    assert stitched["pose_enc"].flatten().tolist() == [
        0.0, 1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 200.0, 201.0, 202.0, 203.0
    ]
