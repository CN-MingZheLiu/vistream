#!/usr/bin/env python3
"""CPU smoke checks for geometry, causal adapters, freezing and local data."""

from tests.test_adapter import (
    test_adapter_gradients_do_not_touch_inputs_and_are_finite,
    test_causal_attention_mask_blocks_future_frames,
    test_single_frame_streaming_adapter_uses_motion_history,
    test_zero_initialized_adapter_is_identity,
)
from tests.test_freezing import test_backbones_are_frozen_and_checkpoint_is_adapter_only
from tests.test_imu import (
    test_causal_gru_encoder_is_rgb_causal,
    test_dual_rate_encoder_is_rgb_causal,
)
from tests.test_lie import test_se3_roundtrip_and_inverse, test_so3_roundtrip_small_and_regular_angles
from tests.test_metrics import test_point_cloud_metrics_identical_clouds, test_se3_and_sim3_alignment
from tests.test_pose_fusion import (
    test_pose_fusion_starts_at_native_pose,
    test_rotation_stage_cannot_change_translation,
)
from tests.test_streaming import (
    test_pose_fusion_streaming_carries_visual_inertial_state,
    test_stream_warmup_then_emits_each_rgb_frame,
)
from tests.test_train_utils import test_imu_augmentation_preserves_masks_and_timestamps


def main() -> None:
    tests = (
        test_so3_roundtrip_small_and_regular_angles,
        test_se3_roundtrip_and_inverse,
        test_zero_initialized_adapter_is_identity,
        test_adapter_gradients_do_not_touch_inputs_and_are_finite,
        test_causal_attention_mask_blocks_future_frames,
        test_single_frame_streaming_adapter_uses_motion_history,
        test_dual_rate_encoder_is_rgb_causal,
        test_causal_gru_encoder_is_rgb_causal,
        test_pose_fusion_starts_at_native_pose,
        test_rotation_stage_cannot_change_translation,
        test_se3_and_sim3_alignment,
        test_point_cloud_metrics_identical_clouds,
        test_backbones_are_frozen_and_checkpoint_is_adapter_only,
        test_stream_warmup_then_emits_each_rgb_frame,
        test_pose_fusion_streaming_carries_visual_inertial_state,
        test_imu_augmentation_preserves_masks_and_timestamps,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} smoke tests passed")


if __name__ == "__main__":
    main()
