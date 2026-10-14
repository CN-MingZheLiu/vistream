#!/usr/bin/env python3
"""Compare expert routes using only visual/IMU/internal kinematics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


visual_root = Path(
    "/home/mingzheliu/MZL_workplace/lingbot-map/outputs/"
    "kitti_00_10_stride6_frozen"
)
imu_root = Path(
    "/home/mingzheliu/VIStream/outputs/"
    "training_free_kitti_raw_imu_stride6_20260813"
)
router_root = Path(
    "/home/mingzheliu/VIStream/outputs/"
    "training_free_expert_router_stride6_20260813"
)


def poses_4x4(values: np.ndarray) -> np.ndarray:
    if values.shape[1:] == (4, 4):
        return values
    output = np.repeat(np.eye(4)[None], len(values), axis=0)
    output[:, :3, :4] = values
    return output


def rotation_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.asarray([
        np.linalg.norm(Rotation.from_matrix(a.T @ b).as_rotvec())
        for a, b in zip(first, second)
    ])


for index in range(11):
    sequence = f"{index:02d}"
    with np.load(visual_root / sequence / "trajectory.npz") as archive:
        visual = poses_4x4(archive["predicted_c2w"])
    with np.load(imu_root / sequence / "candidate_before_gt.npz") as archive:
        imu = poses_4x4(archive["fused_c2w"])
    with np.load(router_root / sequence / "candidate_before_gt.npz") as archive:
        candidate = poses_4x4(archive["c2w"])
    diagnostics = json.loads(
        (router_root / sequence / "router_before_gt.json").read_text()
    )
    visual_steps = np.diff(visual[:, :3, 3], axis=0)
    candidate_steps = np.diff(candidate[:, :3, 3], axis=0)
    step_scale = max(float(np.median(np.linalg.norm(visual_steps, axis=1))), 1e-9)
    rotation_delta = rotation_distance(
        visual[:, :3, :3], imu[:, :3, :3]
    )
    visual_heading_change = rotation_distance(
        visual[:-1, :3, :3], visual[1:, :3, :3]
    )
    imu_heading_change = rotation_distance(
        imu[:-1, :3, :3], imu[1:, :3, :3]
    )
    step_difference = np.linalg.norm(candidate_steps - visual_steps, axis=1) / step_scale
    endpoint_change = float(
        np.linalg.norm(
            (candidate[-1, :3, 3] - candidate[0, :3, 3])
            - (visual[-1, :3, 3] - visual[0, :3, 3])
        )
        / max(float(np.sum(np.linalg.norm(visual_steps, axis=1))), 1e-9)
    )
    print(json.dumps({
        "sequence": sequence,
        "expert": diagnostics["expert"],
        "rotation_delta_q": np.quantile(rotation_delta, [0.5, 0.9, 1.0]).tolist(),
        "visual_turn_sum": float(np.sum(visual_heading_change)),
        "imu_turn_sum": float(np.sum(imu_heading_change)),
        "turn_sum_ratio": float(
            np.sum(imu_heading_change) / max(np.sum(visual_heading_change), 1e-9)
        ),
        "step_difference_q": np.quantile(step_difference, [0.5, 0.9, 1.0]).tolist(),
        "endpoint_change_path_ratio": endpoint_change,
        **diagnostics,
    }))
