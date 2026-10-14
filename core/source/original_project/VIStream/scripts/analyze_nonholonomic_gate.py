#!/usr/bin/env python3
"""Inspect GT-free vehicle-motion observability in frozen KITTI poses."""

from __future__ import annotations

from pathlib import Path

import numpy as np


visual_root = Path(
    "/home/mingzheliu/MZL_workplace/lingbot-map/outputs/"
    "kitti_00_10_stride6_frozen"
)
imu_root = Path(
    "/home/mingzheliu/VIStream/outputs/"
    "training_free_kitti_raw_imu_stride6_20260813"
)


def poses_4x4(values: np.ndarray) -> np.ndarray:
    if values.shape[1:] == (4, 4):
        return values
    output = np.repeat(np.eye(4)[None], len(values), axis=0)
    output[:, :3, :4] = values
    return output


for index in range(11):
    sequence = f"{index:02d}"
    with np.load(visual_root / sequence / "trajectory.npz") as archive:
        visual = poses_4x4(archive["predicted_c2w"])
    with np.load(imu_root / sequence / "candidate_before_gt.npz") as archive:
        imu = poses_4x4(archive["fused_c2w"])
    world_steps = np.diff(visual[:, :3, 3], axis=0)
    norms = np.linalg.norm(world_steps, axis=1)
    valid = norms > np.quantile(norms, 0.1)
    for label, rotations in (
        ("visual", visual[:-1, :3, :3]),
        ("imu", imu[:-1, :3, :3]),
    ):
        local = np.einsum("nij,nj->ni", rotations.transpose(0, 2, 1), world_steps)
        forward = local[:, 2]
        forward_ratio = np.abs(forward) / np.maximum(norms, 1e-9)
        planar_ratio = np.linalg.norm(local[:, [0, 2]], axis=1) / np.maximum(norms, 1e-9)
        sign = np.sign(forward[valid])
        sign_consistency = max(np.mean(sign >= 0), np.mean(sign <= 0))
        print(
            sequence,
            label,
            "forward_q", np.round(np.quantile(forward_ratio[valid], [0.1, 0.5, 0.9]), 4).tolist(),
            "planar_q", np.round(np.quantile(planar_ratio[valid], [0.1, 0.5, 0.9]), 4).tolist(),
            "sign", round(float(sign_consistency), 4),
        )
