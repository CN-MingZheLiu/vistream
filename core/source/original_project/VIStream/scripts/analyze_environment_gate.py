#!/usr/bin/env python3
"""Summarize GT-free loop/IMU agreement features for KITTI candidates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path("/home/mingzheliu/VIStream")
VISUAL_ROOT = Path(
    "/home/mingzheliu/MZL_workplace/lingbot-map/outputs/"
    "kitti_00_10_stride6_frozen"
)
IMU_ROOT = ROOT / "outputs/training_free_kitti_raw_imu_stride6_20260813"
LOOP_ROOT = ROOT / "outputs/training_free_loop_fixed_stride6_kitti_00_10_20260813"


def poses_4x4(values: np.ndarray) -> np.ndarray:
    if values.shape[1:] == (4, 4):
        return values
    output = np.repeat(np.eye(4)[None], len(values), axis=0)
    output[:, :3, :4] = values
    return output


def relative(poses: np.ndarray, src: int, dst: int) -> np.ndarray:
    return np.linalg.inv(poses[dst]) @ poses[src]


def rotation_error(measurement: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            Rotation.from_matrix(
                measurement[:3, :3].T @ prediction[:3, :3]
            ).as_rotvec()
        )
    )


for index in range(11):
    sequence = f"{index:02d}"
    with np.load(VISUAL_ROOT / sequence / "trajectory.npz") as archive:
        visual = poses_4x4(archive["predicted_c2w"])
    with np.load(IMU_ROOT / sequence / "candidate_before_gt.npz") as archive:
        imu = poses_4x4(archive["fused_c2w"])
    with np.load(LOOP_ROOT / sequence / "candidate_before_gt.npz") as archive:
        loop = poses_4x4(archive["c2w"])
    edges = json.loads(
        (LOOP_ROOT / sequence / "loop_edges_before_gt.json").read_text()
    )
    step_norm = np.linalg.norm(np.diff(visual[:, :3, 3], axis=0), axis=1)
    positive_steps = step_norm[step_norm > 1e-9]
    step_scale = float(np.median(positive_steps)) if len(positive_steps) else 1.0
    features = []
    for edge in edges:
        src, dst = edge["src_pos"], edge["dst_pos"]
        measurement = np.asarray(edge["transform_dst_from_src"])
        visual_relative = relative(visual, src, dst)
        imu_relative = relative(imu, src, dst)
        features.append(
            {
                "separation": dst - src,
                "score": edge["retrieval_score"],
                "inlier_ratio": edge["inlier_ratio"],
                "visual_rot_error": rotation_error(measurement, visual_relative),
                "imu_rot_error": rotation_error(measurement, imu_relative),
                "imu_support": rotation_error(measurement, visual_relative)
                - rotation_error(measurement, imu_relative),
                "translation_cycle_steps": float(
                    np.linalg.norm(
                        visual_relative[:3, 3]
                        - measurement[:3, 3]
                    )
                    / max(step_scale, 1e-9)
                ),
            }
        )
    visual_to_loop = np.linalg.norm(
        loop[:, :3, 3] - visual[:, :3, 3], axis=1
    )
    visual_to_imu = np.linalg.norm(
        imu[:, :3, 3] - visual[:, :3, 3], axis=1
    )
    summary = {
        "sequence": sequence,
        "edge_count": len(features),
        "step_scale": step_scale,
        "loop_path_change_median_steps": float(
            np.median(visual_to_loop) / max(step_scale, 1e-9)
        ),
        "imu_path_change_median_steps": float(
            np.median(visual_to_imu) / max(step_scale, 1e-9)
        ),
        "edges": features,
    }
    if features:
        for key in (
            "score",
            "inlier_ratio",
            "visual_rot_error",
            "imu_rot_error",
            "imu_support",
            "translation_cycle_steps",
        ):
            values = np.asarray([item[key] for item in features])
            summary[f"median_{key}"] = float(np.median(values))
    print(json.dumps(summary))
