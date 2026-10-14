#!/usr/bin/env python3
"""GT-free post-hoc fusion study for LingBot-Map and the trained IMU branch.

This is a zero-shot ablation, not an end-to-end trained LingBot+AirIMU model.
All candidate trajectories are constructed before KITTI ground truth is loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from evaluate_kitti_odometry import (
    aligned_poses,
    kitti_segment_metrics,
    load_ground_truth,
    position_metrics,
    relative_pose_metrics,
    umeyama,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LINGBOT = REPO_ROOT / "outputs/kitti_05_stride6_lingbot_map_long/trajectory.npz"
DEFAULT_IMU_PAIRS = Path(
    "/home/mingzheliu/MZL_workplace/IMU-lingbot_weights_unf_e15/"
    "eval_kitti_05_no_gt_leakage/pair_predictions.npz"
)
DEFAULT_KITTI = Path(
    "/data2/huangshuaixin/LingBot-Map-relocalization/kitti_odometry/dataset"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lingbot-trajectory", type=Path, default=DEFAULT_LINGBOT)
    parser.add_argument("--imu-pair-cache", type=Path, default=DEFAULT_IMU_PAIRS)
    parser.add_argument("--kitti-root", type=Path, default=DEFAULT_KITTI)
    parser.add_argument("--sequence", default="05")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
        help="Predeclared IMU-branch interpolation weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/kitti_05_lingbot_airimu_posthoc",
    )
    return parser.parse_args()


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / max(np.linalg.norm(quaternion), 1e-12)
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def current_branch_poses(rel_t: np.ndarray, rel_q: np.ndarray) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(rel_t) + 1, axis=0)
    for index, (translation, quaternion) in enumerate(zip(rel_t, rel_q)):
        poses[index + 1, :3, 3] = (
            poses[index, :3, 3] + poses[index, :3, :3] @ translation
        )
        poses[index + 1, :3, :3] = (
            poses[index, :3, :3] @ quaternion_wxyz_to_matrix(quaternion)
        )
    return poses


def interpolate_rotation(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    residual = first.T @ second
    residual_vector = Rotation.from_matrix(residual).as_rotvec()
    return first @ Rotation.from_rotvec(alpha * residual_vector).as_matrix()


def normalized_translation_blend(
    lingbot: np.ndarray, imu_branch: np.ndarray, alpha: float
) -> np.ndarray:
    lingbot_norm = float(np.linalg.norm(lingbot))
    imu_norm = float(np.linalg.norm(imu_branch))
    if lingbot_norm < 1e-12 or imu_norm < 1e-12:
        return lingbot.copy()
    # Preserve LingBot's per-interval scale and blend only motion direction.
    imu_scaled = imu_branch * (lingbot_norm / imu_norm)
    return (1.0 - alpha) * lingbot + alpha * imu_scaled


def fuse_relative_poses(
    lingbot_poses: np.ndarray,
    imu_rel_t: np.ndarray,
    imu_rel_q: np.ndarray,
    alpha: float,
    blend_translation: bool,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    output = np.empty_like(lingbot_poses)
    output[0] = lingbot_poses[0]
    lingbot_relative = np.linalg.inv(lingbot_poses[:-1]) @ lingbot_poses[1:]
    for index in range(len(imu_rel_t)):
        imu_rotation = quaternion_wxyz_to_matrix(imu_rel_q[index])
        relative = np.eye(4, dtype=np.float64)
        relative[:3, :3] = interpolate_rotation(
            lingbot_relative[index, :3, :3], imu_rotation, alpha
        )
        relative[:3, 3] = (
            normalized_translation_blend(
                lingbot_relative[index, :3, 3], imu_rel_t[index], alpha
            )
            if blend_translation
            else lingbot_relative[index, :3, 3]
        )
        output[index + 1] = output[index] @ relative
    return output


def evaluate(predicted: np.ndarray, ground_truth: np.ndarray) -> tuple[dict, np.ndarray]:
    predicted_positions = predicted[:, :3, 3]
    gt_positions = ground_truth[:, :3, 3]
    se3_positions, se3_rotation, _, se3_translation = umeyama(
        predicted_positions, gt_positions, with_scale=False
    )
    sim3_positions, sim3_rotation, sim3_scale, sim3_translation = umeyama(
        predicted_positions, gt_positions, with_scale=True
    )
    se3_poses = aligned_poses(predicted, se3_rotation, 1.0, se3_translation)
    sim3_poses = aligned_poses(
        predicted, sim3_rotation, sim3_scale, sim3_translation
    )
    result = {
        "ate_se3_aligned": position_metrics(se3_positions, gt_positions),
        "ate_sim3_aligned": position_metrics(sim3_positions, gt_positions),
        "sim3_scale": sim3_scale,
        "rpe_se3_delta_1": relative_pose_metrics(se3_poses, ground_truth, 1),
        "rpe_se3_delta_10": relative_pose_metrics(se3_poses, ground_truth, 10),
        "kitti_segment_drift_sim3_scale": kitti_segment_metrics(sim3_poses, ground_truth),
    }
    return result, sim3_positions


def save_plot(path: Path, trajectories: dict[str, np.ndarray], gt: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    axis.plot(gt[:, 0], gt[:, 2], color="black", linewidth=1.8, label="KITTI GT")
    for name, trajectory in trajectories.items():
        axis.plot(trajectory[:, 0], trajectory[:, 2], linewidth=0.9, label=name)
    axis.set_title("KITTI 05 post-hoc fusion sensitivity (Sim(3) aligned)")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("z [m]")
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.lingbot_trajectory) as data:
        frame_indices = data["frame_indices"].astype(np.int64)
        timestamps = data["timestamps"].astype(np.float64)
        lingbot_poses = data["predicted_c2w"].astype(np.float64)
    with np.load(args.imu_pair_cache) as data:
        imu_rel_t = data["rel_t"].astype(np.float64)
        imu_rel_q = data["rel_q"].astype(np.float64)
        imu_timestamps = data["timestamps"].astype(np.float64)
        valid = data["valid"].astype(bool)
    if not valid.all():
        raise ValueError(f"IMU pair cache contains {(~valid).sum()} invalid intervals")
    if len(lingbot_poses) != len(imu_rel_t) + 1:
        raise ValueError("LingBot pose and IMU interval counts differ")
    if not np.allclose(timestamps, imu_timestamps, atol=1e-6, rtol=0):
        raise ValueError("LingBot and IMU branches do not use identical timestamps")

    # Construct every candidate before opening the KITTI pose file.
    candidates: dict[str, np.ndarray] = {"lingbot": lingbot_poses}
    candidates["current_simple_rgb_imu"] = current_branch_poses(imu_rel_t, imu_rel_q)
    for alpha in args.alphas:
        key = f"rotation_alpha_{alpha:.2f}"
        candidates[key] = fuse_relative_poses(
            lingbot_poses, imu_rel_t, imu_rel_q, alpha, blend_translation=False
        )
        key = f"pose_alpha_{alpha:.2f}"
        candidates[key] = fuse_relative_poses(
            lingbot_poses, imu_rel_t, imu_rel_q, alpha, blend_translation=True
        )

    ground_truth = load_ground_truth(args.kitti_root, args.sequence, frame_indices)
    results: dict[str, dict] = {}
    aligned_for_plot: dict[str, np.ndarray] = {}
    save_arrays: dict[str, np.ndarray] = {
        "frame_indices": frame_indices,
        "timestamps": timestamps,
        "ground_truth_c2w": ground_truth,
    }
    for name, trajectory in candidates.items():
        results[name], aligned_for_plot[name] = evaluate(trajectory, ground_truth)
        save_arrays[name] = trajectory
    np.savez_compressed(args.output_dir / "trajectories.npz", **save_arrays)
    save_plot(
        args.output_dir / "trajectory_sensitivity.png",
        {
            name: trajectory
            for name, trajectory in aligned_for_plot.items()
            if name in ("lingbot", "current_simple_rgb_imu", "pose_alpha_0.25", "pose_alpha_0.50")
        },
        ground_truth[:, :3, 3],
    )
    metadata = {
        "dataset": "KITTI Odometry",
        "sequence": args.sequence,
        "num_frames": len(frame_indices),
        "protocol": "all candidate trajectories built without GT; GT loaded only for final alignment and metrics",
        "integration_type": "post-hoc zero-shot relative-pose interpolation",
        "claim_boundary": (
            "This is not the trained end-to-end LingBot+AirIMU fusion model. "
            "The IMU checkpoint was trained with SimpleVisualEncoder tokens."
        ),
        "lingbot_trajectory": str(args.lingbot_trajectory),
        "imu_pair_cache": str(args.imu_pair_cache),
        "alphas": list(args.alphas),
        "alpha_selection": "predeclared sensitivity sweep; no test-set best alpha is a valid final model",
        "translation_fusion": "IMU direction normalized to LingBot interval magnitude before interpolation",
        "results": results,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    summary = {
        name: {
            "se3_ate_rmse_m": value["ate_se3_aligned"]["rmse_m"],
            "sim3_ate_rmse_m": value["ate_sim3_aligned"]["rmse_m"],
            "sim3_scale": value["sim3_scale"],
        }
        for name, value in results.items()
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
