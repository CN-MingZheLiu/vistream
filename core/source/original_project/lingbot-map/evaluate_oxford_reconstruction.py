#!/usr/bin/env python3
"""Run frozen LingBot depth and evaluate Oxford reconstruction benchmark clouds.

The script performs inference before reading Oxford reconstruction GT.  It uses
the sealed loop-off candidate poses for mapping, then applies the trajectory
Sim(3) alignment to the predicted cloud and compares it with the official GT
cloud.  GT PCD files are read with Open3D after the inference boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LINGBOT_ROOT = Path("/home/mingzheliu/MZL_workplace/lingbot-map")
DATASET_ROOT = Path("/data2/huangshuaixin/datasets/Oxford_Spires")
CANDIDATE_ROOT = Path("/data2/mingzheliu/loop_closure_off_ablation_20260818/candidates_strict_v2/oxford")
VISUAL_ROOT = LINGBOT_ROOT / "outputs/oxford_all_current_long_3840"
SEQUENCES = {
    "keble-college-04": "2024-03-12-keble-college-04",
    "observatory-quarter-01": "2024-03-13-observatory-quarter-01",
    "blenheim-palace-05": "2024-03-14-blenheim-palace-05",
    "christ-church-02": "2024-03-18-christ-church-02",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", choices=sorted(SEQUENCES), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--image-size", type=int, default=518)
    p.add_argument("--patch-size", type=int, default=14)
    p.add_argument("--num-scale-frames", type=int, default=8)
    p.add_argument("--keyframe-interval", type=int, default=12)
    p.add_argument("--kv-cache-sliding-window", type=int, default=64)
    p.add_argument("--max-frame-num", type=int, default=1024)
    p.add_argument("--camera-num-iterations", type=int, default=4)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--pixel-stride", type=int, default=4)
    p.add_argument("--voxel-size", type=float, default=0.10)
    p.add_argument("--confidence-threshold", type=float, default=1.5)
    p.add_argument("--min-depth", type=float, default=0.1)
    p.add_argument("--max-depth", type=float, default=150.0)
    p.add_argument("--gt-voxel-size", type=float, default=0.10)
    p.add_argument("--thresholds", default="0.05,0.10")
    p.add_argument("--smoke-frames", type=int, default=None)
    p.add_argument("--use-sdpa", action="store_true", help="Use PyTorch SDPA instead of FlashInfer (no ninja build required)")
    return p.parse_args()


def ensure_import_path() -> None:
    sys.path.insert(0, str(LINGBOT_ROOT))


def load_images_and_metadata(sequence: str, max_frames: int | None):
    path = VISUAL_ROOT / sequence / "trajectory.npz"
    if sequence == "blenheim-palace-05":
        # The benchmark package contains 365 rectified frames for Blenheim;
        # the earlier native-fusion run evaluated the 354 frames overlapping
        # the IMU stream and records their source indices explicitly.
        source = LINGBOT_ROOT / "outputs/oxford_spires_cache" / SEQUENCES[sequence] / "cam0_undistorted_official_centered_principal_point_v1"
        lingbot = LINGBOT_ROOT.parent / "IMU-lingbot/output/eval_oxford_spires_2024-03-14-blenheim-palace-05_fresh_official_c2w_centered_20260805/lingbot_trajectory.npz"
        evaluated = LINGBOT_ROOT.parent / "IMU-lingbot/output/eval_oxford_spires_2024-03-14-blenheim-palace-05_fresh_official_c2w_centered_20260805/evaluated_trajectories.npz"
        if lingbot.is_file() and evaluated.is_file():
            with np.load(lingbot, allow_pickle=False) as z:
                all_paths = [Path(str(x)) for x in z["image_paths"]]
                all_timestamps = np.asarray(z["timestamps"], dtype=np.float64)
            with np.load(evaluated, allow_pickle=False) as z:
                indices = np.asarray(z["frame_indices"], dtype=np.int64)
            image_paths = [all_paths[int(i)] for i in indices]
            timestamps = all_timestamps[indices]
            if max_frames is not None:
                image_paths, timestamps = image_paths[:max_frames], timestamps[:max_frames]
            return image_paths, timestamps
        image_paths = sorted(source.glob("*.jpg"), key=lambda p: float(p.stem))
        timestamps = np.asarray([float(p.stem) for p in image_paths], dtype=np.float64)
        if max_frames is not None:
            image_paths, timestamps = image_paths[:max_frames], timestamps[:max_frames]
        return image_paths, timestamps
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            image_paths = [Path(str(x)) for x in z["image_paths"]]
            # Some cached trajectories keep only the GT-overlap timestamps in
            # ``timestamps`` while image_paths still contains all 3840 inputs.
            timestamps = np.asarray(z.get("all_timestamps", z["timestamps"]), dtype=np.float64)
        # Strict loop-off candidates may intentionally start after the first
        # one or two frames (their ``input_frame_indices`` record the exact
        # source image indices).  Match the inference stream to that pose
        # stream instead of truncating an unaligned 3840-frame cache.
        candidate_path = CANDIDATE_ROOT / sequence / "candidate_before_gt.npz"
        if candidate_path.is_file():
            with np.load(candidate_path, allow_pickle=False) as candidate_z:
                source_indices = np.asarray(candidate_z["input_frame_indices"], dtype=np.int64)
            if len(source_indices) != len(image_paths) and len(source_indices) <= len(image_paths):
                image_paths = [image_paths[int(i)] for i in source_indices]
                timestamps = timestamps[source_indices]
    else:
        # Blenheim has the same full rectified stream in the general Oxford
        # cache, but no entry in the ten-sequence trajectory cache.
        full_name = SEQUENCES[sequence]
        directory = LINGBOT_ROOT / "outputs/oxford_spires_cache" / full_name / "cam0_undistorted_official_centered_principal_point_v1"
        image_paths = sorted(directory.glob("*.jpg"), key=lambda p: float(p.stem))
        if not image_paths:
            raise FileNotFoundError(f"no Oxford image cache for {sequence}: {directory}")
        candidate_path = CANDIDATE_ROOT / sequence / "candidate_before_gt.npz"
        with np.load(candidate_path, allow_pickle=False) as z:
            target_ns = np.asarray(z["timestamps_ns"], dtype=np.int64)
        source_ts = np.asarray([float(p.stem) for p in image_paths], dtype=np.float64)
        target_ts = target_ns.astype(np.float64) * 1e-9
        nearest = np.searchsorted(source_ts, target_ts, side="left")
        nearest = np.clip(nearest, 0, len(source_ts) - 1)
        previous = np.maximum(nearest - 1, 0)
        choose_previous = np.abs(source_ts[previous] - target_ts) < np.abs(source_ts[nearest] - target_ts)
        nearest[choose_previous] = previous[choose_previous]
        image_paths = [image_paths[int(i)] for i in nearest]
        timestamps = target_ts
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
        timestamps = timestamps[:max_frames]
    if len(image_paths) != len(timestamps):
        raise ValueError("image/timestamp count mismatch")
    missing = next((p for p in image_paths if not p.is_file()), None)
    if missing is not None:
        raise FileNotFoundError(missing)
    return image_paths, timestamps


def trajectory_alignment(pred: np.ndarray, gt: np.ndarray):
    """Umeyama Sim(3) mapping predicted camera centers to GT centers."""
    from evaluate_kitti_odometry import umeyama

    _, rotation, scale, translation = umeyama(
        pred[:, :3, 3], gt[:, :3, 3], with_scale=True
    )
    return float(scale), np.asarray(rotation), np.asarray(translation)


def interpolate_gt(sequence: str, timestamps: np.ndarray) -> np.ndarray:
    """Interpolate Oxford GT base poses and convert them to cam0 C2W."""
    from scipy.spatial.transform import Rotation, Slerp
    import yaml

    seq_root = DATASET_ROOT / "sequences" / SEQUENCES[sequence]
    raw = np.loadtxt(seq_root / "processed/trajectory/gt-tum.txt", dtype=np.float64)
    raw = raw[np.argsort(raw[:, 0])]
    keep = np.r_[True, np.diff(raw[:, 0]) > 0]
    raw = raw[keep]
    clipped = np.clip(timestamps, raw[0, 0], raw[-1, 0])
    xyz = np.column_stack([np.interp(clipped, raw[:, 0], raw[:, axis]) for axis in (1, 2, 3)])
    rot = Slerp(raw[:, 0], Rotation.from_quat(raw[:, 4:8]))(clipped).as_matrix()
    world_base = np.repeat(np.eye(4, dtype=np.float64)[None], len(timestamps), axis=0)
    world_base[:, :3, :3] = rot
    world_base[:, :3, 3] = xyz
    with (DATASET_ROOT / "calibration/cam-lidar-imu.yaml").open(encoding="utf-8") as f:
        ext = yaml.safe_load(f)
    t_cam_lidar = np.asarray(ext["cam0"]["T_cam_lidar"], dtype=np.float64)
    t_base_lidar = np.eye(4, dtype=np.float64)
    t_base_lidar[:3, :3] = Rotation.from_quat([0.0, 0.0, 1.0, 0.0]).as_matrix()
    t_base_lidar[:3, 3] = [0.0, 0.0, 0.124]
    t_base_cam = t_base_lidar @ np.linalg.inv(t_cam_lidar)
    return world_base @ t_base_cam


def voxel_insert(table: dict[tuple[int, int, int], np.ndarray], points: np.ndarray, size: float) -> None:
    if not len(points):
        return
    keys = np.floor(points / size).astype(np.int64)
    for key, point in zip(keys, points):
        table.setdefault((int(key[0]), int(key[1]), int(key[2])), point.astype(np.float32))


def write_ascii_ply(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(f, points, fmt="%.6f")


def load_cloud(path: Path, voxel_size: float) -> np.ndarray:
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if not len(points):
        raise ValueError(f"empty GT cloud: {path}")
    if voxel_size > 0:
        points = np.asarray(
            cloud.voxel_down_sample(voxel_size).points, dtype=np.float64
        )
    return points


def cloud_metrics(pred: np.ndarray, gt: np.ndarray, thresholds: list[float]) -> dict[str, float]:
    from scipy.spatial import cKDTree

    pred_tree = cKDTree(pred)
    gt_tree = cKDTree(gt)
    pred_to_gt = gt_tree.query(pred, k=1)[0]
    gt_to_pred = pred_tree.query(gt, k=1)[0]
    out = {
        "chamfer_l1_m": float(np.mean(pred_to_gt) + np.mean(gt_to_pred)),
        "accuracy_mean_m": float(np.mean(pred_to_gt)),
        "completeness_mean_m": float(np.mean(gt_to_pred)),
        "pred_to_gt_p95_m": float(np.percentile(pred_to_gt, 95)),
        "gt_to_pred_p95_m": float(np.percentile(gt_to_pred, 95)),
    }
    for threshold in thresholds:
        acc = float(np.mean(pred_to_gt <= threshold))
        comp = float(np.mean(gt_to_pred <= threshold))
        f1 = 2 * acc * comp / (acc + comp) if acc + comp else 0.0
        tag = f"{int(round(threshold * 100)):02d}cm"
        out[f"accuracy_{tag}"] = acc
        out[f"completeness_{tag}"] = comp
        out[f"f1_{tag}"] = f1
    return out


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1 or args.pixel_stride < 1:
        raise ValueError("strides must be positive")
    if args.smoke_frames is not None and args.smoke_frames < 2:
        raise ValueError("--smoke-frames must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_import_path()
    from evaluate_kitti_odometry import build_model, choose_dtype, infer, predictions_to_poses
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    max_frames = args.smoke_frames
    image_paths, timestamps = load_images_and_metadata(args.sequence, max_frames)
    images = load_and_preprocess_images(
        [str(x) for x in image_paths], mode="crop", image_size=args.image_size, patch_size=args.patch_size
    )
    device = __import__("torch").device(args.device)
    torch = __import__("torch")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    dtype = choose_dtype("bfloat16", device)
    model_args = SimpleNamespace(
        inference_mode="streaming", image_size=args.image_size, patch_size=args.patch_size,
        max_frame_num=args.max_frame_num, kv_cache_sliding_window=args.kv_cache_sliding_window,
        num_scale_frames=args.num_scale_frames, use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations, model_path=LINGBOT_ROOT / "checkpoints/lingbot-map-long.pt",
        window_size=64, overlap_keyframes=8, window_scale_mode="hybrid",
    )
    model = build_model(model_args, device, dtype)
    print(json.dumps({"sequence": args.sequence, "frames": len(image_paths), "dtype": str(dtype), "device": str(device)}), flush=True)
    predictions, inference_seconds, peak_gpu_gib = infer(
        model, images, model_args, device, dtype, args.keyframe_interval
    )
    pred_c2w, intrinsics = predictions_to_poses(predictions["pose_enc"], tuple(images.shape[-2:]), "opencv_w2c")

    # All model outputs are now materialized.  The candidate and GT are loaded
    # only after inference, preserving the same leakage boundary as the pose run.
    candidate_path = CANDIDATE_ROOT / args.sequence / "candidate_before_gt.npz"
    candidate_key = "c2w"
    if candidate_path.is_file():
        with np.load(candidate_path, allow_pickle=False) as z:
            candidate = np.asarray(z[candidate_key], dtype=np.float64)
    elif args.sequence == "blenheim-palace-05":
        candidate_path = LINGBOT_ROOT.parent / "IMU-lingbot/output/eval_oxford_spires_2024-03-14-blenheim-palace-05_fresh_official_c2w_centered_20260805/evaluated_trajectories.npz"
        with np.load(candidate_path, allow_pickle=False) as z:
            candidate = np.asarray(z["airimu_blend_0p25_global_gain_1"], dtype=np.float64)
        candidate_key = "airimu_blend_0p25_global_gain_1"
    candidate = candidate[: len(image_paths)]
    visual_path = VISUAL_ROOT / args.sequence / "trajectory.npz"
    gt_from_cache = None
    cache_timestamps = timestamps
    if visual_path.is_file():
        with np.load(visual_path, allow_pickle=False) as z:
            gt_from_cache = np.asarray(z["ground_truth_c2w"], dtype=np.float64) if "ground_truth_c2w" in z.files else None
            cache_timestamps = np.asarray(z.get("all_timestamps", z["timestamps"]), dtype=np.float64)
    gt = interpolate_gt(args.sequence, timestamps)
    if len(candidate) != len(gt):
        raise ValueError(f"candidate/GT mismatch: {len(candidate)} vs {len(gt)}")
    scale, rotation, translation = trajectory_alignment(candidate, gt)
    aligned_candidate = candidate.copy()
    aligned_candidate[:, :3, :3] = rotation @ candidate[:, :3, :3]
    aligned_candidate[:, :3, 3] = scale * (candidate[:, :3, 3] @ rotation.T) + translation
    aligned_pred_visual = pred_c2w.copy()
    aligned_pred_visual[:, :3, :3] = rotation @ pred_c2w[:, :3, :3]
    aligned_pred_visual[:, :3, 3] = scale * (pred_c2w[:, :3, 3] @ rotation.T) + translation

    depth = predictions["depth"]
    conf = predictions.get("depth_conf")
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    if hasattr(conf, "detach"):
        conf = conf.detach().cpu().numpy()
    depth = np.asarray(depth)
    conf = np.ones_like(depth) if conf is None else np.asarray(conf)
    if depth.ndim == 5:
        depth = depth[0]
    if depth.shape[-1] == 1:
        depth = depth[..., 0]
    if conf.ndim == 4 and conf.shape[0] == 1:
        conf = conf[0]
    if len(depth) != len(candidate):
        raise ValueError(f"depth/candidate mismatch: {len(depth)} vs {len(candidate)}")
    height, width = depth.shape[1:]
    yy, xx = np.meshgrid(
        np.arange(0, height, args.pixel_stride, dtype=np.float64),
        np.arange(0, width, args.pixel_stride, dtype=np.float64), indexing="ij"
    )
    table: dict[tuple[int, int, int], np.ndarray] = {}
    candidates = kept = 0
    for frame in range(0, len(candidate), args.frame_stride):
        d = depth[frame, :: args.pixel_stride, :: args.pixel_stride].astype(np.float64)
        c = conf[frame, :: args.pixel_stride, :: args.pixel_stride].astype(np.float64)
        K = intrinsics[frame]
        cam = np.stack(((xx - K[0, 2]) * d / K[0, 0], (yy - K[1, 2]) * d / K[1, 1], d), axis=-1).reshape(-1, 3)
        cf = c.reshape(-1)
        valid = np.isfinite(cam).all(axis=1) & np.isfinite(cf) & (cf >= args.confidence_threshold)
        valid &= (cam[:, 2] >= args.min_depth) & (cam[:, 2] <= args.max_depth)
        candidates += len(cam)
        kept += int(valid.sum())
        world = cam[valid] @ candidate[frame, :3, :3].T + candidate[frame, :3, 3]
        voxel_insert(table, world, args.voxel_size)
    points = np.stack(list(table.values()), axis=0) if table else np.empty((0, 3), dtype=np.float32)
    aligned_points = scale * (points @ rotation.T) + translation
    write_ascii_ply(args.output_dir / "predicted_cloud_model_frame.ply", points)
    write_ascii_ply(args.output_dir / "predicted_cloud_gt_frame.ply", aligned_points)

    # GT is loaded only after the predicted cloud has been written.
    gt_path = DATASET_ROOT / "reconstruction_benchmark" / SEQUENCES[args.sequence] / "gt_cloud_merged.pcd"
    gt_points = load_cloud(gt_path, args.gt_voxel_size)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    metrics = cloud_metrics(aligned_points, gt_points, thresholds)
    metrics.update({"sequence": SEQUENCES[args.sequence], "protocol_sequence": args.sequence,
                    "frames": len(image_paths), "frame_stride": args.frame_stride,
                    "pixel_stride": args.pixel_stride, "voxel_size_m": args.voxel_size,
                    "gt_voxel_size_m": args.gt_voxel_size, "confidence_threshold": args.confidence_threshold,
                    "candidate_pixels": candidates, "kept_pixels": kept,
                    "pred_points": int(len(aligned_points)), "gt_points": int(len(gt_points)),
                    "trajectory_sim3_scale": scale, "inference_seconds": inference_seconds,
                    "inference_fps": len(image_paths) / max(inference_seconds, 1e-9),
                    "peak_gpu_gib": peak_gpu_gib,
                    "checkpoint": str(model_args.model_path),
                    "candidate": str(candidate_path), "candidate_pose_key": candidate_key,
                    "gt_cloud": str(gt_path),
                    "alignment_source": "candidate c2w to interpolated Oxford gt-tum camera centers"})
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
