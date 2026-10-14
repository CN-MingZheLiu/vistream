#!/usr/bin/env python3
"""Evaluate frozen LingBot-Map on the ten Oxford Spires 3840-frame streams.

The image paths come from the existing LingBot evaluation caches.  They are
the exact 3840-frame subsets used by the earlier Oxford protocol, including
the official centered fisheye rectification.  Ground truth is read only after
the complete visual inference has finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from evaluate_kitti_odometry import (
    aligned_poses,
    build_model,
    choose_dtype,
    infer,
    position_metrics,
    predictions_to_poses,
    relative_pose_metrics,
    sha256,
    umeyama,
)
from lingbot_map.utils.load_fn import load_and_preprocess_images


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path("/data2/huangshuaixin/datasets/Oxford_Spires")
REFERENCE_RUNS = Path("/home/mingzheliu/PLZ/runs")
IMAGE_CACHE = Path("/data2/mingzheliu/PLZ-runs/oxford_plz_ten_image_cache")
SEQUENCES = {
    "keble-college-02": "2024-03-12-keble-college-02",
    "keble-college-03": "2024-03-12-keble-college-03",
    "keble-college-04": "2024-03-12-keble-college-04",
    "keble-college-05": "2024-03-12-keble-college-05",
    "observatory-quarter-01": "2024-03-13-observatory-quarter-01",
    "observatory-quarter-02": "2024-03-13-observatory-quarter-02",
    "christ-church-02": "2024-03-18-christ-church-02",
    "christ-church-03": "2024-03-18-christ-church-03",
    "christ-church-05": "2024-03-20-christ-church-05",
    "bodleian-library-02": "2024-05-20-bodleian-library-02",
}
REFERENCE_ALIASES = {
    "keble-college-02": "keble02",
    "keble-college-03": "keble03",
    "keble-college-04": "keble04",
    "keble-college-05": "keble05",
    "observatory-quarter-01": "observatory01",
    "observatory-quarter-02": "observatory02",
    "christ-church-02": "christ02",
    "christ-church-03": "christ03",
    "christ-church-05": "christ05",
    "bodleian-library-02": "bodleian02",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, choices=sorted(SEQUENCES))
    parser.add_argument(
        "--model-path", type=Path, default=ROOT / "checkpoints/lingbot-map-long.pt"
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--reference-runs", type=Path, default=REFERENCE_RUNS)
    parser.add_argument("--image-cache", type=Path, default=IMAGE_CACHE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--keyframe-interval", type=int, default=None)
    parser.add_argument("--kv-cache-sliding-window", type=int, default=64)
    parser.add_argument("--max-frame-num", type=int, default=1024)
    parser.add_argument("--camera-num-iterations", type=int, default=4)
    parser.add_argument("--inference-mode", choices=("streaming", "windowed"), default="streaming")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--overlap-keyframes", type=int, default=8)
    parser.add_argument("--window-scale-mode", choices=("median", "depth", "centers", "hybrid"), default="hybrid")
    parser.add_argument("--use-sdpa", action="store_true")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-frames", type=int, default=None, help="Debug smoke-test limit; full protocol uses 3840.")
    return parser.parse_args()


def load_calibration(dataset_root: Path) -> np.ndarray:
    """Return the fixed Oxford base-to-cam0 transform."""
    import yaml

    with (dataset_root / "calibration/cam-lidar-imu.yaml").open(encoding="utf-8") as handle:
        extrinsics = yaml.safe_load(handle)
    t_cam_lidar = np.asarray(extrinsics["cam0"]["T_cam_lidar"], dtype=np.float64)
    t_base_lidar = np.eye(4, dtype=np.float64)
    t_base_lidar[:3, :3] = Rotation.from_quat([0.0, 0.0, 1.0, 0.0]).as_matrix()
    t_base_lidar[:3, 3] = [0.0, 0.0, 0.124]
    return t_base_lidar @ np.linalg.inv(t_cam_lidar)


def load_images(sequence: str, args: argparse.Namespace) -> tuple[list[Path], np.ndarray, str]:
    alias = REFERENCE_ALIASES[sequence]
    reference = args.reference_runs / f"oxford_plz_ten_{alias}/lingbot_trajectory.npz"
    paths: list[Path]
    source = str(reference)
    if reference.is_file():
        with np.load(reference, allow_pickle=False) as cached:
            paths = [Path(str(path)) for path in cached["image_paths"]]
            timestamps = np.asarray(cached["timestamps"], dtype=np.float64)
    else:
        directory = args.image_cache / SEQUENCES[sequence] / "cam0_undistorted_official_centered_principal_point_v1"
        paths = sorted(directory.glob("*.jpg"))[:3840]
        timestamps = np.asarray([float(path.stem) for path in paths], dtype=np.float64)
        source = str(directory)
    if args.max_frames is not None:
        if args.max_frames < 2:
            raise ValueError("--max-frames must be at least 2")
        paths, timestamps = paths[: args.max_frames], timestamps[: args.max_frames]
    if len(paths) != len(timestamps):
        raise RuntimeError(f"Image/timestamp mismatch: {len(paths)} vs {len(timestamps)}")
    if len(paths) != 3840 and args.max_frames is None:
        raise RuntimeError(f"Oxford protocol requires exactly 3840 frames, found {len(paths)}")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing cached image {missing[0]}")
    if not np.all(np.diff(timestamps) > 0.0):
        raise RuntimeError("Oxford image timestamps are not strictly increasing")
    return paths, timestamps, source


def interpolate_gt(path: Path, timestamps: np.ndarray, t_base_cam: np.ndarray) -> np.ndarray:
    raw = np.loadtxt(path, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 8:
        raise ValueError(f"Unexpected GT shape {raw.shape} from {path}")
    raw = raw[np.argsort(raw[:, 0])]
    keep = np.r_[True, np.diff(raw[:, 0]) > 0.0]
    raw = raw[keep]
    if timestamps[0] < raw[0, 0] or timestamps[-1] > raw[-1, 0]:
        raise ValueError(
            f"Image timestamps {timestamps[0]:.9f}..{timestamps[-1]:.9f} "
            f"outside GT {raw[0, 0]:.9f}..{raw[-1, 0]:.9f}"
        )
    translation = np.column_stack([
        np.interp(timestamps, raw[:, 0], raw[:, axis]) for axis in (1, 2, 3)
    ])
    rotation = Slerp(raw[:, 0], Rotation.from_quat(raw[:, 4:8]))(timestamps).as_matrix()
    world_base = np.repeat(np.eye(4, dtype=np.float64)[None], len(timestamps), axis=0)
    world_base[:, :3, :3] = rotation
    world_base[:, :3, 3] = translation
    return world_base @ t_base_cam


def evaluate_convention(predicted: np.ndarray, gt: np.ndarray) -> dict[str, object]:
    pred_positions = predicted[:, :3, 3]
    gt_positions = gt[:, :3, 3]
    se3_positions, se3_rotation, _, se3_translation = umeyama(
        pred_positions, gt_positions, with_scale=False
    )
    sim3_positions, sim3_rotation, sim3_scale, sim3_translation = umeyama(
        pred_positions, gt_positions, with_scale=True
    )
    se3 = aligned_poses(predicted, se3_rotation, 1.0, se3_translation)
    sim3 = aligned_poses(predicted, sim3_rotation, sim3_scale, sim3_translation)
    return {
        "ate_raw": position_metrics(pred_positions, gt_positions),
        "ate_se3_aligned": position_metrics(se3_positions, gt_positions),
        "ate_sim3_aligned": position_metrics(sim3_positions, gt_positions),
        "sim3_scale": float(sim3_scale),
        "rpe_se3_aligned": {
            "delta_1": relative_pose_metrics(se3, gt, 1),
            "delta_10": relative_pose_metrics(se3, gt, 10),
        },
        "rpe_sim3_aligned": {
            "delta_1": relative_pose_metrics(sim3, gt, 1),
            "delta_10": relative_pose_metrics(sim3, gt, 10),
        },
        "alignment": {
            "se3_rotation": se3_rotation.tolist(),
            "se3_translation": se3_translation.tolist(),
            "sim3_rotation": sim3_rotation.tolist(),
            "sim3_translation": sim3_translation.tolist(),
        },
        "se3_c2w": se3,
        "sim3_c2w": sim3,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths, timestamps, image_source = load_images(args.sequence, args)
    keyframe_interval = args.keyframe_interval or max(1, math.ceil(len(paths) / 320))
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    print(
        f"Oxford {args.sequence}: {len(paths)} frames, keyframe interval={keyframe_interval}; "
        f"source={image_source}",
        flush=True,
    )
    started = time.perf_counter()
    images = load_and_preprocess_images(
        [str(path) for path in paths], mode="crop", image_size=args.image_size, patch_size=args.patch_size
    )
    dtype = choose_dtype(args.dtype, device)
    model_args = SimpleNamespace(
        inference_mode=args.inference_mode,
        image_size=args.image_size,
        patch_size=args.patch_size,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        num_scale_frames=args.num_scale_frames,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
        model_path=args.model_path,
        window_size=args.window_size,
        overlap_keyframes=args.overlap_keyframes,
        window_scale_mode=args.window_scale_mode,
    )
    model = build_model(model_args, device, dtype)
    load_seconds = time.perf_counter() - started
    predictions, inference_seconds, peak_gpu_gib = infer(
        model, images, model_args, device, dtype, keyframe_interval
    )
    if int(predictions["pose_enc"].shape[1]) != len(paths):
        raise RuntimeError("Model returned a pose count different from the image count")
    print(f"Inference finished in {inference_seconds:.1f}s", flush=True)

    # Persist the complete visual output before touching any ground truth.
    # This also makes a later metric-boundary adjustment independent of GPU
    # inference and preserves the leakage-free evaluation boundary.
    np.savez_compressed(
        args.output_dir / "predictions_before_gt.npz",
        timestamps=timestamps,
        image_paths=np.asarray([str(path) for path in paths]),
        pose_enc=predictions["pose_enc"].float().cpu().numpy(),
    )

    # This is the deliberate leakage boundary: GT is loaded only now.
    t_base_cam = load_calibration(args.dataset_root)
    gt_path = args.dataset_root / "sequences" / SEQUENCES[args.sequence] / "processed/trajectory/gt-tum.txt"
    raw_gt = np.loadtxt(gt_path, dtype=np.float64)
    raw_gt = raw_gt[np.argsort(raw_gt[:, 0])]
    valid = (timestamps >= raw_gt[0, 0]) & (timestamps <= raw_gt[-1, 0])
    if int(valid.sum()) < 2:
        raise ValueError("Fewer than two Oxford frames overlap the GT trajectory")
    metric_indices = np.flatnonzero(valid)
    gt = interpolate_gt(gt_path, timestamps[metric_indices], t_base_cam)
    decoded: dict[str, np.ndarray] = {}
    results: dict[str, dict[str, object]] = {}
    for convention in ("benchmark_c2w", "opencv_w2c"):
        predicted_all, intrinsics_all = predictions_to_poses(
            predictions["pose_enc"], tuple(images.shape[-2:]), convention
        )
        predicted = predicted_all[metric_indices]
        intrinsics = intrinsics_all[metric_indices]
        evaluated = evaluate_convention(predicted, gt)
        results[convention] = {key: value for key, value in evaluated.items() if not key.endswith("_c2w")}
        decoded[convention] = predicted_all
        if convention == "benchmark_c2w":
            np.savez_compressed(
                args.output_dir / "trajectory.npz",
                timestamps=timestamps[metric_indices],
                all_timestamps=timestamps,
                metric_indices=metric_indices,
                image_paths=np.asarray([str(path) for path in paths]),
                predicted_c2w=predicted,
                predicted_c2w_all=predicted_all,
                ground_truth_c2w=gt,
                se3_c2w=evaluated["se3_c2w"],
                sim3_c2w=evaluated["sim3_c2w"],
                intrinsics=intrinsics,
            )

    metrics = {
        "dataset": "Oxford Spires",
        "sequence": SEQUENCES[args.sequence],
        "protocol_sequence_name": args.sequence,
        "frames": len(paths),
        "frames_evaluated": int(len(metric_indices)),
        "dropped_image_indices_without_gt": np.flatnonzero(~valid).tolist(),
        "duration_s": float(timestamps[-1] - timestamps[0]),
        "image_source": image_source,
        "rectification": "official centered principal point v1 (cached)",
        "model_training": "none; frozen LingBot-Map checkpoint",
        "model": "LingBot-Map GCTStream visual-only",
        "checkpoint": str(args.model_path),
        "checkpoint_sha256": sha256(args.model_path),
        "inference_mode": args.inference_mode,
        "keyframe_interval": keyframe_interval,
        "num_scale_frames": args.num_scale_frames,
        "dtype": str(dtype).removeprefix("torch."),
        "device": str(device),
        "inference_backend": "SDPA" if args.use_sdpa else "FlashInfer",
        "timing": {
            "preprocess_model_load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "fps": len(paths) / max(inference_seconds, 1e-9),
            "peak_gpu_memory_gib": peak_gpu_gib,
        },
        "ground_truth_frame": "Oxford dataset base C2W converted to cam0 C2W",
        "pose_convention_results": results,
        "primary_pose_convention": "benchmark_c2w",
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "sequence": args.sequence,
        "frames": len(paths),
        "benchmark_c2w_se3_ate_m": results["benchmark_c2w"]["ate_se3_aligned"]["rmse_m"],
        "benchmark_c2w_sim3_ate_m": results["benchmark_c2w"]["ate_sim3_aligned"]["rmse_m"],
        "opencv_w2c_se3_ate_m": results["opencv_w2c"]["ate_se3_aligned"]["rmse_m"],
        "opencv_w2c_sim3_ate_m": results["opencv_w2c"]["ate_sim3_aligned"]["rmse_m"],
        "inference_seconds": inference_seconds,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
