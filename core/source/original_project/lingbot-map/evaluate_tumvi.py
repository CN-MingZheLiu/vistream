#!/usr/bin/env python3
"""Evaluate LingBot-Map on TUM-VI without using ground truth during inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_kitti_odometry import (
    aligned_poses,
    build_model,
    choose_dtype,
    infer,
    position_metrics,
    predictions_to_poses,
    sha256,
    umeyama,
)
from lingbot_map.utils.fisheye import (
    load_equidistant_calibration,
    rectify_equidistant_images,
)
from lingbot_map.utils.load_fn import load_and_preprocess_images


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path("/data2/mingzheliu/datasets/TUM_VI")
DEFAULT_MODEL = REPO_ROOT / "checkpoints" / "lingbot-map-long.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--sequence", default="dataset-room2_512_16")
    parser.add_argument("--camera", choices=("cam0", "cam1"), default="cam0")
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--keyframe-interval", type=int, default=None)
    parser.add_argument("--kv-cache-sliding-window", type=int, default=64)
    parser.add_argument("--max-frame-num", type=int, default=1024)
    parser.add_argument("--camera-num-iterations", type=int, default=4)
    parser.add_argument(
        "--inference-mode", choices=("streaming", "windowed"), default="streaming"
    )
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--overlap-keyframes", type=int, default=8)
    parser.add_argument(
        "--window-scale-mode",
        choices=("median", "depth", "centers", "hybrid"),
        default="hybrid",
    )
    parser.add_argument(
        "--preprocess", choices=("raw", "rectified"), default="raw"
    )
    parser.add_argument(
        "--rectified-fov-deg",
        type=float,
        default=90.0,
        help="Horizontal and vertical field of view of the rectified pinhole image.",
    )
    parser.add_argument(
        "--gt-center",
        choices=("imu", "camera"),
        default="imu",
        help="Use IMU/mocap positions for legacy comparisons or calibrated camera centers.",
    )
    parser.add_argument("--gt-match-threshold-ms", type=float, default=10.0)
    parser.add_argument("--use-sdpa", action="store_true")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "tumvi_lingbot_map",
    )
    return parser.parse_args()


def load_camera_rows(
    sequence_root: Path,
    camera: str,
    frame_stride: int,
    max_frames: int | None,
) -> tuple[list[Path], np.ndarray, np.ndarray]:
    if frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")
    csv_path = sequence_root / "mav0" / camera / "data.csv"
    image_root = csv_path.parent / "data"
    timestamps_ns: list[int] = []
    paths: list[Path] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(line for line in handle if not line.startswith("#"))
        for row in rows:
            if not row:
                continue
            timestamps_ns.append(int(row[0]))
            paths.append(image_root / row[1].strip())

    selected = np.arange(0, len(paths), frame_stride, dtype=np.int64)
    if max_frames is not None:
        if max_frames < 2:
            raise ValueError("--max-frames must be at least 2")
        selected = selected[:max_frames]
    return (
        [paths[index] for index in selected],
        np.asarray(timestamps_ns, dtype=np.int64)[selected],
        selected,
    )


def quaternion_wxyz_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)
    w, x, y, z = quaternion.T
    result = np.empty((len(quaternion), 3, 3), dtype=np.float64)
    result[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    result[:, 0, 1] = 2.0 * (x * y - z * w)
    result[:, 0, 2] = 2.0 * (x * z + y * w)
    result[:, 1, 0] = 2.0 * (x * y + z * w)
    result[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    result[:, 1, 2] = 2.0 * (y * z - x * w)
    result[:, 2, 0] = 2.0 * (x * z - y * w)
    result[:, 2, 1] = 2.0 * (y * z + x * w)
    result[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return result


def load_ground_truth(
    sequence_root: Path,
    camera: str,
    timestamps_ns: np.ndarray,
    match_threshold_ms: float,
    center: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mocap = np.loadtxt(
        sequence_root / "mav0" / "mocap0" / "data.csv",
        delimiter=",",
        comments="#",
        dtype=np.float64,
    )
    mocap_timestamps = mocap[:, 0].astype(np.int64)
    insertions = np.searchsorted(mocap_timestamps, timestamps_ns)
    right = np.clip(insertions, 0, len(mocap_timestamps) - 1)
    left = np.clip(insertions - 1, 0, len(mocap_timestamps) - 1)
    use_left = (
        np.abs(timestamps_ns - mocap_timestamps[left])
        <= np.abs(mocap_timestamps[right] - timestamps_ns)
    )
    matched = np.where(use_left, left, right)
    offsets_ns = np.abs(mocap_timestamps[matched] - timestamps_ns)
    valid = offsets_ns <= match_threshold_ms * 1e6

    positions = mocap[matched, 1:4].copy()
    if center == "camera":
        calibration = load_equidistant_calibration(
            sequence_root / "dso" / "camchain.yaml", camera
        )
        rotation_room_imu = quaternion_wxyz_to_rotation(mocap[matched, 4:8])
        transform_camera_imu = np.asarray(calibration["T_cam_imu"], dtype=np.float64)
        center_imu = (
            -transform_camera_imu[:3, :3].T @ transform_camera_imu[:3, 3]
        )
        positions += np.einsum("nij,j->ni", rotation_room_imu, center_imu)
    return positions, valid, offsets_ns.astype(np.float64) / 1e6


def main() -> None:
    args = parse_args()
    sequence_root = args.dataset_root / args.sequence
    paths, timestamps_ns, raw_frame_indices = load_camera_rows(
        sequence_root, args.camera, args.frame_stride, args.max_frames
    )
    keyframe_interval = args.keyframe_interval
    if keyframe_interval is None:
        keyframe_interval = max(1, math.ceil(len(paths) / 320))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"{args.sequence}/{args.camera}: {len(paths)} frames, "
        f"preprocess={args.preprocess}, mode={args.inference_mode}, "
        f"keyframe interval={keyframe_interval}",
        flush=True,
    )

    load_started = time.perf_counter()
    calibration = load_equidistant_calibration(
        sequence_root / "dso" / "camchain.yaml", args.camera
    )
    if args.preprocess == "rectified":
        images = rectify_equidistant_images(
            paths, calibration, args.image_size, args.rectified_fov_deg
        )
    else:
        images = load_and_preprocess_images(
            [str(path) for path in paths],
            mode="crop",
            image_size=args.image_size,
            patch_size=args.patch_size,
        )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    dtype = choose_dtype(args.dtype, device)
    model = build_model(args, device, dtype)
    load_seconds = time.perf_counter() - load_started

    predictions, inference_seconds, peak_gpu_gib = infer(
        model, images, args, device, dtype, keyframe_interval
    )
    predicted_c2w, predicted_intrinsics = predictions_to_poses(
        predictions["pose_enc"], tuple(images.shape[-2:]), "opencv_w2c"
    )
    if len(predicted_c2w) != len(paths):
        raise RuntimeError(
            f"Inference returned {len(predicted_c2w)} poses for {len(paths)} images"
        )

    # Leakage boundary: mocap is first opened after model inference has finished.
    gt_positions_all, valid, offsets_ms = load_ground_truth(
        sequence_root,
        args.camera,
        timestamps_ns,
        args.gt_match_threshold_ms,
        args.gt_center,
    )
    metric_indices = np.flatnonzero(valid)
    predicted_positions = predicted_c2w[metric_indices, :3, 3]
    gt_positions = gt_positions_all[metric_indices]
    se3_positions, se3_rotation, _, se3_translation = umeyama(
        predicted_positions, gt_positions, with_scale=False
    )
    sim3_positions, sim3_rotation, sim3_scale, sim3_translation = umeyama(
        predicted_positions, gt_positions, with_scale=True
    )
    se3_c2w = aligned_poses(
        predicted_c2w[metric_indices], se3_rotation, 1.0, se3_translation
    )
    sim3_c2w = aligned_poses(
        predicted_c2w[metric_indices],
        sim3_rotation,
        sim3_scale,
        sim3_translation,
    )

    chunk_scales = predictions.get("chunk_scales")
    result = {
        "dataset": "TUM-VI",
        "sequence": args.sequence,
        "camera": args.camera,
        "checkpoint": str(args.model_path),
        "checkpoint_sha256": sha256(args.model_path),
        "num_input_frames": len(paths),
        "num_metric_frames": len(metric_indices),
        "frame_stride": args.frame_stride,
        "input_frame_indices": raw_frame_indices.tolist(),
        "image_shape_chw": list(images.shape[1:]),
        "preprocess": args.preprocess,
        "rectified_fov_degrees": (
            args.rectified_fov_deg if args.preprocess == "rectified" else None
        ),
        "gt_center": args.gt_center,
        "gt_match_threshold_ms": args.gt_match_threshold_ms,
        "gt_max_matched_offset_ms": float(offsets_ms[valid].max()),
        "inference_mode": args.inference_mode,
        "num_scale_frames": args.num_scale_frames,
        "keyframe_interval": keyframe_interval,
        "kv_cache_sliding_window": args.kv_cache_sliding_window,
        "window_size_keyframes": (
            args.window_size if args.inference_mode == "windowed" else None
        ),
        "overlap_keyframes": (
            args.overlap_keyframes if args.inference_mode == "windowed" else None
        ),
        "window_scale_mode": (
            args.window_scale_mode if args.inference_mode == "windowed" else None
        ),
        "window_alignment_scales": (
            chunk_scales.float().cpu().numpy().tolist()
            if chunk_scales is not None
            else None
        ),
        "dtype": str(dtype).removeprefix("torch."),
        "attention_backend": "SDPA" if args.use_sdpa else "FlashInfer",
        "device": str(device),
        "timing": {
            "preprocess_and_model_load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "fps_excluding_load": len(paths) / inference_seconds,
            "peak_gpu_memory_gib": peak_gpu_gib,
        },
        "ate_se3_aligned": position_metrics(se3_positions, gt_positions),
        "ate_sim3_aligned": position_metrics(sim3_positions, gt_positions),
        "sim3_scale": sim3_scale,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    np.savez_compressed(
        args.output_dir / "trajectory.npz",
        frame_indices=raw_frame_indices[metric_indices],
        timestamps_ns=timestamps_ns[metric_indices],
        predicted_c2w=predicted_c2w[metric_indices],
        ground_truth_positions=gt_positions,
        se3_c2w=se3_c2w,
        sim3_c2w=sim3_c2w,
        predicted_intrinsics=predicted_intrinsics[metric_indices],
    )
    print("@@RESULT@@" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
