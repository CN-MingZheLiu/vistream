#!/usr/bin/env python3
"""Create a frozen LingBot-Map camera-trajectory cache for one EuRoC sequence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from evaluate_kitti_odometry import (
    build_model,
    choose_dtype,
    infer,
    predictions_to_poses,
    sha256,
)
from lingbot_map.utils.load_fn import load_and_preprocess_images


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-root", type=Path,
        default=Path("/data1/mingzheliu/datasets/EuRoc_flat/MH_01_easy"),
    )
    parser.add_argument(
        "--model-path", type=Path, default=ROOT / "checkpoints/lingbot-map-long.pt"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/euroc_mh01_lingbot_map"
    )
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--keyframe-interval", type=int, default=None)
    parser.add_argument("--kv-cache-sliding-window", type=int, default=64)
    parser.add_argument("--max-frame-num", type=int, default=1024)
    parser.add_argument("--camera-num-iterations", type=int, default=4)
    parser.add_argument("--use-sdpa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16"
    )
    return parser.parse_args()


def load_camera_rows(
    sequence_root: Path, stride: int, maximum: int | None,
) -> tuple[list[Path], np.ndarray, np.ndarray]:
    if stride < 1:
        raise ValueError("--frame-stride must be positive")
    csv_path = sequence_root / "mav0/cam0/data.csv"
    image_root = csv_path.parent / "data"
    timestamps: list[int] = []
    paths: list[Path] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(line for line in handle if not line.startswith("#"))
        for row in rows:
            if row:
                timestamps.append(int(row[0]))
                paths.append(image_root / row[1].strip())
    selected = np.arange(0, len(paths), stride, dtype=np.int64)
    if maximum is not None:
        if maximum < 2:
            raise ValueError("--max-frames must be at least 2")
        selected = selected[:maximum]
    missing = [paths[index] for index in selected if not paths[index].is_file()]
    if missing:
        raise FileNotFoundError(f"Missing EuRoC image {missing[0]}")
    return [paths[index] for index in selected], np.asarray(timestamps)[selected], selected


def main() -> None:
    args = parse_args()
    paths, timestamps_ns, frame_indices = load_camera_rows(
        args.sequence_root, args.frame_stride, args.max_frames
    )
    keyframe_interval = args.keyframe_interval or max(1, math.ceil(len(paths) / 320))
    print(
        f"EuRoC {args.sequence_root.name}: {len(paths)} frames, "
        f"keyframe interval={keyframe_interval}", flush=True,
    )
    images = load_and_preprocess_images(
        [str(path) for path in paths], mode="crop",
        image_size=args.image_size, patch_size=args.patch_size,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    dtype = choose_dtype(args.dtype, device)
    model_args = SimpleNamespace(
        inference_mode="streaming", image_size=args.image_size, patch_size=args.patch_size,
        max_frame_num=args.max_frame_num, kv_cache_sliding_window=args.kv_cache_sliding_window,
        num_scale_frames=args.num_scale_frames, use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations, model_path=args.model_path,
        window_size=64, overlap_keyframes=8, window_scale_mode="hybrid",
    )
    model = build_model(model_args, device, dtype).requires_grad_(False)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable:
        raise RuntimeError(f"LingBot must be frozen, found {trainable} trainable parameters")
    predictions, inference_seconds, peak_gpu_gib = infer(
        model, images, model_args, device, dtype, keyframe_interval
    )
    predicted_c2w, intrinsics = predictions_to_poses(
        predictions["pose_enc"], tuple(images.shape[-2:]), "opencv_w2c"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "trajectory.npz",
        frame_indices=frame_indices, timestamps_ns=timestamps_ns,
        image_paths=np.asarray([str(path) for path in paths]),
        predicted_c2w=predicted_c2w, intrinsics=intrinsics,
    )
    metadata = {
        "dataset": "EuRoC MAV", "sequence": args.sequence_root.name,
        "frames": len(paths), "frame_stride": args.frame_stride,
        "model": "frozen LingBot-Map", "checkpoint": str(args.model_path),
        "checkpoint_sha256": sha256(args.model_path), "trainable_parameters": trainable,
        "inference_seconds": inference_seconds, "peak_gpu_gib": peak_gpu_gib,
        "pose_convention": "OpenCV W2C decoded then inverted to C2W",
        "ground_truth_read": False,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(
        f"Saved {len(predicted_c2w)} frozen poses in {inference_seconds:.1f}s; "
        f"peak GPU {peak_gpu_gib:.2f} GiB", flush=True,
    )


if __name__ == "__main__":
    main()
