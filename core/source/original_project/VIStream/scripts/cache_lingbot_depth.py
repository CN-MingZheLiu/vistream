#!/usr/bin/env python3
"""Cache frozen Lingbot depth artifacts without ever opening ground truth."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_evaluator(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("lingbot_kitti_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--trajectory-cache", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--depth-pixel-stride", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--kv-cache-sliding-window", type=int, default=64)
    parser.add_argument("--max-frame-num", type=int, default=1024)
    parser.add_argument("--camera-num-iterations", type=int, default=4)
    parser.add_argument("--keyframe-interval", type=int, required=True)
    parser.add_argument("--pose-convention", default="benchmark_c2w")
    parser.add_argument("--use-sdpa", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = load_evaluator(args.evaluator.resolve())
    with np.load(args.trajectory_cache, allow_pickle=False) as archive:
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        cached_poses = np.asarray(archive["predicted_c2w"], dtype=np.float64)
    image_dir = args.image_root / args.sequence / "image_2"
    paths = [image_dir / f"{int(frame):06d}.png" for frame in frame_indices]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[0])
    images = evaluator.load_and_preprocess_images(
        [str(path) for path in paths],
        mode="crop",
        image_size=args.image_size,
        patch_size=args.patch_size,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    model_args = SimpleNamespace(
        inference_mode="streaming",
        image_size=args.image_size,
        patch_size=args.patch_size,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        num_scale_frames=args.num_scale_frames,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
        model_path=args.model_path,
        window_size=64,
        overlap_keyframes=8,
        window_scale_mode="hybrid",
    )
    dtype = evaluator.choose_dtype(args.dtype, device)
    model = evaluator.build_model(model_args, device, dtype)
    predictions, inference_seconds, peak_gpu_gib = evaluator.infer(
        model, images, model_args, device, dtype, args.keyframe_interval
    )
    inferred_poses, intrinsics = evaluator.predictions_to_poses(
        predictions["pose_enc"], tuple(images.shape[-2:]), args.pose_convention
    )
    if inferred_poses.shape != cached_poses.shape:
        raise ValueError("inferred pose shape differs from frozen trajectory cache")
    translation_delta = np.linalg.norm(
        inferred_poses[:, :3, 3] - cached_poses[:, :3, 3], axis=1
    )
    depth = predictions["depth"]
    confidence = predictions["depth_conf"]
    if depth.ndim == 5:
        depth = depth[0]
    if depth.shape[-1] == 1:
        depth = depth[..., 0]
    if confidence.ndim == 4:
        confidence = confidence[0]
    stride = args.depth_pixel_stride
    depth = depth[:, ::stride, ::stride].float().cpu().numpy().astype(np.float16)
    confidence = (
        confidence[:, ::stride, ::stride].float().cpu().numpy().astype(np.float16)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        frame_indices=frame_indices,
        depth=depth,
        depth_confidence=confidence,
        intrinsics=intrinsics,
        image_hw=np.asarray(images.shape[-2:], dtype=np.int64),
        depth_pixel_stride=np.asarray(stride, dtype=np.int64),
    )
    metadata = {
        "protocol": "frozen Lingbot inference; ground truth is never opened",
        "sequence": args.sequence,
        "checkpoint": str(args.model_path),
        "checkpoint_sha256": sha256(args.model_path),
        "trajectory_cache": str(args.trajectory_cache),
        "trajectory_cache_sha256": sha256(args.trajectory_cache),
        "frame_count": len(frame_indices),
        "keyframe_interval": args.keyframe_interval,
        "image_hw": list(images.shape[-2:]),
        "depth_shape": list(depth.shape),
        "depth_pixel_stride": stride,
        "pose_translation_delta_median": float(np.median(translation_delta)),
        "pose_translation_delta_max": float(np.max(translation_delta)),
        "inference_seconds": inference_seconds,
        "peak_gpu_memory_gib": peak_gpu_gib,
        "artifact": str(args.output),
        "artifact_sha256": sha256(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
