#!/usr/bin/env python3
"""Run frozen LingBot-Map on the bundled Oxford RGB example.

The example contains images only, so this is an inference and trajectory
integrity test rather than an ATE benchmark. No timestamps, IMU, calibration,
or ground-truth trajectory are fabricated.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_kitti_odometry import (
    build_model,
    choose_dtype,
    infer,
    predictions_to_poses,
    rotation_angle_deg,
    sha256,
)
from lingbot_map.utils.load_fn import load_and_preprocess_images


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=ROOT / "example/oxford")
    parser.add_argument("--model-path", type=Path, default=ROOT / "checkpoints/lingbot-map.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/oxford_lingbot_map_pt")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--keyframe-interval", type=int, default=1)
    parser.add_argument("--kv-cache-sliding-window", type=int, default=64)
    parser.add_argument("--max-frame-num", type=int, default=1024)
    parser.add_argument("--camera-num-iterations", type=int, default=4)
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--use-sdpa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inference-mode", choices=("streaming", "windowed"), default="streaming")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--overlap-keyframes", type=int, default=8)
    parser.add_argument("--window-scale-mode", choices=("median", "depth", "centers", "hybrid"), default="hybrid")
    return parser.parse_args()


def trajectory_diagnostics(c2w: np.ndarray) -> dict[str, object]:
    rotations = c2w[:, :3, :3]
    positions = c2w[:, :3, 3]
    relative = np.linalg.inv(c2w[:-1]) @ c2w[1:]
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    angles = rotation_angle_deg(relative[:, :3, :3])
    centered = positions - positions.mean(axis=0)
    span = np.ptp(positions, axis=0)
    return {
        "all_pose_values_finite": bool(np.isfinite(c2w).all()),
        "max_rotation_orthogonality_error": float(np.max(np.linalg.norm(
            np.swapaxes(rotations, 1, 2) @ rotations - np.eye(3), axis=(1, 2)
        ))),
        "max_rotation_determinant_error": float(np.max(np.abs(
            np.linalg.det(rotations) - 1.0
        ))),
        "trajectory_span_model_units": span.tolist(),
        "trajectory_rms_radius_model_units": float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))),
        "path_length_model_units": float(np.sum(steps)),
        "step_model_units": {
            "median": float(np.median(steps)),
            "p90": float(np.quantile(steps, 0.9)),
            "p99": float(np.quantile(steps, 0.99)),
            "max": float(np.max(steps)),
        },
        "relative_rotation_deg": {
            "median": float(np.median(angles)),
            "p90": float(np.quantile(angles, 0.9)),
            "p99": float(np.quantile(angles, 0.99)),
            "max": float(np.max(angles)),
        },
    }


def save_plot(path: Path, c2w: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    position = c2w[:, :3, 3]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, columns, labels in zip(
        axes, ((0, 2), (0, 1)), (("x", "z"), ("x", "y"))
    ):
        axis.plot(position[:, columns[0]], position[:, columns[1]], linewidth=1.2)
        axis.scatter(position[0, columns[0]], position[0, columns[1]], s=28, label="start")
        axis.scatter(position[-1, columns[0]], position[-1, columns[1]], s=28, label="end")
        axis.set_xlabel(f"{labels[0]} [model units]")
        axis.set_ylabel(f"{labels[1]} [model units]")
        axis.axis("equal")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Oxford example: LingBot-Map predicted trajectory (no GT)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    paths = sorted(args.image_root.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG images found in {args.image_root}")
    if args.keyframe_interval < 1:
        raise ValueError("--keyframe-interval must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    started = time.perf_counter()
    images = load_and_preprocess_images(
        [str(path) for path in paths],
        mode="crop",
        image_size=args.image_size,
        patch_size=args.patch_size,
    )
    dtype = choose_dtype(args.dtype, device)
    model = build_model(args, device, dtype)
    load_seconds = time.perf_counter() - started
    predictions, inference_seconds, peak_gpu_gib = infer(
        model, images, args, device, dtype, args.keyframe_interval
    )
    pose_count = int(predictions["pose_enc"].shape[1])
    if pose_count != len(paths):
        raise RuntimeError(f"Expected {len(paths)} poses, received {pose_count}")
    c2w, intrinsics = predictions_to_poses(
        predictions["pose_enc"], tuple(images.shape[-2:]), "opencv_w2c"
    )
    diagnostics = trajectory_diagnostics(c2w)
    depth = predictions.get("depth")
    confidence = predictions.get("depth_conf")
    if depth is not None:
        diagnostics["finite_depth_fraction"] = float(torch.isfinite(depth).float().mean())
    if confidence is not None:
        diagnostics["finite_depth_confidence_fraction"] = float(
            torch.isfinite(confidence).float().mean()
        )

    np.savez_compressed(
        args.output_dir / "trajectory.npz",
        frame_indices=np.arange(len(paths), dtype=np.int64),
        image_paths=np.asarray([str(path) for path in paths]),
        predicted_c2w=c2w,
        intrinsics=intrinsics,
    )
    save_plot(args.output_dir / "trajectory.png", c2w)
    result = {
        "dataset": "LingBot-Map bundled Oxford example",
        "input": str(args.image_root),
        "num_images": len(paths),
        "image_shape_chw": list(images.shape[1:]),
        "checkpoint": str(args.model_path),
        "checkpoint_sha256": sha256(args.model_path),
        "inference_mode": args.inference_mode,
        "keyframe_interval": args.keyframe_interval,
        "dtype": str(dtype),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "fps": len(paths) / max(inference_seconds, 1e-9),
        "peak_gpu_gib": peak_gpu_gib,
        "pose_convention": "OpenCV W2C decoded then inverted to C2W",
        "ate": None,
        "ate_status": (
            "unavailable: the bundled RGB-only example has no ground-truth "
            "trajectory or timestamps"
        ),
        "airimu_fusion": None,
        "airimu_status": "unavailable: the bundled example has no IMU stream",
        "trajectory_diagnostics": diagnostics,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "num_images": len(paths),
        "inference_seconds": inference_seconds,
        "fps": result["fps"],
        "peak_gpu_gib": peak_gpu_gib,
        "ate_status": result["ate_status"],
        "trajectory_diagnostics": diagnostics,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
