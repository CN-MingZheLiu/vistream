#!/usr/bin/env python3
"""Leakage-free KITTI odometry and reconstruction evaluation for LingBot-Map.

Ground-truth poses are loaded only after model inference. They are never passed
to image preprocessing, GCTStream, scale-frame selection, or KV-cache updates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = REPO_ROOT / "checkpoints" / "lingbot-map-long.pt"
DEFAULT_KITTI = Path(
    "/data2/huangshuaixin/LingBot-Map-relocalization/kitti_odometry/dataset"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--kitti-root", type=Path, default=DEFAULT_KITTI)
    parser.add_argument("--sequence", default="05")
    parser.add_argument("--frame-stride", type=int, default=6)
    parser.add_argument(
        "--metric-stride",
        type=int,
        default=1,
        help="Evaluate every Nth model input while still inferring every input frame.",
    )
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument(
        "--keyframe-interval",
        type=int,
        default=None,
        help="Default: ceil(sampled_frames / 320), matching the official demo.",
    )
    parser.add_argument("--kv-cache-sliding-window", type=int, default=64)
    parser.add_argument("--max-frame-num", type=int, default=1024)
    parser.add_argument("--camera-num-iterations", type=int, default=4)
    parser.add_argument(
        "--inference-mode",
        choices=("streaming", "windowed"),
        default="streaming",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=64,
        help="Keyframes per independent window in windowed mode.",
    )
    parser.add_argument(
        "--overlap-keyframes",
        type=int,
        default=8,
        help="Shared keyframe intervals used for cross-window Sim(3) alignment.",
    )
    parser.add_argument(
        "--window-scale-mode",
        choices=("median", "depth", "centers", "hybrid"),
        default="hybrid",
    )
    parser.add_argument(
        "--pose-convention",
        choices=("benchmark_c2w", "opencv_w2c"),
        default="opencv_w2c",
        help=(
            "pose_enc is OpenCV world-to-camera, as documented by pose_enc.py. "
            "benchmark_c2w retains the older diagnostic interpretation."
        ),
    )
    parser.add_argument("--use-sdpa", action="store_true")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument("--pixel-stride", type=int, default=8)
    parser.add_argument("--point-frame-stride", type=int, default=1)
    parser.add_argument("--confidence-threshold", type=float, default=1.5)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--voxel-size", type=float, default=0.10)
    parser.add_argument("--max-points", type=int, default=2_000_000)
    parser.add_argument("--no-point-cloud", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "kitti_05_stride6_lingbot_map_long",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_dtype(name: str, device: torch.device) -> torch.dtype:
    if name != "auto":
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[name]
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def sampled_inputs(
    root: Path, sequence: str, stride: int
) -> tuple[list[Path], np.ndarray, np.ndarray]:
    if stride < 1:
        raise ValueError(f"frame_stride must be positive, got {stride}")
    sequence_dir = root / "sequences" / sequence
    paths = sorted((sequence_dir / "image_2").glob("*.png"))
    timestamps = np.loadtxt(sequence_dir / "times.txt", dtype=np.float64)
    if len(paths) != len(timestamps):
        raise ValueError(f"Image/time count mismatch: {len(paths)} vs {len(timestamps)}")
    indices = np.arange(0, len(paths), stride, dtype=np.int64)
    if indices[-1] != len(paths) - 1:
        indices = np.append(indices, len(paths) - 1)
    return [paths[i] for i in indices], timestamps[indices], indices


def build_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    if args.inference_mode == "windowed":
        from lingbot_map.models.gct_stream_window import GCTStream
    else:
        from lingbot_map.models.gct_stream import GCTStream

    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_point=False,
        enable_depth=True,
        enable_3d_rope=True,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
    )
    checkpoint = torch.load(
        args.model_path, map_location="cpu", mmap=True, weights_only=False
    )
    state = checkpoint.get("model", checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Official checkpoint did not match the official GCTStream construction: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    del checkpoint, state
    model = model.to(device).eval()
    # This follows demo.py: the GCT aggregator runs in reduced precision while
    # camera/depth heads stay fp32 and explicitly upcast their token inputs.
    if dtype != torch.float32:
        model.aggregator = model.aggregator.to(dtype=dtype)
    return model


def infer(
    model: Any,
    images: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    keyframe_interval: int,
) -> tuple[dict[str, torch.Tensor], float, float | None]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        autocast = torch.amp.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32)
    else:
        autocast = nullcontext()
    started = time.perf_counter()
    with torch.no_grad(), autocast:
        if args.inference_mode == "windowed":
            predictions = model.inference_windowed(
                images,
                window_size=args.window_size,
                overlap_keyframes=args.overlap_keyframes,
                num_scale_frames=args.num_scale_frames,
                scale_mode=args.window_scale_mode,
                keyframe_interval=keyframe_interval,
                output_device=torch.device("cpu"),
            )
        else:
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=keyframe_interval,
                output_device=torch.device("cpu"),
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_gib = float(torch.cuda.max_memory_allocated(device) / 2**30)
    else:
        peak_gib = None
    return predictions, time.perf_counter() - started, peak_gib


def predictions_to_poses(
    pose_enc: torch.Tensor, image_hw: tuple[int, int], convention: str
) -> tuple[np.ndarray, np.ndarray]:
    if pose_enc.ndim == 2:
        pose_enc = pose_enc.unsqueeze(0)
    # pose_enc.py defines the decoded extrinsic as OpenCV W2C and demo.py
    # therefore inverts it. Keep the historical direct-C2W interpretation only
    # as an explicit diagnostic for comparing older result folders.
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        pose_enc.float(), image_hw
    )
    extrinsic_4x4 = torch.zeros(
        (*extrinsic.shape[:-2], 4, 4), dtype=extrinsic.dtype
    )
    extrinsic_4x4[..., :3, :4] = extrinsic.cpu()
    extrinsic_4x4[..., 3, 3] = 1.0
    if convention == "opencv_w2c":
        extrinsic_4x4 = torch.linalg.inv(extrinsic_4x4)
    c2w = extrinsic_4x4[0].numpy().astype(np.float64)
    return c2w, intrinsic[0].cpu().numpy().astype(np.float64)


def load_ground_truth(root: Path, sequence: str, indices: np.ndarray) -> np.ndarray:
    raw = np.loadtxt(root / "poses" / f"{sequence}.txt", dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None]
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(raw), axis=0)
    poses[:, :3, :4] = raw.reshape(-1, 3, 4)
    return poses[indices]


def umeyama(
    predicted: np.ndarray, ground_truth: np.ndarray, with_scale: bool
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    pred_mean = predicted.mean(axis=0)
    gt_mean = ground_truth.mean(axis=0)
    pred_zero = predicted - pred_mean
    gt_zero = ground_truth - gt_mean
    u, singular, vt = np.linalg.svd(pred_zero.T @ gt_zero / len(predicted))
    sign = 1.0 if np.linalg.det(vt.T @ u.T) >= 0 else -1.0
    correction = np.diag([1.0, 1.0, sign])
    rotation = vt.T @ correction @ u.T
    if with_scale:
        variance = float(np.sum(pred_zero**2) / len(predicted))
        scale = float(np.trace(np.diag(singular) @ correction) / max(variance, 1e-12))
    else:
        scale = 1.0
    translation = gt_mean - scale * rotation @ pred_mean
    aligned = (scale * (rotation @ predicted.T)).T + translation
    return aligned, rotation, scale, translation


def aligned_poses(
    poses: np.ndarray, rotation: np.ndarray, scale: float, translation: np.ndarray
) -> np.ndarray:
    result = poses.copy()
    result[:, :3, :3] = np.einsum("ij,njk->nik", rotation, poses[:, :3, :3])
    result[:, :3, 3] = (
        scale * np.einsum("ij,nj->ni", rotation, poses[:, :3, 3]) + translation
    )
    return result


def position_metrics(predicted: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    error = np.linalg.norm(predicted - ground_truth, axis=1)
    return {
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "mean_m": float(np.mean(error)),
        "median_m": float(np.median(error)),
        "max_m": float(np.max(error)),
    }


def rotation_angle_deg(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def relative_pose_metrics(
    predicted: np.ndarray, ground_truth: np.ndarray, delta: int
) -> dict[str, float | int]:
    if delta >= len(predicted):
        return {"pairs": 0, "translation_rmse_m": math.nan, "rotation_rmse_deg": math.nan}
    gt_relative = np.linalg.inv(ground_truth[:-delta]) @ ground_truth[delta:]
    pred_relative = np.linalg.inv(predicted[:-delta]) @ predicted[delta:]
    error = np.linalg.inv(gt_relative) @ pred_relative
    trans = np.linalg.norm(error[:, :3, 3], axis=1)
    rot = rotation_angle_deg(error[:, :3, :3])
    return {
        "pairs": len(error),
        "translation_rmse_m": float(np.sqrt(np.mean(trans**2))),
        "translation_mean_m": float(np.mean(trans)),
        "rotation_rmse_deg": float(np.sqrt(np.mean(rot**2))),
        "rotation_mean_deg": float(np.mean(rot)),
    }


def kitti_segment_metrics(
    predicted: np.ndarray, ground_truth: np.ndarray
) -> dict[str, object]:
    lengths = (100, 200, 300, 400, 500, 600, 700, 800)
    gt_step = np.linalg.norm(np.diff(ground_truth[:, :3, 3], axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(gt_step)])
    trans_errors: list[float] = []
    rot_errors: list[float] = []
    per_length: dict[str, dict[str, float | int]] = {}
    for length in lengths:
        trans_at_length: list[float] = []
        rot_at_length: list[float] = []
        for start in range(len(ground_truth) - 1):
            end = int(np.searchsorted(distance, distance[start] + length))
            if end >= len(ground_truth):
                continue
            gt_delta = np.linalg.inv(ground_truth[start]) @ ground_truth[end]
            pred_delta = np.linalg.inv(predicted[start]) @ predicted[end]
            error = np.linalg.inv(gt_delta) @ pred_delta
            trans_at_length.append(float(np.linalg.norm(error[:3, 3]) / length))
            rot_at_length.append(float(np.radians(rotation_angle_deg(error[:3, :3])) / length))
        if trans_at_length:
            trans_errors.extend(trans_at_length)
            rot_errors.extend(rot_at_length)
            per_length[str(length)] = {
                "segments": len(trans_at_length),
                "translation_percent": 100.0 * float(np.mean(trans_at_length)),
                "rotation_deg_per_100m": math.degrees(float(np.mean(rot_at_length))) * 100.0,
            }
    return {
        "translation_percent": 100.0 * float(np.mean(trans_errors)) if trans_errors else math.nan,
        "rotation_deg_per_100m": (
            math.degrees(float(np.mean(rot_errors))) * 100.0 if rot_errors else math.nan
        ),
        "segments": len(trans_errors),
        "per_length_m": per_length,
    }


def voxel_downsample(
    points: np.ndarray, colors: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0:
        return points, colors
    cells = np.floor(points / voxel_size).astype(np.int64)
    _, selected = np.unique(cells, axis=0, return_index=True)
    return points[selected], colors[selected]


def write_rgb_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertices = np.empty(
        len(points),
        dtype=np.dtype(
            [
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ]
        ),
    )
    vertices["x"], vertices["y"], vertices["z"] = points.astype(np.float32).T
    vertices["red"], vertices["green"], vertices["blue"] = colors.astype(np.uint8).T
    header = "\n".join(
        [
            "ply", "format binary_little_endian 1.0",
            "comment official LingBot-Map KITTI reconstruction with RGB",
            f"element vertex {len(vertices)}", "property float x", "property float y",
            "property float z", "property uchar red", "property uchar green",
            "property uchar blue", "end_header", "",
        ]
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def export_reconstruction(
    output_dir: Path,
    depth: torch.Tensor,
    confidence: torch.Tensor,
    images: torch.Tensor,
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    sim3_rotation: np.ndarray,
    sim3_scale: float,
    sim3_translation: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    if depth.ndim == 5:
        depth = depth[0]
    if depth.shape[-1] == 1:
        depth = depth[..., 0]
    if confidence.ndim == 4:
        confidence = confidence[0]
    if images.ndim == 5:
        images = images[0]
    frame_ids = range(0, len(c2w), args.point_frame_stride)
    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    candidates = 0
    kept = 0
    for frame in frame_ids:
        d = depth[frame, :: args.pixel_stride, :: args.pixel_stride].numpy()
        conf = confidence[frame, :: args.pixel_stride, :: args.pixel_stride].numpy()
        color = (
            images[frame, :, :: args.pixel_stride, :: args.pixel_stride]
            .permute(1, 2, 0)
            .numpy()
        )
        vv, uu = np.meshgrid(
            np.arange(0, depth.shape[1], args.pixel_stride, dtype=np.float64),
            np.arange(0, depth.shape[2], args.pixel_stride, dtype=np.float64),
            indexing="ij",
        )
        camera = np.stack(
            [
                (uu - intrinsics[frame, 0, 2]) * d / intrinsics[frame, 0, 0],
                (vv - intrinsics[frame, 1, 2]) * d / intrinsics[frame, 1, 1],
                d,
            ],
            axis=-1,
        ).reshape(-1, 3)
        conf = conf.reshape(-1)
        color = np.clip(color.reshape(-1, 3) * 255.0, 0, 255).astype(np.uint8)
        valid = (
            np.isfinite(camera).all(axis=1)
            & np.isfinite(conf)
            & (conf >= args.confidence_threshold)
            & (camera[:, 2] >= args.min_depth)
            & (camera[:, 2] <= args.max_depth)
        )
        candidates += len(camera)
        kept += int(valid.sum())
        camera = camera[valid]
        world = camera @ c2w[frame, :3, :3].T + c2w[frame, :3, 3]
        point_chunks.append(world.astype(np.float32))
        color_chunks.append(color[valid])
    points = np.concatenate(point_chunks)
    colors = np.concatenate(color_chunks)
    points, colors = voxel_downsample(points, colors, args.voxel_size)
    after_voxel = len(points)
    if len(points) > args.max_points:
        rng = np.random.default_rng(20260801)
        selected = np.sort(rng.choice(len(points), args.max_points, replace=False))
        points, colors = points[selected], colors[selected]
    model_path = output_dir / "reconstruction_model_frame_rgb.ply"
    aligned_path = output_dir / "reconstruction_sim3_aligned_rgb.ply"
    write_rgb_ply(model_path, points, colors)
    aligned = (
        sim3_scale * np.einsum("ij,nj->ni", sim3_rotation, points)
        + sim3_translation
    )
    write_rgb_ply(aligned_path, aligned, colors)
    return {
        "geometry_source": "official LingBot-Map depth head unprojected with its predicted camera poses/intrinsics",
        "gt_use": "only reconstruction_sim3_aligned_rgb.ply uses the trajectory Sim(3) transform",
        "pixel_stride": args.pixel_stride,
        "frame_stride_within_sampled_stream": args.point_frame_stride,
        "confidence_threshold": args.confidence_threshold,
        "min_depth_m": args.min_depth,
        "max_depth_m": args.max_depth,
        "voxel_size_model_units": args.voxel_size,
        "candidates": candidates,
        "after_filter": kept,
        "after_voxel": after_voxel,
        "exported_points": len(points),
        "model_frame_ply": str(model_path),
        "sim3_aligned_ply": str(aligned_path),
    }


def save_plot(
    path: Path, se3: np.ndarray, sim3: np.ndarray, gt: np.ndarray, sequence: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for axis, predicted, alignment in zip(axes, (se3, sim3), ("SE(3)", "Sim(3)")):
        axis.plot(gt[:, 0], gt[:, 2], color="black", linewidth=1.6, label="KITTI GT")
        axis.plot(predicted[:, 0], predicted[:, 2], color="#d1495b", linewidth=1.1, label="LingBot-Map")
        axis.set_title(f"KITTI {sequence}, {alignment} aligned")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("z [m]")
        axis.axis("equal")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.use_sdpa and shutil.which("ninja") is None:
        print("ninja is unavailable; using the official SDPA fallback", flush=True)
        args.use_sdpa = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    paths, timestamps, frame_indices = sampled_inputs(
        args.kitti_root, args.sequence, args.frame_stride
    )
    if args.metric_stride < 1:
        raise ValueError(f"metric_stride must be positive, got {args.metric_stride}")
    keyframe_interval = args.keyframe_interval
    if keyframe_interval is None:
        keyframe_interval = max(1, math.ceil(len(paths) / 320))
    print(
        f"KITTI {args.sequence}: {len(paths)} frames, raw stride={args.frame_stride}, "
        f"keyframe interval={keyframe_interval}",
        flush=True,
    )
    load_started = time.perf_counter()
    images = load_and_preprocess_images(
        [str(path) for path in paths],
        mode="crop",
        image_size=args.image_size,
        patch_size=args.patch_size,
    )
    dtype = choose_dtype(args.dtype, device)
    model = build_model(args, device, dtype)
    load_seconds = time.perf_counter() - load_started
    print(f"Loaded images/model in {load_seconds:.1f}s; input={tuple(images.shape)}", flush=True)
    predictions, inference_seconds, peak_gpu_gib = infer(
        model, images, args, device, dtype, keyframe_interval
    )
    print(
        f"{args.inference_mode.capitalize()} inference finished in "
        f"{inference_seconds:.1f}s",
        flush=True,
    )

    predicted_frames = predictions["pose_enc"].shape[1]
    if predicted_frames != len(frame_indices):
        raise RuntimeError(
            f"Inference returned {predicted_frames} poses for {len(frame_indices)} inputs"
        )

    # Leakage boundary: GT is deliberately first loaded after inference ends.
    pred_poses_all, intrinsics = predictions_to_poses(
        predictions["pose_enc"], tuple(images.shape[-2:]), args.pose_convention
    )
    metric_indices = np.arange(0, len(frame_indices), args.metric_stride, dtype=np.int64)
    if metric_indices[-1] != len(frame_indices) - 1:
        metric_indices = np.append(metric_indices, len(frame_indices) - 1)
    evaluation_frame_indices = frame_indices[metric_indices]
    evaluation_timestamps = timestamps[metric_indices]
    gt_poses = load_ground_truth(
        args.kitti_root, args.sequence, evaluation_frame_indices
    )
    pred_poses = pred_poses_all[metric_indices]
    pred_positions = pred_poses[:, :3, 3]
    gt_positions = gt_poses[:, :3, 3]
    origin_positions = pred_positions + gt_positions[0] - pred_positions[0]
    se3_positions, se3_rotation, _, se3_translation = umeyama(
        pred_positions, gt_positions, with_scale=False
    )
    sim3_positions, sim3_rotation, sim3_scale, sim3_translation = umeyama(
        pred_positions, gt_positions, with_scale=True
    )
    se3_poses = aligned_poses(pred_poses, se3_rotation, 1.0, se3_translation)
    sim3_poses = aligned_poses(
        pred_poses, sim3_rotation, sim3_scale, sim3_translation
    )

    reconstruction = None
    if not args.no_point_cloud:
        reconstruction = export_reconstruction(
            args.output_dir,
            predictions["depth"],
            predictions["depth_conf"],
            images,
            pred_poses_all,
            intrinsics,
            sim3_rotation,
            sim3_scale,
            sim3_translation,
            args,
        )
    window_alignment = None
    if args.inference_mode == "windowed":
        chunk_scales = predictions.get("chunk_scales")
        window_alignment = {
            "num_windows": int(chunk_scales.shape[1]) if chunk_scales is not None else 1,
            "chunk_scales": (
                chunk_scales.float().cpu().numpy().tolist()
                if chunk_scales is not None
                else [[1.0]]
            ),
        }

    metrics = {
        "dataset": "KITTI Odometry",
        "sequence": args.sequence,
        "protocol": (
            f"{args.inference_mode} causal inference; GT loaded only after inference"
        ),
        "model": "LingBot-Map GCTStream",
        "model_source": str(REPO_ROOT),
        "checkpoint": str(args.model_path),
        "checkpoint_sha256": sha256(args.model_path),
        "num_raw_frames": int(frame_indices[-1] + 1),
        "num_input_frames": len(frame_indices),
        "input_frame_stride": args.frame_stride,
        "input_frame_indices": frame_indices.tolist(),
        "num_evaluation_frames": len(evaluation_frame_indices),
        "metric_stride_within_inputs": args.metric_stride,
        "evaluation_frame_indices": evaluation_frame_indices.tolist(),
        "duration_s": float(evaluation_timestamps[-1] - evaluation_timestamps[0]),
        "image_shape_chw": list(images.shape[1:]),
        "num_scale_frames": args.num_scale_frames,
        "keyframe_interval": keyframe_interval,
        "kv_cache_sliding_window": args.kv_cache_sliding_window,
        "camera_num_iterations": args.camera_num_iterations,
        "inference_mode": args.inference_mode,
        "window_size_keyframes": (
            args.window_size if args.inference_mode == "windowed" else None
        ),
        "overlap_keyframes": (
            args.overlap_keyframes if args.inference_mode == "windowed" else None
        ),
        "window_scale_mode": (
            args.window_scale_mode if args.inference_mode == "windowed" else None
        ),
        "window_alignment": window_alignment,
        "pose_convention": args.pose_convention,
        "inference_backend": "SDPA" if args.use_sdpa else "FlashInfer",
        "dtype": str(dtype).removeprefix("torch."),
        "device": str(device),
        "timing": {
            "preprocess_and_model_load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "fps_excluding_load": len(frame_indices) / inference_seconds,
            "peak_gpu_memory_gib": peak_gpu_gib,
        },
        "ate_raw": position_metrics(pred_positions, gt_positions),
        "ate_origin_aligned": position_metrics(origin_positions, gt_positions),
        "ate_se3_aligned": position_metrics(se3_positions, gt_positions),
        "ate_sim3_aligned": position_metrics(sim3_positions, gt_positions),
        "sim3_scale": sim3_scale,
        "rpe_se3_aligned": {
            "delta_1_sampled_frame": relative_pose_metrics(se3_poses, gt_poses, 1),
            "delta_10_sampled_frames": relative_pose_metrics(se3_poses, gt_poses, 10),
        },
        "rpe_sim3_aligned": {
            "delta_1_sampled_frame": relative_pose_metrics(sim3_poses, gt_poses, 1),
            "delta_10_sampled_frames": relative_pose_metrics(sim3_poses, gt_poses, 10),
        },
        "kitti_segment_drift_sim3_scale": kitti_segment_metrics(sim3_poses, gt_poses),
        "alignment": {
            "se3_rotation": se3_rotation.tolist(),
            "se3_translation": se3_translation.tolist(),
            "sim3_rotation": sim3_rotation.tolist(),
            "sim3_translation": sim3_translation.tolist(),
        },
        "reconstruction": reconstruction,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    np.savez_compressed(
        args.output_dir / "trajectory.npz",
        frame_indices=evaluation_frame_indices,
        timestamps=evaluation_timestamps,
        predicted_c2w=pred_poses,
        ground_truth_c2w=gt_poses,
        se3_c2w=se3_poses,
        sim3_c2w=sim3_poses,
    )
    np.savetxt(
        args.output_dir / "trajectory.csv",
        np.column_stack(
            [
                evaluation_frame_indices,
                evaluation_timestamps,
                pred_positions,
                gt_positions,
                se3_positions,
                sim3_positions,
            ]
        ),
        delimiter=",",
        header=(
            "frame,timestamp,pred_x,pred_y,pred_z,gt_x,gt_y,gt_z,"
            "se3_x,se3_y,se3_z,sim3_x,sim3_y,sim3_z"
        ),
        comments="",
    )
    save_plot(
        args.output_dir / "trajectory_alignment.png",
        se3_positions,
        sim3_positions,
        gt_positions,
        args.sequence,
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
