#!/usr/bin/env python3
"""Evaluate VIStream with SE(3)/Sim(3) ATE and available CD/F1 ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from vistream.checkpoint import load_adapter_checkpoint
from vistream.config import load_yaml, model_config
from vistream.data import (
    EurocWindowDataset,
    KITTIOdometryWindowDataset,
    TartanAirWindowDataset,
    discover_tartanair_sequences,
    kaist_dataset,
    split_tartanair_by_environment,
)
from vistream.metrics import ate_rmse, point_cloud_metrics, umeyama_alignment
from vistream.model.system import VIStream


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _unproject(depth: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray, stride: int = 8) -> np.ndarray:
    depth = depth[::stride, ::stride]
    y, x = np.mgrid[0 : depth.shape[0] * stride : stride, 0 : depth.shape[1] * stride : stride]
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 80.0)
    pixels = np.stack((x[valid], y[valid], np.ones(valid.sum())), axis=-1)
    rays = pixels @ np.linalg.inv(intrinsics).T
    camera = rays * depth[valid, None]
    return _transform_points(pose, camera)


@torch.no_grad()
def evaluate_dataset(
    model: VIStream,
    dataset,
    device: torch.device,
    voxel_size: float,
    thresholds: list[float],
    include_geometry: bool = True,
    bf16: bool = False,
    progress_label: str | None = None,
) -> dict[str, float | str]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    predicted_poses: dict[int, np.ndarray] = {}
    target_poses: dict[int, np.ndarray] = {}
    predicted_clouds: list[np.ndarray] = []
    target_clouds: list[np.ndarray] = []
    processed_frames: set[int] = set()
    model.reset_stream()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for cpu_batch in loader:
        start = int(cpu_batch["start_frame"].item())
        batch = _to_device(cpu_batch, device)
        target = cpu_batch["poses"][0].numpy()
        for local in range(cpu_batch["images"].shape[1]):
            frame = start + local
            if frame in processed_frames:
                continue
            processed_frames.add(frame)
            if progress_label and len(processed_frames) % 500 == 0:
                total_frames = len(getattr(dataset, "image_files", ())) or None
                print(
                    json.dumps(
                        {
                            "evaluation_progress": progress_label,
                            "processed_frames": len(processed_frames),
                            "total_frames": total_frames,
                        }
                    ),
                    flush=True,
                )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=bf16 and device.type == "cuda",
            ):
                output = model.stream_step(
                    batch["images"][:, local],
                    batch["imu"][:, local],
                    batch["imu_dt"][:, local],
                    batch["imu_mask"][:, local],
                    batch.get("rotation_camera_from_imu"),
                    batch.get("gravity_camera"),
                    include_geometry=include_geometry,
                )
            if output is None:
                continue
            predicted_poses[frame] = output["camera_to_world"][0, 0].float().cpu().numpy()
            target_poses[frame] = target[local]
            if "world_points" in output and frame % 5 == 0:
                points = output["world_points"][0, 0, ::8, ::8].float().cpu().numpy().reshape(-1, 3)
                valid = np.isfinite(points).all(axis=-1) & (np.linalg.norm(points, axis=-1) < 1e4)
                predicted_clouds.append(points[valid])
            if cpu_batch["depth"].numel() > 0 and frame % 5 == 0:
                depth = cpu_batch["depth"][0, local, 0].numpy()
                intrinsics = cpu_batch["gt_intrinsics"][0, local].numpy()
                target_clouds.append(_unproject(depth, intrinsics, target[local]))
            elif hasattr(dataset, "load_lidar_camera") and frame % 5 == 0:
                lidar = dataset.load_lidar_camera(frame)
                if len(lidar):
                    target_clouds.append(_transform_points(target[local], lidar[::4]))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    frames = sorted(set(predicted_poses) & set(target_poses))
    predicted_positions = np.stack([predicted_poses[index][:3, 3] for index in frames])
    target_positions = np.stack([target_poses[index][:3, 3] for index in frames])
    se3 = ate_rmse(predicted_positions, target_positions, "se3")
    sim3 = ate_rmse(predicted_positions, target_positions, "sim3")
    result = {
        "frames": len(frames),
        "ate_rmse_se3_m": se3["ate_rmse"],
        "ate_mean_se3_m": se3["ate_mean"],
        "ate_rmse_sim3_m": sim3["ate_rmse"],
        "ate_mean_sim3_m": sim3["ate_mean"],
        "sim3_scale": sim3["scale"],
        "elapsed_seconds": elapsed,
        "frames_per_second": len(frames) / max(elapsed, 1e-9),
    }
    if hasattr(dataset, "imu_source"):
        result.update(
            {
                "imu_source": str(dataset.imu_source),
                "imu_rate_hz": float(dataset.imu_rate_hz),
                "imu_max_interval_samples": int(dataset.max_interval_imu_samples),
                "imu_sample_capacity": int(dataset.max_imu_samples),
                "imu_calibration": str(dataset.imu_calibration_source),
            }
        )
    if predicted_clouds and target_clouds:
        cloud_alignment = umeyama_alignment(
            predicted_positions, target_positions, with_scale=False
        )
        result.update(
            point_cloud_metrics(
                np.concatenate([cloud_alignment.apply(cloud) for cloud in predicted_clouds]),
                np.concatenate(target_clouds),
                thresholds=thresholds,
                voxel_size=voxel_size,
            )
        )
    return result


def _write_results(
    target: Path,
    dataset_name: str,
    navigation_only: bool,
    precision: str,
    attention_backend: str,
    bounded_cache_window: int | None,
    results: dict[str, dict[str, float | str]],
    complete: bool,
) -> None:
    numeric_keys = sorted(
        {
            key
            for scores in results.values()
            for key, value in scores.items()
            if isinstance(value, (int, float))
        }
    )
    mean = {
        key: float(np.mean([scores[key] for scores in results.values() if key in scores]))
        for key in numeric_keys
    }
    payload = {
        "dataset": dataset_name,
        "navigation_only": navigation_only,
        "precision": precision,
        "attention_backend": attention_backend,
        "bounded_cache_window_frames": bounded_cache_window,
        "complete": complete,
        "completed_sequences": list(results),
        "results": results,
        "mean": mean,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vistream_tartanair.yaml")
    parser.add_argument("--adapter", help="Adapter-only checkpoint; omit for native zero-gate baseline")
    parser.add_argument(
        "--dataset", choices=("tartanair", "kitti", "euroc", "kaist"), required=True
    )
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--navigation-only",
        action="store_true",
        help="Skip depth/world-point heads when only trajectory metrics are required.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use BF16 autocast for the frozen aggregator and fusion adapter.",
    )
    parser.add_argument(
        "--bounded-cache-window",
        type=int,
        metavar="FRAMES",
        help=(
            "Use a bounded causal KV cache with this many recent frames, in addition "
            "to the warm-up scale frames. Evicted special tokens are discarded."
        ),
    )
    parser.add_argument(
        "--flashinfer",
        action="store_true",
        help="Use LingBot's paged FlashInfer backend for the aggregator KV cache.",
    )
    parser.add_argument(
        "--evaluation-stride-frames",
        type=int,
        help=(
            "Dataset window stride used only for evaluation. Defaults to the window "
            "length so frames are decoded once while causal model state stays continuous."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an incomplete JSON output and skip sequences already present.",
    )
    args = parser.parse_args()
    if args.evaluation_stride_frames is not None and args.evaluation_stride_frames < 1:
        parser.error("--evaluation-stride-frames must be positive")
    config = load_yaml(args.config)
    evaluation_stride = args.evaluation_stride_frames or config["data"]["window_frames"]
    if args.flashinfer:
        config.setdefault("model", {})["use_sdpa"] = False
    if args.bounded_cache_window is not None:
        if args.bounded_cache_window < 1:
            parser.error("--bounded-cache-window must be at least 1")
        config.setdefault("model", {})["sliding_window_size"] = args.bounded_cache_window
        config["model"]["kv_cache_sliding_window"] = args.bounded_cache_window
        config["model"]["kv_cache_cross_frame_special"] = False
    evaluation = config["evaluation"]
    checkpoint_stage = None
    if args.adapter:
        checkpoint = torch.load(args.adapter, map_location="cpu", weights_only=False)
        checkpoint_stage = checkpoint.get("stage")
        if checkpoint_stage is None:
            checkpoint_stage = checkpoint.get("config", {}).get("adapter", {}).get("stage")
        if checkpoint_stage is None:
            raise RuntimeError("adapter checkpoint does not record its curriculum stage")
        config.setdefault("model", {}).setdefault("adapter", {})["stage"] = checkpoint_stage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VIStream.from_pretrained(model_config(config))
    if args.navigation_only:
        # Geometry heads are never called in this mode. Drop them before the
        # device transfer so long trajectory evaluation does not reserve GPU
        # memory for inactive frozen weights.
        model.lingbot.depth_head = None
        model.lingbot.point_head = None
        model.lingbot.local_point_head = None
    model = model.to(device).eval()
    if args.adapter:
        model.configure_stage(checkpoint_stage)
        load_adapter_checkpoint(args.adapter, model)
    model.arm_backbone_guard()

    if args.dataset == "tartanair":
        all_sequences = discover_tartanair_sequences(config["data"]["tartanair_root"])
        _, validation_sequences = split_tartanair_by_environment(
            all_sequences,
            config["data"]["validation_environments"],
            config["data"]["split_seed"],
        )
        sequences = args.sequences or validation_sequences
        factory = lambda sequence: TartanAirWindowDataset(
            config["data"]["tartanair_root"],
            [sequence],
            image_size=config["model"]["image_size"],
            window_frames=config["data"]["window_frames"],
            stride_frames=evaluation_stride,
            max_imu_samples=config["data"]["max_imu_samples"],
            acc_source=config["data"].get("acc_source", "acc"),
        )
    elif args.dataset == "kitti":
        sequences = args.sequences or evaluation["kitti_sequences"]
        factory = lambda sequence: KITTIOdometryWindowDataset(
            evaluation["kitti_odometry_root"], evaluation["kitti_raw_root"], sequence,
            image_size=config["model"]["image_size"],
            window_frames=config["data"]["window_frames"],
            stride_frames=evaluation_stride,
            max_imu_samples=config["data"].get("max_imu_samples", 32),
            prefer_high_rate_imu=config["data"].get("prefer_high_rate_imu", True),
        )
    elif args.dataset == "euroc":
        sequences = args.sequences or evaluation["euroc_sequences"]
        factory = lambda sequence: EurocWindowDataset(
            evaluation["euroc_root"], sequence,
            image_size=config["model"]["image_size"],
            window_frames=config["data"]["window_frames"],
            stride_frames=evaluation_stride,
        )
    else:
        sequences = args.sequences or None
        if sequences is None:
            from vistream.data import KAIST_SEQUENCE_NAMES
            sequences = list(KAIST_SEQUENCE_NAMES)
        factory = lambda sequence: kaist_dataset(
            evaluation["kaist_converted_root"], sequence,
            camera_yaml=evaluation["kaist_camera_yaml"],
            image_size=config["model"]["image_size"],
            window_frames=config["data"]["window_frames"],
            stride_frames=evaluation_stride,
        )
    target = Path(args.output)
    results: dict[str, dict[str, float | str]] = {}
    if args.resume and target.is_file():
        previous = json.loads(target.read_text())
        if previous.get("dataset") != args.dataset:
            raise RuntimeError(
                f"cannot resume {target}: dataset is {previous.get('dataset')!r}, "
                f"expected {args.dataset!r}"
            )
        results.update(previous.get("results", {}))
        sequences = [sequence for sequence in sequences if sequence not in results]
        print(
            json.dumps(
                {
                    "resuming": str(target),
                    "skipped_sequences": list(results),
                    "remaining_sequences": sequences,
                }
            ),
            flush=True,
        )
    for sequence in sequences:
        print(f"evaluating {args.dataset}/{sequence}", flush=True)
        results[sequence] = evaluate_dataset(
            model,
            factory(sequence),
            device,
            evaluation["voxel_size_m"],
            evaluation["f1_thresholds_m"],
            include_geometry=not args.navigation_only,
            bf16=args.bf16,
            progress_label=f"{args.dataset}/{sequence}",
        )
        model.assert_backbones_intact()
        print(json.dumps(results[sequence], indent=2), flush=True)
        _write_results(
            target,
            args.dataset,
            args.navigation_only,
            "bf16" if args.bf16 else "fp32",
            "flashinfer" if args.flashinfer else "sdpa",
            args.bounded_cache_window,
            results,
            complete=False,
        )
    _write_results(
        target,
        args.dataset,
        args.navigation_only,
        "bf16" if args.bf16 else "fp32",
        "flashinfer" if args.flashinfer else "sdpa",
        args.bounded_cache_window,
        results,
        complete=True,
    )


if __name__ == "__main__":
    main()
