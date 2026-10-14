#!/usr/bin/env python3
"""Fuse a frozen Lingbot trajectory with an arbitrary IMU processor.

The GT-free fused result is always written before an optional ground-truth
archive is opened. This makes the artifact suitable for strict held-out
evaluation: the processor and fusion settings cannot inspect GT while
producing a trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

# Match the project wrapper when this script is invoked directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vistream.metrics import ate_rmse
from vistream.training_free import (
    FusionConfig,
    NPZIMUProcessor,
    RawIMUProcessor,
    TrainingFreeIMUFusion,
    load_processor,
)
from vistream.training_free.io import (
    load_normalized_imu_csv,
    load_normalized_imu_npz,
    load_trajectory_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poses-key", default="predicted_c2w")
    parser.add_argument("--timestamps-key", default=None)
    parser.add_argument("--timestamp-unit", default="auto", choices=("auto", "s", "ms", "us", "ns"))
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--imu-csv", type=Path)
    inputs.add_argument("--imu-npz", type=Path, help="Normalized samples or processed NPZ output")
    parser.add_argument("--imu-order", default="gyro_first", choices=("gyro_first", "accel_first"))
    parser.add_argument("--imu-timestamp-unit", default="auto", choices=("auto", "s", "ms", "us", "ns"))
    parser.add_argument(
        "--processor",
        default="raw",
        help="raw, npz:/path, or python.module:factory. The custom processor receives raw normalized IMU.",
    )
    parser.add_argument("--rotation-camera-from-imu", type=Path, default=None, help="3x3 .npy rotation")
    parser.add_argument("--config", type=Path, default=None, help="JSON object overriding FusionConfig")
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--ground-truth-poses-key", default="ground_truth_c2w")
    parser.add_argument("--ground-truth-timestamps-key", default=None)
    return parser.parse_args()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _as_4x4(poses: np.ndarray) -> np.ndarray:
    if poses.shape[1:] == (4, 4):
        return poses
    result = np.repeat(np.eye(4)[None], len(poses), axis=0)
    result[:, :3, :4] = poses
    return result


def _evaluate(output: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Open ground truth only after the GT-free output archive exists."""
    if args.ground_truth is None:
        return {"evaluated": False}
    visual, timestamps = load_trajectory_npz(
        args.trajectory, args.poses_key, args.timestamps_key, args.timestamp_unit
    )
    ground_truth, ground_timestamps = load_trajectory_npz(
        args.ground_truth,
        args.ground_truth_poses_key,
        args.ground_truth_timestamps_key,
        args.timestamp_unit,
    )
    if len(ground_truth) != len(visual) or not np.allclose(ground_timestamps, timestamps):
        raise ValueError("ground truth must have the same timestamps as the visual trajectory")
    with np.load(output, allow_pickle=False) as archive:
        fused = np.asarray(archive["fused_c2w"], dtype=np.float64)
    valid = np.isfinite(ground_truth).all(axis=(1, 2))
    if np.count_nonzero(valid) < 3:
        raise ValueError("ground truth has fewer than three finite poses")
    visual_positions = _as_4x4(visual)[valid, :3, 3]
    fused_positions = fused[valid, :3, 3]
    target_positions = _as_4x4(ground_truth)[valid, :3, 3]
    result = {
        "evaluated": True,
        "frames": int(np.count_nonzero(valid)),
        "visual": {
            "ate_se3": ate_rmse(visual_positions, target_positions, "se3"),
            "ate_sim3": ate_rmse(visual_positions, target_positions, "sim3"),
        },
        "training_free": {
            "ate_se3": ate_rmse(fused_positions, target_positions, "se3"),
            "ate_sim3": ate_rmse(fused_positions, target_positions, "sim3"),
        },
    }
    for group in ("visual", "training_free"):
        for key in ("ate_se3", "ate_sim3"):
            result[group][key].pop("aligned")
    return result


def main() -> None:
    args = parse_args()
    visual, timestamps = load_trajectory_npz(
        args.trajectory, args.poses_key, args.timestamps_key, args.timestamp_unit
    )
    if args.imu_csv:
        raw_samples = load_normalized_imu_csv(
            args.imu_csv, args.imu_order, args.imu_timestamp_unit
        )
        processor = RawIMUProcessor() if args.processor == "raw" else load_processor(args.processor)
    else:
        raw_samples = np.empty((0, 7), dtype=np.float64)
        processor = NPZIMUProcessor(args.imu_npz) if args.processor == "raw" else load_processor(args.processor)
        if args.processor != "raw":
            raw_samples = load_normalized_imu_npz(args.imu_npz)
    rotation = np.eye(3)
    if args.rotation_camera_from_imu is not None:
        rotation = np.asarray(np.load(args.rotation_camera_from_imu), dtype=np.float64)
    overrides = {} if args.config is None else json.loads(args.config.read_text())
    result = TrainingFreeIMUFusion(FusionConfig(**overrides)).fuse(
        visual,
        timestamps,
        raw_samples,
        processor,
        rotation,
        {"trajectory": str(args.trajectory), "imu_input": str(args.imu_csv or args.imu_npz)},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        timestamps=timestamps,
        visual_c2w=_as_4x4(visual),
        fused_c2w=result.camera_poses_c2w,
        interval_gates=result.interval_gates,
    )
    diagnostics_path = args.output.with_suffix(".json")
    diagnostics = dict(result.diagnostics)
    diagnostics["candidate_path"] = str(args.output)
    diagnostics["ground_truth_loaded_after_candidate_write"] = args.ground_truth is not None
    diagnostics["evaluation"] = _evaluate(args.output, args)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, default=_json_value) + "\n")
    print(json.dumps({
        "candidate": str(args.output),
        "diagnostics": str(diagnostics_path),
        "accepted": result.diagnostics["accepted"],
        "evaluation": diagnostics["evaluation"],
    }, default=_json_value), flush=True)


if __name__ == "__main__":
    main()
