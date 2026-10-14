#!/usr/bin/env python3
"""Summarize the frozen LingBot-Map KITTI and Oxford evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from evaluate_kitti_odometry import (
    aligned_poses,
    kitti_segment_metrics,
    position_metrics,
    relative_pose_metrics,
    umeyama,
)


KITTI_SEQUENCES = [f"{index:02d}" for index in range(11)]
OXFORD_SEQUENCES = [
    "keble-college-02",
    "keble-college-03",
    "keble-college-04",
    "keble-college-05",
    "observatory-quarter-01",
    "observatory-quarter-02",
    "christ-church-02",
    "christ-church-03",
    "christ-church-05",
    "bodleian-library-02",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent / "outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kitti-root", type=Path, default=root / "kitti_all_current_long_stride6"
    )
    parser.add_argument(
        "--oxford-root", type=Path, default=root / "oxford_all_current_long_3840"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "full_current_long_summary"
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def evaluate_poses(predicted: np.ndarray, gt: np.ndarray, kitti: bool) -> dict[str, object]:
    pred_position = predicted[:, :3, 3]
    gt_position = gt[:, :3, 3]
    se3_position, se3_rotation, _, se3_translation = umeyama(
        pred_position, gt_position, with_scale=False
    )
    sim3_position, sim3_rotation, sim3_scale, sim3_translation = umeyama(
        pred_position, gt_position, with_scale=True
    )
    se3 = aligned_poses(predicted, se3_rotation, 1.0, se3_translation)
    sim3 = aligned_poses(predicted, sim3_rotation, sim3_scale, sim3_translation)
    output: dict[str, object] = {
        "ate_se3_aligned": position_metrics(se3_position, gt_position),
        "ate_sim3_aligned": position_metrics(sim3_position, gt_position),
        "sim3_scale": float(sim3_scale),
        "rpe_se3_aligned": {
            "delta_1": relative_pose_metrics(se3, gt, 1),
            "delta_10": relative_pose_metrics(se3, gt, 10),
        },
        "rpe_sim3_aligned": {
            "delta_1": relative_pose_metrics(sim3, gt, 1),
            "delta_10": relative_pose_metrics(sim3, gt, 10),
        },
    }
    if kitti:
        output["kitti_segment_drift_sim3_scale"] = kitti_segment_metrics(sim3, gt)
    return output


def load_kitti(root: Path, allow_incomplete: bool) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    details: list[dict] = []
    for sequence in KITTI_SEQUENCES:
        directory = root / sequence
        metrics_path = directory / "metrics.json"
        trajectory_path = directory / "trajectory.npz"
        if not metrics_path.is_file() or not trajectory_path.is_file():
            if allow_incomplete:
                continue
            raise FileNotFoundError(f"Incomplete KITTI {sequence}: {directory}")
        with metrics_path.open(encoding="utf-8") as handle:
            source_metrics = json.load(handle)
        if source_metrics.get("pose_convention") != "opencv_w2c":
            raise ValueError(f"KITTI {sequence} source is not opencv_w2c")
        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            opencv = trajectory["predicted_c2w"].astype(np.float64)
            gt = trajectory["ground_truth_c2w"].astype(np.float64)
        conventions = {
            "opencv_w2c": evaluate_poses(opencv, gt, True),
            "benchmark_c2w": evaluate_poses(np.linalg.inv(opencv), gt, True),
        }
        dual = {
            "dataset": "KITTI Odometry",
            "sequence": sequence,
            "checkpoint": source_metrics["checkpoint"],
            "checkpoint_sha256": source_metrics["checkpoint_sha256"],
            "num_raw_frames": source_metrics["num_raw_frames"],
            "num_input_frames": source_metrics["num_input_frames"],
            "input_frame_stride": source_metrics["input_frame_stride"],
            "keyframe_interval": source_metrics["keyframe_interval"],
            "inference_mode": source_metrics["inference_mode"],
            "dtype": source_metrics["dtype"],
            "timing": source_metrics["timing"],
            "pose_convention_results": conventions,
        }
        with (directory / "metrics_dual.json").open("w", encoding="utf-8") as handle:
            json.dump(dual, handle, indent=2)
        details.append(dual)
        for convention, result in conventions.items():
            drift = result["kitti_segment_drift_sim3_scale"]
            rows.append({
                "dataset": "KITTI",
                "sequence": sequence,
                "convention": convention,
                "input_frames": source_metrics["num_input_frames"],
                "evaluation_frames": source_metrics["num_evaluation_frames"],
                "se3_ate_rmse_m": result["ate_se3_aligned"]["rmse_m"],
                "sim3_ate_rmse_m": result["ate_sim3_aligned"]["rmse_m"],
                "sim3_scale": result["sim3_scale"],
                "translation_drift_percent": drift["translation_percent"],
                "rotation_drift_deg_per_100m": drift["rotation_deg_per_100m"],
                "inference_seconds": source_metrics["timing"]["inference_seconds"],
            })
    return rows, details


def load_oxford(root: Path, allow_incomplete: bool) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    details: list[dict] = []
    for sequence in OXFORD_SEQUENCES:
        path = root / sequence / "metrics.json"
        if not path.is_file():
            if allow_incomplete:
                continue
            raise FileNotFoundError(f"Incomplete Oxford {sequence}: {path}")
        with path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        details.append(metrics)
        for convention, result in metrics["pose_convention_results"].items():
            rows.append({
                "dataset": "Oxford Spires",
                "sequence": sequence,
                "convention": convention,
                "input_frames": metrics["frames"],
                "evaluation_frames": metrics.get("frames_evaluated", metrics["frames"]),
                "se3_ate_rmse_m": result["ate_se3_aligned"]["rmse_m"],
                "sim3_ate_rmse_m": result["ate_sim3_aligned"]["rmse_m"],
                "sim3_scale": result["sim3_scale"],
                "translation_drift_percent": math.nan,
                "rotation_drift_deg_per_100m": math.nan,
                "inference_seconds": metrics["timing"]["inference_seconds"],
            })
    return rows, details


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    keys = sorted({(row["dataset"], row["convention"]) for row in rows})
    for dataset, convention in keys:
        selected = [row for row in rows if row["dataset"] == dataset and row["convention"] == convention]
        se3 = np.asarray([row["se3_ate_rmse_m"] for row in selected], dtype=np.float64)
        sim3 = np.asarray([row["sim3_ate_rmse_m"] for row in selected], dtype=np.float64)
        output.append({
            "dataset": dataset,
            "convention": convention,
            "sequences": len(selected),
            "macro_mean_se3_ate_rmse_m": float(np.mean(se3)),
            "macro_median_se3_ate_rmse_m": float(np.median(se3)),
            "macro_mean_sim3_ate_rmse_m": float(np.mean(sim3)),
            "macro_median_sim3_ate_rmse_m": float(np.median(sim3)),
        })
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kitti_rows, kitti_details = load_kitti(args.kitti_root, args.allow_incomplete)
    oxford_rows, oxford_details = load_oxford(args.oxford_root, args.allow_incomplete)
    rows = kitti_rows + oxford_rows
    fieldnames = list(rows[0])
    with (args.output_dir / "per_sequence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "kitti_root": str(args.kitti_root),
        "oxford_root": str(args.oxford_root),
        "per_sequence": rows,
        "aggregate": aggregate(rows),
        "completed": {
            "kitti": len(kitti_details),
            "oxford": len(oxford_details),
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"completed": summary["completed"], "aggregate": summary["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
