#!/usr/bin/env python3
"""Two-phase raw-IMU fusion evaluation on a cached KITTI stream."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vistream.metrics import ate_rmse
from vistream.training_free import FusionConfig, RawIMUProcessor, TrainingFreeIMUFusion
from evaluate_training_free_cached import _ensure_coverage, _poses_4x4


SEQUENCES = tuple(f"{index:02d}" for index in range(11))
DATE_BY_SEQUENCE = {
    "00": "2011_10_03",
    "01": "2011_10_03",
    "02": "2011_10_03",
    "03": "2011_09_26",
    **{f"{index:02d}": "2011_09_30" for index in range(4, 11)},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_projection(path: Path, key: str) -> np.ndarray:
    for line in path.read_text().splitlines():
        if line.startswith(f"{key}:"):
            return np.fromstring(line.split(":", 1)[1], sep=" ").reshape(3, 4)
    raise ValueError(f"missing {key} in {path}")


def read_rigid(path: Path) -> np.ndarray:
    values = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, data = line.split(":", 1)
            if key in {"R", "T"}:
                values[key] = np.fromstring(data, sep=" ")
    if values.get("R", np.empty(0)).size != 9 or values.get("T", np.empty(0)).size != 3:
        raise ValueError(f"invalid rigid calibration: {path}")
    transform = np.eye(4)
    transform[:3, :3] = values["R"].reshape(3, 3)
    transform[:3, 3] = values["T"]
    return transform


def camera_from_imu(odometry_calibration: Path, imu_to_velo: Path) -> np.ndarray:
    camera0_from_velo = np.eye(4)
    camera0_from_velo[:3, :4] = read_projection(odometry_calibration, "Tr")
    projection2 = read_projection(odometry_calibration, "P2")
    camera2_from_camera0 = np.eye(4)
    camera2_from_camera0[:3, 3] = np.linalg.solve(
        projection2[:, :3], projection2[:, 3]
    )
    return camera2_from_camera0 @ camera0_from_velo @ read_rigid(imu_to_velo)


def load_kitti_imu(path: Path, timestamps: np.ndarray) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",", dtype=np.float64)
    normalized = np.column_stack((raw[:, 0], raw[:, 15:18], raw[:, 21:24]))
    return _ensure_coverage(normalized, timestamps)


def write_candidate(
    path: Path, timestamps: np.ndarray, frame_indices: np.ndarray,
    visual: np.ndarray, fused: np.ndarray, diagnostics: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        timestamps=timestamps,
        frame_indices=frame_indices,
        visual_c2w=visual,
        fused_c2w=fused,
    )
    path.with_suffix(".diagnostics.json").write_text(
        json.dumps(
            diagnostics,
            indent=2,
            default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value,
        )
        + "\n"
    )


def generate(args: argparse.Namespace) -> None:
    config = FusionConfig(**json.loads(args.config.read_text()))
    entries = []
    for sequence in SEQUENCES:
        cache = args.cache_root / sequence / "trajectory.npz"
        with np.load(cache, allow_pickle=False) as archive:
            visual = _poses_4x4(archive[args.pose_key])
            timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
            frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        if args.visual_convention == "w2c":
            # Historical LingBot decoder caches store OpenCV world-to-camera.
            # Fusion and evaluation use camera-to-world poses throughout.
            visual = np.linalg.inv(visual)
        imu_path = args.imu_root / f"kitti_odo_{sequence}" / "imu.csv"
        calibration_path = args.calibration_root / DATE_BY_SEQUENCE[sequence] / "calib_imu_to_velo.txt"
        if imu_path.exists():
            imu = load_kitti_imu(imu_path, timestamps)
            transform = camera_from_imu(
                args.image_root / sequence / "calib.txt", calibration_path
            )
            result = TrainingFreeIMUFusion(config).fuse(
                visual,
                timestamps,
                imu,
                RawIMUProcessor(),
                transform[:3, :3],
                {"sequence": sequence, "imu": str(imu_path)},
            )
            fused = result.camera_poses_c2w
            diagnostics = result.diagnostics
            status = "fused"
        else:
            fused = visual.copy()
            diagnostics = {
                "training_free": True,
                "accepted": False,
                "fallback_reason": "missing_matching_imu",
                "sequence": sequence,
            }
            status = "visual_fallback"
        candidate = args.output_dir / sequence / "candidate_before_gt.npz"
        write_candidate(candidate, timestamps, frame_indices, visual, fused, diagnostics)
        entry = {
            "sequence": sequence,
            "cache": str(cache),
            "cache_sha256": sha256(cache),
            "candidate": str(candidate),
            "candidate_sha256": sha256(candidate),
            "imu": str(imu_path) if imu_path.exists() else None,
            "imu_sha256": sha256(imu_path) if imu_path.exists() else None,
            "calibration": str(calibration_path) if imu_path.exists() else None,
            "calibration_sha256": sha256(calibration_path) if imu_path.exists() else None,
            "status": status,
        }
        entries.append(entry)
        print(json.dumps(entry), flush=True)
    manifest = {
        "protocol": "all cached stream frames; all raw-IMU candidates before GT load",
        "pose_key": args.pose_key,
        "visual_convention": args.visual_convention,
        "config": config.__dict__,
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "entries": entries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest_before_gt.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def evaluate(args: argparse.Namespace) -> None:
    manifest = json.loads((args.output_dir / "manifest_before_gt.json").read_text())
    rows = []
    for entry in manifest["entries"]:
        candidate_path = Path(entry["candidate"])
        if sha256(candidate_path) != entry["candidate_sha256"]:
            raise ValueError(f"candidate changed after generation: {candidate_path}")
        with np.load(candidate_path, allow_pickle=False) as archive:
            visual = archive["visual_c2w"][:, :3, 3]
            fused = archive["fused_c2w"][:, :3, 3]
        # Ground truth is opened only after every candidate has been hashed.
        with np.load(entry["cache"], allow_pickle=False) as archive:
            target = archive[args.ground_truth_key][:, :3, 3]
        visual_metric = ate_rmse(visual, target, "sim3")
        fused_metric = ate_rmse(fused, target, "sim3")
        rows.append({
            "sequence": entry["sequence"],
            "status": entry["status"],
            "frames": len(target),
            "visual_ate_rmse_m": visual_metric["ate_rmse"],
            "training_free_ate_rmse_m": fused_metric["ate_rmse"],
            "visual_ate_mean_m": visual_metric["ate_mean"],
            "training_free_ate_mean_m": fused_metric["ate_mean"],
        })
    visual = np.asarray([row["visual_ate_rmse_m"] for row in rows])
    fused = np.asarray([row["training_free_ate_rmse_m"] for row in rows])
    summary = {
        "protocol": manifest["protocol"],
        "rows": rows,
        "macro": {
            "visual_ate_rmse_mean_m": float(visual.mean()),
            "training_free_ate_rmse_mean_m": float(fused.mean()),
            "improved_sequences": int(np.sum(fused < visual)),
            "sequences": len(rows),
        },
    }
    (args.output_dir / "summary_after_gt.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--cache-root", type=Path, required=True)
    generate_parser.add_argument("--image-root", type=Path, required=True)
    generate_parser.add_argument("--imu-root", type=Path, required=True)
    generate_parser.add_argument("--calibration-root", type=Path, required=True)
    generate_parser.add_argument("--config", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--pose-key", default="predicted_c2w")
    generate_parser.add_argument(
        "--visual-convention", choices=("c2w", "w2c"), default="c2w",
        help="Convention stored under --pose-key; convert W2C to C2W before fusion.",
    )
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--ground-truth-key", default="ground_truth_c2w")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args) if args.command == "generate" else evaluate(args)


if __name__ == "__main__":
    main()
