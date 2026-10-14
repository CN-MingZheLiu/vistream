#!/usr/bin/env python3
"""Two-phase training-free IMU evaluation on all available frozen caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from vistream.metrics import ate_rmse
from vistream.training_free import FusionConfig, RawIMUProcessor, TrainingFreeIMUFusion

from evaluate_training_free_cached import (
    _ensure_coverage,
    _native_imu,
    _oxford_camera_from_imu,
    _oxford_imu,
    _poses_4x4,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def oxford_sequence_from_image_path(path: str) -> str:
    matches = [part for part in Path(path).parts if part.startswith("2024-")]
    if len(matches) != 1:
        raise ValueError(f"cannot identify Oxford sequence from {path}")
    return matches[0]


def tum_sequence_from_cache(path: Path) -> str:
    for room in (2, 4, 6):
        if f"room{room}" in path.parent.name:
            return f"dataset-room{room}_512_16"
    raise ValueError(f"cannot identify TUM-VI sequence from {path}")


def _write_candidate(
    path: Path, timestamps: np.ndarray, visual: np.ndarray, fused: np.ndarray,
    diagnostics: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, timestamps=timestamps, visual_c2w=visual, fused_c2w=fused
    )
    path.with_suffix(".diagnostics.json").write_text(
        json.dumps(
            diagnostics,
            indent=2,
            default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value,
        )
        + "\n"
    )


def _generate_oxford(
    cache: Path, dataset_root: Path, config: FusionConfig, output: Path
) -> dict[str, Any]:
    with np.load(cache, allow_pickle=False) as archive:
        raw_visual = _poses_4x4(archive["predicted_c2w"])
        raw_timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
        sequence = oxford_sequence_from_image_path(str(archive["image_paths"][0]))
    visual = np.linalg.inv(raw_visual)
    origin = float(raw_timestamps[0])
    timestamps = raw_timestamps - origin
    imu_path = dataset_root / "sequences" / sequence / "raw/imu.csv"
    imu = _ensure_coverage(_oxford_imu(imu_path, origin), timestamps)
    calibration = dataset_root / "calibration/cam-lidar-imu.yaml"
    rotation = _oxford_camera_from_imu(calibration)[:3, :3]
    result = TrainingFreeIMUFusion(config).fuse(
        visual, timestamps, imu, RawIMUProcessor(), rotation
    )
    candidate = output / "oxford" / sequence / "candidate_before_gt.npz"
    _write_candidate(candidate, timestamps, visual, result.camera_poses_c2w, result.diagnostics)
    return {
        "dataset": "oxford",
        "sequence": sequence,
        "cache": str(cache),
        "candidate": str(candidate),
        "candidate_sha256": sha256(candidate),
        "imu": str(imu_path),
        "accepted": result.diagnostics["accepted"],
    }


def _generate_tum(
    cache: Path, dataset_root: Path, config: FusionConfig, output: Path
) -> dict[str, Any]:
    sequence = tum_sequence_from_cache(cache)
    with np.load(cache, allow_pickle=False) as archive:
        visual = _poses_4x4(archive["predicted_c2w"])
        timestamps_ns = np.asarray(archive["timestamps_ns"], dtype=np.int64)
    timestamps = (timestamps_ns - timestamps_ns[0]) * 1e-9
    sequence_root = dataset_root / sequence
    transform = np.asarray(
        yaml.safe_load((sequence_root / "dso/camchain.yaml").read_text())["cam0"]["T_cam_imu"],
        dtype=np.float64,
    )
    imu_path = sequence_root / "mav0/imu0/data.csv"
    imu = _ensure_coverage(_native_imu(imu_path, int(timestamps_ns[0])), timestamps)
    result = TrainingFreeIMUFusion(config).fuse(
        visual, timestamps, imu, RawIMUProcessor(), transform[:3, :3]
    )
    candidate = output / "tumvi" / sequence / "candidate_before_gt.npz"
    _write_candidate(candidate, timestamps, visual, result.camera_poses_c2w, result.diagnostics)
    return {
        "dataset": "tumvi",
        "sequence": sequence,
        "cache": str(cache),
        "candidate": str(candidate),
        "candidate_sha256": sha256(candidate),
        "imu": str(imu_path),
        "accepted": result.diagnostics["accepted"],
    }


def generate(args: argparse.Namespace) -> None:
    config = FusionConfig(**json.loads(args.config.read_text()))
    oxford_caches = sorted(args.lingbot_root.glob("oxford_all_current_long_3840/*/trajectory.npz"))
    tum_caches = sorted(args.lingbot_root.glob("tumvi_improved_room[246]_rectified/trajectory.npz"))
    if len(oxford_caches) != 10 or len(tum_caches) != 3:
        raise RuntimeError(
            f"expected 10 Oxford and 3 TUM-VI caches, got {len(oxford_caches)} and {len(tum_caches)}"
        )
    entries = []
    for cache in oxford_caches:
        entry = _generate_oxford(cache, args.oxford_root, config, args.output_dir)
        entries.append(entry)
        print(json.dumps(entry), flush=True)
    for cache in tum_caches:
        entry = _generate_tum(cache, args.tum_root, config, args.output_dir)
        entries.append(entry)
        print(json.dumps(entry), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "all suite candidates persisted before any ground-truth key is opened",
        "config": config.__dict__,
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "entries": entries,
    }
    (args.output_dir / "manifest_before_gt.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def _metric(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(target).all(axis=1)
    metric = ate_rmse(predicted[valid], target[valid], "sim3")
    return {
        "frames": int(np.sum(valid)),
        "ate_rmse_m": metric["ate_rmse"],
        "ate_mean_m": metric["ate_mean"],
    }


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
        # Ground-truth keys are opened only in this evaluation phase.
        with np.load(entry["cache"], allow_pickle=False) as archive:
            target = (
                archive["ground_truth_c2w"][:, :3, 3]
                if entry["dataset"] == "oxford"
                else archive["ground_truth_positions"]
            )
        rows.append({
            "dataset": entry["dataset"],
            "sequence": entry["sequence"],
            "accepted": entry["accepted"],
            "visual": _metric(visual, target),
            "training_free": _metric(fused, target),
        })
    macro = {}
    for dataset in ("oxford", "tumvi"):
        selected = [row for row in rows if row["dataset"] == dataset]
        visual = np.asarray([row["visual"]["ate_rmse_m"] for row in selected])
        fused = np.asarray([row["training_free"]["ate_rmse_m"] for row in selected])
        macro[dataset] = {
            "sequences": len(selected),
            "visual_ate_rmse_mean_m": float(visual.mean()),
            "training_free_ate_rmse_mean_m": float(fused.mean()),
            "improved_sequences": int(np.sum(fused < visual)),
        }
    summary = {"protocol": manifest["protocol"], "rows": rows, "macro": macro}
    (args.output_dir / "summary_after_gt.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--lingbot-root", type=Path, required=True)
    generate_parser.add_argument("--oxford-root", type=Path, required=True)
    generate_parser.add_argument("--tum-root", type=Path, required=True)
    generate_parser.add_argument("--config", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args) if args.command == "generate" else evaluate(args)


if __name__ == "__main__":
    main()
