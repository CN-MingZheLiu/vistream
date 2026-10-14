#!/usr/bin/env python3
"""GT-free consensus over frozen Lingbot trajectories at fixed input strides."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import interp1d


SEQUENCES = tuple(f"{index:02d}" for index in range(11))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_center = source - source.mean(axis=0)
    target_center = target - target.mean(axis=0)
    covariance = target_center.T @ source_center / len(source)
    left, singular, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right) < 0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right
    variance = max(float(np.mean(np.sum(source_center**2, axis=1))), 1e-12)
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = target.mean(axis=0) - scale * (rotation @ source.mean(axis=0))
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, transform: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = transform
    return scale * (points @ rotation.T) + translation


def interpolate_positions(
    source_frames: np.ndarray, source_positions: np.ndarray, target_frames: np.ndarray
) -> np.ndarray:
    return interp1d(
        source_frames,
        source_positions,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=(source_positions[0], source_positions[-1]),
        assume_sorted=True,
    )(target_frames)


def robust_consensus(trajectories: np.ndarray, iterations: int = 5) -> tuple[np.ndarray, dict]:
    """Generalized Procrustes followed by an IRLS geometric consensus."""
    reference = trajectories[len(trajectories) // 2].copy()
    aligned = trajectories.copy()
    rms_history = []
    for _ in range(iterations):
        aligned = np.asarray([
            apply_similarity(trajectory, similarity(trajectory, reference))
            for trajectory in trajectories
        ])
        median = np.median(aligned, axis=0)
        residual = np.linalg.norm(aligned - median[None], axis=2)
        sequence_scale = np.maximum(np.median(residual, axis=1), 1e-6)
        weights = 1.0 / (1.0 + (residual / (3.0 * sequence_scale[:, None])) ** 2)
        reference = np.sum(weights[..., None] * aligned, axis=0) / np.sum(
            weights, axis=0
        )[:, None]
        rms_history.append(float(np.sqrt(np.mean((aligned - reference[None]) ** 2))))
    return reference, {
        "iterations": iterations,
        "alignment_rms_history": rms_history,
        "per_stride_median_residual": np.median(
            np.linalg.norm(aligned - reference[None], axis=2), axis=1
        ).tolist(),
    }


def load_visual(path: Path, pose_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        poses = np.asarray(archive[pose_key], dtype=np.float64)
        frames = np.asarray(archive["frame_indices"], dtype=np.int64)
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
    return poses, frames, timestamps


def generate(args: argparse.Namespace) -> None:
    strides = tuple(int(value) for value in args.strides.split(","))
    if len(strides) < 3 or len(set(strides)) != len(strides):
        raise ValueError("--strides must contain at least three unique values")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for sequence in SEQUENCES:
        sources = []
        for stride in strides:
            path = args.input_root / f"stride_{stride:02d}" / sequence / "trajectory.npz"
            if not path.exists() and stride == 6 and args.stride6_root is not None:
                path = args.stride6_root / sequence / "trajectory.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            poses, frames, timestamps = load_visual(path, args.pose_key)
            sources.append((stride, path, poses, frames, timestamps))
        target_frames = np.unique(np.concatenate([item[3] for item in sources]))
        interpolated = np.asarray([
            interpolate_positions(frames, poses[:, :3, 3], target_frames)
            for _, _, poses, frames, _ in sources
        ])
        positions, diagnostics = robust_consensus(interpolated, args.iterations)
        anchor = min(sources, key=lambda item: abs(item[0] - args.anchor_stride))
        anchor_poses, anchor_frames = anchor[2], anchor[3]
        output_poses = np.repeat(np.eye(4)[None], len(target_frames), axis=0)
        output_poses[:, :3, 3] = positions
        # Rotation remains from the fixed anchor. The generic IMU backend may
        # replace it later without changing the consensus translation.
        nearest = np.searchsorted(anchor_frames, target_frames, side="left")
        nearest = np.clip(nearest, 0, len(anchor_frames) - 1)
        previous = np.maximum(nearest - 1, 0)
        use_previous = np.abs(target_frames - anchor_frames[previous]) < np.abs(
            target_frames - anchor_frames[nearest]
        )
        nearest[use_previous] = previous[use_previous]
        output_poses[:, :3, :3] = anchor_poses[nearest, :3, :3]
        target_timestamps = np.interp(target_frames, anchor_frames, anchor[4])
        output = args.output_dir / sequence
        output.mkdir(parents=True, exist_ok=True)
        candidate = output / "candidate_before_gt.npz"
        np.savez_compressed(
            candidate,
            c2w=output_poses,
            frame_indices=target_frames,
            timestamps=target_timestamps,
        )
        diagnostic_path = output / "consensus_before_gt.json"
        diagnostic_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
        entries.append({
            "sequence": sequence,
            "candidate": str(candidate),
            "candidate_sha256": sha256(candidate),
            "diagnostics": str(diagnostic_path),
            "sources": [
                {"stride": stride, "path": str(path), "sha256": sha256(path)}
                for stride, path, *_ in sources
            ],
        })
        print(json.dumps({"sequence": sequence, "frames": len(target_frames)}), flush=True)
    manifest = {
        "protocol": "all fixed-stride consensus candidates written before GT load",
        "strides": list(strides),
        "pose_key": args.pose_key,
        "anchor_stride": args.anchor_stride,
        "iterations": args.iterations,
        "entries": entries,
    }
    (args.output_dir / "manifest_before_gt.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def evaluate(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.project_root.resolve()))
    from vistream.metrics import ate_rmse

    manifest = json.loads((args.output_dir / "manifest_before_gt.json").read_text())
    rows = []
    for entry in manifest["entries"]:
        candidate_path = Path(entry["candidate"])
        if sha256(candidate_path) != entry["candidate_sha256"]:
            raise ValueError(f"candidate changed after generation: {candidate_path}")
        with np.load(candidate_path, allow_pickle=False) as archive:
            predicted = archive["c2w"][:, :3, 3]
            frames = archive["frame_indices"].astype(np.int64)
        source = Path(entry["sources"][0]["path"])
        with np.load(source, allow_pickle=False) as archive:
            source_frames = archive["frame_indices"].astype(np.int64)
            source_gt = archive[args.ground_truth_key][:, :3, 3]
        target = interpolate_positions(source_frames, source_gt, frames)
        metric = ate_rmse(predicted, target, "sim3")
        rows.append({
            "sequence": entry["sequence"],
            "frames": len(frames),
            "ate_rmse_m": metric["ate_rmse"],
            "ate_mean_m": metric["ate_mean"],
        })
    summary = {
        "protocol": manifest["protocol"],
        "strides": manifest["strides"],
        "rows": rows,
        "macro_ate_rmse_mean_m": float(np.mean([row["ate_rmse_m"] for row in rows])),
    }
    (args.output_dir / "summary_after_gt.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--input-root", type=Path, required=True)
    generate_parser.add_argument("--stride6-root", type=Path)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--strides", default="1,2,3,4,5,6,7,8,9,10,11,12")
    generate_parser.add_argument("--pose-key", default="predicted_c2w")
    generate_parser.add_argument("--anchor-stride", type=int, default=6)
    generate_parser.add_argument("--iterations", type=int, default=5)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--project-root", type=Path, required=True)
    evaluate_parser.add_argument("--ground-truth-key", default="ground_truth_c2w")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args) if args.command == "generate" else evaluate(args)


if __name__ == "__main__":
    main()
