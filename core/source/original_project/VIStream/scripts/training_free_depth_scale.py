#!/usr/bin/env python3
"""GT-free local scale correction from frozen depth ground-plane distance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def poses_4x4(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1:] == (4, 4):
        return values.copy()
    output = np.repeat(np.eye(4)[None], len(values), axis=0)
    output[:, :3, :4] = values
    return output


def interpolate_log_height(rows: list[dict], count: int) -> tuple[np.ndarray, np.ndarray]:
    valid_indices = np.asarray(
        [index for index, row in enumerate(rows) if row.get("valid")], dtype=np.int64
    )
    if not len(valid_indices):
        raise ValueError("ground-plane artifact has no valid frames")
    heights = np.asarray(
        [rows[index]["distance"] for index in valid_indices], dtype=np.float64
    )
    log_height = np.interp(
        np.arange(count, dtype=np.float64), valid_indices, np.log(heights)
    )
    return log_height, valid_indices


def depth_scale_candidate(
    visual: np.ndarray,
    ground_plane: dict,
    smoothing_frames: float,
    minimum_scale: float,
    maximum_scale: float,
    minimum_valid_fraction: float,
    maximum_relative_mad: float,
    maximum_path_change_ratio: float,
) -> tuple[np.ndarray, dict]:
    rows = ground_plane["rows"]
    if len(rows) != len(visual):
        raise ValueError("ground-plane and visual frame counts differ")
    valid_fraction = float(ground_plane["valid_fraction"])
    relative_mad = float(ground_plane["relative_mad"])
    if valid_fraction < minimum_valid_fraction or relative_mad > maximum_relative_mad:
        return visual.copy(), {
            "accepted": False,
            "fallback_reason": "ground_plane_reliability_rejected",
            "valid_fraction": valid_fraction,
            "relative_mad": relative_mad,
        }
    log_height, valid_indices = interpolate_log_height(rows, len(visual))
    if smoothing_frames > 0.0 and len(log_height) > 1:
        log_height = gaussian_filter1d(log_height, smoothing_frames, mode="nearest")
    reference = float(np.median(log_height[valid_indices]))
    frame_scale = np.clip(
        np.exp(reference - log_height), minimum_scale, maximum_scale
    )
    interval_scale = np.sqrt(frame_scale[:-1] * frame_scale[1:])
    output = visual.copy()
    output[0] = visual[0]
    visual_steps = np.diff(visual[:, :3, 3], axis=0)
    for index, (step, scale) in enumerate(zip(visual_steps, interval_scale)):
        output[index + 1, :3, 3] = output[index, :3, 3] + scale * step
    path_scale = max(
        float(np.linalg.norm(visual[-1, :3, 3] - visual[0, :3, 3])),
        0.1 * float(np.sum(np.linalg.norm(visual_steps, axis=1))),
        1e-9,
    )
    path_change = float(
        np.sqrt(np.mean(np.sum((output[:, :3, 3] - visual[:, :3, 3]) ** 2, axis=1)))
        / path_scale
    )
    accepted = path_change <= maximum_path_change_ratio
    if not accepted:
        output = visual.copy()
    return output, {
        "accepted": accepted,
        "fallback_reason": None if accepted else "path_change_budget_exceeded",
        "valid_fraction": valid_fraction,
        "relative_mad": relative_mad,
        "frame_scale_quantiles": np.quantile(
            frame_scale, [0.05, 0.25, 0.5, 0.75, 0.95]
        ).tolist(),
        "relative_path_change": path_change,
    }


def generate(args: argparse.Namespace) -> None:
    policy = json.loads(args.policy.read_text())
    with np.load(args.trajectory, allow_pickle=False) as archive:
        visual = poses_4x4(archive[args.pose_key])
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
    ground_plane = json.loads(args.ground_plane.read_text())
    candidate, diagnostics = depth_scale_candidate(
        visual,
        ground_plane,
        policy["smoothing_frames"],
        policy["minimum_scale"],
        policy["maximum_scale"],
        policy["minimum_valid_fraction"],
        policy["maximum_relative_mad"],
        policy["maximum_path_change_ratio"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidate_before_gt.npz"
    np.savez_compressed(
        candidate_path,
        c2w=candidate,
        frame_indices=frame_indices,
    )
    diagnostics_path = args.output_dir / "depth_scale_before_gt.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    manifest = {
        "protocol": "frozen-depth scale candidate generated before GT load",
        "trajectory": str(args.trajectory),
        "trajectory_sha256": sha256(args.trajectory),
        "ground_plane": str(args.ground_plane),
        "ground_plane_sha256": sha256(args.ground_plane),
        "policy": policy,
        "policy_path": str(args.policy.resolve()),
        "policy_sha256": sha256(args.policy),
        "candidate": str(candidate_path),
        "candidate_sha256": sha256(candidate_path),
        "diagnostics": str(diagnostics_path),
        "diagnostics_sha256": sha256(diagnostics_path),
    }
    (args.output_dir / "manifest_before_gt.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(diagnostics, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    from vistream.metrics import ate_rmse

    manifest = json.loads((args.output_dir / "manifest_before_gt.json").read_text())
    for path_key, hash_key in (
        ("trajectory", "trajectory_sha256"),
        ("ground_plane", "ground_plane_sha256"),
        ("candidate", "candidate_sha256"),
        ("diagnostics", "diagnostics_sha256"),
    ):
        path = Path(manifest[path_key])
        if sha256(path) != manifest[hash_key]:
            raise ValueError(f"artifact changed after generation: {path}")
    with np.load(manifest["candidate"], allow_pickle=False) as archive:
        candidate = poses_4x4(archive["c2w"])
    # GT is first opened only after the complete manifest has been verified.
    with np.load(manifest["trajectory"], allow_pickle=False) as archive:
        visual = poses_4x4(archive[args.pose_key])
        target = poses_4x4(archive[args.ground_truth_key])
    result = {
        "visual_ate_rmse_m": ate_rmse(
            visual[:, :3, 3], target[:, :3, 3], "sim3"
        )["ate_rmse"],
        "candidate_ate_rmse_m": ate_rmse(
            candidate[:, :3, 3], target[:, :3, 3], "sim3"
        )["ate_rmse"],
    }
    (args.output_dir / "summary_after_gt.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--trajectory", type=Path, required=True)
    generate_parser.add_argument("--ground-plane", type=Path, required=True)
    generate_parser.add_argument("--policy", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--pose-key", default="predicted_c2w")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--pose-key", default="predicted_c2w")
    evaluate_parser.add_argument("--ground-truth-key", default="ground_truth_c2w")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generate(arguments) if arguments.command == "generate" else evaluate(arguments)
