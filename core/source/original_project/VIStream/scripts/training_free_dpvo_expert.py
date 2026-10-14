#!/usr/bin/env python3
"""Generate and audit frozen DPVO trajectory experts without reading GT."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import subprocess
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_set_sha256(image_dir: Path, frame_indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    for frame in frame_indices:
        path = image_dir / f"{int(frame):06d}.png"
        digest.update(path.name.encode("ascii"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def tracking_schedule(
    target_frame_indices: np.ndarray,
    target_timestamps: np.ndarray,
    tracking_times: Path | None,
    tracking_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a dense tracking schedule and exact target-pose sample positions."""
    target_frame_indices = np.asarray(target_frame_indices, dtype=np.int64)
    target_timestamps = np.asarray(target_timestamps, dtype=np.float64)
    if len(target_frame_indices) != len(target_timestamps):
        raise ValueError("target frame indices and timestamps differ")
    if len(target_frame_indices) == 0:
        raise ValueError("empty target trajectory")
    if tracking_stride < 1:
        raise ValueError("tracking stride must be positive")
    if tracking_times is None:
        if tracking_stride != 1:
            raise ValueError("tracking stride requires --tracking-times")
        positions = np.arange(len(target_frame_indices), dtype=np.int64)
        return target_frame_indices.copy(), target_timestamps.copy(), positions

    all_timestamps = np.atleast_1d(
        np.loadtxt(tracking_times, dtype=np.float64)
    )
    first, last = int(target_frame_indices[0]), int(target_frame_indices[-1])
    if np.any(np.diff(target_frame_indices) <= 0):
        raise ValueError("target frame indices must be strictly increasing")
    if first < 0 or last >= len(all_timestamps):
        raise ValueError("tracking timestamps do not cover target frame indices")
    if not np.allclose(
        all_timestamps[target_frame_indices], target_timestamps, rtol=0.0, atol=5e-4
    ):
        raise ValueError("cached and tracking timestamps disagree at target frames")
    regular = np.arange(first, last + 1, tracking_stride, dtype=np.int64)
    tracking_frame_indices = np.unique(
        np.concatenate([regular, target_frame_indices])
    )
    positions = np.searchsorted(tracking_frame_indices, target_frame_indices)
    if not np.array_equal(tracking_frame_indices[positions], target_frame_indices):
        raise AssertionError("target frames are absent from tracking schedule")
    return (
        tracking_frame_indices,
        all_timestamps[tracking_frame_indices],
        positions,
    )


def dpvo_source_state(root: Path) -> tuple[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return commit, hashlib.sha256(diff).hexdigest()


def poses_4x4(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1:] == (4, 4):
        return values.copy()
    if values.shape[1:] != (3, 4):
        raise ValueError(f"expected [N,3,4] or [N,4,4], got {values.shape}")
    output = np.repeat(np.eye(4)[None], len(values), axis=0)
    output[:, :3, :4] = values
    return output


def read_intrinsics(path: Path) -> np.ndarray:
    for line in path.read_text().splitlines():
        if line.startswith("P2:"):
            projection = np.fromstring(line.split(":", 1)[1], sep=" ").reshape(3, 4)
            return np.asarray(
                [projection[0, 0], projection[1, 1], projection[0, 2], projection[1, 2]],
                dtype=np.float32,
            )
    raise ValueError(f"missing P2 in {path}")


def dpvo_pose_matrices(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"expected DPVO [N,7] poses, got {poses.shape}")
    output = np.repeat(np.eye(4)[None], len(poses), axis=0)
    output[:, :3, 3] = poses[:, :3]
    output[:, :3, :3] = Rotation.from_quat(poses[:, 3:7]).as_matrix()
    return output


def direction_transfer(
    visual: np.ndarray,
    source: np.ndarray,
    output_rotations: np.ndarray,
) -> np.ndarray:
    if len(visual) != len(source) or len(output_rotations) != len(visual):
        raise ValueError("visual, source, and output rotations must have equal length")
    visual_steps = np.diff(visual[:, :3, 3], axis=0)
    visual_norms = np.linalg.norm(visual_steps, axis=1)
    source_steps = np.diff(source[:, :3, 3], axis=0)
    source_local = np.einsum(
        "nij,nj->ni", source[:-1, :3, :3].transpose(0, 2, 1), source_steps
    )
    source_norms = np.linalg.norm(source_local, axis=1)
    output = visual.copy()
    output[:, :3, :3] = output_rotations
    output[0, :3, 3] = visual[0, :3, 3]
    for index, (direction, source_norm, visual_norm) in enumerate(
        zip(source_local, source_norms, visual_norms)
    ):
        if source_norm <= 1e-12 or visual_norm <= 1e-12:
            visual_local = visual[index, :3, :3].T @ visual_steps[index]
            direction = visual_local / max(float(np.linalg.norm(visual_local)), 1e-12)
        else:
            direction = direction / source_norm
        output[index + 1, :3, 3] = (
            output[index, :3, 3]
            + output_rotations[index] @ direction * visual_norm
        )
    return output


def path_statistics(poses: np.ndarray) -> dict[str, float]:
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    path_length = float(np.sum(steps))
    endpoint = float(np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3]))
    return {
        "path_length": path_length,
        "endpoint_distance": endpoint,
        "endpoint_to_path_ratio": endpoint / max(path_length, 1e-12),
        "median_step": float(np.median(steps)),
        "minimum_step": float(np.min(steps)),
        "maximum_step": float(np.max(steps)),
    }


def run_dpvo(
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
    image_dir: Path,
    calibration: Path,
    dpvo_root: Path,
    network: Path,
    config: Path,
    seed: int,
    loop_closure: bool,
    backend_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    import torch

    expected_package = (dpvo_root / "dpvo").resolve()
    for name in tuple(sys.modules):
        if name == "dpvo" or name.startswith("dpvo."):
            del sys.modules[name]
    sys.path.insert(0, str(dpvo_root.resolve()))
    importlib.invalidate_caches()
    package = importlib.import_module("dpvo")
    package_path = Path(package.__file__).resolve().parent
    if package_path != expected_package:
        raise ImportError(
            f"DPVO import resolved to {package_path}, expected {expected_package}"
        )
    from dpvo.config import cfg
    from dpvo.dpvo import DPVO

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cfg.defrost()
    cfg.merge_from_file(str(config))
    cfg.BUFFER_SIZE = max(int(cfg.BUFFER_SIZE), len(frame_indices) + 1)
    cfg.LOOP_CLOSURE = loop_closure
    cfg.BACKEND_THRESH = backend_threshold
    cfg.CLASSIC_LOOP_CLOSURE = False
    cfg.freeze()
    intrinsics = read_intrinsics(calibration)
    slam = None
    input_hw = None
    with torch.inference_mode():
        for timestamp, frame in zip(timestamps, frame_indices):
            path = image_dir / f"{int(frame):06d}.png"
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            height, width = image.shape[:2]
            height -= height % 16
            width -= width % 16
            image = image[:height, :width]
            if slam is None:
                input_hw = (height, width)
                slam = DPVO(cfg, str(network), ht=height, wd=width, viz=False)
            if (height, width) != input_hw:
                raise ValueError("DPVO input image sizes differ within one sequence")
            slam(
                float(timestamp),
                torch.as_tensor(image, device="cuda").permute(2, 0, 1),
                torch.as_tensor(intrinsics, dtype=torch.float32, device="cuda"),
            )
        if slam is None:
            raise ValueError("empty image sequence")
        poses, output_timestamps = slam.terminate()
    diagnostics = {
        "input_frames": len(frame_indices),
        "output_poses": len(poses),
        "initialized": bool(slam.is_initialized),
        "keyframes": int(slam.n),
        "input_hw": list(input_hw),
        "dpvo_root": str(dpvo_root),
        "buffer_size": int(cfg.BUFFER_SIZE),
        "seed": seed,
        "loop_closure": bool(cfg.LOOP_CLOSURE),
        "backend_threshold": float(cfg.BACKEND_THRESH),
    }
    return dpvo_pose_matrices(poses), np.asarray(output_timestamps), diagnostics


def generate(args: argparse.Namespace) -> None:
    actual_commit, actual_patch_sha256 = dpvo_source_state(args.dpvo_root)
    if actual_commit != args.dpvo_commit:
        raise ValueError(
            f"DPVO commit differs: expected {args.dpvo_commit}, got {actual_commit}"
        )
    if actual_patch_sha256 != sha256(args.dpvo_compat_patch):
        raise ValueError("DPVO working-tree diff differs from compatibility patch")
    with np.load(args.trajectory, allow_pickle=False) as archive:
        visual = poses_4x4(archive[args.pose_key])
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
    tracking_frame_indices, tracking_timestamps, sample_positions = tracking_schedule(
        frame_indices,
        timestamps,
        args.tracking_times,
        args.tracking_stride,
    )
    tracked_dpvo, output_timestamps, diagnostics = run_dpvo(
        tracking_frame_indices,
        tracking_timestamps,
        args.image_dir,
        args.calibration,
        args.dpvo_root,
        args.network,
        args.dpvo_config,
        args.seed,
        args.loop_closure,
        args.backend_threshold,
    )
    if len(tracked_dpvo) != len(tracking_frame_indices):
        raise ValueError("DPVO did not return one pose per tracking frame")
    if len(output_timestamps) != len(tracking_timestamps) or not np.allclose(
        output_timestamps, tracking_timestamps, rtol=0.0, atol=1e-9
    ):
        raise ValueError("DPVO output timestamps differ from tracking schedule")
    dpvo = tracked_dpvo[sample_positions]
    if len(dpvo) != len(visual):
        raise ValueError("sampled DPVO and Lingbot pose counts differ")
    diagnostics["target_frames"] = len(frame_indices)
    diagnostics["tracking_frames"] = len(tracking_frame_indices)
    diagnostics["tracking_stride"] = args.tracking_stride
    diagnostics["tracking_overlap_factor"] = float(
        len(tracking_frame_indices) / len(frame_indices)
    )
    visual_rotation = direction_transfer(visual, dpvo, visual[:, :3, :3])
    dpvo_rotation = direction_transfer(visual, dpvo, dpvo[:, :3, :3])
    diagnostics["visual"] = path_statistics(visual)
    diagnostics["dpvo"] = path_statistics(dpvo)
    diagnostics["dpvo_direction_visual_rotation"] = path_statistics(visual_rotation)
    diagnostics["dpvo_direction_dpvo_rotation"] = path_statistics(dpvo_rotation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidate_before_gt.npz"
    np.savez_compressed(
        candidate_path,
        frame_indices=frame_indices,
        timestamps=timestamps,
        dpvo_c2w=dpvo,
        dpvo_direction_visual_rotation_c2w=visual_rotation,
        dpvo_direction_dpvo_rotation_c2w=dpvo_rotation,
    )
    diagnostics_path = args.output_dir / "dpvo_before_gt.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    manifest = {
        "protocol": "frozen DPVO experts generated and hashed before GT load",
        "trajectory": str(args.trajectory),
        "trajectory_sha256": sha256(args.trajectory),
        "calibration": str(args.calibration),
        "calibration_sha256": sha256(args.calibration),
        "image_dir": str(args.image_dir),
        "frame_indices": frame_indices.tolist(),
        "image_set_sha256": image_set_sha256(args.image_dir, frame_indices),
        "tracking_frame_indices": tracking_frame_indices.tolist(),
        "tracking_image_set_sha256": image_set_sha256(
            args.image_dir, tracking_frame_indices
        ),
        "tracking_times": (
            str(args.tracking_times) if args.tracking_times is not None else None
        ),
        "tracking_times_sha256": (
            sha256(args.tracking_times) if args.tracking_times is not None else None
        ),
        "tracking_stride": args.tracking_stride,
        "dpvo_root": str(args.dpvo_root),
        "dpvo_commit": args.dpvo_commit,
        "dpvo_compat_patch": str(args.dpvo_compat_patch),
        "dpvo_compat_patch_sha256": sha256(args.dpvo_compat_patch),
        "network": str(args.network),
        "network_sha256": sha256(args.network),
        "dpvo_config": str(args.dpvo_config),
        "dpvo_config_sha256": sha256(args.dpvo_config),
        "seed": args.seed,
        "loop_closure": args.loop_closure,
        "backend_threshold": args.backend_threshold,
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
        ("calibration", "calibration_sha256"),
        ("dpvo_compat_patch", "dpvo_compat_patch_sha256"),
        ("network", "network_sha256"),
        ("dpvo_config", "dpvo_config_sha256"),
        ("candidate", "candidate_sha256"),
        ("diagnostics", "diagnostics_sha256"),
    ):
        path = Path(manifest[path_key])
        if sha256(path) != manifest[hash_key]:
            raise ValueError(f"artifact changed after generation: {path}")
    frame_indices = np.asarray(manifest["frame_indices"], dtype=np.int64)
    if image_set_sha256(Path(manifest["image_dir"]), frame_indices) != manifest["image_set_sha256"]:
        raise ValueError("image set changed after candidate generation")
    if manifest.get("tracking_times") is not None:
        tracking_times = Path(manifest["tracking_times"])
        if sha256(tracking_times) != manifest["tracking_times_sha256"]:
            raise ValueError("tracking timestamps changed after candidate generation")
    tracking_frame_indices = np.asarray(
        manifest.get("tracking_frame_indices", manifest["frame_indices"]),
        dtype=np.int64,
    )
    expected_tracking_hash = manifest.get(
        "tracking_image_set_sha256", manifest["image_set_sha256"]
    )
    if (
        image_set_sha256(Path(manifest["image_dir"]), tracking_frame_indices)
        != expected_tracking_hash
    ):
        raise ValueError("tracking image set changed after candidate generation")
    with np.load(manifest["candidate"], allow_pickle=False) as archive:
        candidates = {
            key.removesuffix("_c2w"): poses_4x4(archive[key])
            for key in archive.files
            if key.endswith("_c2w")
        }
    # GT is opened only after every input, source, and candidate hash verifies.
    with np.load(manifest["trajectory"], allow_pickle=False) as archive:
        visual = poses_4x4(archive[args.pose_key])
        target = poses_4x4(archive[args.ground_truth_key])
    result = {
        "visual_ate_rmse_m": ate_rmse(
            visual[:, :3, 3], target[:, :3, 3], "sim3"
        )["ate_rmse"],
        "candidate_ate_rmse_m": {
            name: ate_rmse(value[:, :3, 3], target[:, :3, 3], "sim3")["ate_rmse"]
            for name, value in candidates.items()
        },
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
    generate_parser.add_argument("--image-dir", type=Path, required=True)
    generate_parser.add_argument("--calibration", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--dpvo-root", type=Path, required=True)
    generate_parser.add_argument("--dpvo-commit", required=True)
    generate_parser.add_argument("--dpvo-compat-patch", type=Path, required=True)
    generate_parser.add_argument("--network", type=Path, required=True)
    generate_parser.add_argument("--dpvo-config", type=Path, required=True)
    generate_parser.add_argument("--tracking-times", type=Path)
    generate_parser.add_argument("--tracking-stride", type=int, default=1)
    generate_parser.add_argument("--seed", type=int, default=1234)
    generate_parser.add_argument("--loop-closure", action="store_true")
    generate_parser.add_argument("--backend-threshold", type=float, default=32.0)
    generate_parser.add_argument("--pose-key", default="predicted_c2w")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--pose-key", default="predicted_c2w")
    evaluate_parser.add_argument("--ground-truth-key", default="ground_truth_c2w")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generate(arguments) if arguments.command == "generate" else evaluate(arguments)
