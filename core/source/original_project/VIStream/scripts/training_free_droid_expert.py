#!/usr/bin/env python3
"""Generate and audit a frozen DROID-SLAM trajectory expert without GT."""

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


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def droid_source_state(root: Path) -> dict[str, str]:
    lietorch = root / "thirdparty" / "lietorch"
    scatter = root / "thirdparty" / "pytorch_scatter"
    combined_diff = "\n".join(
        git_output(path, "diff", "--binary")
        for path in (root, lietorch, scatter)
    )
    return {
        "commit": git_output(root, "rev-parse", "HEAD"),
        "lietorch_commit": git_output(lietorch, "rev-parse", "HEAD"),
        "pytorch_scatter_commit": git_output(scatter, "rev-parse", "HEAD"),
        "tracked_diff_sha256": hashlib.sha256(combined_diff.encode()).hexdigest(),
    }


def poses_4x4(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1:] == (4, 4):
        return values.copy()
    if values.shape[1:] != (3, 4):
        raise ValueError(f"expected [N,3,4] or [N,4,4], got {values.shape}")
    output = np.repeat(np.eye(4)[None], len(values), axis=0)
    output[:, :3, :4] = values
    return output


def pose_matrices(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"expected DROID [N,7] poses, got {poses.shape}")
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
        raise ValueError("visual, source, and output rotations must align")
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
            direction = visual[index, :3, :3].T @ visual_steps[index]
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
        else:
            direction = direction / source_norm
        output[index + 1, :3, 3] = (
            output[index, :3, 3]
            + output_rotations[index] @ direction * visual_norm
        )
    return output


def path_statistics(poses: np.ndarray) -> dict[str, float]:
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    path = float(np.sum(steps))
    endpoint = float(np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3]))
    return {
        "path_length": path,
        "endpoint_distance": endpoint,
        "endpoint_to_path_ratio": endpoint / max(path, 1e-12),
        "median_step": float(np.median(steps)),
        "minimum_step": float(np.min(steps)),
        "maximum_step": float(np.max(steps)),
    }


def read_intrinsics(path: Path) -> np.ndarray:
    for line in path.read_text().splitlines():
        if line.startswith("P2:"):
            projection = np.fromstring(line.split(":", 1)[1], sep=" ").reshape(3, 4)
            return np.asarray(
                [projection[0, 0], projection[1, 1], projection[0, 2], projection[1, 2]],
                dtype=np.float32,
            )
    raise ValueError(f"missing P2 in {path}")


def tracking_schedule(
    target_frame_indices: np.ndarray,
    target_timestamps: np.ndarray,
    tracking_times: Path,
    tracking_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tracking_stride < 1:
        raise ValueError("tracking stride must be positive")
    target_frame_indices = np.asarray(target_frame_indices, dtype=np.int64)
    target_timestamps = np.asarray(target_timestamps, dtype=np.float64)
    all_timestamps = np.atleast_1d(np.loadtxt(tracking_times, dtype=np.float64))
    first, last = int(target_frame_indices[0]), int(target_frame_indices[-1])
    if first < 0 or last >= len(all_timestamps):
        raise ValueError("tracking timestamps do not cover target frames")
    if not np.allclose(
        all_timestamps[target_frame_indices], target_timestamps, rtol=0.0, atol=5e-4
    ):
        raise ValueError("cached and tracking timestamps disagree")
    regular = np.arange(first, last + 1, tracking_stride, dtype=np.int64)
    frames = np.unique(np.concatenate([regular, target_frame_indices]))
    positions = np.searchsorted(frames, target_frame_indices)
    if not np.array_equal(frames[positions], target_frame_indices):
        raise AssertionError("target frames are absent from tracking schedule")
    return frames, all_timestamps[frames], positions


def preprocess(path: Path, intrinsics: np.ndarray) -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    scale = np.sqrt((384.0 * 512.0) / (height * width))
    new_height, new_width = int(height * scale), int(width * scale)
    image = cv2.resize(image, (new_width, new_height))
    image = image[: new_height - new_height % 8, : new_width - new_width % 8]
    scaled = torch.as_tensor(intrinsics.copy())
    scaled[0::2] *= new_width / width
    scaled[1::2] *= new_height / height
    tensor = torch.as_tensor(image).permute(2, 0, 1)[None]
    return tensor, scaled


def run_droid(
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
    image_dir: Path,
    calibration: Path,
    droid_root: Path,
    weights: Path,
    seed: int,
    policy: dict,
) -> tuple[np.ndarray, dict, dict[str, str]]:
    import torch

    module_root = (droid_root / "droid_slam").resolve()
    sys.path.insert(0, str(module_root))
    importlib.invalidate_caches()
    droid_module = importlib.import_module("droid")
    if Path(droid_module.__file__).resolve().parent != module_root:
        raise ImportError("DROID import did not resolve to the audited source root")
    from droid import Droid
    import droid_backends
    import lietorch_backends
    import lietorch_extras

    intrinsics = read_intrinsics(calibration)
    first_image, _ = preprocess(
        image_dir / f"{int(frame_indices[0]):06d}.png", intrinsics
    )
    namespace = argparse.Namespace(
        weights=str(weights),
        buffer=len(frame_indices) + policy["buffer_margin"],
        image_size=[first_image.shape[2], first_image.shape[3]],
        disable_vis=True,
        stereo=False,
        asynchronous=False,
        frontend_device="cuda",
        backend_device="cuda",
        upsample=False,
        **{key: value for key, value in policy.items() if key != "buffer_margin"},
    )

    def stream():
        for timestamp, frame in zip(timestamps, frame_indices):
            image, scaled_intrinsics = preprocess(
                image_dir / f"{int(frame):06d}.png", intrinsics
            )
            yield float(timestamp), image, scaled_intrinsics

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    droid = Droid(namespace)
    for timestamp, image, scaled_intrinsics in stream():
        droid.track(timestamp, image, intrinsics=scaled_intrinsics)
    keyframes = int(droid.video.counter.value)
    with torch.inference_mode():
        trajectory = droid.terminate(stream())
    trajectory = pose_matrices(trajectory)
    diagnostics = {
        "input_frames": len(frame_indices),
        "output_poses": len(trajectory),
        "keyframes": keyframes,
        "initialized": bool(keyframes >= policy["warmup"]),
        "input_hw": list(namespace.image_size),
        "droid_root": str(droid_root),
        "seed": seed,
        "policy": policy,
    }
    extension_paths = {
        "droid_backends": str(Path(droid_backends.__file__).resolve()),
        "lietorch_backends": str(Path(lietorch_backends.__file__).resolve()),
        "lietorch_extras": str(Path(lietorch_extras.__file__).resolve()),
    }
    return trajectory, diagnostics, extension_paths


def generate(args: argparse.Namespace) -> None:
    source = droid_source_state(args.droid_root)
    expected = {
        "commit": args.droid_commit,
        "lietorch_commit": args.lietorch_commit,
        "pytorch_scatter_commit": args.pytorch_scatter_commit,
        "tracked_diff_sha256": args.droid_diff_sha256,
    }
    if source != expected:
        raise ValueError(f"DROID source state differs: {source} != {expected}")
    if sha256(args.weights) != sha256(args.weights_mirror):
        raise ValueError("independent DROID weight mirrors differ")
    policy = json.loads(args.policy.read_text())
    with np.load(args.trajectory, allow_pickle=False) as archive:
        visual = poses_4x4(archive[args.pose_key])
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
    tracking_frames, tracking_timestamps, sample_positions = tracking_schedule(
        frame_indices, timestamps, args.tracking_times, args.tracking_stride
    )
    tracked, diagnostics, extension_paths = run_droid(
        tracking_frames,
        tracking_timestamps,
        args.image_dir,
        args.calibration,
        args.droid_root,
        args.weights,
        args.seed,
        policy,
    )
    if len(tracked) != len(tracking_frames) or not np.isfinite(tracked).all():
        raise ValueError("DROID did not return one finite pose per tracking frame")
    droid = tracked[sample_positions]
    visual_rotation = direction_transfer(visual, droid, visual[:, :3, :3])
    droid_rotation = direction_transfer(visual, droid, droid[:, :3, :3])
    diagnostics.update({
        "target_frames": len(frame_indices),
        "tracking_frames": len(tracking_frames),
        "tracking_stride": args.tracking_stride,
        "tracking_overlap_factor": float(len(tracking_frames) / len(frame_indices)),
        "visual": path_statistics(visual),
        "droid": path_statistics(droid),
        "droid_direction_visual_rotation": path_statistics(visual_rotation),
        "droid_direction_droid_rotation": path_statistics(droid_rotation),
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidate_before_gt.npz"
    np.savez_compressed(
        candidate_path,
        frame_indices=frame_indices,
        timestamps=timestamps,
        droid_c2w=droid,
        droid_direction_visual_rotation_c2w=visual_rotation,
        droid_direction_droid_rotation_c2w=droid_rotation,
    )
    diagnostics_path = args.output_dir / "droid_before_gt.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    manifest = {
        "protocol": "frozen DROID experts generated and hashed before GT load",
        "trajectory": str(args.trajectory),
        "trajectory_sha256": sha256(args.trajectory),
        "calibration": str(args.calibration),
        "calibration_sha256": sha256(args.calibration),
        "image_dir": str(args.image_dir),
        "frame_indices": frame_indices.tolist(),
        "image_set_sha256": image_set_sha256(args.image_dir, frame_indices),
        "tracking_frame_indices": tracking_frames.tolist(),
        "tracking_image_set_sha256": image_set_sha256(args.image_dir, tracking_frames),
        "tracking_times": str(args.tracking_times),
        "tracking_times_sha256": sha256(args.tracking_times),
        "tracking_stride": args.tracking_stride,
        "droid_root": str(args.droid_root),
        "source": source,
        "weights": str(args.weights),
        "weights_sha256": sha256(args.weights),
        "weights_mirror": str(args.weights_mirror),
        "weights_mirror_sha256": sha256(args.weights_mirror),
        "weight_mirror_urls": args.weight_mirror_url,
        "extensions": {
            name: {"path": path, "sha256": sha256(Path(path))}
            for name, path in extension_paths.items()
        },
        "policy": policy,
        "policy_path": str(args.policy.resolve()),
        "policy_sha256": sha256(args.policy),
        "seed": args.seed,
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
        ("tracking_times", "tracking_times_sha256"),
        ("weights", "weights_sha256"),
        ("weights_mirror", "weights_mirror_sha256"),
        ("policy_path", "policy_sha256"),
        ("candidate", "candidate_sha256"),
        ("diagnostics", "diagnostics_sha256"),
    ):
        path = Path(manifest[path_key])
        if sha256(path) != manifest[hash_key]:
            raise ValueError(f"artifact changed after generation: {path}")
    for extension in manifest["extensions"].values():
        if sha256(Path(extension["path"])) != extension["sha256"]:
            raise ValueError("compiled DROID extension changed after generation")
    tracking_frames = np.asarray(manifest["tracking_frame_indices"], dtype=np.int64)
    if (
        image_set_sha256(Path(manifest["image_dir"]), tracking_frames)
        != manifest["tracking_image_set_sha256"]
    ):
        raise ValueError("tracking image set changed after generation")
    with np.load(manifest["candidate"], allow_pickle=False) as archive:
        candidates = {
            key.removesuffix("_c2w"): poses_4x4(archive[key])
            for key in archive.files
            if key.endswith("_c2w")
        }
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
    generate_parser.add_argument("--tracking-times", type=Path, required=True)
    generate_parser.add_argument("--tracking-stride", type=int, default=1)
    generate_parser.add_argument("--droid-root", type=Path, required=True)
    generate_parser.add_argument("--droid-commit", required=True)
    generate_parser.add_argument("--lietorch-commit", required=True)
    generate_parser.add_argument("--pytorch-scatter-commit", required=True)
    generate_parser.add_argument("--droid-diff-sha256", required=True)
    generate_parser.add_argument("--weights", type=Path, required=True)
    generate_parser.add_argument("--weights-mirror", type=Path, required=True)
    generate_parser.add_argument("--weight-mirror-url", action="append", default=[])
    generate_parser.add_argument("--policy", type=Path, required=True)
    generate_parser.add_argument("--seed", type=int, default=1234)
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
