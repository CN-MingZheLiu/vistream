#!/usr/bin/env python3
"""Generate and audit frozen ORB-SLAM3 monocular experts without reading GT."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

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
    for frame in np.asarray(frame_indices, dtype=np.int64):
        path = image_dir / f"{int(frame):06d}.png"
        digest.update(path.name.encode("ascii"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def git_output(root: Path, *arguments: str, binary: bool = False):
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=not binary,
    ).stdout


def repository_state(root: Path) -> tuple[dict[str, str], bytes]:
    commit = git_output(root, "rev-parse", "HEAD").strip()
    diff = git_output(root, "diff", "--binary", binary=True)
    state = {
        "commit": commit,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }
    return state, diff


def poses_4x4(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1:] == (4, 4):
        return values.copy()
    if values.shape[1:] != (3, 4):
        raise ValueError(f"expected [N,3,4] or [N,4,4], got {values.shape}")
    output = np.repeat(np.eye(4)[None], len(values), axis=0)
    output[:, :3, :4] = values
    return output


def parse_kitti_intrinsics(calibration: Path) -> np.ndarray:
    for line in calibration.read_text().splitlines():
        if line.startswith("P2:"):
            projection = np.fromstring(line.split(":", 1)[1], sep=" ").reshape(3, 4)
            return np.asarray(
                [projection[0, 0], projection[1, 1], projection[0, 2], projection[1, 2]],
                dtype=np.float64,
            )
    raise ValueError(f"missing P2 in {calibration}")


def parse_orb_intrinsics(settings: Path) -> np.ndarray:
    values: dict[str, float] = {}
    pattern = re.compile(r"^Camera1\.(fx|fy|cx|cy):\s*([-+0-9.eE]+)\s*$")
    for line in settings.read_text().splitlines():
        match = pattern.match(line.strip())
        if match:
            values[match.group(1)] = float(match.group(2))
    missing = [name for name in ("fx", "fy", "cx", "cy") if name not in values]
    if missing:
        raise ValueError(f"missing ORB camera settings: {missing}")
    return np.asarray([values[name] for name in ("fx", "fy", "cx", "cy")])


def parse_euroc_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
    if rows.ndim != 2 or rows.shape[1] != 8:
        raise ValueError(f"expected EuRoC trajectory [N,8], got {rows.shape}")
    if len(rows) == 0 or not np.isfinite(rows).all():
        raise ValueError("ORB-SLAM3 trajectory is empty or non-finite")
    timestamps = rows[:, 0] / 1e9
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("ORB-SLAM3 timestamps are not strictly increasing")
    quaternion_norms = np.linalg.norm(rows[:, 4:8], axis=1)
    if np.any(quaternion_norms <= 1e-12):
        raise ValueError("ORB-SLAM3 returned a zero quaternion")
    poses = np.repeat(np.eye(4)[None], len(rows), axis=0)
    poses[:, :3, 3] = rows[:, 1:4]
    poses[:, :3, :3] = Rotation.from_quat(rows[:, 4:8]).as_matrix()
    return timestamps, poses


def match_output_frames(
    output_timestamps: np.ndarray,
    all_timestamps: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    if tolerance <= 0:
        raise ValueError("timestamp tolerance must be positive")
    output_timestamps = np.asarray(output_timestamps, dtype=np.float64)
    all_timestamps = np.asarray(all_timestamps, dtype=np.float64)
    right = np.searchsorted(all_timestamps, output_timestamps)
    right = np.clip(right, 0, len(all_timestamps) - 1)
    left = np.clip(right - 1, 0, len(all_timestamps) - 1)
    choose_left = (
        np.abs(all_timestamps[left] - output_timestamps)
        <= np.abs(all_timestamps[right] - output_timestamps)
    )
    indices = np.where(choose_left, left, right).astype(np.int64)
    errors = np.abs(all_timestamps[indices] - output_timestamps)
    if np.any(errors > tolerance):
        worst = int(np.argmax(errors))
        raise ValueError(
            "ORB-SLAM3 timestamp does not match KITTI input: "
            f"{output_timestamps[worst]} versus {all_timestamps[indices[worst]]}"
        )
    if len(np.unique(indices)) != len(indices):
        raise ValueError("multiple ORB-SLAM3 poses map to one KITTI frame")
    return indices


def sample_target_poses(
    output_timestamps: np.ndarray,
    output_poses: np.ndarray,
    all_timestamps: np.ndarray,
    target_frame_indices: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, dict]:
    output_poses = poses_4x4(output_poses)
    if len(output_timestamps) != len(output_poses):
        raise ValueError("ORB timestamps and poses differ in length")
    matched = match_output_frames(output_timestamps, all_timestamps, tolerance)
    frame_to_output = {int(frame): index for index, frame in enumerate(matched)}
    sampled = np.repeat(np.eye(4)[None], len(target_frame_indices), axis=0)
    injected_first_frame = False
    missing_targets = []
    for target_index, frame in enumerate(target_frame_indices):
        frame = int(frame)
        if frame in frame_to_output:
            sampled[target_index] = output_poses[frame_to_output[frame]]
        elif frame == 0:
            injected_first_frame = True
        else:
            missing_targets.append(frame)
    if missing_targets:
        raise ValueError(
            "ORB-SLAM3 did not produce exact target frames: "
            + ", ".join(str(value) for value in missing_targets[:20])
        )
    all_frames = np.arange(len(all_timestamps), dtype=np.int64)
    missing = np.setdiff1d(all_frames, matched, assume_unique=True)
    diagnostics = {
        "output_frames": len(output_poses),
        "matched_frame_indices": matched.tolist(),
        "missing_frame_indices": missing.tolist(),
        "tracking_completeness": float(len(matched) / len(all_timestamps)),
        "injected_first_frame_identity": injected_first_frame,
    }
    return sampled, diagnostics


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


def dynamic_libraries(binary: Path) -> list[dict[str, str]]:
    output = subprocess.run(
        ["ldd", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    libraries = []
    for line in output.splitlines():
        line = line.strip()
        if "=>" in line:
            name, remainder = line.split("=>", 1)
            candidate = remainder.strip().split(" ", 1)[0]
        elif line.startswith("/"):
            candidate = line.split(" ", 1)[0]
            name = Path(candidate).name
        else:
            continue
        if candidate == "not":
            raise ValueError(f"unresolved dynamic dependency: {line}")
        path = Path(candidate)
        if not path.is_file():
            continue
        resolved = path.resolve()
        libraries.append(
            {
                "name": name.strip(),
                "path": str(path),
                "resolved_path": str(resolved),
                "sha256": sha256(resolved),
            }
        )
    if not libraries:
        raise ValueError(f"no dynamic libraries resolved for {binary}")
    return libraries


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path.resolve())}


def run_orbslam3(
    binary: Path,
    vocabulary: Path,
    settings: Path,
    sequence_dir: Path,
    raw_dir: Path,
    timeout_seconds: int,
) -> tuple[Path, Path, Path, list[str]]:
    raw_dir.mkdir(parents=True, exist_ok=False)
    command = [
        str(binary.resolve()),
        str(vocabulary.resolve()),
        str(settings.resolve()),
        str(sequence_dir.resolve()),
    ]
    log_path = raw_dir / "run.log"
    with log_path.open("wb") as log:
        subprocess.run(
            command,
            cwd=raw_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout_seconds,
        )
    frame_path = raw_dir / "FrameTrajectory.txt"
    keyframe_path = raw_dir / "KeyFrameTrajectory.txt"
    if not frame_path.is_file() or not keyframe_path.is_file():
        raise FileNotFoundError("ORB-SLAM3 did not create both trajectory files")
    return frame_path, keyframe_path, log_path, command


def validate_source_state(args: argparse.Namespace) -> tuple[dict, bytes]:
    orb_state, orb_diff = repository_state(args.orb_root)
    pangolin_state, _ = repository_state(args.pangolin_root)
    opencv_state, _ = repository_state(args.opencv_root)
    expected = {
        "orbslam3": {
            "commit": args.orb_commit,
            "tracked_diff_sha256": args.orb_diff_sha256,
        },
        "pangolin": {
            "commit": args.pangolin_commit,
            "tracked_diff_sha256": args.pangolin_diff_sha256,
        },
        "opencv": {
            "commit": args.opencv_commit,
            "tracked_diff_sha256": args.opencv_diff_sha256,
        },
    }
    actual = {
        "orbslam3": orb_state,
        "pangolin": pangolin_state,
        "opencv": opencv_state,
    }
    if actual != expected:
        raise ValueError(f"source state differs: {actual} != {expected}")
    return actual, orb_diff


def generate(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_state, orb_diff = validate_source_state(args)
    source_diff_path = args.output_dir / "orbslam3_source_diff_before_gt.patch"
    source_diff_path.write_bytes(orb_diff)

    sequence_times = args.sequence_dir / "times.txt"
    image_dir = args.sequence_dir / "image_2"
    all_timestamps = np.atleast_1d(np.loadtxt(sequence_times, dtype=np.float64))
    all_frame_indices = np.arange(len(all_timestamps), dtype=np.int64)
    if len(all_timestamps) == 0 or np.any(np.diff(all_timestamps) <= 0):
        raise ValueError("KITTI timestamps are empty or not strictly increasing")
    if not np.allclose(
        parse_kitti_intrinsics(args.calibration),
        parse_orb_intrinsics(args.settings),
        rtol=0.0,
        atol=args.intrinsics_tolerance,
    ):
        raise ValueError("ORB settings do not match KITTI image_2 intrinsics")

    # np.load is lazy for NPZ archives; the GT member is not requested here.
    with np.load(args.trajectory, allow_pickle=False) as archive:
        visual = poses_4x4(archive[args.pose_key])
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
    if len(visual) != len(frame_indices) or len(timestamps) != len(frame_indices):
        raise ValueError("Lingbot visual trajectory arrays differ in length")
    if np.any(np.diff(frame_indices) <= 0):
        raise ValueError("Lingbot frame indices are not strictly increasing")
    if frame_indices[0] < 0 or frame_indices[-1] >= len(all_timestamps):
        raise ValueError("Lingbot frame indices exceed the KITTI sequence")
    if not np.allclose(
        all_timestamps[frame_indices], timestamps, rtol=0.0, atol=args.timestamp_tolerance
    ):
        raise ValueError("Lingbot and KITTI timestamps disagree")

    all_image_hash = image_set_sha256(image_dir, all_frame_indices)
    target_image_hash = image_set_sha256(image_dir, frame_indices)
    binary_dependencies = dynamic_libraries(args.binary)
    frame_path, keyframe_path, log_path, command = run_orbslam3(
        args.binary,
        args.vocabulary,
        args.settings,
        args.sequence_dir,
        args.output_dir / "raw_before_gt",
        args.timeout_seconds,
    )
    output_timestamps, output_poses = parse_euroc_trajectory(frame_path)
    orb, tracking = sample_target_poses(
        output_timestamps,
        output_poses,
        all_timestamps,
        frame_indices,
        args.timestamp_tolerance,
    )
    visual_rotation = direction_transfer(visual, orb, visual[:, :3, :3])
    orb_rotation = direction_transfer(visual, orb, orb[:, :3, :3])
    diagnostics = {
        "input_frames": len(all_timestamps),
        "target_frames": len(frame_indices),
        "keyframes": sum(1 for line in keyframe_path.read_text().splitlines() if line.strip()),
        "timestamp_tolerance_seconds": args.timestamp_tolerance,
        **tracking,
        "visual": path_statistics(visual),
        "orbslam3": path_statistics(orb),
        "orb_direction_visual_rotation": path_statistics(visual_rotation),
        "orb_direction_orb_rotation": path_statistics(orb_rotation),
    }
    candidate_path = args.output_dir / "candidate_before_gt.npz"
    np.savez_compressed(
        candidate_path,
        frame_indices=frame_indices,
        timestamps=timestamps,
        orbslam3_c2w=orb,
        orb_direction_visual_rotation_c2w=visual_rotation,
        orb_direction_orb_rotation_c2w=orb_rotation,
    )
    diagnostics_path = args.output_dir / "orbslam3_before_gt.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")

    artifacts = {
        "trajectory": artifact(args.trajectory),
        "calibration": artifact(args.calibration),
        "times": artifact(sequence_times),
        "settings": artifact(args.settings),
        "vocabulary": artifact(args.vocabulary),
        "binary": artifact(args.binary),
        "orb_library": artifact(args.orb_library),
        "dbow2_library": artifact(args.dbow2_library),
        "g2o_library": artifact(args.g2o_library),
        "source_diff": artifact(source_diff_path),
        "raw_frame_trajectory": artifact(frame_path),
        "raw_keyframe_trajectory": artifact(keyframe_path),
        "run_log": artifact(log_path),
        "candidate": artifact(candidate_path),
        "diagnostics": artifact(diagnostics_path),
    }
    manifest = {
        "protocol": "frozen ORB-SLAM3 experts generated and hashed before GT load",
        "source_state": source_state,
        "sequence_dir": str(args.sequence_dir.resolve()),
        "image_dir": str(image_dir.resolve()),
        "all_frame_indices": all_frame_indices.tolist(),
        "all_image_set_sha256": all_image_hash,
        "target_frame_indices": frame_indices.tolist(),
        "target_image_set_sha256": target_image_hash,
        "pose_key": args.pose_key,
        "command": command,
        "binary_dependencies": binary_dependencies,
        "artifacts": artifacts,
    }
    manifest_path = args.output_dir / "manifest_before_gt.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


def verify_manifest(manifest: dict) -> None:
    for name, value in manifest["artifacts"].items():
        path = Path(value["path"])
        if sha256(path) != value["sha256"]:
            raise ValueError(f"artifact changed after generation: {name}: {path}")
    for library in manifest["binary_dependencies"]:
        path = Path(library["resolved_path"])
        if sha256(path) != library["sha256"]:
            raise ValueError(f"dynamic library changed after generation: {path}")
    image_dir = Path(manifest["image_dir"])
    all_frames = np.asarray(manifest["all_frame_indices"], dtype=np.int64)
    if image_set_sha256(image_dir, all_frames) != manifest["all_image_set_sha256"]:
        raise ValueError("full ORB-SLAM3 image set changed after generation")
    target_frames = np.asarray(manifest["target_frame_indices"], dtype=np.int64)
    if image_set_sha256(image_dir, target_frames) != manifest["target_image_set_sha256"]:
        raise ValueError("target image set changed after generation")


def evaluate(args: argparse.Namespace) -> None:
    from vistream.metrics import ate_rmse

    manifest_path = args.output_dir / "manifest_before_gt.json"
    manifest = json.loads(manifest_path.read_text())
    verify_manifest(manifest)
    candidate_path = Path(manifest["artifacts"]["candidate"]["path"])
    with np.load(candidate_path, allow_pickle=False) as archive:
        candidates = {
            key.removesuffix("_c2w"): poses_4x4(archive[key])
            for key in archive.files
            if key.endswith("_c2w")
        }
    trajectory_path = Path(manifest["artifacts"]["trajectory"]["path"])
    # GT is accessed only after all sealed input, executable, library, and output hashes pass.
    with np.load(trajectory_path, allow_pickle=False) as archive:
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


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    empty_diff = hashlib.sha256(b"").hexdigest()
    parser.add_argument("--orb-root", type=Path, required=True)
    parser.add_argument("--orb-commit", required=True)
    parser.add_argument("--orb-diff-sha256", required=True)
    parser.add_argument("--pangolin-root", type=Path, required=True)
    parser.add_argument("--pangolin-commit", required=True)
    parser.add_argument("--pangolin-diff-sha256", default=empty_diff)
    parser.add_argument("--opencv-root", type=Path, required=True)
    parser.add_argument("--opencv-commit", required=True)
    parser.add_argument("--opencv-diff-sha256", default=empty_diff)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--trajectory", type=Path, required=True)
    generate_parser.add_argument("--sequence-dir", type=Path, required=True)
    generate_parser.add_argument("--calibration", type=Path, required=True)
    generate_parser.add_argument("--binary", type=Path, required=True)
    generate_parser.add_argument("--vocabulary", type=Path, required=True)
    generate_parser.add_argument("--settings", type=Path, required=True)
    generate_parser.add_argument("--orb-library", type=Path, required=True)
    generate_parser.add_argument("--dbow2-library", type=Path, required=True)
    generate_parser.add_argument("--g2o-library", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--pose-key", default="predicted_c2w")
    generate_parser.add_argument("--timeout-seconds", type=int, default=7200)
    generate_parser.add_argument("--timestamp-tolerance", type=float, default=5e-7)
    generate_parser.add_argument("--intrinsics-tolerance", type=float, default=1e-3)
    add_source_arguments(generate_parser)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--pose-key", default="predicted_c2w")
    evaluate_parser.add_argument("--ground-truth-key", default="ground_truth_c2w")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generate(arguments) if arguments.command == "generate" else evaluate(arguments)
