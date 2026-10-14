#!/usr/bin/env python3
"""GT-free adjacent-frame translation-direction expert for KITTI 00--10."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SEQUENCES = tuple(f"{index:02d}" for index in range(11))


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
    if values.shape[1:] != (3, 4):
        raise ValueError(f"expected [N,3,4] or [N,4,4], got {values.shape}")
    output = np.repeat(np.eye(4)[None], len(values), axis=0)
    output[:, :3, :4] = values
    return output


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0))))


def soft_and_probability(margins: list[float]) -> float:
    return sigmoid(min(margins))


def read_intrinsics(path: Path, scale: float) -> np.ndarray:
    for line in path.read_text().splitlines():
        if line.startswith("P2:"):
            projection = np.fromstring(
                line.split(":", 1)[1], sep=" "
            ).reshape(3, 4)
            intrinsics = projection[:, :3].copy()
            intrinsics[:2] *= scale
            return intrinsics
    raise ValueError(f"missing P2 in {path}")


def rotation_error(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.linalg.norm(Rotation.from_matrix(first.T @ second).as_rotvec())
    )


def normalized_blend(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    blended = (1.0 - weight) * first + weight * second
    norm = float(np.linalg.norm(blended))
    return first.copy() if norm <= 1e-12 else blended / norm


def direction_gate_probability(
    inliers: int,
    inlier_ratio: float,
    rotation_innovation_rad: float,
    forward_ratio: float,
    policy: dict,
) -> float:
    return soft_and_probability([
        (inliers - policy["minimum_inliers"]) / policy["inlier_count_width"],
        (inlier_ratio - policy["minimum_inlier_ratio"])
        / policy["inlier_ratio_width"],
        (policy["maximum_rotation_innovation_rad"] - rotation_innovation_rad)
        / policy["rotation_innovation_width_rad"],
        (forward_ratio - policy["minimum_forward_ratio"])
        / policy["forward_ratio_width"],
    ])


def extract_features(
    image_path: Path,
    scale: float,
    detector: cv2.SIFT,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.resize(image, None, fx=scale, fy=scale)
    return detector.detectAndCompute(image, None)


def generate_sequence(
    visual: np.ndarray,
    frame_indices: np.ndarray,
    image_dir: Path,
    intrinsics: np.ndarray,
    policy: dict,
) -> tuple[np.ndarray, list[dict], dict]:
    detector = cv2.SIFT_create(
        nfeatures=policy["sift_nfeatures"],
        contrastThreshold=policy["sift_contrast_threshold"],
    )
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    scale = policy["image_scale"]
    keypoints, descriptors = extract_features(
        image_dir / f"{int(frame_indices[0]):06d}.png", scale, detector
    )
    output = visual.copy()
    output[0] = visual[0]
    rows = []
    accepted = 0
    gate_values = []
    direction_innovations = []
    for index in range(len(visual) - 1):
        next_keypoints, next_descriptors = extract_features(
            image_dir / f"{int(frame_indices[index + 1]):06d}.png",
            scale,
            detector,
        )
        visual_world_step = visual[index + 1, :3, 3] - visual[index, :3, 3]
        step_norm = float(np.linalg.norm(visual_world_step))
        visual_direction = visual[index, :3, :3].T @ visual_world_step
        visual_direction /= max(float(np.linalg.norm(visual_direction)), 1e-12)
        recovered_direction = visual_direction.copy()
        matches = []
        inliers = 0
        inlier_ratio = 0.0
        rotation_innovation = float("inf")
        forward_ratio = 0.0
        gate_probability = 0.0
        direction_innovation = 0.0
        if descriptors is not None and next_descriptors is not None:
            matches = [
                first
                for first, second in matcher.knnMatch(
                    descriptors, next_descriptors, k=2
                )
                if first.distance < policy["ratio_test"] * second.distance
            ]
        if len(matches) >= policy["minimum_matches"]:
            points_a = np.float32(
                [keypoints[item.queryIdx].pt for item in matches]
            )
            points_b = np.float32(
                [next_keypoints[item.trainIdx].pt for item in matches]
            )
            essential, mask = cv2.findEssentialMat(
                points_a,
                points_b,
                intrinsics,
                cv2.RANSAC,
                policy["ransac_probability"],
                policy["ransac_threshold_px"],
            )
            if essential is not None and mask is not None:
                if essential.shape[0] > 3:
                    essential = essential[:3]
                try:
                    inliers, recovered_rotation, recovered_t, _ = cv2.recoverPose(
                        essential,
                        points_a,
                        points_b,
                        intrinsics,
                        mask=mask,
                    )
                    inlier_ratio = float(inliers) / len(matches)
                    visual_relative_rotation = (
                        visual[index + 1, :3, :3].T
                        @ visual[index, :3, :3]
                    )
                    rotation_innovation = rotation_error(
                        visual_relative_rotation, recovered_rotation
                    )
                    recovered_direction = -recovered_rotation.T @ recovered_t[:, 0]
                    recovered_direction /= max(
                        float(np.linalg.norm(recovered_direction)), 1e-12
                    )
                    if recovered_direction[2] * visual_direction[2] < 0.0:
                        recovered_direction *= -1.0
                    forward_ratio = float(abs(recovered_direction[2]))
                    gate_probability = direction_gate_probability(
                        int(inliers),
                        inlier_ratio,
                        rotation_innovation,
                        forward_ratio,
                        policy,
                    )
                    direction_innovation = float(
                        np.arccos(
                            np.clip(
                                np.dot(visual_direction, recovered_direction),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                except cv2.error:
                    pass
        route_enabled = gate_probability >= policy["route_threshold"]
        blend = gate_probability * policy["maximum_direction_blend"]
        selected_direction = (
            normalized_blend(visual_direction, recovered_direction, blend)
            if route_enabled
            else visual_direction
        )
        output[index + 1, :3, :3] = visual[index + 1, :3, :3]
        output[index + 1, :3, 3] = (
            output[index, :3, 3]
            + visual[index, :3, :3] @ selected_direction * step_norm
        )
        accepted += int(route_enabled)
        gate_values.append(gate_probability)
        if route_enabled:
            direction_innovations.append(direction_innovation)
        rows.append({
            "src_frame": int(frame_indices[index]),
            "dst_frame": int(frame_indices[index + 1]),
            "matches": len(matches),
            "inliers": int(inliers),
            "inlier_ratio": inlier_ratio,
            "rotation_innovation_rad": rotation_innovation,
            "forward_ratio": forward_ratio,
            "direction_innovation_rad": direction_innovation,
            "gate_probability": gate_probability,
            "route_enabled": route_enabled,
        })
        keypoints, descriptors = next_keypoints, next_descriptors
    visual_path_scale = max(
        float(np.linalg.norm(visual[-1, :3, 3] - visual[0, :3, 3])),
        0.1 * float(np.sum(np.linalg.norm(np.diff(visual[:, :3, 3], axis=0), axis=1))),
        1e-9,
    )
    relative_path_change = float(
        np.sqrt(np.mean(np.sum((output[:, :3, 3] - visual[:, :3, 3]) ** 2, axis=1)))
        / visual_path_scale
    )
    if relative_path_change > policy["maximum_path_change_ratio"]:
        output = visual.copy()
        fallback_reason = "path_change_budget_exceeded"
    else:
        fallback_reason = None
    summary = {
        "expert": "essential_translation_direction" if fallback_reason is None else "visual",
        "accepted_intervals": accepted,
        "interval_count": len(visual) - 1,
        "accepted_fraction": accepted / max(len(visual) - 1, 1),
        "gate_mean": float(np.mean(gate_values)),
        "median_accepted_direction_innovation_rad": (
            float(np.median(direction_innovations)) if direction_innovations else 0.0
        ),
        "relative_path_change": relative_path_change,
        "fallback_reason": fallback_reason,
    }
    return output, rows, summary


def generate(args: argparse.Namespace) -> None:
    policy = json.loads(args.policy.read_text())
    entries = []
    for sequence in SEQUENCES:
        cache_path = args.cache_root / sequence / args.cache_name
        calibration_path = args.image_root / sequence / "calib.txt"
        image_dir = args.image_root / sequence / "image_2"
        with np.load(cache_path, allow_pickle=False) as archive:
            visual = poses_4x4(archive[args.pose_key])
            frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        candidate, rows, summary = generate_sequence(
            visual,
            frame_indices,
            image_dir,
            read_intrinsics(calibration_path, policy["image_scale"]),
            policy,
        )
        output_dir = args.output_dir / sequence
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = output_dir / "candidate_before_gt.npz"
        np.savez_compressed(
            candidate_path, c2w=candidate, frame_indices=frame_indices
        )
        diagnostics_path = output_dir / "direction_gate_before_gt.json"
        diagnostics_path.write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n"
        )
        entries.append({
            "sequence": sequence,
            "cache": str(cache_path),
            "cache_sha256": sha256(cache_path),
            "calibration": str(calibration_path),
            "calibration_sha256": sha256(calibration_path),
            "candidate": str(candidate_path),
            "candidate_sha256": sha256(candidate_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": sha256(diagnostics_path),
        })
        print(json.dumps({"sequence": sequence, **summary}), flush=True)
    manifest = {
        "protocol": "all essential-direction candidates generated and hashed before GT load",
        "policy": policy,
        "policy_path": str(args.policy.resolve()),
        "policy_sha256": sha256(args.policy),
        "entries": entries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest_before_gt.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def evaluate(args: argparse.Namespace) -> None:
    from vistream.metrics import ate_rmse

    manifest = json.loads((args.output_dir / "manifest_before_gt.json").read_text())
    if manifest["policy"] != json.loads(args.policy.read_text()):
        raise ValueError("evaluation policy differs from generation policy")
    rows = []
    for entry in manifest["entries"]:
        for path_key, hash_key in (
            ("cache", "cache_sha256"),
            ("calibration", "calibration_sha256"),
            ("candidate", "candidate_sha256"),
            ("diagnostics", "diagnostics_sha256"),
        ):
            path = Path(entry[path_key])
            if sha256(path) != entry[hash_key]:
                raise ValueError(f"artifact changed after generation: {path}")
        with np.load(entry["candidate"], allow_pickle=False) as archive:
            candidate = poses_4x4(archive["c2w"])
        # GT is opened only after the candidate and diagnostics hashes verify.
        with np.load(entry["cache"], allow_pickle=False) as archive:
            visual = poses_4x4(archive[args.pose_key])
            target = poses_4x4(archive[args.ground_truth_key])
        diagnostics = json.loads(Path(entry["diagnostics"]).read_text())["summary"]
        rows.append({
            "sequence": entry["sequence"],
            "expert": diagnostics["expert"],
            "accepted_fraction": diagnostics["accepted_fraction"],
            "visual_ate_rmse_m": ate_rmse(
                visual[:, :3, 3], target[:, :3, 3], "sim3"
            )["ate_rmse"],
            "candidate_ate_rmse_m": ate_rmse(
                candidate[:, :3, 3], target[:, :3, 3], "sim3"
            )["ate_rmse"],
        })
    visual = np.asarray([row["visual_ate_rmse_m"] for row in rows])
    candidate = np.asarray([row["candidate_ate_rmse_m"] for row in rows])
    summary = {
        "protocol": manifest["protocol"],
        "policy": manifest["policy"],
        "rows": rows,
        "macro": {
            "visual_ate_rmse_mean_m": float(np.mean(visual)),
            "candidate_ate_rmse_mean_m": float(np.mean(candidate)),
            "improved_sequences": int(np.sum(candidate < visual)),
            "sequence_count": len(rows),
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
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--policy", type=Path, required=True)
    generate_parser.add_argument("--cache-name", default="trajectory.npz")
    generate_parser.add_argument("--pose-key", default="predicted_c2w")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--policy", type=Path, required=True)
    evaluate_parser.add_argument("--pose-key", default="predicted_c2w")
    evaluate_parser.add_argument("--ground-truth-key", default="ground_truth_c2w")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generate(arguments) if arguments.command == "generate" else evaluate(arguments)
