#!/usr/bin/env python3
"""GT-free depth-assisted PnP translation experts for one cached trajectory."""

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


def preprocess_image(
    path: Path, image_size: int, patch_size: int
) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape
    new_width = image_size
    new_height = round(height * (new_width / width) / patch_size) * patch_size
    image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_CUBIC
    )
    if new_height > image_size:
        start = (new_height - image_size) // 2
        image = image[start : start + image_size]
    return image


def depth_points_for_matches(
    keypoints: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    depth: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    pixel_stride: int,
    minimum_depth: float,
    maximum_depth: float,
    minimum_confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_pixels = np.asarray(
        [keypoints[item.queryIdx].pt for item in matches], dtype=np.float64
    )
    uu = np.clip(
        np.rint(source_pixels[:, 0] / pixel_stride).astype(np.int64),
        0,
        depth.shape[1] - 1,
    )
    vv = np.clip(
        np.rint(source_pixels[:, 1] / pixel_stride).astype(np.int64),
        0,
        depth.shape[0] - 1,
    )
    values = depth[vv, uu].astype(np.float64)
    scores = confidence[vv, uu].astype(np.float64)
    valid = (
        np.isfinite(values)
        & np.isfinite(scores)
        & (values >= minimum_depth)
        & (values <= maximum_depth)
        & (scores >= minimum_confidence)
    )
    pixels = source_pixels[valid]
    values = values[valid]
    points = np.column_stack((
        (pixels[:, 0] - intrinsics[0, 2]) * values / intrinsics[0, 0],
        (pixels[:, 1] - intrinsics[1, 2]) * values / intrinsics[1, 1],
        values,
    ))
    return points, valid, scores[valid]


def rotation_error(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.linalg.norm(Rotation.from_matrix(first.T @ second).as_rotvec())
    )


def pnp_gate_probability(
    inliers: int,
    inlier_ratio: float,
    reprojection_error_px: float,
    rotation_innovation_rad: float,
    forward_ratio: float,
    policy: dict,
) -> float:
    return soft_and_probability([
        (inliers - policy["minimum_pnp_inliers"])
        / policy["pnp_inlier_count_width"],
        (inlier_ratio - policy["minimum_pnp_inlier_ratio"])
        / policy["pnp_inlier_ratio_width"],
        (policy["maximum_reprojection_error_px"] - reprojection_error_px)
        / policy["reprojection_error_width_px"],
        (policy["maximum_rotation_innovation_rad"] - rotation_innovation_rad)
        / policy["rotation_innovation_width_rad"],
        (forward_ratio - policy["minimum_forward_ratio"])
        / policy["forward_ratio_width"],
    ])


def normalized_blend(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    blended = (1.0 - weight) * first + weight * second
    norm = float(np.linalg.norm(blended))
    return first.copy() if norm <= 1e-12 else blended / norm


def path_change_ratio(reference: np.ndarray, candidate: np.ndarray) -> float:
    steps = np.linalg.norm(np.diff(reference[:, :3, 3], axis=0), axis=1)
    scale = max(
        float(np.linalg.norm(reference[-1, :3, 3] - reference[0, :3, 3])),
        0.1 * float(np.sum(steps)),
        1e-9,
    )
    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    (candidate[:, :3, 3] - reference[:, :3, 3]) ** 2,
                    axis=1,
                )
            )
        )
        / scale
    )


def generate_candidates(
    visual: np.ndarray,
    frame_indices: np.ndarray,
    image_dir: Path,
    depth: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    image_hw: tuple[int, int],
    pixel_stride: int,
    policy: dict,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    if not (
        len(visual)
        == len(frame_indices)
        == len(depth)
        == len(confidence)
        == len(intrinsics)
    ):
        raise ValueError("trajectory and depth artifacts must have equal frame counts")
    detector = cv2.SIFT_create(
        nfeatures=policy["sift_nfeatures"],
        contrastThreshold=policy["sift_contrast_threshold"],
    )
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    image = preprocess_image(
        image_dir / f"{int(frame_indices[0]):06d}.png",
        policy["image_size"],
        policy["patch_size"],
    )
    if image.shape != image_hw:
        raise ValueError(f"preprocessed image {image.shape} differs from depth {image_hw}")
    keypoints, descriptors = detector.detectAndCompute(image, None)
    visual_world_steps = np.diff(visual[:, :3, 3], axis=0)
    visual_local_steps = np.einsum(
        "nij,nj->ni",
        visual[:-1, :3, :3].transpose(0, 2, 1),
        visual_world_steps,
    )
    visual_norms = np.linalg.norm(visual_local_steps, axis=1)
    estimates: list[np.ndarray | None] = []
    gates = []
    rows = []
    accepted_metric_norms = []
    accepted_visual_norms = []
    for index in range(len(visual) - 1):
        next_image = preprocess_image(
            image_dir / f"{int(frame_indices[index + 1]):06d}.png",
            policy["image_size"],
            policy["patch_size"],
        )
        next_keypoints, next_descriptors = detector.detectAndCompute(
            next_image, None
        )
        matches = []
        if descriptors is not None and next_descriptors is not None:
            matches = [
                first
                for first, second in matcher.knnMatch(
                    descriptors, next_descriptors, k=2
                )
                if first.distance < policy["ratio_test"] * second.distance
            ]
        object_points = np.empty((0, 3), dtype=np.float64)
        image_points = np.empty((0, 2), dtype=np.float64)
        sampled_confidence = np.empty(0, dtype=np.float64)
        if matches:
            object_points, valid, sampled_confidence = depth_points_for_matches(
                keypoints,
                matches,
                depth[index],
                confidence[index],
                intrinsics[index],
                pixel_stride,
                policy["minimum_depth"],
                policy["maximum_depth"],
                policy["minimum_depth_confidence"],
            )
            target_pixels = np.asarray(
                [next_keypoints[item.trainIdx].pt for item in matches],
                dtype=np.float64,
            )
            image_points = target_pixels[valid]
        success = False
        pnp_inliers = np.empty(0, dtype=np.int64)
        recovered_local_step = None
        inlier_ratio = 0.0
        reprojection_error = policy["maximum_reprojection_error_px"] * 10.0
        rotation_innovation = np.pi
        forward_ratio = 0.0
        gate_probability = 0.0
        if len(object_points) >= policy["minimum_depth_matches"]:
            success, rvec, tvec, pnp_inliers_raw = cv2.solvePnPRansac(
                object_points,
                image_points,
                intrinsics[index + 1],
                None,
                iterationsCount=policy["pnp_iterations"],
                reprojectionError=policy["pnp_ransac_error_px"],
                confidence=policy["pnp_confidence"],
                flags=cv2.SOLVEPNP_EPNP,
            )
            if success and pnp_inliers_raw is not None:
                pnp_inliers = pnp_inliers_raw[:, 0].astype(np.int64)
                if len(pnp_inliers) >= 6:
                    rvec, tvec = cv2.solvePnPRefineLM(
                        object_points[pnp_inliers],
                        image_points[pnp_inliers],
                        intrinsics[index + 1],
                        None,
                        rvec,
                        tvec,
                    )
                recovered_rotation = cv2.Rodrigues(rvec)[0]
                projected, _ = cv2.projectPoints(
                    object_points[pnp_inliers],
                    rvec,
                    tvec,
                    intrinsics[index + 1],
                    None,
                )
                reprojection_error = float(
                    np.median(
                        np.linalg.norm(
                            projected[:, 0] - image_points[pnp_inliers], axis=1
                        )
                    )
                )
                inlier_ratio = len(pnp_inliers) / max(len(object_points), 1)
                visual_dst_from_src = (
                    np.linalg.inv(visual[index + 1]) @ visual[index]
                )
                rotation_innovation = rotation_error(
                    visual_dst_from_src[:3, :3], recovered_rotation
                )
                recovered_local_step = -recovered_rotation.T @ tvec[:, 0]
                recovered_norm = float(np.linalg.norm(recovered_local_step))
                if recovered_norm > 1e-12:
                    recovered_local_step /= recovered_norm
                    visual_direction = visual_local_steps[index] / max(
                        visual_norms[index], 1e-12
                    )
                    if np.dot(recovered_local_step, visual_direction) < 0.0:
                        recovered_local_step *= -1.0
                    forward_ratio = float(abs(recovered_local_step[2]))
                    gate_probability = pnp_gate_probability(
                        len(pnp_inliers),
                        inlier_ratio,
                        reprojection_error,
                        rotation_innovation,
                        forward_ratio,
                        policy,
                    )
                    recovered_local_step *= recovered_norm
        route_enabled = bool(
            recovered_local_step is not None
            and gate_probability >= policy["route_threshold"]
        )
        estimates.append(recovered_local_step if route_enabled else None)
        gates.append(gate_probability if route_enabled else 0.0)
        if route_enabled:
            accepted_metric_norms.append(float(np.linalg.norm(recovered_local_step)))
            accepted_visual_norms.append(float(visual_norms[index]))
        rows.append({
            "src_frame": int(frame_indices[index]),
            "dst_frame": int(frame_indices[index + 1]),
            "matches": len(matches),
            "depth_matches": len(object_points),
            "median_depth_confidence": (
                float(np.median(sampled_confidence))
                if len(sampled_confidence) else 0.0
            ),
            "pnp_inliers": len(pnp_inliers),
            "pnp_inlier_ratio": inlier_ratio,
            "reprojection_error_px": reprojection_error,
            "rotation_innovation_rad": rotation_innovation,
            "forward_ratio": forward_ratio,
            "gate_probability": gate_probability,
            "route_enabled": route_enabled,
        })
        keypoints, descriptors = next_keypoints, next_descriptors

    if accepted_metric_norms:
        metric_scale = float(
            np.median(accepted_visual_norms)
            / max(np.median(accepted_metric_norms), 1e-12)
        )
    else:
        metric_scale = 1.0
    direction_output = visual.copy()
    metric_output = visual.copy()
    direction_output[0] = visual[0]
    metric_output[0] = visual[0]
    for index, (estimate, gate) in enumerate(zip(estimates, gates)):
        visual_step = visual_local_steps[index]
        visual_norm = visual_norms[index]
        if estimate is None or visual_norm <= 1e-12:
            direction_step = visual_step
            metric_step = visual_step
        else:
            weight = gate * policy["maximum_translation_blend"]
            visual_direction = visual_step / visual_norm
            pnp_direction = estimate / max(float(np.linalg.norm(estimate)), 1e-12)
            direction = normalized_blend(visual_direction, pnp_direction, weight)
            direction_step = direction * visual_norm
            scaled_pnp = estimate * metric_scale
            ratio = float(np.linalg.norm(scaled_pnp)) / visual_norm
            ratio = np.clip(
                ratio,
                policy["minimum_interval_scale_ratio"],
                policy["maximum_interval_scale_ratio"],
            )
            metric_step = direction * visual_norm * (
                (1.0 - weight) + weight * ratio
            )
        direction_output[index + 1, :3, 3] = (
            direction_output[index, :3, 3]
            + visual[index, :3, :3] @ direction_step
        )
        metric_output[index + 1, :3, 3] = (
            metric_output[index, :3, 3]
            + visual[index, :3, :3] @ metric_step
        )
    direction_change = path_change_ratio(visual, direction_output)
    metric_change = path_change_ratio(visual, metric_output)
    direction_accepted = direction_change <= policy["maximum_path_change_ratio"]
    metric_accepted = metric_change <= policy["maximum_path_change_ratio"]
    if not direction_accepted:
        direction_output = visual.copy()
    if not metric_accepted:
        metric_output = visual.copy()
    summary = {
        "interval_count": len(rows),
        "accepted_intervals": int(np.sum(np.asarray(gates) > 0.0)),
        "accepted_fraction": float(np.mean(np.asarray(gates) > 0.0)),
        "gate_mean": float(np.mean(gates)),
        "metric_scale": metric_scale,
        "direction_accepted": direction_accepted,
        "metric_accepted": metric_accepted,
        "direction_path_change_ratio": direction_change,
        "metric_path_change_ratio": metric_change,
    }
    return direction_output, metric_output, rows, summary


def generate(args: argparse.Namespace) -> None:
    policy = json.loads(args.policy.read_text())
    with np.load(args.trajectory, allow_pickle=False) as archive:
        visual = poses_4x4(archive[args.pose_key])
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
    with np.load(args.depth, allow_pickle=False) as archive:
        depth_frames = np.asarray(archive["frame_indices"], dtype=np.int64)
        depth = archive["depth"].astype(np.float32)
        confidence = archive["depth_confidence"].astype(np.float32)
        intrinsics = archive["intrinsics"].astype(np.float64)
        image_hw = tuple(int(value) for value in archive["image_hw"])
        pixel_stride = int(archive["depth_pixel_stride"])
    if not np.array_equal(frame_indices, depth_frames):
        raise ValueError("trajectory and depth frame indices differ")
    direction, metric, rows, summary = generate_candidates(
        visual,
        frame_indices,
        args.image_dir,
        depth,
        confidence,
        intrinsics,
        image_hw,
        pixel_stride,
        policy,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidate_before_gt.npz"
    np.savez_compressed(
        candidate_path,
        direction_c2w=direction,
        metric_c2w=metric,
        frame_indices=frame_indices,
    )
    diagnostics_path = args.output_dir / "depth_pnp_before_gt.json"
    diagnostics_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n"
    )
    depth_metadata_path = args.depth.with_suffix(".json")
    manifest = {
        "protocol": "depth-PnP candidates generated and hashed before GT load",
        "trajectory": str(args.trajectory),
        "trajectory_sha256": sha256(args.trajectory),
        "depth": str(args.depth),
        "depth_sha256": sha256(args.depth),
        "depth_metadata": str(depth_metadata_path),
        "depth_metadata_sha256": sha256(depth_metadata_path),
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
    print(json.dumps(summary, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    from vistream.metrics import ate_rmse

    manifest = json.loads((args.output_dir / "manifest_before_gt.json").read_text())
    if manifest["policy"] != json.loads(args.policy.read_text()):
        raise ValueError("evaluation policy differs from generation policy")
    for path_key, hash_key in (
        ("trajectory", "trajectory_sha256"),
        ("depth", "depth_sha256"),
        ("depth_metadata", "depth_metadata_sha256"),
        ("candidate", "candidate_sha256"),
        ("diagnostics", "diagnostics_sha256"),
    ):
        path = Path(manifest[path_key])
        if sha256(path) != manifest[hash_key]:
            raise ValueError(f"artifact changed after generation: {path}")
    with np.load(manifest["candidate"], allow_pickle=False) as archive:
        direction = poses_4x4(archive["direction_c2w"])
        metric = poses_4x4(archive["metric_c2w"])
    # GT is first opened only after all source and candidate hashes verify.
    with np.load(manifest["trajectory"], allow_pickle=False) as archive:
        visual = poses_4x4(archive[args.pose_key])
        target = poses_4x4(archive[args.ground_truth_key])
    result = {
        "visual_ate_rmse_m": ate_rmse(
            visual[:, :3, 3], target[:, :3, 3], "sim3"
        )["ate_rmse"],
        "direction_ate_rmse_m": ate_rmse(
            direction[:, :3, 3], target[:, :3, 3], "sim3"
        )["ate_rmse"],
        "metric_ate_rmse_m": ate_rmse(
            metric[:, :3, 3], target[:, :3, 3], "sim3"
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
    generate_parser.add_argument("--depth", type=Path, required=True)
    generate_parser.add_argument("--image-dir", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--policy", type=Path, required=True)
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
