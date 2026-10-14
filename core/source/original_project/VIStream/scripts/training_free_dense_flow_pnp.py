#!/usr/bin/env python3
"""GT-free dense RAFT-flow and frozen-depth PnP translation experts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


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


def preprocess_color(path: Path, image_size: int, patch_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    new_width = image_size
    new_height = round(height * (new_width / width) / patch_size) * patch_size
    image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_CUBIC
    )
    if new_height > image_size:
        start = (new_height - image_size) // 2
        image = image[start : start + image_size]
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def sample_bilinear(values: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float32)
    map_x = pixels[:, 0].reshape(-1, 1)
    map_y = pixels[:, 1].reshape(-1, 1)
    sampled = cv2.remap(
        np.asarray(values),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    return sampled.reshape((len(pixels),) + values.shape[2:])


def dense_depth_correspondences(
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
    depth: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    depth_pixel_stride: int,
    policy: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    height, width = forward_flow.shape[:2]
    margin = policy["grid_margin_px"]
    step = policy["grid_step_px"]
    xx, yy = np.meshgrid(
        np.arange(margin, width - margin, step, dtype=np.float32),
        np.arange(margin, height - margin, step, dtype=np.float32),
    )
    source = np.column_stack((xx.ravel(), yy.ravel()))
    source_flow = sample_bilinear(forward_flow, source)
    target = source + source_flow
    inside = (
        (target[:, 0] >= margin)
        & (target[:, 0] < width - margin)
        & (target[:, 1] >= margin)
        & (target[:, 1] < height - margin)
    )
    backward = sample_bilinear(backward_flow, target)
    fb_error = np.linalg.norm(source_flow + backward, axis=1)
    depth_pixels = source / float(depth_pixel_stride)
    depth_values = sample_bilinear(depth[..., None], depth_pixels)[:, 0]
    confidence_values = sample_bilinear(
        confidence[..., None], depth_pixels
    )[:, 0]
    flow_norm = np.linalg.norm(source_flow, axis=1)
    valid = (
        inside
        & np.isfinite(source_flow).all(axis=1)
        & np.isfinite(backward).all(axis=1)
        & np.isfinite(depth_values)
        & np.isfinite(confidence_values)
        & (fb_error <= policy["maximum_forward_backward_error_px"])
        & (flow_norm >= policy["minimum_flow_magnitude_px"])
        & (depth_values >= policy["minimum_depth"])
        & (depth_values <= policy["maximum_depth"])
        & (confidence_values >= policy["minimum_depth_confidence"])
    )
    source = source[valid].astype(np.float64)
    target = target[valid].astype(np.float64)
    depth_values = depth_values[valid].astype(np.float64)
    valid_confidence_values = confidence_values[valid]
    points = np.column_stack((
        (source[:, 0] - intrinsics[0, 2])
        * depth_values
        / intrinsics[0, 0],
        (source[:, 1] - intrinsics[1, 2])
        * depth_values
        / intrinsics[1, 1],
        depth_values,
    ))
    diagnostics = {
        "grid_points": int(len(valid)),
        "valid_correspondences": int(np.sum(valid)),
        "median_forward_backward_error_px": (
            float(np.median(fb_error[valid])) if np.any(valid) else 1e6
        ),
        "median_flow_magnitude_px": (
            float(np.median(flow_norm[valid])) if np.any(valid) else 0.0
        ),
        "median_depth_confidence": (
            float(np.median(valid_confidence_values)) if np.any(valid) else 0.0
        ),
    }
    return points, target, diagnostics


def rotation_error(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.linalg.norm(Rotation.from_matrix(first.T @ second).as_rotvec())
    )


def dense_pnp_gate_probability(
    correspondences: int,
    inliers: int,
    inlier_ratio: float,
    reprojection_error_px: float,
    rotation_innovation_rad: float,
    forward_ratio: float,
    forward_backward_error_px: float,
    policy: dict,
) -> float:
    return soft_and_probability([
        (correspondences - policy["minimum_correspondences"])
        / policy["correspondence_count_width"],
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
        (
            policy["maximum_median_forward_backward_error_px"]
            - forward_backward_error_px
        ) / policy["forward_backward_error_width_px"],
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
    delta = candidate[:, :3, 3] - reference[:, :3, 3]
    return float(np.sqrt(np.mean(np.sum(delta**2, axis=1))) / scale)


def load_raft(policy: dict, device: str):
    import torch
    from torchvision.models.optical_flow import (
        Raft_Large_Weights,
        Raft_Small_Weights,
        raft_large,
        raft_small,
    )

    if policy["raft_model"] == "small":
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=True)
    elif policy["raft_model"] == "large":
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=True)
    else:
        raise ValueError("raft_model must be small or large")
    model.eval().to(torch.device(device))
    cache_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    return model, weights.transforms(), weights.url, cache_path


def infer_bidirectional_flow(
    model, transforms, first: np.ndarray, second: np.ndarray, device: str,
    updates: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as functional

    first_tensor = torch.from_numpy(first).permute(2, 0, 1).float() / 255.0
    second_tensor = torch.from_numpy(second).permute(2, 0, 1).float() / 255.0
    first_batch = torch.stack((first_tensor, second_tensor))
    second_batch = torch.stack((second_tensor, first_tensor))
    height, width = first.shape[:2]
    pad_height = (-height) % 8
    pad_width = (-width) % 8
    first_batch = functional.pad(first_batch, (0, pad_width, 0, pad_height))
    second_batch = functional.pad(second_batch, (0, pad_width, 0, pad_height))
    first_batch, second_batch = transforms(first_batch, second_batch)
    with torch.inference_mode():
        flow = model(
            first_batch.to(device), second_batch.to(device),
            num_flow_updates=updates,
        )[-1]
    flow = flow[:, :, :height, :width].permute(0, 2, 3, 1).float().cpu().numpy()
    return flow[0], flow[1]


def generate_candidates(
    visual: np.ndarray,
    frame_indices: np.ndarray,
    image_dir: Path,
    depth: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    image_hw: tuple[int, int],
    depth_pixel_stride: int,
    policy: dict,
    device: str,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict, str, Path]:
    model, transforms, weights_url, weights_path = load_raft(policy, device)
    image = preprocess_color(
        image_dir / f"{int(frame_indices[0]):06d}.png",
        policy["image_size"], policy["patch_size"],
    )
    if image.shape[:2] != image_hw:
        raise ValueError(f"preprocessed image {image.shape[:2]} differs from depth {image_hw}")
    world_steps = np.diff(visual[:, :3, 3], axis=0)
    local_steps = np.einsum(
        "nij,nj->ni", visual[:-1, :3, :3].transpose(0, 2, 1), world_steps
    )
    visual_norms = np.linalg.norm(local_steps, axis=1)
    estimates: list[np.ndarray | None] = []
    gates = []
    rows = []
    metric_norms = []
    visual_metric_norms = []
    for index in range(len(visual) - 1):
        next_image = preprocess_color(
            image_dir / f"{int(frame_indices[index + 1]):06d}.png",
            policy["image_size"], policy["patch_size"],
        )
        forward, backward = infer_bidirectional_flow(
            model, transforms, image, next_image, device,
            policy["raft_flow_updates"],
        )
        points, target_pixels, flow_diagnostics = dense_depth_correspondences(
            forward, backward, depth[index], confidence[index], intrinsics[index],
            depth_pixel_stride, policy,
        )
        success = False
        inlier_indices = np.empty(0, dtype=np.int64)
        estimate = None
        inlier_ratio = 0.0
        reprojection_error = policy["maximum_reprojection_error_px"] * 10.0
        rotation_innovation = np.pi
        forward_ratio = 0.0
        probability = 0.0
        if len(points) >= policy["minimum_correspondences"]:
            success, rvec, tvec, raw_inliers = cv2.solvePnPRansac(
                points,
                target_pixels,
                intrinsics[index + 1],
                None,
                iterationsCount=policy["pnp_iterations"],
                reprojectionError=policy["pnp_ransac_error_px"],
                confidence=policy["pnp_confidence"],
                flags=cv2.SOLVEPNP_EPNP,
            )
            if success and raw_inliers is not None:
                inlier_indices = raw_inliers[:, 0].astype(np.int64)
                if len(inlier_indices) >= 6:
                    rvec, tvec = cv2.solvePnPRefineLM(
                        points[inlier_indices], target_pixels[inlier_indices],
                        intrinsics[index + 1], None, rvec, tvec,
                    )
                recovered_rotation = cv2.Rodrigues(rvec)[0]
                projected, _ = cv2.projectPoints(
                    points[inlier_indices], rvec, tvec,
                    intrinsics[index + 1], None,
                )
                reprojection_error = float(np.median(np.linalg.norm(
                    projected[:, 0] - target_pixels[inlier_indices], axis=1
                )))
                inlier_ratio = len(inlier_indices) / len(points)
                visual_dst_from_src = np.linalg.inv(visual[index + 1]) @ visual[index]
                rotation_innovation = rotation_error(
                    visual_dst_from_src[:3, :3], recovered_rotation
                )
                estimate = -recovered_rotation.T @ tvec[:, 0]
                estimate_norm = float(np.linalg.norm(estimate))
                if estimate_norm > 1e-12:
                    estimate /= estimate_norm
                    visual_direction = local_steps[index] / max(visual_norms[index], 1e-12)
                    if np.dot(estimate, visual_direction) < 0.0:
                        estimate *= -1.0
                    forward_ratio = float(abs(estimate[2]))
                    probability = dense_pnp_gate_probability(
                        len(points), len(inlier_indices), inlier_ratio,
                        reprojection_error, rotation_innovation, forward_ratio,
                        flow_diagnostics["median_forward_backward_error_px"], policy,
                    )
                    estimate *= estimate_norm
                else:
                    estimate = None
        route_enabled = bool(
            estimate is not None and probability >= policy["route_threshold"]
        )
        estimates.append(estimate if route_enabled else None)
        gates.append(probability if route_enabled else 0.0)
        if route_enabled:
            metric_norms.append(float(np.linalg.norm(estimate)))
            visual_metric_norms.append(float(visual_norms[index]))
        rows.append({
            "src_frame": int(frame_indices[index]),
            "dst_frame": int(frame_indices[index + 1]),
            **flow_diagnostics,
            "pnp_success": bool(success),
            "pnp_inliers": len(inlier_indices),
            "pnp_inlier_ratio": inlier_ratio,
            "reprojection_error_px": reprojection_error,
            "rotation_innovation_rad": rotation_innovation,
            "forward_ratio": forward_ratio,
            "gate_probability": probability,
            "route_enabled": route_enabled,
        })
        image = next_image

    metric_scale = (
        float(np.median(visual_metric_norms) / max(np.median(metric_norms), 1e-12))
        if metric_norms else 1.0
    )
    direction_output = visual.copy()
    metric_output = visual.copy()
    for index, (estimate, gate) in enumerate(zip(estimates, gates)):
        visual_step = local_steps[index]
        visual_norm = visual_norms[index]
        if estimate is None or visual_norm <= 1e-12:
            direction_step = visual_step
            metric_step = visual_step
        else:
            weight = gate * policy["maximum_translation_blend"]
            visual_direction = visual_step / visual_norm
            recovered_direction = estimate / max(float(np.linalg.norm(estimate)), 1e-12)
            direction = normalized_blend(visual_direction, recovered_direction, weight)
            direction_step = direction * visual_norm
            scaled_norm = float(np.linalg.norm(estimate * metric_scale))
            ratio = np.clip(
                scaled_norm / visual_norm,
                policy["minimum_interval_scale_ratio"],
                policy["maximum_interval_scale_ratio"],
            )
            metric_step = direction * visual_norm * ((1.0 - weight) + weight * ratio)
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
    return (
        direction_output, metric_output, rows, summary, weights_url, weights_path
    )


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
    direction, metric, rows, summary, weights_url, weights_path = generate_candidates(
        visual, frame_indices, args.image_dir, depth, confidence, intrinsics,
        image_hw, pixel_stride, policy, args.device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidate_before_gt.npz"
    np.savez_compressed(
        candidate_path, direction_c2w=direction, metric_c2w=metric,
        frame_indices=frame_indices,
    )
    diagnostics_path = args.output_dir / "dense_flow_pnp_before_gt.json"
    diagnostics_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n"
    )
    depth_metadata_path = args.depth.with_suffix(".json")
    manifest = {
        "protocol": "dense-flow PnP candidates generated and hashed before GT load",
        "trajectory": str(args.trajectory),
        "trajectory_sha256": sha256(args.trajectory),
        "depth": str(args.depth),
        "depth_sha256": sha256(args.depth),
        "depth_metadata": str(depth_metadata_path),
        "depth_metadata_sha256": sha256(depth_metadata_path),
        "image_dir": str(args.image_dir),
        "frame_indices": frame_indices.tolist(),
        "image_set_sha256": image_set_sha256(args.image_dir, frame_indices),
        "raft_weights_url": weights_url,
        "raft_weights": str(weights_path),
        "raft_weights_sha256": sha256(weights_path),
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
        ("raft_weights", "raft_weights_sha256"),
        ("candidate", "candidate_sha256"),
        ("diagnostics", "diagnostics_sha256"),
    ):
        path = Path(manifest[path_key])
        if sha256(path) != manifest[hash_key]:
            raise ValueError(f"artifact changed after generation: {path}")
    frame_indices = np.asarray(manifest["frame_indices"], dtype=np.int64)
    if image_set_sha256(Path(manifest["image_dir"]), frame_indices) != manifest["image_set_sha256"]:
        raise ValueError("image set changed after candidate generation")
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
    generate_parser.add_argument("--device", default="cuda:7")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--policy", type=Path, required=True)
    evaluate_parser.add_argument("--pose-key", default="predicted_c2w")
    evaluate_parser.add_argument("--ground-truth-key", default="ground_truth_c2w")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generate(arguments) if arguments.command == "generate" else evaluate(arguments)
