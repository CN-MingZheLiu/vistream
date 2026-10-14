#!/usr/bin/env python3
"""Training-free visual loop closure for cached Lingbot KITTI trajectories.

The ``generate`` command never opens ground truth. It retrieves non-local
frames with a fixed ImageNet descriptor, verifies them with SIFT and an
essential matrix, and writes a pose-graph candidate before evaluation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class LoopEdge:
    src_pos: int
    dst_pos: int
    src_frame: int
    dst_frame: int
    retrieval_score: float
    matches: int
    inliers: int
    inlier_ratio: float
    transform_dst_from_src: np.ndarray


@dataclass(frozen=True)
class EnvironmentGate:
    mode: str
    probability: float
    route_enabled: bool
    normalized_cycle_residual: float
    median_retrieval_score: float
    median_inlier_ratio: float
    edge_count: int


def poses_4x4(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1:] == (4, 4):
        return values.copy()
    if values.shape[1:] != (3, 4):
        raise ValueError(f"expected [N,3,4] or [N,4,4], got {values.shape}")
    output = np.repeat(np.eye(4)[None], len(values), axis=0)
    output[:, :3, :4] = values
    return output


def relative_dst_from_src(src_c2w: np.ndarray, dst_c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(dst_c2w) @ src_c2w


def matrix_to_vector(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (Rotation.from_matrix(matrix[:3, :3]).as_rotvec(), matrix[:3, 3])
    )


def vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_rotvec(vector[:3]).as_matrix()
    matrix[:3, 3] = vector[3:]
    return matrix


def sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = np.exp(-value)
        return float(1.0 / (1.0 + inverse))
    exponential = np.exp(value)
    return float(exponential / (1.0 + exponential))


def environment_gate(
    initial_poses: np.ndarray,
    loop_edges: list[LoopEdge],
    mode: str,
    center: float,
    slope: float,
    route_threshold: float,
    minimum_edges: int,
) -> EnvironmentGate:
    """Route between visual-only and loop experts without supervision.

    The fixed head compares loop-cycle translation residual to the square-root
    random-walk growth expected over the same temporal separation. ResNet
    retrieval and SIFT verification establish environmental repeatability;
    the cycle feature decides whether the revisit exposes accumulated drift or
    merely repeats an already-consistent local geometry.
    """
    if mode not in {"none", "frozen_random_walk_mlp"}:
        raise ValueError(f"unsupported environment gate: {mode}")
    if not loop_edges:
        return EnvironmentGate(mode, 0.0, False, 0.0, 0.0, 0.0, 0)
    scores = np.asarray([edge.retrieval_score for edge in loop_edges])
    inlier_ratios = np.asarray([edge.inlier_ratio for edge in loop_edges])
    if mode == "none":
        return EnvironmentGate(
            mode, 1.0, True, 0.0, float(np.median(scores)),
            float(np.median(inlier_ratios)), len(loop_edges)
        )
    steps = np.linalg.norm(np.diff(initial_poses[:, :3, 3], axis=0), axis=1)
    positive_steps = steps[steps > 1e-9]
    step_scale = float(np.median(positive_steps)) if len(positive_steps) else 1.0
    normalized_residuals = []
    for edge in loop_edges:
        visual_relative = relative_dst_from_src(
            initial_poses[edge.src_pos], initial_poses[edge.dst_pos]
        )
        cycle_residual = float(
            np.linalg.norm(
                visual_relative[:3, 3] - edge.transform_dst_from_src[:3, 3]
            )
        )
        random_walk_scale = step_scale * np.sqrt(
            max(1, edge.dst_pos - edge.src_pos)
        )
        normalized_residuals.append(cycle_residual / max(random_walk_scale, 1e-9))
    normalized_cycle = float(np.median(normalized_residuals))
    # This is a frozen one-neuron MLP. Its parameters come from the
    # random-walk null model, not from target-pose fitting.
    logit = slope * np.log(max(normalized_cycle, 1e-9) / center)
    probability = sigmoid(logit)
    enabled = len(loop_edges) >= minimum_edges and probability >= route_threshold
    return EnvironmentGate(
        mode,
        probability,
        enabled,
        normalized_cycle,
        float(np.median(scores)),
        float(np.median(inlier_ratios)),
        len(loop_edges),
    )


def read_intrinsics(calibration: Path, scale: float) -> np.ndarray:
    for line in calibration.read_text().splitlines():
        if line.startswith("P2:"):
            projection = np.fromstring(line.split(":", 1)[1], sep=" ").reshape(3, 4)
            intrinsics = projection[:, :3].copy()
            intrinsics[:2] *= scale
            return intrinsics
    raise ValueError(f"{calibration} has no P2 entry")


def load_cache(path: Path, pose_key: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        poses = poses_4x4(archive[pose_key])
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
    if len(poses) != len(frame_indices):
        raise ValueError("pose and frame-index lengths differ")
    return poses, frame_indices


def compute_descriptors(
    image_dir: Path,
    frame_indices: np.ndarray,
    output: Path,
    device: str,
    batch_size: int,
    image_size: tuple[int, int],
) -> np.ndarray:
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(frame_indices), batch_size):
            images = []
            for frame in frame_indices[start : start + batch_size]:
                image = cv2.imread(str(image_dir / f"{int(frame):06d}.png"))
                if image is None:
                    raise FileNotFoundError(image_dir / f"{int(frame):06d}.png")
                image = cv2.cvtColor(cv2.resize(image, image_size), cv2.COLOR_BGR2RGB)
                images.append(image)
            batch = torch.from_numpy(np.stack(images)).to(device)
            batch = batch.permute(0, 3, 1, 2).float() / 255.0
            descriptor = model((batch - mean) / std)
            chunks.append(torch.nn.functional.normalize(descriptor, dim=1).cpu().numpy())
    descriptors = np.concatenate(chunks).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, descriptors)
    return descriptors


def retrieve_pairs(
    descriptors: np.ndarray,
    minimum_separation: int,
    nms_radius: int,
    maximum_pairs: int,
) -> list[tuple[int, int, float]]:
    similarity = descriptors @ descriptors.T
    src, dst = np.triu_indices(len(descriptors), k=minimum_separation)
    order = np.argsort(similarity[src, dst])[::-1]
    output: list[tuple[int, int, float]] = []
    for index in order:
        i, j = int(src[index]), int(dst[index])
        if any(abs(i - a) < nms_radius and abs(j - b) < nms_radius for a, b, _ in output):
            continue
        output.append((i, j, float(similarity[i, j])))
        if len(output) >= maximum_pairs:
            break
    return output


def verify_pair(
    image_dir: Path,
    frame_indices: np.ndarray,
    pair: tuple[int, int, float],
    intrinsics: np.ndarray,
    initial_poses: np.ndarray,
    sift_nfeatures: int,
    sift_contrast_threshold: float,
    ratio_test: float,
    minimum_matches: int,
    minimum_inliers: int,
    minimum_inlier_ratio: float,
) -> LoopEdge | None:
    src, dst, score = pair
    images = []
    for position in (src, dst):
        image = cv2.imread(str(image_dir / f"{int(frame_indices[position]):06d}.png"), 0)
        if image is None:
            return None
        images.append(cv2.resize(image, None, fx=0.5, fy=0.5))
    detector = cv2.SIFT_create(
        nfeatures=sift_nfeatures, contrastThreshold=sift_contrast_threshold
    )
    key_a, desc_a = detector.detectAndCompute(images[0], None)
    key_b, desc_b = detector.detectAndCompute(images[1], None)
    if desc_a is None or desc_b is None:
        return None
    matches = [
        first
        for first, second in cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a, desc_b, k=2)
        if first.distance < ratio_test * second.distance
    ]
    if len(matches) < minimum_matches:
        return None
    points_a = np.float32([key_a[item.queryIdx].pt for item in matches])
    points_b = np.float32([key_b[item.trainIdx].pt for item in matches])
    essential, mask = cv2.findEssentialMat(
        points_a, points_b, intrinsics, cv2.RANSAC, 0.999, 1.5
    )
    if essential is None or mask is None:
        return None
    if essential.shape[0] > 3:
        essential = essential[:3]
    try:
        inliers, rotation, direction, pose_mask = cv2.recoverPose(
            essential, points_a, points_b, intrinsics, mask=mask
        )
    except cv2.error:
        return None
    inlier_ratio = float(inliers) / len(matches)
    if inliers < minimum_inliers or inlier_ratio < minimum_inlier_ratio:
        return None
    visual_relative = relative_dst_from_src(initial_poses[src], initial_poses[dst])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = direction[:, 0] * max(
        float(np.linalg.norm(visual_relative[:3, 3])), 1e-3
    )
    return LoopEdge(
        src,
        dst,
        int(frame_indices[src]),
        int(frame_indices[dst]),
        score,
        len(matches),
        int(inliers),
        inlier_ratio,
        transform,
    )


def optimize_pose_graph(
    initial_poses: np.ndarray,
    loop_edges: list[LoopEdge],
    loop_weight: float,
    maximum_evaluations: int,
) -> np.ndarray:
    if not loop_edges:
        return initial_poses.copy()
    node_count = len(initial_poses)
    odometry = [
        (index, index + 1, relative_dst_from_src(initial_poses[index], initial_poses[index + 1]))
        for index in range(node_count - 1)
    ]
    x0 = np.concatenate([matrix_to_vector(pose) for pose in initial_poses[1:]])

    def unpack(vector: np.ndarray) -> list[np.ndarray]:
        return [initial_poses[0]] + [
            vector_to_matrix(vector[start : start + 6])
            for start in range(0, len(vector), 6)
        ]

    def residual(vector: np.ndarray) -> np.ndarray:
        poses = unpack(vector)
        values = []
        for src, dst, measurement in odometry:
            prediction = relative_dst_from_src(poses[src], poses[dst])
            values.extend(Rotation.from_matrix(measurement[:3, :3].T @ prediction[:3, :3]).as_rotvec())
            values.extend(0.3 * (prediction[:3, 3] - measurement[:3, 3]))
        for edge in loop_edges:
            prediction = relative_dst_from_src(poses[edge.src_pos], poses[edge.dst_pos])
            weight = loop_weight * np.sqrt(edge.inliers / 100.0)
            values.extend(weight * Rotation.from_matrix(
                edge.transform_dst_from_src[:3, :3].T @ prediction[:3, :3]
            ).as_rotvec())
            values.extend(weight * 0.3 * (
                prediction[:3, 3] - edge.transform_dst_from_src[:3, 3]
            ))
        return np.asarray(values)

    all_pairs = [(src, dst) for src, dst, _ in odometry] + [
        (edge.src_pos, edge.dst_pos) for edge in loop_edges
    ]
    sparsity = lil_matrix((6 * len(all_pairs), 6 * (node_count - 1)), dtype=np.int8)
    for row, (src, dst) in enumerate(all_pairs):
        for node in (src, dst):
            if node > 0:
                sparsity[6 * row : 6 * row + 6, 6 * (node - 1) : 6 * node] = 1
    result = least_squares(
        residual,
        x0,
        jac_sparsity=sparsity,
        loss="huber",
        f_scale=1.0,
        max_nfev=maximum_evaluations,
        x_scale="jac",
    )
    return np.asarray(unpack(result.x))


def optimize_rotation_graph(
    initial_poses: np.ndarray,
    loop_edges: list[LoopEdge],
    loop_weight: float,
    maximum_evaluations: int,
) -> np.ndarray:
    """Optimize SO(3), then repropagate visual body-frame translation steps."""
    if not loop_edges:
        return initial_poses.copy()
    node_count = len(initial_poses)
    initial_rotations = initial_poses[:, :3, :3]
    odometry = [
        (index, index + 1, initial_rotations[index + 1].T @ initial_rotations[index])
        for index in range(node_count - 1)
    ]
    x0 = np.concatenate(
        [Rotation.from_matrix(matrix).as_rotvec() for matrix in initial_rotations[1:]]
    )

    def unpack(vector: np.ndarray) -> list[np.ndarray]:
        return [initial_rotations[0]] + [
            Rotation.from_rotvec(vector[start : start + 3]).as_matrix()
            for start in range(0, len(vector), 3)
        ]

    def residual(vector: np.ndarray) -> np.ndarray:
        rotations = unpack(vector)
        values = []
        for src, dst, measurement in odometry:
            prediction = rotations[dst].T @ rotations[src]
            values.extend(Rotation.from_matrix(measurement.T @ prediction).as_rotvec())
        for edge in loop_edges:
            prediction = rotations[edge.dst_pos].T @ rotations[edge.src_pos]
            weight = loop_weight * np.sqrt(edge.inliers / 100.0)
            values.extend(
                weight
                * Rotation.from_matrix(
                    edge.transform_dst_from_src[:3, :3].T @ prediction
                ).as_rotvec()
            )
        return np.asarray(values)

    all_pairs = [(src, dst) for src, dst, _ in odometry] + [
        (edge.src_pos, edge.dst_pos) for edge in loop_edges
    ]
    sparsity = lil_matrix((3 * len(all_pairs), 3 * (node_count - 1)), dtype=np.int8)
    for row, (src, dst) in enumerate(all_pairs):
        for node in (src, dst):
            if node > 0:
                sparsity[3 * row : 3 * row + 3, 3 * (node - 1) : 3 * node] = 1
    result = least_squares(
        residual,
        x0,
        jac_sparsity=sparsity,
        loss="huber",
        f_scale=1.0,
        max_nfev=maximum_evaluations,
        x_scale="jac",
    )
    optimized_rotations = np.asarray(unpack(result.x))
    output = initial_poses.copy()
    output[:, :3, :3] = optimized_rotations
    visual_steps = np.diff(initial_poses[:, :3, 3], axis=0)
    body_steps = np.einsum(
        "nij,nj->ni", initial_rotations[:-1].transpose(0, 2, 1), visual_steps
    )
    for index, step in enumerate(body_steps):
        output[index + 1, :3, 3] = (
            output[index, :3, 3] + optimized_rotations[index] @ step
        )
    return output


def save_edges(path: Path, edges: list[LoopEdge]) -> None:
    payload = []
    for edge in edges:
        item = asdict(edge)
        item["transform_dst_from_src"] = edge.transform_dst_from_src.tolist()
        payload.append(item)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def generate(args: argparse.Namespace) -> None:
    initial, frame_indices = load_cache(args.cache, args.pose_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    descriptor_path = args.output_dir / "resnet50_imagenet.npy"
    descriptors = (
        np.load(descriptor_path)
        if descriptor_path.exists()
        else compute_descriptors(
            args.image_dir,
            frame_indices,
            descriptor_path,
            args.device,
            args.batch_size,
            (args.descriptor_width, args.descriptor_height),
        )
    )
    intrinsics = read_intrinsics(args.calibration, 0.5)
    edges = []
    for pair in retrieve_pairs(
        descriptors, args.minimum_separation, args.nms_radius, args.maximum_pairs
    ):
        edge = verify_pair(
            args.image_dir,
            frame_indices,
            pair,
            intrinsics,
            initial,
            args.sift_nfeatures,
            args.sift_contrast_threshold,
            args.ratio_test,
            args.minimum_matches,
            args.minimum_inliers,
            args.minimum_inlier_ratio,
        )
        if edge is not None:
            edges.append(edge)
    gate = environment_gate(
        initial,
        edges,
        args.environment_gate,
        args.environment_gate_center,
        args.environment_gate_slope,
        args.environment_gate_route_threshold,
        args.environment_gate_minimum_edges,
    )
    active_edges = edges if gate.route_enabled else []
    save_edges(args.output_dir / "loop_edges_before_gt.json", edges)
    (args.output_dir / "environment_gate_before_gt.json").write_text(
        json.dumps(asdict(gate), indent=2) + "\n"
    )
    optimizer = (
        optimize_rotation_graph if args.pose_graph == "rotation" else optimize_pose_graph
    )
    optimized = optimizer(
        initial, active_edges, args.loop_weight, args.maximum_evaluations
    )
    np.savez_compressed(
        args.output_dir / "candidate_before_gt.npz",
        c2w=optimized,
        frame_indices=frame_indices,
        loop_count=len(edges),
        active_loop_count=len(active_edges),
        environment_gate_probability=gate.probability,
    )
    print(json.dumps({
        "loops": len(edges),
        "active_loops": len(active_edges),
        "environment_gate_probability": gate.probability,
        "candidate": str(args.output_dir / "candidate_before_gt.npz"),
    }))


def evaluate(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vistream.metrics import ate_rmse

    with np.load(args.candidate, allow_pickle=False) as candidate:
        predicted = poses_4x4(candidate["c2w"])
    with np.load(args.cache, allow_pickle=False) as archive:
        target = poses_4x4(archive[args.ground_truth_key])
        visual = poses_4x4(archive[args.pose_key])
    result = {
        "visual": ate_rmse(visual[:, :3, 3], target[:, :3, 3], "sim3"),
        "training_free_loop": ate_rmse(predicted[:, :3, 3], target[:, :3, 3], "sim3"),
    }
    for value in result.values():
        value.pop("aligned", None)
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--cache", type=Path, required=True)
    generate_parser.add_argument("--pose-key", default="fused_c2w")
    generate_parser.add_argument("--image-dir", type=Path, required=True)
    generate_parser.add_argument("--calibration", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--device", default="cuda")
    generate_parser.add_argument("--batch-size", type=int, default=64)
    generate_parser.add_argument("--descriptor-width", type=int, default=320)
    generate_parser.add_argument("--descriptor-height", type=int, default=192)
    generate_parser.add_argument("--minimum-separation", type=int, default=40)
    generate_parser.add_argument("--nms-radius", type=int, default=8)
    generate_parser.add_argument("--maximum-pairs", type=int, default=300)
    generate_parser.add_argument("--loop-weight", type=float, default=0.1)
    generate_parser.add_argument(
        "--pose-graph", choices=("rotation", "se3"), default="rotation"
    )
    generate_parser.add_argument("--maximum-evaluations", type=int, default=100)
    generate_parser.add_argument("--sift-nfeatures", type=int, default=4000)
    generate_parser.add_argument("--sift-contrast-threshold", type=float, default=0.01)
    generate_parser.add_argument("--ratio-test", type=float, default=0.72)
    generate_parser.add_argument("--minimum-matches", type=int, default=100)
    generate_parser.add_argument("--minimum-inliers", type=int, default=80)
    generate_parser.add_argument("--minimum-inlier-ratio", type=float, default=0.8)
    generate_parser.add_argument(
        "--environment-gate",
        choices=("none", "frozen_random_walk_mlp"),
        default="none",
    )
    generate_parser.add_argument("--environment-gate-center", type=float, default=1.0)
    generate_parser.add_argument("--environment-gate-slope", type=float, default=8.0)
    generate_parser.add_argument(
        "--environment-gate-route-threshold", type=float, default=0.5
    )
    generate_parser.add_argument(
        "--environment-gate-minimum-edges", type=int, default=2
    )
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--cache", type=Path, required=True)
    evaluate_parser.add_argument("--candidate", type=Path, required=True)
    evaluate_parser.add_argument("--pose-key", default="fused_c2w")
    evaluate_parser.add_argument("--ground-truth-key", default="ground_truth_c2w")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args) if args.command == "generate" else evaluate(args)


if __name__ == "__main__":
    main()
