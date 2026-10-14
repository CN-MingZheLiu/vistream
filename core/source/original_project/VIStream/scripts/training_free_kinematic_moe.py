#!/usr/bin/env python3
"""GT-free environment gate and vehicle-kinematic translation experts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation


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
    """A frozen monotone gate head whose weakest physical cue dominates."""
    return sigmoid(min(margins))


def rotation_angle(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(Rotation.from_matrix(matrix).as_rotvec()))


def midpoint_rotation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    relative = first.T @ second
    half = 0.5 * Rotation.from_matrix(relative).as_rotvec()
    return first @ Rotation.from_rotvec(half).as_matrix()


def robust_cv(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    if median <= 1e-12:
        return 0.0
    return float(1.4826 * np.median(np.abs(values - median)) / median)


def smooth_step_lengths(step_lengths: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("speed smoothing window must be a positive odd integer")
    filtered = median_filter(step_lengths, size=window, mode="nearest")
    source_median = float(np.median(step_lengths))
    filtered_median = float(np.median(filtered))
    if filtered_median > 1e-12:
        filtered *= source_median / filtered_median
    return filtered


def kinematic_trajectory(
    visual: np.ndarray,
    rotations: np.ndarray,
    step_lengths: np.ndarray,
    forward_sign: float,
) -> np.ndarray:
    if len(visual) != len(rotations) or len(step_lengths) != len(visual) - 1:
        raise ValueError("trajectory, rotations, and step lengths disagree")
    output = visual.copy()
    output[:, :3, :3] = rotations
    output[0, :3, 3] = visual[0, :3, 3]
    camera_forward = np.asarray([0.0, 0.0, forward_sign], dtype=np.float64)
    for index, step_length in enumerate(step_lengths):
        direction = midpoint_rotation(rotations[index], rotations[index + 1]) @ camera_forward
        output[index + 1, :3, 3] = (
            output[index, :3, 3] + direction * float(step_length)
        )
    return output


def path_change_ratio(reference: np.ndarray, candidate: np.ndarray) -> float:
    steps = np.linalg.norm(np.diff(reference[:, :3, 3], axis=0), axis=1)
    scale = max(
        float(np.linalg.norm(reference[-1, :3, 3] - reference[0, :3, 3])),
        0.1 * float(np.sum(steps)),
        1e-9,
    )
    displacement = candidate[:, :3, 3] - reference[:, :3, 3]
    return float(np.sqrt(np.mean(np.sum(displacement**2, axis=1))) / scale)


def environment_features(
    visual: np.ndarray,
    imu: np.ndarray | None,
    loop_gate: dict,
) -> dict[str, float | int | bool]:
    world_steps = np.diff(visual[:, :3, 3], axis=0)
    lengths = np.linalg.norm(world_steps, axis=1)
    valid = lengths >= max(float(np.quantile(lengths, 0.1)), 1e-9)
    local_steps = np.einsum(
        "nij,nj->ni", visual[:-1, :3, :3].transpose(0, 2, 1), world_steps
    )
    forward = local_steps[:, 2] / np.maximum(lengths, 1e-9)
    signs = np.sign(forward[valid])
    forward_sign = 1.0 if np.mean(signs >= 0.0) >= np.mean(signs <= 0.0) else -1.0
    visual_turns = np.asarray([
        rotation_angle(visual[index, :3, :3].T @ visual[index + 1, :3, :3])
        for index in range(len(visual) - 1)
    ])
    result: dict[str, float | int | bool] = {
        "frame_count": len(visual),
        "loop_edge_count": int(loop_gate.get("edge_count", 0)),
        "loop_probability": float(loop_gate.get("probability", 0.0)),
        "forward_ratio_median": float(np.median(np.abs(forward[valid]))),
        "forward_sign": forward_sign,
        "forward_sign_consistency": float(
            max(np.mean(signs >= 0.0), np.mean(signs <= 0.0))
        ),
        "visual_turn_median_rad": float(np.median(visual_turns)),
        "visual_turn_p90_rad": float(np.quantile(visual_turns, 0.9)),
        "visual_step_robust_cv": robust_cv(lengths[valid]),
        "imu_available": imu is not None,
    }
    if imu is None:
        result.update({
            "imu_rotation_innovation_median_rad": float(np.pi),
            "imu_rotation_innovation_p90_rad": float(np.pi),
        })
    else:
        innovations = []
        for index in range(len(visual) - 1):
            visual_relative = visual[index, :3, :3].T @ visual[index + 1, :3, :3]
            imu_relative = imu[index, :3, :3].T @ imu[index + 1, :3, :3]
            innovations.append(rotation_angle(visual_relative.T @ imu_relative))
        result.update({
            "imu_rotation_innovation_median_rad": float(np.median(innovations)),
            "imu_rotation_innovation_p90_rad": float(np.quantile(innovations, 0.9)),
        })
    return result


def frozen_environment_gate(features: dict, policy: dict) -> dict:
    open_road_probability = soft_and_probability([
        (policy["maximum_open_road_loop_probability"] - features["loop_probability"])
        / policy["loop_probability_width"],
        (features["forward_ratio_median"] - policy["minimum_forward_ratio"])
        / policy["forward_ratio_width"],
        (
            features["forward_sign_consistency"]
            - policy["minimum_forward_sign_consistency"]
        ) / policy["forward_sign_width"],
        (features["frame_count"] - policy["minimum_open_road_frames"])
        / policy["frame_count_width"],
    ])
    imu_rotation_probability = soft_and_probability([
        10.0 if features["imu_available"] else -10.0,
        (
            policy["maximum_imu_rotation_innovation_median_rad"]
            - features["imu_rotation_innovation_median_rad"]
        ) / policy["imu_rotation_innovation_median_width_rad"],
        (
            policy["maximum_imu_rotation_innovation_p90_rad"]
            - features["imu_rotation_innovation_p90_rad"]
        ) / policy["imu_rotation_innovation_p90_width_rad"],
    ])
    speed_smoothing_probability = sigmoid(
        (features["visual_step_robust_cv"] - policy["speed_cv_smoothing_center"])
        / policy["speed_cv_smoothing_width"]
    )
    route_enabled = open_road_probability >= policy["route_threshold"]
    rotation_source = (
        "imu" if imu_rotation_probability >= policy["route_threshold"] else "visual"
    )
    speed_source = (
        "smooth"
        if speed_smoothing_probability >= policy["route_threshold"]
        else "raw"
    )
    selected_expert = (
        f"{rotation_source}_forward_{speed_source}" if route_enabled else "visual"
    )
    return {
        "open_road_probability": open_road_probability,
        "imu_rotation_probability": imu_rotation_probability,
        "speed_smoothing_probability": speed_smoothing_probability,
        "route_enabled": route_enabled,
        "rotation_source": rotation_source,
        "speed_source": speed_source,
        "selected_expert": selected_expert,
    }


def generate(args: argparse.Namespace) -> None:
    policy = json.loads(args.policy.read_text())
    entries = []
    for sequence in SEQUENCES:
        visual_path = args.visual_root / sequence / args.visual_name
        imu_path = args.imu_root / sequence / args.imu_name
        imu_diagnostics_path = imu_path.with_suffix(".diagnostics.json")
        loop_gate_path = args.loop_root / sequence / args.loop_gate_name
        with np.load(visual_path, allow_pickle=False) as archive:
            visual = poses_4x4(archive[args.visual_pose_key])
            frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        imu = None
        imu_accepted = False
        if imu_path.exists() and imu_diagnostics_path.exists():
            imu_diagnostics = json.loads(imu_diagnostics_path.read_text())
            imu_accepted = bool(
                imu_diagnostics.get("accepted", False)
                and imu_diagnostics.get("fallback_reason") is None
            )
            if imu_accepted:
                with np.load(imu_path, allow_pickle=False) as archive:
                    imu = poses_4x4(archive[args.imu_pose_key])
        loop_gate = json.loads(loop_gate_path.read_text())
        features = environment_features(visual, imu, loop_gate)
        features["imu_source_accepted"] = imu_accepted
        gate = frozen_environment_gate(features, policy)

        step_lengths = np.linalg.norm(
            np.diff(visual[:, :3, 3], axis=0), axis=1
        )
        smooth_lengths = smooth_step_lengths(
            step_lengths, policy["speed_smoothing_window"]
        )
        constant_lengths = np.full_like(step_lengths, np.median(step_lengths))
        rotations = {
            "visual": visual[:, :3, :3],
            "imu": imu[:, :3, :3] if imu is not None else visual[:, :3, :3],
        }
        speeds = {
            "raw": step_lengths,
            "smooth": smooth_lengths,
            "constant": constant_lengths,
        }
        candidates = {
            f"{rotation_name}_forward_{speed_name}": kinematic_trajectory(
                visual,
                rotation_values,
                speed_values,
                float(features["forward_sign"]),
            )
            for rotation_name, rotation_values in rotations.items()
            for speed_name, speed_values in speeds.items()
        }
        selected = visual.copy()
        fallback_reason = None
        if gate["route_enabled"]:
            selected = candidates[gate["selected_expert"]].copy()
            change = path_change_ratio(visual, selected)
            if change > policy["maximum_path_change_ratio"]:
                selected = visual.copy()
                gate["route_enabled"] = False
                gate["selected_expert"] = "visual"
                fallback_reason = "path_change_budget_exceeded"
        else:
            change = 0.0
            fallback_reason = "environment_gate_rejected"
        candidate_changes = {
            name: path_change_ratio(visual, value)
            for name, value in candidates.items()
        }
        diagnostics = {
            "sequence": sequence,
            "features": features,
            "gate": gate,
            "selected_path_change_ratio": change,
            "candidate_path_change_ratios": candidate_changes,
            "fallback_reason": fallback_reason,
        }
        output_dir = args.output_dir / sequence
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = output_dir / "candidate_before_gt.npz"
        np.savez_compressed(
            candidate_path,
            visual=visual,
            selected_c2w=selected,
            frame_indices=frame_indices,
            **{f"candidate_{name}": value for name, value in candidates.items()},
        )
        diagnostics_path = output_dir / "kinematic_gate_before_gt.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
        entry = {
            "sequence": sequence,
            "visual": str(visual_path),
            "visual_sha256": sha256(visual_path),
            "imu": str(imu_path) if imu_path.exists() else None,
            "imu_sha256": sha256(imu_path) if imu_path.exists() else None,
            "imu_diagnostics": (
                str(imu_diagnostics_path) if imu_diagnostics_path.exists() else None
            ),
            "imu_diagnostics_sha256": (
                sha256(imu_diagnostics_path) if imu_diagnostics_path.exists() else None
            ),
            "loop_gate": str(loop_gate_path),
            "loop_gate_sha256": sha256(loop_gate_path),
            "candidate": str(candidate_path),
            "candidate_sha256": sha256(candidate_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": sha256(diagnostics_path),
        }
        entries.append(entry)
        print(json.dumps({"sequence": sequence, **gate}), flush=True)
    manifest = {
        "protocol": "all frozen kinematic experts and routes hashed before GT load",
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

    manifest_path = args.output_dir / "manifest_before_gt.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["policy"] != json.loads(args.policy.read_text()):
        raise ValueError("evaluation policy differs from generation policy")
    verified = []
    for entry in manifest["entries"]:
        for path_key, hash_key in (
            ("visual", "visual_sha256"),
            ("imu", "imu_sha256"),
            ("imu_diagnostics", "imu_diagnostics_sha256"),
            ("loop_gate", "loop_gate_sha256"),
            ("candidate", "candidate_sha256"),
            ("diagnostics", "diagnostics_sha256"),
        ):
            if entry.get(path_key) is None:
                continue
            path = Path(entry[path_key])
            if sha256(path) != entry[hash_key]:
                raise ValueError(f"artifact changed after generation: {path}")
        verified.append(entry)

    rows = []
    candidate_names: list[str] | None = None
    for entry in verified:
        with np.load(entry["candidate"], allow_pickle=False) as archive:
            visual = poses_4x4(archive["visual"])
            selected = poses_4x4(archive["selected_c2w"])
            names = sorted(key for key in archive.files if key.startswith("candidate_"))
            candidates = {key.removeprefix("candidate_"): poses_4x4(archive[key]) for key in names}
        if candidate_names is None:
            candidate_names = sorted(candidates)
        # No GT is opened until every sequence source, route, and candidate is verified.
        with np.load(entry["visual"], allow_pickle=False) as archive:
            target = poses_4x4(archive[args.ground_truth_key])
        diagnostics = json.loads(Path(entry["diagnostics"]).read_text())
        metrics = {
            name: ate_rmse(value[:, :3, 3], target[:, :3, 3], "sim3")["ate_rmse"]
            for name, value in candidates.items()
        }
        row = {
            "sequence": entry["sequence"],
            "selected_expert": diagnostics["gate"]["selected_expert"],
            "visual_ate_rmse_m": ate_rmse(
                visual[:, :3, 3], target[:, :3, 3], "sim3"
            )["ate_rmse"],
            "selected_ate_rmse_m": ate_rmse(
                selected[:, :3, 3], target[:, :3, 3], "sim3"
            )["ate_rmse"],
            "candidate_ate_rmse_m": metrics,
        }
        rows.append(row)
    assert candidate_names is not None
    visual_values = np.asarray([row["visual_ate_rmse_m"] for row in rows])
    selected_values = np.asarray([row["selected_ate_rmse_m"] for row in rows])
    candidate_macro = {
        name: float(np.mean([row["candidate_ate_rmse_m"][name] for row in rows]))
        for name in candidate_names
    }
    oracle_values = np.asarray([
        min([row["visual_ate_rmse_m"], *row["candidate_ate_rmse_m"].values()])
        for row in rows
    ])
    summary = {
        "protocol": manifest["protocol"],
        "policy": manifest["policy"],
        "rows": rows,
        "macro": {
            "visual_ate_rmse_mean_m": float(np.mean(visual_values)),
            "selected_ate_rmse_mean_m": float(np.mean(selected_values)),
            "selected_improved_sequences": int(np.sum(selected_values < visual_values)),
            "candidate_ate_rmse_mean_m": candidate_macro,
            "diagnostic_gt_oracle_mean_m": float(np.mean(oracle_values)),
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
    generate_parser.add_argument("--visual-root", type=Path, required=True)
    generate_parser.add_argument("--imu-root", type=Path, required=True)
    generate_parser.add_argument("--loop-root", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--policy", type=Path, required=True)
    generate_parser.add_argument("--visual-name", default="trajectory.npz")
    generate_parser.add_argument("--imu-name", default="candidate_before_gt.npz")
    generate_parser.add_argument("--loop-gate-name", default="environment_gate_before_gt.json")
    generate_parser.add_argument("--visual-pose-key", default="predicted_c2w")
    generate_parser.add_argument("--imu-pose-key", default="fused_c2w")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--policy", type=Path, required=True)
    evaluate_parser.add_argument("--ground-truth-key", default="ground_truth_c2w")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generate(arguments) if arguments.command == "generate" else evaluate(arguments)
