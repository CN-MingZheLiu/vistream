#!/usr/bin/env python3
"""Frozen GT-free gate for accepting a DPVO trajectory expert."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 3 or len(second) != len(first):
        return 0.0
    first_rank = np.argsort(np.argsort(first, kind="stable"), kind="stable")
    second_rank = np.argsort(np.argsort(second, kind="stable"), kind="stable")
    correlation = np.corrcoef(first_rank, second_rank)[0, 1]
    return float(correlation) if np.isfinite(correlation) else 0.0


def turn_angles(poses: np.ndarray) -> np.ndarray:
    return np.asarray([
        np.linalg.norm(
            Rotation.from_matrix(
                poses[index, :3, :3].T @ poses[index + 1, :3, :3]
            ).as_rotvec()
        )
        for index in range(len(poses) - 1)
    ])


def endpoint_to_path_ratio(poses: np.ndarray) -> float:
    path = float(np.sum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)))
    endpoint = float(np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3]))
    return endpoint / max(path, 1e-12)


def dpvo_gate_features(
    visual: np.ndarray,
    dpvo: np.ndarray,
    loop_gate: dict,
    dpvo_diagnostics: dict,
) -> dict:
    visual_ratio = endpoint_to_path_ratio(visual)
    dpvo_ratio = endpoint_to_path_ratio(dpvo)
    ratio_disagreement = abs(np.log(max(dpvo_ratio, 1e-9) / max(visual_ratio, 1e-9)))
    visual_turns = turn_angles(visual)
    dpvo_turns = turn_angles(dpvo)
    visual_steps = np.linalg.norm(np.diff(visual[:, :3, 3], axis=0), axis=1)
    dpvo_steps = np.linalg.norm(np.diff(dpvo[:, :3, 3], axis=0), axis=1)
    input_frames = int(dpvo_diagnostics.get("input_frames", len(dpvo)))
    target_frames = int(dpvo_diagnostics.get("target_frames", len(dpvo)))
    return {
        "frame_count": len(visual),
        "loop_probability": float(loop_gate.get("probability", 0.0)),
        "loop_edge_count": int(loop_gate.get("edge_count", 0)),
        "visual_endpoint_to_path_ratio": visual_ratio,
        "dpvo_endpoint_to_path_ratio": dpvo_ratio,
        "endpoint_path_log_ratio_disagreement": ratio_disagreement,
        "turn_rank_correlation": rank_correlation(visual_turns, dpvo_turns),
        "step_rank_correlation": rank_correlation(visual_steps, dpvo_steps),
        "keyframe_fraction": (
            float(dpvo_diagnostics["keyframes"]) / max(input_frames, 1)
        ),
        "tracking_complete": bool(
            dpvo_diagnostics.get("initialized", False)
            and dpvo_diagnostics.get("output_poses") == input_frames
            and target_frames == len(visual)
        ),
    }


def frozen_dpvo_gate(features: dict, policy: dict) -> dict:
    probability = soft_and_probability([
        10.0 if features["tracking_complete"] else -10.0,
        (policy["maximum_loop_probability"] - features["loop_probability"])
        / policy["loop_probability_width"],
        (
            policy["maximum_endpoint_path_log_ratio_disagreement"]
            - features["endpoint_path_log_ratio_disagreement"]
        ) / policy["endpoint_path_log_ratio_width"],
        (features["turn_rank_correlation"] - policy["minimum_turn_rank_correlation"])
        / policy["turn_rank_correlation_width"],
        (features["keyframe_fraction"] - policy["minimum_keyframe_fraction"])
        / policy["keyframe_fraction_width"],
    ])
    accepted = probability >= policy["route_threshold"]
    return {
        "probability": probability,
        "route_enabled": accepted,
        "selected_expert": "dpvo" if accepted else "visual",
        "fallback_reason": None if accepted else "dpvo_consensus_rejected",
    }


def generate(args: argparse.Namespace) -> None:
    policy = json.loads(args.policy.read_text())
    with np.load(args.visual, allow_pickle=False) as archive:
        visual = poses_4x4(archive[args.visual_pose_key])
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
    with np.load(args.dpvo, allow_pickle=False) as archive:
        dpvo = poses_4x4(archive[args.dpvo_pose_key])
    if len(visual) != len(dpvo):
        raise ValueError("visual and DPVO frame counts differ")
    loop_gate = json.loads(args.loop_gate.read_text())
    dpvo_diagnostics = json.loads(args.dpvo_diagnostics.read_text())
    features = dpvo_gate_features(visual, dpvo, loop_gate, dpvo_diagnostics)
    gate = frozen_dpvo_gate(features, policy)
    selected = dpvo if gate["route_enabled"] else visual
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidate_before_gt.npz"
    np.savez_compressed(
        candidate_path,
        c2w=selected,
        frame_indices=frame_indices,
        selected_expert=np.asarray(gate["selected_expert"]),
    )
    diagnostics_path = args.output_dir / "dpvo_gate_before_gt.json"
    diagnostics_path.write_text(
        json.dumps({"features": features, "gate": gate}, indent=2) + "\n"
    )
    manifest = {
        "protocol": "frozen DPVO route and candidate hashed before GT load",
        "policy": policy,
        "policy_path": str(args.policy.resolve()),
        "policy_sha256": sha256(args.policy),
        "visual": str(args.visual),
        "visual_sha256": sha256(args.visual),
        "dpvo": str(args.dpvo),
        "dpvo_sha256": sha256(args.dpvo),
        "loop_gate": str(args.loop_gate),
        "loop_gate_sha256": sha256(args.loop_gate),
        "dpvo_diagnostics": str(args.dpvo_diagnostics),
        "dpvo_diagnostics_sha256": sha256(args.dpvo_diagnostics),
        "candidate": str(candidate_path),
        "candidate_sha256": sha256(candidate_path),
        "diagnostics": str(diagnostics_path),
        "diagnostics_sha256": sha256(diagnostics_path),
    }
    (args.output_dir / "manifest_before_gt.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"features": features, "gate": gate}, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    from vistream.metrics import ate_rmse

    manifest = json.loads((args.output_dir / "manifest_before_gt.json").read_text())
    if manifest["policy"] != json.loads(args.policy.read_text()):
        raise ValueError("evaluation policy differs from generation policy")
    for path_key, hash_key in (
        ("visual", "visual_sha256"),
        ("dpvo", "dpvo_sha256"),
        ("loop_gate", "loop_gate_sha256"),
        ("dpvo_diagnostics", "dpvo_diagnostics_sha256"),
        ("candidate", "candidate_sha256"),
        ("diagnostics", "diagnostics_sha256"),
    ):
        path = Path(manifest[path_key])
        if sha256(path) != manifest[hash_key]:
            raise ValueError(f"artifact changed after generation: {path}")
    with np.load(manifest["candidate"], allow_pickle=False) as archive:
        candidate = poses_4x4(archive["c2w"])
    # GT is first opened after every route input and candidate hash verifies.
    with np.load(manifest["visual"], allow_pickle=False) as archive:
        visual = poses_4x4(archive[args.visual_pose_key])
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
    generate_parser.add_argument("--visual", type=Path, required=True)
    generate_parser.add_argument("--dpvo", type=Path, required=True)
    generate_parser.add_argument("--loop-gate", type=Path, required=True)
    generate_parser.add_argument("--dpvo-diagnostics", type=Path, required=True)
    generate_parser.add_argument("--policy", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--visual-pose-key", default="predicted_c2w")
    generate_parser.add_argument("--dpvo-pose-key", default="dpvo_c2w")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--policy", type=Path, required=True)
    evaluate_parser.add_argument("--visual-pose-key", default="predicted_c2w")
    evaluate_parser.add_argument("--ground-truth-key", default="ground_truth_c2w")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generate(arguments) if arguments.command == "generate" else evaluate(arguments)
