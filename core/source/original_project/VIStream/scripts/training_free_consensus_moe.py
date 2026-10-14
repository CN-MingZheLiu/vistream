#!/usr/bin/env python3
"""Exploratory frozen consensus-MoE over sealed trajectory experts."""

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


def increasing(value: float, center: float, width: float) -> float:
    if width <= 0.0:
        raise ValueError("membership width must be positive")
    return sigmoid((value - center) / width)


def decreasing(value: float, center: float, width: float) -> float:
    return increasing(center - value, 0.0, width)


def softmax(values: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    names = list(values)
    logits = np.asarray([values[name] for name in names], dtype=np.float64)
    logits = (logits - np.max(logits)) / temperature
    weights = np.exp(logits)
    weights /= np.sum(weights)
    return {name: float(weight) for name, weight in zip(names, weights)}


def path_length(poses: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)))


def endpoint_to_path_ratio(poses: np.ndarray) -> float:
    path = path_length(poses)
    endpoint = float(np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3]))
    return endpoint / max(path, 1e-12)


def local_directions(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    steps = np.diff(poses[:, :3, 3], axis=0)
    local = np.einsum(
        "nij,nj->ni", poses[:-1, :3, :3].transpose(0, 2, 1), steps
    )
    norms = np.linalg.norm(local, axis=1)
    valid = norms > 1e-10
    directions = np.zeros_like(local)
    directions[valid] = local[valid] / norms[valid, None]
    return directions, valid


def direction_consensus(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first_directions, first_valid = local_directions(first)
    second_directions, second_valid = local_directions(second)
    valid = first_valid & second_valid
    if not np.any(valid):
        return {
            "median_cosine": -1.0,
            "p10_cosine": -1.0,
            "fraction_above_0_8": 0.0,
        }
    cosine = np.sum(first_directions[valid] * second_directions[valid], axis=1)
    return {
        "median_cosine": float(np.median(cosine)),
        "p10_cosine": float(np.quantile(cosine, 0.1)),
        "fraction_above_0_8": float(np.mean(cosine > 0.8)),
    }


def incremental_rotation_disagreement(
    first: np.ndarray, second: np.ndarray
) -> dict[str, float]:
    first_delta = first[:-1, :3, :3].transpose(0, 2, 1) @ first[1:, :3, :3]
    second_delta = second[:-1, :3, :3].transpose(0, 2, 1) @ second[1:, :3, :3]
    error = Rotation.from_matrix(first_delta.transpose(0, 2, 1) @ second_delta).magnitude()
    return {
        "median_rad": float(np.median(error)),
        "p90_rad": float(np.quantile(error, 0.9)),
    }


def tracking_complete(diagnostics: dict, target_count: int) -> bool:
    input_frames = int(diagnostics.get("input_frames", target_count))
    target_frames = int(diagnostics.get("target_frames", target_count))
    return bool(
        diagnostics.get("initialized", False)
        and int(diagnostics.get("output_poses", -1)) == input_frames
        and target_frames == target_count
    )


def consensus_features(
    visual: np.ndarray,
    dense_dpvo: np.ndarray,
    droid: np.ndarray,
    dense_diagnostics: dict,
    droid_diagnostics: dict,
    base_router: dict,
) -> dict:
    if not (len(visual) == len(dense_dpvo) == len(droid)):
        raise ValueError("visual, dense DPVO, and DROID trajectories must align")
    direction = direction_consensus(dense_dpvo, droid)
    rotation = incremental_rotation_disagreement(dense_dpvo, droid)
    dense_path = path_length(dense_dpvo)
    droid_path = path_length(droid)
    return {
        "frame_count": len(visual),
        "loop_probability": float(base_router.get("loop_gate_probability", 0.0)),
        "visual_endpoint_to_path_ratio": endpoint_to_path_ratio(visual),
        "native_scale_log_disagreement": abs(
            float(np.log(max(dense_path, 1e-12) / max(droid_path, 1e-12)))
        ),
        "direction_consensus": direction,
        "rotation_consensus": rotation,
        "dense_keyframe_fraction": float(dense_diagnostics["keyframes"])
        / max(int(dense_diagnostics["input_frames"]), 1),
        "droid_keyframe_fraction": float(droid_diagnostics["keyframes"])
        / max(int(droid_diagnostics["input_frames"]), 1),
        "dense_tracking_complete": tracking_complete(dense_diagnostics, len(visual)),
        "droid_tracking_complete": tracking_complete(droid_diagnostics, len(visual)),
    }


def imu_base_score(base_router: dict) -> float:
    if not (
        base_router.get("expert") == "nonholonomic_imu"
        and base_router.get("route_enabled", False)
    ):
        return 0.0
    evidence = [
        float(base_router.get("imu_route_probability", 0.0)),
        float(base_router.get("forward_sign_consistency", 0.0)),
    ]
    if base_router.get("weak_revisit_short_sequence_override", False):
        evidence.append(1.0)
    else:
        evidence.append(float(base_router.get("weak_revisit_gate_probability", 0.0)))
    return min(evidence)


def frozen_consensus_gate(features: dict, base_router: dict, policy: dict) -> dict:
    """Route with fixed physical memberships and no trainable parameters."""
    loop_probability = features["loop_probability"]
    endpoint_ratio = features["visual_endpoint_to_path_ratio"]
    direction_probability = increasing(
        features["direction_consensus"]["fraction_above_0_8"],
        policy["minimum_direction_consensus_fraction"],
        policy["direction_consensus_width"],
    )
    rotation_probability = decreasing(
        features["rotation_consensus"]["p90_rad"],
        policy["maximum_rotation_disagreement_p90_rad"],
        policy["rotation_disagreement_width_rad"],
    )
    complete_probability = (
        1.0
        if features["dense_tracking_complete"] and features["droid_tracking_complete"]
        else 0.0
    )
    consensus_probability = min(
        direction_probability, rotation_probability, complete_probability
    )

    closed_loop_direction = min(
        loop_probability,
        decreasing(
            endpoint_ratio,
            policy["closed_loop_endpoint_ratio"],
            policy["closed_loop_endpoint_width"],
        ),
        consensus_probability,
    )
    loop_geometry = min(
        loop_probability,
        increasing(
            endpoint_ratio,
            policy["loop_geometry_endpoint_ratio"],
            policy["loop_geometry_endpoint_width"],
        ),
    )
    native_dense = min(
        1.0 - loop_probability,
        increasing(
            endpoint_ratio,
            policy["native_minimum_endpoint_ratio"],
            policy["native_endpoint_width"],
        ),
        decreasing(
            features["native_scale_log_disagreement"],
            policy["native_maximum_scale_log_disagreement"],
            policy["native_scale_log_width"],
        ),
        consensus_probability,
    )
    dense_direction_touring = min(
        1.0 - loop_probability,
        increasing(
            features["frame_count"],
            policy["touring_minimum_frames"],
            policy["touring_frame_width"],
        ),
        increasing(
            endpoint_ratio,
            policy["touring_minimum_endpoint_ratio"],
            policy["touring_minimum_endpoint_width"],
        ),
        decreasing(
            endpoint_ratio,
            policy["touring_maximum_endpoint_ratio"],
            policy["touring_maximum_endpoint_width"],
        ),
        consensus_probability,
    )
    physics_exclusion = max(
        closed_loop_direction, loop_geometry, native_dense, dense_direction_touring
    )
    base_physics = min(imu_base_score(base_router), 1.0 - physics_exclusion)
    scores = {
        "visual": float(policy["visual_abstention_score"]),
        "closed_loop_dense_direction": float(closed_loop_direction),
        "loop_geometry": float(loop_geometry),
        "dense_native_open_road": float(native_dense),
        "dense_direction_touring": float(dense_direction_touring),
        "base_physics": float(base_physics),
    }
    selected_head = max(scores, key=scores.get)
    if scores[selected_head] < float(policy["route_threshold"]):
        selected_head = "visual"
    selected_expert = {
        "visual": "visual",
        "closed_loop_dense_direction": "dpvo_direction_dpvo_rotation",
        "loop_geometry": str(base_router.get("expert", "visual")),
        "dense_native_open_road": "dpvo",
        "dense_direction_touring": "dpvo_direction_dpvo_rotation",
        "base_physics": str(base_router.get("expert", "visual")),
    }[selected_head]
    environments = {
        "visual": "visual_abstention",
        "closed_loop_dense_direction": "strong_closed_loop_direction_consensus",
        "loop_geometry": "loop_geometry",
        "dense_native_open_road": "straight_open_road_native_vo",
        "dense_direction_touring": "long_touring_direction_consensus",
        "base_physics": "weak_revisit_physics",
    }
    ordered = sorted(scores.values(), reverse=True)
    return {
        "network": "frozen_environment_consensus_moe_v2",
        "trainable_parameters": 0,
        "head_scores": scores,
        "head_weights": softmax(scores, float(policy["softmax_temperature"])),
        "winning_margin": float(ordered[0] - ordered[1]),
        "selected_head": selected_head,
        "selected_expert": selected_expert,
        "environment": environments[selected_head],
        "route_enabled": selected_head != "visual",
        "fallback_reason": (
            None if selected_head != "visual" else "no_consensus_head_exceeded_abstention"
        ),
    }


def verify_entry(entry: dict, pairs: tuple[tuple[str, str], ...]) -> None:
    for path_key, hash_key in pairs:
        if entry.get(path_key) is None:
            continue
        path = Path(entry[path_key])
        if sha256(path) != entry[hash_key]:
            raise ValueError(f"source changed after sealing: {path}")


def generate(args: argparse.Namespace) -> None:
    policy = json.loads(args.policy.read_text())
    base_manifest_path = args.base_router_root / "manifest_before_gt.json"
    base_manifest = json.loads(base_manifest_path.read_text())
    base_entries = {entry["sequence"]: entry for entry in base_manifest["entries"]}
    if set(base_entries) != set(SEQUENCES):
        raise ValueError("base router manifest must contain KITTI 00-10 exactly once")

    output_entries = []
    for sequence in SEQUENCES:
        base_entry = base_entries[sequence]
        verify_entry(
            base_entry,
            (
                ("visual", "visual_sha256"),
                ("candidate", "candidate_sha256"),
                ("router", "router_sha256"),
            ),
        )
        dense_manifest_path = args.dense_dpvo_root / sequence / "manifest_before_gt.json"
        droid_manifest_path = args.droid_root / sequence / "manifest_before_gt.json"
        dense_manifest = json.loads(dense_manifest_path.read_text())
        droid_manifest = json.loads(droid_manifest_path.read_text())
        verify_entry(
            dense_manifest,
            (("candidate", "candidate_sha256"), ("diagnostics", "diagnostics_sha256")),
        )
        verify_entry(
            droid_manifest,
            (("candidate", "candidate_sha256"), ("diagnostics", "diagnostics_sha256")),
        )

        visual_path = Path(base_entry["visual"])
        base_candidate_path = Path(base_entry["candidate"])
        base_router_path = Path(base_entry["router"])
        dense_candidate_path = Path(dense_manifest["candidate"])
        dense_diagnostics_path = Path(dense_manifest["diagnostics"])
        droid_candidate_path = Path(droid_manifest["candidate"])
        droid_diagnostics_path = Path(droid_manifest["diagnostics"])
        with np.load(visual_path, allow_pickle=False) as archive:
            visual = poses_4x4(archive[args.visual_pose_key])
            frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        with np.load(base_candidate_path, allow_pickle=False) as archive:
            base_candidate = poses_4x4(archive["c2w"])
            base_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        with np.load(dense_candidate_path, allow_pickle=False) as archive:
            dense_native = poses_4x4(archive["dpvo_c2w"])
            dense_direction = poses_4x4(
                archive["dpvo_direction_dpvo_rotation_c2w"]
            )
            dense_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        with np.load(droid_candidate_path, allow_pickle=False) as archive:
            droid = poses_4x4(archive["droid_c2w"])
            droid_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        if not (
            len(visual) == len(base_candidate) == len(dense_native) == len(droid)
            and np.array_equal(frame_indices, base_indices)
            and np.array_equal(frame_indices, dense_indices)
            and np.array_equal(frame_indices, droid_indices)
        ):
            raise ValueError(f"expert alignment differs for sequence {sequence}")
        base_router = json.loads(base_router_path.read_text())
        dense_diagnostics = json.loads(dense_diagnostics_path.read_text())
        droid_diagnostics = json.loads(droid_diagnostics_path.read_text())
        features = consensus_features(
            visual,
            dense_native,
            droid,
            dense_diagnostics,
            droid_diagnostics,
            base_router,
        )
        gate = frozen_consensus_gate(features, base_router, policy)
        if gate["selected_head"] in {"loop_geometry", "base_physics"}:
            selected = base_candidate
        elif gate["selected_head"] == "dense_native_open_road":
            selected = dense_native
        elif gate["selected_head"] in {
            "closed_loop_dense_direction",
            "dense_direction_touring",
        }:
            selected = dense_direction
        else:
            selected = visual

        output_dir = args.output_dir / sequence
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = output_dir / "candidate_before_gt.npz"
        np.savez_compressed(
            candidate_path,
            c2w=selected,
            frame_indices=frame_indices,
            selected_expert=np.asarray(gate["selected_expert"]),
        )
        diagnostics_path = output_dir / "consensus_moe_before_gt.json"
        diagnostics_path.write_text(
            json.dumps(
                {
                    "sequence": sequence,
                    "features": features,
                    "gate": gate,
                    "base_router": base_router,
                },
                indent=2,
            )
            + "\n"
        )
        output_entries.append(
            {
                "sequence": sequence,
                "visual": str(visual_path),
                "visual_sha256": sha256(visual_path),
                "base_candidate": str(base_candidate_path),
                "base_candidate_sha256": sha256(base_candidate_path),
                "base_router": str(base_router_path),
                "base_router_sha256": sha256(base_router_path),
                "dense_manifest": str(dense_manifest_path),
                "dense_manifest_sha256": sha256(dense_manifest_path),
                "dense_candidate": str(dense_candidate_path),
                "dense_candidate_sha256": sha256(dense_candidate_path),
                "dense_diagnostics": str(dense_diagnostics_path),
                "dense_diagnostics_sha256": sha256(dense_diagnostics_path),
                "droid_manifest": str(droid_manifest_path),
                "droid_manifest_sha256": sha256(droid_manifest_path),
                "droid_candidate": str(droid_candidate_path),
                "droid_candidate_sha256": sha256(droid_candidate_path),
                "droid_diagnostics": str(droid_diagnostics_path),
                "droid_diagnostics_sha256": sha256(droid_diagnostics_path),
                "candidate": str(candidate_path),
                "candidate_sha256": sha256(candidate_path),
                "diagnostics": str(diagnostics_path),
                "diagnostics_sha256": sha256(diagnostics_path),
            }
        )
        print(json.dumps({"sequence": sequence, **gate}), flush=True)

    manifest = {
        "protocol": (
            "exploratory GT-free consensus routes hashed before this evaluation; "
            "KITTI development results were previously observed"
        ),
        "base_manifest": str(base_manifest_path),
        "base_manifest_sha256": sha256(base_manifest_path),
        "policy": policy,
        "policy_path": str(args.policy.resolve()),
        "policy_sha256": sha256(args.policy),
        "entries": output_entries,
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
    if sha256(Path(manifest["base_manifest"])) != manifest["base_manifest_sha256"]:
        raise ValueError("base manifest changed after generation")
    pairs = (
        ("visual", "visual_sha256"),
        ("base_candidate", "base_candidate_sha256"),
        ("base_router", "base_router_sha256"),
        ("dense_manifest", "dense_manifest_sha256"),
        ("dense_candidate", "dense_candidate_sha256"),
        ("dense_diagnostics", "dense_diagnostics_sha256"),
        ("droid_manifest", "droid_manifest_sha256"),
        ("droid_candidate", "droid_candidate_sha256"),
        ("droid_diagnostics", "droid_diagnostics_sha256"),
        ("candidate", "candidate_sha256"),
        ("diagnostics", "diagnostics_sha256"),
    )
    for entry in manifest["entries"]:
        verify_entry(entry, pairs)

    rows = []
    for entry in manifest["entries"]:
        with np.load(entry["candidate"], allow_pickle=False) as archive:
            candidate = poses_4x4(archive["c2w"])
        with np.load(entry["visual"], allow_pickle=False) as archive:
            visual = poses_4x4(archive[args.visual_pose_key])
            target = poses_4x4(archive[args.ground_truth_key])
        diagnostics = json.loads(Path(entry["diagnostics"]).read_text())
        rows.append(
            {
                "sequence": entry["sequence"],
                "environment": diagnostics["gate"]["environment"],
                "expert": diagnostics["gate"]["selected_expert"],
                "visual_ate_rmse_m": ate_rmse(
                    visual[:, :3, 3], target[:, :3, 3], "sim3"
                )["ate_rmse"],
                "candidate_ate_rmse_m": ate_rmse(
                    candidate[:, :3, 3], target[:, :3, 3], "sim3"
                )["ate_rmse"],
            }
        )
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
    generate_parser.add_argument("--base-router-root", type=Path, required=True)
    generate_parser.add_argument("--dense-dpvo-root", type=Path, required=True)
    generate_parser.add_argument("--droid-root", type=Path, required=True)
    generate_parser.add_argument("--policy", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--visual-pose-key", default="predicted_c2w")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--policy", type=Path, required=True)
    evaluate_parser.add_argument("--visual-pose-key", default="predicted_c2w")
    evaluate_parser.add_argument("--ground-truth-key", default="ground_truth_c2w")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generate(arguments) if arguments.command == "generate" else evaluate(arguments)
