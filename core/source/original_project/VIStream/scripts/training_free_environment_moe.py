#!/usr/bin/env python3
"""Frozen evidence-gating network for training-free trajectory experts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


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


def softmax(values: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    names = list(values)
    logits = np.asarray([values[name] for name in names], dtype=np.float64)
    logits = (logits - np.max(logits)) / temperature
    weights = np.exp(logits)
    weights /= np.sum(weights)
    return {name: float(weight) for name, weight in zip(names, weights)}


def base_expert_head(base_router: dict) -> tuple[str | None, float]:
    """Map the old router's physical evidence to one competing gate head."""
    expert = str(base_router.get("expert", "visual"))
    if expert.startswith("environment_gated_loop"):
        return "loop_geometry", float(base_router.get("loop_gate_probability", 0.0))
    if expert == "nonholonomic_imu" and base_router.get("route_enabled", False):
        evidence = [
            float(base_router.get("imu_route_probability", 0.0)),
            float(base_router.get("forward_sign_consistency", 0.0)),
        ]
        if base_router.get("weak_revisit_short_sequence_override", False):
            evidence.append(1.0)
        elif "weak_revisit_gate_probability" in base_router:
            evidence.append(float(base_router["weak_revisit_gate_probability"]))
        return "imu_nonholonomic", min(evidence)
    return None, 0.0


def frozen_evidence_gate(
    base_router: dict,
    dpvo_gate: dict,
    policy: dict,
) -> dict:
    """Run a parameter-free mixture-of-experts head with visual abstention."""
    scores = {"visual": float(policy["visual_abstention_score"])}
    base_head, base_score = base_expert_head(base_router)
    if base_head is not None:
        scores[base_head] = base_score
    scores["dpvo_open_road"] = float(dpvo_gate.get("probability", 0.0))
    weights = softmax(scores, float(policy["softmax_temperature"]))

    selected_head = max(scores, key=scores.get)
    selected_score = scores[selected_head]
    if selected_score < float(policy["route_threshold"]):
        selected_head = "visual"
    if selected_head == "dpvo_open_road":
        selected_expert = "dpvo"
        environment = "open_road_dpvo_consensus"
    elif selected_head == "loop_geometry":
        selected_expert = str(base_router["expert"])
        environment = "strong_loop_geometry"
    elif selected_head == "imu_nonholonomic":
        selected_expert = "nonholonomic_imu"
        environment = "weak_revisit_imu"
    else:
        selected_expert = "visual"
        environment = "visual_abstention"
    ordered_scores = sorted(scores.values(), reverse=True)
    margin = selected_score - ordered_scores[1] if len(ordered_scores) > 1 else 1.0
    return {
        "network": "frozen_evidence_moe_v1",
        "trainable_parameters": 0,
        "head_scores": scores,
        "head_weights": weights,
        "winning_margin": float(margin),
        "selected_head": selected_head,
        "selected_expert": selected_expert,
        "environment": environment,
        "route_enabled": selected_head != "visual",
        "fallback_reason": (
            None if selected_head != "visual" else "no_expert_exceeded_abstention"
        ),
    }


def verified_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


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
    base_manifest = verified_manifest(base_manifest_path)
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
        dpvo_manifest_path = args.dpvo_root / sequence / "manifest_before_gt.json"
        dpvo_manifest = verified_manifest(dpvo_manifest_path)
        verify_entry(
            dpvo_manifest,
            (("candidate", "candidate_sha256"), ("diagnostics", "diagnostics_sha256")),
        )
        dpvo_gate_manifest_path = (
            args.dpvo_gate_root / sequence / "manifest_before_gt.json"
        )
        dpvo_gate_manifest = verified_manifest(dpvo_gate_manifest_path)
        verify_entry(
            dpvo_gate_manifest,
            (("candidate", "candidate_sha256"), ("diagnostics", "diagnostics_sha256")),
        )

        visual_path = Path(base_entry["visual"])
        base_candidate_path = Path(base_entry["candidate"])
        base_router_path = Path(base_entry["router"])
        dpvo_candidate_path = Path(dpvo_manifest["candidate"])
        dpvo_gate_path = Path(dpvo_gate_manifest["diagnostics"])
        with np.load(visual_path, allow_pickle=False) as archive:
            visual = poses_4x4(archive[args.visual_pose_key])
            frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        with np.load(base_candidate_path, allow_pickle=False) as archive:
            base_candidate = poses_4x4(archive["c2w"])
            base_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        with np.load(dpvo_candidate_path, allow_pickle=False) as archive:
            dpvo = poses_4x4(archive[args.dpvo_pose_key])
            dpvo_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        if not (
            len(visual) == len(base_candidate) == len(dpvo)
            and np.array_equal(frame_indices, base_indices)
            and np.array_equal(frame_indices, dpvo_indices)
        ):
            raise ValueError(f"expert alignment differs for sequence {sequence}")
        base_router = json.loads(base_router_path.read_text())
        dpvo_gate_diagnostics = json.loads(dpvo_gate_path.read_text())
        gate = frozen_evidence_gate(
            base_router, dpvo_gate_diagnostics["gate"], policy
        )
        if gate["selected_head"] == "dpvo_open_road":
            selected = dpvo
        elif gate["selected_head"] in {"loop_geometry", "imu_nonholonomic"}:
            selected = base_candidate
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
        diagnostics_path = output_dir / "environment_moe_before_gt.json"
        diagnostics = {
            "sequence": sequence,
            "base_expert": base_router.get("expert", "visual"),
            "base_router": base_router,
            "dpvo_gate": dpvo_gate_diagnostics["gate"],
            "dpvo_features": dpvo_gate_diagnostics["features"],
            "gate": gate,
        }
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
        output_entries.append({
            "sequence": sequence,
            "visual": str(visual_path),
            "visual_sha256": sha256(visual_path),
            "base_candidate": str(base_candidate_path),
            "base_candidate_sha256": sha256(base_candidate_path),
            "base_router": str(base_router_path),
            "base_router_sha256": sha256(base_router_path),
            "dpvo_candidate": str(dpvo_candidate_path),
            "dpvo_candidate_sha256": sha256(dpvo_candidate_path),
            "dpvo_gate": str(dpvo_gate_path),
            "dpvo_gate_sha256": sha256(dpvo_gate_path),
            "dpvo_manifest": str(dpvo_manifest_path),
            "dpvo_manifest_sha256": sha256(dpvo_manifest_path),
            "dpvo_gate_manifest": str(dpvo_gate_manifest_path),
            "dpvo_gate_manifest_sha256": sha256(dpvo_gate_manifest_path),
            "candidate": str(candidate_path),
            "candidate_sha256": sha256(candidate_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": sha256(diagnostics_path),
        })
        print(json.dumps({"sequence": sequence, **gate}), flush=True)

    manifest = {
        "protocol": "all frozen evidence-MoE routes hashed before GT load",
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
        ("dpvo_candidate", "dpvo_candidate_sha256"),
        ("dpvo_gate", "dpvo_gate_sha256"),
        ("dpvo_manifest", "dpvo_manifest_sha256"),
        ("dpvo_gate_manifest", "dpvo_gate_manifest_sha256"),
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
        rows.append({
            "sequence": entry["sequence"],
            "environment": diagnostics["gate"]["environment"],
            "expert": diagnostics["gate"]["selected_expert"],
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
    generate_parser.add_argument("--base-router-root", type=Path, required=True)
    generate_parser.add_argument("--dpvo-root", type=Path, required=True)
    generate_parser.add_argument("--dpvo-gate-root", type=Path, required=True)
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
