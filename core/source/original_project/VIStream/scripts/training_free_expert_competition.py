#!/usr/bin/env python3
"""GT-free competition among visual, loop, and translation experts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training_free_expert_router import (  # noqa: E402
    compose_loop_translation_expert,
    poses_4x4,
)


SEQUENCES = tuple(f"{index:02d}" for index in range(11))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_dst_from_src(src_c2w: np.ndarray, dst_c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(dst_c2w) @ src_c2w


def loop_energy(
    poses: np.ndarray,
    edges: list[dict],
    rotation_weight: float,
    direction_weight: float,
) -> dict[str, float]:
    rotation_residuals = []
    direction_residuals = []
    combined = []
    for edge in edges:
        measurement = np.asarray(edge["transform_dst_from_src"], dtype=np.float64)
        prediction = relative_dst_from_src(
            poses[edge["src_pos"]], poses[edge["dst_pos"]]
        )
        rotation_residual = float(
            np.linalg.norm(
                Rotation.from_matrix(
                    measurement[:3, :3].T @ prediction[:3, :3]
                ).as_rotvec()
            )
        )
        measured_translation = measurement[:3, 3]
        predicted_translation = prediction[:3, 3]
        measured_norm = float(np.linalg.norm(measured_translation))
        predicted_norm = float(np.linalg.norm(predicted_translation))
        if min(measured_norm, predicted_norm) <= 1e-9:
            direction_residual = float(np.pi)
        else:
            direction_residual = float(
                np.arccos(
                    np.clip(
                        np.dot(measured_translation, predicted_translation)
                        / (measured_norm * predicted_norm),
                        -1.0,
                        1.0,
                    )
                )
            )
        rotation_residuals.append(rotation_residual)
        direction_residuals.append(direction_residual)
        combined.append(
            rotation_weight * rotation_residual
            + direction_weight * direction_residual
        )
    if not combined:
        return {
            "energy": 0.0,
            "median_rotation_residual_rad": 0.0,
            "median_direction_residual_rad": 0.0,
            "has_evidence": False,
        }
    return {
        "energy": float(np.median(combined)),
        "median_rotation_residual_rad": float(np.median(rotation_residuals)),
        "median_direction_residual_rad": float(np.median(direction_residuals)),
        "has_evidence": True,
    }


def competition_probability(
    baseline_energy: float,
    candidate_energy: float,
    minimum_relative_improvement: float,
    improvement_width: float,
) -> tuple[float, float]:
    if improvement_width <= 0.0:
        raise ValueError("improvement_width must be positive")
    if not np.isfinite(baseline_energy) or not np.isfinite(candidate_energy):
        return 0.0, 0.0
    relative_improvement = (
        baseline_energy - candidate_energy
    ) / max(baseline_energy, 1e-9)
    logit = (relative_improvement - minimum_relative_improvement) / improvement_width
    probability = float(1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0))))
    return probability, float(relative_improvement)


def generate(args: argparse.Namespace) -> None:
    policy = json.loads(args.policy.read_text())
    entries = []
    for sequence in SEQUENCES:
        cache_path = args.cache_root / sequence / args.cache_name
        loop_path = args.loop_root / sequence / args.loop_name
        edges_path = args.loop_root / sequence / args.edges_name
        direction_path = args.direction_root / sequence / args.direction_name
        direction_diagnostics_path = (
            args.direction_root / sequence / args.direction_diagnostics_name
        )
        with np.load(cache_path, allow_pickle=False) as archive:
            baseline = poses_4x4(archive[args.pose_key])
            frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        with np.load(loop_path, allow_pickle=False) as archive:
            loop = poses_4x4(archive[args.loop_pose_key])
        with np.load(direction_path, allow_pickle=False) as archive:
            direction = poses_4x4(archive[args.direction_pose_key])
        edges = json.loads(edges_path.read_text())
        combined, composition = compose_loop_translation_expert(
            baseline,
            loop,
            direction,
            policy["maximum_composition_path_change_ratio"],
        )
        candidates = {
            "baseline": baseline,
            "loop": loop,
            "translation_direction": direction,
            "loop_translation": combined,
        }
        energies = {
            name: loop_energy(
                value,
                edges,
                policy["rotation_residual_weight"],
                policy["direction_residual_weight"],
            )
            for name, value in candidates.items()
        }
        best_name = min(energies, key=lambda name: energies[name]["energy"])
        probability, relative_improvement = competition_probability(
            energies["baseline"]["energy"],
            energies[best_name]["energy"],
            policy["minimum_relative_energy_improvement"],
            policy["relative_improvement_width"],
        )
        route_enabled = bool(
            len(edges) >= policy["minimum_edges"]
            and best_name != "baseline"
            and probability >= policy["route_threshold"]
        )
        selected_name = best_name if route_enabled else "baseline"
        selected = candidates[selected_name]
        diagnostics = {
            "expert": selected_name,
            "best_expert_before_gate": best_name,
            "route_enabled": route_enabled,
            "route_probability": probability,
            "relative_energy_improvement": relative_improvement,
            "edge_count": len(edges),
            "energies": energies,
            "composition": composition,
            "fallback_reason": (
                None if route_enabled else "expert_competition_rejected"
            ),
        }
        output_dir = args.output_dir / sequence
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = output_dir / "candidate_before_gt.npz"
        np.savez_compressed(
            candidate_path,
            c2w=selected,
            frame_indices=frame_indices,
            selected_expert=np.asarray(selected_name),
        )
        diagnostics_path = output_dir / "competition_gate_before_gt.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
        entries.append({
            "sequence": sequence,
            "cache": str(cache_path),
            "cache_sha256": sha256(cache_path),
            "loop": str(loop_path),
            "loop_sha256": sha256(loop_path),
            "edges": str(edges_path),
            "edges_sha256": sha256(edges_path),
            "direction": str(direction_path),
            "direction_sha256": sha256(direction_path),
            "direction_diagnostics": str(direction_diagnostics_path),
            "direction_diagnostics_sha256": sha256(direction_diagnostics_path),
            "candidate": str(candidate_path),
            "candidate_sha256": sha256(candidate_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": sha256(diagnostics_path),
        })
        print(json.dumps({"sequence": sequence, **diagnostics}), flush=True)
    manifest = {
        "protocol": "all competition-gated candidates generated and hashed before GT load",
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
            ("loop", "loop_sha256"),
            ("edges", "edges_sha256"),
            ("direction", "direction_sha256"),
            ("direction_diagnostics", "direction_diagnostics_sha256"),
            ("candidate", "candidate_sha256"),
            ("diagnostics", "diagnostics_sha256"),
        ):
            path = Path(entry[path_key])
            if sha256(path) != entry[hash_key]:
                raise ValueError(f"artifact changed after generation: {path}")
        with np.load(entry["candidate"], allow_pickle=False) as archive:
            candidate = poses_4x4(archive["c2w"])
        # GT is opened only after the complete manifest has already been written.
        with np.load(entry["cache"], allow_pickle=False) as archive:
            baseline = poses_4x4(archive[args.pose_key])
            target = poses_4x4(archive[args.ground_truth_key])
        diagnostics = json.loads(Path(entry["diagnostics"]).read_text())
        rows.append({
            "sequence": entry["sequence"],
            "expert": diagnostics["expert"],
            "route_probability": diagnostics["route_probability"],
            "baseline_ate_rmse_m": ate_rmse(
                baseline[:, :3, 3], target[:, :3, 3], "sim3"
            )["ate_rmse"],
            "candidate_ate_rmse_m": ate_rmse(
                candidate[:, :3, 3], target[:, :3, 3], "sim3"
            )["ate_rmse"],
        })
    baseline = np.asarray([row["baseline_ate_rmse_m"] for row in rows])
    candidate = np.asarray([row["candidate_ate_rmse_m"] for row in rows])
    summary = {
        "protocol": manifest["protocol"],
        "policy": manifest["policy"],
        "rows": rows,
        "macro": {
            "baseline_ate_rmse_mean_m": float(np.mean(baseline)),
            "candidate_ate_rmse_mean_m": float(np.mean(candidate)),
            "improved_sequences": int(np.sum(candidate < baseline)),
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
    generate_parser.add_argument("--loop-root", type=Path, required=True)
    generate_parser.add_argument("--direction-root", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--policy", type=Path, required=True)
    generate_parser.add_argument("--cache-name", default="trajectory.npz")
    generate_parser.add_argument("--loop-name", default="candidate_before_gt.npz")
    generate_parser.add_argument("--edges-name", default="loop_edges_before_gt.json")
    generate_parser.add_argument("--direction-name", default="candidate_before_gt.npz")
    generate_parser.add_argument(
        "--direction-diagnostics-name", default="direction_gate_before_gt.json"
    )
    generate_parser.add_argument("--pose-key", default="fused_c2w")
    generate_parser.add_argument("--loop-pose-key", default="c2w")
    generate_parser.add_argument("--direction-pose-key", default="c2w")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--policy", type=Path, required=True)
    evaluate_parser.add_argument("--pose-key", default="fused_c2w")
    evaluate_parser.add_argument("--ground-truth-key", default="ground_truth_c2w")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generate(arguments) if arguments.command == "generate" else evaluate(arguments)
