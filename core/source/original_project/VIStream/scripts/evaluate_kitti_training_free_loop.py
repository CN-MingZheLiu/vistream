#!/usr/bin/env python3
"""Two-phase, fixed-policy evaluation of training-free KITTI loop closure.

``generate`` is ground-truth blind: it only reads cached visual poses and
image/calibration files, and writes every candidate before the evaluation
phase is allowed to read a target pose.  ``evaluate`` consumes that manifest
and reports per-sequence and macro Sim(3) ATE metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


SEQUENCES = tuple(f"{index:02d}" for index in range(11))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text())
    required = {
        "minimum_separation",
        "nms_radius",
        "maximum_pairs",
        "loop_weight",
        "pose_graph",
        "maximum_evaluations",
        "descriptor",
        "descriptor_input",
        "verification",
        "sift_nfeatures",
        "sift_contrast_threshold",
        "ratio_test",
        "minimum_matches",
        "minimum_inliers",
        "minimum_inlier_ratio",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError(f"policy is missing keys: {', '.join(missing)}")
    if policy["pose_graph"] not in {"rotation", "se3"}:
        raise ValueError("policy pose_graph must be rotation or se3")
    if policy["descriptor"] != "resnet50_imagenet1k_v2":
        raise ValueError("the fixed runner supports only resnet50_imagenet1k_v2")
    if policy["verification"] != "sift_essential_ransac":
        raise ValueError("the fixed runner supports only sift_essential_ransac")
    if policy["descriptor_input"] != [320, 192]:
        raise ValueError("descriptor_input must be [320, 192]")
    gate = policy.get("environment_gate", "none")
    if gate not in {"none", "frozen_random_walk_mlp"}:
        raise ValueError("unsupported environment_gate")
    if gate != "none":
        gate_keys = {
            "environment_gate_center",
            "environment_gate_slope",
            "environment_gate_route_threshold",
            "environment_gate_minimum_edges",
        }
        missing_gate = sorted(gate_keys - set(policy))
        if missing_gate:
            raise ValueError(
                f"environment gate is missing keys: {', '.join(missing_gate)}"
            )
    return policy


def _cache_path(cache_root: Path, sequence: str, name: str = "trajectory.npz") -> Path:
    return cache_root / sequence / name


def _build_generate_command(
    loop_script: Path,
    cache: Path,
    image_dir: Path,
    calibration: Path,
    output: Path,
    policy: dict[str, Any],
    device: str,
    batch_size: int,
    pose_key: str = "fused_c2w",
) -> list[str]:
    command = [
        sys.executable,
        str(loop_script),
        "generate",
        "--cache",
        str(cache),
        "--pose-key",
        pose_key,
        "--image-dir",
        str(image_dir),
        "--calibration",
        str(calibration),
        "--output-dir",
        str(output),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--descriptor-width",
        str(policy["descriptor_input"][0]),
        "--descriptor-height",
        str(policy["descriptor_input"][1]),
    ]
    for key in (
        "minimum_separation",
        "nms_radius",
        "maximum_pairs",
        "loop_weight",
        "pose_graph",
        "maximum_evaluations",
        "sift_nfeatures",
        "sift_contrast_threshold",
        "ratio_test",
        "minimum_matches",
        "minimum_inliers",
        "minimum_inlier_ratio",
    ):
        command.extend([f"--{key.replace('_', '-')}", str(policy[key])])
    command.extend(["--environment-gate", policy.get("environment_gate", "none")])
    for key in (
        "environment_gate_center",
        "environment_gate_slope",
        "environment_gate_route_threshold",
        "environment_gate_minimum_edges",
    ):
        if key in policy:
            command.extend([f"--{key.replace('_', '-')}", str(policy[key])])
    return command


def _generate(args: argparse.Namespace) -> None:
    policy = _load_policy(args.policy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loop_script = args.loop_script.resolve()
    entries = []
    for sequence in SEQUENCES:
        cache = _cache_path(args.cache_root, sequence, args.cache_name)
        ground_truth_cache = _cache_path(
            args.ground_truth_cache_root or args.cache_root,
            sequence,
            args.ground_truth_cache_name,
        )
        image_dir = args.image_root / sequence / "image_2"
        calibration = args.image_root / sequence / "calib.txt"
        output = args.output_dir / sequence
        for path in (cache, ground_truth_cache, image_dir, calibration):
            if not path.exists():
                raise FileNotFoundError(path)
        command = _build_generate_command(
            loop_script, cache, image_dir, calibration, output, policy,
            args.device, args.batch_size, args.pose_key
        )
        print(f"[generate] KITTI {sequence}: {output}", flush=True)
        subprocess.run(command, check=True)
        candidate = output / "candidate_before_gt.npz"
        edges = output / "loop_edges_before_gt.json"
        gate = output / "environment_gate_before_gt.json"
        if not candidate.exists() or not edges.exists() or not gate.exists():
            raise RuntimeError(f"loop script did not write complete candidate for {sequence}")
        entries.append({
            "sequence": sequence,
            "cache": str(cache),
            "ground_truth_cache": str(ground_truth_cache),
            "candidate": str(candidate),
            "edges": str(edges),
            "environment_gate": str(gate),
            "candidate_sha256": _sha256(candidate),
            "edges_sha256": _sha256(edges),
            "environment_gate_sha256": _sha256(gate),
        })
    manifest = {
        "protocol": "all candidates generated before any ground-truth load",
        "sequences": list(SEQUENCES),
        "policy": policy,
        "policy_path": str(args.policy.resolve()),
        "policy_sha256": _sha256(args.policy),
        "loop_script": str(loop_script),
        "loop_script_sha256": _sha256(loop_script),
        "pose_key": args.pose_key,
        "entries": entries,
    }
    (args.output_dir / "manifest_before_gt.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"manifest": str(args.output_dir / "manifest_before_gt.json")}, indent=2))


def _evaluate(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.project_root.resolve()))
    from vistream.metrics import ate_rmse

    manifest_path = args.output_dir / "manifest_before_gt.json"
    manifest = json.loads(manifest_path.read_text())
    policy = _load_policy(args.policy)
    if manifest["policy"] != policy:
        raise ValueError("evaluation policy differs from the policy used for generation")
    if args.pose_key != manifest["pose_key"]:
        raise ValueError("evaluation pose key differs from the generation pose key")
    rows = []
    for entry in manifest["entries"]:
        cache = Path(entry["cache"])
        ground_truth_cache = Path(entry.get("ground_truth_cache", entry["cache"]))
        candidate_path = Path(entry["candidate"])
        if _sha256(candidate_path) != entry["candidate_sha256"]:
            raise ValueError(f"candidate changed after generation: {candidate_path}")
        for artifact_key, hash_key in (
            ("edges", "edges_sha256"),
            ("environment_gate", "environment_gate_sha256"),
        ):
            if artifact_key not in entry:
                continue
            artifact = Path(entry[artifact_key])
            if _sha256(artifact) != entry[hash_key]:
                raise ValueError(
                    f"{artifact_key} changed after generation: {artifact}"
                )
        with np.load(candidate_path, allow_pickle=False) as candidate:
            predicted = np.asarray(candidate["c2w"], dtype=np.float64)
            candidate_frames = np.asarray(candidate["frame_indices"], dtype=np.int64)
        with np.load(cache, allow_pickle=False) as archive:
            visual = np.asarray(archive[args.pose_key], dtype=np.float64)
            cache_frames = np.asarray(archive["frame_indices"], dtype=np.int64)
        # Ground truth is opened only here, after every candidate is complete.
        with np.load(ground_truth_cache, allow_pickle=False) as archive:
            target = np.asarray(archive[args.ground_truth_key], dtype=np.float64)
            target_frames = np.asarray(archive["frame_indices"], dtype=np.int64)
        if not np.array_equal(candidate_frames, cache_frames):
            raise ValueError(f"frame indices differ for KITTI {entry['sequence']}")
        if not np.array_equal(candidate_frames, target_frames):
            raise ValueError(f"ground-truth frame indices differ for KITTI {entry['sequence']}")
        target = target[:, :3, 3] if target.shape[1:] == (4, 4) else target[:, :, 3]
        visual = visual[:, :3, 3] if visual.shape[1:] == (4, 4) else visual[:, :, 3]
        predicted = predicted[:, :3, 3] if predicted.shape[1:] == (4, 4) else predicted[:, :, 3]
        visual_metric = ate_rmse(visual, target, "sim3")
        candidate_metric = ate_rmse(predicted, target, "sim3")
        with np.load(candidate_path, allow_pickle=False) as candidate:
            loop_count = int(candidate["loop_count"])
            active_loop_count = int(
                candidate["active_loop_count"]
                if "active_loop_count" in candidate.files else loop_count
            )
            gate_probability = float(
                candidate["environment_gate_probability"]
                if "environment_gate_probability" in candidate.files else 1.0
            )
        rows.append({
            "sequence": entry["sequence"],
            "frames": len(target),
            "loops": loop_count,
            "active_loops": active_loop_count,
            "environment_gate_probability": gate_probability,
            "visual_ate_rmse_m": visual_metric["ate_rmse"],
            "candidate_ate_rmse_m": candidate_metric["ate_rmse"],
            "visual_ate_mean_m": visual_metric["ate_mean"],
            "candidate_ate_mean_m": candidate_metric["ate_mean"],
        })
    candidate_rmse = np.asarray([row["candidate_ate_rmse_m"] for row in rows])
    visual_rmse = np.asarray([row["visual_ate_rmse_m"] for row in rows])
    summary = {
        "protocol": manifest["protocol"],
        "policy": policy,
        "rows": rows,
        "macro": {
            "visual_ate_rmse_mean_m": float(visual_rmse.mean()),
            "candidate_ate_rmse_mean_m": float(candidate_rmse.mean()),
            "candidate_sequences_at_or_below_10m": int(np.sum(candidate_rmse <= 10.0)),
            "sequence_count": len(rows),
        },
    }
    path = args.output_dir / "summary_after_gt.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--cache-root", type=Path, required=True)
    generate.add_argument("--cache-name", default="trajectory.npz")
    generate.add_argument("--ground-truth-cache-root", type=Path)
    generate.add_argument("--ground-truth-cache-name", default="trajectory.npz")
    generate.add_argument("--image-root", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--policy", type=Path, required=True)
    generate.add_argument("--loop-script", type=Path, required=True)
    generate.add_argument("--device", default="cuda:7")
    generate.add_argument("--batch-size", type=int, default=64)
    generate.add_argument("--pose-key", default="fused_c2w")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--policy", type=Path, required=True)
    evaluate.add_argument("--project-root", type=Path, required=True)
    evaluate.add_argument("--pose-key", default="fused_c2w")
    evaluate.add_argument("--ground-truth-key", default="ground_truth_c2w")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _generate(args) if args.command == "generate" else _evaluate(args)


if __name__ == "__main__":
    main()
