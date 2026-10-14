#!/usr/bin/env python3
"""Exploratory frozen second-stage gate over v8 and ORB-SLAM3 experts."""

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


def path_statistics(poses: np.ndarray) -> dict[str, float]:
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    path = float(np.sum(steps))
    endpoint = float(np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3]))
    return {
        "path_length": path,
        "endpoint_distance": endpoint,
        "endpoint_to_path_ratio": endpoint / max(path, 1e-12),
    }


def rebase_first_pose(poses: np.ndarray) -> np.ndarray:
    poses = poses_4x4(poses)
    return np.linalg.inv(poses[0])[None] @ poses


def blend_sim3_aligned(
    base: np.ndarray,
    orb: np.ndarray,
    orb_weight: float,
) -> tuple[np.ndarray, dict[str, float]]:
    from vistream.metrics import umeyama_alignment

    base = poses_4x4(base)
    orb = poses_4x4(orb)
    if len(base) != len(orb):
        raise ValueError("base and ORB trajectories must align")
    if not 0.0 <= orb_weight <= 1.0:
        raise ValueError("ORB mixture weight must be within [0,1]")
    if orb_weight == 0.0:
        return base.copy(), {"scale": 1.0, "rotation_angle_rad": 0.0}
    if orb_weight == 1.0:
        return orb.copy(), {"scale": 1.0, "rotation_angle_rad": 0.0}

    alignment = umeyama_alignment(
        base[:, :3, 3], orb[:, :3, 3], with_scale=True
    )
    aligned_base = base.copy()
    aligned_base[:, :3, 3] = alignment.apply(base[:, :3, 3])
    aligned_base[:, :3, :3] = alignment.rotation @ base[:, :3, :3]
    relative = (
        aligned_base[:, :3, :3].transpose(0, 2, 1) @ orb[:, :3, :3]
    )
    relative_vectors = Rotation.from_matrix(relative).as_rotvec()
    blended = np.repeat(np.eye(4)[None], len(base), axis=0)
    blended[:, :3, 3] = (
        (1.0 - orb_weight) * aligned_base[:, :3, 3]
        + orb_weight * orb[:, :3, 3]
    )
    blended[:, :3, :3] = (
        aligned_base[:, :3, :3]
        @ Rotation.from_rotvec(orb_weight * relative_vectors).as_matrix()
    )
    return rebase_first_pose(blended), {
        "scale": float(alignment.scale),
        "rotation_angle_rad": float(Rotation.from_matrix(alignment.rotation).magnitude()),
    }


def raw_orb_features(log_path: Path, frame_path: Path | None) -> dict:
    text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    features = {
        "loop_detection_count": text.count("*Loop detected"),
        "map_count": text.count("Creation of new map with id"),
        "raw_log_available": log_path.is_file(),
        "raw_frame_trajectory_available": bool(frame_path and frame_path.is_file()),
        "raw_timestamp_nonmonotonic_count": None,
    }
    if frame_path is not None and frame_path.is_file():
        rows = np.atleast_2d(np.loadtxt(frame_path, dtype=np.float64, usecols=(0,)))
        timestamps = rows.reshape(-1)
        features["raw_timestamp_nonmonotonic_count"] = int(
            np.sum(np.diff(timestamps) <= 0)
        )
    return features


def frozen_orb_gate(features: dict, policy: dict) -> dict:
    valid = bool(
        features["orb_available"]
        and features["map_count"] == 1
        and features["raw_timestamp_nonmonotonic_count"] == 0
        and features["tracking_completeness"]
        >= float(policy["minimum_tracking_completeness"])
    )
    loop_count = int(features["loop_detection_count"])
    endpoint_ratio = float(features["orb_endpoint_to_path_ratio"])
    if not valid:
        environment = "orb_invalid_or_unavailable_base_fallback"
        orb_weight = 0.0
    elif loop_count > 0 and endpoint_ratio <= float(
        policy["strong_closed_loop_endpoint_ratio"]
    ):
        environment = "orb_confirmed_strong_closed_loop"
        orb_weight = float(policy["strong_closed_loop_orb_weight"])
    elif loop_count > 0 and endpoint_ratio <= float(
        policy["partial_loop_endpoint_ratio"]
    ):
        environment = "orb_confirmed_partial_loop_consensus"
        orb_weight = float(policy["partial_loop_orb_weight"])
    elif loop_count > 0:
        environment = "orb_confirmed_weak_loop_consensus"
        orb_weight = float(policy["weak_loop_orb_weight"])
    elif endpoint_ratio >= float(policy["open_road_endpoint_ratio"]):
        environment = "orb_open_road_consensus"
        orb_weight = float(policy["open_road_orb_weight"])
    else:
        environment = "orb_unconfirmed_touring_advisory"
        orb_weight = float(policy["touring_advisory_orb_weight"])
    return {
        "network": "frozen_orb_environment_mixture_gate_v1",
        "trainable_parameters": 0,
        "valid_orb_head": valid,
        "environment": environment,
        "orb_weight": orb_weight,
        "base_weight": 1.0 - orb_weight,
        "selected_expert": (
            "v8" if orb_weight == 0.0 else "orbslam3" if orb_weight == 1.0 else "v8_orb_sim3_consensus"
        ),
    }


def verify_hash(path: Path, expected: str, label: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"sealed {label} changed: {path}")


def load_base_entry(entry: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_path = Path(entry["candidate"])
    diagnostics_path = Path(entry["diagnostics"])
    visual_path = Path(entry["visual"])
    verify_hash(candidate_path, entry["candidate_sha256"], "v8 candidate")
    verify_hash(diagnostics_path, entry["diagnostics_sha256"], "v8 diagnostics")
    verify_hash(visual_path, entry["visual_sha256"], "visual trajectory")
    with np.load(candidate_path, allow_pickle=False) as archive:
        base = poses_4x4(archive["c2w"])
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
    with np.load(visual_path, allow_pickle=False) as archive:
        visual_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
    if not np.array_equal(frame_indices, visual_indices):
        raise ValueError("v8 candidate and visual frame indices differ")
    return base, frame_indices, visual_indices


def unavailable_orb_record(sequence_dir: Path) -> tuple[dict, dict]:
    raw_dir = sequence_dir / "raw_before_gt"
    log_path = raw_dir / "run.log"
    frame_path = raw_dir / "FrameTrajectory.txt"
    features = raw_orb_features(log_path, frame_path if frame_path.is_file() else None)
    features.update(
        {
            "orb_available": False,
            "tracking_completeness": 0.0,
            "orb_endpoint_to_path_ratio": 0.0,
        }
    )
    evidence = {}
    for name, path in (("raw_log", log_path), ("raw_frame_trajectory", frame_path)):
        if path.is_file():
            evidence[name] = {"path": str(path), "sha256": sha256(path)}
    return features, evidence


def load_orb_entry(
    sequence_dir: Path,
    expected_frame_indices: np.ndarray,
) -> tuple[np.ndarray | None, dict, dict]:
    manifest_path = sequence_dir / "manifest_before_gt.json"
    if not manifest_path.is_file():
        features, evidence = unavailable_orb_record(sequence_dir)
        return None, features, {
            "manifest": None,
            "manifest_sha256": None,
            "failure_evidence": evidence,
        }
    manifest = json.loads(manifest_path.read_text())
    for name in ("candidate", "diagnostics", "raw_frame_trajectory", "run_log"):
        value = manifest["artifacts"][name]
        verify_hash(Path(value["path"]), value["sha256"], f"ORB {name}")
    candidate_path = Path(manifest["artifacts"]["candidate"]["path"])
    diagnostics_path = Path(manifest["artifacts"]["diagnostics"]["path"])
    log_path = Path(manifest["artifacts"]["run_log"]["path"])
    frame_path = Path(manifest["artifacts"]["raw_frame_trajectory"]["path"])
    with np.load(candidate_path, allow_pickle=False) as archive:
        orb = poses_4x4(archive["orbslam3_c2w"])
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
    if not np.array_equal(frame_indices, expected_frame_indices):
        raise ValueError("ORB and v8 frame indices differ")
    diagnostics = json.loads(diagnostics_path.read_text())
    features = raw_orb_features(log_path, frame_path)
    features.update(
        {
            "orb_available": True,
            "tracking_completeness": float(diagnostics["tracking_completeness"]),
            "orb_endpoint_to_path_ratio": float(
                diagnostics["orbslam3"]["endpoint_to_path_ratio"]
            ),
        }
    )
    return orb, features, {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "candidate": str(candidate_path),
        "candidate_sha256": sha256(candidate_path),
        "diagnostics": str(diagnostics_path),
        "diagnostics_sha256": sha256(diagnostics_path),
        "raw_log": str(log_path),
        "raw_log_sha256": sha256(log_path),
        "failure_evidence": {},
    }


def generate(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = json.loads(args.policy.read_text())
    base_manifest_path = args.base_root / "manifest_before_gt.json"
    base_manifest = json.loads(base_manifest_path.read_text())
    base_entries = {entry["sequence"]: entry for entry in base_manifest["entries"]}
    if set(base_entries) != set(SEQUENCES):
        raise ValueError("v8 manifest must contain KITTI 00-10 exactly once")

    entries = []
    for sequence in SEQUENCES:
        base_entry = base_entries[sequence]
        base, frame_indices, _ = load_base_entry(base_entry)
        orb, features, orb_record = load_orb_entry(
            args.orb_root / sequence, frame_indices
        )
        gate = frozen_orb_gate(features, policy)
        if orb is None or gate["orb_weight"] == 0.0:
            selected = base.copy()
            alignment = {"scale": 1.0, "rotation_angle_rad": 0.0}
        else:
            selected, alignment = blend_sim3_aligned(
                base, orb, float(gate["orb_weight"])
            )
        output_dir = args.output_dir / sequence
        output_dir.mkdir(parents=True, exist_ok=False)
        candidate_path = output_dir / "candidate_before_gt.npz"
        np.savez_compressed(
            candidate_path,
            c2w=selected,
            frame_indices=frame_indices,
            selected_expert=np.asarray(gate["selected_expert"]),
            orb_weight=np.asarray(gate["orb_weight"]),
        )
        diagnostics = {
            "sequence": sequence,
            "features": features,
            "gate": gate,
            "sim3_base_to_orb": alignment,
            "base": path_statistics(base),
            "orb": path_statistics(orb) if orb is not None else None,
            "selected": path_statistics(selected),
        }
        diagnostics_path = output_dir / "orb_consensus_before_gt.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
        entries.append(
            {
                "sequence": sequence,
                "visual": base_entry["visual"],
                "visual_sha256": base_entry["visual_sha256"],
                "base_candidate": base_entry["candidate"],
                "base_candidate_sha256": base_entry["candidate_sha256"],
                "base_diagnostics": base_entry["diagnostics"],
                "base_diagnostics_sha256": base_entry["diagnostics_sha256"],
                "orb": orb_record,
                "candidate": str(candidate_path),
                "candidate_sha256": sha256(candidate_path),
                "diagnostics": str(diagnostics_path),
                "diagnostics_sha256": sha256(diagnostics_path),
            }
        )
        print(json.dumps({"sequence": sequence, **gate}), flush=True)

    manifest = {
        "protocol": (
            "exploratory zero-parameter ORB environment mixtures hashed before this "
            "evaluation; KITTI development results were previously observed"
        ),
        "base_manifest": str(base_manifest_path),
        "base_manifest_sha256": sha256(base_manifest_path),
        "policy": policy,
        "policy_path": str(args.policy.resolve()),
        "policy_sha256": sha256(args.policy),
        "entries": entries,
    }
    (args.output_dir / "manifest_before_gt.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def verify_entry(entry: dict) -> None:
    for path_key, hash_key in (
        ("visual", "visual_sha256"),
        ("base_candidate", "base_candidate_sha256"),
        ("base_diagnostics", "base_diagnostics_sha256"),
        ("candidate", "candidate_sha256"),
        ("diagnostics", "diagnostics_sha256"),
    ):
        verify_hash(Path(entry[path_key]), entry[hash_key], path_key)
    orb = entry["orb"]
    for path_key, hash_key in (
        ("manifest", "manifest_sha256"),
        ("candidate", "candidate_sha256"),
        ("diagnostics", "diagnostics_sha256"),
        ("raw_log", "raw_log_sha256"),
    ):
        if orb.get(path_key) is not None:
            verify_hash(Path(orb[path_key]), orb[hash_key], f"ORB {path_key}")
    for name, value in orb.get("failure_evidence", {}).items():
        verify_hash(Path(value["path"]), value["sha256"], f"ORB failure {name}")


def evaluate(args: argparse.Namespace) -> None:
    from vistream.metrics import ate_rmse

    manifest_path = args.output_dir / "manifest_before_gt.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["policy"] != json.loads(args.policy.read_text()):
        raise ValueError("evaluation policy differs from generation policy")
    verify_hash(
        Path(manifest["base_manifest"]),
        manifest["base_manifest_sha256"],
        "base manifest",
    )
    for entry in manifest["entries"]:
        verify_entry(entry)

    rows = []
    for entry in manifest["entries"]:
        with np.load(entry["candidate"], allow_pickle=False) as archive:
            candidate = poses_4x4(archive["c2w"])
        # GT is opened only after all source and candidate hashes verify.
        with np.load(entry["visual"], allow_pickle=False) as archive:
            visual = poses_4x4(archive[args.visual_pose_key])
            target = poses_4x4(archive[args.ground_truth_key])
        diagnostics = json.loads(Path(entry["diagnostics"]).read_text())
        rows.append(
            {
                "sequence": entry["sequence"],
                "environment": diagnostics["gate"]["environment"],
                "expert": diagnostics["gate"]["selected_expert"],
                "orb_weight": diagnostics["gate"]["orb_weight"],
                "visual_ate_rmse_m": ate_rmse(
                    visual[:, :3, 3], target[:, :3, 3], "sim3"
                )["ate_rmse"],
                "candidate_ate_rmse_m": ate_rmse(
                    candidate[:, :3, 3], target[:, :3, 3], "sim3"
                )["ate_rmse"],
            }
        )
    visual_values = np.asarray([row["visual_ate_rmse_m"] for row in rows])
    candidate_values = np.asarray([row["candidate_ate_rmse_m"] for row in rows])
    summary = {
        "protocol": manifest["protocol"],
        "policy": manifest["policy"],
        "rows": rows,
        "macro": {
            "visual_ate_rmse_mean_m": float(np.mean(visual_values)),
            "candidate_ate_rmse_mean_m": float(np.mean(candidate_values)),
            "improved_sequences": int(np.sum(candidate_values < visual_values)),
            "sequences_at_or_below_10m": int(np.sum(candidate_values <= 10.0)),
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
    generate_parser.add_argument("--base-root", type=Path, required=True)
    generate_parser.add_argument("--orb-root", type=Path, required=True)
    generate_parser.add_argument("--policy", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--policy", type=Path, required=True)
    evaluate_parser.add_argument("--visual-pose-key", default="predicted_c2w")
    evaluate_parser.add_argument("--ground-truth-key", default="ground_truth_c2w")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generate(arguments) if arguments.command == "generate" else evaluate(arguments)
